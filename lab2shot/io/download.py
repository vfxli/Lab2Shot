"""Downloads that survive dropped connections: a resumable downloader for bulk fetches (benchmark data). The
extension installer fetches weights through lab2shot/installer/sources.py instead, which chooses between official
sources and verified mirrors, retries according to the administrator's policy and stops when its task is cancelled.
The two implementations overlap and could be merged.

    download(url, dest, sha256=...)   a whole file, resumed from <dest>.part after a drop. Above PARALLEL_MIN_BYTES,
                                      when the server supports byte ranges, the file is fetched as PARALLEL_PARTS
                                      ranges, each resumable in its own <dest>.part<i>, then joined. An HTML page
                                      returned in place of a file is rejected. The sha256 is verified when given (a
                                      mismatching file is deleted so that the next attempt fetches it again).
    Stream(url)                       a read stream over HTTP that reconnects with a Range request at the current
                                      position when the connection drops (a tar.gz is streamed end to end without
                                      disturbing the gzip state)

Errors are reported with message codes (E-DOWNLOAD-*, lab2shot/messages/io.toml)."""

from __future__ import annotations

import http.client
import io
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from ..errors import Invalid
from ..messages import Msg
from .digest import sha256 as file_sha256

PARALLEL_MIN_BYTES = 256 << 20  # smaller files do not benefit from multiple connections
PARALLEL_PARTS = 8
STALL_S = 60  # a connection idle for this many seconds is dropped and the download resumes
CHUNK = 1 << 20


def _request(url: str, headers: dict | None, unredirected: dict | None, **more: str) -> urllib.request.Request:
    """`headers` are sent with every request; `unredirected` (a token) is sent only to the requested host, never across a redirect."""
    request = urllib.request.Request(url, headers={"User-Agent": "lab2shot", **(headers or {}), **more})
    for k, v in (unredirected or {}).items():
        request.add_unredirected_header(k, v)
    return request


def ranged_size(url: str, headers: dict | None = None, unredirected: dict | None = None) -> int | None:
    """Return the file's size when the server (after redirects) supports byte ranges, otherwise None."""
    try:
        with urllib.request.urlopen(_request(url, headers, unredirected, Range="bytes=0-0"), timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")  # "bytes 0-0/<size>"
            if response.status == 206 and "/" in content_range and not content_range.endswith("/*"):
                return int(content_range.rsplit("/", 1)[1])
    except (OSError, ValueError):
        pass
    return None


def _refuse_web_page(response, url: str, dest: Path) -> None:
    """Reject an HTML response for a non-HTML file: it is a login, quota or error page served with status 200."""
    if response.headers.get_content_type() == "text/html" and dest.suffix.lower() not in (".html", ".htm"):
        raise Invalid(Msg("E-DOWNLOAD-WEBPAGE", url=url))


def download(url: str, dest: Path, *, sha256: str | None = None, headers: dict | None = None, unredirected: dict | None = None,
             progress: Callable[[int, int | None], None] | None = None, attempts: int = 6, force: bool = False) -> Path:
    """Download `url` into `dest` (see the module docstring), calling `progress(bytes so far, total or None)` during the transfer."""
    dest = Path(dest)
    if not (dest.exists() and dest.stat().st_size > 0 and not force):
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = ranged_size(url, headers, unredirected)
        if size is not None and size >= PARALLEL_MIN_BYTES:
            _parallel(url, dest, size, headers, unredirected, progress, attempts, force)
        else:
            _single(url, dest, headers, unredirected, progress, attempts, force)
    if sha256 and file_sha256(dest) != sha256:
        got = file_sha256(dest)
        dest.unlink()
        raise Invalid(Msg("E-DOWNLOAD-CHECKSUM", file=dest.name, want=sha256, got=got))
    return dest


def _single(url, dest: Path, headers, unredirected, progress, attempts: int, force: bool) -> None:
    part = dest.with_name(dest.name + ".part")
    if force and part.exists():
        part.unlink()
    total = None
    for _ in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        try:
            with urllib.request.urlopen(_request(url, headers, unredirected, **({"Range": f"bytes={have}-"} if have else {})),
                                        timeout=STALL_S) as response:
                _refuse_web_page(response, url, dest)
                resumed = have and response.status == 206
                length = int(response.headers.get("Content-Length") or 0)
                total = (have + length if resumed else length) or total
                done = have if resumed else 0
                with part.open("ab" if resumed else "wb") as fh:
                    while chunk := response.read(CHUNK):
                        fh.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # nothing left to fetch: the partial file is already complete
                part.replace(dest)
                return
            if exc.code in (401, 403, 404):
                raise Invalid(Msg("E-DOWNLOAD-REFUSED", url=url, status=exc.code)) from None
            continue
        except OSError:  # a dropped or stalled connection (URLError, TimeoutError): resume from the received bytes
            continue
        got = part.stat().st_size
        if total is None or got == total:
            part.replace(dest)
            return
        if got > total:  # the server ignored the range and the full body was appended: restart
            part.unlink()
    raise Invalid(Msg("E-DOWNLOAD-FAILED", url=url, attempts=attempts))


def _parallel(url, dest: Path, size: int, headers, unredirected, progress, attempts: int, force: bool) -> None:
    bounds = [size * i // PARALLEL_PARTS for i in range(PARALLEL_PARTS + 1)]
    parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(PARALLEL_PARTS)]
    if force:
        for part in parts:
            part.unlink(missing_ok=True)
    lock = Lock()
    arrived = [sum(p.stat().st_size for p in parts if p.exists())]

    def fetch(i: int) -> None:
        start, end, part = bounds[i], bounds[i + 1], parts[i]
        for _ in range(attempts):
            have = part.stat().st_size if part.exists() else 0
            if have >= end - start:
                return
            try:
                with urllib.request.urlopen(_request(url, headers, unredirected, Range=f"bytes={start + have}-{end - 1}"),
                                            timeout=STALL_S) as response, part.open("ab") as fh:
                    _refuse_web_page(response, url, dest)
                    if response.status != 206:
                        raise Invalid(Msg("E-DOWNLOAD-NORANGES", url=url))
                    while chunk := response.read(CHUNK):
                        fh.write(chunk)
                        with lock:
                            arrived[0] += len(chunk)
                            if progress is not None:
                                progress(arrived[0], size)
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 404):
                    raise Invalid(Msg("E-DOWNLOAD-REFUSED", url=url, status=exc.code)) from None
            except OSError:  # a dropped connection: resume this range from the received bytes
                pass
        raise Invalid(Msg("E-DOWNLOAD-PARTFAILED", url=url, part=i + 1, attempts=attempts))

    with ThreadPoolExecutor(PARALLEL_PARTS) as pool:
        list(pool.map(fetch, range(PARALLEL_PARTS)))
    joined = dest.with_name(dest.name + ".part")
    with joined.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 24)
    if joined.stat().st_size != size:
        raise Invalid(Msg("E-DOWNLOAD-INCOMPLETE", url=url, got=joined.stat().st_size, want=size))
    joined.replace(dest)
    for part in parts:
        part.unlink()


class Stream(io.RawIOBase):
    """A read-only byte stream over HTTP that reconnects with a Range request at the current offset when the connection
    drops (see the module docstring)."""

    def __init__(self, url: str, headers: dict | None = None, attempts: int = 20, wait_s: float = 5.0):
        super().__init__()
        self.url, self.headers, self.attempts, self.wait_s = url, dict(headers or {}), attempts, wait_s
        self.pos, self.response, self.size = 0, None, None  # size: total file size in bytes, once a response reports it

    def readable(self) -> bool:
        return True

    def _open(self) -> None:
        more = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        self.response = urllib.request.urlopen(_request(self.url, self.headers, None, **more), timeout=STALL_S)
        if self.pos and self.response.status != 206:
            self.response.close()
            raise Invalid(Msg("E-DOWNLOAD-NORANGES", url=self.url))
        content_range = self.response.headers.get("Content-Range", "")  # "bytes <start>-<end>/<size>"
        if "/" in content_range and not content_range.endswith("/*"):
            self.size = int(content_range.rsplit("/", 1)[1])
        elif not self.pos and self.response.headers.get("Content-Length"):
            self.size = int(self.response.headers["Content-Length"])

    def readinto(self, b) -> int:
        for attempt in range(self.attempts):
            try:
                if self.response is None:
                    self._open()
                n = self.response.readinto(b)
                if n or self.size is None or self.pos >= self.size:
                    self.pos += n
                    return n  # data, or the actual end of the file
                # the connection ended before the file did: treat as a drop and reconnect at the current position
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 404):
                    raise Invalid(Msg("E-DOWNLOAD-REFUSED", url=self.url, status=exc.code)) from None
            except (OSError, http.client.HTTPException):  # a dropped connection (IncompleteRead is not an OSError)
                pass
            self.response = None
            time.sleep(min(60.0, self.wait_s * (attempt + 1)))
        raise Invalid(Msg("E-DOWNLOAD-STREAM", url=self.url, at=self.pos))

    def close(self) -> None:
        if self.response is not None:
            self.response.close()
        super().close()
