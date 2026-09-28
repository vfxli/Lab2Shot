"""Where an install gets things from, and how it keeps going when the network does not.

Every resource has an ordered list of sources: the official one first, then mirrors (the admin page's 安装 settings;
by default only well-known ones). A mirror is used only for what can be checked against the official identity:

    a file with a sha256 (model files, micromamba)  checked after download; a mismatch is deleted, logged, next source
    a git commit (the pinned full SHA)              fetched by that SHA and checked out: git names objects by content
    Python packages                                 only from a hash-pinned lock (EnvSpec.lock, --require-hashes)

Anything else (a Hugging Face snapshot without file hashes, packages without a lock) comes from its official source
only. Network errors are retried with exponential backoff (install.retries, install.backoff_max); a download resumes
from what it already has (.part files), across retries and across sources.
"""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import config
from ..messages import Msg
from ..extensions.spec import GitSource, InstallError
from .events import Sink

PARALLEL_MIN_BYTES = 256 << 20  # smaller files are not worth several connections
PARALLEL_PARTS = 8
STALL_S = 60  # a connection that sends nothing this long is dropped and the download resumes
HF = "https://huggingface.co"
GITHUB = "https://github.com/"


@dataclass(frozen=True)
class Source:
    url: str
    official: bool
    name: str  # the host, as the log names it


@dataclass(frozen=True)
class Policy:
    retries: int = 5
    backoff_max: float = 60.0
    hf: tuple[str, ...] = ()
    pypi: tuple[str, ...] = ()
    github: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls) -> "Policy":
        s = config.settings()
        split = lambda key: tuple(u.rstrip("/") for u in str(s.value(key) or "").split())  # noqa: E731
        return cls(int(s.value("install.retries")), float(s.value("install.backoff_max")),
                   split("install.mirror_hf"), split("install.mirror_pypi"), split("install.mirror_github"))

    def delay(self, attempt: int) -> float:
        """Seconds to wait before attempt `attempt + 1`: 1, 2, 4, 8 ... up to backoff_max."""
        return min(self.backoff_max, float(2 ** (attempt - 1)))


class Gated(Exception):
    """A Hugging Face file whose access request is not approved (401/403): never tried on a mirror."""


def host(url: str) -> str:
    return url.split("://", 1)[-1].split("/", 1)[0]


def candidates(url: str, policy: Policy, checkable: bool) -> list[Source]:
    """The official URL, then its mirrors when the result can be checked against the official identity."""
    out = [Source(url, True, host(url))]
    if not checkable:
        return out
    if url.startswith(HF + "/"):
        out += [Source(m + url[len(HF):], False, host(m)) for m in policy.hf]
    elif url.startswith(GITHUB):
        out += [Source(f"{m}/{url}", False, host(m)) for m in policy.github]  # proxy mirrors take the whole URL
    return out


# ------------------------------------------------------------------ retrying

# Network errors as whole words or phrases in command output (the wording of git, uv, pip and curl), plus
# 429/502/503/504 when stated as an HTTP status. Substring matching is not used: "tls" would match "utils", "504"
# would match "line 2504" and "network" would match "networkx", so a compile error would be retried as a network
# failure and reported as one.
# Every pattern has word boundaries; status codes match only in forms such as "HTTP 503", "status 503" and
# "503 Service Unavailable".
NETWORK_SIGNS = re.compile(
    r"\b(?:timed out|read ?timeout|connect(?:ion)? ?timeout|timeout(?:error)? (?:while|when|on|reading|connecting)"
    r"|(?:read|connect)timeouterror|connection (?:refused|reset|aborted|closed|error|broken|timed out|was reset)"
    r"|connectionerror|connecterror|failed to connect|could not connect|could not resolve|temporary failure in name resolution"
    r"|name or service not known|nodename nor servname|no route to host|network is unreachable|network (?:error|failure|down|unreachable)"
    r"|reset by peer|broken pipe|early eof|unexpected eof|rpc failed|unable to access|failed to fetch|failed to download"
    r"|error sending request|dns error|tls (?:handshake|connection|error)|ssl(?:_read|_write|error| error| handshake)|gnutls"
    r"|remote end hung up|curl \d+(?::| error)"
    r"|(?:http|status(?: code)?|error|response) ?[:=]? ?(?:429|502|503|504)\b"
    r"|(?:429|502|503|504) (?:too many requests|bad gateway|service unavailable|gateway time-?out))")


def network_said(text: str) -> bool:
    """Whether this command output (its last lines) names the network as the cause of the failure."""
    return NETWORK_SIGNS.search(text.lower()) is not None


def transient(exc: BaseException) -> bool:
    """Whether the error is worth retrying (a dropped, stalled or refused connection, or a busy server)."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in (408, 425, 429, 500, 502, 503, 504)
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, (NetworkFailure, http.client.HTTPException)):  # a connection cut off mid-read (IncompleteRead)
        return True
    if isinstance(exc, OSError) and not isinstance(exc, (FileNotFoundError, PermissionError, IsADirectoryError)):
        return True
    return False


class NetworkFailure(Exception):
    """A command failed with what its output says is the network (git, uv): retried like a dropped download."""


class NoRanges(Exception):
    """A server answered a byte-range request with the whole file (no 206): fetch() downloads sequentially instead."""


def pause(sink: Sink, seconds: float) -> None:
    """Wait between two attempts; a cancel ends the wait at once."""
    sink.cancel.wait(seconds)
    sink.check()


def retry(fn: Callable[[], object], what: str, sink: Sink, policy: Policy):
    """fn() until it succeeds, retrying network errors with backoff; the last error (or any other) is raised."""
    for attempt in range(1, policy.retries + 1):
        sink.check()
        try:
            return fn()
        except Gated:
            raise
        except Exception as exc:
            if not transient(exc) or attempt == policy.retries:
                raise
            wait = policy.delay(attempt)
            sink.say(Msg("W-INSTALL-RETRY", what=what, attempt=attempt, attempts=policy.retries, seconds=wait,
                         detail=str(exc)[:200] or type(exc).__name__))
            pause(sink, wait)
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ files


def sha256_of(path: Path) -> str:
    from ..io.digest import sha256

    return sha256(path)


def matches(path: Path, want: str) -> bool:
    """Whether the file's sha256 equals `want`. On a mismatch the file is read and hashed a second time.

    Under heavy load, reading the same multi-GB file has been observed to occasionally yield a different sha256, so a
    single mismatch is not treated as corruption: a false negative would delete a valid multi-GB weight file (`_verify`
    deletes on mismatch). A second read is cheap (about 1 s for 2.5 GB); the file is considered corrupt only when both
    reads mismatch.
    """
    return sha256_of(path) == want or sha256_of(path) == want


def fetch_checked(url: str, dest: Path, sha256: str, sink: Sink, policy: Policy, gated: bool = False,
                  download: Callable[..., None] | None = None) -> Source:
    """`url` into `dest` from the first source that works and whose file matches `sha256` (mirrors only when there is
    one). Returns the source used. InstallError when every source failed (E-INSTALL-FORBIDDEN when the official one
    refused with 401/403: the file needs an access request); Gated when the official one refuses a gated file.

    A file already at `dest` that does not match `sha256` is a different version (the adapter pinned a new sha256
    under the same file name), not a broken download: it stays there until the new one is downloaded beside it
    (<dest>.new) and checked, then the new one takes its place."""
    download = download or fetch
    target = dest
    # An existing dest with a different hash means the declared sha256 changed, not that the file is broken (fetch()
    # skips an existing dest, so checking it directly would misreport it as corrupt and blame the source). Environments
    # in use still depend on the old file, so it is kept: the new one is downloaded beside it and replaces it once
    # verified.
    if sha256 and dest.exists() and dest.stat().st_size > 0 and not matches(dest, sha256):
        sink.say(Msg("N-INSTALL-NEWVERSION", what=dest.name))
        target = dest.with_name(dest.name + ".new")
    tried: list[str] = []
    last: BaseException | None = None
    refused: urllib.error.HTTPError | None = None  # the official source said 401 / 403: an access request, not the network
    for src in candidates(url, policy, checkable=bool(sha256)):
        sink.check()
        if not src.official:
            sink.say(Msg("N-INSTALL-MIRROR", what=dest.name, mirror=src.name))
        # fetch() skips a complete file already at the target: then nothing is downloaded and the log must not say
        # the file came from this source (it only says it was checked)
        local = target.exists() and target.stat().st_size > 0
        try:
            retry(lambda: download(src.url, target, sink), dest.name, sink, policy)
        except urllib.error.HTTPError as exc:
            if src.official and exc.code in (401, 403):
                if gated:
                    raise Gated() from exc
                refused = exc
            tried.append(src.name)
            last = exc
            if src.official and exc.code == 404:
                break  # the official address itself is wrong: a mirror of it would not serve the pinned file
            continue
        except InstallError:
            raise
        except Exception as exc:
            if not transient(exc):
                raise
            tried.append(src.name)
            last = exc
            continue
        # matches() re-reads once before treating the file as corrupt (see its docstring).
        if sha256 and not matches(target, sha256):
            got = sha256_of(target)
            target.unlink(missing_ok=True)
            tried.append(src.name)
            sink.say(Msg("W-INSTALL-HASHMISMATCH", what=dest.name, source=src.name, got=got[:16], want=sha256[:16]))
            continue
        if target != dest:
            target.replace(dest)  # the old version is replaced only after verification
        sink.say(Msg("I-INSTALL-LOCALOK", what=dest.name) if local else Msg("I-INSTALL-SOURCE", what=dest.name, source=src.name))
        return src
    if refused is not None:
        # The official source refused (401/403) a file not marked gated: neither the network nor a mirror can help, so
        # the error states that access must be requested (the page is derived as in Weight.page).
        repo, sep, _ = url.partition("/resolve/")
        page = repo if sep and repo.startswith(HF + "/") else url
        raise InstallError(Msg("E-INSTALL-FORBIDDEN", url=url, code=refused.code, page=page)) from refused
    raise InstallError(Msg("E-INSTALL-DOWNLOAD", url=url, sources=tried or [host(url)], detail=str(last or "")[:200]))


def fetch(url: str, dest: Path, sink: Sink) -> None:
    """One attempt at `url` into `dest`, going on from <dest>.part (or its parallel parts); raises on a dropped or
    stalled connection so retry() calls it again. Only a complete file becomes `dest`."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = ranged_size(url)
    if size is not None and size >= PARALLEL_MIN_BYTES:
        try:
            _fetch_parallel(url, dest, size, sink)
            return
        except NoRanges:
            # The probe returned 206 but a part request returned the whole file with 200 (some mirrors honor Range only
            # for the first range). The source is still usable: fall back to a sequential download from the start.
            sink.say(Msg("N-INSTALL-NORANGES", url=url))
            for i in range(PARALLEL_PARTS):
                dest.with_name(f"{dest.name}.part{i}").unlink(missing_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    try:
        response = urllib.request.urlopen(request(url, **({"Range": f"bytes={have}-"} if have else {})), timeout=STALL_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # nothing left to fetch: the partial file is already whole
            part.replace(dest)
            return
        raise
    with response:
        refuse_web_page(response, url, dest)
        resumed = bool(have) and response.status == 206
        length = int(response.headers.get("Content-Length") or 0)
        total = (have + length if resumed else length) or None
        done = have if resumed else 0
        with part.open("ab" if resumed else "wb") as fh:
            while chunk := response.read(1 << 20):
                sink.check()
                fh.write(chunk)
                done += len(chunk)
                sink.progress(dest.name, done, total)
    got = part.stat().st_size
    if total is not None and got != total:
        if got > total:  # the server ignored the range and the data was appended: start over
            part.unlink()
        raise ConnectionError(f"{got} / {total} bytes")
    part.replace(dest)
    sink.progress(dest.name, got, got)


def _fetch_parallel(url: str, dest: Path, size: int, sink: Sink) -> None:
    """`size` bytes in PARALLEL_PARTS byte ranges, each going on from its own <dest>.part<i>, then joined."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    bounds = [size * i // PARALLEL_PARTS for i in range(PARALLEL_PARTS + 1)]
    parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(PARALLEL_PARTS)]
    lock = Lock()
    done = [p.stat().st_size if p.exists() else 0 for p in parts]

    def one(i: int) -> None:
        start, end, part = bounds[i], bounds[i + 1], parts[i]
        have = part.stat().st_size if part.exists() else 0
        if have >= end - start:
            return
        with urllib.request.urlopen(request(url, Range=f"bytes={start + have}-{end - 1}"), timeout=STALL_S) as response, \
                part.open("ab") as fh:
            refuse_web_page(response, url, dest)
            if response.status != 206:
                raise NoRanges(url)  # caught by fetch(), which downloads sequentially from this source
            while chunk := response.read(1 << 20):
                sink.check()
                fh.write(chunk)
                with lock:
                    done[i] += len(chunk)
                    sink.progress(dest.name, sum(done), size)

    with ThreadPoolExecutor(PARALLEL_PARTS) as pool:
        list(pool.map(one, range(PARALLEL_PARTS)))
    if any((p.stat().st_size if p.exists() else 0) < bounds[i + 1] - bounds[i] for i, p in enumerate(parts)):
        raise ConnectionError("a part is not complete")
    joined = dest.with_name(dest.name + ".part")
    with joined.open("wb") as out:
        for part in parts:
            with part.open("rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 24)
    got = joined.stat().st_size
    if got != size:
        joined.unlink()
        for part in parts:  # the parts are of no use either: the next try downloads afresh instead of joining the same again
            part.unlink(missing_ok=True)
        raise InstallError(Msg("E-INSTALL-INCOMPLETE", got=got, want=size, url=url))
    joined.replace(dest)
    for part in parts:
        part.unlink()


def request(url: str, **headers: str) -> urllib.request.Request:
    """A download request. Files on Hugging Face carry the user's token (gated repositories need it), sent to
    huggingface.co only: never along a redirect to the file's storage host, never to a mirror."""
    req = urllib.request.Request(url, headers={"User-Agent": "lab2shot", **headers})
    if url.startswith(HF + "/"):
        from huggingface_hub import get_token

        if token := get_token():
            req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def refuse_web_page(response, url: str, dest: Path) -> None:
    """A weight never comes as an HTML page: that is a login, quota or error page served with status 200."""
    if response.headers.get_content_type() == "text/html" and dest.suffix.lower() not in (".html", ".htm"):
        raise InstallError(Msg("E-INSTALL-WEBPAGE", url=url))


def ranged_size(url: str) -> int | None:
    """The file's size when the server (after redirects) serves byte ranges, else None."""
    try:
        with urllib.request.urlopen(request(url, Range="bytes=0-0"), timeout=30) as response:
            content_range = response.headers.get("Content-Range", "")  # "bytes 0-0/<size>"
            if response.status == 206 and "/" in content_range and not content_range.endswith("/*"):
                return int(content_range.rsplit("/", 1)[1])
    except (OSError, ValueError):
        pass
    return None


# ------------------------------------------------------------------ git


def git_env() -> dict[str, str]:
    """The environment every git command of the installer runs in: the machine's own git configuration stays out of
    it. A url.insteadOf in ~/.gitconfig would redirect the official address, the LFS smudge filter would fetch large
    files on checkout (the pin is a commit, not LFS objects; weights are downloaded separately), global hooks would run
    scripts in the checkout, and a credential prompt would hang without a terminal. Any of these can make a checkout fail
    or produce something else; installing a pinned commit needs none of the user's configuration."""
    return os.environ | {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_LFS_SKIP_SMUDGE": "1",
                         "GIT_TERMINAL_PROMPT": "0"}


def checkout(src: GitSource, dest: Path, sink: Sink, policy: Policy, run: Callable[..., str]) -> str:
    """The pinned commit `src.commit` as a plain checkout in `dest`, from the official URL or a mirror (whatever a
    source hands over must check out as exactly that commit). Returns the host it came from. `dest` is never the
    live checkout (installer/run.py step_repo: a new commit goes into a folder beside it): what is there is a
    leftover of an earlier attempt and is removed first."""
    tried: list[str] = []
    env = git_env()
    for source in candidates(src.url, policy, checkable=True):
        sink.check()
        if not source.official:
            sink.say(Msg("N-INSTALL-MIRROR", what=src.url.rsplit("/", 1)[-1], mirror=source.name))
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)

        def fetch_commit() -> None:
            run(["git", "fetch", "--progress", "--depth", "1", source.url, src.commit], cwd=dest, env=env, network=True, idle=True)

        run(["git", "init", "-q"], cwd=dest, env=env)
        try:
            retry(fetch_commit, src.url.rsplit("/", 1)[-1], sink, policy)
        except NetworkFailure:  # still the network after every retry: the next source
            tried.append(source.name)
            continue
        except InstallError:
            if source.official:
                raise  # the official source answered and has no such commit: a mirror of it would not either
            tried.append(source.name)
            continue
        run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=dest, env=env)
        head = run(["git", "rev-parse", "HEAD"], cwd=dest, env=env).strip()
        if head != src.commit:
            tried.append(source.name)
            sink.say(Msg("W-INSTALL-COMMITMISMATCH", source=source.name, got=head[:12], want=src.commit[:12]))
            continue
        run(["git", "remote", "add", "origin", src.url], cwd=dest, env=env)
        sink.say(Msg("I-INSTALL-SOURCE", what=f"{src.url.rsplit('/', 1)[-1]} @ {src.commit[:10]}", source=source.name))
        return source.name
    raise InstallError(Msg("E-INSTALL-DOWNLOAD", url=src.url, sources=tried, detail=""))


# ------------------------------------------------------------------ files shared by hash between extensions


def weights_index() -> Path:
    """Verified downloads by sha256, across extensions (third_party/_weights_index.json): an extension that needs a
    file another one already has gets a hard link instead of a second download and a second copy on disk."""
    return config.THIRD_PARTY_DIR / "_weights_index.json"


def _index() -> dict:
    try:
        return json.loads(weights_index().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@contextmanager
def _index_lock():
    """One writer of the shared index at a time, across processes (the server's install task and the command line
    both write it; run.py's `locked` is per extension, this file is shared by all of them)."""
    lock = weights_index().with_name(weights_index().name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def remember(sha256: str, path: Path) -> None:
    from ..io.atomic import write_text

    # The read-modify-write runs under the lock and replaces the file atomically. Without the lock, two concurrent
    # installs would lose one entry; a crash mid-write would leave partial JSON that _index() reads as an empty table.
    with _index_lock():
        index = _index()
        index[sha256] = {"path": path.relative_to(config.THIRD_PARTY_DIR).as_posix(), "size": path.stat().st_size}
        write_text(weights_index(), json.dumps(index, indent=1))


def share(sha256: str, dest: Path) -> bool:
    """Make `dest` a hard link to the verified file with this sha256, if one exists elsewhere. True when linked."""
    entry = _index().get(sha256) if sha256 else None
    if not entry:
        return False
    src = config.THIRD_PARTY_DIR / entry["path"]
    if not src.is_file() or src.stat().st_size != entry["size"] or src.resolve() == dest.resolve():
        return False
    if dest.exists() and os.path.samefile(src, dest):
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    from ..io.files import link_or_copy

    tmp = dest.with_name(dest.name + ".link")
    tmp.unlink(missing_ok=True)
    link_or_copy(src, tmp)
    tmp.replace(dest)
    return True


def package_indexes(lock: Path | None, policy: Policy) -> list[Source]:
    """The package index to install from, in order: the official one (uv's default) and, only with a hash lock,
    the PyPI mirrors."""
    out = [Source("", True, "pypi.org")]
    if lock is not None:
        out += [Source(m, False, host(m)) for m in policy.pypi]
    return out

