"""How answers travel: one place for what the network carries back.

- Caching: nothing any shared cache may keep. A tunnel in front of this server (Cloudflare caches `.js` and `.png`
  by default) must never hand code or pictures behind a login to someone without one, so every answer is
  `private`. Content-addressed answers (the page's hashed scripts, a packet's frames) are kept by the browser permanently
  (IMMUTABLE); everything else is asked again (`no-cache`) and, for JSON, answered 304 without a body when it has not
  changed (an ETag of its bytes): polls and reloads cost a few hundred bytes.
- Compression: the page's scripts and styles are compressed once when built (brotli and gzip copies next to them,
  webui/vite.config.ts); Assets sends the one the browser takes. JSON and view data go through GZipMiddleware
  (server/app.py); pictures and archives already are compressed, and files keep their size.
- Every frame the viewer shows is a proxy: when a solve finishes, frames are scaled proportionally to the tier the
  administrator set, compressed and stored (lab2shot/view/proxy.py). One address is one file and one sequence of
  bytes, with no lossless / lossy tiers to switch between; lossless viewing is done by downloading the output (its
  zip) and viewing it in Nuke or a DCC. The 3D path is not here (server/view_data.py: skeletons, point clouds, thinning).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import mimetypes
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TypeVar

import anyio
from anyio.lowlevel import RunVar
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..errors import NotFound
from ..messages import Msg
from ..serving import carried

IMMUTABLE = "private, max-age=31536000, immutable"  # content-addressed: the same URL is always the same bytes
FRESH = "private, no-cache"  # asked again every time (a 304 when unchanged)
NEVER = "private, no-store"  # never kept by any cache (what a result holds is asked for again each time)
HSTS = "max-age=15552000"  # 180 days of "this host is https only"; never for localhost or an address (it would hold for every port of it)
ETAG_MAX = 8 << 20  # JSON answers up to this size get an ETag (bigger ones are streamed as they are)


def admin_answer(said: str) -> str:
    """What an administrator's answer says about caching: never kept, but for a picture at a content-addressed address
    (one whose content or version is in its address), kept in the administrator's own
    browser only: a player scrubbing back never fetches a frame twice."""
    return (said if "private" in said else f"private, {said}") if "immutable" in said else "no-store"


OFF_LOOP_THREADS = 16  # the middlewares' own threads: they never wait behind the routes' (anyio's default 40)
_off_loop: RunVar[anyio.CapacityLimiter] = RunVar("lab2shot_off_loop")  # one per event loop
T = TypeVar("T")


async def off_loop(fn: Callable[..., T], *args) -> T:
    """Blocking work a middleware must do for every request (the guard's session lookup, a database read; hashing or
    scrubbing an answer) on a thread, so the event loop goes on serving every other request meanwhile."""
    try:
        limiter = _off_loop.get()
    except LookupError:
        limiter = anyio.CapacityLimiter(OFF_LOOP_THREADS)
        _off_loop.set(limiter)
    return await anyio.to_thread.run_sync(fn, *args, limiter=limiter)


def _etag(body: bytes) -> str:
    return f'W/"{hashlib.blake2b(body, digest_size=12).hexdigest()}"'


def _private(value: str) -> str:
    v = value.lower()
    return value if "private" in v or "no-store" in v else f"private, {value}"


class Wire:
    """The middleware between the guard (server/access.py) and compression: Cache-Control on every answer, ETags and
    304 on JSON a GET asks for."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        asking = scope["method"] == "GET"
        known = Headers(scope=scope).get("if-none-match", "")
        started: Message | None = None
        held: list[bytes] = []

        async def wired(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = _private(headers.get("Cache-Control", FRESH))
                length = headers.get("content-length", "")
                if (asking and message["status"] == 200 and "etag" not in headers and headers.get("content-type", "").startswith("application/json")
                        and length.isdigit() and int(length) <= ETAG_MAX):
                    started = message  # held until the body is here: its ETag is of the body
                    return
                await send(message)
                return
            if started is None:
                await send(message)
                return
            held.append(message.get("body", b""))
            if message.get("more_body"):
                return
            body = b"".join(held)
            tag = await off_loop(_etag, body)
            headers = MutableHeaders(scope=started)
            headers["ETag"] = tag
            if tag in [t.strip() for t in known.split(",")]:
                del headers["content-length"]
                headers["content-length"] = "0"
                await send({**started, "status": 304})
                await send({"type": "http.response.body", "body": b""})
                return
            await send(started)
            await send({"type": "http.response.body", "body": body})

        await self.app(scope, receive, wired)


class Assets(StaticFiles):
    """The built page's files (webui/dist/assets), by name: hashed names never change, so the browser keeps them
    permanently. A script or style goes as its brotli or gzip copy when the browser takes one (made by the build);
    the copies are never asked for by their own names."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        if path.endswith((".br", ".gz")):
            raise NotFound(Msg("E-SERVER-NOCOPY"))
        accept = Headers(scope=scope).get("accept-encoding", "")
        for coding, suffix in (("br", ".br"), ("gzip", ".gz")):
            if coding not in accept:
                continue
            full = Path(str(self.directory)) / (path + suffix)
            if os.sep + ".." in os.path.normpath(os.sep + path) or not full.is_file():
                continue
            kind = mimetypes.guess_type(path)[0] or "application/octet-stream"
            return FileResponse(full, media_type="text/javascript" if path.endswith(".js") else kind,
                                headers={"Content-Encoding": coding, "Vary": "Accept-Encoding", "Cache-Control": IMMUTABLE})
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = IMMUTABLE
            response.headers["Vary"] = "Accept-Encoding"
        return response


# ------------------------------------------------------------------ view proxies: a frame's picture, a channel
#
# The viewer receives only proxies: when a solve finishes, the same job scales them proportionally to the tier the
# administrator set, compresses and stores them (`lab2shot/view/proxy.py`). This section only covers transport: which
# file is sent and whether the browser may keep it. One address is one file and one sequence of bytes, with no second
# answer negotiated by connection speed or client capability, so `Vary: Accept` is not needed.
#
# The byte layout of a channel is in `lab2shot/view/channels.py` (encoding belongs to the viewing layer, not HTTP).

from ..view.channels import CHANNEL_MEDIA


@cache
def _ahead_kind():
    from ..farm.tasks import Kind
    from ..messages import Msg

    return Kind("view.ahead", title=lambda subject: Msg("I-VIEW-TASKAHEAD", packet=subject.split(":")[0][:12]), lasting=True)


def ahead(subject: str, pictures: Callable[[], list[Callable[[], object]]]) -> None:
    """Fill in, in the background, whatever is missing from a packet's proxies (one background task per account, packet
    and tier: `subject`, the packet and tier, which ahead scopes to the account).

    Normally there is nothing to do: proxies are made in the same job when a solve finishes (`view/proxy.py build`).
    This path covers two cases: a packet whose proxies of this tier were never made, or a tier the administrator
    changed to (proxies of the new tier are made on demand). Each proxy is made only once
    (view/encode.py), so the requested frame is made immediately and the rest follow in the background, and playback
    does not overtake it.
    Never in the way of the answer: a server ending takes none (they are made when asked), and a restart does not
    wait for it."""
    from ..errors import MessageError
    from ..farm import farm
    from ..serving import account

    # one task per account too: each account has its own cache (another account's copy of the same packet being done
    # says nothing about this one's), and one account's tasks tell nothing of another's
    subject = f"{subject}@{account().user_id}"
    kind, tasks = _ahead_kind(), farm().tasks
    last = tasks.latest(kind, subject)
    if last is not None and last.state in ("queued", "running", "done"):
        return

    def work(task) -> None:
        todo = pictures()
        # Several at a time. Serially, writing the PNG of one 1080p display image takes about 0.2 s, so 150 frames take
        # half a minute and playback would catch up and stall. Parallelism scales almost linearly (OIIO releases the GIL
        # while writing PNG). The thread count follows engine/cook.py FRAME_THREADS rather than a separate setting.
        from ..engine.cook import FRAME_THREADS

        done = 0
        with ThreadPoolExecutor(FRAME_THREADS) as pool:
            for i in range(0, len(todo), FRAME_THREADS):
                task.check()
                for _ in pool.map(carried(lambda make: make()), todo[i:i + FRAME_THREADS]):  # the viewer's account's cache
                    done += 1
                task.progress(done, len(todo))

    try:
        tasks.submit(kind, subject, "viewer", work)
    except MessageError:
        pass


def versioned(p, g: str) -> bool:
    """The precondition for one address being one sequence of bytes: the address carries the packet's generation `g`
    (data/packet.py created; the page takes it from the status reply's gens). Present and matching -> True, and the
    answer may be marked immutable; present but not matching -> an old address from before a recompute, 404
    E-VIEW-STALE (the page's next status reply switches to the new generation's address); absent -> False, the answer
    is sent but the browser may not keep it long: marking an address without a generation immutable would leave the
    browser with old bytes for a year after the same fingerprint is recomputed or the administrator changes the
    proxy tier."""
    if not g:
        return False
    if g != (p.created or ""):
        raise NotFound(Msg("E-VIEW-STALE"))
    return True


def picture_answer(picture: Path, kept: bool = True) -> FileResponse:
    """The answer for a frame's proxy picture. When the address carries generation and tier (`versioned`), the same
    address is always the same bytes and the browser keeps it (IMMUTABLE); other addresses are asked for every time."""
    kind = mimetypes.guess_type(picture.name)[0] or "image/png"
    return FileResponse(picture, media_type=kind, headers={"Cache-Control": IMMUTABLE if kept else FRESH})


def channel_answer(blob_gz: Path, accept_encoding: str, kept: bool = True) -> Response:
    """The answer for a proxy channel: stored gzipped on disk (when the address carries generation and tier, the same
    address is always the same bytes and the browser keeps it); a browser that accepts gzip gets the file as is,
    otherwise it is decompressed on the fly."""
    import gzip

    said = {"Cache-Control": IMMUTABLE if kept else FRESH, "Vary": "Accept-Encoding"}
    if "gzip" in (accept_encoding or ""):
        return FileResponse(blob_gz, media_type=CHANNEL_MEDIA, headers={**said, "Content-Encoding": "gzip"})
    return Response(gzip.decompress(blob_gz.read_bytes()), media_type=CHANNEL_MEDIA, headers=said)
