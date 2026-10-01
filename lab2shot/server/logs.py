"""HTTP logging: requests that fail or modify state are written to the server log (lab2shot/logs.py), users can
submit their browser's log window to it, and the admin page reads it."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .repeats import Repeats
from .routes import Access, Body, Limit, Router
from .. import logs
from ..errors import Failed
from ..messages import Msg
from . import auth
from .access import manages
from .farm import client_of
from .wire import off_loop

admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # admin routes of this module, included by app.py
router = Router(prefix="/api", tags=["日志"])
http = logs.get("http")

# Paths too frequent to log on success: status polling while editing, blob uploads (one per frame), queue polling.
QUIET = ("/api/status", "/api/uploads/parts", "/api/uploads/have", "/api/queue", "/api/admin/queue", "/api/admin/log")


def failed(request: Request, exc: BaseException) -> JSONResponse:
    """A request the program failed: its reference and traceback in the log, the answer as every error's (errors.py
    MessageError) with that reference. Whichever layer it failed in: the routes (install), the guard itself
    (server/access.py Guard)."""
    ref = uuid.uuid4().hex[:8]
    logs.say(http, Msg("E-HTTP-FAILED", who=auth.who(request), method=request.method, path=request.url.path, ref=ref),
             logs.error_text(exc))
    said = Failed(Msg("E-SERVER-INTERNAL", ref=ref))
    return JSONResponse(said.answer(), status_code=said.status)


REFUSED_S = 600  # one address's refused requests of one status within this long: one log line, counted


def _refused_again(said: Msg, count: int, last: float) -> None:
    logs.say(http, Msg("W-HTTP-REQUESTSAGAIN", line=said.text, count=count))


# a client sending what is refused again and again (an address no route has, a flood held back) is in the log once a
# while, counted (server/repeats.py): never a way to roll the log over what it keeps
_refused = Repeats(REFUSED_S, lambda key, t, said: logs.say(http, said) or said, _refused_again)


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.time()
        who = auth.who(request)
        if (s := request.scope.get("lab2shot_session")) is not None:  # the account, once resolved by the guard
            who = f"{s.user.username}@{who}"
        try:
            response = await call_next(request)
        except Exception as exc:  # details go to the log only; the response never exposes code or paths
            # Writing a log line is file I/O (including rotation, lab2shot/logs.py) and must not run on the event loop.
            return await off_loop(failed, request, exc)
        path, status = request.url.path, response.status_code
        changes = request.method not in ("GET", "HEAD") and not path.startswith(QUIET)
        if path.startswith("/api/") and status >= 400:
            said = Msg("W-HTTP-REQUEST", who=who, method=request.method, path=path, status=status, ms=int((time.time() - started) * 1000))
            await off_loop(_refused.happened, (auth.who(request), status), said)
        elif path.startswith("/api/") and changes:
            said = Msg("I-HTTP-REQUEST", who=who, method=request.method, path=path, status=status, ms=int((time.time() - started) * 1000))
            await off_loop(logs.say, http, said)
        return response


class ClientLog(Body):
    text: str  # contents of the browser's log window
    client: dict = {}


@router.post("/logs", access=Access.user("把网页的日志发给管理员：只能写，不能读", limit=Limit(body=262144, burst=5, per_s=0.016666666666666666)), summary="把网页的日志窗口发给管理员：写进服务日志，附上是谁发的")
def post_log(req: ClientLog, request: Request) -> dict:
    who = client_of(request, req.client)
    # Size and rate are enforced by the guard through the route's Limit (server/routes.py).
    logs.say(logs.get("client"), Msg("W-LOG-FROMPAGE", who=who.who, ip=who.details.get("ip", ""), app=who.app), req.text)
    return {"ok": True}


@admin.get("/security", access=Access.admin("security.manage"), summary="安全：最近的可疑请求（登录失败、没开放的接口、可疑的路径、请求太频繁……）、各类的次数、暂时封住的来源、现在有几个登录")
def admin_security(request: Request) -> dict:
    from .. import accounts
    from . import auth

    s = auth.session(request)
    return {**auth.guards().watch.view(), "online": accounts.online(lambda owner, role: manages(s, owner, role)), "limits": {
        "rate_per_s": auth.RATE_PER_S, "burst": auth.RATE_BURST, "block_after": auth.BLOCK_AFTER,
        "block_window_min": auth.BLOCK_WINDOW_S // 60, "block_min": auth.BLOCK_S // 60, "free": auth.FREE,
        "client_lock": auth.CLIENT_LOCK, "address_free": auth.ADDRESS_FREE, "address_lock": auth.ADDRESS_LOCK,
        "subject_free": auth.SUBJECT_FREE, "max_wait_s": auth.MAX_WAIT_S, "window_min": auth.WINDOW_S // 60}}


class Unblock(Body):
    client: str


@admin.post("/security/unblock", access=Access.admin("security.manage"), summary="解开一个被暂时封住的来源")
def admin_unblock(req: Unblock) -> dict:
    from . import auth

    auth.guards().watch.unblock(req.client)
    return admin_security()


@admin.get("/log", access=Access.admin("logs.view"), summary="服务日志的最后几行（最新的在最后）")
def admin_log(lines: int = 500) -> dict:
    return {"file": str(logs.log_file()), "lines": logs.tail(min(max(lines, 1), 5000))}
