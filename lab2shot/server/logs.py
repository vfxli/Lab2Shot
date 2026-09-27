"""HTTP logging: requests that fail or modify state are written to the server log (lab2shot/logs.py), users can
submit their browser's log window to it, and the admin page reads it."""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .routes import Access, Limit, Router
from .. import logs
from ..messages import Msg
from .farm import client_of
from .wire import off_loop

admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # admin routes of this module, included by app.py
router = Router(prefix="/api", tags=["日志"])
http = logs.get("http")

# Paths too frequent to log on success: status polling while editing, blob uploads (one per frame), queue polling.
QUIET = ("/api/status", "/api/uploads/parts", "/api/uploads/have", "/api/queue", "/api/admin/queue", "/api/admin/log")


def install(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.time()
        who = request.client.host if request.client else "?"
        if (s := request.scope.get("lab2shot_session")) is not None:  # the account, once resolved by the guard
            who = f"{s.user.username}@{who}"
        try:
            response = await call_next(request)
        except Exception as exc:  # details go to the log only; the response never exposes code or paths
            ref = uuid.uuid4().hex[:8]
            # Writing a log line is file I/O (including rotation, lab2shot/logs.py) and must not run on the event loop.
            await off_loop(logs.say, http, Msg("E-HTTP-FAILED", who=who, method=request.method, path=request.url.path, ref=ref),
                           logs.error_text(exc))
            said = Msg("E-SERVER-INTERNAL", ref=ref)
            return JSONResponse({"detail": said.text, "code": said.code}, status_code=500)
        path, status = request.url.path, response.status_code
        changes = request.method not in ("GET", "HEAD") and not path.startswith(QUIET)
        failed = status >= 400 and not (request.method == "HEAD" and status == 404)  # HEAD 404 means "not uploaded yet"
        if path.startswith("/api/") and (changes or failed):
            said = Msg("W-HTTP-REQUEST" if failed else "I-HTTP-REQUEST", who=who, method=request.method, path=path, status=status,
                       ms=int((time.time() - started) * 1000))
            await off_loop(logs.say, http, said)
        return response


class ClientLog(BaseModel):
    text: str  # contents of the browser's log window
    client: dict = {}


@router.post("/logs", access=Access.user("把网页的日志发给管理员：只能写，不能读", limit=Limit(body=262144, burst=5, per_s=0.016666666666666666)), summary="把网页的日志窗口发给管理员：写进服务日志，附上是谁发的")
def post_log(req: ClientLog, request: Request) -> dict:
    who = client_of(request, req.client)
    # Size and rate are enforced by the guard through the route's Limit (server/routes.py).
    logs.say(logs.get("client"), Msg("W-LOG-FROMPAGE", who=who.who, ip=who.details.get("ip", ""), app=who.app), req.text)
    return {"ok": True}


@admin.get("/security", access=Access.admin("security.manage"), summary="安全：最近的可疑请求（登录失败、没开放的接口、可疑的路径、请求太频繁……）、各类的次数、暂时封住的来源、现在有几个登录")
def admin_security() -> dict:
    from .. import accounts
    from . import auth

    return {**auth.guards().watch.view(), "online": accounts.online(), "limits": {
        "rate_per_s": auth.RATE_PER_S, "burst": auth.RATE_BURST, "block_after": auth.BLOCK_AFTER,
        "block_window_min": auth.BLOCK_WINDOW_S // 60, "block_min": auth.BLOCK_S // 60, "free": auth.FREE,
        "client_lock": auth.CLIENT_LOCK, "global_lock": auth.GLOBAL_LOCK, "window_min": auth.WINDOW_S // 60}}


class Unblock(BaseModel):
    client: str


@admin.post("/security/unblock", access=Access.admin("security.manage"), summary="解开一个被暂时封住的来源")
def admin_unblock(req: Unblock) -> dict:
    from . import auth

    auth.guards().watch.unblock(req.client)
    return admin_security()


@admin.get("/log", access=Access.admin("logs.view"), summary="服务日志的最后几行（最新的在最后）")
def admin_log(lines: int = 500) -> dict:
    return {"file": str(logs.log_file()), "lines": logs.tail(min(max(lines, 1), 5000))}
