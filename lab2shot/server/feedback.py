"""HTTP routes for 提交反馈 (storage is in lab2shot/feedback.py).

Each page submits the user's text together with the diagnostics it collected; the server appends its own
(environment, the user's recent jobs, logs). The 用户反馈 section of the admin page lists, shows, downloads, marks
and deletes submissions."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from functools import cache

from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .routes import Access, Limit, Router
from .. import __version__, feedback, logs
from ..config import ROOT, machine_memory_gb
from ..database import db
from ..engine.resident import available_gb
from ..engine.resident import pool as resident
from ..errors import Invalid, NotFound
from ..messages import Msg
from ..farm import farm
from . import auth, restart
from .farm import client_of

admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # admin routes of this module, included by app.py
router = Router(prefix="/api", tags=["用户反馈"])
log = logs.get("admin")

# Upper bound for the whole request: diagnostics, screenshots (base64 adds one third) and headroom for other fields.
MAX_REQUEST = feedback.MAX_BUNDLE + feedback.MAX_IMAGES * feedback.MAX_IMAGE * 4 // 3 + (256 << 10)


@cache
def revision() -> str:
    """Return the git revision of the running code (with a dirty marker), or 未知 when not running from a clone."""
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True, text=True,
                             timeout=5)
        return out.stdout.strip() or "未知"
    except (OSError, subprocess.SubprocessError):
        return "未知"


def environment() -> dict:
    """Return the current server environment: versions, host, GPUs, memory, queue states and resident models."""
    jobs = farm().jobs_now()
    states: dict[str, int] = {}
    for j in jobs:
        states[j.state] = states.get(j.state, 0) + 1
    return {
        "version": __version__, "revision": revision(), "python": platform.python_version(),
        "platform": platform.platform(), "hostname": platform.node(), "pid": os.getpid(), "boot": restart.BOOT,
        "started": restart.STARTED, "uptime_s": round(time.time() - restart.STARTED), "executable": sys.executable,
        "database_version": db().version, "time": time.time(),
        "memory": {"total_gb": round(machine_memory_gb(), 1), "available_gb": round(available_gb(), 1)},
        "gpus": [{**g.describe(), "takes_jobs": g.uuid in farm().authorized} for g in farm().host.snapshot().gpus],
        "queue": states,
        "resident": [{k: p.get(k) for k in ("extension", "gpu", "state", "models", "vram_mb", "ram_mb", "idle_s")}
                     for p in resident().view()["processes"]],
    }


# ------------------------------------------------------------------ users


@router.post("/feedback", access=Access.user("提交反馈：只能写，不能读", limit=Limit(body=None)), summary="提交反馈：写的问题、类别、截图（最多 3 张，每张 5 MB 以内）和网页收集的诊断资料；服务器补上自己的环境、这个用户最近的任务和日志")
async def submit(request: Request) -> dict:
    size = int(request.headers.get("content-length") or 0)
    if size > MAX_REQUEST:
        raise Invalid(Msg("E-FEEDBACK-REQUESTBIG", size=size / 2**20, max=MAX_REQUEST >> 20))
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_REQUEST:
            raise Invalid(Msg("E-FEEDBACK-REQUESTLIMIT", max=MAX_REQUEST >> 20))
    try:
        data = json.loads(body)
    except ValueError:
        raise Invalid(Msg("E-FEEDBACK-UNREADABLE")) from None
    if not isinstance(data, dict):
        raise Invalid(Msg("E-FEEDBACK-UNREADABLE"))
    declared = data.get("client") if isinstance(data.get("client"), dict) else {}
    page = data.get("diagnostics") if isinstance(data.get("diagnostics"), dict) else {}
    images = [i for i in data.get("images") or [] if isinstance(i, dict)]

    client = client_of(request, declared)

    def keep() -> dict:
        return feedback.submit(str(data.get("text") or ""), str(data.get("category") or ""), client.user, client.full(),
                               page, images, environment())

    row = await run_in_threadpool(keep)
    return {"id": row["id"], "at": row["at"]}


class JobsRequest(BaseModel):
    jobs: list[str] = []


@router.post("/feedback/jobs", access=Access.user("提交反馈前：看自己最近的任务"), summary="提交反馈前先看：服务器会附上的自己最近的任务（状态、出错信息），不含日志全文")
def preview_jobs(req: JobsRequest, request: Request) -> dict:
    return {"jobs": [{k: j[k] for k in ("id", "title", "state", "submitted", "error")}
                     | {"error_log": bool(j["error_log"]), "server_log": len(j["server_log"])}
                     for j in feedback.jobs_of(auth.me(request).id, req.jobs)],
            "environment": {k: v for k, v in environment().items() if k in ("version", "revision", "python", "platform")}}


@router.get("/feedback/mine", access=Access.user("我的反馈：只有自己的，和管理员的回复"), summary="我的反馈：自己提交过的反馈（状态、管理员的回复），和几条有没看过的新回复或状态变化；不含诊断资料和内部备注，也从不含别人的")
def mine(request: Request) -> dict:
    return feedback.mine(auth.me(request).id)


@router.post("/feedback/mine/read", access=Access.user("我的反馈：看过了"), summary="看过了我的反馈：回复和状态变化不再算没看过")
def mine_read(request: Request) -> dict:
    return feedback.mark_read(auth.me(request).id)


# ------------------------------------------------------------------ admin


@admin.get("/feedback", access=Access.admin("feedback.reply"), summary="用户反馈：按时间倒序，可按状态（new/seen/solved）、时间段、账号的名字筛选；和各状态的条数")
def listing(status: str = "", since: float | None = None, until: float | None = None, person: str = "") -> dict:
    return feedback.listing(status or None, since, until, person)


@admin.get("/feedback/{fid}", access=Access.admin("feedback.reply"), summary="一条反馈的全部：写的问题、截图、诊断资料（节点图、日志、错误、任务和出错日志、环境）")
def detail(fid: str) -> dict:
    return feedback.detail(fid)


@admin.get("/feedback/{fid}/files/{name}", access=Access.admin("feedback.reply"), summary="反馈附的截图")
def image(fid: str, name: str) -> FileResponse:
    if name not in feedback.get(fid)["images"]:
        raise NotFound(Msg("E-FEEDBACK-NOSHOT"))
    return FileResponse(feedback.folder(fid) / name)


@admin.get("/feedback/{fid}/download", access=Access.admin("feedback.reply"), summary="把整条反馈下载成一个 zip：feedback.json（问题、谁、状态、诊断资料）和截图")
def download(fid: str) -> Response:
    row = feedback.get(fid)
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(row["at"]))
    return Response(feedback.archive(fid), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="feedback-{stamp}-{fid}.zip"'})


class AnswerRequest(BaseModel):
    status: str
    reply: str = ""  # visible to the submitting user
    note: str = ""  # visible to administrators only


@admin.put("/feedback/{fid}", access=Access.admin("feedback.reply"), summary="回应一条反馈：状态（新 / 已看 / 已解决）和给用户的回复（用户在「我的反馈」里看到），和只有管理员看得到的内部备注")
def answer(fid: str, req: AnswerRequest, request: Request) -> dict:
    return feedback.answer(fid, req.status, req.reply, req.note, auth.me(request).username)


@admin.delete("/feedback/{fid}", access=Access.admin("feedback.delete"), summary="删除一条反馈（连同截图和诊断资料；数据库备份里的留到备份轮换掉）")
def delete(fid: str, request: Request) -> dict:
    feedback.delete(fid, auth.me(request).username)
    return {"ok": True}
