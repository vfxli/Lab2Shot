"""The admin page's 设置, 概览 and 重启服务: the machine's settings (one schema, lab2shot/config.py) with what the
server can tell about each group, the server's state at a glance, and restarting it (server/restart.py). Every page
asks /api/server which run of the server answers and whether a restart is coming."""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .routes import Access, Router
from .. import __version__, feedback, logs
from ..config import SCHEMA, WEBUI_DIST, InvalidSettings, machine_memory_gb, port_problem, settings
from ..errors import Conflict, message_of
from ..engine.resident import available_gb, keep_free_gb
from ..messages import Msg
from ..engine.resident import pool as resident
from ..farm import farm, policy
from . import notice, restart

log = logs.get("admin")


admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # this module's admin routes (app.py includes it)


def _gb(n: float) -> str:
    return f"{n / 1024:.1f} TB" if n >= 1024 else f"{n:.0f} GB"


def address() -> str:
    """Where users reach this server (as it runs now)."""
    s = settings()
    host = socket.gethostname() if s["server.host"] == "0.0.0.0" else "localhost"
    return f"{'https' if s['server.https'] else 'http'}://{host}:{s['server.port']}"


def _huggingface() -> tuple[str, str]:
    """Is this machine logged in to Hugging Face (never the token itself): what it says, and where it comes from."""
    from huggingface_hub import constants, get_token

    if not get_token():
        return "没有登录", "要先申请权限的模型（gated）下载不了：在服务器上运行 hf auth login，或设置环境变量 HF_TOKEN"
    if os.environ.get("HF_TOKEN"):
        return "已登录", "令牌来自环境变量 HF_TOKEN（这里从不显示令牌本身）"
    return "已登录", f"令牌存在 {constants.HF_TOKEN_PATH}（这里从不显示令牌本身）"


def _third_party_free() -> float:
    from .. import config
    from ..installer.preflight import free_gb

    return free_gb(config.THIRD_PARTY_DIR)


def status() -> dict[str, list[dict]]:
    """Per group, what the server can tell that goes with its settings: {label, value, tip}."""
    s = settings()
    work = s.work_dir
    work.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(work)
    names = {g.uuid: f"GPU {g.index} · {g.short_name}" for g in farm().host.snapshot().gpus}
    takers = [names.get(u, u) for u in farm().authorized]
    hf, hf_tip = _huggingface()
    tls = work / "tls" / "ca.pem"
    from ..database import db

    accounts = db().row("SELECT COUNT(*) AS n FROM users WHERE deleted IS NULL")["n"]
    return {
        "people": [{"label": "账号", "value": f"{accounts} 个",
                    "tip": "在「用户」里新建账号、改到期时间和部门；部门表就是这里的「部门」"}],
        "queue": [{"label": "识别到的显卡", "value": f"{len(names)} 张" + ("：" + "、".join(names.values()) if names else ""),
                   "tip": "这台机器上找到的显卡，不管接不接任务。下面几项调度参数按「接任务的」张数算，换机器不用改设置"},
                  {"label": "接任务的显卡", "value": f"{len(takers)} 张" + ("：" + "、".join(takers) if takers else ""),
                   "tip": "哪些显卡接任务，在「概览」或「队列」里用显卡上的开关改；没授权的显卡不接任务"},
                  {"label": "单账号最多占", "value": policy.share_says(len(takers)),
                   "tip": "按上面的「单账号占卡」和接任务的张数算出来的：别人也在排队时，一个账号同时最多占这么多张卡"}],
        "memory": [{"label": "现在可用", "value": f"{available_gb():.0f} GB，共 {machine_memory_gb():.0f} GB",
                    "tip": "这台机器现在可用的内存（所有程序一起算，包括别人的训练和渲染）"}],
        "storage": [{"label": "硬盘剩余", "value": f"{_gb(disk.free / 2**30)}，共 {_gb(disk.total / 2**30)}",
                     "tip": f"工作文件夹 {work} 所在的盘；各类内容占多少，看「硬盘」"}],
        # 「视图」这一组只有设置，没有服务器能另外告诉的现状（点云有多大是每份数据自己的事，不是机器的状态）
        "view": [],
        "network": [{"label": "现在的地址", "value": address(),
                     "tip": "用户在浏览器里打开这个地址，DCC 插件和命令行 --server 也连它；局域网里把 localhost 换成这台机器的名字或 IP"},
                    *([{"label": "证书", "value": str(tls),
                        "tip": "用 HTTPS 时每台用户电脑装一次：浏览器打开 /api/tls/ca.pem 下载，装进系统的「受信任的根证书颁发机构」"}] if tls.exists() else [])],
        "install": [{"label": "扩展包硬盘", "value": _gb(_third_party_free()),
                     "tip": "扩展包所在的盘还剩多少：新环境建在旧环境旁边，自检通过才换上，旧的留着可以回退，所以安装时要多留一份空间"}],
        "env": [{"label": "Hugging Face", "value": hf, "tip": hf_tip},
                {"label": "版本", "value": f"Lab2Shot {__version__} · Python {platform.python_version()}",
                 "tip": f"服务的 Python：{sys.executable}"}],
    }


def view() -> dict:
    return {**settings().describe(), "status": status()}


@admin.get("/settings", access=Access.admin("settings.edit"), summary="设置：每项的值、默认、范围、说明、改了要不要重启；每组附带服务器的现状（只读）")
def admin_settings() -> dict:
    return view()


class SettingsChange(BaseModel):
    values: dict[str, object]  # key -> new value


@admin.put("/settings", access=Access.admin("settings.edit"), summary="保存设置（全部检查通过才保存）：马上生效的立刻用上，要重启的等重启；出错时 errors 按项说明原因")
def admin_save_settings(req: SettingsChange):
    s = settings()
    before = {k: s.value(k) for k in req.values if k in SCHEMA}
    try:
        changed = s.save(req.values)
    except InvalidSettings as exc:
        return JSONResponse({"detail": str(exc), "code": exc.code, "errors": {k: m.text for k, m in exc.problems.items()},
                             "codes": {k: m.code for k, m in exc.problems.items()}}, status_code=400)
    for k in changed:
        spec = SCHEMA[k]
        logs.say(log, Msg("I-SETTINGS-CHANGEDRESTART" if spec.restart else "I-SETTINGS-CHANGED", label=spec.label, before=str(before.get(k)), after=str(s.value(k))))
    resident().tidy()  # models no longer kept end now
    farm().wake()  # jobs waiting for memory look again
    return view()


@admin.get("/overview", access=Access.admin("server.view"), summary="概览：内存、硬盘、常驻模型、服务本身（版本、地址、启动时间、进程号），和等重启生效的设置")
def admin_overview() -> dict:
    s = settings()
    disk = shutil.disk_usage(s.work_dir) if s.work_dir.exists() else None
    procs = resident().view()["processes"]
    return {
        "memory": {"total_gb": round(machine_memory_gb(), 1), "available_gb": round(available_gb(), 1),
                   "keep_free_gb": keep_free_gb()},
        "disk": {"path": str(s.work_dir), "total": disk.total if disk else 0, "free": disk.free if disk else 0},
        "resident": {"processes": len(procs), "vram_mb": sum(p["vram_mb"] for p in procs)},
        "server": {"version": __version__, "boot": restart.BOOT, "started": restart.STARTED, "pid": os.getpid(),
                   "address": address(), "command": " ".join(sys.orig_argv)},
        "pending": [SCHEMA[k].label for k in s.pending()],
        "feedback_new": feedback.new_count(),
    }


# ------------------------------------------------------------------ restarting (server/restart.py)

public = Router(prefix="/api", tags=["设置"])


@public.get("/server", access=Access.open("服务的这次启动和重启：页面据此知道服务在不在、要不要刷新"), summary="服务本身：这次启动的编号（重启后变）、版本、界面的版本、正在进行的重启（没有是 null），和管理员通知上次修改的时间（变了就重新读 /api/notice）")
def server_state() -> dict:
    return server_now()


def server_now() -> dict:
    """服务本身这一份：这次启动的编号、版本、界面的版本、正在进行的重启、通知改动时间。

    登录着的页面不为它单独发请求：`/api/load`（server/farm.py load）把这一份捎上，页面只剩一条轮询。
    一次往返的字节大半是请求头和会话 cookie，省流量只能靠少问。两条路由共用这个函数。"""
    index = WEBUI_DIST / "index.html"
    # 本机代理的两个数（管理员的设置，只有浏览器用）也捎在这一份里，页面不为它们另发请求
    s = settings()
    return {"boot": restart.BOOT, "started": restart.STARTED, "version": __version__,
            "ui": index.stat().st_mtime if index.exists() else 0, "restart": restart.restarter().view(),
            "notice": notice.changed_at(),
            "view": {"local_px": int(s["view.local_px"]), "local_cache_gb": int(s["view.local_cache_gb"])}}


class RestartRequest(BaseModel):
    mode: str  # drain: after the jobs running; now: stop them


@admin.post("/restart", access=Access.admin("server.restart"), summary="重启服务：drain 等计算中的任务算完（不再开始新任务），now 立即停下它们；排队的任务重启后接着排")
def admin_restart(req: RestartRequest) -> dict:
    s = settings()
    if s.value("server.port") != s["server.port"] and (problem := port_problem(int(s.value("server.port")))):
        raise Conflict(Msg("E-SETTINGS-RESTARTPORT", reason=problem))
    try:
        restart.restarter().request(req.mode)
    except RuntimeError as exc:
        raise Conflict(message_of(exc)) from exc
    return server_state()


@admin.post("/restart/cancel", access=Access.admin("server.restart"), summary="不重启了（还在等计算中的任务时）：队列照常继续")
def admin_restart_cancel() -> dict:
    restart.restarter().call_off()
    logs.say(log, Msg("I-RESTART-CALLEDOFF"))
    return server_state()
