"""The admin page's settings pages, 概览 and 重启服务: the machine's settings (one schema, lab2shot/config.py, laid out
on the pages of config.PAGES) with what the server can tell about each group, the server's state at a glance, and
restarting it (server/restart.py). Every page asks /api/server which run of the server answers and whether a restart
is coming."""

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
from .. import __version__, accounts, feedback, logs, periods, process, registration, roles, traffic
from ..config import PAGE_OF, SCHEMA, WEBUI_DIST, InvalidSettings, machine_memory_gb, port_problem, settings
from ..errors import Conflict, message_of
from ..engine.resident import available_gb, keep_free_gb
from ..messages import Msg
from ..traffic import SCOPE_USER
from ..nodes import tags  # noqa: F401  (provides the options of 注册可用模型类别: config.provide_choices)
from ..view import proxy  # noqa: F401  (provides the options of 视图代理尺寸: config.provide_choices)
from ..engine.resident import pool as resident
from ..farm import farm, usage
from . import auth, available, notice, restart
from .access import audit

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


def _source(request: Request) -> dict:
    """What this server sees of where users come from, judged by this very request (the administrator's page):
    their real address, a trusted proxy's word for it, or nothing at all (a tunnel that only passes the connection
    through: every request is this machine's loopback). Said in the 网络 group, beside 可信代理."""
    src = auth.client_source(request)
    if src.via and src.apart:
        value, tip = f"看得到：{src.ip}（可信代理 {src.via} 转告的）", "按 IP 的限制（注册、输错邀请码、请求频率）按代理转告的真实地址算"
    elif src.via:
        value, tip = (f"看不到用户的真实 IP：可信代理 {src.via} 没有转告地址",
                      "这个请求从可信代理来，却没带 X-Forwarded-For / X-Real-IP：代理要设成转发这个头（nginx：proxy_set_header "
                      "X-Forwarded-For $proxy_add_x_forwarded_for;），否则按 IP 的限制不起作用")
    elif src.apart:
        value, tip = f"看得到：{src.ip}（连接本身的地址）", "用户直接连到这台服务器（局域网），按 IP 的限制按连接的地址算"
    else:
        value, tip = (f"看不到用户的真实 IP：来源都是本机 {src.ip}，按 IP 的限制不起作用",
                      "请求是这台机器自己转进来的（本机浏览器，或者 frp 这类内网穿透只转发 TCP 连接到这里的端口）：看不出谁是谁，"
                      "注册按 IP 和网段的限制、输错邀请码按来源的计数就不算，免得把所有人当成一个人一起挡住；全站注册上限、邀请码、"
                      "工作量证明照常。要看到真实 IP：让云服务器上的 nginx / Caddy / frp https2http 解开 HTTPS、用 HTTP 转发过来"
                      "并带上 X-Forwarded-For，再把它连过来的地址填进「可信代理」")
    if src.ignored:
        tip += f"\n这个请求带着 X-Forwarded-For 之类的转发头，但它来自的 {request.client.host if request.client else '?'} 不在「可信代理」里，这些头都没有采用"
    return {"label": "用户的 IP", "value": value, "tip": tip}


def _registering() -> dict:
    """The 注册 group's state: how many registered lately against the limits, and whether that paused registering."""
    c = registration.counts()
    state = "没开放" if not c["open"] else "已自动暂停" if c["paused"] else "开放中"
    return {"label": "注册情况", "value": f"{state} · 最近一小时 {c['hour']} / {c['per_hour']} 个，最近一天 {c['day']} / {c['per_day']} 个",
            "tip": "全站自己注册的账号数和两个上限；到了上限注册自动暂停，过了那一小时 / 那一天自己恢复。邀请码和谁用它注册了在这一页下面看"}


def status(request: Request) -> dict[str, list[dict]]:
    """Per group, what the server can tell that goes with its settings: {label, value, tip}. `request`: the
    administrator's own, for what this server sees of where users come from (_source)."""
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
        "accounts": [{"label": "账号", "value": f"{accounts} 个",
                      "tip": "在「用户」里新建账号、改到期时间和环节；环节表就是这里的「环节」"}],
        "register": [_registering(), _source(request)],
        "gpu": [{"label": "识别到的显卡", "value": f"{len(names)} 张" + ("：" + "、".join(names.values()) if names else ""),
                 "tip": "这台机器上找到的显卡，不管接不接任务"},
                {"label": "接任务的显卡", "value": f"{len(takers)} 张" + ("：" + "、".join(takers) if takers else ""),
                 "tip": "哪些显卡接任务，在「概览」或「队列」里用显卡上的开关改；没授权的显卡不接任务。每张接任务的卡同时算一个显卡节点"}],
        "memory": [{"label": "现在可用", "value": f"{available_gb():.0f} GB，共 {machine_memory_gb():.0f} GB",
                    "tip": "这台机器现在可用的内存（所有程序一起算，包括别人的训练和渲染）"}],
        "storage": [{"label": "硬盘剩余", "value": f"{_gb(disk.free / 2**30)}，共 {_gb(disk.total / 2**30)}",
                     "tip": f"工作文件夹 {work} 所在的盘；各类内容占多少，看「硬盘」"}],
        # 「视图」这一组只有设置，没有服务器能另外告诉的现状（点云有多大是每份数据自己的事，不是机器的状态）
        "view": [],
        "network": [_source(request),
                    {"label": "现在的地址", "value": address(),
                     "tip": "用户在浏览器里打开这个地址，DCC 插件和命令行 --server 也连它；局域网里把 localhost 换成这台机器的名字或 IP"},
                    *([{"label": "证书", "value": str(tls),
                        "tip": "用 HTTPS 时每台用户电脑装一次：浏览器打开 /api/tls/ca.pem 下载，装进系统的「受信任的根证书颁发机构」"}]
                      # only while this run serves HTTPS: a certificate left from an earlier run is no one's to install
                      if s["server.https"] and tls.exists() else [])],
        "install": [{"label": "Hugging Face", "value": hf, "tip": hf_tip},
                    {"label": "扩展包硬盘", "value": _gb(_third_party_free()),
                     "tip": "扩展包所在的盘还剩多少：新环境建在旧环境旁边，自检通过才换上，旧的留着可以回退，所以安装时要多留一份空间"}],
        "env": [{"label": "版本", "value": f"Lab2Shot {__version__} · Python {platform.python_version()}",
                 "tip": f"服务的 Python：{sys.executable}"}],
    }


def _locked(s, key: str) -> Msg | None:
    """Why this session may not change setting `key` (None: it may): the capability it needs besides settings.edit
    (roles.setting_needs), when the session does not hold it."""
    need = roles.setting_needs(key)
    return None if need is None or s.can(need) else Msg("E-SETTINGS-NEEDS", setting=SCHEMA[key].label if key in SCHEMA else key,
                                                       label=roles.CAPABILITIES[need].label, roles=roles.holders(need))


def view(request: Request) -> dict:
    """The settings as config.describe gives them, each with `locked`: why this login may not change it ("" it may)."""
    s, described = auth.session(request), settings().describe()
    for item in described["settings"]:
        item["locked"] = why.text if (why := _locked(s, item["key"])) else ""
    return {**described, "status": status(request)}


@admin.get("/settings", access=Access.admin("settings.edit"), summary="设置：每项的值、默认、范围、说明、改了要不要重启；每组附带服务器的现状（只读）")
def admin_settings(request: Request) -> dict:
    return view(request)


class SettingsChange(BaseModel):
    values: dict[str, object]  # key -> new value


@admin.put("/settings", access=Access.admin("settings.edit"), summary="保存设置（全部检查通过才保存）：马上生效的立刻用上，要重启的等重启；出错时 errors 按项说明原因")
def admin_save_settings(req: SettingsChange, request: Request):
    session = auth.session(request)
    if locked := {k: m for k in req.values if (m := _locked(session, k))}:
        for k in locked:
            audit(Msg("W-AUDIT-REFUSED", who=session.user.label, role=roles.label(session.user.role),
                      what=roles.CAPABILITIES[roles.setting_needs(k)].what, method="PUT", path=str(request.url.path),
                      code="E-SETTINGS-NEEDS"), session=session, method="PUT", path=str(request.url.path))
        return JSONResponse({"detail": Msg("E-SETTINGS-INVALID", problems=list(locked.values())).text, "code": "E-SETTINGS-NEEDS",
                             "errors": {k: m.text for k, m in locked.items()}, "codes": {k: m.code for k, m in locked.items()}},
                            status_code=403)
    s = settings()
    before = {k: s.value(k) for k in req.values if k in SCHEMA}
    try:
        changed = s.save(req.values)
    except InvalidSettings as exc:
        return JSONResponse({"detail": str(exc), "code": exc.code, "errors": {k: m.text for k, m in exc.problems.items()},
                             "codes": {k: m.code for k, m in exc.problems.items()}}, status_code=400)
    for said in process.apply_changed(changed):  # 「保留核心数」「复用内存上限」 hold for this server at once
        logs.say(log, said)
    for k in changed:
        spec = SCHEMA[k]
        logs.say(log, Msg("I-SETTINGS-CHANGEDRESTART" if spec.restart else "I-SETTINGS-CHANGED", label=spec.label, before=str(before.get(k)), after=str(s.value(k))))
    resident().tidy()  # end the resident models the saved settings do not keep
    farm().wake()  # jobs waiting for memory look again
    return view(request)


@admin.get("/overview", access=Access.admin("server.view", hides={"recent.feedback": ("feedback_new",)}), summary="概览：内存、硬盘、常驻模型、服务本身（版本、地址、启动时间、进程号），和等重启生效的设置；看得了用户反馈的还有几条新反馈")
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
        # each with the settings page it is on, for the page's mark in the side list and the overview's link
        "pending": [{"label": SCHEMA[k].label, "page": PAGE_OF[SCHEMA[k].group]} for k in s.pending()],
        "feedback_new": feedback.new_count(),
        "registering": registration.counts(),  # 自行注册: open or not, lately against the limits, paused or not
    }


@admin.get("/overview/recent", access=Access.admin("server.view", hides={f"recent.{g}": (g,) for g in available.RECENT}),
           summary="概览的「今天和最近」（按服务器的本地时间：今日从零点、近 7 天含今天、本周从周一、本月从 1 号）："
                   "访问（在线、登录人数、登录次数、登录失败）、注册（新账号，其中自己注册的）、任务（提交的，按结果）、"
                   "流量（发出的字节）、反馈（新收的、未解决的）；每组只给看得了它对应那一页的登录，带那一页的 section")
def admin_overview_recent() -> dict:
    p = periods.now()
    found = {"access": accounts.logins_in(p), "accounts": registration.new_accounts(p), "tasks": usage.jobs_in(p),
             "traffic": traffic.totals(p), "feedback": feedback.counts_in(p)}
    return {"days": {k: p.first(k) for k in ("today", "days7", "week", "month")},
            **{g: {"section": available.RECENT[g], **v} for g, v in found.items()}}


# ------------------------------------------------------------------ restarting (server/restart.py)

public = Router(prefix="/api", tags=["设置"])


@public.get("/server", access=Access.open("服务的这次启动和重启：页面据此知道服务在不在、要不要刷新"), summary="服务本身：这次启动的编号（重启后变）、版本、界面的版本、正在进行的重启（没有是 null），和管理员通知上次修改的时间（变了就重新读 /api/notice），以及这个浏览器现在登录的账号（没登录是 null）")
def server_now(request: Request) -> dict:
    """服务本身这一份：这次启动的编号、版本、界面的版本、正在进行的重启、通知改动时间，和问的这一次登录的账号。

    登录着的页面不为它单独发请求：`/api/load`（server/farm.py load）把这一份捎上，页面只剩一条轮询。
    一次往返的字节大半是请求头和会话 cookie，省流量只能靠少问。两条路由共用这个函数。

    `account`：这个浏览器现在登录的账号（没登录是 null），取自 access.py Guard 记在请求上的归属，不另查一次。
    同一个浏览器的登录是各个标签页共用的：别的窗口换了账号，这一页的请求就已经是另一个账号的了；页面据此
    发现自己不再属于这个登录，重新打开（webui/src/platform/http.ts sawAccount）。"""
    index = WEBUI_DIST / "index.html"
    # 本机代理的两个数（管理员的设置，只有浏览器用）也捎在这一份里，页面不为它们另发请求
    s = settings()
    return {"boot": restart.BOOT, "started": restart.STARTED, "version": __version__,
            "ui": index.stat().st_mtime if index.exists() else 0, "restart": restart.restarter().view(),
            "notice": notice.changed_at(), "account": request.scope.get(SCOPE_USER) or None,
            "view": {"local_px": int(s["view.local_px"]), "local_cache_gb": int(s["view.local_cache_gb"])}}


class RestartRequest(BaseModel):
    mode: str  # drain: after the jobs running; now: stop them


@admin.post("/restart", access=Access.admin("server.restart"), summary="重启服务：drain 等计算中的任务算完（不再开始新任务），now 立即停下它们；排队的任务重启后接着排")
def admin_restart(req: RestartRequest, request: Request) -> dict:
    s = settings()
    if s.value("server.port") != s["server.port"] and (problem := port_problem(int(s.value("server.port")))):
        raise Conflict(Msg("E-SETTINGS-RESTARTPORT", reason=problem))
    try:
        restart.restarter().request(req.mode)
    except RuntimeError as exc:
        raise Conflict(message_of(exc)) from exc
    return server_now(request)


@admin.post("/stop", access=Access.admin("server.restart"), summary="停止服务（配置菜单的「停止服务」与一键更新用，只有本机令牌会调它）：drain 等计算中的任务算完（不再开始新任务），now 立即停下它们；排队的任务留给下一次启动的服务接着排")
def admin_stop(req: RestartRequest, request: Request) -> dict:
    try:
        restart.restarter().request(req.mode, then="stop")
    except RuntimeError as exc:
        raise Conflict(message_of(exc)) from exc
    return server_now(request)


@admin.post("/restart/cancel", access=Access.admin("server.restart"), summary="不重启了（还在等计算中的任务时）：队列照常继续")
def admin_restart_cancel(request: Request) -> dict:
    restart.restarter().call_off()
    logs.say(log, Msg("I-RESTART-CALLEDOFF"))
    return server_now(request)
