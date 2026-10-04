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

from .routes import Access, Body, Router
from .. import __version__, i18n, accounts, logs, periods, process, roles, traffic
from ..site import feedback, registration
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
from .access import audit, manages

log = logs.get("admin")


admin = Router(prefix="/api/admin", tags=["admin"])  # this module's admin routes (app.py includes it)


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
        return i18n.t("settings.state.hf.out"), i18n.t("settings.state.hf.out_tip")
    if os.environ.get("HF_TOKEN"):
        return i18n.t("settings.state.hf.in"), i18n.t("settings.state.hf.env_tip")
    return i18n.t("settings.state.hf.in"), i18n.t("settings.state.hf.file_tip", path=constants.HF_TOKEN_PATH)


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
        value, tip = i18n.t("settings.state.ip.via", ip=src.ip, via=src.via), i18n.t("settings.state.ip.via_tip")
    elif src.via:
        value, tip = i18n.t("settings.state.ip.via_none", via=src.via), i18n.t("settings.state.ip.via_none_tip")
    elif src.apart:
        value, tip = i18n.t("settings.state.ip.direct", ip=src.ip), i18n.t("settings.state.ip.direct_tip")
    else:
        value, tip = i18n.t("settings.state.ip.local", ip=src.ip), i18n.t("settings.state.ip.local_tip")
    if src.ignored:
        tip += "\n" + i18n.t("settings.state.ip.ignored", host=request.client.host if request.client else "?")
    return {"label": i18n.t("settings.state.ip.label"), "value": value, "tip": tip}


def _registering() -> dict:
    """The registration group's state: how many registered lately against the limits, and whether that paused registering."""
    c = registration.counts()
    state = i18n.t("settings.state.register.closed") if not c["open"] else \
        i18n.t("settings.state.register.paused") if c["paused"] else i18n.t("settings.state.register.open")
    return {"label": i18n.t("settings.state.register.label"),
            "value": i18n.t("settings.state.register.value", state=state, hour=c["hour"], per_hour=c["per_hour"],
                            day=c["day"], per_day=c["per_day"]),
            "tip": i18n.t("settings.state.register.tip")}


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
    sep = i18n.separator()
    return {
        "accounts": [{"label": i18n.t("settings.state.accounts.label"), "value": i18n.t("settings.state.accounts.value", n=accounts),
                      "tip": i18n.t("settings.state.accounts.tip")}],
        "register": [_registering(), _source(request)],
        "gpu": [{"label": i18n.t("settings.state.gpus.label"),
                 "value": i18n.t("settings.state.gpus.value", n=len(names), names=sep.join(names.values()) or "-"),
                 "tip": i18n.t("settings.state.gpus.tip")},
                {"label": i18n.t("settings.state.takers.label"),
                 "value": i18n.t("settings.state.gpus.value", n=len(takers), names=sep.join(takers) or "-"),
                 "tip": i18n.t("settings.state.takers.tip")}],
        "memory": [{"label": i18n.t("settings.state.memory.label"),
                    "value": i18n.t("settings.state.memory.value", free=f"{available_gb():.0f}", total=f"{machine_memory_gb():.0f}"),
                    "tip": i18n.t("settings.state.memory.tip")}],
        "storage": [{"label": i18n.t("settings.state.disk.label"),
                     "value": i18n.t("settings.state.disk.value", free=_gb(disk.free / 2**30), total=_gb(disk.total / 2**30)),
                     "tip": i18n.t("settings.state.disk.tip", work=work)}],
        # the viewer group has settings only: nothing the server can add (how large a point cloud is is each data's own)
        "view": [],
        "network": [_source(request),
                    {"label": i18n.t("settings.state.address.label"), "value": address(),
                     "tip": i18n.t("settings.state.address.tip")},
                    *([{"label": i18n.t("settings.state.certificate.label"), "value": str(tls),
                        "tip": i18n.t("settings.state.certificate.tip")}]
                      # only while this run serves HTTPS: a certificate left from an earlier run is no one's to install
                      if s["server.https"] and tls.exists() else [])],
        "install": [{"label": "Hugging Face", "value": hf, "tip": hf_tip},
                    {"label": i18n.t("settings.state.extensions_disk.label"), "value": _gb(_third_party_free()),
                     "tip": i18n.t("settings.state.extensions_disk.tip")}],
        "env": [{"label": i18n.t("settings.state.version.label"), "value": f"Lab2Shot {__version__} · Python {platform.python_version()}",
                 "tip": i18n.t("settings.state.version.tip", python=sys.executable)}],
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


@admin.get("/settings", access=Access.admin("settings.edit"), summary="Settings: each one's value, default, range, help and whether a change needs a restart; each group with the server's current state (read-only)")
def admin_settings(request: Request) -> dict:
    return view(request)


class SettingsChange(Body):
    values: dict[str, object]  # key -> new value


@admin.put("/settings", access=Access.admin("settings.edit"), summary="Save settings (only when all pass their checks): those that apply at once are used at once, the others after a restart; on error, errors says why per setting")
def admin_save_settings(req: SettingsChange, request: Request):
    session = auth.session(request)
    if locked := {k: m for k in req.values if (m := _locked(session, k))}:
        lacking = dict.fromkeys(roles.CAPABILITIES[roles.setting_needs(k)].what for k in locked)  # one request, one row
        audit(Msg("W-AUDIT-REFUSED", who=session.user.label, role=roles.word(session.user.role), what=i18n.Both.of(lambda: i18n.separator().join(lacking)),
                  method="PUT", path=str(request.url.path), code="E-SETTINGS-NEEDS"),
              session=session, method="PUT", path=str(request.url.path))
        raise InvalidSettings(locked, 403)  # answered field by field (errors.FieldErrors), as a value that does not fit
    s = settings()
    before = {k: s.value(k) for k in req.values if k in SCHEMA}
    changed = s.save(req.values)  # InvalidSettings: answered field by field (server/app.py, errors.FieldErrors)
    for said in process.apply_changed(changed):  # 「保留核心数」「复用内存上限」 hold for this server at once
        logs.say(log, said)
    for k in changed:
        spec = SCHEMA[k]
        logs.say(log, Msg("I-SETTINGS-CHANGEDRESTART" if spec.restart else "I-SETTINGS-CHANGED", label=i18n.Word(f"setting.{k}.label"), before=str(before.get(k)), after=str(s.value(k))))
    resident().tidy()  # end the resident models the saved settings do not keep
    farm().wake()  # jobs waiting for memory look again
    return view(request)


@admin.get("/overview", access=Access.admin("server.view", hides={"recent.feedback": ("feedback_new",)}), summary="Overview: memory, disk, resident models, the server itself (version, address, start time, process id), and settings waiting for a restart; for a login that may read feedback, how many new reports there are")
def admin_overview() -> dict:
    from ..farm.policy import space

    s = settings()
    disk = shutil.disk_usage(s.work_dir) if s.work_dir.exists() else None
    procs = resident().view()["processes"]
    return {
        "memory": {"total_gb": round(machine_memory_gb(), 1), "available_gb": round(available_gb(), 1),
                   "keep_free_gb": keep_free_gb()},
        "disk": {"path": str(s.work_dir), "total": disk.total if disk else 0, "free": disk.free if disk else 0},
        # the data disk against 暂停新计算的剩余空间: below it every account's new computing pauses (farm/policy.py)
        "space": {**space(), "path": str(s.data_dir)},
        "resident": {"processes": len(procs), "vram_mb": sum(p["vram_mb"] for p in procs)},
        "server": {"version": __version__, "boot": restart.BOOT, "started": restart.STARTED, "pid": os.getpid(),
                   "address": address(), "command": " ".join(sys.orig_argv)},
        # each with the settings page it is on, for the page's mark in the side list and the overview's link
        "pending": [{"label": SCHEMA[k].label, "page": PAGE_OF[SCHEMA[k].group]} for k in s.pending()],
        "feedback_new": feedback.new_count(),
        "registering": registration.counts(),  # 自行注册: open or not, lately against the limits, paused or not
    }


@admin.get("/overview/recent", access=Access.admin("server.view", hides={f"recent.{g}": (g,) for g in available.RECENT}),
           summary="The overview's Today and Recent (server local time: today from midnight, the last 7 days including today, "
                   "this week from Monday, this month from the 1st): access (online, users logged in, logins, failed logins), "
                   "registration (new accounts, of which self-registered), jobs (submitted, by outcome), traffic (bytes sent), "
                   "feedback (new, unresolved); each group only for a login that may open its page, with that page's section")
def admin_overview_recent(request: Request) -> dict:
    p = periods.now()
    s = auth.session(request)
    found = {"access": accounts.logins_in(p, lambda owner, role: manages(s, owner, role)), "accounts": registration.new_accounts(p),
             "tasks": usage.jobs_in(p),
             "traffic": traffic.totals(p), "feedback": feedback.counts_in(p)}
    return {"days": {k: p.first(k) for k in ("today", "days7", "week", "month")},
            **{g: {"section": available.RECENT[g], **v} for g, v in found.items()}}


# ------------------------------------------------------------------ restarting (server/restart.py)

public = Router(prefix="/api", tags=["settings"])


@public.get("/server", access=Access.open("This run of the server and restarts: the page learns whether the server is up and whether to reload"), summary="The server itself: this run's id (changes on restart), version, the web page's version, a restart in progress (null: none), when the administrator notice last changed (read /api/notice again when it does), and the account this browser is logged in as (null: none); before logging in only the run id, the page version and where a restart reconnects")
def server_now(request: Request) -> dict:
    """服务本身这一份：这次启动的编号、版本、界面的版本、正在进行的重启、通知改动时间，和问的这一次登录的账号。

    登录着的页面不为它单独发请求：`/api/load`（server/farm.py load）把这一份捎上，页面只剩一条轮询。
    一次往返的字节大半是请求头和会话 cookie，省流量只能靠少问。两条路由共用这个函数。

    `account`：这个浏览器现在登录的账号（没登录是 null），取自 access.py Guard 记在请求上的归属，不另查一次。
    同一个浏览器的登录是各个标签页共用的：别的窗口换了账号，这一页的请求就已经是另一个账号的了；页面据此
    发现自己不再属于这个登录，重新打开（webui/src/platform/http.ts sawAccount）。"""
    index = WEBUI_DIST / "index.html"
    ui = index.stat().st_mtime if index.exists() else 0
    going = restart.restarter().view()
    account = request.scope.get(SCOPE_USER) or None
    if account is None:
        # 没登录的页面（登录页）只需知道服务换没换、界面换没换、重启到哪里接着连：版本、计算中的任务数、后台任务的标题、
        # 通知和各项设置都不给不认识的人
        where = {k: going[k] for k in ("state", "mode", "since", "port", "https")} if going else None
        return {"boot": restart.BOOT, "ui": ui, "restart": where, "account": None}
    # 本机代理的两个数（管理员的设置，只有浏览器用）和「任务保留天数」（浏览器里暂停着的上传按同一个天数清，
    # webui/src/transfer/uploads.ts purgeStale）也捎在这一份里，页面不为它们另发请求
    s = settings()
    return {"boot": restart.BOOT, "started": restart.STARTED, "version": __version__, "ui": ui, "restart": going,
            "notice": notice.changed_at(), "account": account,
            "view": {"local_px": int(s["view.local_px"]), "local_cache_gb": int(s["view.local_cache_gb"])},
            "tasks": {"keep_days": int(s["tasks.keep_days"])}}


class RestartRequest(Body):
    mode: str  # drain: after the jobs running; now: stop them


@admin.post("/restart", access=Access.admin("server.restart"), summary="Restart the server: drain waits for running jobs to finish (starting no new ones), now stops them at once; queued jobs stay queued after the restart")
def admin_restart(req: RestartRequest, request: Request) -> dict:
    s = settings()
    if s.value("server.port") != s["server.port"] and (problem := port_problem(int(s.value("server.port")))):
        raise Conflict(Msg("E-SETTINGS-RESTARTPORT", reason=problem))
    try:
        restart.restarter().request(req.mode)
    except RuntimeError as exc:
        raise Conflict(message_of(exc)) from exc
    return server_now(request)


@admin.post("/stop", access=Access.admin("server.restart", local=True), summary="Stop the server (only the command line on this server, with the local token: the setup menu's Stop Server and the one-step update): drain waits for running jobs to finish (starting no new ones), now stops them at once; queued jobs wait for the next start")
def admin_stop(req: RestartRequest, request: Request) -> dict:
    try:
        restart.restarter().request(req.mode, then="stop")
    except RuntimeError as exc:
        raise Conflict(message_of(exc)) from exc
    return server_now(request)


@admin.post("/restart/cancel", access=Access.admin("server.restart"), summary="Call off the restart (while still waiting for running jobs): the queue carries on")
def admin_restart_cancel(request: Request) -> dict:
    restart.restarter().call_off()
    logs.say(log, Msg("I-RESTART-CALLEDOFF"))
    return server_now(request)
