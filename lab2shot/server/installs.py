"""Extension installation routes for the 「扩展包」 section of the admin page.

Every route requires the installs.run capability (declared per route via Access.admin); other sessions are refused by
the guard and are never shown install details (server/available.py extension()).

An install runs as a farm background task (farm/tasks.py; INSTALL below). Only one install runs at a time, since
installs download gigabytes and may compile. Installs are cancellable and report in the common progress format: steps
as done/total plus the current step, and the log as the task's output lines. An install stops immediately when the
farm closes or the server restarts; a later request resumes from the interrupted step using the installer's own
records (installer/run.py). The installer itself lives in lab2shot/installer; this module provides its routes and the
server state an install must respect (Live: running jobs that use the extension, resident worker processes, memory
that can be freed).

Manual downloads (lab2shot.extensions.manual): the inbox and the state of each item (reading it also sorts the inbox),
the licence text of a file that requires the user's consent, and the user's 同意并安装 action.
"""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel

from ..errors import Invalid, Unavailable
from ..extensions import get_extension, manual
from ..farm.tasks import Kind, Task
from ..installer import Live, TaskSink, install, plan, previous, rollback, uninstall
from ..installer.preflight import preflight
from ..messages import Msg
from . import auth
from .farm import admin
from .routes import Access


def _title(name: str) -> str:
    """Return the extension's display title, or its folder name when it is not loaded (a task from before a restart)."""
    from ..errors import MessageError

    try:
        return get_extension(name).title
    except MessageError:
        return name


INSTALL = Kind("install", title=lambda name: Msg("I-INSTALL-TASK", name=_title(name)), group="install",
               refused=lambda busy: Msg("E-INSTALL-BUSY", name=_title(busy.subject)),
               failed=lambda task, why: Msg("E-INSTALL-FAILED", name=_title(task.subject), why=why),
               cancelled=lambda task: Msg("N-INSTALL-CANCELLED", name=_title(task.subject)))


def busy(name: str) -> int:
    """Return the number of jobs that use this extension: running jobs that compute a node of it, and any job one of
    whose nodes of it holds a place (granted a moment before its job is marked running)."""
    from ..farm import farm
    from ..nodes import node_types

    types = node_types()
    jobs = {job.id for job in farm().running() if any(w.type in types and types[w.type].runtime == name for w in job.works)}
    return len(jobs | {t.task for t in farm().pools.now() if t.granted and t.need.runtime == name})


def _switched(name: str) -> None:
    from ..engine.resident import pool

    pool().end_extension(name)  # end resident processes of the old environment (busy ones have already been awaited)
    _changed()


def _changed() -> None:
    from ..engine.evaluations import EVALUATIONS

    EVALUATIONS.bump()  # the state of the extension's nodes may have changed


def live() -> Live:
    from ..engine.resident import pool

    from ..farm import farm

    return Live(busy=busy, holding=lambda name: farm().pools.withholding(name), switched=_switched,
                free_ram=lambda gb: pool().free_ram(gb))


def _work(name: str, force: bool):
    """Return the task work that installs `name`, resuming from the last completed step and reporting to the task
    (installer/events.py TaskSink)."""

    def work(task: Task) -> Msg:
        ext = get_extension(name)
        try:
            install(ext, TaskSink(task), force=force, live=live())
        finally:
            _changed()  # whether completed, partial or rolled forward, the extension's cards and nodes must be re-evaluated
        return Msg("I-INSTALL-DONE", title=ext.title)

    return work


def view(task: Task, since: int | None = None) -> dict:
    """Return an install task in the common progress format (Task.json with output lines after `since`; None omits
    them), plus each step's state for the step bar: steps before the current one are done (or skipped), the current
    step is running (or failed or cancelled), and later steps are waiting."""
    on = task.done if task.total else -1  # index of the current step; -1 until the installer has reported its plan
    at = {"running": "running", "failed": "failed", "cancelled": "cancelled"}.get(task.state, "waiting")
    rows = [{"id": s.id, "label": s.label,
             "state": "done" if task.state == "done" or i < on else at if i == on else "waiting"}
            for i, s in enumerate(plan.steps(get_extension(task.subject)))]
    return {**task.json(task.lines_total if since is None else since), "steps": rows}


def revision() -> tuple:
    """Return the revision key for responses that include install tasks (server/revisions.py installs): the running
    tasks and their progress. Starting, advancing or ending a task changes the key, which invalidates cached
    responses (ETag)."""
    from ..farm import farm

    return tuple((t.id, t.state, t.done, t.lines_total) for t in farm().tasks.running(INSTALL))


def latest(name: str) -> Task | None:
    from ..farm import farm

    return farm().tasks.latest(INSTALL, name)


def installs(session) -> bool:
    """Return whether the session holds installs.run; only such sessions receive install details."""
    return session is not None and session.can("installs.run")


def attribution_notice(status: dict) -> str | None:
    """Return the licence attribution to show on the extension's row, taken from the first downloaded weight that
    declares a `notice` (status["weights"]; lab2shot/extensions/spec.py Weight.notice). A weight used only by an
    optional feature contributes its notice only once it is downloaded. The notice is declared by the weight; this
    function does not name any project."""
    return next((w["notice"] for w in status["weights"] if w.get("notice") and w["status"] == "ok"), None)


def install_view(ext, session, everything: bool = False) -> dict:
    """Return an extension's install state as visible to the session: the full state with available actions
    (server/available.py extension) and the latest install task for sessions holding installs.run; only the public
    state (status.public) for other sessions. With `everything` and no session (command line), the full state is
    returned without actions.

    This is the single source of install state and available actions (extensions/status.py + available.extension);
    the 「扩展包」 section of the admin page reads it."""
    from ..extensions.status import extension_status, public
    from . import available

    status = extension_status(ext)
    notice = attribution_notice(status)
    if everything and session is None:
        return {**status, "notice": notice, "actions": {"available": [], "inactive": {}}, "job": None}
    if not installs(session):
        return {**public(status), "notice": notice, "actions": {"available": [], "inactive": {}}, "job": None}
    f = facts(ext)
    return {**status, "notice": notice, "actions": available.extension(session, f), "job": f["job"]}


def facts(ext) -> dict:
    """Return the facts that determine the admin page's actions on one extension (server/available.py extension)."""
    from ..extensions.status import extension_status

    task = latest(ext.name)
    return {"title": ext.title, "installed": extension_status(ext)["installed"], "job": view(task) if task else None,
            "previous": previous(ext) is not None, "busy": busy(ext.name)}


def _refuse_while_installing(ext) -> None:
    if (task := latest(ext.name)) is not None and task.active:
        raise Unavailable(Msg("N-INSTALL-QUEUED", title=ext.title))


class Start(BaseModel):
    name: str
    force: bool = False  # rerun every step into a fresh environment next to the live one


@admin.post("/installs", access=Access.admin("installs.run"), summary="开始安装一个扩展包（后台任务，同一时间只装一个）：先体检，齐全才开始；已完成的步骤跳过，从停下的那步接着装")
def start_install(req: Start, request: Request) -> dict:
    from ..farm import farm

    ext = get_extension(req.name)
    checklist = preflight(ext)
    if not checklist.ready:
        blocking = checklist.blocking
        raise Unavailable(Msg("B-INSTALL-NOTREADY", title=ext.title, count=len(blocking), items=[c.message for c in blocking]))
    s = auth.session(request)
    return view(farm().tasks.submit(INSTALL, ext.name, s.user.label if s else "", _work(ext.name, req.force)), 0)


@admin.get("/extensions", access=Access.admin("installs.run"), summary="每个扩展包装了没有、缺什么、这个登录能对它做什么，和它最近一次安装任务")
def extension_list(request: Request) -> dict:
    """Return the data for the 「扩展包」 section of the admin page (webui/src/admin/Extensions.tsx). All install routes
    require installs.run. Install state and actions come from install_view (extensions/status.py +
    available.extension), the single source for them."""
    from ..extensions import extensions
    from ..nodes import node_types

    session = auth.session(request)
    types = node_types()
    rows = []
    for name, ext in sorted(extensions().items()):
        state = install_view(ext, session)
        rows.append({"name": name, "title": ext.title, "summary": ext.summary,
                     "nodes": sum(1 for t in types.values() if t.runtime == name),
                     "installed": state["installed"], "ready": state["ready"], "label": state["label"],
                     "reason": state["reason"], "manual": state["manual"],
                     "actions": state["actions"], "job": state["job"]})
    return {"extensions": rows}


@admin.get("/installs", access=Access.admin("installs.run"), summary="正在排队或进行中的安装任务（后台任务的统一格式，不带输出行）")
def install_jobs() -> dict:
    from ..farm import farm

    return {"jobs": [view(t) for t in farm().tasks.running(INSTALL)]}


@admin.get("/installs/{job_id}", access=Access.admin("installs.run"), summary="一个安装任务：后台任务的统一格式（since 之后的输出行、做到第几步）和每一步的状态")
def install_job(job_id: str, since: int = 0) -> dict:
    from ..farm import farm

    return view(farm().tasks.get(job_id, INSTALL), since)


@admin.post("/installs/{job_id}/cancel", access=Access.admin("installs.run"), summary="取消一个安装任务：停在当前步骤，再点安装从这一步接着装")
def cancel_install(job_id: str) -> dict:
    from ..farm import farm

    return view(farm().tasks.cancel(job_id, INSTALL))


@admin.get("/extensions/{name}/preflight", access=Access.admin("installs.run"), summary="安装前体检：手动下载、许可同意、Hugging Face 申请、硬盘、显卡兼容，每项缺什么、怎么办")
def install_preflight(name: str) -> dict:
    return preflight(get_extension(name)).json()


@admin.post("/extensions/{name}/rollback", access=Access.admin("installs.run"), summary="回退到上一个环境（上次安装切换之前的那个，还在硬盘上时）")
def rollback_extension(name: str) -> dict:
    ext = get_extension(name)
    _refuse_while_installing(ext)
    rollback(ext, live())
    return facts(ext)


@admin.post("/extensions/{name}/uninstall", access=Access.admin("installs.run"), summary="卸载：删掉环境、代码、缓存和安装记录；模型文件保留（重装不用再下载）")
def uninstall_extension(name: str) -> dict:
    ext = get_extension(name)
    _refuse_while_installing(ext)
    uninstall(ext, live())
    return facts(ext)


# ------------------------------------------------------------------ downloaded by hand


@admin.get("/manual", access=Access.admin("installs.run"), summary="手动下载：收件文件夹、每一项的状态和用在哪、认不出的文件（先把认得出的装好）")
def manual_downloads() -> dict:
    return manual.check()


@admin.get("/manual/licence", access=Access.admin("licence.consent"), summary="收件文件夹里一个文件要用户同意的许可协议原文（从安装程序里取出，没有同意）")
def manual_licence(file: str) -> dict:
    return manual.licence(file)  # ManualError (a MessageError) names the file and the reason


class Accept(BaseModel):
    file: str  # file name in the inbox
    licence: str  # sha256 of the licence text shown to the user
    client: dict = {}  # client that submitted the consent (see lab2shot/farm/clients.py)


@admin.post("/manual/accept", access=Access.admin("licence.consent"), summary="用户看过许可协议后点「同意并安装」：记下谁、什么时候、哪个版本，然后安装")
def manual_accept(req: Accept, request: Request) -> dict:
    from .farm import client_of

    try:
        return manual.accept(req.file, req.licence, client_of(request, req.client).full())
    except manual.ManualError as exc:
        raise Invalid(exc.message) from None  # a refusal, reported as 400 with its message rather than a server error
