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

from contextlib import ExitStack, contextmanager

from fastapi import Request

from ..errors import Invalid, Unavailable
from ..extensions import get_extension, manual
from ..farm.tasks import Kind, Task
from ..installer import Live, TaskSink, install, plan, previous, rollback, uninstall
from ..installer.preflight import preflight
from ..messages import Msg
from . import auth
from .routes import Access, Body, Router

admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])  # this module's admin routes (app.py includes each module's)


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
    """Return the number of jobs that use this extension's environment: running jobs that compute a node of it or of
    an extension running in it (registry.sharing_env), and any job one of whose such nodes holds a place (granted a
    moment before its job is marked running)."""
    from ..extensions.registry import sharing_env
    from ..farm import farm
    from ..nodes import node_types

    names = set(sharing_env(name))
    types = node_types()
    jobs = {job.id for job in farm().running() if any(w.type in types and types[w.type].runtime in names for w in job.works)}
    return len(jobs | {t.task for t in farm().pools.now() if t.granted and t.need.runtime in names})


@contextmanager
def _holding(name: str):
    """No node of this extension, nor of one running in its environment, starts while it lasts."""
    from ..extensions.registry import sharing_env
    from ..farm import farm

    with ExitStack() as held:
        for each in sharing_env(name):
            held.enter_context(farm().pools.withholding(each))
        yield


def _switched(name: str) -> None:
    from ..engine.resident import pool
    from ..extensions.registry import sharing_env

    for each in sharing_env(name):  # end resident processes of the old environment (busy ones have already been awaited)
        pool().end_extension(each)
    _changed()


def _changed() -> None:
    from ..engine.evaluations import EVALUATIONS

    EVALUATIONS.bump()  # the state of the extension's nodes may have changed


def live() -> Live:
    from ..engine.resident import pool

    return Live(busy=busy, holding=_holding, switched=_switched, free_ram=lambda gb: pool().free_ram(gb))


def _work(name: str, rebuild: bool):
    """Return the task work that installs `name`, resuming from the last completed step and reporting to the task
    (installer/events.py TaskSink)."""

    def work(task: Task) -> Msg:
        ext = get_extension(name)
        try:
            install(ext, TaskSink(task), rebuild=rebuild, live=live())
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


class Start(Body):
    name: str
    # rerun every step into a fresh environment next to the live one. Changes to the original code are never thrown
    # away from the page: they stop the install, and whoever made them decides on the command line (--revert)
    rebuild: bool = False


@admin.post("/installs", access=Access.admin("installs.run"), summary="Start installing an extension (a background job, one install at a time): checks first and starts only when "
                                                                      "everything is there; steps already done are skipped, it continues from where it stopped")
def start_install(req: Start, request: Request) -> dict:
    from ..farm import farm

    ext = get_extension(req.name)
    checklist = preflight(ext)
    if not checklist.ready:
        blocking = checklist.blocking
        raise Unavailable(Msg("B-INSTALL-NOTREADY", title=ext.title, count=len(blocking), items=[c.message for c in blocking]))
    s = auth.session(request)
    return view(farm().tasks.submit(INSTALL, ext.name, s.user.label if s else "", _work(ext.name, req.rebuild)), 0)


@admin.get("/extensions", access=Access.admin("installs.run"), summary="Whether each extension is installed, what is missing, what this login may do with it, and its latest install "
                                                                       "job")
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
        if ext.is_base:  # no nodes of its own: its state is its features' (their rows), installing one installs it
            continue
        state = install_view(ext, session)
        rows.append({"name": name, "title": ext.title, "summary": ext.summary,
                     "nodes": sum(1 for t in types.values() if t.runtime == name),
                     "installed": state["installed"], "ready": state["ready"], "label": state["label"],
                     "reason": state["reason"], "manual": state["manual"],
                     "actions": state["actions"], "job": state["job"]})
    return {"extensions": rows}


@admin.get("/installs", access=Access.admin("installs.run"), summary="Install jobs queued or running (the background jobs' common form, without output lines)")
def install_jobs() -> dict:
    from ..farm import farm

    return {"jobs": [view(t) for t in farm().tasks.running(INSTALL)]}


@admin.get("/installs/{job_id}", access=Access.admin("installs.run"), summary="One install job: the background jobs' common form (output lines since `since`, the step reached) and the state "
                                                                              "of each step")
def install_job(job_id: str, since: int = 0) -> dict:
    from ..farm import farm

    return view(farm().tasks.get(job_id, INSTALL), since)


@admin.post("/installs/{job_id}/cancel", access=Access.admin("installs.run"), summary="Cancel an install job: it stops at the current step; installing again continues from that step")
def cancel_install(job_id: str) -> dict:
    from ..farm import farm

    return view(farm().tasks.cancel(job_id, INSTALL))


@admin.get("/extensions/{name}/preflight", access=Access.admin("installs.run"), summary="Pre-install check: manual downloads, licence consent, Hugging Face access requests, disk, GPU compatibility; "
                                                                                        "for each, what is missing and what to do")
def install_preflight(name: str) -> dict:
    return preflight(get_extension(name)).json()


@admin.post("/extensions/{name}/rollback", access=Access.admin("installs.run"), summary="Roll back to the previous environment (the one before the last install switched, while it is still on disk)")
def rollback_extension(name: str) -> dict:
    ext = get_extension(name)
    _refuse_while_installing(ext)
    rollback(ext, live())
    return facts(ext)


@admin.post("/extensions/{name}/uninstall", access=Access.admin("installs.run"), summary="Uninstall: delete the environment, code, cache and install records; model files are kept (no download again "
                                                                                         "when reinstalling)")
def uninstall_extension(name: str) -> dict:
    ext = get_extension(name)
    _refuse_while_installing(ext)
    uninstall(ext, live())
    return facts(ext)


# ------------------------------------------------------------------ downloaded by hand


@admin.get("/manual", access=Access.admin("installs.run"), summary="Manual downloads: the inbox folder, each item's state and where it is used, files not recognised (install the "
                                                                   "recognised ones first)")
def manual_downloads() -> dict:
    return manual.check()


@admin.get("/manual/licence", access=Access.admin("licence.consent"), summary="The licence text a file in the inbox folder needs the user to agree to (taken from its installer, not yet "
                                                                              "agreed)")
def manual_licence(file: str) -> dict:
    return manual.licence(file)  # ManualError (a MessageError) names the file and the reason


class Accept(Body):
    file: str  # file name in the inbox
    licence: str  # sha256 of the licence text shown to the user
    client: dict = {}  # client that submitted the consent (see lab2shot/farm/clients.py)


@admin.post("/manual/accept", access=Access.admin("licence.consent"), summary="After reading the licence the user clicks Agree and Install: who, when and which version is recorded, then it "
                                                                              "installs")
def manual_accept(req: Accept, request: Request) -> dict:
    from .farm import client_of

    try:
        return manual.accept(req.file, req.licence, client_of(request, req.client).full())
    except manual.ManualError as exc:
        raise Invalid(exc.message) from None  # a refusal, reported as 400 with its message rather than a server error
