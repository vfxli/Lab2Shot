"""Ownership resolvers for route data, declared per route with `owned=` on its access (server/routes.py). The router
runs the resolver before the handler, and the handler reads the loaded object from request.state.owned. Handlers do
not check ownership themselves, and no router or dependency does either: ownership is always declared here, never
checked inline (`lab2shot check routes` holds every route naming a packet, job, task or template to it).

Two checks decide, both in server/access.py: a record with an owner (a job, a task, a template) is returned only to
the owning account or to sessions holding the right its route declares for other people's data (access.mine); a
packet is read only from the asking account's own cache (access.readable). Any other caller is told the object does
not exist, never that it belongs to someone else. Each resolver receives the request and the handler's parameters by
name."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request

from ..errors import NotFound
from ..messages import Msg

from . import auth
from .access import mine, readable


def job(request: Request, job_id: str):
    """Return a job (farm/queue.py Job), owned by the submitting account."""
    from ..farm import farm

    found = farm().get(job_id)
    mine(request, found.client.user, Msg("E-JOB-GONE"))
    return found


def job_seen_through(request: Request, job_id: str, camera: str | None = None):
    """A job (`job`), and the camera packet a partial result of it is placed with, when the route names one (`packets`)."""
    found = job(request, job_id)
    packets(request, camera=camera)
    return found


def packets(request: Request, fp: str | None = None, camera: str | None = None, through: str | None = None) -> None:
    """The packets a route reads: its {fp}, the camera it places points with (`camera`), the camera it looks through
    (`through`: its lens undistorts the plate, so reading through another account's camera would read its lens)."""
    for one in (fp, camera, through):
        if one:
            readable(request, one)


def packet_inputs(request: Request, req: Any = None) -> None:
    """The packets a request body names as a node's inputs (`inputs`: port -> fingerprint)."""
    for fp in (getattr(req, "inputs", None) or {}).values():
        readable(request, fp)


def job_record(request: Request, job_id: str) -> dict:
    """Return a job's stored record and graph (farm.job_row), owned by the submitting account."""
    from ..farm import job_row

    row = job_row(job_id)
    mine(request, row["user"], Msg("E-JOB-NOGRAPH"))
    return row


def feedback(request: Request, fid: str) -> dict:
    """Return a feedback (lab2shot/site/feedback.py get), to a login that may act on its sender's account (access.mine: an
    administrator's feedback is not a 二级管理员's to read or answer)."""
    from ..site import feedback as kept

    row = kept.get(fid)
    mine(request, row["user"], Msg("E-FEEDBACK-NOTFOUND"))
    return row


def task(request: Request, task_id: str) -> str:
    """Return a task's id (transfer/tasks.py), owned by the account that submitted it: its outputs are fetched only by
    that account (and a login holding data.others); anyone else is told there is no such result."""
    from ..transfer import tasks

    try:
        owner = tasks.owner(task_id)
    except NotFound:
        raise NotFound(Msg("E-OUTPUT-GONE", days=tasks.keep_days())) from None
    mine(request, owner, Msg("E-OUTPUT-GONE", days=tasks.keep_days()))
    return task_id


def task_output(request: Request, task_id: str, pkg: str) -> dict:
    """Return one finished output of a task (transfer/outputs.py record), owned as the task is (`task`)."""
    from ..transfer import outputs

    task(request, task_id)
    return outputs.record(task_id, pkg)


def saved_graph(request: Request, gid: str) -> dict:
    """Return a user's template file (「我的模板」 in server/library.py; id is user~<username>~<file name>), owned by
    the account named in the id. Sessions holding data.others also have access; the restore and permanent-delete
    actions on the admin 「用户」 detail page rely on this."""
    from .. import accounts
    from ..site import library

    kind, username, stem = library.parse_id(gid)
    account = accounts.by_username(username) if kind == "user" else None
    if account is None:
        raise NotFound(Msg("E-LIBRARY-NOSUCH"))
    mine(request, account.id, Msg("E-LIBRARY-NOSUCH"))
    return library.user_get(username, stem)


def own_jobs(request: Request, req: Any = None) -> list[str]:
    """The jobs the request body names (`jobs`) that are the asking account's own, in the order named: any other is
    left out as if it did not exist (a list is filtered, not refused: a page sends what it remembers, some of it gone).
    Strictly the account's own, whatever rights the login holds: these routes act on one's own jobs only."""
    from ..database import db

    asked = list(dict.fromkeys(j for j in (getattr(req, "jobs", None) or []) if isinstance(j, str)))
    if not asked:
        return []
    own = {r["id"] for r in db().rows(f"SELECT id FROM jobs WHERE user_id = ? AND id IN ({','.join('?' * len(asked))})",
                                      (auth.me(request).id, *asked))}
    return [j for j in asked if j in own]


def job_followed(request: Request, req: Any = None) -> str:
    """The job a new one follows (the body's `follows`, server/farm.py JobRequest): the asking account's own, never
    another's, whatever rights it holds (data.others included); refused with the same words whether it is another's or
    not there at all. "" when it follows none."""
    from ..database import db

    job_id = getattr(req, "follows", None) or ""
    if not job_id:
        return ""
    row = db().row("SELECT user_id FROM jobs WHERE id = ?", (job_id,))
    if row is None or row["user_id"] != auth.me(request).id:
        raise NotFound(Msg("B-JOB-FOLLOWS", job=job_id[:64]))
    return job_id


def saved_graph_named(request: Request, req: Any = None):
    """Return the template named by the request body's `id` (「保存到我的模板」 overwriting an existing template), or
    None when the id is empty (a new template, with no ownership to check)."""
    gid = str(getattr(req, "id", "") or "")
    return saved_graph(request, gid) if gid else None


def upload(request: Request, ref: str | None = None, req: Any = None):
    """Return the server location of the upload named by the route's `ref` parameter or the body's `ref`, for the
    requesting account; uploads of other accounts are reported as absent (transfer/uploads.py resolve is the single
    place that decides this, using the account resolved by the guard). Return None when no upload is named."""
    from ..transfer import uploads

    ref = ref if ref is not None else getattr(req, "ref", None)
    if not ref or not uploads.is_ref(ref):
        return None
    try:
        return uploads.resolve(ref)
    except NotFound:
        # An upload that has been declared but whose bytes have not arrived yet (declare first, transfer later) also
        # belongs to this account: its folder does not exist yet, but `declare_set` has recorded it under the account.
        # Without this case the file parameter would report the file as missing although it is still on the user's
        # machine. `declared()` checks the account as strictly as `resolve()`; other accounts' uploads remain absent.
        sid, _ = uploads.ref_parts(ref) or ("", "")
        if uploads.declared(sid) is None:
            raise
        return sid
