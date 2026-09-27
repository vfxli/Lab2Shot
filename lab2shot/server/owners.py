"""Ownership resolvers for route data, declared per route with `owned=` on its access (server/routes.py). The router
runs the resolver before the handler, and the handler reads the loaded object from request.state.owned. Handlers do
not check ownership themselves; ownership is always declared, never checked inline.

Data is returned only to the owning account or to sessions holding data.others; any other caller is told the object
does not exist (access.mine), never that it belongs to someone else. Each resolver receives the request and the
handler's parameters by name."""

from __future__ import annotations

from typing import Any

from starlette.requests import Request

from ..errors import NotFound
from ..messages import Msg

from .access import mine


def job(request: Request, job_id: str):
    """Return a job (farm/queue.py Job), owned by the submitting account."""
    from ..farm import farm

    found = farm().get(job_id)
    mine(request, found.client.user, Msg("E-JOB-GONE"))
    return found


def job_record(request: Request, job_id: str) -> dict:
    """Return a job's stored record and graph (farm.job_row), owned by the submitting account."""
    from ..farm import job_row

    row = job_row(job_id)
    mine(request, row["user"], Msg("E-JOB-NOGRAPH"))
    return row


def delivery(request: Request, run: str, node: str) -> dict:
    """Return a delivery record (transfer/deliveries.py), owned by the account whose cook produced it."""
    from ..transfer import deliveries

    record = deliveries.record(run, node)
    mine(request, record.get("user"), Msg("E-DELIVERY-GONE", days=deliveries.keep_days()))
    return record


def delivery_batch(request: Request, run: str, node: str) -> list[dict]:
    """Return every package one 「输出」 node delivered in one run (one per item of a 逐项处理 block), owned by the
    account whose cook produced them."""
    from ..transfer import deliveries

    found = deliveries.batch(run, node)
    if not found:
        raise NotFound(Msg("E-DELIVERY-GONE", days=deliveries.keep_days()))
    mine(request, found[0].get("user"), Msg("E-DELIVERY-GONE", days=deliveries.keep_days()))
    return found


def saved_graph(request: Request, gid: str) -> dict:
    """Return a user's template file (「我的模板」 in server/library.py; id is user~<username>~<file name>), owned by
    the account named in the id. Sessions holding data.others also have access; the restore and permanent-delete
    actions on the admin 「用户」 detail page rely on this."""
    from .. import accounts, library

    kind, username, stem = library.parse_id(gid)
    account = accounts.by_username(username) if kind == "user" else None
    if account is None:
        raise NotFound(Msg("E-LIBRARY-NOSUCH"))
    mine(request, account.id, Msg("E-LIBRARY-NOSUCH"))
    return library.user_get(username, stem)


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
        sid = ref[len(uploads.PREFIX):].partition("/")[0]
        if uploads.declared(sid) is None:
            raise
        return sid
