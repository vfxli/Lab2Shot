"""The top bar's 「更新说明」: the release notes (lab2shot/releases.py, CHANGELOG.toml), for any logged-in user. The file
is read once, when this module is imported at server start; a change to it shows after a restart, which is how a new
version arrives anyway."""

from __future__ import annotations

from fastapi import Request

from .. import releases, roles
from ..errors import Failed
from ..messages import Msg
from . import auth
from .routes import Access, Router

router = Router(prefix="/api", tags=["设置"])

releases.current()  # read now, at start: the first request does not pay for it


@router.get("/releases", access=Access.user("顶部栏「更新说明」：项目地址和各版本的更新"),
            summary="更新说明：项目地址，各版本（最新在前）的名称、日期、说明和更新条目；管理员与二级管理员另外看到给后台的条目")
def release_notes(request: Request) -> dict:
    """Everyone gets each version's `changes`; only an administrator or a 二级管理员 also gets its `admin` lines. The
    server leaves them out for everyone else, so they never reach an ordinary user's page."""
    notes = releases.current()
    if notes.problems:
        raise Failed(Msg("E-RELEASES-BROKEN", file=releases.FILE.name, why=notes.problems[0]))
    staff = auth.me(request).role in (roles.ADMIN, roles.DEPUTY_ROLE)
    return {"project": notes.project,
            "releases": [r if staff else {k: v for k, v in r.items() if k != "admin"} for r in notes.releases]}
