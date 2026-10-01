"""The top bar's 「更新说明」: the release notes (lab2shot/releases.py, CHANGELOG.toml), for any logged-in user. The file
is read once, when this module is imported at server start; a change to it shows after a restart, which is how a new
version arrives anyway."""

from __future__ import annotations

from .. import releases
from ..errors import Failed
from ..messages import Msg
from .routes import Access, Router

router = Router(prefix="/api", tags=["设置"])

releases.current()  # read now, at start: the first request does not pay for it


@router.get("/releases", access=Access.user("顶部栏「更新说明」：项目地址和各版本的更新", hides={"releases.admin": ("releases[].admin",)}),
            summary="更新说明：项目地址，各版本（最新在前）的名称、日期、说明和更新条目；管理员与二级管理员另外看到给后台的条目")
def release_notes() -> dict:
    """Every version whole: the guard leaves each one's `admin` lines out for a login that does not work in the back
    office (server/available.py FIELDS releases.admin), so they never reach an ordinary user's page."""
    notes = releases.current()
    if notes.problems:
        raise Failed(Msg("E-RELEASES-BROKEN", file=releases.FILE.name, why=notes.problems[0]))
    return {"project": notes.project, "releases": notes.releases}
