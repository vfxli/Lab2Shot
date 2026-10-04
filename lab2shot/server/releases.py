"""The top bar's 「更新说明」: the release notes (lab2shot/releases.py, CHANGELOG.toml), for any logged-in user. The file
is read once, when this module is imported at server start; a change to it shows after a restart, which is how a new
version arrives anyway."""

from __future__ import annotations

from .. import releases
from ..errors import Failed
from ..messages import Msg
from .routes import Access, Router

router = Router(prefix="/api", tags=["Settings"])

releases.current()  # read now, at start: the first request does not pay for it


@router.get("/releases", access=Access.user("Top bar Release Notes: the project address and each version's changes", hides={"releases.admin": ("releases[].admin",)}),
            summary="Release notes: the project address and each version (newest first) with its name, date, notes and items; "
                    "administrators and deputy administrators also see the items for the admin side")
def release_notes() -> dict:
    """Every version whole: the guard leaves each one's `admin` lines out for a login that does not work in the back
    office (server/available.py FIELDS releases.admin), so they never reach an ordinary user's page."""
    notes = releases.current()
    if notes.problems:
        raise Failed(Msg("E-RELEASES-BROKEN", file=releases.FILE.name, why=notes.problems[0]))
    return {"project": notes.project, "releases": notes.releases}
