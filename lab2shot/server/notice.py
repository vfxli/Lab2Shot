"""The administrator notice: a single non-dismissable bar shown at the top of the editor and the admin page. It is
server state stored in the database (meta `server.notice`):

    text     the message (at most NOTICE_CHARS; enabling it with empty text is refused)
    tone     the colour, one of the four design-token tones (TONES): info 信息, notice 提醒, warn 警告,
             risk 生产风险 (red, reserved for conditions that can cause a production incident)
    on       whether it is shown (the text is retained while hidden)
    updated  time of the last change (0: never), with `by` naming who made it

Changing it requires `settings.notice` (declared on the route, lab2shot/roles.py). Pages detect a change through the
server-state poll they already perform (GET /api/server `notice`: time of the last change) and then fetch the notice
(GET /api/notice, any logged-in user). No part of it is served before login."""

from __future__ import annotations

import time

from fastapi import Request

from ..database import db
from ..errors import Invalid
from ..messages import Msg
from . import auth
from .routes import Access, Body, Router

KEY = "server.notice"
TONES = ("info", "notice", "warn", "risk")
NOTICE_CHARS = 200
NONE = {"text": "", "tone": "info", "on": False, "updated": 0.0, "by": "", "by_id": None}

router = Router(prefix="/api", tags=["Settings"])
admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])


def current() -> dict:
    """Return the current notice (NONE until an administrator first sets one)."""
    return {**NONE, **db().meta(KEY, {})}


def changed_at() -> float:
    """Return the time of the last change (0: never); included in the server state so pages know when to refetch."""
    return float(current()["updated"])


class Notice(Body):
    text: str
    tone: str
    on: bool


def check(req: Notice) -> dict:
    """Return the normalized notice to store, or raise Invalid describing the problem."""
    text = req.text.strip()
    if req.tone not in TONES:
        raise Invalid(Msg("E-NOTICE-TONE", tone=req.tone, tones=list(TONES)))
    if len(text) > NOTICE_CHARS:
        raise Invalid(Msg("E-NOTICE-TOOLONG", count=len(text), most=NOTICE_CHARS))
    if req.on and not text:
        raise Invalid(Msg("B-NOTICE-EMPTY"))
    return {"text": text, "tone": req.tone, "on": req.on}


@router.get("/notice", access=Access.user("Admin notice: the bar at the top of the page"), summary="Admin notice: text, color (info / notice / warn / risk: production risk), whether shown, when last changed")
def notice() -> dict:
    return {k: v for k, v in current().items() if k not in ("by", "by_id")}


@admin.get("/notice", access=Access.admin("settings.notice"), summary="Admin notice (for editing): with who changed it last")
def admin_notice() -> dict:
    return current()


@admin.put("/notice", access=Access.admin("settings.notice"), summary="Change the admin notice: text (at most 200 characters), color, on/off; every page takes it up the next time it "
                                                                      "asks for the server state")
def set_notice(req: Notice, request: Request) -> dict:
    by = auth.actor(request)
    kept = {**check(req), "updated": time.time(), "by": by.label, "by_id": by.id}
    db().set_meta(KEY, kept)
    return kept
