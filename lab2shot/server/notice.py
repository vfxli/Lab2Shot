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

router = Router(prefix="/api", tags=["设置"])
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])


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


@router.get("/notice", access=Access.user("管理员通知：页面顶部的通知条"), summary="管理员通知：文字、颜色（info 信息 / notice 提醒 / warn 警告 / risk 生产风险）、是否显示、上次修改的时间")
def notice() -> dict:
    return {k: v for k, v in current().items() if k not in ("by", "by_id")}


@admin.get("/notice", access=Access.admin("settings.notice"), summary="管理员通知（编辑用）：连同上次是谁改的")
def admin_notice() -> dict:
    return current()


@admin.put("/notice", access=Access.admin("settings.notice"), summary="改管理员通知：文字（最多 200 字）、颜色、开关；所有页面在下一次查服务器状态时换上")
def set_notice(req: Notice, request: Request) -> dict:
    by = auth.actor(request)
    kept = {**check(req), "updated": time.time(), "by": by.label, "by_id": by.id}
    db().set_meta(KEY, kept)
    return kept
