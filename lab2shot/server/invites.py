"""后台「注册设置」页下面的邀请码：自行注册用的邀请码（lab2shot/site/registration.py），和按邀请码或时间段批量停用自己注册的账号。

新建（自己填或随机生成）、改备注 / 可用次数 / 到期、停用和启用、删除；列表写着每个码的全文（随时可以复制）、用了几次、谁用它
注册了。日志和留底只写码的前几位（registration.HINT），从不写全文。

路由都要 invites.manage（lab2shot/roles.py）；批量停用另外只停本会话管得着的账号（roles.manages：普通用户要
users.manage_normal），停用走的是「用户」页改账号的同一个函数（accounts.update）。每个改动都写进管理操作留底（access.audit）。"""

from __future__ import annotations

import math

from fastapi import Request

from .routes import Access, Body, Moment, Router
from .words import Word
from .. import accounts, roles
from ..site import registration
from ..errors import Invalid
from ..messages import Msg
from . import auth
from .access import audit
from .access import manages, particulars
from .users import day_text

admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])  # 本模块的管理路由（由 app.py 引入）


def _view() -> dict:
    return {"invites": registration.invites(), "registering": registration.counts(),
            "rules": {"min": registration.TYPED_MIN, "max": registration.TYPED_MAX, "note_most": registration.NOTE_MOST,
                     "uses_most": registration.USES_MOST}}


def _moment(t: float) -> str | Word:
    import time

    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if math.isfinite(t) else day_text(None)


def _uses(n: int | None) -> Word:
    return Word("server.invite.uses_unlimited") if n is None else Word("server.invite.uses", count=n)


@admin.get("/invites", access=Access.admin("invites.manage"), summary="Invite codes: each code in full, note, uses allowed, uses so far, expiry, enabled, usable, who registered with "
                                                                      "it; plus the registration state now (open, invite needed, how many registered in the last hour and day, "
                                                                      "paused)")
def invites() -> dict:
    return _view()


class NewInvite(Body):
    code: str = ""  # 空：随机生成
    note: str = ""
    uses_max: int | None = None  # None：不限次数
    expires: Moment | None = None  # 自纪元起的秒数；None：不过期


@admin.post("/invites", access=Access.admin("invites.manage"), summary="New invite code: type one (6 to 32 letters and digits) or leave empty for a random one; optionally a note, a "
                                                                       "use limit and an expiry")
def create(req: NewInvite, request: Request) -> dict:
    s = auth.signed_in(request)
    made = registration.create_invite(req.code or None, req.note, req.uses_max, req.expires, accounts.Actor.of(s.user))
    audit(Msg("I-AUDIT-INVITECREATED", who=s.user.label, role=roles.word(s.user.role), hint=made["hint"],
              note=made["note"] or Word("server.invite.no_note"), uses=_uses(made["uses_max"]), expires=day_text(made["expires"]),
              how=Word("server.invite.typed") if req.code else Word("server.invite.random")),
          session=s, method="POST", path=str(request.url.path))
    return {"made": made, **_view()}


class InviteChange(Body):
    note: str
    uses_max: int | None
    expires: Moment | None
    enabled: bool


@admin.put("/invites/{invite_id}", access=Access.admin("invites.manage"), summary="Change an invite code: note, uses allowed (empty for no limit, not fewer than already used), expiry (empty for "
                                                                                  "never), disable or enable; effective at once")
def change(invite_id: int, req: InviteChange, request: Request) -> dict:
    s = auth.signed_in(request)
    done = registration.change_invite(invite_id, req.note, req.uses_max, req.expires, req.enabled)
    audit(Msg("I-AUDIT-INVITECHANGED", who=s.user.label, role=roles.word(s.user.role), hint=done["hint"],
              note=done["note"] or Word("server.invite.no_note"), uses=_uses(done["uses_max"]), expires=day_text(done["expires"]),
              state=Word("server.invite.enabled") if done["enabled"] else Word("server.invite.disabled")),
          session=s, method="PUT", path=str(request.url.path))
    return _view()


@admin.delete("/invites/{invite_id}", access=Access.admin("invites.manage"), summary="Delete an invite code: nobody can register with it any more; accounts registered with it stay, and records "
                                                                                     "keep its first characters")
def delete(invite_id: int, request: Request) -> dict:
    s = auth.signed_in(request)
    gone = registration.delete_invite(invite_id)
    audit(Msg("I-AUDIT-INVITEDELETED", who=s.user.label, role=roles.word(s.user.role), hint=gone["hint"],
              note=gone["note"] or Word("server.invite.no_note"), used=gone["used"]),
          session=s, method="DELETE", path=str(request.url.path))
    return _view()


@admin.get("/registrations", access=Access.admin("invites.manage"), summary="Self-registered accounts (not deleted), filtered by invite code (invite) or registration period (since, until: "
                                                                            "seconds since the epoch), each with its registration time, the first characters of its code, its source "
                                                                            "address and whether it is enabled; accounts this login does not manage are marked")
def registrations(request: Request, invite: int | None = None, since: float | None = None, until: float | None = None) -> dict:
    s = auth.signed_in(request)
    rows = registration.registered(invite, since, until)
    # an account this login does not manage: marked so, and without where it registered from (access.particulars)
    return {"accounts": [{**particulars(s, r["id"], r), "managed": manages(s, r["id"], r["role"]) and r["id"] != accounts.ADMIN_ID}
                         for r in rows]}


class Disable(Body):
    invite: int | None = None
    since: Moment | None = None
    until: Moment | None = None


@admin.post("/registrations/disable", access=Access.admin("invites.manage"), summary="Disable self-registered accounts in bulk: by invite code, or by registration period (both given: both must "
                                                                                     "hold); only those this login manages and still enabled, whose logins end at once. The answer says which were "
                                                                                     "disabled")
def disable(req: Disable, request: Request) -> dict:
    s = auth.signed_in(request)
    rows = registration.registered(req.invite, req.since, req.until)
    targets = [r["id"] for r in rows if r["enabled"] and manages(s, r["id"], r["role"]) and r["id"] != accounts.ADMIN_ID]
    if not targets:
        raise Invalid(Msg("E-REGISTER-NONETODISABLE"))
    done = registration.disable(targets)
    by: list[Word] = [Word("server.invite.by_code", code=rows[0]["invite"])] if req.invite is not None else []
    if req.since is not None or req.until is not None:
        by.append(Word("server.invite.by_time", since=_moment(req.since) if req.since is not None else Word("server.invite.earliest"),
                       until=_moment(req.until) if req.until is not None else Word("server.invite.now")))
    audit(Msg("I-AUDIT-REGISTEREDDISABLED", who=s.user.label, role=roles.word(s.user.role), by=by, count=len(done),
              usernames=list(done)),
          session=s, method="POST", path=str(request.url.path))
    return {"disabled": done, **_view()}
