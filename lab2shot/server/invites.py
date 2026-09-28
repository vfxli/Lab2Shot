"""后台「注册设置」页下面的邀请码：自行注册用的邀请码（lab2shot/registration.py），和按邀请码或时间段批量停用自己注册的账号。

新建（自己填或随机生成）、改备注 / 可用次数 / 到期、停用和启用、删除；列表写着每个码的全文（随时可以复制）、用了几次、谁用它
注册了。日志和留底只写码的前几位（registration.HINT），从不写全文。

路由都要 invites.manage（lab2shot/roles.py）；批量停用另外只停本会话管得着的账号（roles.manages：普通用户要
users.manage_normal），停用走的是「用户」页改账号的同一个函数（accounts.update）。每个改动都写进管理操作留底（access.audit）。"""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel

from .routes import Access, Router
from .. import accounts, registration, roles
from ..errors import Invalid
from ..messages import Msg
from . import auth
from .access import audit
from .users import day_text, sees

admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # 本模块的管理路由（由 app.py 引入）


def _session(request: Request) -> accounts.Session:
    s = auth.session(request)
    assert s is not None  # 守卫只放行具有该路由所需能力的会话
    return s


def _view() -> dict:
    return {"invites": registration.invites(), "registering": registration.counts(),
            "rules": {"min": registration.TYPED_MIN, "max": registration.TYPED_MAX, "note_most": registration.NOTE_MOST,
                     "uses_most": registration.USES_MOST}}


def _moment(t: float) -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def _uses(n: int | None) -> str:
    return "不限次数" if n is None else f"{n} 次"


@admin.get("/invites", access=Access.admin("invites.manage"), summary="邀请码：每个码的全文、备注、可用次数、用了几次、到期、开没开、能不能用，谁用它注册了；还有现在的注册情况（开没开放、要不要邀请码、最近一小时和一天注册了几个、暂停没有）")
def invites() -> dict:
    return _view()


class NewInvite(BaseModel):
    code: str = ""  # 空：随机生成
    note: str = ""
    uses_max: int | None = None  # None：不限次数
    expires: float | None = None  # 自纪元起的秒数；None：不过期


@admin.post("/invites", access=Access.admin("invites.manage"), summary="新建邀请码：自己填一个（6 到 32 个字母和数字），或留空随机生成；可以写备注、限次数、限到期")
def create(req: NewInvite, request: Request) -> dict:
    s = _session(request)
    made = registration.create_invite(req.code or None, req.note, req.uses_max, req.expires, s.user.label)
    audit(Msg("I-AUDIT-INVITECREATED", who=s.user.label, role=roles.label(s.user.role), hint=made["hint"],
              note=made["note"] or "没写备注", uses=_uses(made["uses_max"]), expires=day_text(made["expires"]),
              how="自己填的" if req.code else "随机生成"),
          session=s, method="POST", path=str(request.url.path))
    return {"made": made, **_view()}


class InviteChange(BaseModel):
    note: str
    uses_max: int | None
    expires: float | None
    enabled: bool


@admin.put("/invites/{invite_id}", access=Access.admin("invites.manage"), summary="改一个邀请码：备注、可用次数（空为不限，不能少于已经用掉的）、到期（空为不过期）、停用或启用；马上生效")
def change(invite_id: int, req: InviteChange, request: Request) -> dict:
    s = _session(request)
    done = registration.change_invite(invite_id, req.note, req.uses_max, req.expires, req.enabled)
    audit(Msg("I-AUDIT-INVITECHANGED", who=s.user.label, role=roles.label(s.user.role), hint=done["hint"],
              note=done["note"] or "没写备注", uses=_uses(done["uses_max"]), expires=day_text(done["expires"]),
              state="开着" if done["enabled"] else "停用"),
          session=s, method="PUT", path=str(request.url.path))
    return _view()


@admin.delete("/invites/{invite_id}", access=Access.admin("invites.manage"), summary="删除一个邀请码：以后不能再用它注册；用它注册的账号照旧，记录里还留着它的前几位")
def delete(invite_id: int, request: Request) -> dict:
    s = _session(request)
    gone = registration.delete_invite(invite_id)
    audit(Msg("I-AUDIT-INVITEDELETED", who=s.user.label, role=roles.label(s.user.role), hint=gone["hint"],
              note=gone["note"] or "没写备注", used=gone["used"]),
          session=s, method="DELETE", path=str(request.url.path))
    return _view()


@admin.get("/registrations", access=Access.admin("invites.manage"), summary="自己注册的账号（没删的）：按邀请码（invite）或注册时间段（since 起、until 止，自纪元起的秒数）筛选，每个写着注册时间、用的码的前几位、来源地址、现在开没开；这个登录管不着的账号标出来")
def registrations(request: Request, invite: int | None = None, since: float | None = None, until: float | None = None) -> dict:
    s = _session(request)
    rows = registration.registered(invite, since, until)
    return {"accounts": [{**r, "managed": sees(s, r["role"]) and r["id"] != accounts.ADMIN_ID} for r in rows]}


class Disable(BaseModel):
    invite: int | None = None
    since: float | None = None
    until: float | None = None


@admin.post("/registrations/disable", access=Access.admin("invites.manage"), summary="批量停用自己注册的账号：按邀请码，或按注册时间段（两个都给就两个都要满足）；只停这个登录管得着、现在还开着的，它们的登录马上失效。回答停了哪些")
def disable(req: Disable, request: Request) -> dict:
    s = _session(request)
    rows = registration.registered(req.invite, req.since, req.until)
    targets = [r["id"] for r in rows if r["enabled"] and sees(s, r["role"]) and r["id"] != accounts.ADMIN_ID]
    if not targets:
        raise Invalid(Msg("E-REGISTER-NONETODISABLE"))
    done = registration.disable(targets)
    by = f"邀请码 {rows[0]['invite']}…" if req.invite is not None else ""
    if req.since is not None or req.until is not None:
        by += ("、" if by else "") + f"注册时间 {_moment(req.since) if req.since is not None else '最早'} 到 {_moment(req.until) if req.until is not None else '现在'}"
    audit(Msg("I-AUDIT-REGISTEREDDISABLED", who=s.user.label, role=roles.label(s.user.role), by=by, count=len(done),
              usernames="、".join(done)),
          session=s, method="POST", path=str(request.url.path))
    return {"disabled": done, **_view()}
