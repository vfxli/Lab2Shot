"""后台「用户」页：管理账号（lab2shot/accounts.py）。管理员或二级管理员可以创建账号，设置初始密码、姓名、环节、
有效期，重设密码，停用和延期；管理员还可以设置其可用范围（tags）、角色，以及删除账号。谁可以对谁执行哪些操作由
lab2shot/roles.py 决定（路由所需能力由守卫检查，对目标账号的 roles.manages 在此处检查）；页面提供的操作见
server/available.py。每次修改都连同消息代码写入服务器日志（access.audit）。"""

from __future__ import annotations

import math
import time

from fastapi import Request

from .routes import Access, Body, Moment, Router
from .wire import SECRETS
from .words import Word
from .. import accounts, i18n, roles
from ..site import account_removal, resources
from ..errors import Forbidden, Invalid
from ..messages import Msg
from ..nodes import tags
from .. import traffic
from . import auth
from .access import audit, manages
from . import available
from .available import account


admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])  # 本模块的管理路由（由 app.py 引入）


def _managed(s: accounts.Session, target: accounts.User) -> None:
    """本会话管得着该账号（access.manages），否则抛出 Forbidden。"""
    if not manages(s, target.id, target.role):
        raise Forbidden(Msg("E-ROLES-NOTYOURS", role=i18n.Both.of(lambda: roles.label(s.user.role)), username=target.username, target=i18n.Both.of(lambda: roles.label(target.role))))


def _no_tags(s: accounts.Session, given: list[str] | None) -> None:
    """许可标签只能由具有 users.tags 的账号（完整管理员）设置。"""
    if given is not None and not s.can("users.tags"):
        raise Forbidden(Msg("E-ROLES-TAGS", role=i18n.Both.of(lambda: roles.label(s.user.role)), default=[tags.TAGS[t].label for t in sorted(tags.ALLOWED_NEW)]))


def _may_make(s: accounts.Session, role: str) -> None:
    if not manages(s, role=role):
        raise Forbidden(Msg("E-ROLES-ROLE", target=i18n.Both.of(lambda: roles.label(role)), role=i18n.Both.of(lambda: roles.label(s.user.role))))


def _view(request: Request) -> dict:
    s = auth.signed_in(request)
    visible = {r for r in roles.ROLES if manages(s, role=r)}
    # 流量和磁盘配额是同一问题的两个方面（该账号占用了多少资源）：在一行中一并提供，页面无需再次请求
    # （server/traffic.py；可见性由该路由自身声明的能力决定，此处不判断角色）。
    used = traffic.of_users()
    rows = [{**row, "traffic": dict(used.get(row["id"], traffic.EMPTY)), "applies": account(s, row).json()}
            for row in accounts.listing() if row["role"] in visible]
    return {"users": rows, "departments": accounts.departments_shown(), "tags": tags.describe(),
            "allowed_new": sorted(tags.ALLOWED_NEW), "online": accounts.online(lambda owner, role: manages(s, owner, role)), "default_role": roles.DEFAULT,
            "roles": [{"id": r.id, "label": r.label, "tip": r.tip} for r in roles.ROLES.values() if r.id in visible]}


@admin.get("/users", access=Access.admin("users.manage_normal"), summary="Users: each visible account (a deputy administrator sees only normal users) with username, display name, "
                                                                         "department, role, expiry, whether usable, which tags' nodes it may use, last login, job count, whether online, "
                                                                         "network traffic used (today / last 7 days / in all), and what this login may do with it; the departments, tags "
                                                                         "and roles that can be made")
def users(request: Request) -> dict:
    return _view(request)


# the rule for 「异地频繁」, in the route's description (English, for developers) and in words on the page (threshold_tip)
SUSPICIOUS = {"ips": accounts.SUSPICIOUS_IPS_7D, "devices": accounts.SUSPICIOUS_DEVICES_7D, "failures": accounts.SUSPICIOUS_FAILURES_DAY}


@admin.get("/users/{user_id}/logins", access=Access.admin("logins.view"),
           summary="One user's recent logins: where they are online now (browser and each DCC plugin count as one place each), "
                   "recent login attempts, 7-day and 30-day summaries; the rule for \"frequent logins from many places\": "
                   f"more than {SUSPICIOUS['ips']} different IPs or more than {SUSPICIOUS['devices']} different devices "
                   f"within 7 days, or more than {SUSPICIOUS['failures']} failures in one day")
def user_logins(user_id: int, request: Request) -> dict:
    _managed(auth.signed_in(request), accounts.get(user_id))  # 账号从未存在时返回 404
    return {"online": accounts.online_now(user_id), "recent": accounts.login_recent(user_id),
            "summary": accounts.login_summary(user_id), "threshold_tip": i18n.t("server.logins.threshold", **SUSPICIOUS)}


@admin.get("/users/{user_id}/resources", access=Access.admin("users.manage_normal"), summary="Which tabs of things an account owns there are: each kind of resource (jobs, uploaded media, feedback, "
                                                                                             "deliveries, benchmark media, benchmark runs, login records, live logins, results fetched, admin actions) and "
                                                                                             "its count; only those this login may see, Admin Actions only on accounts that manage something")
def user_resources(user_id: int, request: Request) -> dict:
    s = auth.signed_in(request)
    _managed(s, accounts.get(user_id))
    return {"user": user_id, "tabs": resources.counts(user_id, s.capabilities)}


@admin.get("/users/{user_id}/resources/{kind}", access=Access.admin("users.manage_normal"), summary="One page of one kind of resource an account owns: column definitions (display names), which column the time "
                                                                                                    "and state filters use, which states there are, the total and this page's rows; q filters by the text in the "
                                                                                                    "columns, since shows only after this time, state only this state")
def user_resource(user_id: int, kind: str, request: Request, offset: int = 0, limit: int = 50, q: str = "",
                  since: float = 0.0, state: str = "") -> dict:
    s = auth.signed_in(request)
    _managed(s, accounts.get(user_id))
    found = resources.page(kind, user_id, s.capabilities, offset, limit, q, since, state)
    # 每行可执行的操作：由登记表声明，可用性按其调用的路由计算（server/available.py），不可用的不发送给网页
    return {**found, "acts": available.resource_acts(s, found["acts"])}


class NewUser(Body):
    username: str
    password: str
    name: str
    department: str
    expires: Moment  # 自纪元起的秒数
    tags: list[str] | None = None  # None：新账号可使用的范围（nodes/tags.py ALLOWED_NEW）；仅在具有 users.tags 时可设置
    role: str = roles.DEFAULT


@admin.post("/users", access=Access.admin("users.manage_normal", lane=SECRETS), summary="New account: username, first password, display name, department, expiry; an administrator also picks the role "
                                                                                        "and which tags' nodes it may use (none given: commercial use only). A deputy administrator can only make "
                                                                                        "normal users and cannot set tags")
def create(req: NewUser, request: Request) -> dict:
    s = auth.signed_in(request)
    role = roles.check(req.role)
    _may_make(s, role)
    _no_tags(s, req.tags)
    given = req.tags if req.tags is not None else sorted(tags.ALLOWED_NEW)
    u = accounts.create(req.username, accounts.new_password(req.password), req.name, req.department, req.expires, given, role=role,
                        by=accounts.by_role(s.user.role))
    audit(Msg("I-AUDIT-USERCREATED", who=s.user.label, role=roles.word(s.user.role), target=roles.word(u.role),
              username=u.username, name=u.name, department=accounts.department_label(u.department), expires=day_text(u.expires),
              tags=[tags.TAGS[t].label for t in sorted(u.tags)] or Word("server.tags.basic_only")),
          about=u.id, session=s, method="POST", path=str(request.url.path))
    return {"user": u.full(), **_view(request)}


class Change(Body):
    name: str | None = None
    department: str | None = None
    expires: Moment | None = None
    enabled: bool | None = None
    tags: list[str] | None = None  # 仅在具有 users.tags 时可设置
    role: str | None = None  # 仅在具有 admins.manage 时可设置
    queue_first: bool | None = None  # 队列优先：仅在具有 queue.manage 时可设置（available.py account.queue_first）


@admin.put("/users/{user_id}", access=Access.admin("users.manage_normal"), summary="Change an account: display name, department, expiry (extend), disable or enable; an administrator also changes "
                                                                                   "its tags and role. Disabling, expiry or a lower role take effect at once. Queue "
                                                                                   "priority (with queue.manage): its tasks submitted from then on wait ahead of every "
                                                                                   "other account's waiting task; nothing running is stopped")
def change(user_id: int, req: Change, request: Request) -> dict:
    s = auth.signed_in(request)
    target = accounts.get(user_id)
    _managed(s, target)
    _no_tags(s, req.tags)
    if req.role is not None and req.role != target.role:
        _may_make(s, roles.check(req.role))
    if req.queue_first is not None and not s.can("queue.manage"):  # 队列优先 decides whose task starts next: the queue's
        raise Forbidden(Msg("E-ROLES-QUEUEFIRST", role=i18n.Both.of(lambda: roles.label(s.user.role))))
    u = accounts.update(user_id, name=req.name, department=req.department, expires=req.expires, enabled=req.enabled,
                        allowed=req.tags, role=req.role, queue_first=req.queue_first)
    audit(Msg("I-AUDIT-USERCHANGED", who=s.user.label, role=roles.word(s.user.role), target=roles.word(target.role),
              username=u.username, changes=[f"{k}={v}" for k, v in req.model_dump(exclude_none=True).items()]),
          about=target.id, session=s, method="PUT", path=str(request.url.path))
    return {"user": u.full(), **_view(request)}


class Reset(Body):
    password: str


@admin.post("/users/{user_id}/password", access=Access.admin("users.manage_normal", lane=SECRETS), summary="Set a new password for an account (when forgotten): all its logins end at once, it logs in again with the new "
                                                                                                           "password")
def reset(user_id: int, req: Reset, request: Request) -> dict:
    s = auth.signed_in(request)
    u = accounts.get(user_id)
    _managed(s, u)
    if u.deleted:
        raise Invalid(Msg("E-USERS-DELETED"))
    # 内置管理员账号（ADMIN_ID）不得由他人修改（roles.py）：`accounts.update` / `delete` 均有此保护，重设密码也必须保护，
    # 否则拥有 admins.manage 的二级管理员可以为内置管理员设置新密码，从而接管该账号。
    # 内置管理员本人可以修改（浏览器上自己的那一行、本机命令行的机器令牌都以 ADMIN_ID 登录）；忘记密码时使用口令或 lab2shot admin password
    if u.owner and s.user.id != u.id:
        raise Forbidden(Msg("E-ACCOUNT-ADMINPASSWORD", username=u.username))
    keep = auth.token_of(request) if u.id == s.user.id else None  # 重设自己的密码时保留当前登录
    accounts.set_password(user_id, req.password, accounts.by_role(s.user.role), keep)
    audit(Msg("I-AUDIT-PASSWORDRESET", who=s.user.label, role=roles.word(s.user.role), target=roles.word(u.role), username=u.username),
          about=u.id, session=s, method="POST", path=str(request.url.path))
    return _view(request)


@admin.delete("/users/{user_id}", access=Access.admin("users.delete"), summary="Delete an account (administrators only): logins end at once, queued and cooking jobs stop, results and upload "
                                                                               "references are deleted; job records stay, counted as Deleted User in the statistics, and feedback stays")
def delete(user_id: int, request: Request) -> dict:
    s = auth.signed_in(request)
    u = accounts.get(user_id)
    _managed(s, u)
    done = account_removal.delete(user_id)
    audit(Msg("I-AUDIT-USERDELETED", who=s.user.label, role=roles.word(s.user.role), target=roles.word(u.role),
              username=u.username, name=u.name, jobs=done["jobs_stopped"], outputs=done["outputs"]),
          about=u.id, session=s, method="DELETE", path=str(request.url.path))
    return {**done, **_view(request)}


@admin.delete("/users/{user_id}/purge", access=Access.admin("users.delete"), summary="Permanently delete an account already deleted (administrators only): the account row is deleted and the "
                                                                                     "username may be reused by someone new; job records, feedback and login records stay, shown as Deleted User in "
                                                                                     "the statistics; graphs kept on the server are deleted with it")
def purge(user_id: int, request: Request) -> dict:
    """永久删除。分两步：先「删除」移入「已删除」，再「永久删除」，以避免一次误操作造成过大损失。保留和删除的内容
    由 account_removal.purge 统一决定。"""
    s = auth.signed_in(request)
    u = accounts.get(user_id)
    _managed(s, u)
    done = account_removal.purge(user_id)
    audit(Msg("I-AUDIT-USERPURGED", who=s.user.label, role=roles.word(s.user.role), target=roles.word(u.role),
              user=user_id, jobs=done["jobs"], feedback=done["feedback"], graphs=done["graphs"]),
          about=user_id, session=s, method="DELETE", path=str(request.url.path))
    return {**done, **_view(request)}


# ------------------------------------------------------------------ 输错密码的计数：由本机命令行清零


@admin.post("/security/unlock", access=Access.admin("security.manage", local=True), summary="Reset all counts of wrong passwords (slowed accounts and locked sources may try again at once): only the "
                                                                                            "command line on this server (machine token) may call it, not an administrator logged in from a browser")
def unlock_logins(request: Request) -> dict:
    """输错密码按账号、按来源计数（server/auth.py Limiter）：从没登录过的设备把一个账号试错多了，之后每次都要等。
    此处为内置管理员提供一条无需更换密码的清零途径：由本机命令行使用机器令牌调用此路由。只接受机器令牌：计数挡的
    正是用密码猜，若浏览器上的管理员登录可以清零，就会成为另一条绕过它的途径。"""
    s = auth.signed_in(request)
    cleared = auth.guards().limiter.clear()
    audit(Msg("I-AUDIT-LOGINUNLOCKED", who=s.user.label, count=cleared), session=s, method="POST", path=str(request.url.path))
    return {"cleared": cleared}


# ------------------------------------------------------------------ 二级管理员的权限
#
# 哪些能力属于二级管理员，是由一级管理员在「用户」中勾选的数据（roles.granted），不在代码中固定。
# 三条路由都要求 admins.manage：只有能管理其他管理员的账号才能修改其权限。修改后下一个请求即生效，无需重新登录。


class Rights(Body):
    role: str
    on: list[str]  # 该角色勾选的全部能力（完整列表，而非增量）


def _rights(request: Request) -> dict:
    """每个可分配的角色一张勾选表，由服务器排版（分组、短名、说明、是否勾选、条数）：网页按此绘制，不自行计算。"""
    return {"sheets": [roles.sheet(r) for r in roles.ASSIGNABLE]}


@admin.get("/rights", access=Access.admin("admins.manage"), summary="The deputy administrators' rights table: each right sorted by group with its short name, what ticking it lets "
                                                                    "one do and whether it is ticked, plus how many are ticked, whether it is still the default and who changed it "
                                                                    "last")
def rights(request: Request) -> dict:
    return _rights(request)


def _changes(was: frozenset[str], now: frozenset[str]) -> list[Word] | Word:
    """审计记录中的语句：增加了哪些项、去掉了哪些项（按短名，与页面上显示的一致）。"""
    parts = []
    for word, which in (("server.rights.added", now - was), ("server.rights.removed", was - now)):
        if which:
            parts.append(Word(word, items=i18n.separator().join(str(roles.CAPABILITIES[c].label) for c in sorted(which))))
    return parts or Word("server.rights.unchanged")


@admin.put("/rights", access=Access.admin("admins.manage"), summary="Assign capabilities to a role (only deputy administrators can be assigned now): the full list given is the new "
                                                                    "one, effective from the next request, nobody needs to log in again")
def set_rights(req: Rights, request: Request) -> dict:
    s = auth.signed_in(request)
    role = roles.assignable(req.role)
    was = roles.granted(role)
    now = roles.grant(role, req.on, by=accounts.Actor.of(s.user))
    audit(Msg("I-AUDIT-RIGHTSSET", who=s.user.label, target=roles.word(role), count=len(now), changes=_changes(was, now)),
          session=s, method="PUT", path=str(request.url.path))
    return _rights(request)


@admin.delete("/rights/{role}", access=Access.admin("admins.manage"), summary="Restore defaults: the role's rights go back to the ones in the code (the default of lab2shot/roles.py) and the "
                                                                              "assigned row is deleted")
def reset_rights(role: str, request: Request) -> dict:
    s = auth.signed_in(request)
    role = roles.assignable(role)
    now = roles.reset(role)
    audit(Msg("I-AUDIT-RIGHTSDEFAULT", who=s.user.label, target=roles.word(role), count=len(now)),
          session=s, method="DELETE", path=str(request.url.path))
    return _rights(request)


def day_text(t: float | None) -> str | Word:
    """An expiry date as the audit lines say it; one that is not a finite time (written before requests' numbers were
    bounded: routes.Body) never comes, like None."""
    return Word("server.never_expires") if t is None or not math.isfinite(t) else time.strftime("%Y-%m-%d", time.localtime(t))
