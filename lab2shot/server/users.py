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
from .. import accounts, resources, roles
from ..errors import Forbidden, Invalid
from ..messages import Msg
from ..nodes import tags
from .. import traffic
from . import auth
from .access import audit, manages
from . import available
from .available import account


admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])  # 本模块的管理路由（由 app.py 引入）


def _managed(s: accounts.Session, target: accounts.User) -> None:
    """本会话管得着该账号（access.manages），否则抛出 Forbidden。"""
    if not manages(s, target.id, target.role):
        raise Forbidden(Msg("E-ROLES-NOTYOURS", role=roles.label(s.user.role), username=target.username, target=roles.label(target.role)))


def _no_tags(s: accounts.Session, given: list[str] | None) -> None:
    """许可标签只能由具有 users.tags 的账号（完整管理员）设置。"""
    if given is not None and not s.can("users.tags"):
        raise Forbidden(Msg("E-ROLES-TAGS", role=roles.label(s.user.role), default="、".join(tags.TAGS[t].label for t in sorted(tags.ALLOWED_NEW))))


def _may_make(s: accounts.Session, role: str) -> None:
    if not manages(s, role=role):
        raise Forbidden(Msg("E-ROLES-ROLE", target=roles.label(role), role=roles.label(s.user.role)))


def _view(request: Request) -> dict:
    s = auth.signed_in(request)
    visible = {r for r in roles.ROLES if manages(s, role=r)}
    # 流量和磁盘配额是同一问题的两个方面（该账号占用了多少资源）：在一行中一并提供，页面无需再次请求
    # （server/traffic.py；可见性由该路由自身声明的能力决定，此处不判断角色）。
    used = traffic.of_users()
    rows = [{**row, "traffic": dict(used.get(row["id"], traffic.EMPTY)), "applies": account(s, row).json()}
            for row in accounts.listing() if row["role"] in visible]
    return {"users": rows, "departments": accounts.departments(), "tags": tags.describe(),
            "allowed_new": sorted(tags.ALLOWED_NEW), "online": accounts.online(lambda owner, role: manages(s, owner, role)), "default_role": roles.DEFAULT,
            "roles": [{"id": r.id, "label": r.label, "tip": r.tip} for r in roles.ROLES.values() if r.id in visible]}


@admin.get("/users", access=Access.admin("users.manage_normal"), summary="用户：看得到的每个账号（二级管理员只看得到普通用户）的用户名、中文名、环节、角色、到期时间、能不能用、能用哪些标签的节点、最近登录、任务数、在不在线、用掉的网络流量（今天 / 近 7 天 / 总计），和这个登录能对它做什么；环节表、标签表、能新建的角色")
def users(request: Request) -> dict:
    return _view(request)


SUSPICIOUS_TIP = (f"7 天内超过 {accounts.SUSPICIOUS_IPS_7D} 个不同 IP，或超过 {accounts.SUSPICIOUS_DEVICES_7D} 个不同设备，"
                  f"或有一天失败 {accounts.SUSPICIOUS_FAILURES_DAY} 次以上")


@admin.get("/users/{user_id}/logins", access=Access.admin("logins.view"), summary="一个用户的最近登录：现在在哪在线（浏览器、DCC 插件各算一处）、最近的登录尝试、7 天和 30 天的汇总，"
                                              f"「异地频繁」的判断标准：{SUSPICIOUS_TIP}")
def user_logins(user_id: int, request: Request) -> dict:
    _managed(auth.signed_in(request), accounts.get(user_id))  # 账号从未存在时返回 404
    return {"online": accounts.online_now(user_id), "recent": accounts.login_recent(user_id),
            "summary": accounts.login_summary(user_id), "threshold_tip": SUSPICIOUS_TIP}


@admin.get("/users/{user_id}/resources", access=Access.admin("users.manage_normal"), summary="一个账号名下的东西有哪些页签：每种资源（任务、上传的素材、反馈、交付、基准素材、基准运行、登录记录、在线的登录、取回过的结果、管理操作）和条数；只列这个登录看得了的，「管理操作」只在管着事的账号上出现")
def user_resources(user_id: int, request: Request) -> dict:
    s = auth.signed_in(request)
    _managed(s, accounts.get(user_id))
    return {"user": user_id, "tabs": resources.counts(user_id, s.capabilities)}


@admin.get("/users/{user_id}/resources/{kind}", access=Access.admin("users.manage_normal"), summary="一个账号名下某一种资源的一页：列定义（中文列名）、按时间和按状态筛选看哪一列、有哪些状态、总数和这一页的行；q 按列里的文字筛选，since 只看这个时间之后的，state 只看这个状态的")
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


@admin.post("/users", access=Access.admin("users.manage_normal", lane=SECRETS), summary="新建账号：用户名、第一次的密码、中文名、环节、到期时间；管理员另外选角色和能用哪些标签的节点（不给：只有可商用）。二级管理员只能新建普通用户，不能设标签")
def create(req: NewUser, request: Request) -> dict:
    s = auth.signed_in(request)
    role = roles.check(req.role)
    _may_make(s, role)
    _no_tags(s, req.tags)
    given = req.tags if req.tags is not None else sorted(tags.ALLOWED_NEW)
    u = accounts.create(req.username, accounts.new_password(req.password), req.name, req.department, req.expires, given, role=role,
                        by=roles.label(s.user.role))
    audit(Msg("I-AUDIT-USERCREATED", who=s.user.label, role=roles.label(s.user.role), target=roles.label(u.role),
              username=u.username, name=u.name, department=u.department, expires=day_text(u.expires),
              tags="、".join(tags.TAGS[t].label for t in sorted(u.tags)) or "只有基础"),
          about=u.id, session=s, method="POST", path=str(request.url.path))
    return {"user": u.full(), **_view(request)}


class Change(Body):
    name: str | None = None
    department: str | None = None
    expires: Moment | None = None
    enabled: bool | None = None
    tags: list[str] | None = None  # 仅在具有 users.tags 时可设置
    role: str | None = None  # 仅在具有 admins.manage 时可设置


@admin.put("/users/{user_id}", access=Access.admin("users.manage_normal"), summary="改一个账号：中文名、环节、到期时间（延期）、停用或启用；管理员另外改能用哪些标签和角色。停用、到期或降了角色马上生效")
def change(user_id: int, req: Change, request: Request) -> dict:
    s = auth.signed_in(request)
    target = accounts.get(user_id)
    _managed(s, target)
    _no_tags(s, req.tags)
    if req.role is not None and req.role != target.role:
        _may_make(s, roles.check(req.role))
    u = accounts.update(user_id, name=req.name, department=req.department, expires=req.expires, enabled=req.enabled,
                        allowed=req.tags, role=req.role)
    audit(Msg("I-AUDIT-USERCHANGED", who=s.user.label, role=roles.label(s.user.role), target=roles.label(target.role),
              username=u.username, changes="、".join(f"{k}={v}" for k, v in req.model_dump(exclude_none=True).items())),
          about=target.id, session=s, method="PUT", path=str(request.url.path))
    return {"user": u.full(), **_view(request)}


class Reset(Body):
    password: str


@admin.post("/users/{user_id}/password", access=Access.admin("users.manage_normal", lane=SECRETS), summary="给一个账号设新密码（忘了密码时）：它所有的登录马上失效，用新密码重新登录")
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
    accounts.set_password(user_id, req.password, roles.label(s.user.role), keep)
    audit(Msg("I-AUDIT-PASSWORDRESET", who=s.user.label, role=roles.label(s.user.role), target=roles.label(u.role), username=u.username),
          about=u.id, session=s, method="POST", path=str(request.url.path))
    return _view(request)


@admin.delete("/users/{user_id}", access=Access.admin("users.delete"), summary="删除一个账号（只有管理员）：登录马上失效，排队和计算中的任务停下，结果和上传的引用删掉；任务记录留着，统计里算作「已删除的用户」，反馈留着")
def delete(user_id: int, request: Request) -> dict:
    s = auth.signed_in(request)
    u = accounts.get(user_id)
    _managed(s, u)
    done = accounts.delete(user_id)
    audit(Msg("I-AUDIT-USERDELETED", who=s.user.label, role=roles.label(s.user.role), target=roles.label(u.role),
              username=u.username, name=u.name, jobs=done["jobs_stopped"], outputs=done["outputs"]),
          about=u.id, session=s, method="DELETE", path=str(request.url.path))
    return {**done, **_view(request)}


@admin.delete("/users/{user_id}/purge", access=Access.admin("users.delete"), summary="永久删除一个已经删掉的账号（只有管理员）：账号行删掉，用户名可以给新人重用；任务记录、反馈、登录记录留着，统计里显示成「已删除的用户」；存在服务器上的节点图一起删掉")
def purge(user_id: int, request: Request) -> dict:
    """永久删除。分两步：先「删除」移入「已删除」，再「永久删除」，以避免一次误操作造成过大损失。保留和删除的内容
    由 accounts.purge 统一决定。"""
    s = auth.signed_in(request)
    u = accounts.get(user_id)
    _managed(s, u)
    done = accounts.purge(user_id)
    audit(Msg("I-AUDIT-USERPURGED", who=s.user.label, role=roles.label(s.user.role), target=roles.label(u.role),
              user=user_id, jobs=done["jobs"], feedback=done["feedback"], graphs=done["graphs"]),
          about=user_id, session=s, method="DELETE", path=str(request.url.path))
    return {**done, **_view(request)}


# ------------------------------------------------------------------ 输错密码的计数：由本机命令行清零


@admin.post("/security/unlock", access=Access.admin("security.manage", local=True), summary="输错密码的计数全部清零（被拖慢的账号、被锁住的来源马上可以再试）：只有这台服务器上的命令行（机器令牌）能调，浏览器上的管理员登录不行")
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


@admin.get("/rights", access=Access.admin("admins.manage"), summary="二级管理员的权限表：每条按分组排好，写明短名、勾上让人干什么、现在勾没勾，以及一共勾了几条、是不是还是默认那一份、谁最近改的")
def rights(request: Request) -> dict:
    return _rights(request)


def _changes(was: frozenset[str], now: frozenset[str]) -> str:
    """审计记录中的语句：增加了哪些项、去掉了哪些项（按短名，与页面上显示的一致）。"""
    parts = []
    for word, which in (("加上", now - was), ("去掉", was - now)):
        if which:
            parts.append(f"{word} " + "、".join(roles.CAPABILITIES[c].label for c in sorted(which)))
    return "、".join(parts) or "没有变化"


@admin.put("/rights", access=Access.admin("admins.manage"), summary="给一个角色分配能力（现在只有二级管理员可分配）：给的就是这一份完整清单，改完下一个请求就生效，正在用的人不用重新登录")
def set_rights(req: Rights, request: Request) -> dict:
    s = auth.signed_in(request)
    role = roles.assignable(req.role)
    was = roles.granted(role)
    now = roles.grant(role, req.on, by=accounts.Actor.of(s.user))
    audit(Msg("I-AUDIT-RIGHTSSET", who=s.user.label, target=roles.label(role), count=len(now), changes=_changes(was, now)),
          session=s, method="PUT", path=str(request.url.path))
    return _rights(request)


@admin.delete("/rights/{role}", access=Access.admin("admins.manage"), summary="恢复默认：这个角色的权限回到代码里那一份（lab2shot/roles.py 的默认），分配过的那一行删掉")
def reset_rights(role: str, request: Request) -> dict:
    s = auth.signed_in(request)
    role = roles.assignable(role)
    now = roles.reset(role)
    audit(Msg("I-AUDIT-RIGHTSDEFAULT", who=s.user.label, target=roles.label(role), count=len(now)),
          session=s, method="DELETE", path=str(request.url.path))
    return _rights(request)


def day_text(t: float | None) -> str:
    """An expiry date as the audit lines say it; one that is not a finite time (written before requests' numbers were
    bounded: routes.Body) never comes, like None."""
    return "不过期" if t is None or not math.isfinite(t) else time.strftime("%Y-%m-%d", time.localtime(t))
