"""Roles and what each may do: the one permission model of the admin side, in one place.

A capability is one named thing an account may do beyond using the editor (manage 普通用户, see the whole queue,
switch a template off, restart the server, ...). A role is a set of capabilities:

    管理员      (admin)     every capability, always
    二级管理员  (deputy)    what the 管理员 gave it (the `role_rights` table, 后台「用户」→「二级管理员权限」);
                            DEPUTY below is only what it starts with, before anyone assigns anything
    普通用户    (user)      none: the editor, their own jobs, results, uploads and feedback

Nothing else decides who may do what:
  - every /api/admin/ route declares the capability it needs (server/routes.py Access.admin, on the route); the guard refuses the others
    (a route without one answers 404: fail closed);
  - an action on another account also needs the right to manage that account's role (manages): 普通用户 through
    users.manage_normal, 管理员 and 二级管理员 only through admins.manage;
  - a few settings need a capability besides settings.edit (setting_needs): the system ones settings.system, the
    licence tags of self-registered accounts users.tags;
  - what the pages show follows from the same declarations, resolved by the one availability mechanism
    (lab2shot/availability.py) with the conditions below (Can, Manages: the capability kind, hidden when not held;
    RightsFresh, NotOwnersAccount: the data kind, greyed with why), declared per subject in server/available.py.

What a 二级管理员 may do is data, not code: a 管理员 ticks the capabilities in 后台「用户」
(admins.manage, server/users.py /api/admin/rights), granted() reads that row, and nothing else looks at DEPUTY.
Changing it takes effect on the next request of whoever is signed in, without a new login. Without a row (a fresh database)
granted() is DEPUTY. 「恢复默认」 deletes the row again.

There is no second, coarser view / edit / hidden switch over this: a capability already is that distinction
(templates.manage changes, stats.view only reads, no capability at all means the section is not there). The only place
the wording for it lives is CAPABILITIES below; each entry states whether ticking it grants reading or changing.

The built-in administrator account (accounts.ADMIN_ID) is always 管理员: it cannot be demoted, disabled or deleted."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .availability import CAPABILITY, Cond
from .messages import Msg


@dataclass(frozen=True)
class Capability:
    """One capability: the band it is ticked under in 后台「二级管理员权限」, its short name (one line, no brackets),
    and what holding it lets one do: the sentence the refusal message, the audit line and the tooltip all use."""

    group: str
    label: str
    what: str


def _c(group: str, label: str, what: str) -> Capability:
    return Capability(group, label, what)


# capability -> what it lets one do. The order is the order the rights sheet lists them in; `group` bands it.
CAPABILITIES: dict[str, Capability] = {
    "users.manage_normal": _c("账号", "普通用户", "管理普通用户：新建、停用和启用、重设密码、改到期时间、环节和磁盘配额"),
    "users.tags": _c("账号", "许可标签", "给账号设能用的标签（许可）：只有管理员"),
    "users.delete": _c("账号", "删除账号", "删除账号：他的结果、上传和登录一起删掉"),
    "admins.manage": _c("账号", "管理员", "新建、修改、删除管理员和二级管理员，改任何账号的角色，分配二级管理员的权限"),
    "logins.view": _c("账号", "最近登录", "看自己能管的账号的最近登录"),
    "invites.manage": _c("账号", "邀请码", "邀请码与自行注册：新建、改、停用、删除邀请码，看谁用它注册了，按邀请码或注册时间段批量停用"
                                         "自己注册的账号（只停自己管得着的）"),
    "terms.edit": _c("账号", "用户协议", "用户协议与隐私政策：看原文、改文字、恢复程序自带的文字（改了以后所有账号下次使用前要重新同意）"),
    "queue.manage": _c("队列与显卡", "队列", "看所有人的队列和任务记录，取消任何人的任务，插队，开关显卡任务和计算任务；删除别人的任务、给别人的组改名要「别人的数据」"),
    "gpu.authorize": _c("队列与显卡", "显卡授权", "授权哪些显卡接任务"),
    "farm.cards": _c("队列与显卡", "显卡详情", "显卡：每张卡的型号、显存、架构、在跑什么，扩展包在它上面能跑 / 不能跑 / 未知，参数档位和每个节点实测的显存；没有这项权限的人，回答里就没有这些"),
    "models.manage": _c("队列与显卡", "常驻模型", "常驻模型：看、卸载到内存、完全卸载"),
    "feedback.reply": _c("用户反馈", "回复反馈", "看用户反馈、标记状态、回复"),
    "feedback.delete": _c("用户反馈", "删除反馈", "删除用户反馈（连同截图和诊断资料）"),
    "templates.create": _c("模板", "管理模板", "在「模板」弹窗里管理预设模板：文件菜单「保存为预设模板」，复制、删除、开关卡片，改名字和简介，"
                                              "新建和整理模板的分类，把卡片拖到别的分类；关掉的模板也看得到"),
    "menu.edit": _c("模板", "管理节点分类", "在加节点的菜单里整理节点：新建、改名、删除、排序分类，把节点拖到别的分类，改节点的名字和说明"),
    "templates.restore": _c("模板", "找回模板", "帮用户找回他误删的模板（只能恢复，删不了）"),
    "stats.view": _c("统计", "看统计", "看使用统计（只读）"),
    "usage.reset": _c("统计", "统计清零", "使用统计清零和撤销"),
    "data.others": _c("数据与节点", "别人的数据", "看、取回、改、删别人的结果和数据：输出结果、提交的节点图、上传、队列里别人的任务和组名、别人模板的回收站和永久删除、硬盘清理"),
    "nodes.all": _c("数据与节点", "不看标签", "什么节点都能用，不看标签"),
    "server.view": _c("服务器", "概览", "概览：内存、硬盘、服务本身的地址和启动命令"),
    "settings.edit": _c("服务器", "服务器设置", "服务器设置：注册、账号配额、队列、显卡、内存、存储、日志、视图……（系统设置除外）"),
    "settings.system": _c("服务器", "系统设置", "系统设置：网络（端口、访问范围、HTTPS、域名、可信代理）、安装和编译用的镜像与编译器、数据位置、"
                                                "OCIO 配置：改了能让服务器下次安装或编译时跑别的代码、换数据的位置、信任别的来源"),
    "settings.notice": _c("服务器", "通知条", "管理员通知：编辑器和后台顶部的通知条（文字、颜色、开关）"),
    "server.restart": _c("服务器", "重启服务", "重启服务"),
    "installs.run": _c("服务器", "安装扩展", "安装扩展包、看手动下载的收件文件夹"),
    "licence.consent": _c("服务器", "同意许可", "替用户同意第三方许可协议"),
    "db.backup_restore": _c("服务器", "数据库", "数据库：看状态、备份、检查"),
    "security.manage": _c("服务器", "安全", "安全：可疑请求、解开封住的来源"),
    "logs.view": _c("服务器", "日志", "服务日志；回答里看得到服务器上的文件位置，不受请求频率限制"),
    "audit.view": _c("留底与帮助", "管理留底", "管理操作留底：谁建了谁、改了谁的哪几项、重设密码、删用户、改设置、被拒绝的尝试"),
    "openapi.view": _c("留底与帮助", "接口描述", "机器可读的 HTTP 接口描述（OpenAPI），给写插件和脚本的人"),
}

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(c.group for c in CAPABILITIES.values()))

# 二级管理员 starts with these. Only a default: once a 管理员 ticks the sheet, the `role_rights` row is what counts,
# and 「恢复默认」 comes back here.
DEPUTY: frozenset[str] = frozenset({
    "users.manage_normal", "logins.view", "queue.manage", "feedback.reply", "stats.view",
    # Lets a 二级管理员 recover templates a user deleted by mistake: restore only, not delete. Permanent deletion is
    # not here: it needs data.others, which only the primary administrator holds.
    "templates.restore",
})

# Capabilities held only by 管理员 and never assignable to 二级管理员, since assigning them would hand over the full
# administrator power (a 二级管理员 with admins.manage could create administrator accounts and grant itself every right;
# users.delete / data.others delete accounts and view or clear other users' data; users.tags governs licences).
# settings.system: what runs at the next install or build (mirrors, compilers), where data lives, whom the server
# trusts and how it is reached: handing it over hands over the machine.
OWNER_ONLY: frozenset[str] = frozenset({"admins.manage", "users.tags", "users.delete", "data.others", "settings.system"})

# Settings whose change needs a capability besides settings.edit (setting_needs; server/settings.py checks it on save
# and says per setting why a login may not change it): by key, or by section when the key ends with a dot.
# register.tags gives every future self-registered account licence tags: the same power as users.tags.
SETTING_NEEDS: dict[str, str] = {"server.": "settings.system", "install.": "settings.system", "build.": "settings.system",
                                 "paths.": "settings.system", "color.": "settings.system", "register.tags": "users.tags"}

ADMIN, DEPUTY_ROLE, USER = "admin", "deputy", "user"
ASSIGNABLE: tuple[str, ...] = (DEPUTY_ROLE,)  # the roles whose capabilities a 管理员 assigns; the other two are fixed
DEFAULTS: dict[str, frozenset[str]] = {DEPUTY_ROLE: DEPUTY}


@dataclass(frozen=True)
class Role:
    id: str
    label: str
    tip: str


ROLES: dict[str, Role] = {r.id: r for r in (
    Role(ADMIN, "管理员", "什么都能做：服务器、安装、数据库、所有账号和标签"),
    Role(DEPUTY_ROLE, "二级管理员", "管理员在「用户」里分配给他的那些：默认是普通用户、队列、反馈、模板和看统计，不碰服务器、安装、数据库、标签和别的管理员"),
    Role(USER, "普通用户", "用编辑器，只看得到自己的任务、结果、上传和反馈"),
)}
DEFAULT = USER


# ------------------------------------------------------------------ capabilities of 二级管理员: editable data

_CACHE: dict[str, tuple[float, frozenset[str]]] = {}  # role -> (the row's `updated`, what it grants)


def _row(role: str) -> Mapping[str, Any] | None:
    """That role's assigned row, or None: nobody assigned anything yet (a fresh database, or 恢复默认)."""
    from .database import db

    return db().row("SELECT * FROM role_rights WHERE role = ?", (role,))


def granted(role: str) -> frozenset[str]:
    """What `role` may do now: the 管理员's assignment, or DEFAULTS[role] while there is none. Read on every request,
    so a change takes effect at once; nobody has to log in again."""
    row = _row(role)
    if row is None:
        return DEFAULTS[role]
    at = float(row["updated"])
    hit = _CACHE.get(role)
    if hit is not None and hit[0] == at:
        return hit[1]
    from .database import json_of

    kept = frozenset(str(c) for c in json_of(row["capabilities"], []))  # written by grant(), which checks them
    _CACHE[role] = (at, kept)
    return kept


def assignable(role: object) -> str:
    """A role whose capabilities may be assigned, or Invalid (管理员 always has all, 普通用户 never any)."""
    from .errors import Invalid

    if role not in ASSIGNABLE:
        raise Invalid(Msg("E-RIGHTS-FIXEDROLE", role=label(str(role)),
                          roles=[label(r) for r in ASSIGNABLE]))
    return str(role)


def check_capabilities(given: Iterable[object]) -> frozenset[str]:
    """The capability names as kept, or Invalid naming the ones nobody declares."""
    from .errors import Invalid

    found = frozenset(str(c) for c in given)
    unknown = found - frozenset(CAPABILITIES)
    if unknown:
        raise Invalid(Msg("E-RIGHTS-UNKNOWN", names=sorted(unknown)))
    if reserved := found & OWNER_ONLY:
        raise Invalid(Msg("E-RIGHTS-OWNERONLY", names=[CAPABILITIES[c].label for c in sorted(reserved)]))
    return found


def grant(role: str, given: Iterable[object], by: str) -> frozenset[str]:
    """Assign exactly these capabilities to `role` (by: who did it, for the row; the audit line is the caller's)."""
    from .database import db, json_text

    role = assignable(role)
    kept = check_capabilities(given)
    with db().write() as c:
        c.execute("INSERT INTO role_rights (role, capabilities, updated, updated_by) VALUES (?, ?, ?, ?) "
                  "ON CONFLICT (role) DO UPDATE SET capabilities = excluded.capabilities, updated = excluded.updated, "
                  "updated_by = excluded.updated_by",
                  (role, json_text(sorted(kept)), time.time(), by))
    _CACHE.pop(role, None)
    return kept


def reset(role: str) -> frozenset[str]:
    """恢复默认: drop the assignment, so DEFAULTS[role] counts again."""
    from .database import db

    role = assignable(role)
    with db().write() as c:
        c.execute("DELETE FROM role_rights WHERE role = ?", (role,))
    _CACHE.pop(role, None)
    return DEFAULTS[role]


def sheet(role: str) -> dict:
    """One assignable role, ready to lay out: its name, every capability banded and ticked or not, how many are on,
    whether that is still the default, and who last changed it. The page renders this and works nothing out itself
    (it never groups, counts, or names a role or a capability of its own)."""
    row = _row(role)
    on = granted(role)
    groups = [{"label": g, "items": [{"id": k, "label": c.label, "what": c.what, "on": k in on}
                                     for k, c in CAPABILITIES.items() if c.group == g and k not in OWNER_ONLY]} for g in GROUPS]
    groups = [g for g in groups if g["items"]]
    return {"role": role, "label": label(role), "tip": ROLES[role].tip, "groups": groups,
            "count": len(on), "total": len(CAPABILITIES) - len(OWNER_ONLY), "default": row is None, "default_count": len(DEFAULTS[role]),
            "updated": float(row["updated"]) if row else 0.0, "updated_by": str(row["updated_by"]) if row else ""}


def check(role: object) -> str:
    """A role id as kept, or Invalid."""
    from .errors import Invalid

    if role not in ROLES:
        raise Invalid(Msg("E-ROLES-UNKNOWN", role=str(role), roles=[r.label for r in ROLES.values()]))
    return str(role)


def capabilities(role: str) -> frozenset[str]:
    """What a role may do: the one place it is worked out (管理员: everything; 二级管理员: what it was granted)."""
    if role == ADMIN:
        return frozenset(CAPABILITIES)
    if role in ASSIGNABLE:
        return granted(role)
    return frozenset()


def holders(capability: str) -> list[str]:
    """The roles that hold `capability` now, by their names: what a refusal says to go and ask."""
    return [r.label for r in ROLES.values() if capability in capabilities(r.id)]


def label(role: str) -> str:
    return ROLES[role].label if role in ROLES else role


def setting_needs(key: str) -> str | None:
    """The capability changing setting `key` needs besides settings.edit (SETTING_NEEDS; None: settings.edit alone)."""
    return next((c for k, c in SETTING_NEEDS.items() if key == k or (k.endswith(".") and key.startswith(k))), None)


def manages(caps: frozenset[str], role: str) -> bool:
    """Whether someone with `caps` may act on an account of `role` (change it, reset its password, see its logins)."""
    return "users.manage_normal" in caps if role == "user" else "admins.manage" in caps


# ------------------------------------------------------------------ conditions on who looks (availability.Cond)


@dataclass(frozen=True)
class SessionFacts:
    """What is known of whoever looks, for the conditions below: the capabilities its role reaches in this session
    (a browser's even when its three days ran out, since they come back with the password; a client's token: none), whether
    those rights ran out (lapsed), and the account row it looks at (server/available.py ACCOUNT)."""

    capabilities: frozenset[str] = frozenset()
    lapsed: bool = False
    account: Mapping[str, Any] = field(default_factory=dict)
    user_id: int = 0  # who looks (0: nobody logged in)

    @classmethod
    def of(cls, s, account: Mapping[str, Any] | None = None) -> SessionFacts:
        """From an accounts.Session (None: nobody logged in)."""
        if s is None:
            return cls(account=account or {})
        return cls(s.user.capabilities if s.lapsed else s.capabilities, s.lapsed, account or {}, int(s.user.id))


@dataclass(frozen=True)
class Can(Cond):
    """The session's role has this capability: not its tool otherwise (hidden)."""

    capability: str
    kind = CAPABILITY

    def holds(self, f: SessionFacts) -> bool:
        return self.capability in f.capabilities


@dataclass(frozen=True)
class Manages(Cond):
    """The session manages the role of the account row it looks at (manages): not its account otherwise (hidden)."""

    kind = CAPABILITY

    def holds(self, f: SessionFacts) -> bool:
        return manages(f.capabilities, str(f.account.get("role", "")))


@dataclass(frozen=True)
class RightsFresh(Cond):
    """The rights are not run out: greyed otherwise, the password again brings them back."""

    def holds(self, f: SessionFacts) -> bool:
        return not f.lapsed

    def why(self, f: SessionFacts, words: Any) -> Msg:
        return Msg("N-ACCESS-RIGHTSAGAIN")


@dataclass(frozen=True)
class NotOwnersAccount(Cond):
    """The account row is not the built-in administrator account (accounts.ADMIN_ID, always 管理员): greyed otherwise,
    since it can never be changed that way (E-ACCOUNT-ADMINFIXED), with `delete` deleted (E-ACCOUNT-ADMINDELETE), or,
    with `password`, have its password reset by anyone but itself (E-ACCOUNT-ADMINPASSWORD)."""

    delete: bool = False
    password: bool = False  # the built-in administrator resets their own (server/users.py reset); nobody else does

    def holds(self, f: SessionFacts) -> bool:
        if self.password and f.account.get("owner"):
            return f.account.get("id") == f.user_id
        return not f.account.get("owner")

    def why(self, f: SessionFacts, words: Any) -> Msg:
        code = "E-ACCOUNT-ADMINDELETE" if self.delete else "E-ACCOUNT-ADMINPASSWORD" if self.password else "E-ACCOUNT-ADMINFIXED"
        return Msg(code, username=f.account.get("username", ""))


def describe() -> dict:
    """For the rights checkbox table in 后台「用户」: every capability (its band, short name and what it lets one do)
    and every role with what it may do now. 后台「二级管理员权限」 reads sheet() instead, which is already laid out."""
    return {"capabilities": [{"id": k, "group": c.group, "label": c.label, "what": c.what} for k, c in CAPABILITIES.items()],
            "groups": list(GROUPS),
            "roles": [{"id": r.id, "label": r.label, "tip": r.tip, "capabilities": sorted(capabilities(r.id)),
                       "assignable": r.id in ASSIGNABLE} for r in ROLES.values()]}
