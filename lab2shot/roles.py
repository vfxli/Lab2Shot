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
from typing import TYPE_CHECKING, Any

from . import i18n
from .availability import CAPABILITY, Cond
from .messages import Msg

if TYPE_CHECKING:
    from .accounts import Actor


@dataclass(frozen=True)
class Capability:
    """One capability: the band it is ticked under in the back office's deputy rights sheet, and, in the catalogue
    (lab2shot/i18n/<lang>/roles.toml, read in the language now), its short name (right.<id>.label: one line, no
    brackets) and what holding it lets one do (right.<id>.what: the sentence the refusal message, the audit line and
    the tooltip all use). The band's name: right.band.<band>."""

    id: str
    band: str

    @property
    def group(self) -> str:
        """Its band's name."""
        return i18n.t(f"right.band.{self.band}")

    @property
    def label(self) -> str:
        return i18n.Word(f"right.{self.id}.label")  # every language: a message says it in its reader's

    @property
    def what(self) -> str:
        return i18n.t(f"right.{self.id}.what")

    @property
    def what_word(self) -> i18n.Word:
        """`what` kept as its word: said in whoever's language reads it (an audit row, i18n.Word)."""
        return i18n.Word(f"right.{self.id}.what")


# capability -> what it lets one do. The order is the order the rights sheet lists them in; `group` bands it.
CAPABILITIES: dict[str, Capability] = {
    "users.manage_normal": Capability("users.manage_normal", "accounts"),
    "users.tags": Capability("users.tags", "accounts"),
    "users.delete": Capability("users.delete", "accounts"),
    "admins.manage": Capability("admins.manage", "accounts"),
    "logins.view": Capability("logins.view", "accounts"),
    "invites.manage": Capability("invites.manage", "accounts"),
    "terms.edit": Capability("terms.edit", "accounts"),
    "queue.manage": Capability("queue.manage", "queue"),
    "gpu.authorize": Capability("gpu.authorize", "queue"),
    "farm.cards": Capability("farm.cards", "queue"),
    "models.manage": Capability("models.manage", "queue"),
    "feedback.reply": Capability("feedback.reply", "feedback"),
    "feedback.delete": Capability("feedback.delete", "feedback"),
    "templates.create": Capability("templates.create", "templates"),
    "menu.edit": Capability("menu.edit", "templates"),
    "templates.restore": Capability("templates.restore", "templates"),
    "stats.view": Capability("stats.view", "stats"),
    "usage.reset": Capability("usage.reset", "stats"),
    "data.others": Capability("data.others", "data"),
    "nodes.all": Capability("nodes.all", "data"),
    "server.view": Capability("server.view", "server"),
    "settings.edit": Capability("settings.edit", "server"),
    "settings.system": Capability("settings.system", "server"),
    "settings.notice": Capability("settings.notice", "server"),
    "server.restart": Capability("server.restart", "server"),
    "installs.run": Capability("installs.run", "server"),
    "licence.consent": Capability("licence.consent", "server"),
    "db.backup_restore": Capability("db.backup_restore", "server"),
    "security.manage": Capability("security.manage", "server"),
    "logs.view": Capability("logs.view", "server"),
    "audit.view": Capability("audit.view", "audit"),
    "openapi.view": Capability("openapi.view", "audit"),
}

GROUPS: tuple[str, ...] = tuple(dict.fromkeys(c.band for c in CAPABILITIES.values()))  # the bands, in order

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
# users.delete / data.others delete accounts and view or clear other users' data; users.tags and nodes.all govern
# licences: who may use which node, and using every node whatever its tags).
# settings.system: what runs at the next install or build (mirrors, compilers), where data lives, whom the server
# trusts and how it is reached: handing it over hands over the machine.
OWNER_ONLY: frozenset[str] = frozenset({"admins.manage", "users.tags", "nodes.all", "users.delete", "data.others",
                                        "settings.system"})

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
    """A role; its name and what it may do, in the catalogue: right.role.<id>.label / .tip."""

    id: str

    @property
    def label(self) -> str:
        return i18n.Word(f"right.role.{self.id}.label")  # every language: a message says it in its reader's

    @property
    def tip(self) -> str:
        return i18n.t(f"right.role.{self.id}.tip")


ROLES: dict[str, Role] = {r.id: r for r in (Role(ADMIN), Role(DEPUTY_ROLE), Role(USER))}
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

    # written by grant(), which checks them; a capability the code no longer declares, or no longer lets anyone but the
    # 管理员 hold (OWNER_ONLY), is dropped here, so it is neither held nor counted on the 权限 page, nor sent back with the
    # ticks and refused (check_capabilities) when that role is next saved
    kept = frozenset(str(c) for c in json_of(row["capabilities"], [])) & (frozenset(CAPABILITIES) - OWNER_ONLY)
    _CACHE[role] = (at, kept)
    return kept


def assignable(role: object) -> str:
    """A role whose capabilities may be assigned, or Invalid (管理员 always has all, 普通用户 never any)."""
    from .errors import Invalid

    if role not in ASSIGNABLE:
        raise Invalid(Msg("E-RIGHTS-FIXEDROLE", role=i18n.Both.of(lambda: label(str(role))),
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


def grant(role: str, given: Iterable[object], by: "Actor") -> frozenset[str]:
    """Assign exactly these capabilities to `role` (by: who did it, for the row; the audit line is the caller's)."""
    from .database import db, json_text

    role = assignable(role)
    kept = check_capabilities(given)
    with db().write() as c:
        c.execute("INSERT INTO role_rights (role, capabilities, updated, updated_by, updated_by_id) VALUES (?, ?, ?, ?, ?) "
                  "ON CONFLICT (role) DO UPDATE SET capabilities = excluded.capabilities, updated = excluded.updated, "
                  "updated_by = excluded.updated_by, updated_by_id = excluded.updated_by_id",
                  (role, json_text(sorted(kept)), time.time(), by.label, by.id))
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
    groups = [{"label": i18n.t(f"right.band.{g}"), "items": [{"id": k, "label": c.label, "what": c.what, "on": k in on}
                                     for k, c in CAPABILITIES.items() if c.band == g and k not in OWNER_ONLY]} for g in GROUPS]
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


def holders(capability: str) -> list[i18n.Word | str]:
    """The roles that hold `capability` now, by their words (word): what a refusal says to go and ask."""
    return [word(r.id) for r in ROLES.values() if capability in capabilities(r.id)]


def label(role: str) -> str:
    return ROLES[role].label if role in ROLES else role


def word(role: str) -> i18n.Word | str:
    """A role as a message's parameter: kept by its id (the word right.role.<id>.label), said in whoever's language
    reads the message, so an audit row reads in its reader's language, not its writer's; a role no longer known, as
    it was stored."""
    return i18n.Word(f"right.role.{role}.label") if role in ROLES else role


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
class Staff(Cond):
    """The session's role has any capability (管理员, or a 二级管理员 given some): the back office is its."""

    kind = CAPABILITY

    def holds(self, f: SessionFacts) -> bool:
        return bool(f.capabilities)


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
