"""What of the admin side is there for a session: its subjects and their conditions, declared once as data (the one
availability mechanism, lab2shot/availability.py, resolves them; lab2shot/roles.py gives the conditions on who looks).

    SECTIONS  a section of the admin page (webui/src/admin/sections.tsx, by id): route(the route it reads)
    ACTIONS   what the pages offer besides reading a section: route(the route it calls), or capability(...) for what is
              only a field of a route (licence tags, a role)
    PAGES     a page as a whole: AnyOf its sections (there when any of them is; greyed when the best of them is)
    ACCOUNT   what can be done to one account (a row of 用户): Manages() its role, the route, and NotOwnersAccount(why)
              for what the owner's account never allows

route(key) is the route's own capability (declared on the route, server/routes.py, never written twice) and the rights still
fresh: a role without the capability does not see it (hidden), a browser whose three days ran out sees it greyed with
why (N-ACCESS-RIGHTSAGAIN). The answer (availability.Availability.json(), under `applies` in the login state and on
every row of 用户) is read by the web page through one module (webui/src/api/applies.ts); no component looks at a role.

A branch adding a control declares it here: the queue's cancel button is ACTIONS["queue.cancel"]; the installer
declares its own subjects and conditions and calls availability.resolve the same way."""

from __future__ import annotations

import re
from typing import Any


from dataclasses import dataclass
from ..messages import Msg
from ..accounts import Session
from ..availability import DATA, All, AnyOf, Availability, Cond, resolve
from ..roles import Can, Manages, NotOwnersAccount, RightsFresh, SessionFacts
from . import routes

FRESH = RightsFresh()


def capability(cap: str) -> Cond:
    """What needs `cap`: the role has it, and its rights are fresh."""
    return All(Can(cap), FRESH)


class RouteCan(Can):
    """Can, for the capability a route declares on itself (server/routes.py Access), looked up when it is asked: the
    subject names the route, and the capability stays written once, on the route (the modules declaring routes may
    load after this one)."""

    def __init__(self, key: str) -> None:
        object.__setattr__(self, "key", key)

    @property
    def capability(self) -> str:  # type: ignore[override]
        if self.key not in routes.DECLARED:  # asked before the server's modules declared their routes (a script, a test)
            from . import app  # noqa: F401
        return routes.DECLARED[self.key].needs


def route(key: str) -> Cond:
    """What calls or reads a route ("GET /api/admin/queue", "POST /api/admin/installs"): the
    capability the route declares, and fresh rights."""
    return All(RouteCan(key), FRESH)


SECTIONS: dict[str, Cond] = {
    "overview": route("GET /api/admin/overview"),
    "queue": route("GET /api/admin/queue"),
    "cards": route("GET /api/admin/cards"),
    "resident": route("GET /api/admin/resident"),
    "usage": route("GET /api/admin/usage"),
    "feedback": route("GET /api/admin/feedback"),
    "users": route("GET /api/admin/users"),
    # 「扩展包」：装扩展在后台这一区（读写的路由一律要 installs.run，只有管理员装得了）
    "extensions": route("GET /api/admin/extensions"),
    "disk": route("GET /api/admin/disk"),
    "database": route("GET /api/admin/db"),
    "security": route("GET /api/admin/security"),
    "logs": route("GET /api/admin/log"),
    "settings": route("GET /api/admin/settings"),
}

ACTIONS: dict[str, Cond] = {
    # 机器可读的接口描述（写插件和脚本的人用）
    "openapi": route("GET /api/admin/openapi.json"),
    "server.status": route("GET /api/admin/overview"),
    "server.restart": route("POST /api/admin/restart"),
    "settings.notice": route("PUT /api/admin/notice"),
    "queue.cancel": route("POST /api/admin/jobs/{job_id}/cancel"),
    "queue.switches": route("PUT /api/admin/queue/switches"),
    # 拖拽插队: the queue table is draggable only for a login the server says may move a job
    "queue.reorder": route("POST /api/admin/jobs/{job_id}/place"),
    "queue.authorize": route("PUT /api/admin/gpus"),
    "queue.graph": route("GET /api/admin/jobs/{job_id}/graph"),
    "users.create": route("POST /api/admin/users"),
    # 模板的后台管理：改分类树、把一张节点图录入成模板
    "categories.edit": route("PUT /api/admin/categories"),
    "categories.remove": route("DELETE /api/admin/categories/{cid}"),
    "templates.delete": route("DELETE /api/admin/templates/{template_id}"),
    "templates.switch": route("PUT /api/admin/templates/{template_id}"),
    "templates.place": route("PUT /api/admin/templates/{template_id}/place"),
    "templates.copy": route("POST /api/admin/templates/{template_id}/copy"),
    # 录入模板做在前台：编辑器右上角头像菜单里那一项，把当前这张节点图直接录成一张卡片
    "templates.create": route("POST /api/admin/templates"),
    "menu.edit": route("PUT /api/admin/menu/categories"),
    "menu.place": route("PUT /api/admin/menu/nodes/{type_id}/place"),
    "menu.order": route("PUT /api/admin/menu/categories/order"),
    "menu.text": route("PUT /api/admin/menu/nodes/{type_id}/text"),
    "templates.edit": route("PUT /api/admin/templates/{template_id}/text"),
    "categories.order": route("PUT /api/admin/categories/order"),
    # 二级管理员的权限由一级管理员分配（server/users.py /api/admin/rights）：后台「用户」里的那张勾选表
    "rights.edit": route("PUT /api/admin/rights"),
    "users.tags": capability("users.tags"),
    # 管理操作: a tab of the account page, not a route of its own — the registry checks it per kind
    "audit.view": capability("audit.view"),
    "users.role": capability("admins.manage"),
    "feedback.delete": route("DELETE /api/admin/feedback/{fid}"),
    "usage.reset": route("POST /api/admin/usage/reset"),
    "help.install": route("POST /api/admin/installs"),
    "help.installprogress": route("GET /api/admin/installs/{job_id}"),
    "help.installcancel": route("POST /api/admin/installs/{job_id}/cancel"),
    "help.preflight": route("GET /api/admin/extensions/{name}/preflight"),
    "help.rollback": route("POST /api/admin/extensions/{name}/rollback"),
    "help.uninstall": route("POST /api/admin/extensions/{name}/uninstall"),
    "help.manual": route("GET /api/admin/manual"),  # the admin page's 「扩展包」 手动下载 list with the inbox (installer and hand downloads)
    "help.consent": route("POST /api/admin/manual/accept"),  # accepting a hand download's licence
}

PAGES: dict[str, Cond] = {"page.admin": AnyOf(*SECTIONS.values())}

_FIXED = NotOwnersAccount()  # the owner's account: always 管理员, never expires or is disabled

ACCOUNT: dict[str, Cond] = {
    "account.edit": All(Manages(), route("PUT /api/admin/users/{user_id}")),
    "account.expiry": All(Manages(), route("PUT /api/admin/users/{user_id}"), _FIXED),
    "account.enable": All(Manages(), route("PUT /api/admin/users/{user_id}"), _FIXED),
    "account.tags": All(Manages(), capability("users.tags"), _FIXED),
    "account.role": All(Manages(), capability("admins.manage"), _FIXED),
    "account.password": All(Manages(), route("POST /api/admin/users/{user_id}/password"), NotOwnersAccount(password=True)),
    "account.delete": All(Manages(), route("DELETE /api/admin/users/{user_id}"), NotOwnersAccount(delete=True)),
    "account.logins": All(Manages(), route("GET /api/admin/users/{user_id}/logins")),
    # 磁盘配额（在用户管理页面里）：看这个账号占了多少、按账号改上限
    "account.quota": All(Manages(), route("GET /api/admin/users/{user_id}/quota")),
    "account.quota_set": All(Manages(), route("PUT /api/admin/users/{user_id}/quota"), _FIXED),
}


@dataclass(frozen=True)
class Usable(Cond):
    """A node type's project is usable now: its extension installed, ready and built from this code (lab2shot/adapters.py
    ProjectFacts.available); greyed otherwise, with what the project says (E-EXT-NOTINSTALLED, E-EXT-OUTDATED ...)."""

    said: Msg | None
    kind = DATA

    def holds(self, f: Any) -> bool:
        return self.said is None

    def why(self, f: Any, words: Any) -> Msg:
        return self.said  # type: ignore[return-value]


def nodes(s: Session | None) -> dict[str, Cond]:
    """node:<type id> for every node type the session's account may use (its licence tags: server/access.py
    node_types_for); a type its tags do not allow is no subject at all, so the page never shows it. Each project is
    looked at once."""
    if s is None:
        return {}
    from .access import node_types_for

    said: dict[str, Msg | None] = {}
    subjects: dict[str, Cond] = {}
    for t in node_types_for(s.user).values():
        if t.runtime not in said:
            said[t.runtime] = t.project.available()
        subjects[f"node:{t.id}"] = Usable(said[t.runtime])
    return subjects


def session(s: Session | None) -> Availability:
    """The sections, actions and pages of this session, and the node types its account may use (node:<type id>)."""
    return resolve({**SECTIONS, **ACTIONS, **PAGES, **nodes(s)}, SessionFacts.of(s))


# the admin page's 「扩展包」 actions on one extension (server/installs.py facts): the same declared subjects, resolved by the
# same one mechanism — a session without installs.run sees nothing of it (the card says 未安装 only)
EXTENSION: dict[str, Cond] = {
    "install": route("POST /api/admin/installs"),
    "preflight": route("GET /api/admin/extensions/{name}/preflight"),
    "progress": route("GET /api/admin/installs/{job_id}"),
    "cancel": route("POST /api/admin/installs/{job_id}/cancel"),
    "rollback": route("POST /api/admin/extensions/{name}/rollback"),
    "uninstall": route("POST /api/admin/extensions/{name}/uninstall"),
}


def extension(s: Session | None, facts: dict) -> dict:
    """What this session may do to one extension on the admin page's 「扩展包」 (server/installs.py facts: its latest install job,
    installed, an environment kept for rollback, jobs using it now). Nothing at all for a session without installs.run:
    such an account sees an uninstalled card with 「未安装」 only. Cancel and the progress are there only while there
    is a job to cancel or show; the rest greyed with why (an install of it running now, nothing to roll back to, no
    environment to remove, jobs using it)."""
    caps = s.capabilities if s is not None else frozenset()
    job = facts.get("job")
    active = job is not None and job["state"] in ("queued", "running")
    queued = Msg("N-INSTALL-QUEUED", title=facts["title"]) if active else None
    available: list[str] = []
    inactive: dict[str, dict] = {}
    who = SessionFacts.of(s)
    for id_, cond in EXTENSION.items():
        if not cond.holds(who):
            continue
        if (id_ == "cancel" and not active) or (id_ == "progress" and job is None):
            continue
        why = None
        if id_ == "install":
            why = queued
        elif id_ == "rollback":
            why = queued or (None if facts["previous"] else Msg("N-INSTALL-NOPREVIOUS", title=facts["title"]))
        elif id_ == "uninstall":
            why = queued or (None if facts["installed"] else Msg("N-INSTALL-NOTINSTALLED", title=facts["title"])) or (
                Msg("N-INSTALL-INUSE", title=facts["title"], count=facts["busy"]) if facts["busy"] else None)
        if why is None:
            available.append(id_)
        else:  # greyed with why, like every other subject (an install running now: N-INSTALL-QUEUED)
            inactive[id_] = why.json()
    return {"available": available, "inactive": inactive}




# ------------------------------------------------------------------ fields of answers
#
# A field of an answer that not everyone gets is a subject too: its condition declared once here, resolved like every
# other subject; where the field is, each route says in its own declaration (server/routes.py Access `hides`: subject ->
# JSON paths). The guard (server/access.py) takes every field whose subject is not available for the session out of the
# answer before it leaves, so no response builder decides who sees what, and a page never gets what it may not show.

FIELDS: dict[str, Cond] = {
    # the cards (artists never see or choose cards; card and verification information is the administrator's): card
    # names, VRAM measured on a card, what runs on which card. Everyone else is told only 「排队中」 unless somebody has
    # to act (farm/queue.py `_told`: N-QUEUE-NOMACHINEEVER / NOCARD / PAUSED, none of which names a card). The routes
    # carrying them: their `hides`.
    "farm.cards": capability("farm.cards"),
    # 测出来的结论（模板上的结论、节点口和参数上的 `finding`）：只给管得了模板的账号看，其他账号的模板、节点、
    # 参数上都不带
    "templates.create": capability("templates.create"),
    "menu.edit": capability("menu.edit"),
}


def hidden_paths(s: Session | None, method: str, path: str) -> tuple[str, ...]:
    """The JSON paths of the answer to `method path` this session does not get (its route's `hides`, for the subjects
    of FIELDS not available to the session)."""
    found = routes.match(method, path)
    if found is None or not found[1].hides:
        return ()
    shown = set(resolve(FIELDS, SessionFacts.of(s)).available)
    return tuple(p for name, paths in found[1].hides.items() if name not in shown for p in paths)


_PATH = re.compile(r"\[\]|[^.\[\]]+")


def strip(value: Any, path: str) -> Any:
    """`value` without the field at `path` (in place; also returned)."""
    parts = _PATH.findall(path)

    def go(v: Any, i: int) -> None:
        part, last = parts[i], i == len(parts) - 1
        if part == "[]":
            for item in v if isinstance(v, list) else ():
                go(item, i + 1)
        elif part == "*":
            for item in list(v.values()) if isinstance(v, dict) else ():
                go(item, i + 1)
        elif isinstance(v, dict) and part in v:
            if last:
                del v[part]
            else:
                go(v[part], i + 1)

    go(value, 0)
    return value


def account(s: Session, row: dict) -> Availability:
    """What this session may do to one account (a row of accounts.listing())."""
    return resolve(ACCOUNT, SessionFacts.of(s, row))


def resource_acts(s: Session | None, acts: list[dict]) -> list[dict]:
    """The row actions of one resource this session may use (lab2shot/resources.py Act: 恢复, 永久删除 …). Each act
    names the route it calls, and that route declares the capability it needs — resolved here by the same one
    mechanism as every other button, so the capability stays written once and the page gets only what it may use."""
    who = SessionFacts.of(s)
    kept = [a for a in acts if route(a["route"]).holds(who)]
    return [{k: v for k, v in a.items() if k != "route"} for a in kept]
