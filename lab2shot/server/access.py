"""Who may reach what through this server's port, and what of it is theirs: every rule is here, in one place.

The server may be shared over the internet (a port-forwarding tunnel; every request may then come from this machine's
own address, so an address proves nothing). Three levels of route:

    open   anyone: the login page and the few files it is made of (the gate), logging in, the administrator's
           recovery with the 口令, the certificate users install for HTTPS, whether the server is up (Access.open, Access.page)
    user   whoever logged in with an account (server/auth.py; or the command line on this machine): the editor,
           the queue, their own jobs, uploads, results and feedback, and the rest of the web page's code (Access.user)
    admin  an account whose role has rights (lab2shot/roles.py: 管理员, 二级管理员), with the password typed within the
           last three days: the code of the admin page, and each route under /api/admin/ with the
           capability it declares (Access.admin); what the pages then show of it is server/available.py's

Any other /api/ path is not served: a route answers only as it declares itself where it is registered (server/routes.py, once;
fail closed). Outside /api/ the same: only the page's entries (PAGE_ENTRY: /, /admin,
and below them) answer, with index.html; the built page's files (webui/dist/assets) take their
level from the build's manifest (the gate's are open, the admin page's are admin, everything else is the
user's), and a file the manifest does not name is not served at all.

What is a user's own, and only theirs and whoever holds the right its route declares for other people's data
(others_right: data.others for all but the few admin routes that name a narrower one, such as
templates.restore for giving a user back a template they deleted):
  - jobs, task outputs, feedback: by the account they were made under (mine);
  - uploads: each account's own, in its own folder (transfer/uploads.py): the same bytes sent by two accounts are
    kept twice, and knowing a sha or an upload id is never enough to use or even learn of another's;
  - results (packets): each account's own cache (data/store.py <数据位置>/cache/<account id>/): a packet is readable
    when its fingerprint is there, in the requesting account's own cache (readable); the same graph cooked by
    two accounts is cooked and kept twice, and a fingerprint of another account's is simply not there;
  - node types: only those whose tags the account may use (nodes/tags.py). The others do not exist for it: each
    answer that describes node types, choices or templates is made for the account (node_types_for, describe_for,
    templates_for; the catalogue's menu places only those), and a graph with one is refused like one with a type this
    server lacks (admit). Nothing filters the words of an answer after it is made: what an account wrote itself (its
    file names, template names, notes) comes back exactly as it wrote it, whatever those words are.

Guard, the one middleware in front of everything, applies it to every request, and also
  - lets an account that has not agreed to the current 用户协议 and 隐私政策 (lab2shot/terms owed: one an
    administrator made, or anyone after the texts changed) reach nothing but the login's own routes (/api/auth/:
    agreeing, logging out) and the open ones, until it agrees (403 marked `terms`: the page asks for it);
  - counts requests per client (auth.Rate, auth.client_key: a session, else where it comes from, else its connection),
    every session's alike: a flood is refused (429). A route with a rate of its own (its declared Limit: logging in,
    registering) is counted in its own bucket instead, so what fills the shared one never refuses it;
  - refuses every write that relies on the cookie (or on no credential: logging in) and did not come from a page of
    this site (403: its Origin, or Referer, names another site or none); a bearer or machine token in a header is
    never sent by another site's page;
  - refuses a request body larger than MAX_BODY, or its route's own limit (its declared Limit), and one whose size it is
    not told (chunked), whatever the method, before any of it is read; a route that reads its own bytes as they come
    (Limit body None: an upload's part) counts them against the size it declared, and nothing reads them before it
    (routes.Route reads a body only for a route that takes one, and never past its Limit);
  - adds the security headers (HEADERS) to every answer, and over HTTPS (this server's, or the tunnel's in front of
    it) to a named host also HSTS: the browser then never asks it over plain http;
  - takes this server's own folders out of what anyone without logs.view gets back (scrub): an error, a job's
    log, a result's description never show where the server keeps things, only file names;
  - notes what looks like probing (auth.Watch, the admin page's 安全): refused routes, odd paths, floods;
  - says whose work a request is while it is served (lab2shot/serving.py): the session's account, or every account
    for one holding data.others; transfer/uploads.py resolve answers by it. Nothing else in the server asks whose
    an upload is.
"""

from __future__ import annotations

import gzip
import json
import re
import time
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import compile_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import i18n, logs, roles, terms
from ..accounts import User
from ..availability import resolve
from ..config import ROOT, WEBUI_DIST, settings
from ..database import db, json_text
from ..engine.graph import Graph, GraphError
from ..errors import (Forbidden, Invalid, LengthRequired, MessageError, Misdirected, NotFound, NotSignedIn,
                      TooLarge, TooManyTries)
from ..messages import Msg, wire as kept
from ..text import decimal
from ..nodes import tags
from ..traffic import SCOPE_USER
from . import auth, lang, wire
from .repeats import Repeats
from .routes import ADMIN, MAX_BODY, MAX_NODES, Access, declared_as, match
from .wire import off_loop
from .words import Word

# Each route's level, capability and limits are declared with the route, once (server/routes.py Access).

HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Content-Security-Policy": "; ".join((
        "default-src 'self'", "script-src 'self'", "style-src 'self' 'unsafe-inline'", "img-src 'self' data: blob:",
        "font-src 'self' data:", "connect-src 'self'", "media-src 'self' data: blob:", "worker-src 'self' blob:",
        "object-src 'none'", "base-uri 'self'", "form-action 'self'", "frame-ancestors 'none'")),
}
SCRUBBED = ("application/json", "text/event-stream", "text/plain")
AUTH = "/api/auth/"  # the login's own routes: answered before an account agreed to the texts (logging in and out, agreeing)
VERBATIM = compile_path("/api/tasks/{task_id}/outputs/{pkg}/file/{name:path}")[0]  # the user's own files, as they are


ROUTE = "lab2shot_route"  # the scope's record of the route a request is for (route_of)


def route_of(request: Request) -> tuple[str, Access] | None:
    """The route a request is for and its declared access (routes.match; None: no route of that method and path),
    looked up once per request and kept on its scope: the guard and everything after it read this one. A HEAD is its
    GET's, as FastAPI answers it (the GET's handler, without the body)."""
    if ROUTE not in request.scope:
        request.scope[ROUTE] = match(declared_as(request.scope["method"]), request.scope["path"])
    return request.scope[ROUTE]


audit_log = logs.get("admin")


DONE = "done"  # admin_actions.status: an id (its words server.audit.<status>)
REFUSED = "refused"
FAILED = "failed"
STATUS_OF = {"I": DONE, "W": REFUSED, "E": FAILED}  # an audit message's level -> how the action went

# the audit rows written while one admin write is served (Guard): the route's own, where the action was done or
# refused; the guard writes one only when the route wrote none, from how the request ended
_audited: ContextVar[list[str] | None] = ContextVar("lab2shot_audited", default=None)


REPEAT_S = 600  # the same refusal of one account at one address within this long is one row, counted (audit)


def audit(message: Msg, session=None, method: str = "", path: str = "", about: int | None = None) -> None:
    """One admin action, written in one place to both records: the server log (with its code: who, role,
    what, on what, as the message states) and the `admin_actions` table, which only ever grows and has no route that
    changes or removes a row. The table is what 后台「用户」详情页's 管理操作 tab lists, per account. How it went is the
    message's level (STATUS_OF): an action is written where it was done (I), refused (W) or failed (E), once.

    A refusal repeated (the same account, the same refusal, the same address within REPEAT_S: a script hammering a
    route it may not use, a flood the guard slows) is counted on the row of its first (`repeats`, `last`: server/repeats.py
    tells the count within its FLUSH_S, and when the process ends), and says nothing more in the log.

    `session`: whose action it was (None: nobody was logged in, i.e. a refused attempt). `method`/`path`: the request it
    came from ("" for an action from the command line). `about`: the account it was done to, by id (permanent deletion
    finds the rows about an account by it: accounts.py PURGE)."""
    if (written := _audited.get()) is not None:
        written.append(message.code)
    user = getattr(session, "user", None)
    row = (time.time(), getattr(user, "id", None), getattr(user, "role", ""), message.code,
           json_text({k: kept(v) for k, v in message.params.items()}), method, path, STATUS_OF[message.level], about)
    if message.level == "W":
        _refusals.happened((row[1], message.code, method, path), {"row": row, "message": message})
    else:
        _write(row, message)


def _write(row: tuple, message: Msg) -> int | None:
    logs.say(audit_log, message)
    try:
        with db().write() as c:
            return c.execute("INSERT INTO admin_actions (at, user_id, role, code, params, method, path, status, target_id) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row).lastrowid
    except Exception as exc:  # the log line above is the record that must not be lost: never fail the action itself
        logs.say(audit_log, Msg("W-AUDIT-NOTKEPT", code=message.code, why=str(exc)))
        return None


def _count_repeats(row: int, count: int, t: float) -> None:
    try:
        with db().write() as c:
            c.execute("UPDATE admin_actions SET repeats = repeats + ?, last = ? WHERE id = ?", (count, t, row))
    except Exception as exc:  # a count is not worth failing a request for
        logs.say(audit_log, Msg("W-AUDIT-NOTKEPT", code="repeats", why=str(exc)))


_refusals = Repeats(REPEAT_S, lambda key, t, said: _write(said["row"], said["message"]), _count_repeats)


def ended(s, need: str, method: str, path: str, status: int) -> Msg:
    """An admin write whose route wrote no audit of its own, as it ended: done, refused (4xx) or failed (5xx)."""
    who, role = named(s)
    code = "I-AUDIT-ACTION" if status < 400 else "W-AUDIT-ACTIONREFUSED" if status < 500 else "E-AUDIT-ACTIONFAILED"
    return Msg(code, who=who, role=role, what=roles.CAPABILITIES[need].what_word, method=method, path=path, status=status)


def named(s) -> tuple[str, str]:
    """Who a session is and its role, as an audit line's words say them (`who`, `role`)."""
    return (s.user.label, roles.word(s.user.role)) if s is not None else (Word("server.not_signed_in"), "")
ADMIN_PAGES = ("src/admin/AdminApp.tsx",)  # the build's entries of the admin's own pages


@lru_cache(maxsize=4)
def _asset_levels(manifest: Path, mtime: float) -> dict[str, str]:
    """/assets/<file> -> its level, from the build's manifest (webui/dist/.vite/manifest.json): what the entry (the
    gate) imports is open; what the page's code imports, the user's; what only the admin pages import, the admin's."""
    chunks = json.loads(manifest.read_text(encoding="utf-8"))

    def closure(keys) -> set[str]:
        seen, todo = set(), list(keys)
        while todo:
            k = todo.pop()
            if k in seen or k not in chunks:
                continue
            seen.add(k)
            todo += chunks[k].get("imports", [])
        return seen

    admin_roots = [k for k in chunks if k.endswith(ADMIN_PAGES)]
    gate = closure(k for k, c in chunks.items() if c.get("isEntry"))
    page = closure(k for k, c in chunks.items() if c.get("isDynamicEntry") and k not in admin_roots)
    admin = closure(admin_roots)
    rank = {"open": 0, "user": 1, "admin": 2}
    levels: dict[str, str] = {}
    for k, c in chunks.items():
        where = "open" if k in gate else "user" if k in page else "admin" if k in admin else "user"
        for f in [c["file"], *c.get("css", []), *c.get("assets", [])]:  # a file two levels need: the lower one's
            path = "/assets/" + f.removeprefix("assets/")
            if rank[where] < rank.get(levels.get(path, ""), 3):
                levels[path] = where
    return levels


def asset_level(path: str) -> str | None:
    """The level of a file of the built page; None for one the build's manifest does not name (not served: nothing
    else in that folder is ever sent, and without a build there is no page file at all)."""
    manifest = WEBUI_DIST / ".vite" / "manifest.json"
    if not manifest.is_file():
        return None
    return _asset_levels(manifest, manifest.stat().st_mtime).get(path)


# the addresses the page answers (webui/src/site.tsx): the editor at /, and the admin page with
# whatever it adds below it. Every other address outside /api/ and /assets/ is not served: letting an address
# through that the page has no view for would send the editor's shell to a page that does not exist.
PAGE_ENTRY = re.compile(r"^/(?:admin(?:/.*)?)?$")


def level(method: str, path: str, found: tuple[str, Access] | None) -> str | None:
    """"open", "user", "admin", or None (not served). `found`: the request's route (route_of)."""
    if path.startswith(ADMIN):
        return "admin"
    if path.startswith("/assets/"):
        return asset_level(path)
    if not path.startswith("/api/"):
        if method not in ("GET", "HEAD"):
            return None
        # the page's entry (index.html), or a page address of its own declared at a fixed path (Access.page: /embed,
        # server/auth.py), never the entry's catch-all
        fixed = found is not None and found[1].level == "page" and "{" not in found[0]
        return "open" if PAGE_ENTRY.match(path) or fixed else None
    return found[1].level if found is not None and found[1].level in ("open", "user") else None


def same_site(request: Request) -> bool:
    """Whether the request came from a page of this site: its Origin (or, without one, its Referer) names the host it was
    sent to (or the host a trusted proxy says it forwarded: auth.host)."""
    source = request.headers.get("origin") or request.headers.get("referer") or ""
    host = urlsplit(source).netloc.lower() if source and source != "null" else ""
    here = {request.headers.get("host", "").lower(), auth.host(request)}
    return bool(host) and host in here - {""}


def refused(error: MessageError, **extra) -> JSONResponse:
    """The guard's answer to a request it does not let through: the error's own answer and status, as every error is
    answered (errors.py MessageError, server/app.py), and what the page acts on (login, admin, kicked, terms)."""
    return JSONResponse({**error.answer(), **extra}, status_code=error.status)


# ------------------------------------------------------------------ what is whose


def account_of(s):
    """Whose work a request is (lab2shot/serving.py Account): its own account's; every account's for a session
    holding data.others (the back office reads a user's graph, job and results as that user has them). A request
    without a session (not signed in) owns nothing."""
    from ..serving import NOBODY, Account

    return NOBODY if s is None else Account(s.user.id, all_accounts=s.can("data.others"))


def others_right(request: Request) -> str:
    """The capability a request's route requires to access other accounts' data: for admin routes, the one the route
    declares (capabilities are declared once; `templates.restore` lets a 二级管理员 give a user back a template deleted
    by mistake without granting data.others); data.others for every other route."""
    found = route_of(request)
    return found[1].needs if found is not None and found[1].level == "admin" and found[1].needs else "data.others"


def mine(request: Request, owner: int | None, gone: Msg) -> None:
    """Only the account it belongs to, or a login holding the right its route declares for other people's data
    (others_right) that also manages that account's role (roles.manages: a 二级管理员 restoring templates acts on a
    user's, never an administrator's, as on the 用户 page): anyone else is told it is not there (`gone`), never that it
    is someone else's. An account deleted for good (no owner, or none any more) is only the right's to look at."""
    u = auth.me(request)
    if owner != u.id and not (auth.can(request, others_right(request)) and manages(auth.session(request), owner)):
        raise NotFound(gone)


def manages(viewer, owner: int | None = None, role: str | None = None) -> bool:
    """Whether the login `viewer` (a Session) may act on, or see the particulars of, an account: `owner` (its id), or
    any account of `role`. The one answer to who manages whom (roles.manages: a 二级管理员 manages users, never an
    administrator; nobody another account of the owner's), read wherever other accounts' things are acted on or shown:
    the 用户 page (server/users.py), ownership (mine), the admin page's queue and history (server/farm.py), feedback,
    registrations (server/invites.py), and what is trimmed of a row (particulars). Its own account a login always
    manages; an account deleted for good (none, or not there any more) is only its right's."""
    if viewer is None:
        return False
    if owner is not None and owner == viewer.user.id:
        return True
    if role is None and owner is not None:
        role = _role_of(owner, accounts_revision())
    return role is None or roles.manages(viewer.capabilities, role)


@lru_cache(maxsize=4096)
def _role_of(owner: int, revision: str) -> str | None:
    """An account's role at `revision` (accounts.revision: a new one with every change of a role)."""
    from .. import accounts

    try:
        return accounts.get(owner).role
    except NotFound:
        return None


def accounts_revision() -> str:
    from .. import accounts

    return accounts.revision()


# what a row of another account's says of where it was (an address, a browser, a computer: farm/clients.py full's
# `details`, a registration's `ip` / `net`): only for a login that manages that account (particulars)
PARTICULARS = ("details", "ip", "net", "user_agent", "agent", "hostname", "device", "device_id")


def particulars(viewer, owner: int | None, row: dict) -> dict:
    """`row` (another account's, `owner`) as `viewer` may see it: whole when it manages that account (manages), else
    without what says where it was (PARTICULARS). The one trimming of such rows, wherever a list shows them."""
    return row if manages(viewer, owner) else {k: v for k, v in row.items() if k not in PARTICULARS}


def node_types_for(u: User | None) -> dict:
    """The node types that exist for an account: those whose tags it may use."""
    from ..nodes import node_types

    given = u.allowed if u else None
    return {k: t for k, t in node_types().items() if tags.may(tags.node_tags(t), given)}


def templates_for(u: User | None) -> list[dict]:
    """The templates an account may see: every node of its graph one the account may use (node_types_for: its licence
    class and capability gates, 生成式扩散 among them, within the account's tags) — what touches anything not given is
    not seen at all, as the node itself is not (no switch or route is judged, nothing greyed) — and,
    unless this login manages the templates (it sees them all), only the ones that are switched on (switched in
    server/templates.py). This is the one place that filter lives, so the template panel, the tools routes
    (server/tools.py) and a DCC plugin agree without either deciding anything. A parameter's options the account may
    not use stay hidden on the node (params_for), not on the card.

    Two sources, one list (lab2shot/site/library.py presets): the project's
    templates/ (including the ones an administrator saved there from a node graph) and each adapter's own templates/;
    they are cards of the same shape, so nothing downstream tells them apart."""
    from ..site.library import presets as templates

    manages = u is not None and "templates.create" in u.capabilities
    if manages:
        return list(templates())
    usable = node_types_for(u)
    return [t for t in templates()
            if t["enabled"] and all(n.get("type") in usable for n in (t["graph"].get("nodes") or ()))]


def _hidden_choices(given: frozenset[str] | None, node_type) -> dict[str, list]:
    """Parameter -> the values of it someone given `given` may not use: the ones whose licence class or registration
    (nodes/applies.py OptionTrait: 非商用, 仅限研究, 需注册) the account was not given, judged as a node's own class
    is (tags.may)."""
    from ..nodes.applies import licensed_choices

    out = {name: [v for v, chosen in values.items() if not tags.may(frozenset(chosen), given)]
           for name, values in licensed_choices(node_type).items()}
    return {name: values for name, values in out.items() if values}


def sees_cards(s) -> bool:
    """Whether a login gets the cards' figures (server/available.py FIELDS farm.cards): card names, what a setting was
    measured to take on a card."""
    from ..roles import SessionFacts
    from .available import FIELDS

    return "farm.cards" in resolve({"farm.cards": FIELDS["farm.cards"]}, SessionFacts.of(s)).available


def params_for(s, node_type) -> list[dict]:
    """A node type's parameters (NodeDef.interface_specs) as a login sees them, in every answer that describes them (the
    catalogue: describe_for; the tools a DCC plugin or a script reads: server/tools.py): the choices its account may not
    use are not among a parameter's options, and a default that is one of them becomes the first it may use; a parameter
    whose only other value is one it may not use (a switch) is not there, staying at its default; what a setting was
    measured to take on a card (`measured`) is there only for a login that sees the cards (sees_cards)."""
    hidden = _hidden_choices(s.user.allowed if s is not None else None, node_type)
    cards = sees_cards(s)
    out = []
    for p in node_type.interface_specs():
        if gone := hidden.get(p["name"], []):
            options = [o for o in (p.get("options") or []) if o not in gone]
            if not options:  # a switch: stays at its default (never one it may not use: nodes/tags.py declares the value)
                continue
            labels = {k: v for k, v in (p.get("option_labels") or {}).items() if k not in {str(g) for g in gone}}
            p = {**p, "options": options, "option_labels": labels, **({"default": options[0]} if p.get("default") in gone else {})}
        out.append(p if cards else _unmeasured(p))
    return out


def _unmeasured(spec):
    """A parameter's description without `measured`, its own and its parts' (a list parameter's items describe theirs)."""
    if isinstance(spec, dict):
        return {k: _unmeasured(v) for k, v in spec.items() if k != "measured"}
    return [_unmeasured(v) for v in spec] if isinstance(spec, list) else spec


def describe_for(s, node_type) -> dict:
    """A node type as the editor of a login sees it (NodeDef.describe): its parameters as params_for gives them, the
    defaults and the lookup table of what choices change without the choices its account may not use."""
    from ..nodes.applies import option_key

    from ..site.catalog import describe

    desc = describe(node_type)
    hidden = _hidden_choices(s.user.allowed if s is not None else None, node_type)
    params, defaults = params_for(s, node_type), dict(desc["defaults"])
    for name, gone in hidden.items():
        if defaults.get(name) in gone:
            shown = next((p for p in params if p["name"] == name), None)
            defaults[name] = shown["options"][0] if shown is not None else (
                not defaults[name] if isinstance(defaults[name], bool) else None)
    traits = {name: {key: row for key, row in rows.items() if key not in {option_key(v) for v in hidden.get(name, [])}}
              for name, rows in desc["option_traits"].items()}
    return {**desc, "params": params, "defaults": defaults, "option_traits": traits}


UNKNOWN = Msg("E-GRAPH-UNKNOWNNODES")


def admit(request: Request, data: dict) -> Graph:
    """A graph from an account, before anything is done with it: every node type one it may use (one it may not is
    refused exactly like a type this server lacks), every choice within its tags. The administrator is told which type
    is missing and why. Uploads are not looked at here at all: one that is not this account's is not there when it is
    read (transfer/uploads.py resolve, for the account this request serves), so that node fails at itself and the
    rest of the graph is answered and cooked as usual."""
    return admitted(auth.me(request), data)


def admitted(u: User, data: dict) -> Graph:
    """admit's rule for an account: the graph read, or GraphError saying why it is refused. The one rule a graph is
    taken by: every request with a graph goes through admit, and the tools a DCC plugin is offered (server/tools.py)
    are the ones it lets through, so what is listed can be submitted."""
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise GraphError(Msg("E-GRAPH-NOTAGRAPH"))
    if len(data["nodes"]) > MAX_NODES:  # every graph reaches the server through here: status, plan, a job, a client's
        raise GraphError(Msg("E-GRAPH-TOOMANYNODES", count=len(data["nodes"]), most=MAX_NODES))
    usable = node_types_for(u)
    if "nodes.all" not in u.capabilities and any(not isinstance(n, dict) or n.get("type") not in usable for n in data["nodes"]):
        raise GraphError(UNKNOWN)
    # what it says when it cannot be read is said to everyone alike: a parameter it refuses is named with the value
    # given and what it takes, never with the list of its options (nodes/params.py refusal)
    graph = Graph.from_json(data)
    given = u.allowed if u else None
    for nid, n in graph.nodes.items():
        if not tags.may(graph.resolved(nid).licence.tags, given):
            raise GraphError(Msg("E-GRAPH-UNKNOWNCHOICE", node=n.label))
    return graph


def readable(request: Request, fp: str) -> None:
    """A packet this account may read: one in its own cache (data/store.py; the request is served as its account,
    lab2shot/serving.py, so packet_dir names that cache), else not found — exactly as for a result never computed. An
    administrator holding data.others reads its own cache too: another account's is read by naming that account
    (data/store.py cache_of), never by a bare fingerprint."""
    from ..data.packet import base_of, packet_dir

    if not packet_dir(base_of(fp)).is_dir():  # packet_dir refuses what is not a fingerprint (E-PACKET-FINGERPRINT)
        raise NotFound(Msg("E-ACCESS-NORESULT"))


def _strip_bytes(data: bytes, paths: tuple[str, ...], stream: bool) -> bytes:
    """An answer without the fields at `paths` (server/available.py strip): a JSON body whole, or each event of a
    stream on its own. What is not JSON is left as it is."""
    from .available import strip

    def stripped(text: str) -> str:
        value = json.loads(text)
        for p in paths:
            strip(value, p)
        return json.dumps(value, ensure_ascii=False)

    try:
        if stream:
            return "\n".join("data: " + stripped(line[6:]) if line.startswith("data: ") else line
                             for line in data.decode("utf-8").split("\n")).encode()
        return stripped(data.decode("utf-8")).encode()
    except (ValueError, UnicodeDecodeError):
        return data


# ------------------------------------------------------------------ this server's own folders, out of every answer


def _places() -> re.Pattern:
    """A path under one of this server's own folders (every folder the settings place things in, the program's, the
    home folder, Python's), as configured and as resolved, up to where it ends in a JSON string or a line."""
    import sys

    return _places_of(tuple(str(f) for f in (*settings().folders, ROOT, Path.home(), Path(sys.prefix), Path(sys.base_prefix))))


@lru_cache(maxsize=4)
def _places_of(folders: tuple[str, ...]) -> re.Pattern:
    """The pattern of _places, made once per set of folders (resolving each is a walk of the file system: never once
    per answer). A folder counts only as a whole path where one begins and ends (not /data inside upload:…/data/, not
    /home/me inside /home/me2): what an account wrote that merely holds those letters is left as it is."""
    roots = {str(p) for f in folders for p in (Path(f), Path(f).resolve())}
    alts = "|".join(re.escape(r) for r in sorted((r for r in roots if len(r) > 1), key=len, reverse=True))
    return re.compile(rf"(?<![\w/:.~-])(?:{alts})(?=[/\s\"'<>\\,;)]|$)(?:/[^\s\"'<>\\,;)]*)?".encode())


def scrub(data: bytes) -> bytes:
    """This server's own folders out of what goes back to someone without logs.view: a path there becomes
    its file name alone (a job's error still says which file, never where it is)."""
    return _places().sub(lambda m: m.group(0).rsplit(b"/", 1)[-1] or b"...", data)


# what probing looks like: in the path as sent, going up a folder, encoded dots, slashes and NUL, backslashes (never in
# an address a client of ours sends); asking for the repo's own folders and files (PROBE_OWN), except on a route of an
# account's own things, whose names may well be config/shot.toml or comp..v2.py (probe_path); in the query, going up or
# NUL (a query may encode slashes: an upload's reference)
PROBE_PATH = re.compile(rb"(?i)((?:^|/)\.\.(?:[/;]|$)|%2e|%2f|%5c|%00|\\|\x00)")
PROBE_OWN = re.compile(rb"(?i)(\.\.|/\.git|/\.env|/lab2shot/|/adapters/|/third_party/|/work/|/config/|\.py$|\.toml$|\.db$|\.lock$)")
PROBE_QUERY = re.compile(rb"(?i)(\.\.|%2e%2e|%00|\x00)")


def probe_path(raw: bytes, found: tuple[str, Access] | None, s) -> bool:
    """Whether the path as sent looks like probing. A signed-in account on a route of its own (user or admin) asks for
    its things by their names: counting a name as probing would block the account's own session (auth.Watch) for
    fetching its outputs, and the route checks the name itself (io/files.py inside). Everyone else, and every address
    that is not such a route (the page's entry, one this server does not have), is checked for the repo's files too."""
    theirs = s is not None and found is not None and found[1].level in ("user", "admin")
    return bool(PROBE_PATH.search(raw) or (not theirs and PROBE_OWN.search(raw)))


_SLASHES = re.compile(r"/{2,}")  # an address no page of this site ever asks for: refused, never quietly collapsed


def _named(host: str) -> bool:
    """A host name of its own (a tunnel's domain), not localhost or an address."""
    import ipaddress

    name = host.split(",")[0].strip().rsplit(":", 1)[0].strip("[]").lower()
    try:
        ipaddress.ip_address(name)
        return False
    except ValueError:
        return bool(name) and name != "localhost" and not name.endswith(".localhost")


class Guard:
    """The middleware in front of everything (see the module doc)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        method, path = scope["method"], scope["path"]
        # the request's language (server/lang.py), before anything is said: what it asks for itself, then (once its
        # login is read) its account's choice
        spoken = i18n.set_current(lang.before_login(request))
        try:
            await self._guard(request, scope, receive, send, method, path)
        finally:
            i18n.reset(spoken)

    async def _guard(self, request: Request, scope: Scope, receive: Receive, send: Send, method: str, path: str) -> None:
        try:
            s, need, refused, scrubbing, fields = await off_loop(self.look, request, method, path)
        except Exception as exc:  # the guard's own fault: answered as every failure is, with its reference (logs.failed)
            from .logs import failed

            await (await off_loop(failed, request, exc))(scope, receive, send)
            return
        # Who the request belongs to, recorded on the scope; the outermost layer (server/traffic.py Meter) uses it to
        # attribute the response bytes to this account. This is the only place in the server that identifies the
        # request's owner; traffic accounting does not identify it again.
        scope[SCOPE_USER] = s.user.id if s is not None else 0
        said_in = lang.of(request, s)
        i18n.set_current(said_in)
        scope[lang.SCOPE_KEY] = said_in
        transforming = scrubbing or bool(fields)
        hsts = auth.https(request) and _named(auth.host(request))

        started: Message | None = None
        held: list[bytes] = []
        status = 0

        def clean(body: bytes, stream: bool, zipped: bool = False) -> bytes:
            if zipped:  # an answer its route compressed itself (a 3D view's description): cleaned as what it says
                return gzip.compress(clean(gzip.decompress(body), stream), compresslevel=5) if body else body
            if scrubbing:
                body = scrub(body)
            # an answer's fields are the answer's: an error answer (never one to strip) goes as it is
            return _strip_bytes(body, fields, stream) if fields and body and 200 <= status < 300 else body

        async def guarded(message: Message) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                for k, v in HEADERS.items():
                    headers.setdefault(k, v)
                headers["Content-Language"] = said_in  # what it is said in, and what that depends on (server/lang.py)
                headers["Vary"] = ", ".join(v for v in (headers.get("vary", ""), lang.VARY) if v)
                if hsts:
                    headers["Strict-Transport-Security"] = wire.HSTS
                if path.startswith((ADMIN, "/api/auth/")):
                    headers["Cache-Control"] = wire.admin_answer(headers.get("cache-control", ""))
                if not (transforming and headers.get("content-type", "").startswith(SCRUBBED)):
                    await send(message)
                elif "content-length" in headers:
                    started = message  # the whole answer is changed at once: its length changes
                elif headers.get("content-encoding"):
                    raise RuntimeError(f"{path}: an encoded answer the guard must change is sent whole (Content-Length)")
                else:
                    await send(message)  # a stream (a job's events): each piece on its own
                    started = {"stream": True}
                return
            if started is None:
                await send(message)
            elif "stream" in started:
                await send({**message, "body": clean(message.get("body", b""), True)})
            else:
                held.append(message.get("body", b""))
                if not message.get("more_body"):
                    zipped = MutableHeaders(scope=started).get("content-encoding", "") == "gzip"
                    body = await off_loop(clean, b"".join(held), False, zipped)
                    MutableHeaders(scope=started)["content-length"] = str(len(body))
                    await send(started)
                    await send({"type": "http.response.body", "body": body})

        from ..serving import serving

        written = _audited.set([])
        try:
            if refused is not None:
                await refused(scope, receive, guarded)
            else:
                with serving(account_of(s)):  # whose work this request is while it is served (uploads.resolve reads it)
                    await self.app(scope, receive, guarded)
        finally:
            said = _audited.get()
            _audited.reset(written)
        # every admin write of a login, once, as it went: refused here or by its route, done, or failed (the route's own
        # line when it wrote one). A stranger's is not one (nobody's action); a login's repeated refusals are counted on
        # one row (audit), so no number of them fills the table
        if need is not None and s is not None and method not in ("GET", "HEAD") and not said:
            await off_loop(lambda: audit(ended(s, need, method, path, status or 500), session=s, method=method, path=path))

    def look(self, request: Request, method: str, path: str):
        """What the guard must know before a request goes on, all of it blocking (the session is a database read): run
        off the event loop (wire.off_loop). Returns the session, the capability the route needs, the refusal (None: it
        may go on), whether its answer is scrubbed of the server's folders, and the fields of its answer this session
        does not get (server/available.py FIELDS: declared once, resolved like every other subject)."""
        from . import available

        s, found = auth.session(request), route_of(request)
        i18n.set_current(lang.of(request, s))  # a refusal below speaks the account's language (this thread's context)
        need = found[1].needs if found is not None else None
        refused = self.refusal(request, method, path, s, found)
        privileged = s is not None and s.can("logs.view")  # sees where the server keeps things
        scrubbing = not privileged and method != "HEAD" and not VERBATIM.match(path)
        fields = available.hidden_paths(s, found) if refused is None and method != "HEAD" else ()
        return s, need, refused, scrubbing, fields

    def refusal(self, request: Request, method: str, path: str, s, found: tuple[str, Access] | None) -> JSONResponse | None:
        g, need = auth.guards(), found[1].needs if found is not None else None
        if blocked := g.watch.is_blocked(request):
            return refused(TooManyTries(Msg("E-ACCESS-BLOCKED", minutes=int(blocked / 60) + 1)))
        if not auth.host_known(request):  # another site's name pointing here (DNS rebinding), or a name not yet listed
            g.watch.note(request, "unknown_host", request.headers.get("host", "")[:200])
            return refused(Misdirected(Msg("E-ACCESS-UNKNOWNHOST", host=request.headers.get("host", "")[:200])))
        key, own = auth.client_key(request), found[1].limit if found is not None else None
        rated = own is not None and own.burst is not None
        if not (g.rate.take(f"{found[0]} {key}", own.burst, own.per_s) if rated else g.rate.take(key)):
            g.watch.note(request, "too_fast")  # every session counts, the administrator's too
            return refused(TooManyTries(Msg("E-ACCESS-TOOFAST")))
        raw, query = request.scope.get("raw_path") or request.scope["path"].encode(), request.scope.get("query_string") or b""
        if probe_path(raw, found, s) or PROBE_QUERY.search(query):
            g.watch.note(request, "odd_path", (raw + (b"?" + query if query else b"")).decode("latin-1")[:200])
        if PROBE_QUERY.search(query):  # going up or NUL in a parameter: never a name any page sends (defence in depth)
            return refused(Invalid(Msg("E-ACCESS-PROBE")))
        if _SLASHES.search(path):  # "//api//admin/x": the routes match the path as sent, so the guard must too;
            g.watch.note(request, "odd_path", path[:200])  # collapsing it here would classify one address and serve another
            return refused(NotFound(Msg("E-ACCESS-NOROUTE")))
        where = level(method, path, found)
        if where is None:
            # Logged in, and the request says it comes from a page of this site, yet the endpoint is not here. During an
            # upgrade the page may be newer than the server or older, and an open page keeps asking: counting that would
            # lock a person out after a few dozen clicks. A script holding someone's login looks just the same (Origin is
            # a header anyone can send), so the note says only what is known, never which of the two it was. It is
            # recorded under the admin page's 「安全」 with the account, not counted toward blocking.
            ours = s is not None and same_site(request)
            g.watch.note(request, "no_route", Word("server.watch_detail.ours") if ours else "", counts=not ours)
            return refused(NotFound(Msg("E-ACCESS-NOROUTE")))
        writes = method not in ("GET", "HEAD")
        if where == "admin" and (s is None or not s.capabilities):  # no rights at all now: the page asks for the password
            g.watch.note(request, "admin_only")
            return refused(NotSignedIn(Msg("E-ACCESS-ADMINONLY")), admin=True)
        if where == "admin" and path.startswith(ADMIN):
            if need is None:
                g.watch.note(request, "no_route")
                return refused(NotFound(Msg("E-ACCESS-NOROUTE")))
            if not s.can(need):  # a role without this one: refused with what it lacks and who has it
                g.watch.note(request, "no_capability", need)
                return refused(Forbidden(Msg("E-ACCESS-NOCAPABILITY", role=roles.word(s.user.role),
                                             what=roles.CAPABILITIES[need].what_word, roles=roles.holders(need))))
            if found[1].local and not auth.machine(request):  # stopping the server, clearing the counts
                g.watch.note(request, "local_only")  # of wrong passwords: never from a browser, whatever its rights
                return refused(Forbidden(Msg("E-ACCESS-LOCALONLY")))
        if where == "user" and s is None:
            if path.startswith("/api/") and auth.token_of(request):
                # a login of ours that ended (another login took its place, or it lapsed) is answered and not counted:
                # the page that was kicked keeps asking for the queue and the load until someone looks at it, and
                # counting those would block the very person who had been kicked from logging in again. Only a token
                # this server never gave out is odd.
                info = auth.accounts.kicked_info(auth.credential(request))
                if info:
                    kicked = auth.kicked_detail(info)
                    return refused(NotSignedIn(kicked), login=True, kicked={**info, "detail": kicked.text, "code": kicked.code})
                if not auth.accounts.once_issued(auth.credential(request)):
                    g.watch.note(request, "foreign_credential")
            return refused(NotSignedIn(Msg("E-ACCESS-SIGNIN")), login=True)
        if where != "open" and not path.startswith(AUTH) and (owed := terms.owed(s.user)) is not None:
            return refused(Forbidden(Msg("E-TERMS-OWED")), terms=owed)
        if writes and not auth.by_header(request) and not same_site(request):  # the cookie's, or logging in
            g.watch.note(request, "cross_site")
            return refused(Forbidden(Msg("E-ACCESS-CROSSSITE")))
        most = own.body if own is not None else MAX_BODY
        if most is not None:  # whatever the method: a GET may carry a body too, and routes.Route reads any JSON one
            length = request.headers.get("content-length")
            if length is None and "chunked" in request.headers.get("transfer-encoding", "").lower():
                return refused(LengthRequired(Msg("E-ACCESS-NOLENGTH")))
            if length is not None and (decimal(length) is None or decimal(length) > most):
                g.watch.note(request, "too_big", Word("server.watch_detail.bytes", count=length))
                return refused(TooLarge(Msg("E-ACCESS-TOOBIG", mb=round(most / (1 << 20), 2))))
        return None
