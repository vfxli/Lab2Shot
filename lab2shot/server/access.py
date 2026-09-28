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
    when its fingerprint is there, in the requesting account's own cache (packets_readable); the same graph cooked by
    two accounts is cooked and kept twice, and a fingerprint of another account's is simply not there;
  - node types: only those whose tags the account may use (nodes/tags.py). The others do not exist for it: not in the
    catalog or templates, a graph with one is refused like one with a type this server lacks, and
    no answer names them (redact).

Guard, the one middleware in front of everything, applies it to every request, and also
  - lets an account that has not agreed to the current 用户协议 and 隐私政策 (lab2shot/terms owed: one an
    administrator made, or anyone after the texts changed) reach nothing but the login's own routes (/api/auth/:
    agreeing, logging out) and the open ones, until it agrees (403 marked `terms`: the page asks for it);
  - counts requests per client (auth.Rate), every session's alike, and per client on a route with a rate of its own
    (its declared Limit): a flood is refused (429);
  - refuses every write that relies on the cookie (or on no credential: logging in) and did not come from a page of
    this site (403: its Origin, or Referer, names another site or none); a bearer or machine token in a header is
    never sent by another site's page;
  - refuses a request body larger than MAX_BODY, or its route's own limit (its declared Limit: uploads and feedback check theirs
    as they read);
  - adds the security headers (HEADERS) to every answer, and over HTTPS (this server's, or the tunnel's in front of
    it) to a named host also HSTS: the browser then never asks it over plain http;
  - takes this server's own folders out of what anyone without logs.view gets back (scrub): an error, a job's
    log, a result's description never show where the server keeps things, only file names;
  - takes what an account may not use out of every answer it gets (redact);
  - notes what looks like probing (auth.Watch, the admin page's 安全): refused routes, odd paths, floods;
  - says whose work a request is while it is served (lab2shot/serving.py): the session's account, or every account
    for one holding data.others; transfer/uploads.py resolve answers by it. Nothing else in the server asks whose
    an upload is.
"""

from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import compile_path
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .. import logs, roles, terms
from ..accounts import User
from ..config import ROOT, WEBUI_DIST, settings
from ..database import db, json_text
from ..engine.graph import Graph, GraphError
from ..errors import NotFound
from ..messages import Msg, plain
from ..nodes import tags
from ..traffic import SCOPE_USER
from . import auth, wire
from .routes import ADMIN, MAX_BODY, MAX_NODES, Limit, match
from .wire import off_loop

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


def limit_of(method: str, path: str) -> tuple[str, Limit] | None:
    """The route and its own limits (its declared Limit), when it has any."""
    found = match(method, path)
    return (found[0], found[1].limit) if found is not None and found[1].limit is not None else None


def need_of(method: str, path: str) -> str | None:
    """The capability a request's route declares (an admin route's always; a product page's action's besides a login);
    None: a login is enough, or no such route."""
    found = match(method, path)
    return found[1].needs if found is not None else None


audit_log = logs.get("admin")


def audit(message: Msg, session=None, method: str = "", path: str = "") -> None:
    """One admin action, written in one place to both records: the server log (with its code: who, role,
    what, on what, as the message states) and the `admin_actions` table, which only ever grows and has no route that
    changes or removes a row. The table is what 后台「用户」详情页's 管理操作 tab lists, per account.

    `session`: whose action it was (None: nobody was logged in, i.e. a refused attempt). `method`/`path`: the request it
    came from ("" for an action from the command line)."""
    logs.say(audit_log, message)
    user = getattr(session, "user", None)
    try:
        with db().write() as c:
            c.execute(
                "INSERT INTO admin_actions (at, user_id, role, code, params, method, path, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), getattr(user, "id", None), getattr(user, "role", ""), message.code,
                 json_text({k: plain(v) for k, v in message.params.items()}), method, path,
                 REFUSED if message.level == "W" else DONE),
            )
    except Exception as exc:  # the log line above is the record that must not be lost: never fail the action itself
        logs.say(audit_log, Msg("W-AUDIT-NOTKEPT", code=message.code, why=str(exc)))


DONE = "成功"
REFUSED = "拒绝"


def actor(s) -> tuple[str, str]:
    """Who a session is and its role, as audit lines name them."""
    return (s.user.label, roles.label(s.user.role)) if s is not None else ("未登录", "")
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


def level(method: str, path: str) -> str | None:
    """"open", "user", "admin", or None (not served)."""
    if path.startswith(ADMIN) or path == ADMIN.rstrip("/"):
        return "admin"
    if path.startswith("/assets/"):
        return asset_level(path)
    if not path.startswith("/api/"):
        return "open" if method in ("GET", "HEAD") and PAGE_ENTRY.match(path) else None  # the page's entry: index.html
    found = match(method, path)
    return found[1].level if found is not None and found[1].level in ("open", "user") else None


def same_site(request: Request) -> bool:
    """Whether the request came from a page of this site: its Origin (or, without one, its Referer) names the host it was
    sent to (or the host a trusted proxy says it forwarded: auth.host)."""
    source = request.headers.get("origin") or request.headers.get("referer") or ""
    host = urlsplit(source).netloc.lower() if source and source != "null" else ""
    here = {request.headers.get("host", "").lower(), auth.host(request)}
    return bool(host) and host in here - {""}


def refused(message: Msg, status: int, **extra) -> JSONResponse:
    """The guard's answer to a request it does not let through: the message's text and code (like every error answer,
    server/app.py), and what the page acts on (login, admin, kicked)."""
    return JSONResponse({"detail": message.text, "code": message.code, **extra}, status_code=status)


# ------------------------------------------------------------------ what is whose


def account_of(s):
    """Whose work a request is (lab2shot/serving.py Account): its own account's; every account's for a session
    holding data.others (the back office reads a user's graph, job and results as that user has them). A request
    without a session (not signed in) owns nothing."""
    from ..serving import NOBODY, Account

    return NOBODY if s is None else Account(s.user.id, all_accounts=s.can("data.others"))


def others_right(method: str, path: str) -> str:
    """The capability a route requires to access other accounts' data: for admin routes, the one the route declares
    (capabilities are declared once; `templates.restore` lets a 二级管理员 give a user back a template deleted by
    mistake without granting data.others); data.others for every other route."""
    found = match(method, path)
    return found[1].needs if found is not None and found[1].level == "admin" and found[1].needs else "data.others"


def mine(request: Request, owner: int | None, gone: Msg) -> None:
    """Only the account it belongs to, or a login holding the right its route declares for other people's data
    (others_right): anyone else is told it is not there (`gone`), never that it is someone else's."""
    u = auth.me(request)
    if owner != u.id and not auth.can(request, others_right(request.method, request.url.path)):
        raise NotFound(gone)


# ------------------------------------------------------------------ node types an account may use (nodes/tags.py)


def node_types_for(u: User | None) -> dict:
    """The node types that exist for an account: those whose tags it may use."""
    from ..nodes import node_types

    given = u.allowed if u else None
    return {k: t for k, t in node_types().items() if tags.may(tags.node_tags(t), given)}


def templates_for(u: User | None) -> list[dict]:
    """The templates an account may use: every node and choice of it within its tags and, unless this login manages
    the templates, only the ones that are switched on (switched in server/templates.py; this is the one place that
    filter lives, so the template panel and the tools routes (server/tools.py) agree without either deciding anything).

    Two sources, one list (engine/templates.py templates, read by lab2shot/library.py presets): the project's
    templates/ (including the ones an administrator saved there from a node graph) and each adapter's own templates/;
    they are cards of the same shape, so nothing downstream tells them apart."""
    from ..engine.templates import templates

    given = u.allowed if u else None
    manages = u is not None and "templates.create" in u.capabilities
    return [t for t in templates() if tags.may(frozenset(t["licence"]), given) and (manages or t["enabled"])]


def _hidden_choices(given: frozenset[str] | None, node_type) -> dict[str, list]:
    """Parameter -> the values of it someone given `given` may not use (the ones that switch to non-commercial parts,
    nodes/applies.py OptionTrait)."""
    from ..nodes.applies import noncommercial_choices

    if given is None or tags.may(frozenset({tags.NONCOMMERCIAL}), given):
        return {}
    return {name: list(values) for name, values in noncommercial_choices(node_type).items()}


def describe_for(u: User | None, node_type) -> dict:
    """A node type as the editor of an account sees it (NodeDef.describe): the choices it may not use are not among
    its parameter's options (nor in its lookup table of what choices change), and a default that is one of them becomes
    the first it may use; a parameter whose only other value is one it may not use (a switch) is not shown, staying at
    its default."""
    from ..nodes.applies import option_key

    from ..catalog import describe

    desc = describe(node_type)
    hidden = _hidden_choices(u.allowed if u else None, node_type)
    if not hidden:
        return desc
    params, defaults = [], dict(desc["defaults"])
    for p in desc["params"]:
        gone = hidden.get(p["name"], [])
        if not gone:
            params.append(p)
            continue
        options = [o for o in (p.get("options") or []) if o not in gone]
        if not options:  # a switch: stays at its default (never one it may not use: nodes/tags.py declares the value)
            if defaults.get(p["name"]) in gone:
                defaults[p["name"]] = not defaults[p["name"]] if isinstance(defaults[p["name"]], bool) else None
            continue
        labels = {k: v for k, v in (p.get("option_labels") or {}).items() if k not in {str(g) for g in gone}}
        params.append({**p, "options": options, "option_labels": labels})
        if defaults.get(p["name"]) in gone:
            defaults[p["name"]] = options[0]
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
    u = auth.me(request)
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


def packets_readable(request: Request) -> None:
    """(A dependency of every route that reads packets: server/packets.py, server/view.py) the packets a request
    names (its {fp}, a camera it draws with, a camera it looks through) are the account's to read."""
    # `through` (the camera a frame is viewed through: server/packets.py frame) is also a packet fingerprint and must be
    # checked too; otherwise another account's camera packet fingerprint would allow viewing through its lens and
    # reading its distortion.
    for fp in (request.path_params.get("fp"), request.query_params.get("camera"), request.query_params.get("through")):
        if fp:
            readable(request, fp)


def readable(request: Request, fp: str) -> None:
    """A packet this account may read: one in its own cache (data/store.py; the request is served as its account,
    lab2shot/serving.py, so packet_dir names that cache), else not found — exactly as for a result never computed. An
    administrator holding data.others reads its own cache too: another account's is read by naming that account
    (data/store.py cache_of), never by a bare fingerprint."""
    from ..data.packet import base_of, packet_dir

    if not packet_dir(base_of(fp)).is_dir():  # packet_dir refuses what is not a fingerprint (E-PACKET-FINGERPRINT)
        raise NotFound(Msg("E-ACCESS-NORESULT"))


# ------------------------------------------------------------------ what an account may not use, out of every answer


def _case_classes() -> dict[int, str]:
    """Each character the regular-expression engine takes as the same letter as another one when case is ignored,
    beyond what lowercasing gives (the dotless i and i, the long s and s, the two sigmas ...: re/_casefix.py), mapped
    to one of them."""
    from re import _casefix

    first: dict[int, int] = {}
    for k, others in _casefix._EXTRA_CASES.items():
        m = min(k, *others)
        for x in (k, *others):
            first[x] = min(first.get(x, m), m)
    return {k: chr(v) for k, v in first.items() if k != v}


_CASE_CLASSES = _case_classes()
_CASE_CHARS = re.compile("[" + "".join(re.escape(chr(k)) for k in _CASE_CLASSES) + "]")


def _fold(text: str) -> str:
    """`text` with case taken out exactly as a case-insensitive pattern takes it out, one character for one: İ is i
    (lowercasing makes it two), then lowercase, then the letters the engine counts as one (_case_classes). Two strings
    a case-insensitive literal pattern matches at a place are equal here character by character. Each step only when
    the text has something for it (most answers have none of those letters)."""
    if "\u0130" in text:
        text = text.replace("\u0130", "i")
    text = text.lower()
    return text.translate(_CASE_CLASSES) if _CASE_CHARS.search(text) else text


class Hidden:
    """The names of what an account may not use, as `pattern`: whole words, case-insensitive (what an answer loses).
    `search` answers exactly as `pattern.search` does, fast: first a plain look for any of the names anywhere in the
    text case-folded (_fold), which every match of `pattern` needs, and `pattern` itself only when that finds one. A
    name that holds another is not looked for (the shorter one is there wherever it is). A big case-insensitive
    alternation with lookarounds tries every name at every place: hundreds of ms on a long answer; the look is a few
    substring searches."""

    def __init__(self, terms: set[str]) -> None:
        alts = "|".join(re.escape(x) for x in sorted(terms, key=len, reverse=True))
        self.pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_])(?:{alts})(?![A-Za-z0-9_])")
        needed: list[str] = []
        for name in sorted({_fold(x) for x in terms}, key=len):
            if not any(n in name for n in needed):
                needed.append(name)
        self._needed = tuple(needed)

    def search(self, text: str) -> re.Match | None:
        folded = _fold(text)
        return self.pattern.search(text) if any(n in folded for n in self._needed) else None


@lru_cache(maxsize=16)
def _hidden_terms(given: frozenset[str], _text_stamp: tuple) -> Hidden | None:
    """The names of what does not exist for someone given `given` (hidden projects: their ids and titles; hidden
    node types: their ids and labels; hidden choices: their values that are names of their own, like a model's), as
    one pattern; None when nothing is hidden. `_text_stamp`: the node text files' stamp the labels were taken from
    (nodes/text.py refresh); it is part of the key, so a renamed node re-derives the pattern."""
    from ..extensions import extensions
    from ..nodes import node_types

    shown = {t.runtime for t in node_types().values() if tags.may(tags.node_tags(t), given)}
    terms: set[str] = set()
    for t in node_types().values():
        if not tags.may(tags.node_tags(t), given):
            terms |= {t.id, t.label}
        for values in _hidden_choices(given, t).values():
            terms |= {v for v in values if isinstance(v, str) and len(v) >= 8}
    for name, ext in extensions().items():
        if name not in shown and not tags.may(frozenset({ext.license.tag}), given):
            # Only the extension name and the whole title are matched (whole word, case-insensitive), without splitting
            # the title into words: words such as "NVIDIA" or "Fast" on their own do not refer to this extension, and
            # listing them would make redact_text remove every sentence containing them from a normal user's answers.
            terms |= {name, ext.title}
    terms = {x for x in terms if len(x) >= 3}
    if not terms:
        return None
    return Hidden(terms)


def hidden_pattern(u: User | None) -> Hidden | None:
    given = u.allowed if u else None
    if given is None:
        return None
    from ..nodes import node_types, text

    # the classes say the files' words before their labels go into the pattern; the stamp those words match is the
    # cache key, so a renamed hidden node has its new name hidden too
    return _hidden_terms(given, text.refresh(node_types()))


_SENTENCE = re.compile(r"(?<=[。；！？\n])|(?<=[.;!?])(?=\s)")


def redact_text(text: str, hidden: Hidden) -> str:
    """A text without the sentences (lines, clauses ended by 。；！？) that name what is hidden."""
    if not hidden.search(text):
        return text
    return "".join(part for part in _SENTENCE.split(text) if not hidden.search(part)).strip()


def redact(value, hidden: Hidden):
    """Every text in an answer without what is hidden: sentences naming it go, an entry keyed by it goes."""
    if isinstance(value, str):
        return redact_text(value, hidden)
    if isinstance(value, list):
        return [redact(v, hidden) for v in value]
    if isinstance(value, dict):
        return {k: redact(v, hidden) for k, v in value.items() if not hidden.search(k)}
    return value


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


def _redact_bytes(data: bytes, hidden: Hidden, stream: bool) -> bytes:
    text = data.decode("utf-8", "replace")  # decoded once leniently here; a strict decode would raise UnicodeDecodeError (a 500) on a split multi-byte character
    if not hidden.search(text):
        return data
    if stream:  # server-sent events: each "data: {...}" line on its own
        lines = []
        for line in text.split("\n"):
            if line.startswith("data: "):
                try:
                    line = "data: " + json.dumps(redact(json.loads(line[6:]), hidden), ensure_ascii=False)
                except ValueError:
                    line = redact_text(line, hidden)
            lines.append(line)
        return "\n".join(lines).encode()
    try:
        return json.dumps(redact(json.loads(text), hidden), ensure_ascii=False).encode()
    except ValueError:
        return redact_text(text, hidden).encode()


# ------------------------------------------------------------------ this server's own folders, out of every answer


def _places() -> re.Pattern:
    """A path under one of this server's own folders (every folder the settings place things in, the program's, the
    home folder, Python's), as configured and as resolved, up to where it ends in a JSON string or a line."""
    import sys

    folders = [*settings().folders, ROOT, Path.home(), Path(sys.prefix), Path(sys.base_prefix)]
    roots = {str(p) for f in folders for p in (f, f.resolve())}
    alts = "|".join(re.escape(r) for r in sorted((r for r in roots if len(r) > 1), key=len, reverse=True))
    return re.compile(rf"(?:{alts})(?:/[^\s\"'<>\\,;)]*)?".encode())


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


def probe_path(raw: bytes, method: str, path: str, s) -> bool:
    """Whether the path as sent looks like probing. A signed-in account on a route of its own (user or admin) asks for
    its things by their names: counting a name as probing would block the account's own session (auth.Watch) for
    fetching its outputs, and the route checks the name itself (io/files.py inside). Everyone else, and every address
    that is not such a route (the page's entry, one this server does not have), is checked for the repo's files too."""
    found = match(method, path)
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
        s, need, refused, scrubbing, hidden, fields = await off_loop(self.look, request, method, path)
        # Who the request belongs to, recorded on the scope; the outermost layer (server/traffic.py Meter) uses it to
        # attribute the response bytes to this account. This is the only place in the server that identifies the
        # request's owner; traffic accounting does not identify it again.
        scope[SCOPE_USER] = s.user.id if s is not None else 0
        transforming = scrubbing or bool(fields)
        hsts = auth.https(request) and _named(auth.host(request))

        started: Message | None = None
        held: list[bytes] = []
        status = 0

        def clean(body: bytes, stream: bool) -> bytes:
            if scrubbing:
                body = scrub(body)
            if fields and body:
                body = _strip_bytes(body, fields, stream)
            return _redact_bytes(body, hidden, stream) if hidden is not None and body else body

        async def guarded(message: Message) -> None:
            nonlocal started, status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                for k, v in HEADERS.items():
                    headers.setdefault(k, v)
                if hsts:
                    headers["Strict-Transport-Security"] = wire.HSTS
                if path.startswith((ADMIN, "/api/auth/")):
                    headers["Cache-Control"] = wire.admin_answer(headers.get("cache-control", ""))
                if not (transforming and headers.get("content-type", "").startswith(SCRUBBED)):
                    await send(message)
                elif "content-length" in headers:
                    started = message  # the whole answer is changed at once: its length changes
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
                    body = await off_loop(clean, b"".join(held), False)
                    MutableHeaders(scope=started)["content-length"] = str(len(body))
                    await send(started)
                    await send({"type": "http.response.body", "body": body})

        from ..serving import serving

        if refused is not None:
            await refused(scope, receive, guarded)
            return
        with serving(account_of(s)):  # whose work this request is while it is served (uploads.resolve reads it)
            await self.app(scope, receive, guarded)
        if need is not None and method not in ("GET", "HEAD"):  # every admin action that went through, in the log
            who, role = actor(s)
            said = Msg("I-AUDIT-ACTION", who=who, role=role, what=roles.CAPABILITIES[need].what, method=method, path=path, status=status)
            await off_loop(lambda: audit(said, session=s, method=method, path=path))

    def look(self, request: Request, method: str, path: str):
        """What the guard must know before a request goes on, all of it blocking (the session is a database read; the
        names hidden from its account are built once per set of tags): run off the event loop (wire.off_loop).
        Returns the session, the capability the route needs, the refusal (None: it may go on), whether its answer is
        scrubbed of the server's folders, the pattern of names hidden from it, and the fields of its answer this session
        does not get (server/available.py FIELDS: declared once, resolved like every other subject)."""
        from . import available

        s = auth.session(request)
        need = need_of(method, path)
        refused = self.refusal(request, method, path, s, need)
        privileged = s is not None and s.can("logs.view")  # sees where the server keeps things
        scrubbing = not privileged and method != "HEAD" and not VERBATIM.match(path)
        hidden = hidden_pattern(s.user) if s is not None and scrubbing else None
        fields = available.hidden_paths(s, method, path) if refused is None and method != "HEAD" else ()
        return s, need, refused, scrubbing, hidden, fields

    def refusal(self, request: Request, method: str, path: str, s, need: str | None) -> JSONResponse | None:
        g = auth.guards()
        if blocked := g.watch.is_blocked(request):
            return refused(Msg("E-ACCESS-BLOCKED", minutes=int(blocked / 60) + 1), 429)
        if not auth.host_known(request):  # another site's name pointing here (DNS rebinding), or a name not yet listed
            g.watch.note(request, "陌生的域名", request.headers.get("host", "")[:200])
            return refused(Msg("E-ACCESS-UNKNOWNHOST", host=request.headers.get("host", "")[:200]), 421)
        key, own = auth.client_key(request), limit_of(method, path)
        if not g.rate.take(key) or (own is not None and own[1].burst is not None
                                    and not g.rate.take(f"{own[0]} {key}", own[1].burst, own[1].per_s)):
            g.watch.note(request, "请求太频繁")  # every session counts, the administrator's too
            return refused(Msg("E-ACCESS-TOOFAST"), 429)
        raw, query = request.scope.get("raw_path") or request.scope["path"].encode(), request.scope.get("query_string") or b""
        if probe_path(raw, method, path, s) or PROBE_QUERY.search(query):
            g.watch.note(request, "路径可疑", (raw + (b"?" + query if query else b"")).decode("latin-1")[:200])
        if PROBE_QUERY.search(query):  # going up or NUL in a parameter: never a name any page sends (defence in depth)
            return refused(Msg("E-ACCESS-PROBE"), 400)
        if _SLASHES.search(path):  # "//api//admin/x": the routes match the path as sent, so the guard must too;
            g.watch.note(request, "路径可疑", path[:200])  # collapsing it here would classify one address and serve another
            return refused(Msg("E-ACCESS-NOROUTE"), 404)
        where = level(method, path)
        if where is None:
            # Logged in, and the request says it comes from a page of this site, yet the endpoint is not here. During an
            # upgrade the page may be newer than the server or older, and an open page keeps asking: counting that would
            # lock a person out after a few dozen clicks. A script holding someone's login looks just the same (Origin is
            # a header anyone can send), so the note says only what is known, never which of the two it was. It is
            # recorded under the admin page's 「安全」 with the account, not counted toward blocking.
            ours = s is not None and same_site(request)
            g.watch.note(request, "未开放的接口", "已登录，请求自称来自本站页面；不计入封禁" if ours else "", counts=not ours)
            return refused(Msg("E-ACCESS-NOROUTE"), 404)
        writes = method not in ("GET", "HEAD")
        if where == "admin" and (s is None or not s.capabilities):  # no rights at all now: the page asks for the password
            g.watch.note(request, "没有管理员权限就访问管理接口")
            return refused(Msg("E-ACCESS-ADMINONLY"), 401, admin=True)
        if where == "admin" and path.startswith(ADMIN):
            if need is None:
                g.watch.note(request, "未开放的接口")
                return refused(Msg("E-ACCESS-NOROUTE"), 404)
            if not s.can(need):  # a role without this one: refused with what it lacks and who has it
                g.watch.note(request, "角色没有这项权限", need)
                said = Msg("E-ACCESS-NOCAPABILITY", role=roles.label(s.user.role), what=roles.CAPABILITIES[need].what,
                           roles=roles.holders(need))
                who, role = actor(s)
                if method not in ("GET", "HEAD"):
                    audit(Msg("W-AUDIT-REFUSED", who=who, role=role, what=roles.CAPABILITIES[need].what, method=method, path=path, code=said.code),
                          session=s, method=method, path=path)
                return refused(said, 403)
        if where == "user" and s is None:
            if path.startswith("/api/") and auth.token_of(request):
                # a login of ours that ended (another login took its place, or it lapsed) is answered and not counted:
                # the page that was kicked keeps asking for the queue and the load until someone looks at it, and
                # counting those would block the very person who had been kicked from logging in again. Only a token
                # this server never gave out is odd.
                info = auth.accounts.kicked_info(auth.credential(request))
                if info:
                    kicked = auth.kicked_detail(info)
                    return refused(kicked, 401, login=True, kicked={**info, "detail": kicked.text, "code": kicked.code})
                if not auth.accounts.once_issued(auth.credential(request)):
                    g.watch.note(request, "登录凭证不是这台服务器发的")
            return refused(Msg("E-ACCESS-SIGNIN"), 401, login=True)
        if where != "open" and not path.startswith(AUTH) and (owed := terms.owed(s.user)) is not None:
            return refused(Msg("E-TERMS-OWED"), 403, terms=owed)
        if where == "user" and need is not None and not s.can(need):  # a product page's action that is not everyone's
            if need in s.user.capabilities:  # the role has it, its three days ran out: the password again
                return refused(Msg("E-ACCESS-ADMINONLY"), 401, admin=True)
            g.watch.note(request, "角色没有这项权限", need)
            said = Msg("E-ACCESS-NOCAPABILITY", role=roles.label(s.user.role), what=roles.CAPABILITIES[need].what,
                       roles=roles.holders(need))
            if writes:
                who, role = actor(s)
                audit(Msg("W-AUDIT-REFUSED", who=who, role=role, what=roles.CAPABILITIES[need].what, method=method, path=path, code=said.code),
                      session=s, method=method, path=path)
            return refused(said, 403)
        if writes and not auth.by_header(request) and not same_site(request):  # the cookie's, or logging in
            g.watch.note(request, "跨站请求")
            return refused(Msg("E-ACCESS-CROSSSITE"), 403)
        most = own[1].body if own is not None else MAX_BODY
        if writes and most is not None:
            length = request.headers.get("content-length")
            if length is None and "chunked" in request.headers.get("transfer-encoding", "").lower():
                return refused(Msg("E-ACCESS-NOLENGTH"), 411)
            if length is not None and (not length.isdigit() or int(length) > most):
                g.watch.note(request, "请求太大", f"{length} 字节")
                return refused(Msg("E-ACCESS-TOOBIG", mb=round(most / (1 << 20), 2)), 413)
        return None
