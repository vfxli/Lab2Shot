"""Every route of this server says who may use it where it is registered, once: the decorator carries its access,
and nothing else lists routes again.

    for example, on router.get("/status", ...):     access=Access.user("编辑器：算到哪了")
    and on admin.put("/gpus", ...):                 access=Access.admin("gpu.authorize")

    Access.open(why)                 anyone, before logging in (the gate, logging in, whether the server is up)
    Access.user(why, needs=...)      whoever logged in; `needs`: a capability besides (lab2shot/roles.py), for a product
                                     page's action that is not everyone's (no route uses it at present)
    Access.admin(capability)         a route under /api/admin/, for a role with that capability and fresh rights
    Access.page(why)                 the built web page (outside /api/)
    limit=Limit(...)                 the route's own body size and rate, instead of the guard's defaults

A Router (FastAPI's APIRouter) refuses a route without an access, one declared twice, and an admin level off /api/admin/
(or a route under it without one), when the module is imported. server/access.py's guard looks each request up here
(match); server/available.py derives what the pages show from the same declarations (needs)."""

from __future__ import annotations

import functools
import inspect
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import compile_path

from ..transfer import BODY_MAX

ADMIN = "/api/admin/"

# What this server takes from one account at once, whatever it sends. These are safety limits, not product settings
# (a number that can blow up memory or time is not left to a settings form) — they are declared here, once, and the one
# place that enforces each is named next to it. A limit that stops a real piece of work is a bug: raise the limit.
MAX_BODY = BODY_MAX  # bytes one request may carry: a graph, a form — far below this (declared in transfer/__init__.py: uploads size the head by it)
MAX_NODES = 1000  # nodes one graph may hold (the biggest template is far below this): server/access.py admit()
MAX_STREAMS = 16  # event streams one account may keep open at once (a page follows a few jobs): server/auth.py Streams


@dataclass(frozen=True)
class Limit:
    """A route's own limits, instead of the guard's defaults. `body`: bytes a request may carry (None: the route checks
    its own, larger limit as it reads: uploads, feedback). `burst` and `per_s`: requests one client may send to it at
    once and a second after that (None: only the rate every request counts against, auth.Rate)."""

    body: int | None = MAX_BODY
    burst: float | None = None
    per_s: float | None = None


@dataclass(frozen=True)
class Access:
    level: str  # open, user, admin, page
    why: str = ""  # what it is for, said to whoever reads the declarations (the OpenAPI summary)
    needs: str | None = None  # the capability it needs (admin: always; user: a product page's action)
    limit: Limit | None = None
    # fields of its answer not everyone gets: a subject of server/available.py FIELDS -> the JSON paths of those fields
    # (keys by dots, `[]` every item, `*` every value; the last key is removed). The guard strips them for a session
    # the subject is not available to: a JSON body whole, each event of a stream on its own.
    hides: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # an answer made from slowly changing state: its key (server/revisions.py) is its ETag; a request whose
    # If-None-Match has it gets 304 without the handler running, and one with the same key gets the answer made once
    keyed: Callable[[Request], str] | None = None
    # whose the data it reads or changes is (server/owners.py): checked before the handler, which finds what it loaded in
    # request.state.owned; no handler checks it itself
    owned: Callable[..., object] | None = None

    @classmethod
    def open(cls, why: str, limit: Limit | None = None) -> Access:
        return cls("open", why, None, limit)

    @classmethod
    def user(cls, why: str, needs: str | None = None, limit: Limit | None = None,
             hides: Mapping[str, tuple[str, ...]] | None = None, keyed: Callable[[Request], str] | None = None,
             owned: Callable[..., object] | None = None) -> Access:
        return cls("user", why, needs, limit, hides or {}, keyed, owned)

    @classmethod
    def admin(cls, needs: str, limit: Limit | None = None, hides: Mapping[str, tuple[str, ...]] | None = None,
              owned: Callable[..., object] | None = None) -> Access:
        return cls("admin", "", needs, limit, hides or {}, None, owned)

    @classmethod
    def page(cls, why: str) -> Access:
        return cls("page", why)


DECLARED: dict[str, Access] = {}  # "METHOD /path" -> its access, in the order the modules declared them
_table: list[tuple[str, re.Pattern, str, Access]] = []


class RouteDeclarationError(Exception):
    """A route declared against the rules above: a bug in this server's code, raised when its module is imported."""


def declare(method: str, path: str, access: Access) -> None:
    key = f"{method} {path}"
    if not isinstance(access, Access):
        raise RouteDeclarationError(key)
    if key in DECLARED:
        raise RouteDeclarationError(f"{key} twice")
    admin, page = access.level == "admin", access.level == "page"
    if admin != path.startswith(ADMIN) or page == path.startswith("/api/") or (admin and access.needs is None) \
            or (access.level in ("open", "page") and access.needs is not None):
        raise RouteDeclarationError(f"{key} {access.level}")
    DECLARED[key] = access
    _table.clear()


def match(method: str, path: str) -> tuple[str, Access] | None:
    """The route a request is for and its access (None: no route of that method and path). The route whose fixed part
    before its first parameter is longest wins: POST /api/admin/users/{user_id}/password over .../users/{user_id}, and every
    route over the page's GET /{path:path}."""
    if not _table:
        keys = sorted(DECLARED, key=lambda k: -len(k.split(" ", 1)[1].split("{", 1)[0]))
        _table.extend((k.split(" ", 1)[0], compile_path(k.split(" ", 1)[1])[0], k, DECLARED[k]) for k in keys)
    return next(((key, a) for m, rx, key, a in _table if m == method and rx.match(path)), None)


KEPT_ANSWERS = 32  # answers kept per keyed route (one per account kind and state: a handful in practice)


def _answered_by_key(handler: Callable, keyed: Callable[[Request], str]) -> Callable:
    """`handler` (a sync route taking `request`) answered from its key: 304 when the page has it, else the answer made
    once for that key (the last KEPT_ANSWERS kept), with the key as its ETag."""
    kept: OrderedDict[str, bytes] = OrderedDict()
    lock = threading.Lock()

    @functools.wraps(handler)
    def answer(*args, **kwargs):
        request: Request = kwargs["request"]
        tag = f'W/"{keyed(request)}"'
        if tag in (t.strip() for t in request.headers.get("if-none-match", "").split(",")):
            return Response(status_code=304, headers={"ETag": tag})
        with lock:
            body = kept.get(tag)
            if body is not None:
                kept.move_to_end(tag)
        if body is None:
            body = JSONResponse(jsonable_encoder(handler(*args, **kwargs))).body
            with lock:
                kept[tag] = body
                while len(kept) > KEPT_ANSWERS:
                    kept.popitem(last=False)
        return Response(body, media_type="application/json", headers={"ETag": tag})

    return answer


def _owner_first(handler: Callable, owned: Callable[..., object], key: str) -> Callable:
    """`handler` (a sync route taking `request`) run after its owner (server/owners.py), what the owner loaded in
    request.state.owned. The owner takes the request and, by name, whichever of the handler's parameters it asks for."""
    if inspect.iscoroutinefunction(handler) or "request" not in inspect.signature(handler).parameters:
        raise RouteDeclarationError(f"{key}: a route that declares whose its data is takes `request` and is not async")
    wanted = [n for n in inspect.signature(owned).parameters if n != "request"]

    @functools.wraps(handler)
    def answer(*args, **kwargs):
        request: Request = kwargs["request"]
        request.state.owned = owned(request, **{n: kwargs[n] for n in wanted if n in kwargs})
        return handler(*args, **kwargs)

    return answer


class Router(APIRouter):
    """FastAPI's router whose routes each declare their access (see the module doc)."""

    def _declared(self, method: str, path: str, access: Access, register: Callable) -> Callable:
        declare(method, self.prefix + path, access)
        if access.keyed is None and access.owned is None:
            return register

        def decorate(handler: Callable) -> Callable:
            if access.owned is not None:
                handler = _owner_first(handler, access.owned, f"{method} {self.prefix}{path}")
            if access.keyed is not None:
                handler = _answered_by_key(handler, access.keyed)
            return register(handler)

        return decorate

    def get(self, path: str, *, access: Access, **kw) -> Callable:  # type: ignore[override]
        return self._declared("GET", path, access, super().get(path, **kw))

    def post(self, path: str, *, access: Access, **kw) -> Callable:  # type: ignore[override]
        return self._declared("POST", path, access, super().post(path, **kw))

    def put(self, path: str, *, access: Access, **kw) -> Callable:  # type: ignore[override]
        return self._declared("PUT", path, access, super().put(path, **kw))

    def patch(self, path: str, *, access: Access, **kw) -> Callable:  # type: ignore[override]
        return self._declared("PATCH", path, access, super().patch(path, **kw))

    def delete(self, path: str, *, access: Access, **kw) -> Callable:  # type: ignore[override]
        return self._declared("DELETE", path, access, super().delete(path, **kw))


def mount(app, path: str, served, *, access: Access, name: str) -> None:
    """A mounted folder of files (the built page's scripts), declared like a route."""
    declare("MOUNT", path, access)
    app.mount(path, served, name=name)
