"""Every route of this server says who may use it where it is registered, once: the decorator carries its access,
and nothing else lists routes again.

    for example, on router.get("/status", ...):     access=Access.user("编辑器：算到哪了")
    and on admin.put("/gpus", ...):                 access=Access.admin("gpu.authorize")

    Access.open(why)                 anyone, before logging in (the gate, logging in, whether the server is up)
    Access.user(why)                 whoever logged in
    Access.admin(capability)         a route under /api/admin/, for a role with that capability and fresh rights
    Access.admin(capability, local=True)  ... and only from the command line on this machine (its machine token),
                                     whatever rights a login in a browser has
    Access.page(why)                 the built web page (outside /api/)
    limit=Limit(...)                 the route's own body size and rate, instead of the guard's defaults
    lane=...                         the threads its handler runs on, instead of the routes' shared ones (server/wire.py
                                     LANE_THREADS): a handler that may wait long (for a password check, the view
                                     worker, a slot) waits there, and never takes a thread every other route needs
    snapshot=True                    a poll that reads state already kept (the queue, the load): it never waits for
                                     its account's slots (Route)

A Router (FastAPI's APIRouter) refuses a route without an access, one declared twice, and an admin level off /api/admin/
(or a route under it without one), when the module is imported. server/access.py's guard looks each request up here
(match); server/available.py derives what the pages show from the same declarations (needs)."""

from __future__ import annotations

import functools
import inspect
import json
import math
import re
import threading
import types
import typing
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Annotated

from fastapi import APIRouter
from fastapi.routing import APIRoute
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import compile_path

from .. import i18n
from ..config import QUOTA_GB_MAX
from ..errors import Invalid, TooLarge
from ..messages import Msg, localized
from ..transfer import BODY_MAX

ADMIN = "/api/admin/"

# What this server takes from one account at once, whatever it sends. These are safety limits, not product settings
# (a number that can blow up memory or time is not left to a settings form) — they are declared here, once, and the one
# place that enforces each is named next to it. A limit that stops a real piece of work is a bug: raise the limit.
MAX_BODY = 4 << 20  # bytes a request may carry unless its route declares more (Limit): a graph, a form — far below this; server/access.py
BIG_BODY = BODY_MAX  # what the upload routes that take many names or a file's head declare (transfer/__init__.py: its head is sized by it)
OPEN_BODY = 64 << 10  # bytes a request to an open route may carry (logging in, registering: a form): Access.open
MAX_NODES = 1000  # nodes one graph may hold (the biggest template is far below this): server/access.py admit()
MAX_DEPTH = 64  # levels of nesting a JSON body may have (a graph is under ten): read_json, for every route (Route) and feedback
MAX_STREAMS = 16  # event streams one account may keep open at once (a page follows a few jobs): server/auth.py Streams
ACCOUNT_SLOTS = 2  # handlers one account runs at once on the routes' shared threads, and on each lane; the rest wait: Route
STRANGER_SLOTS = 8  # requests without a login a lane holds at once, waiting or running (logging in); past it refused: Route


@dataclass(frozen=True)
class Limit:
    """A route's own limits, instead of the guard's defaults. `body`: bytes a request may carry (None: the route checks
    its own, larger limit as it reads: an upload's parts). `burst` and `per_s`: requests one client may send to it at
    once and a second after that, counted apart from its other requests (None: it counts against the rate every other
    request does, auth.Rate)."""

    body: int | None = MAX_BODY
    burst: float | None = None
    per_s: float | None = None


@dataclass(frozen=True)
class Access:
    level: str  # open, user, admin, page
    why: str = ""  # what it is for, said to whoever reads the declarations (the OpenAPI summary)
    needs: str | None = None  # the capability it needs (admin routes only, always one)
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
    # the lane of threads its (sync) handler runs on (server/wire.py LANE_THREADS); None: the routes' shared threads
    lane: str | None = None
    # a poll reading state already kept: answered without waiting for its account's slots (Route)
    snapshot: bool = False
    # only the command line on this machine may use it (server/access.py refusal): stopping the server, clearing the
    # counts of wrong passwords
    local: bool = False
    # the fields of its request body naming one account's data (a job, a packet, a template...), which its `owned`
    # resolver checks or filters to the asking account's: `lab2shot check routes` holds every body field named like
    # account data (cli/check_arch.py OWNED_BODY_FIELDS) to being declared here, and every one declared to an owner
    body_ids: tuple[str, ...] = ()

    @classmethod
    def open(cls, why: str, limit: Limit | None = None, lane: str | None = None) -> Access:
        """Anyone's, before logging in: whatever else its Limit says, its body is never more than OPEN_BODY (a stranger
        may not make this server take MAX_BODY at every login attempt)."""
        return cls("open", why, None, replace(limit or Limit(), body=OPEN_BODY), lane=lane)

    @classmethod
    def user(cls, why: str, limit: Limit | None = None, hides: Mapping[str, tuple[str, ...]] | None = None,
             keyed: Callable[[Request], str] | None = None, owned: Callable[..., object] | None = None,
             lane: str | None = None, snapshot: bool = False, body_ids: tuple[str, ...] = ()) -> Access:
        return cls("user", why, None, limit, hides or {}, keyed, owned, lane, snapshot, body_ids=body_ids)

    @classmethod
    def admin(cls, needs: str, limit: Limit | None = None, hides: Mapping[str, tuple[str, ...]] | None = None,
              owned: Callable[..., object] | None = None, keyed: Callable[[Request], str] | None = None,
              lane: str | None = None, local: bool = False, body_ids: tuple[str, ...] = ()) -> Access:
        return cls("admin", "", needs, limit, hides or {}, keyed, owned, lane, local=local, body_ids=body_ids)

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
    if admin != path.startswith(ADMIN) or page == path.startswith("/api/") or admin != (access.needs is not None):
        raise RouteDeclarationError(f"{key} {access.level}")
    if access.keyed is not None and access.owned is not None:  # a 304 by its key would answer before its owner is checked
        raise RouteDeclarationError(f"{key}: keyed and owned")
    if access.body_ids and access.owned is None:  # body fields naming account data, and nobody checking whose
        raise RouteDeclarationError(f"{key}: body_ids without owned")
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
        tag = f'W/"{keyed(request)}.{i18n.current()}"'  # the same state said in another language is another answer
        if tag in (t.strip() for t in request.headers.get("if-none-match", "").split(",")):
            return Response(status_code=304, headers={"ETag": tag})
        with lock:
            body = kept.get(tag)
            if body is not None:
                kept.move_to_end(tag)
        if body is None:
            body = JSONResponse(localized(jsonable_encoder(handler(*args, **kwargs)))).body
            with lock:
                kept[tag] = body
                while len(kept) > KEPT_ANSWERS:
                    kept.popitem(last=False)
        return Response(body, media_type="application/json", headers={"ETag": tag})

    return answer


def _answered_here(handler: Callable) -> Callable:
    """`handler` (a sync route) made to answer with its bytes: what it returns is encoded as JSON on its own thread,
    never by FastAPI on the event loop (a big answer, tens of MB of names, would hold every request up while it is
    written). One that sets its answer's headers itself (a `response` it takes: a cookie) is left to FastAPI, which
    carries them."""
    hints = typing.get_type_hints(handler)
    if inspect.iscoroutinefunction(handler) or any(isinstance(t, type) and issubclass(t, Response)
                                                    for name, t in hints.items() if name != "return"):
        return handler

    @functools.wraps(handler)
    def answer(*args, **kwargs):
        got = handler(*args, **kwargs)
        if isinstance(got, Response):
            return got
        # every message in it said in the request's language: what was kept or made elsewhere (a job's record, a
        # packet's manifest) too (messages.localized)
        content = localized(jsonable_encoder(got))
        try:
            return JSONResponse(content)
        except ValueError:  # a number JSON cannot say (nan, inf: a packet's meta may hold one): said as null
            return JSONResponse(_finite(content))

    return answer


def _finite(value):
    """`value` (a JSON value) with every number JSON cannot write (nan, inf) as None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [_finite(v) for v in value]
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    return value


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


def _on_lane(handler: Callable, lane: str, key: str) -> Callable:
    """`handler` (a sync route) run on a thread of `lane` (wire.off_loop), the request's context with it."""
    from .wire import LANE_THREADS, off_loop

    if lane not in LANE_THREADS or inspect.iscoroutinefunction(handler):
        raise RouteDeclarationError(f"{key}: lane {lane} is for a sync handler, one of {sorted(LANE_THREADS)}")

    @functools.wraps(handler)
    async def answer(*args, **kwargs):
        return await off_loop(functools.partial(handler, *args, **kwargs), lane=lane)

    return answer


INT_RANGE = (-(1 << 63), (1 << 63) - 1)  # what an SQLite INTEGER holds
MOMENT_MAX = 253402300799.0  # 9999-12-31 23:59:59 UTC: the latest time every date function can still write
DAYS_MAX = 36500  # a hundred years


def _bound(kind):
    """`kind` with every number it holds bounded, the one rule for a number a request carries: an int within INT_RANGE,
    a float finite (never nan or inf, which JSON as Python reads it lets through), whether it stands alone, may be None,
    or sits in a list or a mapping. A number already Annotated keeps what its kind declares (Moment, Gigabytes, Days,
    Count). For the path and the query (_bounded) and for every body (Body)."""
    if kind is int:
        return Annotated[int, Field(ge=INT_RANGE[0], le=INT_RANGE[1])]
    if kind is float:
        return Annotated[float, Field(allow_inf_nan=False)]
    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if origin is list and args:
        return list[_bound(args[0])]
    if origin is dict and len(args) == 2:
        return dict[args[0], _bound(args[1])]
    if origin is types.UnionType or origin is typing.Union:
        return typing.Union[tuple(_bound(a) for a in args)]
    return kind


class Body(BaseModel):
    """What every request body is read as: each model a route takes is one of these, and each of its fields is
    bounded (_bound) where the model is defined. What a number means narrows it further where its kind is declared,
    once (Moment, Gigabytes, Days, Count), never by a route's own check."""

    model_config = ConfigDict(allow_inf_nan=False)

    @classmethod
    def __pydantic_init_subclass__(cls, **kw) -> None:
        super().__pydantic_init_subclass__(**kw)
        bounded = {}
        for name, f in cls.model_fields.items():
            if f.metadata:  # a kind of its own (Moment ...): its bounds are declared with it
                continue
            if (kind := _bound(f.annotation)) != f.annotation:
                bounded[name] = FieldInfo.merge_field_infos(f, annotation=kind)
        if bounded:
            cls.model_fields.update(bounded)
            cls.model_rebuild(force=True)


Moment = Annotated[float, Field(ge=0, le=MOMENT_MAX, allow_inf_nan=False)]  # a time (seconds since 1970): an expiry, a day
Gigabytes = Annotated[float, Field(ge=0, le=QUOTA_GB_MAX, allow_inf_nan=False)]  # a disk quota, as the settings' own
Days = Annotated[float, Field(ge=0, le=DAYS_MAX, allow_inf_nan=False)]  # how many days back
Count = Annotated[int, Field(ge=0, le=INT_RANGE[1])]  # a size in bytes, a number of things


def _bounded(handler: Callable) -> Callable:
    """`handler` with each number it takes from the path or the query bounded where FastAPI reads it (_bound). A
    request with any other (20 digits, nan, inf) is refused as bad input before the handler runs, whichever route it
    is: the handlers never see one. Its body is a Body, bounded where that model is defined."""
    hints = typing.get_type_hints(handler, include_extras=True)
    sig = inspect.signature(handler)
    handler.__signature__ = sig.replace(parameters=[p.replace(annotation=_bound(hints.get(p.name, p.annotation)))
                                                    for p in sig.parameters.values()],
                                        return_annotation=hints.get("return", sig.return_annotation))
    return handler


def read_json(body: bytes | bytearray):
    """A request's JSON body, or Invalid: not JSON, or nested deeper than MAX_DEPTH. Python's parser and everything
    after it (validation, its error answer, the handlers) walk a body recursively; one nested thousands deep would stop
    them with a RecursionError, a 500, so the depth is bounded here, once, without recursing. What a body builds deeper
    than it is itself (a graph patch's path: each key a level) is bounded where that is declared (graphs.Patch)."""
    try:
        value = json.loads(body)
    except RecursionError:
        raise Invalid(Msg("E-ACCESS-TOODEEP", most=MAX_DEPTH)) from None
    except ValueError:
        raise Invalid(Msg("E-ACCESS-NOTJSON")) from None
    level = [(value, 1)]
    while level:
        item, depth = level.pop()
        if isinstance(item, (dict, list)):
            if depth > MAX_DEPTH:
                raise Invalid(Msg("E-ACCESS-TOODEEP", most=MAX_DEPTH))
            level.extend((v, depth + 1) for v in (item.values() if isinstance(item, dict) else item))
    return value


async def read_body(request: Request, most: int) -> bytes:
    """A request's body, read as it comes and refused (TooLarge) the moment it is more than `most`, whatever its
    Content-Length said: the one way a route's body is read (Route). Kept on the request for whatever reads it next."""
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > most:
            raise TooLarge(Msg("E-ACCESS-TOOBIG", mb=round(most / (1 << 20), 2)))
    request._body = bytes(body)
    return request._body


def declared_as(method: str) -> str:
    """The method a request's route is declared under: a HEAD is its GET's (Router.get registers both)."""
    return "GET" if method == "HEAD" else method


def _json_typed(request: Request) -> bool:
    """The body is one FastAPI reads as JSON: application/json or application/*+json (without a type it does not)."""
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    return kind == "application/json" or (kind.startswith("application/") and kind.endswith("+json"))


class Route(APIRoute):
    """The body of a route that takes one is read here (read_body: never past its Limit, whatever the request says) and,
    JSON, parsed on a thread (read_json) before FastAPI reads it: a bad one is answered like any other bad input (400
    with its message) instead of by FastAPI's own words or a 500, and parsing never holds up the event loop. A route
    that takes no body has none read for it: an upload's part reads its own bytes as they come.

    And here, once, a signed-in account's handler that runs on threads (a sync handler: the routes' shared threads, or
    its lane's) holds one of that account's ACCOUNT_SLOTS on those threads (server/wire.py account_slot): its other
    requests there wait their turn. The process has one interpreter and each lane a few threads for everyone: without
    it, a few requests of one account that are heavy on the CPU (declaring tens of thousands of files) or that wait
    long (the view worker, a slot for a picture's planes) take what every other account's requests need. Only the
    handler holds the slot, never the sending of its answer (a download, a stream). A route declared snapshot never
    waits for one, so a page still sees the queue and the load while its account's heavy requests go on."""

    def get_route_handler(self) -> Callable:
        from ..traffic import SCOPE_USER
        from .wire import account_slot, off_loop, stranger_slot

        handle = super().get_route_handler()
        declared = [DECLARED.get(f"{declared_as(m)} {self.path}") for m in self.methods]
        lane = next((a.lane for a in declared if a is not None and a.lane), None)
        counted = (lane is not None or not inspect.iscoroutinefunction(self.dependant.call)) and all(
            a is not None and a.level in ("user", "admin") and not a.snapshot for a in declared)
        # the body is read here only for a route that takes one (a model), never for one that reads its own bytes as
        # they come (Limit body None: an upload's part, which counts them against the size it declared)
        reads = bool(self.dependant.body_params)
        most = next((a.limit.body if a.limit is not None else MAX_BODY for a in declared if a is not None), MAX_BODY)
        if reads and most is None:
            raise RouteDeclarationError(f"{self.path}: a route that takes a body declares how big it may be (Limit body)")

        async def read_and_handle(request: Request) -> Response:
            if reads:
                body = await read_body(request, most)
                if _json_typed(request) and body:
                    # parsed on a thread (wire.off_loop), never on the event loop every account's requests are answered on
                    request._json = await off_loop(read_json, body)  # starlette's Request.json(), which FastAPI calls next, answers with it
            return await handle(request)

        async def handler(request: Request) -> Response:
            if counted and (user := request.scope.get(SCOPE_USER, 0)):
                async with account_slot(user, lane, ACCOUNT_SLOTS):  # its body read and parsed in its slot too
                    return await read_and_handle(request)
            if lane is not None:  # nobody's (logging in, registering): all of them together hold a few places of it
                async with stranger_slot(lane, STRANGER_SLOTS):
                    return await read_and_handle(request)
            return await read_and_handle(request)

        return handler


class Router(APIRouter):
    """FastAPI's router whose routes each declare their access (see the module doc), and read their body (Route)."""

    def __init__(self, **kw) -> None:
        super().__init__(route_class=Route, **kw)

    def _declared(self, method: str, path: str, access: Access, register: Callable) -> Callable:
        declare(method, self.prefix + path, access)

        def decorate(handler: Callable) -> Callable:
            handler = _bounded(handler)
            if access.keyed is None:  # a keyed answer is made into bytes by its key's cache, on the same thread
                handler = _answered_here(handler)
            if access.owned is not None:
                handler = _owner_first(handler, access.owned, f"{method} {self.prefix}{path}")
            if access.keyed is not None:
                handler = _answered_by_key(handler, access.keyed)
            if access.lane is not None:
                handler = _on_lane(handler, access.lane, f"{method} {self.prefix}{path}")
            return register(handler)

        return decorate

    def get(self, path: str, *, access: Access, **kw) -> Callable:  # type: ignore[override]
        # a HEAD is the GET without its body, under the GET's declaration (server/access.py route_of); not listed apart
        get, head = super().get(path, **kw), super().head(path, **{**kw, "include_in_schema": False})

        def register(handler: Callable) -> Callable:
            head(handler)
            return get(handler)

        return self._declared("GET", path, access, register)

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
