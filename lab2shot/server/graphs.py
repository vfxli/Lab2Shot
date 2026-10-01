"""Graph transfer for editor requests: each graph version is sent in full at most once.

The page identifies each version of its graph by a key. The first request for a version carries either the whole
graph or a patch against a version the server already holds; subsequent requests carry only the key. Requests from
scripts and DCC plugins always carry the whole graph. The server keeps a small number of recent versions per browser
session. A request naming a version the server does not hold (after a restart or eviction) is answered with 409
{"graph": "unknown"}, and the page resends the whole graph.

A patch operates on the graph in tree form, with nodes keyed by id and their order stored separately (the same
steps as webui/src/model/graphPatch.ts): a list of {"p": [path...], "v": value} (set) and {"p": [path...]} (remove).
"""

from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict
from typing import Annotated, Any

from fastapi import Request
from pydantic import Field

from ..engine.graph import Graph, GraphError, check_shape
from ..errors import Invalid, MessageError, TooLarge
from ..messages import Msg
from . import auth
from .routes import MAX_DEPTH, Body

KEEP_PER_SESSION = 6  # versions kept per browser session
# the versions one session keeps, counted as their JSON (held as dicts they take several times as much memory, about
# 5-8x): a graph is tens to hundreds of KB. One version alone is never more (resolve: a patch that makes it bigger is
# refused), and past it the session's own oldest go
KEEP_SESSION_BYTES = 4 << 20
KEEP_SESSIONS = 2000  # sessions whose versions are kept: past it, the one used longest ago forgets all of its
_lock = threading.Lock()
# session -> its versions, key -> (graph as a tree, its JSON size); both least recently used first
_kept: OrderedDict[str, OrderedDict[str, tuple[dict, int]]] = OrderedDict()


class Patch(Body):
    """One change of a patch: set (`v` given) or remove (no `v`) what `p` names, key by key from the top. The path is
    no deeper than a body may nest (MAX_DEPTH): each key makes a level of the graph, which later reads walk."""

    p: Annotated[list[str], Field(max_length=MAX_DEPTH)]
    v: Any = None


class GraphRequest(Body):
    """The graph of a request: the whole `graph` (scripts, plugins, first request of a version), a `key` naming a
    version the server holds, or `base` + `patch` producing version `key` from a held version."""

    graph: dict | None = None
    key: str = ""
    base: str = ""
    patch: list[Patch] | None = None


def _session(request: Request | None) -> str:
    """Return the session that owns the versions; keys of other browser sessions are never consulted."""
    return auth.client_key(request) if request is not None else ""


def tree(graph: dict) -> dict:
    """A graph as versions are kept and patched: its nodes by id, their order apart. Only a graph of the file's shape
    (check_shape: GraphError otherwise, as for a graph sent whole)."""
    check_shape(graph)
    nodes = graph["nodes"]
    return {**graph, "nodes": {n["id"]: n for n in nodes}, "node_order": [n["id"] for n in nodes]}


def untree(t: dict) -> dict:
    """The graph a kept (or patched) version is: GraphError when a patch left it no graph's shape."""
    order, nodes = t.get("node_order", []), t.get("nodes", {})
    if not isinstance(order, list) or not isinstance(nodes, dict) or not all(isinstance(i, str) for i in order):
        raise GraphError(Msg("E-GRAPH-BADPATCH"))
    return {**{k: v for k, v in t.items() if k != "node_order"}, "nodes": [nodes[i] for i in order if i in nodes]}


def apply(t: dict, ops: list[Patch]) -> dict:
    """Return a copy of `t` with the patch operations applied."""
    out = copy.deepcopy(t)
    for op in ops:
        path, setting = op.p, "v" in op.model_fields_set
        if not path:
            if not setting or not isinstance(op.v, dict):
                raise Invalid(Msg("E-GRAPH-BADPATCH"))
            out = copy.deepcopy(op.v)
            continue
        at = out
        for k in path[:-1]:
            if not isinstance(at.get(k), dict):
                at[k] = {}
            at = at[k]
        if setting:
            at[path[-1]] = copy.deepcopy(op.v)
        else:
            at.pop(path[-1], None)
    return out


def _keep(session: str, key: str, t: dict, size: int) -> None:
    """Keep version `key` of the session's graph (one admitted for its account: resolve): the session's oldest go while it
    keeps more than KEEP_PER_SESSION or KEEP_SESSION_BYTES (never the one kept now), and past KEEP_SESSIONS sessions the
    one used longest ago. What one session sends never pushes out another's versions."""
    with _lock:
        mine = _kept.setdefault(session, OrderedDict())
        _kept.move_to_end(session)
        mine.pop(key, None)
        mine[key] = (t, size)
        while len(mine) > 1 and (len(mine) > KEEP_PER_SESSION or sum(v[1] for v in mine.values()) > KEEP_SESSION_BYTES):
            mine.popitem(last=False)
        while len(_kept) > KEEP_SESSIONS:
            _kept.popitem(last=False)


def _get(session: str, key: str) -> dict | None:
    with _lock:
        mine = _kept.get(session)
        got = mine.get(key) if mine is not None else None
        if got:
            _kept.move_to_end(session)
            mine.move_to_end(key)
        return got[0] if got else None


class UnknownGraph(MessageError):
    """The request names a version the server does not hold: the page sends the whole graph again ("graph": "unknown")."""

    status = 409

    def __init__(self) -> None:
        super().__init__(Msg("E-GRAPH-UNKNOWNVERSION"))

    def answer(self) -> dict:
        return {**super().answer(), "graph": "unknown"}


def resolve(req: GraphRequest, request: Request) -> tuple[dict, Graph]:
    """The graph a request names, as JSON and admitted for its account (server/access.py admit); kept under its key for
    subsequent requests only once admitted: what is no graph this account may use never takes a place."""
    from .access import admit

    session = _session(request)
    if req.key and (t := _get(session, req.key)) is not None:
        data = untree(t)
        return data, admit(request, data)
    if req.graph is not None:
        t = tree(req.graph) if req.key else None
    elif req.key and req.base and req.patch is not None:
        base = _get(session, req.base)
        if base is None:
            raise UnknownGraph()
        t = apply(base, req.patch)
    elif req.key:
        raise UnknownGraph()
    else:
        raise Invalid(Msg("E-GRAPH-MISSING"))
    data = req.graph if req.graph is not None else untree(t)
    graph = admit(request, data)
    if req.key:
        size = len(json.dumps(t, separators=(",", ":")))
        if size > KEEP_SESSION_BYTES:  # a chain of patches makes a version of any size; a request never makes more
            raise TooLarge(Msg("E-GRAPH-TOOBIG", mb=KEEP_SESSION_BYTES >> 20))
        _keep(session, req.key, t, size)
    return data, graph


