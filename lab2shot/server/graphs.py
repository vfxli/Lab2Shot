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

from fastapi import Request
from pydantic import BaseModel

from ..errors import Invalid, MessageError
from ..messages import Msg
from . import auth

KEEP_PER_SESSION = 6  # versions kept per browser session
KEEP_BYTES = 64 << 20  # total JSON size of all kept versions across sessions
_lock = threading.Lock()
_kept: OrderedDict[tuple[str, str], tuple[dict, int]] = OrderedDict()  # (session, key) -> (graph as a tree, size)
_size = 0


class GraphRequest(BaseModel):
    """The graph of a request: the whole `graph` (scripts, plugins, first request of a version), a `key` naming a
    version the server holds, or `base` + `patch` producing version `key` from a held version."""

    graph: dict | None = None
    key: str = ""
    base: str = ""
    patch: list | None = None


def _session(request: Request | None) -> str:
    """Return the session that owns the versions; keys of other browser sessions are never consulted."""
    return auth.client_key(request) if request is not None else ""


def tree(graph: dict) -> dict:
    nodes = graph.get("nodes") or []
    return {**graph, "nodes": {str(n.get("id")): n for n in nodes}, "node_order": [str(n.get("id")) for n in nodes]}


def untree(t: dict) -> dict:
    order = t.get("node_order") or []
    nodes = t.get("nodes") or {}
    return {**{k: v for k, v in t.items() if k != "node_order"}, "nodes": [nodes[i] for i in order if i in nodes]}


def apply(t: dict, ops: list) -> dict:
    """Return a copy of `t` with the patch operations applied."""
    out = copy.deepcopy(t)
    for op in ops:
        if not isinstance(op, dict) or not isinstance(op.get("p"), list) or not all(isinstance(k, str) for k in op["p"]):
            raise Invalid(Msg("E-GRAPH-BADPATCH"))
        path = op["p"]
        if not path:
            if "v" not in op or not isinstance(op["v"], dict):
                raise Invalid(Msg("E-GRAPH-BADPATCH"))
            out = copy.deepcopy(op["v"])
            continue
        at = out
        for k in path[:-1]:
            if not isinstance(at.get(k), dict):
                at[k] = {}
            at = at[k]
        if "v" in op:
            at[path[-1]] = copy.deepcopy(op["v"])
        else:
            at.pop(path[-1], None)
    return out


def _keep(session: str, key: str, t: dict) -> None:
    global _size
    size = len(json.dumps(t, separators=(",", ":")))
    with _lock:
        if (session, key) in _kept:
            _size -= _kept.pop((session, key))[1]
        _kept[(session, key)] = (t, size)
        _size += size
        mine = [k for k in _kept if k[0] == session]
        for k in mine[:-KEEP_PER_SESSION]:
            _size -= _kept.pop(k)[1]
        while _size > KEEP_BYTES and len(_kept) > 1:
            _size -= _kept.popitem(last=False)[1]


def _get(session: str, key: str) -> dict | None:
    with _lock:
        got = _kept.get((session, key))
        if got:
            _kept.move_to_end((session, key))
        return got[0] if got else None


class UnknownGraph(MessageError):
    """The request names a version the server does not hold; answered with 409 {"graph": "unknown"} (server/app.py)."""

    def __init__(self) -> None:
        super().__init__(Msg("E-GRAPH-UNKNOWNVERSION"))


def resolve(req: GraphRequest, request: Request | None) -> dict:
    """Return the graph a request names, as JSON, and keep it under its key for subsequent requests."""
    session = _session(request)
    if req.key and (t := _get(session, req.key)) is not None:
        return untree(t)
    if req.graph is not None:
        if req.key:
            _keep(session, req.key, tree(req.graph))
        return req.graph
    if req.key and req.base and req.patch is not None:
        base = _get(session, req.base)
        if base is None:
            raise UnknownGraph()
        t = apply(base, req.patch)
        _keep(session, req.key, t)
        return untree(t)
    if req.key:
        raise UnknownGraph()
    raise Invalid(Msg("E-GRAPH-MISSING"))


