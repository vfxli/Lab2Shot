"""Shared cache of Evaluation objects (engine/evaluation.py), one per (session, graph key, cache generation), so that
repeated queries about the same version of the same graph are evaluated once. Every read-only caller that has a graph
and a key for it (/api/status, /api/plan, the farm's internal bookkeeping) obtains its Evaluation from this cache
instead of building one.

The generation is a single server-wide counter. It is incremented whenever a cached packet that a plan relied on may
have changed: the engine commits a packet (engine/cook.py, Engine._run_node), the disk cleaner removes cache items or
uploads (farm/disk.py), or an install switches an extension's environment or finishes (server/installs.py: its nodes
may behave differently afterwards).
Incrementing loses nothing: stale Evaluations are no longer handed out, and the next request for their key builds a
fresh one.

A different kind of staleness, where a graph's external file changes identity (NodeDef.source_identity) without any
global change and therefore without a new generation, is detected per Evaluation by the inexpensive `still_true()`
check before the cache returns it."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

from .evaluation import Evaluation
from .graph import Graph

if TYPE_CHECKING:
    from ..serving import Account


def content_key(graph: Graph) -> str:
    """Return a cache key computed from the graph's content alone (each node's type, label, parameters and promoted
    parameters, the edges, the frame range). Graphs with identical content yield the same key, stable across processes
    and independent of how the graph was obtained. Callers that already have a key (the page's graph version,
    server/graphs.py) use that instead; this function serves scripts, DCC plugins and the farm's internal bookkeeping,
    which have only the parsed Graph."""
    nodes = {nid: {"type": n.type.id, "label": n.label, "params": n.params, "promoted": list(n.promoted)}
             for nid, n in graph.nodes.items()}
    edges = sorted((f"{dst}.{dport}", [f"{s}.{sp}" for s, sp in wires]) for (dst, dport), wires in graph.inputs.items() if wires)
    blob = {"frames": list(graph.frames) if graph.frames else None, "nodes": nodes, "edges": edges}
    from ..io.digest import key

    return key(blob, 64)


class EvaluationCache:
    """`size`: maximum number of Evaluations kept in total; `per_session`: maximum kept for one session (a browser
    tab, "farm" for the queue's internal bookkeeping, or a script/DCC identity). Eviction drops the session's oldest
    entries first, then the oldest overall once the total exceeds `size`."""

    def __init__(self, size: int = 64, per_session: int = 6) -> None:
        self.size = size
        self.per_session = per_session
        self.generation = 0
        self._lock = threading.Lock()
        self._by_key: OrderedDict[tuple[str, str, int], Evaluation] = OrderedDict()

    def bump(self) -> None:
        """Start a new generation because a cached packet that a plan relied on may have changed. Evaluations of the
        ending generation are no longer handed out, but remain stored until normal eviction, so a cook in progress
        that still reads its own Evaluation keeps working (see the Engine docstring)."""
        with self._lock:
            self.generation += 1

    def reset(self) -> None:
        """Clear the cache and start a new generation. Used with an isolated cache directory (e.g. a test harness's
        work folder, analogous to farm.queue.forget() resetting the queue); otherwise a graph whose content matches
        one from an earlier test, whose work folder no longer exists, would receive a stale result from this
        process-wide cache."""
        with self._lock:
            self.generation += 1
            self._by_key.clear()

    def get(self, session: str, graph_key: str, build: Callable[[], Graph], account: "Account | None" = None) -> Evaluation:
        """Return the Evaluation of `graph_key` for `session` and `account` at the current generation: the cached one
        if present and its external files (still_true) have not changed identity; otherwise a new one built from
        `build()` (called only on a cache miss). `account` (lab2shot/serving.py Account) is part of the key because the
        same graph resolves differently for different accounts: uploads belonging to another account are not visible,
        and the farm's session is shared by all accounts."""
        from ..serving import ANYONE

        account = account or ANYONE  # no account given: the process serves no account (a tool or a test)
        with self._lock:
            gen = self.generation
            key = (session, f"{graph_key}?{account.user_id}{'+' if account.all_accounts else ''}", gen)
            hit = self._by_key.get(key)
        if hit is not None and hit.still_true():
            with self._lock:
                if self._by_key.get(key) is hit:  # not evicted in the meantime
                    self._by_key.move_to_end(key)
            return hit
        ev = Evaluation(build(), account)
        with self._lock:
            if self.generation == gen:  # otherwise already stale when built: returned but not cached
                self._by_key[key] = ev
                self._evict(session)
        return ev

    def _evict(self, session: str) -> None:
        """Drop `session`'s oldest entries beyond `per_session`, then the overall oldest beyond `size`. The caller must
        hold the lock. `max(0, ...)` is required: with fewer than `per_session` entries a negative slice would drop
        from the end instead of doing nothing."""
        mine = [k for k in self._by_key if k[0] == session]
        cut = max(0, len(mine) - self.per_session)
        for k in mine[:cut]:
            del self._by_key[k]
        while len(self._by_key) > self.size:
            self._by_key.popitem(last=False)


EVALUATIONS = EvaluationCache()

# a removal from the cache starts a new generation: packet.remove is the one way a cache entry goes, and it tells
# this module, so no removal path (cook, disk.clean, quota, a reader node's invalidation, deleting tasks and footage)
# has to call bump itself
from ..data.packet import on_removed  # noqa: E402

on_removed(lambda _fp, _why: EVALUATIONS.bump())
