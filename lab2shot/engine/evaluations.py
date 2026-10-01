"""Shared cache of Evaluation objects (engine/evaluation.py), one per (session, graph key, cache generation), so that
repeated queries about the same version of the same graph are evaluated once. The read-only callers that answer a
session about a graph it may send again (/api/status, /api/plan) obtain their Evaluation from this cache instead of
building one. Two keep their own instead, for a reason: a queued job holds one for its whole life (farm/queue.py Job:
one job, one evaluation, for the account that submitted it; its graph never changes and nothing else asks about it),
and an Engine cooks on a fresh one (it mutates plans as it cooks: engine/cook.py Engine).

An Evaluation is stale when something it planned changed on disk: a packet committed or removed, a failure record
written or cleared, in its account's cache (records.changed: each Evaluation is listed under the fingerprints it
planned, so a cook of another graph, or of another account, leaves it standing). Its external files changing identity
(NodeDef.source_identity) and a packet it found cached going away are found by the inexpensive `still_true()` check
before the cache returns it. What changes every graph at once (an extension installed or its environment switched:
server/installs.py; uploads cleared: farm/disk.py) starts a new generation (`bump`): Evaluations of the ending one are
no longer handed out, but remain stored until normal eviction."""

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
    entries first, then the oldest overall once the total exceeds `size`, or once what they remember together
    (Evaluation.remembered: entries of every table, roughly 0.9 KB each: the default million is about 0.9 GB in all)
    exceeds `entries`. An evaluation goes on remembering after it was put in (a status read, a look), so the bound is
    checked on every hit as well as on every insertion; the Engines' and the jobs' own evaluations are not in it."""

    def __init__(self, size: int = 64, per_session: int = 6, entries: int = 1_000_000) -> None:
        self.size = size
        self.per_session = per_session
        self.entries = entries
        self.generation = 0
        self._lock = threading.Lock()
        self._by_key: OrderedDict[tuple[str, str, int], Evaluation] = OrderedDict()

    def bump(self) -> None:
        """Start a new generation: something every graph may depend on changed (an extension's environment, uploads
        cleared). Evaluations of the ending generation are no longer handed out, but remain stored until normal
        eviction."""
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
        # a fault of the program while checking it (a node type's source_identity) is said and taken as stale: it is
        # rebuilt, and the new one's answers say the fault at that node (Status.guarded), never a 500 on every look
        if hit is not None and not hit.stale and hit.guarded("still true", hit.still_true, False):
            with self._lock:
                if self._by_key.get(key) is hit:  # not evicted in the meantime
                    self._by_key.move_to_end(key)
                    self._evict(session)  # what it holds may have grown since it was put in
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
        held = sum(ev.remembered() for ev in self._by_key.values())
        while len(self._by_key) > 1 and held > self.entries:  # the oldest first; the newest always stays
            _key, ev = self._by_key.popitem(last=False)
            held -= ev.remembered()


EVALUATIONS = EvaluationCache()

# a packet committed or removed makes stale only the evaluations that planned it (records.changed, by the account
# whose cache it is in): packet.commit and packet.remove are the one way an entry comes and goes, and they tell this
# module, so no path (a cook, cleaning, quota, deleting tasks and footage) has to say it itself; every other graph's,
# and every other account's, evaluation stays in the cache
from ..data.packet import on_committed, on_removed  # noqa: E402
from ..serving import account as _account  # noqa: E402
from .records import changed as _changed  # noqa: E402

on_committed(lambda fp: _changed(_account().user_id, fp))
on_removed(lambda fp, _why: _changed(_account().user_id, fp))
