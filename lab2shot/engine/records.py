"""What an evaluation keeps about instances (engine/evaluation.py and its parts: presence, routing, demand, status):
the one table type they remember answers in (Memo, and its concurrency rule), a node instance's plan (NodePlan), why
an instance has no result (Outcome), where its last failure is kept (failure_file), and the errors that mean an
instance can't be planned (PLAN_ERRORS)."""

from __future__ import annotations

import threading
import weakref

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from ..data.packet import packet_dir, valid
from ..errors import CookError, GraphError
from .scopes import ItemPath

K = TypeVar("K")
V = TypeVar("V")

PLAN_ERRORS = (GraphError, CookError, OSError, ValueError)
Wire = tuple[str, str, ItemPath]  # into an instance: (from node, its port, the from instance's path)
Dropped = tuple[str, str, str, ItemPath]  # (input, from node, its port, its path)


class Memo(Generic[K, V]):
    """One table an Evaluation remembers answers in, and the one place its concurrency rule lives. One Evaluation is
    shared across threads (EvaluationCache: a status request, the queue placing a job, a policy walk), so:
    - each key is written once: `put` keeps the first value written and hands it back, so two threads that worked the
      same answer out at once both go on with the same object (computing it twice is wasteful, never wrong);
    - whoever walks a table walks a snapshot (`items`, `values`), never the live dict;
    - `drop` / `drop_where` are only for the Engine that owns the evaluation while it cooks (Evaluation.forget): an
      Engine cooks on an Evaluation of its own, never one from EvaluationCache (the farm builds a fresh one per cook,
      farm/queue.py; the cache only ever holds ones it built itself for reading).
    No lock: a reader's every operation is one atomic dict operation; `put` and the drops touch the index as well, which
    is safe because only the Engine that owns the evaluation drops (never two at once, under its own lock).

    `of`: what a key is about, for Evaluation.forget, which drops what a cook changed from every table the evaluation
    made (Evaluation._memo registers each: a table added is forgotten with the rest, never left out of a list):
    INSTANCE (the key is an Inst), a function from a key to the Inst it is about, WHOLE (what a cook takes as a whole:
    worked out again after any change that is not only an instance's outputs appearing) or KEPT (never stale: a
    packet's manifest, an instance's first source identity, which still_true compares with)."""

    __slots__ = ("_d", "of", "_about")

    def __init__(self, of: Any = None) -> None:
        self._d: dict[K, V] = {}
        self.of = of
        # a table whose keys are about an Inst through a function: Inst -> its keys, so forgetting an instance drops
        # its keys without walking the whole table (a block's first cook forgets thousands of instances)
        self._about: dict[Any, set[K]] | None = {} if callable(of) or of == INSTANCE else None

    def __contains__(self, key: K) -> bool:
        return key in self._d

    def __getitem__(self, key: K) -> V:
        return self._d[key]

    def __len__(self) -> int:
        return len(self._d)

    def put(self, key: K, value: V) -> V:
        """Remember `value` for `key` unless another thread already did; the value kept."""
        if self._about is not None:
            self._about.setdefault(key if self.of == INSTANCE else self.of(key), set()).add(key)
        return self._d.setdefault(key, value)

    def items(self) -> list[tuple[K, V]]:
        return list(self._d.items())

    def values(self) -> list[V]:
        return list(self._d.values())

    def drop(self, key: K) -> None:
        self._d.pop(key, None)

    def drop_where(self, gone: Callable[[K], bool]) -> None:
        for k in [k for k in list(self._d) if gone(k)]:
            self._d.pop(k, None)

    def drop_about(self, insts, where: Callable[[Any], bool] | None = None) -> None:
        """(A table whose `of` is a function) drop every key about one of `insts`, and, with `where`, about an Inst
        it holds true for (walking the Insts it has keys about, not every key)."""
        about = self._about if self._about is not None else {}
        for inst in [*insts, *([i for i in list(about) if where(i)] if where else [])]:
            for k in about.pop(inst, ()):
                self._d.pop(k, None)


INSTANCE, WHOLE, KEPT = "instance", "whole", None  # Memo.of

# which evaluations planned what: (account id, fingerprint) -> the evaluations that planned an instance with that
# fingerprint (its node's, for its failure record; each output's, for its packet). When one of them changes on disk
# (committed, removed, a failure written or cleared: `changed`), those evaluations and only those are stale: the cache
# of evaluations (engine/evaluations.py) hands them out no more. A cook of another graph, or another account's, leaves
# every other evaluation standing. An evaluation's entries go with it: when it is collected it is noted (`_dead`, a
# finalizer only appends: it may run inside this module's own lock, wherever the collector happens to run) and its
# entries are dropped under the lock by the next `planned` or `changed` (_bury), so what a thousand versions of a
# dragged parameter planned is not kept after them.
_planned_by: dict[tuple[int, str], dict[int, weakref.ref]] = {}  # key -> id(evaluation) -> it
_keys_of: dict[int, set[tuple[int, str]]] = {}  # id(evaluation) -> the keys it is under
_dead: list[int] = []
_planned_lock = threading.Lock()


def planned(ev, user_id: int, fingerprints) -> None:
    """`ev` planned an instance with these fingerprints (Evaluation._plan)."""
    with _planned_lock:
        _bury()
        if (me := id(ev)) not in _keys_of:
            _keys_of[me] = set()
            weakref.finalize(ev, _dead.append, me)
        for fp in fingerprints:
            _planned_by.setdefault((user_id, fp), {})[me] = weakref.ref(ev)
            _keys_of[me].add((user_id, fp))


def _bury() -> None:
    """Drop the entries of the evaluations collected since (a key no other evaluation is under goes with them). The
    caller holds the lock; an id is noted before its memory can be reused, and buried before it is registered again."""
    while _dead:
        me = _dead.pop()
        for key in _keys_of.pop(me, ()):
            if (under := _planned_by.get(key)) is not None:
                under.pop(me, None)
                if not under:
                    del _planned_by[key]


def changed(user_id: int, fingerprint: str) -> None:
    """What `fingerprint` names in `user_id`'s cache changed: every evaluation that planned it is stale."""
    with _planned_lock:
        _bury()
        found = _planned_by.pop((user_id, fingerprint), {})
        for me in found:
            _keys_of.get(me, set()).discard((user_id, fingerprint))
    for ref in found.values():
        if (ev := ref()) is not None:
            ev.stale = True


def _hash(obj: Any) -> str:
    from ..io.digest import key

    return key(obj, 24)


@dataclass
class NodePlan:
    fingerprint: str
    outputs: dict[str, str]  # port -> packet fingerprint

    def has(self, ports) -> bool:
        """Whether a plan's packets for these ports are on disk (valid, as presence.present says: complete, the outside
        files it reads unchanged, the packets it names there): what Demand.cached asks (what a cook still writes is
        Demand.missing, what still_true checks is each packet's generation). Asked for no port, or only for ports it has none of (a wire from an output that is gone), and
        for a node without outputs (「输出」, it always cooks): False, never 「已算」 by all([]); a caller for whom
        nothing asked means something else says so itself (satisfied: nothing wanted is nothing missing). Only an
        answer: a packet found invalid is left where it is (a read never writes the cache; the cook that needs it writes
        it again under its lock, fresh_dir, and cleaning removes what nothing uses, farm/disk.py)."""
        if not any(port in self.outputs for port in ports):
            return False
        return bool(self.outputs) and all(valid(packet_dir(o)) for port, o in self.outputs.items() if port in ports)


def failure_file(fingerprint: str):
    """Where the error an instance last failed with at this fingerprint is kept (engine/cook.py writes it): a
    computation's result like its packets (a refresh, another tab, a later status all still see it) until a cook
    tries it again or another cook of it commits its result (Engine._run_node removes it then); read only while the
    result is not there (Evaluation.failure): the packets are the fact, the record says why they are missing. The fingerprint is the instance's own (its inputs are its item's), so this is per instance."""
    return packet_dir(fingerprint + "_failed") / "failed.json"


@dataclass(frozen=True)
class Outcome:
    """An instance that has no result because of an error: it
    "failed" (its own error, `message` with its log), is in "error" (it can't be planned: its planning error, root
    itself; engine/scopes.py ERROR) or is "skipped" (a required input comes from an instance that failed, is in error
    or was skipped, or what it waits for does: `message` says which, `root` is the node that failed).
    `failure`: False for what is only outside the frame range (N-EACH-OUTSIDE) and what is skipped behind it: no
    error anywhere. `chain`: a node that delivers failed because a line into it is broken (_chains_broken): it is
    never tried, it fails as it stands. `retryable`: all that stands in its way is the record a past cook left
    (failure_file; for what is skipped or a broken line, every failure at its root is): the next cook tries it again
    (Engine._begin clears the record), so it never refuses a cook (Demand.readiness); a planning error, a missing
    file or an extension that can't be used stays whatever is cooked.
    `blocked`: skipped because a 「阻断」 (gate) on the way is set to block (its root is that gate): no error
    anywhere (failure False), and quiet — what goes without it (an optional input, one wire of a multi input) says
    nothing (Evaluation.unused_inputs), the page shows 「已跳过（被阻断）」."""

    state: str
    root: str
    message: dict
    failure: bool = True
    chain: bool = False
    retryable: bool = False
    blocked: bool = False
