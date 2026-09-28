"""Everything a graph's state can tell without cooking anything: fingerprints, cached outputs, effective parameters,
wired values, sources, usage warnings, the status the editor polls. One `Evaluation` is one version of one graph
(engine/evaluations.py keeps one per graph version): every method here remembers its answer per node instance (a
node and its item path, engine/scopes.py Inst; a node outside every 逐项处理 block has the one instance with the empty
path, which is what every `path=()` below means), so asking the same question twice (the parameter panel, the footer,
the status reply, a second /api/status call with nothing changed) costs nothing the second time. `Engine`
(engine/cook.py) holds one to plan and answer status from, and cooks on top of it; a plan `Evaluation.plan()` makes is
the same object `Engine` mutates in place as instances actually cook (`.cached` flips true once its outputs are
written), so a running cook's `Engine` never shares its `Evaluation` with another one from the cache (see
EvaluationCache.get in evaluations.py).

What can only be known once something above is cooked (a wired parameter's value, a block's items, the input a switch
chooses) is pending (engine/scopes.py Pending), found by the one function `_known_or_pending`; an instance that waits
is planned with what is known, never remembered, and planned again once it is (`waits`, `order`).

Nothing here writes to disk: a cached packet's "last used" time is set only where a packet is actually read for
cooking (Engine._run_node) or shown (server/packets.py's GETs), never by looking at what is cached (see
Packet.used, data/packet.py)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Generic, TypeVar

from typing import TYPE_CHECKING

from ..data.contracts import KEEPS, shot_of
from ..data.packet import CACHE_VERSION, FACTS_VERSION, Packet, discard_invalid, packet_dir, valid
from ..errors import CookError, GraphError, Invalid, NotFound, Refused, message_of
from ..messages import Msg
from ..nodes.applies import NodeFacts, Resolved, effective_params, resolve, standing_notices
from ..nodes.base import Info
from ..nodes.port import PARAM
from ..nodes.formats import ImportNode
from ..nodes.output import OutputSettings
from . import scopes as sc
from .graph import Graph
from .lint import warnings
from .scopes import BEGIN, END, Inst, ItemAt, ItemPath, Pending

if TYPE_CHECKING:
    from ..serving import Account

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
    - `drop` / `drop_where` are only for the Engine that owns the evaluation while it cooks (Evaluation.forget), which
      shares it with nobody (EvaluationCache.get hands a running cook's evaluation to no one else).
    No lock: every operation here is one atomic dict operation."""

    __slots__ = ("_d",)

    def __init__(self) -> None:
        self._d: dict[K, V] = {}

    def __contains__(self, key: K) -> bool:
        return key in self._d

    def __getitem__(self, key: K) -> V:
        return self._d[key]

    def __len__(self) -> int:
        return len(self._d)

    def put(self, key: K, value: V) -> V:
        """Remember `value` for `key` unless another thread already did; the value kept."""
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


def _hash(obj: Any) -> str:
    from ..io.digest import key

    return key(obj, 24)


@dataclass
class NodePlan:
    fingerprint: str
    outputs: dict[str, str]  # port -> packet fingerprint
    cached: bool  # every output something needs (Evaluation.needed_outputs) is there

    def has(self, ports) -> bool:
        """The single formula for "the packets of these ports are on disk": `cached` (by who wires it in the whole graph,
        `needed_outputs`), `Evaluation.satisfied` (by the ports this cook wants of it, `demand`), `still_true` and the
        cook's `_settled` all call it, with no duplicate. A node without outputs (「输出」) is always False: it always cooks.
        On disk means valid (data/packet.py check): complete, the external files it reads present and unchanged, the
        packets it references present. Invalid packets found are removed at once (discard_invalid -> remove); when
        downstream needs them again they are uploaded and cooked again."""
        return bool(self.outputs) and all(discard_invalid(packet_dir(o)).ok for port, o in self.outputs.items() if port in ports)


def failure_file(fingerprint: str):
    """Where the error an instance last failed with at this fingerprint is kept (engine/cook.py writes it): a
    computation's result like its packets (a refresh, another tab, a later status all still see it) until a cook
    tries it again. The fingerprint is the instance's own (its inputs are its item's), so this is per instance."""
    return packet_dir(fingerprint + "_failed") / "failed.json"


@dataclass(frozen=True)
class Outcome:
    """An instance that has no result because of an error: it
    "failed" (its own error, `message` with its log) or is "skipped" (a required input comes from an instance that
    failed or was skipped, or what it waits for does: `message` says which, `root` is the node that failed).
    `failure`: False for what is only outside the frame range (N-EACH-OUTSIDE) and what is skipped behind it: no
    error anywhere. `chain`: a node that delivers failed because a line into it is broken (_chains_broken): it is
    never tried, it fails as it stands."""

    state: str
    root: str
    message: dict
    failure: bool = True
    chain: bool = False


class Evaluation:
    """One version of one graph, memoised per node instance."""

    def __init__(self, graph: Graph, account: "Account | None" = None) -> None:
        from ..serving import ANYONE

        self.graph = graph
        # whose the graph's files are read as (lab2shot/serving.py Account): an upload that is not this account's is
        # not there: the node fails at itself with E-UPLOAD-GONE (source_missing below, exactly as for one that was
        # cleaned away), what needs it is skipped, a multi input goes on without it, and the file is never opened,
        # not even to identify it. Kept here so this evaluation answers the same wherever it is read from (the farm's
        # own thread reads a job's), and part of its cache key (engine/evaluations.py).
        self.account = account or ANYONE
        # every table it remembers answers in is a Memo (its concurrency rule)
        self._plans: Memo[Inst, NodePlan] = Memo()
        self._infos: Memo[Inst, Info] = Memo()  # before the frame range: a frame source's every frame
        # why an instance could not be planned (a refused file, a precondition): kept so status, checks and the plan
        # of what comes after read its file once, not once each
        self._plan_errors: Memo[Inst, Exception] = Memo()
        self._manifests: Memo[str, dict | None] = Memo()  # packet fingerprint -> its manifest (type, meta, node), once
        self._source_identities: Memo[Inst, Any] = Memo()  # its source_identity() the first time plan() saw it
        self._gone: Memo[Inst, Msg] = Memo()  # its source file is not there (cleaned, or not this account's): why
        self._source_errors: Memo[Inst, Exception] = Memo()  # its source file could not be read: why (asked once)
        self._outcomes: Memo[Inst, Outcome | None] = Memo()
        self._dropped: Memo[Inst, dict[Dropped, Outcome]] = Memo()
        self._resolved: Memo[Inst, Resolved] = Memo()
        self._checks: Memo[Inst, list[dict]] = Memo()
        self._shots: Memo[tuple, dict] = Memo()  # (node, output, path) -> what was photographed, before any cook
        self._provenance: Memo[Inst, dict] = Memo()
        self._wires: Memo[Inst, dict[str, list[Wire]]] = Memo()  # input -> the wires the instance takes
        self._items: Memo[Inst, list[ItemAt]] = Memo()  # a begin and the path its scope sits at -> the items
        self._orders: Memo[tuple[str, ...], tuple[list[Inst], frozenset[str]]] = Memo()
        # the ports this cook wants of each instance (`demand`) and whether they are all on disk (`satisfied`):
        # both tables are invalidated together with order (forget)
        self._demands: Memo[tuple[tuple[str, ...], frozenset[str]], dict[Inst, frozenset[str]]] = Memo()
        self._satisfied: Memo[tuple[Inst, frozenset[str]], bool] = Memo()

    # ------------------------------------------------------------------ manifests (read at most once per fingerprint)

    def manifest(self, fp: str) -> dict | None:
        """A packet's manifest (type, meta, node), read from disk at most once per fingerprint in this evaluation's
        lifetime (packets are immutable once committed, so this never goes stale under it). None: not committed; this is
        never remembered, since a cook of this evaluation's own graph may commit it in a moment (a packet that is
        there, on the other hand, stays what it is)."""
        if fp in self._manifests:
            return self._manifests[fp]
        d = packet_dir(fp)
        if not Packet.exists(d):
            return None
        return self._manifests.put(fp, {"type": (p := Packet.load(d)).type, "meta": p.meta, "node": p.node, "messages": p.messages,
                                        "created": p.created})

    def packet(self, fp: str) -> Packet | None:
        """The packet a fingerprint names, from `manifest` (no second read of manifest.json)."""
        m = self.manifest(fp)
        return None if m is None else Packet(packet_dir(fp), m["type"], m["meta"], m["node"], m["messages"])

    # ------------------------------------------------------------------ blocks: instances, items, wires

    def instances(self, node_id: str) -> tuple[list[ItemPath], list[Pending]]:
        """The node's instances known now (their paths; one empty path outside every block), and the item lists of its
        blocks still pending."""
        paths: list[ItemPath] = [()]
        pending: list[Pending] = []
        for begin in self.graph.scopes.chain(node_id):
            deeper = []
            for p in paths:
                items = self.item_list(begin, p[:self.graph.scopes.depth(begin) - 1])
                if isinstance(items, Pending):
                    pending.append(items)
                else:
                    deeper += [p + (i.key,) for i in items]
            paths = deeper
        return paths, pending

    def item_list(self, begin: str, path: ItemPath = ()) -> list[ItemAt] | Pending:
        """The items of the block `begin` starts, at the path of the blocks around it: from the packet wired into its
        item input, once that is known."""
        key = Inst(begin, path)
        if key in self._items:
            return self._items[key]
        g, node = self.graph, self.graph.nodes[begin]
        wires = g.inputs.get((begin, node.type.item_input), [])
        if not wires:
            return self._items.put(key, [])
        src, sport = wires[0]
        got = self._known_or_pending("items", Inst(src, path[:g.scopes.depth(src)]), sport)
        if isinstance(got, Pending):
            return got
        found = [] if got.meta.get("empty") else node.type.scope_items(node.params, got)
        return self._items.put(key, [ItemAt(i.name, i.key, i.packet, n, len(found)) for n, i in enumerate(found)])

    def item_at(self, begin: str, path: ItemPath) -> ItemAt:
        """The item of `begin`'s block an instance at `path` stands for (its last key at that block's depth). A path
        that does not reach that depth names no instance of it at all (something asked about the node itself, not
        about one of its items): that is an error of the asking, said as one, never an IndexError."""
        depth = self.graph.scopes.depth(begin)
        items = self.item_list(begin, path[:depth - 1])
        found = (None if isinstance(items, Pending) or len(path) < depth
                 else next((i for i in items if i.key == path[depth - 1]), None))
        if found is None:
            raise GraphError(Msg("E-GRAPH-NONODE", node=f"{begin}@{'/'.join(path)}"))
        return found

    def _known_or_pending(self, kind: str, on: Inst, port: str) -> Packet | Pending:
        """THE one place something an instance needs before it can be planned is found out not to be known yet
        (engine/scopes.py「Pending」): a wired parameter's value ("value"), a block's items ("items"), the input a
        switch chooses ("condition"). What `on`'s `port` gives when known (Evaluation.known), else what waits for it."""
        packet = self.known(on.node, port, on.path)
        return Pending(kind, on, port) if packet is None else packet

    def _chosen(self, node_id: str, path: ItemPath) -> frozenset[str] | Pending | None:
        """The inputs a switch needs with its condition (None: the node is not a switch)."""
        g, node = self.graph, self.graph.nodes[node_id]
        if not sc.chooses(node.type):
            return None
        cond = node.type.condition_input
        wires = g.inputs.get((node_id, cond), [])
        if not wires:  # no condition wired: its check says so (Graph.check_inputs)
            return frozenset({cond})
        src, sport = wires[0]
        got = self._known_or_pending("condition", Inst(src, path[:g.scopes.depth(src)]), sport)
        if isinstance(got, Pending):
            return got
        return frozenset(node.type.chosen_inputs(node.params, got)) | {cond}

    def wires(self, node_id: str, path: ItemPath = ()) -> dict[str, list[Wire]]:
        """Every input of the instance (a promoted parameter's too) -> the wires it takes, each with the path of the
        instance it comes from: the same path cut to that node's depth; into a block's end, one per item of the
        block (and none while they are pending); into a switch, only the inputs its condition chooses (only the
        condition while that is pending). A wire the rules of a block refuse is left out (the node's error says so)."""
        key = Inst(node_id, path)
        if key in self._wires:
            return self._wires[key]
        g = self.graph
        scopes = g.scopes
        chosen = self._chosen(node_id, path)
        begin_of_end = scopes.ended.get(node_id)
        items = self.item_list(begin_of_end, path) if begin_of_end is not None else None
        out: dict[str, list[Wire]] = {}
        inner = scopes.chain(node_id) + (begin_of_end,) if begin_of_end is not None else None
        for port in g.input_ports(node_id):
            taken: list[Wire] = []
            if not isinstance(chosen, Pending) or port.name == g.nodes[node_id].type.condition_input:
                if chosen is None or isinstance(chosen, Pending) or port.name in chosen:
                    for src, sport in g.inputs.get((node_id, port.name), []):
                        depth = scopes.depth(src)
                        if inner is not None and not port.param and scopes.chain(src) == inner:
                            taken += [(src, sport, path + (i.key,)) for i in items] if not isinstance(items, Pending) else []
                        elif depth <= len(path) and scopes.chain(src) == scopes.chain(node_id)[:depth]:
                            taken.append((src, sport, path[:depth]))
            out[port.name] = taken
        if isinstance(chosen, Pending) or isinstance(items, Pending):
            return out
        return self._wires.put(key, out)

    def waits(self, node_id: str, path: ItemPath = ()) -> tuple[Pending, ...]:
        """What the instance waits for before it can be planned for good (engine/scopes.py Pending)."""
        node = self.graph.nodes[node_id]
        out: list[Pending] = []
        inactive = self.resolved(node_id, path).params.inactive
        for name in node.promoted:
            if name not in inactive and isinstance(w := self._wired_packet(node_id, name, path), Pending):
                out.append(w)
        if isinstance(chosen := self._chosen(node_id, path), Pending):
            out.append(chosen)
        if (begin := self.graph.scopes.ended.get(node_id)) is not None and isinstance(items := self.item_list(begin, path), Pending):
            out.append(items)
        return tuple(out)

    def deps(self, inst: Inst) -> list[Inst]:
        """The instances `inst` needs cooked first: those wired into it (as `wires` takes them) and what it waits for."""
        found = [Inst(src, p) for wires in self.wires(*inst).values() for src, _, p in wires]
        found += [w.on for w in self.waits(*inst)]
        return list(dict.fromkeys(found))

    def order(self, targets: list[str]) -> tuple[list[Inst], frozenset[str]]:
        """Cooking `targets` (every instance of each): the instances it needs, dependencies first, as far as is known
        now; and the nodes behind what is still pending (a block's items, a switch's condition) that it will need once
        that is known. Worked out again after every cook that changes what is known (forget)."""
        key = tuple(targets)
        if key in self._orders:
            return self._orders[key]
        g = self.graph
        order: list[Inst] = []
        seen: set[Inst] = set()
        visiting: set[Inst] = set()
        behind: set[str] = set()

        def visit(inst: Inst) -> None:
            if inst in seen:
                return
            if inst in visiting:
                raise GraphError(Msg("B-GRAPH-CYCLE"))
            visiting.add(inst)
            for w in self.waits(*inst):
                if w.kind != "value":  # a value's node is wired in: planned anyway; the rest is behind this
                    ports = [p for p in g.input_ports(inst.node) if w.kind == "items" or p.name != g.nodes[inst.node].type.condition_input]
                    behind.update(n for p in ports for src, _ in g.inputs.get((inst.node, p.name), []) for n in g.upstream_order(src))
            for d in self.deps(inst):
                visit(d)
            visiting.discard(inst)
            seen.add(inst)
            order.append(inst)

        for t in targets:
            if t not in g.nodes:
                raise GraphError(Msg("E-GRAPH-NONODE", node=t))
            paths, pending = self.instances(t)
            for p in pending:
                visit(p.on)
                behind.update(g.upstream_order(t))
            for p in paths:
                visit(Inst(t, p))
        return self._orders.put(key, (order, frozenset(behind - {i.node for i in order})))

    def needed_outputs(self, node_id: str) -> frozenset[str]:
        """The outputs of the node something needs: those a wire takes; every one when no wire takes any (what is
        cooked to be shown or delivered, a sink). What `cached` means (NodePlan)."""
        taken = {sport for dst, dport in self.graph.outputs_by_node.get(node_id, ())
                 for src, sport in self.graph.inputs.get((dst, dport), []) if src == node_id}
        return frozenset(taken) if taken else frozenset(p.name for p in self.graph.outputs(node_id))

    # ------------------------------------------------------------------ what this cook requires (Demand)

    def demand(self, targets: list[str], shown: frozenset[str] = frozenset()) -> dict[Inst, frozenset[str]]:
        """For cooking `targets`, the ports each instance must give. This is the project's single answer; the cook
        (Engine._begin, _compute), `computes`, `_case` and the queue's cache marks all ask it. The two sides must not
        use different criteria: if the cook wrote by the current demand while the status judged by the whole graph's
        `needed_outputs`, then with node A's port x wired to B and port y to C, cooking only B writes only x, A would
        count as never cached by the whole graph, and every later cook would compute A again only to find nothing to
        write.

        An instance's wanted ports = those taken from it by instances in the current order (wires and waits) + those
        the target itself waits on in blocks not yet expanded + `shown` (the ports shown for the target) + the sources
        of ports made from another port of the same node (`Port.made_from`).
        An instance absent from the table: none of its ports are wanted now (`frozenset()`)."""
        key = (tuple(targets), shown)
        if key in self._demands:
            return self._demands[key]
        want: dict[Inst, set[str]] = {}
        add = lambda inst, port: want.setdefault(inst, set()).add(port)  # noqa: E731
        order, _ = self.order(list(targets))
        # the target's own ports: those shown (`shown`), otherwise all of them (「计算」 wants the whole node);
        # without this the target itself would always count as satisfied and never cook
        own = {t: (shown or frozenset(p.name for p in self.graph.outputs(t))) for t in targets}
        for i in order:
            for wires in self.wires(*i).values():
                for src, sport, p in wires:
                    add(Inst(src, p), sport)
            for w in self.waits(*i):  # what it is pending on (a block's items, a switch's condition): wanted too
                add(w.on, w.port)
            if i.node in own:
                want.setdefault(i, set()).update(own[i.node])
        for t in targets:  # the target itself inside a block whose items are not known yet: the list it waits on
            for p in self.instances(t)[1]:
                add(p.on, p.port)
        out: dict[Inst, frozenset[str]] = {}
        for inst, ports in want.items():
            outs = {p.name: p for p in self.graph.outputs(inst.node)}
            for name in list(ports):  # a port made from another of the node's own (point cloud = depth map + camera): its sources too
                ports.update(n for n in getattr(outs.get(name), "made_from", ()) if n in outs)
            out[inst] = frozenset(ports)
        return self._demands.put(key, out)

    def satisfied(self, inst: Inst, wanted: frozenset[str]) -> bool:
        """Whether the ports this instance must give (per `demand`) are all on disk. A begin's item port is never written
        (it is the list's packet) and is not counted. An instance that cannot be planned (a file is gone): False; the
        cook will fail there and say why."""
        key = (inst, wanted)
        if key in self._satisfied:
            return self._satisfied[key]
        try:
            plan = self.plan(*inst)
        except PLAN_ERRORS:
            return self._satisfied.put(key, False)
        t = self.graph.nodes[inst.node].type
        item = t.item_output if sc.role(t) == sc.BEGIN else None
        return self._satisfied.put(key, plan.has(frozenset(p for p in wanted if p != item)))

    # ------------------------------------------------------------------ errors and what they leave out

    def failure(self, node_id: str, path: ItemPath = ()) -> dict | None:
        """The error the instance last failed with at its current fingerprint (its message, and its log), None when it
        did not (or can't be planned: that is said as its status error). Before any cook, a node without a result whose
        extension can't be used now (ProjectFacts.available: not installed, built from older code, weights missing) has
        failed already (E-COOK-UNAVAILABLE), and so has one whose source file is not there (cleaned away, or not this
        account's, which is the same answer: source_missing, E-UPLOAD-GONE); what needs it is skipped, a multi input
        goes on without it."""
        if (gone := self.source_missing(node_id, path)) is not None:
            return gone
        try:
            plan = self.plan(node_id, path)
        except PLAN_ERRORS:
            return None
        node = self.graph.nodes[node_id]
        project = node.type.project
        if not plan.cached and project.extension is not None and (why := project.available()) is not None:
            return Msg("E-COOK-UNAVAILABLE", node=node.label, reason=why).json()  # the one answer the session's node:<type> reads too
        path_ = failure_file(plan.fingerprint)
        if not path_.exists():
            return None
        try:
            return json.loads(path_.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _gathers(self, node_id: str, port) -> bool:
        """An input that goes on while one of its wires works: a multi input, and every input of a block's end (one
        wire per item)."""
        return port.multi or sc.role(self.graph.nodes[node_id].type) == END

    def outcome(self, node_id: str, path: ItemPath = ()) -> Outcome | None:
        """Why the instance has no result because of an error, None when nothing stands in its way: a required input
        from an instance that failed or was skipped (checked first: the error to fix is up there; for a multi input
        and a block's end, every wire into it), what it waits for failed or was skipped, or its own failure."""
        key = Inst(node_id, path)
        if key in self._outcomes:
            return self._outcomes[key]
        g, node = self.graph, self.graph.nodes[node_id]
        found = self._outside_frames(node_id, path)
        wires = self.wires(node_id, path)
        for port in g.input_ports(node_id):
            if port.optional:
                continue
            ups = [o for src, _, p in wires[port.name] if (o := self.outcome(src, p)) is not None]
            if ups and (not self._gathers(node_id, port) or len(ups) == len(wires[port.name])):
                found = self._skipped(node, port.label, ups[0])
                break
        # alternative wirings (NodeDef.input_choice): when every used wire is gone because upstream failed or was
        # skipped, this node is skipped too. Otherwise it would cook and report "nothing wired" itself, saying the same
        # thing twice and hiding the real error (the real error is reported once at the failing node; downstream is
        # marked skipped)
        if found is None and node.type.input_choice:
            live = [p for p in g.input_ports(node_id) if p.name in node.type.choice_inputs() and wires[p.name]]
            gone = [(p.label, o) for p in live for src, _, sub in wires[p.name] if (o := self.outcome(src, sub)) is not None]
            if live and len(gone) == len(live):
                found = self._skipped(node, gone[0][0], gone[0][1])
        if found is None:
            for w in self.waits(node_id, path):
                if (up := self.outcome(*w.on)) is not None:
                    port = g.input_port(node_id, node.type.item_input if w.kind == "items" and sc.role(node.type) == BEGIN else w.port)
                    found = self._skipped(node, port.label if port else g.nodes[w.on.node].label, up)
                    break
        if node.type.delivers and (broken := self._chains_broken(node_id, path)) is not None:
            found = broken  # 「输出」 packs only whole lines: what stands above it decides before anything else
        if found is None and (failed := self.failure(node_id, path)) is not None:
            found = Outcome("failed", node_id, failed)
        return self._outcomes.put(key, found)

    def _chains_broken(self, node_id: str, path: ItemPath) -> Outcome | None:
        """A node that delivers (「输出」) and a line into it that is not whole: somewhere above one of the
        output-settings nodes wired into it a node failed, was skipped behind a failure, or a block's end gathered
        without an item that failed (an output packs only when every line into it is complete, so what it hands over
        is never partial without saying so; the user cuts the broken line or fixes it). Fails it, naming each broken line and the node
        at its root (and the item, inside a block). None: every line is whole (or not known to be broken yet)."""
        g = self.graph
        seen: set[Inst] = set()
        said = []
        try:
            wires = self.wires(node_id, path)
        except PLAN_ERRORS:
            return None
        for port in g.input_ports(node_id):
            for src, _sport, p in wires[port.name]:
                root = self._failure_above(src, p, seen)
                if root is None:
                    continue
                at, names = root
                chain = g.nodes[src].label
                said.append(Msg("I-OUTPUT-CHAINITEM", chain=chain, root=g.nodes[at].label, item=" / ".join(names))
                            if names else Msg("I-OUTPUT-CHAIN", chain=chain, root=g.nodes[at].label))
        if not said:
            return None
        msg = Msg("E-OUTPUT-CHAINFAILED", node=g.nodes[node_id].label, count=len(said), chains=said)
        return Outcome("failed", node_id, msg.json(), chain=True)

    def _failure_above(self, node_id: str, path: ItemPath, seen: set[Inst]) -> tuple[str, list[str]] | None:
        """The first failure at or above the instance (the node at its root and the names of its items), None when
        there is none: its own outcome, a wire it goes without (dropped: an optional input, a gathered item), then
        every instance it takes something from or waits for."""
        inst = Inst(node_id, path)
        if inst in seen:
            return None
        seen.add(inst)
        try:
            o = self.outcome(node_id, path)
            if o is not None and o.failure:
                return o.root, self._root_names(o.root, path)
            for (_port, _src, _sport, p), up in self.dropped(node_id, path).items():
                if up.failure:
                    return up.root, self._root_names(up.root, p)
            ups = [(src, p) for ws in self.wires(node_id, path).values() for src, _, p in ws]
            ups += [(w.on.node, w.on.path) for w in self.waits(node_id, path)]
        except PLAN_ERRORS:
            return None
        for src, p in ups:
            if (found := self._failure_above(src, p, seen)) is not None:
                return found
        return None

    def _root_names(self, root: str, path: ItemPath) -> list[str]:
        """The item names of the failed node's instance, from a path at or below it (a node outside every block: none)."""
        depth = len(self.graph.scopes.chain(root))
        if not depth or len(path) < depth:
            return []
        try:
            return [n for n in self._names(root, path[:depth]) if n]
        except PLAN_ERRORS:
            return []

    def _outside_frames(self, node_id: str, path: ItemPath) -> Outcome | None:
        """An item whose own frames are all outside the graph's frame range is skipped, not an error: the frame
        range narrows every item on its own, and the remaining items cook as usual."""
        if not path:
            return None
        try:
            self.info(node_id, path)
        except CookError as exc:
            if exc.message.code == "E-FRAMES-OUTSIDE" and self.graph.frames:
                lo, hi = self.graph.frames
                return Outcome("skipped", node_id, Msg("N-EACH-OUTSIDE", node=self.graph.nodes[node_id].label,
                                                       item=self._names(node_id, path)[-1], lo=lo, hi=hi).json(), failure=False)
        except PLAN_ERRORS:
            return None
        return None

    def _skipped(self, node, input_label: str, up: Outcome) -> Outcome:
        return Outcome("skipped", up.root, Msg("N-COOK-SKIPPED", node=node.label, input=input_label,
                                               root=self.graph.nodes[up.root].label).json(), failure=up.failure)

    def dropped(self, node_id: str, path: ItemPath = ()) -> dict[Dropped, Outcome]:
        """The wires whose source failed or was skipped into the instance's optional inputs (a parameter's input too)
        and into a multi input (or a block's end) that still has a wire that works: (input, source node, its output,
        its path) -> that outcome. The instance is cooked without them (its fingerprint says so)."""
        key = Inst(node_id, path)
        if key not in self._dropped:
            out: dict[Dropped, Outcome] = {}
            wires = self.wires(node_id, path)
            for port in self.graph.input_ports(node_id):
                gone = {(port.name, src, sport, p): o for src, sport, p in wires[port.name] if (o := self.outcome(src, p)) is not None}
                if port.optional or (self._gathers(node_id, port) and len(gone) < len(wires[port.name])):
                    out.update(gone)
            return self._dropped.put(key, out)
        return self._dropped[key]

    def gathered(self, node_id: str, path: ItemPath = ()) -> list[ItemAt]:
        """The items a block's end gathers at `path`: those whose results on every input it takes are there (an item
        with one failed or skipped is left out whole)."""
        begin = self.graph.scopes.ended.get(node_id)
        items = self.item_list(begin, path) if begin is not None else []
        if isinstance(items, Pending):
            return []
        out = {k[3][len(path)] for k in self.dropped(node_id, path) if len(k[3]) > len(path)}
        return [i for i in items if i.key not in out]

    def taken(self, node_id: str, path: ItemPath = ()) -> dict[str, list[Wire]]:
        """`wires` without the dropped ones and, into a block's end, without the items left out (gathered): what the
        instance is planned and cooked with."""
        dropped = self.dropped(node_id, path)
        keep = None
        if self.graph.scopes.ended.get(node_id) is not None:
            keep = {i.key for i in self.gathered(node_id, path)}
        return {port: [(src, sport, p) for src, sport, p in wires if (port, src, sport, p) not in dropped
                       and (keep is None or len(p) <= len(path) or p[len(path)] in keep)]
                for port, wires in self.wires(node_id, path).items()}

    def unused_inputs(self, node_id: str, path: ItemPath = ()) -> list[tuple[Msg, str]]:
        """What the instance says about the inputs it goes without (dropped): W-INPUT-UNUSED and the input it is about;
        a block's end says the items it gathers without once (W-EACH-FAILED)."""
        g, node = self.graph, self.graph.nodes[node_id]
        out: list[tuple[Msg, str]] = []
        missing: dict[str, Outcome] = {}
        begin = g.scopes.ended.get(node_id)
        for (port, src, _, p), up in self.dropped(node_id, path).items():
            if begin is not None and len(p) > len(path):
                missing.setdefault(p[len(path)], up)
                continue
            reason = (Msg("I-INPUT-FAILED", source=g.nodes[src].label) if up.state == "failed"
                      else Msg("I-INPUT-SKIPPED", source=g.nodes[src].label, root=g.nodes[up.root].label))
            label = g.input_port(node_id, port).label
            said = Msg("W-INPUT-UNUSED", node=node.label, input=label, reason=reason)
            if (said, port) not in out:
                out.append((said, port))
        if missing:
            items = self.item_list(begin, path)
            names = {i.key: i.name for i in items} if not isinstance(items, Pending) else {}
            listed = [Msg("I-EACH-ITEMFAILED", name=names.get(k, k), root=g.nodes[o.root].label) for k, o in missing.items()]
            out.append((Msg("W-EACH-FAILED", node=node.label, count=len(listed), items=listed), ""))
        return out

    def resolved(self, node_id: str, path: ItemPath = ()) -> Resolved:
        """The node's declarations resolved (nodes/applies.py) as the instance is cooked: the graph's, without the
        wires it goes without (dropped: a graph wire is gone when none of its instance wires is left; a parameter it
        drove keeps its typed value)."""
        key = Inst(node_id, path)
        if key in self._resolved:
            return self._resolved[key]
        dropped = self.dropped(node_id, path)
        if not dropped:
            r = self.graph.resolved(node_id)
        else:
            node, f = self.graph.nodes[node_id], self.graph.facts(node_id)
            left = {(port, src, sport) for port, wires in self.wires(node_id, path).items() for src, sport, p in wires
                    if (port, src, sport, p) not in dropped}
            wired, params = {}, dict(f.params)
            for port, types in f.wired.items():
                wires = self.graph.inputs.get((node_id, port), [])
                kept = tuple(t for (src, sport), t in zip(wires, types) if (port, src, sport) in left)
                if kept:
                    wired[port] = kept
                elif port.startswith(PARAM):
                    params[port.removeprefix(PARAM)] = node.params.get(port.removeprefix(PARAM))
            r = resolve(node.type, NodeFacts(params, wired, f.own, f.incoming, f.wired_out))
        return self._resolved.put(key, r)

    def provenance(self, node_id: str, path: ItemPath = ()) -> dict:
        """Which third-party projects produced what the instance takes (it and every node above it), whether all of
        them allow commercial use, and where the values those nodes were given came from (a focal length wired from
        AnyCalib over the camera's): what an output-settings node records next to its files (CookContext.provenance)."""
        key = Inst(node_id, path)
        if key not in self._provenance:
            sources, values = [], []
            scopes = self.graph.scopes
            for nid in self.graph.upstream_order(node_id):
                n = self.graph.nodes[nid]
                if n.type.runtime != "core":  # learned: it runs a model (not a format module reading or writing a file)
                    sources.append({"node": n.type.id, "label": n.label, "project": n.type.project.title,
                                    "commercial": self.graph.resolved(nid).licence.commercial,
                                    "learned": not issubclass(n.type, (ImportNode, OutputSettings))})
                depth = scopes.depth(nid)
                if depth <= len(path) and scopes.chain(nid) == scopes.chain(node_id)[:depth] and (said := self.sources(nid, path[:depth])):
                    values.append({"node": n.type.id, "label": n.label, "params": said})
            return self._provenance.put(key, {"sources": sources, "commercial": all(s["commercial"] for s in sources),
                                              **({"values": values} if values else {})})
        return self._provenance[key]

    def forget(self, node_ids) -> None:
        """What this evaluation remembers of every instance of these nodes, dropped: a cook just changed what is known
        of them (an instance cooked, failed or tried again) and so of what comes after (engine/cook.py)."""
        gone = set(node_ids)
        self._shots.drop_where(lambda k: k[0] in gone)  # keyed by (node, output, path), not by instance
        for memo in (self._plans, self._plan_errors, self._infos, self._outcomes, self._dropped, self._resolved, self._checks,
                     self._provenance, self._wires, self._items, self._gone, self._source_errors):
            memo.drop_where(lambda k: k.node in gone)
        self._orders.drop_where(lambda k: True)
        self._demands.drop_where(lambda k: True)
        self._satisfied.drop_where(lambda k: True)

    def shot(self, node_id: str, port: str, path: ItemPath = ()) -> dict:
        """What was photographed in the instance's output `port`, as far as the graph can tell before anything is
        cooked (data/contracts.py SHOT_KEYS: the plate's lens state, its pixel aspect, the lens it carries): what the
        node says of itself from its parameters (NodeDef.said_shot: a reader's 镜头状态 and 像素比) with whatever its
        output port takes from its picture input on top (contracts.shot_of), the same rule the engine settles a cooked
        packet by, so a warning that depends on it does not wait for a cook.

        {} for a key nothing says yet. Once the node is cooked its packet says it instead (engine/lint.py reads the
        manifest first)."""
        key = (node_id, port, path)
        if key in self._shots:
            return self._shots[key]
        g = self.graph
        node = g.nodes[node_id]
        shape = next((p.shape for p in g.outputs(node_id) if p.name == port), KEEPS)
        try:
            said = node.type.said_shot(self.params(node_id, path))
        except (OSError, ValueError, KeyError):  # its parameters are not readable yet: it says nothing
            said = {}
        upstream: dict = {}
        follows = shape.follows or node.type.picture
        if follows:
            for src, sport, p in self.taken(node_id, path).get(follows, ())[:1]:
                upstream = self.shot(src, sport, p)
        return self._shots.put(key, {**said, **shot_of(shape, upstream, said)})

    def checks(self, node_id: str, path: ItemPath = ()) -> list[dict]:
        """What the instance says before it is cooked: its usage checks (engine/lint.py; a B- one refuses it)
        and what it can foresee from its parameters and what it will cover (NodeDef.foresee: gaps in a sequence)."""
        key = Inst(node_id, path)
        if key not in self._checks:
            said = warnings(self, node_id, path)
            node = self.graph.nodes[node_id]
            # what the node always says (applies.py standing_notices: the size a node decides for itself), unless a
            # check of this node already spoke about the same thing: one message per subject, never a standing line
            # beside a warning that says more about it. What "the same thing" is, is the message's own module (the
            # middle of its code): there is no second list of which codes cover which
            spoken = {m["code"].split("-")[1] for m in said}
            said += [m.json() for m in standing_notices(node.type) if m.code.split("-")[1] not in spoken]
            try:  # only an instance that plans: one that can't says why as its error, without reading its file again here
                self.plan(node_id, path)
                said += [m.json() for m in node.type.foresee(self.params(node_id, path), self.info(node_id, path))]
            except PLAN_ERRORS:  # not plannable yet: its status says why
                pass
            return self._checks.put(key, said)
        return self._checks[key]

    # ------------------------------------------------------------------ parameters

    def _typed(self, node_id: str, path: ItemPath = ()) -> dict:
        """The node's own parameters as the instance is cooked: those its connections or other settings make inactive
        at their defaults."""
        node = self.graph.nodes[node_id]
        return effective_params(node.type, node.params, self.resolved(node_id, path))

    def params(self, node_id: str, path: ItemPath = ()) -> dict:
        """What the instance is cooked with: its own parameters (_typed), each driven by a wire at the wired value once
        that is known (the node feeding it cooked, or giving it from its parameters alone, a constant). A wired value
        it can't take leaves the typed one here; the graph status and the cook say why (wired_values)."""
        try:
            wired = self.wired_values(node_id, path)
        except ValueError:
            wired = {}
        return {**self._typed(node_id, path), **{name: one for name, (one, _) in wired.items()}}

    def _wired(self, node_id: str, path: ItemPath = ()) -> dict[str, list[Inst]]:
        """Input port -> the instances wired into it, its declared ones and one per row of a ports_from table that makes
        inputs (「多层 EXR 输出设置」's 图层): what the instance's result covers comes from these; a value driving a
        parameter covers nothing of the picture. Without the wires it goes without (taken)."""
        node = self.graph.nodes[node_id]
        taken = self.taken(node_id, path)
        return {port.name: [Inst(src, p) for src, _, p in taken.get(port.name, [])] for port in node.type.input_ports(node.params)}

    # ------------------------------------------------------------------ parameters driven by wires

    def known(self, node_id: str, port: str, path: ItemPath = ()) -> Packet | None:
        """What an instance's output gives, when that is known before cooking downstream: its packet once cooked, or
        what the node gives from its parameters alone (NodeDef.known_outputs: a constant value node, nodes/core/values.py); None not yet."""
        try:
            plan = self.plan(node_id, path)
        except PLAN_ERRORS:
            plan = None  # it cannot be cooked yet (no picture wired into 「AnyCalib 镜头标定」): a value that comes out
            # of its own parameters is known all the same (the graph tells it without a cook)
        if plan is not None and port in plan.outputs and valid(packet_dir(plan.outputs[port])):  # each output on its own (Engine: wanted)
            try:
                return self.packet(plan.outputs[port])
            except FileNotFoundError:  # removed between valid() and the manifest read (a clean, a recook): not known,
                pass  # like any packet not there; never a 500 on /api/status
        node = self.graph.nodes[node_id]
        try:
            meta = node.type.known_outputs(self.params(node_id, path)).get(port)
        except PLAN_ERRORS:
            return None
        if meta is None:
            return None
        # a value read out of its own meta needs no folder; it gets the cache folder it will land in once there is one
        where = packet_dir(plan.outputs[port]) if plan is not None and port in plan.outputs else Path(".")
        return Packet(where, self.graph.output_type(node_id, port), meta)

    def _wired_packet(self, node_id: str, name: str, path: ItemPath = ()) -> tuple[str, str, Packet] | Pending | None:
        """The wire into a promoted parameter: (node, port, its packet) once known, what it waits for while not; None
        when there is none or it gives nothing (an empty packet counts as not connected: the parameter keeps its own
        value)."""
        wires = self.wires(node_id, path).get(PARAM + name) or []
        if not wires:
            return None
        src, sport, p = wires[0]
        if self.graph.wire_problem(src, sport, node_id, PARAM + name):  # a wrong wire gives nothing: its node says why
            return None
        if (PARAM + name, src, sport, p) in self.dropped(node_id, path):  # its source failed: the typed value (W-INPUT-UNUSED says so)
            return None
        got = self._known_or_pending("value", Inst(src, p), sport)
        if isinstance(got, Pending):
            return got
        return None if got.meta.get("empty") else (src, sport, got)

    def _list_values(self, packet: Packet) -> list:
        """The values a list packet holds, in order: what a wire into a table parameter carries (「畸变参数」 driven by
        AnyCalib's 「畸变系数」). One item per row, each an ordinary value packet of its own."""
        from ..data.values import list_values

        return list_values(packet, self.packet)  # self.packet: this evaluation's manifest cache, no second read

    def _where(self, src: str, sport: str) -> str:
        out = next((p for p in self.graph.outputs(src) if p.name == sport), None)
        return f"「{self.graph.nodes[src].label}」的「{out.label if out else sport}」"

    def wired_values(self, node_id: str, path: ItemPath = ()) -> dict[str, tuple[Any, Any]]:
        """The instance's active parameters driven by a wire whose value is known (the node feeding it is cooked) ->
        (the value the parameter takes, the Value in the parameter's unit: nodes/values.py for_param). Raises
        ValueError saying what is wrong: a unit that doesn't convert, one value per frame that changes into a parameter
        taking one, a value out of the parameter's range."""
        from pydantic import ValidationError

        from ..data.values import describe_value, for_param, read, rows_for_param
        from ..nodes.params import range_said

        node = self.graph.nodes[node_id]
        inactive = self.resolved(node_id, path).params.inactive
        specs = {s["name"]: s for s in node.type.param_specs()}
        out = {}
        # tables last: their rows follow another parameter (「畸变参数」 follows 「镜头模型」), which may itself be wired
        for name in sorted(node.promoted, key=lambda n: specs[n]["items"] is not None if n in specs else False):
            wire = None if name in inactive else self._wired_packet(node_id, name, path)
            if wire is None or isinstance(wire, Pending):
                continue
            src, sport, packet = wire
            spec, where = specs[name], self._where(src, sport)
            if spec["items"] is not None:
                so_far = {**self._typed(node_id, path), **{n: v for n, (v, _) in out.items()}}
                rows = node.type.derive(so_far).get(name, so_far.get(name)) or []
                one = rows_for_param(spec, rows, self._list_values(packet), where)
                out[name] = (one, None)
                continue
            one, value = for_param(spec, read(packet), where)
            candidates = [one]
            if value.per_frame and value.numeric:  # the parameter's range holds on every frame
                lo, hi = value.span()
                candidates += [type(one)(lo), type(one)(hi)] if spec["widget"] != "vec3" else []
            for c in candidates:
                try:
                    node.type.Params(**{**self._typed(node_id, path), name: c})
                except ValidationError:
                    raise Invalid(Msg("E-PARAM-WIREDRANGE", name=spec["label"], where=where, value=describe_value(value), reason=range_said(spec))) from None
            out[name] = (one, value)
        return out

    def wired_from(self, node_id: str, path: ItemPath = ()) -> dict[str, str]:
        """Parameter -> the label of the node driving it with a wire. `sources` says the same thing in a sentence for
        a person to read; a node that has to *decide* something by it (「LensDistortion」: a lens it was handed is an
        estimate, one typed on it is a lens sheet someone measured) reads this instead of parsing that sentence."""
        out = {}
        for name in self.graph.nodes[node_id].promoted:
            wire = self._wired_packet(node_id, name, path)
            if wire is not None and not isinstance(wire, Pending):
                out[name] = self.graph.nodes[wire[0]].label
        return out

    def sources(self, node_id: str, path: ItemPath = ()) -> dict[str, str]:
        """Where the instance's parameters get their values, for those that say something about it: driven by a wire
        ("Focal Length 38.6 mm · 来自 AnyCalib 镜头标定"), or set over what a connected input would give (P(overrides=),
        "（覆盖相机的 Focal Length）"), or left to that input ("Focal Length · 来自相机（ViPE 相机解算）"). The node's footer and
        parameter panel show it, its worker's job and the provenance of what it makes keep it."""
        from ..data.values import describe_value, option_label, read, say, unit_problem

        g = self.graph
        node = g.nodes[node_id]
        connected = g.connected(node_id)
        inactive = self.resolved(node_id, path).params.inactive
        params = self.params(node_id, path)
        ports = {p.name: p.label for p in node.type.input_ports(node.params)}
        out = {}
        from ..nodes.applies import param_conditions, supplying_port

        supplies = {n: supplying_port(c) for n, c in param_conditions(node.type).items()}
        for spec in node.type.param_specs():
            name, label = spec["name"], spec["label"]
            if name in inactive:
                # the parameter is inactive because an input supplies the value (with a camera wired, Focal Length and
                # Filmback come from the camera): the footer still names the camera, otherwise the node would show only
                # a disabled parameter without indicating whose lens is used
                port = supplies.get(name) or ""
                if port and g.inputs.get((node_id, port)):
                    out[name] = f"{label} · 来自{ports.get(port, port)}（{g.nodes[g.inputs[(node_id, port)][0][0]].label}）"
                continue
            over = [p for p in spec["overrides"] if p in connected]
            wire = self._wired_packet(node_id, name, path)
            if wire is not None:
                src = wire.on.node if isinstance(wire, Pending) else wire[0]
                value = ""
                if not isinstance(wire, Pending) and spec["items"] is not None:
                    value = "：" + "、".join(describe_value(v) for v in self._list_values(wire[2])) if wire[2].meta.get("items") else ""
                elif not isinstance(wire, Pending):
                    v = read(wire[2])
                    v = v.in_unit(spec["unit"]) if spec["unit"] and not unit_problem(v.unit, spec["unit"]) else v
                    # a choice parameter is described by its own label (「OpenCV 鱼眼」), not the id sent on the wire
                    # (「opencv_fisheye」); a wired value not among its options is shown as is, and the refusal is
                    # reported separately
                    choice = option_label(spec, v.value) if not v.per_frame else ""
                    value = " " + (choice or describe_value(v))
                text = f"{label}{value} · 来自 {g.nodes[src].label}"
            elif over and params.get(name) is not None:
                text = f"{label} {say(spec, params[name])} · 手填"
            elif over:
                src = g.inputs[(node_id, over[0])][0][0]
                out[name] = f"{label} · 来自{ports[over[0]]}（{g.nodes[src].label}）"
                continue
            elif spec["assumed"] and params.get(name) is None:
                # an empty parameter is not necessarily ineffective: a parameter declaring `assumed` is still computed
                # with some value when empty, and without stating it in the footer the user cannot tell what the result
                # is based on. Parameters whose name and placeholder already make this clear (「已知 Focal Length」,
                # 「Filmback」, nodes/lens.py) do not declare `assumed`; no parameter currently uses it, and the rule is
                # kept for parameters that would otherwise be unclear.
                out[name] = f"{label} · {spec['assumed']}"
                continue
            else:
                continue
            if over:
                text += f"（覆盖{'、'.join(ports[p] for p in over)}的{label}）"
            out[name] = text
        return out

    # ------------------------------------------------------------------ frames

    def _all_frames(self, node_id: str, path: ItemPath = ()) -> Info:
        """NodeDef.info with the frame range not yet applied to this node (its inputs have it). While the instance
        waits for something (waits), what it says holds for what is known: worked out again once it is."""
        key = Inst(node_id, path)
        if key in self._infos:
            return self._infos[key]
        if self.source_missing(node_id, path) is not None:  # its file is not there: nothing of it is read
            return self._infos.put(key, Info())
        node = self.graph.nodes[node_id]
        self.graph.check_inputs(node_id)
        inputs = {port: [self.info(*src) for src in srcs] for port, srcs in self._wired(node_id, path).items()}
        try:
            info = node.type.info(self.params(node_id, path), inputs)
        except (OSError, ValueError) as exc:  # e.g. no file chosen: say which node
            raise CookError(node_id, Msg("E-COOK-FAILED", node=node.label, reason=exc)) from exc
        if not self.waits(node_id, path):
            return self._infos.put(key, info)
        return info

    def info(self, node_id: str, path: ItemPath = ()) -> Info:
        """What the instance's result will cover (frames, picture size), known before cooking: a frame source keeps the
        frames of its own that fall in the graph's frame range."""
        info = self._all_frames(node_id, path)
        node = self.graph.nodes[node_id]
        if not node.type.frame_source or info.still or self.graph.frames is None:
            return info
        lo, hi = self.graph.frames
        kept = tuple(f for f in info.frames if lo <= f <= hi)
        if not kept:
            raise CookError(node_id, Msg("E-FRAMES-OUTSIDE", node=node.label, first=info.frames[0], last=info.frames[-1], lo=lo, hi=hi))
        return replace(info, frames=kept)

    def work(self, node_id: str, path: ItemPath = ()) -> Info:
        """How much the instance processes: what its inputs cover (a source: what it emits). Timing records and time
        estimates scale by it."""
        ins = [self.info(*src) for srcs in self._wired(node_id, path).values() for src in srcs]
        return Info.merge(ins) if ins else self.info(node_id, path)

    def frame_range(self, targets: list[str]) -> tuple[int, int] | None:
        """The frames the inputs of cooking `targets` cover, first and last (several sources: from the earliest
        first to the latest last); None without a frame source. A source that can't be read yet is left out."""
        frames = []
        for nid in self.graph.needed(targets):
            if self.graph.nodes[nid].type.frame_source:
                for p in self.instances(nid)[0]:
                    try:
                        info = self._all_frames(nid, p)
                    except (GraphError, CookError):
                        continue
                    if not info.still and info.frames:
                        frames += [info.frames[0], info.frames[-1]]
        return (min(frames), max(frames)) if frames else None

    def check_frames(self, targets: list[str]) -> None:
        """A frame range narrows what the inputs cover; it never widens it."""
        full = self.frame_range(targets)
        if self.graph.frames and full and not full[0] <= self.graph.frames[0] <= self.graph.frames[1] <= full[1]:
            lo, hi = self.graph.frames
            raise GraphError(Msg("B-FRAMES-WIDER", lo=lo, hi=hi, first=full[0], last=full[1]))

    # ------------------------------------------------------------------ planning

    def plan(self, node_id: str, path: ItemPath = ()) -> NodePlan:
        key = Inst(node_id, path)
        if key in self._plans:
            return self._plans[key]
        if key in self._plan_errors:
            raise self._plan_errors[key]
        try:
            return self._plan(node_id, path)
        except PLAN_ERRORS as exc:
            if not self.waits(node_id, path):  # else tried again once what it waits for is known
                self._plan_errors.put(key, exc)
            raise

    def _plan(self, node_id: str, path: ItemPath) -> NodePlan:
        g = self.graph
        node = g.nodes[node_id]
        g.check_inputs(node_id)
        refused = next((w for w in warnings(self, node_id, path) if w["level"] == "B" and not w.get("refused")), None)
        if refused is not None:  # a check it can't get past (nodes/expects.py FrameCount, DistinctNames): the whole
            # check travels with the error, so the panel offers the same one click as it does for a refused wire
            raise Refused(refused)
        # every input's packets, a wired parameter's too, unless its connections or settings make it do nothing
        inactive = self.resolved(node_id, path).params.inactive
        wired = {PARAM + name for name in g.wired_params(node_id) if name not in inactive}
        dropped = self.dropped(node_id, path)  # its source failed: cooked without it, so a fixed source is cooked in later
        taken = self.taken(node_id, path)
        begins = sc.role(node.type) == BEGIN
        item = self.item_at(node_id, path) if begins else None
        input_fps = {port.name: [self.plan(src, p).outputs[sport] for src, sport, p in taken[port.name]]
                     for port in g.input_ports(node_id) if not port.param or port.name in wired}
        list_packet = ""
        if begins:  # its item, not the whole list: adding one item leaves the instances already there alone
            list_packet = next(iter(input_fps[node.type.item_input]), "")
            input_fps[node.type.item_input] = [item.packet]
        affecting = node.type.affecting_params()
        blob = {
            "cache": CACHE_VERSION,
            **({"facts": FACTS_VERSION} if FACTS_VERSION else {}),  # version of the fact fields (data/packet.py FACTS_VERSION)
            "type": node.type.id,
            "version": node.type.version,
            # effective values enter the fingerprint (nodes/base.py fingerprint_params): a parameter left empty and
            # filled in by the node gives the same fingerprint whether or not it was entered
            "params": {k: v for k, v in node.type.fingerprint_params(self.params(node_id, path)).items() if k in affecting},
            "inputs": input_fps,
            "source": self._source_identity(node_id, path),
        }
        # the node's extension and that extension's code / environment / weights (nodes/services.py
        # ProjectFacts.result_identity): after reinstalling the extension, changing weights or changing torch, old
        # results no longer hit the cache. Core nodes have no extension; the entry is absent and the fingerprint unchanged
        if node.type.project.result_identity:
            blob["ext"] = node.type.project.result_identity
        if node.type.frame_source:
            blob["frames"] = list(self.info(node_id, path).frames)
        if node.type.named_result:  # the node's name is in what it gives (an import's folder): renamed, cooked again
            blob["label"] = node.label
        if dropped:
            blob["without"] = sorted(f"{port}<-{src}.{sport}" + (f"@{'/'.join(p)}" if p else "") for port, src, sport, p in dropped)
        if item is not None:  # a begin instance: its item (what each output is addressed by is its own, below)
            blob["item"] = {"name": item.name, "index": item.index}
        if g.scopes.ended.get(node_id) is not None:  # an end: the items it gathers, by name, in order
            blob["items"] = [i.name for i in self.gathered(node_id, path)]
        fp = _hash(blob)
        outputs = {p.name: _hash([fp, p.name]) for p in g.outputs(node_id)}
        if item is not None:  # a block's begin: the item is its own packet, the others are addressed by what they
            # depend on (engine/scopes.py port_fp)
            outputs[node.type.item_output] = item.packet
            outputs.update({port: fp for port, (fp, _) in node.type.item_outputs(node.params, item, list_packet).items()})
        plan = NodePlan(fp, outputs, False)
        plan.cached = plan.has(self.needed_outputs(node_id))
        if not self.waits(node_id, path):  # else planned again once what it waits for is known
            return self._plans.put(Inst(node_id, path), plan)
        return plan

    def source_missing(self, node_id: str, path: ItemPath = ()) -> dict | None:
        """Its source file is not there (cleaned away, or, for an upload, not this account's, which is the same thing)
        (transfer/uploads.py resolve): the message to show, None when nothing is missing. The node has failed before
        any cook (failure), so what needs it is skipped and an optional or multi input goes on without it; the file is
        never opened, and another account's is never told apart from one that is gone."""
        key = Inst(node_id, path)
        if key not in self._gone:
            try:
                self._source_identity(node_id, path)
            except PLAN_ERRORS:  # any other trouble is that instance's own error, said where it happens
                pass
        return self._gone[key].json() if key in self._gone else None

    def _source_identity(self, node_id: str, path: ItemPath = ()) -> Any:
        from ..serving import serving

        key = Inst(node_id, path)
        if key in self._source_identities:  # one version of one graph: asked once per instance
            return self._source_identities[key]
        if key in self._source_errors:  # it could not be read: its file is not opened again to hear the same thing
            raise self._source_errors[key]
        node = self.graph.nodes[node_id]
        try:
            with serving(self.account):  # read as the account this evaluation is for, whatever thread asks
                identity = node.type.source_identity(node.params)
        except NotFound as exc:  # not there (cleaned, or not this account's): the node's own failure, said as it is
            self._gone.put(key, exc.message)
            raise self._source_errors.put(key, CookError(node_id, exc.message)) from exc
        except (OSError, ValueError) as exc:  # e.g. the input file is missing: say which node
            raise self._source_errors.put(key, CookError(node_id, Msg("E-COOK-FAILED", node=node.label, reason=exc))) from exc
        return self._source_identities.put(key, identity)

    def still_true(self) -> bool:
        """Whether this evaluation still holds: every external file it has already looked at (NodeDef.source_identity,
        of the instances `plan()` has actually planned so far) still has the identity it saw, and every packet it found
        cached is still on disk, a cheap stat check either way (uploads.resolve and Packet.exists touch the file
        system, not the file's bytes or manifest), not a re-read of anything. False: EvaluationCache.get() rebuilds
        rather than reusing this one. The cache generation is not guaranteed to bump when the cache directory changes
        from outside cook or disk.tidy (a test's own cleanup, an operator's `rm`), so the next look finds it out for
        itself instead of going stale forever; a node it never planned can't have gone stale under it, so this costs
        nothing for the untouched majority of a large graph."""
        from ..serving import serving

        # plan() may add entries while this runs on another thread: Memo walks a snapshot
        with serving(self.account):  # its files as this evaluation's account sees them, whatever thread asks
            for inst, was in self._source_identities.items():
                node = self.graph.nodes[inst.node]
                try:
                    now = node.type.source_identity(node.params)
                except (OSError, ValueError):
                    now = None
                if now != was:
                    return False
        for inst, plan in self._plans.items():
            if plan.cached and not plan.has(self.needed_outputs(inst.node)):
                return False
        return True

    def computes(self, targets: list[str], force: bool = False) -> list[str]:
        """The nodes cooking `targets` runs, dependencies first: those with an instance whose result is not cached
        (「输出」 always runs: it delivers and keeps nothing), forced targets, and the nodes behind what is still pending
        (they will run once it is known). An instance that can't even be planned (its file is not there) is one of
        them: the cook runs it, it fails at its own node and the rest of the cook goes on; a cook is
        never refused as a whole because one node of it is broken."""
        order, behind = self.order(targets)
        d = self.demand(targets)
        found = [i.node for i in order if not self.satisfied(i, d.get(i, frozenset())) or (force and i.node in targets)]
        return list(dict.fromkeys(found + [n for n in self.graph.needed(targets) if n in behind]))

    # ------------------------------------------------------------------ status

    def state(self, node_id: str, path: ItemPath, used: set[Inst] | None) -> str:
        """The instance's state as the editor shows it (engine/scopes.py STATES)."""
        if used is not None and Inst(node_id, path) not in used:
            return sc.UNUSED
        if (o := self.outcome(node_id, path)) is not None:
            return o.state
        try:
            plan = self.plan(node_id, path)
            self.wired_values(node_id, path)
        except PLAN_ERRORS:
            return sc.ERROR
        if self.waits(node_id, path):
            return sc.PENDING
        return sc.CACHED if plan.cached else sc.TODO

    def _used(self) -> tuple[set[Inst], frozenset[str]] | None:
        """The instances some result of the graph needs (every node no wire leaves, through the inputs switches
        choose) and the nodes behind what is pending; None when that can't be told (a cycle)."""
        sinks = [n for n in self.graph.nodes if not self.graph.outputs_by_node.get(n)]
        try:
            order, behind = self.order(sinks)
        except PLAN_ERRORS:
            return None
        return set(order), behind

    def _value_outputs(self, nid: str, path: ItemPath, ports: list[str]) -> dict[str, str]:
        """The instance's value outputs as the panel says them (「38.6 mm」), for those already known: cooked, or known
        from the node's own parameters (NodeDef.known_outputs). One place, used whether or not it can be planned."""
        from ..data.values import describe, is_value

        return {port: describe(pk) for port in ports
                if is_value(self.graph.output_type(nid, port))
                and (pk := self.known(nid, port, path)) is not None and not pk.meta.get("empty")}

    def _instance_status(self, nid: str, path: ItemPath, state: dict, used: set[Inst] | None) -> dict:
        """One instance's entry: its plan, outcome, messages and value outputs (the same fields as a node outside every
        block)."""
        g, node = self.graph, self.graph.nodes[nid]
        # every check is listed as it is, B- ones included (the page draws it by its level letter, 「提交前拦下」):
        # a B- one is also the instance's error below, the same way a refused wire is both
        entry = {"messages": [*state["messages"], *(m for m in self.checks(nid, path) if m not in state["messages"])]}
        entry["messages"] += [{**said.json(), **({"port": port} if port else {})} for said, port in self.unused_inputs(nid, path)]
        try:
            p = self.plan(nid, path)
        except PLAN_ERRORS as exc:
            entry.update({"fingerprint": None, "cached": False, "state": self.state(nid, path, used)})
            # an instance with no plan at all (its file is not there, a precondition it can't get past) still says
            # whose error it is: its own when it failed here, the one above it when it is only skipped
            if (o := self.outcome(nid, path)) is not None:
                entry["outcome"] = {"state": o.state, "root": o.root}
                entry["error" if o.state == "failed" else "skipped"] = o.message
            else:
                entry["error"] = message_of(exc).json()
            # an instance that cannot cook may still give values: outputs determined by its own parameters
            # (NodeDef.known_outputs, e.g. 「镜头模型」 of 「AnyCalib 镜头标定」) are sent as usual, so downstream panels
            # can show 「OpenCV 鱼眼 · 来自 AnyCalib」 before a picture is wired
            if values := self._value_outputs(nid, path, [port.name for port in g.outputs(nid)]):
                entry["values"] = values
            return entry
        entry.update({"fingerprint": p.fingerprint, "cached": p.cached, "outputs": p.outputs,
                      "present": sorted(port for port, fp in p.outputs.items() if valid(packet_dir(fp))),
                      # the generation of each port on disk (the packet's commit time, data/packet.py created): the
                      # page includes it in its cache key, so a recook of the same fingerprint gives new keys, and the
                      # page drops stale packet descriptions when it changes
                      "gens": {port: (self.manifest(fp) or {}).get("created", "") for port, fp in p.outputs.items() if valid(packet_dir(fp))},
                      # the ports of this node that are needed (`needed_outputs`: those with outgoing wires, or all
                      # when none). Sent so the page can decide which channels to upload this time: only wired channels
                      # are uploaded. The decision is made only in `needed_outputs`; the page follows this list.
                      "needed": sorted(self.needed_outputs(nid))})
        # the channels to upload (`NodeDef.upload_channels`, answerable only by file-reading nodes): `{"take", "write"}`;
        # absent means the whole file is uploaded (PNG / JPG, all channels needed, or nodes that read no file). The
        # mapping from ports to channel names lives only in the node; the page decodes per this list in a worker and
        # uploads with the chunked protocol (`webui/src/transfer/planes.ts`)
        if (planes := self._upload_channels(nid, path)) is not None:
            entry["channels"] = planes
        if (o := self.outcome(nid, path)) is not None:  # no result because of an error: its own, or one above it
            entry["outcome"] = {"state": o.state, "root": o.root}
            entry["error" if o.state == "failed" else "skipped"] = o.message
        if p.cached:  # what it said while it was cooked, kept with its result (Packet.commit)
            said = next((m["messages"] for o in p.outputs.values() if (m := self.manifest(o)) and m["messages"]), [])
            entry["messages"] = [*entry["messages"], *(m for m in said if m not in entry["messages"])]
        try:
            self.wired_values(nid, path)
        except ValueError as exc:
            entry["error"] = Msg("E-COOK-WIRED", node=node.label, reason=exc).json()
        if values := self._value_outputs(nid, path, list(p.outputs)):
            entry["values"] = values
        if strip := self.strip_values(nid, path):
            entry["strip"] = strip
        entry["state"] = self.state(nid, path, used)
        return entry

    def strip_values(self, node_id: str, path: ItemPath = ()) -> list[dict]:
        """Current values of the parameters shown in the strip under the view (NodeDef.strip), one {label, text} each: a
        wired value as on the wire (the same number `sources` states), an entered value as entered, otherwise empty
        (the page draws 「—」). These are not output ports (see nodes/base.py strip)."""
        from ..data.values import describe_value, option_label, read, say, unit_problem

        node = self.graph.nodes[node_id]
        if not node.type.strip:
            return []
        params = self.params(node_id, path)
        specs = {s["name"]: s for s in node.type.param_specs()}
        out = []
        for name, label in node.type.strip.items():
            spec = specs[name]
            text = ""
            wire = self._wired_packet(node_id, name, path)
            if wire is not None and not isinstance(wire, Pending):
                v = read(wire[2])
                v = v.in_unit(spec["unit"]) if spec["unit"] and not unit_problem(v.unit, spec["unit"]) else v
                text = (option_label(spec, v.value) if not v.per_frame else "") or describe_value(v)
            elif wire is None and params.get(name) is not None and params.get(name) != "":
                text = say(spec, params[name])
            out.append({"label": label, "text": text})
        return out

    def _upload_channels(self, nid: str, path: ItemPath = ()) -> dict | None:
        """The channels of this node's source to upload for the current cook (NodeDef.upload_channels; None: upload the
        whole file, or the node reads no file). A header not read yet, parameters not filled in or an upload already
        cleaned only mean there is no answer: the whole file is uploaded, and the graph status must not fail."""
        node = self.graph.nodes[nid]
        try:
            return node.type.upload_channels(self.params(nid, path), self.needed_outputs(nid))
        except Exception:  # noqa: BLE001 no answer means the whole file (reading reports its own errors)
            return None

    def view_path(self, node_id: str, view: dict[str, str] | None = None) -> ItemPath | None:
        """The instance of the node the view is on: the item each block around it is showing (`view`: begin -> item
        key, from the request), the first item of it otherwise; None while an item list is not known yet."""
        path: ItemPath = ()
        for n, begin in enumerate(self.graph.scopes.chain(node_id)):
            items = self.item_list(begin, path[:n])
            if isinstance(items, Pending) or not items:
                return None
            want = (view or {}).get(begin)
            path += (next((i.key for i in items if i.key == want), items[0].key),)
        return path

    def items(self, node_id: str, offset: int = 0, limit: int = 50) -> dict:
        """Item by item for one node inside a block (what the status reply does not carry, asked for on
        its own): {"total", "offset", "items": [one entry per instance, as the status has it for the view item, with
        "item": {"path", "names"}]}."""
        paths, _pending = self.instances(node_id)
        used = (self._used() or (None, frozenset()))[0]
        window = paths[offset:offset + limit] if limit else paths[offset:]
        return {"total": len(paths), "offset": offset,
                "items": [{**self._instance_status(node_id, p, {"messages": []}, used),
                           "item": {"path": list(p), "names": self._names(node_id, p)}} for p in window]}

    def status(self, view: dict[str, str] | None = None) -> dict:
        """The graph as the editor reads it, everything worked out here so the
        page never derives a rule itself (webui/src/graph/rules.ts only reads it); the shape of the block parts
        ("scopes", a node's "state", "items", "summary") is written in engine/scopes.py:
        - "nodes": per node its fingerprint, whether its result is cached, its outputs' packets, the parameters its
          connections and settings make inactive (with why), its cost and licence with its parameters, its usage
          warnings (engine/lint.py), where its parameters get their values (sources) and, once cooked, what its value
          outputs hold ("values": port -> "38.6 mm"); its ports as they are in this graph (Graph.ports), the viewer
          handles that apply (Graph.handles), what a click on 计算 cooks ("policy") and its state. A wired
          value it can't take (once known) is its error.
        - "wires": every wire's type and state (Graph.wire_states).
        - "scopes": every 逐项处理 block, its members and its item lists.
        - "deliver": what 交付 cooks (every 「输出」 together), null without one.
        A node inside a block answers for the item the view is on (`view`: begin -> item key; the first item by
        default) and says how all its items stand ("summary"); item by item is `items()`.
        The server adds the graph's key, the page's cook-inputs version and the shown node's plan (server/packets.py)."""
        g = self.graph
        used_behind = self._used()
        used = None if used_behind is None else used_behind[0]
        behind = frozenset() if used_behind is None else used_behind[1]
        out = {}
        for nid in g.nodes:
            node = g.nodes[nid]
            inside = bool(g.scopes.depth(nid))
            r = self.resolved(nid) if not inside else g.resolved(nid)
            # what its declarations say here (nodes/applies.py): the parameters that do nothing and why, those a cook
            # still has to tell, what it costs and whose licence it is with these parameters
            state = {"applies": r.params.json(),
                     "cost": r.cost.describe(), "licence": r.licence.describe(),
                     "messages": [], "sources": self.sources(nid) if not inside else {},
                     "ports": g.ports(nid), "handles": g.handles(nid),
                     # where a placing node puts what it gives (nodes/handles.py Places), for the viewer's preview
                     **({"places": node.type.places.placement()} if node.type.places else {})}
            if not inside:
                entry = self._instance_status(nid, (), state, used if nid not in behind else None)
                # a B- check is its error (plan) and is listed among its messages with its one click, like a refused wire
                out[nid] = {**state, **entry}
                continue
            # inside a block: the item the view is on, and how the items stand together (item by item
            # is `items()`, asked for on its own)
            paths, pending = self.instances(nid)
            states = [self.state(nid, p, used) for p in paths]
            summary = {"total": len(paths), **sc.summary(states)}
            shown = self.view_path(nid, view)
            up = next((o for w in pending if (o := self.outcome(*w.on)) is not None), None)
            entry = ({**self._instance_status(nid, shown, state, used), "item": {"path": list(shown), "names": self._names(nid, shown)},
                      "sources": self.sources(nid, shown)} if shown is not None
                     else {"fingerprint": None, "cached": False, "outputs": {}, "messages": [],
                           "item": {"path": [], "names": []}, "state": sc.PENDING})
            entry["summary"] = summary
            entry["cached"] = bool(paths) and all(s == sc.CACHED for s in states)
            entry["state"] = sc.SKIPPED if up is not None and not paths else sc.node_state(states, bool(pending))
            if up is not None and not paths:
                o = self._skipped(node, g.nodes[up.root].label, up)
                entry.update({"outcome": {"state": o.state, "root": o.root}, "skipped": o.message})
            out[nid] = {**state, **entry}
        for nid in g.nodes:
            out[nid]["policy"] = self._case(self._targets(g.cook_targets, nid))  # what a click on 「计算」 cooks
        deliveries = g.deliveries()
        return {"nodes": out, "wires": g.wire_states(), "scopes": self._scopes_status(used),
                "deliver": self._case(deliveries) if deliveries else None}

    def item_names(self, node_id: str, path: ItemPath) -> list[str]:
        """The item names of an instance's path, outer block first ([] outside every block)."""
        return self._names(node_id, path) if path else []

    def _names(self, node_id: str, path: ItemPath) -> list[str]:
        """The item names of an instance's path, outer first."""
        names = []
        for n, begin in enumerate(self.graph.scopes.chain(node_id)):
            items = self.item_list(begin, path[:n])
            names.append(next((i.name for i in items if i.key == path[n]), "") if not isinstance(items, Pending) else "")
        return names

    def _scopes_status(self, used: set[Inst] | None) -> list[dict]:
        """Every 逐项处理 block: its members, its items and how they stand together. 「3 条 · 2/3 已算」 is worked out
        here, like every other rule, so the page only shows it."""
        out = []
        for begin, s in self.graph.scopes.scopes.items():
            lists = []
            for p in self._parent_paths(begin):
                items = self.item_list(begin, p)
                lists.append({"path": list(p), "pending": True} if isinstance(items, Pending)
                             else {"path": list(p), "items": [{"key": i.key, "name": i.name} for i in items],
                                   "summary": self._scope_summary(s, p, items, used)})
            out.append({**s.describe(), "lists": lists})
        return out

    def _scope_summary(self, s: sc.Scope, path: ItemPath, items: list[ItemAt], used: set[Inst] | None) -> dict:
        """How the items of one block stand: {"total": how many, then the STATES that have any}. An item counts as
        what its members make of it together, the same rule a node's own instances follow (engine/scopes.py
        node_state), so a block is 「已算」 only once every node in it has that item's result."""
        by_item: dict[str, list[str]] = {i.key: [] for i in items}
        depth = len(path)
        for nid in s.members:
            paths, _pending = self.instances(nid)
            for ip in paths:
                if len(ip) > depth and ip[:depth] == path and ip[depth] in by_item:
                    by_item[ip[depth]].append(self.state(nid, ip, used))
        states = [sc.node_state(v, not v) for v in by_item.values()]
        return {"total": len(items), **sc.summary(states)}

    def _parent_paths(self, begin: str) -> list[ItemPath]:
        """The paths the block `begin` starts sits at, as far as the item lists of the blocks around it are known."""
        scopes = self.graph.scopes
        chain = scopes.chain(begin)[:-1]
        paths: list[ItemPath] = [()]
        for b in chain:
            deeper = []
            for p in paths:
                items = self.item_list(b, p[:scopes.depth(b) - 1])
                if not isinstance(items, Pending):
                    deeper += [p + (i.key,) for i in items]
            paths = deeper
        return paths

    @staticmethod
    def _targets(of, node_id: str) -> list[str]:
        try:
            return of(node_id)
        except GraphError:  # a graph the rules refuse (a cycle): the node alone
            return [node_id]

    def _case(self, targets: list[str]) -> dict:
        """What cooking `targets` computes, and whether it delivers (collects and packs files: a 「输出」 among them),
        judged on the nodes it computes: those with an instance that does not have every port this cook wants of it
        (`demand` / `satisfied`, the same rule `computes` and the queue apply, never the whole graph's `cached`, which
        would not match what the cook actually writes)."""
        try:
            order, behind = self.order(targets)
            d = self.demand(targets)
            needed = list(dict.fromkeys([i.node for i in order] + [n for n in self.graph.needed(targets) if n in behind]))
            done = {n for n in needed if n not in behind and all(self.satisfied(i, d.get(i, frozenset())) for i in order if i.node == n)}
        except PLAN_ERRORS:  # a cycle: what can be ordered, as a walk that stops where it closes
            needed = self._walk(targets)
            done = set()
        computes = [n for n in needed if n not in done]
        return {"targets": targets, "computes": computes,
                "delivers": any(self.graph.nodes[n].type.delivers for n in computes if n in self.graph.nodes)}

    def _walk(self, targets: list[str]) -> list[str]:
        needed: list[str] = []
        seen: set[str] = set()

        def visit(n: str) -> None:  # dependencies first, in the order the inputs are declared (a cycle stops)
            if n in seen:
                return
            seen.add(n)
            for port in self.graph.input_ports(n):
                for src, _ in self.graph.inputs.get((n, port.name), []):
                    visit(src)
            needed.append(n)

        for t in targets:
            visit(t)
        return needed
