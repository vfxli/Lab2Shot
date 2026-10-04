"""Everything a graph's state can tell without cooking anything: fingerprints, cached outputs, effective parameters,
wired values, sources, usage warnings, the status the editor polls. Without cooking a node, not without reading
files: planning an import reads what its file holds (ImportNode.listing, through its extension's worker for a format
the core does not read; cached per file), which the engine does ahead, outside its lock (NodeDef.ready). One `Evaluation` is one version of one graph
(engine/evaluations.py keeps one per graph version): every method here remembers its answer per node instance (a
node and its item path, engine/scopes.py Inst; a node outside every 逐项处理 block has the one instance with the empty
path, which is what every `path=()` below means), so asking the same question twice (the parameter panel, the footer,
the status reply, a second /api/status call with nothing changed) costs nothing the second time. `Engine`
(engine/cook.py) holds one to plan and answer status from, and cooks on top of it, forgetting what each cook changes
(Demand.forget), so an Engine cooks on an Evaluation of its own, never one from the cache (the farm builds a fresh one
per cook). Whether an instance is cached is asked when needed (Demand.cached), never kept in its plan. Evaluation is made of parts, each the one place of its concept: presence (is it there), routing (which way a
switch goes), demand (what a cook needs), status (what the editor is told); this module keeps instances, wires,
outcomes, parameters, frames and planning.

What can only be known once something above is cooked (a wired parameter's value, a block's items, the input a switch
chooses) is pending (engine/scopes.py Pending), found by the one function `lookup` (engine/presence.py). The wait passes
down: an instance that waits gives nothing known (Presence.known), so what reads it waits too. What is worked out for an
instance that waits (its plan, info, warnings, what its wires bring) is remembered like any other answer; the cook of
what it waits for forgets it and what follows (Demand.forget), and it is worked out again then (`waits`, `order`).

Looking at the cache never marks a packet used: its "last used" time is set only where a packet is actually read for
cooking (Engine._run_node) or shown (server/packets.py's GETs) (Packet.used, data/packet.py). Reading writes nothing:
a packet found invalid is only answered for (NodePlan.has); the cook that needs it writes it again."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..data.contracts import KEEPS, shot_of
from ..data.packet import CACHE_VERSION, FACTS_VERSION, Packet
from ..errors import CookError, GraphError, Invalid, NotFound, Refused, message_of
from .. import i18n
from ..messages import Both, Msg, again
from ..nodes.applies import NodeFacts, Resolved, effective_params, resolve, standing_notices
from ..nodes.base import Info
from ..nodes.port import PARAM
from ..nodes.formats import ImportNode
from ..nodes.output import OutputSettings
from . import scopes as sc
from .demand import Demand
from .graph import Graph, walk, with_table
from .lint import warnings
from .presence import Presence, generation, present
from .records import INSTANCE, KEPT, PLAN_ERRORS, Dropped, Memo, NodePlan, Outcome, Wire, _hash, failure_file, planned
from .routing import Routing
from .scopes import BEGIN, END, Inst, ItemAt, ItemPath, Pending
from .status import Status

if TYPE_CHECKING:
    from ..serving import Account

__all__ = ["Evaluation", "Inst", "NodePlan", "Outcome", "PLAN_ERRORS", "failure_file"]



def _again(exc: BaseException) -> BaseException:
    """A kept error, to be raised again: its value, without the frames of every earlier raise (each raise of the same
    object adds its own to __traceback__: a kept evaluation polled a thousand times would hold them all)."""
    exc.__context__ = None
    return exc.with_traceback(None)


def _said_label(port) -> str:
    """A port's name for a message, in both languages (messages.Both): read in whoever's language follows it."""
    return Both.of(lambda: port.label)


class Evaluation(Presence, Routing, Demand, Status):
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
        # every table it remembers answers in is a Memo (its concurrency rule), made by _memo: registered, so forget
        # reaches every one
        self._tables: list[Memo] = []
        self.stale = False  # something it planned changed on disk (records.changed): the cache hands it out no more
        self._plans: Memo[Inst, NodePlan] = self._memo()
        self._infos: Memo[Inst, Info] = self._memo()  # before the frame range: a frame source's every frame
        # why an instance could not be planned (a refused file, a precondition): kept so status, checks and the plan
        # of what comes after read its file once, not once each
        self._plan_errors: Memo[Inst, Exception] = self._memo()
        self._source_identities: Memo[Inst, Any] = self._memo(KEPT)  # its source_identity() the first time plan() saw it
        self._gone: Memo[Inst, Msg] = self._memo()  # its source file is not there (cleaned, or not this account's): why
        self._source_errors: Memo[Inst, Exception] = self._memo()  # its source file could not be read: why (asked once)
        self._outcomes: Memo[Inst, Outcome | None] = self._memo()
        self._dropped: Memo[Inst, dict[Dropped, Outcome]] = self._memo()
        self._resolved: Memo[Inst, Resolved] = self._memo()
        self._checks: Memo[Inst, list[dict]] = self._memo()
        self._warnings: Memo[Inst, list[dict]] = self._memo()  # its usage checks (engine/lint.py warnings)
        self._shots: Memo[tuple, dict] = self._memo(lambda k: Inst(k[0], k[2]))  # (node, output, path) -> what was photographed, before any cook
        self._provenance: Memo[Inst, dict] = self._memo()
        self._wires: Memo[Inst, dict[str, list[Wire]]] = self._memo()  # input -> the wires the instance takes
        self._waits: Memo[Inst, tuple[Pending, ...]] = self._memo()  # what it waits for (forget: once that is cooked)
        self._items: Memo[Inst, list[ItemAt]] = self._memo()  # a begin and the path its scope sits at -> the items
        self._warming = threading.local()  # this thread is filling the tables from the sources up (_sources_first)
        self._open_presence()
        self._open_routing()
        self._open_demand()
        self._open_status()

    def _planned(self, inst: Inst) -> bool:
        return inst in self._plans or inst in self._plan_errors

    def _sources_first(self, node_id: str, path: ItemPath, known, *fill) -> None:
        """Before an answer about an instance that is worked out from its sources' answers (plan, outcome, info,
        provisional: each asks its sources, which ask theirs), work theirs out first, from the far end of the graph
        down (walk: no recursion), so the call itself goes one step up, never down a whole chain (a chain of a
        thousand nodes would overflow Python's stack). `fill`: what to work out for each, in turn; `known(inst)`: its
        answer is remembered already, so the walk goes no further up from it (what is above was worked out with it).
        Along the wires the instance takes (`wires`: a switch's route) once known, else every wire the graph draws into
        it, a block's end into each item's instance (as `deps`), so a chain inside a block is walked too. Once per outermost call, on this thread: the calls it makes find their sources worked out."""
        if getattr(self._warming, "on", False):
            return
        self._warming.on = True

        start = Inst(node_id, path)

        g = self.graph

        def sources(inst: Inst) -> list[Inst]:
            if inst != start and known(inst):
                return []
            if inst in self._wires:  # the wires it takes, once known (a switch's route)
                return [Inst(src, p) for ws in self._wires[inst].values() for src, _, p in ws]
            # else every wire the graph draws into it, where its source is at one instance: working out which it takes
            # asks its sources' answers itself (a switch's condition), which is what this walk is there to do first
            out = []
            for src in dict.fromkeys(g.sources_of(inst.node)):
                if (at := g.scopes.source_path(src, inst.node, inst.path)) is not None:
                    out.append(Inst(src, at))
                elif g.scopes.depth(src) > len(inst.path):  # deeper: a block's end gathers it per item, as deps does
                    try:
                        items_of, _ = self.instances(src)
                    except PLAN_ERRORS:
                        continue
                    out += [Inst(src, p) for p in items_of if p[:len(inst.path)] == inst.path]
            return out

        try:
            for inst in walk([start], sources)[:-1]:
                if known(inst):
                    continue
                for fn in fill:
                    try:
                        fn(*inst)
                    except PLAN_ERRORS:  # the same error the answer itself meets, and says, when it gets there
                        pass
        finally:
            self._warming.on = False

    def remembered(self) -> int:
        """How many answers its tables hold: what it costs to keep (EvaluationCache bounds the sum)."""
        return sum(len(memo) for memo in self._tables)

    def _memo(self, of: Any = INSTANCE) -> Memo:
        """A new table of this evaluation, registered (Memo.of says what its keys are about, for forget)."""
        memo = Memo(of)
        self._tables.append(memo)
        return memo

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
        got = self.wired_input(begin, node.type.item_input, path, "items")
        if isinstance(got, Pending):
            return got
        found = [] if got is None else node.type.scope_items(node.params, got[2])  # none, refused or empty: no items
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

    def wires(self, node_id: str, path: ItemPath = ()) -> dict[str, list[Wire]]:
        """Every input of the instance (a promoted parameter's too) -> the wires it takes, each with the path of the
        instance it comes from: the same path cut to that node's depth; into a block's end, one per item of the
        block (and none while they are pending); into a switch, only the inputs its condition chooses (only the
        condition while that is pending). A wire the rules of a block refuse is left out (the node's error says so)."""
        key = Inst(node_id, path)
        if key in self._wires:
            return self._wires[key]
        self._one_instance(node_id, path)
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
                        if inner is not None and not port.param and scopes.chain(src) == inner:
                            taken += [(src, sport, path + (i.key,)) for i in items] if not isinstance(items, Pending) else []
                        elif (at := scopes.source_path(src, node_id, path)) is not None:
                            taken.append((src, sport, at))
            out[port.name] = taken
        if isinstance(chosen, Pending) or isinstance(items, Pending):
            return out
        return self._wires.put(key, out)

    def waits(self, node_id: str, path: ItemPath = ()) -> tuple[Pending, ...]:
        """What the instance waits for before it can be planned for good (engine/scopes.py Pending). Remembered either
        way: what it waits for changes only when something above it is cooked, and that forgets it (Demand.forget: a
        value read before planning forgets everything downstream)."""
        key = Inst(node_id, path)
        if key in self._waits:
            return self._waits[key]
        self._sources_first(node_id, path, self._waits.__contains__, self.waits, self.outcome)
        node = self.graph.nodes[node_id]
        out: list[Pending] = []
        inactive = self.resolved(node_id, path).params.inactive
        for name in node.promoted:
            if name not in inactive and isinstance(w := self._wired_packet(node_id, name, path), Pending):
                out.append(w)
        for port in node.type.params_inputs:
            if isinstance(w := self._input_value(node_id, path, port), Pending):
                out.append(w)
        if isinstance(chosen := self._chosen(node_id, path), Pending):
            out.append(chosen)
        if (begin := self.graph.scopes.ended.get(node_id)) is not None and isinstance(items := self.item_list(begin, path), Pending):
            out.append(items)
        return self._waits.put(key, tuple(out))

    def deps(self, inst: Inst) -> list[Inst]:
        """The instances `inst` needs cooked first: those wired into it (as `wires` takes them), what it waits for, and,
        for a block's end, the list its items come from: that decides them even when there are none (an end over an
        empty list gathers nothing, and the list and what made it are what said so: used, not left grey)."""
        g = self.graph
        found = [Inst(src, p) for wires in self.wires(*inst).values() for src, _, p in wires]
        found += [w.on for w in self.waits(*inst)]
        if (begin := g.scopes.ended.get(inst.node)) is not None:
            for src, _ in g.inputs.get((begin, g.nodes[begin].type.item_input), [])[:1]:
                if (at := g.scopes.source_path(src, begin, inst.path)) is not None:
                    found.append(Inst(src, at))
        return list(dict.fromkeys(found))

    # ------------------------------------------------------------------ errors and what they leave out

    def failure(self, node_id: str, path: ItemPath = ()) -> Outcome | None:
        """The instance's own failure, None when it has none (or can't be planned: that is said as its status error).
        Before any cook, a node without a result whose extension can't be used now (ProjectFacts.available: not
        installed, built from older code, weights missing) has failed already (E-COOK-UNAVAILABLE), and so has one
        whose source file is not there (cleaned away, or not this account's, which is the same answer: source_missing,
        E-UPLOAD-GONE); otherwise the error it last failed with at its current fingerprint (its message, and its log),
        the one failure the next cook tries again (retryable). What needs it is skipped, a multi input goes on
        without it."""
        if (gone := self.source_missing(node_id, path)) is not None:
            return Outcome(sc.FAILED, node_id, gone)
        try:
            plan = self.plan(node_id, path)
        except PLAN_ERRORS:
            return None
        node = self.graph.nodes[node_id]
        project = node.type.project
        if project.extension is not None and (why := project.available()) is not None and not self._has_result(node_id, plan):
            return Outcome(sc.FAILED, node_id, Msg("E-COOK-UNAVAILABLE", node=node.label, reason=why).json())  # the one
            # answer the session's node:<type> reads too
        path_ = failure_file(plan.fingerprint)
        if not path_.exists() or self._has_result(node_id, plan):  # a record beside the result it lacks: another cook
            # of it succeeded (the packets on disk are the fact; a record only says why they are not there)
            return None
        try:
            return Outcome(sc.FAILED, node_id, json.loads(path_.read_text(encoding="utf-8")), retryable=True)
        except (OSError, ValueError):
            return None

    def _has_result(self, node_id: str, plan: NodePlan) -> bool:
        """Whether the instance has its result on disk, as its own failure (failure) asks it: every output a wire in
        the graph takes from it, whichever way of a switch that wire goes into (every output when none is wired; a
        「有没有」's input takes nothing: Routing._choose). Not `cached`, which is about the outputs its readers need
        along the routes switches take now: a route can hang on what this answer decides — a 「有没有」 driving a
        switch one of whose ways this node feeds (present -> the source's plan, which leaves out the wires whose source
        failed -> failure -> which outputs a switch takes -> the 「有没有」 again: the recursion that took /api/status
        down), or any value known before cooking that drives such a switch."""
        g = self.graph
        wired = frozenset(sport for dst, dport in g.outputs_by_node.get(node_id, ())
                          if dport != g.nodes[dst].type.presence_of
                          for src, sport in g.inputs.get((dst, dport), []) if src == node_id and sport in plan.outputs)
        return plan.has(wired or frozenset(plan.outputs))

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
        self._sources_first(node_id, path, self._outcomes.__contains__, self.plan, self.outcome)
        g, node = self.graph, self.graph.nodes[node_id]
        if (behind := self._behind_missing(node_id, path)) is not None:  # an input it needs has nothing because a
            # reader up the line has no file: that reader is the cause, said there (not this node's waiting wire)
            return self._outcomes.put(key, behind)
        try:  # its own wiring is wrong: that is its own error, whatever stands above it (the first thing its plan says)
            g.check_inputs(node_id, only=self.taken_ports(node_id, path))
        except GraphError as exc:
            if not self.waits(node_id, path):
                return self._outcomes.put(key, Outcome(sc.ERROR, node_id, exc.message.json()))
        if (blocked := self._blocked(node_id, path)) is not None:  # a 「阻断」 set to block: nothing above it is taken
            return self._outcomes.put(key, blocked)
        found = self._outside_frames(node_id, path)
        wires = self.wires(node_id, path)
        for port in g.input_ports(node_id):
            if not self.requires(node_id, path, port):
                continue
            ups =[o for src, _, p in wires[port.name] if (o := self.outcome(src, p)) is not None]
            if ups and (not self._gathers(node_id, port) or len(ups) == len(wires[port.name])):
                found = self._skipped(node, _said_label(port), ups[0])
                break
        # alternative wirings (NodeDef.input_choice): when every used wire is gone because upstream failed or was
        # skipped, this node is skipped too. Otherwise it would cook and report "nothing wired" itself, saying the same
        # thing twice and hiding the real error (the real error is reported once at the failing node; downstream is
        # marked skipped)
        if found is None and node.type.input_choice:
            live = [p for p in g.input_ports(node_id) if p.name in node.type.choice_inputs() and wires[p.name]]
            gone = [(_said_label(p), o) for p in live for src, _, sub in wires[p.name] if (o := self.outcome(src, sub)) is not None]
            if live and len(gone) == len(live):
                found = self._skipped(node, gone[0][0], gone[0][1])
        if found is None and getattr(node.type, "collects", False):  # a node that only gathers what comes in (多层
            # EXR 输出设置's rows): every input wired into it blocked on the way, it has nothing to gather — blocked too
            found = self._all_blocked(node_id, path)
        if found is None:
            for w in self.waits(node_id, path):
                if (up := self.outcome(*w.on)) is not None:
                    port = g.input_port(node_id, node.type.item_input if w.kind == "items" and sc.role(node.type) == BEGIN else w.port)
                    found = self._skipped(node, _said_label(port) if port else g.nodes[w.on.node].label, up)
                    break
        if node.type.delivers and (broken := self._chains_broken(node_id, path)) is not None:
            found = broken  # 「输出」 packs only whole lines: what stands above it decides before anything else
        if found is None and not self.waits(node_id, path):  # a value wired into a parameter that it can't take (out of
            # range, a unit that doesn't convert): known now, whatever is cooked again — its own error, like a plan's
            try:
                self.plan(node_id, path)
                self.wired_values(node_id, path)
            except ValueError as exc:
                if not isinstance(exc, (GraphError, CookError)):
                    found = Outcome(sc.ERROR, node_id, Msg("E-COOK-WIRED", node=node.label, reason=exc).json())
            except PLAN_ERRORS:
                pass  # said below
        if found is None:
            found = self.failure(node_id, path)
        if found is None and not self.waits(node_id, path):  # it can't be planned (and waits for nothing that could
            # change that): its own error, an outcome like a failure, so what needs it is skipped behind it (root: it) and
            # a block's end gathers without that item (W-EACH-FAILED), never taking its error as their own
            try:
                self.plan(node_id, path)
            except PLAN_ERRORS as exc:
                found = Outcome(sc.ERROR, node_id, message_of(exc).json())
        return self._outcomes.put(key, found)

    def _chains_broken(self, node_id: str, path: ItemPath) -> Outcome | None:
        """A node that delivers (「输出」) and a line into it that is not whole: somewhere above one of the
        output-settings nodes wired into it a node failed, was skipped behind a failure, or a block's end gathered
        without an item that failed (an output packs only when every line into it is complete, so what it hands over
        is never partial without saying so; the user cuts the broken line or fixes it). Fails it, naming each broken line and the node
        at its root (and the item, inside a block). None: every line is whole (or not known to be broken yet)."""
        g = self.graph
        seen: set[Inst] = set()
        broken: dict[tuple[str, tuple[str, ...]], list[str]] = {}  # (root, its items) -> the lines it breaks, in order
        # what went wrong at each root, said to whoever uses a card (E-OUTPUT-CHAINFAILED's app words: no lines, no
        # node names, only what to put right — 「没有选择图像序列」)
        reasons: dict[tuple[str, tuple[str, ...]], Msg] = {}
        retryable = True
        try:
            wires = self.wires(node_id, path)
        except PLAN_ERRORS:
            return None
        for port in g.input_ports(node_id):
            for src, _sport, p in wires[port.name]:
                root = self._failure_above(src, p, seen)
                if root is None:
                    continue
                (at, names), retryable = root[:2], retryable and root[2]
                broken.setdefault((at, tuple(names)), []).append(g.nodes[src].label)
                if (at, tuple(names)) not in reasons and (why := self._root_message(at, p)) is not None:
                    reasons[(at, tuple(names))] = why
        if not broken:
            return None
        # one root cause is said once: lines broken by the same node (the same items) are named together, not one
        # sentence each that all point at it (「FBX 输出设置」「USD 输出设置」这 2 条线上的「导入 FBX」出错了)
        said = []
        for (at, names), chains in broken.items():
            root, item = g.nodes[at].label, " / ".join(names)
            if len(chains) > 1:
                lines = i18n.Both.of(lambda: "".join(i18n.t("engine.quoted", name=c) for c in dict.fromkeys(chains)))  # same-named lines once; the count says how many
                said.append(Msg("I-OUTPUT-CHAINSITEM", chains=lines, count=len(chains), root=root, item=item) if names
                            else Msg("I-OUTPUT-CHAINS", chains=lines, count=len(chains), root=root))
            else:
                said.append(Msg("I-OUTPUT-CHAINITEM", chain=chains[0], root=root, item=item) if names
                            else Msg("I-OUTPUT-CHAIN", chain=chains[0], root=root))
        msg = Msg("E-OUTPUT-CHAINFAILED", node=g.nodes[node_id].label, count=sum(map(len, broken.values())), chains=said,
                  reasons=list(reasons.values()) or [i18n.Word("engine.chain_unknown")])
        return Outcome("failed", node_id, msg.json(), chain=True, retryable=retryable)

    def _failure_above(self, node_id: str, path: ItemPath, seen: set[Inst]) -> tuple[str, list[str], bool] | None:
        """The first failure at or above the instance (the node at its root, the names of its items, whether it is
        retryable), None when there is none: its own outcome, a wire it goes without (dropped: an optional input, a
        gathered item), then every instance it takes something from or waits for."""
        stack = [Inst(node_id, path)]  # depth first, each one's sources in order, without recursion
        while stack:
            inst = stack.pop()
            if inst in seen:
                continue
            seen.add(inst)
            try:
                o = self.outcome(*inst)
                if o is not None and o.failure:
                    return o.root, self._root_names(o.root, inst.path), o.retryable
                for (_port, _src, _sport, p), up in self.dropped(*inst).items():
                    if up.failure:
                        return up.root, self._root_names(up.root, p), up.retryable
                ups = [Inst(src, p) for ws in self.wires(*inst).values() for src, _, p in ws]
                ups += [w.on for w in self.waits(*inst)]
            except PLAN_ERRORS:
                continue
            stack.extend(reversed(ups))
        return None

    def _root_message(self, root: str, path: ItemPath) -> Msg | None:
        """The failed node's own message (its outcome's, as a message again), from a path at or below its instance;
        None when it has none to say (it failed only behind something, or its message can't be said again)."""
        depth = len(self.graph.scopes.chain(root))
        try:
            o = self.outcome(root, tuple(path[:depth]) if depth else ())
        except PLAN_ERRORS:
            return None
        if o is None or o.root != root or not isinstance(o.message, dict):
            return None
        return again(o.message)

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
        if up.blocked:  # behind a 「阻断」 set to block: skipped quietly, said as information only
            return Outcome("skipped", up.root, Msg("I-COOK-BLOCKED", node=node.label, input=input_label,
                                                   root=self.graph.nodes[up.root].label).json(), failure=False, blocked=True)
        return Outcome("skipped", up.root, Msg("N-COOK-SKIPPED", node=node.label, input=input_label,
                                               root=self.graph.nodes[up.root].label).json(), failure=up.failure,
                       retryable=up.retryable)

    def _blocked(self, node_id: str, path: ItemPath) -> Outcome | None:
        """A 「阻断」 (a node type that `blocks`, gate) whose route is known and takes nothing: blocked, quietly (no
        error anywhere: its root is itself). None for any other node, or while its route is not known."""
        node = self.graph.nodes[node_id]
        if not getattr(node.type, "blocks", False) or (taken := self.taken_ports(node_id, path)) is None:
            return None
        if taken - {node.type.condition_input}:
            return None
        return Outcome("skipped", node_id, Msg("I-GATE-BLOCKED", node=node.label).json(), failure=False, blocked=True)

    def _all_blocked(self, node_id: str, path: ItemPath) -> Outcome | None:
        """A node that collects (NodeDef.collects): blocked when every input wired into it gives nothing because of a
        block — each of its wires dropped behind a 「阻断」, or the input not taken at all (a reader with no file, a
        mode that has no use for it), at least one of them blocked. None otherwise."""
        g = self.graph
        wires, dropped = self.wires(node_id, path), self.dropped(node_id, path)
        first = None
        for port in g.input_ports(node_id):
            if port.param or not g.inputs.get((node_id, port.name)):
                continue
            for src, sport, p in wires[port.name]:
                up = dropped.get((port.name, src, sport, p))
                if up is None or not up.blocked:
                    return None
                first = first or (port, up)
        return None if first is None else self._skipped(g.nodes[node_id], _said_label(first[0]), first[1])

    def _behind_missing(self, node_id: str, path: ItemPath) -> Outcome | None:
        """An input the instance takes and cannot go without, every wire of which comes from what surely gives nothing
        (gives_nothing: a reader with no file picked, or a switch / passing 「阻断」 on the way to one): skipped behind
        that reader, so what is said — on this node, on what follows it, on an 「输出」 whose line it breaks — names the
        node that is really missing something (「导入 FBX」没有选择文件), never a wire waiting on the way (THE rule for
        a missing material: the cause is reported at its source, all along the line)."""
        g, node = self.graph, self.graph.nodes[node_id]
        taken = self.taken_ports(node_id, path)
        for port in g.input_ports(node_id):
            if port.param or (taken is not None and port.name not in taken) or not self.requires(node_id, path, port):
                continue
            wires = [(src, g.scopes.source_path(src, node_id, path)) for src, _ in g.inputs.get((node_id, port.name), [])]
            if not wires or not all(at is not None and self.gives_nothing(src, at) for src, at in wires):
                continue
            root = self._missing_at(*wires[0])
            if root is not None and (up := self.outcome(*root)) is not None:
                return self._skipped(node, _said_label(port), up)
        return None

    def _missing_at(self, node_id: str, path: ItemPath) -> tuple[str, ItemPath] | None:
        """The reader with no file a source that surely gives nothing comes down to (through the switches and passing
        gates on the way): gives_nothing's own walk, kept to the first wire."""
        from ..nodes.base import ReadsFile

        g = self.graph
        while True:
            t = g.nodes[node_id].type
            if issubclass(t, ReadsFile):
                return node_id, path
            if not sc.chooses(t) or (taken := self.taken_ports(node_id, path)) is None:
                return None
            ways = [w for w in taken if w != t.condition_input]
            wire = next((w for way in ways for w in g.inputs.get((node_id, way), [])), None)
            if wire is None or (at := g.scopes.source_path(wire[0], node_id, path)) is None:
                return None
            node_id, path = wire[0], at

    def present(self, node_id: str, path: ItemPath = ()) -> bool:
        """Whether something comes in on the input a 「有没有」 (NodeDef.presence_of) tells of (comes)."""
        return self.comes(node_id, self.graph.nodes[node_id].type.presence_of, path)

    def comes(self, node_id: str, port: str, path: ItemPath = (), kind: str = "") -> bool:
        """Whether something comes in on input `port` of the instance, known before anything is cooked: no for a wire
        from an output the source does not have with its parameters (a reader with that kind not picked: Port.when),
        from what surely gives nothing (a reader with no file picked, or a switch / passing gate on the way to one:
        gives_nothing), from what is blocked (behind a 「阻断」 set to block), or from an output on disk that is empty;
        yes for any other wire (what is still to be cooked is taken to come). No wire: no. A 「有没有」's answer
        (present) and a node's refusal before the cook (NodeDef.plan_refusals) ask it. `kind`: only what may be of
        this data type counts (gives_types: a skeleton-only FBX's way through the switches brings no 角色; nothing
        known of the way's types counts as may be)."""
        g = self.graph
        for src, sport in g.inputs.get((node_id, port), []):
            if all(p.name != sport for p in g.outputs(src)):
                continue
            at = g.scopes.source_path(src, node_id, path)
            if at is None or self.gives_nothing(src, at):
                continue
            o = self.outcome(src, at)
            if o is not None and o.blocked:
                continue
            fp = self.on_disk(src, sport, at)
            if fp and (m := self.manifest(fp)) is not None and m["meta"].get("empty"):
                continue
            if kind and (types := self.gives_types(src, sport, at)) and kind not in types:
                continue  # none known on the way (a reader not picked yet): it may be of that type
            return True
        return False

    def gives_types(self, node_id: str, port: str, path: ItemPath = (), _seen: frozenset = frozenset()) -> frozenset[str]:
        """The data types output `port` of the instance may carry on the route its switches take now: a switch gives
        what the ways it takes bring, a port that follows one input (Port.type_from "input:<port>") what that input
        brings, any other its type in the graph (Graph.output_type; each of an open "a|b"). Known before cooking."""
        g = self.graph
        key = (node_id, port, path)
        t = g.nodes[node_id].type
        if key in _seen:
            return frozenset(filter(None, g.output_type(node_id, port).split("|")))
        seen = _seen | {key}
        ways: list[str] | None = None
        if sc.chooses(t):
            taken = self.taken_ports(node_id, path)
            ways = None if taken is None else [w for w in taken if w != t.condition_input]
        else:
            decl = next((p for p in g.outputs(node_id) if p.name == port), None)
            follows = getattr(decl, "type_from", "") if decl is not None else ""
            if follows.startswith("input:") and "#" not in follows and "," not in follows:
                ways = [follows[len("input:"):]]
        if ways is None:
            return frozenset(filter(None, g.output_type(node_id, port).split("|")))
        out: set[str] = set()
        for way in ways:
            for src, sport in g.inputs.get((node_id, way), []):
                if all(p.name != sport for p in g.outputs(src)):
                    continue
                at = g.scopes.source_path(src, node_id, path)
                if at is None or self.gives_nothing(src, at):
                    continue
                out |= self.gives_types(src, sport, at, seen)
        return frozenset(out)

    def quiet_inputs(self, node_id: str, path: ItemPath = ()) -> frozenset[str]:
        """The inputs wired in the graph that bring nothing this time and are not a mistake to tell of: not taken (a
        reader with no file on the way, a mode with no use for them: taken_ports), or every wire of them dropped behind
        a 「阻断」 set to block. A node that would say an input it has a row for is not wired (多层 EXR 输出设置) leaves
        these out silently (CookContext.quiet)."""
        g = self.graph
        wires, dropped = self.wires(node_id, path), self.dropped(node_id, path)
        out = set()
        for port in g.input_ports(node_id):
            if not g.inputs.get((node_id, port.name)):
                continue
            ws = wires[port.name]
            if not ws or all((u := dropped.get((port.name, *w))) is not None and u.blocked for w in ws):
                out.add(port.name)
        return frozenset(out)

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
                if not self.requires(node_id, path, port) or (self._gathers(node_id, port) and len(gone) < len(wires[port.name])):
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
            if up.blocked:  # behind a 「阻断」 set to block: going without it is what was asked for, nothing to say
                continue
            if begin is not None and len(p) > len(path):
                missing.setdefault(p[len(path)], up)
                continue
            reason = (Msg("I-INPUT-FAILED", source=g.nodes[src].label) if up.state != "skipped"
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
        plain = not self.dropped(node_id, path) and not self._wired_choices(node_id, path)
        r = self.graph.resolved(node_id) if plain else resolve(self.graph.nodes[node_id].type, self.facts(node_id, path))
        return self._resolved.put(key, r)

    def _wired_choices(self, node_id: str, path: ItemPath) -> dict[str, Any]:
        """The parameters with a choice (options, a checkbox) driven by a wire whose value is known: name -> that value.
        What the node's licence and cost are judged on (nodes/applies.py may_hold: a value still to be cooked counts as
        the strictest and costliest it may be; one known is the value the node runs with, never the one typed)."""
        from ..data.values import read

        node = self.graph.nodes[node_id]
        specs = {p["name"]: p for p in node.type.param_specs()}
        out = {}
        for name in node.promoted:
            spec = specs.get(name)
            if spec is None or not (spec.get("options") or spec["type"] == "boolean"):
                continue
            wire = self._wired_packet(node_id, name, path)
            if wire is None or isinstance(wire, Pending):
                continue
            try:
                value = read(wire[2]).one()
            except (ValueError, KeyError, TypeError):
                continue
            if spec["type"] == "boolean" or value in (spec.get("options") or ()):
                out[name] = bool(value) if spec["type"] == "boolean" else value
        return out

    def facts(self, node_id: str, path: ItemPath = ()) -> NodeFacts:
        """What the instance's declarations are resolved on (nodes/applies.py): the graph's facts of the node
        (Graph.facts), without the wires it goes without (dropped). `resolved` resolves these; a cook that learns more
        about its node (Engine._say_unused: CookContext.fact) resolves them again with that on top, the same basis."""
        f = self.graph.facts(node_id)
        if choices := self._wired_choices(node_id, path):  # what the wires set them to, now it is known
            f = NodeFacts({**f.params, **choices}, f.wired, f.own, f.incoming, f.wired_out, f.wired_data)
        dropped = self.dropped(node_id, path)
        if not dropped:
            return f
        node = self.graph.nodes[node_id]
        left = {(port, src, sport) for port, wires in self.wires(node_id, path).items() for src, sport, p in wires
                if (port, src, sport, p) not in dropped}
        wired, data, params = {}, {}, dict(f.params)
        for port, types in f.wired.items():
            wires = self.graph.inputs.get((node_id, port), [])
            kept = tuple(t for (src, sport), t in zip(wires, types) if (port, src, sport) in left)
            if kept:
                wired[port] = kept
                data[port] = tuple(d for (src, sport), d in zip(wires, f.wired_data.get(port, ())) if (port, src, sport) in left)
            elif port.startswith(PARAM):
                params[port.removeprefix(PARAM)] = node.params.get(port.removeprefix(PARAM))
        return NodeFacts(params, with_table(node.type, node.params, wired), f.own, f.incoming, f.wired_out,
                         with_table(node.type, node.params, data))

    def lineage(self, node_id: str, path: ItemPath = ()) -> dict[str, list[dict]]:
        """Per input of the instance, per wire in wire order: what the wire brings, as its source's provenance as
        delivered (delivered_provenance: the third-party projects above it and itself that gave something this time,
        upstream first, with their licences; the same account a delivery's sidecar gives) and the source node's id
        and label. For a node that weighs several results of one kind against each other (the ensembles: which model
        each one is, which share a guide, the strictest licence of them all, CookContext.lineage); worked out only
        for a node that asks (NodeDef.reads_lineage), since it walks everything above every wire, and only when it
        cooks (everything above it has then)."""
        g = self.graph
        return {port: [{"id": src, "label": g.nodes[src].label, "node": g.nodes[src].type.id,
                        **self.delivered_provenance(src, at)}
                       for src, _sport, at in wires]
                for port, wires in self.wires(node_id, path).items() if wires}

    def provenance(self, node_id: str, path: ItemPath = ()) -> dict:
        """Which third-party projects produced what the instance takes (it and every node above it), whether all of
        them allow commercial use, and where the values those nodes were given came from (a focal length wired from
        AnyCalib over the camera's): what an output-settings node records next to its files (CookContext.provenance)."""
        key = Inst(node_id, path)
        if key not in self._provenance:
            sources, values = [], []
            scopes = self.graph.scopes
            for nid in self.upstream(node_id, path):  # the routes its switches take: what really went into it
                n = self.graph.nodes[nid]
                if n.type.runtime != "core":  # a third-party project (not the core); `learned` below says whether it runs a model
                    sources.append({"id": nid, "node": n.type.id, "label": n.label, "project": n.type.project.title,
                                    "commercial": self.resolved(nid, at).licence.commercial if (at := scopes.source_path(nid, node_id, path)) is not None
                                    else self.graph.resolved(nid).licence.commercial,
                                    "learned": n.type.learned and not issubclass(n.type, (ImportNode, OutputSettings))})
                at = scopes.source_path(nid, node_id, path)
                if at is not None and (said := self.sources(nid, at)):
                    values.append({"node": n.type.id, "label": n.label, "params": said})
            return self._provenance.put(key, {"sources": sources, "commercial": all(s["commercial"] for s in sources),
                                              **({"values": values} if values else {})})
        return self._provenance[key]

    def delivered_provenance(self, node_id: str, path: ItemPath = ()) -> dict:
        """`provenance` as it stands once everything above the instance has cooked (what an output-settings node
        records and names its files by: OutputSettings.stem, the sidecar): without the projects whose node gave nothing
        this time — a translation with no text to translate (NothingToCook, an empty packet), an instance skipped
        behind a 「阻断」 — so a delivery names only the methods that really made it, and its `commercial` is theirs."""
        whole = self.provenance(node_id, path)
        scopes = self.graph.scopes
        # the same walk as provenance's, one source per node not of the core, in its order
        above = [nid for nid in self.upstream(node_id, path) if self.graph.nodes[nid].type.runtime != "core"]
        gave = [self._gave_something(nid, scopes.source_path(nid, node_id, path)) for nid in above]
        if all(gave) or len(above) != len(whole["sources"]):
            return whole
        sources = [s for s, kept in zip(whole["sources"], gave) if kept]
        return {**whole, "sources": sources, "commercial": all(s["commercial"] for s in sources)}

    def _gave_something(self, node_id: str, at: ItemPath | None) -> bool:
        """Whether the instance gave anything in this cook: it has no outcome (failed, skipped, blocked) and one of its
        outputs is a packet that is not empty (engine/cook.py _give_nothing writes empty ones)."""
        if at is None:
            return True  # not one instance to ask: counted as given (never drops a project on a guess)
        try:
            if self.outcome(node_id, at) is not None:
                return False
            fps = self.plan(node_id, at).outputs.values()
        except PLAN_ERRORS:
            return True
        packets = [p for fp in fps if (p := self.packet(fp)) is not None]
        return not packets or any(not p.meta.get("empty") for p in packets)

    def shot(self, node_id: str, port: str, path: ItemPath = ()) -> dict:
        """What was photographed in the instance's output `port`, as far as the graph can tell before anything is
        cooked (data/contracts.py SHOT_KEYS: the plate's lens state, its pixel aspect, the lens it carries): what the
        node says of itself from its parameters (NodeDef.said_shot: a reader's 镜头状态 and 像素比) with whatever its
        output port takes from its picture input on top (contracts.shot_of), the same rule the engine settles a cooked
        packet by, so a warning that depends on it does not wait for a cook.

        {} for a key nothing says yet. Once the node is cooked its packet says it instead (engine/lint.py reads the
        manifest first)."""
        g = self.graph

        def step(key: tuple) -> tuple[Any, dict, tuple | None]:  # its shape, what it says itself, the output it follows
            node_id, port, path = key
            node = g.nodes[node_id]
            shape = next((p.shape for p in g.outputs(node_id) if p.name == port), KEEPS)
            try:
                said = node.type.said_shot(self.params(node_id, path))
            except (OSError, ValueError, KeyError):  # its parameters are not readable yet: it says nothing
                said = {}
            follows = shape.follows or node.type.picture
            up = next(iter(self.taken(node_id, path).get(follows, ())), None) if follows else None
            return shape, said, up

        # up the picture it follows until one is known, then down again (a chain of any length, no recursion)
        chain, key = [], (node_id, port, path)
        while key is not None and key not in self._shots:
            shape, said, up = step(key)
            chain.append((key, shape, said))
            key = up
        upstream = self._shots[key] if key is not None else {}
        for key, shape, said in reversed(chain):
            upstream = self._shots.put(key, {**said, **shot_of(shape, upstream, said)})
        return self._shots[(node_id, port, path)]

    def warnings(self, node_id: str, path: ItemPath = ()) -> list[dict]:
        """The instance's usage checks (engine/lint.py warnings), worked out once: planning (a B- one refuses it),
        `checks` (the status) and the cook's context (Engine._context) all read this. Remembered like its plan, while it
        waits too: what it waits for being cooked forgets it (Demand.forget)."""
        key = Inst(node_id, path)
        if key in self._warnings:
            return self._warnings[key]
        said = warnings(self, node_id, path)
        return self._warnings.put(key, said)

    def checks(self, node_id: str, path: ItemPath = ()) -> list[dict]:
        """What the instance says before it is cooked: its usage checks (engine/lint.py; a B- one refuses it)
        and what it can foresee from its parameters and what it will cover (NodeDef.foresee: gaps in a sequence)."""
        key = Inst(node_id, path)
        if key not in self._checks:
            said = list(self.warnings(node_id, path))
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
            wired = self.wired_values(node_id, path, inputs=False)  # _set_by_inputs below drops what it can't take
        except ValueError:
            wired = {}
        params = {**self._typed(node_id, path), **{name: one for name, (one, _) in wired.items()}}
        if getattr(self.graph.nodes[node_id].type, "presence_of", ""):  # 「有没有」: its answer, known before cooking
            params["present"] = self.present(node_id, path)
        return {**params, **self._set_by_inputs(node_id, path)} if self.graph.nodes[node_id].type.params_inputs else params

    def _set_by_inputs(self, node_id: str, path: ItemPath, strict: bool = False) -> dict:
        """The parameters the node's value inputs set (NodeDef.params_inputs / params_from_input), from the value wired
        into each once it is known; nothing for one not wired, still pending (the instance waits for it: waits) or
        giving nothing. Held to the node's Params like a wired parameter is (wired_values): values the parameters
        can't take are not set (the typed ones stay); `strict`: say so instead (E-PARAM-INPUTSET), as the status and
        the cook do (wired_values asks it)."""
        from pydantic import ValidationError

        from ..data.values import read

        node = self.graph.nodes[node_id]
        out: dict = {}
        for port in node.type.params_inputs:
            packet = self._input_value(node_id, path, port)
            if packet is None or isinstance(packet, Pending):
                continue
            try:
                value = read(packet).value
            except (OSError, ValueError, KeyError):
                continue
            got = node.type.params_from_input(port, value)
            try:
                node.type.Params(**{**self._typed(node_id, path), **out, **got})
            except ValidationError as exc:
                if strict:
                    where = self._where(*next(iter(self.graph.inputs[(node_id, port)])))
                    raise Invalid(Msg("E-PARAM-INPUTSET", node=node.label, where=where, reason=exc.errors()[0].get("msg", str(exc)))) from None
                continue
            out.update(got)
        return out

    def _input_value(self, node_id: str, path: ItemPath, port: str) -> Packet | Pending | None:
        """What the first wire into a parameter-setting input (NodeDef.params_inputs) gives (wired_input); None: nothing."""
        got = self.wired_input(node_id, port, path, "value")
        return got if got is None or isinstance(got, Pending) else got[2]

    def _wired(self, node_id: str, path: ItemPath = ()) -> dict[str, list[Inst]]:
        """Input port -> the instances wired into it, its declared ones and one per row of a ports_from table that makes
        inputs (「多层 EXR 输出设置」's 图层): what the instance's result covers comes from these; a value driving a
        parameter covers nothing of the picture. Without the wires it goes without (taken)."""
        node = self.graph.nodes[node_id]
        taken = self.taken(node_id, path)
        return {port.name: [Inst(src, p) for src, _, p in taken.get(port.name, [])] for port in node.type.input_ports(node.params)}

    # ------------------------------------------------------------------ parameters driven by wires

    def _wired_packet(self, node_id: str, name: str, path: ItemPath = ()) -> tuple[str, str, Packet] | Pending | None:
        """The wire into a promoted parameter: (node, port, its packet) once known, what it waits for while not; None
        when there is none or it gives nothing (an empty packet counts as not connected: the parameter keeps its own
        value)."""
        return self.wired_input(node_id, PARAM + name, path, "value")

    def _list_values(self, packet: Packet) -> list:
        """The values a list packet holds, in order: what a wire into a table parameter carries (「畸变参数」 driven by
        AnyCalib's 「畸变系数」). One item per row, each an ordinary value packet of its own."""
        from ..data.values import list_values

        return list_values(packet, self.packet)  # self.packet: this evaluation's manifest cache, no second read

    def _where(self, src: str, sport: str) -> str:
        out = next((p for p in self.graph.outputs(src) if p.name == sport), None)
        return i18n.Word("engine.port_of", node=self.graph.nodes[src].label, port=out.label if out else sport)

    def wired_values(self, node_id: str, path: ItemPath = (), inputs: bool = True) -> dict[str, tuple[Any, Any]]:
        """The instance's active parameters driven by a wire whose value is known (the node feeding it is cooked) ->
        (the value the parameter takes, the Value in the parameter's unit: nodes/values.py for_param). Raises
        ValueError saying what is wrong: a unit that doesn't convert, one value per frame that changes into a parameter
        taking one, a value out of the parameter's range; with `inputs`, a value a value input sets (params_inputs) that the
        parameters can't take (E-PARAM-INPUTSET: the status and the cook say it; params() only leaves it out)."""
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
        if inputs and node.type.params_inputs:  # what the value inputs set is held to the same parameters (said, not only dropped)
            self._set_by_inputs(node_id, path, strict=True)
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
        """`source_notes`' full sentences (the footer, the worker's job, the provenance)."""
        return self.source_notes(node_id, path)[0]

    def source_notes(self, node_id: str, path: ItemPath = ()) -> tuple[dict[str, str], dict[str, str]]:
        """Where the instance's parameters get their values, for those that say something about it: driven by a wire
        ("Focal Length 38.6 mm · 来自 AnyCalib 镜头标定"), or set over what a connected input would give (P(overrides=),
        "（覆盖相机的 Focal Length）"), or left to that input ("Focal Length · 来自相机（ViPE 相机解算）"). The node's footer and
        parameter panel show it, its worker's job and the provenance of what it makes keep it. The second table holds,
        for a parameter set over a connected input, that clause on its own (「覆盖相机的 Focal Length」): a field of its
        own, so the page never reads it back out of the sentence."""
        from ..data.values import describe_value, option_label, read, say, unit_problem

        g = self.graph
        node = g.nodes[node_id]
        connected = g.connected(node_id)
        inactive = self.resolved(node_id, path).params.inactive
        params = self.params(node_id, path)
        ports = {p.name: p.label for p in node.type.input_ports(node.params)}
        out: dict[str, str] = {}
        overrides: dict[str, str] = {}
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
                    out[name] = Msg("I-SOURCE-FROMINPUT", label=label, input=ports.get(port, port), node=g.nodes[g.inputs[(node_id, port)][0][0]].label).text
                continue
            over = [p for p in spec["overrides"] if p in connected]
            wire = self._wired_packet(node_id, name, path)
            if wire is not None:
                src = wire.on.node if isinstance(wire, Pending) else wire[0]
                value = ""
                if not isinstance(wire, Pending) and spec["items"] is not None:
                    value = i18n.Both.of(lambda: i18n.t("engine.values", values=i18n.separator().join(describe_value(v) for v in self._list_values(wire[2])))) if wire[2].meta.get("items") else ""
                elif not isinstance(wire, Pending):
                    v = read(wire[2])
                    v = v.in_unit(spec["unit"]) if spec["unit"] and not unit_problem(v.unit, spec["unit"]) else v
                    # a choice parameter is described by its own label (「OpenCV 鱼眼」), not the id sent on the wire
                    # (「opencv_fisheye」); a wired value not among its options is shown as is, and the refusal is
                    # reported separately
                    choice = option_label(spec, v.value) if not v.per_frame else ""
                    value = " " + (choice or describe_value(v))
                text = Msg("I-SOURCE-WIRED", label=label, value=value, node=g.nodes[src].label).text
            elif over and params.get(name) is not None:
                text = Msg("I-SOURCE-TYPED", label=label, value=say(spec, params[name])).text
            elif over:
                src = g.inputs[(node_id, over[0])][0][0]
                out[name] = Msg("I-SOURCE-FROMINPUT", label=label, input=ports[over[0]], node=g.nodes[src].label).text
                continue
            elif spec["assumed"] and params.get(name) is None:
                # an empty parameter is not necessarily ineffective: a parameter declaring `assumed` is still computed
                # with some value when empty, and without stating it in the footer the user cannot tell what the result
                # is based on. Parameters whose name and placeholder already make this clear (「已知 Focal Length」,
                # 「Filmback」, nodes/lens.py) do not declare `assumed`; no parameter currently uses it, and the rule is
                # kept for parameters that would otherwise be unclear.
                out[name] = Msg("I-SOURCE-ASSUMED", label=label, assumed=spec["assumed"]).text
                continue
            else:
                continue
            if over:
                inputs = i18n.separator().join(ports[p] for p in over)
                text += Msg("I-SOURCE-OVERRIDES", inputs=inputs, label=label).text
                overrides[name] = Msg("I-SOURCE-OVERRIDESNOTE", inputs=inputs, label=label).text
            out[name] = text
        return out, overrides

    # ------------------------------------------------------------------ frames

    def _all_frames(self, node_id: str, path: ItemPath = ()) -> Info:
        """NodeDef.info with the frame range not yet applied to this node (its inputs have it). While the instance
        waits for something (waits), what it says holds for what is known: worked out again once it is."""
        key = Inst(node_id, path)
        if key in self._infos:
            return self._infos[key]
        self._sources_first(node_id, path, self._infos.__contains__, self.info)
        if self.source_missing(node_id, path) is not None:  # its file is not there: nothing of it is read
            return self._infos.put(key, Info())
        node = self.graph.nodes[node_id]
        self.graph.check_inputs(node_id, only=self.taken_ports(node_id, path))
        inputs = {port: [self.info(*src) for src in srcs] for port, srcs in self._wired(node_id, path).items()}
        try:
            info = node.type.info(self.params(node_id, path), inputs)
        except (OSError, ValueError) as exc:  # e.g. no file chosen: say which node
            raise CookError(node_id, Msg("E-COOK-FAILED", node=node.label, reason=exc)) from exc
        return self._infos.put(key, info)

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
        """How much the instance processes: the frames its result covers (its own `info`, which says which input its
        frames come from: 「动作重定向」's target and 「线性蒙皮变形」's character only lend a skeleton, often a one-frame rest
        pose, so their frame 0 is not work), or every frame its inputs cover when its result is one still made from a
        shot (a point cloud from a whole sequence); the picture size of what it reads (a source: what it emits). Timing
        records and the progress bar's budget scale by it, and node_done says its frames."""
        ins = [self.info(*src) for srcs in self._wired(node_id, path).values() for src in srcs]
        if not ins:
            return self.info(node_id, path)
        merged, own = Info.merge(ins), self.info(node_id, path)
        return merged if own.still or not own.frames else replace(merged, frames=own.frames)

    def frame_range(self, targets: list[str]) -> tuple[int, int] | None:
        """The frames the inputs of cooking `targets` cover, first and last (several sources: from the earliest
        first to the latest last); None without a frame source. A source that can't be read yet is left out."""
        frames = []
        for nid in self.needed(targets):
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

    def _one_instance(self, node_id: str, path: ItemPath) -> None:
        """The path names one instance of the node: as deep as the blocks around it (engine/scopes.py). Asked about
        otherwise (a node in a block at the top level), E-GRAPH-NONODE: never a plan of an instance that is not there."""
        if len(path) != self.graph.scopes.depth(node_id):
            raise GraphError(Msg("E-GRAPH-NONODE", node=f"{node_id}@{'/'.join(path)}"))

    def full_tier(self, node_id: str, path: ItemPath = ()) -> bool | None:
        """Whether a node that steps down on a smaller card (nodes/applies.py Cost.vram_full_gb) runs its full tier on
        this machine: a card authorized for jobs holds it (nodes/services.py PlanEnv.holds). Its cook then asks for that
        much (vram_need), so it runs only on such a card; the answer is in its fingerprint, so a stepped-down result is
        never reused for a full one. None: the node does not step down."""
        cost = self.resolved(node_id, path).cost
        if not cost.gpu or cost.vram_full_gb <= cost.vram_gb:
            return None
        from ..nodes.services import services

        return services().plan.holds(self.graph.nodes[node_id].type.runtime, cost.vram_full_gb)

    def vram_need(self, node_id: str, path: ItemPath = ()) -> float:
        """The VRAM its cook asks a card for: its full tier's where this machine runs it (full_tier), else its
        resolved cost's; 0 off a GPU."""
        cost = self.resolved(node_id, path).cost
        if not cost.gpu:
            return 0.0
        return cost.vram_full_gb if self.full_tier(node_id, path) else cost.vram_gb

    def plan(self, node_id: str, path: ItemPath = ()) -> NodePlan:
        key = Inst(node_id, path)
        if key in self._plans:
            return self._plans[key]
        self._one_instance(node_id, path)
        if key in self._plan_errors:
            raise _again(self._plan_errors[key])
        self._sources_first(node_id, path, self._planned, self.plan, self.outcome)
        try:
            return self._plan(node_id, path)
        except PLAN_ERRORS as exc:
            self._plan_errors.put(key, exc)
            raise

    def _plan(self, node_id: str, path: ItemPath) -> NodePlan:
        g = self.graph
        node = g.nodes[node_id]
        g.check_inputs(node_id, only=self.taken_ports(node_id, path))
        refused = next((w for w in self.warnings(node_id, path) if w["level"] == "B" and not w.get("refused")), None)
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
        route = None
        if sc.chooses(node.type):  # a switch: which way it takes, never what decided it (a condition that changes and
            # still picks the same way leaves it and everything after it cached); what comes in on that way is in inputs
            cond = node.type.condition_input
            route = sorted((self.taken_ports(node_id, path) or frozenset()) - {cond})
            input_fps.pop(cond, None)
            affecting = affecting - {cond.removeprefix(PARAM)}
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
        if (full := self.full_tier(node_id, path)) is not None:  # a node that steps down on a smaller card: which
            # tier it runs at here is its result's (nodes/applies.py Cost.vram_full_gb)
            blob["tier"] = "full" if full else "stepped"
        if node.type.frame_source:
            blob["frames"] = list(self.info(node_id, path).frames)
        if node.type.named_result:  # the node's name is in what it gives (an import's folder): renamed, cooked again
            blob["name"] = node.id
        if route is not None:
            blob["route"] = route
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
        plan = NodePlan(fp, outputs)
        planned(self, self.account.user_id, [fp, *outputs.values()])  # stale once any of them changes (records.changed)
        return self._plans.put(Inst(node_id, path), plan)

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
            raise _again(self._source_errors[key])
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
        of the instances `plan()` has actually planned so far) still has the identity it saw, every one that was not there
        (or not this account's) still is not, and every packet it found
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
            # a file that was not there (or not this account's, the same answer: _gone): there now, and this evaluation's
            # E-UPLOAD-GONE is stale. Asked again only for those: a file that was there but could not be read is not
            # opened again on every look (an import's worker would run each time), a new version of it comes with a
            # new cache generation
            for inst, _ in self._gone.items():
                node = self.graph.nodes[inst.node]
                try:
                    node.type.source_identity(node.params)
                except (OSError, ValueError, NotFound, CookError):
                    continue
                return False
        # every packet it found an instance cached by is still the one it found (its generation: one stat, not a
        # re-read of anything; Demand.cached kept which)
        return all(generation(fp) == gen for _inst, seen in self._cached_seen.items() for fp, gen in seen.items())
