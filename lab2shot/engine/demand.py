"""What cooking some targets requires: the instances in dependency order and what is behind what is pending (`order`),
the ports each instance must give (`demand`), whether it has them (`satisfied`), the nodes a cook runs (`computes`)
and what a click on 「计算」 cooks (`_case`); and forgetting what a cook changed (`forget`). One of the parts
Evaluation is made of (engine/evaluation.py; the others: presence, routing, status)."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import GraphError, message_of
from ..messages import Msg, again
from ..nodes.port import PARAM
from . import scopes as sc
from .graph import walk
from .presence import generation, present
from .records import INSTANCE, KEPT, PLAN_ERRORS, WHOLE, Memo
from .scopes import Inst, ItemPath


@dataclass(frozen=True)
class Step:
    """One instance in the instance graph (Demand._step)."""

    deps: tuple[Inst, ...]  # what it needs cooked first
    takes: tuple[tuple[Inst, str], ...]  # (source instance, its output) for each thing it takes or waits on
    behind: frozenset[Inst]  # what is behind what it waits on (Demand._behind_at: an instance, or a deeper node's prefix)


@dataclass(frozen=True)
class Readiness:
    """What cooking some targets comes to, before it starts (Demand.readiness, given the outputs shown): the one answer
    the queue's submission, the look before it (farm/timings.py), the cook itself (Engine.cook) and a finished job's
    cache mark (farm/queue.py _mark) read, each with the same targets, force and shown (a job's record keeps its show)."""

    refused: Msg | None  # why it is refused as a whole: a B- error of the graph (its wiring, by Routing.surely_takes;
    # the frame range), or no target can give its result (Demand.comes_through: the first error that stands in the
    # way for good; a past cook's failure is tried again, Outcome.retryable, never this); one instance's error among
    # others that would come through is that instance's failure, never this
    failing: frozenset[Inst]  # the instances it fails at without running (Demand.fate FAIL)
    computing: tuple[Inst, ...]  # what it runs, dependencies first (fate COMPUTE or RETRY): instances, then what is
    # behind what is pending (an instance, or a deeper node's prefix: Demand._behind_at)
    cached: tuple[Inst, ...]  # the instances it needs that have everything it wants of them already (fate CACHED)
    skipped: frozenset[Inst] = frozenset()  # the instances skipped behind what fails (fate SKIP): none of them runs


# what cooking does with an instance (Demand.fate): the one classification readiness (and so the submission, the
# look, the policies, the cache mark) and the cook (Engine._begin) read
COMPUTE, CACHED, RETRY, FAIL, SKIP = "compute", "cached", "retry", "fail", "skip"


class Demand:
    """The demand part of Evaluation (a mixin: every method reads the evaluation's own graph and tables)."""

    def _open_demand(self) -> None:
        # the instance graph (_step, _post): what each instance needs first, takes and has behind what it waits on,
        # and its own dependencies-first order; order, demand and each_case all read it
        self._steps: Memo[Inst, Step] = self._memo()
        self._posts: Memo[Inst, list[Inst]] = self._memo()
        self._orders: Memo[tuple[str, ...], tuple[list[Inst], frozenset[Inst]]] = self._memo(WHOLE)
        # the ports a cook wants of each instance (`demand`) and whether they are all on disk (`satisfied`)
        self._demands: Memo[tuple[tuple[str, ...], frozenset[str]], dict[Inst, frozenset[str]]] = self._memo(WHOLE)
        self._satisfied: Memo[tuple[Inst, frozenset[str]], bool] = self._memo(lambda k: k[0])
        # found cached (`cached`): the packets it was found by, which still_true checks are still there
        self._cached_seen: Memo[Inst, dict[str, tuple]] = self._memo()  # packet -> its generation (manifest identity)
        # whether it is 「已算」 (cached), per (instance, shown): once per evaluation, as a packet found there stays
        # (still_true tells a removed or rewritten one by its generation, and the evaluation is built again)
        self._cacheds: Memo[tuple, bool] = self._memo(lambda k: k[0])
        # what the instance's readers need of it (needed_outputs): it follows the routes and what results use
        # (_used), so it goes whenever they may change (WHOLE: with `_orders`)
        self._needs: Memo[Inst, frozenset[str]] = self._memo(WHOLE)

    def order(self, targets: list[str]) -> tuple[list[Inst], frozenset[Inst]]:
        """Cooking `targets` (every instance of each): the instances it needs, dependencies first, as far as is known
        now; and what is behind what is still pending (a block's items, a switch's condition), which it will need once
        that is known: instances (Step.behind), not nodes — a node some of whose instances are in the order may have
        another behind (a block's new item on a way its switch has not chosen yet). Worked out again after every cook
        that changes what is known (forget)."""
        key = tuple(targets)
        if key in self._orders:
            return self._orders[key]
        g = self.graph
        behind: set[Inst] = set()
        starts: list[Inst] = []
        for t in targets:
            if t not in g.nodes:
                raise GraphError(Msg("E-GRAPH-NONODE", node=t))
            paths, pending = self.instances(t)
            for p in pending:
                starts.append(p.on)
                behind.update(self._behind_at(self.upstream(t), p.on.path))
            starts += [Inst(t, p) for p in paths]
        order = self._post_of(starts)
        for inst in order:
            behind |= self._step(inst).behind
        return self._orders.put(key, (order, frozenset(behind - set(order))))

    # ------------------------------------------------------------------ the instance graph

    def _wanting(self, order: list[Inst], behind) -> list[Inst]:
        """The instances whose takes a cook wants: those in its order, and those behind what is pending that already
        are instances (not a deeper node's prefix): what each would take if the pending thing went its way, so a node
        feeding both a switch's condition and, further up, one of its ways gives every output it may be asked for in
        the one cook (the demand is the union over the ways), never cooked a second time once the route is known."""
        depth = self.graph.scopes.depth
        return order + sorted((b for b in behind if depth(b.node) == len(b.path)), key=repr)

    def _behind_at(self, nodes, path: ItemPath) -> set[Inst]:
        """The instances of `nodes` at `path` (cut to each node's depth), or, for a node deeper than `path` (in a block
        whose items are not known), the prefix its instances will have: what stands behind a pending thing there."""
        depth = self.graph.scopes.depth
        return {Inst(n, path[:depth(n)]) if depth(n) <= len(path) else Inst(n, path) for n in nodes}

    def _step(self, inst: Inst) -> Step:
        """The instance in the instance graph: what it needs cooked first (deps), what it takes from each source and
        what it waits on (takes, with what it may take once that is known: Routing.wanted_while_pending), and what is
        behind what it waits on (the nodes up the inputs a pending condition or item list leaves open, at its path).
        Remembered, while it waits too (what it waits for being cooked forgets it: Demand.forget)."""
        if inst in self._steps:
            return self._steps[inst]
        g = self.graph
        takes = [(Inst(src, p), sport) for wires in self.wires(*inst).values() for src, sport, p in wires]
        behind: set[Inst] = set()
        waits = self.waits(*inst)
        for w in waits:
            takes.append((w.on, w.port))
            takes += [(Inst(src, p), sport) for src, sport, p in self.wanted_while_pending(inst.node, inst.path, w)]
            if w.kind != "value":  # a value's node is wired in: planned anyway; the rest is behind this
                cond = g.nodes[inst.node].type.condition_input if hasattr(g.nodes[inst.node].type, "condition_input") else None
                ports = [p for p in g.input_ports(inst.node) if w.kind == "items" or p.name != cond]
                behind |= {b for p in ports for src, _ in g.inputs.get((inst.node, p.name), [])
                           for b in self._behind_at(self.upstream(src, inst.path), inst.path)}
        step = Step(tuple(self.deps(inst)), tuple(takes), frozenset(behind))
        return self._steps.put(inst, step)

    def _post_of(self, starts: list[Inst]) -> list[Inst]:
        """The instances `starts` need, dependencies first, each once: each instance's own order (_posts) worked out
        once, from the far end (walk), and shared by every start and every node's case that reaches it."""
        local: dict[Inst, list[Inst]] = {}  # every post this call works out or reads, fixed for the call
        steps: dict[Inst, Step] = {}  # each instance's step as the walk saw it (a pending one may change meanwhile)

        def known(i: Inst) -> list[Inst] | None:
            if i not in local and i in self._posts:
                local[i] = self._posts[i]
            return local.get(i)

        def deps(x: Inst) -> list[Inst]:
            if known(x) is not None:
                return []
            steps[x] = self._step(x)
            return list(steps[x].deps)

        for i in walk(starts, deps):
            if known(i) is None:
                post = local[i] = list(dict.fromkeys([j for d in steps[i].deps for j in known(d)] + [i]))
                self._posts.put(i, post)  # (a cook changing a step forgets it and the posts after it)
        return list(dict.fromkeys(j for s in starts for j in known(s)))

    def needed_outputs(self, node_id: str, path: ItemPath = ()) -> frozenset[str]:
        """The outputs of the instance something needs: those a wire takes into an input its reader takes along the
        route a switch takes (taken_ports; while the route is pending, every way: the union); every one when nothing
        takes any (what is cooked to be shown or delivered, a sink). What `cached` means, the one rule the status, the
        cook and the uploads read (a port wired only into a way not taken is needed by nobody)."""
        key = Inst(node_id, path)
        if key in self._needs:
            return self._needs[key]
        found = self._needed_outputs(node_id, path)
        if not getattr(self._warming, "used", False):  # asked while _used is worked out: not known yet, not kept
            self._needs.put(key, found)
        return found

    def _needed_outputs(self, node_id: str, path: ItemPath) -> frozenset[str]:
        g = self.graph
        have = frozenset(p.name for p in g.outputs(node_id))
        taken = {sport for dst, dport in g.outputs_by_node.get(node_id, ())
                 for src, sport in g.inputs.get((dst, dport), [])
                 if src == node_id and sport in have and self._takes_input(dst, dport, path)}
        wired = any(src == node_id and sport in have for dst, dport in g.outputs_by_node.get(node_id, ())
                    for src, sport in g.inputs.get((dst, dport), []))
        if taken or not wired:
            return frozenset(taken) if taken else have  # a wire from an output it no longer has asks for nothing of it
        return frozenset()  # wired, but only into ways no switch takes: nothing of it is needed

    def _takes_input(self, node_id: str, port: str, path: ItemPath) -> bool:
        """Whether the instances of `node_id` related to `path` take their input `port`: they are used by a result of the
        graph (Status._used: along the routes switches take, or behind what is pending), and take that input
        (taken_ports: a switch on the route it takes; anything else, and a switch whose route is pending, every input)."""
        depth = self.graph.scopes.depth(node_id)
        used = self._used()  # (the instances some result needs, along the routes switches take; the nodes behind what is pending)
        if depth <= len(path):
            paths, pending = [path[:depth]], []
        else:  # a reader inside a block the instance is outside of: its instances under this path
            try:
                known, pending = self.instances(node_id)
            except PLAN_ERRORS:
                return True
            paths = [p for p in known if p[:len(path)] == path]
        if pending or not paths:
            return True
        return any(((t := self.taken_ports(node_id, p)) is None or port in t)
                   and (used is None or Inst(node_id, p) in used[0] or node_id in used[1]) for p in paths)

    def cached(self, node_id: str, path: ItemPath = (), shown: frozenset[str] = frozenset()) -> bool:
        """Whether every output the instance's readers need (needed_outputs) is on disk now: worked out when asked, as
        the route it depends on may have been decided since the instance was planned. Needed by nobody (wired only
        into ways no switch takes): 「已算」 when its every output is there (cooked on its own), else not (its state
        says unused). `shown`: the
        outputs the viewer shows of it, as 「计算」 would be given them (demand): what it has of those, the same answer
        the readiness of cooking it with them gives."""
        key = (Inst(node_id, path), frozenset(shown))
        if key in self._cacheds:
            return self._cacheds[key]
        try:
            plan = self.plan(node_id, path)
        except PLAN_ERRORS:
            return False
        need = self.asked_of(Inst(node_id, path), shown) & frozenset(plan.outputs)
        found = plan.has(need)
        if found:
            self._cached_seen.put(Inst(node_id, path), {fp: generation(fp) for p in need if (fp := plan.outputs.get(p))})
        return self._cacheds.put(key, found)

    def asked_of(self, inst: Inst, shown: frozenset[str] = frozenset()) -> frozenset[str]:
        """What is asked of the instance, the one rule its 「已算」 (cached) and a click on its 「计算」 (demand: a target's
        own ports) both read: the outputs the viewer shows of it, else what its readers need (needed_outputs), else
        (needed by nobody: a sink, or on a way no switch takes) all of them."""
        outs = frozenset(p.name for p in self.graph.outputs(inst.node))
        return (frozenset(shown) & outs) or (self.needed_outputs(*inst) & outs) or outs

    def demand(self, targets: list[str], shown: frozenset[str] = frozenset()) -> dict[Inst, frozenset[str]]:
        """For cooking `targets`, the ports each instance must give: what the instances the cook wants take of it
        (_wanting: its order, and what is behind what is pending, so a node feeding a condition and a way gives both at
        once), what the targets wait on, the target's own ports (asked_of, with `shown`: the outputs the viewer shows of
        the one target) and the sources of ports made from another of the same
        node (Port.made_from). Contract: the one answer the cook (what to write), `satisfied`,
        readiness and the policies all read (why: a cook writing by one rule while the status judged by another would
        count a node never cached, and cook it again for nothing). Absent from the table: nothing wanted of it."""
        key = (tuple(targets), shown)
        if key in self._demands:
            return self._demands[key]
        order, behind = self.order(list(targets))
        # the target's own ports, what is asked of it (asked_of: shown, else what its readers need, else all), the
        # same its 「已算」 is judged on; without this the target itself would always count as satisfied and never cook
        own = set(targets)
        among = list(self._wanting(order, behind))
        takers = self._takers(among)
        mine = frozenset(shown) if len(targets) == 1 else frozenset()
        extra: dict[Inst, set[str]] = {i: set(self.asked_of(i, mine)) for i in among if i.node in own}
        for t in targets:  # the target itself inside a block whose items are not known yet: the list it waits on
            for p in self.instances(t)[1]:
                extra.setdefault(p.on, set()).add(p.port)
        return self._demands.put(key, {i: self._wants(i, takers, None, extra.get(i, ())) for i in {*takers, *extra}})

    def _takers(self, among) -> dict[Inst, set[Inst]]:
        """Who among these instances takes something from each instance (its wires, what it waits on and may take once
        known: _step): what `_wants` sums, for `demand` and `each_case` alike."""
        takers: dict[Inst, set[Inst]] = {}
        for i in among:
            for src, _ in self._step(i).takes:
                takers.setdefault(src, set()).add(i)
        return takers

    def _wants(self, inst: Inst, takers: dict[Inst, set[Inst]], among: set[Inst] | None, extra=()) -> frozenset[str]:
        """What is wanted of `inst`: the ports its takers (those in `among`; None: all of them) take of it, and `extra`
        (a target's own ports, the list a target waits on), with the ports those are made from (Port.made_from: point
        cloud = depth map + camera). The one sum `demand` and `each_case` both make."""
        ports = {port for t in takers.get(inst, ()) if among is None or t in among
                 for src, port in self._step(t).takes if src == inst} | set(extra)
        outs = {p.name: p for p in self.graph.outputs(inst.node)}
        for name in list(ports):
            ports.update(n for n in getattr(outs.get(name), "made_from", ()) if n in outs)
        return frozenset(ports)

    def satisfied(self, inst: Inst, wanted: frozenset[str]) -> bool:
        """Whether the ports this instance must give (per `demand`) are all on disk: nothing `missing`. A node that
        keeps no outputs (「输出」) never is: it always cooks. An instance that cannot be planned (a file is gone): False;
        the cook will fail there and say why."""
        key = (inst, wanted)
        if key in self._satisfied:
            return self._satisfied[key]
        try:
            plan = self.plan(*inst)
        except PLAN_ERRORS:
            return self._satisfied.put(key, False)
        return self._satisfied.put(key, bool(plan.outputs) and not self.missing(inst.node, plan, wanted))

    def missing(self, node_id: str, plan, wanted: frozenset[str]) -> frozenset[str]:
        """Of the ports wanted of an instance, those it has that are not on disk (presence.present): the one answer
        `satisfied` (nothing missing) and the cook (Engine._to_give: what it writes) read, so a cook never reports as
        cached what the readiness counted as computing. A port it does not have (a wire from an output that is gone) is
        not missing: nothing can write it; a begin's item port is never written (it is the list's packet)."""
        t = self.graph.nodes[node_id].type
        item = t.item_output if sc.role(t) == sc.BEGIN else None
        return frozenset(p for p in wanted if p in plan.outputs and p != item and not present(plan.outputs[p]))

    def computes(self, targets: list[str], force: bool = False) -> list[str]:
        """The nodes cooking `targets` runs, dependencies first: those with an instance whose result is not cached
        (「输出」 always runs: it delivers and keeps nothing), forced targets, and the nodes behind what is still pending
        (they will run once it is known). An instance that can't even be planned (its file is not there) is one of
        them: the cook runs it, it fails at its own node and the rest of the cook goes on; a cook is
        never refused as a whole because one node of it is broken."""
        return self._computing(targets, force)[1]

    def readiness(self, targets: list[str], force: bool = False, shown: frozenset[str] = frozenset()) -> Readiness:
        """Cooking `targets` before it starts (Readiness). Refused as a whole for:
        - a B- wiring error of a node every instance needs (Graph.check_inputs along the routes switches take; where a
          route or the items are not known yet, what every instance surely takes: Routing.surely_takes);
        - a frame range wider than the inputs;
        - no target that can give its result (Demand.comes_through: each of its instances fails for good or is skipped
          behind what does; a past cook's failure is tried again, Outcome.retryable), with the first such error;
        - a fault of the program on the way (a node type's bug): E-FARM-INTERNAL (Status.guarded), never a 500.
        Anything else wrong with one instance is its own failure: the rest is cooked. `shown`: the outputs the viewer
        shows of the one target (demand); submitting, the look and Engine.cook pass the same. An instance computes when
        it lacks a port this cook wants of it (`demand` / `satisfied`), when it is behind what is pending, or when it is
        a target and the cook is forced; the others are cached."""
        def ready() -> Readiness:
            try:
                return self._readiness(targets, force, frozenset(shown))
            except PLAN_ERRORS as exc:  # something on the way can't be planned at all (a node it names is not there)
                return Readiness(message_of(exc), frozenset(), (), ())

        return self.guarded("readiness", ready, lambda said: Readiness(said, frozenset(), (), ()))

    def _readiness(self, targets: list[str], force: bool, shown: frozenset[str]) -> Readiness:
        g, depth = self.graph, self.graph.scopes.depth
        refused = None
        try:
            for nid in g.needed(targets, self.surely_takes):
                if not depth(nid) and self._behind_missing(nid, ()) is not None:  # a wire waiting on a reader with no
                    continue  # file: the reader is the cause, said below as what does not come through (_behind_missing)
                if depth(nid) or (sc.chooses(g.nodes[nid].type) and self.taken_ports(nid) is None):
                    # in a block, or a switch whose route is not decided yet: what holds whichever route each
                    # instance takes (its wires), the rest per instance once it is known (its own error then)
                    g.check_inputs(nid, every_item=True)
                else:  # a switch on the route it takes; any other node, all of it
                    g.check_inputs(nid, only=self.taken_ports(nid))
            self.check_frames(targets)
        except GraphError as exc:
            refused = exc.message
        order, behind = self.order(targets)
        d = self.demand(targets, frozenset(shown))
        computing, cached, failing, skipped = [], [], set(), set()
        for i, f in self._classified(order, d, lambda i: force and i.node in targets).items():
            (computing.append if f in (COMPUTE, RETRY) else cached.append if f == CACHED
             else failing.add if f == FAIL else skipped.add)(i)
        later = {n: k for k, n in enumerate(self.needed(targets))}
        computing += sorted(behind, key=lambda i: (later.get(i.node, len(later)), repr(i)))
        # no target can give its result whatever is cooked (comes_through: every instance of it fails for good or is
        # skipped behind what does; a record a past cook left is tried again, never this): refused too, in the words
        # of the first that fails for good ("读取序列没有选择序列图"), before anything is queued. What the rest of the
        # graph has cooked or not has no say; one bad item or branch among others is not this: the rest is cooked
        if refused is None and not any(self.comes_through(t, retried=True) for t in targets):
            first = next((i for i in order if i in failing), None)
            said = self.outcome(*first).message if first is not None else next(
                o.message for t in targets for o in self._fates(t) if o is not None)
            # made again with its words kept (messages.again: `args`), so it reads in whoever's language asks
            refused = again(said) or Msg(said["code"], **(said.get("params") or {}))
        return Readiness(refused, frozenset(failing), tuple(computing), tuple(cached), frozenset(skipped))

    def fate(self, inst: Inst, wanted: frozenset[str] = frozenset(), force: bool = False, now: bool = False,
             anew: set[Inst] | None = None) -> str:
        """What a cook does with the instance, before it starts (the one classification; readiness and Engine._begin
        read it): FAIL, it fails without running (known now, whatever runs: a planning error, a value wired into a
        parameter it can't take, its file not there, an extension that can't be used, a 「输出」 with a broken line);
        SKIP, it is skipped behind what fails so; RETRY, a past cook's failure (Outcome.retryable) is run again;
        COMPUTE, it runs (it lacks what is wanted of it, `wanted`, or is forced; what is skipped only behind a failure
        that is run again too); CACHED, it has it all. `now`: asked by the cook as the instance's turn comes, everything
        above it through: what stands in its way now is final (a skip or broken line behind a retried failure too).
        `anew` (before a cook, walked dependencies first: _classified): the instances this cook makes anew because of
        a failure it runs again; what goes without one of those (a block's end without a failed item, an optional input)
        or takes from one computes too, the same rule as what is skipped behind it — and is added to it."""
        if (o := self.outcome(*inst)) is not None:
            if o.retryable and not (now and (o.chain or o.state == sc.SKIPPED)):
                fate = RETRY if o.state == sc.FAILED and not o.chain else COMPUTE
            else:
                return SKIP if o.state == sc.SKIPPED else FAIL
        elif anew is not None and (any(up.retryable for up in self.dropped(*inst).values())
                                   or (anew and any(d in anew for d in self.deps(inst)))):
            fate = COMPUTE
        else:
            return COMPUTE if force or not self.satisfied(inst, wanted) else CACHED
        if anew is not None:
            anew.add(inst)
        return fate

    def _classified(self, order: list[Inst], wanted: dict[Inst, frozenset[str]], forced=lambda i: False) -> dict[Inst, str]:
        """Each instance of `order` (dependencies first) and its fate, the one walk readiness and each_case make."""
        anew: set[Inst] = set()
        return {i: self.fate(i, wanted.get(i, frozenset()), forced(i), anew=anew) for i in order}

    def _fates(self, target: str) -> list:
        """The outcome of each instance of `target` (None: nothing stands in its way), and of each list it waits on."""
        paths, pending = self.instances(target)
        return [self.outcome(target, p) for p in paths] + [self.outcome(*w.on) for w in pending]

    def comes_through(self, target: str, retried: bool = False) -> bool:
        """Whether `target` gives its result: an instance of it with nothing in its way (a block's end over the other
        items); a 「输出」 only when no item of it failed (it packs only what is whole; one only outside the frame range
        is not a failure). `retried`: before a cook, a past cook's failure (Outcome.retryable) is not in its way, it is
        tried again. The one answer readiness (refused when no target can) and the cook (Engine._deliver: which
        targets have their result, and so whether another branch came through) read."""
        paths, _ = self.instances(target)
        stops = [o for o in self._fates(target) if o is not None and not (retried and o.retryable)]
        if not stops:
            return True
        if self.graph.nodes[target].type.delivers and any(o.failure for o in stops):
            return False
        return len(stops) < len(paths)

    def _computing(self, targets: list[str], force: bool = False, shown: frozenset[str] = frozenset()) -> tuple[list[str], list[str]]:
        """(what cooking `targets` takes from, along the routes switches take: `needed`; the nodes it computes): the
        nodes of readiness(...).computing, in the order the instances come (dependencies first), then the nodes only
        behind what is pending. `computes` and `_case` give this."""
        uses = self.needed(targets)
        order, _ = self.order(targets)
        computing = {i.node for i in self.readiness(targets, force, shown).computing}
        needed = list(dict.fromkeys([i.node for i in order] + [n for n in uses if n in computing]))
        return uses, [n for n in needed if n in computing]

    def _case(self, targets: list[str], shown: frozenset[str] = frozenset()) -> dict:
        """What cooking `targets` computes, and whether it delivers (collects and packs files: a 「输出」 among them),
        judged on the nodes it computes: those with an instance that does not have every port this cook wants of it
        (`demand` / `satisfied`, the same rule `computes` and the queue apply, never the whole graph's `cached`, which
        would not match what the cook actually writes). "uses": every node the cook takes something from, cached or
        not, along the routes switches take (Evaluation.needed): what the page checks before submitting (a file not
        chosen), instead of walking every wire itself."""
        try:
            uses, computes = self._computing(targets, False, shown)
        except PLAN_ERRORS:  # something on the way can't be planned (its status says why): the graph as drawn
            computes = uses = walk(targets, self.graph.sources_of)
        return {"targets": targets, "computes": computes, "uses": uses,
                "delivers": any(self.graph.nodes[n].type.delivers for n in computes if n in self.graph.nodes)}

    def each_case(self, nodes: list[str]) -> dict[str, dict]:
        """`_case([n])` for each of `nodes` (the status reply's "policy": what a click on 「计算」 cooks), the same
        answers worked out together: every node's order comes from the one instance graph (`order`: _post, _step), and
        what an instance must give is the sum `demand` makes (_takers, _wants), made once over everything that takes
        from it where every taker is in the case, and what runs is the fate readiness reads (_classified), instead of per node (N nodes in a chain: N sums of up to N each); one
        node whose case can't be worked out gets the graph as drawn, as in `_case`."""
        g = self.graph
        node_post: dict[str, list[str]] = {}  # a node's upstream (Routing.upstream at the top level)
        orders: dict[str, tuple[list[Inst], frozenset[Inst], list[ItemPath], list]] = {}
        for nid in nodes:  # every order first: then every instance's takers are known when what they want is summed
            try:
                paths, pending = self.instances(nid)
                orders[nid] = (*self.order([nid]), paths, pending)
            except PLAN_ERRORS:
                pass
        takers = self._takers({i for order, behind, *_ in orders.values() for i in self._wanting(order, behind)})

        def upstream_of(target: str) -> list[str]:
            for n in walk([target], lambda x: [] if x in node_post else g.sources_of(x, self.taken_ports(x))):
                if n not in node_post:
                    ups = g.sources_of(n, self.taken_ports(n))
                    node_post[n] = list(dict.fromkeys([m for u in ups for m in node_post[u]] + [n]))
            return node_post[target]

        whole: dict[Inst, frozenset[str]] = {}  # what every taker wants of it: most instances in a node's order
        out = {}
        for nid in nodes:
            try:
                if nid not in orders:
                    raise GraphError(Msg("E-GRAPH-NONODE", node=nid))  # (its order could not be worked out)
                order, behind_at, paths, pending = orders[nid]
                behind = {i.node for i in behind_at}
                uses = upstream_of(nid)
                among = set(self._wanting(order, behind_at))
                own = {Inst(nid, p) for p in paths}
                ons = {p.on: p.port for p in pending}
                wants = {}
                for i in order:
                    if i in own or i in ons:  # the target's own ports, and the list it waits on: this case alone
                        wants[i] = self._wants(i, takers, among, self.asked_of(i) if i in own else {ons[i]})
                    elif all(t in among for t in takers.get(i, ())):
                        wants[i] = whole[i] if i in whole else whole.setdefault(i, self._wants(i, takers, None))
                    else:
                        wants[i] = self._wants(i, takers, among)
                # what runs: the one classification readiness reads (fate, walked as _classified walks it)
                short = {i.node for i, f in self._classified(order, wants).items() if f in (COMPUTE, RETRY)}
                needed = list(dict.fromkeys([i.node for i in order] + [n for n in uses if n in behind]))
                computes = [n for n in needed if n in behind or n in short]
            except PLAN_ERRORS:  # as _case: something on the way can't be planned, the graph as drawn
                uses = walk([nid], g.sources_of)
                computes = uses
            out[nid] = {"targets": [nid], "computes": computes, "uses": uses,
                        "delivers": any(g.nodes[n].type.delivers for n in computes if n in g.nodes)}
        return out

    def forget(self, inst: Inst, cooked: bool = False) -> None:
        """A cook just changed what is known of this instance: the one way what this evaluation remembers is dropped,
        by instance (every table's index), never a walk of every key.
        - `cooked` (its outputs are there now, its plan and outcome as they were): the instance itself (its `cached`,
          what it satisfies) and the instances its wires lead into (their checks read its packets: lint's Seen); what
          comes after those is planned from fingerprints that did not change, and stays. Another instance of the same
          fingerprint (two readers of one file) is not told: what the cook still has to write is asked of the disk
          (Engine._wanting_more, _to_give: Demand.missing), never of a remembered answer.
        - failed or tried again (its outcome changed): it and every instance downstream of it at a related path (its own
          item and whatever gathers the items; a sibling item does not depend on it), whose outcome and what they go
          without follow it. What a cook takes (`_orders`, `_demands`, the WHOLE tables) stays: no item list and no
          route is known that was not before.
        - cooked, and read before it can be planned (a wired parameter, a switch's condition, a block's items:
          `lookup`): what was pending is known now: everything downstream, and the WHOLE tables too."""
        g = self.graph
        readers = [(dst, dport) for dst, dport in g.outputs_by_node.get(inst.node, ())
                   if any(src == inst.node for src, _ in g.inputs.get((dst, dport), []))]
        known_now = cooked and any(self._read_before_planning(dst, dport) for dst, dport in readers)
        nodes = {dst for dst, _ in readers} if cooked and not known_now else set(walk([inst.node], g.targets_of)) - {inst.node}
        affected = {inst} | {a for node in nodes for a in self._related(node, inst.path)}
        for memo in self._tables:  # every table the evaluation made (Evaluation._memo), by what its keys are about
            if memo.of == WHOLE:
                if known_now:
                    memo.drop_where(lambda k: True)
            elif memo.of is not KEPT:
                memo.drop_about(affected)

    def _related(self, node: str, path: ItemPath) -> set[Inst]:
        """The instances of `node` whose path is related to `path` (one a prefix of the other), with the prefixes that
        stand for what is behind a pending list: what an instance at `path` changing may change of `node`."""
        n, depth = len(path), self.graph.scopes.depth(node)
        out = {Inst(node, path[:j]) for j in range(min(n, depth) + 1)}
        if depth > n:  # into a block it is outside of: every item of it under this path, and their prefixes
            try:
                known, _ = self.instances(node)
            except PLAN_ERRORS:
                known = []
            for p in known:
                if p[:n] == path:
                    out |= {Inst(node, p[:j]) for j in range(n + 1, len(p) + 1)}
        return out

    def _read_before_planning(self, node_id: str, port: str) -> bool:
        """The input is read before its node can be planned (Presence.lookup): a wired parameter or one an input sets
        ("value"), a switch's condition ("condition"), a block's list ("items")."""
        t = self.graph.nodes[node_id].type
        return (port.startswith(PARAM) or port in t.params_inputs or port == getattr(t, "condition_input", None)
                or (sc.role(t) == sc.BEGIN and port == t.item_input))
