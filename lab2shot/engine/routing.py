"""Which inputs a switch takes: THE one source (engine/evaluation.py Evaluation is made of this and the other parts:
presence, demand, status). `taken_ports` answers 「is this input in use」; everything that follows a switch's route
reads it: the wires an instance takes (Evaluation.wires), the routed walk up the graph (`upstream`, `needed`), the wires
drawn as unused (`unchosen_wires`) and the packets that stand for an input before it is cooked (`stand_ins`), the inputs
it cannot go without (`requires`), the switch's own wire check (Graph.check_inputs(only=…) and _no_route, which without
an evaluation falls back on NodeDef.chosen_inputs, the same rule _chosen reads) and the switch's cook (it is handed the
packet of the way taken and nothing else). The graph's static type rules for switches (their common type,
`Graph.takes`) stay in engine/graph.py."""

from __future__ import annotations

from . import scopes as sc
from .graph import walk

from .records import WHOLE, Wire
from .scopes import BEGIN, Inst, ItemPath, Pending


class Routing:
    """The routing part of Evaluation (a mixin: every method reads the evaluation's own graph and tables)."""

    def _open_routing(self) -> None:
        self._chosens = self._memo()  # instance -> its switch's inputs, or what that waits for (forget: once known)
        self._upstreams = self._memo(WHOLE)  # (target, path) -> upstream: it follows the routes (WHOLE: with them)

    def _chosen(self, node_id: str, path: ItemPath) -> frozenset[str] | Pending | None:
        """The inputs a switch needs with its condition (None: the node is not a switch, or its condition's wire is
        refused: then it takes no route at all, it is in error for that wire, and no way is drawn as not taken).
        Remembered per instance, Pending too, like what the wire brings (Presence.wired_input)."""
        key = Inst(node_id, path)
        if key in self._chosens:
            return self._chosens[key]
        return self._chosens.put(key, self._choose(node_id, path))

    def _choose(self, node_id: str, path: ItemPath) -> frozenset[str] | Pending | None:
        g, node = self.graph, self.graph.nodes[node_id]
        if not sc.chooses(node.type):
            return None
        cond = node.type.condition_input
        wire = next(iter(g.inputs.get((node_id, cond), [])), None)
        if wire is not None and g.wire_problem(*wire, node_id, cond):
            return None
        got = self.wired_input(node_id, cond, path, "condition")  # none or nothing given: its own 「走哪一路」
        if isinstance(got, Pending):
            return got
        return frozenset(node.type.chosen_inputs(node.params, None if got is None else got[2])) | {cond}

    def taken_ports(self, node_id: str, path: ItemPath = ()) -> frozenset[str] | None:
        """THE answer to 「is this input in use」 where a switch (core.switch) decides it: a switch whose choice is known
        now, the input ports it takes (its condition among them); None for any other node, or while the choice is
        pending (every input may be). Everything that follows a switch's route reads it: the wires an instance takes
        (`wires`), the routed walk up the graph (`upstream`, `needed`: what is behind a pending block, the cook's
        up-front check, provenance, the frame range, the queue's cache marks), the switch's own wire check
        (Graph.check_inputs(only=…)) and the wires drawn as unused (`unchosen_wires`). A path that is not one instance
        of the node (a switch in a block asked about at the top level): None, it has one route per item."""
        if len(path) != self.graph.scopes.depth(node_id):
            return None
        chosen = self._chosen(node_id, path)
        return chosen if isinstance(chosen, frozenset) else None

    def surely_takes(self, node_id: str) -> frozenset[str] | None:
        """The inputs every instance of the node takes whichever route it comes to (the cook's up-front check walks the
        graph by it): taken_ports where the route is known; a switch whose route is not decided yet (its condition
        still to be cooked, or decided per item in a block) surely takes only its condition, what is up its ways is
        checked once the route is known, as that instance's own error (the same rule at the top level and in a block);
        None: all of them."""
        node = self.graph.nodes[node_id]
        if (taken := self.taken_ports(node_id)) is None and sc.chooses(node.type):
            return frozenset({node.type.condition_input})
        return taken

    def requires(self, node_id: str, path: ItemPath, port) -> bool:
        """An input the instance cannot go without (its source failed: it is skipped; it gave nothing: it has nothing to
        cook): a required port, or a switch's input on the route it takes. A switch's ways are optional ports (only
        the one it takes needs a wire), but the one it takes is all it passes on: a failure there is the source's, the
        switch is skipped behind it, never cooked without it. Evaluation.outcome, dropped and Engine._context read it."""
        return not port.optional or port.name in (self.taken_ports(node_id, path) or ())

    def upstream(self, target: str, path: ItemPath = ()) -> list[str]:
        """`target` and what it depends on, dependencies first, as Graph.upstream_order, except that a switch whose
        route is known (taken_ports) leads only up the inputs it takes: an import on a route the switch does not take is
        not needed, so nothing asks for its file, it is not checked before a cook, its project is not in the provenance
        (「载入角色」 with the FBX route chosen is not stopped by the USD import that has no file; a camera from FBX does
        not mark the delivery as ViPE's). `path`: the instance of `target` (each node's route at its own depth)."""
        key = (target, tuple(path))
        if key in self._upstreams:
            return self._upstreams[key]
        depth = self.graph.scopes.depth
        return self._upstreams.put(key, self.graph.upstream_order(target, lambda n: self.taken_ports(n, path[:depth(n)])))

    def needed(self, targets: list[str]) -> list[str]:
        """`targets` and what they depend on, each once, along the routes switches take (upstream): the nodes a cook of
        them may compute, as far as that is known now."""
        return list(dict.fromkeys(n for t in targets for n in self.upstream(t)))

    def unchosen_wires(self) -> frozenset[tuple[str, str, str, str]]:
        """The wires into routes no instance of a switch takes now: drawn as "unused", their problems not reported
        (Graph.wire_states). A switch in a block counts once every instance's route is known (the union of theirs)."""
        g = self.graph
        out = set()
        for node_id in g.nodes:
            if not sc.chooses(g.nodes[node_id].type):
                continue
            paths, pending = self.instances(node_id)
            routes = [self.taken_ports(node_id, p) for p in paths]
            if pending or not routes or any(r is None for r in routes):
                continue
            taken = frozenset().union(*routes)
            for port in g.input_ports(node_id):
                if port.name not in taken:
                    out.update((src, sport, node_id, port.name) for src, sport in g.inputs.get((node_id, port.name), []))
        return frozenset(out)

    def wanted_while_pending(self, node_id: str, path: ItemPath, w: Pending) -> list[Wire]:
        """What the instance may take once what it waits on is known, wanted of its sources already, so a node that
        feeds it writes those outputs in the one cook rather than being run again for them (the rule for both ways
        demand grows as a cook unfolds): a switch whose condition is pending, every way (`_ways`); a block's end whose
        items are pending, every wire from outside the block into a node of it (a top-level node's 「条数」 read only
        inside). Whatever this does not foresee, the cook finds when it is known (Engine._cook_all: an instance it
        cooked already that no longer has every output now wanted is cooked again for the missing ones)."""
        if w.kind == "condition":
            return self._ways(node_id, path)
        g, scopes = self.graph, self.graph.scopes
        begin = scopes.ended.get(node_id)
        if w.kind != "items" or begin is None:
            return []
        members = scopes.scopes[begin].members
        out: list[Wire] = []
        for m in sorted(members):
            for port in g.input_ports(m):
                for src, sport in g.inputs.get((m, port.name), []):
                    depth = scopes.depth(src)
                    if src in members or depth > len(path) or scopes.chain(src) != scopes.chain(m)[:depth]:
                        continue
                    out.append((src, sport, path[:depth]))
        return list(dict.fromkeys(out))

    def _ways(self, node_id: str, path: ItemPath) -> list[Wire]:
        """A switch's wires into every way it may take (not its condition), each with its source instance's path."""
        g, scopes = self.graph, self.graph.scopes
        cond = g.nodes[node_id].type.condition_input
        out: list[Wire] = []
        for port in g.input_ports(node_id):
            if port.name == cond:
                continue
            for src, sport in g.inputs.get((node_id, port.name), []):
                if (at := scopes.source_path(src, node_id, path)) is not None:
                    out.append((src, sport, at))
        return out

    def stand_ins(self, node_id: str, path: ItemPath = ()) -> dict[str, str]:
        """For the input ports a parameter's options come from (P(choices_from), NodeDef.choices): the packet that
        stands for what is wired in (port -> fingerprint), so an option editor opens before the node itself can cook
        (「动作重定向」's 对应关系 before its block has run). THE rule, on the engine's own roles, never a node type id:
        the packet the wire brings once it is on disk; before that, up through what only passes data along — a
        block's begin (its item: the list wired into it), a node that splits data into a list (NodeDef.list_role
        "split": the data wired into it) and a switch (the route it takes: taken_ports). The choices route turns a list
        or a whole into the item the port takes (server/app.py representative). A port with nothing to stand for it
        yet is left out."""
        node = self.graph.nodes[node_id]
        wanted = {n for spec in node.type.param_specs() for n in spec["choices_from"]}
        out = {}
        for port in self.graph.input_ports(node_id):
            if port.name in wanted and (fp := self._stand_in(node_id, port.name, path)):
                out[port.name] = fp
        return out

    def _stand_in(self, node_id: str, port: str, path: ItemPath) -> str | None:
        """The first packet on disk up the wires into `port`, through what only passes data along (stand_ins): depth
        first in wire order, without recursion and without a cap on how far (the graph has no cycle)."""
        g = self.graph

        def into(dst: str, dport: str) -> list[tuple[str, str, str]]:
            return [(src, sport, dst) for src, sport in g.inputs.get((dst, dport), [])]

        stack, seen = into(node_id, port)[::-1], set()
        while stack:
            wire = stack.pop()
            if wire in seen:
                continue
            seen.add(wire)
            src, sport, dst = wire
            t = g.nodes[src].type
            at = g.scopes.source_path(src, dst, path)  # its instance, when the path reaches it
            if at is not None and (fp := self.on_disk(src, sport, at)):
                return fp
            ups: list[str] = []
            if sc.role(t) == BEGIN and sport == t.item_output:
                ups = [t.item_input]
            elif getattr(t, "list_role", "") == "split":
                ups = [p.name for p in g.input_ports(src) if not p.param]
            elif sc.chooses(t) and (taken := self.taken_ports(src, at or ())) is not None:
                ups = [p.name for p in g.input_ports(src) if p.name in taken and p.name != t.condition_input]
            stack += [w for up in ups for w in into(src, up)][::-1]
        return None
