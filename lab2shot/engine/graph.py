"""A node graph loaded from its JSON file, validated against the node registry.

A node's inputs in a graph are its type's declared inputs and one more per promoted parameter: a parameter driven by a
wire ("promoted": ["focal_mm"] on the node in the file, its wires ordinary edges into "param:focal_mm"). Everything that
walks a node's inputs goes through Graph.input_ports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..errors import GraphError
from ..messages import Msg
from ..data.types import element_of, is_list, list_of, type_label
from ..nodes import DATA_TYPES, accepts, node_types
from ..nodes.applies import NodeFacts, Resolved, facts_for, output_ports, resolve, waiting_port
from ..nodes.base import NodeDef, Port
from ..nodes.port import PARAM
from ..nodes.output import FILES
from .scopes import Scopes, rules

SCHEMA = "lab2shot.graph/1"
_BAD_NAME = re.compile(r'[\\/:*?"<>|]')


def _takes_files(node: GNode, port: str) -> bool:
    return any(p.name == port and p.type == FILES for p in node.type.input_ports(node.params))


def _common(types: list[str]) -> str:
    """The type every one of these carries: the widest of them that takes all the others (data/types.py accepts),
    "" when they have nothing in common."""
    return next((t for t in types if all(accepts(t, other) for other in types)), "")


def _shown_as_inputs(data_type: str) -> bool:
    """The viewer shows data of this type as what it was made from (nodes/types.py ROLES_2D "inputs": files).

    `data_type` is empty while the port does not exist yet (`Graph.output_type` returns `""` for an unknown port, e.g.
    a wire from an import node before the user has picked in its hierarchy); this counts as not shown as inputs, and
    `""` must not be used to index `DATA_TYPES`, otherwise the KeyError surfaces as a 500 from /api/status."""
    kind = element_of(data_type.split("|")[0])
    got = DATA_TYPES.get(kind)
    return got is not None and got.in_2d == "inputs"


def _list_node(role: str) -> str:
    """The node type that plays `role` between a list and a single (nodes/core/flow.py NodeDef.list_role): "one" takes
    one item out, "make" makes a list of several wires, "split" makes a list of what one piece of data holds (人物框 of
    several people). Found by declaration, like a block's begin and end (engine/scopes.py): the engine names no node."""
    from ..nodes import node_types

    return next((t.id for t in node_types().values() if getattr(t, "list_role", "") == role), "")


@dataclass
class GNode:
    id: str
    type: type[NodeDef]
    label: str
    params: dict[str, Any]
    promoted: tuple[str, ...] = ()  # parameters driven by a wire: each has an input "param:<name>" (NodeDef.param_port)


@dataclass
class Graph:
    nodes: dict[str, GNode]
    # (to_node, to_port) -> [(from_node, from_port), ...] in file order
    inputs: dict[tuple[str, str], list[tuple[str, str]]] = field(default_factory=dict)
    # the frames to cook, first and last: the sources emit only these, everything downstream follows; None: every
    # frame the inputs have (saved in the graph file as "frames": [first, last] or null)
    frames: tuple[int, int] | None = None
    # node -> what is wrong with the wires into it (an output that is gone, a type that no longer fits): raised when
    # that node is planned, so the rest of the graph still plans and shows its state
    wiring: dict[str, list[Msg]] = field(default_factory=dict)
    # node -> the refused wires into it one node put in between would make right (a 蒙皮角色 into a format without
    # skeletons: 「烘焙成模型」; a camera and a cloud into one that holds no clouds: 「按种类取出」): (its input, the node type to
    # insert, the refusal, the button's label: fix_label), offered as one click like the usage checks' fixes (engine/lint.py)
    fixes: dict[str, list[tuple[str, str, Msg, str]]] = field(default_factory=dict)
    # the wires among `wiring` that wait for their output rather than being wrong (NodeDef.waiting: an import node's
    # kind with nothing selected yet, e.g. a template's before its file is picked): (from, its port, to, its port)
    waiting: set[tuple[str, str, str, str]] = field(default_factory=set)
    # indexes built once by from_json, from the final `inputs` (never rebuilt after: a Graph is never edited in
    # place). node -> its input ports that have a wire (Graph.connected, without scanning every edge each call).
    inputs_by_node: dict[str, frozenset[str]] = field(default_factory=dict, repr=False, compare=False)
    # node -> (dst, dport) pairs its outputs feed (Graph.delivered_by, the reverse of `inputs`).
    outputs_by_node: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict, repr=False, compare=False)
    # per-node input_ports(), built the first time it is asked for (GNode.type.inputs is static; only the promoted
    # parameters' Port objects cost anything, param_port() being a linear scan of the spec).
    _input_ports: dict[str, tuple[Port, ...]] = field(default_factory=dict, repr=False, compare=False)
    # scene_kinds(node, port) memo, keyed by the outermost call (see scene_kinds: _seen threads recursion, the memo
    # only remembers a call that started the walk, never a partial one truncated by a cycle guard mid-walk).
    _scene_kinds: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict, repr=False, compare=False)
    # resolved(node) memo: what the node's declarations say with its parameters and wires (nodes/applies.py)
    _resolved: dict[str, Resolved] = field(default_factory=dict, repr=False, compare=False)
    # the 逐项处理 blocks: their members, every node's chain, the wires their rules refuse (engine/scopes.py rules),
    # worked out once by from_json from the final wires
    scopes: Scopes = field(default_factory=Scopes, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: dict) -> Graph:
        if data.get("schema") != SCHEMA:
            raise GraphError(Msg("E-GRAPH-NOTGRAPH"))
        frames = data.get("frames")
        if frames is not None and not (isinstance(frames, list) and len(frames) == 2
                                       and all(isinstance(f, int) and not isinstance(f, bool) for f in frames) and frames[0] <= frames[1]):
            raise GraphError(Msg("E-GRAPH-FRAMES", frames=str(frames)))
        registry = node_types()
        nodes: dict[str, GNode] = {}
        for n in data["nodes"]:
            node_type = registry.get(n["type"])
            if node_type is None:
                from ..nodes.registry import why_missing

                raise GraphError(Msg("E-GRAPH-UNKNOWNTYPE", type=n["type"], reason=why_missing(n["type"])))
            try:
                params = node_type.load_params(n.get("params", {}))
                # the node type's permanent wired ports (NodeDef.wired_ports) combined with the parameters this
                # instance promoted to the node, so every path below that follows promoted (input ports, wired values,
                # evaluation cache keys) recognises both
                promoted = tuple(dict.fromkeys(node_type.wired_ports + tuple(n.get("promoted", ()))))
                for name in promoted:
                    node_type.param_port(name)  # a parameter it has, one a wire can drive
            except ValueError as exc:  # unknown or invalid parameters (pydantic's ValidationError is a ValueError)
                raise GraphError(Msg("E-GRAPH-PARAMS", node=n["id"], reason=exc)) from exc
            if len(set(promoted)) != len(promoted):
                raise GraphError(Msg("E-GRAPH-PROMOTEDTWICE", node=n["id"]))
            nodes[n["id"]] = GNode(n["id"], node_type, n.get("label") or node_type.label, params, promoted)
        g = cls(nodes, {}, tuple(frames) if frames else None)
        for e in data["edges"]:
            (src, sport), (dst, dport) = e["from"], e["to"]
            g._connect(src, sport, dst, dport)
        g._index()
        g.scopes = rules(g)
        g._check_wires()
        return g

    def _index(self) -> None:
        """Build inputs_by_node / outputs_by_node from the final `inputs` (called once, after every wire is known)."""
        by_dst: dict[str, set[str]] = {}
        by_src: dict[str, list[tuple[str, str]]] = {}
        for (dst, dport), wires in self.inputs.items():
            if wires:
                by_dst.setdefault(dst, set()).add(dport)
            for src, sport in wires:
                by_src.setdefault(src, []).append((dst, dport))
        self.inputs_by_node = {k: frozenset(v) for k, v in by_dst.items()}
        self.outputs_by_node = {k: tuple(dict.fromkeys(v)) for k, v in by_src.items()}

    def input_ports(self, node_id: str) -> tuple[Port, ...]:
        """The node's inputs in this graph: its type's, then one per promoted parameter (in the order promoted),
        cached (a promoted parameter's Port comes from a linear scan of the spec, param_port)."""
        cached = self._input_ports.get(node_id)
        if cached is None:
            n = self.nodes[node_id]
            cached = self._input_ports[node_id] = n.type.input_ports(n.params) + tuple(
                n.type.param_port(name) for name in n.promoted if name not in n.type.wired_ports)  # the permanent ones are already in input_ports
        return cached

    def input_port(self, node_id: str, port: str) -> Port | None:
        return next((p for p in self.input_ports(node_id) if p.name == port), None)

    def wired_params(self, node_id: str) -> dict[str, tuple[str, str]]:
        """The node's promoted parameters that have a wire -> the output it comes from (node, port)."""
        return {name: wires[0] for name in self.nodes[node_id].promoted if (wires := self.inputs.get((node_id, PARAM + name)))}

    def _connect(self, src: str, sport: str, dst: str, dport: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise GraphError(Msg("E-GRAPH-NOSUCHWIRENODE", source=src, target=dst))
        inp = self.input_port(dst, dport)
        if inp is None:
            code = "E-GRAPH-NOTPROMOTED" if dport.startswith(PARAM) else "E-GRAPH-NOSUCHPORT"
            raise GraphError(Msg(code, source=src, output=sport, target=dst, input=dport))
        wires = self.inputs.setdefault((dst, dport), [])
        if wires and not inp.multi:
            raise GraphError(Msg("B-GRAPH-ONEWIRE", node=self.nodes[dst].label, input=inp.label))
        wires.append((src, sport))

    def outputs(self, node_id: str) -> tuple[Port, ...]:
        """The node's outputs with its parameters (nodes/applies.py output_ports: 读取序列 has one per layer it lists)."""
        n = self.nodes[node_id]
        return output_ports(n.type, n.params)

    def facts(self, node_id: str) -> NodeFacts:
        """What the graph knows of a node for its declarations (nodes/applies.py): its parameters (those a wire drives
        as WIRED), the types wired into each input, the facts it knows from its parameters."""
        n = self.nodes[node_id]
        wired = {}
        for port in self.input_ports(node_id):
            wires = self.inputs.get((node_id, port.name), [])
            if wires:
                wired[port.name] = tuple(self.output_type(s, sp) for s, sp in wires)
        if n.type.ports_from and n.type.ports_from_side == "inputs":  # the table's rows together, under the table's name
            rows = {p.name for p in n.type.input_ports(n.params)} - {p.name for p in n.type.inputs}
            if together := tuple(x for name in sorted(rows) for x in wired.get(name, ())):
                wired[n.type.ports_from] = together
        # which of its own outputs go anywhere (a parameter that only shapes one of them: WiredOut)
        out = {sport for wires in self.inputs.values() for src, sport in wires if src == node_id}
        return facts_for(n.type, n.params, wired, tuple(self.wired_params(node_id)), out, self._incoming(node_id))

    def _incoming(self, node_id: str) -> dict[str, dict]:
        """Each input port of this node -> the facts the node wired into it declares about what it gives
        (NodeDef.facts of the source; nodes/applies.py Incoming). Several wires into one port: only facts every one
        of them agrees on, so a condition never reads one of two sources. Empty where nothing is wired or the source
        cannot tell yet; Incoming then says 'not known' and leaves the subject usable."""
        got: dict[str, dict] = {}
        for port, wires in self.inputs.items():
            if port[0] != node_id or not wires:
                continue
            each = [self.nodes[src].type.facts(dict(self.nodes[src].params)) for src, _ in wires if src in self.nodes]
            if not each:
                continue
            shared = {k: v for k, v in each[0].items()
                      if all(k in other and other[k].value == v.value for other in each[1:])}
            if shared:
                got[port[1]] = shared
        return got

    def resolved(self, node_id: str) -> Resolved:
        """The node's declarations resolved in this graph (nodes/applies.py resolve): its outputs, the parameters that do
        nothing here and why, its cost and licence. Remembered: a Graph is never edited in place."""
        got = self._resolved.get(node_id)
        if got is None:
            got = self._resolved[node_id] = resolve(self.nodes[node_id].type, self.facts(node_id))
        return got

    def output_unit(self, node_id: str, port: str) -> str:
        """The unit a value output gives: its port's, or the one its node's parameter says ("param:unit": the constant
        nodes)."""
        out = next(p for p in self.outputs(node_id) if p.name == port)
        if out.unit.startswith(PARAM):
            return str(self.nodes[node_id].params.get(out.unit.removeprefix(PARAM)) or "")
        return out.unit

    def output_type(self, node_id: str, port: str, _seen: frozenset[str] = frozenset()) -> str:
        """The data type an output carries in this graph: its port's type or, for a port that follows its inputs
        (Port.type_from), what they carry; open ("image|map") while nothing is. Three ways to follow, the way the
        port writes it:
        - "input:<port>": what is wired into that input (「LensDistortion」's 镜头 a camera);
        - "input:<port>#item": one layer of list off it (「逐项开始」's 条目: an 图像序列[] gives 图像序列);
        - "input:<port>#list": the list of it (「逐项结束」's 列表, 「合成列表」's; what already is a list stays one,
          and an inner block's list is flattened);
        - "input:<port>,<port>...#common": the type every one of them carries (「切换」's branches; none in common: the
          port's own type, and the wires say so: B-SWITCH-TYPES).

        Returns an empty string (rather than raising) when the node has no such port now: some ports appear with
        parameters (「导入 USD」 has a port only after a model is picked in its hierarchy), while wires in the graph may
        exist first. A wire to a port that does not exist yet means the type is not known yet, not a server fault;
        raising would turn /api/status into a 500 and the graph could not be opened. The empty string is how this
        chain expresses a missing port (wire_problem and describe read it that way)."""
        out = next((p for p in self.outputs(node_id) if p.name == port), None)
        if out is None:
            return ""
        if not out.type_from or node_id in _seen:
            return out.type
        named, _, how = out.type_from.removeprefix("input:").partition("#")
        got = self._carried(node_id, named.split(","), _seen | {node_id}, first=how != "common")
        if not got:
            return out.type
        if how == "item":
            return element_of(got[0])
        if how == "list":
            return list_of(got[0])
        if how == "common":
            return _common(got) or out.type
        return got[0]

    def _carried(self, node_id: str, ports: list[str], _seen: frozenset[str], first: bool) -> list[str]:
        """What is wired into these inputs carries (`first`: only the first wire of each)."""
        got = []
        for name in ports:
            for src, sport in self.inputs.get((node_id, name), []):
                if any(o.name == sport for o in self.outputs(src)):
                    got.append(self.output_type(src, sport, _seen))
                    if first:
                        break
        return got

    def scene_kinds(self, node_id: str, port: str, _seen: frozenset[str] = frozenset()) -> frozenset[str]:
        """The kinds of 3D data an output carries in this graph (types.SCENE_KINDS, and types.DEFORMING for a 模型
        that deforms), known before anything is cooked: what its node knows from its parameters (NodeDef.output_kinds:
        an import node, the kinds selected); for a port that follows an input (type_from), what that input carries;
        a single-kind type, its kind and the port's `kinds`; the umbrella 场景, what the node's 3D inputs carry (「USD
        打包」: all of them together; 3D nodes that pass their input on the same). Empty: not 3D data, or nothing known.
        The one derivation the wiring check, the output checks and the viewer use (the editor reads it from the status reply, Graph.ports; was store.ts
        sceneKinds). Memoized by (node, port): only at the outermost call (`_seen` empty), never for a call still
        inside another's recursion: a cycle guard mid-walk (`node_id in _seen`) gives a truncated answer that is
        only valid at that point in that particular walk, not a fact about the node worth remembering."""
        if not _seen and (cached := self._scene_kinds.get((node_id, port))) is not None:
            return cached
        result = self._scene_kinds_of(node_id, port, _seen)
        if not _seen:
            self._scene_kinds[(node_id, port)] = result
        return result

    def _scene_kinds_of(self, node_id: str, port: str, _seen: frozenset[str]) -> frozenset[str]:
        from ..data.types import kind_of

        n = self.nodes[node_id]
        out = next((p for p in self.outputs(node_id) if p.name == port), None)
        # the type it carries in this graph (a port that follows an input: what is wired in, 「LensDistortion」's 镜头 a camera)
        if out is None or node_id in _seen or not self.output_type(node_id, port).startswith("scene"):
            return frozenset()
        known = self.resolved(node_id).kinds.get(port)  # what the node knows (Port.kinds_from)
        if known is not None:
            return known
        seen = _seen | {node_id}
        if out.narrows:  # its own kind of what the input carries (Port.narrows: 「按种类取出」)
            kind = kind_of(out.type)
            carried = frozenset().union(*(self.scene_kinds(src, sport, seen) for src, sport in self.inputs.get((node_id, out.narrows.removeprefix("input:")), [])
                                          if any(o.name == sport for o in self.outputs(src))))
            return frozenset(k for k in carried if k.split(".")[0] == kind)
        if out.type_from:  # what those inputs carry, however the port follows them (#item, #list, #common)
            ports = out.type_from.removeprefix("input:").partition("#")[0].split(",")
        elif kind := kind_of(out.type):
            return frozenset({kind, *out.kinds})
        else:
            ports = [p.name for p in n.type.input_ports(n.params) if p.type.startswith("scene")]
        return frozenset().union(*(self.scene_kinds(src, sport, seen) for p in ports for src, sport in self.inputs.get((node_id, p), [])
                                   if any(o.name == sport for o in self.outputs(src))))

    # ------------------------------------------------------------------ what the editor reads (status reply)

    def ports(self, node_id: str) -> dict:
        """The node's ports in this graph as the editor draws them: its inputs (promoted parameters' too), its outputs
        with the type, unit and 3D kinds they carry here, and the declared outputs that wait for a parameter (with what
        brings them). The page never works any of this out itself (webui/src/graph/rules.ts reads it)."""
        from ..data.types import KIND_ORDER

        outputs = [{**p.describe(), "type": (t := self.output_type(node_id, p.name)), "unit": self.output_unit(node_id, p.name),
                    "type_label": type_label(t),  # the type it carries here, in the artist's words (the page never spells one)
                    "tip": p.tip(t),  # what it says here: that same type, not the alternatives it declares
                    "list": is_list(t),  # a list here and now (a port that follows a list input: 「逐项结束」)
                    "kinds": [k for k in KIND_ORDER if k in self.scene_kinds(node_id, p.name)]} for p in self.outputs(node_id)]
        # every port of a graph says what the pointer says over it (nodes/base.py Port.tip): the catalogue carries none
        # of it, so this is the one place a port's sentence comes from
        # applicability of input ports (Port.applies): ports not applicable now are sent to the page with the reason
        # to be shown disabled (the same algorithm as for parameters)
        from ..availability import resolve as _resolve

        t = self.nodes[node_id].type
        facts = self.facts(node_id)
        conds = {p.name: c for p in t.inputs if (c := t.port_applies(p)) is not None}
        off = _resolve(conds, facts, t).inactive
        # output ports have applicability too (with a distortion-free 「拟合模型」, 「镜头模型」 and 「畸变系数」 of COLMAP /
        # AnyCalib / GeoCalib are disabled, nodes/lens.py only_when_distorting): the port stays in place, disabled,
        # with the reason on hover, and cannot be wired. It is computed in nodes/applies.py resolve()
        # (`Resolved.out_inactive`) and only read here, not resolved again
        out_off = self.resolved(node_id).out_inactive
        outputs = [{**o, **({"inactive": out_off[o["name"]].json()} if o["name"] in out_off else {})} for o in outputs]
        # `.json()` is required: `Msg.text` is a property, not a field, so serialising the object directly yields only
        # code and params; the page would read `p.inactive.text` as `undefined` and print "undefined" in the port hover
        return {"inputs": [{**p.describe(), "tip": p.tip(), **({"inactive": off[p.name].json()} if p.name in off else {})}
                           for p in self.input_ports(node_id)], "outputs": outputs,
                "waiting": [{**p.describe(), "tip": p.tip(), "kinds": []} for p in self.resolved(node_id).waiting]}

    def handles(self, node_id: str) -> list[int]:
        """The node's viewer handles that apply with its parameters and wires now (Handle.when), by their index."""
        from ..availability import AVAILABLE, level

        facts = self.facts(node_id)
        return [i for i, h in enumerate(self.nodes[node_id].type.handles) if level(h.when, facts) is AVAILABLE]

    def wire_states(self) -> list[dict]:
        """Every wire as the editor draws it: the type it carries, "ok", "waiting" for its output (an import node's kind
        not selected yet) or "wrong" with the problem (a message) and the node type that puts it right ("" none)."""
        out = []
        for (dst, dport), wires in self.inputs.items():
            for src, sport in wires:
                problem = self.wire_problem(src, sport, dst, dport)
                state = "ok" if problem is None else "waiting" if (src, sport, dst, dport) in self.waiting else "wrong"
                out.append({"from": [src, sport], "to": [dst, dport], "type": self.output_type(src, sport),
                            "state": state, "problem": problem.json() if problem is not None else None,
                            "fix": self.fix_for(src, sport, dst, dport) if problem is not None else ""})
        return out

    @classmethod
    def at_defaults(cls, type_id: str) -> dict:
        """A node type's ports and handles on its own at its default parameters, nothing wired: what the editor shows
        for a node just added, before the status reply for it arrives (the catalogue's `at_defaults`).

        Without what the pointer says over each port (`tip`): that sentence repeats every type's description on every
        port, which on 122 node types is 34 KB of the catalogue every first load. It travels with a
        graph's own ports instead (`ports`), and a node just added says its own description for the moment before its
        status arrives."""
        g = cls.from_json({"schema": SCHEMA, "nodes": [{"id": "n", "type": type_id, "params": {}}], "edges": []})
        ports = {side: [{k: v for k, v in p.items() if k != "tip"} for p in row] for side, row in g.ports("n").items()}
        return {"ports": ports, "handles": g.handles("n")}

    def _check_wires(self) -> None:
        """Every wire against the outputs as the parameters make them and the types they carry now, once all wires are
        known (a type can follow an input wired later in the file). A wire from an output that is gone (a layer taken
        out of 读取序列's list), whose type no longer fits (the layer taken as another kind) or that carries a kind of
        3D data the node it goes to refuses (NodeDef.refuses: a format that can't hold it) or a value in a unit the
        parameter it drives can't take (px into mm) is the problem of the node it feeds."""
        for (dst, dport), wires in self.inputs.items():
            for src, sport in wires:
                if problem := self.wire_problem(src, sport, dst, dport):
                    self.wiring.setdefault(dst, []).append(problem)
                    if not any(p.name == sport for p in self.outputs(src)) and waiting_port(self.nodes[src].type, sport, self.nodes[src].params):
                        self.waiting.add((src, sport, dst, dport))
                    if via := self.fix_for(src, sport, dst, dport):
                        self.fixes.setdefault(dst, []).append((dport, via, problem, self.fix_label(src, sport, dst, via)))

    def fix_for(self, src: str, sport: str, dst: str, dport: str) -> str:
        """The node type that, put between them, makes a wrong wire right, when it takes what the wire carries: a
        conversion into what the input takes (NodeDef.converts: 「置信度转遮罩」 for a 置信度 into a mask's input), or
        what the node names for data it refuses (NodeDef.refusal_fix); "" none."""
        from ..nodes import node_types
        from ..nodes.registry import converter

        out = next((p for p in self.outputs(src) if p.name == sport), None)
        if (src, sport, dst, dport) in self.scopes.problems and self._fits(src, sport, dst, dport):
            return self.scopes.problems[(src, sport, dst, dport)][1]  # the rules of a block: an end put in between
        if out is None or self.input_port(dst, dport).param:
            return ""
        t = self.output_type(src, sport)
        if self._list_problem(src, sport, dst, dport) is not None:  # a list where a single is meant, or the reverse
            if is_list(t):
                return _list_node("one")  # 「取一条」 (「逐项开始」 is the other way: it opens a block, so it is not one click)
            from ..data.items import kind_of as items_of

            return _list_node("split" if items_of(element_of(t.split("|")[0])) is not None else "make")
        if not accepts(inp := self.input_port(dst, dport).type, t):
            return converter(t, inp)
        via = node_types().get(self.nodes[dst].type.refusal_fix(t, self.scene_kinds(src, sport)))
        return via.id if via and any(accepts(p.type, t) for p in via.inputs) else ""

    def fix_label(self, src: str, sport: str, dst: str, via: str) -> str:
        """The one click's button: the node inserted, and the kind of 3D data it lets through when it names one
        (「插入「按种类取出」（相机）」: NodeDef.fix_kind)."""
        from ..data.types import kind_label
        from ..nodes import node_types

        kind = self.nodes[dst].type.fix_kind(self.output_type(src, sport), self.scene_kinds(src, sport))
        return f"插入「{node_types()[via].label}」" + (f"（{kind_label(kind)}）" if kind else "")

    def wire_problem(self, src: str, sport: str, dst: str, dport: str) -> Msg | None:
        """What is wrong with one wire (None nothing): see _check_wires."""
        from ..data.values import unit_problem

        kind = type_label  # 「图像序列」, 「图像序列列表」, 「深度图或遮罩」 (data/types.py)
        inp = self.input_port(dst, dport)
        to = {"node": self.nodes[dst].label, "input": inp.label}
        source = self.nodes[src].label
        # the input port is not applicable now (Port.applies: with 「图像」 wired, 「序列图输出设置」 takes no individual
        # channels, and vice versa). It is disabled on the node and the page does not allow the wire; the server
        # refuses it as well, preventing submissions that bypass the page
        cond = self.nodes[dst].type.port_applies(inp)
        if cond is not None:
            from ..availability import resolve as _resolve

            off = _resolve({dport: cond}, self.facts(dst), self.nodes[dst].type).inactive
            if dport in off:
                return Msg("B-WIRE-INACTIVE", why=off[dport], **to)
        out = next((p for p in self.outputs(src) if p.name == sport), None)
        # the source output port is not applicable now (Port.applies, e.g. a solver's 「相机」 only passes through once
        # a camera is wired): disabled on the node, not allowed by the page, and refused by the server as well
        # (preventing submissions that bypass the page)
        if out is not None and out.applies is not None:
            from ..availability import resolve as _resolve_out

            gone = _resolve_out({sport: out.applies}, self.facts(src), self.nodes[src].type).inactive
            if sport in gone:
                return Msg("B-WIRE-OUTINACTIVE", source=source, output=out.label, why=gone[sport])
        if out is None and (waits := waiting_port(self.nodes[src].type, sport, self.nodes[src].params)):
            return Msg("B-WIRE-WAITS", source=source, what=waits.label, waits=waits.waits, **to)
        if out is None:
            return Msg("B-WIRE-GONE", source=source, output=sport, **to)
        if (said := self._list_problem(src, sport, dst, dport)) is not None:
            return said
        if not accepts(inp.type, t := self.output_type(src, sport)):
            from ..nodes.registry import converter

            if not inp.param and (via := converter(t, inp.type)):  # a conversion exists: never done on the quiet
                return Msg("B-WIRE-CONVERT", source=source, output=out.label, got=kind(t), want=kind(inp.type),
                           via=node_types()[via].label, **to)
            return Msg("B-WIRE-TYPE", source=source, output=out.label, got=kind(t), want=kind(inp.type), **to)
        if not inp.param and (why := self.nodes[dst].type.refuses(t, self.scene_kinds(src, sport))):  # a parameter's input takes its value
            return Msg("B-WIRE-REFUSED", source=source, output=out.label, got=kind(t), node=to["node"], reason=why)
        if not inp.param and (why := self.nodes[dst].type.param_refuses(t, self.nodes[dst].params)):
            return Msg("B-WIRE-REFUSED", source=source, output=out.label, got=kind(t), node=to["node"], reason=why)
        if not inp.param and (other := self._uncommon(dst, dport, t)):  # 「切换」: every branch the same kind of data
            return Msg("B-SWITCH-TYPES", node=to["node"], input=inp.label, got=kind(t), other=kind(other))
        # incompatible units cannot be wired, for parameter ports and data ports alike. Checking only `inp.param` would
        # let a data port declaring a unit (「Focal Length（px）」 / 「Focal Length（mm）」 of 「Focal Length 换算」) accept a
        # millimetre value and fail only midway through the cook.
        if inp.unit and (why := unit_problem(self.output_unit(src, sport), inp.unit)):
            return Msg("B-WIRE-UNIT", source=source, output=out.label, reason=why, **to)
        if (src, sport, dst, dport) in self.scopes.problems:  # the rules of a block (engine/scopes.py)
            return self.scopes.problems[(src, sport, dst, dport)][0]
        return None

    def _list_problem(self, src: str, sport: str, dst: str, dport: str) -> Msg | None:
        """A list where a single is meant, or the other way round (None: neither). The items never go through
        on the quiet: 「合成列表」 or 「拆成列表」 makes the list, 「取一条」 or 「逐项开始」 takes the items out, the one
        click on the wire (fix_for). A list into a node that would make a list of it is one level too deep
        (B-LIST-NESTED): the type system has one level."""
        inp = self.input_port(dst, dport)
        out = next((p for p in self.outputs(src) if p.name == sport), None)
        if out is None:
            return None
        t = self.output_type(src, sport)
        other = element_of(t) if is_list(t) else list_of(t)
        if accepts(inp.type, t) or not accepts(inp.type, other):  # it fits, or it does not fit either way (a wrong type)
            return None
        to = {"node": self.nodes[dst].label, "input": inp.label}
        if is_list(t) and self._makes_list_of(dst, dport):
            return Msg("B-LIST-NESTED", source=self.nodes[src].label, output=out.label, got=type_label(t), **to)
        return Msg("B-WIRE-LIST", source=self.nodes[src].label, output=out.label, got=type_label(t),
                   want=type_label(inp.type), **to)

    def _makes_list_of(self, node_id: str, port: str) -> bool:
        """The node makes a list of what this input carries (an output following it with "#list")."""
        return any(out.type_from.endswith("#list") and port in out.type_from.removeprefix("input:").partition("#")[0].split(",")
                   for out in self.outputs(node_id))

    def _uncommon(self, node_id: str, port: str, carried: str) -> str:
        """For an input a "common:" output follows (「切换」's branches): the type another of them carries that this one
        has nothing in common with, "" none."""
        for out in self.outputs(node_id):
            named, _, how = out.type_from.removeprefix("input:").partition("#")
            if how != "common" or port not in named.split(","):
                continue
            for other in self._carried(node_id, [p for p in named.split(",") if p != port], frozenset({node_id}), first=False):
                if not _common([carried, other]):
                    return other
        return ""

    def _fits(self, src: str, sport: str, dst: str, dport: str) -> bool:
        """The wire is wrong only by the rules of a block: its output is there and its type fits."""
        out = next((p for p in self.outputs(src) if p.name == sport), None)
        return out is not None and accepts(self.input_port(dst, dport).type, self.output_type(src, sport))

    def upstream_order(self, target: str) -> list[str]:
        """`target` and everything it depends on, dependencies first."""
        order: list[str] = []
        visiting: set[str] = set()

        def visit(n: str) -> None:
            if n in order:
                return
            if n in visiting:
                raise GraphError(Msg("B-GRAPH-CYCLE"))
            visiting.add(n)
            for port in self.input_ports(n):
                for src, _ in self.inputs.get((n, port.name), []):
                    visit(src)
            visiting.discard(n)
            order.append(n)

        if target not in self.nodes:
            raise GraphError(Msg("E-GRAPH-NONODE", node=target))
        visit(target)
        return order

    def needed(self, targets: list[str]) -> list[str]:
        """`targets` and everything they depend on, dependencies first, each once."""
        return list(dict.fromkeys(n for t in targets for n in self.upstream_order(t)))

    def connected(self, node_id: str) -> frozenset[str]:
        """The node's input ports that have a wire."""
        return self.inputs_by_node.get(node_id, frozenset())

    def wired_types(self, node_id: str) -> dict[str, str]:
        """The data type each port actually carries in this graph (port name -> type id): inputs by the wire feeding
        them, outputs by `output_type`. Generic nodes (ports declared as any data) depend on it: `core.split_items`
        carrying 人物框 can be split in the browser, while carrying a scene it requires USD and stays on the server
        (`types` of NodeDef.browser_ops)."""
        # input and output ports may share a name (「选人」 uses boxes on both sides): outputs are set first and
        # inputs override them, since type-dependent decisions always concern what is wired in
        out = {port.name: self.output_type(node_id, port.name) for port in self.outputs(node_id)}
        for port in self.input_ports(node_id):
            wires = self.inputs.get((node_id, port.name)) or ()
            if wires:
                out[port.name] = self.output_type(*wires[0])
        return out

    def check_inputs(self, node_id: str) -> None:
        """The node can be planned: its wires are right (see _check_wires), every required input has one, and it keeps
        the rules of writing files (check_delivery)."""
        if node_id in self.wiring:
            wrong = self.wiring[node_id]
            if len(wrong) == 1:
                raise GraphError(wrong[0])
            # the combined message keeps the right tone: when every wire is only waiting for an upstream choice (an
            # import node without a picked model / camera, `waiting`), it still says a choice is pending; only when at
            # least one wire is actually wrong does it say the wiring is wrong. Otherwise two waiting wires would be
            # reported as "2 wires are wrong" and the artist would assume a wiring mistake
            waits = sum(1 for (_, _, dst, _) in self.waiting if dst == node_id)
            code = "B-WIRE-WAITSEVERAL" if waits == len(wrong) else "B-WIRE-SEVERAL"
            raise GraphError(Msg(code, count=len(wrong), problems=wrong))
        if node_id in self.scopes.unpaired:
            raise GraphError(self.scopes.unpaired[node_id][0])
        node = self.nodes[node_id]
        for port in self.input_ports(node_id):
            if not port.optional and not self.inputs.get((node_id, port.name)):
                raise GraphError(Msg("B-GRAPH-NOWIRE", node=node.label, input=port.label))
        # none of the alternative wirings (NodeDef.input_choice: the whole 「图像」 of 「序列图输出设置」, or individual
        # R G B A) is used: refused before submission, with the page stating which ports to wire, rather than failing
        # midway through the cook
        need = node.type.choice_inputs()
        if need and not any(self.inputs.get((node_id, name)) for name in need):
            labels = [p.label for p in self.input_ports(node_id) if p.name in need]
            raise GraphError(Msg("B-GRAPH-NOWIREANY", node=node.label, inputs=labels))
        self.check_delivery(node_id)

    # ------------------------------------------------------------------ writing files (「输出」, nodes/output.py)
    # Output-settings nodes write files (an output of type "files"); 「输出」 (a node taking "files") delivers them to
    # the user, each in a sub-folder of its 名字. The rules live here and nowhere else: the server refuses a cook that
    # breaks them, and the editor shows the same reasons before submitting (the graph's status).

    def gives_files(self, node_id: str) -> bool:
        return any(p.type == FILES for p in self.outputs(node_id))

    def delivered_by(self, node_id: str) -> list[str]:
        """The 「输出」 nodes the files of `node_id` are wired into."""
        return list(dict.fromkeys(dst for dst, port in self.outputs_by_node.get(node_id, ()) if _takes_files(self.nodes[dst], port)))

    def cook_targets(self, target: str) -> list[str]:
        """What cooking `target` means: a node that writes files is cooked by delivering them, through the 「输出」 it
        is wired into (not wired: itself, which the rules then refuse)."""
        return (self.delivered_by(target) or [target]) if self.gives_files(target) else [target]

    def deliveries(self) -> list[str]:
        """Every 「输出」 in the graph (NodeDef.delivers), in node order: what 交付 (POST /api/jobs with deliver) cooks
        together, one job for all of them."""
        return [nid for nid, node in self.nodes.items() if node.type.delivers]

    def computed_by(self, target: str) -> list[str]:
        """The nodes whose results cooking `target` computes: an 「输出」 only delivers what the settings nodes wired
        into it wrote (no work of its own), so its cook is theirs. What the time estimate covers, before a place to deliver to
        is chosen."""
        out = []
        for t in self.cook_targets(target):
            fed = [src for p in self.nodes[t].type.input_ports(self.nodes[t].params) if p.type == FILES
                   for src, _ in self.inputs.get((t, p.name), [])]
            out += fed or [t]
        return list(dict.fromkeys(out))

    def shown_by(self, target: str) -> list[str]:
        """The nodes whose results the viewer shows for `target` (webui/src/view/plan.ts displayPlan): its own; a result shown as
        what it was made from (the files an output-settings node wrote) and a node without results (「输出」) show
        what is wired into it, the same way. Showing a node cooks these: a delivery is never among them."""
        made = [_shown_as_inputs(self.output_type(target, p.name)) for p in self.outputs(target)]
        own = [target] if not all(made) else []
        return list(dict.fromkeys(own + (self._made_from(target) if not made or any(made) else [])))

    def _made_from(self, node_id: str) -> list[str]:
        """The nodes whose results are wired into the node's inputs (its declared ones, and one per row of a
        ports_from table that makes inputs: 「多层 EXR 输出设置」's 图层), through results shown as what they were made
        from."""
        return [n for port in self.input_ports(node_id) for src, sport in self.inputs.get((node_id, port.name), [])
                for n in (self._made_from(src) if _shown_as_inputs(self.output_type(src, sport)) else [src])]

    def check_delivery(self, node_id: str) -> None:
        """An output-settings node's files go into an 「输出」; an 「输出」 has somewhere to deliver to, and the settings
        nodes wired into it are named so each gets a folder of its own (a valid, distinct name). Raises GraphError
        saying what to change."""
        node = self.nodes[node_id]
        if self.gives_files(node_id) and not self.delivered_by(node_id):
            raise GraphError(Msg("B-DELIVER-NOTWIRED", node=node.label))
        for port in self.input_ports(node_id):
            if not _takes_files(node, port.name):
                continue
            named: dict[str, list[str]] = {}
            for src, _ in self.inputs.get((node_id, port.name), []):
                out = self.nodes[src]
                name = str(out.params.get("name") or "").strip()
                if not name or name in (".", "..") or _BAD_NAME.search(name):
                    raise GraphError(Msg("B-DELIVER-BADNAME", node=out.label, name=name))
                named.setdefault(name.lower(), []).append(f"「{out.label}」")
            same = [Msg("B-DELIVER-NAMED", outputs=labels, name=name) for name, labels in named.items() if len(labels) > 1]
            if same:
                raise GraphError(Msg("B-DELIVER-SAMENAME", node=node.label, same=same))
        for spec in node.type.param_specs() if any(_takes_files(node, p.name) for p in self.input_ports(node_id)) else ():
            if spec["widget"] == "deliver" and not node.params.get(spec["name"]):
                raise GraphError(Msg("B-DELIVER-NOPATH", node=node.label, setting=spec["label"]))
