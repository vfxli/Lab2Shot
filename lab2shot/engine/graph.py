"""A node graph loaded from its JSON file, validated against the node registry.

A node's inputs in a graph are its type's declared inputs and one more per promoted parameter: a parameter driven by a
wire ("promoted": ["focal_mm"] on the node in the file, its wires ordinary edges into "param:focal_mm"). Whatever walks
all of a node's inputs, promoted ones included, goes through Graph.input_ports; what is about its declared inputs alone
(the data wired in, not the values driving its parameters: what its result covers, Evaluation._wired; the labels of its
parameters' supplying inputs; files into 「输出」; the 3D kinds wired in) reads its type's NodeDef.input_ports.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import i18n
from ..errors import GraphError
from ..messages import Msg
from ..data.types import ABSTRACT_TYPES, DATA_TYPES, element_of, is_list, list_of, type_label, within
from ..nodes import accepts, node_types
from ..nodes.applies import NodeFacts, Resolved, facts_for, output_ports, resolve, waiting_port
from ..nodes.base import NodeDef, Port
from ..nodes.port import PARAM
from ..nodes.output import FILES, file_name, name_key
from .naming import node_ref
from .scopes import Scopes, chooses, rules

SCHEMA = "lab2shot.graph/1"


def _takes_files(node: GNode, port: str) -> bool:
    return any(p.name == port and p.type == FILES for p in node.type.input_ports(node.params))


def _common(types: list[str]) -> str:
    """The type every one of these carries: the widest of them that takes all the others (data/types.py accepts); else
    when they share a kind above them in the type tree (the part before a dot: 骨架动画 scene.skeleton and 蒙皮角色
    scene.character are both 场景 scene) that is a type of its own and not an open family root (ABSTRACT_TYPES:
    image, value stand for "not known yet", so image.3 with image.1 and value.int with value.float stay apart), the
    alternatives themselves ("scene.skeleton|scene.character": it carries one of them); a list of them the same way. "" when they have nothing in common (B-SWITCH-TYPES). A way whose type is still open (a
    port following an input that is not wired: 「Kimodo」's 骨架动画 or 蒙皮角色) may carry any of its alternatives,
    so the common type has to take every one of them, not just one (accepts takes an open type when any fits)."""
    if found := next((t for t in types if all(accepts(t, alt) for other in types for alt in other.split("|"))), ""):
        return found
    if not types or len({is_list(t) for t in types}) != 1:
        return ""
    parts = [element_of(t).split(".") for t in types]
    shared = []
    for level in zip(*parts):
        if len(set(level)) != 1:
            break
        shared.append(level[0])
    while shared:
        kind = ".".join(shared)
        if kind in DATA_TYPES and kind not in ABSTRACT_TYPES:
            # what it carries is one of them, by the way it takes (骨架动画 on one way, 蒙皮角色 on the other): the
            # alternatives themselves, not the kind above them — an input that takes either (「重定向预处理」's 动作)
            # takes it, as it takes a port that follows an input not wired yet (accepts: any alternative fits)
            return "|".join(dict.fromkeys(alt for t in types for alt in t.split("|")))
        shared.pop()
    return ""


def insert_fix(via: str, kind: str = "") -> dict:
    """THE one click that inserts a node, as every side offers it (a refused wire: Graph.fixes; a usage check:
    engine/lint.py; a message a cook says: CookContext.say): {"insert": the node type, "label": its button,
    「插入「…」」, with the kind of 3D data it lets through when it names one (NodeDef.fix_kind)}."""
    from ..data.types import kind_label
    from ..nodes import node_types

    said = Msg("I-FIX-INSERTKIND", via=node_types()[via].subtitle, kind=kind_label(kind)) if kind else Msg("I-FIX-INSERT", via=node_types()[via].subtitle)
    return {"insert": via, "label": said.text}


def walk(starts, next_of: Callable[[str], Any]) -> list[str]:
    """THE way everything reachable is collected (Graph.upstream_order, Demand.order and each_case, forget, the
    evaluation's and the graph's sources-first filling, templates): every node reachable from `starts`
    by `next_of` (a node -> the nodes it leads to, in order), each once, each after every node it leads to
    (dependencies first when `next_of` goes upstream), without recursion (a chain of any length). A graph has no cycle
    (Graph.from_json refuses one).

    What is not a collection of everything reachable walks on its own, each for its reason: finding the cycle itself
    (Graph._on_cycle, Kahn: a walk assumes there is none); searches that stop at the first hit, in order
    (Evaluation._failure_above, Routing._stand_in: an explicit stack); one chain followed up a single wire
    (Evaluation.shot: what was photographed, up the one picture input each output follows, a loop until an answer is
    known). The answers remembered per instance or per port that
    ask their sources' answers (plan, outcome, info, provisional; output_type, scene_kinds) recurse one step only:
    _sources_first fills their tables through a walk first, and every one of them is remembered, for an instance that
    waits too (a chain of waiting instances is worked out once each, not once per way of asking)."""
    order: list[str] = []
    done: set[str] = set()
    open_: set[str] = set()
    for start in starts:
        if start in done:
            continue
        stack = [(start, iter(next_of(start)))]
        open_.add(start)
        while stack:
            node, rest = stack[-1]
            nxt = next(rest, None)
            if nxt is None:
                stack.pop()
                open_.discard(node)
                done.add(node)
                order.append(node)
            elif nxt not in done and nxt not in open_:
                open_.add(nxt)
                stack.append((nxt, iter(next_of(nxt))))
    return order


def with_table(node_type, params: dict, wired: dict[str, tuple]) -> dict[str, tuple]:
    """`wired` (input -> what each wire into it brings: its type, or whether it is values) with an input-making table's rows (NodeDef.ports_from on the inputs:
    多层 EXR 的「图层」, 切换的「路」) together under the table's name, what a condition on the table reads
    (nodes/applies.py WiredType("layers", …)). The one place that sum is made: the graph's facts and an instance's with
    some wires gone (Evaluation.resolved) both call it."""
    wired = {k: v for k, v in wired.items() if k != node_type.ports_from}
    if node_type.ports_from and node_type.ports_from_side == "inputs":
        rows = {p.name for p in node_type.made_ports(params)}
        if together := tuple(x for name in sorted(rows) for x in wired.get(name, ())):
            wired[node_type.ports_from] = together
    return wired


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
    params: dict[str, Any]
    promoted: tuple[str, ...] = ()  # parameters driven by a wire: each has an input "param:<name>" (NodeDef.param_port)

    @property
    def label(self) -> str:
        """How a message points at this node: `name（type）` (engine/naming.py node_ref, the one place)."""
        return node_ref(self.id, self.type.id)


# what a node id may not hold: it is the key of `node.param` (templates' targets), `node@item` and `dst.dport`
# (evaluations.content_key), so none of those separators, and nothing invisible: no white space, no control
# character, no format character (Unicode Cf: a zero-width space, a direction mark), which would make two ids look
# alike; and at most ID_MOST characters (the editor's own are a type name and a number)
_ID = re.compile(r"[^./@:\s\x00-\x1f\x7f]+")
ID_MOST = 64


def _good_id(node_id: str) -> bool:
    return len(node_id) <= ID_MOST and bool(_ID.fullmatch(node_id)) and not any(unicodedata.category(c) == "Cf" for c in node_id)


def data_kind_word(data: bool) -> str:
    """How a message names what a port carries (Port.data): values (a normal map, motion vectors) or a picture."""
    return i18n.Word("engine.kind.data" if data else "engine.kind.picture")


def check_shape(data: dict) -> None:
    """The graph file's structure, checked where a graph is read (Graph.from_json, and first thing by every helper that
    reads the file before it: engine/templates.py apply_values, exposed_params, check_exposed, file_params) so nothing
    below meets a missing field: `nodes` a list of {id, type, params?: {…}, promoted?: [names]} (ids distinct), `edges` (none:
    no wires) a list of {from: [node, port], to: [node, port]}. GraphError naming what is wrong (E-GRAPH-SHAPE,
    E-GRAPH-SAMEID)."""
    nodes, edges = data.get("nodes"), data.get("edges", [])
    if not isinstance(nodes, list):
        raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.nodes")))
    if not isinstance(edges, list):
        raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.edges")))
    seen: set[str] = set()
    for i, n in enumerate(nodes, 1):
        if not isinstance(n, dict) or not isinstance(n.get("id"), str) or not n["id"] or not isinstance(n.get("type"), str):
            raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.node", n=i)))
        if not _good_id(n["id"]):  # the id is a key everywhere: node.param, node@path, content_key's dst.port
            raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.id", id=n["id"][:ID_MOST], most=ID_MOST)))
        if "comment" in n and not isinstance(n["comment"], str):
            raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.comment", id=n["id"])))
        if not isinstance(n.get("params", {}), dict):
            raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.params", id=n["id"])))
        if not (isinstance(n.get("promoted", []), list) and all(isinstance(x, str) for x in n.get("promoted", []))):
            raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.promoted", id=n["id"])))
        if n["id"] in seen:
            raise GraphError(Msg("E-GRAPH-SAMEID", id=n["id"]))
        seen.add(n["id"])
    end = lambda x: isinstance(x, list) and len(x) == 2 and all(isinstance(v, str) for v in x)  # noqa: E731
    for i, e in enumerate(edges, 1):
        if not isinstance(e, dict) or not end(e.get("from")) or not end(e.get("to")):
            raise GraphError(Msg("E-GRAPH-SHAPE", what=i18n.Word("engine.shape.edge", n=i)))


def _check_file_name(node) -> str:
    """An output-settings node's 名字, which names its files and its sub-folder: one plain file name, never a path.
    Returns it; GraphError when it is empty, "." / "..", or holds a separator or a character a file name may not."""
    return file_name(node.params.get("name"), node.label)


def _params_only(cond) -> bool:
    """A condition that reads the node's parameters and nothing else (no wire, no fact about the data)."""
    params, inputs, facts = cond.names()
    return bool(params) and not inputs and not facts


@dataclass
class Graph:
    nodes: dict[str, GNode]
    # (to_node, to_port) -> [(from_node, from_port), ...] in file order
    inputs: dict[tuple[str, str], list[tuple[str, str]]] = field(default_factory=dict)
    # the frames to cook, first and last: the sources emit only these, everything downstream follows; None: every
    # frame the inputs have (saved in the graph file as "frames": [first, last] or null)
    frames: tuple[int, int] | None = None
    # node -> what is wrong with the wires into it (an output that is gone, a type that no longer fits), each with the
    # inputs it is about (its own; none for ways of a switch that don't agree: the switch itself): raised when that node
    # is planned, so the rest of the graph still plans and shows its state; a switch is judged on the inputs it takes,
    # a problem counting only while every input it is about is taken (check_inputs(only=…))
    wiring: dict[str, list[tuple[frozenset[str], Msg]]] = field(default_factory=dict)
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
    # every wire is in (from_json's _index ran; a graph with a cycle is refused there): then what a port carries is a
    # fact of the graph, whichever call asks first, and output_type remembers every answer
    _settled: bool = field(default=False, repr=False, compare=False)
    # this thread is filling a table from the sources up (_sources_first); a graph is read by several threads at once
    _filling: threading.local = field(default_factory=threading.local, repr=False, compare=False)
    # node -> (dst, dport) pairs its outputs feed (Graph.delivered_by, the reverse of `inputs`).
    outputs_by_node: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict, repr=False, compare=False)
    # per-node input_ports(), built the first time it is asked for (GNode.type.inputs is static; only the promoted
    # parameters' Port objects cost anything, param_port() being a linear scan of the spec).
    _input_ports: dict[str, tuple[Port, ...]] = field(default_factory=dict, repr=False, compare=False)
    _outputs: dict[str, tuple[Port, ...]] = field(default_factory=dict, repr=False, compare=False)  # outputs(node) memo
    # scene_kinds(node, port) memo (once `_settled`, like output_type's)
    _scene_kinds: dict[tuple[str, str], frozenset[str]] = field(default_factory=dict, repr=False, compare=False)
    # output_type(node, port) memo (once `_settled`): without it a chain of switches each following all its ways
    # re-walks every way below it, exponential in the chain's length
    _output_types: dict[tuple[str, str], str] = field(default_factory=dict, repr=False, compare=False)
    _output_data: dict[tuple[str, str], bool | None] = field(default_factory=dict, repr=False, compare=False)
    # resolved(node) memo: what the node's declarations say with its parameters and wires (nodes/applies.py)
    _resolved: dict[str, Resolved] = field(default_factory=dict, repr=False, compare=False)
    # the 逐项处理 blocks: their members, every node's chain, the wires their rules refuse (engine/scopes.py rules),
    # worked out once by from_json from the final wires
    scopes: Scopes = field(default_factory=Scopes, repr=False, compare=False)

    @classmethod
    def from_json(cls, data: dict) -> Graph:
        """THE way a graph is read: a GraphError for anything wrong with it, never another exception (a fault of the
        program reading a graph it did not foresee is said as E-GRAPH-UNREADABLE, in the server's log with its trace,
        so /api/status answers instead of failing with a 500)."""
        try:
            return cls._from_json(data)
        except GraphError:
            raise
        except Exception as exc:  # noqa: BLE001
            from .. import logs

            said = Msg("E-GRAPH-UNREADABLE", detail=f"{type(exc).__name__}: {exc}")
            logs.say(logs.get("farm"), said, logs.error_text(exc), about="graph")
            raise GraphError(said) from exc

    @classmethod
    def _from_json(cls, data: dict) -> Graph:
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise GraphError(Msg("E-GRAPH-NOTGRAPH"))
        check_shape(data)
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
                raise GraphError(Msg("E-GRAPH-PARAMS", node=node_ref(n["id"], node_type.id), reason=exc)) from exc
            nodes[n["id"]] = GNode(n["id"], node_type, params, promoted)
        g = cls(nodes, {}, tuple(frames) if frames else None)
        for e in data.get("edges", []):
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
        if on_cycle := self._on_cycle():
            raise GraphError(Msg("E-GRAPH-CYCLE", nodes=i18n.Both.of(lambda: i18n.separator().join(i18n.Word("engine.quoted", name=self.name(n)) for n in on_cycle))))
        self._settled = True

    def _on_cycle(self) -> list[str]:
        """The nodes on a cycle of wires, in node order ([] none): a graph is a DAG or is refused (from_json), so
        nothing downstream ever meets a cycle. Kahn from both ends: what is left is on a cycle (or between two). Every
        wire in the file counts, one into an input its table no longer makes too (B-WIRE-GONEIN): the same wires
        outputs_by_node holds, so the two directions agree (a loop through a removed way of 「切换」 is a loop)."""
        into: dict[str, set[str]] = {n: set() for n in self.nodes}
        for (dst, _dport), wires in self.inputs.items():
            into[dst].update(src for src, _ in wires)
        out = {n: set(self.targets_of(n)) for n in self.nodes}
        for edges, back in ((into, out), (out, into)):
            ready = [n for n in edges if not edges[n]]
            while ready:
                n = ready.pop()
                for m in back.pop(n, ()):
                    if m in edges:
                        edges[m].discard(n)
                        if not edges[m]:
                            ready.append(m)
                edges.pop(n, None)
        return [n for n in self.nodes if n in into and n in out]

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

    def name(self, node_id: str) -> str:
        """The node as a message names it (GNode.label); one that is not in the graph (the end of a wire that points
        nowhere) by the id it was given."""
        return self.nodes[node_id].label if node_id in self.nodes else node_id

    def _connect(self, src: str, sport: str, dst: str, dport: str) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise GraphError(Msg("E-GRAPH-NOSUCHWIRENODE", source=self.name(src), target=self.name(dst)))
        inp = self.input_port(dst, dport)
        n = self.nodes[dst]
        if inp is None and not dport.startswith(PARAM) and n.type.ports_from and n.type.ports_from_side == "inputs":
            # an input the node's own table made and no longer has (a way taken out of 「切换」 with its wire still in):
            # that wire's problem (B-WIRE-GONEIN, drawn wrong, to be taken out), as a wire from an output gone is
            self.inputs.setdefault((dst, dport), []).append((src, sport))
            return
        if inp is None:
            out = next((p.label for p in self.outputs(src) if p.name == sport), sport)
            if dport.startswith(PARAM):
                spec = next((p for p in self.nodes[dst].type.param_specs() if p["name"] == dport[len(PARAM):]), None)
                raise GraphError(Msg("E-GRAPH-NOTPROMOTED", source=self.name(src), output=out, target=self.name(dst),
                                     input=spec["label"] if spec else dport[len(PARAM):]))
            raise GraphError(Msg("E-GRAPH-NOSUCHPORT", source=self.name(src), output=out, target=self.name(dst), input=dport))
        wires = self.inputs.setdefault((dst, dport), [])
        if wires and not inp.multi:
            raise GraphError(Msg("B-GRAPH-ONEWIRE", node=self.name(dst), input=inp.label))
        wires.append((src, sport))

    def outputs(self, node_id: str) -> tuple[Port, ...]:
        """The node's outputs with its parameters (nodes/applies.py output_ports: 读取序列 has one per layer it lists)."""
        if (found := self._outputs.get(node_id)) is None:
            n = self.nodes[node_id]
            found = self._outputs[node_id] = tuple(output_ports(n.type, n.params))
        return found

    def facts(self, node_id: str) -> NodeFacts:
        """What the graph knows of a node for its declarations (nodes/applies.py): its parameters (those a wire drives
        as WIRED), the types wired into each input, the facts it knows from its parameters."""
        n = self.nodes[node_id]
        wired, data = {}, {}
        for port in self.input_ports(node_id):
            wires = self.inputs.get((node_id, port.name), [])
            if wires:
                wired[port.name] = tuple(self.output_type(s, sp) for s, sp in wires)
                data[port.name] = tuple(self.output_data(s, sp) for s, sp in wires)
        wired, data = with_table(n.type, n.params, wired), with_table(n.type, n.params, data)
        # which of its own outputs go anywhere (a parameter that only shapes one of them: WiredOut)
        out = {sport for dst, dport in self.outputs_by_node.get(node_id, ()) for src, sport in self.inputs[(dst, dport)] if src == node_id}
        return facts_for(n.type, n.params, wired, tuple(self.wired_params(node_id)), out, self._incoming(node_id), data)

    def _incoming(self, node_id: str) -> dict[str, dict]:
        """Each input port of this node -> the facts the node wired into it declares about what it gives
        (NodeDef.facts of the source; nodes/applies.py Incoming). Several wires into one port: only facts every one
        of them agrees on, so a condition never reads one of two sources. Empty where nothing is wired or the source
        cannot tell yet; Incoming then says 'not known' and leaves the subject usable."""
        got: dict[str, dict] = {}
        for name in self.inputs_by_node.get(node_id, ()):
            wires = self.inputs[(node_id, name)]
            each = [self.nodes[src].type.facts(dict(self.nodes[src].params)) for src, _ in wires if src in self.nodes]
            if not each:
                continue
            shared = {k: v for k, v in each[0].items()
                      if all(k in other and other[k].value == v.value for other in each[1:])}
            if shared:
                got[name] = shared
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

    def _sources_first(self, node_id: str, table: dict, fill) -> None:
        """Before a type (or 3D kinds) is worked out from what is wired in, work out those of everything above it
        first, from the far end down (walk: no recursion), so the call itself goes one step up, never down a whole
        chain: a chain of a few hundred type-following nodes written downstream first would overflow Python's stack.
        Only once the graph is settled (no cycle, every wire in: what is remembered then holds); once per outermost call
        (the calls it makes find their sources done); a node whose every output is remembered already stops the walk."""
        if not self._settled or getattr(self._filling, "on", False):
            return
        self._filling.on = True
        try:
            def known(n: str) -> bool:
                return all((n, p.name) in table for p in self.outputs(n))

            for n in walk([node_id], lambda n: [s for s in dict.fromkeys(self.sources_of(n)) if not known(s)])[:-1]:
                for p in self.outputs(n):
                    fill(n, p.name)
        finally:
            self._filling.on = False

    def output_type(self, node_id: str, port: str, _seen: frozenset[str] = frozenset()) -> str:
        """The data type an output carries in this graph: its port's type or, for a port that follows its inputs
        (Port.type_from), what they carry; open ("image|map") while nothing is. Four ways to follow, the way the
        port writes it:
        - "input:<port>": what is wired into that input (「LensDistortion」's 镜头 a camera);
        - "input:<port>#item": one layer of list off it (「逐项开始」's 条目: an 图像序列[] gives 图像序列);
        - "input:<port>#list": the list of it (「逐项结束」's 列表, 「合成列表」's; what already is a list stays one,
          and an inner block's list is flattened);
        - "input:<port>,<port>...#common": the type every one of them carries (「切换」's branches; none in common: the
          port's own type, and the wires say so: B-SWITCH-TYPES).
        An input-making table's name stands for its rows, the ones the node has now (`_followed`: 「切换」 follows
        "input:ways#common", however many ways it has).

        Returns an empty string (rather than raising) when the node has no such port now: some ports appear with
        parameters (「导入 USD」 has a port only after a model is picked in its hierarchy), while wires in the graph may
        exist first. A wire to a port that does not exist yet means the type is not known yet, not a server fault;
        raising would turn /api/status into a 500 and the graph could not be opened. The empty string is how this
        chain expresses a missing port (wire_problem and describe read it that way). Remembered per (node, port) once
        the graph is settled (every wire in, no cycle: `_settled`)."""
        if (cached := self._output_types.get((node_id, port))) is not None:
            return cached
        self._sources_first(node_id, self._output_types, self.output_type)
        found = self._output_type(node_id, port, _seen)
        if self._settled:
            self._output_types[(node_id, port)] = found
        return found

    def output_data(self, node_id: str, port: str, _seen: frozenset[str] = frozenset()) -> bool | None:
        """Whether an image output gives values (True) or a picture (False) in this graph, None when that is not known
        (nothing wired into the input it follows yet, two inputs that disagree, not an image). The port's declaration
        (Port.data), else by its type: one or two channels values, three or four a picture, a port following an input
        (type_from) what that input carries. The one answer: conditions (WiredPicture) and the engine's check of what a
        node wrote (Engine._settle_outputs) both read it. Remembered like output_type."""
        if (node_id, port) in self._output_data:
            return self._output_data[(node_id, port)]
        self._sources_first(node_id, self._output_data, self.output_data)
        found = self._output_data_of(node_id, port, _seen)
        if self._settled:
            self._output_data[(node_id, port)] = found
        return found

    def _output_data_of(self, node_id: str, port: str, _seen: frozenset[str]) -> bool | None:
        from ..data.types import channels_of

        out = next((p for p in self.outputs(node_id) if p.name == port), None)
        if out is None:
            return None
        if out.data in (True, False):
            return out.data
        kind = self.output_type(node_id, port)
        if is_list(kind) or not channels_of(kind):
            return None
        if channels_of(kind) <= 2:
            return True
        if not out.type_from or node_id in _seen:
            return False
        named, _how = self._followed(node_id, out)
        got = {self.output_data(src, sport, _seen | {node_id}) for name in named
               for src, sport in self.inputs.get((node_id, name), []) if any(o.name == sport for o in self.outputs(src))}
        return got.pop() if len(got) == 1 else None

    def _output_type(self, node_id: str, port: str, _seen: frozenset[str]) -> str:
        out = next((p for p in self.outputs(node_id) if p.name == port), None)
        if out is None:
            return ""
        if not out.type_from or node_id in _seen:
            return out.type
        named, how = self._followed(node_id, out)
        got = self._carried(node_id, named, _seen | {node_id}, first=how != "common")
        if not got:
            return out.type
        if how == "item":
            return within(element_of(got[0]), out.type)
        if how == "list":
            return within(list_of(got[0]), out.type)
        if how == "common":
            # a way that carries no more than the port's own open type (a switch inside it whose every way is not
            # known yet) is not known yet either: the common type is the other ways', as a way from a port neither
            # there nor declared is left out (_carried)
            return _common([t for t in got if t != out.type] or got) or out.type
        return within(got[0], out.type)

    def takes(self, port_type: str, src: str, sport: str) -> bool:
        """Whether an input declared `port_type` takes what this output carries in this graph (data/types.py accepts on
        output_type). An output that is one of several ways (「切换」, "#common") carries whichever way is taken, each a
        concrete type of its own: its common type may be the wider kind above them (场景 over 骨架动画 and 蒙皮角色,
        _common), and an input that takes every one of the ways takes it too (「动作重定向」's 动作 takes 骨架动画 or
        蒙皮角色: a switch between the two goes straight in)."""
        if accepts(port_type, self.output_type(src, sport)):
            return True
        ways = self.ways(src, sport)
        return bool(ways) and all(accepts(port_type, w) for w in ways)

    def ways(self, node_id: str, port: str) -> list[str]:
        """For an output that is one of several ways ("#common": 「切换」), the type each way carries in this graph, each
        once, in the ways' order; [] for any other output."""
        out = next((p for p in self.outputs(node_id) if p.name == port), None)
        if out is None or not out.type_from:
            return []
        named, how = self._followed(node_id, out)
        return list(dict.fromkeys(self._carried(node_id, named, frozenset({node_id}), first=False))) if how == "common" else []

    def _followed(self, node_id: str, out: Port) -> tuple[list[str], str]:
        """The inputs an output following them (Port.type_from, "input:<port>,<port>...#how") follows on this node, and
        how. The one place type_from is read: a name that is the node's input-making table (NodeDef.ports_from) stands
        for its rows as they are now, the way the table's name stands for them in a condition (facts), so 「切换」's
        output follows every way it has and no other."""
        named, _, how = out.type_from.removeprefix("input:").partition("#")
        n = self.nodes[node_id]
        table = n.type.ports_from if n.type.ports_from_side == "inputs" else ""
        rows = [p.name for p in n.type.made_ports(n.params)] if table else []
        return [r for name in named.split(",") for r in (rows if table and name == table else [name])], how

    def _carried(self, node_id: str, ports: list[str], _seen: frozenset[str], first: bool) -> list[str]:
        """What is wired into these inputs carries (`first`: only the first wire of each). Into the ways of a switch
        (not `first`: every one of them), a wire from a port its node does not have now but declares (an import's 角色
        before a file is picked: NodeDef.outputs, its ports by its parameters) carries what the port is declared to
        carry, the type that way has once it is there: the switch is typed by all its ways, whichever it takes. Any
        other wire from a port not there is left out (nothing comes on it: a node following an optional reference
        with no file picked, 「Kimodo」's 角色, stays open), as is one from a port not declared with a type of its own."""
        got = []
        for name in ports:
            for src, sport in self.inputs.get((node_id, name), []):
                if any(o.name == sport for o in self.outputs(src)):
                    got.append(self.output_type(src, sport, _seen))
                elif not first and (declared := next((o for o in self.nodes[src].type.outputs if o.name == sport), None)) is not None and not declared.type_from:
                    got.append(declared.type)
                else:
                    continue
                if first:
                    break
        return got

    def scene_kinds(self, node_id: str, port: str, _seen: frozenset[str] = frozenset()) -> frozenset[str]:
        """The kinds of 3D data an output carries in this graph (types.SCENE_KINDS, and types.DEFORMING for a 模型
        that deforms), known before anything is cooked: what its node knows from its parameters (NodeDef.output_kinds:
        an import node, the kinds selected); for a port that follows an input (type_from), what that input carries;
        a single-kind type, its kind and the port's `kinds`; the umbrella 场景, what the node's 3D inputs carry (「USD
        打包」: all of them together; 3D nodes that pass their input on the same). Empty: not 3D data, or nothing known.
        The one derivation the wiring check, the output checks and the viewer use (the editor reads it from the status
        reply, Graph.ports). Remembered per (node, port) once the graph is settled (every wire in, no cycle:
        `_settled`), at every level of the walk as output_type is: a chain of 3D switches each following all its ways
        would otherwise walk every way below it again, exponential in the chain's length."""
        if (cached := self._scene_kinds.get((node_id, port))) is not None:
            return cached
        self._sources_first(node_id, self._scene_kinds, self.scene_kinds)
        result = self._scene_kinds_of(node_id, port, _seen)
        if self._settled:
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
            ports = self._followed(node_id, out)[0]
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
                    "kinds": [k for k in KIND_ORDER if k in self.scene_kinds(node_id, p.name)],
                    # one of several ways (「切换」, "#common"): the type each way carries; an input that takes every one
                    # of them takes this output (Graph.takes; the page's wiring check reads the same)
                    **({"ways": ways} if len(ways := self.ways(node_id, p.name)) > 1 else {})} for p in self.outputs(node_id)]
        # every port of a graph says what the pointer says over it (nodes/base.py Port.tip): the catalogue carries none
        # of it, so this is the one place a port's sentence comes from
        # applicability of input ports (Port.applies): ports not applicable now are sent to the page with the reason
        # to be shown disabled (the same algorithm as for parameters)
        from ..availability import resolve as _resolve

        t = self.nodes[node_id].type
        facts = self.facts(node_id)
        conds = {p.name: c for p in t.inputs if (c := t.port_applies(p)) is not None}
        off = _resolve(conds, facts, t).inactive
        # output ports have applicability too (Port.applies): the port stays in place, disabled,
        # with the reason on hover, and cannot be wired. It is computed in nodes/applies.py resolve()
        # (`Resolved.out_inactive`) and only read here, not resolved again
        out_off = self.resolved(node_id).out_inactive
        outputs = [{**o, **({"inactive": out_off[o["name"]].json()} if o["name"] in out_off else {})} for o in outputs]
        # `.json()` is required: `Msg.text` is a property, not a field, so serialising the object directly yields only
        # code and params; the page would read `p.inactive.text` as `undefined` and print "undefined" in the port hover
        # two kinds of 「off」: switched off by the node's own parameters (a mode: the port stays wireable, its wire is
        # simply not taken now — mode_off, engine/routing.py — and is drawn unused), or by what is wired (a wiring
        # mistake: `inactive`, the page refuses the wire, wire_problem B-WIRE-INACTIVE). The page reads the two apart
        # (webui graph/wireRule.ts)
        unused = {name for name, c in conds.items() if _params_only(c)}

        def state(name: str) -> dict:
            if name not in off:
                return {}
            return {"unused": off[name].json()} if name in unused else {"inactive": off[name].json()}

        return {"inputs": [{**p.describe(), "tip": p.tip(), **state(p.name)} for p in self.input_ports(node_id)], "outputs": outputs,
                "waiting": [{**p.describe(), "tip": p.tip(), "kinds": []} for p in self.resolved(node_id).waiting]}

    def mode_off(self, node_id: str) -> frozenset[str]:
        """The node's input ports its own parameters switch off now (Port.applies reading parameters only: a
        generator's 「参考图」 outside its multi-reference mode). Wires into them are not taken (engine/routing.py
        _choose); an input switched off by what is wired is a wiring mistake instead (wire_problem)."""
        from ..availability import resolve as _resolve

        t = self.nodes[node_id].type
        conds = {p.name: c for p in t.inputs if (c := t.port_applies(p)) is not None and _params_only(c)}
        return frozenset(_resolve(conds, self.facts(node_id), t).inactive) if conds else frozenset()

    def handles(self, node_id: str) -> list[int]:
        """The node's viewer handles that apply with its parameters and wires now (Handle.when), by their index."""
        from ..availability import AVAILABLE, level

        facts = self.facts(node_id)
        return [i for i, h in enumerate(self.nodes[node_id].type.handles) if level(h.when, facts) is AVAILABLE]

    def wire_states(self, unused: frozenset[tuple[str, str, str, str]] = frozenset()) -> list[dict]:
        """Every wire as the editor draws it: the type it carries, "ok", "waiting" for its output (an import node's kind
        not selected yet), "unused" (into a route a switch does not take now: whatever is wrong there is not a problem,
        Evaluation.unchosen_wires) or "wrong" with the problem (a message) and the node type that puts it right ("" none)."""
        out = []
        for (dst, dport), wires in self.inputs.items():
            for src, sport in wires:
                problem = None if (src, sport, dst, dport) in unused else self.wire_problem(src, sport, dst, dport)
                state = ("unused" if (src, sport, dst, dport) in unused else "ok" if problem is None
                         else "waiting" if (src, sport, dst, dport) in self.waiting else "wrong")
                out.append({"from": [src, sport], "to": [dst, dport], "type": self.output_type(src, sport),
                            "state": state, "problem": problem.json() if problem is not None else None,
                            "fix": self.fix_for(src, sport, dst, dport) if problem is not None else ""})
        return out

    @classmethod
    def at_defaults(cls, type_id: str) -> dict:
        """A node type's ports and handles on its own at its default parameters, nothing wired: what the editor shows
        for a node just added, before the status reply for it arrives (the catalogue's `at_defaults`).

        Without what the pointer says over each port (`tip`): that sentence repeats every type's description on every
        port, which over every node type adds tens of KB to the catalogue on every first load. It travels with a
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
            if len(wires) > 1 and (together := self.nodes[dst].type.several_refused(dport, tuple(self.output_type(s, p) for s, p in wires))):
                self.wiring.setdefault(dst, []).append((frozenset({dport}), together))  # the wires together (NodeDef.several_refused)
            for src, sport in wires:
                if problem := self.wire_problem(src, sport, dst, dport):
                    # the inputs it is about: its own (not a problem on a way a switch does not take); ways of a switch
                    # that don't agree are about the switch itself, whichever way it takes: what its output carries is
                    # a fact of the graph, it can't be a type on one item and another on the next
                    about = frozenset() if problem.code == "B-SWITCH-TYPES" else frozenset({dport})
                    self.wiring.setdefault(dst, []).append((about, problem))
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
        if out is None or (inp := self.input_port(dst, dport)) is None or inp.param:
            return ""
        t = self.output_type(src, sport)
        if self._list_problem(src, sport, dst, dport) is not None:  # a list where a single is meant, or the reverse
            if is_list(t):
                return _list_node("one")  # 「取一条」 (「逐项开始」 is the other way: it opens a block, so it is not one click)
            from ..data.items import kind_of as items_of

            return _list_node("split" if items_of(element_of(t).split("|")[0]) is not None else "make")
        if not self.takes(inp := self.input_port(dst, dport).type, src, sport):
            return converter(t, inp)
        via = node_types().get(self.nodes[dst].type.refusal_fix(t, self.scene_kinds(src, sport)))
        return via.id if via and any(accepts(p.type, t) for p in via.inputs) else ""

    def fix_label(self, src: str, sport: str, dst: str, via: str) -> str:
        """The one click's button for a refused wire (insert_fix, with the kind of 3D data the node lets through)."""
        return insert_fix(via, self.nodes[dst].type.fix_kind(self.output_type(src, sport), self.scene_kinds(src, sport)))["label"]

    def wire_problem(self, src: str, sport: str, dst: str, dport: str) -> Msg | None:
        """What is wrong with one wire (None nothing): see _check_wires."""
        from ..data.values import unit_problem

        kind = type_label  # 「图像序列」, 「图像序列列表」, 「深度图或遮罩」 (data/types.py)
        inp = self.input_port(dst, dport)
        # the nodes' labels only for a message: most wires are fine, and a label is worked out in the language now
        def source() -> str:
            return self.nodes[src].label

        def to() -> dict:
            return {"node": self.nodes[dst].label, "input": inp.label}

        if inp is None:  # an input its table no longer makes (_connect)
            out = next((p.label for p in self.outputs(src) if p.name == sport), sport)
            return Msg("B-WIRE-GONEIN", node=self.nodes[dst].label, input=dport, source=source(), output=out)
        # the input port is not applicable now (Port.applies: with 「图像」 wired, 「序列图输出设置」 takes no individual
        # channels, and vice versa). It is disabled on the node and the page does not allow the wire; the server
        # refuses it as well, preventing submissions that bypass the page
        # An input the node's own parameters switch off (a mode with no use for it) is not refused: the wire is not
        # taken this time, drawn as unused and its source not computed (engine/routing.py), so a card can keep the
        # inputs of every mode wired
        cond = self.nodes[dst].type.port_applies(inp)
        if cond is not None and not _params_only(cond):
            from ..availability import resolve as _resolve

            off = _resolve({dport: cond}, self.facts(dst), self.nodes[dst].type).inactive
            if dport in off:
                return Msg("B-WIRE-INACTIVE", why=off[dport], **to())
        out = next((p for p in self.outputs(src) if p.name == sport), None)
        # the source output port is not applicable now (Port.applies, e.g. a solver's 「相机」 only passes through once
        # a camera is wired): disabled on the node, not allowed by the page, and refused by the server as well
        # (preventing submissions that bypass the page)
        if out is not None and out.applies is not None:
            from ..availability import resolve as _resolve_out

            gone = _resolve_out({sport: out.applies}, self.facts(src), self.nodes[src].type).inactive
            if sport in gone:
                return Msg("B-WIRE-OUTINACTIVE", source=source(), output=out.label, why=gone[sport])
        if out is None and (waits := waiting_port(self.nodes[src].type, sport, self.nodes[src].params)):
            if dport == self.nodes[dst].type.presence_of:
                # a 「有没有」 asks whether it comes: an output the source does not give with its parameters is its
                # answer 「没有」 (Evaluation.comes, the same judgement), not a wire waiting for the user (e.g.
                # a skeleton-only FBX leaves 角色 unselected, the card's 「有没有」 then takes the skeleton's way)
                return None
            return Msg("B-WIRE-WAITS", source=source(), what=waits.label, waits=waits.waits_text, **to())
        if out is None:
            return Msg("B-WIRE-GONE", source=source(), output=sport, **to())
        if (said := self._list_problem(src, sport, dst, dport)) is not None:
            return said
        if not self.takes(inp.type, src, sport):
            t = self.output_type(src, sport)
            from ..nodes.registry import converter

            if not inp.param and (via := converter(t, inp.type)):  # a conversion exists: never done on the quiet
                return Msg("B-WIRE-CONVERT", source=source(), output=out.label, got=kind(t), want=kind(inp.type),
                           via=node_types()[via].subtitle, **to())
            return Msg("B-WIRE-TYPE", source=source(), output=out.label, got=kind(t), want=kind(inp.type), **to())
        t = self.output_type(src, sport)
        # a picture where values are, or values where a picture is (Port.data on both ends; output_data): a normal map
        # read by a model as a photograph, or premultiplied as colour, is wrong with no error anywhere
        if inp.data in (True, False) and (have := self.output_data(src, sport)) is not None and have != inp.data:
            return Msg("B-WIRE-DATAKIND", source=source(), output=out.label, got=data_kind_word(have),
                       want=data_kind_word(inp.data), **to())
        if not inp.param and (why := self.nodes[dst].type.refuses(t, self.scene_kinds(src, sport))):  # a parameter's input takes its value
            return Msg("B-WIRE-REFUSED", source=source(), output=out.label, got=kind(t), node=to()["node"], reason=why)
        if not inp.param and (why := self.nodes[dst].type.param_refuses(t, self.nodes[dst].params)):
            return Msg("B-WIRE-REFUSED", source=source(), output=out.label, got=kind(t), node=to()["node"], reason=why)
        if not inp.param and (other := self._uncommon(dst, dport, t)):  # 「切换」: every branch the same kind of data
            return Msg("B-SWITCH-TYPES", node=to()["node"], input=inp.label, got=kind(t), other=kind(other))
        # incompatible units cannot be wired, for parameter ports and data ports alike. Checking only `inp.param` would
        # let a data port declaring a unit (「Focal Length（px）」 / 「Focal Length（mm）」 of 「Focal Length 换算」) accept a
        # millimetre value and fail only midway through the cook.
        if inp.unit and (why := unit_problem(self.output_unit(src, sport), inp.unit)):
            return Msg("B-WIRE-UNIT", source=source(), output=out.label, reason=why, **to())
        if inp.plain and (have := self.output_unit(src, sport)):  # a plain number only (Port.plain)
            return Msg("B-WIRE-UNIT", source=source(), output=out.label, reason=Msg("B-VALUES-PLAIN", have=have), **to())
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
            return Msg("B-LIST-NESTED", source=self.nodes[src].label, output=out.label, got=i18n.Both.of(lambda: type_label(t)), **to)
        return Msg("B-WIRE-LIST", source=self.nodes[src].label, output=out.label, got=i18n.Both.of(lambda: type_label(t)),
                   want=i18n.Both.of(lambda: type_label(inp.type)), **to)

    def _makes_list_of(self, node_id: str, port: str) -> bool:
        """The node makes a list of what this input carries (an output following it with "#list")."""
        return any(out.type_from.endswith("#list") and port in self._followed(node_id, out)[0] for out in self.outputs(node_id))

    def _uncommon(self, node_id: str, port: str, carried: str) -> str:
        """For an input a "common:" output follows (「切换」's branches): the type another of them carries that this one
        has nothing in common with, "" none."""
        for out in self.outputs(node_id):
            if not out.type_from:
                continue
            named, how = self._followed(node_id, out)
            if how != "common" or port not in named:
                continue
            for other in self._carried(node_id, [p for p in named if p != port], frozenset({node_id}), first=False):
                if not _common([carried, other]):
                    return other
        return ""

    def _fits(self, src: str, sport: str, dst: str, dport: str) -> bool:
        """The wire is wrong only by the rules of a block: its output is there and its type fits."""
        out = next((p for p in self.outputs(src) if p.name == sport), None)
        return out is not None and (inp := self.input_port(dst, dport)) is not None and self.takes(inp.type, src, sport)

    def upstream_order(self, target: str, routes: Callable[[str], frozenset[str] | None] | None = None) -> list[str]:
        """`target` and everything it depends on, dependencies first. `routes`: the inputs a node takes (a switch whose
        route is known: Evaluation.taken_ports), None for all of them; without it every wire is followed (the graph as
        drawn). The routed walk is Evaluation.upstream: what a cook of `target` really needs."""
        if target not in self.nodes:
            raise GraphError(Msg("E-GRAPH-NONODE", node=target))
        return walk([target], lambda n: self.sources_of(n, routes(n) if routes is not None else None))

    def sources_of(self, node_id: str, taken: frozenset[str] | None = None) -> list[str]:
        """The nodes wired into this one, in the order its inputs are declared (`taken`: only into these inputs)."""
        return [src for port in self.input_ports(node_id) if taken is None or port.name in taken
                for src, _ in self.inputs.get((node_id, port.name), [])]

    def targets_of(self, node_id: str) -> list[str]:
        """The nodes this one's outputs are wired into."""
        return list(dict.fromkeys(dst for dst, _ in self.outputs_by_node.get(node_id, ())))

    def needed(self, targets: list[str], routes: Callable[[str], frozenset[str] | None] | None = None) -> list[str]:
        """`targets` and everything they depend on, dependencies first, each once (`routes`: see upstream_order)."""
        return list(dict.fromkeys(n for t in targets for n in self.upstream_order(t, routes)))

    def connected(self, node_id: str) -> frozenset[str]:
        """The node's input ports that have a wire."""
        return self.inputs_by_node.get(node_id, frozenset())

    def check_inputs(self, node_id: str, only: frozenset[str] | None = None, every_item: bool = False) -> None:
        """The node can be planned: its wires are right (see _check_wires), every required input has one, and it keeps
        the rules of writing files (check_delivery). `only`: a switch whose choice is known (engine/evaluation.py
        Evaluation.taken_ports) is checked on the inputs it takes; a wire waiting on a route it does not take (an import node on the
        other route without a picked camera) is not its problem. Ways that don't agree on a type (B-SWITCH-TYPES) are,
        whichever way it takes: the type its output carries is one fact of the graph (_check_wires).
        `every_item`: what holds for every instance of a node in a 逐项处理 block, checked before its items are known
        (the cook's up-front check, its status while they are not): all of it, except that a switch whose condition
        is wired, whose route each item decides, is held only to what is about no one way (B-SWITCH-TYPES); its ways
        are checked per instance, once its items are known."""
        kind, per_item = self.nodes[node_id].type, False
        if every_item and chooses(kind):
            per_item = bool(self.inputs.get((node_id, kind.condition_input)))
            if not per_item:  # its own 「走哪一路」: the same route for every item (NodeDef.chosen_inputs)
                only = frozenset(kind.chosen_inputs(self.nodes[node_id].params, None)) | {kind.condition_input}
        if node_id in self.wiring:
            wrong = [m for about, m in self.wiring[node_id] if (not about if per_item else only is None or about <= only)]
            if len(wrong) == 1:
                raise GraphError(wrong[0])
            if wrong:
                # the combined message keeps the right tone: when every wire is only waiting for an upstream choice (an
                # import node without a picked model / camera, `waiting`), it still says a choice is pending; only when at
                # least one wire is actually wrong does it say the wiring is wrong. Otherwise two waiting wires would be
                # reported as "2 wires are wrong" and the artist would assume a wiring mistake
                waits = sum(1 for (_, _, dst, dport) in self.waiting if dst == node_id and not per_item and (only is None or dport in only))
                code = "B-WIRE-WAITSEVERAL" if waits == len(wrong) else "B-WIRE-SEVERAL"
                raise GraphError(Msg(code, count=len(wrong), problems=wrong))
        if node_id in self.scopes.unpaired:
            raise GraphError(self.scopes.unpaired[node_id][0])
        node = self.nodes[node_id]
        if chooses(node.type) and not per_item and (said := self._no_route(node_id, only)) is not None:
            raise GraphError(said)
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

    def _no_route(self, node_id: str, only: frozenset[str] | None) -> Msg | None:
        """A switch whose 「走哪一路」 names no way it has, or a way with no wire (None: it names a wired one, or which it
        names is not known yet): said before anything is planned rather than when the switch cooks. The way it takes is
        routing's answer (`only`, Evaluation.taken_ports); read without one (lab2shot check templates, lint), a condition
        not wired is its own 「走哪一路」, the one rule NodeDef.chosen_inputs."""
        node = self.nodes[node_id]
        cond = node.type.condition_input
        ways = [p for p in self.input_ports(node_id) if not p.param]
        wired = bool(self.inputs.get((node_id, cond)))
        if only is None and wired:
            return None
        chosen = (only - {cond}) if only is not None else node.type.chosen_inputs(node.params, None)
        if not chosen and getattr(node.type, "blocks", False):  # a 「阻断」 set to block takes nothing: that is its route
            return None
        if not chosen:
            if wired:
                return Msg("B-SWITCH-WIREDRANGE", node=node.label, count=len(ways))
            return Msg("B-SWITCH-RANGE", node=node.label, value=node.params.get(cond.removeprefix(PARAM)), count=len(ways))
        if not any(self.inputs.get((node_id, way)) for way in chosen):
            label = next((p.label for p in ways if p.name in chosen), "")
            return Msg("B-SWITCH-NOWIRE", node=node.label, way=label)
        return None

    # ------------------------------------------------------------------ writing files (「输出」, nodes/output.py)
    # Output-settings nodes write files (an output of type "files"); 「输出」 (a node taking "files") delivers them to
    # the user, each in a sub-folder of its 名字. The rules live here and nowhere else: the server refuses a cook that
    # breaks them, and the editor shows the same reasons before submitting (the graph's status).

    def gives_files(self, node_id: str) -> bool:
        return any(p.type == FILES for p in self.outputs(node_id))

    def delivered_by(self, node_id: str) -> list[str]:
        """The 「输出」 nodes the files of `node_id` are wired into."""
        return list(dict.fromkeys(dst for dst, port in self.outputs_by_node.get(node_id, ()) if _takes_files(self.nodes[dst], port)))

    def deliveries(self) -> list[str]:
        """Every 「输出」 in the graph (NodeDef.delivers), in node order: what POST /api/jobs with `deliver: []` cooks
        together (a DCC plugin, `lab2shot cook`), one task for all of them."""
        return [nid for nid, node in self.nodes.items() if node.type.delivers]

    def check_delivery(self, node_id: str) -> None:
        """An output-settings node's files go into an 「输出」, and the settings nodes wired into one are named so each
        gets a folder of its own (a valid, distinct name). Raises GraphError saying what to change."""
        node = self.nodes[node_id]
        if self.gives_files(node_id):
            # its own name is checked whenever it is cooked, not only when an 「输出」 is: the name becomes a file name.
            # A 名字 driven by a wire is known only when the node cooks: checked there (OutputSettings.stem)
            if "name" not in self.wired_params(node_id):
                _check_file_name(node)
            if not self.delivered_by(node_id):
                raise GraphError(Msg("B-DELIVER-NOTWIRED", node=node.label))
        for port in self.input_ports(node_id):
            if not _takes_files(node, port.name):
                continue
            named: dict[str, list[str]] = {}
            for src, _ in self.inputs.get((node_id, port.name), []):
                out = self.nodes[src]
                if "name" in self.wired_params(src):  # its 名字 comes by wire: compared when 「输出」 cooks (core/output.py)
                    continue
                name = _check_file_name(out)
                named.setdefault(name_key(name), []).append(i18n.Word("engine.quoted", name=out.label))
            same = [Msg("B-DELIVER-NAMED", outputs=labels, name=name) for name, labels in named.items() if len(labels) > 1]
            if same:
                raise GraphError(Msg("B-DELIVER-SAMENAME", node=node.label, same=same))
