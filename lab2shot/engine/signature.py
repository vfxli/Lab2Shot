"""A tool's typed signature (B2): what data it reads from files and what it delivers, worked out from its node graph
alone, like its exposed parameters (engine/templates.py exposed_params). Every client that runs a graph as a tool (DCC
plugins, scripts) reads it to tell which of its own objects goes into which file parameter and what comes back, without
knowing the graph:

- `inputs`: one entry per file parameter (engine/templates.py file_params) — `param` its outside name (the template's
  exposed name, the fixed names of templates/_conventions.md: input, cam_fbx_path, char_fbx_path …; else node.param),
  `key` node.param, `widget` (file / sequence), `label`, `type` the data its node reads out of the file along the
  wires the graph has from it ("a|b" when several), `kinds` the 3D kinds among them (data/types.py SCENE_KINDS),
  `optional` (the graph's deliveries do not need it with its values as they stand: engine/templates.py delivered_by)
  `when` ({outside name: value}: the one menu value that makes the deliveries need it, None when no single value
  does or it is needed anyway) and `accept` the file name suffixes its node reads ([]: any; a client whose object
  exports to one format binds it to the input that reads that format: cam_usd_path, not cam_fbx_path).
- `delivers`: one entry per 「输出」 — `node`, `label`, `types` / `kinds` everything wired into the output-settings
  nodes that feed it, and `settings` one entry per such node (its node id, label, node type, `format` its 3D format
  name when it is one: engine/deliver_formats.py, `name` its 名字, `types`, `kinds`).
- `focus`: the node a client opens the graph's job on in the page (「在网页里打开」, `#job=…&focus=<node>`): the node
  worked on in the viewer, by its declarations — a node with viewer handles (NodeDef.handles; 「在视图里点选」 buttons
  are made for the parameters of such a handle). Among several: one whose handle's parameters (or their 「在视图里点选」)
  the interface exposes, then one the deliveries need with the values as they stand (delivered_by), then one whose
  handle applies now, then the first in the graph. None: the first 「输出」; none either: None. Worked out here, from the
  graph, so every client focuses the same node.

Nothing here knows an application; the same graph gives the same signature to every client.

A signature depends on the graph and on the node types (their ports, labels, what each delivers): it is remembered per
graph content for the registry it was worked out with (nodes/registry.py registry_key), and worked out again once that
changes. The templates' and the single-node tools' are worked out in the background when the server starts
(server/tools.py warm), so the first request of a plugin does not wait for them; a graph a client brings along is never
remembered (remember=False), so no client can push the others out.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict

from .. import i18n
from ..data.types import kind_of
from .graph import Graph, GraphError

REMEMBERED = 512
_DONE: OrderedDict[str, str] = OrderedDict()  # graph text -> its signature, for the registry in _REGISTRY
_GOING: dict[str, threading.Event] = {}  # being worked out now: a second asker waits for it instead of doing it again
_REGISTRY: list = [None]
_LOCK = threading.Lock()


def signature(data: dict, registry: tuple | None = None, *, remember: bool = True) -> dict:
    """{"inputs": [...], "delivers": [...]} of a graph (the module docstring); a graph that does not read gives both
    empty (the client is told why when it submits). Remembered per graph content and registry (`registry`: its
    registry_key, when the caller asks it once for many graphs) unless `remember` is False."""
    try:
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return {"inputs": [], "delivers": []}
    if not remember:
        return json.loads(_work_out(text))
    if registry is None:
        from ..nodes.registry import registry_key

        registry = registry_key()
    return json.loads(_remembered(text, registry))


def _remembered(text: str, registry: tuple) -> str:
    while True:
        with _LOCK:
            if _REGISTRY[0] != registry:  # another set of node types, or their words changed: nothing remembered holds
                _DONE.clear()
                _REGISTRY[0] = registry
            if text in _DONE:
                _DONE.move_to_end(text)
                return _DONE[text]
            going = _GOING.get(text)
            if going is None:
                going = _GOING[text] = threading.Event()
                break
        going.wait()  # the same graph is being worked out (the warm-up, another request): its answer, then
    try:
        out = _work_out(text)
        with _LOCK:
            if _REGISTRY[0] == registry:
                _DONE[text] = out
                while len(_DONE) > REMEMBERED:
                    _DONE.popitem(last=False)
        return out
    finally:
        with _LOCK:
            _GOING.pop(text, None)
        going.set()


def _work_out(text: str) -> str:
    data = json.loads(text)
    try:
        graph = Graph.from_json(data)
    except GraphError:
        return json.dumps({"inputs": [], "delivers": [], "focus": None})
    from .templates import delivered_by

    needed = delivered_by(data)
    return json.dumps({"inputs": _inputs(data, graph, needed), "delivers": _delivers(graph),
                       "focus": _focus(data, graph, needed)}, ensure_ascii=False)


def _focus(data: dict, graph: Graph, needed: list[str] | None) -> str | None:
    """The node to focus the page on (the module docstring)."""
    from ..nodes.params import PICKED_IN_VIEW
    from .templates import exposed_items, targets_of

    exposed = {key for x, _ in exposed_items(data) for key in targets_of(x)}
    best: tuple | None = None
    for i, (nid, node) in enumerate(graph.nodes.items()):
        handles = node.type.handles
        if not handles:
            continue
        try:
            applies = bool(graph.handles(nid))
        except Exception:  # noqa: BLE001 (facts that cannot be told now: its handles count as declared only)
            applies = False
        worked = {p for h in handles for p in h.params.values()}
        worked |= {f"{p}_pick" for p in worked if p in PICKED_IN_VIEW}  # its 「在视图里点选」 (nodes/params.py pick_button)
        rank = (not any(f"{nid}.{p}" in exposed for p in worked), needed is not None and nid not in needed, not applies, i)
        if best is None or rank < best[0]:
            best = (rank, nid)
    if best is not None:
        return best[1]
    return next(iter(graph.deliveries()), None)


def _join(types: list[str]) -> str:
    return "|".join(dict.fromkeys(a for t in types for a in t.split("|") if a))


def _ordered(kinds: set[str]) -> list[str]:
    from ..data.types import KIND_ORDER

    return [k for k in KIND_ORDER if k in kinds]


def _kinds(types: list[str]) -> list[str]:
    return _ordered({k for t in types for a in t.split("|") if (k := kind_of(a))})


def _read_types(graph: Graph, node_id: str) -> list[str]:
    """What a reading node gives along the wires leaving it, as its type declares those outputs (an import node's
    selection ports exist only once something is selected, the wires are there before); its declared data outputs
    when no wire leaves it."""
    from ..nodes.applies import all_outputs

    node = graph.nodes[node_id]
    declared = {p.name: p.type for p in all_outputs(node.type)}
    for p in graph.outputs(node_id):
        declared.setdefault(p.name, p.type)
    used = [declared.get(sport) or graph.output_type(node_id, sport)
            for (src, sport), _dst in _leaving(graph, node_id)]
    data = [t for t in used if t and not t.startswith("value")]
    if data:
        return data
    return [t for t in declared.values() if not t.startswith("value")]


def _leaving(graph: Graph, node_id: str) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    return [((src, sport), (dst, dport)) for (dst, dport), wires in graph.inputs.items()
            for src, sport in wires if src == node_id]


def _menus(data: dict) -> list[tuple[str, list]]:
    """The exposed choices a client picks between: (outside name, its values) for every menu, and every parameter
    with options of its own."""
    from .templates import route_variants

    picks: dict[str, list] = {}
    for pick, _chosen in route_variants(data)[1:]:
        for name, value in pick.items():
            if not any(value == v and type(value) is type(v) for v in picks.setdefault(name, [])):
                picks[name].append(value)
    return list(picks.items())


def _inputs(data: dict, graph: Graph, needed: list[str] | None) -> list[dict]:
    from .templates import apply_values, delivered_by, exposed_items, exposed_label, file_params, targets_of

    labels = {key: exposed_label(x) for x, _ in exposed_items(data) for key in targets_of(x)}
    menus = None
    chosen_by: dict[tuple[int, int], list[str] | None] = {}  # (menu, value) -> delivered_by: the same for every file input

    def chosen(i: int, j: int, name: str, v) -> list[str] | None:
        if (i, j) not in chosen_by:
            try:
                chosen_by[i, j] = delivered_by(apply_values(data, {name: v}))
            except GraphError:
                chosen_by[i, j] = None
        return chosen_by[i, j]

    out = []
    for f in file_params(data):
        node_id = f["key"].split(".", 1)[0]
        types = _read_types(graph, node_id) if node_id in graph.nodes else []
        optional = needed is not None and node_id not in needed
        when = None
        if optional:
            menus = _menus(data) if menus is None else menus
            for i, (name, values) in enumerate(menus):
                for j, v in enumerate(values):
                    if node_id in (chosen(i, j, name, v) or ()):
                        when = {name: v}
                        break
                if when:
                    break
        out.append({"param": f["name"] or f["key"], "key": f["key"], "widget": f["widget"],
                    "label": labels.get(f["key"]) or f["label"], "node": f["node"],
                    "type": _join(types), "kinds": _kinds(types), "optional": optional, "when": when,
                    "accept": f["accept"]})
    return out


def _delivers(graph: Graph) -> list[dict]:
    from ..nodes.port import PARAM
    from .deliver_formats import format_of

    out = []
    for d in graph.deliveries():
        settings, all_types = [], []
        for (dst, dport), wires in graph.inputs.items():
            if dst != d:
                continue
            for src, _sport in wires:
                node = graph.nodes[src]  # an output-settings node feeding the 「输出」
                types, kinds = [], set()
                for p in graph.input_ports(src):
                    if p.name.startswith(PARAM):
                        continue
                    for s2, p2 in graph.inputs.get((src, p.name), []):
                        types.append(graph.output_type(s2, p2))
                        kinds |= set(graph.scene_kinds(s2, p2))
                name = node.params.get("name") if isinstance(node.params, dict) else None
                settings.append({"node": src, "label": node.label, "type": node.type.id, "format": format_of(node.type.id),
                                 "name": str(name or ""), "types": _join(types).split("|") if types else [],
                                 "kinds": _ordered(kinds | set(_kinds(types)))})
                all_types += types
        out.append({"node": d, "label": graph.nodes[d].label, "types": _join(all_types).split("|") if all_types else [],
                    "kinds": _ordered({k for s in settings for k in s["kinds"]}), "settings": settings})
    return out
