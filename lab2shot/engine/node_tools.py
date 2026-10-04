"""Single nodes as tools (B1): a node that works on data a client can hand over as files, wrapped by the server into
the smallest graph that runs it — 「读入 → 节点 → 输出设置 → 输出」 — and listed beside the templates, so every node
can be used from a DCC plugin or a script without anyone making a template for it. The wrapped graph is an ordinary
graph with an ordinary parameter interface: submitted, signed (engine/signature.py) and rewritten for formats
(engine/deliver_formats.py) like any template.

How a node is wrapped, all from its declarations:
- each required input gets a reading node by its data type: the node type declaring `reads` (nodes/base.py Reads)
  with an output that the input takes, the lowest `rank` first, trying the input's alternatives in order (pictures
  「读取序列」, a video 「读取视频」, a list of pictures 「读取多条序列」, 3D data the import nodes: FBX before Alembic before
  PLY before USD, as each declares). Nothing here names a reading node or its parameters: a new format's import node
  is used once it declares `reads`. The first reader of the tool's main material (Reads.main) gives its file and the
  parameters read with it the outside names templates/_conventions.md binds to them (`input`, `colorspace_in`,
  `first_frame`, `last_frame`); any other reader's file is `<node id>_<file parameter>`. A required input no reading
  node gives (人物框, 2D tracks, values …) means the node is not offered. Optional inputs stay unwired.
- every output carrying data an output-settings node writes is delivered: all 3D outputs together into one 3D
  settings node (USD by default, the format holding every kind; a client asks for its own with `formats`), each other
  one into the first settings node that takes it; everything into one 「输出」. A node with nothing to deliver, a
  reading node itself, a settings node, flow and value nodes, and a node declaring `tool = False` (one that only works
  between others of a graph) are not offered.
- the interface: the node's own parameters (a fixed name of templates/_conventions.md where its table binds one to
  that parameter, else 「节点 id_参数名」), then the reading nodes' files, then `unit` / `fps` of the 3D settings.

A wrapped graph that does not read cleanly (a wire the graph refuses, an interface check_exposed faults) is not
offered: what is listed always runs.
"""

from __future__ import annotations

import re
from functools import lru_cache

PREFIX = "node~"
SKIPPED_CATEGORIES = ("flow", "list", "value", "math")
_ID = re.compile(r"[^a-z0-9_]+")


def tool_id(type_id: str) -> str:
    return PREFIX + type_id


def type_of_tool(tid: str) -> str:
    return tid[len(PREFIX):] if tid.startswith(PREFIX) else ""


def _node_id(word: str, taken: set[str]) -> str:
    base = _ID.sub("_", word.lower()).strip("_") or "node"
    if not base[0].isalpha():
        base = "n" + base
    if len(base) < 3:
        base += "_in"
    nid, i = base, 1
    while nid in taken:
        i += 1
        nid = f"{base}_{i}"
    taken.add(nid)
    return nid


def readers() -> list[tuple[str, type]]:
    """The reading node types (those declaring `reads`), lowest rank first, the registry's order between equals."""
    from ..nodes import node_types

    found = [(k, t) for k, t in node_types().items() if getattr(t, "reads", None) is not None]
    return sorted(found, key=lambda kt: kt[1].reads.rank)


def _gives(t) -> list[tuple[str, str]]:
    """(output port, data type) a reading node reads its file into: its declaration's, else its declared data outputs."""
    from ..nodes.applies import all_outputs

    return list(t.reads.gives) or [(p.name, p.type) for p in all_outputs(t) if not p.type.startswith("value")]


def _reader_for(port_type: str):
    """(reading node type, its output port) for data of this port's type, or None: for each of the port's
    alternatives in order, the first reader (by rank) with an output of it that the port takes."""
    from ..data.types import accepts

    found = readers()
    for alt in port_type.split("|"):
        for k, t in found:
            port = next((name for name, given in _gives(t) if accepts(alt, given) and accepts(port_type, given)), None)
            if port is not None:
                return k, port
    return None


def _settings_for(data_type: str):
    """The first output-settings node type (no input-making table) with an input taking this type: (type, port)."""
    from ..data.types import accepts
    from ..nodes import node_types
    from ..nodes.output import OutputSettings

    best = None
    for k, t in node_types().items():
        if not (isinstance(t, type) and issubclass(t, OutputSettings)) or t.ports_from:
            continue
        port = next((p for p in t.inputs if accepts(p.type, data_type)), None)
        if port is None:
            continue
        full = sum(w.how == "full" for w in t.writes.values()) if t.writes else 0
        if best is None or full > best[0]:
            best = (full, k, port.name)
    return (best[1], best[2]) if best else None


def _fixed_names() -> dict[tuple[str, str], str]:
    from ..config import TEMPLATES_DIR
    from .conventions import read_table

    path = TEMPLATES_DIR / "_conventions.md"
    if not path.is_file():
        return {}
    out: dict[tuple[str, str], str] = {}
    for name, row in read_table(path).items():
        for target in row["targets"]:
            out.setdefault(target, name)
    return out


def wrap(type_id: str) -> dict | None:
    """The graph that runs one node as a tool, or None when the node is not offered (the module docstring)."""
    from ..nodes import node_types
    from ..nodes.applies import all_outputs
    from ..nodes.output import OutputSettings
    from .conventions import derived_name
    from .graph import SCHEMA, Graph, GraphError
    from .templates import check_exposed

    types = node_types()
    t = types.get(type_id)
    if t is None or t.delivers or (isinstance(t, type) and issubclass(t, OutputSettings)) or t.ports_from:
        return None
    if not getattr(t, "tool", True) or getattr(t, "reads", None) is not None:
        return None
    if getattr(t, "category", "") in SKIPPED_CATEGORIES or any(s["widget"] in ("file", "sequence") for s in t.param_specs()):
        return None
    fixed = _fixed_names()
    taken: set[str] = {"deliver"}
    main = _node_id(type_id.rsplit(".", 1)[-1], taken)
    nodes = [{"id": main, "type": type_id, "params": {}}]
    edges: list[dict] = []
    exposed: list[dict] = []
    names: set[str] = set()

    def expose(name: str, target: str, label: str) -> None:
        if name in names:
            return
        names.add(name)
        exposed.append({"name": name, "label": label, "target": target})

    for spec in t.interface_specs():
        if spec["widget"] == "button":
            continue
        name = fixed.get((type_id, spec["name"]))
        expose(name if name and name not in names else derived_name(main, spec["name"], t), f"{main}.{spec['name']}",
               spec["label"])
    main_material = True  # the first reader of the main material takes the fixed names (Reads.main)
    for port in t.inputs:
        if port.optional:
            continue
        reader = _reader_for(port.type)
        if reader is None:
            return None
        rtype, rport = reader
        rid = _node_id(port.name, taken)
        nodes.append({"id": rid, "type": rtype, "params": {}})
        edges.append({"from": [rid, rport], "to": [main, port.name]})
        rt = types[rtype]
        reads = rt.reads
        specs = {s["name"]: s for s in rt.param_specs()}
        if reads.main and main_material:
            main_material = False
            for param in (reads.file, *reads.goes_with):
                if param not in specs:
                    continue
                name = fixed.get((rtype, param))
                expose(name if name and name not in names else f"{rid}_{param}", f"{rid}.{param}",
                       port.label if param == reads.file else specs[param]["label"])
        else:
            expose(f"{rid}_{reads.file}", f"{rid}.{reads.file}", port.label)
    scene_ports, others = [], []
    for out in all_outputs(t):
        if out.type.startswith("scene"):
            scene_ports.append(out)
        elif not out.type.startswith(("value", "files")):
            others.append(out)
    settings: list[str] = []
    if scene_ports:
        found = _settings_for("scene")
        if found:
            sid = _node_id("scene", taken)
            nodes.append({"id": sid, "type": found[0], "params": {"name": "scene"}})
            for out in scene_ports:
                edges.append({"from": [main, out.name], "to": [sid, found[1]]})
            settings.append(sid)
            st = types[found[0]]
            for param in ("unit", "fps"):
                if param in st.Params.model_fields and fixed.get((found[0], param)) == param:
                    spec = next(s for s in st.param_specs() if s["name"] == param)
                    expose(param, f"{sid}.{param}", spec["label"])
    for out in others:
        found = _settings_for(out.type)
        if not found:
            continue
        sid = _node_id(f"{out.name}_out", taken)
        st = types[found[0]]
        params = {"name": out.name} if "name" in st.Params.model_fields else {}
        nodes.append({"id": sid, "type": found[0], "params": params})
        edges.append({"from": [main, out.name], "to": [sid, found[1]]})
        settings.append(sid)
    if not settings:
        return None
    nodes.append({"id": "deliver", "type": next(k for k, v in types.items() if v.delivers), "params": {}})
    for sid in settings:
        edges.append({"from": [sid, "files"], "to": ["deliver", "files"]})
    data = {"schema": SCHEMA, "meta": {"name": t.subtitle, "intro": t.description}, "exposed": exposed,
            "nodes": nodes, "edges": edges}
    try:
        graph = Graph.from_json(data)
    except GraphError:
        return None
    waits = {w[2] for w in graph.waiting}  # a reading node's kind waits for its file to be picked: not a fault
    if any(problems and nid not in waits for nid, problems in graph.wiring.items()):
        return None
    if check_exposed(data):
        return None
    return data


@lru_cache(maxsize=4)
def _all(registry: tuple) -> tuple:
    import traceback

    out = []
    for type_id in registry[0]:
        try:
            data = wrap(type_id)
        except Exception:  # noqa: BLE001 - one node that cannot be wrapped must not hide the others
            from .. import logs
            from ..messages import Msg

            logs.say(logs.get("farm"), Msg("W-TOOL-WRAPFAILED", type=type_id),
                     traceback.format_exc(), about=type_id)
            continue
        if data is not None:
            out.append((type_id, data))
    return tuple(out)


def node_tools() -> list[tuple[str, dict]]:
    """Every node offered as a tool: (its type id, its wrapped graph), in the registry's order. Worked out once per
    registry (nodes/registry.py registry_key: the types loaded and their words)."""
    from ..nodes.registry import registry_key

    return list(_all(registry_key()))
