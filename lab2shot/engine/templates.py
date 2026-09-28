"""Node graph files, the built-in templates, and templates used as tools.

A graph file is plain JSON with "schema": "lab2shot.graph/1". Users keep graph files on their own machines: the editor
opens and saves them in the browser, and `lab2shot cook` reads them locally. The server holds only the templates:
graph files (templates/, adapters/<name>/templates/, work/users/<name>/templates/; see lab2shot/library.py) whose
"exposed" list names the parameters intended to be set externally (the web UI, the command line, DCC plugins).

A template is used as a tool: set its exposed parameters, cook its 「输出」 nodes, and receive what they pack (one zip
per 「输出」, and the same files unpacked on the server, lab2shot/transfer/outputs.py). File parameters are handled by the
client, which uploads the given input files and fetches the results (see lab2shot/client.py); file_params() identifies
those parameters.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from ..errors import NotFound
from ..messages import Msg
from .graph import SCHEMA, Graph, GraphError


def parse_graph(text: str, name: str) -> dict:
    """Parse a graph file's content (`name`: the file name, used in messages)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphError(Msg("E-GRAPH-BADJSON", file=name, detail=exc)) from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise GraphError(Msg("E-GRAPH-SCHEMA", file=name, schema=SCHEMA))
    return data


def load_graph(path: str | Path) -> dict:
    """Load a graph file from the local machine (a template, or a file given on the command line)."""
    p = Path(path).expanduser()
    if p.suffix.lower() != ".json" or not p.is_file():
        raise GraphError(Msg("E-GRAPH-NOTFILE", path=str(p)))
    return parse_graph(p.read_text(encoding="utf-8"), p.name)


def templates() -> list[dict]:
    """Return every preset card (the project's templates/ and each adapter's) via lab2shot/library.py presets(), the
    single loader of template files."""
    from ..library import presets

    return presets()


def order(cards: list[dict]) -> list[dict]:
    """Sort templates into the canonical order. The server determines the order; pages only render it. Cards are
    ordered by category and subcategory as in the templates tree (lab2shot/categories.py), then by name within a
    subcategory, with cards that `follow` another card placed directly after it.

    `meta.follows` is the only declared relation between two cards: 「同一条流程，逐人 / 逐段各做一遍」 (多人, 批量).
    It must be declared explicitly and is never parsed from names, since names are written for people and may be
    reworded. Its sole use is this ordering; pages never receive it."""
    from ..categories import templates as tree

    cats = tree.tree()
    rank = {c["id"]: (ci, -1) for ci, c in enumerate(cats)}  # cards placed directly in the category come before its subcategories
    rank.update({s["id"]: (ci, si) for ci, c in enumerate(cats) for si, s in enumerate(c["subs"])})
    by_id = {t["id"]: t for t in cards}
    last = (len(cats), 0)  # 未分类 sorts last

    def lead(t: dict) -> dict:
        """Return the card that `t` is placed after (`t` itself when it follows none). A card whose lead is absent
        from the list (e.g. not visible to the account) is placed on its own rather than dropped."""
        seen = {t["id"]}
        while (f := t.get("follows")) and f in by_id and f not in seen:
            seen.add(f)
            t = by_id[f]
        return t

    def key(t: dict) -> tuple:
        head = lead(t)
        return (*rank.get(head["deliverable"], last), head["name"], t is not head, t["name"])

    return sorted(cards, key=key)


def core_project(data: dict, types) -> str:
    """The card's core third-party project: walking upstream from the graph's output nodes (「输出」, or an
    output-settings node of any format, a subclass of OutputSettings), the project of the first third-party node met.
    That node produces the card's deliverable; a card involving several projects takes only this one. Judged from the
    node graph alone, never from the card's category (the administrator sets categories; they say nothing of the
    deliverable). A third-party node wired to no output does not count. A card without an output node walks up from
    every end node; with no third-party node above the ends, it takes the project of the graph's first third-party node;
    a card made only of core nodes has no project, and so no year (server/app.py)."""
    from ..nodes.output import OutputSettings

    nodes = data.get("nodes", [])
    runtime = {n["id"]: types[n["type"]].runtime for n in nodes if n["type"] in types}
    third = [n["id"] for n in nodes if runtime.get(n["id"], "core") != "core"]
    feeds: dict[str, list[str]] = {}
    sources = set()
    for e in data.get("edges", []):
        feeds.setdefault(e["to"][0], []).append(e["from"][0])
        sources.add(e["from"][0])
    sinks = [n for n in nodes if n["id"] not in sources]  # nodes with no outgoing wires
    writer = {n["id"] for n in nodes if n["type"] in types and isinstance(types[n["type"]], type) and issubclass(types[n["type"]], OutputSettings)}
    is_out = lambda n: n["type"].startswith("core.output") or n["id"] in writer  # noqa: E731
    queue = [n["id"] for n in sinks if is_out(n)] or [n["id"] for n in sinks]
    third = [nid for nid in third if nid not in writer]  # a format writer (FBX, Alembic) only writes the deliverable; it does not produce it
    seen = set(queue)
    while queue:
        nid = queue.pop(0)
        if runtime.get(nid, "core") != "core" and nid not in writer:
            return runtime[nid]
        for up in feeds.get(nid, []):
            if up not in seen:
                seen.add(up)
                queue.append(up)
    return runtime[third[0]] if third else ""


def template(ref: str, among: list[dict] | None = None) -> dict:
    """Find a template in `among` (default: all templates) by id (e.g. sam_3d_body_moving_camera) or display name."""
    for t in templates() if among is None else among:
        if ref in (t["id"], t["name"]):
            return t
    raise NotFound(Msg("E-TEMPLATE-NOTFOUND", name=ref))


# ------------------------------------------------------------------ templates as tools


def wired_from(data: dict, node_id: str, param: str) -> str:
    """Describe the node and output whose wire drives the parameter, e.g. 「AnyCalib 镜头标定」的「Focal Length」;
    "" when the parameter is set directly."""
    from ..nodes import node_types
    from ..nodes.applies import all_outputs
    from ..nodes.port import PARAM

    edge = next((e for e in data.get("edges", []) if e["to"] == [node_id, PARAM + param]), None)
    if edge is None:
        return ""
    src = next((n for n in data.get("nodes", []) if n["id"] == edge["from"][0]), None)
    src_type = node_types().get(src["type"]) if src else None
    label = (src or {}).get("label") or (src_type.label if src_type else edge["from"][0])
    port = next((p.label for p in all_outputs(src_type) if p.name == edge["from"][1]), edge["from"][1]) if src_type else edge["from"][1]
    return f"「{label}」的「{port}」"


def exposed_params(data: dict) -> list[dict]:
    """Return the template's exposed parameters, each with the target parameter's description (type, default,
    choices) and, for a parameter driven by a wire, its source ("wired"; clients show it as not settable)."""
    from ..nodes import node_types

    registry = node_types()
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    out = []
    for x in data.get("exposed", []):
        node_id, _, param = x["target"].partition(".")
        node = nodes.get(node_id)
        node_type = registry.get(node["type"]) if node else None
        spec = next((p for p in node_type.param_specs() if p["name"] == param), None) if node_type else None
        value = (node or {}).get("params", {}).get(param, spec["default"] if spec else None)
        out.append({**x, "value": value, "param": spec, "wired": wired_from(data, node_id, param)})
    return out


# parameter widget -> file direction: every file parameter is one the client uploads (what 「输出」 packs is fetched
# from its task: lab2shot/transfer/outputs.py)
FILE_WIDGETS = {"file": "in", "sequence": "in"}


def file_params(data: dict) -> list[dict]:
    """Return every file parameter of the graph: key (node.param), exposed name if any, direction ("in": the client
    uploads a file for it), widget and label."""
    from ..nodes import node_types

    registry = node_types()
    exposed = {x["target"]: x["name"] for x in data.get("exposed", [])}
    out = []
    for n in data.get("nodes", []):
        node_type = registry.get(n["type"])
        for spec in node_type.param_specs() if node_type else []:
            if spec["widget"] in FILE_WIDGETS:
                key = f"{n['id']}.{spec['name']}"
                out.append({"key": key, "name": exposed.get(key), "direction": FILE_WIDGETS[spec["widget"]],
                            "widget": spec["widget"], "label": spec["label"], "node": n.get("label") or node_type.label})
    return out


def apply_values(data: dict, values: dict[str, Any]) -> dict:
    """Return a copy of the graph with the given parameters set. Keys are exposed names, or node.param otherwise. A
    parameter driven by a wire cannot be set, since the wire supplies its value (the GraphError names the source).
    Parameters derived from the ones set here are computed as the editor does (NodeDef.derive: e.g. selecting the
    only camera of a chosen file, listing an EXR's layers), unless they are also set explicitly."""
    from ..nodes import node_types

    data = copy.deepcopy(data)
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    by_name = {x["name"]: x["target"] for x in data.get("exposed", [])}
    changed: dict[str, set[str]] = {}
    for key, value in values.items():
        target = by_name.get(key, key)
        node_id, dot, param = target.partition(".")
        if not dot or node_id not in nodes:
            raise GraphError(Msg("E-TEMPLATE-NOPARAM", key=key, names=list(by_name)) if by_name else Msg("E-TEMPLATE-NOEXPOSED", key=key))
        if wire := wired_from(data, node_id, param):
            raise GraphError(Msg("E-TEMPLATE-WIRED", key=key, wire=wire))
        nodes[node_id].setdefault("params", {})[param] = value
        changed.setdefault(node_id, set()).add(param)
    for node_id, names in changed.items():  # derived parameters, computed as in the editor (NodeDef.derive)
        t = node_types().get(nodes[node_id]["type"])
        follows = {s["name"] for s in t.param_specs() if set(s["derived_from"]) & names} - names if t else set()
        if follows:
            try:
                derived = t.derive(t.load_params(nodes[node_id]["params"]))
            except ValueError as exc:
                raise GraphError(Msg("E-TEMPLATE-DERIVE", node=nodes[node_id].get("label") or t.label, reason=exc)) from exc
            nodes[node_id]["params"].update({k: v for k, v in derived.items() if k in follows})
    return data


def pick_targets(graph: Graph, targets: list[str] | None = None) -> list[str]:
    """Return the nodes to cook: `targets` (Graph.cook_targets), or by default every 「输出」."""
    if targets is None:
        targets = graph.deliveries()
        if not targets:
            raise GraphError(Msg("B-TEMPLATE-NOTARGET"))
    for t in targets:
        if t not in graph.nodes:
            raise GraphError(Msg("E-GRAPH-NONODE", node=t))
    return list(dict.fromkeys(c for t in targets for c in graph.cook_targets(t)))

