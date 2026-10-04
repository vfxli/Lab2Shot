"""Delivery formats chosen by whoever submits (B3): `POST /api/jobs` with `formats: {kind: format}` — a kind of 3D data
(data/types.py SCENE_KINDS) and the format it is wanted in — rewrites that job's graph before it is queued, so what the
job records is what it cooks.

A format is a 3D output-settings node type (nodes/output.py OutputSettings with `writes`), named by the Format it
declares (`format`: its short name and how a person reads it); its type id and its label name nothing, so renaming
either leaves the contract as it is (`lab2shot check formats`: every settings node declares one, no name twice).
`formats()` lists them with what each holds of every kind (Writes: full / static / no), the table a client picks from.

The rewrite (`with_formats`), for every 3D output-settings node of the graph: the kinds each wire into it carries
(Graph.scene_kinds, known before anything cooks) go to the format asked for them (a kind not asked keeps the node's own
format). A node whose kinds all go to one other format becomes a node of that format in place — same id, its wires,
and the parameters both have (名字, 帧率, 单位) kept; the interface entries of parameters the new one lacks leave the
interface. When the kinds of one wire go to several formats, 「按种类取出」 (split_scene) splits the wire and each format
gets its own settings node beside it (名字 + _<format>, kept distinct under the same 「输出」). One content is delivered
once: a kind that would move into a format another settings node of the same 「输出」 already writes it in, from the
same source, is left out of the move (a node left with nothing to write goes). A kind the format asked
for cannot hold (Writes `no`, or a deforming 模型 into a format that holds still ones only) refuses the job with the
kind, the format and the format's own reason: never another format on the quiet.
"""

from __future__ import annotations

import copy

from ..errors import Invalid
from .. import i18n
from ..messages import Msg


def format_of(type_id: str) -> str:
    """The format name of a 3D output-settings node type ("" for any other node type)."""
    return _names().get(type_id, "")


def _settings_types() -> dict:
    """The 3D output-settings node types that declare their format: the formats a client may ask for."""
    from ..nodes import node_types
    from ..nodes.output import OutputSettings

    return {k: t for k, t in node_types().items()
            if isinstance(t, type) and issubclass(t, OutputSettings) and t.writes and t.format is not None}


def _names() -> dict[str, str]:
    return {k: t.format.name for k, t in _settings_types().items()}


def _type_of(fmt: str):
    return next((t for t in _settings_types().values() if t.format.name == fmt), None)


def formats() -> list[dict]:
    """Every format a client may ask for: {name, label (both the node's Format), node (its type id), writes: {kind:
    full / static / no}}."""
    return [{"name": t.format.name, "label": t.format.label, "node": k,
             "writes": {kind: w.how for kind, w in t.writes.items()}} for k, t in _settings_types().items()]


def _check_request(asked: dict) -> None:
    from ..data.types import SCENE_KINDS

    known = {f["name"] for f in formats()}
    for kind, fmt in asked.items():
        if kind not in SCENE_KINDS:
            raise Invalid(Msg("E-FORMAT-NOKIND", kind=str(kind), kinds=i18n.Both.of(lambda: i18n.separator().join(SCENE_KINDS))))
        if fmt not in known:
            raise Invalid(Msg("E-FORMAT-NOFORMAT", format=str(fmt), formats=i18n.Both.of(lambda: i18n.separator().join(sorted(known)))))


def _refuse_unwritable(kind: str, deforming: bool, fmt: str) -> None:
    from ..data.types import DEFORMING, kind_label

    t = _type_of(fmt)
    w = t.writes_of(kind)
    probe = DEFORMING if kind == "model" and deforming else kind
    if not w.takes(probe):
        raise Invalid(Msg("E-FORMAT-CANTWRITE", kind=kind_label(probe), format=fmt, why=w.note or "-"))


def with_formats(data: dict, asked: dict[str, str]) -> dict:
    """The graph with its 3D deliveries in the formats asked (the module docstring); the graph itself when nothing
    changes. Raises Invalid (E-FORMAT-…) when a kind cannot be written as asked."""
    from ..data.types import DEFORMING
    from ..nodes.port import PARAM
    from .graph import Graph

    asked = {str(k): str(v) for k, v in (asked or {}).items()}
    if not asked:
        return data
    _check_request(asked)
    graph = Graph.from_json(data)
    out = copy.deepcopy(data)
    nodes = {n["id"]: n for n in out["nodes"]}
    names = _names()
    changed = False
    settings = {}  # settings node -> (its format, its wires: (port, source, source port, kinds, deforming))
    for sid, gnode in graph.nodes.items():
        own = names.get(gnode.type.id)
        if not own:
            continue
        ports = [p for p in graph.input_ports(sid) if not p.name.startswith(PARAM)]
        # every wire into the node with the kinds it carries and where each kind goes
        wires = []
        for p in ports:
            for src, sport in graph.inputs.get((sid, p.name), []):
                kinds = graph.scene_kinds(src, sport)
                deforming = DEFORMING in kinds
                plain = sorted(k.split(".")[0] for k in kinds if k != DEFORMING)
                wires.append((p.name, src, sport, plain, deforming))
        settings[sid] = (own, wires)
    # One delivery of each content (按内容去重): the same kind from the same source, in the same format, into the same
    # 「输出」 is delivered once. A kind a settings node already writes in its own format stays there; one that would
    # move into a format another settings node already writes it in, for the same 「输出」, is left out of the move
    # (Pi3: the camera into usd, asked as nuke_camera, beside the template's own nuke_camera of the same camera).
    delivered: dict[tuple, list[tuple[str, frozenset]]] = {}

    def downstream(sid: str) -> frozenset:
        return frozenset((dst, dport) for (src, _sp), (dst, dport) in _leaving_wires(graph, sid))

    for sid, (own, wires) in settings.items():
        for _p, src, sport, kinds, _d in wires:
            for k in kinds:
                if asked.get(k, own) == own:
                    delivered.setdefault((src, sport, k, own), []).append((sid, downstream(sid)))
    for sid, (own, wires) in settings.items():
        wanted = {asked.get(k, own) for _p, _s, _sp, kinds, _d in wires for k in kinds}
        if wanted <= {own}:
            continue
        for _p, _s, _sp, kinds, deforming in wires:
            for k in kinds:
                if asked.get(k, own) != own:
                    _refuse_unwritable(k, deforming, asked[k])
        dropped: dict[tuple, set] = {}
        mine = downstream(sid)
        for _p, src, sport, kinds, _d in wires:
            for k in kinds:
                fmt = asked.get(k, own)
                if fmt == own:
                    continue
                key = (src, sport, k, fmt)
                if any(other != sid and there & mine for other, there in delivered.get(key, [])):
                    dropped.setdefault((src, sport), set()).add(k)
                else:
                    delivered.setdefault(key, []).append((sid, mine))
        changed = True
        _rewrite(out, nodes, graph, sid, own, wires, asked, dropped)
    return out if changed else data


def _leaving_wires(graph, node_id: str) -> list:
    return [((src, sport), (dst, dport)) for (dst, dport), wires in graph.inputs.items()
            for src, sport in wires if src == node_id]


def _unique_id(nodes: dict, base: str) -> str:
    i, nid = 1, base
    while nid in nodes:
        i += 1
        nid = f"{base}_{i}"
    return nid


def _carried_params(old: dict, new_type) -> dict:
    """The parameters of the old settings node the new type also has, each kept only when the new type takes it."""
    keep = {k: v for k, v in (old.get("params") or {}).items() if k in new_type.Params.model_fields}
    good = {}
    for k, v in keep.items():
        try:
            new_type.load_params({**good, k: v})
            good[k] = v
        except ValueError:
            continue
    return good


def _rewrite(out: dict, nodes: dict, graph, sid: str, own: str, wires: list, asked: dict, dropped: dict) -> None:
    from ..nodes import node_types
    from ..nodes.formats import PORT_OF
    from ..nodes.output import TAKE, name_key
    from ..nodes.port import PARAM

    types = node_types()
    old = nodes[sid]
    edges = out.setdefault("edges", [])
    # the wires into the node now, its files leaving it, its parameters driven by wires
    into = [e for e in edges if e["to"][0] == sid and not e["to"][1].startswith(PARAM)]
    param_wires = [e for e in edges if e["to"][0] == sid and e["to"][1].startswith(PARAM)]
    leaving = [e for e in edges if e["from"][0] == sid]
    edges[:] = [e for e in edges if e not in into]
    groups: dict[str, list] = {}  # format -> [(source node, its port)] to wire into that format's node
    for _port, src, sport, kinds, _deforming in wires:
        gone = dropped.get((src, sport), set())  # delivered by another settings node already (with_formats)
        kept = [k for k in kinds if k not in gone]
        by_format: dict[str, list[str]] = {}
        for k in kept:
            by_format.setdefault(asked.get(k, own), []).append(k)
        if not kinds:  # nothing known about it: it stays with the node, whatever the node becomes
            by_format = {"": []}
        elif not kept:  # all of it delivered elsewhere: the wire goes
            continue
        if len(by_format) == 1 and not gone:
            groups.setdefault(next(iter(by_format)), []).append((src, sport))
            continue
        from ..data.types import is_list

        if is_list(graph.output_type(src, sport)):
            raise Invalid(Msg("E-FORMAT-LIST", node=graph.nodes[sid].label))
        take = _unique_id(nodes, f"{sid}_take")
        nodes[take] = {"id": take, "type": TAKE, "params": {}, "ui": copy.deepcopy(old.get("ui") or {})}
        out["nodes"].append(nodes[take])
        edges.append({"from": [src, sport], "to": [take, "scene"]})
        for fmt, ks in by_format.items():
            for k in ks:
                groups.setdefault(fmt, []).append((take, PORT_OF[k]))
    stay = groups.pop("", [])
    if not groups and not stay:  # everything it wrote is delivered by other settings nodes: it goes
        out["nodes"][:] = [n for n in out["nodes"] if n["id"] != sid]
        nodes.pop(sid, None)
        edges[:] = [e for e in edges if sid not in (e["from"][0], e["to"][0])]
        _drop_exposed(out, sid, set())
        return
    keep_own = own in groups
    order = ([own] if keep_own else []) + [f for f in groups if f != own]
    downstream = {tuple(e["to"]) for e in leaving}
    taken_names = {name_key(str((nodes[e["from"][0]].get("params") or {}).get("name") or ""))
                   for e in edges if tuple(e["to"]) in downstream and e["from"][0] != sid}
    base_name = str((old.get("params") or {}).get("name") or types[old["type"]].Params.model_fields["name"].default)
    for i, fmt in enumerate(order):
        new_type = _type_of(fmt)
        port = next(p.name for p in new_type.inputs)
        sources = groups[fmt] + (stay if i == 0 else [])
        if i == 0:
            target = sid
            if fmt != own:  # in place: the same node, now of the format asked
                old["type"] = new_type.id
                old.pop("label", None)  # its own type's name now (a label of the old format would mislead)
                old["params"] = _carried_params(old, new_type)
                old["promoted"] = [p for p in old.get("promoted") or [] if p in new_type.Params.model_fields]
                for e in param_wires:
                    if e["to"][1].removeprefix(PARAM) not in new_type.Params.model_fields:
                        edges.remove(e)
                _drop_exposed(out, sid, {s["name"] for s in new_type.interface_specs()})
            taken_names.add(name_key(base_name))
        else:
            target = _unique_id(nodes, f"{sid}_{fmt}")
            name = f"{base_name}_{fmt}"
            n = 2
            while name_key(name) in taken_names:
                name, n = f"{base_name}_{fmt}_{n}", n + 1
            taken_names.add(name_key(name))
            params = {**_carried_params(old, new_type), "name": name}
            nodes[target] = {"id": target, "type": new_type.id, "params": params,
                             "promoted": [p for p in old.get("promoted") or [] if p in new_type.Params.model_fields and p != "name"],
                             "ui": copy.deepcopy(old.get("ui") or {})}
            out["nodes"].append(nodes[target])
            for e in param_wires:
                pname = e["to"][1].removeprefix(PARAM)
                if pname in nodes[target]["promoted"] and e in edges:
                    edges.append({"from": list(e["from"]), "to": [target, e["to"][1]]})
            for e in leaving:
                edges.append({"from": [target, e["from"][1]], "to": list(e["to"])})
        for src, sport in sources:
            edges.append({"from": [src, sport], "to": [target, port]})


def _drop_exposed(out: dict, node_id: str, kept: set[str]) -> None:
    """Interface entries of the node's parameters it no longer has leave the interface (a group left empty goes too)."""
    from .templates import is_group, split_target, targets_of

    def walk(entries):
        result = []
        for x in entries if isinstance(entries, list) else []:
            if is_group(x):
                children = walk(x.get("children"))
                if children:
                    result.append({**x, "children": children})
                continue
            keys = targets_of(x)
            if keys:  # an entry driving several parameters keeps the ones still there (none: it goes)
                left = [k for k in keys if not ((t := split_target(k))[0] == node_id and t[1] not in kept)]
                if not left:
                    continue
                if len(left) < len(keys):
                    x = {**x, "target": left[0] if len(left) == 1 else left}
            result.append(x)
        return result

    if isinstance(out.get("exposed"), list):
        out["exposed"] = walk(out["exposed"])
