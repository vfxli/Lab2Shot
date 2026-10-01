"""Node graph files, the built-in templates, and templates used as tools.

A graph file is plain JSON with "schema": "lab2shot.graph/1". Users keep graph files on their own machines: the editor
opens and saves them in the browser, and `lab2shot cook` reads them locally. The server holds only the templates:
graph files (templates/, adapters/<name>/templates/, work/users/<name>/templates/; see lab2shot/library.py) whose
"exposed" list names the parameters intended to be set externally (the web UI, the command line, DCC plugins).

"exposed" is the template's parameter interface, a tree (Houdini's Edit Parameter Interface): a list whose entries are
parameters {name, label, target, widget?, options?, hide_when?, disable_when?, show_on_change?} and groups {kind: "group", label, collapsed, children: [...]},
a group's children again such entries (a subgroup is a group in a group). An older file's flat list of parameters is
the same thing with no groups: read as it is, nothing to convert. `name` is the parameter's outside name (lab2shot cook
--set, DCC plugins: apply_values), `target` "node.param"; `widget` overrides how the page shows it: "menu" (a pull-down
of `options` [{value, label, hide_when?}], values of the target's own type; an option's own `hide_when` true leaves it out
of the pull-down — the parameter's value is kept even when it is the one left out) or "checkbox" (a boolean target, or an integer one
whose options are exactly 0 and 1); `hide_when` and `disable_when` condition expressions (engine/conditions.py;
Houdini's Hide When and Disable When): `hide_when` true hides it (a group all of whose entries are hidden hides too),
`disable_when` true greys it out. `show_on_change` (true / false, default false): once the parameter is changed in the
parameter panel's interface tree (a template's parameters, 应用模式), the viewer shows its node, as a double click on
it would (webui/src/editor/ParamPanel.tsx). An older file's `when` (true: may be edited) reads as disable_when = not (when).
A target may also be a button (nodes/params.py Button: "<node id>.cook" 「计算」 on every node, "<output id>.download"
「下载」 on 「输出」): it takes no widget or options, and no value.
check_exposed says what is wrong with one, each problem with where it is.

A template is used as a tool: set its exposed parameters, cook its 「输出」 nodes, and receive what they pack (one zip
per 「输出」, and the same files unpacked on the server, lab2shot/transfer/outputs.py). File parameters are handled by the
client, which uploads the given input files and fetches the results (see lab2shot/client.py); file_params() identifies
those parameters.

Setting values from outside (apply_values: lab2shot cook, DCC plugins) keeps to what the interface offers where it says
so: a parameter the interface shows as a menu takes only its menu's values (E-TEMPLATE-NOTOPTION), besides what the
target parameter itself accepts. An entry's own Hide When / Disable When only decides how the page shows it: a client may
set a hidden or greyed parameter (the page would let it be set once the condition changes), so it is not checked here.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
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
    """The card's main third-party project (the year its card shows, server/app.py): the one its file declares
    (meta.project: a card whose name leads with another project than the graph would tell, `lab2shot check templates`
    holds it to a project of a node on the card); otherwise among the third-party nodes its
    output is made from (walking up from the output nodes: 「输出」 and every format's output settings; a card without
    one, from every end), the one that takes the most of the others' results — a solver after its detector and its
    camera, a matting after its coarse mask — and, on a tie, the nearest to the output. Only by declaration, never by a
    node's id: a node that only reads a file (ImportNode), writes one (OutputSettings) or finishes another's result
    (NodeDef.finishes: edge unmixing) is never it. Judged from the graph alone, never from the card's category. A card
    with no such node above its ends takes the graph's first one; one made only of core nodes has no project."""
    from ..nodes.formats import ImportNode
    from ..nodes.output import OutputSettings
    from .graph import walk

    if isinstance(declared := (data.get("meta") or {}).get("project"), str) and declared:
        return declared
    nodes = [n for n in data.get("nodes", []) if n.get("type") in types]
    kind = {n["id"]: types[n["type"]] for n in nodes}
    feeds: dict[str, list[str]] = {}
    fed: set[str] = set()
    for e in data.get("edges", []):
        feeds.setdefault(e["to"][0], []).append(e["from"][0])
        fed.add(e["from"][0])
    main = lambda nid: (kind[nid].runtime != "core" and not issubclass(kind[nid], (ImportNode, OutputSettings))  # noqa: E731
                        and not kind[nid].finishes)
    ends = [nid for nid in kind if nid not in fed]
    outs = [nid for nid in ends if kind[nid].delivers or issubclass(kind[nid], OutputSettings)] or ends
    distance, ring = {nid: 0 for nid in outs}, list(outs)  # how many wires up from an output (for a tie)
    while ring:
        nxt = []
        for nid in ring:
            for up in feeds.get(nid, []):
                if up in kind and up not in distance:
                    distance[up] = distance[nid] + 1
                    nxt.append(up)
        ring = nxt
    candidates = [nid for nid in distance if main(nid)]
    if not candidates:
        return next((kind[nid].runtime for nid in kind if main(nid)), "")
    above = {nid: sum(1 for up in walk([nid], lambda n: feeds.get(n, [])) if up != nid and up in candidates) for nid in candidates}
    best = min(candidates, key=lambda nid: (-above[nid], distance[nid]))
    return kind[best].runtime


def template(ref: str, among: list[dict] | None = None) -> dict:
    """Find a template in `among` (default: all templates) by its card id (admin~sam_3d_body_moving_camera), its display
    name, or its file name alone (sam_3d_body_moving_camera, the part after the last ~) when only one card has it
    (E-TEMPLATE-AMBIGUOUS names them otherwise)."""
    cards = templates() if among is None else among
    for t in cards:
        if ref in (t["id"], t["name"]):
            return t
    short = [t for t in cards if t["id"].rsplit("~", 1)[-1] == ref]
    if len(short) == 1:
        return short[0]
    if short:
        raise NotFound(Msg("E-TEMPLATE-AMBIGUOUS", name=ref, ids="、".join(t["id"] for t in short)))
    raise NotFound(Msg("E-TEMPLATE-NOTFOUND", name=ref))


# ------------------------------------------------------------------ templates as tools


def wire_source(data: dict, node_id: str, param: str):
    """The output whose wire drives the parameter: (「节点」的「口」 as a message names it, the output Port or None when it
    can't be found), or None when the parameter is set directly. The output is looked up among the node's outputs as
    its parameters make them (output_ports: 「读取序列」's layers come from its file), not only the declared."""
    from ..nodes import node_types
    from ..nodes.applies import all_outputs, output_ports
    from ..nodes.port import PARAM

    edge = next((e for e in data.get("edges", []) if e["to"] == [node_id, PARAM + param]), None)
    if edge is None:
        return None
    src = next((n for n in data.get("nodes", []) if n["id"] == edge["from"][0]), None)
    src_type = node_types().get(src["type"]) if src else None
    label = (src or {}).get("label") or (src_type.label if src_type else edge["from"][0])
    port = None
    if src_type is not None:
        try:
            ports = output_ports(src_type, src_type.load_params(src.get("params") or {}))
        except (OSError, ValueError):
            ports = ()
        port = next((p for p in (*ports, *all_outputs(src_type)) if p.name == edge["from"][1]), None)
    return f"「{label}」的「{port.label if port else edge['from'][1]}」", port


def wired_from(data: dict, node_id: str, param: str) -> str:
    """Describe the node and output whose wire drives the parameter, e.g. 「AnyCalib 镜头标定」的「Focal Length」;
    "" when the parameter is set directly."""
    got = wire_source(data, node_id, param)
    return got[0] if got else ""


def is_group(x: Any) -> bool:
    """An entry of the parameter interface that is a group (else a parameter)."""
    return isinstance(x, dict) and x.get("kind") == "group"


def exposed_items(data: dict) -> list[tuple[dict, list[str]]]:
    """Every parameter of the interface tree in the order it shows, each with the names of the groups it sits in
    (outermost first; [] at the top). Entries that are neither a group nor a parameter are passed over here
    (check_exposed says what is wrong with them)."""
    out: list[tuple[dict, list[str]]] = []

    def walk(entries: Any, path: list[str]) -> None:
        for x in entries if isinstance(entries, list) else []:
            if is_group(x):
                walk(x.get("children"), [*path, str(x.get("label") or "")])
            elif isinstance(x, dict) and isinstance(x.get("name"), str) and isinstance(x.get("target"), str):
                out.append((x, path))

    walk(data.get("exposed"), [])
    return out


def disable_when(x: dict) -> str | None:
    """A parameter's Disable When (true: greyed out), an older file's `when` read as its opposite: engine/conditions.py
    disable_when_of, the one rule (the page reads a file by the same one)."""
    from .conditions import disable_when_of

    return disable_when_of(x)


def exposed_params(data: dict, specs=None) -> list[dict]:
    """Return the template's exposed parameters in the order the interface tree shows them, one flat list (clients
    read each by `name`), each with the target parameter's description (type, default, choices), for a parameter
    driven by a wire its source ("wired"; clients show it as not settable, unless "fallback": that output may give
    nothing, Port.may_be_empty, so a value set here is used when it does — the rule apply_values keeps), and its place
    in the interface: `label` (its outside name when the file gives none), `group`
    (the names of the groups it sits in, outermost first) and `widget` / `options` / `hide_when` / `disable_when`
    (None when not set; an older file's `when` given as its disable_when). A button (param.widget "button": nodes/params.py Button, e.g. 「计算」, 「下载」) is listed like a
    parameter, with no value: it takes none (apply_values passes a value given to it over). `specs`: a node type's
    parameters as whoever asks sees them (server/access.py params_for); None all of them (NodeDef.interface_specs)."""
    from .graph import check_shape
    from ..nodes import node_types

    check_shape(data)  # a malformed file: E-GRAPH-SHAPE, as reading it says (never a KeyError)

    registry = node_types()
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    out = []
    for x, path in exposed_items(data):
        node_id, _, param = x["target"].partition(".")
        node = nodes.get(node_id)
        node_type = registry.get(node["type"]) if node else None
        listed = (specs(node_type) if specs else node_type.interface_specs()) if node_type else []
        spec = next((p for p in listed if p["name"] == param), None)
        button = bool(spec and spec["widget"] == "button")
        value = None if button else (node or {}).get("params", {}).get(param, spec["default"] if spec else None)
        item = {k: v for k, v in x.items() if k != "when"}
        wire = None if button else wire_source(data, node_id, param)
        out.append({"widget": None, "options": None, "show_on_change": False, **item, "label": str(x.get("label") or x["name"]), "hide_when": x.get("hide_when"), "disable_when": disable_when(x),
                    "group": path, "value": value, "param": spec, "wired": wire[0] if wire else "",
                    "fallback": bool(wire and wire[1] is not None and wire[1].may_be_empty)})
    return out


EXPOSED_WIDGETS = {"menu": "下拉", "checkbox": "复选框"}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _refuses(spec: dict, value: Any) -> str | None:
    """Why the target parameter would not take `value` (None: it would)."""
    if value is None:
        return None if spec["nullable"] else "这个参数不能为空"
    kind = spec["type"]
    fits = {"boolean": isinstance(value, bool), "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": _is_number(value), "string": isinstance(value, str)}.get(kind, False)
    if not fits:
        return {"boolean": "要 true 或 false", "integer": "要整数", "number": "要数字", "string": "要文字"}.get(kind, "这个参数不能用下拉")
    if spec.get("options") and not any(value == o and type(value) is type(o) or (_is_number(value) and _is_number(o) and value == o)
                                       for o in spec["options"]):
        return "只能是 " + " / ".join(map(str, spec["options"])) + " 之一"
    if _is_number(value):
        if spec.get("minimum") is not None and value < spec["minimum"]:
            return f"不能小于 {spec['minimum']:g}"
        if spec.get("maximum") is not None and value > spec["maximum"]:
            return f"不能大于 {spec['maximum']:g}"
    return None


# 参数界面的三条值规则（_refuses、_widget_refused、_never）：网页 platform/conditions.ts valueRefused / widgetRefused /
# neverValue 逐字相同（「编辑参数界面」即时标红用）
# （lab2shot check conditions）。改一边要改另一边。


def _widget_refused(widget: str, spec: dict, options: Any) -> str | None:
    """Why the target parameter cannot show as this widget ("menu" / "checkbox"; None: it can)."""
    if spec["widget"] == "button":
        return "按钮没有值"
    if spec["type"] == "array" or spec["widget"] in FILE_WIDGETS or spec["widget"] in ("table", "hierarchy"):
        return "这个参数不是单个值"
    if widget == "menu":
        return None if isinstance(options, list) and options else "下拉至少要有一项"
    if spec["type"] == "boolean":
        return None
    values = [o.get("value") for o in options] if isinstance(options, list) and all(isinstance(o, dict) for o in options) else None
    zero_one = values is not None and len(values) == 2 and all(_is_number(v) for v in values) and sorted(values) == [0, 1]
    return None if spec["type"] == "integer" and zero_one else "目标参数要是布尔，或者是整数且选项恰好是 0 和 1 两项"


def _never(name: str, value: Any, options: Any, spec: dict) -> str | None:
    """Why the exposed parameter `name` can never be `value` (None: it can): a menu's own values, else what the target
    takes (_refuses). A condition such as `cam_src != 7` that always holds is most likely a mistake."""
    from . import conditions

    if spec["widget"] == "button":
        return None
    if isinstance(options, list) and options and all(isinstance(o, dict) and "value" in o for o in options):
        if not any(conditions.same(value, o["value"]) for o in options):
            return (f"比的值不对：「{name}」只会是 " + " / ".join(json.dumps(o["value"], ensure_ascii=False) for o in options)
                    + f" 之一，不会是 {json.dumps(value, ensure_ascii=False)}")
        return None
    why = _refuses(spec, value)
    return f"比的值不对：「{name}」不会是 {json.dumps(value, ensure_ascii=False)}（{why}）" if why else None


def check_exposed(data: dict) -> list[Msg]:
    """What is wrong with the graph's parameter interface (the module docstring says what it holds), each problem
    naming where it is (「组 / 子组 / 显示名」): the tree's shape, a target that no longer exists, two parameters with
    one outside name, a widget the target cannot show as, a menu value the target would not take, a condition that
    does not read or uses a name no parameter has. [] when it is fine."""
    from .graph import check_shape
    from ..nodes import node_types
    from . import conditions

    check_shape(data)  # a malformed file: E-GRAPH-SHAPE, as reading it says (never a KeyError)

    registry = node_types()
    nodes = {n.get("id"): n for n in data.get("nodes", []) if isinstance(n, dict)}
    raw = data.get("exposed")
    if raw is None:
        return []
    if not isinstance(raw, list):
        return [Msg("E-EXPOSED-BAD", where="参数界面", why="exposed 要是一个列表")]
    problems: list[Msg] = []
    names = [x["name"] for x, _ in exposed_items(data)]
    seen: set[str] = set()

    def spec_of(x: dict) -> dict | None:
        node_id, dot, param = x["target"].partition(".")
        node = nodes.get(node_id) if dot else None
        node_type = registry.get(node.get("type")) if node else None
        return next((p for p in node_type.interface_specs() if p["name"] == param), None) if node_type else None

    def never(name: str, value: Any) -> str | None:
        """Why the exposed parameter `name` can never be `value` (None: it can, or that cannot be told): _never on its
        entry's options and its target's spec."""
        x = next((x for x, _ in exposed_items(data) if x["name"] == name), None)
        spec = spec_of(x) if x is not None else None
        return _never(name, value, x.get("options"), spec) if spec is not None else None

    def expr_problem(expr: Any) -> str | None:
        """What is wrong with one Hide When / Disable When (an entry's, or one menu option's own hide_when): not text,
        does not read, a name no parameter has, or a value it compares with that the parameter never has (never)."""
        if not isinstance(expr, str):
            return "要写成文字"
        why = conditions.problem(expr, names)
        if why is None:
            why = next((w for n, v in conditions.compared(conditions.parse(expr)) if (w := never(n, v))), None)
        return why

    def where(path: list[str], last: str) -> str:
        return " / ".join([*path, last] if last else path) or "参数界面"

    def walk(entries: list, path: list[str]) -> None:
        for i, x in enumerate(entries, 1):
            if is_group(x):
                label = x.get("label")
                here = where(path, str(label or f"第 {i} 项"))
                if not isinstance(label, str) or not label.strip():
                    problems.append(Msg("E-EXPOSED-BAD", where=here, why="组要有名字"))
                if not isinstance(x.get("children"), list):
                    problems.append(Msg("E-EXPOSED-BAD", where=here, why="组的 children 要是一个列表"))
                else:
                    walk(x["children"], [*path, str(label or "")])
                continue
            if not isinstance(x, dict) or not isinstance(x.get("name"), str) or not isinstance(x.get("target"), str):
                problems.append(Msg("E-EXPOSED-BAD", where=where(path, f"第 {i} 项"), why="既不是组，也不是带 name 和 target 的参数"))
                continue
            here = where(path, str(x.get("label") or x["name"]))
            item(x, here)

    def item(x: dict, here: str) -> None:
        name = x["name"]
        if not name.strip():
            problems.append(Msg("E-EXPOSED-BAD", where=here, why="对外名字不能空着"))
        elif name in seen:
            problems.append(Msg("E-EXPOSED-SAMENAME", where=here, name=name))
        seen.add(name)
        spec = spec_of(x)
        if spec is None:
            problems.append(Msg("E-EXPOSED-NOTARGET", where=here, target=x["target"]))
        widget, options = x.get("widget"), x.get("options")
        if spec is not None and spec["widget"] == "button" and (widget is not None or options is not None):
            problems.append(Msg("E-EXPOSED-WIDGET", where=here, widget=EXPOSED_WIDGETS.get(widget, "下拉或复选框"), why="按钮没有值"))
        elif widget is not None and widget not in EXPOSED_WIDGETS:
            problems.append(Msg("E-EXPOSED-BAD", where=here, why=f"控件只能是 menu（下拉）或 checkbox（复选框），不是 {widget!r}"))
        elif widget is not None and spec is not None and (why := _widget_refused(widget, spec, options)):
            problems.append(Msg("E-EXPOSED-WIDGET", where=here, widget=EXPOSED_WIDGETS[widget], why=why))
        if options is not None:
            if not isinstance(options, list):
                problems.append(Msg("E-EXPOSED-BAD", where=here, why="options 要是一个列表"))
            else:
                values_seen: list = []
                for n, o in enumerate(options, 1):
                    if not isinstance(o, dict) or "value" not in o or not isinstance(o.get("label"), str) or not o["label"].strip():
                        problems.append(Msg("E-EXPOSED-OPTION", where=here, n=n, value=json.dumps(o, ensure_ascii=False)[:40], why="每一项要有值和显示名"))
                        continue
                    why = _refuses(spec, o["value"]) if spec is not None else None
                    if why is None and any(conditions.same(o["value"], v) for v in values_seen):
                        why = "和前面一项的值重复"
                    values_seen.append(o["value"])
                    # 这一项自己的 Hide When（成立时下拉里不列这一项；当前值恰好是它时值照留）：规则同条目级（expr_problem）
                    if why is None and o.get("hide_when") is not None:
                        bad = expr_problem(o["hide_when"])
                        why = f"Hide When「{o['hide_when']}」{bad}" if bad else None
                    if why:
                        problems.append(Msg("E-EXPOSED-OPTION", where=here, n=n, value=json.dumps(o["value"], ensure_ascii=False), why=why))
        if "show_on_change" in x and not isinstance(x["show_on_change"], bool):
            problems.append(Msg("E-EXPOSED-BAD", where=here, why="show_on_change（修改后在视图里显示这个节点）只能是 true 或 false"))
        # Hide When / Disable When (Houdini's hide-when / disable-when); `when`: an older file's opposite of disable_when
        for key, code in (("hide_when", "E-EXPOSED-HIDE"), ("disable_when", "E-EXPOSED-DISABLE"), ("when", "E-EXPOSED-WHEN")):
            expr = x.get(key)
            if expr is not None:
                why = expr_problem(expr)
                if why:
                    problems.append(Msg(code, where=here, when=str(expr), why=why))

    walk(raw, [])
    return problems


def exposed_errors(data: dict) -> None:
    """Refuse a graph whose parameter interface is wrong (saving it as a template): Invalid with the first problem
    (check_exposed), which says where it is."""
    from ..errors import Invalid

    problems = check_exposed(data)
    if problems:
        raise Invalid(problems[0])


# parameter widget -> file direction: every file parameter is one the client uploads (what 「输出」 packs is fetched
# from its task: lab2shot/transfer/outputs.py)
FILE_WIDGETS = {"file": "in", "sequence": "in"}


def file_params(data: dict) -> list[dict]:
    """Return every file parameter of the graph: key (node.param), exposed name if any, direction ("in": the client
    uploads a file for it), widget and label."""
    from .graph import check_shape
    from ..nodes import node_types

    check_shape(data)  # a malformed file: E-GRAPH-SHAPE, as reading it says (never a KeyError)

    registry = node_types()
    exposed = {x["target"]: x["name"] for x, _ in exposed_items(data)}
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
    only camera of a chosen file, listing an EXR's layers), unless they are also set explicitly. A menu of the
    interface takes only its own values (the module docstring); one whose current option the new values hide falls to
    the first it lists, as on the page (conditions.settled)."""
    from .graph import check_shape
    from ..nodes import node_types
    from . import conditions

    check_shape(data)  # a malformed file: E-GRAPH-SHAPE, as reading it says (never a KeyError)

    data = copy.deepcopy(data)
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    by_name = {x["name"]: x["target"] for x, _ in exposed_items(data)}
    # the interface's rules go by what is set, however it is named: an exposed parameter set by node.param is held to
    # its menu and its hidden options all the same (its exposed name for the messages)
    name_of = {x["target"]: x["name"] for x, _ in exposed_items(data)}
    menus = {x["target"]: x["options"] for x, _ in exposed_items(data)
             if x.get("widget") == "menu" and isinstance(x.get("options"), list)}
    changed: dict[str, set[str]] = {}
    named: dict[str, Any] = {}
    for key, value in values.items():
        target = by_name.get(key, key)
        if target in name_of:
            named[name_of[target]] = value
        if target in menus and not any(isinstance(o, dict) and conditions.same(value, o.get("value")) for o in menus[target]):
            said = " / ".join(f"{json.dumps(o.get('value'), ensure_ascii=False)}（{o.get('label', '')}）" for o in menus[target] if isinstance(o, dict))
            raise GraphError(Msg("E-TEMPLATE-NOTOPTION", key=key, value=json.dumps(value, ensure_ascii=False), options=said))
        node_id, dot, param = target.partition(".")
        if not dot or node_id not in nodes:
            raise GraphError(Msg("E-TEMPLATE-NOPARAM", key=key, names=list(by_name)) if by_name else Msg("E-TEMPLATE-NOEXPOSED", key=key))
        t = node_types().get(nodes[node_id]["type"])
        if t and param not in t.Params.model_fields and any(b["name"] == param and b["widget"] == "button" for b in t.interface_specs()):
            continue  # a button (「计算」, 「下载」) holds no value: a client sending every exposed name sends it too
        # driven by a wire: set here only when that output may give nothing (Port.may_be_empty: the source's value when
        # it has one, else this one), as the page lets it be typed
        if (wire := wire_source(data, node_id, param)) and not (wire[1] is not None and wire[1].may_be_empty):
            raise GraphError(Msg("E-TEMPLATE-WIRED", key=key, wire=wire[0]))
        nodes[node_id].setdefault("params", {})[param] = value
        changed.setdefault(node_id, set()).add(param)
    # a menu the values set hid the current option of falls to the first it lists, as on the page (conditions.settled:
    # one rule, both sides); what was set here is held to its menu instead (hidden_values: said, not moved)
    fall = [{"name": x["name"], "options": x["options"]} for x, _ in exposed_items(data)
            if x["target"] in menus and x["name"] not in named]
    for name, _was, now in conditions.settled(fall, exposed_values(data)):
        node_id, _dot, param = by_name[name].partition(".")
        nodes[node_id].setdefault("params", {})[param] = now
        changed.setdefault(node_id, set()).add(param)
    hidden_values(data, named)
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


def exposed_values(data: dict) -> dict[str, Any]:
    """每个公开参数现在的值（按对外名字；没写的取节点类型的默认值）：条件表达式里的名字取的就是它（网页
    graph/exposedTree.ts exposedValues 同一口径）。按钮没有值，不列。"""
    from ..nodes import node_types

    nodes = {n.get("id"): n for n in data.get("nodes", []) if isinstance(n, dict)}
    out: dict[str, Any] = {}
    for x, _ in exposed_items(data):
        node_id, _dot, param = x["target"].partition(".")
        node = nodes.get(node_id)
        t = node_types().get(node.get("type")) if node else None
        spec = next((p for p in t.param_specs() if p["name"] == param), None) if t else None
        if spec is None:
            continue
        own = node.get("params") or {}
        out[x["name"]] = own[param] if param in own else spec["default"]
    return out


def hidden_values(data: dict, values: dict[str, Any]) -> None:
    """设进来的值（apply_values）不能是下拉里现在被藏起的那一项（选项自己的 hide_when 成立，conditions.shown_options）：
    网页里使用者选不到它，命令行 / 插件同样不收，说现在能选哪些。"""
    from . import conditions

    now = exposed_values(data)
    for x, _ in exposed_items(data):
        if x["name"] not in values or x.get("widget") != "menu" or not isinstance(x.get("options"), list):
            continue
        value = values[x["name"]]
        option = next((o for o in x["options"] if isinstance(o, dict) and conditions.same(o.get("value"), value)), None)
        if option is not None and conditions.holds(option.get("hide_when"), now):
            shown = conditions.shown_options(x["options"], now)
            raise GraphError(Msg("E-TEMPLATE-HIDDENOPTION", key=x["name"], value=json.dumps(value, ensure_ascii=False), when=option["hide_when"],
                                 shown=" / ".join(f"{json.dumps(o['value'], ensure_ascii=False)}（{o.get('label', '')}）" for o in shown) or "（没有）"))


def delivered_by(data: dict) -> list[str] | None:
    """The nodes a card's deliveries (every 「输出」; a card without one, every node no wire leaves) are made from with
    its values as they stand: along the routes its switches take (Routing.needed, the one answer the cook reads; a
    route still to be computed counts every way). What its licence is judged on (lab2shot/library.py, nodes/tags.py
    graph_tags): a node on a way no default takes is not used. Judged against an empty cache, so what anyone happens to
    have cooked never changes a card's licence. None when the graph can't tell (it does not read)."""
    with _empty_cache():
        return _delivered(data)


@contextmanager
def _empty_cache():
    """An empty cache to judge a card's routes in (nothing cooked counts), as account 1 of it."""
    import tempfile

    from ..data.store import Store, using
    from ..serving import Account, serving

    with tempfile.TemporaryDirectory() as tmp, using(Store(Path(tmp) / "work", Path(tmp) / "data")), serving(Account(1)):
        yield


def _delivered(data: dict) -> list[str] | None:
    from ..serving import Account
    from .evaluation import PLAN_ERRORS, Evaluation

    try:
        graph = Graph.from_json(data)
        targets = graph.deliveries() or [n for n in graph.nodes if not graph.outputs_by_node.get(n)]
        return Evaluation(graph, Account(1)).needed(targets)
    except (GraphError, *PLAN_ERRORS):
        return None


def card_tags(data: dict) -> frozenset[str]:
    """A card's licence tags: those of the nodes its deliveries are made from with its defaults (delivered_by;
    nodes/tags.py graph_tags). The chip its card shows."""
    from ..nodes.tags import graph_tags

    return graph_tags(data, delivered_by(data))


ROUTES_MOST = 256  # route choices a card is judged on at most: every combination of its menus up to this, else one at a time


def route_variants(data: dict) -> list[tuple[dict, dict]]:
    """Every route a user can take through the card, as (the values picked, the card with them set): the defaults
    first, then every combination of the values of its interface's choices — its menus (the ones a menu's hide_when
    leaves: a combination apply_values refuses is no route), its checkboxes (true and false: a boolean picks a
    switch's way as much as a menu does), and the parameters with options exposed as they are (an option that switches
    a node's model counts as much as one that switches a way). Up to ROUTES_MOST combinations, else each choice's
    values one at a time. The one enumeration route_tags (who may use the card) and `lab2shot check templates` (every
    route loads and plans) read."""
    import itertools

    from ..nodes import node_types

    types, nodes = node_types(), {n.get("id"): n for n in data.get("nodes", []) if isinstance(n, dict)}

    def choices(x: dict) -> list:  # the values a user can pick: the menu's own, else the parameter's
        if x.get("widget") == "menu" and isinstance(x.get("options"), list):
            return [o.get("value") for o in x["options"] if isinstance(o, dict)]
        node_id, _dot, param = x["target"].partition(".")
        t = types.get((nodes.get(node_id) or {}).get("type"))
        spec = next((p for p in t.param_specs() if p["name"] == param), None) if t else None
        if spec is None or spec.get("widget") == "button":
            return []
        if spec["type"] == "boolean":
            return [True, False]
        return list(spec["options"] or [])

    menus = [(x["name"], values) for x, _ in exposed_items(data) if (values := choices(x))]
    count = 1
    for _name, values in menus:
        count *= max(len(values), 1)
    if count <= ROUTES_MOST:
        picks = [dict(zip([n for n, _ in menus], combo)) for combo in itertools.product(*[v for _, v in menus])]
    else:
        picks = [{name: v} for name, values in menus for v in values]
    out = [({}, data)]
    for pick in picks:
        try:
            out.append((pick, apply_values(data, pick)))
        except GraphError:  # a value its other choices hide: not a route anyone can take
            continue
    return out


def route_tags(data: dict) -> list[frozenset[str]]:
    """The licence tags of each route a user can take through the card (route_variants), each judged as card_tags
    judges the defaults (what its deliveries are made from). Distinct, the defaults' first. Who may use the card
    (server/access.py templates_for): anyone some route is open to; its chip stays the defaults' (card_tags)."""
    from ..nodes.tags import graph_tags

    out: list[frozenset[str]] = []
    with _empty_cache():
        for _pick, chosen in route_variants(data):
            tags = graph_tags(chosen, _delivered(chosen))
            if tags not in out:
                out.append(tags)
    return out


def route_problems(data: dict, waits: tuple[str, ...] = ()) -> list[str]:
    """Each route a user can take through the card (route_variants) loads as a graph and its deliveries plan: no
    wiring error (a B- refusal) on the way they take with those values — what the defaults alone would not show (a
    lens model chosen that leaves a wire wrong). `waits`: refusal codes that only wait for the user (a file to pick)."""
    from ..serving import Account
    from .evaluation import Evaluation

    said = []
    with _empty_cache():
        for pick, chosen in route_variants(data):
            where = "、".join(f"{k}={v!r}" for k, v in pick.items()) or "默认值"
            try:
                graph = Graph.from_json(chosen)
            except GraphError as exc:
                said.append(f"{where}：节点图读不进来：{exc.message.text}")
                continue
            targets = graph.deliveries() or [n for n in graph.nodes if not graph.outputs_by_node.get(n)]
            refused = Evaluation(graph, Account(1)).readiness(targets).refused
            if refused is not None and refused.code.startswith("B-") and not refused.code.startswith(waits):
                said.append(f"{where}：交付路线规划不了：{refused.text}")
    return said


def menu_routes(data: dict) -> list[str]:
    """Each menu of the interface that decides a switch's route (its target is the switch's 「走哪一路」, or a node
    whose value, known from its parameters alone, NodeDef.known_outputs, is wired into it) against the ways the
    switch has: a value of the menu that takes none of them (the switch would stop the card with
    B-SWITCH-RANGE / B-SWITCH-WIREDRANGE once it is picked), each in a sentence. `lab2shot check templates` says them."""
    from ..nodes import node_types
    from ..nodes.port import PARAM
    from .scopes import chooses

    types = node_types()
    nodes = {n["id"]: n for n in data.get("nodes", []) if isinstance(n, dict)}

    def params_of(nid: str, **over) -> dict:
        t = types[nodes[nid]["type"]]
        return {**{p["name"]: p["default"] for p in t.param_specs()}, **(nodes[nid].get("params") or {}), **over}

    said = []
    for x, _ in exposed_items(data):
        if x.get("widget") != "menu" or not isinstance(x.get("options"), list):
            continue
        nid, _dot, param = x["target"].partition(".")
        if nid not in nodes or nodes[nid].get("type") not in types:
            continue
        kind = types[nodes[nid]["type"]]
        # (switch, how an option's value becomes its 「走哪一路」)
        driven = [(nid, lambda v: v)] if chooses(kind) and PARAM + param == kind.condition_input else []
        for e in data.get("edges", []):
            src, dst = e.get("from") or [None, None], e.get("to") or [None, None]
            if src[0] == nid and dst[0] in nodes and nodes[dst[0]].get("type") in types:
                sw = types[nodes[dst[0]]["type"]]
                if chooses(sw) and dst[1] == sw.condition_input:
                    driven.append((dst[0], lambda v, out=src[1]: (kind.known_outputs(params_of(nid, **{param: v})).get(out) or {}).get("value")))
        for sw_id, value_of in driven:
            sw = types[nodes[sw_id]["type"]]
            for o in x["options"]:
                v = value_of(o.get("value")) if isinstance(o, dict) else None
                if v is not None and not sw.chosen_inputs(params_of(sw_id, which=v), None):
                    said.append(f"下拉「{x['name']}」选「{o.get('label', o.get('value'))}」时「{nodes[sw_id].get('label') or sw.label}」"
                                f"走第 {v} 路，它只有 {len(sw.way_names(params_of(sw_id)))} 路")
    return said


def shared_interfaces(cards: list[tuple[str, dict]]) -> list[str]:
    """For each node type whose parameters the cards expose alike (NodeDef.same_on_cards): what every card that
    exposes some of them exposes of it (the parameter -> its outside name, label, and the shape of its Hide When and Disable When: which names they
    compare how, the values aside, as each card's menu numbers its choices its own way) against what most of them do,
    each card that differs in a sentence. A card whose meta lists the type in `own_interface` departs on purpose and is left out.
    `lab2shot check templates` says them."""
    from collections import Counter

    from ..nodes import node_types
    from .conditions import ConditionError, disable_when_of, parse

    def shape(text) -> str:  # the condition with its values left out ("target_src == ?")
        def walk(x):
            if isinstance(x, tuple) and x[:1] == ("lit",):
                return "?"
            return tuple(map(walk, x)) if isinstance(x, (tuple, list)) else x
        try:
            return repr(walk(parse(text or "")))
        except ConditionError:
            return str(text)

    types = node_types()
    shared = [t for t, k in types.items() if getattr(k, "same_on_cards", False)]
    said = []
    for type_id in shared:
        seen: dict[str, dict] = {}
        shown: dict[str, dict] = {}  # as written, for the sentence
        for name, data in cards:
            if type_id in ((data.get("meta") or {}).get("own_interface") or ()):
                continue
            ids = {n["id"] for n in data.get("nodes", []) if n.get("type") == type_id}
            if not ids:
                continue
            mine = [x for x, _ in exposed_items(data) if x["target"].partition(".")[0] in ids]
            if not mine:  # the node is on the card, none of its parameters on the interface: nothing to compare
                continue
            seen[name] = {x["target"].partition(".")[2]: (x["name"], x.get("label", ""), shape(x.get("hide_when")),
                                                         shape(disable_when_of(x))) for x in mine}
            shown[name] = {x["target"].partition(".")[2]: (x["name"], x.get("label", ""), x.get("hide_when") or "",
                                                          disable_when_of(x) or "") for x in mine}
        if len(seen) < 2:
            continue
        common, _ = Counter(json.dumps(v, sort_keys=True, ensure_ascii=False) for v in seen.values()).most_common(1)[0]
        want = json.loads(common)
        like = next(n for n, v in seen.items() if json.dumps(v, sort_keys=True, ensure_ascii=False) == common)
        for name, got in seen.items():
            if got == {k: tuple(v) for k, v in want.items()}:
                continue
            gone = sorted(set(want) - set(got))
            more = sorted(set(got) - set(want))
            other = sorted(k for k in set(want) & set(got) if tuple(want[k]) != got[k])
            parts = ([f"少了 {'、'.join(gone)}"] if gone else []) + ([f"多了 {'、'.join(more)}"] if more else []) + \
                [f"{k} 的对外名 / 显示名 / 条件的写法不同（{' / '.join(shown[name][k])}；多数卡如「{like}」是 {' / '.join(shown[like][k])}）" for k in other]
            said.append(f"{name}: {type_id} 公开的参数与其余 {len(seen) - 1} 张卡不一致：{'；'.join(parts)}"
                        f"（有意不同就在 meta.own_interface 里写上 {type_id}）")
    return said


def pick_targets(graph: Graph, targets: list[str] | None = None) -> list[str]:
    """Return the nodes to cook: `targets` (cooking a node is that node and what it needs, whichever node it is; an
    output-settings node writes its files, only the 「输出」 it is wired into packs them), or by default every 「输出」."""
    if targets is None:
        targets = graph.deliveries()
        if not targets:
            raise GraphError(Msg("B-TEMPLATE-NOTARGET"))
    for t in targets:
        if t not in graph.nodes:
            raise GraphError(Msg("E-GRAPH-NONODE", node=t))
    return list(dict.fromkeys(targets))

