"""The templates' conventions (templates/_conventions.md, written for people: what each outside name means), checked
where they can be told from a graph (`lab2shot check templates`). The rules are read from that file's fixed-name table,
never repeated here: a row whose target column names `<node type>.<parameter>` binds its outside names to those
targets. Besides the table, the rules its prose states:
- a card that delivers one file names it `file_name`; one that delivers several, `exr_name` / `usd_name` / …, never
  `file_name`;
- the last step's button (「输出」's 「计算」) is labelled 「打包」 when the card has other stage buttons, else 「计算」;
- on a card whose EXR carries data layers (anything but colour pictures), the template's menu for `exr_compression` lists
  only what the table allows for it (the lossless ones);
- a card that exposes `focal` exposes its node's `filmback` too, greyed out while no focal length is given
  (`disable_when` containing `not focal`);
- a parameter comes before every stage button it acts on: no parameter of a node upstream of a button's node (its
  own included, through wires into parameters too) comes after that button (a person would press it, then change
  what it already used).
Each problem names the card's parameter it is about."""

from __future__ import annotations

import re
from pathlib import Path

from .templates import exposed_items, is_group

_TICK = re.compile(r"`([^`]+)`")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_TARGET = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+\.[a-z0-9_]+$")  # <extension>.<node>.<parameter>
SEVERAL = ("exr_name", "usd_name", "nuke_name", "curves_name", "tracks_name")


def read_table(path: Path) -> dict[str, dict]:
    """The fixed-name table of the conventions file: outside name -> {"targets": {(node type, parameter)}, "values":
    the backticked words of its notes, "unread": what its target column names that is not a target (every target is
    written `<extension>.<node>.<parameter>`, a bare parameter after one being on the same node type), "unpaired":
    (names, target groups) when a row of several names has not one group of targets each}. Only rows
    whose names are plain words are read (a pattern row, `<内容>_name`, binds no name); a row of them that binds
    nothing is a problem of the table (table_problems), never silently a row that checks nothing."""
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not line.lstrip().startswith("|") or set(cells[0]) <= {"-", " "}:
            continue
        names = [n for n in _TICK.findall(cells[0]) if _NAME.match(n)]
        groups: list[list[tuple[str, str]]] = []  # the target column split at 「/」: one group per name, in order
        unread: list[str] = []
        for part in cells[2].split("/"):
            group: list[tuple[str, str]] = []  # a bare parameter after a full target is on that target's node type
            for t in _TICK.findall(part):
                if _TARGET.match(t):
                    group.append(tuple(t.rsplit(".", 1)))
                elif (group or groups) and _NAME.match(t):
                    group.append(((group or groups[-1])[-1][0], t))
                else:
                    unread.append(t)
            if group:
                groups.append(group)
        values = [v for v in _TICK.findall(cells[3]) if _NAME.match(v)]
        # one name: every target is its; several: 「a / b」 against 「x … / y …」, one group each, in order, and as many
        # groups as names (otherwise every name would be allowed every target: said by table_problems)
        paired = len(names) > 1
        for i, n in enumerate(names):
            mine = groups[i] if paired and len(groups) == len(names) else [t for g in groups for t in g]
            rows[n] = {"targets": set(mine), "values": values, "unread": unread, "notes": cells[3],
                       "unpaired": (len(names), len(groups)) if paired and len(groups) != len(names) else None}
    return rows


def table_problems(rows: dict[str, dict]) -> list[str]:
    """What is wrong with the table itself, each in a sentence: a name whose targets can't be read (none, or a word of
    its target column that is not `<extension>.<node>.<parameter>`), or a target that names no node type's parameter."""
    from ..nodes import node_types

    types, said = node_types(), []
    for name, row in rows.items():
        if row.get("unpaired"):
            said.append(f"_conventions.md 固定名表「{name}」这一行有 {row['unpaired'][0]} 个名字、{row['unpaired'][1]} 组目标"
                        f"（按「/」分组）：名字与目标组要一一对应，或者每个名字各占一行")
        if row["unread"] or not row["targets"]:
            what = "、".join(f"`{t}`" for t in row["unread"]) or "（没有）"
            said.append(f"_conventions.md 固定名表「{name}」的目标读不出：{what}（写成 `扩展.节点.参数`）")
        for node_type, param in sorted(row["targets"]):
            kind = types.get(node_type)
            if kind is None:
                said.append(f"_conventions.md 固定名表「{name}」的目标 {node_type}.{param}：没有 {node_type} 这种节点")
            elif param not in {p["name"] for p in kind.param_specs()} | {b["name"] for b in kind.interface_specs()}:
                said.append(f"_conventions.md 固定名表「{name}」的目标 {node_type}.{param}：{node_type} 没有参数 {param}")
    return said


def read_prefixes(path: Path) -> list[str]:
    """The role prefixes rule 2 of the conventions file registers (the backticked `xxx_` words of its line 「2.」),
    and the name patterns it writes anywhere (`<内容>_name`, `colorspace_in_<内容>`, `input_*`), as the regular
    expressions they stand for (a name matching one is a name of the conventions, like one of the fixed table)."""
    text = path.read_text(encoding="utf-8")
    line = next((l for l in text.splitlines() if l.lstrip().startswith("2.")), "")
    prefixes = [t for t in _TICK.findall(line) if re.fullmatch(r"[a-z0-9]+_", t)]
    patterns = [re.sub(r"<[^<>]+>", "[a-z0-9_]+", t).replace("*", "[a-z0-9_]*") for t in _TICK.findall(text)
                if re.fullmatch(r"(?:[a-z0-9_]|<[^<>`]+>|\*)+", t) and ("<" in t or "*" in t)]
    return prefixes + [f"re:{p}" for p in patterns]


def problems(data: dict, table: dict[str, dict], prefixes: list[str] = ()) -> list[str]:
    """What the card breaks of the conventions (see the module docstring), each in a sentence."""
    from ..nodes import node_types
    from ..nodes.output import OutputSettings

    types = node_types()
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    kind = {nid: types.get(n["type"]) for nid, n in nodes.items()}
    items = [x for x, _ in exposed_items(data)]
    by_name = {x["name"]: x for x in items}
    said: list[str] = []

    def target(x: dict) -> tuple[str, str, str]:
        node_id, _, param = x["target"].partition(".")
        return node_id, (nodes.get(node_id) or {}).get("type", ""), param

    # the table: a name bound to its targets
    for x in items:
        row = table.get(x["name"])
        if row and row["targets"]:
            _, node_type, param = target(x)
            if (node_type, param) not in row["targets"]:
                want = "、".join(f"{t}.{p}" for t, p in sorted(row["targets"]))
                said.append(f"对外名 {x['name']} 只指 {want}，这里指的是 {node_type}.{param}")
    # file_name on a card that delivers one file only
    writers = [nid for nid, t in kind.items() if t is not None and isinstance(t, type) and issubclass(t, OutputSettings)]
    if "file_name" in by_name and len(writers) > 1:
        said.append(f"卡上有 {len(writers)} 个输出设置（多种交付）：用 {' / '.join(SEVERAL)}，不用 file_name")
    if len(writers) == 1 and (several := [n for n in SEVERAL if n in by_name]):
        said.append(f"卡上只有一种交付：它的名字叫 file_name，不叫 {'、'.join(several)}")
    # the last step's label
    cooks = [x for x in items if x["target"].endswith(".cook")]
    last = [x for x in cooks if kind.get(target(x)[0]) is not None and kind[target(x)[0]].delivers]
    for x in last:
        want = "打包" if len(cooks) > len(last) else "计算"
        if x.get("label") != want:
            said.append(f"最后一步的按钮 {x['name']} 应叫「{want}」（{'卡上有其他阶段按钮' if want == '打包' else '卡上没有其他阶段按钮'}），现在叫「{x.get('label')}」")
    # EXR compression on a card with data layers
    menu = by_name.get("exr_compression")
    allowed = set((table.get("exr_compression") or {}).get("values") or ())
    if menu and allowed and isinstance(menu.get("options"), list) and _has_data_layers(data, target(menu)[0]):
        if extra := [o.get("value") for o in menu["options"] if isinstance(o, dict) and o.get("value") not in allowed]:
            said.append(f"exr_compression 的下拉在带数据层的卡上只能列无损的 {' / '.join(sorted(allowed))}，多了 {extra}")
    # focal and its filmback
    if focal := by_name.get("focal"):
        node_id, _, _ = target(focal)
        film = next((x for x in items if x["target"] == f"{node_id}.filmback_mm"), None)
        if film is None:
            said.append(f"公开了 focal（{focal['target']}）就要公开同一节点的 filmback（{node_id}.filmback_mm）")
        elif "not focal" not in str(film.get("disable_when") or ""):
            said.append(f"{film['name']} 的 Disable When 要含 not focal（没填焦距时置灰），现在是 {film.get('disable_when')!r}")
    # a parameter before every stage button it acts on
    said += _after_button(data, items, kind)
    # rules 1 / 2: a name without a registered role prefix is the card's core node's parameter (its main project's,
    # meta.project or core_project), or a name of the fixed table
    said += _prefixed(data, items, kind, table, prefixes)
    # the menus of the fixed table: each value means on this card what the table says it means (「1 FBX / 2 USD」)
    said += _meanings(data, items, table)
    return said


def _prefixed(data: dict, items: list[dict], kind: dict, table: dict[str, dict], prefixes) -> list[str]:
    from ..nodes import node_types
    from ..nodes.params import COOK_BUTTON
    from .templates import core_project

    from ..nodes.output import OutputSettings

    main = core_project(data, node_types())
    said = []

    def written(x: dict, t, param: str) -> bool:  # a name the conventions register: prefix, pattern or the table
        for p in prefixes:
            if not p.startswith("re:"):
                if x["name"].startswith(p):
                    return True
            elif re.fullmatch(p[3:], x["name"]):
                # 「<内容>_name」 is a delivery's name (rule 4): only an output-settings node's `name` takes it; any other
                # *_name (a node's own character name) is judged by rules 1 / 2 like every other name
                if not p[3:].endswith("_name") or (isinstance(t, type) and issubclass(t, OutputSettings) and param == "name"):
                    return True
        return x["name"] in table

    for x in items:
        nid, _, param = x["target"].partition(".")
        t = kind.get(nid)
        if t is None or written(x, t, param):
            continue
        if param == COOK_BUTTON.name or any(b["name"] == param and b.get("widget") == "button" for b in t.interface_specs()):
            continue  # a button is no value (rule 7)
        if t.runtime != main:
            said.append(f"对外名 {x['name']} 指的是辅助节点 {t.id} 的参数：带上通则 2 登记的角色前缀（{' '.join(p for p in prefixes if not p.startswith('re:'))}），"
                        f"或写进固定名表（不带前缀的名字只给核心节点 {main or '（无）'}，通则 1）")
    return said


_MEANING = re.compile(r"(\d+)\s*([^/（(]+)")


def _meanings(data: dict, items: list[dict], table: dict[str, dict]) -> list[str]:
    """A fixed-table menu whose notes say what each value means (「1 FBX / 2 USD / 3 ViPE」): each option of it on the
    card with that value has a label that says so; a card the notes name as an exception (「TRAM 例外」, a word of
    the card's name) is left out."""
    name = str((data.get("meta") or {}).get("name") or "")
    said = []
    for x in items:
        row = table.get(x["name"])
        if not row or x.get("widget") != "menu" or not isinstance(x.get("options"), list):
            continue
        notes = row.get("notes", "")
        if any(w and w in name for w in re.findall(r"([A-Za-z0-9-]+)\s*例外", notes)):
            continue
        plain = re.sub(r"（[^）]*）|\([^)]*\)", "", notes)
        meant = {int(n): w.strip() for n, w in _MEANING.findall(plain) if w.strip()}
        for o in x["options"]:
            v = o.get("value") if isinstance(o, dict) else None
            if isinstance(v, int) and not isinstance(v, bool) and v in meant and meant[v] not in str(o.get("label", "")):
                said.append(f"{x['name']} = {v} 在固定名表里是「{meant[v]}」，这张卡上是「{o.get('label', '')}」（取值的含义各卡一致，通则 3）")
    return said


def _has_data_layers(data: dict, node_id: str) -> bool:
    """Whether the EXR output settings node (the one `exr_compression` is set on) takes anything but colour pictures
    (image.3 / image.4) into the rows of its layer table (NodeDef.made_ports)."""
    from ..errors import GraphError
    from .graph import Graph

    try:
        graph = Graph.from_json(data)
    except GraphError:  # a card that does not read is said by the other template checks
        return False
    node = graph.nodes.get(node_id)
    if node is None:
        return False
    rows = [p.name for p in node.type.made_ports(node.params)]
    return any(graph.output_type(src, sport) not in ("image.3", "image.4")
               for row in rows for src, sport in graph.inputs.get((node_id, row), []))


def _after_button(data: dict, items: list[dict], kind: dict) -> list[str]:
    """Parameters that come, in the interface's order, after a stage button (a node's 「计算」, not the last step's that
    delivers) whose node takes from theirs: what the button computes from (the node and everything upstream of it,
    wires into parameters included)."""
    feeds: dict[str, list[str]] = {}
    for e in data.get("edges", []):
        feeds.setdefault(e["to"][0], []).append(e["from"][0])

    def buttons_of(nid: str) -> set[str]:
        t = kind.get(nid)
        return {b["name"] for b in t.interface_specs() if b.get("widget") == "button"} if t is not None else set()

    from ..nodes.params import COOK_BUTTON

    said = []
    for i, x in enumerate(items):
        nid, _, param = x["target"].partition(".")
        if param != COOK_BUTTON.name or kind.get(nid) is None or kind[nid].delivers:  # a stage: a node's 「计算」
            continue
        up, stack = set(), [nid]
        while stack:
            if (n := stack.pop()) not in up:
                up.add(n)
                stack += feeds.get(n, [])
        late = [y["name"] for y in items[i + 1:] if (t := y["target"].partition("."))[0] in up and t[2] not in buttons_of(t[0])]
        if late:
            said.append(f"参数 {'、'.join(late)} 作用于阶段按钮 {x.get('label') or x['name']}（{x['target']}），却排在它之后：挪到按钮前面")
    return said
