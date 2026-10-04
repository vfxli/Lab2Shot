"""The templates' conventions (templates/_conventions.md, written for people: what each outside name means), checked
where they can be told from a graph (`lab2shot check templates`). The rules are read from that file's fixed-name table,
never repeated here: a row whose target column names `<node type>.<parameter>` binds its outside names to those
targets. Besides the table, the rules its prose states:
- a card that delivers one file names it `file_name`; one that delivers several, `exr_name` / `usd_name` / …, never
  `file_name`;
- the last step's button (「输出」's 「计算」) names the shared word `pack` (「打包」) when the card has other stage buttons,
  else `cook` (「计算」): its `word`, so no shown text is compared;
- on a card whose EXR carries data layers (anything but colour pictures), the template's menu for `exr_compression` lists
  only what the table allows for it (the lossless ones);
- a card that exposes `focal` exposes its node's `filmback` too, greyed out while no focal length is given
  (`disable_when` containing `not focal`);
- a parameter comes before every stage button it acts on: no parameter of a node upstream of a button's node (its
  own included, through wires into parameters too) comes after that button (a person would press it, then change
  what it already used);
- every other outside name is derived, never judged: a value is `<node id>_<parameter>`; the delivering node's 「计算」
  is `cook` and its 「下载」 `download`, any other stage's 「计算」 `cook_<node id>`, any other button
  `<node id>_<button>`. No node of a card ranks above another (a card is a graph), so nothing decides which node's
  parameters may go without a prefix;
- a node id is a readable word (rule 2: it is part of every derived name).
Each problem names the card's parameter it is about."""

from __future__ import annotations

import re
from pathlib import Path

from .. import i18n
from .naming import delivery_names
from .templates import LAST_STEP_WORD, exposed_items, exposed_label, first_target, is_group, split_target, targets_of

_TICK = re.compile(r"`([^`]+)`")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
NODE_ID = re.compile(r"^[a-z][a-z0-9_]{2,}$")  # rule 2: a readable word, the head of every derived name
_TARGET = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)?\.[a-z0-9_]+$")  # <node type>.<parameter>; a type is `retarget` or `<extension>.<task>`


def read_table(path: Path) -> dict[str, dict]:
    """The fixed-name table of the conventions file: outside name -> {"targets": {(node type, parameter)}, "values":
    the backticked words of its notes, "unread": what its target column names that is not a target (every target is
    written `<node type>.<parameter>`, a bare parameter after one being on the same node type), "unpaired":
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
    its target column that is not `<node type>.<parameter>`), or a target that names no node type's parameter."""
    from ..nodes import node_types

    types, said = node_types(), []
    for name, row in rows.items():
        if row.get("unpaired"):
            said.append(i18n.t("conventions.unpaired", name=name, names=row["unpaired"][0], groups=row["unpaired"][1]))
        if row["unread"] or not row["targets"]:
            what = i18n.separator().join(f"`{t}`" for t in row["unread"]) or i18n.t("conventions.none")
            said.append(i18n.t("conventions.unread", name=name, what=what))
        for node_type, param in sorted(row["targets"]):
            kind = types.get(node_type)
            if kind is None:
                said.append(i18n.t("conventions.no_type", name=name, type=node_type, param=param))
            elif param not in {p["name"] for p in kind.param_specs()} | {b["name"] for b in kind.interface_specs()}:
                said.append(i18n.t("conventions.no_param", name=name, type=node_type, param=param))
    return said


def problems(data: dict, table: dict[str, dict]) -> list[str]:
    """What the card breaks of the conventions (see the module docstring), each in a sentence."""
    from ..nodes import node_types
    from ..nodes.output import OutputSettings

    types = node_types()
    nodes = {n["id"]: n for n in data.get("nodes", [])}
    kind = {nid: types.get(n["type"]) for nid, n in nodes.items()}
    items = [x for x, _ in exposed_items(data)]
    by_name = {x["name"]: x for x in items}
    said: list[str] = []

    def target(x: dict) -> tuple[str, str, str]:  # the first target (the one the interface shows)
        node_id, param = first_target(x)
        return node_id, (nodes.get(node_id) or {}).get("type", ""), param

    # the table: a name bound to its targets (every target of an entry that drives several)
    for x in items:
        row = table.get(x["name"])
        if row and row["targets"]:
            for key in targets_of(x):
                node_id, param = split_target(key)
                node_type = (nodes.get(node_id) or {}).get("type", "")
                if (node_type, param) not in row["targets"]:
                    want = i18n.separator().join(f"{t}.{p}" for t, p in sorted(row["targets"]))
                    said.append(i18n.t("conventions.bound", name=x["name"], want=want, type=node_type, param=param))
    # file_name on a card that delivers one file only
    writers = [nid for nid, t in kind.items() if t is not None and isinstance(t, type) and issubclass(t, OutputSettings)]
    if "file_name" in by_name and len(writers) > 1:
        said.append(i18n.t("conventions.several_outputs", n=len(writers), names=" / ".join(delivery_names())))
    if len(writers) == 1 and (several := [n for n in delivery_names() if n in by_name]):
        said.append(i18n.t("conventions.one_output", names=i18n.separator().join(several)))
    # the last step's label
    cooks = [x for x in items if target(x)[2] == "cook"]
    last = [x for x in cooks if kind.get(target(x)[0]) is not None and kind[target(x)[0]].delivers]
    for x in last:
        # judged by the shared word the button names (its `word`, button.<word>), never by its shown text
        stages = len(cooks) > len(last)
        word = "pack" if stages else "cook"
        if x.get("word") != word:
            said.append(i18n.t("conventions.last_button_staged" if stages else "conventions.last_button_alone",
                               name=x["name"], word=word, want=i18n.t(LAST_STEP_WORD), label=exposed_label(x)))
    # EXR compression on a card with data layers
    menu = by_name.get("exr_compression")
    allowed = set((table.get("exr_compression") or {}).get("values") or ())
    if menu and allowed and isinstance(menu.get("options"), list) and _has_data_layers(data, target(menu)[0]):
        if extra := [o.get("value") for o in menu["options"] if isinstance(o, dict) and o.get("value") not in allowed]:
            said.append(i18n.t("conventions.exr_lossless", allowed=" / ".join(sorted(allowed)), extra=extra))
    # focal and its filmback
    if focal := by_name.get("focal"):
        node_id, _, _ = target(focal)
        film = next((x for x in items if f"{node_id}.filmback_mm" in targets_of(x)), None)
        if film is None:
            said.append(i18n.t("conventions.focal_filmback", target=" / ".join(targets_of(focal)), node=node_id))
        elif "not focal" not in str(film.get("disable_when") or ""):
            said.append(i18n.t("conventions.filmback_when", name=film["name"], now=repr(film.get("disable_when"))))
    # a parameter before every stage button it acts on
    said += _after_button(data, items, kind)
    # rules 1 / 2 / 6: a name of the fixed table, else the one derived from its target; node ids that read
    said += _derived(items, kind, table)
    said += [i18n.t("conventions.node_id", id=repr(nid)) for nid in nodes if not NODE_ID.match(nid)]
    # the menus of the fixed table: each value means on this card what the table says it means (「1 FBX / 2 USD」)
    said += _meanings(data, items, table)
    return said


def derived_name(node_id: str, param: str, t) -> str:
    """The outside name a target gets when it is no name of the fixed table (rules 1 and 6)."""
    from ..nodes.params import COOK_BUTTON

    if param == COOK_BUTTON.name:
        return "cook" if t is not None and t.delivers else f"cook_{node_id}"
    if param == "download" and t is not None and t.delivers:
        return "download"
    return f"{node_id}_{param}"


def _derived(items: list[dict], kind: dict, table: dict[str, dict]) -> list[str]:
    """Every outside name outside the fixed table is the one its target derives (derived_name): nothing to judge."""
    said = []
    for x in items:
        nid, param = first_target(x)  # an entry that drives several is named after the first
        t = kind.get(nid)
        want = derived_name(nid, param, t)
        row = table.get(want)
        if row and row["targets"] and (getattr(t, "id", ""), param) not in row["targets"]:
            # the derived name is a name of the table that means something else: the node id is the one to change
            said.append(i18n.t("conventions.derived_taken", id=repr(nid), param=param, want=want))
            continue
        if x["name"] in table:
            continue  # bound to its targets by the table check
        if want != x["name"]:
            said.append(i18n.t("conventions.derived_name", name=x["name"], target=" / ".join(targets_of(x)), want=want))
    return said


_OPEN, _CLOSE = chr(0xFF08), chr(0xFF09)  # the full-width brackets of Chinese text, by code point
_MEANING = re.compile(rf"(\d+)\s*([^/{_OPEN}(]+)")


def _meanings(data: dict, items: list[dict], table: dict[str, dict]) -> list[str]:
    """A fixed-table menu whose notes say what each value means (「1 FBX / 2 USD / 3 ViPE」): each option of it on the
    card with that value has a label that says so; a card the notes name as an exception (「TRAM 例外」, a word of
    the card's name, in either language) is left out."""
    meta_name = (data.get("meta") or {}).get("name")
    name = " ".join(i18n.pick(meta_name, lang) for lang in i18n.LANGS)  # its name in both languages ({zh, en})
    said = []
    for x in items:
        row = table.get(x["name"])
        if not row or x.get("widget") != "menu" or not isinstance(x.get("options"), list):
            continue
        notes = row.get("notes", "")
        # 「TRAM 例外」: the word the notes mark an exception with, in either language (conventions.exception)
        marks = "|".join(re.escape(i18n.t("conventions.exception", in_lang=lang)) for lang in i18n.LANGS)
        if any(w and w in name for w in re.findall(rf"([A-Za-z0-9-]+)\s*(?:{marks})", notes)):
            continue
        plain = re.sub(rf"{_OPEN}[^{_CLOSE}]*{_CLOSE}|\([^)]*\)", "", notes)
        meant = {int(n): w.strip() for n, w in _MEANING.findall(plain) if w.strip()}
        for o in x["options"]:
            v = o.get("value") if isinstance(o, dict) else None
            label = i18n.pick(o.get("label"), "zh")  # the table's notes are in Chinese
            if isinstance(v, int) and not isinstance(v, bool) and v in meant and meant[v] not in label:
                said.append(i18n.t("conventions.meaning", name=x["name"], value=v, meant=meant[v], label=i18n.pick(o.get("label"))))
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
        nid, param = first_target(x)
        if param != COOK_BUTTON.name or kind.get(nid) is None or kind[nid].delivers:  # a stage: a node's 「计算」
            continue
        up, stack = set(), [nid]
        while stack:
            if (n := stack.pop()) not in up:
                up.add(n)
                stack += feeds.get(n, [])
        late = [y["name"] for y in items[i + 1:]
                if any((t := split_target(k))[0] in up and t[1] not in buttons_of(t[0]) for k in targets_of(y))]
        if late:
            said.append(i18n.t("conventions.after_button", params=i18n.separator().join(late),
                               button=exposed_label(x), target=" / ".join(targets_of(x))))
    return said
