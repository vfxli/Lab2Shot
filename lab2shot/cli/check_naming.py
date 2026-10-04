"""`lab2shot check naming`: node type names and node names (engine/naming.py).

- a type name is `[a-z][a-z0-9_]*`, at most one dot: a node of the core has no prefix (`retarget`), or a format's
  (`fbx.import`, `usd.output`: engine/naming.py format_prefixes); an extension's node is `<extension>.<task>`;
- the formats' prefixes are reserved: no extension under adapters/ takes one of them as its name, except the core's
  own format extensions (format_extensions: fbx, alembic); no core type without a prefix is an extension's name;
- in every template: node names follow the rule (engine/naming.py NAME, NAME_MOST), are unique in their graph, and
  everything that names a node (wires, exposed targets, boxes, the node shown) names one that is there."""

from __future__ import annotations

import re

from .. import i18n
from ..engine.templates import targets_of

TYPE_NAME = re.compile(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?")


def _exposed_targets(entries) -> list[str]:
    out: list[str] = []
    for x in entries or []:
        if isinstance(x, dict) and x.get("kind") == "group":
            out += _exposed_targets(x.get("children"))
        elif isinstance(x, dict):
            out += targets_of(x)  # one parameter or several (engine/templates.py targets_of)
    return out


def graph_name_problems(data: dict) -> list[str]:
    """What is wrong with the node names of one graph file and the references to them, each in a sentence."""
    from ..engine.naming import NAME, NAME_MOST

    said: list[str] = []
    ids = [n.get("id") for n in data.get("nodes", []) if isinstance(n, dict)]
    for nid in ids:
        if not isinstance(nid, str) or not NAME.fullmatch(nid) or len(nid) > NAME_MOST:
            said.append(i18n.t("cli.check.naming.bad_name", name=repr(nid), most=NAME_MOST))
    if dupes := sorted({i for i in ids if ids.count(i) > 1}):
        said.append(i18n.t("cli.check.naming.duplicate", names=dupes))
    there = set(ids)
    for e in data.get("edges", []):
        for end in (e.get("from"), e.get("to")):
            if isinstance(end, list) and end and end[0] not in there:
                said.append(i18n.t("cli.check.naming.wire", start=e.get("from"), end=e.get("to"), node=repr(end[0])))
    for target in _exposed_targets(data.get("exposed")):
        if target.split(".")[0] not in there:
            said.append(i18n.t("cli.check.naming.exposed", target=repr(target)))
    for b in data.get("boxes", []) or []:
        if gone := [m for m in b.get("members", []) or [] if m not in there]:
            said.append(i18n.t("cli.check.naming.box", box=repr(i18n.pick(b.get("label")) or b.get("id")), members=gone))
    shown = (data.get("view") or {}).get("display")
    if shown is not None and shown not in there:
        said.append(i18n.t("cli.check.naming.display", node=repr(shown)))
    return said


def check_naming(r) -> None:
    from ..config import ADAPTERS_DIR
    from ..engine.naming import core_format_prefixes, format_prefixes
    from ..site.library import presets
    from ..nodes.registry import node_types

    types = node_types()
    prefixes, core_prefixes = format_prefixes(), core_format_prefixes()
    extensions = {p.name for p in ADAPTERS_DIR.iterdir() if (p / "nodes.py").is_file()} if ADAPTERS_DIR.is_dir() else set()
    bad: list[str] = []
    for tid, t in sorted(types.items()):
        if not TYPE_NAME.fullmatch(tid):
            bad.append(i18n.t("cli.check.naming.type_chars", type=tid))
            continue
        head, dot, _ = tid.partition(".")
        if t.runtime == "core":
            if dot and head not in prefixes:
                bad.append(i18n.t("cli.check.naming.type_prefix", type=tid))
            if not dot and head in extensions:
                bad.append(i18n.t("cli.check.naming.type_extension", type=tid, extension=head))
        elif not dot or head != t.runtime:
            bad.append(i18n.t("cli.check.naming.type_third_party", type=tid, library=t.runtime))
    for name in sorted(extensions & core_prefixes):
        bad.append(i18n.t("cli.check.naming.reserved", name=name))
    for x in bad:
        r.bad(f"naming: {x}")
    if not bad:
        r.ok(i18n.t("cli.check.naming.types_ok", count=len(types), prefixes=len(prefixes)))
    cards = presets()
    n = 0
    for t in cards:
        for said in graph_name_problems(t["graph"]):
            n += 1
            r.bad(f"naming {t['name']}: {said}")
    if not n:
        r.ok(i18n.t("cli.check.naming.templates_ok", count=len(cards)))
