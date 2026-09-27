"""Category trees, stored entirely as data; the code defines no categories. There is no automatic categorisation:
categories are assigned by the administrator in the panel. There are two trees, each in its own file, sharing one
format and one read/write implementation:

    templates/_categories.json   category tree of the templates panel (kept beside the project's preset templates)
    menu/categories.json         category tree of the node menu; a first-level category carries `section`: the upper
                                 band (tools, Lab2Shot's own nodes) or the lower band (deliver, third-party projects)
    menu/nodes.json              node placement: node type id -> category or subcategory id

A template's placement is recorded in its own file (meta.deliverable, lab2shot/library.py place); a node's placement
is recorded in menu/nodes.json. Anything not placed is 「未分类」: newly added nodes and newly saved templates start
there until the administrator drags them into a category.

File format: `{"<id>": {"parent": "", "label", "tip", "color", "rank", "section"?}}`; an empty parent denotes a
first-level category, otherwise the row is a subcategory of that parent. Before each change the previous version is
kept as `.bak`. An unreadable (corrupt) file does not block the panel: the tree is treated as empty and everything is
shown as 「未分类」, while `problem()` reports what is wrong and where the previous version is. Writes use atomic
replacement (io/atomic.py), so a partially written file cannot occur.

This module is product-layer storage and does not import `lab2shot.server` (the layering rule). The routes are in
server/categories.py, which handles only the HTTP layer.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from pathlib import Path

from .config import MENU_DIR, TEMPLATES_DIR
from .errors import Invalid, NotFound
from .io.atomic import write_text
from .messages import Msg

ID = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
LABEL_CHARS = 10  # fits on one line of the interface (interface text is never wrapped or truncated)
TIP_CHARS = 120
COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
DEFAULT_COLOR = "#8E8E93"


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _row_problem(k: str, r: dict) -> str:
    """Why one row of a tree file is unusable ("" when it is fine): the fields the tree reads must have their types,
    or the whole file counts as broken (reported, never partially read into a 500)."""
    for name in ("parent", "label", "tip", "color", "section"):
        if name in r and not isinstance(r[name], str):
            return f"{k} 的 {name} 不是文字"
    if "rank" in r and not _number(r["rank"]):
        return f"{k} 的 rank 不是数"
    if str(r.get("parent") or "") == k:
        return f"{k} 挂在了自己下面"
    return ""


class Tree:
    """One category tree in one file. `sections`: the bands a first-level category must name (the node menu's two;
    none for the templates panel)."""

    def __init__(self, file: Path, sections: tuple[str, ...] = ()):
        self.file = file
        self.bak = file.with_name(file.name + ".bak")
        self.sections = sections
        self.why = ""  # the last read's reason when the file could not be read

    # ---- reading

    def _read(self) -> tuple[dict[str, dict], str]:
        """(rows, problem): the file's rows, or no rows and the reason (a missing file is not a problem: it is an
        empty tree)."""
        if not self.file.is_file():
            return {}, ""
        try:
            got = json.loads(self.file.read_text(encoding="utf-8"))
            if not isinstance(got, dict) or not all(isinstance(v, dict) for v in got.values()):
                raise ValueError("不是一张 id → 分类 的表")
            for k, r in got.items():
                if why := _row_problem(str(k), r):
                    raise ValueError(why)
                parent = str(r.get("parent") or "")
                if parent and (parent not in got or got[parent].get("parent")):
                    raise ValueError(f"{k} 的上级 {parent} 不是一级分类")  # a subcategory hangs under a first-level one only
        except (OSError, ValueError) as exc:
            self.why = str(exc)[:120]
            return {}, Msg("W-CATEGORIES-BROKEN", file=str(self.file), why=self.why,
                           bak=str(self.bak) if self.bak.is_file() else "没有上一版").text
        return {str(k): v for k, v in got.items()}, ""

    def rows(self) -> dict[str, dict]:
        return self._read()[0]

    def writable(self) -> dict[str, dict]:
        """The rows, for modification. A broken file is refused here with its reason and is never read as an empty tree
        that a write would then replace; its .bak is the only good copy, and the administrator must restore it first."""
        rows, problem = self._read()
        if problem:
            raise Invalid(Msg("E-CATEGORIES-BROKEN", why=self.why))
        return rows

    def problem(self) -> str:
        """"" when the file is readable; otherwise the sentence the panel shows (the tree is then shown empty)."""
        return self._read()[1]

    def changed_at(self) -> float:
        try:
            return self.file.stat().st_mtime
        except OSError:
            return 0.0

    def tree(self) -> list[dict]:
        """First-level categories in rank order, each with its subcategories in rank order:
        {id, label, tip, color, rank, section, subs: [{id, label, tip, rank}]}."""
        rows = self.rows()
        out = [{"id": k, "label": str(r.get("label") or k), "tip": str(r.get("tip") or ""),
                "color": str(r.get("color") or DEFAULT_COLOR), "rank": float(r.get("rank") or 0),
                "section": str(r.get("section") or (self.sections[0] if self.sections else "")), "subs": []}
               for k, r in rows.items() if not r.get("parent")]
        by_id = {c["id"]: c for c in out}
        for k, r in rows.items():
            parent = by_id.get(str(r.get("parent") or ""))
            if parent is not None:
                parent["subs"].append({"id": k, "label": str(r.get("label") or k), "tip": str(r.get("tip") or ""),
                                       "rank": float(r.get("rank") or 0)})
        out.sort(key=lambda c: (c["rank"], c["label"]))
        for c in out:
            c["subs"].sort(key=lambda s: (s["rank"], s["label"]))
        return out

    def subs(self) -> dict[str, str]:
        """Subcategory id -> its first-level category's id."""
        return {s["id"]: c["id"] for c in self.tree() for s in c["subs"]}

    def category_of(self, where: str) -> str:
        """The first-level category an item placed at `where` (a subcategory or a first-level id) sits in; "" when there
        is no such place (the item is 未分类)."""
        if not where:
            return ""
        got = self.subs().get(where, "")
        if got:
            return got
        return where if where in {c["id"] for c in self.tree()} else ""

    def known(self, where: str) -> bool:
        return bool(self.category_of(where))

    # ---- writing

    def _save(self, rows: dict[str, dict], before: dict[str, dict]) -> bool:
        """Write only when something other than a timestamp differs from `before` (the rows the change was based on);
        returns False when there is nothing to write or audit. The .bak beside the file is thus the version before the
        last real change and is never overwritten by a no-op (a card dropped back where it was)."""
        stripped = lambda table: {k: {f: v for f, v in r.items() if f != "updated"} for k, r in table.items()}  # noqa: E731
        if stripped(before) == stripped(rows):
            return False
        if self.file.is_file():
            shutil.copyfile(self.file, self.bak)  # the version before this change, for manual one-step rollback
        write_text(self.file, json.dumps(rows, ensure_ascii=False, indent=1))
        return True

    def _check(self, id_: str, parent: str, label: str, tip: str, color: str, section: str, rank: float, rows: dict) -> None:
        if not ID.match(id_):
            raise Invalid(Msg("E-CATEGORY-BADID", id=id_))
        if not _number(rank):
            raise Invalid(Msg("E-CATEGORY-RANK", rank=str(rank)))
        # A category may not be its own parent, and one that has subcategories must stay first-level (otherwise its
        # subcategories would be orphaned and vanish from the tree along with everything placed under them).
        if parent and (parent == id_ or any(r.get("parent") == id_ for r in rows.values())):
            raise Invalid(Msg("E-CATEGORY-PARENT", id=id_, parent=parent))
        if not label.strip() or len(label) > LABEL_CHARS:
            raise Invalid(Msg("E-CATEGORY-LABEL", most=LABEL_CHARS))
        if len(tip) > TIP_CHARS:
            raise Invalid(Msg("E-CATEGORY-TIP", most=TIP_CHARS))
        if color and not COLOR.match(color):
            raise Invalid(Msg("E-CATEGORY-COLOR", color=color))
        if parent and (parent not in rows or rows[parent].get("parent")):
            raise NotFound(Msg("E-CATEGORY-NOSUCH", id=parent))  # a subcategory hangs under a first-level one only
        if not parent and self.sections and section not in self.sections:
            raise Invalid(Msg("E-CATEGORY-SECTION", section=section, sections="、".join(self.sections)))

    def put(self, id_: str, *, parent: str = "", label: str, tip: str = "", color: str = "", rank: float = 0.0,
            section: str = "") -> bool:
        """Create a category (or a subcategory, with `parent`) or change one: name, tip, colour, rank, band. Returns
        False when nothing differs from the file."""
        before = self.writable()
        rows = dict(before)
        was = rows.get(id_, {})
        section = section or str(was.get("section") or (self.sections[0] if self.sections else ""))
        self._check(id_, parent, label, tip, color, section, float(rank), rows)
        now = time.time()
        rows[id_] = {"parent": parent, "label": label.strip(), "tip": tip.strip(), "color": color, "rank": float(rank),
                     **({"section": section} if not parent and self.sections else {}),
                     "created": was.get("created", now), "updated": now}
        return self._save(rows, before)

    def reorder(self, ids: list[str], parent: str = "") -> bool:
        """Put the categories under `parent` ("": the first level, within one band) in this order, with one write and
        one audit entry (a drag in the panel is one change, not one per row). Returns False when the order is unchanged."""
        before = self.writable()
        rows = dict(before)
        if parent and (parent not in rows or rows[parent].get("parent")):
            raise NotFound(Msg("E-CATEGORY-NOSUCH", id=parent))
        if not ids:
            raise Invalid(Msg("E-CATEGORY-ORDER"))
        default = self.sections[0] if self.sections else ""  # band used for a first-level row without one (tree())
        sections = {str(rows[k].get("section") or default) for k in ids if k in rows} if not parent and self.sections else set()
        if len(sections) > 1:
            raise Invalid(Msg("E-CATEGORY-ORDER"))  # one band's order at a time
        for i, k in enumerate(ids):
            if k not in rows or str(rows[k].get("parent") or "") != parent:
                raise NotFound(Msg("E-CATEGORY-NOSUCH", id=k))
            rows[k] = {**rows[k], "rank": float(i), "updated": time.time()}
        return self._save(rows, before)

    def remove(self, id_: str) -> set[str]:
        """Remove a category together with its subcategories and return the removed ids. The caller un-places the items
        that were in them; items are never deleted with a category."""
        rows = self.writable()
        if id_ not in rows:
            raise NotFound(Msg("E-CATEGORY-NOSUCH", id=id_))
        gone = {k for k, r in rows.items() if k == id_ or r.get("parent") == id_}
        self._save({k: r for k, r in rows.items() if k not in gone}, rows)
        return gone


class Placements:
    """Where each node type sits in the node menu: one file, type id -> a category or subcategory id."""

    def __init__(self, file: Path):
        self.file = file
        self.bak = file.with_name(file.name + ".bak")
        self.why = ""

    def _read(self) -> tuple[dict[str, str], str]:
        if not self.file.is_file():
            return {}, ""
        try:
            got = json.loads(self.file.read_text(encoding="utf-8"))
            if not isinstance(got, dict) or not all(isinstance(v, str) for v in got.values()):
                raise ValueError("不是一张 节点 → 分类 的表")
        except (OSError, ValueError) as exc:
            self.why = str(exc)[:120]
            return {}, Msg("W-CATEGORIES-BROKEN", file=str(self.file), why=self.why,
                           bak=str(self.bak) if self.bak.is_file() else "没有上一版").text
        return {str(k): str(v) for k, v in got.items() if v}, ""

    def rows(self) -> dict[str, str]:
        return self._read()[0]

    def writable(self) -> dict[str, str]:
        rows, problem = self._read()
        if problem:
            raise Invalid(Msg("E-PLACEMENTS-BROKEN", why=self.why))
        return rows

    def problem(self) -> str:
        return self._read()[1]

    def changed_at(self) -> float:
        try:
            return self.file.stat().st_mtime
        except OSError:
            return 0.0

    def _save(self, rows: dict[str, str], before: dict[str, str]) -> bool:
        if before == rows:
            return False  # nothing changed: the .bak stays the version before the last real change
        if self.file.is_file():
            shutil.copyfile(self.file, self.bak)
        write_text(self.file, json.dumps(dict(sorted(rows.items())), ensure_ascii=False, indent=1))
        return True

    def place(self, type_id: str, where: str) -> bool:
        """`where`: a category or subcategory id, or "" (未分类). Returns False when the type is already placed there."""
        before = self.writable()
        rows = dict(before)
        if where:
            rows[type_id] = where
        else:
            rows.pop(type_id, None)
        return self._save(rows, before)

    def unplace(self, gone: set[str]) -> None:
        """The places in `gone` no longer exist; everything placed there becomes 未分类."""
        rows = self.writable()
        self._save({k: v for k, v in rows.items() if v not in gone}, rows)


# the two trees and the node placements
templates = Tree(TEMPLATES_DIR / "_categories.json")
MENU_SECTIONS = (
    {"id": "tools", "label": "Lab2Shot", "tip": "我们自己的节点：读取、图像、遮罩、相机、几何、场景、积木、输出"},
    {"id": "deliver", "label": "第三方项目", "tip": "第三方项目解算出来的东西"},
)
menu = Tree(MENU_DIR / "categories.json", sections=tuple(s["id"] for s in MENU_SECTIONS))
nodes = Placements(MENU_DIR / "nodes.json")


def changed_at() -> tuple:
    """The modification time of each of the three files. The catalogue and template routes use it as their cache key;
    times are kept per file, so restoring an older version also counts as a change."""
    return (templates.changed_at(), menu.changed_at(), nodes.changed_at())


def describe_menu() -> dict:
    """The node menu as the catalogue sends it: its two bands, its tree, where every node sits, and whether a file is
    broken (in which case the menu shows everything as 未分类 and reports the problem)."""
    return {"sections": list(MENU_SECTIONS), "categories": menu.tree(), "placed": nodes.rows(),
            "problem": menu.problem() or nodes.problem()}
