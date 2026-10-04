"""Category trees, stored entirely as data; the code defines no categories. There is no automatic categorisation:
categories are assigned by the administrator in the panel. There are two trees, each in its own file, sharing one
format and one read/write implementation:

    templates/_categories.json   category tree of the templates panel (kept beside the project's preset templates);
                                 it is also the node menu's lower band (deliver: nodes backed by a third-party project),
                                 so an algorithm sits in the same place in both
    menu/categories.json         the node menu's upper band only (tools: Lab2Shot's own nodes, plus the file-format
                                 readers and writers); every first-level row carries `section` "tools"
    menu/nodes.json              node placement: node type id -> category or subcategory id (of either band)

The node menu's tree (`menu` below) is the two files read as one: the tools rows of menu/categories.json, then the
templates tree with `section` "deliver" on its first-level rows. A change made from the node menu is written to the
file its band lives in, so the two never need to be kept in step by hand. A category id is unique across both files.

A template's placement is recorded in its own file (meta.deliverable, lab2shot/site/library.py place); a node's placement
is recorded in menu/nodes.json. Anything not placed is 「未分类」: newly added nodes and newly saved templates start
there until the administrator drags them into a category.

File format: `{"<id>": {"parent": "", "color", "rank", "section"?, "templates_only"?}}`; an empty
parent denotes a first-level category, otherwise the row is a subcategory of that parent. `templates_only` (a
first-level row of the templates tree): it holds templates alone (「工作流」), so the node menu leaves it out. Before each change the previous version is
kept as `.bak`. An unreadable (corrupt) file does not block the panel: the tree is treated as empty and everything is
shown as 「未分类」, while `problem()` reports what is wrong and where the previous version is. Writes use atomic
replacement (io/atomic.py), so a partially written file cannot occur.

A category's words are not in the tree files but in the catalogue, one file per language
(lab2shot/i18n/<lang>/categories.toml: category.<id>.label, the node menu's bands menu_section.<id>.label),
read in the language now; a category whose word is missing shows its id. The administrator edits both languages at
once (`put`: the label as {lang: text}); `Words` writes them back, beside the tree file's own write.

This module is product-layer storage and does not import `lab2shot.server` (the layering rule). The routes are in
server/categories.py, which handles only the HTTP layer.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections.abc import Collection
from pathlib import Path

from . import i18n
from .config import MENU_DIR, TEMPLATES_DIR
from .errors import Invalid, NotFound
from .io.atomic import write_text
from .messages import Msg

ID = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
# fits on one line of the interface (interface text is never wrapped or truncated), per language
LABEL_CHARS = {"zh": 10, "en": 32}
COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
DEFAULT_COLOR = "#8E8E93"


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _row_problem(k: str, r: dict) -> str:
    """Why one row of a tree file is unusable ("" when it is fine): the fields the tree reads must have their types,
    or the whole file counts as broken (reported, never partially read into a 500)."""
    for name in ("parent", "label", "color", "section"):
        if name in r and not isinstance(r[name], str):
            return f"{k}: {name} is not text"
    if "templates_only" in r and not isinstance(r["templates_only"], bool):
        return f"{k}: templates_only is not true or false"
    if "rank" in r and not _number(r["rank"]):
        return f"{k}: rank is not a number"
    if str(r.get("parent") or "") == k:
        return f"{k} hangs under itself"
    return ""


class Words:
    """The categories' words, one catalogue file per language (lab2shot/i18n/<lang>/categories.toml): category.<id>.label,
    and the node menu's bands menu_section.<id>.label. Read through lab2shot/i18n like every word; written
    here when the administrator creates, renames or removes a category (the whole file, in a fixed form)."""

    HEADER = ("# Category words (lab2shot/categories.py Words): the node menu's bands, and the categories of both trees\n"
              "# (menu/categories.json, templates/_categories.json hold their structure). Written by the admin page when a\n"
              "# category is created, changed or removed: the whole file is rewritten, comments added by hand are not kept.\n")

    @staticmethod
    def file(lang: str) -> Path:
        return i18n.DIR / lang / "categories.toml"

    def _tables(self, lang: str) -> dict[str, dict[str, dict[str, str]]]:
        import tomllib

        f = self.file(lang)
        data = tomllib.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
        return {"menu_section": dict(data.get("menu_section") or {}), "category": dict(data.get("category") or {})}

    def of(self, cid: str) -> dict[str, dict[str, str]]:
        """A category's words in every language: {"label": {lang: text}} (the admin's editor)."""
        out: dict[str, dict[str, str]] = {"label": {}}
        for lang in i18n.LANGS:
            row = self._tables(lang)["category"].get(cid) or {}
            for part in out:
                out[part][lang] = str(row.get(part) or "")
        return out

    def write(self, changed: dict[str, dict[str, dict[str, str]]], gone: Collection[str] = ()) -> bool:
        """Set the words of the categories in `changed` (id -> {"label": {lang: text}}) and drop
        those in `gone`; returns whether a file changed. The catalogue is read afresh afterwards (i18n.clear)."""
        wrote = False
        for lang in i18n.LANGS:
            tables = self._tables(lang)
            cats = tables["category"]
            before = json.dumps(tables, sort_keys=True, ensure_ascii=False)
            for cid in gone:
                cats.pop(cid, None)
            for cid, parts in changed.items():
                cats[cid] = {"label": str((parts.get("label") or {}).get(lang) or "").strip()}
            if json.dumps(tables, sort_keys=True, ensure_ascii=False) == before and self.file(lang).is_file():
                continue
            write_text(self.file(lang), self.HEADER + "".join(
                f"\n[{table}.{key}]\n" + "".join(f"{k} = {json.dumps(v, ensure_ascii=False)}\n" for k, v in row.items())
                for table in ("menu_section", "category") for key, row in tables[table].items()))
            wrote = True
        if wrote:
            i18n.clear()
        return wrote

    def changed_at(self) -> float:
        try:
            return max(self.file(lang).stat().st_mtime for lang in i18n.LANGS)
        except OSError:
            return 0.0


words = Words()


def _word(cid: str, part: str) -> str:
    """A category's label in the language now ("" when it has none; a label then shows the id)."""
    return i18n.lookup(f"category.{cid}.{part}") or ""


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
                raise ValueError("not a table of id -> category")
            for k, r in got.items():
                if why := _row_problem(str(k), r):
                    raise ValueError(why)
                parent = str(r.get("parent") or "")
                if parent and (parent not in got or got[parent].get("parent")):
                    raise ValueError(f"{k}: its parent {parent} is not a first-level category")  # a subcategory hangs under a first-level one only
        except (OSError, ValueError) as exc:
            self.why = str(exc)[:120]
            return {}, Msg("W-CATEGORIES-BROKEN", file=str(self.file), why=self.why,
                           bak=i18n.Both.of(lambda: str(self.bak) if self.bak.is_file() else i18n.t("categories.no_bak"))).text
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
        {id, label, color, rank, section, subs: [{id, label, rank}]}."""
        rows = self.rows()
        out = [{"id": k, "label": _word(k, "label") or k,
                "color": str(r.get("color") or DEFAULT_COLOR), "rank": float(r.get("rank") or 0),
                "section": str(r.get("section") or (self.sections[0] if self.sections else "")), "subs": []}
               for k, r in rows.items() if not r.get("parent")]
        by_id = {c["id"]: c for c in out}
        for k, r in rows.items():
            parent = by_id.get(str(r.get("parent") or ""))
            if parent is not None:
                parent["subs"].append({"id": k, "label": _word(k, "label") or k,
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

    def _check(self, id_: str, parent: str, label: dict[str, str], color: str, section: str, rank: float,
               rows: dict) -> None:
        if not ID.match(id_):
            raise Invalid(Msg("E-CATEGORY-BADID", id=id_))
        if not _number(rank):
            raise Invalid(Msg("E-CATEGORY-RANK", rank=str(rank)))
        # A category may not be its own parent, and one that has subcategories must stay first-level (otherwise its
        # subcategories would be orphaned and vanish from the tree along with everything placed under them).
        if parent and (parent == id_ or any(r.get("parent") == id_ for r in rows.values())):
            raise Invalid(Msg("E-CATEGORY-PARENT", id=id_, parent=parent))
        for lang in i18n.LANGS:  # a name in every language, each on one line
            name = str(label.get(lang) or "").strip()
            if not name or len(name) > LABEL_CHARS[lang]:
                raise Invalid(Msg("E-CATEGORY-LABEL", most=LABEL_CHARS[lang], lang=i18n.Word(f"lang.{lang}")))
        if color and not COLOR.match(color):
            raise Invalid(Msg("E-CATEGORY-COLOR", color=color))
        if parent and (parent not in rows or rows[parent].get("parent")):
            raise NotFound(Msg("E-CATEGORY-NOSUCH", id=parent))  # a subcategory hangs under a first-level one only
        if not parent and self.sections and section not in self.sections:
            raise Invalid(Msg("E-CATEGORY-SECTION", section=section, sections=i18n.Both.of(lambda: i18n.separator().join(self.sections))))

    def put(self, id_: str, *, parent: str = "", label: dict[str, str], color: str = "",
            rank: float = 0.0, section: str = "") -> bool:
        """Create a category (or a subcategory, with `parent`) or change one: name in every language ({lang:
        text}, the catalogue's: Words), colour, rank, band. Returns False when nothing differs."""
        before = self.writable()
        rows = dict(before)
        was = rows.get(id_, {})
        section = section or str(was.get("section") or (self.sections[0] if self.sections else ""))
        self._check(id_, parent, label, color, section, float(rank), rows)
        now = time.time()
        rows[id_] = {"parent": parent, "color": color, "rank": float(rank),
                     **({"section": section} if not parent and self.sections else {}),
                     **({"templates_only": True} if was.get("templates_only") else {}),  # kept through a rename
                     "created": was.get("created", now), "updated": now}
        said = words.write({id_: {"label": label}})
        return self._save(rows, before) or said

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
        words.write({}, gone)
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
                raise ValueError("not a table of node type -> category")
        except (OSError, ValueError) as exc:
            self.why = str(exc)[:120]
            return {}, Msg("W-CATEGORIES-BROKEN", file=str(self.file), why=self.why,
                           bak=i18n.Both.of(lambda: str(self.bak) if self.bak.is_file() else i18n.t("categories.no_bak"))).text
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


TOOLS, ALGORITHMS = "tools", "deliver"  # the node menu's two bands


class MenuTree(Tree):
    """The node menu's tree: its tools band is its own file, its algorithms band is the templates tree (one source,
    the module docstring). Reads join the two; a write goes to the file each row's band lives in, each keeping its own
    .bak. Rows of the algorithms band left in the tools file are not read (lab2shot check reports them)."""

    def __init__(self, file: Path, algorithms: Tree, sections: tuple[str, ...]):
        super().__init__(file, sections)
        self.algorithms = algorithms

    @staticmethod
    def _band(rows: dict[str, dict], k: str) -> str:
        r = rows.get(k, {})
        root = rows.get(str(r.get("parent") or ""), r)
        return str(root.get("section") or TOOLS)

    def own_rows(self) -> dict[str, dict]:
        """The tools file's rows as written (lab2shot check reads it for stray rows of the algorithms band)."""
        return Tree._read(self)[0]

    def _read(self) -> tuple[dict[str, dict], str]:
        own, problem = super()._read()
        if problem:
            return {}, problem
        alg, problem = self.algorithms._read()
        if problem:
            self.why = self.algorithms.why
            return {}, problem
        own = {k: r for k, r in own.items() if self._band(own, k) == TOOLS}
        # a first-level category of templates alone (「工作流」: whole workflows, `templates_only` in the templates
        # tree) holds no node: not part of the node menu at all, nor its subcategories
        alone = self._alone(alg)
        alg = {k: r for k, r in alg.items() if k not in alone}
        if clash := sorted(set(own) & set(alg)):
            self.why = f"ids {', '.join(clash[:4])} are in both files"
            return {}, Msg("W-CATEGORIES-BROKEN", file=str(self.file), why=self.why,
                           bak=i18n.Both.of(lambda: str(self.bak) if self.bak.is_file() else i18n.t("categories.no_bak"))).text
        return {**own, **{k: ({**r, "section": ALGORITHMS} if not r.get("parent") else r) for k, r in alg.items()}}, ""

    def _split(self, rows: dict[str, dict]) -> tuple[dict[str, dict], dict[str, dict]]:
        own = {k: r for k, r in rows.items() if self._band(rows, k) == TOOLS}
        alg = {k: {f: v for f, v in r.items() if f != "section"} for k, r in rows.items() if k not in own}
        return own, alg

    def _save(self, rows: dict[str, dict], before: dict[str, dict]) -> bool:
        own, alg = self._split(rows)
        own_before, alg_before = self._split(before)
        kept = {k: r for k, r in Tree._read(self)[0].items() if k not in own_before and k not in alg_before}
        wrote = super()._save({**kept, **own}, {**kept, **own_before})
        # the templates-only rows the menu never showed (_read) are written back as they are, never dropped by a menu edit
        alone = self._alone(self.algorithms._read()[0])
        return self.algorithms._save({**alone, **alg}, {**alone, **alg_before}) or wrote

    @staticmethod
    def _alone(alg: dict[str, dict]) -> dict[str, dict]:
        """The templates tree's rows of a templates-only first-level category (and its subcategories)."""
        top = {k for k, r in alg.items() if not r.get("parent") and r.get("templates_only")}
        return {k: r for k, r in alg.items() if k in top or str(r.get("parent") or "") in top}

    def changed_at(self) -> float:
        return max(super().changed_at(), self.algorithms.changed_at())


# the two trees and the node placements
templates = Tree(TEMPLATES_DIR / "_categories.json")
MENU_SECTIONS = (TOOLS, ALGORITHMS)  # their words: menu_section.<id>.label (categories.toml)
menu = MenuTree(MENU_DIR / "categories.json", templates, sections=(TOOLS, ALGORITHMS))
nodes = Placements(MENU_DIR / "nodes.json")


def changed_at() -> tuple:
    """The modification time of each of the three files and of the words. The catalogue and template routes use it as their cache key;
    times are kept per file, so restoring an older version also counts as a change."""
    return (templates.changed_at(), menu.changed_at(), nodes.changed_at(), words.changed_at())


def describe_menu(shown: Collection[str] | None = None) -> dict:
    """The node menu as the catalogue sends it: its two bands, its tree, where each node sits, and whether a file is
    broken (in which case the menu shows everything as 未分类 and reports the problem). `shown`: the node types the
    account asking may use (server/access.py node_types_for), the only ones placed; None every one (the admin pages)."""
    placed = nodes.rows()
    sections = [{"id": b, "label": i18n.t(f"menu_section.{b}.label")} for b in MENU_SECTIONS]
    return {"sections": sections, "categories": menu.tree(),
            "placed": placed if shown is None else {k: v for k, v in placed.items() if k in shown},
            "problem": menu.problem() or nodes.problem()}
