"""The routes of the two category trees (the trees themselves are lab2shot/categories.py: product-layer storage, like
「我的模板」).

The templates panel's tree belongs to templates.create; the node menu's tree, the node placements and the nodes' names
and descriptions to menu.edit: two separate rights, which an administrator gives a deputy one by one. The page shows
what the server says is available. Every step writes an admin audit line, the two trees' lines apart. The node menu's
algorithms band is the templates panel's tree (lab2shot/categories.py MenuTree): both edit templates/_categories.json,
and removing one of its categories puts the templates and nodes under it into Uncategorized.

A category's name, and a node's name and description, are edited in every language at once ({lang: text}):
categories in lab2shot/i18n/<lang>/categories.toml (categories.Words), nodes in their own catalogue files
(nodes/text.py save, one language at a time).
"""

from __future__ import annotations

from fastapi import Request

from .. import categories, i18n
from ..site import library
from ..errors import Forbidden, Invalid
from ..messages import Msg
from . import auth
from .access import audit
from .routes import Access, Body, Router
from .words import Word

admin = Router(prefix="/api/admin", tags=["admin"])


class CategoryIn(Body):
    id: str
    parent: str = ""  # "": a first-level category
    # its name in every language ({lang: text}, lab2shot/i18n LANGS); a plain string is the language now's,
    # the other languages keeping theirs (a new category: the same text until it is written in them)
    label: dict[str, str] | str
    color: str = ""
    rank: float = 0.0
    section: str = ""  # the node menu's tree: the band a first-level category is in (tools / deliver); not the templates tree


class Order(Body):
    ids: list[str]  # the categories under `parent` in their new order
    parent: str = ""  # "": the first level


def _both(cid: str, part: str, given: dict[str, str] | str | None) -> dict[str, str]:
    """A category's `part` in every language: `given` as it is ({lang: text}), or a string for the language now with
    the others as they are written (or the same string where there is none)."""
    had = categories.words.of(cid)[part]
    if given is None:
        return {lang: had.get(lang, "") for lang in i18n.LANGS}
    if isinstance(given, dict):  # a language left out keeps its words (only an empty string given clears it)
        return {lang: str(given[lang] or "") if lang in given else had.get(lang, "") for lang in i18n.LANGS}
    return {lang: given if lang == i18n.current() else (had.get(lang) or given) for lang in i18n.LANGS}


def _words(tree: categories.Tree) -> dict:
    """Every category's name in every language, as written (the editor's two languages): id -> {"label":
    {lang: text}}."""
    return {cid: categories.words.of(cid) for cid in tree.rows()}


def _menu() -> dict:
    """The node menu as the catalogue has it, with its categories' words in every language (the admin's answers)."""
    return {**categories.describe_menu(), "words": _words(categories.menu)}


def _label(tree: categories.Tree, cid: str) -> str:
    """A category's name in the language now (its id when it has none)."""
    return categories._word(cid, "label") or cid


def _top() -> Word:
    return Word("server.categories.first_level")


# ---- the templates panel's tree


@admin.get("/categories", access=Access.admin("templates.create"), summary="The templates panel's category tree (templates/_categories.json), and what is wrong when the file does not read")
def listing(request: Request) -> dict:
    return {"categories": categories.templates.tree(), "problem": categories.templates.problem(), "words": _words(categories.templates)}


@admin.put("/categories", access=Access.admin("templates.create"), summary="Templates panel: create a category or subcategory, or change one: name (every language), color, rank")
def upsert(req: CategoryIn, request: Request) -> dict:
    if req.id in categories.menu.own_rows():  # this tree is also the node menu's algorithms band: ids are shared
        raise Invalid(Msg("E-CATEGORY-TAKEN", id=req.id))
    label = _both(req.id, "label", req.label)
    if categories.templates.put(req.id, parent=req.parent, label=label, color=req.color, rank=req.rank):
        audit(Msg("I-AUDIT-CATEGORYSET", who=auth.actor(request).label, name=i18n.Both.of(lambda: i18n.pick(label))),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"categories": categories.templates.tree(), "words": _words(categories.templates)}


@admin.put("/categories/order", access=Access.admin("templates.create"), summary="Templates panel: the new order of the first-level categories (or of one category's subcategories), in one write")
def order(req: Order, request: Request) -> dict:
    if categories.templates.reorder(req.ids, req.parent):
        audit(Msg("I-AUDIT-CATEGORYORDER", who=auth.actor(request).label, tree=Word("server.categories.tree_templates"), group=_label(categories.templates, req.parent) if req.parent else _top(), count=len(req.ids)),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"categories": categories.templates.tree(), "words": _words(categories.templates)}


@admin.delete("/categories/{cid}", access=Access.admin("templates.create"), summary="Templates panel: remove a category (with its subcategories); the templates in it become uncategorized, never deleted with it")
def drop(cid: str, request: Request) -> dict:
    categories.nodes.writable()  # the tree is also the node menu's algorithms band: nodes placed there are un-placed too
    name = _label(categories.templates, cid)
    gone = categories.templates.remove(cid)
    categories.nodes.unplace(gone)
    stuck = library.unplace(gone)
    audit(Msg("I-AUDIT-CATEGORYGONE", who=auth.actor(request).label, name=name),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    if stuck:
        audit(Msg("I-AUDIT-TEMPLATESTUCK", who=auth.actor(request).label, name=name, count=len(stuck), names=i18n.Both.of(lambda: i18n.separator().join(stuck))),
              session=auth.session(request), method="DELETE", path=str(request.url.path))
    # the category is gone either way; files that could not be rewritten still name it (they show as 未分类): said
    # beside the tree, since the removal itself succeeded
    problem = Msg("E-CATEGORY-UNPLACED", count=len(stuck), names=i18n.Both.of(lambda: i18n.separator().join(stuck))).text if stuck else ""
    return {"categories": categories.templates.tree(), "problem": problem, "words": _words(categories.templates)}


# ---- the node menu's tree and the node placements


def _algorithms_need_templates(request: Request, *ids: str, section: str = "") -> None:
    """The algorithms band's categories are the templates panel's tree: changing them (create, change, reorder, remove)
    needs templates.create as well."""
    rows = categories.menu.rows()
    band = section or next((categories.MenuTree._band(rows, i) for i in ids if i in rows), "")
    if band == categories.ALGORITHMS and not auth.can(request, "templates.create"):
        raise Forbidden(Msg("E-CATEGORY-ALGORITHMS"))


@admin.get("/menu/categories", access=Access.admin("menu.edit"), summary="The node menu (bands, category tree, node placements) with every category's name in every language, as the editor shows them")
def menu_listing() -> dict:
    return _menu()


@admin.put("/menu/categories", access=Access.admin("menu.edit"), summary="Node menu: create a category (saying which band) or subcategory, or change one: name (every language), color, rank")
def menu_upsert(req: CategoryIn, request: Request) -> dict:
    _algorithms_need_templates(request, req.parent or req.id, section="" if req.parent else req.section)
    label = _both(req.id, "label", req.label)
    if categories.menu.put(req.id, parent=req.parent, label=label, color=req.color, rank=req.rank,
                           section=req.section):
        audit(Msg("I-AUDIT-MENUCATEGORYSET", who=auth.actor(request).label, name=i18n.Both.of(lambda: i18n.pick(label))),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return _menu()


@admin.put("/menu/categories/order", access=Access.admin("menu.edit"), summary="Node menu: the new order of one band's first-level categories (or of one category's subcategories), in one write")
def menu_order(req: Order, request: Request) -> dict:
    _algorithms_need_templates(request, req.parent, *req.ids)
    if categories.menu.reorder(req.ids, req.parent):
        audit(Msg("I-AUDIT-CATEGORYORDER", who=auth.actor(request).label, tree=Word("server.categories.tree_menu"), group=_label(categories.menu, req.parent) if req.parent else _top(), count=len(req.ids)),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return _menu()


@admin.delete("/menu/categories/{cid}", access=Access.admin("menu.edit"), summary="Node menu: remove a category (with its subcategories); the nodes in it become uncategorized")
def menu_drop(cid: str, request: Request) -> dict:
    _algorithms_need_templates(request, cid)
    categories.nodes.writable()  # both files are written: a broken one is refused before the tree loses anything
    name = _label(categories.menu, cid)
    gone = categories.menu.remove(cid)
    categories.nodes.unplace(gone)
    library.unplace(gone)  # a category of the algorithms band is the templates tree's: its cards become 未分类 too
    audit(Msg("I-AUDIT-MENUCATEGORYGONE", who=auth.actor(request).label, name=name),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    return _menu()


class NodeWords(Body):
    # its subtitle and description in every language ({lang: text}, lab2shot/i18n LANGS); plain strings are the
    # language now's, the other languages left as they are
    subtitle: dict[str, str] | str
    description: dict[str, str] | str = ""


def _node_words(type_id: str) -> dict:
    """A node type's subtitle and description in every language, as its catalogue files hold them now."""
    from ..errors import NotFound
    from ..nodes import text
    from ..nodes.registry import node_types

    cls = node_types().get(type_id)
    if cls is None:
        raise NotFound(Msg("E-NODE-NOSUCH", type=type_id))
    text.refresh()
    out: dict = {"id": type_id, "subtitle": {}, "description": {}}
    for lang in i18n.LANGS:  # as written in each language, no fallback to another (what the editor fills in)
        written = i18n.words(lang)
        out["subtitle"][lang] = written.get(f"node.{type_id}.subtitle", "")
        out["description"][lang] = written.get(f"node.{type_id}.description", "")
    return out


@admin.get("/menu/nodes/{type_id}/text", access=Access.admin("menu.edit"), summary="A node's subtitle and description in every language (the node menu's Edit on a node)")
def menu_text_get(type_id: str) -> dict:
    return _node_words(type_id)


@admin.put("/menu/nodes/{type_id}/text", access=Access.admin("menu.edit"), summary="Change a node's subtitle and description in every language (the node menu's Edit on a node): written into the catalogue file of each language that holds its words, in effect at once")
def menu_text(type_id: str, req: NodeWords, request: Request) -> dict:
    from ..nodes import text

    if isinstance(req.subtitle, str):
        given = {i18n.current(): (req.subtitle, req.description if isinstance(req.description, str) else "")}
    else:
        said = req.description if isinstance(req.description, dict) else {}
        given = {lang: (str(req.subtitle.get(lang) or ""), str(said.get(lang) or "")) for lang in i18n.LANGS}
    for lang, (subtitle, description) in given.items():  # every subtitle first, so a refused one changes nothing
        with i18n.using(lang):
            text._check(subtitle, description)
    changed = False
    for lang, (subtitle, description) in given.items():
        with i18n.using(lang):
            changed = text.save(type_id, subtitle, description)[1] or changed
    if changed:
        audit(Msg("I-AUDIT-NODETEXT", who=auth.actor(request).label, node=type_id, name=i18n.Both.of(lambda: i18n.pick(req.subtitle) if isinstance(req.subtitle, dict) else req.subtitle)),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return _node_words(type_id)


class PlaceNode(Body):
    where: str  # a category or subcategory id of the menu tree, or "" (uncategorized)


@admin.put("/menu/nodes/{type_id}/place", access=Access.admin("menu.edit"), summary="Node menu: place a node type in a category or subcategory (dragged in the menu); an empty string makes it uncategorized. A node type not loaded (its extension uninstalled) can only be un-placed")
def menu_place(type_id: str, req: PlaceNode, request: Request) -> dict:
    from ..errors import NotFound
    from ..nodes.registry import node_types

    categories.menu.writable()  # a broken tree file is said as such, not as "no such category"
    categories.nodes.writable()
    # a type not loaded now (its extension uninstalled) may still be un-placed, so its old placing is cleaned; any other
    # unknown name is refused rather than written down and audited
    if type_id not in node_types() and not (req.where == "" and type_id in categories.nodes.rows()):
        raise NotFound(Msg("E-NODE-NOSUCH", type=type_id))
    if req.where and not categories.menu.known(req.where):
        raise NotFound(Msg("E-CATEGORY-NOSUCH", id=req.where))
    if categories.nodes.place(type_id, req.where):
        audit(Msg("I-AUDIT-NODEPLACED", who=auth.actor(request).label, node=type_id, where=_label(categories.menu, req.where) if req.where else Word("server.categories.uncategorized")),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return _menu()
