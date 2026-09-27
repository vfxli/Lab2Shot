"""两棵分类树的路由（树本身在 lab2shot/categories.py：产品这一层的存储，和「我的模板」一样）。

模板面板的树归「管理模板」（templates.create），节点菜单的树、节点归属和节点的名字说明归「管理节点分类」（menu.edit）：两项权限分开，
二级管理员由一级分别给。页面按服务器算的可用性显示。每一步写一条「管理操作」留底，两棵树的留底分开写。
"""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel

from .. import categories, library
from ..messages import Msg
from . import auth
from .access import audit
from .routes import Access, Router

admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])


class CategoryIn(BaseModel):
    id: str
    parent: str = ""  # 空：一级分类
    label: str
    tip: str = ""
    color: str = ""
    rank: float = 0.0
    section: str = ""  # 节点菜单的树：一级分类在哪个区（tools / deliver）；模板的树不用


class Order(BaseModel):
    ids: list[str]  # the categories under `parent` in their new order
    parent: str = ""  # 空：一级分类


def _label(tree: categories.Tree, cid: str) -> str:
    return str(tree.rows().get(cid, {}).get("label") or cid)


# ---- 模板面板的树


@admin.get("/categories", access=Access.admin("templates.create"), summary="模板面板的分类树（templates/_categories.json），和文件读不出来时的那句话")
def listing(request: Request) -> dict:
    return {"categories": categories.templates.tree(), "problem": categories.templates.problem()}


@admin.put("/categories", access=Access.admin("templates.create"), summary="模板面板：新建一个分类或二级分类，或改一个已有的：名字、说明、颜色、排在第几")
def upsert(req: CategoryIn, request: Request) -> dict:
    if categories.templates.put(req.id, parent=req.parent, label=req.label, tip=req.tip, color=req.color, rank=req.rank):
        audit(Msg("I-AUDIT-CATEGORYSET", who=auth.label(request), name=req.label),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"categories": categories.templates.tree()}


@admin.put("/categories/order", access=Access.admin("templates.create"), summary="模板面板：一级分类（或一个分类下的二级分类）的新次序，一次写完")
def order(req: Order, request: Request) -> dict:
    if categories.templates.reorder(req.ids, req.parent):
        audit(Msg("I-AUDIT-CATEGORYORDER", who=auth.label(request), tree="模板", group=_label(categories.templates, req.parent) if req.parent else "一级", count=len(req.ids)),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"categories": categories.templates.tree()}


@admin.delete("/categories/{cid}", access=Access.admin("templates.create"), summary="模板面板：删掉一个分类（连同它的二级分类）；归在它下面的模板进「未分类」，不会跟着删")
def drop(cid: str, request: Request) -> dict:
    name = _label(categories.templates, cid)
    gone = categories.templates.remove(cid)
    stuck = library.unplace(gone)
    audit(Msg("I-AUDIT-CATEGORYGONE", who=auth.label(request), name=name),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    if stuck:
        audit(Msg("I-AUDIT-TEMPLATESTUCK", who=auth.label(request), name=name, count=len(stuck), names="、".join(stuck)),
              session=auth.session(request), method="DELETE", path=str(request.url.path))
    # the category is gone either way; files that could not be rewritten still name it (they show as 未分类): said
    # beside the tree, since the removal itself succeeded
    problem = Msg("E-CATEGORY-UNPLACED", count=len(stuck), names="、".join(stuck)).text if stuck else ""
    return {"categories": categories.templates.tree(), "problem": problem}


# ---- 节点菜单的树和节点归属


@admin.put("/menu/categories", access=Access.admin("menu.edit"), summary="节点菜单：新建一个分类（说明在哪个区）或二级分类，或改一个已有的：名字、说明、颜色、排在第几")
def menu_upsert(req: CategoryIn, request: Request) -> dict:
    if categories.menu.put(req.id, parent=req.parent, label=req.label, tip=req.tip, color=req.color, rank=req.rank,
                           section=req.section):
        audit(Msg("I-AUDIT-MENUCATEGORYSET", who=auth.label(request), name=req.label),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return categories.describe_menu()


@admin.put("/menu/categories/order", access=Access.admin("menu.edit"), summary="节点菜单：一个区的一级分类（或一个分类下的二级分类）的新次序，一次写完")
def menu_order(req: Order, request: Request) -> dict:
    if categories.menu.reorder(req.ids, req.parent):
        audit(Msg("I-AUDIT-CATEGORYORDER", who=auth.label(request), tree="节点菜单", group=_label(categories.menu, req.parent) if req.parent else "一级", count=len(req.ids)),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return categories.describe_menu()


@admin.delete("/menu/categories/{cid}", access=Access.admin("menu.edit"), summary="节点菜单：删掉一个分类（连同它的二级分类）；归在它下面的节点进「未分类」")
def menu_drop(cid: str, request: Request) -> dict:
    categories.nodes.writable()  # both files are written: a broken one is refused before the tree loses anything
    name = _label(categories.menu, cid)
    gone = categories.menu.remove(cid)
    categories.nodes.unplace(gone)
    audit(Msg("I-AUDIT-MENUCATEGORYGONE", who=auth.label(request), name=name),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    return categories.describe_menu()


class NodeWords(BaseModel):
    label: str
    description: str = ""


@admin.put("/menu/nodes/{type_id}/text", access=Access.admin("menu.edit"), summary="改一个节点的名字和说明（节点菜单里节点的「编辑」）：写进它所在文件夹的 nodes.json，立刻生效")
def menu_text(type_id: str, req: NodeWords, request: Request) -> dict:
    from ..nodes import text

    entry, changed = text.save(type_id, req.label, req.description)
    if changed:
        audit(Msg("I-AUDIT-NODETEXT", who=auth.label(request), node=type_id, name=entry["label"]),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": type_id, **entry}


class PlaceNode(BaseModel):
    where: str  # a category or subcategory id of the menu tree, or "" (未分类)


@admin.put("/menu/nodes/{type_id}/place", access=Access.admin("menu.edit"), summary="节点菜单：把一个节点类型归到一个分类或二级分类（菜单里拖过去）；空字符串进「未分类」。不在的节点类型（扩展包卸了）只能清掉归属")
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
        audit(Msg("I-AUDIT-NODEPLACED", who=auth.label(request), node=type_id, where=req.where or "未分类"),
              session=auth.session(request), method="PUT", path=str(request.url.path))
    return categories.describe_menu()
