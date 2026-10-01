"""模板的管理路由，界面在前台的模板弹窗里（后台没有单独的模板管理页）。

存储全在文件里（lab2shot/library.py：templates/ 是项目预设，adapters/<包名>/templates/ 是接入层自带的、只读）。
这里的每条路由就是一个文件操作：新建、复制、删除、开关、拖动归类、改名字和简介。一项能力管全部：`templates.create`
（roles.py「管理模板」）——一级管理员天然有，二级管理员看一级给不给。每一步写一条「管理操作」留底（server/access.py audit）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from .. import library
from ..messages import Msg
from . import auth
from .access import audit
from .routes import Access, Body, Router

admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])

disabled = library.disabled  # server/access.py templates_for reads it


class NewTemplate(Body):
    name: str
    intro: str = ""
    deliverable: str = ""  # a subcategory or first-level category id; "" 未分类
    graph: dict
    replace: bool = False  # 管理员已确认：同名的项目预设用这张图覆盖（没有这个标志而同名时回 E-TEMPLATES-SAMENAME）


@admin.post("/templates", access=Access.admin("templates.create"), summary="「保存为预设模板」（文件菜单，管理员）：把手上这张节点图写成 templates/ 下的一个文件。和已有的项目预设同名时先回 409 E-TEMPLATES-SAMENAME，网页问过之后带 replace 再发，覆盖那个文件（文件名、创建时间、作者不变）；和接入层自带的同名则不能存")
def create(req: NewTemplate, request: Request) -> dict:
    from .. import categories
    from ..engine.templates import exposed_errors
    from ..errors import Conflict

    exposed_errors(req.graph)  # 参数界面（exposed 树）有错不存，消息说清是哪一项（engine/templates.py check_exposed）
    if req.deliverable:
        categories.templates.writable()  # a broken tree file is said as such, not as "no such category"
        if not categories.templates.known(req.deliverable):
            from ..errors import NotFound

            raise NotFound(Msg("E-CATEGORY-NOSUCH", id=req.deliverable))
    same = library.same_name_preset(req.name)
    if same is not None and not req.replace:
        raise Conflict(Msg("E-TEMPLATES-SAMENAME", template=same["name"]))
    if same is not None:
        found = library.replace_preset(same["id"], req.graph, req.name, req.intro, req.deliverable)
        told = Msg("I-AUDIT-GRAPHREPLACED", who=auth.actor(request).label, name=found["name"], file=Path(found["path"]).name)
    else:
        found = library.save_preset(req.graph, req.name, req.intro, req.deliverable, auth.actor(request).id)
        told = Msg("I-AUDIT-GRAPHLISTED", who=auth.actor(request).label, name=found["name"])
    audit(told, session=auth.session(request), method="POST", path=str(request.url.path))
    return {"id": found["id"], "name": found["name"], "replaced": same is not None}


class Switch(Body):
    enabled: bool


@admin.put("/templates/{template_id}", access=Access.admin("templates.create"), summary="开或关一张模板卡：关掉的，没有管理权限的账号在模板面板看不到；已经打开或保存的节点图不受影响。写进它自己的文件")
def switch(template_id: str, req: Switch, request: Request) -> dict:
    found = library.set_enabled(template_id, req.enabled)
    audit(Msg("I-AUDIT-TEMPLATESWITCH", who=auth.actor(request).label, template=found["name"],
              state=Msg("I-TEMPLATES-ON" if req.enabled else "I-TEMPLATES-OFF").text),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": template_id, "enabled": found["enabled"]}


class Place(Body):
    where: str  # a subcategory id, a first-level category id, or "" (未分类)


@admin.put("/templates/{template_id}/place", access=Access.admin("templates.create"), summary="把一张模板卡归到另一个二级分类或一级分类（模板弹窗里拖过去）：写进它自己文件的 meta.deliverable；空字符串进「未分类」")
def place(template_id: str, req: Place, request: Request) -> dict:
    from .. import categories

    categories.templates.writable()  # a broken tree file is said as such, not as "no such category"
    if req.where and not categories.templates.known(req.where):
        from ..errors import NotFound

        raise NotFound(Msg("E-CATEGORY-NOSUCH", id=req.where))
    found = library.place(template_id, req.where)
    audit(Msg("I-AUDIT-TEMPLATEPLACED", who=auth.actor(request).label, template=found["name"], where=req.where or "未分类"),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": template_id, "where": req.where}


class Words(Body):
    name: str
    intro: str = ""


@admin.put("/templates/{template_id}/text", access=Access.admin("templates.create"), summary="改一张模板卡的名字和简介（模板弹窗卡片菜单「编辑」）：写进它自己的文件")
def edit(template_id: str, req: Words, request: Request) -> dict:
    found = library.edit_preset(template_id, req.name, req.intro)
    audit(Msg("I-AUDIT-TEMPLATEEDITED", who=auth.actor(request).label, name=found["name"]),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": template_id, "name": found["name"], "intro": found["intro"]}


class Copy(Body):
    name: str = ""  # "" : the original's name with 「副本」


@admin.post("/templates/{template_id}/copy", access=Access.admin("templates.create"), summary="把一张模板卡复制成一个新的项目预设文件（节点图一样，归同一分类，名字后加「副本」），之后可改可删")
def copy(template_id: str, req: Copy, request: Request) -> dict:
    source = library.preset(template_id)
    found = library.copy_preset(template_id, req.name, auth.actor(request).id)
    audit(Msg("I-AUDIT-TEMPLATECOPIED", who=auth.actor(request).label, template=source["name"], name=found["name"]),
          session=auth.session(request), method="POST", path=str(request.url.path))
    return {"id": found["id"], "name": found["name"]}


@admin.delete("/templates/{template_id}", access=Access.admin("templates.create"), summary="删掉一个项目预设（templates/ 下的那个文件）；接入层自带的删不了，只能关闭")
def delete(template_id: str, request: Request) -> dict:
    found = library.delete_preset(template_id)
    audit(Msg("I-AUDIT-TEMPLATEDELETED", who=auth.actor(request).label, name=found["name"]),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    return {"id": template_id}
