"""模板的管理路由，界面在前台的模板弹窗里（后台没有单独的模板管理页）。

存储全在文件里（lab2shot/site/library.py：templates/ 是项目预设，adapters/<包名>/templates/ 是接入层自带的、只读）。
这里的每条路由就是一个文件操作：新建、复制、删除、开关、拖动归类、改名字和简介。一项能力管全部：`templates.create`
（roles.py「管理模板」）——一级管理员天然有，二级管理员看一级给不给。每一步写一条「管理操作」留底（server/access.py audit）。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Request

from ..site import library
from ..messages import Msg
from . import auth
from .access import audit
from .routes import Access, Body, Router
from .words import Word

admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])

disabled = library.disabled  # server/access.py templates_for reads it


class NewTemplate(Body):
    name: str
    intro: str = ""
    deliverable: str = ""  # a subcategory or first-level category id; "" 未分类
    graph: dict
    replace: bool = False  # 管理员已确认：同名的项目预设用这张图覆盖（没有这个标志而同名时回 E-TEMPLATES-SAMENAME）


@admin.post("/templates", access=Access.admin("templates.create"), summary="Save as Preset Template (File menu, administrators): write the graph at hand as a file under templates/. With "
                                                                           "the same name as an existing project preset the answer is first 409 E-TEMPLATES-SAMENAME; after the page asks, "
                                                                           "it sends again with replace and overwrites that file (file name, creation time and author kept); one with the "
                                                                           "same name as a template an integration ships cannot be saved")
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


@admin.put("/templates/{template_id}", access=Access.admin("templates.create"), summary="Enable or disable a template card: accounts without management rights do not see disabled ones in the template "
                                                                                        "panel; graphs already opened or saved are not affected. Written to its own file")
def switch(template_id: str, req: Switch, request: Request) -> dict:
    found = library.set_enabled(template_id, req.enabled)
    audit(Msg("I-AUDIT-TEMPLATESWITCH", who=auth.actor(request).label, template=found["name"],
              state=Msg("I-TEMPLATES-ON" if req.enabled else "I-TEMPLATES-OFF").text),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": template_id, "enabled": found["enabled"]}


class Place(Body):
    where: str  # a subcategory id, a first-level category id, or "" (未分类)


@admin.put("/templates/{template_id}/place", access=Access.admin("templates.create"), summary="Place a template card in another subcategory or category (dragged in the template dialog): written to its own "
                                                                                              "file's meta.deliverable; an empty string means Uncategorized")
def place(template_id: str, req: Place, request: Request) -> dict:
    from .. import categories

    categories.templates.writable()  # a broken tree file is said as such, not as "no such category"
    if req.where and not categories.templates.known(req.where):
        from ..errors import NotFound

        raise NotFound(Msg("E-CATEGORY-NOSUCH", id=req.where))
    found = library.place(template_id, req.where)
    audit(Msg("I-AUDIT-TEMPLATEPLACED", who=auth.actor(request).label, template=found["name"], where=req.where or Word("server.uncategorized")),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": template_id, "where": req.where}


class Words(Body):
    name: str
    intro: str = ""


@admin.put("/templates/{template_id}/text", access=Access.admin("templates.create"), summary="Change a template card's name and intro (Edit in the card menu of the template dialog): written to its own "
                                                                                             "file")
def edit(template_id: str, req: Words, request: Request) -> dict:
    found = library.edit_preset(template_id, req.name, req.intro)
    audit(Msg("I-AUDIT-TEMPLATEEDITED", who=auth.actor(request).label, name=found["name"]),
          session=auth.session(request), method="PUT", path=str(request.url.path))
    return {"id": template_id, "name": found["name"], "intro": found["intro"]}


class Copy(Body):
    name: str = ""  # "" : the original's name with 「副本」


@admin.post("/templates/{template_id}/copy", access=Access.admin("templates.create"), summary="Copy a template card to a new project preset file (the same graph, in the same category, the name followed by "
                                                                                              "Copy), which may then be changed or deleted")
def copy(template_id: str, req: Copy, request: Request) -> dict:
    source = library.preset(template_id)
    found = library.copy_preset(template_id, req.name, auth.actor(request).id)
    audit(Msg("I-AUDIT-TEMPLATECOPIED", who=auth.actor(request).label, template=source["name"], name=found["name"]),
          session=auth.session(request), method="POST", path=str(request.url.path))
    return {"id": found["id"], "name": found["name"]}


@admin.delete("/templates/{template_id}", access=Access.admin("templates.create"), summary="Delete a project preset (that file under templates/); templates an integration ships cannot be deleted, only "
                                                                                           "disabled")
def delete(template_id: str, request: Request) -> dict:
    found = library.delete_preset(template_id)
    audit(Msg("I-AUDIT-TEMPLATEDELETED", who=auth.actor(request).label, name=found["name"]),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    return {"id": template_id}
