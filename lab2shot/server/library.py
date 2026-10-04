"""「我的模板」路由：各账号自己的模板文件（work/users/<用户名>/templates/，见 lab2shot/site/library.py），以及管理员对这些
文件的恢复、放入回收站和永久删除（后台「用户」详情页登记表中的按钮，见 lab2shot/site/resources.py）。

用户路由位于 /api/my/templates 下，仅操作本人数据，登录即可使用。管理员路由位于 /api/admin/graphs/{gid} 下，gid 为卡片
id（user~<用户名>~<文件名>）。每条路由声明所需能力（server/routes.py），网页据此决定显示哪些按钮，组件中不做角色判断。
"""

from __future__ import annotations

from fastapi import Request

from ..site import library
from ..database import json_text
from ..messages import Msg
from . import auth, owners
from .access import audit
from .routes import Access, Body, Router

router = Router(tags=["My Templates"])
admin = Router(prefix="/api/admin", tags=["Admin (/admin page)"])


def _view(username: str, user_id: int) -> dict:
    """返回「我的模板」页面数据，并附带磁盘占用，使页面无需另行请求。"""
    from . import quota

    return {"mine": library.user_cards(username), "usage": quota.usage(user_id)}


class Saved(Body):
    name: str
    intro: str = ""
    graph: dict
    id: str = ""  # 要覆盖的已有模板 id；为空时新建
    replace: bool = False  # 用户已确认：同名的自己的模板用这张图覆盖（没有这个标志而同名时回 E-LIBRARY-SAMENAME）


@router.get("/api/my/templates", access=Access.user("Editor: your own graphs kept on the server"), summary="My Templates: the graphs this account keeps on the server (there on any computer it logs in from) and this "
                                                                                                                 "account's disk usage")
def my_templates(request: Request) -> dict:
    u = auth.me(request)
    return _view(u.username, u.id)


@router.post("/api/my/templates", access=Access.user("Editor: save the current graph to My Templates", owned=owners.saved_graph_named, body_ids=("id",)), summary="Save the current graph as your own template file: only the graph and parameters, media are not saved with it "
                                                                                                                                                        "and chosen each time; over the disk quota it says how much is used. With the same name as one of your "
                                                                                                                                                        "templates the answer is first 409 E-LIBRARY-SAMENAME; after the page asks, it sends again with replace and "
                                                                                                                                                        "overwrites that one")
def save_mine(req: Saved, request: Request) -> dict:
    from . import quota
    from ..engine.templates import exposed_errors
    from ..errors import Conflict

    exposed_errors(req.graph)  # 参数界面（exposed 树）有错不存，消息说清是哪一项（engine/templates.py check_exposed）
    u = auth.me(request)  # 覆盖已有模板时，归属已由路由的 `owned` 校验（server/owners.py）
    stem = library.parse_id(req.id)[2] if req.id else ""
    if not stem:  # 同名的只在这个账号自己的文件夹里找：归属不用再查
        stem = library.user_same_name(u.username, req.name)
        if stem and not req.replace:
            raise Conflict(Msg("E-LIBRARY-SAMENAME", name=req.name.strip()))
    quota.room_for(u.id, len(json_text(req.graph).encode("utf-8")))
    library.user_save(u.username, req.graph, req.name, req.intro, stem=stem)
    return {**_view(u.username, u.id), "replaced": bool(stem)}


@router.get("/api/my/templates/{gid}", access=Access.user("Editor: open one of your saved graphs", owned=owners.saved_graph), summary="Open one of your saved templates: the graph itself")
def open_mine(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    return library.user_get(username, stem)


@router.delete("/api/my/templates/{gid}", access=Access.user("Editor: move one of your templates to the recycle bin", owned=owners.saved_graph), summary="Move one of your templates to the recycle bin: not deleted, an administrator can restore it")
def bin_mine(gid: str, request: Request) -> dict:
    u = auth.me(request)
    _, username, stem = library.parse_id(gid)
    library.user_bin(username, stem, "user")
    return _view(u.username, u.id)


# 恢复仅需 templates.restore（二级管理员也可为用户找回误删的模板）；放入回收站和永久删除需要 data.others。
def _owner_of(gid: str) -> int | None:
    """The account a saved graph's id names (lab2shot/site/library.py parse_id), for the audit line about it."""
    from .. import accounts

    _, username, _ = library.parse_id(gid)
    found = accounts.by_username(username)
    return found.id if found is not None else None


@admin.post("/graphs/{gid}/restore", access=Access.admin("templates.restore", owned=owners.saved_graph), summary="Restore a template a user deleted: it shows in their My Templates again")
def admin_restore(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    found = library.user_restore(username, stem)
    audit(Msg("I-AUDIT-GRAPHRESTORED", who=auth.actor(request).label, name=found["name"]),
          about=_owner_of(gid), session=auth.session(request), method="POST", path=str(request.url.path))
    return found


@admin.post("/graphs/{gid}/bin", access=Access.admin("data.others", owned=owners.saved_graph), summary="Move a user's template to the recycle bin: they no longer see it, it can be restored at any time")
def admin_bin(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    found = library.user_bin(username, stem, "admin")
    audit(Msg("I-AUDIT-GRAPHBINNED", who=auth.actor(request).label, name=found["name"]),
          about=_owner_of(gid), session=auth.session(request), method="POST", path=str(request.url.path))
    return found


@admin.delete("/graphs/{gid}", access=Access.admin("data.others", owned=owners.saved_graph), summary="Delete a user's template file permanently: it cannot be recovered (put it in the recycle bin first, delete "
                                                                                                     "once sure)")
def admin_purge(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    found = library.user_purge(username, stem)
    audit(Msg("I-AUDIT-GRAPHPURGED", who=auth.actor(request).label, name=found["name"]),
          about=_owner_of(gid), session=auth.session(request), method="DELETE", path=str(request.url.path))
    return found
