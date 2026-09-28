"""「我的模板」路由：各账号自己的模板文件（work/users/<用户名>/templates/，见 lab2shot/library.py），以及管理员对这些
文件的恢复、放入回收站和永久删除（后台「用户」详情页登记表中的按钮，见 lab2shot/resources.py）。

用户路由位于 /api/my/templates 下，仅操作本人数据，登录即可使用。管理员路由位于 /api/admin/graphs/{gid} 下，gid 为卡片
id（user~<用户名>~<文件名>）。每条路由声明所需能力（server/routes.py），网页据此决定显示哪些按钮，组件中不做角色判断。
"""

from __future__ import annotations

from fastapi import Request
from pydantic import BaseModel

from .. import library
from ..database import json_text
from ..messages import Msg
from . import auth, owners
from .access import audit
from .routes import Access, Router

router = Router(tags=["我的模板"])
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])


def _view(username: str, user_id: int) -> dict:
    """返回「我的模板」页面数据，并附带磁盘占用，使页面无需另行请求。"""
    from . import quota

    return {"mine": library.user_cards(username), "usage": quota.usage(user_id)}


class Saved(BaseModel):
    name: str
    intro: str = ""
    graph: dict
    id: str = ""  # 要覆盖的已有模板 id；为空时新建


@router.get("/api/my/templates", access=Access.user("编辑器：自己存在服务器上的节点图"), summary="「我的模板」：这个账号存在服务器上的节点图（换台电脑登录也在），和这个账号的磁盘占用")
def my_templates(request: Request) -> dict:
    u = auth.me(request)
    return _view(u.username, u.id)


@router.post("/api/my/templates", access=Access.user("编辑器：把当前节点图存到「我的模板」", owned=owners.saved_graph_named), summary="把当前节点图存成自己的模板文件：只存节点图和参数，素材不随模板保存，每次自己选；超出磁盘配额时说清占了多少")
def save_mine(req: Saved, request: Request) -> dict:
    from . import quota

    u = auth.me(request)  # 覆盖已有模板时，归属已由路由的 `owned` 校验（server/owners.py）
    quota.room_for(u.id, len(json_text(req.graph).encode("utf-8")))
    library.user_save(u.username, req.graph, req.name, req.intro, stem=library.parse_id(req.id)[2] if req.id else "")
    return _view(u.username, u.id)


@router.get("/api/my/templates/{gid}", access=Access.user("编辑器：打开自己存的一张节点图", owned=owners.saved_graph), summary="打开自己存的一张模板：节点图本身")
def open_mine(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    return library.user_get(username, stem)


@router.delete("/api/my/templates/{gid}", access=Access.user("编辑器：把自己的一张模板放进回收站", owned=owners.saved_graph), summary="把自己的一张模板放进回收站：不是删掉，管理员在后台能恢复")
def bin_mine(gid: str, request: Request) -> dict:
    u = auth.me(request)
    _, username, stem = library.parse_id(gid)
    library.user_bin(username, stem, "user")
    return _view(u.username, u.id)


# 恢复仅需 templates.restore（二级管理员也可为用户找回误删的模板）；放入回收站和永久删除需要 data.others。
@admin.post("/graphs/{gid}/restore", access=Access.admin("templates.restore", owned=owners.saved_graph), summary="把一个用户删掉的模板恢复回去：他在「我的模板」里又看得到它")
def admin_restore(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    found = library.user_restore(username, stem)
    audit(Msg("I-AUDIT-GRAPHRESTORED", who=auth.label(request), name=found["name"]),
          session=auth.session(request), method="POST", path=str(request.url.path))
    return found


@admin.post("/graphs/{gid}/bin", access=Access.admin("data.others", owned=owners.saved_graph), summary="把一个用户的模板放进回收站：他看不到了，随时能恢复")
def admin_bin(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    found = library.user_bin(username, stem, "admin")
    audit(Msg("I-AUDIT-GRAPHBINNED", who=auth.label(request), name=found["name"]),
          session=auth.session(request), method="POST", path=str(request.url.path))
    return found


@admin.delete("/graphs/{gid}", access=Access.admin("data.others", owned=owners.saved_graph), summary="永久删除一个用户的模板文件：删了就找不回来了（先放回收站，确认不要了再删）")
def admin_purge(gid: str, request: Request) -> dict:
    _, username, stem = library.parse_id(gid)
    found = library.user_purge(username, stem)
    audit(Msg("I-AUDIT-GRAPHPURGED", who=auth.label(request), name=found["name"]),
          session=auth.session(request), method="DELETE", path=str(request.url.path))
    return found
