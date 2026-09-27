"""Templates as tools, for DCC plugins, scripts and `lab2shot cook`.

  GET  /api/tools             templates with their exposed parameters
  GET  /api/tools/{id}        one template: its graph, exposed parameters and file parameters
  POST /api/tools/describe    the same for a graph the client brings along
  POST /api/jobs              set parameters (and a frame range), queue a cook of its 「输出」 nodes (a job):
                              {template, values, deliver: [], frames} (server/farm.py JobRequest)
  GET  /api/jobs/{job}/state?since  poll progress; output events say which files to fetch

File parameters travel through the client: it uploads the input files (/api/uploads) and puts the references it gets
into the values, and fetches the files its 「输出」 nodes deliver (/api/deliveries) to where the user asked for them.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from pydantic import BaseModel

from .routes import Access, Router
from .. import __version__
from ..engine.graph import GraphError
from ..messages import Msg
from ..engine.templates import exposed_params, file_params, template
from . import auth
from .access import admit, templates_for

router = Router(prefix="/api", tags=["模板即工具（DCC 插件和脚本）"])


def describe(data: dict) -> dict:
    return {"exposed": exposed_params(data), "files": file_params(data)}


@router.get("/tools", access=Access.user("DCC 插件和脚本：这个账号能用的模板和对外参数"), summary="这个账号能用的模板和它们的对外参数（类型、默认值、可选值）")
def tools(request: Request) -> dict:
    return {"version": __version__, "tools": [{**{k: t[k] for k in ("id", "name", "intro")}, "exposed": exposed_params(t["graph"])}
                                              for t in templates_for(auth.me(request))]}


@router.get("/tools/{tool_id}", access=Access.user("DCC 插件和脚本：一个模板"), summary="一个模板：节点图、对外参数、文件参数（哪些要上传、哪些会写出）")
def tool(tool_id: str, request: Request) -> dict:
    t = template(tool_id, templates_for(auth.me(request)))
    return {"id": t["id"], "name": t["name"], "intro": t["intro"], "graph": t["graph"], **describe(t["graph"])}


class DescribeRequest(BaseModel):
    graph: dict


@router.post("/tools/describe", access=Access.user("DCC 插件和脚本：自己带来的节点图的对外参数"), summary="客户端自己带来的节点图：对外参数、文件参数")
def describe_graph(req: DescribeRequest, request: Request) -> dict:
    admit(request, req.graph)  # only node types the account may use (local paths are just names until uploaded)
    return describe(req.graph)
