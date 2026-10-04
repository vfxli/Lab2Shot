"""Graphs as tools, for DCC plugins, scripts and `lab2shot cook`.

  GET  /api/tools             every tool this account may use, with its exposed parameters and typed signature, and
                              the delivery formats a client may ask for (`formats`)
  GET  /api/tools/{id}        one tool: its graph, exposed parameters, file parameters and typed signature
  POST /api/tools/describe    the same for a graph the client brings along (a node graph file on the user's machine)
  POST /api/jobs              set parameters (and a frame range), queue a cook of its 「输出」 nodes (a job):
                              {template | graph, values, deliver: [], frames, formats} (server/farm.py JobRequest)
  GET  /api/jobs/{job}/state?since  poll progress; output events say what each 「输出」 packed
  GET  /api/tasks/{task}/outputs/{pkg}[/zip | /file/<name>]   an output: its file list, its zip, one of its files

A tool is a node graph with a parameter interface; there are four sources of them and one shape for all
(`entry`): {id, name, intro, source, category, exposed, inputs, delivers}:
  preset  the templates (templates/, the adapters'; server/access.py templates_for)
  node    a single node wrapped into 「读入 → 节点 → 输出设置 → 输出」 (engine/node_tools.py), id node~<node type>
  mine    the account's own saved graphs (「我的模板」, lab2shot/site/library.py user_cards), id user~<username>~<file>
  local   a graph file the client reads itself and hands to /api/tools/describe (never listed here)
`category` {id, path, ids}: where the tool sits — a preset in the templates tree (templates/_categories.json), a node in
the node menu (menu/), mine and local nowhere ("", [], []); `categories` (beside the tools) every category of both, in
the page's order, with its colour, for a client's tabs. `inputs` / `delivers` / `focus`: engine/signature.py, a pure
function of the graph like `exposed`. A client submits any of them the same way: the graph (GET /api/tools/{id}) with
values, or the id as `template`. What is listed is what admit takes (server/access.py admitted, the one rule): a node
tool or one of 「我的模板」 the account's tags do not allow is not listed, exactly as submitting it would be refused; a
preset is listed when every node of it is one the account may use (access.templates_for, the template panel's rule).

The signatures of the presets and the node tools are worked out in the background when the server starts (`warm`), so
a plugin's first GET /api/tools does not wait for them.

File parameters travel through the client: it uploads the input files (/api/uploads) and puts the references it gets
into the values, and fetches what its 「输出」 nodes packed (server/transfer.py: the zip whole, or the files it needs
from the unpacked folder) to where the user asked for them.
"""

from __future__ import annotations

import threading
from functools import lru_cache

from fastapi import Request

from .routes import Access, Body, Router
from .. import __version__
from ..engine.deliver_formats import formats
from ..engine.node_tools import node_tools, tool_id, type_of_tool
from ..engine.signature import signature
from ..engine.templates import exposed_params, file_params, template
from ..engine.graph import GraphError
from ..errors import NotFound
from ..messages import Msg
from .. import i18n
from ..nodes.registry import registry_key
from . import auth
from .access import admit, admitted, params_for, templates_for

router = Router(prefix="/api", tags=["Templates as Tools (DCC plugins and scripts)"])


def _specs(request: Request):
    """A node type's parameters as this login sees them (access.params_for), for the exposed parameters' descriptions."""
    s = auth.session(request)
    return lambda node_type: params_for(s, node_type)


def describe(data: dict, request: Request, *, remember: bool = True) -> dict:
    return {"exposed": exposed_params(data, _specs(request)), "files": file_params(data),
            **signature(data, remember=remember)}


def _path(rows: dict, where: str) -> dict:
    """{id, path, ids}: a place in a category tree (rows: id -> {parent, …}, structure only), the labels down to it in
    the language now (category.<id>.label, lab2shot/i18n/<lang>/categories.toml; one with no words as its id) and the
    ids alongside them (ids[0]: its first-level category, one of `categories`)."""
    path, ids, here = [], [], where
    while here and here in rows:
        path.insert(0, i18n.lookup(f"category.{here}.label") or here)
        ids.insert(0, here)
        here = str(rows[here].get("parent") or "")
    return {"id": where if path else "", "path": path, "ids": ids}


def categories() -> list[dict]:
    """Every category a tool can sit in, in the order the page shows them: the templates tree (工作流 first), then the
    node menu's own (its tools band: Lab2Shot's own nodes); each {id, label, color, rank, parent, section} ("" parent:
    first level; section "templates" for the templates tree, the node menu's band otherwise), a first-level one before
    its subcategories. A client draws its tabs and marks from these (category.ids of each tool)."""
    from ..categories import menu, templates as tree

    out, seen = [], set()
    for t, band in ((tree, "templates"), (menu, "")):
        for c in t.tree():
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            section = band or c["section"]
            out.append({"id": c["id"], "label": c["label"], "color": c["color"], "rank": c["rank"], "parent": "",
                        "section": section})
            for s in c["subs"]:
                if s["id"] not in seen:
                    seen.add(s["id"])
                    out.append({"id": s["id"], "label": s["label"], "color": c["color"], "rank": s["rank"],
                                "parent": c["id"], "section": section})
    return out


def _template_category(where: str) -> dict:
    from ..categories import templates as tree

    return _path(tree.rows(), where)


def _node_category(type_id: str) -> dict:
    from ..categories import menu, nodes

    return _path(menu.rows(), nodes.rows().get(type_id, ""))


def entry(tid: str, name: str, intro: str, source: str, category: dict, data: dict, specs,
          registry: tuple | None = None) -> dict:
    """One tool as every source lists it (the module docstring)."""
    return {"id": tid, "name": name, "intro": intro, "source": source, "category": category,
            "exposed": exposed_params(data, specs), **signature(data, registry)}


def _usable(u, data: dict) -> bool:
    """Whether admit takes the graph from this account (server/access.py admitted): what is listed can be submitted."""
    try:
        admitted(u, data)
    except GraphError:
        return False
    return True


@lru_cache(maxsize=16)
def _node_tools_for(allowed: frozenset[str] | None, everything: bool, registry: tuple) -> tuple[str, ...]:
    """The node tools admit takes from an account with these tags (and 「所有节点」 or not): the same for every account
    alike, so worked out once per registry."""
    from types import SimpleNamespace

    u = SimpleNamespace(allowed=allowed, capabilities=frozenset({"nodes.all"} if everything else ()))
    return tuple(t for t, d in node_tools() if _usable(u, d))


def _nodes_for(u) -> list[tuple[str, dict]]:
    allowed = None if u.allowed is None else frozenset(u.allowed)
    usable = set(_node_tools_for(allowed, "nodes.all" in u.capabilities, registry_key()))
    return [(t, d) for t, d in node_tools() if t in usable]


def _mine(u) -> list[dict]:
    """The account's own templates admit takes from it (the others it can still open in the page, where it is told why
    one cannot be run)."""
    from ..site import library

    out = []
    for row in library.user_cards(u.username):
        try:
            got = library.user_get(u.username, row["stem"])
        except NotFound:
            continue
        if _usable(u, got["graph"]):
            out.append(got)
    return out


def _one_of_mine(u, tid: str) -> dict | None:
    """One of the account's own templates by its tool id (user~<username>~<file>), when admit takes it."""
    from ..site import library

    head = library.user_card_id(u.username, "")
    if not tid.startswith(head):
        return None
    try:
        got = library.user_get(u.username, tid[len(head):])
    except NotFound:
        return None
    return got if got.get("deleted") is None and _usable(u, got["graph"]) else None


def warm() -> None:
    """Work out the presets' and the node tools' signatures in the background (engine/signature.py remembers them for
    this registry): called once the server is up (server/restart.py serve), so the first GET /api/tools of a plugin
    answers at once instead of after seconds of working them out."""
    from .. import logs
    from ..site.library import presets as templates

    def run() -> None:
        try:
            registry = registry_key()
            for t in templates():
                signature(t["graph"], registry)
            for _type_id, data in node_tools():
                signature(data, registry)
        except Exception:  # noqa: BLE001 (the first request works them out itself)
            import traceback

            logs.say(logs.get("farm"), Msg("W-TOOL-WARMFAILED"), traceback.format_exc())

    threading.Thread(target=run, name="tools-warm", daemon=True).start()


def find(tid: str, request: Request) -> tuple[dict, str, dict]:
    """A tool by its id (or a preset's name, as `template` takes it) among those this account may use: (its graph, its
    source, its listing fields). E-TOOL-NOTFOUND when there is none."""
    u = auth.me(request)
    if node_type := type_of_tool(tid):
        data = dict(_nodes_for(u)).get(node_type)
        if data is None:
            raise NotFound(Msg("E-TOOL-NOTFOUND", name=tid))
        meta = data.get("meta") or {}
        return data, "node", {"id": tid, "name": meta.get("name", tid), "intro": meta.get("intro", ""),
                               "category": _node_category(node_type)}
    if tid.startswith("user~"):
        got = _one_of_mine(u, tid)
        if got is None:
            raise NotFound(Msg("E-TOOL-NOTFOUND", name=tid))
        return got["graph"], "mine", {"id": got["id"], "name": got["name"], "intro": got["intro"],
                                      "category": {"id": "", "path": [], "ids": []}}
    t = template(tid, templates_for(u))
    return t["graph"], "preset", {"id": t["id"], "name": t["name"], "intro": t["intro"],
                                  "category": _template_category(t["deliverable"])}


@router.get("/tools", access=Access.user("DCC plugins and scripts: the tools this account may use (preset templates, single nodes, my templates) and "
                                         "their exposed parameters"),
            summary="Tools this account may use: preset templates, single nodes, my templates, each with its exposed parameters, "
                    "which data it reads and which it delivers; and the delivery formats that may be asked for")
def tools(request: Request) -> dict:
    u = auth.me(request)
    specs = _specs(request)
    registry = registry_key()
    listed = [entry(t["id"], t["name"], t["intro"], "preset", _template_category(t["deliverable"]), t["graph"], specs,
                    registry) for t in templates_for(u)]
    for type_id, data in _nodes_for(u):
        meta = data.get("meta") or {}
        listed.append(entry(tool_id(type_id), meta.get("name", type_id), meta.get("intro", ""), "node",
                            _node_category(type_id), data, specs, registry))
    for g in _mine(u):
        listed.append(entry(g["id"], g["name"], g["intro"], "mine", {"id": "", "path": [], "ids": []}, g["graph"], specs,
                            registry))
    return {"version": __version__, "formats": formats(), "categories": categories(), "tools": listed}


@router.get("/tools/{tool_id}", access=Access.user("DCC plugins and scripts: one tool"),
            summary="One tool: graph, exposed parameters, file parameters (which are uploaded), data types read and delivered")
def tool(tool_id: str, request: Request) -> dict:
    data, source, fields = find(tool_id, request)
    return {**fields, "source": source, "graph": data, **describe(data, request)}


class DescribeRequest(Body):
    graph: dict


@router.post("/tools/describe", access=Access.user("DCC plugins and scripts: the exposed parameters of a graph they bring"),
             summary="A graph the client brings itself (a graph file on its machine): exposed parameters, file parameters, data "
                     "types read and delivered")
def describe_graph(req: DescribeRequest, request: Request) -> dict:
    admit(request, req.graph)  # only node types the account may use (local paths are just names until uploaded)
    meta = req.graph.get("meta") if isinstance(req.graph.get("meta"), dict) else {}
    return {"name": i18n.pick(meta.get("name")), "intro": i18n.pick(meta.get("intro")), "source": "local",
            "category": {"id": "", "path": [], "ids": []}, **describe(req.graph, request, remember=False)}
