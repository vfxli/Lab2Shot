"""Web UI backend: node catalog, templates, OCIO lists; the routers for cooking, files and the queue."""

from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES, GZipMiddleware

from .. import __version__, catalog
from .. import library as library_cards  # the template cards (lab2shot/library.py); `library` here is its routes
from ..config import WEBUI_DIST
from ..errors import Invalid, MessageError, NotFound, Unavailable
from ..messages import Msg
from ..nodes import describe_layer_ports, describe_scene_kinds, describe_types, tags
from ..nodes.registry import type_tables
from ..data.values import UNIT_KINDS, UNITS
from .. import categories as trees  # the two category trees and the node placements (data files)
from . import access, auth, categories, farm, feedback, installs, invites, library, logs, notice, owners, packets, quota, records, register, settings, terms, tools, traffic, transfer, users, view, wire  # noqa: F401 (installs, users: admin routes)
from . import templates as admin_templates  # its own admin routes: the 模板 page's switches
from . import revisions
from . import releases
from .routes import Access, Body, Router, mount
from . import view_worker
from ..farm.queue import on_scene_done

catalog.install()  # the node types and what planning needs, for every route below (lab2shot/catalog.py)
# 视图预生成交给 farm：一个 scene.* 数据包算完时由视图 worker 在后台预先生成它的三维块（farm 不认识 server，
# 由这里把上层的函数交给这个进程的 farm，它在造出来时拿到，同 catalog.install 的做法）
on_scene_done(view_worker.scene_done)

# no /docs, /redoc or /openapi.json of FastAPI's own: the machine-readable one is /api/admin/openapi.json, for the administrator
app = FastAPI(title="Lab2Shot", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
logs.install(app)
app.add_middleware(access.Guard)  # who may reach what, on every request (server/access.py)
app.add_middleware(wire.Wire)  # private caching, ETags and 304s, on the answer as the guard let it out (server/wire.py)
# answers travel compressed where it pays (JSON, view data); the page's scripts and styles come compressed from the
# build (wire.Assets); pictures and archives already are, and files keep their size (a download shows its progress).
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5,
                   exclude_content_types=(*DEFAULT_EXCLUDED_CONTENT_TYPES, "image/*", "application/octet-stream"))
# a part of a file (206) is never compressed either; zip and gzip are among the defaults
# The outermost layer counts the bytes actually sent for a request and attributes them to the requesting account
# (server/traffic.py). Placed outside compression, it counts the bytes that actually go over the wire: port 8765 is
# exposed through frp and billed by traffic, so 后台「用户」 must show who used how much.
app.add_middleware(traffic.Meter)


# What goes wrong for the requester, raised anywhere, is answered here in one place: an error with a message with the
# status its class declares (errors.py MessageError.status) and what it says (MessageError.answer), and a request whose
# path, query or body does not read as its route declares (FastAPI's validation) as Invalid, naming the field. Anything
# else is the program's fault, a library's error too (a file gone, a value it refused): 500 with its reference
# (server/logs.py). Where a user's input is what went wrong, the code says so with Invalid or NotFound.
# what is wrong with a field, by the kind pydantic names (its `type`), in words of the catalogue; a kind not here is said
# in pydantic's own words (E-FIELD-OTHER)
_WHY = {"missing": "E-FIELD-MISSING", "int_parsing": "E-FIELD-INTEGER", "int_type": "E-FIELD-INTEGER",
        "int_from_float": "E-FIELD-INTEGER", "float_parsing": "E-FIELD-NUMBER", "float_type": "E-FIELD-NUMBER",
        "finite_number": "E-FIELD-NUMBER", "string_type": "E-FIELD-TEXT", "list_type": "E-FIELD-LIST",
        "dict_type": "E-FIELD-OBJECT", "model_type": "E-FIELD-OBJECT", "model_attributes_type": "E-FIELD-OBJECT",
        "bool_parsing": "E-FIELD-SWITCH", "bool_type": "E-FIELD-SWITCH", "greater_than_equal": "E-FIELD-TOOSMALL",
        "greater_than": "E-FIELD-TOOSMALL", "less_than_equal": "E-FIELD-TOOBIG", "less_than": "E-FIELD-TOOBIG",
        "too_long": "E-FIELD-TOOLONG", "base64_decode": "E-FIELD-BASE64"}


def _unreadable(request, exc: RequestValidationError) -> JSONResponse:
    first = (exc.errors() or [{}])[0]
    where = [str(x) for x in first.get("loc", ())]
    field = ".".join(where[1:]) or "请求体"  # the body itself (not an object, not JSON of the route's shape)
    code = _WHY.get(str(first.get("type", "")))
    why = Msg(code) if code else Msg("E-FIELD-OTHER", detail=str(first.get("msg", ""))[:200])
    said = Invalid(Msg("E-ACCESS-BADFIELD", field=field, why=why))
    return JSONResponse({**said.answer(), "field": field}, status_code=said.status)


app.add_exception_handler(MessageError, lambda request, exc: JSONResponse(exc.answer(), status_code=exc.status))
app.add_exception_handler(RequestValidationError, _unreadable)


router = Router()  # the editor's own routes (paths in full)
admin = Router(prefix="/api/admin", tags=["管理（/admin 页面）"])
pages = Router()  # the built web page: after every other route


# The machine-readable API description, for plugin and script authors (there is no help page; installing extensions
# from 后台「扩展包」 is a separate feature, server/installs.py)
@admin.get("/openapi.json", access=Access.admin("openapi.view"), summary="HTTP 接口的机器可读描述（OpenAPI），给写插件和脚本的开发人员")
def openapi() -> dict:
    return app.openapi()


app.include_router(logs.router)
app.include_router(tools.router)
app.include_router(farm.router)
app.include_router(packets.router)
app.include_router(view.router)
app.include_router(transfer.router)
app.include_router(settings.public)
app.include_router(notice.router)
app.include_router(feedback.router)
app.include_router(auth.router)
app.include_router(register.router)  # the login page's 「注册」 (lab2shot/registration.py)
app.include_router(terms.router)  # 用户协议 and 隐私政策 (lab2shot/terms), and agreeing to them
app.include_router(library.router)  # 我的模板 (user templates): saving to the server, opening, moving to the bin, restoring
app.include_router(quota.router)  # the account's own disk usage and cleanup
app.include_router(releases.router)  # the top bar's 「更新说明」 (lab2shot/releases.py)
for part in (farm, installs, records, settings, notice, terms, feedback, logs, users, invites, admin_templates, library, categories, quota):  # each module's own admin routes
    app.include_router(part.admin)
app.include_router(admin)


@router.get("/api/catalog", access=Access.user("编辑器：数据类型和这个账号能用的节点类型", hides={
    # GPU figures only 显卡详情 (farm.cards) sees (nodes/applies.py ResolvedCost; a setting's measured VRAM: access.params_for)
    "farm.cards": ("nodes[].at_defaults.cost.vram_gb", "nodes[].at_defaults.cost.vram_measured",
                   "nodes[].at_defaults.cost.measured_on")}, keyed=revisions.catalog), tags=["节点图"], summary="所有数据类型、场景里的几种数据、节点菜单（两个区、分类树、每个节点归在哪）、数值的单位、标签，和这个账号能用的节点类型（端口、参数、参数的输入口、3D 输出设置写得了什么）")
def catalog(request: Request) -> dict:
    from ..nodes import text
    from ..nodes.registry import node_types

    u = auth.me(request)
    text.refresh(node_types())  # a nodes.json restored or edited outside the server: its words before this answer (under its lock)
    usable = access.node_types_for(u)
    return {
        "version": __version__,
        "types": describe_types(),
        "layer_ports": describe_layer_ports(),  # port name -> the EXR layer it is delivered as (the page fills in the layer name from it when a wire is drawn)
        "scene_kinds": describe_scene_kinds(),
        # The node menu: two sections, the category tree and each node's placement (lab2shot/categories.py: all data
        # files, edited by administrators in the menu)
        "menu": trees.describe_menu(usable),
        "units": {u: {"kind": kind, "kind_label": UNIT_KINDS[kind], "factor": f} for u, (kind, f) in UNITS.items()},  # which convert
        "tags": tags.describe(),
        # how many built-in templates this account may use: the top bar says it beside 「模板」 and must not download
        # all of them, with their graphs, to count
        "templates": len(access.templates_for(u)),
        **type_tables(),  # accepts, converters: looked up while a wire is drawn (webui/src/graph/rules.ts)
        "nodes": [_with_defaults(access.describe_for(auth.session(request), n), n) for n in usable.values()],
    }


@lru_cache(maxsize=None)
def _at_defaults(node_type) -> dict:
    """A node type's ports and handles on its own at its defaults (engine Graph.at_defaults), once per type. A type that
    cannot stand alone at its defaults shows none until its status arrives: one broken type never takes the catalogue."""
    from ..engine import Graph

    try:
        return Graph.at_defaults(node_type.id)
    except (ValueError, KeyError, OSError):  # GraphError is a ValueError
        from ..logs import get as log

        log("catalog").warning("%s: no ports at its defaults", node_type.id, exc_info=True)
        return {"ports": {"inputs": [], "outputs": [], "waiting": []}, "handles": []}


def _with_defaults(desc: dict, node_type) -> dict:
    """The catalogue's node with its ports and handles at its defaults beside its cost and licence: what a node just
    added shows until the status reply for it arrives."""
    return {**desc, "at_defaults": {**desc["at_defaults"], **_at_defaults(node_type)}}


def _node_type(request: Request, type_id: str):
    """A node type the account may use; any other is not there (the same words whether it exists or not)."""
    node = access.node_types_for(auth.me(request)).get(type_id)
    if node is None:
        raise NotFound(Msg("E-GRAPH-NODETYPE"))
    return node


class DeriveRequest(Body):
    params: dict


@router.post("/api/nodes/{type_id}/derive", access=Access.user("编辑器：从参数推出来的参数，比如选的文件有哪些图层（只能是自己上传的文件）"), tags=["节点图"], summary="节点从别的参数推出来的参数（读取序列：选的文件有哪些图层、各是什么）")
def derive(type_id: str, req: DeriveRequest, request: Request) -> dict:
    node = _node_type(request, type_id)  # an upload of another account is not there for it (uploads.resolve)
    return node.derive(node.load_params(req.params))


class PasteRequest(Body):
    text: str


@router.post("/api/nodes/{type_id}/paste", access=Access.user("编辑器：把别的软件的一段文字读成这个节点的一组参数"), tags=["节点图"], summary="粘进来的一段文字 → 这个节点的一组参数值（3DE / Nuke 的镜头数据 →「LensDistortion」的整张镜头表）")
def paste(type_id: str, req: PasteRequest, request: Request) -> dict:
    """The reverse of copying to other applications: a node declares which application it can read
    (`Pasteable.paste`) and recognizes its own format (`read_pasted`). This route knows no application or format; it
    only hands the text to the node class.

    When the text cannot be read, the node raises Invalid with the reason (no lens, several lenses, a model that
    cannot be read yet, coefficients out of range), and the page shows it to the user as is. Pasted text is user data:
    it is read only as text, never executed or treated as instructions."""
    node = _node_type(request, type_id)
    if not getattr(node, "paste", ""):
        raise NotFound(Msg("E-GRAPH-NODETYPE"))  # an undeclared node type: this route does not exist for it
    return node.read_pasted(req.text)


class ChoicesRequest(Body):
    params: dict
    inputs: dict[str, str] = {}  # port -> fingerprint of the packet cooked for it


@router.post("/api/nodes/{type_id}/choices", access=Access.user("编辑器：参数里来自输入的选项（只能是自己节点图的结果）", owned=owners.packet_inputs), tags=["节点图"], summary="参数的可选项里来自输入的部分（补帧：接进来的骨骼有哪些关节、自动猜的是哪个）")
def choices(type_id: str, req: ChoicesRequest, request: Request) -> dict:
    from ..data.packet import Packet, packet_dir

    node = _node_type(request, type_id)  # an upload of another account is not there for it (uploads.resolve)
    inputs = {port: Packet.load(packet_dir(fp)) for port, fp in req.inputs.items() if Packet.exists(packet_dir(fp))}
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        return node.choices(node.load_params(req.params), representative(node, inputs, scratch))


def representative(node, inputs: dict, scratch: str) -> dict:
    """网页没有这个口自己的包时会送来能代表它的包（状态回复里节点的 stand_ins：engine/evaluation.py Evaluation.stand_ins）：逐项块里的节点收到的是块外
    的列表，或者「拆成列表」还没算时它要拆的那份整数据。这个口要单个而收到列表：取列表的第一条；收到这个口不收的整份
    数据而它能拆（data/items.py）：拆出第一条这个口收得下的（写在 `scratch` 临时文件夹里，只读来答这一次，不进缓存）。同一个解算器
    每个人的骨架相同，拿第一个人配对就够（「动作重定向」的「对应关系」在重定向块跑之前就能编辑）。"""
    from pathlib import Path

    from ..data.items import names as item_names, split as split_item
    from ..data.packet import Packet, items_of, packet_dir
    from ..data.types import accepts, is_list

    wants = {p.name: p.type for p in node.inputs}
    out = dict(inputs)
    for port, packet in inputs.items():
        want = wants.get(port)
        if not want:
            continue
        if is_list(packet.type) and not any(is_list(t) for t in want.split("|")):
            first = items_of(packet)[:1]
            got = Packet.load(packet_dir(first[0][1])) if first and Packet.exists(packet_dir(first[0][1])) else None
        elif not accepts(want, packet.type) and item_names(packet):
            # 第一条这个口收得下的（整个场景里可能先是相机）
            # （拆出来的一条类型可能仍是笼统的「场景」：没有收得下的就用第一条笼统的，让节点自己读；别的类型，例如相机，
            # 排最后）
            got, rank = None, 3
            for i, name in enumerate(item_names(packet)):
                where = Path(scratch) / f"{port}_{i}"
                where.mkdir()
                one = split_item(packet, name, where, lambda items, work: (work(x) for x in items))
                r = 0 if accepts(want, one.type) else 1 if one.type == packet.type else 2
                if r < rank:
                    got, rank = one, r
                if r == 0:
                    break
        else:
            continue
        if got is None:
            out.pop(port)
        else:
            out[port] = got
    return out


@router.get("/api/templates", access=Access.user("编辑器：这个账号能用的模板", keyed=revisions.templates), tags=["节点图"], summary="这个账号能用的模板：名字、简介、归在哪个分类、节点图、用到的项目、许可、来源和文件属性，加上模板面板的分类树；管模板的账号还拿到关掉的")
def templates(request: Request) -> dict:
    from ..extensions import extensions
    from ..nodes import tags as nodes_tags

    exts = extensions()
    u = auth.me(request)

    def card(t: dict) -> dict:
        return {**{k: t[k] for k in ("id", "name", "intro", "deliverable", "category", "graph", "licence", "owner", "adapter",
                                     "path", "created", "updated", "bytes")},
                "author": library_cards.author_of(t),
                # a template the administrator turned off: only a login that manages them gets it at all (access.py),
                # and it is told so, to draw it greyed as 「已停用」
                "enabled": t["enabled"],
                "projects": [{"name": r, "title": exts[r].title if r in exts else r} for r in t["runtimes"]],
                # The year in the card's lower-right corner: the paper or release year of the card's core project
                # (engine/templates.py core_project searches upstream from the graph's end), read from that project's
                # docs.md (Extension.year)
                "year": exts[t["project"]].year if t["project"] in exts else None,
                # one word on the card: the strictest licence its parts carry, the tag table's own
                # word, and whether that word means results may be used commercially (the page never reads the word)
                "licence_word": nodes_tags.strictest(frozenset(t["licence"])),
                "commercial": nodes_tags.commercial(frozenset(t["licence"])),
                # the best route another choice of its menus gives (「默认路线 非商用；有可商用路线」): its word and whether
                # that word means commercial use; the same as the chip's when no choice does better
                "best_licence_word": nodes_tags.strictest(best := nodes_tags.least_strict([frozenset(r) for r in t["route_licences"]])),
                "best_commercial": nodes_tags.commercial(best)}

    # the cards with the tree they sit in (lab2shot/categories.py templates: the administrator's file) and, when that
    # file cannot be read, the one sentence the panel shows (every card is 未分类 then)
    return {"templates": [card(t) for t in access.templates_for(u)], "categories": trees.templates.tree(),
            "problem": trees.templates.problem()}


@router.get("/api/tls/ca.pem", access=Access.open("用 HTTPS 时用户电脑要装的证书：证书本来就是公开的，装了才打得开登录页"), tags=["设置"], summary="这台服务器的证书（用 HTTPS 时，每台用户电脑装一次）")
def tls_authority() -> FileResponse:
    from . import tls

    f = tls.authority_file()
    if f is None:
        raise NotFound(Msg("E-SERVER-NOHTTPS"))
    return FileResponse(f, media_type="application/x-x509-ca-cert", filename="lab2shot-ca.crt")


@router.get("/api/ocio", access=Access.user("编辑器：色彩空间列表"), tags=["设置"], summary="当前 OCIO 配置的色彩空间和工作空间")
def ocio() -> dict:
    """The 「色彩空间」 dropdown options and the working space name (io/color.py: input is converted to it and output is
    converted from it; no display or view transform applies, since the working space is the sRGB shown on screen)."""
    from ..io.color import load_config, working_space

    cfg = load_config()
    return {
        "name": cfg.name,
        "origin": cfg.origin,
        "colorspaces": [cs.getName() for cs in cfg.config.getColorSpaces()],
        "working": working_space(cfg),
    }


# ---------------------------------------------------------------- built UI (server/routes.py Access.page)


@pages.get("/{path:path}", access=Access.page("网页的入口 index.html（只是登录页的外壳）：别的地址都是它，不发任何别的文件"), include_in_schema=False)
def spa(path: str) -> FileResponse:
    """The page's entry (webui/dist/index.html: only the gate's shell) for the addresses the page answers: the guard
    lets only those reach here (server/access.py PAGE_ENTRY); the page's own files come from /assets, at their level."""
    index = WEBUI_DIST / "index.html"
    if not index.is_file():
        raise Unavailable(Msg("E-SERVER-NOPAGE"))
    return FileResponse(index, headers={"Cache-Control": wire.FRESH})


app.include_router(router)
mount(app, "/assets", wire.Assets(directory=WEBUI_DIST / "assets", check_dir=False), name="assets",
      access=Access.page("网页的脚本和样式（webui/dist/assets）：按构建清单分级，登录页的公开，别的要登录，管理页面的要管理员"))
app.include_router(pages)  # last: the addresses no other route answers
