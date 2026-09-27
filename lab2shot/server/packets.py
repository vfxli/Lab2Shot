"""Graph status and cooked results (packets) for the viewer. Cooks are submitted through the farm (server/farm.py).
A graph is first admitted for the requesting account; its status response grants the account access to the results it
references, and a result can be read only once granted (server/access.py)."""

from __future__ import annotations

import json

import numpy as np
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from pathlib import Path

from .routes import Access, Router
from ..engine import EVALUATIONS, Evaluation, Graph, Packet
from ..engine.evaluations import content_key
from ..data.packet import packet_dir
from ..errors import Invalid, NotFound, Unviewable
from ..io import FileProblem
from ..messages import Msg
from . import auth, graphs, wire
from .graphs import GraphRequest

from .access import account_of, admit, grant_packets, grant_status, packets_readable

router = Router(prefix="/api", tags=["节点状态与结果"], dependencies=[Depends(packets_readable)])



def graph_of(req: GraphRequest, request: Request) -> Graph:
    """Return the graph a request names (sent once per version, server/graphs.py), admitted for the requesting account
    (server/access.py admit)."""
    return admit(request, graphs.resolve(req, request))


def evaluation_of(req: GraphRequest, request: Request) -> Evaluation:
    """Return the Evaluation (engine/evaluation.py) of the request's graph from the shared cache
    (engine/evaluations.py). It is built once per (session, graph key, cache generation), so a repeated /api/status or
    /api/plan with no changes performs no reads. Admission (`graph_of`) still runs on every request: it costs a few
    milliseconds even for hundreds of nodes, and it ensures that permission changes and unknown node types are detected
    immediately. The cache key includes the account, because the same graph resolves differently per account: uploads
    of other accounts are treated as absent (transfer/uploads.py resolve)."""
    graph = graph_of(req, request)
    key = req.key or content_key(graph)
    return EVALUATIONS.get(auth.client_key(request), key, lambda: graph, account_of(auth.session(request)))


class StatusRequest(GraphRequest):
    cook_inputs: int = -1  # the page's cook-input version, echoed back; the page accepts only answers for its own version
    display: str | None = None  # the node shown in the viewer; its plan is included in the response
    # The item each 逐项处理 block is showing (begin -> item key); nodes inside a block report on that item. This is a
    # view setting of the page (webui/src/state/items.ts) and does not affect the cook, so the response carries the
    # same `cook_inputs` and the results are unchanged.
    view: dict[str, str] = Field(default_factory=dict)


@router.post("/status", access=Access.user("编辑器：自己节点图的每个节点有没有缓存、能不能算", hides={
    "farm.cards": ("nodes.*.cost.vram_gb", "nodes.*.cost.vram_measured", "nodes.*.cost.measured_on", "nodes.*.cost.rating.tip")}), summary="编辑器一次编辑只问这一个：每个节点有没有缓存、能不能算、为什么不能算，端口、连线、计算策略，和显示节点的计划（节点图每一版只发一次：graph、或 key / base + patch，见 server/graphs.py）")
def status(req: StatusRequest, request: Request) -> dict:
    """Return Evaluation.status plus server-side information: the graph version answered (its key and the echoed
    cook-input version) and the plan of the displayed node (farm/timings.py look)."""
    from ..farm import farm, timings

    ev = evaluation_of(req, request)
    found = ev.status(view=req.view or None)
    grant_status(request, ev, found)  # the account may read these results, since they come from its own graph
    plan = timings.look(ev, req.display, False, farm().models()) if req.display in ev.graph.nodes else None
    return {"graph": req.key or content_key(ev.graph), "cook_inputs": req.cook_inputs, **found, "plan": plan}


ITEMS_PAGE = 50  # default page size of node_items; blocks with hundreds of items are read page by page
ITEMS_MOST = 200


@router.get("/status/{graph}/node/{node}/items", access=Access.user("编辑器：块内一个节点每一条的状态"), summary="块内一个节点的每一条（逐项处理的条目）：状态、有没有缓存、出错或跳过的原因、条目名；一页一页取（offset 从第几条起，limit 一页几条，最多 200）。节点图按状态回复里的 graph 键指名（这一版服务器记得的那一版）")
def node_items(graph: str, node: str, request: Request, offset: int = 0, limit: int = ITEMS_PAGE) -> dict:
    """Return per-item status for one node inside a block. The status response is per node (a block node reports its
    summary and the item currently viewed); the per-item detail, which may cover hundreds of items, is requested here
    only for the node being inspected."""
    ev = evaluation_of(GraphRequest(key=graph), request)
    if node not in ev.graph.nodes:
        raise NotFound(Msg("B-COOK-NOSUCHNODE", node=node))
    if offset < 0 or limit < 1:
        raise Invalid(Msg("E-STATUS-ITEMSPAGE", most=ITEMS_MOST))
    found = ev.items(node, offset, min(limit, ITEMS_MOST))
    grant_packets(request, ev, (fp for i in found["items"] for fp in (i.get("outputs") or {}).values()))
    return {"graph": graph, "node": node, **found}


# ------------------------------------------------------------------ packets for the viewer


def _packet(fp: str) -> Packet:
    d = packet_dir(fp)
    if not Packet.exists(d):
        raise NotFound(Msg("E-ACCESS-NORESULT"))
    return Packet.load(d)


def _nothing(p: Packet, **shape) -> dict | None:
    """Return the empty response shape for an empty packet (a node that found or cooked nothing), or None for a packet
    with data. An empty packet has no payload, so the viewer receives the usual shape with no content, as for an empty
    scene (view_data.scene_view), rather than an error."""
    if not p.meta.get("empty"):
        return None
    return {"frames": [], "width": int(p.meta.get("width", 0)), "height": int(p.meta.get("height", 0)), "empty": True, **shape}


@router.get("/packet/{fp}", access=Access.user("看结果：数据包的类型"), summary="数据包的类型、元数据和一份摘要（这份数据是什么：尺寸、帧范围、Focal Length、人数……端口提示和数据信息面板都读它）；显示成原样上传文件的图像另有每帧内容的 sha256（blobs：浏览器有同样的文件就不用下载）")
def manifest(fp: str) -> dict:
    from ..data import summary as data_summary

    p = _packet(fp)
    # The summary is produced by the data layer from the type's declaration; the port tooltip, the 数据信息 panel and
    # 「取信息」 all read this response, and the page does not compose any of it.
    from ..data.payloads import is_data
    from ..data.types import channels_of

    from ..data.payloads import channel_list

    out = {"type": p.type, "meta": p.meta, "fingerprint": fp, "created": p.created, "summary": data_summary.describe(p)}
    # Channel names: frame_channel requests data by channel name, so the names must be available here. The type id
    # gives the channel count (image.3 has three) but not the names: packets with a valid channel have one more, and
    # video decodes to R G B. Names are declared by the core (data/payloads.py channel_list), not composed by the page.
    if names := channel_list(p):
        out["channels"] = {"names": list(names)}
        # Frame size in the viewer at the current proxy tier (administrator setting view.proxy_px). The page includes
        # it in its cache key (packet, frame, channel, proxy tier), so switching away and back hits the cache; after
        # the administrator changes the tier, a reload uses the new tier.
        from ..view.proxy import sized, tier

        w, h = sized(int(p.meta.get("width", 0)), int(p.meta.get("height", 0)))
        out["proxy"] = {"px": tier(), "width": w, "height": h}
    # Only 2D images can have a file shown to the browser unchanged. _shown_blobs reads colorspace and files, which
    # non-2D packets (cameras, point clouds) do not have, so it must not be called for them.
    if channels_of(p.type) and not is_data(p) and (blobs := _shown_blobs(p)):
        out["blobs"] = blobs
    return out


def _shown_blobs(p: Packet) -> dict[str, str]:
    """Return, per frame, the sha256 of the uploaded file shown unchanged in the viewer, so the page can use its local
    copy when it has one. Empty when frames are converted for viewing."""
    from ..data.payloads import image_files, worker_ready
    from ..io.color import load_config
    from ..transfer import uploads

    if not worker_ready(p, load_config()):
        return {}
    shas = {str(f): uploads.sha_of(path) for f, path in image_files(p).items()}
    return {f: sha for f, sha in shas.items() if sha}


def _from_here(frames: list[int], here: int) -> list[int]:
    """Return the frames reordered to start at `here` and wrap around, so background work stays ahead of playback."""
    at = next((i for i, f in enumerate(frames) if f >= here), 0)
    return frames[at:] + frames[:at]


@router.get("/packet/{fp}/frame/{frame}.png", access=Access.user("看结果：一帧的视图代理图"), summary="数据包某一帧给视图看的**代理图**（显示空间）：算完就按管理员设定的那一档（「设置 · 视图 · 视图代理尺寸」，默认 512）等比缩、有损压缩存起来，网页只拿这一份。交付写出去的文件永远是原尺寸、原精度、无损，不受它影响。内容按地址不变，浏览器一直留着")
def frame(fp: str, frame: int, through: str = "", at: str = "", g: str = "", px: int | None = None) -> FileResponse:
    """Return the proxy picture of one frame (`lab2shot/view/proxy.py`).

    `through` (a camera packet's fingerprint) with `at` (that camera's path in the scene) views the frame through that
    camera: when the camera has lens distortion, the plate is undistorted with its lens
    (`view/proxy.py through_picture_file`) so that it matches the 3D projection. Without them the plain proxy is
    returned. The 3D stage passes them when viewing through a distorted camera (`webui/src/view/Stage3D.tsx`).

    This route serves views that combine several colour channels (decided by `webui/src/transfer/route.ts`; the
    picture is the smallest carrier of the three channels after the OCIO display transform). Single-channel views use
    frame_channel below.

    There is a single quality level: the proxy is already compressed, with no lossless variant.

    The proxy is normally built at cook time (`view/proxy.py build`, handed to the Engine by
    `lab2shot/farm/queue.py`). It is built here on demand in two cases: the packet predates proxies, or the
    administrator has just changed the tier. The requested frame is built first and the rest follow in the background
    (`wire.ahead`)."""
    from ..data.payloads import channel_list
    from ..view import proxy

    p = _packet(fp)
    if not channel_list(p):
        raise Invalid(Msg("E-VIEW-NOTIMAGE"))
    if (nothing := _nothing(p)) is not None:  # empty packets have no files; answer the empty shape instead of a 500
        return nothing
    Packet.used(p.dir)  # viewing counts as use, so cleanup (least recently used first) does not evict viewed packets
    kept = wire.versioned(p, g)  # the URL is immutable only with the generation (`g=`, packet created) and tier (`px=`)
    px = proxy.tier_of(px)
    cam = _packet(through) if through else None
    picture = (lambda f: proxy.through_picture_file(p, f, cam, at, px)) if cam else (lambda f: proxy.picture_file(p, f, px))
    try:
        path = picture(frame)
    except FileProblem as exc:  # an unreadable user file (cannot open, missing channel): report it, not a 500
        raise Unviewable(exc.message) from None
    if path is None:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame))
    wire.ahead(f"{fp}:pic.{px}" + (f":through.{through}.{at}" if cam else ""),
               lambda: [lambda f=f: picture(f) for f in _from_here(proxy.frames_of(p), frame)])
    return wire.picture_answer(path, kept)


# ------------------------------------------------------------------ per-channel data (not rendered pictures)

# `/frame/{frame}.png` above serves a rendered display picture (the smallest carrier for combined colour channels).
# The two routes below serve raw data and the display transform; all viewing operations are computed in the browser.
# Both serve proxies, whose size and compression are defined solely in lab2shot/view/proxy.py.
# Which route a view uses is decided solely by webui/src/transfer/route.ts.


@router.get("/packet/{fp}/frame/{frame}/channel/{name}", access=Access.user("看结果：一帧里的一条通道"), summary="一帧里**一条通道**的代理数据（不是做好的图）：浏览器拿它自己画——取通道、黑白点、着色、合成都在浏览器算，切看法一次网络都不用。头 16 字节说清格式（u8 / u16 / 半精度 / float32，对应显卡的 R8 / R16 / R16F / R32F）和宽高，后面是宽×高个值。尺寸是管理员设定的那一档（「设置 · 视图 · 视图代理尺寸」）。内容按地址不变，浏览器一直留着")
def frame_channel(fp: str, frame: int, name: str, request: Request, g: str = "", px: int | None = None) -> Response:
    """Return the proxy data of one channel of one frame.

    `name` must be one of the packet's channels as listed in /api/packet/{fp} `channels` (R G B A, plus valid when a
    validity channel was written). Any other name is refused: the name comes from the URL, so only known names are
    read and no file is touched otherwise.

    Proxies keep the channel structure: each channel produced by the solver is resized, compressed and served
    separately, and only the requested channel is sent. Values are the data's own values with no view applied: linear
    images stay linear (the display transform is the table from /api/packet/{fp}/lut, applied in the browser), value
    maps stay numeric (display range in meta.range, mapped in the browser), and images with alpha are premultiplied.

    There is a single quality level: the proxy is already compressed."""
    from ..data.payloads import channel_list
    from ..view import proxy

    p = _packet(fp)
    Packet.used(p.dir)  # viewing counts as use (see frame)
    names = channel_list(p)
    if not names:
        raise Invalid(Msg("E-VIEW-NOTIMAGE"))
    if name not in names:
        raise NotFound(Msg("E-VIEW-NOCHANNEL", channel=name, channels=" ".join(names)))
    if (nothing := _nothing(p)) is not None:  # empty packets have no files (see frame)
        return nothing
    kept = wire.versioned(p, g)
    px = proxy.tier_of(px)
    try:
        blob = proxy.channel_file(p, frame, name, px)
    except FileNotFoundError:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame)) from None
    except FileProblem as exc:  # an unreadable user file (cannot open, missing channel): report it, not a 500
        raise Unviewable(exc.message) from None
    wire.ahead(f"{fp}:ch.{name}.{px}",
               lambda: [lambda f=f: proxy.channel_file(p, f, name, px) for f in _from_here(proxy.frames_of(p), frame)])
    return wire.channel_answer(blob, request.headers.get("accept-encoding", ""), kept)


@router.get("/packet/{fp}/lut", access=Access.user("看结果：这个数据包的显示变换（查找表）"), summary="这个数据包的显示变换，烤成浏览器能查的一张表（每个包要一次，不逐帧烤进像素）：OCIO 的显示变换在对数网格上的 RGB。数值图和本来就是显示空间的像素回 mode=raw（原样画）")
def packet_lut(fp: str) -> dict:
    """Return the packet's display transform baked into a lookup table. The browser has no OCIO, so the transform is
    baked once here and applied per frame in the browser (same format as /api/view/lut for local EXR preview).

    Value maps (depth, masks, normals and similar) are not colour managed, and video already carries display pixels;
    both return mode "raw"."""
    from ..data.payloads import channel_list, is_data
    from .view import lut_for

    p = _packet(fp)
    if not channel_list(p):
        raise Invalid(Msg("E-VIEW-NOTIMAGE"))
    raw = p.type == "video" or is_data(p) or not p.meta.get("colorspace")
    return lut_for(None if raw else p.meta["colorspace"])


@router.get("/packet/{fp}/clipboard", access=Access.user("看结果：复制到别的软件的节点文字"),
            summary="结果里带的节点文字（Nuke 的 .nk 片段）：网页的「复制到 Nuke」把它放进剪贴板")
def clipboard(fp: str) -> dict:
    """Return the snippet a node wrote for pasting into another application: the target application, the file name
    it was written as, and its text. The node declares it (nodes/clipboard.py Pasteable) and writes it into the result
    (via put_clipboard, or, for an output-settings node, among its delivered files); this route is format-agnostic.
    Results without a snippet answer not found.

    Any result type may carry a snippet, not only 文件: for example, a lens calibration's snippet is attached to the
    port carrying the lens, which is a 数值."""
    p = _packet(fp)
    got = p.meta.get("clipboard") or {}
    name = str(got.get("file") or "")
    # Only a file inside this result is accepted (a delivered file of an output-settings node, or the file written by
    # put_clipboard); paths leading outside the packet are rejected.
    inside = bool(name) and ".." not in name and not name.startswith("/") and p.path(name).is_file()
    if not got.get("app") or not inside:
        raise NotFound(Msg("E-CLIPBOARD-NONE"))
    # A message to show when copying (for content the target application cannot represent). The node records the code
    # and parameters; the text comes from the server's message catalogue (messages/) and the page only displays it.
    said = got.get("said")
    return {"app": got["app"], "file": name, "text": p.path(name).read_text(encoding="utf-8"),
            **({"said": Msg(said["code"], **(said.get("params") or {})).json()} if said else {})}


@router.get("/packet/{fp}/boxes", access=Access.user("看结果：人物框"), summary="人物框数据包：每个人每帧的框")
def boxes(fp: str) -> dict:
    from ..data.payloads import read_boxes

    p = _packet(fp)
    if p.type != "boxes":
        raise Invalid(Msg("E-VIEW-NOTBOXES"))
    if (nothing := _nothing(p, people=[])) is not None:
        return nothing
    return {**p.meta, "people": read_boxes(p)}  # meta["people"] holds only ids; the box data replaces it


@router.get("/packet/{fp}/tracks", access=Access.user("看结果：跟踪点"), summary="跟踪点数据包：每个点每帧的位置")
def tracks(fp: str) -> dict:
    """Return 2D tracks for the viewer: per point a list of [x, y] (null where hidden), aligned with meta.frames. For
    trackers that score their points, "confidence" gives per point and frame a value in 0..1 rounded to two decimals.
    A planar track (points with the plane's homography) also returns its outline on every frame, including estimated
    frames ("outline": per frame the ordered corner points and whether the plane was seen)."""
    from ..data.payloads import read_tracks

    p = _packet(fp)
    if p.type != "tracks2d":
        raise Invalid(Msg("E-VIEW-NOTTRACKS"))
    if (nothing := _nothing(p, points=[], names=[])) is not None:
        return nothing
    t = read_tracks(p)
    xy = np.round(t["tracks"], 2)
    points = [[[float(a), float(b)] if v else None for (a, b), v in zip(xy[i], t["visible"][i])] for i in range(len(xy))]
    out = {**p.meta, "points": points}
    if "confidence" in t:
        out["confidence"] = np.round(t["confidence"], 2).tolist()
    if "homography" in t:
        out["outline"] = [{"corners": xy[:, j].tolist(), "seen": bool(t["visible"][:, j].any())} for j in range(xy.shape[1])]
    return out


@router.get("/packet/{fp}/curves", access=Access.user("看结果：曲线"), summary="曲线数据包，或者每帧一个的数值：每条曲线每帧的值")
def curves(fp: str) -> dict:
    """Return curves for the viewer's strip: from a curves packet, from a 「SMPL 人体」 packet's parameters (three
    axis-angle curves per joint plus translation), or from a per-frame value (浮点 or 整数: one curve; 向量: three,
    X Y Z; 布尔: 0 and 1)."""
    from ..data.payloads import read_curves
    from ..data.values import LENS, TEXT, VECTOR, is_value, read

    p = _packet(fp)
    if (p.type == "curves" or is_value(p.type)) and (nothing := _nothing(p, names=[], values=[], range=[0.0, 0.0])) is not None:
        return nothing
    if p.type == "curves":
        return {**p.meta, "values": np.round(read_curves(p), 5).T.tolist()}  # per curve, frame-aligned
    if not is_value(p.type) or "frames" not in p.meta or p.type in (TEXT, LENS):  # text and lens intrinsics are not curves
        raise Invalid(Msg("E-VIEW-NOTCURVES"))
    v = read(p)
    values = np.asarray(v.values, np.float64).reshape(len(v.frames), -1)
    names = ["X", "Y", "Z"] if p.type == VECTOR else [p.meta.get("curve") or "值"]
    return {"frames": list(v.frames), "names": names, "unit": v.unit,
            "range": [float(values.min()), float(values.max())], "values": np.round(values, 5).T.tolist()}


@router.get("/packet/{fp}/scene", access=Access.user("看结果：3D 视图的描述"), summary="3D 视图用的描述：模型、蒙皮角色、点云、相机、灯光，以及每一部分数据的地址")
def scene(fp: str, request: Request) -> Response:
    from .view_data import respond_description
    from .view_worker import describe

    p = _packet(fp)  # part URLs carry the packet generation, so a recook yields new URLs (wire.versioned)
    return respond_description(describe(("scene", fp), f"/api/packet/{fp}/view/{{part}}?g={p.created or ''}"), request.headers.get("accept-encoding", ""))


@router.get("/packet/{fp}/view/{part}", access=Access.user("看结果：3D 视图的数据（一段）"), summary="3D 视图的一部分数据（二进制，gzip）：先是小的，再是大的静止数据，再按帧分段")
def scene_part(fp: str, part: str, request: Request, g: str = "") -> Response:
    """Return one part of the 3D view data. Each URL maps to a single immutable byte sequence with no preview variant:
    3D data is small, and point clouds are decimated to the 「点云上限」 setting."""
    from .view_data import respond
    from .view_worker import part as view_part

    kept = wire.versioned(_packet(fp), g)
    return respond(view_part(("scene", fp), part), request.headers.get("accept-encoding", ""), kept=kept)
