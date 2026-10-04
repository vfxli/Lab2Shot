"""Graph status and cooked results (packets) for the viewer. Cooks are submitted through the farm (server/farm.py).
A graph is first admitted for the requesting account and answered from that account's own cache (data/store.py); a
result is read only from the requesting account's own cache (each route declares owned=owners.packets)."""

from __future__ import annotations


import numpy as np
from fastapi import Request
from fastapi.responses import FileResponse, Response
from pydantic import Field


from ..recent import Recent, packet_key
from .routes import Access, Router
from ..engine import EVALUATIONS, Evaluation, Graph, Packet
from ..engine.evaluations import content_key
from ..data.packet import packet_dir
from ..errors import Invalid, NotFound
from ..io.files import inside
from .. import i18n
from ..messages import Msg
from . import auth, graphs, owners, wire
from .wire import WAITS
from .graphs import GraphRequest

from .access import account_of

router = Router(prefix="/api", tags=["Node Status and Results"])



def graph_of(req: GraphRequest, request: Request) -> Graph:
    """Return the graph a request names (sent once per version, server/graphs.py), admitted for the requesting account
    (server/access.py admit)."""
    return graphs.resolve(req, request)[1]


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
    show: list[str] | None = None  # the outputs of it the viewer shows, as 「计算」 would submit them (JobRequest.show)
    # The item each 逐项处理 block is showing (begin -> item key); nodes inside a block report on that item. This is a
    # view setting of the page (webui/src/state/items.ts) and does not affect the cook, so the response carries the
    # same `cook_inputs` and the results are unchanged.
    view: dict[str, str] = Field(default_factory=dict)
    # the `key` of `handle_node`'s handle data the page already holds: the same comes back as its key alone (the
    # skeletons of 「骨架姿势」 are tens of KB and a status reply is asked for on every edit)
    handle_key: str = ""
    # the node whose handle data comes back; "" none. Which node's handles the page draws is the view's choice
    # (webui/src/state/handleView.ts: the open 「对应关系」 editor, or the pose handle being edited in the viewer), not
    # the displayed node's: the server does not guess it
    handle_node: str = ""
    # the nodes whose 「计算」 the page offers besides the shown one (a card's buttons: 「计算并打包」 on its 「输出」, a stage's
    # 「解算摄影机」): for each that can't be cooked now, why (`holds`), so every such button is greyed with its reason
    # before it is clicked, whatever the viewer shows (webui/src/graph/actions.ts planError)
    holds: list[str] = Field(default_factory=list)


AGAIN_MS = 2000  # how soon the page asks again while a file is being read for a listing


@router.post("/status", access=Access.user("Editor: whether each node of your own graph is cached and can cook", hides={
    "farm.cards": ("nodes.*.cost.vram_gb", "nodes.*.cost.vram_measured", "nodes.*.cost.measured_on", "nodes.*.cost.vram_full_gb")}), summary="The one question an editor asks per edit: whether each node is cached, whether it can cook and why not, ports, "
                                                                                                                "wires, cook policy, and the plan of the shown node (each version of the graph is sent once: graph, or key / "
                                                                                                                "base + patch, see server/graphs.py)")
def status(req: StatusRequest, request: Request) -> dict:
    """Return Evaluation.status plus server-side information: the graph version answered (its key and the echoed
    cook-input version) and the plan of the displayed node (farm/timings.py look)."""
    from ..engine.external import reading_now, readings
    from ..serving import account
    from ..farm import timings

    ev = evaluation_of(req, request)
    # the shown node's state and policy are of the cook 「计算」 would submit (its show), as its plan below is
    found = ev.status(view=req.view or None, shown={req.display: frozenset(req.show)} if req.display and req.show else None)
    plan = timings.look(ev, req.display, False, req.show) if req.display in ev.graph.nodes else None
    holds = {nid: said for nid in dict.fromkeys(req.holds) if nid != req.display and nid in ev.graph.nodes
             and (said := timings.look(ev, nid, False).get("error"))}
    about = handle_about(req, ev.graph)
    data = _handle_data(ev, about, found["nodes"].get(about) or {}) if about else None
    if data and data["key"] == req.handle_key:
        data = {"node": data["node"], "key": data["key"]}  # the page has these already
    # a file is being read in the background for a listing (engine/external.py ask_worker): the page asks again soon,
    # so what is read shows by itself, with nothing held on the server meanwhile
    mine = readings(account().user_id)
    again = {"again_ms": AGAIN_MS, **({"readings": mine} if mine else {})} if reading_now() else {}
    return {"graph": req.key or content_key(ev.graph), "cook_inputs": req.cook_inputs, **found, "plan": plan,
            **({"holds": holds} if holds else {}), **({"handle_data": data} if data else {}), **again}


@router.post("/readings/stop", access=Access.user("Editor: stop reading files in the background (read before Cook was clicked, so not in the queue)"),
             summary="Stop the file reading this account runs in the background (read automatically when an import node lists its "
                     "contents; there is no job, so the queue cannot cancel it): the reading processes end at once; returns how many "
                     "were stopped")
def stop_readings() -> dict:
    from ..engine.external import stop_readings as stop
    from ..serving import account

    return {"stopped": stop(account().user_id)}


def handle_about(req: StatusRequest, graph: Graph) -> str | None:
    """The node whose handle data a status reply carries: the one the page names (`handle_node`), when the graph has
    it; None without one, whatever node is displayed."""
    return req.handle_node if req.handle_node in graph.nodes else None


_HANDLE_DATA: Recent = Recent(16)  # the last few answers of NodeDef.handle_data, by what they were read from


def _handle_data(ev: Evaluation, nid: str, entry: dict) -> dict | None:
    """A node's handle data (StatusRequest.handle_node's; NodeDef.handle_data: what a 「骨架姿势」 handle draws), read from the packets
    standing for its inputs (its entry's `stand_ins`, the same the option editors read; a list or a whole the port does
    not take is turned into its item as for choices: app.py representative). Kept by (node type, parameters, those
    packets, as the account asking has them: whose they are and their generation, Packet.created): a status reply asked again
    with nothing changed reads nothing, another account's packets of the same fingerprint are never what it reads, and a
    recook is read afresh. None when the node declares no such handle or nothing is wired in yet; a fault reading it (a
    bug) leaves it out and is logged, never taking the reply."""
    import json
    import tempfile

    from .. import logs
    from ..serving import account
    from .app import representative

    node = ev.graph.nodes[nid]
    t = node.type
    if not any(h.wants_data for h in t.handles):
        return None
    stand = entry.get("stand_ins") or {}
    if not stand:
        return None
    try:
        inputs = {port: Packet.load(packet_dir(fp)) for port, fp in stand.items() if Packet.exists(packet_dir(fp))}
    except (OSError, ValueError):  # a packet removed or rewritten while this reply reads it: no handle data this time
        return None
    key = (t.id, json.dumps(node.params, sort_keys=True, default=str),
           tuple(sorted((port, packet_key(inputs[port]) if port in inputs else (account(), fp, ""))
                        for port, fp in stand.items())))
    data = _HANDLE_DATA.get(key)
    if data is None:
        try:
            with tempfile.TemporaryDirectory() as scratch:
                got = t.handle_data(t.load_params(node.params), representative(t, inputs, scratch))
        except Exception as exc:  # noqa: BLE001 the handle's data never takes the status reply down
            logs.say(logs.get("farm"), Msg("E-FARM-INTERNAL", detail=f"{type(exc).__name__}: {exc}"), logs.error_text(exc),
                     about="handle_data")
            return None
        for k, one in got.items():  # an item split into scratch is gone now: the packet the stage loads is the stand-in
            if "packet" in one and not Packet.exists(packet_dir(one["packet"])):
                one["packet"] = stand.get(t.handles[k].source or "", "")
        data = _HANDLE_DATA.put(key, {str(k): v for k, v in got.items()})
    if not data:
        return None
    from ..io.digest import key as digest_key

    digest = digest_key(key, 16)  # one way of making a key (io/digest.py), stable across processes (not repr)
    return {"node": nid, "key": digest, "handles": data}


ITEMS_PAGE = 50  # default page size of node_items; blocks with hundreds of items are read page by page
ITEMS_MOST = 200


@router.get("/status/{graph}/node/{node}/items", access=Access.user("Editor: the state of each item of a node in a block"), summary="Each item of a node in a block (the items processed one by one): state, whether cached, why it failed or was "
                                                                                                                                 "skipped, item name; fetched page by page (offset: from which item, limit: items per page, at most 200). The "
                                                                                                                                 "graph is named by the graph key of the status answer (the version the server remembers)")
def node_items(graph: str, node: str, request: Request, offset: int = 0, limit: int = ITEMS_PAGE) -> dict:
    """Return per-item status for one node inside a block. The status response is per node (a block node reports its
    summary and the item currently viewed); the per-item detail, which may cover hundreds of items, is requested here
    only for the node being inspected."""
    ev = evaluation_of(GraphRequest(key=graph), request)
    if node not in ev.graph.nodes:
        raise NotFound(Msg("B-COOK-NOSUCHNODE", node=node))
    if offset < 0 or limit < 1:
        raise Invalid(Msg("E-STATUS-ITEMSPAGE", most=ITEMS_MOST))
    # a fault of the program in one node's items is that node's error, never a 500 (Evaluation.guarded, as the status)
    found = ev.guarded("items", lambda: ev.items(node, offset, min(limit, ITEMS_MOST)),
                       lambda said: {"total": 0, "offset": offset, "items": [], "error": said.json()})
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


@router.get("/packet/{fp}", access=Access.user("View results: a packet's type", owned=owners.packets), summary="A packet's type, metadata and a summary (what the data is: size, frame range, Focal Length, number of "
                                                                                                                "people... the port tooltip and the info panel read it); images shown as the uploaded file itself also have "
                                                                                                                "each frame's content sha256 (blobs: a browser holding the same file need not download it)")
def manifest(fp: str, request: Request) -> dict:
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
        from ..view.proxy import form_of, sized, tier

        w, h = sized(int(p.meta.get("width", 0)), int(p.meta.get("height", 0)))
        # `form`: how its proxies are made when not the usual way (编号图 "ids": nearest, exact values; view/proxy.py
        # form_of). The page puts it in the address (`pf=`) and its cache keys beside the tier, so proxies made the
        # other way before (kept by the browser as immutable) are never taken for these
        form = form_of(p)
        out["proxy"] = {"px": tier(), "width": w, "height": h, **({"form": form} if form else {})}
    # Only 2D images can have a file shown to the browser unchanged. _shown_blobs reads colorspace and files, which
    # non-2D packets (cameras, point clouds) do not have, so it must not be called for them.
    if channels_of(p.type) and not is_data(p) and (blobs := _shown_blobs(p)):
        out["blobs"] = blobs
    return out


def _shown_blobs(p: Packet) -> dict[str, str]:
    """Return, per frame, the sha256 of the uploaded file shown unchanged in the viewer, so the page can use its local
    copy when it has one. Empty when frames are converted for viewing."""
    from ..data.payloads import image_files, shown_as_is
    from ..io.color import load_config
    from ..transfer import uploads

    if not shown_as_is(p, load_config()):
        return {}
    shas = {str(f): uploads.sha_of(path) for f, path in image_files(p).items()}
    return {f: sha for f, sha in shas.items() if sha}


def _from_here(frames: list[int], here: int) -> list[int]:
    """Return the frames reordered to start at `here` and wrap around, so background work stays ahead of playback."""
    at = next((i for i, f in enumerate(frames) if f >= here), 0)
    return frames[at:] + frames[:at]


@router.get("/packet/{fp}/frame/{frame}.png", access=Access.user("View results: one frame's viewer proxy image", owned=owners.packets), summary="The **proxy image** of one frame of a packet for the viewer (display space): made when cooked at the size the "
                                                                                                                                        "administrator set (Settings > View > Viewer Proxy Size, default 512), scaled to fit and lossy-compressed; the "
                                                                                                                                        "page only fetches this. Delivered files are always full size, full precision and lossless, unaffected by it. "
                                                                                                                                        "The content of an address never changes, so the browser keeps it")
def frame(fp: str, request: Request, frame: int, through: str = "", at: str = "", g: str = "", px: int | None = None, pf: str = "",
          cg: str = "") -> FileResponse:
    """Return the proxy picture of one frame (`lab2shot/view/proxy.py`).

    `through` (a camera packet's fingerprint) with `at` (that camera's path in the scene) views the frame through that
    camera: when the camera has lens distortion, the plate is undistorted with its lens
    (`view/proxy.py through_picture_file`) so that it matches the 3D projection. Without them the plain proxy is
    returned. The 3D stage passes them when viewing through a distorted camera (`webui/src/view/Stage3D.tsx`), with
    `cg`, that camera packet's generation: the undistorted picture depends on both packets, so the address is one
    sequence of bytes only when it carries both generations (as server/view.py points does).

    This route serves colour pictures (decided by `webui/src/transfer/route.ts`): the screen is 8-bit sRGB and a colour
    result is shown as it is, so one 8-bit lossy WebP is the smallest carrier of its three channels (about a tenth of
    the channel values). Everything but an HDRI is sRGB already; an HDRI is converted to sRGB when its proxy is made.
    Value maps and single channels use frame_channel below.

    There is a single quality level: the proxy is already compressed, with no lossless variant.

    The proxy is normally built at cook time (`view/proxy.py build`, handed to the Engine by
    `lab2shot/farm/queue.py`). It is built here on demand in two cases: the packet's proxies of this tier
    were never made, or the administrator changed the tier. The requested frame is built first and the rest follow in the background
    (`wire.ahead`)."""
    from ..data.payloads import channel_list
    from ..view import proxy

    p = _packet(fp)
    if not channel_list(p):
        raise Invalid(Msg("E-VIEW-NOTIMAGE"))
    if (nothing := _nothing(p)) is not None:  # empty packets have no files; answer the empty shape instead of a 500
        return nothing
    Packet.used(p.dir)  # viewing counts as use, so cleanup (least recently used first) does not evict viewed packets
    # the URL is immutable only with the generation (`g=`, packet created), tier (`px=`) and the proxy's form (`pf=`,
    # view/proxy.py form_of): an address naming another form (a page from before the form changed) is answered with
    # the proxy as it is made now, but not kept. Through a camera, its generation too (`cg=`): the same camera
    # fingerprint recomputed with another lens makes another picture (the file on disk is named by it already,
    # proxy.through_picture_file); a stale `cg` is E-VIEW-STALE, an address without one is answered but not kept
    cam = _packet(through) if through else None
    # the tier too: an address without `px=` is answered at the current tier, which the administrator may change
    kept = wire.versioned(p, g) and px is not None and pf == proxy.form_of(p) and (wire.versioned(cam, cg) if cam is not None else True)
    px = proxy.tier_of(px)
    picture = (lambda f: proxy.through_picture_file(p, f, cam, at, px)) if cam else (lambda f: proxy.picture_file(p, f, px))
    path = picture(frame)  # an unreadable user file (cannot open, missing channel) is FileProblem's own answer
    if path is None:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame))
    wire.ahead(f"{fp}:pic.{px}" + (f":through.{through}.{at}" if cam else ""),
               lambda: [lambda f=f: picture(f) for f in _from_here(proxy.frames_of(p), frame)])
    return wire.picture_answer(path, kept)


# ------------------------------------------------------------------ per-channel data (not rendered pictures)

# `/frame/{frame}.png` above serves a rendered display picture (the smallest carrier for combined colour channels).
# The route below serves one channel's raw values; picking a channel, black / white points, tinting and compositing
# are done by the browser's display code (webui/src/view/look.ts). It serves proxies, whose size and compression are
# defined solely in lab2shot/view/proxy.py. Which route a view uses is decided solely by webui/src/transfer/route.ts.


@router.get("/packet/{fp}/frame/{frame}/channel/{name}", access=Access.user("View results: one channel of a frame", owned=owners.packets), summary="Proxy data of **one channel** of a frame (not a finished image): the browser draws it itself; picking "
                                                                                                                                                   "channels, black and white points, colouring and compositing are all done in the browser, so switching views "
                                                                                                                                                   "needs no network at all. The first 16 bytes give the format (u8 / u16 / half / float32, the GPU's R8 / R16 / "
                                                                                                                                                   "R16F / R32F) and width and height, then width x height values. The size is the tier the administrator set "
                                                                                                                                                   "(Settings > View > Viewer Proxy Size). The content of an address never changes, so the browser keeps it")
def frame_channel(fp: str, frame: int, name: str, request: Request, g: str = "", px: int | None = None,
                  pf: str = "") -> Response:
    """Return the proxy data of one channel of one frame.

    `name` must be one of the packet's channels as listed in /api/packet/{fp} `channels` (R G B A, plus valid when a
    validity channel was written). Any other name is refused: the name comes from the URL, so only known names are
    read and no file is touched otherwise.

    Proxies keep the channel structure: each channel produced by the solver is resized, compressed and served
    separately, and only the requested channel is sent. Values are the data's own values with no view applied: value
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
    kept = wire.versioned(p, g) and px is not None and pf == proxy.form_of(p)  # immutable only for this tier and form (see frame)
    px = proxy.tier_of(px)
    try:
        blob = proxy.channel_file(p, frame, name, px)
    except FileNotFoundError:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame)) from None
    wire.ahead(f"{fp}:ch.{name}.{px}",
               lambda: [lambda f=f: proxy.channel_file(p, f, name, px) for f in _from_here(proxy.frames_of(p), frame)])
    return wire.channel_answer(blob, request.headers.get("accept-encoding", ""), kept)


@router.get("/packet/{fp}/clipboard", access=Access.user("View results: node text for copying to another application", owned=owners.packets),
            summary="Node text carried by a result (a Nuke .nk snippet): the page's Copy to Nuke puts it on the clipboard")
def clipboard(fp: str, request: Request) -> dict:
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
    # put_clipboard); a name leading outside the packet, by its path or through a link, is none.
    try:
        f = inside(p.dir, name)
    except Invalid:
        f = None
    if not got.get("app") or f is None or not f.is_file():
        raise NotFound(Msg("E-CLIPBOARD-NONE"))
    # A message to show when copying (for content the target application cannot represent). The node records the code
    # and parameters; the text comes from the server's message catalogue (messages/) and the page only displays it.
    said = got.get("said")
    return {"app": got["app"], "file": name, "text": f.read_text(encoding="utf-8"),
            **({"said": Msg(said["code"], **(said.get("params") or {})).json()} if said else {})}


@router.get("/packet/{fp}/boxes", access=Access.user("View results: bounding boxes", owned=owners.packets), summary="Bounding box packet: each person's box in each frame")
def boxes(fp: str, request: Request) -> dict:
    from ..data.payloads import read_boxes

    p = _packet(fp)
    if p.type != "boxes":
        raise Invalid(Msg("E-VIEW-NOTBOXES"))
    if (nothing := _nothing(p, people=[])) is not None:
        return nothing
    return {**p.meta, "people": read_boxes(p)}  # meta["people"] holds only ids; the box data replaces it


@router.get("/packet/{fp}/tracks", access=Access.user("View results: tracks", owned=owners.packets), summary="Track packet: each point's position in each frame")
def tracks(fp: str, request: Request) -> dict:
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


@router.get("/packet/{fp}/curves", access=Access.user("View results: curves", owned=owners.packets), summary="Curves packet, or a value per frame: each curve's value in each frame")
def curves(fp: str, request: Request) -> dict:
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
    names = ["X", "Y", "Z"] if p.type == VECTOR else [p.meta.get("curve") or i18n.t("server.curve_value")]
    return {"frames": list(v.frames), "names": names, "unit": v.unit,
            "range": [float(values.min()), float(values.max())], "values": np.round(values, 5).T.tolist()}


@router.get("/packet/{fp}/scene", access=Access.user("View results: the 3D viewport's description", owned=owners.packets, lane=WAITS), summary="The description for the 3D viewport: geometry, skinned characters, point clouds, cameras, lights, and the "
                                                                                                                                  "address of each part of the data")
def scene(fp: str, request: Request) -> Response:
    from .view_data import respond_description
    from .view_worker import describe

    p = _packet(fp)  # part URLs carry the packet generation, so a recook yields new URLs (wire.versioned)
    return respond_description(describe(("scene", fp), f"/api/packet/{fp}/view/{{part}}?g={p.created or ''}"), request.headers.get("accept-encoding", ""))


@router.get("/packet/{fp}/view/{part}", access=Access.user("View results: the 3D viewport's data (one part)", owned=owners.packets, lane=WAITS), summary="Part of the 3D viewport's data (binary, gzip): small parts first, then large static data, then per frame")
def scene_part(fp: str, part: str, request: Request, g: str = "") -> Response:
    """Return one part of the 3D view data. Each URL maps to a single immutable byte sequence with no preview variant:
    3D data is small, and point clouds are decimated to the 「点云上限」 setting."""
    from .view_data import respond, stored_chunk
    from .view_worker import part as view_part

    p = _packet(fp)
    kept = wire.versioned(p, g)
    # made ahead and kept on disk (view_data.store_chunk): sent as it is, without asking a view worker lane
    stored = stored_chunk(("scene", fp), (p.created,), part, request.query_params) if kept else None
    data = stored.read_bytes() if stored is not None else view_part(("scene", fp), part)
    return respond(data, request.headers.get("accept-encoding", ""), kept=kept)
