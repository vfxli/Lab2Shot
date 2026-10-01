"""查看器所绘制、但数据包中不以原样保存的内容：视频帧的代理图、深度或位置图在场景中的点预览、浏览器应用于本地 EXR
的显示 LUT。此处的一切仅用于显示：不作为结果缓存，也不参与计算。"""

from __future__ import annotations

import numpy as np
from fastapi import Request
from fastapi.responses import FileResponse, Response

from . import owners
from .routes import Access, Router
from ..engine import Packet
from ..data.packet import packet_dir
from ..errors import Invalid, NotFound
from ..messages import Msg
from . import wire
from .wire import WAITS

router = Router(prefix="/api/view", tags=["视图"])  # 只提供本账号自己缓存里的结果：每条路由声明 owned=owners.packets


def _packet(fp: str) -> Packet:
    d = packet_dir(fp)
    if not Packet.exists(d):
        raise NotFound(Msg("E-ACCESS-NORESULT"))
    return Packet.load(d)


@router.get("/{fp}/video/{frame}.png", access=Access.user("看结果：视频的一帧", owned=owners.packets), summary="视频某一帧的**代理图**：按管理员设定的那一档（「设置 · 视图 · 视图代理尺寸」）等比缩、有损压缩；看了第一帧，其余的在后台补齐")
def video_frame(fp: str, request: Request, frame: int, g: str = "", px: int | None = None) -> FileResponse:
    """视频帧的代理（`lab2shot/view/proxy.py`）。视频像素位于容器中，先解码为显示用 PNG
    （`view/frames.py video_frames`，与「按通道取」读取的是同一张），再按 `px` 档位缩放并压缩。只有代理这一档，
    没有原图 / 打包 / 预览等其他获取方式。"""
    from ..view import proxy

    p = _packet(fp)
    if p.type != "video":
        raise Invalid(Msg("E-VIEW-NOTVIDEO"))
    if frame not in p.meta["frames"]:
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame))
    kept = wire.versioned(p, g) and px is not None  # without `px=` it is answered at the current tier: not kept
    px = proxy.tier_of(px)
    try:
        path = proxy.video_picture_file(p, frame, px)
    except FileNotFoundError:  # 容器头报告的帧数多于实际可解码的帧数（view/frames.py video_frames）
        raise NotFound(Msg("E-VIEW-NOFRAME", frame=frame)) from None
    wire.ahead(f"{fp}:video.{px}",
               lambda: [lambda f=f: proxy.video_picture_file(p, f, px) for f in p.meta["frames"]])
    return wire.picture_answer(path, kept)


LUT_SIZE, LUT_LO, LUT_HI = 48, -10.0, 8.0  # 浏览器使用的表：每边的点数，覆盖线性值的 log2 范围


def lut_for(space: str | None) -> dict:
    """将「转到工作空间」的变换烘焙为浏览器可查询的表：对数网格上每个点的 RGB，0..65535，小端，base64。
    浏览器没有 OCIO，因此变换烘焙一次后发送，而不是逐帧烘焙进像素（本机预览 EXR 也用这张表，webui/src/transfer/exr.ts）。
    工作空间即屏幕显示的 sRGB（io/color.py），因此「转到工作空间」即屏幕上的显示效果，没有额外的显示变换。

    `space` 为 None，或该色彩空间本身即工作空间（PNG、视频帧、读取节点已转换的数据包）、或不经过色彩管理（数值图）时，
    mode 返回 "raw"：像素值即屏幕上的值，浏览器原样绘制。该表只与（色彩空间, 工作空间）两项有关，与具体数据包无关，
    因此浏览器按 head 中的这几项缓存一份即可。`display`（工作空间名）和 `view`（"working"）是浏览器的缓存键。"""
    import base64

    from ..io.color import load_config, working_space, working_table

    cfg = load_config()
    head = {"display": working_space(cfg), "view": "working", "size": LUT_SIZE, "lo": LUT_LO, "hi": LUT_HI}
    if space is None:
        return {**head, "colorspace": "", "mode": "raw", "data": ""}
    cfg.colorspace(space)  # 配置中不存在的名称会报错说明
    table = working_table(cfg, space, LUT_SIZE, LUT_LO, LUT_HI)
    if table is None:
        return {**head, "colorspace": space, "mode": "raw", "data": ""}
    data = np.round(table * 65535).astype("<u2").tobytes()
    return {**head, "colorspace": space, "mode": "lut", "data": base64.b64encode(data).decode("ascii")}


@router.get("/lut", access=Access.user("本地预览 EXR：服务器「转到工作空间」的变换烤成的查找表"), summary="本地预览 EXR 用的查找表：服务器的 OCIO「转到工作空间」变换在对数网格上烤成（工作空间就是屏幕显示的 sRGB，没有另外的显示变换；space：节点的色彩空间，空着按文件名规则）")
def display_lut(space: str = "", file: str = "") -> dict:
    """使用者选择的 EXR 经本服务器「转到工作空间」变换后的效果（lut_for），供浏览器在服务器取得文件之前显示
    （webui/src/transfer/exr.ts）：每个网格点的 RGB，0..65535，小端，base64。
    mode "raw"：按原值显示（已是工作空间或数值图），与服务器的处理一致。"""
    from ..io.color import load_config

    from pathlib import PurePosixPath

    # 只传入文件名而非完整路径：`colorspace_for_file` 对匹配默认规则的图会 stat 该路径（检查文件是否存在、是否为
    # 浮点），传入完整路径等于允许登录用户逐一探测服务器上存在哪些文件。色彩空间规则本身只依据扩展名和文件名。
    return lut_for(space or load_config().colorspace_for_file(PurePosixPath(file.replace("\\", "/")).name or "plate.exr"))


@router.get("/{fp}/points", access=Access.user("看结果：深度图显示成的点云", owned=owners.packets, lane=WAITS), summary="深度图（配上相机）或位置图显示成的点云：描述和每一部分数据的地址")
def points(fp: str, request: Request, camera: str | None = None) -> Response:
    from .view_data import respond_description
    from .view_worker import describe

    p = _packet(fp)
    # 每一块的地址带有代次（wire.versioned）：深度 / 位置图的 g，和放置它的相机的 cg——块的字节同样取决于相机，
    # 相机按同一指纹重算（地址里的 camera= 不变）后，旧地址答 E-VIEW-STALE，浏览器与本机硬盘里的旧块不会被当成新的用
    query = ([f"camera={camera}", f"cg={_packet(camera).created or ''}"] if camera else []) + [f"g={p.created or ''}"]
    url = f"/api/view/{fp}/points/{{part}}?" + "&".join(query)
    return respond_description(describe(("points", fp, camera), url), request.headers.get("accept-encoding", ""))


@router.get("/{fp}/points/{part}", access=Access.user("看结果：点云的数据（一段）", owned=owners.packets, lane=WAITS), summary="点云预览的一部分数据（二进制，gzip）：每帧每个有值的像素一个点；超过「点云上限」就每 N 个点取一个，坐标一个位不变，视图里写着显示了多少")
def points_part(fp: str, part: str, request: Request, camera: str | None = None, g: str = "", cg: str = "") -> Response:
    """每个有值的像素，在查看器请求时按帧块分批读取（server/view_data.py）。

    抽稀是唯一的大小控制手段：超过「点云上限」（设置 view.points_max_mb）时每 N 个点保留一个。
    只对显示用副本抽稀，查看器始终显示所显示点数占总点数的比例；不存在低质量的预览档位。"""
    from .view_data import respond, stored_chunk
    from .view_worker import part as view_part

    p = _packet(fp)
    cam = _packet(camera) if camera else None
    # 两个包的代次都对上才是「一个地址一份字节」（地址见 points：g、cg）；带相机却没带 cg 的旧地址照答，但不让浏览器长留
    kept = wire.versioned(p, g) and (wire.versioned(cam, cg) if cam is not None else True)
    # 已预先生成并存盘的块（view_data.store_chunk）直接发文件，不经过视图 worker 的通道；存盘文件名带深度图和相机两者的代次
    stored = None
    if kept:
        seen_from = cam.created if cam is not None else 0
        stored = stored_chunk(("points", fp, camera), (p.created, seen_from), part, request.query_params)
    data = stored.read_bytes() if stored is not None else view_part(("points", fp, camera), part)
    return respond(data, request.headers.get("accept-encoding", ""), kept=kept)


