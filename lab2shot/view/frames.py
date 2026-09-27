"""查看器对图像和映射图数据包的显示：显示参考的 PNG 只生成一次并缓存在源文件旁边。frames_for_worker（同样的转换，
但用于 worker 输入而非查看器）及其辅助函数 worker_ready / _convert_for_worker 位于 lab2shot/data/payloads.py：
engine/external.py 在准备 worker 任务时需要它们，而 engine 位于本视图层之下。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..data.packet import Packet, file_path, packet_dir
from ..data.payloads import channel_list, file_at, has_alpha, image_files, read_map, window_of, worker_ready
from ..io import images
from ..io.color import load_config, to_srgb8, to_working_picture
from .encode import write_once


def view_png(target: Path, pixels: Callable[[], np.ndarray]) -> Path:
    """为查看器生成的画面，只生成一次（view/encode.py：多个查看器可能同时请求；`pixels()` 只由负责生成的一方
    执行）：8 位 RGB，或预乘的浮点 RGBA（PNG 以非预乘方式保存 alpha）。"""

    def write(part: Path) -> None:
        made = pixels()
        (images.write_image if made.shape[-1] == 4 else images.write_png)(part, made)

    return write_once(target, write)


def _first_of(p: Packet, path: Path) -> int:
    """显示 `path` 的第一帧（静帧为每一帧列出同一个文件：其视图只生成一次）。"""
    return min(f for f, q in image_files(p).items() if q == path)


def display_frame(p: Packet, frame: int, cache: Path | None = None) -> Path | None:
    """图像数据包在 `frame` 处供查看器显示的画面（None：没有该帧）：显示参考的 PNG 原样使用；带 alpha 的画面
    为 RGBA PNG（显示参考颜色、非预乘 alpha，即浏览器合成所用的形式，查看器的「棋盘格」和「Alpha」据此显示）；
    否则为 frames_for_worker 生成的帧，尚未生成时将该帧经同一 OCIO 视图转换。首次请求时生成，写入数据包的
    `_view`（或 `cache`，即流式节点的临时视图；不写入数据包自身的 `_view`，因为其最终范围要到镜头计算完成后才确定）：
    查看器无需等待整个镜头转换完成。"""
    path = file_at(p, frame)
    if path is None:
        return None
    cfg = load_config()
    if worker_ready(p, cfg):
        return path
    target = (cache or p.dir / "_view") / f"frame.{_first_of(p, path)}.png"
    if has_alpha(p):
        return view_png(target, lambda: to_working_picture(images.read_rgba(path), cfg, p.meta["colorspace"]))
    # 视图始终取画幅（显示窗口，即 meta 的 width x height）区域。frames_for_worker 生成的 `_display` 供解算器使用，
    # 它按数据窗口转换整张画布（payloads._convert_for_worker，带扩边的去畸变图即为整张画布），尺寸与画幅不同；
    # 若不加区分直接使用，同一帧的地址在解算前为画幅、解算后为画布，叠加的人物框和跟踪点会整体偏移一个扩边，
    # 通过相机查看的去畸变（view/proxy.py）也会按错误的尺寸计算。因此只有数据窗口即画幅时才复用 `_display`，
    # 否则只转换该帧的画幅区域（read_rgb 不指定 box 时读取的即显示窗口）。
    made = packet_dir(f"{p.fingerprint}_display")
    if window_of(p).same and Packet.exists(made):
        return file_at(Packet.load(made), frame)
    return view_png(target, lambda: to_srgb8(images.read_rgb(path), cfg, p.meta["colorspace"]))


def view_frames(p: Packet) -> dict[int, Path]:
    """查看器对图像数据包每一帧的显示（逐帧调用 display_frame）。"""
    return {f: display_frame(p, f) for f in image_files(p)}


def map_view_frame(p: Packet, frame: int, cache: Path | None = None) -> Path | None:
    """数值图在 `frame` 处供视图显示的 PNG（None：没有该帧），首次请求时生成。
    `cache`：边算边看时的临时文件夹，不写入数据包自身。

    只按数据包自身的范围生成一张，不带黑白点参数：数值图通过「按通道取」（server/packets.py frame_channel）获取，浏览器
    持有的是值本身（代理档位、尾数低位置零的半精度，半精度无法容纳的保留 float32：view/channels.py channel_blob），黑白点
    在显卡上计算，无需回服务器重绘。"""
    path = file_at(p, frame)
    if path is None:
        return None
    target = (cache if cache else p.dir / "_view") / f"frame.{_first_of(p, path)}.png"
    return view_png(target, lambda: images.as_uint8(_map_rgb(p, path)))


def _map_rgb(p: Packet, path: Path) -> np.ndarray:
    """将一帧数值图绘制为 RGB：只依据通道数和数据包自带的显示范围（meta range），不识别「深度」「遮罩」
    「法线」等名称：通道的含义由使用它的节点决定，视图不按名称判断。

    单通道 → 按范围映射的灰度；双通道 → 红绿两色（UV、ST-map 通常按此查看）；三或四通道 → 前三个通道按范围映射。
    范围反向（hi < lo）时翻转斜坡：「近亮远暗」一类显示方式由写出该图的节点通过 range 声明，而非由视图推测。
    无值的像素（valid 通道为 0）绘制为黑色。
    """
    lo, hi = p.meta.get("range", [0.0, 1.0])
    span = hi - lo if abs(hi - lo) > 1e-6 else 1e-6
    data, alpha = read_map(path)
    valid = alpha > 0
    scaled = np.clip((data - lo) / span, 0.0, 1.0)
    if data.shape[-1] == 1:
        return np.repeat(np.where(valid, scaled[..., 0], 0.0)[..., None], 3, axis=2)
    if data.shape[-1] == 2:  # u 为红，v 为绿
        return np.where(valid[..., None], np.concatenate([scaled, np.zeros_like(scaled[..., :1])], axis=2), 0.0)
    return np.where(valid[..., None], scaled[..., :3], 0.0)


# ------------------------------------------------------------------ 单个通道（按通道取）


VIDEO_AHEAD = 12  # 单独请求一帧时额外解码的帧数（见下方 video_frames 的说明）


def video_frames(p: Packet, frames: list[int], ahead: int = 0) -> None:
    """将视频数据包的若干帧解码为显示用 PNG（每张只写一次：view/encode.py），一次解码完成这些帧。

    视频的像素位于容器中，不像序列图那样每帧一个文件，因此某帧的画面需要先解码；解码结果
    即视图显示的画面，也是「按通道取」读取通道的来源（两条路径不得各自解码）。

    `ahead`：在所需的最后一帧之后额外解码的帧数（io/sources.py 不支持定位，解码视频只能从头解到该帧：取帧路径
    逐帧请求，每帧都从头解码一次将是 O(n²)；额外解码一小段，后续几帧被请求时即已存在）。

    容器头报告的帧数（stream.frames）可能与实际可解码的帧数不同：末尾无法解码的帧不会有 PNG，请求方按
    「没有该帧」处理（channel_values、proxy._shown），而不是返回 500。"""
    from ..io.color import load_config
    from ..io.sources import open_source

    if ahead and frames:
        last = max(frames)
        frames = [*frames, *(f for f in p.meta["frames"] if last < f <= last + ahead)]
    frames = [f for f in frames if not (p.dir / "_view" / f"frame.{f}.png").exists()]
    if not frames:
        return
    src = open_source(file_path(p, p.meta["path"]), start_frame=0)
    cfg = load_config()
    space = src.colorspace or cfg.colorspace_for_file(src.colorspace_hint)
    for f, rgb in src.iter_frames(frames):
        view_png(p.dir / "_view" / f"frame.{f}.png", lambda rgb=rgb: _display(rgb, cfg, space))


def _display(rgb, cfg, space) -> np.ndarray:
    return to_srgb8(rgb, cfg, space)


def channel_key(p: Packet, frame: int) -> int | None:
    """该帧通道数据的存储键：同一文件的多帧（静帧整段指向同一张图）视为同一份，只处理一次。
    None：该数据包没有此帧。"""
    if p.type == "video":
        return frame if frame in p.meta["frames"] else None
    path = file_at(p, frame)
    return None if path is None else _first_of(p, path)


def channel_values(p: Packet, frame: int, name: str) -> np.ndarray:
    """一帧中单个通道的值本身（[H, W] float32，画幅大小，从左上角开始），不附加任何显示处理：
    不映射黑白点、不着色、不合成，这些都在浏览器中计算。

    数据来源：序列图和数值图读取数据包自身的文件（EXR 按通道名读取，只读取所需的通道，其他通道不解码）；
    视频读取其解码出的显示 PNG（video_frames，与视图显示的是同一张）。
    值即数据自身的值：线性画面即为线性（显示变换由浏览器使用 /api/packet/{fp}/lut 的表完成），
    数值图即为数值（显示范围在 meta.range 中，由浏览器自行映射），带 alpha 的画面为预乘，与 Nuke 一致。
    """
    if name not in channel_list(p):
        raise ValueError(f"{p.type} has no channel {name!r}")
    if p.type == "video":
        video_frames(p, [frame], ahead=VIDEO_AHEAD)
        made = p.dir / "_view" / f"frame.{frame}.png"
        if not made.exists():  # 容器头报告的帧数多于实际可解码的帧数：该帧不存在（见 video_frames 的说明）
            raise FileNotFoundError(f"no frame {frame}")
        return images.read_named(made, [name])[name]
    path = file_at(p, frame)
    if path is None:
        raise ValueError(f"no frame {frame}")
    # box=None 即画幅本身（显示窗口，io/images.py read_named）：与显示图所见区域一致，
    # 数据包在画幅之外的溢出像素不进入这条视图路径。
    stored = _in_file(path, name)
    return images.read_named(path, [stored])[stored]


def _in_file(path: Path, name: str) -> str:
    """该声明通道在文件中的名称（channel_in_file，依据文件自身的通道名）。"""
    return channel_in_file(images.channel_names(path), name)


def channel_in_file(have: list[str], name: str) -> str:
    """数据包声明的通道在文件中的名称。数据包按类型声明 R G B（payloads.channel_list）；文件中有同名通道即使用它；
    原样引用的灰度图只有一个通道（Y），由其代替三个通道（与 io/images.py read_rgb 一致）。其他情况一律使用原名：
    双通道文件（R G 的 ST-map）请求 B 时即不存在 B，由 E-IMAGE-NOCHANNEL 如实报告；不能将不足三个通道的情况
    一律用第一个通道代替，否则 G 会读成 R。`lab2shot check` 覆盖这些情况。"""

    if name in have or not have:
        return name
    if name in ("R", "G", "B") and len(have) == 1:
        return have[0]
    return name
