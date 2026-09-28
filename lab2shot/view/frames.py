"""查看器对图像和映射图数据包的显示：显示参考的 PNG 只生成一次，缓存在数据包的 `_view` 中。frames_for_worker（同样的转换，
但用于 worker 输入而非查看器）及其辅助函数 worker_ready / _convert_for_worker，以及查看器原样显示的判断 shown_as_is，
位于 lab2shot/data/payloads.py：engine/external.py 在准备 worker 任务时需要它们，而 engine 位于本视图层之下。
查看器原样显示的条件比 worker 宽：灰度 PNG 查看器原样显示（浏览器能解），交给 worker 时才转成 RGB。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..data.packet import Packet, file_path, packet_dir
from ..data.payloads import channel_list, file_at, has_alpha, image_files, is_data, read_map, shown_as_is, window_of
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
    否则为 frames_for_worker 生成的帧，尚未生成时将该帧从数据包的色彩空间转成 sRGB 8 位（只有 HDRI 这类线性结果需要转）。首次请求时生成，写入数据包的
    `_view`（或 `cache`，即流式节点的临时视图；不写入数据包自身的 `_view`，因为其最终范围要到镜头计算完成后才确定）：
    查看器无需等待整个镜头转换完成。"""
    plan = _display_plan(p, frame, cache)
    return plan if plan is None or isinstance(plan, Path) else view_png(*plan)


def _display_plan(p: Packet, frame: int, cache: Path | None) -> Path | tuple[Path, Callable[[], np.ndarray]] | None:
    """display_frame 如何显示该帧：原样显示的文件，或 (要生成的 PNG, 生成它的像素)；None：没有该帧。"""
    path = file_at(p, frame)
    if path is None:
        return None
    cfg = load_config()
    if shown_as_is(p, cfg):
        return path
    target = (cache or p.dir / "_view") / f"frame.{_first_of(p, path)}.png"
    if has_alpha(p):
        return target, lambda: to_working_picture(images.read_rgba(path), cfg, p.meta["colorspace"])
    # 视图始终取画幅（显示窗口，即 meta 的 width x height）区域。frames_for_worker 生成的 `_display` 供解算器使用，
    # 它按数据窗口转换整张画布（payloads._convert_for_worker，带扩边的去畸变图即为整张画布），尺寸与画幅不同；
    # 若不加区分直接使用，同一帧的地址在解算前为画幅、解算后为画布，叠加的人物框和跟踪点会整体偏移一个扩边，
    # 通过相机查看的去畸变（view/proxy.py）也会按错误的尺寸计算。因此只有数据窗口即画幅时才复用 `_display`，
    # 否则只转换该帧的画幅区域（read_rgb 不指定 box 时读取的即显示窗口）。
    made = packet_dir(f"{p.fingerprint}_display")
    if window_of(p).same and Packet.exists(made):
        return file_at(Packet.load(made), frame)
    return target, lambda: to_srgb8(images.read_rgb(path), cfg, p.meta["colorspace"])


def shown_now(p: Packet, frame: int) -> Path | np.ndarray | None:
    """该帧供查看器显示的画面（图像数据包即 display_frame，数值图即 map_view_frame），给马上用它生成代理图的一方
    （view/proxy.py 的显示图代理）：有现成的文件（显示参考的原文件、`_display`、之前已生成的 PNG）即为该文件；
    否则为本应写入 PNG 的 8 位 RGB 像素本身，不写 PNG。None：没有该帧。

    视图只接收代理，这张原尺寸 PNG 只在生成代理时被读回一次；写它（1080p 一帧约 0.1–0.2 秒、1–2 MB）比生成
    代理的其余步骤加起来还慢，100 帧的镜头在 `_view` 中多占约 200 MB。像素相同（PNG 无损），代理的字节相同。
    带 alpha 的画面仍写 PNG 再读：PNG 存非预乘 alpha，读回时 OpenImageIO 重新预乘，代理用的是这一往返后的值。"""
    plan = (_map_plan if is_data(p) else _display_plan)(p, frame, None)
    if plan is None or isinstance(plan, Path):
        return plan
    target, pixels = plan
    if target.exists() or (not is_data(p) and has_alpha(p)):
        return view_png(target, pixels)
    return pixels()


def map_view_frame(p: Packet, frame: int, cache: Path | None = None) -> Path | None:
    """数值图在 `frame` 处供视图显示的 PNG（None：没有该帧），首次请求时生成。
    `cache`：边算边看时的临时文件夹，不写入数据包自身。

    只按数据包自身的范围生成一张，不带黑白点参数：数值图通过「按通道取」（server/packets.py frame_channel）获取，浏览器
    持有的是值本身（代理档位、尾数低位置零的半精度，半精度无法容纳的保留 float32：view/channels.py channel_blob），黑白点
    在显卡上计算，无需回服务器重绘。"""
    plan = _map_plan(p, frame, cache)
    return plan if plan is None else view_png(*plan)


def _map_plan(p: Packet, frame: int, cache: Path | None) -> tuple[Path, Callable[[], np.ndarray]] | None:
    """map_view_frame 要生成的 PNG 与生成它的像素；None：没有该帧。"""
    path = file_at(p, frame)
    if path is None:
        return None
    target = (cache if cache else p.dir / "_view") / f"frame.{_first_of(p, path)}.png"
    return target, lambda: images.as_uint8(_map_rgb(p, path))


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
    from concurrent.futures import ThreadPoolExecutor

    from ..engine.cook import FRAME_THREADS
    from ..io.parallel import in_flight
    from ..serving import carried

    src = open_source(file_path(p, p.meta["path"]), start_frame=0)
    cfg = load_config()
    space = src.colorspace or cfg.colorspace_for_file(src.colorspace_hint)

    @carried  # 在该数据包所属账号的缓存中写（线程池的线程本身不属于任何账号）
    def made(f: int, rgb) -> None:
        view_png(p.dir / "_view" / f"frame.{f}.png", lambda: _display(rgb, cfg, space))

    # 解码只能按顺序一帧接一帧（一帧 1080p 约 8 ms），转换并写出 PNG（约 0.2 秒，OpenImageIO 写 PNG 时释放解释器锁）
    # 交给多个线程同时做：否则 100 帧的视频光写 PNG 就要 20 秒。同时在做的帧数有上限，解出的帧不会堆满内存。
    with ThreadPoolExecutor(FRAME_THREADS, thread_name_prefix="l2s-video") as pool:
        for _ in in_flight(pool, ((f, made, f, rgb) for f, rgb in src.iter_frames(frames)), FRAME_THREADS * 2):
            pass


def _display(rgb, cfg, space) -> np.ndarray:
    return to_srgb8(rgb, cfg, space)


def channel_key(p: Packet, frame: int) -> int | None:
    """该帧通道数据的存储键：同一文件的多帧（静帧整段指向同一张图）视为同一份，只处理一次。
    None：该数据包没有此帧。"""
    if p.type == "video":
        return frame if frame in p.meta["frames"] else None
    path = file_at(p, frame)
    return None if path is None else _first_of(p, path)


def channel_values(p: Packet, frame: int, names: list[str]) -> dict[str, np.ndarray]:
    """一帧中若干通道的值本身（通道名 -> [H, W] float32，画幅大小，从左上角开始），不附加任何显示处理：
    不映射黑白点、不着色、不合成，这些都在浏览器中计算。各通道从该帧的文件中一次读出（文件只打开、解码一次，
    不是每个通道各读一遍：JPEG、PNG 每读一个通道都要解码整张图）。

    数据来源：序列图和数值图读取数据包自身的文件（EXR 按通道名读取，只读取所需的通道，其他通道不解码）；
    视频读取其解码出的显示 PNG（video_frames，与视图显示的是同一张）。
    值即数据自身的值，不经过显示变换：
    数值图即为数值（显示范围在 meta.range 中，由浏览器自行映射），带 alpha 的画面为预乘，与 Nuke 一致。
    """
    have = channel_list(p)
    for name in names:
        if name not in have:
            raise ValueError(f"{p.type} has no channel {name!r}")
    if p.type == "video":
        video_frames(p, [frame], ahead=VIDEO_AHEAD)
        made = p.dir / "_view" / f"frame.{frame}.png"
        if not made.exists():  # 容器头报告的帧数多于实际可解码的帧数：该帧不存在（见 video_frames 的说明）
            raise FileNotFoundError(f"no frame {frame}")
        return images.read_named(made, list(names))
    path = file_at(p, frame)
    if path is None:
        raise ValueError(f"no frame {frame}")
    # box=None 即画幅本身（显示窗口，io/images.py read_named）：与显示图所见区域一致，
    # 数据包在画幅之外的溢出像素不进入这条视图路径。
    in_file = images.channel_names(path)
    stored = {name: channel_in_file(in_file, name) for name in names}
    got = images.read_named(path, list(dict.fromkeys(stored.values())))
    return {name: got[stored[name]] for name in names}


def channel_in_file(have: list[str], name: str) -> str:
    """数据包声明的通道在文件中的名称。数据包按类型声明 R G B（payloads.channel_list）；文件中有同名通道即使用它；
    原样引用的灰度图除 A 外只有一个通道（Y），由其代替三个通道（与 io/images.py read_rgb 一致）。其他情况一律使用原名：
    双通道文件（R G 的 ST-map）请求 B 时即不存在 B，由 E-IMAGE-NOCHANNEL 如实报告；不能将不足三个通道的情况
    一律用第一个通道代替，否则 G 会读成 R。`lab2shot check` 覆盖这些情况。"""

    if name in have or not have:
        return name
    grey = [c for c in have if c != "A"] if len(have) > 1 else have
    if name in ("R", "G", "B") and len(grey) == 1:
        return grey[0]
    return name
