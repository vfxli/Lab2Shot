"""视图代理：解算完成后，在同一次任务中将每一帧的二维结果等比缩放到管理员设定的档位（TIERS 三档之一）、
分别压缩后存为代理。视图只接收代理，没有无损路径；需要无损查看时，使用交付的文件在 Nuke / DCC 中查看。
代理的生成不向使用者报告进度，使用者看到的即为「已完成」。

代理的形式：每帧、每个通道一份（不合并为一张 RGB 图，使用者用到哪个通道就只传输哪个），分别缩放到该档位、
分别压缩，分别通过「按通道取」的地址发送。同时查看多个颜色通道时走图片路径
（由 `webui/src/transfer/route.ts` 决定，该图即三个通道经显示变换后打包成的最小载体），该图同样是
代理：同样缩放到该档位，同样有损压缩。

本层只负责查看：交付写出的文件始终为原尺寸、原精度、无损，不改动任何字节。

代理在此处统一生成，而不是在各 `adapters/<名称>/` 中：若由各适配器自行实现，必然会有遗漏。第三方接入新项目时，
代理自动具备：engine 每计算完一个数据包都会调用 `build()`（`lab2shot/farm/queue.py` 将该函数交给 Engine；
engine 本身不依赖 view 层，核心各层只向下 import）。

代理在 CPU 上生成：GPU 计算完成后应立即释放给下一个任务。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..data.packet import Packet
from ..data.payloads import channel_list, image_files, is_data
from .channels import channel_blob
from .encode import write_once
from .frames import VIDEO_AHEAD, channel_key, channel_values, display_frame, map_view_frame, video_frames

# 管理员可选的三档（长边像素）。本表是唯一数据源：设置项的选项（lab2shot/config.py view.proxy_px）由它生成。
TIERS = (512, 1024, 2048)

# 代理图的 WebP 质量。1920×1080 的实拍帧缩到 512 档时，一张代理约 2.4 KB，
# 而原尺寸无损显示图约 1.5 MB：缩放与压缩共节省三个数量级。
PROXY_QUALITY = 80
# 单个通道保留的尾数位数见 `view/channels.py HALF_MANTISSA`，此处不另存。


def tier() -> int:
    """本服务器当前的代理档位（长边像素数）：由一项设置决定，修改后立即生效，无需重启。不在三档中的值按最小档处理。"""
    from ..config import settings

    px = int(settings()["view.proxy_px"])
    return px if px in TIERS else TIERS[0]


def tier_of(px: int | None) -> int:
    """地址中携带的档位：属于三档之一时按其处理（页面按数据包说明中的 proxy.px 拼接地址），否则按当前设置。"""
    return px if px in TIERS else tier()


def sized(width: int, height: int, px: int | None = None) -> tuple[int, int]:
    """在该档位下，`width` × `height` 的画面缩放后的尺寸：等比缩放，长边为 `px`。

    小于该档位的源图不放大：放大等于凭空生成像素，且代理会比原图更大。400×300 的图保持 400×300。"""
    px = tier() if px is None else px
    longest = max(int(width), int(height))
    if longest <= px or longest <= 0:
        return int(width), int(height)
    scale = px / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


# ------------------------------------------------------------------ 等比缩放：按面积平均


@lru_cache(maxsize=8)  # 每份是 n_out x n_in 的 float32 稠密矩阵（4K 缩到 2048 一份约 30 MB）：只保留常用的几对，64 份可能占用超过 1 GB
def _weights(n_in: int, n_out: int) -> np.ndarray:
    """一个轴上将 `n_in` 个像素缩为 `n_out` 个的权重矩阵（每行之和为 1）：每个输出像素为其覆盖的
    源像素区间按面积加权的平均值。

    不使用双线性（`lab2shot/data/maps.py resize`）：双线性只考虑相邻两个源像素，1920 缩到 512 时每三个源像素中
    有一个完全未被采样，细纹会产生摩尔纹，而代理是使用者唯一能看到的画面。按面积平均是缩图的标准方法
    （OIIO 的 box、OpenCV 的 INTER_AREA 均为此法），每个源像素都参与计算，且不裁掉任何行列
    （整数倍下采样的 reshape 写法会裁掉余数，使画面整体偏移，叠加的人物框便无法对齐）。"""
    edges = np.linspace(0.0, n_in, n_out + 1)
    w = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        lo, hi = edges[i], edges[i + 1]
        for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            w[i, j] = min(hi, j + 1.0) - max(lo, float(j))
    total = w.sum(axis=1, keepdims=True)
    return w / np.where(total > 0, total, 1.0)


def shrink(values: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 [h, w] 或 [h, w, C] 的 float32 按面积平均缩放到 width × height（已是该尺寸时原样返回）。

    NaN 和无穷不参与平均：深度图、位置图中「该像素无值」即写为 NaN，若参与计算会使一大片
    变为 NaN。某块中没有任何有限值时输出 NaN（仍表示无值）。"""
    h, w = values.shape[:2]
    if (w, h) == (width, height):
        return np.ascontiguousarray(values, np.float32)
    flat = np.ascontiguousarray(values, np.float32).reshape(h, w, -1)
    good = np.isfinite(flat)
    vals = np.where(good, flat, 0.0).astype(np.float32)
    wy, wx = _weights(h, height), _weights(w, width)

    def across(a: np.ndarray) -> np.ndarray:
        # 先纵向（[h, w*C] 一次矩阵乘法），再横向（每个通道 [w] × [w, width]）：两步各为一次 BLAS 调用
        c = a.shape[2]
        down = (wy @ a.reshape(h, w * c)).reshape(height, w, c)
        return np.ascontiguousarray(down.transpose(0, 2, 1) @ wx.T).transpose(0, 2, 1)

    summed, seen = across(vals), across(good.astype(np.float32))
    out = np.where(seen > 0, summed / np.where(seen > 0, seen, 1.0), np.nan).astype(np.float32)
    # 必须是连续内存：上述两步转置后通道是跨步存储的，而 OpenImageIO 的 write_image 遇到这种数组时
    # （"Can't handle numpy array with noncontiguous channels"）只返回 False 而不抛出异常，写出的代理图全黑，
    # 且仍会被当作已完成的结果移到位并发送给使用者。
    return np.ascontiguousarray(out.reshape(height, width, *values.shape[2:]))


# ------------------------------------------------------------------ 单个通道的代理


def _blob(values: np.ndarray) -> bytes:
    """单个通道缩放后编码为传输字节（见 `lab2shot/view/channels.py channel_blob` 的格式表）。

    代理始终经过压缩，不存在「无损 / 压缩」两种模式，因此 `channel_blob` 只有一种编码方式：8 位数据本身即为
    一个字节，尾数低位置零既不节省空间也没有意义，它会自行识别。"""
    return channel_blob(values)


def channel_file(p: Packet, frame: int, name: str, px: int | None = None) -> Path:
    """该帧该通道的代理（gzip 文件），生成一次后保留（view/encode.py write_once）。

    名称中包含所属帧的文件、通道和档位：静帧整段指向同一个文件，`channel_key` 使其落在
    同一个名称上，只需生成一次；档位改变时名称随之改变，旧文件自然失效，新文件按需生成。"""
    px = tier() if px is None else px
    key = channel_key(p, frame)
    if key is None:
        raise FileNotFoundError(f"no frame {frame}")
    target = p.dir / "_view" / "proxy" / f"{key}.{name}.{px}.bin.gz"

    def write(part: Path) -> None:
        import gzip

        values = channel_values(p, frame, name)
        h, w = values.shape[:2]
        part.write_bytes(gzip.compress(_blob(shrink(values, *sized(w, h, px))), 5))

    return write_once(target, write)


# ------------------------------------------------------------------ 单帧显示图的代理


def picture_file(p: Packet, frame: int, px: int | None = None) -> Path | None:
    """该帧显示图的代理（None：该数据包没有此帧）。

    同时查看多个颜色通道时走此路径（`webui/src/transfer/route.ts`：该图即三个通道经 OCIO 显示
    变换后打包成的最小载体）。它同样是代理：缩放到该档位并使用有损 WebP，否则最常用的查看方式无法节省流量。"""
    px = tier() if px is None else px
    source = _shown(p, frame)
    if source is None:
        return None
    key = channel_key(p, frame)
    return picture_of(source, p.dir / "_view" / "proxy" / f"{key if key is not None else frame}.{px}", px)


def _shown(p: Packet, frame: int) -> Path | None:
    """该帧显示图本身（未缩放），适用于任何类型的二维像素数据包；None：没有此帧。
    视频的像素位于容器中，需先解码为显示 PNG（view/frames.py video_frames）；视频数据包的 meta 中没有 files，
    不能直接使用 display_frame。"""
    if p.type == "video":
        if channel_key(p, frame) is None:
            return None
        video_frames(p, [frame], ahead=VIDEO_AHEAD)
        made = p.dir / "_view" / f"frame.{frame}.png"
        return made if made.exists() else None  # 容器头报告的帧数多于实际可解码的帧数：该帧不存在（见 video_frames 的说明）
    return (map_view_frame if is_data(p) else display_frame)(p, frame)


def video_picture_file(p: Packet, frame: int, px: int | None = None) -> Path:
    """视频某帧显示图的代理（视频像素位于容器中，先解码为显示 PNG，与「按通道取」读取的是同一张）。"""
    px = tier() if px is None else px
    video_frames(p, [frame], ahead=VIDEO_AHEAD)
    source = p.dir / "_view" / f"frame.{frame}.png"
    if not source.exists():  # 容器头报告的帧数多于实际可解码的帧数：该帧不存在（见 video_frames 的说明）
        raise FileNotFoundError(f"no frame {frame}")
    return picture_of(source, p.dir / "_view" / "proxy" / f"{frame}.{px}", px)


def picture_of(source: Path, base: Path, px: int | None = None, write=None) -> Path:
    """任意显示图的代理。

    边算边看的路径同样使用它（`lab2shot/server/farm.py partial_frame`）：此时显示图生成在数据包的
    `_partial` 中，不进入数据包自身的 `_view`，但发送给使用者的仍只能是代理。

    `base` 为生成结果的存放位置：可写 WebP 时为 `.webp`，无法写出时（通道数不支持、编码器未安装）
    为 `.png`，文件名与其中的字节始终一致（发送时按后缀声明 Content-Type，声明错误即相当于发送了一张
    自称 WebP 的 PNG）。两种格式都是缩放后的结果：不得因编码器不可用而发送原尺寸图。生成一次后保留
    （view/encode.py write_once）。

    `write(source, part, px, kind)`：将该显示图生成为该结果的方式（默认为 `_write_picture`：缩放并编码）。
    通过带畸变的相机查看时改用 `_write_through`（缩放后再按该相机的镜头去畸变），其余相同。"""
    px = tier() if px is None else px
    write = _write_picture if write is None else write
    # 以追加方式拼接，而不使用 `with_suffix`：`base` 的名称中包含档位（`2001.512`），
    # `with_suffix` 会将 `.512` 视为后缀替换掉，导致代理图丢失档位、两个档位互相覆盖。
    webp, png = base.with_name(base.name + ".webp"), base.with_name(base.name + ".png")
    if png.is_file():   # 上次已发现该图无法写为 WebP：不再重复尝试
        return png
    try:
        return write_once(webp, lambda part: write(source, part, px, ".webp"))
    except OSError:
        return write_once(png, lambda part: write(source, part, px, ".png"))


def _write_picture(source: Path, part: Path, px: int, kind: str, warp=None) -> None:
    """将显示图缩放到该档位并编码为 `kind` 格式（无法写出时抛出 OSError，由 `picture_of` 改用另一种格式）。
    `warp(pixels)`：缩放后对该 [h, w, C] float32 执行的附加步骤（去畸变），没有时不执行。"""
    import OpenImageIO as oiio

    buf = oiio.ImageBuf(str(source))
    spec = buf.spec()
    pixels = buf.get_pixels(oiio.UINT8)
    width, height = sized(spec.width, spec.height, px)
    if (width, height) != (spec.width, spec.height) or warp is not None:
        pixels = shrink(pixels.astype(np.float32), width, height)
        pixels = np.rint(np.clip(pixels if warp is None else warp(pixels), 0, 255))
    # 必须连续：OIIO 的 write_image 收到通道不连续的数组时返回 False 而不抛出异常，留下一张全黑的图
    pixels = np.ascontiguousarray(pixels, np.uint8)
    out = oiio.ImageOutput.create(str(part))
    if out is not None:
        want = oiio.ImageSpec(width, height, spec.nchannels, oiio.UINT8)
        if kind == ".webp":
            # 只单独设置 CompressionQuality，不设置其他参数：OIIO 的 WebP 写入器只要设置了 `compression`
            # 就会忽略该值
            want.attribute("CompressionQuality", PROXY_QUALITY)
        if out.open(str(part), want):
            # 需检查是否写入成功（原因同上：失败时返回 False 而不抛出异常）
            wrote = out.write_image(pixels)
            out.close()
            if wrote:
                return
            part.unlink(missing_ok=True)
    # 内部错误（`picture_of` 捕获后改用另一种格式重试），不面向使用者，因此不进入消息目录
    raise OSError(f"cannot write a {kind} proxy of {source} ({spec.nchannels} channels)")


# ------------------------------------------------------------------ 通过带畸变的相机查看：底图按其镜头去畸变


def through_picture_file(p: Packet, frame: int, camera: Packet, at: str = "", px: int | None = None) -> Path | None:
    """通过带畸变的相机查看时该帧底图的代理：显示图缩放到该档位后，再按该相机自带的镜头去畸变
    （None：该数据包没有此帧）。相机不带畸变时即为普通的 `picture_file`。

    三维视图中的相机为针孔模型（页面 view/camera3d.tsx 按焦距和片门投影），针孔所见即去畸变后的画面；
    由带畸变模型（如 COLMAP 的鱼眼）解出的相机若以原图作为底图，点云与底图将无法对齐。因此底图需按相机所带的
    畸变去除后才能与点云贴合：镜头即相机数据包自带的那一套（data/camera.py lens_properties，由解算节点
    kit/cameras.py solved_camera 写入），使用者无需另接 LensDistortion / STMap。

    先缩放后去畸变（而非先去畸变后缩放）：代理档位只有几百像素，ST-map 按此尺寸计算，每帧几十毫秒；
    整幅 1080p 先去畸变再缩放约需一秒，对代理而言是无谓的开销。画布即画幅本身（fit_canvas 的 "none"）：三维相机的
    片门即画幅、没有扩边，底图也按同一框裁切。像素位置按比例换算到相机的画幅坐标后查询镜头
    （`Lens.raster`），因此底图与相机画幅尺寸不同（其他代理档位、上游已缩放）时也能对齐。

    文件名中包含镜头的摘要（`_camera_lens`）：同一台相机重新解出另一套畸变时即为另一份，旧文件自然失效。"""
    px = tier() if px is None else px
    lens, tag = _camera_lens(camera.fingerprint, at, _generation(camera))
    if lens is None:
        return picture_file(p, frame, px)
    source = _shown(p, frame)
    if source is None:
        return None
    key = channel_key(p, frame)
    when = frame if lens.animated else None  # 镜头整段固定时各帧共用同一张 ST-map；变焦时每帧一张
    base = p.dir / "_view" / "proxy" / f"{key if key is not None else frame}.{px}.through-{tag}{'' if when is None else f'.{when}'}"
    return picture_of(source, base, px, write=lambda src, part, px_, kind: _write_picture(src, part, px_, kind, warp=lambda pixels: _undistort(pixels, lens, when)))


def _undistort(pixels: np.ndarray, lens, frame: int | None) -> np.ndarray:
    """将 [h, w, C] 的显示图按 `lens` 去畸变，画布即其自身的框：每个输出像素通过镜头的畸变公式查询「该针孔位置
    在实拍画面上的落点」（data/lens_models.py lens_stmaps 的 undistort 映射，换算为该档位的尺寸），再用与
    「STMap」节点相同的采样核（apply_stmap，双三次）取样。落在画面之外的像素为黑色。"""
    from ..data.lens_models import apply_stmap, encode, pixel_centres

    h, w = pixels.shape[:2]
    raster_w, raster_h = lens.raster
    centres = pixel_centres(w, h) * (raster_w / w, raster_h / h)  # 该档位的像素中心，换算到相机画幅的像素坐标
    mapped = lens.map_px(centres, "distort", frame)  # 针孔位置 -> 实拍画面上的位置（画幅像素）
    st = encode(mapped.points, (raster_w, raster_h))  # 按画幅归一化：乘以该档位的尺寸即为该图上的位置
    out, valid = apply_stmap(np.ascontiguousarray(pixels, np.float32), st, (w, h), bicubic=True)
    return np.where(valid[..., None] > 0, out, 0.0)


def _generation(p: Packet) -> str:
    """该数据包的代次：manifest 中的 created（同一指纹重新计算后即为另一个 created，data/packet.py Packet.created），
    没有该字段的旧数据包使用 manifest 文件的修改时间。按指纹缓存的内容一律附带代次：若只按指纹缓存，相机数据包重新计算
    （fresh_dir 替换了文件夹）后，本进程会一直使用旧的镜头。"""
    from ..data.packet import MANIFEST

    if p.created:
        return p.created
    try:
        return str((p.dir / MANIFEST).stat().st_mtime_ns)
    except OSError:
        return ""


@lru_cache(maxsize=32)
def _camera_lens(camera_fp: str, at: str, generation: str = ""):
    """相机数据包 `camera_fp` 中 `at` 所指相机（为空时取包中唯一的相机）所带的镜头 -> (Lens, 摘要)；不带畸变时为 (None, "")。
    按数据包指纹及其代次（`generation`，_generation）缓存：数据包写好后不再改变，同一台相机的每一帧都会查询，
    无需每帧打开一次 USD；重新计算的数据包属于另一代次，不会取到旧结果。
    提供 `at` 时它必须是该数据包中的一台相机：地址中的任意字符串若不指向相机即立即拒绝（4xx），
    否则非相机的 prim 会一直传到 CameraSamples.from_prim 中引发 TypeError，导致 500。"""
    import hashlib
    import json

    from pxr import UsdGeom

    from ..data.camera import CameraSamples
    from ..data.lens_models import Lens
    from ..data.packet import packet_dir
    from ..data.scene import open_scene, the_camera
    from ..errors import Invalid
    from ..messages import Msg

    camera = Packet.load(packet_dir(camera_fp))
    stage = open_scene([camera])
    if at:
        prim = stage.GetPrimAtPath(at)
        if prim is None or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise Invalid(Msg("E-VIEW-NOCAMERAAT", at=at))
    else:
        prim = the_camera(stage, "场景")
    width, height = int(camera.meta.get("width") or 0), int(camera.meta.get("height") or 0)
    samples = CameraSamples.from_prim(prim, list(camera.meta.get("frames") or []), width=width, height=height)
    if not samples.lens:
        return None, ""
    meta = samples.lens_meta()
    lens = Lens.from_meta(meta, (width, height) if width and height else None)
    if lens is None:
        return None, ""
    return lens, hashlib.sha1(json.dumps(meta, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ 计算完成后生成（在同一次任务中）


def has_proxy(p: Packet) -> bool:
    """该数据包是否应有代理：仅限逐帧的二维像素数据（画面、遮罩、深度、法线、运动矢量等）。

    相机、点云、人物框、跟踪点、曲线、数值本身只是少量数字，按各自的路径发送；三维数据不使用代理。空数据包没有像素。"""
    return bool(channel_list(p)) and not p.meta.get("empty")


def frames_of(p: Packet) -> list[int]:
    return sorted(p.meta.get("frames") or ()) if p.type == "video" else sorted(image_files(p))


def build(p: Packet, threads: int = 0) -> None:
    """生成该数据包的全部代理（每帧 × 每个通道，另加每帧的显示图）。不报告进度，不发送任何事件。

    engine 每计算完一个节点，就对其输出的每个数据包调用一次（`lab2shot/farm/queue.py` 将该函数交给 Engine）。
    已生成的不再重复生成（write_once 按文件名识别），因此重复调用只是若干次 `stat`，不会重新生成。

    出错不影响本次解算：代理只涉及查看方式，计算结果本身已写好；本层出现问题时，由取帧路径在
    首次请求时再生成一次。"""
    if not has_proxy(p):
        return
    from ..engine.cook import FRAME_THREADS

    px, names, frames = tier(), channel_list(p), frames_of(p)
    if p.type == "video":
        # 视频先一次解码完成：其像素位于容器中，逐帧请求意味着将同一文件打开数百次
        # （`view/frames.py video_frames` 一次解码这些帧，已解码的跳过）。下面每个通道、每张图读取的
        # 都是解码出的 PNG，因此该步骤必须在启动线程之前执行。
        from .frames import video_frames

        video_frames(p, list(frames))
    jobs: list = [lambda f=f, n=n: channel_file(p, f, n, px) for f in frames for n in names]
    jobs += [lambda f=f: (video_picture_file if p.type == "video" else picture_file)(p, f, px) for f in frames]
    if not jobs:
        return
    with ThreadPoolExecutor(threads or FRAME_THREADS) as pool:
        for _ in pool.map(_quietly, jobs):
            pass


def _quietly(make) -> None:
    try:
        make()
    except Exception as exc:  # noqa: BLE001 — 见 build() 的说明：代理做不出来不算这次解算失败
        from .. import logs
        from ..messages import Msg

        logs.say(logs.get("view"), Msg("I-VIEW-PROXYLATER", reason=logs.error_text(exc)))
