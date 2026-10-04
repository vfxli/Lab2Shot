"""Payload conventions per data type, and helpers to read/write them. frames_for_worker converts a plate to
display-referred PNGs (what a worker's model wants, and — through the cache it makes — what the viewer shows of a
picture without an alpha too); the viewer's own per-frame lookups (falling back to converting just that one frame,
never a whole shot) are lab2shot/view/frames.py.

image.1-4  逐帧的二维数据，一至四条通道（二维数据只有一种类型，区别仅在通道数）。meta: frames,
        files {frame: path}, width, height, colorspace, alpha（四通道画面带 alpha，预乘，与 Nuke 一致；
        io/images.py），数值图另有 range（视图的显示范围）；files 可指向包外的文件，例如使用者自己的素材
boxes   boxes.json: {"people": [{"id", "boxes": {frame: [x1,y1,x2,y2]}, "prominence"}]}; meta: people (the ids), chosen
        (selected by 「选人」; otherwise everyone a detection found), frames, width, height
tracks2d tracks.npz: tracks [N,F,2] pixels (centres at +0.5), visible [N,F], query_frames [N], optional confidence
        [N,F] (0..1, the tracker's own); meta: names
curves  curves.npz: values [F,C]; meta: names [C], range
video   meta: path, count (frames; numbered by video to sequence), width, height
scene*  scene.usd (cm, Y-up, source frame numbers) + meta: frames
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from lab2shot_worker.files import write_exr

from ..io import images
from ..io.color import DATA, is_working, load_config, space, to_srgb8, to_working_picture, working_space
from ..errors import Invalid
from ..messages import Msg
from .packet import Packet, file_path, file_ref, packet_dir, produce
from .units import DEFAULT_FPS
from .windows import Window

SCENE_FILE = "scene.usd"


# ------------------------------------------------------------------ image


def image_packet(directory: Path, files: dict[int, Path | str], width: int, height: int, colorspace: str,
                 alpha: bool = False, window: Window | None = None) -> Packet:
    """`width`, `height`: the plate frame (display window). `window` (data/windows.py): the same size with the data
    window the files keep, when that reaches past the plate frame or stops short of it; None: the plate frame itself."""
    from .contracts import NOT_SAID

    frames = sorted(files)
    window = window or Window(width, height)
    if window.plate != (width, height):
        raise ValueError(f"window {window.plate} is not the picture's {width} x {height}")
    return Packet(
        directory,
        "image.4" if alpha else "image.3",
        {
            "frames": frames,
            "files": {str(f): file_ref(directory, files[f]) for f in frames},
            "colorspace": colorspace,
            "alpha": bool(alpha),
            **window.meta(),
            **NOT_SAID,  # what was photographed: not said here, filled in from the picture the node worked on (contracts.settle)
        },
    )


def video_packet(directory: Path, path: Path, count: int, fps: float, width: int, height: int, **meta) -> Packet:
    """A video the user picked (「读取视频」): its file and what it holds; its frames count from 0 (「视频转序列」
    numbers them). What was photographed comes from the node that read it, else it is not said.

    `fps` is this file's own recorded rate, a property of the container like `width`, not a frame rate the shot runs
    at. It is the only packet type that carries one, and only 「视频转序列」 reads it, to turn its 「区间（秒）」 into frame
    positions inside this same file. The sequence it makes carries frame numbers and nothing else."""
    from .contracts import NOT_SAID

    return Packet(directory, "video", {"path": file_ref(directory, Path(path)), "count": int(count), "frames": list(range(int(count))),
                                       "fps": float(fps), "width": int(width), "height": int(height), **NOT_SAID, **meta})


def window_of(p: Packet) -> Window:
    """A picture or data map packet's window (data/windows.py Window)."""
    return Window.of(p.meta)


def has_overscan(p: Packet) -> bool:
    """Whether an image or map packet keeps pixels past its plate frame."""
    return window_of(p).has_overscan


def has_alpha(p: Packet) -> bool:
    """Whether an image packet's pictures carry an alpha, as its meta states."""
    return bool(p.meta["alpha"])


def is_labels(p: Packet) -> bool:
    """该图存储的是类别编号（分割）而非连续数值，自身附带类别表。编号之间不得插值，
    写多层 EXR 时写为 Cryptomatte。"""
    return bool(p.meta.get("classes"))


def read_picture(p: Packet, path: Path, box: tuple | None = None) -> np.ndarray:
    """One picture of an image packet as a node that keeps alpha takes it: RGBA premultiplied when the packet has an
    alpha, else RGB. Every pixel the packet keeps (its data window, overscan included); `box` (x, y, w, h from the
    plate frame's corner) reads another part — `window_of(p).display_box` for the plate frame alone."""
    return images.read_picture(path, has_alpha(p), window_of(p).data if box is None else box)


def ingest_picture(directory: Path, files: dict[int, Path], width: int, height: int, colorspace: str, alpha: bool,
                   window: Window, each_done=None) -> Packet:
    """A reader's pictures into the working space (io/color.py module docstring): a file whose pixels already are
    it (PNG, JPG, SDR video frames) is referenced as it is; a scene-referred one (an EXR in ACEScg, a log plate) is
    converted once, here, into half-float EXR frames in `directory`. Either way the packet states the working space;
    nothing downstream carries or converts colour. `each_done`: the cook's per-frame runner (ctx.each_done: the frames
    converted side by side, with progress); None converts them one after another."""
    cfg = load_config()
    if is_working(colorspace, cfg):
        return image_packet(directory, files, width, height, working_space(cfg), alpha, window)
    writer = ExrWriter(directory, 4 if alpha else 3, half=True, colorspace=working_space(cfg), window=window,
                       **({"alpha": True} if alpha else {}))
    items = sorted(files.items())
    first: dict[Path, int] = {}  # a still (one file for every frame) is converted once, at the first frame showing it
    for frame, path in items:
        first.setdefault(path, frame)

    def convert(job):
        frame, path = job
        writer.add(frame, to_working_picture(images.read_picture(path, alpha, window.data), cfg, colorspace))

    jobs = [(frame, path) for path, frame in first.items()]
    list((each_done or (lambda jobs, work: map(work, jobs)))(jobs, convert))
    for frame, path in items:
        if first[path] != frame:
            writer.files[frame] = writer.files[first[path]]
    return writer.packet()


def image_files(p: Packet) -> dict[int, Path]:
    """The packet's files by frame, at their place (a packet holds its own frames; it may also read a plate without
    holding it, named relative to the packet's folder with `..`)."""
    return {int(f): file_path(p, v) for f, v in p.meta["files"].items()}


def file_at(p: Packet, frame: int) -> Path | None:
    """The file of an image or map packet at `frame` (a still: its one picture at any frame); None when it has no
    such frame. Looked up by the frame's key in the meta (image_packet writes `str(frame)`), not by building every
    frame's path: a node asks once per frame, and a shot of F frames would otherwise cost F x F paths."""
    ref = p.meta["files"].get(str(frame))
    if ref is not None:
        return file_path(p, ref)
    files = image_files(p)  # a key written some other way ("0001"), or a still: every file by its frame
    if frame in files:
        return files[frame]
    return next(iter(files.values()), None) if p.meta.get("still") else None


def frames_for_worker(p: Packet, emit=None) -> Packet:
    """The packet as sRGB RGB PNGs: what a worker's ML model wants as its input (and, through the cache it makes,
    what the viewer shows of a picture without an alpha. The name reflects the worker use, which is what every caller
    except the viewer needs: engine/external.py's job preparation).

    A worker receives RGB PNGs, 8 or 16 bit, and nothing else; this is the one place that holds it (worker_ready), so
    no adapter checks what its decoder (cv2, PIL, torchvision, an upstream loader) makes of other PNGs. Every picture
    packet is in the working space (io/color.py): an RGB PNG is used as it is, with no copy; anything else (an EXR, a
    grey or palette PNG referenced as it is by 「读取序列」) is written out once as 8-bit RGB PNGs, cached next to the
    source; no colour transform happens here. A picture's alpha is deliberately left out (the models take RGB): what
    they see is its premultiplied colour, the picture over black, as Nuke shows RGB.
    """
    cfg = load_config()
    files = image_files(p)
    if worker_ready(p, cfg, files):
        return p
    # two cooks (or viewers) of the same plate convert it once: data/packet.py produce holds the entry's lock
    return produce(f"{p.fingerprint}_display", lambda d: _convert_for_worker(p, d, cfg, files, emit))


def shown_as_is(p: Packet, cfg) -> bool:
    """Working-space PNGs without an alpha, their data window the plate frame: the viewer shows (and the browser
    decodes) them as they are, whatever their samples (a grey PNG too)."""
    return (is_working(p.meta["colorspace"], cfg) and all(f.suffix.lower() == ".png" for f in image_files(p).values())
            and not has_alpha(p) and window_of(p).same)


def worker_ready(p: Packet, cfg, files: dict[int, Path]) -> bool:
    """Shown as it is and every file an 8- or 16-bit RGB PNG (a header read per file; a still's one file once): a
    worker reads the packet's own files. `files`: image_files(p)."""
    return shown_as_is(p, cfg) and all(images.is_rgb_png(f) for f in set(files.values()))


def _convert_for_worker(p: Packet, d: Path, cfg, files: dict[int, Path], emit) -> Packet:
    # every pixel the packet keeps goes to the worker: an undistorted plate's canvas is what a pinhole solve needs
    # and a PNG has no windows, so the canvas is the picture here
    window = window_of(p)
    out: dict[int, Path] = {}
    seen: dict[Path, Path] = {}  # a still (e.g. an HDRI) lists one file for every frame: convert it once
    for i, (frame, path) in enumerate(sorted(files.items())):
        if path in seen:
            out[frame] = seen[path]
            continue
        rgb8 = to_srgb8(images.read_rgb(path, window.data), cfg, p.meta["colorspace"])
        out[frame] = seen[path] = d / f"frame.{frame}.png"
        images.write_png(out[frame], rgb8)
        if emit:
            emit(i + 1, len(files))
    view = image_packet(d, out, *window.canvas, working_space(cfg))
    if p.meta.get("still"):
        view.meta["still"] = True
    return view.commit("data.frames_for_worker")


# ------------------------------------------------------------------ data maps (depth / position / mask)


def map_packet(directory: Path, channels: int, files: dict[int, Path], width: int, height: int,
               value_range: tuple[float, float], window: Window | None = None) -> Packet:
    """逐帧的数值图：存储原始数值，不经过色彩管理。`channels` 为通道数（一至四条），
    `value_range` 用作视图的显示范围。"""
    p = image_packet(directory, files, width, height, space(DATA), window=window)
    p.type = f"image.{channels}"
    p.meta["alpha"] = False
    p.meta["values"] = True  # 数值而非颜色：视图按范围映射显示，不经过 OCIO 显示变换
    p.meta["range"] = [float(value_range[0]), float(value_range[1])]
    # 真实的最小值与最大值（供视图的「贴合」使用）：此处先按显示范围填写，实测值由 ExrWriter 覆盖
    p.meta["full_range"] = [float(value_range[0]), float(value_range[1])]
    return p


UNIT = (0.0, 1.0)  # 遮罩、置信度、ST-map、UV、粗糙度、金属度：0 至 1
SIGNED = (-1.0, 1.0)  # 法线等带符号的数据：-1 至 1

VALIDITY = "valid"  # 附加通道：该像素是否有值（0 = 无值）
CHANNEL_LETTERS = "RGBA"  # 一至四条通道依次命名为 R G B A，不另设命名


def channel_names(channels: int, validity: bool = False) -> tuple[str, ...]:
    """Lab2Shot 自行写出的 EXR 的通道名：取 R G B A 的前若干条，带有效通道时再加一条 valid。

    此处不区分「深度」「遮罩」「ST-map」：二维数据只有一种类型，区别仅在通道数。
    交付给 Nuke 时按含义命名的通道（depth.Z、forward.u、Cryptomatte）定义在 data/layers.py，
    属于外部软件的约定，不是本处的类型。"""
    if not 1 <= channels <= 4:
        raise ValueError(f"{channels} channels: a float image has one to four")
    names = tuple(CHANNEL_LETTERS[:channels])
    return (*names, VALIDITY) if validity else names


def channel_list(p: Packet) -> tuple[str, ...]:
    """该包的通道名列表。此为核心层的声明（各类型如何传输由引擎负责），
    视图的「按通道取」（server/packets.py frame_channel）和数据包说明（/api/packet/{fp} 的 channels）
    均读取此处，页面不另行维护。

    image.1-4 为 R G B A 的前若干条，文件中带 valid 通道时再加一条（channel_names，与写 EXR 时
    使用同一套名称）；视频为解码得到的 R G B 三条。非二维像素数据的包（相机、点云、数值、人物框）
    没有通道，返回空元组；这些数据按各自的方式传输。"""
    from .types import channels_of

    if p.type == "video":
        return channel_names(3)
    count = channels_of(p.type)
    return channel_names(count, has_validity(p)) if count else ()


def has_validity(p: Packet) -> bool:
    """该包的每帧 EXR 是否带有 valid 通道（标记哪些像素有值）。"""
    return bool(p.meta.get("validity"))


def is_data(p: Packet) -> bool:
    """判断二维数据是数值图（深度、遮罩、法线、ST-map 等）还是画面（照片、渲染、HDRI）。

    判断仅在此处进行，画面即 `not is_data(p)`。三通道的 image.3 既可能是照片也可能是法线图，
    通道数无法区分，因此依据包上的标记 values：计算完成的包由引擎按输出端口的声明写上（Port.data，
    engine/graph.py output_data；engine/cook.py _settle_outputs 同时核对写出的是不是这一种），界面上
    「输出色彩空间」等条件读的也是这份声明（applies.py WiredPicture），两处不会各判各的。
    计算中的包：map_packet 先写上 values，其余依据色彩空间（数值图使用配置中的 DATA 角色）；
    边算边传的临时包两者均未写入，按通道数推断：一至两条为数值，三至四条为画面；推断错误仅影响计算中各帧的预览。

    非二维像素数据的包（相机、点云、数值等）均返回 False，且必须最先排除：
    否则相机（0 条通道）会落入最后的 `<= 2` 判断而被视为数值图，而逐帧数值包
    meta 中的 values 是数值列表，bool() 结果同样为真。"""
    from .types import channels_of

    if not channels_of(p.type):
        return False
    if "values" in p.meta:
        return bool(p.meta["values"])
    if (cs := p.meta.get("colorspace")) is not None:
        return cs == space(DATA)
    return channels_of(p.type) <= 2  # 按通道数：一至两条为数值，三至四条为画面


def read_map(path: Path, box: tuple | None = None) -> tuple[np.ndarray, np.ndarray]:
    """读取 Lab2Shot 自行写出的一帧二维数据：(值 [H,W,C]，有效 [H,W] 0..1)。

    EXR 文件自身记录通道，按通道名读取，不依赖类型或查表；无 valid 通道时视为处处有效。
    `box`：读取文件的区域（包的数据窗口，data/windows.py），None 表示画幅本身。"""
    # 第二个位置参数传错（例如将类型名作为 box 传入）时在此处立即报错，
    # 避免到 read_named 内部 int("image.1"[0]) 处才产生难以理解的异常。
    if box is not None and not (isinstance(box, tuple) and len(box) == 4):
        # 属于开发者错误（调用方传参错误），不是面向使用者的消息，因此不进入消息目录，与其他断言一样使用英文
        raise TypeError(f"read_map's second argument is a data window (x, y, w, h), not {box!r}: for a packet's own window use data/maps.py map_at")
    # 一次打开、一次读出：值通道（R G B A 的前若干条）在前，有 valid 通道时放在最后一并读出
    block, chosen = images.read_stack(path, lambda names: [*channel_names(len(names) - (VALIDITY in names)),
                                                            *([VALIDITY] if VALIDITY in names else [])], box)
    if chosen[-1] != VALIDITY:
        return block, np.ones(block.shape[:2], np.float32)
    return np.ascontiguousarray(block[..., :-1]), block[..., -1].astype(np.float32)


RANGE_SAMPLE = 1_000_000  # the most values of a frame a map's display range is taken over (a fixed stride: reproducible)

# The compression of the EXR frames the engine keeps between nodes (ExrWriter): ZIP, lossless and readable everywhere.
# Measured with OpenImageIO 3.1 (the core's writer) on 1080p frames, ms per frame, one thread / eight frames at once:
#                   mask (half, 1 ch)          picture (half, 3 ch)       depth (float, 1 ch)
#                   write     read   MB        write     read   MB        write     read   MB
#   zip (level 4)  4.2/1.9  5.0/1.0  0.01     9.8/5.8 12.4/3.3  2.95     7.2/4.2  2.1/1.5  7.85
#   zip level 1    4.1/1.8  6.2/1.0  0.01     9.0/4.6 13.9/3.2  3.18     8.2/3.4  2.9/1.9  7.86   <- used
#   zips          19.4/2.9 19.1/2.4  0.06    27.3/6.3 28.2/3.8  3.78    27.6/5.1 19.0/1.9  8.30
#   rle           19.2/3.6 19.1/3.4  0.09    28.2/5.0 30.6/3.7  6.49    28.2/5.1 18.9/3.2  8.31
#   none          25.3/4.8 19.0/3.6  4.16    34.0/6.5 22.3/4.6 12.46    27.7/4.8 18.9/3.1  8.31
#   piz            4.4/3.3  1.9/1.1  0.03     9.4/5.5  5.6/3.4  1.96    20.6/11.4 3.5/2.2 6.95
# "Lighter" formats are not faster here: OpenEXR splits a 16-scanline ZIP file into blocks it compresses and
# decompresses on its own threads, while none / rle / zips store one scanline per block and go through it line by
# line, and write 4-400 times the bytes. PIZ reads pictures faster but writes float data at half the speed. DWAA / DWAB
# and PXR24 lose precision (on float data), which a data map may not. So the cache keeps ZIP at its fastest level: the
# same format every reader already takes (the level is not part of the file), a few percent larger, faster to write.
CACHE_COMPRESSION = "zip"
CACHE_LEVEL = 1


class ExrWriter:
    """向输出文件夹逐帧写入 EXR，最后生成该序列的包。

    `channels`：通道数（一至四条），决定包类型为 image.1 / image.2 / image.3 / image.4。
    给出 `colorspace` 时为画面（使用该色彩空间），未给出时为数值（配置中的 data 角色）。
    `half` 未给出时的取值见下方说明。
    `validity`：每帧附加一条 valid 通道，标记该像素是否有值，由图像自身表明数据范围，
    不另附「有效像素」遮罩。
    数值图未给定 `value_range` 时，视图的显示范围取整段有效像素的 1%–99% 分位。

    `add` 可以在多个线程中同时调用（逐帧并行的节点在各自的线程里写出本帧，engine/cook.py CookContext.each_done）：
    各帧的文件互不相干，共用的记录（文件表、取值范围）在锁内合并；合并只取最小值与最大值，与各帧写出的先后无关，
    包的内容与逐帧依次写出时完全相同（文件表在包中按帧号排列，image_packet）。
    """

    def __init__(self, directory: Path, channels: int, *, validity: bool = False, half: bool | None = None,
                 colorspace: str | None = None, value_range: tuple[float, float] | None = None,
                 window: Window | None = None, **meta):
        self.count = channels
        self.validity = validity
        self.channels = channel_names(channels, validity)
        self.picture = colorspace is not None
        self.directory = directory
        # the window every frame is written with (data/windows.py): the plate frame it belongs to and the pixels it
        # keeps around it. None: the frames' own size, nothing past it
        self.window = window
        self.colorspace = colorspace or space(DATA)  # none given: values, the config's data role
        self.fixed = value_range
        # 精度选择：未给出时使用全精度，不根据值域推断。若按「值域在 -1..1 内即用半精度」判断，ST-map
        # 也会被判为半精度；其值虽在 0..1 内，却需乘以画面宽度作为坐标使用，而半精度在 0.5–1 之间的步长为 2^-11，
        # 在 2K 宽度上误差约一个像素。数值精度是否足够只有写入节点能判断，因此由节点显式指定 half=True（遮罩、法线等）。
        self.half = bool(half)
        # 始终测量真实的最小值与最大值（不限于未给定固定值域的情况），声明 0..1 的输出同样测量。
        # 黑白点对应两项不同的信息：「显示范围」默认 0..1 不裁切（fixed），
        # 「贴合」按数据真实的最小值与最大值拉伸（full_range）。若给定 fixed 即不测量，
        # 声明 0..1 的输出将无法使用「贴合」。
        # 1%–99% 分位（lo/hi，显示范围）只在未给定 fixed 时计算：给定时显示范围就是 fixed，分位不会被用到，
        # 而它是这里最贵的一步（两次部分排序），遮罩这类固定值域的输出每帧省下它
        self.track = not self.picture
        self.meta = meta
        self.files: dict[int, Path] = {}
        self.lo, self.hi = np.inf, -np.inf
        # 真实的最小值与最大值（不裁切）：视图的「贴合」据此拉伸黑点与白点。上方的 lo/hi 为 1%–99% 分位，
        # 本身已经裁切，不能作为数据的实际范围
        self.least, self.most = np.inf, -np.inf
        self.size = (0, 0)
        self._top = -np.inf  # the highest frame written so far (its size is the packet's)
        self._lock = threading.Lock()

    def add(self, frame: int, data: np.ndarray, valid: np.ndarray | None = None) -> None:
        """`valid` (bool, or 0..1): pixels outside it are written as 0, marked invalid in the map's validity channel
        and left out of the range. `data` covers the writer's window (its data window's pixels, a picture's overscan
        passed on), else its own size is the picture. Safe to call from several threads at once (see the class)."""
        data = np.ascontiguousarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data[..., None]
        mask = None if valid is None else np.asarray(valid, np.float32).reshape(data.shape[:2])
        if mask is not None:
            data = np.where(mask[..., None] > 0, data, 0.0).astype(np.float32)
        if self.validity:
            data = np.concatenate([data, (np.ones(data.shape[:2], np.float32) if mask is None else mask)[..., None]], axis=-1)
        if data.shape[-1] != len(self.channels):
            raise Invalid(Msg("E-EXR-CHANNELS", count=len(self.channels), channels=" ".join(self.channels), given=data.shape[-1]))
        if self.window is not None and (data.shape[1], data.shape[0]) != self.window.canvas:
            raise Invalid(Msg("E-EXR-SIZE", width=data.shape[1], height=data.shape[0],
                              want_width=self.window.canvas[0], want_height=self.window.canvas[1]))
        path = self.directory / f"frame.{frame}.exr"
        write_exr(path, data, self.channels, half=self.half, compression=CACHE_COMPRESSION, level=CACHE_LEVEL,
                  windows=None if self.window is None else self.window.exr_windows)
        found = self._range(data, mask) if self.track else None
        with self._lock:
            self.files[frame] = path
            if frame >= self._top:  # the last frame's size, as when the frames are written one after another
                self._top = frame
                self.size = self.window.plate if self.window is not None else (data.shape[1], data.shape[0])
            if found is not None:
                lo, hi, least, most = found
                self.lo, self.hi = min(self.lo, lo), max(self.hi, hi)
                self.least, self.most = min(self.least, least), max(self.most, most)

    def _range(self, data: np.ndarray, mask: np.ndarray | None) -> tuple[float, float, float, float] | None:
        """One frame's (1% quantile, 99% quantile, least, most) over its finite values (inside `mask`), taken over
        every n-th of them past RANGE_SAMPLE; None when it has none. With a fixed range the quantiles are not taken
        (they would not be used): (inf, -inf, least, most)."""
        values = data[..., : self.count]
        values = values[mask > 0] if mask is not None else values.reshape(-1, values.shape[-1])
        if self.fixed is not None and values.size and np.isfinite(values.min()) and np.isfinite(values.max()):
            # every value finite (a NaN or an infinity would show in the least or the most): the finite values
            # below are all of them, in the same order, so this skips only the filtering copy
            values = values.reshape(-1)
        else:
            values = values[np.isfinite(values)]
        if values.size > RANGE_SAMPLE:  # a hint for the viewer's range: every n-th value, the same each time
            values = values[:: -(-values.size // RANGE_SAMPLE)]
        if not values.size:
            return None
        least, most = float(values.min()), float(values.max())
        if self.fixed is not None:
            return np.inf, -np.inf, least, most
        return float(np.percentile(values, 1)), float(np.percentile(values, 99)), least, most

    def packet(self) -> Packet:
        if self.picture:
            p = image_packet(self.directory, self.files, *self.size, self.colorspace,
                             self.count == 4, self.window)
        else:
            rng = self.fixed or ((self.lo, self.hi) if np.isfinite(self.lo) else (0.0, 1.0))
            p = map_packet(self.directory, self.count, self.files, *self.size, rng, self.window)
            if np.isfinite(self.least):  # 不裁切的范围，供视图的「贴合」使用
                p.meta["full_range"] = [self.least, self.most]
        p.meta["validity"] = self.validity  # 文件是否带 valid 通道由包自身声明，下游写入的节点据此沿用
        p.meta.update(self.meta)
        return p


def still_packet(directory: Path, path: Path, frames: list[int], width: int, height: int, *,
                 channels: int = 3, colorspace: str | None = None, value_range: tuple[float, float] | None = None,
                 window: Window | None = None) -> Packet:
    """整段仅有一张图（HDRI、ST-map）：每一帧指向同一文件，并标记 still。
    给出 `colorspace` 时为画面，未给出时为数值。"""
    files = {f: path for f in frames}
    if colorspace is not None:
        p = image_packet(directory, files, width, height, colorspace, channels == 4, window)
    else:
        p = map_packet(directory, channels, files, width, height, value_range or UNIT, window)
    p.meta["still"] = True
    return p


def display_rgb(image: Packet, frame: int, width: int | None = None, height: int | None = None) -> np.ndarray:
    """One frame as sRGB RGB 0..1: the working space clamped (only that frame is converted), optionally resampled
    (nearest)."""
    src = file_at(image, frame)
    cached = packet_dir(f"{image.fingerprint}_display")
    cfg = load_config()
    if is_working(image.meta["colorspace"], cfg) and src.suffix.lower() == ".png":
        rgb = images.read_rgb(src)
    elif window_of(image).same and Packet.exists(cached):
        # `_display` 按数据窗口转换整张画布（_convert_for_worker）：带扩边的包其尺寸大于画幅，下方按 width x height
        # 重采样会将画布压缩进画幅，导致取色整体偏移一个扩边宽度；因此与 view/frames.py display_frame 一样，仅在两者尺寸相同时使用
        rgb = images.read_rgb(file_at(Packet.load(cached), frame))
    else:
        rgb = to_srgb8(images.read_rgb(src), cfg, image.meta["colorspace"]).astype(np.float32) / 255.0
    if width and height and rgb.shape[:2] != (height, width):
        ys = (np.arange(height) * rgb.shape[0] / height).astype(int)
        xs = (np.arange(width) * rgb.shape[1] / width).astype(int)
        rgb = rgb[ys][:, xs]
    return rgb


# ------------------------------------------------------------------ boxes


def write_boxes(directory: Path, people: list[dict], width: int, height: int, frames: list[int],
                chosen: bool = False) -> Packet:
    """chosen: the people were selected (「选人」), rather than everyone a detection found."""
    from .contracts import NOT_SAID

    (directory / "boxes.json").write_text(json.dumps({"people": people}, ensure_ascii=False), encoding="utf-8")
    return Packet(directory, "boxes", {"people": [p["id"] for p in people], "chosen": chosen, "width": width,
                                       "height": height, "frames": frames, **NOT_SAID})


def read_boxes(p: Packet) -> list[dict]:
    return json.loads(p.path("boxes.json").read_text(encoding="utf-8"))["people"]


# ------------------------------------------------------------------ 2D tracks and curves


def tracks_packet(directory: Path, frames: list[int], width: int, height: int, tracks: np.ndarray,
                  visible: np.ndarray, query_frames: np.ndarray, names: list[str] | None = None,
                  homography: np.ndarray | None = None, confidence: np.ndarray | None = None, **meta) -> Packet:
    """Points followed through the shot: tracks [N,F,2] in pixels at width x height, visible [N,F] and, from a tracker
    that scores its points, how sure it is of each on each frame (confidence [N,F], 0..1: visible is a yes / no drawn
    from it, the score itself is kept). Points on one plane (a planar track: its corners) may carry the plane's
    homography [F,3,3], from their query frame's pixels to each frame's (apply_homography)."""
    tracks = np.asarray(tracks, np.float32)
    extra = {} if homography is None else {"homography": np.asarray(homography, np.float64)}
    if confidence is not None:
        extra["confidence"] = np.clip(np.nan_to_num(np.asarray(confidence, np.float32)), 0.0, 1.0)
    np.savez_compressed(directory / "tracks.npz", tracks=tracks, visible=np.asarray(visible, bool),
                        query_frames=np.asarray(query_frames, np.int64), **extra)
    from .contracts import NOT_SAID

    names = names or [f"p{i:04d}" for i in range(len(tracks))]
    return Packet(directory, "tracks2d", {"frames": [int(f) for f in frames], "width": width, "height": height,
                                          "count": int(len(tracks)), "names": names, **NOT_SAID, **meta})


def read_tracks(p: Packet) -> dict[str, np.ndarray]:
    """tracks, visible, query_frames; confidence from a tracker that scores its points; homography when the points
    are a planar track's."""
    d = np.load(p.path("tracks.npz"))
    return {k: d[k] for k in ("tracks", "visible", "query_frames", "confidence", "homography") if k in d}


def apply_homography(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Pixels xy [...,2] through the homography h [3,3] (pixel centres at +0.5 on both sides)."""
    xy = np.asarray(xy, np.float64)
    p = xy @ h[:2, :2].T + h[:2, 2]
    w = xy @ h[2, :2] + h[2, 2]
    return p / w[..., None]


def curves_packet(directory: Path, frames: list[int], names: list[str], values: np.ndarray, **meta) -> Packet:
    """Named per-frame curves: values [F, C]."""
    values = np.asarray(values, np.float32)
    assert values.shape == (len(frames), len(names)), values.shape
    np.savez_compressed(directory / "curves.npz", values=values)
    finite = values[np.isfinite(values)]
    rng = [float(finite.min()), float(finite.max())] if finite.size else [0.0, 1.0]
    return Packet(directory, "curves", {"frames": [int(f) for f in frames], "names": list(names), "range": rng, **meta})


def read_curves(p: Packet) -> np.ndarray:
    return np.load(p.path("curves.npz"))["values"]


# ------------------------------------------------------------------ scene


def scene_packet(directory: Path, frames: list[int], type_: str = "scene", **meta) -> Packet:
    """A 3D packet over the scene file already written into `directory`: what it holds is read once here (`contents`
    per kind, `top` the names under /shot), so everything that says what a result is reads meta only (data/summary.py)."""
    return Packet(directory, type_, {"frames": [int(f) for f in frames],
                                     **_scene_contents(directory), **meta})


def _scene_contents(directory: Path) -> dict:
    """{contents: {kind: how many}, top: the names under /shot} of the scene file in `directory` ({} when it is not
    written yet: the packet's maker passes them itself)."""
    from pxr import Usd

    from ..io import usd

    path = directory / SCENE_FILE
    if not path.is_file():
        return {}
    from .scene import prims_by_kind

    from .subsets import stage_subsets

    stage = Usd.Stage.Open(str(path))
    found = prims_by_kind(stage)
    shot = stage.GetPrimAtPath(usd.ROOT_PATH)
    return {"contents": {k: len(prims) for k, prims in found.items() if prims},
            "top": [c.GetName() for c in shot.GetChildren()] if shot else [],
            # 网格的分区（USD 的 GeomSubset）：名称 -> 面数。与曲线条数相同，在生成包时统计一次，
            # 此后摘要、端口提示和「按分区取出」的选项均只读 meta（data/summary.py 不打开文件）
            **({"subsets": parts} if (parts := stage_subsets(stage)) else {}),
            # 三维曲线的总条数与总点数：摘要据此描述，显示预算按总点数计算，
            # 因此在生成包时统计一次，此后需要其规模的地方均只读 meta（data/summary.py 不打开文件）
            **(dict(zip(("strands", "curve_points"), curve_totals(found["curves"]))) if found.get("curves") else {}),
            # 点和曲线上附带的属性：模型推理结果中点上的附加信息需要保留，属性名与属性类型须在节点的
            # 信息面板中可见。与分区、曲线条数相同，在生成包时统计一次，此后摘要、端口提示、「AttribDelete」的选项
            # 均只读 meta（data/summary.py 不打开文件）
            **({"attributes": attrs} if (attrs := point_attributes(found)) else {})}


# USD 类型名 -> 面向美术人员的类型（id；名称 attr.type.<id>，data/attributes.py attribute_said）
ATTR_SAID = {"float": "float", "double": "float", "half": "float", "int": "int", "int64": "int", "uint": "int",
             "bool": "toggle", "string": "string", "token": "string", "float2": "vec2", "float3": "vec3", "double3": "vec3",
             "color3f": "color", "normal3f": "normal", "point3f": "position", "vector3f": "vector", "texCoord2f": "uv",
             "quatf": "quaternion", "matrix4d": "matrix"}


def point_attributes(found: dict) -> list[dict]:
    """点云和三维曲线上附带的属性：[{name, type, per}]，`per` 为 point（逐点）或 curve（逐条）。

    模型输出的置信度、法线、类别、点编号等均经由此处列出，否则这些推理结果将无法呈现。
    坐标本身不计为属性。"""
    from pxr import UsdGeom

    vertex = (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying)  # 线性曲线的 varying 同样按逐点处理
    seen: dict[tuple[str, str, str], dict] = {}
    for kind in ("points", "curves"):
        for prim in found.get(kind) or ():
            named: list[tuple[str, object, str]] = []
            if prim.IsA(UsdGeom.Points):
                cloud = UsdGeom.Points(prim)
                named += [("ids", cloud.GetIdsAttr(), UsdGeom.Tokens.vertex),
                          ("velocities", cloud.GetVelocitiesAttr(), UsdGeom.Tokens.vertex)]
            elif prim.IsA(UsdGeom.BasisCurves):
                curves = UsdGeom.BasisCurves(prim)
                named += [("widths", curves.GetWidthsAttr(), curves.GetWidthsInterpolation()),
                          ("normals", curves.GetNormalsAttr(), curves.GetNormalsInterpolation())]
            named += [(pv.GetPrimvarName(), pv.GetAttr(), pv.GetInterpolation()) for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars()]
            for name, attr, interp in named:
                if not attr or not attr.HasAuthoredValue() or interp not in (*vertex, UsdGeom.Tokens.uniform):
                    continue
                said = str(attr.GetTypeName().scalarType or attr.GetTypeName())
                per = "point" if interp in vertex else "curve"  # attr.per.<per>
                seen.setdefault((name, said, per), {"name": name, "type": ATTR_SAID.get(said, said), "per": per})
    return sorted(seen.values(), key=lambda a: a["name"])


def curve_totals(prims) -> tuple[int, int]:
    """一组三维曲线（UsdGeom.BasisCurves prim）在第一个采样上的总条数与总点数。"""
    from pxr import Usd, UsdGeom

    strands = points = 0
    for prim in prims:
        attr = UsdGeom.BasisCurves(prim).GetCurveVertexCountsAttr()
        times = attr.GetTimeSamples()
        counts = attr.Get(Usd.TimeCode(times[0]) if times else Usd.TimeCode.Default()) or []
        strands += len(counts)
        points += int(sum(counts))
    return strands, points


def partial_frames(directory: Path) -> dict[int, str]:
    """The frames a node is writing into `directory` right now, by frame number -> the file's name inside it: the same
    naming ExrWriter uses above, read back here so nothing outside this layer depends on it (边算边传: the server layer
    never globs "frame.*.exr" itself). The folder has no manifest yet, so the file names are the only source."""
    out = {}
    for f in sorted(directory.glob("frame.*.exr")) if directory.is_dir() else ():
        # 只接受恰好为 `frame.<帧号>.exr` 的文件：write_exr（worker_sdk/lab2shot_shared/exr.py）先写 `frame.<帧号>.part.exr`
        # 再重命名，该临时名同样匹配此 glob；若只按第二段取帧号，会把正在写入的帧报告为已完成，
        # 导致浏览器读到不完整的文件
        parts = f.name.split(".")
        if len(parts) == 3 and f.is_file() and parts[1].lstrip("-").isdigit():
            out[int(parts[1])] = f.name
    return out


DEPTH_GRID = "lab2shot:depth_grid"  # a cloud's customData: the depth map, camera, matte and confidence it was made from


def points_packet(directory: Path, frames: list[int], name: str, points: list, colors: list | None = None, *,
                  scale: str, primvars: dict | None = None, width_cm: float = 0.2, info: dict | None = None,
                  depth_grid: dict | None = None, **meta) -> Packet:
    """/shot/<name>: per-frame point clouds (cm, USD space); `scale` "metric" (real distances) or "relative" (right up
    to one factor, like a solve without scale). `depth_grid`: what a cloud made from a depth map was made from (the
    packets' fingerprints, the step, the space, the confidence threshold), kept on the prim (DEPTH_GRID) wherever the
    prim goes (packed, moved): the 3D viewer sends such a cloud as its depth map and rebuilds the points itself
    (server/view_data.py depth_grid), after checking every frame gives exactly these points."""
    from ..io import usd

    stage = usd.create_stage(frames, info)
    cloud = usd.write_points(stage, f"{usd.ROOT_PATH}/{name}", frames, points, colors, primvars, width_cm=width_cm)
    if depth_grid:
        cloud.GetPrim().SetCustomDataByKey(DEPTH_GRID, {k: v for k, v in depth_grid.items() if v is not None})
    usd.save_stage(stage, directory / SCENE_FILE)
    return scene_packet(directory, frames, "scene.points", scale=scale, **meta)


def scene_curves_packet(directory: Path, frames: list[int], name: str, vertex_counts: list, points: list,
                  colors: list | None = None, widths: list | None = None, normals: list | None = None, *,
                  primvars: dict | None = None, width_cm: float = 0.2, info: dict | None = None, **meta) -> Packet:
    """/shot/<name>: per-frame 三维曲线 (cm, USD space). `vertex_counts` per frame [C] how many points each curve has,
    `points` per frame [N,3] their points one curve after another; `widths` one per point (hair is thick at the root
    and thin at the tip) or, left out, the one `width_cm` throughout."""
    from ..io import usd

    stage = usd.create_stage(frames, info)
    usd.write_curves(stage, f"{usd.ROOT_PATH}/{name}", frames, vertex_counts, points, widths, colors, normals,
                     primvars, width_cm=width_cm)
    usd.save_stage(stage, directory / SCENE_FILE)
    return scene_packet(directory, frames, "scene.curves", **meta)


def tracked_points_packet(directory: Path, frames: list[int], name: str, xyz: np.ndarray, visible: np.ndarray, *,
                          scale: str, colors: np.ndarray | None = None, width_cm: float = 1.0, info: dict | None = None,
                          confidence: np.ndarray | None = None, **meta) -> Packet:
    """/shot/<name>: points followed through the shot (3D tracks, locators): xyz [N,F,3] (cm, USD space; nan where not
    known), visible [N,F]. Every point is there on every frame with its id (0..N-1, Houdini's `id`), a `visible` primvar
    (1 seen, 0 hidden or not known: its position is the tracker's guess, or between the frames it is known on, held
    before the first and after the last) and its velocity (cm per second, Houdini's `v`); `colors` [N,3] (0..1) its
    colour, e.g. from the plate where it starts; `confidence` [N,F] (0..1) how sure its tracker was of it on each frame
    (a `confidence` primvar: a point attribute in Houdini). A point known on no frame sits at the origin, hidden."""
    xyz = np.array(xyz, np.float64)
    visible = np.asarray(visible, bool) & np.isfinite(xyz).all(-1)
    n, f = visible.shape
    known = np.isfinite(xyz).all(-1)
    for i in np.flatnonzero(~known.all(1)):
        have = np.flatnonzero(known[i])
        for a in range(3):
            xyz[i, :, a] = np.interp(np.arange(f), have, xyz[i, have, a]) if len(have) else 0.0
    t = np.asarray(frames, np.float64) / DEFAULT_FPS  # USD velocities are per second: the stage's own time base (io/usd.py)
    velocity = np.gradient(xyz, t, axis=1) if f > 1 else np.zeros_like(xyz)
    ids = np.arange(n, dtype=np.int64)
    cols = None if colors is None else [np.asarray(colors, np.float32)] * f
    from ..io import usd

    stage = usd.create_stage(frames, info)
    usd.write_points(stage, f"{usd.ROOT_PATH}/{name}", frames, list(xyz.transpose(1, 0, 2)), cols,
                     {"visible": list(visible.T.astype(np.int32)),
                      **({} if confidence is None else {"confidence": list(np.asarray(confidence, np.float32).T)})},
                     width_cm=width_cm, ids=[ids] * f,
                     velocities=list(velocity.transpose(1, 0, 2)))
    usd.save_stage(stage, directory / SCENE_FILE)
    return scene_packet(directory, frames, "scene.points", scale=scale, **meta)


def start_colours(image: Packet, tracks: dict[str, np.ndarray], frames: list[int]) -> np.ndarray:
    """Each tracked point's colour [N,3] (display, 0..1): the plate's under it on the frame it starts at (its query
    frame; the first frame when that is not in `frames` or the plate has no picture there: grey)."""
    w, h = image.meta["width"], image.meta["height"]
    colours = np.full((len(tracks["tracks"]), 3), 0.7)
    starts = np.asarray(tracks["query_frames"])
    at = np.where(np.isin(starts, frames), starts, frames[0])
    for f in np.unique(at).tolist():
        ids = np.flatnonzero(at == f)
        if file_at(image, f) is None:
            continue
        xy = tracks["tracks"][ids, frames.index(f)]
        rgb = display_rgb(image, f, w, h)
        colours[ids] = rgb[np.clip(np.floor(xy[:, 1]).astype(int), 0, h - 1), np.clip(np.floor(xy[:, 0]).astype(int), 0, w - 1)]
    return colours


