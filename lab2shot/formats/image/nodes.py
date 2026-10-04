"""图像文件的输出设置节点（接口见 nodes/output.py）：「序列图输出设置」（序列，每帧一个文件）与
「多层 EXR 输出设置」（多个结果作为每帧一个 EXR 的各个图层）。

格式参数与 Nuke 一致：EXR 的位深与压缩，PNG 的位深 / 压缩强度 / alpha，JPG 的质量。数据图（深度、法线、运动矢量、
位置、ST-map、置信度、Cryptomatte）必须保持无损和全精度：「序列图输出设置」无论格式设置如何都原样复制其文件
（格式设置只作用于画面，并在接入数据图时置灰说明）。「多层 EXR 输出设置」无论自身「位深」如何都将数据层写为
32 位浮点无损（ExrOutput.write）；其「压缩」是作用于整个文件的实际选项，因此接入数据层时拒绝 DWAA / DWAB / PXR24
（ExrOutput.param_refuses）。
"""

from __future__ import annotations

from typing import Any, Literal

from ...nodes.port import EITHER
from ...availability import All, AnyOf
from ...data.types import channels_of
from ...errors import Invalid
from ... import i18n
from ...messages import Msg
from ...nodes.base import NodeParams, P, Port, colorspace_param
from ...nodes.output import Format, OutputSettings, name_param
from ...nodes.applies import Param, Wired, WiredPicture

# 「序列图输出设置」每个通道一个端口：名称采用 CG 通用写法（与 Nuke 一致），接入哪个端口即写为哪个通道
CHANNEL_PORTS = ("R", "G", "B", "A")
# 由单个通道组成的画面：只有这两种通道组合可写为 PNG / JPG（其他组合只能写 EXR）
PICTURE_LETTERS = (["R", "G", "B"], ["R", "G", "B", "A"])


def writes_a_picture():
    """「序列图输出设置」本次写出的是否为画面（PNG / JPG 可容纳的 RGB / RGBA）：
    「图像」端口接入了画面（WiredPicture：来源端口声明它不是数值图，engine/graph.py output_data），
    或 R G B 三个端口均已单独接入。

    其他情况（数据图原样复制、只接入 A 等零散通道、未接入任何内容）只能写为 EXR，
    因此「格式」和「色彩空间」在这些情况下置灰并说明原因。"""
    return AnyOf(WiredPicture("image"),
                 All(Wired("R"), Wired("G"), Wired("B")))

# ------------------------------------------------------------------ EXR / PNG / JPG 共用的格式参数

# 采用 Nuke 的名称；lab2shot_shared/exr.py EXR_COMPRESSIONS 中有各项对应的 OpenEXR/OpenImageIO 名称
# 作用于文件所有通道（包括数据层）的压缩方式：接入数据层时一律不安全
LOSSY_EXR_COMPRESSIONS = ("dwaa", "dwab", "pxr24")


def exr_bit_depth_param(*, images: bool) -> Any:
    """`images`：用于「序列图输出设置」（其唯一输入可以是数据图，数据图文件总是原样复制，此参数只影响画面）；
    否则用于「多层 EXR 输出设置」（数据层无论此参数如何都写为 32 位浮点，所以只在接了画面时生效）。"""
    return P("half", group="file",
             applies=Param("format").one_of("exr") if images else WiredPicture("layers"))


def exr_compression_param(*, images: bool) -> Any:
    return P("zips", group="file",
              **({"applies": Param("format").one_of("exr")} if images else {}))


class ImageOutput(OutputSettings):
    id = "image.output"
    merges_into = "exr.output"  # 几个序列图输出设置合并成一个多层 EXR 输出设置（ExrOutput），一个节点一层
    format = Format("images")
    # 2：转色彩空间时 alpha 为 0 的像素保留颜色（io/color.py apply_premultiplied），原来被清成 0；
    # 3：交付清单带上画面的色彩空间（file_colorspace），缓存里的旧结果没有这一项，重算
    version = 3
    category = "out_picture"
    # 一个节点支持两种接法：
    #   1.「图像」：整张图（RGB、RGBA 或一张数据图）原样写出；
    #   2. R G B A：每个浮点通道一个端口，接入哪个端口即写为哪个通道，未接入的端口不产生对应通道
    #     （只接 A 则只写出 A，接入 R G B 则为 RGB，四个都接则为 RGBA）。
    # 两种接法只能使用一种（见下方 input_choice）：接入「图像」后 R G B A 四个端口置灰且不可连接；
    # 单独接入任一通道后「图像」置灰。两种都未使用时在提交前拦下。
    inputs = (Port("image", "image", optional=True, alpha=True, data=EITHER),
              *(Port(c, "image.1", c, optional=True)
                for c in CHANNEL_PORTS))
    # 两种接法只能使用一种，且至少使用一种。置灰说明、禁止连接以及未接入时的提交前拦截，
    # 均由框架根据此声明推导（NodeDef.input_choice / port_applies），节点本身不编写条件。
    input_choice = (("image",), CHANNEL_PORTS)
    on_node = ("name", "format")

    class Params(NodeParams):
        name: str = name_param("images")
        layer: str = P("", group="file",
                       applies=Param("format").one_of("exr"))
        # 格式只在写出画面时生效：接入了三或四通道的整张图，或 R G B 三个端口均已接入。
        # 其他情况（数据图原样复制、只接入 A 等零散通道）PNG / JPG 无法容纳，一律写为 EXR，此处置灰并说明原因。
        format: Literal["exr", "png", "jpg"] = P(
            "exr", group="file",
            applies=writes_a_picture(),
        )
        exr_bit_depth: Literal["half", "float"] = exr_bit_depth_param(images=True)
        exr_compression: Literal["none", "zips", "zip", "piz", "pxr24", "dwaa", "dwab"] = exr_compression_param(images=True)
        png_bit_depth: Literal[8, 16] = P(8, group="file",
                                          applies=Param("format").one_of("png"))
        png_compress_level: int = P(6, group="file", ge=0, le=9, applies=Param("format").one_of("png"))
        png_alpha: Literal["auto", "yes", "no"] = P("auto", group="file",
                                                    applies=Param("format").one_of("png"))
        jpg_quality: int = P(95, group="file", ge=1, le=100, applies=Param("format").one_of("jpg"))
        colorspace: str | None = colorspace_param(choices_from=("format",),
                                                  applies=WiredPicture("image"))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """「色彩空间」的默认值取决于「格式」（io/color.py colorspace_for_file，与读取节点使用同一规则），由网页写入参数。"""
        from ...io.color import load_config

        return {"colorspace": {"options": [], "default": load_config().colorspace_for_file(f"x.{params.get('format') or 'exr'}")}}

    @classmethod
    def target_space(cls, params: dict, cfg) -> str:
        """写出所用的色彩空间：「色彩空间」参数，未填写时按「格式」确定（与 choices 提供给网页的值相同）。"""
        return params["colorspace"] or cfg.colorspace_for_file(f"x.{params['format']}")

    @classmethod
    def file_colorspace(cls, ctx) -> str:
        """The files' colour space when the whole picture was written (a data map, or channels wired one by one: values)."""
        from ...data.payloads import is_data
        from ...io.color import load_config

        src = ctx.input("image")
        return cls.target_space(ctx.params, load_config()) if src is not None and not is_data(src) else ""

    @classmethod
    def write(cls, ctx) -> str:
        """按接入内容写出：整张图由 _write_whole 处理，单独接入的通道由 _write_channels 处理。"""
        wired = {c: p for c in CHANNEL_PORTS if (p := ctx.input(c)) is not None}
        src = ctx.input("image")
        if wired:
            return cls._write_channels(ctx, src, wired)
        if src is None:
            raise Invalid(Msg("E-IMAGES-NOINPUT"))
        return cls._write_whole(ctx, src)

    @classmethod
    def _write_whole(cls, ctx, src) -> str:
        import shutil
        from pathlib import Path

        import numpy as np

        from ...data import layers
        from ...data.payloads import has_alpha, image_files, is_data, is_labels, read_map, read_picture, window_of
        from ...data.types import channels_of
        from ...io import images
        from ...io.color import convert_picture, load_config
        from lab2shot_worker.files import write_exr

        exr = ctx.params["format"] == "exr"
        window = window_of(src)
        kept = window.has_overscan and exr and not is_data(src)  # 数据图的文件连同窗口原样复制
        if window.has_overscan and not exr and not is_data(src):
            o = window.overscan
            ctx.say("N-OUTPUT-OVERSCAN", param="format", format=ctx.params["format"].upper(),
                    left=o[0], top=o[1], right=o[2], bottom=o[3])
        still = bool(src.meta.get("still"))  # 整个镜头只有一张图（ST-map、HDRI）：一个文件，不带帧号
        items = sorted(image_files(src).items())[:1] if still else sorted(image_files(src).items())
        data = is_data(src)  # 数据图：其文件原样复制
        suffix = items[0][1].suffix.lower() if data and items else f".{ctx.params['format']}"
        if not data and has_alpha(src) and ctx.params["format"] == "jpg":
            ctx.say("W-OUTPUT-JPGALPHA", param="format")
        target = (lambda f: cls.out_file(ctx, suffix)) if still else (lambda f: cls.out_file(ctx, suffix, f))
        convert = None
        if not data:  # 画面：从数据包声明的色彩空间转换到文件的色彩空间
            cfg = load_config()
            space = cls.target_space(ctx.params, cfg)  # not `target`: that name is a frame's file path in this function
            convert = lambda picture: convert_picture(picture, cfg, src.meta["colorspace"], space)  # noqa: E731
        # 交付的通道名按「图层名 + 通道数」生成（data/layers.py）：图层名即本输出设置的「图层」参数，
        # 运动矢量写为 Nuke 的 forward.u … backward.v，深度写为 depth.Z；
        # 名称与内部数据包中的 R G B A 相同时不重写，直接原样复制（速度快且不改动任何字节）。
        labelled = data and is_labels(src)
        layer = layers.layer_names([ctx.params["layer"]], [channels_of(src.type)])[0] if data else ""
        names = layers.checked_channels(layer, channels_of(src.type), labelled, "") if data else []
        out_names = layers.layer_channels(layer, names, False, labelled) if data else []
        # 未填写图层名时原样复制（速度快且不改动任何字节）；填写后才按交付约定重写通道名
        rename = data and exr and bool(ctx.params["layer"]) and out_names != list(layers.CHANNEL_LETTERS[: len(out_names)])
        # 编号图填了图层名：通道名是 Cryptomatte 的一组（layer_channels），像素也要编成 Cryptomatte、带上清单，
        # 和「多层 EXR 输出设置」写编号图的做法相同；不填图层名时原样复制编号图本身（整数编号，float32）
        crypto = layers.crypto_layer_header(layer, src.meta.get("classes") or []) if labelled and rename else None
        def write(job):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            frame, path = job
            if data and not rename:
                shutil.copyfile(path, target(frame))
                return
            if data:  # 像素不变，仅改为交付用的通道名
                values, valid = read_map(path, window.data)
                if crypto is not None:  # 编号图写为 Cryptomatte（见上）
                    block = layers.crypto_encode(values[..., 0], src.meta["classes"])
                else:
                    block = np.concatenate([values, valid[..., None]], axis=-1) if len(out_names) > values.shape[-1] else values
                write_exr(target(frame), np.ascontiguousarray(block, np.float32), out_names, half=False, header=crypto,
                          compression=ctx.params["exr_compression"], windows=window.exr_windows if window.has_overscan else None)
                return
            picture = read_picture(src, path, window.data if kept else window.display_box)  # 有 alpha 时为预乘 RGBA
            if convert:
                picture = convert(picture)
            images.write_image(
                target(frame), np.ascontiguousarray(picture),
                exr_half=ctx.params["exr_bit_depth"] == "half", exr_compression=ctx.params["exr_compression"],
                png_bit_depth=ctx.params["png_bit_depth"], png_compress_level=ctx.params["png_compress_level"],
                alpha=ctx.params["png_alpha"], jpg_quality=ctx.params["jpg_quality"],
                windows=window.exr_windows if kept else None,
            )

        list(ctx.each_done(items, write))
        return Path(target(0)).name if still else cls.sequence_main(ctx, suffix)

    @classmethod
    def _write_channels(cls, ctx, src, wired: dict) -> str:
        """逐通道接法：R、G、B、A 四个端口，接入哪个端口即写为哪个通道。

        不做任何推测：端口名即通道名。同时接入「图像」时它仅作为底层，按位置分配到 R G B A 上，
        再由单独接入的通道覆盖（相当于 Nuke 中 Copy 接 Write 的做法）。未接入的端口不产生对应通道。
        此处不做预乘，也不做色彩转换（单个通道是数值而非颜色；需要预乘时使用「遮罩转 Alpha」）。
        """
        from pathlib import Path

        import numpy as np

        from ...data import layers
        from ...data.maps import map_at, same_size
        from ...data.payloads import file_at, is_data, read_picture, window_of
        from ...data.types import channels_of
        from ...io import images
        from ...io.color import convert_picture, load_config
        from lab2shot_worker.files import write_exr

        used = [p for p in (src, *wired.values()) if p is not None]
        same_size({ctx.node_type.port_label("image"): src, **wired})
        # 每个通道字母的来源：(数据包, 其通道序号)。先按「图像」分配，再由单独接入的端口覆盖
        parts: dict[str, tuple] = {}
        if src is not None:
            for i, letter in enumerate(CHANNEL_PORTS[:channels_of(src.type)]):
                parts[letter] = (src, i)
        parts.update({letter: (p, 0) for letter, p in wired.items()})
        letters = [c for c in CHANNEL_PORTS if c in parts]
        # 有损压缩会改变数值：此路径会重新编码（而非原样复制），因此只要有一个通道来自数值图即拦下，
        # 与「多层 EXR 输出设置」规则相同（param_refuses / E-EXR-LOSSYDATA）。
        data_from = {c for c, (p, _) in parts.items() if is_data(p)}
        if data_from and ctx.params["exr_compression"] in LOSSY_EXR_COMPRESSIONS:
            raise Invalid(Msg("E-EXR-LOSSYDATA", compression=cls.option_label("exr_compression", ctx.params["exr_compression"])))
        # PNG / JPG 只能容纳 RGB / RGBA，其他通道组合一律写为 EXR（此时「格式」已置灰并说明原因）
        exr = ctx.params["format"] == "exr" or letters not in PICTURE_LETTERS
        suffix = ".exr" if exr else f".{ctx.params['format']}"
        if not exr and "A" in letters and ctx.params["format"] == "jpg":
            ctx.say("W-OUTPUT-JPGALPHA", param="format")
        window = window_of(used[0])
        kept = window.has_overscan and exr
        if window.has_overscan and not exr:
            o = window.overscan
            ctx.say("N-OUTPUT-OVERSCAN", param="format", format=ctx.params["format"].upper(),
                    left=o[0], top=o[1], right=o[2], bottom=o[3])
        box = window.data if kept else window.display_box
        convert = None
        if src is not None and not is_data(src):
            cfg = load_config()
            space = cls.target_space(ctx.params, cfg)  # not `target`: that name is a frame's file path in this function
            convert = lambda picture: convert_picture(picture, cfg, src.meta["colorspace"], space)  # noqa: E731
        shots = [p for p in used if not p.meta.get("still")]
        still = not shots
        frames = sorted({f for p in (shots or used) for f in p.meta["frames"]})[:1 if still else None]
        target = (lambda f: cls.out_file(ctx, suffix)) if still else (lambda f: cls.out_file(ctx, suffix, f))
        # 通道名即端口名；填写「图层」时挂在该图层下（如 depth.R），留空时为不带前缀的 R G B A
        names = layers.layer_channels(ctx.params["layer"] or layers.RGBA, letters)
        # 画面按「位深」写出，数值图通道始终为 32 位无损（与「多层 EXR 输出设置」相同：半精度会改变数值）
        halves = [ctx.params["exr_bit_depth"] == "half" and c not in data_from for c in letters]

        def write(frame):
            """一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）。返回该帧缺少输入的通道。"""
            missing = []
            picture = None
            if src is not None and (path := file_at(src, frame)) is not None:
                picture = read_picture(src, path, box)
                if convert is not None:
                    picture = convert(picture)
            block = np.zeros((box[3], box[2], len(letters)), np.float32)
            for i, letter in enumerate(letters):
                packet, index = parts[letter]
                if packet is src:
                    if picture is None:
                        missing.append(cls.port_label("image"))
                        continue
                    block[..., i] = picture[..., index]
                    continue
                got = map_at(packet, frame, box)
                if got is None:
                    missing.append(letter)
                    continue
                block[..., i] = got[0][..., 0]
            if exr:
                write_exr(target(frame), np.ascontiguousarray(block), names, half=halves,
                          compression=ctx.params["exr_compression"], windows=window.exr_windows if kept else None)
            else:
                images.write_image(
                    target(frame), np.ascontiguousarray(block),
                    png_bit_depth=ctx.params["png_bit_depth"], png_compress_level=ctx.params["png_compress_level"],
                    alpha=ctx.params["png_alpha"], jpg_quality=ctx.params["jpg_quality"],
                )
            return missing

        lacking: dict[str, list[int]] = {}
        for frame, missing in zip(frames, ctx.each_done(frames, write)):
            for channel in missing:
                lacking.setdefault(channel, []).append(frame)
        for channel, missing in lacking.items():
            ctx.say("N-IMAGES-GAP", channel=channel, count=len(missing), first=missing[0])
        return Path(target(0)).name if still else cls.sequence_main(ctx, suffix)


class ExrLayerEntry(NodeParams):
    """「多层 EXR 输出设置」「图层」表中的一行：其输入端口（稳定不变，连线和节点图文件引用的是端口而非标签）、
    写入 EXR 的图层名以及写入的通道（可随时编辑，修改任一项都不影响连线）。每行均明确给出
    来源线 → 目标图层名 → 通道名，不含隐式内容；行内未指定时，通道取连接类型自身的通道（data/layers.py row_channels）。"""

    name: str = P(..., widget="fixed")
    label: str = P(...)
    channels: str = P("")


class ExrOutput(OutputSettings):
    id = "exr.output"
    format = Format("exr")
    # 2：转色彩空间时 alpha 为 0 的像素保留颜色（io/color.py apply_premultiplied），原来被清成 0；
    # 3：交付清单带上画面的色彩空间（file_colorspace），缓存里的旧结果没有这一项，重算
    version = 3
    category = "out_picture"
    ports_from = "layers"
    ports_from_side = "inputs"
    ports_from_type = "image"
    on_node = ("name",)
    # 只收各行接进来的图层：接着的全被「阻断」时它自己也被阻断（不出空文件）；被阻断 / 没选文件的那一行不写、不留空层
    collects = True

    class Params(NodeParams):
        name: str = name_param("layers")
        layers: list[ExrLayerEntry] = P(
            [], widget="table", group="layers",
        )
        exr_bit_depth: Literal["half", "float"] = exr_bit_depth_param(images=False)
        exr_compression: Literal["none", "zips", "zip", "piz", "pxr24", "dwaa", "dwab"] = exr_compression_param(images=False)
        colorspace: str | None = colorspace_param(applies=WiredPicture("layers"))

    @classmethod
    def choices(cls, params: dict, inputs: dict) -> dict:
        """「色彩空间」的默认值 ACEScg（OCIO 配置的 scene_linear）：由网页写入参数，对使用者可见。"""
        from ...io.color import SCENE_LINEAR, load_config, space

        return {"colorspace": {"options": [], "default": space(SCENE_LINEAR, load_config())}}

    @classmethod
    def target_space(cls, params: dict, cfg) -> str:
        from ...io.color import SCENE_LINEAR, space

        return params["colorspace"] or space(SCENE_LINEAR, cfg)

    @classmethod
    def file_colorspace(cls, ctx) -> str:
        """The picture layers' colour space ("" when every layer is a data map)."""
        from ...data.payloads import is_data
        from ...io.color import load_config

        wired = [q for e in ctx.params["layers"] if (q := ctx.input(e["name"])) is not None]
        return cls.target_space(ctx.params, load_config()) if any(not is_data(q) for q in wired) else ""

    @classmethod
    def param_refuses(cls, data_type: str, params: dict) -> Msg | None:
        # 有损压缩会改变数值。此处只能获知类型：单通道或双通道（深度、遮罩、ST-map 等）必然是数值图，予以拦下；
        # 三或四通道既可能是照片也可能是法线图，无法按通道数区分，由接线者负责。
        if channels_of(data_type) not in (1, 2) or params["exr_compression"] not in LOSSY_EXR_COMPRESSIONS:
            return None
        return Msg("E-EXR-LOSSYDATA", compression=cls.option_label("exr_compression", params["exr_compression"]))

    @classmethod
    def write(cls, ctx) -> str:
        import json

        import numpy as np
        from lab2shot_worker.files import write_exr

        from ...data import layers
        from ...data.payloads import has_alpha, has_validity, image_files, is_data, is_labels, read_map, read_picture, window_of
        from ...data.types import channels_of
        from ...io.color import convert_picture, load_config
        from ...data.contracts import own_meta

        p = ctx.params
        # 加了行但还没接线的图层（网页上点「＋」/「添加」加的空行；行的口都是可选口，nodes/base.py made_ports）：
        # 跳过这一行，提示一次（列出所有跳过的图层名），不报错。一行都没接上时与表为空相同
        # 被「阻断」或接着没选文件的读取（ctx.quiet：这次有意不给）的行：不写这一层、不留空层，也不提示「没接上」
        unwired = [e for e in p["layers"] if ctx.input(e["name"]) is None and e["name"] not in getattr(ctx, "quiet", ())]
        rows = [e for e in p["layers"] if ctx.input(e["name"]) is not None]
        if not rows:
            raise Invalid(Msg("E-LAYERS-EMPTY"))
        if unwired:
            ctx.say("N-LAYERS-UNWIRED", layers=i18n.Both.of(lambda: i18n.separator().join(e["label"] or e["name"] for e in unwired)), count=len(unwired))
        results = [ctx.input(e["name"]) for e in rows]
        sizes = sorted({(q.meta["width"], q.meta["height"]) for q in results})
        if len(sizes) > 1:
            raise Invalid(Msg("E-LAYERS-SIZES", sizes=[f"{w}×{h}" for w, h in sizes]))
        w, h = sizes[0]
        counts = [channels_of(q.type) for q in results]
        # 图层名由该行自身填写（网页接线时按来源端口名预填：depth、normal、mask 等），不按数据类型确定
        # a row left without a name takes the one its data goes by (an ensemble's result: kit/ensemble.py ensemble_name)
        names = layers.layer_names([e["label"] or q.meta.get("layer", "") for e, q in zip(rows, results)], counts)
        cfg = load_config()
        target = cls.target_space(p, cfg)
        # 每一层是画面还是数值图、是否为编号图，由数据自带的色彩空间和类别表决定，不按通道数推测
        pictures = [not is_data(q) for q in results]
        labelled = [is_labels(q) for q in results]
        # 各画面层从其数据包声明的色彩空间转换到文件的色彩空间
        sources = {i: q.meta["colorspace"] for i, q in enumerate(results) if pictures[i]}
        # 来源线 → 目标图层名 → 通道名，均按各行的设置写出：表中未指定的位置不写入任何内容。
        # 数值图自带「有效区域」通道时，交付时写为该层的 A（Nuke 中为 depth.A）；画面的 alpha 同理。
        extra = [has_alpha(q) if pictures[i] else has_validity(q) for i, q in enumerate(results)]
        wanted = [layers.checked_channels(n, c - (1 if pic and a else 0), lab, e.get("channels", ""))
                  for n, c, pic, a, lab, e in zip(names, counts, pictures, extra, labelled, rows)]
        channels = [layers.layer_channels(n, c, a, lab) for n, c, a, lab in zip(names, wanted, extra, labelled)]
        # 画面层按所选位深写出（半精度或浮点）；数据层始终为 32 位浮点无损
        # （Cryptomatte 的 id 同样如此：半精度浮点无法表示超过 2048 的整数）
        halves = [(p["exr_bit_depth"] == "half") if pic else False
                  for pic, ch in zip(pictures, channels) for _ in ch]
        # 在文件头中注明最后的 A 通道表示有效区域而非数据：交付给 Nuke 时采用 depth.A 约定，
        # 读回时据此还原为 image.1 + 有效通道，而不是当作两个数据通道。
        told = {n: {**own_meta(q), **({"colorspace": target} if pic else {"values": True}),
                    **({"validity": True} if a and not pic else {})}
                for n, q, pic, a in zip(names, results, pictures, extra)}
        header = {key: v for n, q, lab in zip(names, results, labelled) if lab
                  for key, v in layers.crypto_layer_header(n, q.meta.get("classes") or []).items()}
        # 任一输入包含的所有帧；静帧（整个镜头只有一张图）出现在每一帧中
        shots = [q for q in results if not q.meta.get("still")]
        frames = sorted({f for q in shots or results for f in q.meta["frames"]})
        for q, n in zip(results, names):
            lacking = [f for f in frames if str(f) not in q.meta["files"]] if not q.meta.get("still") else []
            if lacking:
                ctx.say("N-LAYERS-GAP", layer=n, count=len(lacking), first=lacking[0])
        files = [image_files(q) for q in results]

        def write(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            paths = [next(iter(fs.values())) if q.meta.get("still") else fs.get(f) for q, fs in zip(results, files)]
            # 每一层自身的数据窗口（包括画幅外的像素），放在该帧所有窗口的并集中
            wins = {i: window_of(q).data for i, (q, path) in enumerate(zip(results, paths)) if path is not None}
            union = _union(wins.values()) if wins else (0, 0, w, h)
            blocks = []
            for i, (q, ch, path) in enumerate(zip(results, channels, paths)):
                if path is None:
                    blocks.append(np.zeros((union[3], union[2], len(ch)), np.float32))
                    continue
                if pictures[i]:  # 有 alpha 时为预乘 RGBA
                    block = convert_picture(read_picture(q, path), cfg, sources[i], target)
                elif labelled[i]:  # 编号图写为 Cryptomatte（Nuke 可直接使用）
                    block = layers.crypto_encode(read_map(path, window_of(q).data)[0][..., 0], q.meta["classes"])
                else:
                    data, alpha = read_map(path, window_of(q).data)
                    block = np.concatenate([data, alpha[..., None]], axis=-1) if len(ch) > data.shape[-1] else data
                blocks.append(_placed_in(block, wins[i], union))
            write_exr(cls.out_file(ctx, ".exr", f), np.concatenate(blocks, axis=-1), [c for ch in channels for c in ch], half=halves,
                      attrs={layers.HEADER: json.dumps(told, ensure_ascii=False)}, header=header, compression=p["exr_compression"],
                      windows=((0, 0, w, h), union))

        list(ctx.each_done(frames, write))
        return cls.sequence_main(ctx, ".exr")


def _union(windows) -> tuple[int, int, int, int]:
    """包含全部 `windows` 的最小窗口（x, y, 宽, 高）。"""
    boxes = [(x, y, x + w, y + h) for x, y, w, h in windows]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    return x0, y0, max(b[2] for b in boxes) - x0, max(b[3] for b in boxes) - y0


def _placed_in(block, window, union):
    """图层位于 `window` 的像素 [h,w,C] 放入该帧的 `union` 窗口中，周围补 0。"""
    import numpy as np

    if tuple(window) == tuple(union):
        return block
    (x, y, w, h), (ux, uy, uw, uh) = window, union
    out = np.zeros((uh, uw, block.shape[-1]), np.float32)
    out[y - uy:y - uy + h, x - ux:x - ux + w] = block
    return out


NODES = (ImageOutput, ExrOutput)
