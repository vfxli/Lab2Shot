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

from ...availability import All, AnyOf
from ...data.types import channels_of
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeParams, P, Port, colorspace_param
from ..output import OutputSettings, name_param
from ..applies import Param, Wired, WiredType

# 「序列图输出设置」每个通道一个端口：名称采用 CG 通用写法（与 Nuke 一致），接入哪个端口即写为哪个通道
CHANNEL_PORTS = ("R", "G", "B", "A")
# 由单个通道组成的画面：只有这两种通道组合可写为 PNG / JPG（其他组合只能写 EXR）
PICTURE_LETTERS = (["R", "G", "B"], ["R", "G", "B", "A"])


def writes_a_picture():
    """「序列图输出设置」本次写出的是否为画面（PNG / JPG 可容纳的 RGB / RGBA）：
    「图像」端口接入了三或四通道的整张图，或 R G B 三个端口均已单独接入。

    其他情况（数据图原样复制、只接入 A 等零散通道、未接入任何内容）只能写为 EXR，
    因此「格式」和「色彩空间」在这些情况下置灰并说明原因。"""
    return AnyOf(All(Wired("image"), WiredType("image", "image.3", "image.4")),
                 All(Wired("R"), Wired("G"), Wired("B")))

# ------------------------------------------------------------------ EXR / PNG / JPG 共用的格式参数

EXR_BIT_DEPTH_LABELS = {"half": "16 位半浮点", "float": "32 位浮点"}
# 采用 Nuke 的名称；worker_sdk.files.EXR_COMPRESSIONS 中有各项对应的 OpenEXR/OpenImageIO 名称
EXR_COMPRESSION_LABELS = {"none": "无", "zips": "ZIP 逐行", "zip": "ZIP 16 行", "piz": "PIZ", "pxr24": "PXR24", "dwaa": "DWAA", "dwab": "DWAB"}
# 作用于文件所有通道（包括数据层）的压缩方式：接入数据层时一律不安全
LOSSY_EXR_COMPRESSIONS = ("dwaa", "dwab", "pxr24")
PNG_BIT_DEPTH_LABELS = {"8": "8 位", "16": "16 位"}
PNG_ALPHA_LABELS = {"auto": "自动", "yes": "带", "no": "不带"}


def exr_bit_depth_param(*, images: bool) -> Any:
    """`images`：用于「序列图输出设置」（其唯一输入可以是数据图，数据图文件总是原样复制，此参数只影响画面）；
    否则用于「多层 EXR 输出设置」（数据层无论此参数如何都写为 32 位浮点）。"""
    note = "；数据图（深度、法线、位置……）原样复制它的 EXR，总是 32 位，不受这个影响（这里只对图像起作用）" if images \
        else "；数据类的图层（深度、法线、位置……）总是 32 位，不受这个影响"
    return P("half", label="位深", group="文件", option_labels=EXR_BIT_DEPTH_LABELS,
              **({"applies": Param("format").one_of("exr")} if images else {}),
              help=f"写成半精度（16 位）还是全精度（32 位）浮点{note}")


def exr_compression_param(*, images: bool) -> Any:
    note = "；数据图（深度、法线……）原样复制它的 EXR，不受这个影响（这里只对图像起作用）" if images \
        else "；一层是数据图的（深度、法线……）时选 PXR24、DWAA、DWAB 会被拦下：这三种连数据图层也会压坏，换一种压缩"
    return P("zips", label="压缩", group="文件", option_labels=EXR_COMPRESSION_LABELS,
              **({"applies": Param("format").one_of("exr")} if images else {}),
              help=f"EXR 的压缩方式：无不压缩；ZIP 逐行（Nuke 默认）、ZIP 16 行、PIZ 都是无损；PXR24、DWAA、DWAB 有损，文件更小但会改动数值{note}")


class ImageOutput(OutputSettings):
    id = "core.output_images"
    category = "out_picture"
    # 一个节点支持两种接法：
    #   1.「图像」：整张图（RGB、RGBA 或一张数据图）原样写出；
    #   2. R G B A：每个浮点通道一个端口，接入哪个端口即写为哪个通道，未接入的端口不产生对应通道
    #     （只接 A 则只写出 A，接入 R G B 则为 RGB，四个都接则为 RGBA）。
    # 两种接法只能使用一种（见下方 input_choice）：接入「图像」后 R G B A 四个端口置灰且不可连接；
    # 单独接入任一通道后「图像」置灰。两种都未使用时在提交前拦下。
    inputs = (Port("image", "image", "图像", optional=True, alpha=True,
                   help="一整张图照原样写：RGB、RGBA，或深度、遮罩这类数据图（按通道写出）。"
                        "要逐条接通道就用右边的 R G B A，两种接法只能用一种"),
              *(Port(c, "image.1", c, optional=True,
                     help=f"一条浮点通道写成 EXR 的 {c} 通道：接到哪个口就是哪个通道，不做任何判断。"
                          f"没接的口就没有这条通道{'；这里不替你预乘颜色，要预乘用「遮罩转 Alpha」' if c == 'A' else ''}")
                for c in CHANNEL_PORTS))
    # 两种接法只能使用一种，且至少使用一种。置灰说明、禁止连接以及未接入时的提交前拦截，
    # 均由框架根据此声明推导（NodeDef.input_choice / port_applies），节点本身不编写条件。
    input_choice = (("image",), CHANNEL_PORTS)
    on_node = ("name", "format")

    class Params(NodeParams):
        name: str = name_param("images")
        layer: str = P("", label="图层", group="文件", placeholder="按通道",
                       applies=Param("format").one_of("exr"),
                       help="写成 EXR 时通道叫什么，按这个图层名给（和 Nuke 的图层一个意思）：留空就是我们自己的 R G B A。"
                            "填 motion 写成 Nuke 标准的 forward.u/v、backward.u/v（VectorBlur、IDistort 拿来就用）；"
                            "填 depth 写成 depth.Z；填 N、P 写成 N.R、P.R 这样。几样结果写进同一个 EXR 用「多层 EXR 输出设置」")
        # 格式只在写出画面时生效：接入了三或四通道的整张图，或 R G B 三个端口均已接入。
        # 其他情况（数据图原样复制、只接入 A 等零散通道）PNG / JPG 无法容纳，一律写为 EXR，此处置灰并说明原因。
        format: Literal["exr", "png", "jpg"] = P(
            "exr", label="格式", group="文件", option_labels={"exr": "EXR", "png": "PNG", "jpg": "JPG"},
            applies=writes_a_picture(),
            help="图像写成哪种文件，文件名是 名字.1001.exr 这样，帧号不变；只有一张的图（HDRI、ST-map）没有帧号。数据图（深度图、遮罩……）总是原样复制它的 EXR。"
                 "带 alpha 的图像：EXR 写 RGBA（预乘，Nuke 的约定），PNG 写 RGBA（按 PNG 标准不预乘），JPG 没有 alpha，只写颜色（会提醒）。"
                 "单条通道接 R、G、B、A 时，只有凑齐 R G B 才写得成 PNG / JPG，别的组合只能写 EXR",
        )
        exr_bit_depth: Literal["half", "float"] = exr_bit_depth_param(images=True)
        exr_compression: Literal["none", "zips", "zip", "piz", "pxr24", "dwaa", "dwab"] = exr_compression_param(images=True)
        png_bit_depth: Literal[8, 16] = P(8, label="位深", group="文件", option_labels=PNG_BIT_DEPTH_LABELS,
                                          applies=Param("format").one_of("png"), help="每个通道 8 位还是 16 位；PNG 总是无损压缩，位深只影响精度")
        png_compress_level: int = P(6, label="压缩强度", group="文件", ge=0, le=9, applies=Param("format").one_of("png"),
                                    help="PNG 的压缩强度，0（不压缩，最快）到 9（压得最狠，最慢）；不影响画质，PNG 总是无损")
        png_alpha: Literal["auto", "yes", "no"] = P("auto", label="带 alpha", group="文件", option_labels=PNG_ALPHA_LABELS,
                                                    applies=Param("format").one_of("png"), help="图像带 alpha 就带上（自动），总是带上（没有的补一个不透明的），或者总是不带")
        jpg_quality: int = P(95, label="质量", group="文件", ge=1, le=100, applies=Param("format").one_of("jpg"),
                             help="JPG 压缩质量，1 到 100：越大画质越好、文件越大。JPG 不能带 alpha，带 alpha 的图像只写颜色（压在黑底上）")
        colorspace: str | None = colorspace_param("写出去的文件是什么色彩空间：画面从工作空间（sRGB）转过去。按「格式」填好默认值：EXR 是 ACEScg，"
                                                  "PNG / JPG 是 sRGB；要别的（ACES2065-1、Linear Rec.709）在这里改。"
                                                  "数据图和单条接进 R G B A 的通道都不做转换：那是数值，不是颜色",
                                                  choices_from=("format",),
                                                  applies=All(Wired("image"), WiredType("image", "image.3", "image.4")))

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
            ctx.say("N-OUTPUT-OVERSCAN", param="format", node=cls.label, format=ctx.params["format"].upper(),
                    left=o[0], top=o[1], right=o[2], bottom=o[3])
        still = bool(src.meta.get("still"))  # 整个镜头只有一张图（ST-map、HDRI）：一个文件，不带帧号
        items = sorted(image_files(src).items())[:1] if still else sorted(image_files(src).items())
        data = is_data(src)  # 数据图：其文件原样复制
        suffix = items[0][1].suffix.lower() if data and items else f".{ctx.params['format']}"
        if not data and has_alpha(src) and ctx.params["format"] == "jpg":
            ctx.say("W-OUTPUT-JPGALPHA", param="format")
        target = (lambda f: cls.out_file(ctx, suffix)) if still else (lambda f: cls.out_file(ctx, suffix, f))
        convert = None
        if not data:  # 画面：从工作色彩空间（读取节点的输出；旧数据包使用其声明的空间）转换到文件的色彩空间
            cfg = load_config()
            target = cls.target_space(ctx.params, cfg)
            convert = lambda picture: convert_picture(picture, cfg, src.meta["colorspace"], target)  # noqa: E731
        # 交付的通道名按「图层名 + 通道数」生成（data/layers.py）：图层名即本输出设置的「图层」参数，
        # 运动矢量写为 Nuke 的 forward.u … backward.v，深度写为 depth.Z；
        # 名称与内部数据包中的 R G B A 相同时不重写，直接原样复制（速度快且不改动任何字节）。
        labelled = data and is_labels(src)
        layer = layers.layer_names([ctx.params["layer"]], [channels_of(src.type)])[0] if data else ""
        names = layers.checked_channels(layer, channels_of(src.type), labelled, "") if data else []
        out_names = layers.layer_channels(layer, names, False, labelled) if data else []
        # 未填写图层名时原样复制（速度快且不改动任何字节）；填写后才按交付约定重写通道名
        rename = data and exr and bool(ctx.params["layer"]) and out_names != list(layers.CHANNEL_LETTERS[: len(out_names)])
        for frame, path in ctx.each(items):
            if data and not rename:
                shutil.copyfile(path, target(frame))
                continue
            if data:  # 像素不变，仅改为交付用的通道名
                values, valid = read_map(path, window.data)
                block = np.concatenate([values, valid[..., None]], axis=-1) if len(out_names) > values.shape[-1] else values
                write_exr(target(frame), np.ascontiguousarray(block, np.float32), out_names, half=False,
                          compression=ctx.params["exr_compression"], windows=window.exr_windows if window.has_overscan else None)
                continue
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
        same_size({"图像": src, **wired})
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
            raise Invalid(Msg("E-EXR-LOSSYDATA", compression=EXR_COMPRESSION_LABELS[ctx.params["exr_compression"]]))
        # PNG / JPG 只能容纳 RGB / RGBA，其他通道组合一律写为 EXR（此时「格式」已置灰并说明原因）
        exr = ctx.params["format"] == "exr" or letters not in PICTURE_LETTERS
        suffix = ".exr" if exr else f".{ctx.params['format']}"
        if not exr and "A" in letters and ctx.params["format"] == "jpg":
            ctx.say("W-OUTPUT-JPGALPHA", param="format")
        window = window_of(used[0])
        kept = window.has_overscan and exr
        if window.has_overscan and not exr:
            o = window.overscan
            ctx.say("N-OUTPUT-OVERSCAN", param="format", node=cls.label, format=ctx.params["format"].upper(),
                    left=o[0], top=o[1], right=o[2], bottom=o[3])
        box = window.data if kept else window.display_box
        convert = None
        if src is not None and not is_data(src):
            cfg = load_config()
            target = cls.target_space(ctx.params, cfg)
            convert = lambda picture: convert_picture(picture, cfg, src.meta["colorspace"], target)  # noqa: E731
        shots = [p for p in used if not p.meta.get("still")]
        still = not shots
        frames = sorted({f for p in (shots or used) for f in p.meta["frames"]})[:1 if still else None]
        target = (lambda f: cls.out_file(ctx, suffix)) if still else (lambda f: cls.out_file(ctx, suffix, f))
        # 通道名即端口名；填写「图层」时挂在该图层下（如 depth.R），留空时为不带前缀的 R G B A
        names = layers.layer_channels(ctx.params["layer"] or layers.RGBA, letters)
        # 画面按「位深」写出，数值图通道始终为 32 位无损（与「多层 EXR 输出设置」相同：半精度会改变数值）
        halves = [ctx.params["exr_bit_depth"] == "half" and c not in data_from for c in letters]
        lacking: dict[str, list[int]] = {}
        for frame in ctx.each(frames):
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
                        lacking.setdefault("图像", []).append(frame)
                        continue
                    block[..., i] = picture[..., index]
                    continue
                got = map_at(packet, frame, box)
                if got is None:
                    lacking.setdefault(letter, []).append(frame)
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
        for channel, missing in lacking.items():
            ctx.say("N-IMAGES-GAP", channel=channel, count=len(missing), first=missing[0])
        return Path(target(0)).name if still else cls.sequence_main(ctx, suffix)


class ExrLayerEntry(NodeParams):
    """「多层 EXR 输出设置」「图层」表中的一行：其输入端口（稳定不变，连线和节点图文件引用的是端口而非标签）、
    写入 EXR 的图层名以及写入的通道（可随时编辑，修改任一项都不影响连线）。每行均明确给出
    来源线 → 目标图层名 → 通道名，不含隐式内容；行内未指定时，通道取连接类型自身的通道（data/layers.py row_channels）。"""

    name: str = P(..., label="端口", widget="fixed", help="节点图里这一行的接线口，不随图层名改变")
    label: str = P(..., label="图层名", help="写进 EXR 的图层名（和 Nuke 的图层一样）：rgba、depth、N、P、mask、stmap、motion、confidence、"
                                            "CryptoObject……接线时按类型自动给，随便改；重名会自动编号")
    channels: str = P("", label="通道", placeholder="按类型",
                      help="这一层写进哪几个通道，空格分开（深度图写 Z 通道，法线图写 R G B，遮罩写 A，ST-map 写 R G，置信度写 Y）：接线时按类型自动给，"
                           "可以改成别的名字（如法线写成 X Y Z），个数要和类型一致。标出哪些像素有值的 A 通道按类型自动带上；"
                           "分割（Cryptomatte）和运动矢量按 Nuke 的标准命名，不能改")


class ExrOutput(OutputSettings):
    id = "core.output_exr"
    category = "out_picture"
    ports_from = "layers"
    ports_from_side = "inputs"
    ports_from_type = "image"
    on_node = ("name",)

    class Params(NodeParams):
        name: str = name_param("layers")
        layers: list[ExrLayerEntry] = P(
            [], label="图层", widget="table", group="图层",
            help="每一行是 EXR 的一层，从上到下就是写出的图层顺序，写明「哪根线 → 哪个图层 → 哪几个通道」：往节点上拖一根线自动加一行，"
                 "图层名和通道按接的类型给默认值，都能改；拖动一行调顺序；删掉一行，连它的口和线一起去掉",
        )
        exr_bit_depth: Literal["half", "float"] = exr_bit_depth_param(images=False)
        exr_compression: Literal["none", "zips", "zip", "piz", "pxr24", "dwaa", "dwab"] = exr_compression_param(images=False)
        colorspace: str | None = colorspace_param("图像图层写成这个色彩空间，默认 ACEScg（OCIO 配置的 scene_linear）；画面从工作空间（sRGB）转过去。Nuke 读 EXR 默认当它是线性。数据图不做转换",
                                                  applies=WiredType("layers", "image.3", "image.4"))

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
    def param_refuses(cls, data_type: str, params: dict) -> Msg | None:
        # 有损压缩会改变数值。此处只能获知类型：单通道或双通道（深度、遮罩、ST-map 等）必然是数值图，予以拦下；
        # 三或四通道既可能是照片也可能是法线图，无法按通道数区分，由接线者负责。
        if channels_of(data_type) not in (1, 2) or params["exr_compression"] not in LOSSY_EXR_COMPRESSIONS:
            return None
        return Msg("E-EXR-LOSSYDATA", compression=EXR_COMPRESSION_LABELS[params["exr_compression"]])

    @classmethod
    def write(cls, ctx) -> str:
        import json

        import numpy as np
        from lab2shot_worker.files import write_exr

        from ...data import layers
        from ...data.payloads import has_alpha, has_validity, image_files, is_data, is_labels, read_map, read_picture, window_of
        from ...data.types import channels_of
        from ...io import images
        from ...io.color import convert_picture, load_config
        from ...data.contracts import own_meta

        p = ctx.params
        rows = p["layers"]
        if not rows:
            raise Invalid(Msg("E-LAYERS-EMPTY"))
        results = [ctx.input(e["name"]) for e in rows]
        sizes = sorted({(q.meta["width"], q.meta["height"]) for q in results})
        if len(sizes) > 1:
            raise Invalid(Msg("E-LAYERS-SIZES", sizes=[f"{w}×{h}" for w, h in sizes]))
        (w, h), kinds = sizes[0], [q.type for q in results]
        counts = [channels_of(q.type) for q in results]
        # 图层名由该行自身填写（网页接线时按来源端口名预填：depth、normal、mask 等），不按数据类型确定
        names = layers.layer_names([e["label"] for e in rows], counts)
        cfg = load_config()
        target = cls.target_space(p, cfg)
        # 每一层是画面还是数值图、是否为编号图，由数据自带的色彩空间和类别表决定，不按通道数推测
        pictures = [not is_data(q) for q in results]
        labelled = [is_labels(q) for q in results]
        # 各画面层从工作色彩空间（旧数据包使用其声明的空间）转换到文件的色彩空间
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
                  for key, v in layers.crypto_header(
                      n, list(layers.class_names([c for c in (q.meta.get("classes") or []) if c["index"] > 0]).values())
                  ).items()}
        # 任一输入包含的所有帧；静帧（整个镜头只有一张图）出现在每一帧中
        shots = [q for q in results if not q.meta.get("still")]
        frames = sorted({f for q in shots or results for f in q.meta["frames"]})
        for q, n in zip(results, names):
            lacking = [f for f in frames if str(f) not in q.meta["files"]] if not q.meta.get("still") else []
            if lacking:
                ctx.say("N-LAYERS-GAP", layer=n, count=len(lacking), first=lacking[0])
        for f in ctx.each(frames):
            files = [image_files(q) for q in results]
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
