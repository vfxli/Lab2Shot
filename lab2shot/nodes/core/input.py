"""输入：读取序列、视频、视频转序列。来自 DCC 文件的三维数据由格式模块的导入节点读入（lab2shot/nodes/formats.py）。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...data.units import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH
from ...errors import Invalid
from ...io.sequence import IMAGE_EXTS, VIDEO_EXTS
from ...messages import Msg
from ..base import HEAVY, Info, NodeDef, NodeParams, P, Port, ReadsFile, colorspace_param
from ...data.contracts import Shape
from ..applies import Cost, Param

def _short_names(channels: list[str]) -> list[str]:
    """读取该图层时各通道使用的名称：通常为短名称（R、Z、u）。短名称在该层内重复时
    （如 Nuke 的 motion 层为 forward.u forward.v backward.u backward.v，短名称为 U V U V），
    整层使用完整名称，否则 backward 的两个通道会读成 forward 的值。"""
    from ...data.layers import role

    short = [role(c) for c in channels]
    return list(channels) if len(set(short)) != len(short) else short


def _outputs_of(found: dict[str, dict]) -> list[tuple[str, str, str, list[str]]]:
    """文件中的图层 → 节点的输出端口，每层一个，顺序与文件一致：(端口名, 显示名称, 图层名, 读取的通道)。

    不提供图层或通道的选择：端口与文件内容一一对应，与 Nuke 的 Read 相同。
    端口（made_ports）与读取（cook）都以此处为唯一来源：端口与图层、通道的对应关系只在此计算一次。"""
    out: list[tuple[str, str, str, list[str]]] = []
    for layer, info in found.items():
        name = "image" if layer == "rgba" else layer
        while any(o[0] == name for o in out):
            name += "_"
        # Cryptomatte 是一整套规范（一组 RGBA 层 + 元数据 + 名称表），解码后为一张编号图，即一个「id」通道
        picked = ["id"] if "manifest" in info else _short_names(info["channels"])
        if info.get("validity") and len(picked) > 1:  # 最后的 A 通道表示有效区域，不是数据通道
            picked = picked[:-1]
        # 端口名称表明其内容：单通道的 rgba 层（灰度 PNG、单通道 TIFF）不是图像，命名为「Mask」，
        # 双通道为「UV」，三或四通道为 RGB / RGBA，与端口类型的名称一致。其他层沿用文件中的图层名。
        label = layer if layer != "rgba" else {1: "Mask", 2: "UV", 3: "RGB"}.get(len(picked), "RGBA")
        out.append((name, label, layer, picked))
    return out


@lru_cache(maxsize=64)
def _opened(path: str):
    """将该上传内容打开为一份素材。已上传的文件不会改变（transfer/uploads.py 按内容寻址：内容不同即为另一个引用），
    因此每份素材只列一次目录。输出端口直接跟随文件中的图层，每次检查连线、计算状态都会调用此函数
    （nodes/applies.py output_ports），不能每次都重新扫描目录。"""
    from ...io.sources import open_source

    return open_source(path)


@lru_cache(maxsize=64)
def _layers_of(path: str, view: str | None = None) -> dict[str, dict]:
    """上传文件的图层；立体文件时为其中一个视角的图层（上传内容不会改变，因此每份只描述一次）。

    文件夹中链接的是通道子集时，图层以原文件的申报为准（`transfer/uploads.py declared_layers_for`）：
    子集 EXR 只包含已连线的通道，而节点端口需要原文件的全部图层，否则上传 rgba 一个通道后，
    depth、N 等端口会从节点上消失。申报结果与打开完整文件读取的结果由同一段代码得出（`describe_file`），
    子集增大时结果不变，因此仍按路径缓存。"""
    from ...data.layers import describe_file
    from ..services import services

    return services().plan.declared_layers(Path(path)) or describe_file(path, view)


@lru_cache(maxsize=64)
def _views_of(path: str) -> tuple[str, ...]:
    """上传的立体文件的各视角，按文件中的顺序；单视角画面时为 ()。

    唯一的调用方是 `ReadSequence._view`：读取立体 EXR 时需要知道第一个视角的名称（不指定视角读取时，
    两只眼的通道会混在一起）。"""
    from ...io.images import views

    return tuple(views(path))


def _declared(params) -> dict | None:
    """字节尚未上传时，申报阶段从文件头读出的描述（`{"of", "layers", "frames", "size"}`），
    没有时为 None。

    选择文件后不立即上传，只上传已连线的通道；但端口来自打开文件后的统计，没有文件就没有端口，也就无法连线。
    为此，选择文件时只上传第一帧开头的几十 KB，服务器用其自身的 `describe_file` 读出图层。
    节点只能通过此函数获取该信息。

    不做缓存：结果按当前账号给出（transfer/uploads.py declared：其他账号的申报对当前账号不存在），
    按 ref 缓存会导致账号之间串用。它只读取几 KB 的 JSON，开销很小；
    `_layers_of` 等缓存可以按路径存储，是因为路径已经过 `resolve` 的账号校验。"""
    from ..services import services

    ref = params.get("path")
    if not ref:
        return None
    try:
        return services().plan.declared_upload(str(ref))
    except (OSError, ValueError):
        return None


class ReadSequence(ReadsFile, NodeDef):
    id = "core.read_sequence"
    version = 4  # 图像数据包现在总会注明是否带 alpha
    frame_source = True
    no_file = "没有选择序列图"
    # 选择文件后不上传任何字节（`base.py ReadsFile.head_is_enough`）：
    # 本节点依靠两条回退仍能回答查询：端口按申报时从文件头读出的图层生成（`made_ports` 的 `_declared`），
    # 色彩空间按文件名判断（`choices` 中的相应部分，OCIO 文件规则只依据名称）。
    head_is_enough = True
    category = "read_plate"

    class Params(NodeParams):
        path: str = P("", label="序列图", help="从你的电脑选择序列图：在对话框里把所有帧一起选中，或者选整个文件夹，也可以直接拖进来；帧号在名字哪里都认得（plate.1001.exr、000001_left.png）。文件夹里有好几段序列（左右眼、不同的 pass）时列出来让你选。名字里没有数字的一张图（HDRI、照片）当作一帧。文件会上传到服务器，同样的文件只传一次。命令行和 DCC 插件里写路径时，plate.####.exr、plate.%04d.exr、plate.$F4.exr 这些写法都认得。只算其中一段帧用计算按钮旁的帧范围", widget="sequence", group="文件", accept=sorted(IMAGE_EXTS))
        colorspace: str | None = colorspace_param(choices_from=("path",))  # 其「自动」项显示按文件规则得到的色彩空间

    @classmethod
    def _source(cls, params):
        src = _opened(str(cls.path(params)))
        if src.kind == "video":
            raise Invalid(Msg("E-READ-VIDEO", file=Path(src.display).name))
        return src

    @classmethod
    def _view(cls, src) -> str | None:
        """立体 EXR（一个文件包含左右眼）读取文件中的第一个视角，通常为 left；
        单视角画面（绝大多数文件）返回 None。此处不询问、不报告、不提示。

        读取第一个视角而不是不指定视角：EXR 中的多视角按通道名区分（`left.R`、`right.R`，或每个
        part 带 `view` 属性）。不指定视角时，`data/layers.py describe_file` 会列出两只眼的全部通道，
        立体文件会产生双倍的输出端口，两只眼的 rgba / depth 混在一起，画面也不正确。指定第一个视角后，
        读出的即单只眼的画面（名称中不含视角部分），与普通序列相同。

        不提供选择视角的参数：实际流程中立体项目的左右眼分为两条序列（`plate_left.####.exr` /
        `plate_right.####.exr`，各接一个「读取序列」），不会放在同一个文件中。"""
        found = _views_of(str(next(iter(src.files().values()))))
        return found[0] if len(found) > 1 else None

    @classmethod
    def _layers(cls, src) -> dict[str, dict]:
        """文件第一帧中的图层（立体文件按其第一个视角，见 `_view`）。

        文件图层信息在全项目中只有这一个来源：data/layers.py describe_file（`_layers_of` 为其缓存）。
        输出端口（made_ports）和读取（cook）均由此生成，不经过参数表，因此不存在参数表中的图层在文件中不存在的情况。"""
        return _layers_of(str(next(iter(src.files().values()))), cls._view(src))

    @classmethod
    def made_ports(cls, params) -> tuple[Port, ...]:
        """输出端口 = 所选文件中的全部图层，按名称逐一列出。端口类型按该层读出的通道数确定
        （一个为 Mask，两个为 UV，三个为 RGB，四个为 RGBA）。

        尚未选择文件或文件暂时无法读取（正在上传、误选了视频）时，先提供一个家族类型的「图像」端口：
        模板中的连线接的即是该端口（家族类型可先接入任何接受二维数据的端口，data/types.py accepts），
        选择文件后通道数才确定。此函数不得抛出异常：每次检查连线都会调用（nodes/applies.py output_ports）。"""
        try:
            found = cls._layers(cls._source(params)) if params.get("path") else {}
        except (OSError, ValueError, StopIteration):  # Invalid / GraphError 均为 ValueError
            found = {}
        if not found:  # 字节尚未上传：端口按申报时从文件头读出的图层生成（见 `_declared`）
            # 申报结果同样按文件中的第一个视角读出（`transfer/uploads.py describe_head`），
            # 与字节上传完成后打开文件得到的端口一致；否则立体文件在申报时两只眼混在一起产生双倍端口，
            # 上传完成后又变回一半，端口在使用者面前变化却没有任何提示。
            found = (_declared(params) or {}).get("layers") or {}
        if not found:
            return (Port("image", "image", "图像"),)
        return tuple(Port(name, f"image.{len(picked)}" if 1 <= len(picked) <= 4 else "image", label)
                     for name, label, _layer, picked in _outputs_of(found))

    @classmethod
    def upload_channels(cls, params, needed):
        """已连线的端口 → 文件中需上传的通道（`NodeDef.upload_channels`）：只上传用到的通道。
        端口与图层的对应关系只由 `_outputs_of` 确定，此处据此查找：所需端口属于哪一层，就上传该层的全部通道
        （Cryptomatte 的 id、数值图的有效区域 A 通道都在层内）。文件是否为 EXR、所需是否已是全部通道、
        立体文件名称如何对应，由 `transfer/uploads.py channel_plan` 判断，其返回 None 时上传完整文件。
        只按申报结果计算：字节尚未上传时只有申报结果可用。"""
        from ..services import services

        said = _declared(params)
        found = (said or {}).get("layers") or {}
        if not found or not needed:
            return None
        wanted: list[str] = []
        for name, _label, layer, _picked in _outputs_of(found):
            if name in needed:
                wanted += [c for c in found[layer]["channels"] if c not in wanted]
        if not wanted:
            return None
        return services().plan.channel_plan(str(params.get("path") or ""), wanted)

    @classmethod
    def choices(cls, params, inputs):
        """所选文件的「色彩空间」初始值，按格式确定（io/color.py colorspace_for_file：EXR 为 ACEScg，
        PNG / JPG 为 sRGB，视频为 Rec.709），由网页写入参数；未选择文件时为 "先选择文件"。"""
        from ...io.color import load_config

        if not params.get("path"):
            return {"colorspace": {"options": [], "empty": "先选择文件"}}
        try:
            first = next(iter(cls._source(params).files().values()))
        except (OSError, ValueError, StopIteration):
            # 字节尚未上传：色彩空间按文件名确定（OCIO 文件规则只依据名称，无需打开文件）
            said = _declared(params)
            if not said:
                return {"colorspace": {"options": [], "empty": "先选择文件"}}
            return {"colorspace": {"options": [], "default": load_config().colorspace_for_file(said.get("of", ""))}}
        return {"colorspace": {"options": [], "default": load_config().colorspace_for_file(str(first))}}

    @classmethod
    def fingerprint_params(cls, params):
        """「色彩空间」留空即取按格式确定的值（choices 提供给页面的 default）：填写与否对应同一个指纹。"""
        if params.get("colorspace"):
            return params
        guessed = (cls.choices(params, {}).get("colorspace") or {}).get("default")
        return {**params, "colorspace": guessed} if guessed else params

    @classmethod
    def foresee(cls, params, info):
        """计算前即可从文件得知的提示：序列在其输出帧范围内缺帧（N-READ-GAP：查看器在每个缺口处保持前一帧）；
        作为分割使用的 Cryptomatte 层，其对象名存放在旁边未随之上传的文件中（N-READ-NOMANIFEST：各对象仍按 id 区分）。"""
        if not info.frames:
            return []
        try:
            src = cls._source(params)
            found = _layers_of(str(next(iter(src.files().values()))), cls._view(src))
        except (OSError, ValueError, StopIteration):
            return []
        said = [Msg("N-READ-NOMANIFEST", file=Path(src.display).name, layer=layer, manifest=info["missing_manifest"])
                for layer, info in found.items() if "missing_manifest" in info]
        missing = [f for f in src.missing() if info.frames[0] <= f <= info.frames[-1]] if src.kind == "sequence" else []
        if not missing:
            return said
        from ...io.sequence import format_frame_range

        return [Msg("N-READ-GAP", file=Path(src.display).name, count=len(missing), frames=format_frame_range(missing)), *said]

    @classmethod
    def info(cls, params, inputs):
        """该段包含的帧和画面尺寸。

        字节尚未上传时也必须能够回答，否则会被判为 `E-COOK-FAILED`「上传的文件已经不在服务器上了」，
        节点报错、下游全部「已跳过」，而使用者并无操作错误，素材仍在其本机。
        帧号由申报的文件名清单计算，画面尺寸从保存的文件头读取（`transfer/uploads.py _declared_frames`）。"""
        try:
            src = cls._source(params)
        except (OSError, ValueError):
            said = _declared(params)
            if not said or not said.get("frames") or not said.get("size"):
                raise
            w, h = said["size"]
            return Info(tuple(said["frames"]), w, h, still=len(said["frames"]) <= 1)
        cls._layers(src)
        return Info(src.frames, src.width, src.height, still=src.kind == "still")

    @classmethod
    def cook(cls, ctx):
        from ...data import layers
        from ...data.payloads import VALIDITY, ExrWriter, ingest_picture
        from ...data.windows import Window
        from ...io import images
        from ...io.color import is_working, load_config, to_working_picture, working_space

        p = ctx.params
        src = cls._source(p)
        found = cls._layers(src)
        view = cls._view(src)  # 立体文件：每一层都从其第一个视角中读取
        files = {f: path for f, path in src.files().items() if f in ctx.frames}
        first = next(iter(files.values()))
        w, h = images.image_size(first)
        # 文件的所有图层在同一区域内读取：取各帧数据窗口的并集，使画面框外的像素不被丢弃，
        # 且画面的 alpha 与颜色来自相同的像素。
        window = Window.of_files(files.values(), w, h)
        cfg = load_config()
        guessed = cfg.colorspace_for_file(str(first))  # 按格式确定，与网页写入「色彩空间」的值相同
        out = {}
        for name, _label, layer, picked in _outputs_of(found):
            if name not in ctx.wanted:  # 未被连线的图层不读取
                continue
            info = found[layer]
            # 色彩空间优先取此处的设置，其次取 Lab2Shot 写出的 EXR 对画面的声明，最后按格式确定。
            # 此后画面均处于工作色彩空间（io/color.py）：已处于该空间的文件直接引用，场景参考空间（ACEScg、log）
            # 的文件在此转换一次，下游不再接触 `cs`。
            cs = p["colorspace"] or info.get("colorspace") or guessed
            crypto = {k: info[k] for k in ("manifest", "classes")} if "manifest" in info else None
            whole = _short_names(info["channels"])
            # 完整彩色画面且未挑选通道时，文件原样作为结果，包括 alpha（不重新写出）。
            # 必须同时满足三个前提：
            #  1. 三或四个通道：灰度 PNG 或只有 R G 的 ST-map EXR 的端口为 image.1 / image.2，
            #     原样直通会使数据包变为 image.3，与端口不一致；
            #  2. 该层即整个文件：多层渲染 EXR 的 rgba 层只是其中四个通道，原样直通会交出
            #     二十余个通道的完整文件，下游按数据包声明的四个通道读取时会直接出错（而非被拒绝）；
            #  3. 不是立体文件的某一只眼，也不是 Cryptomatte。
            one_layer = len(found) == 1 and len(info["channels"]) == len(images.channel_names(str(first)))
            if layer == "rgba" and picked == whole and len(whole) in (3, 4) and one_layer and view is None and crypto is None:
                alpha = any(layers.role(c) == "A" for c in info["channels"])
                packet = ingest_picture(ctx.outputs[name], files, w, h, cs, alpha, window, each=lambda items: ctx.each(items, "转到工作空间"))
            else:
                # 文件自身记录的信息（本项目写出的 EXR 的 lab2shot:layers 头）原样保留：这些来自文件本身，
                # 而非使用者的指定，包括尺度、坐标系、投影方式、方向以及置信度的来源。
                said = {k: info[k] for k in ("scale", "space", "projection", "direction", "model") if k in info}
                channels = list(layers.crypto_rank(info["channels"])) if crypto else layers.channels_named(info["channels"], picked)
                count = 1 if crypto else len([c for c in channels if layers.role(c) != VALIDITY])
                picture = count >= 3 and not crypto and info.get("values") is not True
                # 数值图一律带有效区域（Houdini 的 Z 中天空写为 inf），否则下游无法得知读出的 inf 和 NaN 的含义
                keeps = crypto is None and not picture
                if info.get("validity"):  # 本项目写出的多层 EXR：最后的 A 通道即有效区域，读取时需一并带上
                    channels = channels + layers.channels_named(info["channels"], ["A"])
                writer = ExrWriter(ctx.outputs[name], count, validity=keeps,
                                   colorspace=working_space(cfg) if picture else None, window=window,
                                   **({"alpha": True} if picture and count == 4 else {}), **said)
                ingest = picture and not is_working(cs, cfg)  # 场景参考的画面层：读取时转换
                labels: set[int] = set()
                for f, path in ctx.each(files.items(), f"图层 {layer}"):
                    data, valid = layers.read(path, channels, crypto, view, box=window.data,
                                              says_valid=bool(info.get("validity")))
                    writer.add(f, to_working_picture(data, cfg, cs) if ingest else data, valid)
                    if crypto is not None:
                        labels.update(int(v) for v in np.unique(data) if v > 0)
                packet = writer.packet()
                if crypto is not None:  # 类别表：取文件自带的（Cryptomatte），否则按找到的 id 各建一项
                    classes = info.get("classes") or [{"index": v, "name": f"id {v}"} for v in sorted(labels)]
                    packet.meta.update(classes=classes, range=[0.0, float(max([c["index"] for c in classes], default=1))])
            if src.kind == "still":
                packet.meta["still"] = True
            packet.meta.update(cls.said_shot(p))  # 拍摄内容在此处声明一次
            out[name] = packet
        return out


class ReadPicture(ReadSequence):
    """「读取图片」：读取单张图而非序列。部分算法本身针对单帧，接入序列会计算很久，因此单独提供读取单帧的节点。

    它与「读取序列」读取同一类文件，使用同一段代码（EXR 的图层同样逐层输出），区别只有两点：
    文件对话框只允许选择一张；所选文件为序列时立即拒绝，并说明帧数及应使用的节点。
    HDRI 光照探针、面部去光照、图像对位的参考图、在参考帧上绘制的修补均使用本节点。"""

    id = "core.read_picture"
    no_file = "没有选择图像"

    class Params(ReadSequence.Params):
        path: str = P("", label="文件", widget="file", group="文件", accept=sorted(IMAGE_EXTS),
                      help="从你的电脑选一张图：HDRI、照片、参考图、在参考帧上画好的修补都行。"
                           "选中的是一段序列（名字里带帧号的一批文件）时会拒绝，并告诉你共几帧——那种要用「读取序列」。"
                           "文件会上传到服务器，同样的文件只传一次")

    @classmethod
    def _source(cls, params):
        src = super()._source(params)
        files = src.files()
        if len(files) > 1:  # 序列：直接拒绝，不静默只读取第一帧
            raise Invalid(Msg("E-READ-NOTONEPICTURE", count=len(files), file=Path(src.display).name))
        return src


class SequenceEntry(NodeParams):
    """「读取多条序列」在文件夹中找到的一条序列：对应哪条序列、在列表中的名称以及是否读取。`pattern` 是标识该行
    对应序列的唯一依据；`frames` 仅记录列目录时找到的帧，因此镜头后续渲染了更多帧时仍是同一行。"""

    pattern: str = P(..., label="序列", widget="fixed", help="文件夹里认出的这一段序列，帧号写成 ####")
    frames: str = P("", label="帧", widget="fixed", help="列出这个文件夹时这一段有哪些帧、共几帧；只是说明，换了帧数还是同一段")
    name: str = P("", label="名字", help="这一条在列表里的名字：写出的层级名、交付的文件夹名。默认用序列名（在子文件夹里的用子文件夹名）")
    use: bool = P(True, label="读", help="勾上的才读进来。用不到的取消勾选，计算更快")


class ReadSequences(ReadsFile, NodeDef):
    id = "core.read_sequences"
    frame_source = True
    no_file = "没有选择文件夹"
    category = "read_plate"
    picture = ""
    outputs = (Port("list", "image.3[]", "列表"),)

    class Params(NodeParams):
        folder: str = P("", label="文件夹", widget="sequence", group="文件", accept=sorted(IMAGE_EXTS),
                      help="从你的电脑选一个文件夹：里面的每一段序列图都是列表里的一条，子文件夹里的也认得（shots/sh010/plate.####.exr）。"
                           "文件会上传到服务器，同样的文件只传一次")
        colorspace: str | None = colorspace_param(choices_from=("folder",))
        sequences: list[SequenceEntry] = P(
            [], label="序列", widget="table", group="序列", derived_from=("folder",), validate_default=True,
            help="文件夹里认出的每一段序列，勾上的读进来。选了文件夹自动列出")

    @classmethod
    def path(cls, params) -> Path:
        """使用者选择的文件夹（ReadsFile 读取「path」参数；本节点使用自己的「folder」参数，
        以保证同一参数名在各处对应同一标签）。"""
        from ...nodes.services import services

        if not params.get("folder"):
            raise ValueError(cls.no_file)
        return services().plan.upload(params["folder"])

    @classmethod
    def _found(cls, params) -> list[tuple[str, object]]:
        """所选文件夹中的全部序列 (名称, 序列)：包括文件夹本身及其下一级子文件夹（shots/sh010/plate.####.exr
        为一个镜头，以子文件夹名作为名称）。同一文件夹内较长者在前。"""
        from ...io.sequence import find_sequences

        folder = cls.path(params)
        if not folder.is_dir():
            raise Invalid(Msg("E-READ-NOTFOLDER", path=folder.name))
        out = []
        for where in [folder, *sorted(p for p in folder.iterdir() if p.is_dir())]:
            found = find_sequences(where)
            for seq in found:
                stem = seq.head.rstrip("._- ") or seq.tail.lstrip(".")
                out.append((where.name if where is not folder and len(found) == 1 else stem, seq))
        return out

    @classmethod
    def _chosen(cls, params) -> list[tuple[str, object]]:
        """表中已勾选的序列及其名称（文件夹中已不存在的序列对应的条目被忽略：以文件夹的实际内容为准）。

        条目只在「序列」列中标识其对应的序列，如 sh010.####.png。列目录时找到的内容（帧及帧数）单独成列，
        因此镜头后续渲染了更多帧时仍是同一行；若关键列同时包含帧数，出现新帧后便无法再匹配。"""
        by_pattern = {seq.name: (name, seq) for name, seq in cls._found(params)}
        out = []
        for e in params["sequences"]:
            found = by_pattern.get(e["pattern"])
            if e["use"] and found:
                name, seq = found
                out.append((e["name"].strip() or name, seq))
        return out

    @classmethod
    def derive(cls, params):
        """文件夹中的全部序列，均为勾选状态：选择文件夹后表格显示的内容。"""
        if not params.get("folder"):
            return {"sequences": []}
        return {"sequences": [{"pattern": seq.name, "frames": seq.found(), "name": name, "use": True}
                              for name, seq in cls._found(params)]}

    @classmethod
    def choices(cls, params, inputs):
        from ...io.color import load_config

        if not params.get("folder"):
            return {"colorspace": {"options": [], "empty": "先选择文件夹"}}
        first = next((seq.path(seq.first) for _, seq in cls._chosen(params)), None)
        if not first:
            return {"colorspace": {"options": [], "empty": "先选择文件夹"}}
        return {"colorspace": {"options": [], "default": load_config().colorspace_for_file(str(first))}}

    @classmethod
    def fingerprint_params(cls, params):
        """「色彩空间」留空即取按格式确定的值（choices 提供给页面的 default）：填写与否对应同一个指纹。"""
        if params.get("colorspace"):
            return params
        guessed = (cls.choices(params, {}).get("colorspace") or {}).get("default")
        return {**params, "colorspace": guessed} if guessed else params

    @classmethod
    def source_identity(cls, params):
        """所读取的每条序列的每一帧，按内容识别（nodes/services.py PlanEnv.content_id）：文件夹中替换了文件即为
        另一份结果。不按大小和修改时间识别，因为原地替换且修改时间不变的情况无法识别。"""
        from ...nodes.services import services  # 在函数内导入，以避免循环导入

        cls.path(params)  # 引用仍存在于本服务器
        content = services().plan.content_id
        return [[seq.pattern, [[f, content(seq.path(f))] for f in seq.frames]]
                for _, seq in cls._chosen(params)]

    @classmethod
    def info(cls, params, inputs):
        """输出内容：所读取的全部序列的帧（每个条目在 cook 中保留各自的帧），尺寸取第一条序列。"""
        chosen = cls._chosen(params)
        if not chosen:
            raise Invalid(Msg("E-READ-NOSEQUENCES", path=cls.path(params).name))
        from ...io import images

        first = chosen[0][1]
        w, h = images.image_size(first.path(first.first))
        return Info(tuple(sorted({f for _, seq in chosen for f in seq.frames})), w, h)

    @classmethod
    def cook(cls, ctx):
        from ...data import layers
        from ...data.packet import Packet, item_fingerprint, items_meta, produce, valid
        from ...data.payloads import ingest_picture
        from ...data.windows import Window
        from ...io import images
        from ...io.color import load_config

        p = ctx.params
        want = set(ctx.frames)
        parts, outside = [], []
        for name, seq in cls._chosen(p):
            frames = [f for f in seq.frames if f in want]
            if not frames:  # 该序列的所有帧均在计算帧范围之外
                outside.append(name)
                continue
            files = {f: seq.path(f) for f in frames}
            from ...nodes.services import services

            content = services().plan.content_id  # 同样按内容识别，与 source_identity 使用同一处实现
            fp = item_fingerprint(cls.id, cls.version, [[f, str(path), content(path)] for f, path in files.items()],
                                  p["colorspace"] or "")

            def make(out: Path, files=files, first=files[frames[0]]) -> Packet:
                w, h = images.image_size(first)
                found = _layers_of(str(first))
                if "rgba" not in found:  # 只有数据层的文件：「读取序列」逐层输出这些数据层
                    raise Invalid(Msg("E-READ-NOPICTURE", file=first.name, layers=list(found)))
                alpha = any(layers.role(c) == "A" for c in found["rgba"]["channels"])
                cs = p["colorspace"] or found["rgba"].get("colorspace") or load_config().colorspace_for_file(str(first))
                return ingest_picture(out, files, w, h, cs, alpha, Window.of_files(files.values(), w, h)).commit(cls.id)  # 转换到工作色彩空间

            # 条目的素材被清除或改变时重新生成（data/packet.py check）；两个卡片同时读取同一条目时只处理一次（produce 持有该条目的锁）
            produce(fp, make)
            parts.append((name, fp))
        if outside:
            ctx.say("N-READ-OUTSIDE", node=ctx.label, count=len(outside), names=outside,
                    lo=ctx.frames[0] if ctx.frames else 0, hi=ctx.frames[-1] if ctx.frames else 0)
        return {"list": Packet(ctx.outputs["list"], ctx.output_types["list"], items_meta(parts))}


def _convert_frame(rgb, ocio_uri: str, colorspace: str, out: str) -> None:
    """工作进程：一帧视频 → 数据包中的显示用 sRGB PNG。"""
    from ...io import images
    from ...io.color import load_config, to_srgb8

    images.write_png(out, to_srgb8(rgb, load_config(ocio_uri), colorspace))  # SDR 视频原样输出；HDR 视频做一次色调映射


class ReadVideo(ReadsFile, NodeDef):
    id = "core.read_video"
    no_file = "没有选择视频"
    category = "read_plate"
    outputs = (Port("video", "video", "视频"),)

    class Params(NodeParams):
        path: str = P("", label="视频", help="从你的电脑选择 mov、mp4 等视频文件（也可以直接拖进来），会上传到服务器，同样的文件只传一次。手机竖拍会自动转正，HDR（HLG / PQ）视频会自动转成正常显示的画面", widget="file", group="文件", accept=sorted(VIDEO_EXTS))

    @classmethod
    def info(cls, params, inputs):
        count, fps, width, height = _video_facts(str(cls.path(params)))
        # 视频流自身的帧，从 0 计数；`video_fps` 为该文件自身的帧率，只由「视频转序列」读取
        return Info(tuple(range(count)), width, height, video_fps=fps)

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import video_packet

        path = cls.path(ctx.params)
        count, fps, width, height = _video_facts(str(path))
        return {"video": video_packet(ctx.outputs["video"], path, count, fps, width, height,
                                      **cls.said_shot(ctx.params))}  # 拍摄内容在此处声明一次


@lru_cache(maxsize=32)
def _video_facts(path: str) -> tuple[int, float, int, int]:
    """视频的 (帧数, fps, 宽, 高)；上传内容不会改变，因此每个文件只打开一次。

    帧率不属于数据：下游不会获得它（视频逐帧读取）。它仅用于把「区间（秒）」换算为该文件内的帧位置，
    并由 `lab2shot info` 输出。"""
    from ...io.sources import open_source

    src = open_source(path, 0)
    return len(src.frames), src.fps or DEFAULT_FPS, src.width, src.height


SCALES = {"full": 1.0, "half": 0.5, "quarter": 0.25}


class VideoToSequence(NodeDef):
    id = "core.video_to_sequence"
    version = 2  # 图像数据包现在总会注明是否带 alpha
    # 开销：解码整段视频并写出每一帧，使用多个进程
    cost = Cost(lane=HEAVY)
    frame_source = True
    on_node = ("range_sec", "step", "downscale")
    category = "read_plate"
    inputs = (Port("video", "video", "视频"),)
    # 视频是其画面输入：拍摄内容取自视频（data/contracts.py settle），尺寸则不是（由「分辨率」缩放），
    # 因此端口声明由节点决定画面，info() 在计算前报告尺寸。
    picture = "video"
    outputs = (Port("image", "image.3", "RGB", shape=Shape(window="node")),)

    class Params(NodeParams):
        range_sec: str | None = P(None, label="区间", unit="秒", help="视频里哪一段是这个镜头，单位秒：如 0-10 取前 10 秒，5-8 取第 5 到 8 秒；留空取整段。起始帧号从这一段的开头数。只想先算镜头里的几帧，用计算按钮旁的帧范围", group="区间", placeholder="整段")
        step: Literal[1, 2, 3] = P(
            1, label="隔帧", group="区间",
            option_labels={"1": "每帧", "2": "每 2 帧取 1", "3": "每 3 帧取 1"},
            help="高帧率视频（如 60fps）可以隔帧取，后面的计算快一倍；帧号保持连续（一秒里的帧数少一半，写文件时在输出设置里填帧率）",
        )
        start_frame: int = P(1001, label="起始帧号", help="转出的第一帧用哪个帧号，影视习惯从 1001 开始。写进 USD 和序列图文件名的帧号都从这里数", group="区间")
        downscale: Literal["full", "half", "quarter"] = P(
            "full", label="分辨率", group="画面",
            option_labels={"full": "原始", "half": "一半", "quarter": "四分之一"},
            help="缩小后再往下传：4K 素材先用一半分辨率试参数更快；最终结果建议用原始分辨率",
        )

    @staticmethod
    def _indices(p: dict, count: int, fps: float) -> list[int]:
        """镜头所用的视频帧（从 0 计数）：镜头帧 start_frame + k 对应视频帧 [k]。"""
        import math

        lo, hi = 0, count - 1
        if p["range_sec"]:
            a, _, b = p["range_sec"].replace(" ", "").partition("-")
            try:
                start, end = float(a or 0), float(b) if b else None
            except ValueError:
                raise Invalid(Msg("E-VIDEO-RANGEFORMAT", text=p["range_sec"])) from None
            lo = max(0, int(math.floor(start * fps)))
            if end is not None:
                hi = min(hi, int(math.ceil(end * fps)) - 1)
            if hi < lo:
                raise Invalid(Msg("E-VIDEO-RANGEOUT", text=p["range_sec"], seconds=count / fps))
        return list(range(lo, hi + 1, p["step"]))

    @classmethod
    def info(cls, params, inputs):
        video = inputs["video"][0]
        shot = len(cls._indices(params, len(video.frames), video.video_fps or DEFAULT_FPS))
        scale = SCALES[params["downscale"]]
        w, h = (max(2, int(round(n * scale / 2)) * 2) for n in (video.width, video.height))  # 与解码器的缩放方式一致
        return Info(tuple(range(params["start_frame"], params["start_frame"] + shot)), w, h)

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import image_packet
        from ...io.parallel import process_pool
        from ...io import images
        from ...io.color import load_config
        from ...io.sources import open_source

        p = ctx.params
        video = ctx.input("video")
        path = file_path(video, video.meta["path"])  # 相对于数据包的位置（旧数据包为绝对路径，file_path 两种均可识别）
        src = open_source(path, 0)  # 视频帧从 0 开始编号；序列保留图片自身的编号
        fps = src.fps or DEFAULT_FPS
        # 源的自身帧号 → 镜头帧号，仅限计算帧范围选中的帧（按整个镜头编号）。_indices 统计的是在源中的位置，
        # 与帧号不同：000001.jpg… 这样的序列从 1 开始编号，而「读取视频」也接受序列，因此按位置索引 src.frames
        # 可以避免对从 1 开始的序列请求第 0 帧。
        picked = [src.frames[i] for i in cls._indices(p, len(src.frames), fps)]
        numbers = {n: p["start_frame"] + k for k, n in enumerate(picked) if p["start_frame"] + k in ctx.frames}
        cfg = load_config()
        cs = src.colorspace or cfg.colorspace_for_file(str(path))
        out = ctx.outputs["image"]
        files: dict[int, Path] = {}
        # 解码很快；色彩转换和 PNG 写出在并行进程中执行
        with process_pool() as pool:
            pending = []
            for k, (i, rgb) in enumerate(src.iter_frames(list(numbers), scale=SCALES[p["downscale"]])):
                frame = numbers[i]
                files[frame] = out / f"frame.{frame}.png"
                pending.append(pool.submit(_convert_frame, rgb, cfg.uri, cs, str(files[frame])))
                while len(pending) > 16:  # 限制内存占用
                    pending.pop(0).result()
                ctx.progress(k + 1, len(numbers), f"帧 {frame}")
            for f in pending:
                f.result()
        h, w = images.read_rgb(next(iter(files.values()))).shape[:2]
        from ...io.color import working_space

        packet = image_packet(out, files, w, h, working_space(cfg))  # PNG 帧：原样作为工作色彩空间
        packet.meta.update(source_colorspace=cs)  # 镜头自身的属性取自视频（settle）
        return {"image": packet}




class Constant(NodeDef):
    """不使用任何素材的纯色画面，对应 Nuke 的 Constant 节点。"""

    id = "core.constant"
    category = "read_plate"
    frame_source = True
    on_node = ("width", "height", "frames")
    outputs = (Port("image", "image.3", "RGB"),)

    class Params(NodeParams):
        width: int = P(DEFAULT_WIDTH, label="画面宽度", unit="px", gt=0, le=4096, group="画面",
                       help="这段画面多宽，单位像素。画布只是给你画东西的底，画大画小不改变结果，够看清就行")
        height: int = P(DEFAULT_HEIGHT, label="画面高度", unit="px", gt=0, le=4096, group="画面",
                        help="这段画面多高，单位像素。和「画面宽度」一起决定这张空画布的尺寸")
        first: int = P(1001, label="首帧", group="帧", help="第一帧的帧号，和整个项目一样用原始帧号")
        # 帧数只提供若干档位，不允许任意填写：每一帧都会实际写出一张 EXR，一万帧将占用数十 GB。
        # 最长 192 帧，恰好在「Sketch2Anim 动作生成」可生成的范围内（9.8 秒）。
        frames: Literal[24, 48, 96, 144, 192] = P(
            96, label="帧数", group="帧",
            option_labels={"24": "24 帧 · 1 秒", "48": "48 帧 · 2 秒", "96": "96 帧 · 4 秒",
                           "144": "144 帧 · 6 秒", "192": "192 帧 · 8 秒"},
            help="一共几帧。后面那个秒数是按电影的 24 帧每秒算的。"
                 "「Sketch2Anim 动作生成」生成的动作就是这么长")
        colour: tuple[float, float, float] = P((0.46, 0.46, 0.46), label="颜色", widget="vec3", parts=("R", "G", "B"), group="画面",
                                               help="画面的颜色，0–1 的 sRGB 值，和屏幕上看到的一样（0.46 是中灰，线性的 0.18）。画火柴人、画遮罩时中灰最好看")

    @classmethod
    def info(cls, params, inputs):
        first = int(params["first"])
        return Info(tuple(range(first, first + int(params["frames"]))), int(params["width"]), int(params["height"]))

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import ExrWriter
        from ...io.color import working_space

        p = ctx.params
        w, h = int(p["width"]), int(p["height"])
        one = np.empty((h, w, 3), np.float32)
        one[:] = np.asarray(p["colour"], np.float32)
        # 画面按工作色彩空间写出（sRGB 值即工作色彩空间的值，io/color.py），不是数值图
        writer = ExrWriter(ctx.outputs["image"], 3, colorspace=working_space())
        for frame in ctx.frames:
            writer.add(frame, one)
        return {"image": writer.packet()}


NODES = (ReadSequence, ReadPicture, ReadSequences, ReadVideo, VideoToSequence, Constant)
