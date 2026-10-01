"""图像处理：STMap、遮罩转 Alpha、合并通道、图像合成、FrameHold、拆网格图、Crop、方向场。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ..port import EITHER
from ..kit.ports import rgb_port
from ...errors import Invalid
from ...messages import Msg
from ...data.contracts import Shape, warped_by
from ..applies import Param
from ..base import Info, NodeDef, NodeParams, P, Port
from ..expects import SameShot


class StmapWarp(NodeDef):
    id = "core.stmap_warp"
    version = 4  # 结果变化时递增，work/ 中的旧结果随之不再命中缓存（engine/cook.py）
    category = "img_warp"
    picture = "src"
    inputs = (Port("src", "image", "源", alpha=True, data=EITHER), Port("stmap", "image.2", "ST-map"))
    # 按一张自身不了解的映射图移动像素：结果位于映射图的窗口上，数据自带的全部描述随之传递，本节点不声明任何自身属性。
    # 本节点没有参数，也不添加任何标记：与 Nuke 的 STMap 一样只是按坐标查表变形，画面是否有畸变由使用者掌握，
    # 不维护镜头状态。lens="keep"：生成该 ST-map 所用的镜头由契约从 ST-map 复制到结果上（端口跟随 stmap 输入），
    # 因此在去畸变画面上解出的相机仍可交付回畸变状态。
    outputs = (Port("image", "image", "结果", type_from="input:src",
                    shape=replace(warped_by("stmap"), lens="keep")),)

    @staticmethod
    def carried(src_still: bool, src_frames, st_still: bool, st_frames) -> bool:
        """源图是否为一张需要沿用到 ST-map 序列每一帧上的画面（只绘制一次的一帧：静帧，或只有一帧的序列，
        如在 Nuke 中绘制的带编号文件被读取时的情况）。"""
        return not st_still and (src_still or (len(src_frames) == 1 and len(st_frames) > 1))

    @classmethod
    def info(cls, params, inputs):
        """源图的各帧（由 ST-map 序列携带的单张画面时取映射图的帧），尺寸为 ST-map 的尺寸（带扩边生成的映射图
        大于原画面）。"""
        src, st = inputs["src"][0], inputs["stmap"][0]
        frames = st if cls.carried(src.still, src.frames, st.still, st.frames) else src
        return replace(frames, width=st.width, height=st.height)

    @classmethod
    def cook(cls, ctx):
        from ...data.contracts import own_meta
        from ...data.lens_models import apply_stmap
        from ...data.payloads import (ExrWriter, file_at, has_alpha, has_validity, is_data, is_labels, read_map,
                                      read_picture, window_of)
        from ...data.types import channels_of

        src, st = ctx.input("src"), ctx.input("stmap")
        # 本节点接受任意二维数据，因此「画面还是数值图」「是否为编号图」由数据自带的信息决定，不按通道数推测
        picture, labels = not is_data(src), is_labels(src)
        keeps_alpha = has_validity(src)  # 源图自带「有效区域」通道时，结果同样保留
        # 结果位于 ST-map 的窗口上（其画面框，以及存在时的画布）
        writer = ExrWriter(ctx.outputs["image"], channels_of(src.type), validity=keeps_alpha,
                           colorspace=src.meta["colorspace"] if picture else None,
                           value_range=tuple(src.meta["range"]) if not picture else None,
                           window=window_of(st), **own_meta(src),
                           **({"alpha": True} if picture and has_alpha(src) else {}))
        src_window = window_of(src)
        src_box, st_box = src_window.data, window_of(st).data

        carried = cls.carried(bool(src.meta.get("still")), src.meta["frames"], bool(st.meta.get("still")), st.meta["frames"])
        frames = list(st.meta["frames"] if carried else src.meta["frames"])

        def path_at(packet, frame, what):
            path = file_at(packet, packet.meta["frames"][0] if carried and packet is src else frame)
            if path is None:
                raise Invalid(Msg("E-STMAP-NOFRAME", source=what, frame=frame, first=packet.meta["frames"][0], last=packet.meta["frames"][-1]))
            return path

        def read_src(path):
            if picture:  # 绘制块的 alpha 随之变形（预乘：按原样采样）
                return read_picture(src, path, src_box), None
            return read_map(path, src_box)

        # 每帧读取的两个文件先在主线程中确定；整段只有一张的（被沿用的源图、静帧 ST-map）在进入循环前读取一次。
        # 不使用共享缓存字典：多帧并发计算时线程之间会互相清除。
        jobs = [(f, path_at(src, f, "源"), path_at(st, f, "ST-map ")) for f in frames]
        one_src = read_src(jobs[0][1]) if jobs and len({j[1] for j in jobs}) == 1 else None
        one_st = read_map(jobs[0][2], st_box)[0] if jobs and len({j[2] for j in jobs}) == 1 else None

        def warp(job):
            """一帧的全部工作：读取源图、读取 ST-map、重采样、写出。只涉及本帧，多帧可并发计算（ctx.each_done）。"""
            frame, src_path, st_path = job
            values, alpha = read_src(src_path) if one_src is None else one_src
            stmap = read_map(st_path, st_box)[0] if one_st is None else one_st
            # Nuke 的 STMap：ST-map 的值按源图画幅归一化。源图带扩边时（去畸变画面、带 overscan 的 CG 渲染），
            # 读入的是完整数据窗口，画幅左上角在其中的位置即 box.offset；加畸变 ST-map 指向画布上的位置，
            # 因此需要该偏移才能正确取样（无扩边时 offset 为 (0,0)）。
            # 编号图（带类别表）取最近邻，在编号之间插值会产生不存在的编号；其余均为双线性，与 Nuke 的 STMap 一致。
            out, valid = apply_stmap(values, stmap, src_window.plate, src_window.offset,
                                     alpha=alpha if keeps_alpha else None, nearest=labels)
            # 该帧没有任何像素取到值时，额外测量该 ST-map 的指向范围（仅在结果为空时计算）
            span = None if valid.any() else _off_the_source(stmap, src_window.plate, src_window.offset, values.shape[:2])
            writer.add(frame, out, valid)
            return span

        off = list(ctx.each_done(jobs, warp))  # 各帧的说明按帧号顺序返回
        # 整段每一帧均为空，且每一帧都指向源图之外时，在输出空图之前说明原因。
        # 取样落在源图内部却为空（源图该区域本身无值）的情况不在此处说明，那属于上游的问题。
        if off and all(span is not None for span in off):
            plate_w, plate_h = src_window.plate
            ctx.say("N-STMAP-NOTHING", port="stmap", low=min(span[0] for span in off),
                    high=max(span[1] for span in off), width=plate_w, height=plate_h)
        packet = writer.packet()
        if src.meta.get("still") and st.meta.get("still"):
            packet.meta["still"] = True
        # 生成该 ST-map 所用的镜头由契约从 ST-map 复制（lens="keep"，见输出端口部分）；本节点自身不声明任何内容。
        return {"image": packet}


def _off_the_source(stmap: np.ndarray, plate: tuple[int, int], offset: tuple[int, int],
                    size: tuple[int, int]) -> tuple[float, float] | None:
    """「STMap」某帧没有任何像素取到值时调用：检查该 ST-map 指向的位置是否全部落在源图（[h, w] 像素范围，
    即 data/maps.py sample_band 使用的范围）之外。若是，返回该图自身的取值范围：ST-map 的取值是按源图画幅
    归一化的 0–1 坐标，超出 0–1 的数值即为原因，应写入提示使用者的消息。只要有一个取样落在源图内即返回 None：
    此时结果为空并非 ST-map 指向错误，而是源图该区域本身无值，应由上游节点说明。"""
    from ...data.lens_models import INVALID, decode

    h, w = size
    pos = decode(stmap, plate, offset)
    x, y = pos[..., 0], pos[..., 1]
    if (np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x <= w) & (y >= 0) & (y <= h)).any():
        return None
    said = stmap[(stmap[..., 0] != INVALID) | (stmap[..., 1] != INVALID)]  # 表示「此处无值」的哨兵值不计入
    return (float(said.min()), float(said.max())) if said.size else None


class AlphaMerge(NodeDef):
    id = "core.alpha_merge"
    category = "img_channel"
    inputs = (rgb_port(), Port("alpha", "image.1", "Alpha", expects=(SameShot("image"),)))
    outputs = (Port("image", "image.4", "RGBA"),)

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import ExrWriter, file_at, has_alpha, read_picture, window_of
        from ...data.maps import map_at, same_size
        from ...io import images
        from ...io.color import load_config, to_working_picture, working_space

        image, matte = ctx.input("image"), ctx.input("alpha")
        same_size({"图像": image, "Alpha": matte})
        cfg = load_config()
        window = window_of(image)
        # 画面按其声明的色彩空间转到工作色彩空间（读取节点的输出本就在工作空间中），本节点只附加 alpha。
        out = ExrWriter(ctx.outputs["image"], 4, half=True, colorspace=working_space(cfg), alpha=True,
                        window=window)
        def merge(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）；返回该帧是否缺少遮罩
            picture = to_working_picture(read_picture(image, file_at(image, f)), cfg, image.meta["colorspace"])
            rgb = np.ascontiguousarray(images.unpremultiply(picture) if has_alpha(image) else picture)  # 非预乘颜色
            got = map_at(matte, f, window.data)  # 静态遮罩：作用于每一帧，使用画面自身的窗口
            a = np.zeros(rgb.shape[:2], np.float32) if got is None else np.clip(got[0][..., 0], 0.0, 1.0)
            out.add(f, np.concatenate([rgb * a[..., None], a[..., None]], axis=-1))
            return got is None

        frames = image.meta["frames"]
        lacking = [f for f, gap in zip(frames, ctx.each_done(frames, merge)) if gap]
        if lacking:
            ctx.say("N-ALPHA-GAP", count=len(lacking), first=lacking[0])
        packet = out.packet()
        if image.meta.get("still"):
            packet.meta["still"] = True
        return {"image": packet}


class ChannelMerge(NodeDef):
    """三张单通道 Mask → 一张 RGB 图像。读取序列读取灰度图输出 image.1，无法接入需要 RGB 的端口；
    类型转换由一个显式节点完成（三个端口接同一张灰度图即可），读取序列和 COLMAP 不为此修改端口。"""

    id = "core.channel_merge"
    category = "img_channel"
    inputs = (Port("r", "image.1", "R"),
              Port("g", "image.1", "G", expects=(SameShot("r"),)),
              Port("b", "image.1", "B", expects=(SameShot("r"),)))
    outputs = (Port("image", "image.3", "RGB"),)
    picture = "r"  # 结果的尺寸和窗口取自 R 输入（三张尺寸必须一致，由 same_size 检查）

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import ExrWriter, window_of
        from ...data.maps import map_at, same_size
        from ...io.color import working_space

        r, g, b = ctx.input("r"), ctx.input("g"), ctx.input("b")
        same_size({"R": r, "G": g, "B": b})
        window = window_of(r)
        out = ExrWriter(ctx.outputs["image"], 3, half=False, colorspace=working_space(), window=window)
        def merge(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）；返回该帧是否缺少某条通道
            planes = [map_at(p, f, window.data) for p in (r, g, b)]  # 静帧作用于每一帧，使用 R 自身的窗口
            if any(p is None for p in planes):
                return True
            out.add(f, np.stack([p[0][..., 0] for p in planes], axis=-1).astype(np.float32))
            return False

        frames = r.meta["frames"]
        lacking = [f for f, gap in zip(frames, ctx.each_done(frames, merge)) if gap]
        if lacking:
            ctx.say("N-CHANNELMERGE-GAP", count=len(lacking), first=lacking[0])
        packet = out.packet()
        if r.meta.get("still"):
            packet.meta["still"] = True
        return {"image": packet}


# 每种运算的中文名只定义一处：「图像合成」的选项名与提示消息（N-IMGMERGE-EMPTY）共用
MERGE_OPERATIONS = {"multiply": "留下", "stencil": "挡掉", "plus": "相加", "minus": "相减",
                    "min": "取小", "max": "取大", "over": "盖上（A over B）"}
OVER = Param("operation").one_of("over")  # 「盖上」用「底图」口，其余六种用「遮罩」口
MASKED = Param("operation").one_of(*(k for k in MERGE_OPERATIONS if k != "over"))


class ImageMerge(NodeDef):
    """两张图按一种运算合成为一张（Nuke 的 Merge）。

    让解算器只计算画面的一部分时，节点链为

        人物检测 → 「人物框转遮罩」→（需要羽化时再接「遮罩调整」）→ 「图像合成」相乘 → 解算器的「图像」端口

    遮挡在送入解算器之前完成，在节点图上可见。重建节点不各自带一个在模型计算之后擦除结果像素的「遮罩」端口
    （那样使用者无法看到，出现问题也难以排查）；本实现由所有解算器共用。
    """

    id = "core.image_merge"
    # 不使用单独的「Merge」作为名称：Merge 在 Nuke 中指 2D 合成，在 Houdini 中指 3D 合并，含义不同。
    # 3D 合并是「合成场景」（core.usd_pack），因此本 2D 节点的名称包含「图像」。
    category = "img_channel"
    on_node = ("operation",)
    # B 固定为单通道遮罩类型：若两个端口都声明为通用的「图像」，1 通道遮罩可能接入 3 通道端口，
    # 直到计算中途才报「合不到一起」。端口如实声明所接受的通道数，不兼容的连线在节点图上即无法连接。
    # 「盖上」（Nuke 的 A over B）的另一边是一张图而不是遮罩：单独一个「底图」口（任意画面，通道数随 A）。两个口二选一
    # （input_choice：至少接一个），哪一个可用由「运算」决定（applies：选「盖上」时「遮罩」变灰，反之「底图」变灰）。
    inputs = (Port("a", "image", "图像", data=EITHER, help="「盖上」时是盖在上面的那张（预乘 alpha 的 RGBA；没有 alpha 就整张盖住）"),
              Port("b", "image.1", "遮罩", optional=True, expects=(SameShot("a"),), applies=MASKED,
                   help="一条通道的遮罩，自动铺到图像的每条通道上"),
              Port("bg", "image", "底图", optional=True, data=EITHER, expects=(SameShot("a"),), applies=OVER,
                   help="「盖上」时垫在下面的那张：A 透明的地方露出它（比如 AllTracker 的贴片、RoMa 对齐的图合到原图上看）"))
    input_choice = (("b",), ("bg",))
    outputs = (Port("image", "image", "图像", type_from="input:a"),)
    # 算法定义不在此处，而在算法目录（lab2shot/ops/ops.toml）中：七种运算各占一条（image.merge.<运算>）。
    # 本节点只负责读取像素和写出结果。
    ops = tuple(f"image.merge.{k}" for k in MERGE_OPERATIONS)

    class Params(NodeParams):
        operation: Literal["multiply", "stencil", "plus", "minus", "min", "max", "over"] = P(
            "multiply", label="运算", group="合成", option_labels=MERGE_OPERATIONS)
        # 「盖上」的不透明度（Nuke Merge 的 mix）：A 没有 alpha（RoMa 对齐过来的整张图）时，调低就透出底图
        mix: float = P(1.0, label="不透明度", ge=0.0, le=1.0, group="合成", applies=OVER)

    @classmethod
    def fingerprint_params(cls, params):
        # 不透明度 1 与这个参数出现之前的「盖上」是同一个结果：不进指纹，加参数不让已有的缓存失效
        return {k: v for k, v in params.items() if not (k == "mix" and float(v) == 1.0)}

    @classmethod
    def cook(cls, ctx):
        from ...data.contracts import own_meta
        from ...data.payloads import (ExrWriter, file_at, has_validity, is_data, read_map, read_picture,
                                      window_of)
        from ...data.maps import map_at, same_size
        from ...data.types import channels_of
        from ...ops import run as run_op

        over = ctx.params["operation"] == "over"
        a, b = ctx.input("a"), ctx.input("bg" if over else "b")
        if b is None:  # 接的是另一个口（它此刻变灰）：说清这个运算要哪个口
            raise Invalid(Msg("E-IMGMERGE-OTHERPORT", operation=MERGE_OPERATIONS[ctx.params["operation"]],
                              wanted="底图" if over else "遮罩"))
        same_size({"A": a, "B": b})
        merge_op = f"image.merge.{ctx.params['operation']}"
        window = window_of(a)
        n = channels_of(a.type)
        if over and is_data(a):
            raise Invalid(Msg("E-IMGMERGE-OVERDATA"))
        # A 接受任意二维数据，而不仅是画面（端口类型为通用的「图像」）：深度图、法线图、ST-map、分割图、遮罩
        # 均可被遮挡。「画面还是数值图」由数据自带的信息决定，不按通道数推测（payloads.is_data，与「STMap」规则相同），
        # 三通道的 image.3 既可能是照片也可能是法线图。
        # 不能一律使用 read_picture（它总是返回 RGB / RGBA 三或四个通道）：单通道深度图或双通道 ST-map
        # 接入 A 时会在 E-EXR-CHANNELS 或广播时失败。
        picture, keeps_valid = not is_data(a), has_validity(a)
        out = ExrWriter(ctx.outputs["image"], n,
                        # 画面写为半精度（EXR 的常规做法）；数值图一律使用全精度：ST-map 的 0..1 需乘以画面宽度
                        # 作为坐标，半精度在 2K 宽度下误差约一个像素
                        half=picture, validity=keeps_valid,
                        # 数值图不写固定值域：此步骤实际改变了数值（相乘、相加、取小），
                        # 源图声明的最小值和最大值不再有效，由写出方重新测量
                        colorspace=a.meta.get("colorspace") if picture else None,
                        window=window, **own_meta(a))  # 深度尺度、法线坐标系、分割类别表原样传递
        def merge(f):
            """一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）。返回（该帧是否缺少 B，结果是否有内容）。"""
            path = file_at(a, f)
            if picture:
                pa, valid = np.asarray(read_picture(a, path), np.float32), None
            else:
                values, ok = read_map(path, window.data)
                pa, valid = np.asarray(values, np.float32), (ok if keeps_valid else None)
            if over:
                under = file_at(b, f)  # 静态底图（整段一张）垫在每一帧下
                if under is None:
                    out.add(f, pa, valid)
                    return True, bool(np.any(pa))
                pb = np.asarray(read_picture(b, under, window.data), np.float32)
                if pb.shape[-1] != pa.shape[-1]:  # 底图有无 alpha 与 A 不同：RGB 补一条不透明的 alpha，或去掉 alpha
                    pb = np.concatenate([pb, np.ones_like(pb[..., :1])], -1) if pb.shape[-1] < pa.shape[-1] else pb[..., :pa.shape[-1]]
                alpha = pa[..., 3:4] if pa.shape[-1] == 4 else np.ones_like(pa[..., :1])
                merged = run_op(merge_op, {"a": pa, "b": pb, "alpha": alpha, "mix": float(ctx.params["mix"])})["value"]
                out.add(f, merged, valid)
                return False, bool(np.any(merged))
            got = map_at(b, f, window.data)  # 静态 B（整段一张图）作用于每一帧
            if got is None:
                out.add(f, pa, valid)
                return True, bool(np.any(pa))
            pb = np.asarray(got[0], np.float32)[..., :1]  # 端口类型保证其为单通道（image.1）
            # 单通道遮罩广播到图像各通道由执行器完成
            merged = run_op(merge_op, {"a": pa, "b": pb})["value"]
            out.add(f, merged, valid)
            return False, bool(np.any(merged))

        frames = a.meta["frames"]
        found = list(ctx.each_done(frames, merge))
        lacking = [f for f, (gap, _) in zip(frames, found) if gap]
        anything = any(some for _, some in found)
        if lacking:  # B 缺少部分帧：这些帧原样输出并给出提示
            ctx.say("N-IMGMERGE-GAP", count=len(lacking), first=lacking[0])
        # 整段合成结果处处为 0 不属于错误（上游未检测到任何内容时即会如此，例如与没有天空的天空概率图相乘），
        # 但对全黑结果不加说明会使使用者难以判断问题出在哪一步。
        if not anything:
            ctx.say("N-IMGMERGE-EMPTY", operation=MERGE_OPERATIONS[ctx.params["operation"]])
        packet = out.packet()
        if a.meta.get("still") and b.meta.get("still"):
            packet.meta["still"] = True
        return {"image": packet}


class FrameHold(NodeDef):
    id = "core.frame_hold"
    version = 2  # 2：留空取首帧（原为镜头中间一帧）；参数没变，不加一的话留空时缓存的中间帧会被原样复用
    category = "img_adjust"
    inputs = (Port("image", "image", "图像", alpha=True, data=EITHER),)
    outputs = (Port("image", "image", "图像", type_from="input:image"),)

    class Params(NodeParams):
        frame: int | None = P(None, label="帧", placeholder="首帧")

    on_node = ("frame",)

    @classmethod
    def info(cls, params, inputs):
        """在计算之前声明只输出一帧。

        若不声明，默认会原样报告输入的帧（`NodeDef.info` → `Info.merge`），提交前的检查会看到整段帧数，
        只接受单帧的下游节点（如 LuxDiT 环境光探针）会被 `B-EXPECT-MANYFRAMES` 拦下，而该节点链本身是正确的，
        本节点正是用于选帧。

        选哪一帧与 `cook` 用同一条规则（`held`）；填写的帧号超出范围时不在此处判断，仍原样报告，由 `cook` 报出
        `E-FRAMEHOLD-RANGE`。"""
        got = Info.merge(inputs.get("image") or [])
        if got.still or not got.frames:
            return got
        frame = cls.held(params.get("frame"), got.frames)
        return replace(got, frames=(frame,)) if frame in got.frames else got

    @staticmethod
    def held(want: int | None, frames) -> int:
        """定住哪一帧（info 与 cook 共用这一处）：填了帧号就是那一帧，留空就是首帧（参数的 placeholder、nodes.json 的描述、
        E-FRAMEHOLD-RANGE 的文字都照这条写）。"""
        return want if want is not None else frames[0]

    @classmethod
    def cook(cls, ctx):
        # 只选取一帧，不修改像素：该帧文件硬链接到本节点的输出文件夹（跨文件系统时复制，
        # io/files.py link_or_copy）。每个节点的结果必须位于自己的文件夹中，
        # 不能只引用上游的文件（上游缓存可能被清除）。
        from ...data.packet import Packet
        from ...data.payloads import image_files, read_map
        from ...io.files import link_or_copy

        src = ctx.input("image")
        frames = src.meta["frames"]
        frame = cls.held(ctx.params["frame"], frames)
        files = image_files(src) if isinstance(src.meta.get("files"), dict) else {}
        if frame not in frames or frame not in files:
            raise Invalid(Msg("E-FRAMEHOLD-RANGE", frame=frame, first=frames[0], last=frames[-1]))
        one = files[frame]
        out = ctx.outputs["image"]
        out.mkdir(parents=True, exist_ok=True)
        here = out / one.name
        if not here.exists():
            link_or_copy(one, here)
        # 选中的帧整幅为 0（上游该帧未产生结果，或本身为黑场）：给出提醒而不报错。只读取这一帧
        data, alpha = read_map(here)
        if not ((alpha > 0) & (np.abs(data).max(axis=-1) > 0)).any():
            ctx.say("N-FRAMEHOLD-EMPTY", frame=frame)
        meta = {**src.meta, "frames": [frame], "files": {str(frame): here.name}}
        return {"image": Packet(out, src.type, meta)}


# 本节点没有模板卡属于设计意图：它是工具箱中的组件，与「Crop」「色彩转换」「FrameHold」同类，
# 不含算法、不产生交付物，素材为拼图时按需添加。
class SplitGrid(NodeDef):
    id = "core.split_grid"
    category = "img_adjust"
    on_node = ("rows", "cols")
    inputs = (Port("image", "image", "网格图", alpha=True, data=EITHER),)
    # 每格尺寸由「行」「列」「边距」决定，而非输入图的尺寸，因此该端口声明「尺寸由节点决定」，
    # 由 info() 在计算之前报告尺寸和帧数（与「视频转序列」的「分辨率」机制相同）。
    outputs = (Port("image", "image", "图像", type_from="input:image", shape=Shape(window="node")),)
    most_frames = 1  # 每次拆分一张：输入为序列时在提交前拒绝，可一键插入「FrameHold」选择其中一张
    # 一张图拆出多帧：帧号由本节点生成（从输入帧的帧号开始依次递增），
    # 因此与「视频转序列」一样属于帧来源，计算时的帧范围在此选择拆出的格（engine/evaluation.py info）
    frame_source = True

    class Params(NodeParams):
        rows: int = P(2, label="行", ge=1, le=64, group="网格")
        cols: int = P(2, label="列", ge=1, le=64, group="网格")
        order: Literal["row", "col"] = P(
            "row", label="顺序", group="网格", option_labels={"row": "先横后竖", "col": "先竖后横"})
        margin: int = P(0, label="边距", unit="px", ge=0, le=512, group="网格")

    @classmethod
    def cells(cls, params: dict, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """每格在网格图中的位置（x, y, 宽, 高），按「顺序」排列：info 与 cook 共用。"""
        rows, cols, m = params["rows"], params["cols"], params["margin"]
        cw, ch = width // cols, height // rows
        pairs = [(r, c) for r in range(rows) for c in range(cols)] if params["order"] == "row" else \
                [(r, c) for c in range(cols) for r in range(rows)]
        return [(c * cw + m, r * ch + m, cw - 2 * m, ch - 2 * m) for r, c in pairs]

    @classmethod
    def info(cls, params, inputs):
        """拆出的帧数与每格尺寸：本节点结果覆盖的帧与输入不同，需告知节点图（base.py info）。"""
        got = Info.merge(inputs.get("image") or [])
        if not got.width:
            return got
        boxes = cls.cells(params, got.width, got.height)
        if not got.frames:  # 输入为无帧号的画面（HDRI、照片）：格尺寸可立即确定，帧号需计算后才能确定
            return replace(got, width=boxes[0][2], height=boxes[0][3])
        first = int(got.frames[0])
        return Info(tuple(range(first, first + len(boxes))), boxes[0][2], boxes[0][3], False)

    @classmethod
    def cook(cls, ctx):
        from ...data.contracts import own_meta
        from ...data.payloads import (ExrWriter, file_at, has_alpha, has_validity, is_data, read_map, read_picture)
        from ...data.types import channels_of

        src, p = ctx.input("image"), ctx.params
        # 同一声明（most_frames）的第二道检查：提交前已按其拒绝一次（nodes/expects.py FrameCount(most=)），
        # 此处保证计算过程中也不会静默只拆第一帧（与「环境光」的处理相同）。
        if cls.most_frames and len(src.meta["frames"]) > cls.most_frames:
            raise Invalid(Msg("E-GRID-ONEFRAME", frames=len(src.meta["frames"])))
        w, h = src.meta["width"], src.meta["height"]
        boxes = cls.cells(p, w, h)
        if boxes[0][2] <= 0 or boxes[0][3] <= 0:
            raise Invalid(Msg("E-GRID-CELL", rows=p["rows"], cols=p["cols"], margin=p["margin"],
                              width=w // p["cols"], height=h // p["rows"]))
        picture = not is_data(src)
        # 半精度仅用于画面，数值图一律使用全精度（与「图像合成」「STMap」规则相同）：半精度最大值为 65504，
        # 深度图（厘米）、光流、位置图超出后会写成 inf/NaN 且节点不报任何提示；ST-map 的 0..1 需乘以画面宽度
        # 作为坐标，半精度在 2K 宽度下误差约一个像素。
        writer = ExrWriter(ctx.outputs["image"], channels_of(src.type),
                           validity=has_validity(src), half=picture,
                           colorspace=src.meta["colorspace"] if picture else None,
                           value_range=tuple(src.meta["range"]) if not picture else None, **own_meta(src),
                           **({"alpha": True} if picture and has_alpha(src) else {}))
        path = file_at(src, src.meta["frames"][0])
        values, valid = (read_picture(src, path), None) if picture else read_map(path)
        first = int(src.meta["frames"][0])

        def cut(job):  # 一格：各格互不相干，由引擎并行写出（ctx.each_done）；返回该格是否有内容
            i, (x, y, cw, ch) = job
            cell = values[y:y + ch, x:x + cw]
            writer.add(first + i, cell, None if valid is None else valid[y:y + ch, x:x + cw])
            return bool(np.any(cell))

        if not any(list(ctx.each_done(enumerate(boxes), cut))):  # 拆出的每一格都为 0：空结果不属于错误，但需说明原因
            ctx.say("N-GRID-EMPTY", rows=p["rows"], cols=p["cols"], margin=p["margin"], param="rows")
        return {"image": writer.packet()}


class Crop(NodeDef):
    id = "core.crop"
    category = "img_adjust"
    inputs = (Port("image", "image", "图像", alpha=True, data=EITHER),)
    # 本节点修改像素并更换画面框：尺寸由节点自身决定（计算前即可确定，见 said）。
    # 镜头数据随之失效：畸变参数针对特定画面测得，裁剪后同样的系数不再适用于该区域
    # （与 warped_by 同理）；镜头状态和像素比不变，裁剪不涉及光学和像素形状。
    outputs = (Port("image", "image", "图像", type_from="input:image",
                    shape=Shape(window="node", lens="unknown", said="I-SHAPE-CROP")),)

    class Params(NodeParams):
        left: int = P(0, label="左", unit="px", ge=0, group="裁剪")
        right: int = P(0, label="右", unit="px", ge=0, group="裁剪")
        top: int = P(0, label="上", unit="px", ge=0, group="裁剪")
        bottom: int = P(0, label="下", unit="px", ge=0, group="裁剪")

    on_node = ("left", "right", "top", "bottom")

    @classmethod
    def info(cls, params, inputs):
        """裁剪不改变帧：输出帧数与输入画面相同。"""
        return inputs["image"][0] if inputs.get("image") else Info()

    @classmethod
    def _passed_through(cls, ctx, src):
        """无需裁剪：像素不做任何修改，每一帧硬链接到本节点的输出文件夹（跨文件系统时复制）。
        每个节点的结果必须位于自己的文件夹中，不能只引用上游的文件（上游缓存可能被清除），
        做法与「FrameHold」相同。"""
        from ...data.packet import Packet
        from ...data.payloads import image_files
        from ...io.files import link_or_copy

        out = ctx.outputs["image"]
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        for f, one in image_files(src).items():
            here = out / one.name
            if not here.exists():
                link_or_copy(one, here)
            files[str(f)] = here.name
        return Packet(out, src.type, {**src.meta, "files": files})

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import ExrWriter, file_at, has_alpha, read_picture, window_of
        from ...data.types import channels_of
        from ...data.windows import Window

        src = ctx.input("image")
        window = window_of(src)
        w, h = window.plate
        cut = {k: int(ctx.params[k]) for k in ("left", "right", "top", "bottom")}
        new_w, new_h = w - cut["left"] - cut["right"], h - cut["top"] - cut["bottom"]
        if new_w < 1 or new_h < 1:
            raise Invalid(Msg("E-CROP-ALLCUT", width=w, height=h, **cut))
        if new_w == w and new_h == h:  # 四条边均为 0：不修改像素，每帧硬链接到本节点的文件夹
            return {"image": cls._passed_through(ctx, src)}
        channels = channels_of(src.type)
        out = ExrWriter(ctx.outputs["image"], channels, half=True,
                        colorspace=src.meta["colorspace"], alpha=has_alpha(src), window=Window(new_w, new_h))
        dx, dy, dw, dh = window.data          # 数据窗口在画面框坐标中的位置
        # 待裁剪区域在画面框坐标中：与数据窗口求交，不相交部分补 0
        x0, y0 = max(cut["left"], dx), max(cut["top"], dy)
        x1, y1 = min(w - cut["right"], dx + dw), min(h - cut["bottom"], dy + dh)
        def crop(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            picture = read_picture(src, file_at(src, f))
            tile = np.zeros((new_h, new_w, channels), np.float32)
            if x1 > x0 and y1 > y0:
                tile[y0 - cut["top"]:y1 - cut["top"], x0 - cut["left"]:x1 - cut["left"]] = \
                    picture[y0 - dy:y1 - dy, x0 - dx:x1 - dx, :channels]
            out.add(f, tile)

        list(ctx.each_done(src.meta["frames"], crop))
        packet = out.packet()
        if src.meta.get("still"):
            packet.meta["still"] = True
        ctx.say("I-CROP-DONE", width=new_w, height=new_h, was_width=w, was_height=h)
        return {"image": packet}


class OrientationField(NodeDef):
    id = "core.orientation_field"
    category = "img_channel"
    inputs = (Port("image", "image", "图像", alpha=True, data=False),
              Port("mask", "image.1", "遮罩", optional=True,
                   help="只在这块区域里量方向，外面的可信度给 0。可信度是按整张图里最集中的那个像素归一化的，"
                        "接上遮罩就只在遮罩里归一化，背景的杂纹不会把主体的可信度压下去",
                   expects=(SameShot("image"),)))
    outputs = (Port("orientation", "image.2", "方向场",
                    help="第一条通道是方向，单位度，0–180；第二条是可信度 0–1。角度量的是纹理本身的走向"),)

    class Params(NodeParams):
        period: float = P(3.0, label="纹理周期", unit="px", ge=1.0, le=64.0)
        reach: int = P(31, label="取样范围", unit="px", ge=5, le=201)
        angles: Literal[90, 180, 360] = P(180, label="角度数", group="精度",
                                          option_labels={"90": "90 档", "180": "180 档", "360": "360 档"})

    on_node = ("period", "reach")

    @classmethod
    def cook(cls, ctx):
        from ...data.maps import map_at
        from ..kit.orientation import HALF_TURN, orientation_field
        from ...data.payloads import ExrWriter, window_of

        image = ctx.input("image")
        mask = ctx.input("mask")
        window = window_of(image)
        out = ExrWriter(ctx.outputs["orientation"], 2, window=window)
        reach = int(ctx.params["reach"]) | 1  # 卷积核需要中心格，因此取奇数
        def measure(f):  # 一帧：各帧互不相干，由引擎逐帧并行（ctx.each_done）
            got = map_at(image, f, window.data)
            if got is None:
                return
            values = got[0]
            # 接受任意通道数：方向只取决于明暗变化，因此合并为单通道。三个及以上通道按亮度（照片的正确算法），
            # 单通道原样使用，两个通道（ST-map、方向场等数值图）取平均，因其没有亮度的概念。
            gray = (values[..., :3] @ np.array([0.2126, 0.7152, 0.0722]) if values.shape[-1] >= 3
                    else values.mean(axis=-1))
            direction, confidence = orientation_field(gray, size=reach, wavelength=float(ctx.params["period"]),
                                                      sigma=float(ctx.params["period"]) * 2.0 / 3.0,
                                                      angles=int(ctx.params["angles"]))
            if mask is not None:
                inside = map_at(mask, f, window.data)
                keep = np.zeros(gray.shape, bool) if inside is None else inside[0][..., 0] > 0.5
                if keep.any():  # 只在遮罩内归一化，再将遮罩外清零
                    confidence = confidence / max(float(confidence[keep].max()), 1e-12)
                confidence = np.where(keep, np.clip(confidence, 0.0, 1.0), 0.0).astype(np.float32)
            degrees = direction * (180.0 / HALF_TURN)
            out.add(f, np.stack([degrees, confidence], axis=-1).astype(np.float32))

        list(ctx.each_done(image.meta["frames"], measure))
        return {"orientation": out.packet()}


NODES = (StmapWarp, AlphaMerge, ChannelMerge, ImageMerge, FrameHold, SplitGrid, Crop, OrientationField)
