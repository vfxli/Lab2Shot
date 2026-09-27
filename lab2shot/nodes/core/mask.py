"""遮罩：Roto、遮罩合并、遮罩调整、人物框转遮罩、分割转遮罩、深度转遮罩、置信度转遮罩。本模块对所有来源通用：
各抠像、分割、检测或深度节点的结果都在此转换为遮罩，遮罩的组合也在此完成，因此新增方法无需编写自己的遮罩代码。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port, parse_picks, parse_shapes, person_ids, typed_list
from ..expects import KnownPeople, Metric, SameShot
from ..handles import Handle


# 「方式」的三个选项只定义一处：参数上的名称与空结果提示使用同一词语
MERGE_MODES = {"union": "合并", "subtract": "相减", "intersect": "相交"}


class MaskMerge(NodeDef):
    id = "core.mask_merge"
    version = 2  # 静帧遮罩（整个镜头一张图）作用于另一张遮罩的每一帧
    category = "mask_edit"
    on_node = ("mode",)
    inputs = (Port("a", "image.1", "遮罩 A"), Port("b", "image.1", "遮罩 B", expects=(SameShot("a"),)))
    outputs = (Port("mask", "image.1", "遮罩"),)

    class Params(NodeParams):
        mode: Literal["union", "subtract", "intersect"] = P(
            "union", label="方式", group="遮罩", option_labels=MERGE_MODES,
            help="合并：两个遮罩里有一个是 1 就是 1（取较大值，Nuke 的 max）；相减：A 里去掉 B 盖住的部分（A × (1 − B)，Nuke 的 "
                 "stencil）；相交：两个都有的部分（取较小值，Nuke 的 min）。软边照样保留",
        )

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter, window_of
        from ...data.frames import shot_frames
        from ...data.maps import map_at, same_size

        a, b = ctx.input("a"), ctx.input("b")
        same_size({"遮罩 A": a, "遮罩 B": b})
        mode = ctx.params["mode"]
        window = window_of(a)  # 两张遮罩都在第一张的窗口中读取（有画布时为其画布）
        out = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True, window=window)
        blank = np.zeros(window.canvas[::-1], np.float32)

        def at(mask, frame):  # 遮罩缺少的帧视为空；静帧则每一帧都使用该图
            got = map_at(mask, frame, window.data)
            return blank if got is None else got[0][..., 0]

        had = left = False  # 分别记录输入是否有内容、输出是否有内容（空结果不属于错误，但需说明原因）
        for f in ctx.each(shot_frames([a, b])):
            ma, mb = at(a, f), at(b, f)
            # 遮罩取值为 0..1（该端口的值域声明为 UNIT）：相减后截断，不写出无效像素；
            # 输入不是遮罩时（两张深度图相减可达 -3e5），半精度会溢出为 inf。
            merged = np.clip(np.maximum(ma, mb) if mode == "union" else ma * (1.0 - mb) if mode == "subtract"
                             else np.minimum(ma, mb), 0.0, 1.0)
            had = had or bool(ma.any()) or bool(mb.any())
            left = left or bool(merged.any())
            out.add(f, merged)
        if had and not left:  # 输入有内容而输出为空：数据在本节点中丢失，必须给出提示
            ctx.say("N-MASK-EMPTYMERGE", mode=MERGE_MODES[mode])
        return {"mask": out.packet()}


class MaskAdjust(NodeDef):
    id = "core.mask_adjust"
    category = "mask_edit"
    on_node = ("grow", "feather", "invert")
    inputs = (Port("mask", "image.1", "遮罩"),)
    outputs = (Port("mask", "image.1", "遮罩"),)

    class Params(NodeParams):
        grow: float = P(0.0, label="扩缩", unit="px", ge=-500, le=500, group="遮罩",
                        help="正数向外扩（圆形笔刷，软边整体外移，Nuke 的 Erode 取负值），负数向内收。给跟踪、修补留余量常用 10–30")
        feather: float = P(0.0, label="羽化", unit="px", ge=0, le=500, group="遮罩",
                           help="边缘变软的宽度：边缘从 0 过渡到 1 大约用这么多像素。0 = 保持原样")
        invert: bool = P(False, label="反转", group="遮罩", help="最后把遮罩反过来：1 变 0、0 变 1，比如从「人」得到「人以外的背景」")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter, window_of
        from ...data.maps import feather, grow, map_at

        src, p = ctx.input("mask"), ctx.params
        out = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True, window=window_of(src))
        for f in ctx.each(src.meta["frames"]):
            m = feather(grow(map_at(src, f)[0][..., 0], p["grow"]), p["feather"])
            out.add(f, np.clip(1.0 - m if p["invert"] else m, 0.0, 1.0))
        return {"mask": out.packet()}


class BoxesMask(NodeDef):
    id = "core.boxes_mask"
    on_node = ("ids",)
    category = "mask_make"
    inputs = (Port("boxes", "boxes", "人物框", expects=(KnownPeople(),)),)
    outputs = (Port("mask", "image.1", "遮罩"),)
    # 算法定义不在此处，而在算法目录（lab2shot/ops/ops.toml）中：
    # 选人使用 people.select（此处只用其「按编号」规则），框的抗锯齿覆盖率使用 boxes.to_mask。
    # 浏览器按同一份描述执行（webui/src/ops/run.ts），因此修改框之后画面立即更新，无需提交计算。
    ops = ("people.select", "boxes.to_mask")

    class Params(NodeParams):
        ids: str = P("", label="编号", group="人物", placeholder="全部",
                     help="只要这些人：填人物编号，多个用逗号，如 1,3（1 号最显眼，2D 视图里框上有标）；留空 = 框里的所有人")

    @classmethod
    def browser_ops(cls, params, types):
        """框转遮罩在浏览器中由若干浮点数绘制出整张遮罩：输入为人物框，
        无需传输任何像素，因此修改框或编号后画面立即更新。"""
        ids = sorted(person_ids(params["ids"]))  # 框中不存在的编号由端口的用法检查提示（KnownPeople）
        return ({"op": "people.select", "args": {"rule": "ids" if ids else "all", "ids": ids}},
                {"op": "boxes.to_mask", "args": {}})

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter, read_boxes
        from ...ops import run as run_op

        src = ctx.input("boxes")
        people = read_boxes(src)
        # 参数到算法参数的转换只在 browser_ops 一处定义：浏览器使用同一结果
        pick, paint = cls.browser_ops(ctx.params, {})
        chosen = run_op(pick["op"], {"items": people, **pick["args"]})
        people = [people[i] for i in chosen["indices"]]
        w, h = src.meta["width"], src.meta["height"]
        out = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True)
        for f in ctx.each(src.meta["frames"]):
            boxes = [person["boxes"][str(f)] for person in people if str(f) in person["boxes"]]
            out.add(f, run_op(paint["op"], {"boxes": boxes, "width": w, "height": h, **paint["args"]})["mask"])
        return {"mask": out.packet()}


class SegmentSelect(NodeDef):
    id = "core.segment_select"
    category = "mask_make"
    inputs = (Port("segmentation", "image.1", "分割图"),)
    outputs = (Port("mask", "image.1", "遮罩"),)
    handles = (Handle("points", {"points": "picks"}, source="segmentation"),)

    class Params(NodeParams):
        classes: str = P("", label="类别", widget="classes", group="选区", placeholder="全部物体", choices_from=("segmentation",),
                         help="要哪些类别：上游算过以后点下面列出的类别；也可以填类别名或编号，多个用逗号，如 Hair, Face_Neck 或 1,3。"
                              "类别和点选都空着 = 除背景以外的全部")
        picks: list[str] = P([], label="点选", widget="picks", group="选区", placeholder="在 2D 视图里点要的区域",
                             help="在 2D 视图里点分割图上的区域，点到哪一类就加上哪一类（和「类别」一起算）；点错了在这里删掉")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter, window_of
        from ...data.maps import map_at

        src = ctx.input("segmentation")
        chosen = cls.chosen(ctx.params["classes"], src.meta.get("classes") or [])
        for frame, x, y in parse_picks(ctx.params["picks"]):
            got = map_at(src, frame)
            h, w = src.meta["height"], src.meta["width"]
            if got is None or not (0 <= x < w and 0 <= y < h):
                ctx.say("N-MASK-PICKOUTSIDE", frame=frame, x=x, y=y, param="picks")
                continue
            label = int(round(float(got[0][int(y), int(x), 0])))
            if label == 0:
                ctx.say("N-MASK-PICKBACKGROUND", frame=frame, x=x, y=y, param="picks")
                continue
            chosen = (chosen or set()) | {label}
        out = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True, window=window_of(src))
        got_any = False
        for f in ctx.each(src.meta["frames"]):
            labels = np.rint(map_at(src, f)[0][..., 0]).astype(np.int64)
            picked = (labels > 0) if chosen is None else np.isin(labels, list(chosen))
            got_any = got_any or bool(picked.any())
            out.add(f, picked)
        if not got_any:  # 未选中任何像素：输出空遮罩并给出提示（空结果不属于错误，但需提示）
            ctx.say("N-MASK-NOPIXELS", classes="、".join(sorted(str(c) for c in chosen)) if chosen else "背景以外的全部")
        return {"mask": out.packet()}

    @classmethod
    def choices(cls, params, inputs):
        """已连接分割结果中可点选的类别（不含背景）：以中文名显示，也可通过输入编号或中文名选择。"""
        src = inputs.get("segmentation")
        classes = [c for c in (src.meta.get("classes") or []) if c["index"] > 0] if src is not None else []
        if not classes:
            return {}
        return {"classes": {"options": [c["name"] for c in classes], "labels": {c["name"]: c.get("name_zh") or c["name"] for c in classes},
                            "aliases": {c["name"]: [str(c["index"]), *([c["name_zh"]] if c.get("name_zh") else [])] for c in classes}}}

    @staticmethod
    def chosen(text: str, classes: list[dict]) -> set[int] | None:
        """输入列表所指定的类别索引（类别编号、名称或中文名，不区分大小写）；未输入时为 None。"""
        wanted = typed_list(text)
        if not wanted:
            return None
        names = {}
        for c in classes:
            for key in (str(c["index"]), c.get("name"), c.get("name_zh")):
                if key:
                    names[str(key).casefold()] = c["index"]
        unknown = [t for t in wanted if t.casefold() not in names]
        if unknown:
            known = [c.get("name_zh") or c["name"] for c in classes]
            if not known:
                raise Invalid(Msg("E-MASK-NOCLASSES", unknown=unknown))
            raise Invalid(Msg("E-MASK-NOCLASS", unknown=unknown, known=known))
        return {names[t.casefold()] for t in wanted}


class DepthKey(NodeDef):
    id = "core.depth_key"
    category = "mask_make"
    on_node = ("near", "far", "soft")
    inputs = (Port("depth", "image.1", "深度图", expects=(Metric(),)),)
    outputs = (Port("mask", "image.1", "遮罩"),)

    class Params(NodeParams):
        near: float = P(0.0, label="近", unit="cm", ge=0, group="距离",
                        help="从多远开始算（离相机的距离，沿镜头方向）。真实尺度的深度填厘米；要手动缩放的深度按视图里显示的数值填")
        far: float | None = P(None, label="远", unit="cm", gt=0, group="距离", placeholder="无限远",
                              help="到多远为止；留空 = 一直到最远。比如近 0、远 300 抠出离相机 3 米以内的物体")
        soft: float = P(10.0, label="软边", unit="cm", ge=0, group="距离",
                        help="范围外面再过渡这么远，从 1 渐变到 0，边缘不生硬。0 = 硬边")
        include_empty: bool = P(False, label="含无值像素", group="距离",
                                help="深度图里没有值的像素（天空、太远或模型没把握的地方）也算进遮罩。「远」留空、想抠整个背景连天空时打开")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter, window_of
        from ...data.maps import map_at, smoothstep

        src, p = ctx.input("depth"), ctx.params
        if p["far"] is not None and p["far"] <= p["near"]:
            raise Invalid(Msg("E-MASK-FARNEAR", far=p["far"], near=p["near"]))
        out = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True, window=window_of(src))
        found = False
        for f in ctx.each(src.meta["frames"]):
            z, alpha = map_at(src, f)
            z = z[..., 0]
            m = smoothstep(p["near"] - p["soft"], p["near"], z)
            if p["far"] is not None:
                m *= 1.0 - smoothstep(p["far"], p["far"] + p["soft"], z)
            m = np.where(alpha > 0, m, 1.0 if p["include_empty"] else 0.0)
            found = found or bool((m > 0).any())
            out.add(f, m)
        if not found:
            if p["far"] is not None:
                ctx.say("N-MASK-EMPTYRANGE", near=p["near"], far=p["far"])
            else:
                ctx.say("N-MASK-EMPTYBEYOND", near=p["near"])
        return {"mask": out.packet()}


class ConfidenceMask(NodeDef):
    id = "core.confidence_mask"
    category = "mask_make"
    on_node = ("conf_threshold", "soft", "keep")
    inputs = (Port("confidence", "image.1", "置信度"),)
    outputs = (Port("mask", "image.1", "遮罩"),)
    # 置信度和遮罩都是单通道图，可直接连接，不会被标为错误连线，因此此处不声明「一键插入」。
    # 本节点的功能是按门槛二值化、软边和反选，而不是类型转换。

    class Params(NodeParams):
        conf_threshold: float = P(0.5, label="置信度门槛", ge=0, le=1, group="遮罩",
                             help="置信度高于它算可信（0–1）。只和同一个模型的置信度比：换了模型要重新挑。显示上游的置信度时，"
                                  "鼠标放在画面上，右下角写着那里的值，照着挑")
        soft: float = P(0.0, label="软边", ge=0, le=0.5, group="遮罩",
                        help="门槛上下这么宽的一段置信度从 0 渐变到 1，遮罩边缘不生硬；0 = 硬边（不是 0 就是 1）")
        keep: Literal["trusted", "doubtful"] = P(
            "trusted", label="区域", group="遮罩", option_labels={"trusted": "可信的", "doubtful": "不可信的"},
            help="可信的：置信度高的地方是 1（只在这些地方跟点、对齐、建点云）；不可信的：反过来，标出模型没把握的地方（去修补、排除）")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter, window_of
        from ...data.maps import map_at, smoothstep

        src, p = ctx.input("confidence"), ctx.params
        t, soft = p["conf_threshold"], p["soft"]
        out = ExrWriter(ctx.outputs["mask"], 1, value_range=UNIT, half=True, window=window_of(src))
        found = False
        for f in ctx.each(src.meta["frames"]):
            trusted = smoothstep(t - soft, t + soft, map_at(src, f)[0][..., 0])
            m = trusted if p["keep"] == "trusted" else 1.0 - trusted
            found = found or bool((m > 0).any())
            out.add(f, m)
        if not found:
            ctx.say("N-MASK-NOTRUSTED" if p["keep"] == "trusted" else "N-MASK-NODOUBTFUL", threshold=t, param="conf_threshold")
        packet = out.packet()
        if src.meta.get("still"):
            packet.meta["still"] = True
        return {"mask": packet}


class Roto(NodeDef):
    id = "core.draw_mask"
    category = "mask_make"
    # 节点上不显示参数：形状是绘制出的一组条目，不是可在节点上修改的数值或开关
    # （nodes/base.py simple_kind；「分割转遮罩」的「点选」同样只在面板中）。
    inputs = (Port("image", "image", "图像", alpha=True),)
    outputs = (Port("mask", "image.1", "遮罩"),)
    # 画布属于现有手柄体系中的一种（nodes/handles.py），不另建交互：在视图中绘制，完成后写入该参数
    handles = (Handle("canvas", {"shapes": "shapes"}),)

    class Params(NodeParams):
        shapes: list[str] = P([], label="形状", widget="canvas", group="遮罩", placeholder="在 2D 视图里拖出轮廓",
                              help="手画的轮廓：在 2D 视图里按住左键沿着要的范围拖一圈，松手就闭合成一个形状；"
                                   "右键点形状里面删掉它。可以画好几个，合在一起是一张遮罩。"
                                   "每个形状记着是在哪一帧画的，但整段镜头都算数")

    @classmethod
    def cook(cls, ctx):
        from lab2shot_worker.files import write_exr

        from ...data.maps import polygon_coverage
        from ...data.payloads import UNIT, channel_names, still_packet, window_of

        src = ctx.input("image")
        w, h = src.meta["width"], src.meta["height"]
        window = window_of(src)  # 输入画面带扩边时，输出遮罩覆盖同一画布（端口声明为「跟随画面」）
        ox, oy = window.offset  # 绘制坐标以画面左上角为原点，有扩边时画布原点位于其外侧
        shapes = parse_shapes(ctx.params["shapes"])
        mask = polygon_coverage([[(x + ox, y + oy) for x, y in shape] for shape in shapes], *window.canvas)
        if not shapes:  # 未绘制任何形状：输出空遮罩并给出提示（空结果不属于错误）
            ctx.say("N-ROTO-NOSHAPES", param="shapes")
        elif not mask.any():  # 已绘制但全部位于画面之外
            ctx.say("N-ROTO-OUTSIDE", count=len(shapes), width=w, height=h, param="shapes")
        out = ctx.outputs["mask"]
        out.mkdir(parents=True, exist_ok=True)
        frames = list(src.meta["frames"]) or [1001]
        path = out / f"frame.{frames[0]}.exr"
        write_exr(path, mask[..., None], channel_names(1), half=True, windows=window.exr_windows)
        packet = still_packet(out, path, frames, w, h, channels=1, value_range=UNIT, window=window)
        # 手绘结果即抠像 alpha（data/contracts.py MEANING 中的 matte）：视图默认将其半透明叠加在上游画面上，
        # 交付时写入 EXR 的 a 通道。绘制时可看到画面依赖于此，无需另外实现「绘制时显示上游」。
        packet.meta["matte"] = True
        return {"mask": packet}


NODES = (Roto, MaskMerge, MaskAdjust, BoxesMask, SegmentSelect, DepthKey, ConfidenceMask)
