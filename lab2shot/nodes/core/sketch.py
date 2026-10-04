"""草图输入：「手画简笔画」节点，输出草图（2D 跟踪点），接给「Sketch2Anim 动作生成」的「草图」口。

绘制火柴人是独立于解算器的工具：解算器只负责解算，节点图上应能看出草图的来源，并可替换为其他来源。
上游 Sketch2Anim 的输入是火柴人的关节坐标（`third_party/sketch2anim/repo/demo_kp_traj_2d.py` 的
`batch['pose_2d'] / ['hint_2d'] / ['text']`），不使用任何像素。

    「手画简笔画」（本模块）────────────────────────────────┐
    「拆网格图」（split_grid）→「手画简笔画」的图像口 ─┼→「Sketch2Anim 动作生成」的「草图」口
    「读取序列」（file）→ 同上 ──────────────┘

棋盘格图和序列图接入「手画简笔画」而非解算器：上游不使用像素，画面仅作为绘制时参照的分镜图，
因此它们是「手画简笔画」的底图，而不是解算器的输入。
"""

from __future__ import annotations

from ..port import EITHER
from ...errors import Invalid
from ... import i18n
from ...messages import Msg

from ..base import NodeDef, NodeParams, P, Port, parse_figures, say_bad_entries
from ...data.units import DEFAULT_HEIGHT, DEFAULT_WIDTH
from ..handles import FIGURE_JOINTS, figure_handle

# 画布尺寸：未接底图时使用 data/units.py 的默认尺寸。该数值不影响结果（上游只使用姿势与位置的变化），
# 但视图绘制和坐标解释都需要一个画布。默认值仅在 data/units.py 中定义。


class DrawFigure(NodeDef):
    id = "draw_figure"
    # 计算结果变化时必须递增版本，否则 work/ 中的旧结果仍会命中缓存
    # （指纹由 type id + version + 参数构成，见 engine/cook.py）。
    version = 2
    # 跟踪点归「几何」类（火柴人的 18 个关节即一组跟踪点），不单独设分类。
    category = "geometry_tools"
    # 底图可选：没有分镜图时也应能绘制。底图的像素不传给下游：上游 Sketch2Anim 输入网络的只有火柴人
    # 18 个关节的坐标（`third_party/sketch2anim/repo/demo_kp_traj_2d.py` 的 `batch['pose_2d']`），上游仓库中
    # 也没有从图像识别关节的步骤（官方 demo 使用动捕 3D 关节经 `project2D()` 投影得到的 2D 坐标）。
    # 本节点只接收一段序列，不识别宫格图，没有行列参数，也不做拆分；宫格图应先经「拆网格图」拆成序列再接入。
    # 端口提示和节点说明中必须写明「底图不参与计算」，以免被误解为基于图像生成。
    inputs = (Port("image", "image", optional=True, data=EITHER),)
    outputs = (Port("sketch", "tracks2d"),)
    # 画布复用现有的手柄机制（nodes/handles.py），不另建交互。
    handles = (figure_handle("poses"),)

    class Params(NodeParams):
        # widget「figure」包括面板上的两个「添加帧」按钮和已绘制帧的列表（webui ParamControls.tsx FigureFrames）。
        # 每帧最多一个火柴人（由 cook 中的 E-FIGURE-TWICE 校验）：帧是该数据的单位，因此面板按帧操作，
        # 视图中只负责拖动关节。不使用 widget「canvas」（手画遮罩所用）：拖框创建容易产生失真的人体比例，
        # 且在同一帧上创建第二个会替换第一个。
        poses: list[str] = P([], widget="figure", group="sketch")

    @classmethod
    def cook(cls, ctx):
        import numpy as np

        from ...data.payloads import tracks_packet

        src = ctx.input("image")          # 未连接时为 None
        width, height = (src.meta["width"], src.meta["height"]) if src is not None else (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        say_bad_entries(ctx, "poses")
        drawn = parse_figures(cls, "poses", ctx.params["poses"])
        if not drawn:  # 未绘制任何姿势：输出空草图并给出提示（空结果不视为错误）
            ctx.say("N-FIGURE-NOPOSES", param="poses")
        # 每帧最多一个：帧是该数据的单位。若不校验，下方 `frames.index(frame)` 会使同一帧的后一个静默覆盖前一个，
        # 既不报错也无法在图上察觉。界面无法产生重复帧（该帧已有姿势时按钮禁用），但手工编辑的节点图文件仍须校验。
        seen: set[int] = set()
        for frame, _ in drawn:
            if frame in seen:
                raise Invalid(Msg("E-FIGURE-TWICE", frame=frame))
            seen.add(frame)
        # 接入序列时，草图覆盖整段序列（已绘制帧的 `visible` 为真，其余为假），而不只包含已绘制的帧：
        # 18 条轨迹覆盖整段镜头，仅在关键姿势帧上有值（下游 `adapters/sketch2anim/nodes.py figures_in`
        # 会跳过火柴人不完整的帧）。
        # 必须覆盖整段的原因：时间线范围取自当前显示节点的结果（`webui/src/view/plan.ts`）。若只输出已绘制的帧，
        # 第一个姿势计算完成后时间线即收缩为该帧，无法切换到其他帧，「添加帧」也只能作用于同一帧。
        shot = list(src.meta["frames"]) if src is not None else []
        outside = sorted(f for f in seen if shot and f not in shot)
        if outside:  # 仅手工编辑的节点图文件可能出现：姿势位于序列之外的帧（给出提示并继续计算）
            ctx.say("N-FIGURE-OUTSIDE", count=len(outside), frames=i18n.Both.of(lambda: i18n.separator().join(str(f) for f in outside)),
                    lo=shot[0], hi=shot[-1], param="poses")
        # 未接序列时，覆盖第一个到最后一个关键姿势之间的整段（理由同上：若只输出两帧，时间线上只有这两帧，
        # 无法在中间插入）。在最后一帧之后添加姿势时，于最后一帧执行添加，下次计算后时间线范围随之扩展。
        frames = shot or (list(range(min(seen), max(seen) + 1)) if seen else [1001])
        names = [label for _, label in FIGURE_JOINTS]
        tracks = np.zeros((len(names), len(frames), 2), np.float32)
        visible = np.zeros((len(names), len(frames)), bool)
        for frame, xy in drawn:
            if frame not in frames:
                continue
            k = frames.index(frame)
            for j, (x, y) in enumerate(xy):
                tracks[j, k] = (x, y)
                visible[j, k] = True
        out = ctx.outputs["sketch"]
        out.mkdir(parents=True, exist_ok=True)
        return {"sketch": tracks_packet(out, frames, width, height, tracks, visible,
                                        np.asarray(frames[:1] * len(names), np.int64), names=names, figure=True)}


NODES = (DrawFigure,)
