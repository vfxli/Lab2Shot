import type { Frame } from "./overlays";
import { t } from "../i18n/t";
import { CORNER_COLOR, FONT, moved, parse, pointUnder, removeSmallest, type Entry, type HandleTool2D, type Pt } from "./handleParts";

/** 火柴人手柄（「figure」，见 `lab2shot/nodes/handles.py` FIGURE_JOINTS）的全部逻辑：
 * 默认姿势、逐帧添加、绘制、拖动关节与右键删除；`figureTool` 是它在二维手柄表（`view/handles2d.ts` TOOLS_2D）里的一项。
 *
 * 姿势按帧存储，每帧一个，便于逐帧检查序列的连续性；新增帧使用默认 T-pose，
 * 或原样复制前一帧的姿势。不通过拖框创建，以免人体比例变形；新增姿势不得覆盖已有姿势。 */

/** 火柴人的骨骼连接（「figure」，见 nodes/handles.py FIGURE_JOINTS）：哪些关节之间构成骨骼。
 * 关节编号按 FIGURE_JOINTS 顺序；关节名称来自手柄的 `labels`，本文件不写任何身体部位名称。 */
const FIGURE_BONES: [number, number][] = [
  [0, 1], [0, 2], [0, 3], [3, 10], [10, 13],
  [1, 4], [4, 6], [6, 8], [2, 5], [5, 7], [7, 9],
  [10, 11], [10, 12], [11, 14], [14, 16], [12, 15], [15, 17],
];
/** 默认姿势：站立的 T-pose，比例固定为真人比例，顺序同 FIGURE_JOINTS。
 * 每项为 `[到身体中线的水平距离, 到头顶的垂直距离]`，单位为身高，因此仅需一个身高值即可摆放，
 * 比例不会变形。数值取人体测量学常见比例：髋关节位于 0.53 身高处、膝 0.285、踝 0.04、
 * 肩宽 ±0.095、上臂 0.19、前臂 0.145（T-pose 两臂水平伸直，指尖跨度约 0.86 身高）。
 *
 * 比例固定而不由拖框决定，原因是拖框的宽高取决于手动操作，会使人体被拉伸或压扁；
 * 因此只保留身高一个参数，比例由本表决定。
 *
 * 人物的左侧位于画面右侧：人体模型的 +X 为人物左侧，而画面 x 轴向右增长，因此面向观察者的人形呈镜像，
 * 与现实中面对面时一致。方向若颠倒，所有生成的动画都会左右肢体互换。 */
const FIGURE_TPOSE: [number, number][] = [
  [0.000, 0.470], [0.055, 0.480], [-0.055, 0.480], [0.000, 0.300],
  [0.055, 0.715], [-0.055, 0.715], [0.055, 0.955], [-0.055, 0.955],
  [0.075, 1.000], [-0.075, 1.000], [0.000, 0.145],
  [0.095, 0.165], [-0.095, 0.165], [0.000, 0.060],
  [0.285, 0.165], [-0.285, 0.165], [0.430, 0.165], [-0.430, 0.165],
];
/** 一个火柴人绘制的关节数，等于上表的行数；其他位置不得另写常量 18
 * （服务器端的表在 `lab2shot/nodes/handles.py FIGURE_JOINTS`，两端的关节数须一致）。 */
export const FIGURE_COUNT = FIGURE_TPOSE.length;

const TPOSE_SPAN = 0.86; // 指尖跨度（单位为身高，对应上表 ±0.43），用于判断放入画面时是否超出宽度

/** 条目与关节坐标之间的相互转换。 */
export const joints = (e: Entry): Pt[] => FIGURE_TPOSE.map((_, k) => ({ x: e.v[2 * k], y: e.v[2 * k + 1] }));
export const figureEntry = (frame: number, pts: Pt[]) =>
  `${frame}:${pts.map((q) => `${q.x.toFixed(1)},${q.y.toFixed(1)}`).join(",")}`;

/** 在画面正中放置默认 T-pose：身高取画面高度的 80%，两臂展开超出画面宽度时按宽度缩小，
 * 保证窄画面中手部不超出画面。 */
const tposeIn = (width: number, height: number): Pt[] => {
  const tall = Math.min(height * 0.8, (width * 0.9) / TPOSE_SPAN);
  const [cx, top] = [width / 2, (height - tall) / 2];
  return FIGURE_TPOSE.map(([dx, dy]) => ({ x: cx + dx * tall, y: top + dy * tall }));
};

/** 该手柄参数中已有姿势的帧，升序排列。每帧最多一个姿势（见 nodes/core/sketch.py 的 E-FIGURE-TWICE）。 */
export const figureFrames = (values: string[]): number[] => values.map((s) => parse(s).frame).sort((a, b) => a - b);

/** 计算「添加帧」的目标帧及其对应的「前一帧」。
 *
 * 目标帧：当前帧尚无姿势时即为当前帧（与各 DCC 中在当前帧设置关键帧的行为一致）；
 * 当前帧已有姿势时，取其后第一个无姿势的帧。每帧只能有一个姿势，因此顺延而不覆盖已有姿势。
 * 连续添加多次即得到连续多帧，不会丢失。
 *
 * 前一帧：目标帧之前最近的已有姿势的帧；不存在时为 null（此时「基于前一帧」变灰并附原因，
 * 不隐藏）。 */
export function figureAdd(values: string[], frame: number): { target: number; previous: number | null } {
  const frames = figureFrames(values);
  let target = frame;
  while (frames.includes(target)) target += 1;
  const before = frames.filter((f) => f < target);
  return { target, previous: before.length ? before[before.length - 1] : null };
}

/** 添加一帧：默认放置 T-pose；`basedOnPrevious` 为真时原样复制前一帧的姿势。
 * 返回新的参数值与目标帧，面板据此将时间线跳转到该帧，便于使用者确认添加位置。
 * 需要复制但不存在任何已有姿势时返回 null（此时对应按钮已处于禁用状态）。 */
export function addFigure(values: string[], frame: number, size: { width: number; height: number },
                          basedOnPrevious = false): { values: string[]; frame: number } | null {
  const { target, previous } = figureAdd(values, frame);
  if (basedOnPrevious && previous === null) return null;
  const src = basedOnPrevious ? values.find((s) => parse(s).frame === previous) : undefined;
  const pts = src ? joints(parse(src)) : tposeIn(size.width, size.height);
  return { values: [...values, figureEntry(target, pts)], frame: target };
}
/** 火柴人的包围框，供右键删除时命中判断使用。 */
export const figureBox = (pts: Pt[]) => ({
  x1: Math.min(...pts.map((q) => q.x)), y1: Math.min(...pts.map((q) => q.y)),
  x2: Math.max(...pts.map((q) => q.x)), y2: Math.max(...pts.map((q) => q.y)),
});

/** 绘制火柴人。`named` 为需要标注名称的关节（即正在拖动的关节）。在分镜尺寸的人形上同时标注全部 18 个名称
 * 会相互重叠而无法阅读，且界面文字不得截断或重叠；人形本身足以表明各点含义，拖动时再显示该关节名称。 */
export function drawFigure(f: Frame, pts: Pt[], faint: boolean, note: string, labels: string[], named = -1, noteRow = 0) {
  const { ctx, at } = f;
  const xy = pts.map((q) => [at.x + q.x * at.s, at.y + q.y * at.s]);
  ctx.save();
  ctx.globalAlpha = faint ? 0.35 : 1;
  ctx.lineWidth = f.line * 1.5;
  ctx.strokeStyle = ctx.fillStyle = CORNER_COLOR; // 与平面四角使用同一种黄色，二者均为「先放置后调整」类手柄
  if (faint) ctx.setLineDash([5, 4]);
  ctx.beginPath();
  for (const [a, b] of FIGURE_BONES) {
    ctx.moveTo(xy[a][0], xy[a][1]);
    ctx.lineTo(xy[b][0], xy[b][1]);
  }
  ctx.stroke();
  ctx.setLineDash([]);
  if (!faint)
    xy.forEach(([x, y], k) => {
      ctx.beginPath();
      ctx.arc(x, y, k === 0 ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
      if (k === named && labels[k]) ctx.fillText(labels[k], x + 8, y - 6);
    });
  // `noteRow`：两个洋葱皮标注各占一行。「基于前一帧」复制出的相邻两帧姿势完全相同，
  // 在中间帧上两个淡影会完全重合，标注文字若不分行将重叠而无法辨认。
  if (note) ctx.fillText(note, xy[13][0] + 8, xy[13][1] - 14 - noteRow * 14);
  ctx.restore();
}

/** 火柴人手柄的工具（`view/handles2d.ts` TOOLS_2D）。 */
export const figureTool: HandleTool2D = {
  hint: "ui.view.hint.figure",
  draws: false,
  marksFrames: true,
  // The current frame's figure is drawn solid (with the name of the joint being dragged); the previous and the next
  // posed frame are laid over it faintly, one each — onion skin, as in 2D animation software, for checking continuity
  // frame by frame. Not every frame faintly: a dozen overlapping figures make continuity impossible to judge.
  draw: (h, f, values, drag) => {
    const { frame } = f;
    f.ctx.font = FONT;
    const dragged = drag && drag.joint !== undefined ? drag : null;
    const all = figureFrames(values);
    const prev = all.filter((k) => k < frame).pop();
    const next = all.find((k) => k > frame);
    const noteOf = (at: number) => (at === prev ? t("ui.view.previous_frame", { frame: at }) : at === next ? t("ui.view.next_frame", { frame: at }) : "");
    // faint ones first, then the solid one: the current frame's figure is always on top, and it is what the pointer grabs
    const order = values.map(parse).filter((e) => e.frame === prev || e.frame === next || e.frame === frame)
                        .sort((a, b) => Number(a.frame === frame) - Number(b.frame === frame));
    for (const e of order) {
      let pts = joints(e);
      if (dragged && e.frame === frame)
        pts = pts.map((q, k) => (dragged.whole || k === dragged.joint ? moved(q, dragged) : q));
      drawFigure(f, pts, e.frame !== frame, noteOf(e.frame), h.labels,
                 dragged && e.frame === frame ? (dragged.whole ? 0 : dragged.joint!) : -1,
                 e.frame === next ? 1 : 0);
    }
  },
  // Only the joints of the current frame's figure are grabbed; a drag creates nothing (creating by dragging a box would
  // distort the body's proportions). Poses are added with 「添加帧」 in the parameter panel, proportioned by FIGURE_TPOSE
  start: (_h, values, frame, p, scale) => {
    const here = values.map(parse).find((e) => e.frame === frame);
    if (!here) return null;
    const j = pointUnder(joints(here), p, scale);
    return j < 0 ? null : { from: p, to: p, joint: j, whole: j === 0 }; // the pelvis moves the whole body
  },
  // a drag only adjusts a pose and never creates one: creating a figure by dragging would silently replace the one already on this frame
  finish: (_h, values, frame, drag) => {
    const at = values.findIndex((text) => parse(text).frame === frame);
    if (drag.joint === undefined || at < 0) return null;
    const pts = joints(parse(values[at]));
    const posed = pts.map((q, k) => (drag.whole || k === drag.joint ? moved(q, drag) : q));
    return values.map((text, i) => (i === at ? figureEntry(frame, posed) : text));
  },
  // the figure's box with a margin of 10 screen pixels, the one reach of the 2D handles (pointUnder)
  remove: (_h, values, frame, p, scale) => {
    const reach = 10 / Math.max(scale, 1e-6);
    return removeSmallest(values, (e) => {
      if (e.frame !== frame) return null;
      const b = figureBox(joints(e));
      return { inside: p.x >= b.x1 - reach && p.x <= b.x2 + reach && p.y >= b.y1 - reach && p.y <= b.y2 + reach, area: (b.x2 - b.x1 + 2 * reach) * (b.y2 - b.y1 + 2 * reach) };
    });
  },
};
