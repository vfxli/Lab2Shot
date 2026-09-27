import type { BoxesData, HandleDef, TracksData } from "../api";
import type { Frame } from "./overlays";
import { ERROR_COLOR } from "../platform/palette";
import { CORNER_COLOR, type Entry, type Pt, parse } from "./handleParts";
// 火柴人类手柄的全部逻辑位于 view/figure2d.ts
import { drawFigure, figureBox, figureEntry, figureFrames, joints } from "./figure2d";

export type { Pt };
export { addFigure, figureAdd, figureFrames } from "./figure2d";

/** The 2D handles (nodes/handles.py): one tool per kind, bound to a list parameter of the node. A tool draws the
 * parameter's entries on the current frame and turns clicks and drags into new entries; a right click removes the
 * entry under the pointer. Entries are "frame:x,y[,label]" (points, picks), "frame:x1,y1,x2,y2" (boxes),
 * "frame:x1,y1,...,x4,y4" (a plane's corners, in order around it) and "frame:x1,y1,..." with as many points as were
 * drawn (canvas: one closed outline each), image pixels from the top-left corner. */

const FONT = "600 11px -apple-system, 'PingFang SC', 'Microsoft YaHei UI', sans-serif";
const POINT_COLORS = ["#FF8FC7", ERROR_COLOR]; // the first label (a point to track, "this"), the second ("not this")
const near = (a: Pt, b: Pt, s: number) => Math.hypot(a.x - b.x, a.y - b.y) * s < 10;
const DRAWN_COLOR = "#64D2FF"; // the colour of what the pointer draws on the picture: a box, a freehand outline ("canvas": 「手画遮罩」)
const CANVAS_STEP = 1.5; // image pixels a freehand drag must travel before another point is added

/** A drag in progress: from where to where, (corners) the corner being moved, (canvas) the outline drawn so far,
 * (figure) the joint being moved, or the whole figure when the pelvis is grabbed, (points) the point being moved. */
export interface Drag {
  from: Pt;
  to: Pt;
  corner?: number;
  path?: Pt[];
  joint?: number;
  whole?: boolean;
  point?: number; // which entry of a points handle is being moved (its index in `values`)
}

const quad = (e: Entry): Pt[] => [0, 1, 2, 3].map((k) => ({ x: e.v[2 * k], y: e.v[2 * k + 1] }));
/** An entry's points: a quad's four, an outline's as many as were drawn. */
const polygon = (e: Entry): Pt[] => e.v.flatMap((_, k) => (k % 2 ? [] : [{ x: e.v[k], y: e.v[k + 1] }]));

/** Whether p is inside the polygon (even-odd rule). */
function insidePolygon(p: Pt, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [a, b] = [poly[i], poly[j]];
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** `noteColor`：说明文字的颜色（默认与框同色；结果过期时使用错误色）。 */
function drawQuad(f: Frame, pts: Pt[], faint: boolean, note: string, noteColor = CORNER_COLOR) {
  const { ctx, at } = f;
  const xy = pts.map((q) => [at.x + q.x * at.s, at.y + q.y * at.s]);
  ctx.save();
  ctx.globalAlpha = faint ? 0.45 : 1;
  ctx.lineWidth = f.line;
  ctx.strokeStyle = ctx.fillStyle = CORNER_COLOR;
  if (faint) ctx.setLineDash([6, 4]);
  ctx.beginPath();
  xy.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = faint ? 0.1 : 0.14;
  ctx.fill();
  ctx.globalAlpha = faint ? 0.45 : 1;
  xy.forEach(([x, y], k) => {
    ctx.strokeRect(x - 5, y - 5, 10, 10);
    ctx.fillText(String(k + 1), x + 8, y - 7);
  });
  if (note) { ctx.fillStyle = noteColor; ctx.fillText(note, xy[0][0], xy[0][1] - 20); }
  ctx.restore();
}

/** `tracked`: this node's own result, when it has one (a planar track's outline per frame, overlays.ts drawTracks);
 * `stale`: that result predates the parameter change (view/plan.ts `stale`), so it shows the last tracked position rather than the current one. */
export function drawHandle(h: HandleDef, f: Frame, values: string[], drag: Drag | null, tracked: TracksData | null = null, stale = false) {
  const { ctx, at, frame } = f;
  ctx.font = FONT;
  if (h.kind === "points") {
    values.forEach((text, i) => {
      const e = parse(text);
      if (e.frame !== frame) return;
      const label = e.v[2] ?? 0;
      // the one being dragged follows the pointer (the entry is only rewritten when the drag ends)
      const at_ = drag && drag.point === i ? drag.to : { x: e.v[0], y: e.v[1] };
      const [cx, cy] = [at.x + at_.x * at.s, at.y + at_.y * at.s];
      ctx.strokeStyle = ctx.fillStyle = POINT_COLORS[label] ?? POINT_COLORS[0];
      ctx.lineWidth = f.line;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
        ctx.moveTo(cx + dx * 3, cy + dy * 3);
        ctx.lineTo(cx + dx * 10, cy + dy * 10);
      }
      ctx.stroke();
      ctx.fillText(h.labels.length ? h.labels[label] ?? "" : `user_${String(i + 1).padStart(2, "0")}`, cx + 9, cy - 8);
    });
  } else if (h.kind === "box") {
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = f.line;
    ctx.strokeStyle = DRAWN_COLOR;
    for (const text of values) {
      const e = parse(text);
      if (e.frame !== frame) continue;
      ctx.strokeRect(at.x + e.v[0] * at.s, at.y + e.v[1] * at.s, (e.v[2] - e.v[0]) * at.s, (e.v[3] - e.v[1]) * at.s);
    }
    if (drag) {
      ctx.strokeStyle = "#FFFFFF";
      ctx.strokeRect(at.x + drag.from.x * at.s, at.y + drag.from.y * at.s, (drag.to.x - drag.from.x) * at.s, (drag.to.y - drag.from.y) * at.s);
    }
    ctx.setLineDash([]);
  } else if (h.kind === "corners") {
    // the quad on its own frame; faint on the other frames, labelled with the frame it belongs to (where tracking starts)
    const e = values.length ? parse(values[0]) : null;
    if (drag && drag.corner === undefined) {
      const [a, b] = [drag.from, drag.to];
      drawQuad(f, [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }], false, "");
    } else if (e) {
      const pts = quad(e);
      if (drag?.corner !== undefined && e.frame === frame) pts[drag.corner] = drag.to;
      // 计算出结果后，其他帧上的手柄跟随结果（与 Nuke 平面跟踪的做法相同）：起始帧以外的帧画面已经移动，
      // 若仍按起始帧的像素位置淡色绘制，框下方已是其他内容。有该帧跟踪结果时绘制在结果位置，只有起始帧可拖动
      const followed = e.frame !== frame && tracked?.outline ? tracked.outline[tracked.frames.indexOf(frame)] : undefined;
      // 结果过期时仍跟随绘制，但须明确标示：参数已修改但尚未重算时绘制的是上一次跟踪到的位置，以错误色标注，避免将旧结果误认为新结果
      if (followed && stale) drawQuad(f, followed.corners.map(([x, y]) => ({ x, y })), true, `四个角在第 ${e.frame} 帧画的，上一次跟踪到的位置（参数改过，结果已过期）`, ERROR_COLOR);
      else if (followed) drawQuad(f, followed.corners.map(([x, y]) => ({ x, y })), true, `四个角在第 ${e.frame} 帧画的，这是跟踪到的位置`);
      else drawQuad(f, pts, e.frame !== frame, e.frame !== frame ? `四个角在第 ${e.frame} 帧` : "");
    }
  } else if (h.kind === "canvas") {
    // every outline is shown on every frame, with a faint fill and a solid edge: a hand-drawn matte applies to the whole
    // shot (nodes/core/roto.py), so what is on screen is exactly the mask
    ctx.save();
    ctx.lineWidth = f.line;
    ctx.strokeStyle = ctx.fillStyle = DRAWN_COLOR;
    for (const pts of [...values.map((text) => polygon(parse(text))), ...(drag?.path ? [drag.path] : [])]) {
      if (pts.length < 2) continue;
      ctx.beginPath();
      pts.forEach((q, k) => (k ? ctx.lineTo(at.x + q.x * at.s, at.y + q.y * at.s) : ctx.moveTo(at.x + q.x * at.s, at.y + q.y * at.s)));
      ctx.closePath();
      ctx.globalAlpha = 0.18;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
    ctx.restore();
  } else if (h.kind === "figure") {
    // 当前帧的火柴人实线绘制（并标注正在拖动的关节名称）；另外以淡色叠加前一帧和后一帧各一个，
    // 即洋葱皮，与二维动画软件中的做法相同，用于逐帧检查连续性。
    // 不以淡色绘制所有帧：绘制十余帧后会有十余个人形重叠，反而无法判断连续性。
    const dragged = drag && drag.joint !== undefined ? drag : null;
    const all = figureFrames(values);
    const prev = all.filter((k) => k < frame).pop();
    const next = all.find((k) => k > frame);
    const noteOf = (at: number) => (at === prev ? `上一帧 ${at}` : at === next ? `下一帧 ${at}` : "");
    // 先绘制淡色，再绘制实线：当前帧的人形始终位于最上层，指针抓取的也是它
    const order = values.map(parse).filter((e) => e.frame === prev || e.frame === next || e.frame === frame)
                        .sort((a, b) => Number(a.frame === frame) - Number(b.frame === frame));
    for (const e of order) {
      let pts = joints(e);
      if (dragged && e.frame === frame)
        pts = dragged.whole
          ? pts.map((q) => ({ x: q.x + dragged.to.x - dragged.from.x, y: q.y + dragged.to.y - dragged.from.y }))
          : pts.map((q, k) => (k === dragged.joint ? dragged.to : q));
      drawFigure(f, pts, e.frame !== frame, noteOf(e.frame), h.labels,
                 dragged && e.frame === frame ? (dragged.whole ? 0 : dragged.joint!) : -1,
                 e.frame === next ? 1 : 0);
    }
  }
  // "person": its picks are shown as the chosen people's boxes (the overlay of the node's input)
}

export function personAt(boxes: BoxesData | null, frame: number, p: Pt): number | null {
  const hits = (boxes?.people ?? []).flatMap((t) => {
    const b = t.boxes?.[String(frame)];
    return b && p.x >= b[0] && p.x <= b[2] && p.y >= b[1] && p.y <= b[3] ? [{ id: t.id, area: (b[2] - b[0]) * (b[3] - b[1]) }] : [];
  });
  return hits.sort((a, b) => a.area - b.area)[0]?.id ?? null; // the smallest box: the person in front
}

/** A click: the new entries, or null when it adds nothing. `label`: the points label chosen in the toolbar. */
export function clickHandle(h: HandleDef, values: string[], frame: number, p: Pt, label: number, source: BoxesData | null): string[] | null {
  if (h.kind === "points") return [...values, `${frame}:${p.x.toFixed(1)},${p.y.toFixed(1)}${h.labels.length ? `,${label}` : ""}`];
  if (h.kind === "person") return personAt(source, frame, p) === null ? null : [...values, `${frame}:${Math.round(p.x)},${Math.round(p.y)}`];
  return null;
}

/** Where a drag of this handle starts: on a corner of the quad of this frame (moves it), else a new box or quad;
 * null where nothing can be dragged. */
export function dragStart(h: HandleDef, values: string[], frame: number, p: Pt & { inside: boolean }, scale: number): Drag | null {
  if (h.kind === "points") {
    // 点偏后可拖回，无需删除重点。抓取当前帧上距离指针最近的点，
    // 与「四个角」抓取角点、火柴人抓取关节的机制相同：手柄是视图组件，使用它的节点无需任何修改
    const i = values.findIndex((text) => {
      const e = parse(text);
      return e.frame === frame && near({ x: e.v[0], y: e.v[1] }, p, scale);
    });
    return i < 0 ? null : { from: p, to: p, point: i };
  }
  if (h.kind === "corners" && values.length) {
    const e = parse(values[0]);
    const corner = e.frame === frame ? quad(e).findIndex((q) => near(q, p, scale)) : -1;
    if (corner >= 0) return { from: p, to: p, corner };
  }
  if (h.kind === "canvas") return p.inside ? { from: p, to: p, path: [{ x: p.x, y: p.y }] } : null;
  if (h.kind === "figure") {
    // 只抓取当前帧火柴人的关节，拖动不创建任何内容（拖框创建会使人体比例失真）。
    // 添加姿势通过参数面板的「添加帧」，比例由 FIGURE_TPOSE 表决定
    const here = values.map(parse).find((e) => e.frame === frame);
    if (!here) return null;
    const j = joints(here).findIndex((q) => near(q, p, scale));
    return j < 0 ? null : { from: p, to: p, joint: j, whole: j === 0 }; // the pelvis moves the whole body
  }
  return (h.kind === "box" || h.kind === "corners") && p.inside ? { from: p, to: p } : null;
}

/** The pointer moved during a drag: the drag's current state. A freehand outline (canvas) adds another point once the
 * pointer has moved far enough, so the entry holds the drawn line rather than every pointer event. */
export function moveDrag(h: HandleDef, drag: Drag, p: Pt): Drag {
  if (h.kind !== "canvas" || !drag.path) return { ...drag, to: p };
  const last = drag.path[drag.path.length - 1];
  return Math.hypot(p.x - last.x, p.y - last.y) < CANVAS_STEP ? { ...drag, to: p } : { ...drag, to: p, path: [...drag.path, { x: p.x, y: p.y }] };
}

/** A drag finished: the new entries, or null for a drag too small to form a box. A box adds one entry; a quad (corners)
 * replaces the plane's, its corners ordered clockwise from the top-left; dragging a corner moves that corner. */
export function dragHandle(h: HandleDef, values: string[], frame: number, drag: Drag): string[] | null {
  const { from, to } = drag;
  if (h.kind === "points") {
    if (drag.point === undefined) return null;
    const e = parse(values[drag.point]);
    const label = e.v.length > 2 ? `,${e.v[2]}` : "";
    return values.map((text, i) => (i === drag.point ? `${e.frame}:${to.x.toFixed(1)},${to.y.toFixed(1)}${label}` : text));
  }
  if (h.kind === "canvas") {
    const pts = drag.path ?? [];
    if (pts.length < 3) return null; // a click or a flick: no area, so nothing is drawn
    return [...values, `${frame}:${pts.map((q) => `${q.x.toFixed(1)},${q.y.toFixed(1)}`).join(",")}`];
  }
  if (h.kind === "figure") {
    // 拖动只调整姿势，从不创建：若拖动创建新火柴人，同一帧上已有的火柴人会被静默覆盖
    const at = values.findIndex((text) => parse(text).frame === frame);
    if (drag.joint === undefined || at < 0) return null;
    const pts = joints(parse(values[at]));
    const moved = drag.whole
      ? pts.map((q) => ({ x: q.x + to.x - from.x, y: q.y + to.y - from.y }))
      : pts.map((q, k) => (k === drag.joint ? to : q));
    return values.map((text, i) => (i === at ? figureEntry(frame, moved) : text));
  }
  if (h.kind === "corners" && drag.corner !== undefined) {
    const e = parse(values[0]);
    const v = [...e.v];
    [v[2 * drag.corner], v[2 * drag.corner + 1]] = [Number(to.x.toFixed(1)), Number(to.y.toFixed(1))];
    return [`${e.frame}:${v.join(",")}`, ...values.slice(1)];
  }
  if ((h.kind !== "box" && h.kind !== "corners") || Math.abs(to.x - from.x) < 4 || Math.abs(to.y - from.y) < 4) return null;
  const [x1, x2] = [Math.min(from.x, to.x), Math.max(from.x, to.x)].map(Math.round);
  const [y1, y2] = [Math.min(from.y, to.y), Math.max(from.y, to.y)].map(Math.round);
  if (h.kind === "corners") return [`${frame}:${x1},${y1},${x2},${y1},${x2},${y2},${x1},${y2}`];
  return [...values, `${frame}:${x1},${y1},${x2},${y2}`];
}

/** A right click: the entries without the one under the pointer on this frame (null: nothing there). */
export function removeAt(h: HandleDef, values: string[], frame: number, p: Pt, scale: number): string[] | null {
  const i = values.findIndex((text) => {
    const e = parse(text);
    if (h.kind === "canvas") return insidePolygon(p, polygon(e)); // an outline applies to every frame: it can be removed from any frame
    if (e.frame !== frame) return false;
    if (h.kind === "figure") {
      const b = figureBox(joints(e));
      return p.x >= b.x1 - 8 && p.x <= b.x2 + 8 && p.y >= b.y1 - 8 && p.y <= b.y2 + 8;
    }
    if (h.kind === "box") return p.x >= e.v[0] && p.x <= e.v[2] && p.y >= e.v[1] && p.y <= e.v[3];
    if (h.kind === "corners") return insidePolygon(p, quad(e));
    return near({ x: e.v[0], y: e.v[1] }, p, scale);
  });
  return i < 0 ? null : values.filter((_, k) => k !== i);
}

/** The pointer's action with this handle, for the cursor and the toolbar hint. */
export const HANDLE_HINT: Record<HandleDef["kind"], string> = {
  // 每条提示都须在窄窗口中放得下：该文字绘制在视图底部的药丸中，`white-space: nowrap`，
  // 放不下时不得换行或使用省略号，只能换用更短的措辞：最长一条约 330 px，1100 宽的窗口也能容纳。
  // 由 tests/ui/walk_text.py 检查（视图上药丸的 scrollWidth 不得超过 clientWidth）。
  points: "点一下加一个点 · 拖动调位置 · 右键删掉",
  box: "拖出一个框，右键框里删掉",
  canvas: "按住左键拖一圈圈出范围，松手闭合 · 右键删掉",
  figure: "拖关节摆姿势 · 拖髋整体移动 · 右键删掉",
  corners: "拖框当平面 · 四个角各拖到位 · 右键框里删掉",
  person: "点画面里的人选中（点在人物框里）",
  transform: "拖动手柄移动、旋转、缩放",
};
