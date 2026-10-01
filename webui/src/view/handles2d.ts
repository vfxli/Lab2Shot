import type { BoxesData, HandleDef, TracksData } from "../api";
import type { Frame } from "./overlays";
import { ERROR_COLOR } from "../platform/palette";
import { CORNER_COLOR, type Entry, type Pt, parse } from "./handleParts";
import { personAt } from "../model/people";
// all of the stick-figure handle's logic lives in view/figure2d.ts
import { drawFigure, figureBox, figureEntry, figureFrames, joints } from "./figure2d";

export type { Pt };
export { personAt };
export { addFigure, figureAdd, figureFrames } from "./figure2d";

/** The 2D handles (nodes/handles.py): one tool per kind, bound to a list parameter of the node. A tool draws the
 * parameter's entries on the current frame and turns clicks and drags into new entries; a right click removes the
 * entry under the pointer. Entries are "frame:x,y[,label]" (points, picks), "frame:x1,y1,x2,y2" (boxes),
 * "frame:x1,y1,...,x4,y4" (a plane's corners, in order around it) and "frame:x1,y1,..." with as many points as were
 * drawn (canvas: one closed outline each), image pixels from the top-left corner. */

const FONT = "600 11px -apple-system, 'PingFang SC', 'Microsoft YaHei UI', sans-serif";
const POINT_COLORS = ["#FF8FC7", ERROR_COLOR]; // the first label (a point to track, "this"), the second ("not this")
/** The 2D handles' one hit rule, in two forms: what the pointer is on is the nearest point within 10 screen pixels
 * (`pointUnder`), or of the shapes containing it the smallest (`smallest`: a box drawn inside another, an outline inside an
 * outline); never the first in the list. Grabbing and right-click removing both use them. Index; -1 none; `null`
 * entries are not candidates. */
function pointUnder(pts: readonly (Pt | null)[], p: Pt, scale: number): number {
  let best = -1, d = Infinity;
  pts.forEach((q, i) => {
    if (!q) return;
    const di = Math.hypot(q.x - p.x, q.y - p.y) * scale;
    if (di < 10 && di < d) (best = i), (d = di);
  });
  return best;
}
function smallest(shapes: readonly ({ inside: boolean; area: number } | null)[]): number {
  let best = -1, a = Infinity;
  shapes.forEach((s, i) => {
    if (s?.inside && s.area < a) (best = i), (a = s.area);
  });
  return best;
}
/** The size `smallest` compares, for every kind of shape: the area of its bounding box (a box and a figure are measured
 * that way anyway). Not the polygon's own area: a self-crossing outline (a figure eight) has lobes that cancel in the
 * shoelace sum, while `insidePolygon` counts the pointer inside either lobe, so the two would not be about the same shape. */
const areaOf = (pts: readonly Pt[]): number => {
  if (!pts.length) return 0;
  const xs = pts.map((q) => q.x), ys = pts.map((q) => q.y);
  return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
};
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

/** Where a grabbed point is during / after a drag: where it was, moved by the pointer's travel since the press (grabbing
 * it a few pixels off does not make it jump under the pointer; the same rule as the 3D handles' increment). */
const moved = (q: Pt, d: Drag): Pt => ({ x: q.x + d.to.x - d.from.x, y: q.y + d.to.y - d.from.y });

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

function drawQuad(f: Frame, pts: Pt[], faint: boolean, note: string) {
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
  if (note) { ctx.fillStyle = CORNER_COLOR; ctx.fillText(note, xy[0][0], xy[0][1] - 20); }
  ctx.restore();
}

/** Where a four-corner entry is on `frame`, as drawn and as hit (drawHandle, removeAt): on its own frame where it was
 * drawn; on another frame, once there is a tracked result for it, where the result is (as Nuke's planar tracker does:
 * away from the start frame the picture has moved, so the start frame's pixel positions would sit over something else);
 * else at the start frame's positions, faint. Only the start frame can be dragged. */
function quadShown(e: Entry, frame: number, tracked: TracksData | null): { pts: Pt[]; followed: boolean } {
  const at = e.frame !== frame && tracked?.outline ? tracked.outline[tracked.frames.indexOf(frame)] : undefined;
  return at ? { pts: at.corners.map(([x, y]) => ({ x, y })), followed: true } : { pts: quad(e), followed: false };
}

/** `tracked`: this node's own result, when it is current (a planar track's outline per frame, overlays.ts drawTracks;
 * view/plan.ts underHandles leaves a result out while it does not match the handle). */
export function drawHandle(h: HandleDef, f: Frame, values: string[], drag: Drag | null, tracked: TracksData | null = null) {
  const { ctx, at, frame } = f;
  ctx.font = FONT;
  if (h.kind === "points") {
    values.forEach((text, i) => {
      const e = parse(text);
      if (e.frame !== frame) return;
      const label = e.v[2] ?? 0;
      // the one being dragged follows the pointer (the entry is only rewritten when the drag ends)
      const at_ = drag && drag.point === i ? moved({ x: e.v[0], y: e.v[1] }, drag) : { x: e.v[0], y: e.v[1] };
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
      const shown = quadShown(e, frame, tracked);
      const pts = shown.pts;
      if (drag?.corner !== undefined && e.frame === frame) pts[drag.corner] = moved(pts[drag.corner], drag);
      if (shown.followed) drawQuad(f, pts, true, `四个角在第 ${e.frame} 帧画的，这是跟踪到的位置`);
      else drawQuad(f, pts, e.frame !== frame, e.frame !== frame ? `四个角在第 ${e.frame} 帧` : "");
    }
  } else if (h.kind === "canvas") {
    // every outline is shown on every frame, with a faint fill and a solid edge: a hand-drawn matte applies to the whole
    // shot (nodes/core/mask.py), so what is on screen is exactly the mask
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
    // The current frame's figure is drawn solid (with the name of the joint being dragged); the previous and the next
    // posed frame are laid over it faintly, one each — onion skin, as in 2D animation software, for checking continuity
    // frame by frame. Not every frame faintly: a dozen overlapping figures make continuity impossible to judge.
    const dragged = drag && drag.joint !== undefined ? drag : null;
    const all = figureFrames(values);
    const prev = all.filter((k) => k < frame).pop();
    const next = all.find((k) => k > frame);
    const noteOf = (at: number) => (at === prev ? `上一帧 ${at}` : at === next ? `下一帧 ${at}` : "");
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
  }
  // "person": its picks are shown as the chosen people's boxes (the overlay of the node's input, pickedPeople)
}

/** The people a "person" handle's picks point at: each pick ("frame:x,y") chooses the person whose box holds it
 * (model/people.ts personAt: the server's at_point rule, which the cook of 「选人」 runs). What the 2D stage lights on
 * the input's boxes while the node's own result does not match the picks (view/plan.ts underHandles): a display of the
 * parameters, not a computation. */
export function pickedPeople(boxes: BoxesData | null, picks: string[]): Set<number> {
  const out = new Set<number>();
  for (const text of picks) {
    const e = parse(text);
    const id = personAt(boxes, e.frame, { x: e.v[0], y: e.v[1] });
    if (id !== null) out.add(id);
  }
  return out;
}

/** A click: the new entries, or null when it adds nothing. `label`: the points label chosen in the toolbar. */
export function clickHandle(h: HandleDef, values: string[], frame: number, p: Pt, label: number, source: BoxesData | null): string[] | null {
  if (h.kind === "points") return [...values, `${frame}:${p.x.toFixed(1)},${p.y.toFixed(1)}${h.labels.length ? `,${label}` : ""}`];
  if (h.kind === "person") return personAt(source, frame, p) === null ? null : [...values, `${frame}:${Math.round(p.x)},${Math.round(p.y)}`];
  return null;
}

/** Where a drag of this handle starts: on a point, a corner of this frame's quad or a joint of this frame's figure
 * (moves it), a new freehand outline (canvas), else a new box or quad; null where nothing can be dragged. */
export function dragStart(h: HandleDef, values: string[], frame: number, p: Pt & { inside: boolean }, scale: number): Drag | null {
  if (h.kind === "points") {
    // A misplaced point can be dragged back instead of deleted and placed again. The point on the current frame nearest
    // the pointer is grabbed, by the same mechanism as the corners of 「四个角」 and a figure's joints: the handle is a
    // view component, and the nodes using it need nothing of their own
    const i = pointUnder(values.map((text) => { const e = parse(text); return e.frame === frame ? { x: e.v[0], y: e.v[1] } : null; }), p, scale);
    return i < 0 ? null : { from: p, to: p, point: i };
  }
  if (h.kind === "corners" && values.length) {
    const e = parse(values[0]);
    const corner = e.frame === frame ? pointUnder(quad(e), p, scale) : -1;
    if (corner >= 0) return { from: p, to: p, corner };
  }
  if (h.kind === "canvas") return p.inside ? { from: p, to: p, path: [{ x: p.x, y: p.y }] } : null;
  if (h.kind === "figure") {
    // Only the joints of the current frame's figure are grabbed; a drag creates nothing (creating by dragging a box
    // would distort the body's proportions). Poses are added with 「添加帧」 in the parameter panel, proportioned by
    // the FIGURE_TPOSE table
    const here = values.map(parse).find((e) => e.frame === frame);
    if (!here) return null;
    const j = pointUnder(joints(here), p, scale);
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
    const q = moved({ x: e.v[0], y: e.v[1] }, drag);
    return values.map((text, i) => (i === drag.point ? `${e.frame}:${q.x.toFixed(1)},${q.y.toFixed(1)}${label}` : text));
  }
  if (h.kind === "canvas") {
    const pts = drag.path ?? [];
    if (pts.length < 3) return null; // a click or a flick: no area, so nothing is drawn
    return [...values, `${frame}:${pts.map((q) => `${q.x.toFixed(1)},${q.y.toFixed(1)}`).join(",")}`];
  }
  if (h.kind === "figure") {
    // a drag only adjusts a pose and never creates one: creating a figure by dragging would silently replace the one already on this frame
    const at = values.findIndex((text) => parse(text).frame === frame);
    if (drag.joint === undefined || at < 0) return null;
    const pts = joints(parse(values[at]));
    const posed = pts.map((q, k) => (drag.whole || k === drag.joint ? moved(q, drag) : q));
    return values.map((text, i) => (i === at ? figureEntry(frame, posed) : text));
  }
  if (h.kind === "corners" && drag.corner !== undefined) {
    const e = parse(values[0]);
    const v = [...e.v];
    const q = moved({ x: v[2 * drag.corner], y: v[2 * drag.corner + 1] }, drag);
    [v[2 * drag.corner], v[2 * drag.corner + 1]] = [Number(q.x.toFixed(1)), Number(q.y.toFixed(1))];
    return [`${e.frame}:${v.join(",")}`, ...values.slice(1)];
  }
  if ((h.kind !== "box" && h.kind !== "corners") || Math.abs(to.x - from.x) < 4 || Math.abs(to.y - from.y) < 4) return null;
  const [x1, x2] = [Math.min(from.x, to.x), Math.max(from.x, to.x)].map(Math.round);
  const [y1, y2] = [Math.min(from.y, to.y), Math.max(from.y, to.y)].map(Math.round);
  if (h.kind === "corners") return [`${frame}:${x1},${y1},${x2},${y1},${x2},${y2},${x1},${y2}`];
  return [...values, `${frame}:${x1},${y1},${x2},${y2}`];
}

/** A right click: the entries without the one under the pointer on this frame (null: nothing there). */
export function removeAt(h: HandleDef, values: string[], frame: number, p: Pt, scale: number, source: BoxesData | null = null, tracked: TracksData | null = null): string[] | null {
  // 「选人」: what is drawn is the person's box, lit (the pick itself is not drawn): a right click in that box removes the
  // picks of that person on this frame
  if (h.kind === "person") {
    const id = personAt(source, frame, p);
    if (id === null) return null;
    const rest = values.filter((text) => { const e = parse(text); return !(e.frame === frame && personAt(source, frame, { x: e.v[0], y: e.v[1] }) === id); });
    return rest.length === values.length ? null : rest;
  }
  if (h.kind === "points") {
    const i = pointUnder(values.map((text) => { const e = parse(text); return e.frame === frame ? { x: e.v[0], y: e.v[1] } : null; }), p, scale);
    return i < 0 ? null : values.filter((_, k) => k !== i);
  }
  // the shapes as drawn, containing the pointer, the smallest taken (the inner of nested boxes / outlines); the margin
  // around a figure is 10 screen pixels, the one reach of the 2D handles (pointUnder)
  const reach = 10 / Math.max(scale, 1e-6);
  const i = smallest(values.map((text) => {
    const e = parse(text);
    if (h.kind === "canvas") { const pts = polygon(e); return { inside: insidePolygon(p, pts), area: areaOf(pts) }; } // an outline applies to every frame
    // a quad is drawn on every frame (its start frame solid, the others dashed or where it was tracked to): removable
    // from any of them, where it is drawn there (quadShown, the same as drawHandle)
    if (h.kind === "corners") { const q = quadShown(e, frame, tracked).pts; return { inside: insidePolygon(p, q), area: areaOf(q) }; }
    if (e.frame !== frame) return null;
    if (h.kind === "figure") {
      const b = figureBox(joints(e));
      return { inside: p.x >= b.x1 - reach && p.x <= b.x2 + reach && p.y >= b.y1 - reach && p.y <= b.y2 + reach, area: (b.x2 - b.x1 + 2 * reach) * (b.y2 - b.y1 + 2 * reach) };
    }
    // a box as drawn, whichever corner the entry starts from
    const [x1, x2] = [Math.min(e.v[0], e.v[2]), Math.max(e.v[0], e.v[2])];
    const [y1, y2] = [Math.min(e.v[1], e.v[3]), Math.max(e.v[1], e.v[3])];
    return { inside: p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2, area: (x2 - x1) * (y2 - y1) };
  }));
  return i < 0 ? null : values.filter((_, k) => k !== i);
}

/** The pointer's action with this handle, for the cursor and the toolbar hint. */
export const HANDLE_HINT: Record<HandleDef["kind"], string> = {
  // Every hint must fit a narrow window: it is drawn in the pill at the bottom of the view, `white-space: nowrap`,
  // and when it does not fit it may neither wrap nor use an ellipsis, only shorter wording: the longest is about
  // 330 px, which a 1100-wide window holds.
  points: "点一下加一个点 · 拖动调位置 · 右键删掉",
  box: "拖出一个框，右键框里删掉",
  canvas: "按住左键拖一圈圈出范围，松手闭合 · 右键删掉",
  figure: "拖关节摆姿势 · 拖髋整体移动 · 右键删掉",
  corners: "拖框当平面 · 四个角各拖到位 · 右键框里删掉",
  person: "点画面里的人选中（点在人物框里）",
  transform: "拖动手柄移动、旋转、缩放",
  skeleton_pose: "点关节选中 · 拖手柄转、移、缩放 · 右边面板填数、镜像、复位",
};
