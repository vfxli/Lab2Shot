/** 二维手柄工具的公共定义：每种手柄一个工具（`view/handles2d.ts` 的表 TOOLS_2D，每种一个文件：points2d / box2d /
 * corners2d / canvas2d / person2d / figure2d），这里是它们共用的东西：工具的接口、条目结构与解析、拖动状态、
 * 唯一的命中规则（pointUnder / smallest），以及几种手柄共用的颜色。
 */

import type { BoxesData, HandleDef, TracksData } from "../api";
import type { Frame } from "./overlays";

export interface Pt {
  x: number; // 图像像素坐标
  y: number;
}

export interface Entry {
  frame: number;
  v: number[];
}

/** 解析一条条目，格式为「帧:x1,y1,…」（语法见 `lab2shot/nodes/handles.py` HANDLE_KINDS）。 */
export const parse = (s: string): Entry => {
  const [f, rest] = s.split(":");
  return { frame: Number(f), v: rest.split(",").map(Number) };
};

/** 「先放置后调整」类手柄（平面四角、火柴人）统一使用的黄色，用于提示该手柄可拖动。 */
export const CORNER_COLOR = "#FFD60A";
/** 指针在画面上画出的东西（框、手画轮廓）的颜色。 */
export const DRAWN_COLOR = "#64D2FF";
export const FONT = "600 11px -apple-system, 'PingFang SC', 'Microsoft YaHei UI', sans-serif";

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

/** One kind of 2D handle (nodes/handles.py): everything the 2D stage, the viewer's toolbar and the parameter panel need
 * of it. The stage looks a handle's tool up in TOOLS_2D (view/handles2d.ts) and never names a kind. A part a kind does
 * not have is left out (no click, no drag...): the stage then does nothing for it.
 * - `hint`: the key of the words for the pointer's action (the pill at the bottom of the view);
 * - `draws`: the pointer draws on the empty picture (crosshair cursor); otherwise it acts only on what is under it
 *   (`people.at`: pointer cursor over a person);
 * - `labelled`: a click takes the label chosen in the view's toolbar (「点的种类」), from the handle's `labels`;
 * - `marksFrames`: each entry is a frame drawn (the timeline's ruler marks the frames that have one);
 * - `people`: its entries pick people on its input's boxes (「选人」): who is at a point, which people the picks light;
 *   the parameter panel lists such picks as people (editor/pickedPeople.tsx). */
export interface HandleTool2D {
  hint: string;
  draws: boolean;
  labelled?: boolean;
  marksFrames?: boolean;
  people?: {
    at: (source: BoxesData | null, frame: number, p: Pt) => number | null;
    lit: (source: BoxesData | null, picks: string[]) => Set<number>;
  };
  /** Draws the parameter's entries on the current frame (`tracked`: this node's own current result, a planar track). */
  draw?: (h: HandleDef, f: Frame, values: string[], drag: Drag | null, tracked: TracksData | null) => void;
  /** A click: the new entries, or null when it adds nothing. `label`: the label chosen in the toolbar. */
  click?: (h: HandleDef, values: string[], frame: number, p: Pt, label: number, source: BoxesData | null) => string[] | null;
  /** Where a drag starts (null: nothing can be dragged there). */
  start?: (h: HandleDef, values: string[], frame: number, p: Pt & { inside: boolean }, scale: number) => Drag | null;
  /** The pointer moved during a drag (default: only `to` follows it). */
  move?: (drag: Drag, p: Pt) => Drag;
  /** A drag finished: the new entries, or null. */
  finish?: (h: HandleDef, values: string[], frame: number, drag: Drag) => string[] | null;
  /** A right click: the entries without the one under the pointer on this frame (null: nothing there). */
  remove: (h: HandleDef, values: string[], frame: number, p: Pt, scale: number, source: BoxesData | null, tracked: TracksData | null) => string[] | null;
}

/** Where a grabbed point is during / after a drag: where it was, moved by the pointer's travel since the press (grabbing
 * it a few pixels off does not make it jump under the pointer; the same rule as the 3D handles' increment). */
export const moved = (q: Pt, d: Drag): Pt => ({ x: q.x + d.to.x - d.from.x, y: q.y + d.to.y - d.from.y });

/** The 2D handles' one hit rule, in two forms: what the pointer is on is the nearest point within 10 screen pixels
 * (`pointUnder`), or of the shapes containing it the smallest (`smallest`: a box drawn inside another, an outline inside an
 * outline); never the first in the list. Grabbing and right-click removing both use them. Index; -1 none; `null`
 * entries are not candidates. */
export function pointUnder(pts: readonly (Pt | null)[], p: Pt, scale: number): number {
  let best = -1, d = Infinity;
  pts.forEach((q, i) => {
    if (!q) return;
    const di = Math.hypot(q.x - p.x, q.y - p.y) * scale;
    if (di < 10 && di < d) (best = i), (d = di);
  });
  return best;
}
export function smallest(shapes: readonly ({ inside: boolean; area: number } | null)[]): number {
  let best = -1, a = Infinity;
  shapes.forEach((s, i) => {
    if (s?.inside && s.area < a) (best = i), (a = s.area);
  });
  return best;
}
/** A right click on shapes: the entries without the smallest one containing the pointer (`shape`: an entry as drawn and
 * hit on this frame, null when it is not a candidate); null when the pointer is in none. */
export function removeSmallest(values: string[], shape: (e: Entry) => { inside: boolean; area: number } | null): string[] | null {
  const i = smallest(values.map((text) => shape(parse(text))));
  return i < 0 ? null : values.filter((_, k) => k !== i);
}
/** The size `smallest` compares, for every kind of shape: the area of its bounding box (a box and a figure are measured
 * that way anyway). Not the polygon's own area: a self-crossing outline (a figure eight) has lobes that cancel in the
 * shoelace sum, while `insidePolygon` counts the pointer inside either lobe, so the two would not be about the same shape. */
export const areaOf = (pts: readonly Pt[]): number => {
  if (!pts.length) return 0;
  const xs = pts.map((q) => q.x), ys = pts.map((q) => q.y);
  return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
};

/** An entry's points: a quad's four, an outline's as many as were drawn. */
export const polygon = (e: Entry): Pt[] => e.v.flatMap((_, k) => (k % 2 ? [] : [{ x: e.v[k], y: e.v[k + 1] }]));

/** Whether p is inside the polygon (even-odd rule). */
export function insidePolygon(p: Pt, poly: Pt[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [a, b] = [poly[i], poly[j]];
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** The entries of a points-like handle on this frame, as points (null: on another frame): what pointUnder grabs. */
export const pointsOn = (values: string[], frame: number): (Pt | null)[] =>
  values.map((text) => { const e = parse(text); return e.frame === frame ? { x: e.v[0], y: e.v[1] } : null; });
