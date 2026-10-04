import { drawFigure, FIGURE_COUNT } from "./figure2d";
import type { BoxesData, TracksData } from "../api";
import { ACCENT_COLOR, ERROR_COLOR } from "../platform/palette";
import { type Bg } from "../model/view2d";
import { CHECKER_DARK, CHECKER_LIGHT } from "../platform/palette";
import { personTint } from "../model/viewOptions";
import { t as word } from "../i18n/t"; // `t` is a track here

/** Overlays drawn by the 2D stage in addition to the picture's own frame: people boxes and tracked points (in the
 * picture's pixel space; `at` maps image pixels to the canvas) and the transparency checkerboard.
 *
 * Drawing depends only on the channel count and the packet's own meta, never on whether a channel holds depth, a
 * mask or confidence: the meaning of a channel is decided by its consumer, and the data does not declare it.
 *
 * The per-pixel look of the picture itself (channel selection, black/white points, tinting, compositing) is not
 * handled here; view/look.ts computes it on the GPU every frame. */

export interface Frame {
  ctx: CanvasRenderingContext2D;
  frame: number;
  at: { x: number; y: number; s: number }; // canvas = x + pixel * s
  width: number; // the picture's size in pixels
  height: number;
  quiet: boolean; // the input a handle works on: drawn quieter than the node's own results
  line: number; // line width in pixels (the display option "线宽", shared with the 3D stage)
}

const FONT = "600 11px -apple-system, 'PingFang SC', 'Microsoft YaHei UI', sans-serif";
const PEOPLE = "#BF5AF2";

/** A colour at the quiet overlay's strength (a handle's input, not lit). */
const dimmed = (hex: string, alpha = 0.55): string =>
  `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(hex.slice(5, 7), 16)},${alpha})`;

/** People boxes. `chosen` (the input of a 「选人」 whose picks point at these people, or the node's own result): who is
 * picked, drawn so the difference is plain at a glance — the picked in the accent colour (ACCENT_COLOR = --accent), a
 * solid thicker line and a solid label with white text; the others thin and half transparent, their label on a dark
 * translucent chip so the number stays readable. Hover lifts one to full strength (not the accent: it is not picked yet).
 * Without `chosen` (a node's own boxes) every box is drawn the same, in the people colour or per person. */
export function drawBoxes(f: Frame, data: BoxesData, hover: number | null, chosen: Set<number> | null, byPerson = false) {
  const { ctx, at, frame } = f;
  for (const t of data.people) {
    const b = t.boxes?.[String(frame)];
    if (!b) continue;
    // "按人物" mode: each person gets its own colour from the overlay palette
    const base = byPerson ? personTint(t.id) : PEOPLE;
    const picked = !!chosen?.has(t.id);
    const hovered = hover === t.id;
    const other = !!chosen && !picked && !hovered; // picking, and this one is not picked: pushed back
    const color = picked ? ACCENT_COLOR : hovered ? base : other || f.quiet ? dimmed(base, other ? 0.4 : 0.55) : base;
    const [x1, y1, x2, y2] = [at.x + b[0] * at.s, at.y + b[1] * at.s, at.x + b[2] * at.s, at.y + b[3] * at.s];
    ctx.lineWidth = picked ? f.line * 2.4 : hovered ? f.line * 1.6 : other ? Math.max(1, f.line * 0.75) : f.line;
    ctx.strokeStyle = color;
    ctx.beginPath();
    ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 6);
    ctx.stroke();
    const label = picked ? word("ui.view.person_picked", { id: t.id }) : word("ui.view.person", { id: t.id });
    ctx.font = FONT;
    const w = ctx.measureText(label).width + 14;
    const ly = Math.max(at.y + 2, y1 - 22);
    // the label: solid colour with white text; a pushed-back one on a dark chip, its text still near white
    ctx.fillStyle = other ? "rgba(0, 0, 0, 0.6)" : color;
    ctx.beginPath();
    ctx.roundRect(x1, ly, w, 18, 9);
    ctx.fill();
    ctx.fillStyle = other ? "rgba(255, 255, 255, 0.82)" : "#fff";
    ctx.fillText(label, x1 + 7, ly + 13);
  }
}

const TRAIL = 12; // frames of history drawn behind each tracked point (fewer when there are many points)

/** 2D tracks: each point's trail over the last frames and its dot, each point in its own colour. */
export function drawTracks(f: Frame, data: TracksData) {
  const { ctx, at, frame } = f;
  const fi = data.frames.indexOf(frame);
  // A stick figure (declared by the packet's `figure` field) is drawn as a body.
  //
  // A sketch stores one track per joint. Drawing it as tracked points would join key poses on adjacent frames with
  // long lines; trails are meaningful for tracker output (to show drift and loss) but not for key poses, which jump
  // from one to the next. Only the current frame's pose is drawn, without trails.
  if (data.figure) {
    const pts = data.points.map((p) => p[fi]);
    // A pose requires all joints (the same criterion as `adapters/sketch2anim/nodes.py figures_in`).
    if (fi >= 0 && pts.length === FIGURE_COUNT && pts.every(Boolean))
      drawFigure(f, pts.map((q) => ({ x: q![0], y: q![1] })), false, "", []);
    return;
  }
  const n = data.points.length;
  const quad = data.outline?.[fi];
  if (quad) {
    // a planar track: its outline, solid where the plane was seen, dashed in the error colour where it is only estimated
    ctx.save();
    ctx.lineWidth = f.line * 1.2;
    ctx.strokeStyle = ctx.fillStyle = quad.seen ? "#FFD60A" : ERROR_COLOR;
    if (!quad.seen) ctx.setLineDash([6, 4]);
    ctx.beginPath();
    quad.corners.forEach(([x, y], k) => (k ? ctx.lineTo(at.x + x * at.s, at.y + y * at.s) : ctx.moveTo(at.x + x * at.s, at.y + y * at.s)));
    ctx.closePath();
    ctx.stroke();
    if (!quad.seen) ctx.fillText(word("ui.view.plane_unconfirmed"), at.x + quad.corners[0][0] * at.s, at.y + quad.corners[0][1] * at.s - 8);
    ctx.restore();
  }
  const trail = n > 200 ? 3 : n > 50 ? 6 : TRAIL;
  ctx.lineWidth = f.line * (n > 200 ? 0.55 : 0.8);
  const dot = Math.max(1, f.line / 1.5) * (n > 200 ? 1.8 : 2.8);
  data.points.forEach((pts, i) => {
    const color = `hsl(${(i * 137.5) % 360} 85% 62%)`;
    ctx.strokeStyle = color;
    ctx.beginPath();
    let drawing = false;
    for (let k = Math.max(0, fi - trail); k <= fi; k++) {
      const q = pts[k];
      if (!q) {
        drawing = false;
        continue;
      }
      const [px, py] = [at.x + q[0] * at.s, at.y + q[1] * at.s];
      if (drawing) ctx.lineTo(px, py);
      else ctx.moveTo(px, py);
      drawing = true;
    }
    ctx.stroke();
    const q = pts[fi];
    if (q) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(at.x + q[0] * at.s, at.y + q[1] * at.s, dot, 0, Math.PI * 2);
      ctx.fill();
    }
  });
}

// ------------------------------------------------------------------ background

const CHECKER_PX = 12; // screen pixels per square, so it reads as "transparent" at any zoom without rescaling

let checkerTile: HTMLCanvasElement | null = null;

function checkerboardTile(): HTMLCanvasElement {
  if (checkerTile) return checkerTile;
  const tile = document.createElement("canvas");
  tile.width = tile.height = CHECKER_PX * 2;
  const g = tile.getContext("2d")!;
  g.fillStyle = CHECKER_DARK;
  g.fillRect(0, 0, CHECKER_PX * 2, CHECKER_PX * 2);
  g.fillStyle = CHECKER_LIGHT;
  g.fillRect(0, 0, CHECKER_PX, CHECKER_PX);
  g.fillRect(CHECKER_PX, CHECKER_PX, CHECKER_PX, CHECKER_PX);
  checkerTile = tile;
  return tile;
}

/** Draws the background beneath the picture: a checkerboard (squares of fixed screen size at any zoom) or a solid
 * colour. It is drawn in all three modes, independent of processing, so that an imported image with alpha shows
 * transparency in the "仅原图" mode as well. */
export function drawBackground(ctx: CanvasRenderingContext2D, bg: Bg, colour: string, x: number, y: number, w: number, h: number) {
  ctx.save();
  ctx.fillStyle = bg === "solid" ? colour : ctx.createPattern(checkerboardTile(), "repeat")!;
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}
