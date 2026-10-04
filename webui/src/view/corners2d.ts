import type { TracksData } from "../api";
import type { Frame } from "./overlays";
import { t } from "../i18n/t";
import { CORNER_COLOR, FONT, areaOf, insidePolygon, moved, parse, pointUnder, removeSmallest, type Entry, type HandleTool2D, type Pt } from "./handleParts";
import { dragRect } from "./box2d";

/** 平面四角（「corners」，nodes/handles.py）：一个平面一项「帧:x1,y1,…,x4,y4」（绕平面一圈的顺序）。拖出一个框放下四角
 * （顺时针、从左上起），之后拖角挪动；平面跟踪从这一帧开始，其他帧上画成虚线（有跟踪结果时画在跟到的位置）。 */

const quad = (e: Entry): Pt[] => [0, 1, 2, 3].map((k) => ({ x: e.v[2 * k], y: e.v[2 * k + 1] }));

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

/** Where a four-corner entry is on `frame`, as drawn and as hit (draw, remove): on its own frame where it was drawn; on
 * another frame, once there is a tracked result for it, where the result is (as Nuke's planar tracker does: away from
 * the start frame the picture has moved, so the start frame's pixel positions would sit over something else); else at
 * the start frame's positions, faint. Only the start frame can be dragged. */
function quadShown(e: Entry, frame: number, tracked: TracksData | null): { pts: Pt[]; followed: boolean } {
  const at = e.frame !== frame && tracked?.outline ? tracked.outline[tracked.frames.indexOf(frame)] : undefined;
  return at ? { pts: at.corners.map(([x, y]) => ({ x, y })), followed: true } : { pts: quad(e), followed: false };
}

export const cornersTool: HandleTool2D = {
  hint: "ui.view.hint.corners",
  draws: true,
  // the quad on its own frame; faint on the other frames, labelled with the frame it belongs to (where tracking starts)
  draw: (_h, f, values, drag, tracked) => {
    const { frame } = f;
    f.ctx.font = FONT;
    const e = values.length ? parse(values[0]) : null;
    if (drag && drag.corner === undefined) {
      const [a, b] = [drag.from, drag.to];
      drawQuad(f, [a, { x: b.x, y: a.y }, b, { x: a.x, y: b.y }], false, "");
    } else if (e) {
      const shown = quadShown(e, frame, tracked);
      const pts = shown.pts;
      if (drag?.corner !== undefined && e.frame === frame) pts[drag.corner] = moved(pts[drag.corner], drag);
      if (shown.followed) drawQuad(f, pts, true, t("ui.view.corners_tracked", { frame: e.frame }));
      else drawQuad(f, pts, e.frame !== frame, e.frame !== frame ? t("ui.view.corners_on_frame", { frame: e.frame }) : "");
    }
  },
  // a corner of this frame's quad is moved; elsewhere inside the picture a new quad is dragged out
  start: (_h, values, frame, p, scale) => {
    if (values.length) {
      const e = parse(values[0]);
      const corner = e.frame === frame ? pointUnder(quad(e), p, scale) : -1;
      if (corner >= 0) return { from: p, to: p, corner };
    }
    return p.inside ? { from: p, to: p } : null;
  },
  // dragging a corner moves that corner; a new quad replaces the plane's, its corners clockwise from the top-left
  finish: (_h, values, frame, drag) => {
    if (drag.corner !== undefined) {
      const e = parse(values[0]);
      const v = [...e.v];
      const q = moved({ x: v[2 * drag.corner], y: v[2 * drag.corner + 1] }, drag);
      [v[2 * drag.corner], v[2 * drag.corner + 1]] = [Number(q.x.toFixed(1)), Number(q.y.toFixed(1))];
      return [`${e.frame}:${v.join(",")}`, ...values.slice(1)];
    }
    const r = dragRect(drag);
    return r ? [`${frame}:${r.x1},${r.y1},${r.x2},${r.y1},${r.x2},${r.y2},${r.x1},${r.y2}`] : null;
  },
  // a quad is drawn on every frame (its start frame solid, the others dashed or where it was tracked to): removable
  // from any of them, where it is drawn there (quadShown, the same as draw)
  remove: (_h, values, frame, p, _scale, _source, tracked) =>
    removeSmallest(values, (e) => { const q = quadShown(e, frame, tracked).pts; return { inside: insidePolygon(p, q), area: areaOf(q) }; }),
};
