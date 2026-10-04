import { ERROR_COLOR } from "../platform/palette";
import { FONT, moved, parse, pointUnder, pointsOn, type HandleTool2D } from "./handleParts";

/** 点手柄（「points」，nodes/handles.py）：每次点击一项「帧:x,y[,标签]」，标签是工具条上选的「点的种类」（主体 / 排除）；
 * 拖动挪动当前帧上的点，右键去掉指针下的点。 */

const POINT_COLORS = ["#FF8FC7", ERROR_COLOR]; // the first label (a point to track, "this"), the second ("not this")

export const pointsTool: HandleTool2D = {
  hint: "ui.view.hint.points",
  draws: true,
  labelled: true,
  draw: (h, f, values, drag) => {
    const { ctx, at, frame } = f;
    ctx.font = FONT;
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
  },
  click: (h, values, frame, p, label) => [...values, `${frame}:${p.x.toFixed(1)},${p.y.toFixed(1)}${h.labels.length ? `,${label}` : ""}`],
  // A misplaced point can be dragged back instead of deleted and placed again. The point on the current frame nearest
  // the pointer is grabbed, by the same mechanism as the corners of 「四个角」 and a figure's joints: the handle is a
  // view component, and the nodes using it need nothing of their own
  start: (_h, values, frame, p, scale) => {
    const i = pointUnder(pointsOn(values, frame), p, scale);
    return i < 0 ? null : { from: p, to: p, point: i };
  },
  finish: (_h, values, _frame, drag) => {
    if (drag.point === undefined) return null;
    const e = parse(values[drag.point]);
    const label = e.v.length > 2 ? `,${e.v[2]}` : "";
    const q = moved({ x: e.v[0], y: e.v[1] }, drag);
    return values.map((text, i) => (i === drag.point ? `${e.frame}:${q.x.toFixed(1)},${q.y.toFixed(1)}${label}` : text));
  },
  remove: (_h, values, frame, p, scale) => {
    const i = pointUnder(pointsOn(values, frame), p, scale);
    return i < 0 ? null : values.filter((_, k) => k !== i);
  },
};
