import { DRAWN_COLOR, FONT, parse, removeSmallest, type Drag, type HandleTool2D, type Pt } from "./handleParts";

/** 框手柄（「box」，nodes/handles.py）：拖出一个框加一项「帧:x1,y1,x2,y2」，右键去掉指针所在的最小的框。 */

/** A drag's rectangle in whole pixels, or null when it is too small to be one (a click or a flick). Shared with the
 * corners handle, whose new quad is drawn the same way. */
export function dragRect(drag: Drag): { x1: number; y1: number; x2: number; y2: number } | null {
  const { from, to } = drag;
  if (Math.abs(to.x - from.x) < 4 || Math.abs(to.y - from.y) < 4) return null;
  const [x1, x2] = [Math.min(from.x, to.x), Math.max(from.x, to.x)].map(Math.round);
  const [y1, y2] = [Math.min(from.y, to.y), Math.max(from.y, to.y)].map(Math.round);
  return { x1, y1, x2, y2 };
}

export const boxTool: HandleTool2D = {
  hint: "ui.view.hint.box",
  draws: true,
  draw: (_h, f, values, drag) => {
    const { ctx, at, frame } = f;
    ctx.font = FONT;
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
  },
  start: (_h, _values, _frame, p) => (p.inside ? { from: p, to: p } : null),
  finish: (_h, values, frame, drag) => {
    const r = dragRect(drag);
    return r ? [...values, `${frame}:${r.x1},${r.y1},${r.x2},${r.y2}`] : null;
  },
  remove: (_h, values, frame, p) =>
    removeSmallest(values, (e) => {
      if (e.frame !== frame) return null;
      return boxHit(p, e.v[0], e.v[1], e.v[2], e.v[3]);
    }),
};

/** A box as drawn, whichever corner the entry starts from: whether p is in it, and its area. */
function boxHit(p: Pt, ax: number, ay: number, bx: number, by: number): { inside: boolean; area: number } {
  const [x1, x2] = [Math.min(ax, bx), Math.max(ax, bx)];
  const [y1, y2] = [Math.min(ay, by), Math.max(ay, by)];
  return { inside: p.x >= x1 && p.x <= x2 && p.y >= y1 && p.y <= y2, area: (x2 - x1) * (y2 - y1) };
}
