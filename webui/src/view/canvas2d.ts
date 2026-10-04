import { DRAWN_COLOR, FONT, areaOf, insidePolygon, parse, polygon, removeSmallest, type HandleTool2D } from "./handleParts";

/** 手画遮罩（「canvas」，nodes/handles.py）：拖动画一个闭合轮廓，每个轮廓一项「帧:x1,y1,…」（画了几点就几点）；
 * 轮廓对整段镜头生效（nodes/core/mask.py），所以每一帧都画出全部轮廓。右键去掉指针所在的最小的轮廓。 */

const CANVAS_STEP = 1.5; // image pixels a freehand drag must travel before another point is added

export const canvasTool: HandleTool2D = {
  hint: "ui.view.hint.canvas",
  draws: true,
  // every outline is shown on every frame, with a faint fill and a solid edge: a hand-drawn matte applies to the whole
  // shot, so what is on screen is exactly the mask
  draw: (_h, f, values, drag) => {
    const { ctx, at } = f;
    ctx.font = FONT;
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
  },
  start: (_h, _values, _frame, p) => (p.inside ? { from: p, to: p, path: [{ x: p.x, y: p.y }] } : null),
  // the outline adds another point once the pointer has moved far enough, so the entry holds the drawn line rather than
  // every pointer event
  move: (drag, p) => {
    if (!drag.path) return { ...drag, to: p };
    const last = drag.path[drag.path.length - 1];
    return Math.hypot(p.x - last.x, p.y - last.y) < CANVAS_STEP ? { ...drag, to: p } : { ...drag, to: p, path: [...drag.path, { x: p.x, y: p.y }] };
  },
  finish: (_h, values, frame, drag) => {
    const pts = drag.path ?? [];
    if (pts.length < 3) return null; // a click or a flick: no area, so nothing is drawn
    return [...values, `${frame}:${pts.map((q) => `${q.x.toFixed(1)},${q.y.toFixed(1)}`).join(",")}`];
  },
  // an outline applies to every frame: removable from any of them
  remove: (_h, values, _frame, p) => removeSmallest(values, (e) => { const pts = polygon(e); return { inside: insidePolygon(p, pts), area: areaOf(pts) }; }),
};
