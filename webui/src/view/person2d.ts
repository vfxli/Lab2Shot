import type { BoxesData } from "../api";
import { personAt } from "../model/people";
import { parse, type HandleTool2D } from "./handleParts";

/** 选人（「person」，nodes/handles.py）：每次点击一项「帧:x,y」，选中框里含这一点的那个人（model/people.ts personAt：
 * 服务端 at_point 的同一条规则，「选人」计算时跑的就是它）。点本身不画：画的是手柄输入上被选中的人物框（舞台的叠加层，
 * `people.lit`）；右键在某人的框里去掉这一帧上点到这个人的全部点击。 */

/** The people the picks point at: each pick chooses the person whose box holds it. What the 2D stage lights on the
 * input's boxes while the node's own result does not match the picks (view/plan.ts underHandles): a display of the
 * parameters, not a computation. */
function lit(boxes: BoxesData | null, picks: string[]): Set<number> {
  const out = new Set<number>();
  for (const text of picks) {
    const e = parse(text);
    const id = personAt(boxes, e.frame, { x: e.v[0], y: e.v[1] });
    if (id !== null) out.add(id);
  }
  return out;
}

export const personTool: HandleTool2D = {
  hint: "ui.view.hint.person",
  draws: false,
  people: { at: personAt, lit },
  click: (_h, values, frame, p, _label, source) => (personAt(source, frame, p) === null ? null : [...values, `${frame}:${Math.round(p.x)},${Math.round(p.y)}`]),
  remove: (_h, values, frame, p, _scale, source) => {
    const id = personAt(source, frame, p);
    if (id === null) return null;
    const rest = values.filter((text) => { const e = parse(text); return !(e.frame === frame && personAt(source, frame, { x: e.v[0], y: e.v[1] }) === id); });
    return rest.length === values.length ? null : rest;
  },
};
