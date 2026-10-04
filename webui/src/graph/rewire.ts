/** Ctrl 拎线：按住 Ctrl 点一个口，拎起这个口上的全部连线；再点另一个同侧的口，整组改接到那里（editor/graphPointer.ts
 * 管手势和画线，这里只管「能不能接、接完是什么样」）。
 *
 * - 输出口拎起：它的全部下游改由新的输出口供数据（一次换掉所有下游的来源）。只能落在输出口上。
 * - 输入口拎起：接进来的全部线一起挪到新的输入口（多输入口的几根一起带走）。只能落在输入口上。
 *
 * 规则与拖一根线完全相同，只是对每一根都查一遍：类型（graph/rules.ts portAccepts，非参数口可经转换节点接上、线标虚线，
 * 同 canWire / connect）、口现在用不用得上（inactive）、不能成环（graph/nodes.ts wouldCycle）、单输入口只能接一根（原来接着的
 * 那根被换掉，同 connect）。有一根不行就整组不接，说为什么（B-WIRE-BUNDLE 包着单根线的那条理由），不做「部分成功」。
 * 落到「＋」（表格加一层的口）上不行：「＋」一次只加一层、接一根。
 *
 * 纯函数：读 GraphView（graph/snapshot.ts 的快照即可）与现在的连线，返回新的整份连线；写回、撤销（一步）归调用处。 */

import { msg, type Message } from "../messages/message";
import { wouldCycle } from "./nodes";
import { ADD_ROW, inputPort, outputPort, wireKey, type GraphView } from "./rules";
import { wireProblem } from "./wireRule";
import { wordIn } from "./naming";

export interface End {
  node: string;
  port: string;
}

export interface Wire {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

/** 拎起的一组线：从哪个口（`side`：输出口还是输入口）拎起、拎起了哪几根。 */
export interface Held {
  side: "output" | "input";
  at: End;
  wires: Wire[];
}

export type Rewired =
  /** 不是这组线的落点（原来那个口、另一侧的口、不存在的口）：什么也不做，线还拎着 */
  | null
  /** 接不下：说这一句，线还拎着 */
  | { refused: Message }
  /** 接得下：改接后的整份连线；`replaced`：单输入口上被换掉的那根（同 connect 的 I-WIRE-REPLACED） */
  | { edges: Wire[]; replaced: Wire | null };

/** 这个口上的全部连线（没有线为 null）。 */
export function holdWires(edges: Wire[], at: End, side: "output" | "input"): Held | null {
  const wires = edges.filter((e) => (side === "output" ? e.source === at.node && e.sourceHandle === at.port : e.target === at.node && e.targetHandle === at.port));
  return wires.length ? { side, at, wires } : null;
}

const labelOf = (s: GraphView, id: string) => wordIn(s.nodes, id); // a message's parameter (graph/naming.ts nodeWord)

/** 把拎起的线落到 `to`（`side` 那一侧的口）上。 */
export function rewire(s: GraphView, edges: Wire[], held: Held, to: End, side: "output" | "input"): Rewired {
  if (side !== held.side || (to.node === held.at.node && to.port === held.at.port)) return null;
  const refuse = (why: Message): Rewired => ({ refused: msg("B-WIRE-BUNDLE", { count: held.wires.length, why: why.text }) });
  const taken = new Set(held.wires.map((w) => w.id));
  const rest = edges.filter((e) => !taken.has(e.id));
  let replaced: Wire | null = null;
  let made: Wire[];

  // 每一根都问同一条规则（graph/wireRule.ts，拖线标绿、connect 也问它）；成环按拿走这组之后的连线查，整组改完再查一遍
  const unfit = (from: End, into: End): Message | null => wireProblem(s, from, into, rest);

  if (held.side === "output") {
    if (!outputPort(s, to.node, to.port)) return null;
    for (const w of held.wires) {
      const why = unfit(to, { node: w.target, port: w.targetHandle });
      if (why) return refuse(why);
    }
    made = held.wires.map((w) => ({ ...w, id: wireKey(to.node, to.port, w.target, w.targetHandle), source: to.node, sourceHandle: to.port }));
  } else {
    const port = inputPort(s, to.node, to.port);
    if (!port) return null;
    if (port.name === ADD_ROW) return refuse(msg("B-WIRE-BUNDLEPLUS", { node: labelOf(s, to.node) }));
    if (!port.multi && held.wires.length > 1)
      return refuse(msg("B-WIRE-ONLYONE", { node: labelOf(s, to.node), input: port.label, count: held.wires.length }));
    for (const w of held.wires) {
      const why = unfit({ node: w.source, port: w.sourceHandle }, to);
      if (why) return refuse(why);
    }
    if (!port.multi) replaced = rest.find((e) => e.target === to.node && e.targetHandle === to.port) ?? null;
    made = held.wires.map((w) => ({ ...w, id: wireKey(w.source, w.sourceHandle, to.node, to.port), target: to.node, targetHandle: to.port }));
  }

  // 已经有同样一根的（多输入口上本来就接着这个来源）不重复
  const kept = replaced ? rest.filter((e) => e !== replaced) : rest;
  const have = new Set(kept.map((e) => e.id));
  const fresh = made.filter((w, i) => !have.has(w.id) && made.findIndex((m) => m.id === w.id) === i);
  const next = [...kept, ...fresh];
  // 成环：改接后的图里，每根新线都不能让目标回到来源
  for (const w of fresh) {
    if (wouldCycle(next.filter((e) => e !== w), w.source, w.target))
      return refuse(msg("B-WIRE-CYCLE", { source: labelOf(s, w.source), node: labelOf(s, w.target) }));
  }
  return { edges: next, replaced };
}
