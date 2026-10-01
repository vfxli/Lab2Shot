/** 一根线 from → into 能不能接、不能时为什么：唯一的判定。拖线时哪些口标绿（editor/graphPointer.ts canWire，xyflow 的
 * isValidConnection）、落下（graph/edit.ts connect）、Ctrl 整组改接（graph/rewire.ts，对每一根问一遍）都问它，所以
 * 标成能接的口松手不会被拒，拒了说的理由也是同一句。
 *
 * 依次：两头的口在不在；源输出口现在用不上（解算器接了相机后它的「相机」只原样透传）；目标口现在用不上（服务器按参数
 * 判的 Port.applies）；类型（graph/rules.ts takes，与服务端同一规则；非参数口可经转换节点接上，线标虚线、一键插入）；
 * 成环（`edges`：现在的连线，接上以后目标能走回来源就不行，自己接自己也算）。 */

import { msg, type Message } from "../messages/message";
import { wouldCycle } from "./nodes";
import { PARAM, converter, inputPort, outputPort, takes, type GraphView } from "./rules";

export interface WireEnd {
  node: string;
  port: string;
}

const labelOf = (s: GraphView, id: string) => s.nodes.find((n) => n.id === id)?.data.label ?? id;

export function wireProblem(s: GraphView, from: WireEnd, into: WireEnd, edges: { source: string; target: string }[]): Message | null {
  const out = outputPort(s, from.node, from.port);
  const port = inputPort(s, into.node, into.port);
  const at = { source: labelOf(s, from.node), node: labelOf(s, into.node), input: port?.label ?? into.port };
  if (!out || !port) return msg("B-WIRE-MISMATCH", { ...at, got: out?.type ?? from.port, want: port?.type ?? into.port });
  if (out.inactive) return msg("B-WIRE-OUTINACTIVE", { source: at.source, output: out.label, why: out.inactive.text }, { port: out.name });
  if (port.inactive) return msg("B-WIRE-INACTIVE", { node: at.node, input: port.label, why: port.inactive.text }, { port: port.name });
  if (!takes(s.catalog, port.type, out) && (port.name.startsWith(PARAM) || !converter(s.catalog, out.type, port.type)))
    return msg("B-WIRE-MISMATCH", { ...at, got: out.type, want: port.type }, { port: port.name });
  if (wouldCycle(edges, from.node, into.node)) return msg("B-WIRE-CYCLE", { source: at.source, node: at.node });
  return null;
}
