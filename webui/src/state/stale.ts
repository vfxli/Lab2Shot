/** 「过期」状态只在此处判定。
 *
 * 节点的结果指纹由自身参数和所有输入的指纹决定。修改本节点或上游节点的参数都会改变指纹，服务器随即报告
 * 「该指纹没有结果」；但用户调整参数时仍需反复播放上一次的结果对比效果。因此：
 *
 * - 结构（连线、上游节点、读取的文件、节点类型）未变，仅指纹改变 → 过期：
 *   继续绘制上一次有结果的包，时间线显示为土黄色，节点右上角显示「已过期」，点击「计算」后更新。
 * - 结构改变（更换连线、文件或节点）→ 作废：画面回退到上游底图，不保留任何内容。
 *
 * 修改 A 的参数：A 及其下游节点过期（其结构均未变），A 的上游节点不受影响。
 * 「上一次有结果」按节点记录，随节点图一并保存在浏览器中（刷新后仍保留）；视图（view/plan.ts）、时间线、
 * 右上角状态格均只读取此处的结果。 */
import { create } from "zustand";
import { readPref, writePref } from "../platform/storage";
import { useCookInputs } from "./cookInputs";
import { useResults } from "./results";

export interface GoodResult {
  structure: string;
  fingerprint: string; // 当时节点的指纹：仅当当前指纹与之不同时才算过期（相同则是同一结果，只是尚未计算或包已被清理）
  outputs: Record<string, string>; // 端口 -> 当时的包指纹
  present: string[]; // 当时实际有包的端口
}

interface NodeLike { typeId: string; params: Record<string, unknown> }
type Nodes = (id: string) => NodeLike | undefined;
interface EdgeLike { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }

const FILE_REF = "upload:";

/** 节点的结构：该节点及其每个上游节点的类型、连线、读取的文件（以 `upload:` 开头的参数值即为文件）。
 * 不包含参数值，这正是「过期」与「作废」的分界。 */
export function structureKey(nodeOf: Nodes, edges: readonly EdgeLike[], id: string): string {
  const seen: string[] = [];
  const queue = [id];
  while (queue.length) {
    const n = queue.shift()!;
    if (seen.includes(n)) continue;
    seen.push(n);
    for (const e of edges) if (e.target === n) queue.push(e.source);
  }
  return JSON.stringify(seen.sort().map((n) => {
    const node = nodeOf(n);
    const files = node ? Object.entries(node.params).filter(([, v]) => typeof v === "string" && v.startsWith(FILE_REF)).sort() : [];
    const wires = edges.filter((e) => e.target === n).map((e) => `${e.source}.${e.sourceHandle}>${e.targetHandle}`).sort();
    return [n, node?.typeId ?? "", wires, files];
  }));
}

const key = (graphId: string) => `l2s.lastGood.${graphId || "-"}`;

interface State { byGraph: Record<string, Record<string, GoodResult>> }
export const useLastGood = create<State>(() => ({ byGraph: {} }));

function loaded(graphId: string): Record<string, GoodResult> {
  const had = useLastGood.getState().byGraph[graphId];
  if (had) return had;
  const fromDisk = readPref<Record<string, GoodResult>>(key(graphId), {});
  useLastGood.setState((s) => ({ byGraph: { ...s.byGraph, [graphId]: fromDisk } }));
  return fromDisk;
}

/** 状态回复或 node_done 报告该节点已有包：记录此次的结构与端口（仅在变化时写入）。 */
export function rememberGood(graphId: string, nodeOf: Nodes, edges: readonly EdgeLike[], id: string, fingerprint: string | null | undefined,
                             outputs: Record<string, string> | undefined, present: string[] | undefined): void {
  if (!fingerprint || !outputs || !present?.length) return;
  const all = loaded(graphId);
  const next: GoodResult = { structure: structureKey(nodeOf, edges, id), fingerprint, outputs, present: [...present].sort() };
  const had = all[id];
  if (had && had.structure === next.structure && had.fingerprint === fingerprint && JSON.stringify(had.outputs) === JSON.stringify(next.outputs) && had.present.join() === next.present.join()) return;
  const merged = { ...all, [id]: next };
  useLastGood.setState((s) => ({ byGraph: { ...s.byGraph, [graphId]: merged } }));
  writePref(key(graphId), merged);
}

/** 过期须同时满足三个条件：① 上一次有包；② 当前指纹已改变（`currentFp`：状态回复中该节点的当前指纹；
 * 尚无可信回复时不判定，此时画面按「无结果」处理，回复到达后即正确）；③ 结构未变。
 * ② 不可省略：重新选择同一份素材时，指纹与结构均相同而服务器尚未计算，这属于同一结果尚未计算而非过期，
 * 不应显示为土黄色，也不应绘制服务器上的旧包。 */
const isStale = (g: GoodResult | undefined, currentFp: string | null | undefined, structure: () => string): g is GoodResult =>
  !!g && !!g.fingerprint && g.present.length > 0 && !!currentFp && currentFp !== g.fingerprint && structure() === g.structure;

/** 若该端口已过期，返回上一次的包指纹；null 表示未过期。 */
export function staleFp(graphId: string, nodeOf: Nodes, edges: readonly EdgeLike[], id: string, port: string,
                        currentFp: string | null | undefined): string | null {
  const g = loaded(graphId)[id];
  if (!g?.present.includes(port)) return null;
  return isStale(g, currentFp, () => structureKey(nodeOf, edges, id)) ? (g.outputs[port] ?? null) : null;
}

/** 该节点当前是否有可用的过期结果（任一端口）。 */
export function staleNode(graphId: string, nodeOf: Nodes, edges: readonly EdgeLike[], id: string, currentFp: string | null | undefined): boolean {
  return isStale(loaded(graphId)[id], currentFp, () => structureKey(nodeOf, edges, id));
}

/** 该节点当前是否为「已过期」（供右上角状态格使用）：服务器报告该指纹无包，且上一次结果的结构未变。
 * 状态回复尚未跟上当前版本的节点图时（`forCookInputs` 落后），同样按上一次的结果判定，即参数刚修改后的情形。 */
export function useStaleNode(id: string): boolean {
  const graphId = useCookInputs((s) => s.graphId);
  const version = useCookInputs((s) => s.version);
  const nodes = useCookInputs((s) => s.nodes);
  const edges = useCookInputs((s) => s.edges);
  const byGraph = useLastGood((s) => s.byGraph);
  const results = useResults((s) => s.results);
  const forCookInputs = useResults((s) => s.forCookInputs);
  const trusted = forCookInputs === version;
  const present = trusted ? (results[id]?.present?.length ?? 0) > 0 : false;
  void byGraph;
  return !present && staleNode(graphId, (n) => nodes[n], edges, id, trusted ? results[id]?.fingerprint : undefined);
}
