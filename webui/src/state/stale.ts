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
 * 右上角状态格均只读取此处的结果。记录中还有该结果是用哪一组参数算出的（`params`）：变换手柄据此把节点自己的
 * 结果按当前参数重新摆放（view/Stage3D.tsx TransformHandle），不必等重新计算。 */
import { create } from "zustand";
import { same, type Json } from "../model/graphPatch";
import { prefKeys, readPref, removePref, writePref } from "../platform/storage";
import { useCookInputs } from "./cookInputs";

interface GoodResult {
  structure: string;
  fingerprint: string; // 当时节点的指纹：仅当当前指纹与之不同时才算过期（相同则是同一结果，只是尚未计算或包已被清理）
  outputs: Record<string, string>; // 端口 -> 当时的包指纹
  present: string[]; // 当时实际有包的端口
  // 算出这份结果时节点的参数：只在状态回复与页面当前版本一致时才知道（此时回复中的指纹就是这组参数的指纹）；
  // 不知道时为 undefined
  params?: Record<string, unknown>;
}

interface NodeLike { typeId: string; params: Record<string, unknown> }
type Nodes = (id: string) => NodeLike | undefined;
interface EdgeLike { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }

const FILE_REF = "upload:";

/** 节点的结构：该节点及其每个上游节点的类型、连线、读取的文件（以 `upload:` 开头的参数值即为文件）。
 * 不包含参数值，这正是「过期」与「作废」的分界。 */
function structureKey(nodeOf: Nodes, edges: readonly EdgeLike[], id: string): string {
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

const PREFIX = "l2s.lastGood.";
const key = (graphId: string) => `${PREFIX}${graphId || "-"}`;
const RECENT = "l2s.lastGoodRecent"; // 节点图的 id，最近写入的在前
const KEEP_GRAPHS = 20; // 浏览器中保留「上一次有结果」的节点图数量；更早的节点图在重新计算前不再显示过期结果

interface State { byGraph: Record<string, Record<string, GoodResult>> }
export const useLastGood = create<State>(() => ({ byGraph: {} }));

/** 这张图的记录读进 store（从浏览器存储）：只在渲染之外做——换了文档（订阅 graphId）、写入新记录（rememberGood）时。
 * 渲染里只读（goodOf），不写 store。 */
function ensureLoaded(graphId: string): Record<string, GoodResult> {
  const had = useLastGood.getState().byGraph[graphId];
  if (had) return had;
  const fromDisk = readPref<Record<string, GoodResult>>(key(graphId), {});
  useLastGood.setState((s) => ({ byGraph: { ...s.byGraph, [graphId]: fromDisk } }));
  return fromDisk;
}
useCookInputs.subscribe((s, p) => {
  if (s.graphId !== p.graphId) ensureLoaded(s.graphId);
});

const NONE: Record<string, GoodResult> = {};
/** 这张图记下的「上一次有结果」（没读进来之前为空：换文档时就读进来了）。 */
const goodOf = (graphId: string): Record<string, GoodResult> => useLastGood.getState().byGraph[graphId] ?? NONE;

/** 状态回复或 node_done 报告该节点已有包：记录此次的结构与端口（仅在变化时写入）。
 * `trusted`：报告回答的就是页面当前的版本，此时节点现在的参数就是算出这份结果的参数，一并记下；
 * 不是时，同一指纹沿用已记下的参数，否则不记。 */
export function rememberGood(graphId: string, nodeOf: Nodes, edges: readonly EdgeLike[], id: string, fingerprint: string | null | undefined,
                             outputs: Record<string, string> | undefined, present: string[] | undefined, trusted: boolean): void {
  if (!fingerprint || !outputs || !present?.length) return;
  const all = ensureLoaded(graphId);
  const had = all[id];
  const params = trusted ? nodeOf(id)?.params : had?.fingerprint === fingerprint ? had.params : undefined;
  const next: GoodResult = { structure: structureKey(nodeOf, edges, id), fingerprint, outputs, present: [...present].sort(), params };
  if (had && had.structure === next.structure && had.fingerprint === fingerprint && same(had.outputs, next.outputs)
      && had.present.join() === next.present.join() && same(had.params as Json | undefined, params as Json | undefined)) return;
  const merged = { ...all, [id]: next };
  useLastGood.setState((s) => ({ byGraph: { ...s.byGraph, [graphId]: merged } }));
  writePref(key(graphId), merged);
  const recent = [graphId, ...readPref<string[]>(RECENT, []).filter((g) => g !== graphId)];
  for (const g of recent.splice(KEEP_GRAPHS)) removePref(key(g));
  writePref(RECENT, recent);
}

/** 浏览器存储已满（editor/autosave.ts 存不下工作副本）时调用：删去其他节点图的「上一次有结果」，只留当前这张。
 * 这些记录只用于在重新计算前显示过期结果，删去不丢任何工作。 */
export function forgetOtherGraphs(graphId: string): void {
  for (const k of prefKeys(PREFIX)) if (k !== key(graphId)) removePref(k);
  writePref(RECENT, [graphId]);
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
  const g = goodOf(graphId)[id];
  if (!g?.present.includes(port)) return null;
  return isStale(g, currentFp, () => structureKey(nodeOf, edges, id)) ? (g.outputs[port] ?? null) : null;
}

/** 节点上一次有结果时的参数（见 `GoodResult.params`；undefined：不知道）。 */
export function cookedWith(graphId: string, id: string): Record<string, unknown> | undefined {
  return goodOf(graphId)[id]?.params;
}

/** 该节点当前是否有可用的过期结果（任一端口）。`good`：它记下的那一份（组件用 useLastGood 选出来，随之重画）。 */
export function staleNode(good: GoodResult | undefined, nodeOf: Nodes, edges: readonly EdgeLike[], id: string, currentFp: string | null | undefined): boolean {
  return isStale(good, currentFp, () => structureKey(nodeOf, edges, id));
}
