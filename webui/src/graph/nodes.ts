import type { DataType, NodeStatus as Status, NodeTypeDef, ParamDef, Plan, SceneKind, ServerMessage, StatusReply, WritesPart } from "../api";
import { fromServer, msg, type Message } from "../messages/message";
import { greyed, nodeUsable, nodeWhy, why as whyOf } from "../api/applies";
import type { GBox, GNode, GraphState, NodeData } from "../state/graph";

/** Pure graph-reading helpers: functions that read a plain snapshot of the graph (nodes, edges, node definitions, the last
 * status reply) and answer a question the page itself owns, such as a node's rows on its body, a group box's members, or
 * what still prevents a click from being submitted. None reads a store, and none works out a rule of the server's (whether
 * a wire fits, a node's ports, what a cook is): those are read from the status reply and the catalogue (graph/rules.ts),
 * and no copy of a server rule is kept in this file. */


export const BOX_COLORS = ["#8E8E93", "#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#FF375F"];
export const BOX_HEAD = 34; // header height; a collapsed box consists of its header only

export const nodeSize = (n: Pick<GNode, "measured">) => ({ w: n.measured?.width ?? 240, h: n.measured?.height ?? 96 });

/** Nodes whose centre lies inside the box: they move, collapse and hide with it. */
export function boxContents(box: GBox, nodes: GNode[]): string[] {
  if (box.collapsed) return box.members;
  return nodes
    .filter((n) => {
      const { w, h } = nodeSize(n);
      const cx = n.position.x + w / 2;
      const cy = n.position.y + h / 2;
      return cx > box.x && cx < box.x + box.w && cy > box.y + BOX_HEAD && cy < box.y + box.h;
    })
    .map((n) => n.id);
}

/** 目录中不存在的类型使用此灰色：比 Mask 的灰色暗一档，不用于任何已知类型（与 lab2shot/data/types.py UNKNOWN_COLOUR 取值相同）。 */
export const UNKNOWN_COLOR = "#6b6b70";

/** Color of a port: its first accepted type. A list takes its items' colour (a list port is drawn as a list OF its items'
 * type, never with a colour of its own), so "scene[]" is the scene colour, not the grey of an unknown type.
 * How a port's type is described is never decided here: the server names it (PortDef.type_label).
 *
 * 颜色规范（见 lab2shot/data/types.py COLOURS 的注释）：暖色 = 三维，冷色 = 二维，灰 = Mask，紫红 = 数值；
 * 同组成员以明显不同的色相或明度区分，不使用微小偏移；列表与单值同色，列表以方形端口加双线区分。
 * 色值由服务器提供，此处不定义任何色值。 */
export function portColor(types: Record<string, DataType>, portType: string): string {
  return types[portType.split("|")[0].replace(/\[\]$/, "")]?.color ?? UNKNOWN_COLOR;
}

/** A 3D settings node's table (OutputSettings.writes) in the catalog's order of kinds. */
export function writesTable(kinds: SceneKind[], def: NodeTypeDef): (WritesPart & { kind: SceneKind })[] {
  const writes = def.writes ?? {};
  return kinds.filter((k) => writes[k.id]).map((k) => ({ kind: k, ...writes[k.id] }));
}

/** 该节点自身「提升到节点」的参数（不含节点类型声明的常驻接线口），其端口绘制在参数所在行上。
 * 常驻口（NodeDef.wired_ports）是节点的输入口，绘制在上方的端口列表中；两处都绘制会导致同一端口出现两次。 */
export const promotedByHand = (def: NodeTypeDef | undefined, promoted: string[] | undefined): string[] =>
  (promoted ?? []).filter((n) => !def?.wired_ports?.includes(n));

/** 返回节点上显示的参数行，与参数值同源，以 json 为准。
 * 节点图文件中该节点的 `ui.on_node` 即为结果（graph/document.ts 负责读写）；
 * `NodeDef.on_node` 仅为出厂默认：在新建节点时写入 json（graph/edit.ts addNode，与写入
 * `...def.defaults` 参数默认值相同），并在 json 未包含该项时作为兜底。
 *
 * 参数行上有两种标记，含义不同：在节点上显示（即此名单，仅影响显示，见 ParamPanel.tsx OnNodePin），以及
 * 提升到节点（`promoted`：增加输入口 `param:<name>`，该行同时可直接编辑）。
 * 使用者可移除出厂默认中的任意一行：这些行已是 json 中的数据，不存在类型层面固定不可移除的行。 */
export const chosenOnNode = (def: NodeTypeDef, onNode: string[] | undefined): string[] => onNode ?? def.on_node;

/** The rows a node's body shows, in the order of its parameters: the ones it shows (chosenOnNode) and the ones this
 * node has 提升到节点 (each of those also carries its own input on its row); 展开 shows every simple parameter. */
export function nodeRows(def: NodeTypeDef | undefined, data: Pick<NodeData, "promoted" | "onNode">, expanded: boolean): ParamDef[] {
  if (!def) return [];
  const chosen = chosenOnNode(def, data.onNode);
  const own = promotedByHand(def, data.promoted);
  return def.params.filter((p) => p.simple && (own.includes(p.name) || expanded || chosen.includes(p.name)));
}

export const hiddenOnNode = (def: NodeTypeDef | undefined, data: Pick<NodeData, "promoted" | "onNode">): number =>
  def ? def.params.filter((p) => p.simple).length - nodeRows(def, data, false).length : 0;


export function wouldCycle(edges: { source: string; target: string }[], source: string, target: string): boolean {
  if (source === target) return true;
  const stack = [target];
  const seen = new Set<string>();
  while (stack.length) {
    const n = stack.pop()!;
    if (n === source) return true;
    if (seen.has(n)) continue;
    seen.add(n);
    for (const e of edges) if (e.source === n) stack.push(e.target);
  }
  return false;
}

/** What is wrong with the typed frame range to cook: whole numbers, in order, inside what the shown node's inputs cover
 * (`plan`: the shown node's, as the server planned it for these cook inputs; null not known). */
export function rangeProblem(cookRange: [string, string] | null, plan: Pick<Plan, "range"> | null): Message | null {
  if (!cookRange) return null;
  const nums = cookRange.every((v) => /^-?\d+$/.test(v.trim())) ? ([Number(cookRange[0]), Number(cookRange[1])] as [number, number]) : null;
  if (!nums) return msg("B-RANGE-NOTINT", { first: cookRange[0], last: cookRange[1] });
  const [first, last] = nums;
  if (first > last) return msg("B-RANGE-ORDER", { first, last });
  // 计算范围随素材变化：范围保存在节点图中，素材重新上传后（例如删除开头一段）旧范围仍然保留，不应因此阻止提交。
  // 与素材有重叠即放行（实际计算交集，见 cookSpan），仅在完全不重叠时阻止，
  // 此时使用者填写的是另一段镜头的帧号，需要修改。
  const full = plan?.range;
  if (full && (last < full[0] || first > full[1])) return msg("B-RANGE-OUTSIDE", { first, last, start: full[0], end: full[1] });
  return null;
}

/** 本次实际计算的范围：使用者填写的范围与素材的交集（未填写时为素材全部；`plan` 未知时按填写值）。
 * 素材更换后范围仅部分落在素材内时，计算该部分，不阻止提交（见上方 rangeProblem 的注释）。 */
export function cookSpan(cookRange: [string, string] | null, plan: Pick<Plan, "range"> | null): [number, number] | null {
  const full = (plan?.range as [number, number] | undefined) ?? null;
  if (!cookRange) return full;
  const nums = cookRange.every((v) => /^-?\d+$/.test(v.trim())) ? ([Number(cookRange[0]), Number(cookRange[1])] as [number, number]) : null;
  if (!nums) return full;
  if (!full) return nums;
  const span: [number, number] = [Math.max(nums[0], full[0]), Math.min(nums[1], full[1])];
  return span[0] <= span[1] ? span : full;
}

/** 单次提交可计算的最大帧数（由管理员在后台「设置」中配置，随队列响应返回；`most` 为 0 表示尚未获取，不做限制）。
 * 网页据此在点击前拦截；服务器在 farm/queue.py _submit 中独立再次校验，因此绕过网页直接提交同样会被拒绝。 */
export function frameLimitProblem(span: [number, number] | null, most: number): Message | null {
  if (!span || most <= 0) return null;
  const frames = span[1] - span[0] + 1;
  return frames > most ? msg("B-JOB-TOOMANYFRAMES", { frames, most, first: span[0], last: span[1] }) : null;
}

/** A node's cook status in words: the node's footer and its 信息 panel say the same. */
export const STATUS_TEXT: Record<import("../state/graph").NodeStatus, string> = { idle: "未计算", queued: "排队", cooked: "已缓存", cooking: "计算中…", error: "出错", skipped: "已跳过" };

/** A node its job is still working on: waiting in the queue (its note: why, as the server says) or cooking. */
export const isLive = (status: import("../state/graph").NodeStatus | undefined): boolean => status === "queued" || status === "cooking";

export function waitText(job: { position: number | null }): string {
  return job.position == null ? "排队中" : `排队第 ${job.position} 位`;
}

/** Every 「输出」 in the graph, in node order: what 提交 delivers, together, as one job. */
export function deliveryNodes(nodes: GNode[], nodeDefs: Record<string, NodeTypeDef>): string[] {
  return nodes.filter((n) => nodeDefs[n.data.typeId]?.delivers).map((n) => n.id);
}

const FILE_IN = ["file", "sequence"];

/** 「提交」 has something to cook only when the graph has an 「输出」: every one of them collects and packs, together. */
export function deliverBlocked(nodes: GNode[], nodeDefs: Record<string, NodeTypeDef>, outputs = deliveryNodes(nodes, nodeDefs)): Blocker | null {
  return outputs.length ? null : { node: "", message: msg("B-DELIVER-NOOUTPUT") };
}

/** Why a cook or a delivery can't be submitted yet (a B- message of lab2shot/messages/web.toml, or the server's own
 * reason for the node), on its node ("" the graph itself); the message's `param` anchor: the parameter to go to. */
export interface Blocker {
  node: string;
  message: Message;
}

/** What blockers() and standing() read beyond GraphState: the status reply for these cook inputs (trusted: a click
 * waits for it, graph/actions.ts currentReply), the shown node's plan, and what only the page knows (uploads going up)
 * as a plain callback, so this file need not import transfer/uploads.ts. */
export interface BlockContext extends GraphState {
  reply: StatusReply;
  cookRange: [string, string] | null;
  plan: Plan | null;
  uploadBlocked: (id: string, param: string, label: string) => Message | undefined; // undefined: nothing going up
  applies: import("../api/applies").Availability | null | undefined; // the login's answer: which node types are usable now
}

/** 本次计算涉及的节点（目标及其全部上游），仅供本文件检查问题与汇总消息使用。
 *
 * 不用于决定计算前上传哪些素材：上传哪些素材、每份上传哪些通道由服务器决定，即状态回复中的
 * `policy.computes`（本次实际计算的节点）与各节点的 `channels`（`graph/apply.ts pickedFor / sendPicked`）。
 * 网页按连线自行反推会产生第二份答案；该判断仅在服务器 `needed_outputs` 中进行。 */
const nodesCooked = (ctx: GraphState, targets: string[]) => [...new Set(targets.flatMap((t) => upstream(t, ctx.edges).reverse()))];

/** Why cooking `targets` (the reply's policy says which) can't be submitted yet: what the page sees before sending
 * anything (a range typed wrong, a file not chosen or still going up, a choice the data wired in can't take), the wires the
 * server judged wrong, and a node's own error from the server. */
export function blockers(ctx: BlockContext, targets: string[]): Blocker[] {
  const range = rangeProblem(ctx.cookRange, ctx.plan);
  const out: Blocker[] = range ? [{ node: "", message: range }] : [];
  for (const id of nodesCooked(ctx, targets)) {
    const node = ctx.nodes.find((n) => n.id === id);
    const def = node && ctx.nodeDefs[node.data.typeId];
    if (!node || !def) continue;
    const name = node.data.label;
    const status = ctx.reply.nodes[id];
    const own: Message[] = [];
    if (!nodeUsable(ctx.applies, def.id)) own.push(msg("B-COOK-UNAVAILABLE", { node: name, reason: nodeWhy(ctx.applies, def.id) }));
    for (const w of ctx.reply.wires) if (w.to[0] === id && w.state !== "ok" && w.problem) own.push({ ...fromServer(w.problem), port: w.to[1] });
    for (const p of def.params) {
      const value = node.data.params[p.name];
      if (greyed(status?.applies, p.name)) continue;
      const going = FILE_IN.includes(p.widget ?? "") ? ctx.uploadBlocked(id, p.name, name) : undefined;
      if (going) own.push({ ...going, param: p.name });
      else if (!value && FILE_IN.includes(p.widget ?? "")) own.push(msg("B-COOK-NOFILE", { node: name, what: p.label }, { param: p.name }));
      // 当前选中的选项不可用：接入的数据类型不符合要求（option_applies，由服务器计算，id 为 "<参数>=<选项>"）。
      // 在提交前拦截，不交由服务器报错
      const why = whyOf(status?.applies, `${p.name}=${String(value)}`);
      if (why) own.push(msg("B-COOK-OPTION", { node: name, setting: p.label, option: p.option_labels?.[String(value)] ?? String(value), reason: why }, { param: p.name }));
    }
    // the server's error stops the click only when the node can't be planned (a required input not wired, a refused
    // file, a wired value it can't take); a failure kept from an earlier cook (its outcome: failed, or skipped under
    // one) is tried again by the click (engine/cook.py), never a reason not to submit
    const server = !status?.outcome ? status?.error : undefined;
    if (!own.length && server) own.push(fromServer(server));
    out.push(...own.map((message) => ({ node: id, message })));
  }
  return out;
}

/** The same reason reported for several nodes (the server names the node that is really wrong on each one below it):
 * said once, where it names a parameter to go to if any does. */
export function dedupeBlockers(found: Blocker[]): Blocker[] {
  const byText = new Map<string, Blocker>();
  for (const b of found) {
    const cur = byText.get(b.message.text);
    if (!cur || (!cur.message.param && b.message.param)) byText.set(b.message.text, b);
  }
  return [...byText.values()];
}

/** What still stands on the nodes cooking `targets` covers (their usage checks and what they said when cooked, W and N:
 * a refused wire is a blocker, information stays in the log): a click's summary starts with them, so a warning on a
 * node already cooked is listed again when it is cooked again, not only on its mark. */
export function standing(ctx: BlockContext, targets: string[]): Blocker[] {
  return nodesCooked(ctx, targets).flatMap((id) => nodeMessages(ctx.reply.nodes, id).filter((m) => !m.refused && (m.level === "W" || m.level === "N")).map((m) => ({ node: id, message: fromServer(m) })));
}

/** What the server says about a node: its usage checks and what it said when it was cooked (lab2shot/messages). */
export function nodeMessages(results: Record<string, Status>, id: string): ServerMessage[] {
  return results[id]?.messages ?? [];
}

export function upstream(id: string, edges: { source: string; target: string }[]): string[] {
  const out: string[] = [];
  const queue = [id];
  while (queue.length) {
    const n = queue.shift()!;
    if (out.includes(n)) continue;
    out.push(n);
    for (const e of edges) if (e.target === n) queue.push(e.source);
  }
  return out;
}

