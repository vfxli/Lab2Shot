import type { CookCase, DataType, NodeStatus as Status, NodeTypeDef, ParamDef, Plan, SceneKind, ServerMessage, StatusReply, WritesPart } from "../api";
import { fromServer, msg, type Message } from "../messages/message";
import { greyed, nodeUsable, nodeWhy, why as whyOf } from "../api/applies";
import type { GBox, GNode, GraphState, NodeData } from "../state/graph";
import { optionName } from "./rules";
import { nodeWord } from "./naming";
import type { Said } from "../messages/format";
import { t } from "../i18n/t";
import { noteText, noteWord, type NoteWord } from "../model/nodeOutcome";

/** 纯读取节点图的辅助函数：读节点图的普通快照（节点、连线、节点定义、上一次状态回复），回答页面自己负责的问题，例如
 * 节点体上显示哪些参数行、分组框里有哪些节点、一次点击还有什么挡着不能提交。它们都不读 store，也都不推算服务器的规则
 * （线能不能接、节点有哪些口、一次计算算什么）：那些从状态回复和目录里读（graph/rules.ts），本文件不留服务器规则的副本。 */


export const BOX_COLORS = ["#8E8E93", "#0A84FF", "#30D158", "#FF9F0A", "#BF5AF2", "#FF375F"];
export const BOX_HEAD = 34; // 标题栏高度；折叠的框只剩标题栏
export const BOX_FOLD_W = 280; // 折叠的框最宽这么宽（再窄的框照它自己的宽）

export const nodeSize = (n: Pick<GNode, "measured">) => ({ w: n.measured?.width ?? 240, h: n.measured?.height ?? 96 });

/** 中心落在框内的节点：随框移动、折叠、隐藏。 */
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

/** Color of a port: its first accepted type (grey when its candidates are of different roots). A list takes its items' colour (a list port is drawn as a list OF its items'
 * type, never with a colour of its own), so "scene[]" is the scene colour, not the grey of an unknown type.
 * How a port's type is described is never decided here: the server names it (PortDef.type_label).
 *
 * 颜色规范（见 lab2shot/data/types.py COLOURS 的注释）：暖色 = 三维，冷色 = 二维，灰 = Mask，紫红 = 数值；
 * 同组成员以明显不同的色相或明度区分，不使用微小偏移；列表与单值同色，列表以方形端口加双线区分。
 * 色值由服务器提供，此处不定义任何色值。 */
export function portColor(types: Record<string, DataType>, portType: string): string {
  const alts = portType.split("|").map((t) => t.replace(/\[\]$/, ""));
  // candidates of different roots (the open "anything" of a port that follows nothing yet): no one type's colour
  if (new Set(alts.map((t) => t.split(".")[0])).size > 1) return UNKNOWN_COLOR;
  return types[alts[0]]?.color ?? UNKNOWN_COLOR;
}

/** 三维输出设置节点的表（OutputSettings.writes），按目录里种类的顺序。 */
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

/** 节点体上显示的参数行，按参数的顺序：它选了显示的（chosenOnNode）与这个节点「提升到节点」的（后者在自己那一行上
 * 还带一个输入口）；「展开」时显示每个简单参数。 */
export function nodeRows(def: NodeTypeDef | undefined, data: Pick<NodeData, "promoted" | "onNode">, expanded: boolean): ParamDef[] {
  if (!def) return [];
  const chosen = chosenOnNode(def, data.onNode);
  const own = promotedByHand(def, data.promoted);
  // 按钮参数（「计算」每个节点都有）只在使用者选了「在节点上显示」时上节点，「展开」不带它们
  return def.params.filter((p) => p.simple && (own.includes(p.name) || (expanded && p.simple !== "button") || chosen.includes(p.name)));
}

export const hiddenOnNode = (def: NodeTypeDef | undefined, data: Pick<NodeData, "promoted" | "onNode">): number =>
  def ? def.params.filter((p) => p.simple && p.simple !== "button").length - nodeRows(def, data, false).filter((p) => p.simple !== "button").length : 0;


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

/** 计算范围两格里写的是什么——范围只在这里解析（rangeProblem、cookSpan、存盘与提交的 frames 都读它）：null 没填；
 * `bad` 写错（不是整数 / 先大后小）；否则 `span` [first, last]。 */
export type ParsedSpan = null | { bad: "notint" | "order" } | { span: [number, number] };
export function parseSpan(cookRange: [string, string] | null): ParsedSpan {
  if (!cookRange) return null;
  if (!cookRange.every((v) => /^-?\d+$/.test(v.trim()))) return { bad: "notint" };
  const span: [number, number] = [Number(cookRange[0]), Number(cookRange[1])];
  return span[0] > span[1] ? { bad: "order" } : { span };
}

/** 填写的计算范围哪里不对：须为整数、先小后大、与显示节点的输入覆盖的范围有重叠（`plan`：它的范围；null 为未知）。
 * 与素材只是部分重叠不算错——提交的是交集（cookSpan），范围本身照写的留着：
 * 素材换短了，节点图里的旧范围不改写，也不挡提交；完全不重叠（填的是另一段镜头的帧号）才要使用者改。 */
export function rangeProblem(cookRange: [string, string] | null, plan: Pick<Plan, "range"> | null): Message | null {
  const got = parseSpan(cookRange);
  if (!got) return null;
  if ("bad" in got) return got.bad === "notint" ? msg("B-RANGE-NOTINT", { first: cookRange![0], last: cookRange![1] }) : msg("B-RANGE-ORDER", { first: Number(cookRange![0]), last: Number(cookRange![1]) });
  const [first, last] = got.span;
  const full = plan?.range;
  if (full && (last < full[0] || first > full[1])) return msg("B-RANGE-OUTSIDE", { first, last, start: full[0], end: full[1] });
  return null;
}

/** 实际提交给服务器的计算范围（toJSON 的 frames；帧数上限也按它数）：没填、写错为 null（整段）；与素材部分重叠为交集
 * （服务器不收比输入宽的范围，engine/evaluation.py check_frames）；素材范围未知或完全不重叠时照写的（后者由 rangeProblem
 * 挡在提交前）。只算、不改文档：节点图里存的永远是使用者写的。 */
export function cookSpan(cookRange: [string, string] | null, plan: Pick<Plan, "range"> | null): [number, number] | null {
  const got = parseSpan(cookRange);
  if (!got || "bad" in got) return null;
  const full = (plan?.range as [number, number] | undefined) ?? null;
  if (!full) return got.span;
  const span: [number, number] = [Math.max(got.span[0], full[0]), Math.min(got.span[1], full[1])];
  return span[0] <= span[1] ? span : got.span;
}

/** 单次提交可计算的最大帧数（由管理员在后台「设置」中配置，随队列响应返回；`most` 为 0 表示尚未获取，不做限制）。
 * 网页据此在点击前拦截；服务器在 farm/queue.py _submit 中独立再次校验，因此绕过网页直接提交同样会被拒绝。 */
export function frameLimitProblem(span: [number, number] | null, most: number): Message | null {
  if (!span || most <= 0) return null;
  const frames = span[1] - span[0] + 1;
  return frames > most ? msg("B-JOB-TOOMANYFRAMES", { frames, most, first: span[0], last: span[1] }) : null;
}

/** 节点计算状态的文字：节点底部与它的「信息」面板说的是同一句。 */
const STATUS_KEY: Record<import("../state/graph").NodeStatus, string> = {
  idle: "ui.graph.status_idle", queued: "ui.graph.status_queued", cooked: "ui.graph.status_cooked",
  cooking: "ui.graph.status_cooking", error: "ui.graph.status_error", skipped: "ui.graph.status_skipped",
};
export const statusText = (status: import("../state/graph").NodeStatus): string => t(STATUS_KEY[status]);

/** 任务还在处理的节点：在队列里等（注记是原因，按服务器说的）或正在计算。 */
export const isLive = (status: import("../state/graph").NodeStatus | undefined): boolean => status === "queued" || status === "cooking";

export function waitText(job: { position: number | null }): string {
  return noteText(waitWord(job));
}

/** The same kept as its word (model/nodeOutcome.ts NoteWord): said when shown, in the language then. */
export function waitWord(job: { position: number | null }): NoteWord {
  return job.position == null ? noteWord("ui.graph.waiting") : noteWord("ui.graph.waiting_at", { position: job.position });
}

/** 图里的全部「输出」，按节点顺序：「提交」把它们一起作为一个任务交付。 */
export function deliveryNodes(nodes: GNode[], nodeDefs: Record<string, NodeTypeDef>): string[] {
  return nodes.filter((n) => nodeDefs[n.data.typeId]?.delivers).map((n) => n.id);
}

const FILE_IN = ["file", "sequence"];

/** 节点上现在要用、却还没选文件的文件参数（置灰的不算）：B-COOK-NOFILE 说的就是它们，blockers() 里「这根线在等的
 * 是不是一个没选文件的读取节点」也按这一条认。 */
function missingFiles(def: NodeTypeDef, params: Record<string, unknown>, status: Status | undefined): ParamDef[] {
  return def.params.filter((p) => FILE_IN.includes(p.widget ?? "") && !params[p.name] && !greyed(status?.applies, p.name));
}

/** 「提交」（graph/actions.ts deliverAll：Ctrl+Shift+Enter，打包全部「输出」；顶栏没有这个按钮）只在图里有「输出」时
 * 才有东西可算：每一个「输出」一起收集、打包。 */
export function deliverBlocked(nodes: GNode[], nodeDefs: Record<string, NodeTypeDef>, outputs = deliveryNodes(nodes, nodeDefs)): Blocker | null {
  return outputs.length ? null : { node: "", message: msg("B-DELIVER-NOOUTPUT") };
}

/** 一次计算或交付为什么还不能提交（lab2shot/messages/web.toml 的一条 B- 消息，或服务器对该节点给的原因），落在哪个节点上
 * （"" 为整张图）；消息的 `param` 锚点：要跳到的参数。 */
export interface Blocker {
  node: string;
  message: Message;
}

/** blockers() 与 standing() 在 GraphState 之外要读的：与这些计算输入对应的状态回复（可信的：点击会等它，
 * graph/actions.ts currentReply）、显示节点的 plan，以及只有页面知道的（正在上传的）——作为普通回调传入，本文件因此不必
 * import transfer/uploads.ts。 */
export interface BlockContext extends GraphState {
  reply: StatusReply;
  cookRange: [string, string] | null;
  plan: Plan | null;
  uploadBlocked: (id: string, param: string, label: Said | string) => Message | undefined; // undefined：没有在传的
  applies: import("../api/applies").Availability | null | undefined; // 登录给的答案：现在哪些节点类型可用
}

/** 本次计算涉及的节点（目标及其按选路走得到的上游，有缓存的也算），仅供本文件检查问题与汇总消息使用。
 *
 * 读服务器的答案：状态回复里这次计算（`policy` 或 `deliver` 中 targets 相同的那一项）的 `uses`
 * （engine/evaluation.py needed：沿切换实际走的那一路往上找，taken_ports 是「走哪一路」唯一的判定）。网页不自己沿
 * 全部连线往上找、再按节点的 unused 去：unused 是「全图没有哪个结果要它」，不是相对这次计算的目标，一个节点在
 * 没走的那一路上、却另有别的终点用它时会被当成阻碍，单算没走那一路上的节点时它自己缺的文件又查不出来。
 * 回复里找不到这一项（没有 uses）时只查目标本身。
 *
 * 不用于决定计算前上传哪些素材：上传哪些素材、每份上传哪些通道由服务器决定，即状态回复中的
 * `policy.computes`（本次实际计算的节点）与各节点的 `channels`（`graph/apply.ts pickedFor / sendPicked`）。 */
function nodesCooked(reply: StatusReply, targets: string[]): string[] {
  const same = (c: CookCase | null | undefined) => !!c && c.targets.length === targets.length && c.targets.every((t, i) => t === targets[i]);
  const cook = same(reply.deliver) ? reply.deliver : Object.values(reply.nodes).map((n) => n.policy).find(same);
  return cook?.uses ?? targets;
}

/** 计算 `targets`（哪些由回复的 policy 说）为什么还不能提交：页面在发送前就能看到的（范围写错、文件没选或还在传、
 * 接入的数据不接受的选项）、服务器判为不对的连线，以及服务器给节点自身的错误。 */
export function blockers(ctx: BlockContext, targets: string[]): Blocker[] {
  const range = rangeProblem(ctx.cookRange, ctx.plan);
  const out: Blocker[] = range ? [{ node: "", message: range }] : [];
  // 同一根因只说一次：读取节点没选文件时，从它出来的线都在「等」（B-WIRE-WAITS「还没有选蒙皮角色」）。缺的是文件，由它自己
  // 那一条 B-COOK-NOFILE 说（它在这次计算里时）；线上不再各说一遍。它不在这次计算里（只被「有没有」看一眼，engine/
  // evaluation.py present）时也不拦：服务器同样照收（engine/demand.py：缺素材的原因报在源头）。选了文件、只是还没选
  // 里面的哪一项，线上的这一句就是唯一的说法，照常拦
  const fileless = (src: string) => {
    const n = ctx.nodes.find((m) => m.id === src);
    const d = n && ctx.nodeDefs[n.data.typeId];
    return !!n && !!d && missingFiles(d, n.data.params, ctx.reply.nodes[src]).length > 0;
  };
  for (const id of nodesCooked(ctx.reply, targets)) {
    const node = ctx.nodes.find((n) => n.id === id);
    const def = node && ctx.nodeDefs[node.data.typeId];
    if (!node || !def) continue;
    const name = nodeWord(id, node.data.typeId); // the messages' parameter (graph/naming.ts)
    const status = ctx.reply.nodes[id];
    const own: Message[] = [];
    if (!nodeUsable(ctx.applies, def.id)) own.push(msg("B-COOK-UNAVAILABLE", { node: name, reason: nodeWhy(ctx.applies, def.id) }));
    for (const w of ctx.reply.wires)
      if (w.to[0] === id && w.state !== "ok" && w.problem && !(w.state === "waiting" && fileless(w.from[0]))) own.push({ ...fromServer(w.problem), port: w.to[1] });
    for (const p of def.params) {
      const value = node.data.params[p.name];
      if (greyed(status?.applies, p.name)) continue;
      const going = FILE_IN.includes(p.widget ?? "") ? ctx.uploadBlocked(id, p.name, name) : undefined;
      if (going) own.push({ ...going, param: p.name });
      else if (missingFiles(def, node.data.params, status).includes(p)) own.push(msg("B-COOK-NOFILE", { node: name, what: p.label }, { param: p.name }));
      // 当前选中的选项不可用：接入的数据类型不符合要求（option_applies，由服务器计算，id 为 "<参数>=<选项>"）。
      // 在提交前拦截，不交由服务器报错
      const why = whyOf(status?.applies, `${p.name}=${String(value)}`);
      if (why) own.push(msg("B-COOK-OPTION", { node: name, setting: p.label, option: optionName(p, value), reason: why }, { param: p.name }));
    }
    // 服务器的错误只在节点排不出计划时才挡点击（必需的输入没接、文件被拒、接进来的值它不收）；之前计算留下的失败（它的
    // outcome：失败，或因上游失败而跳过）会由这次点击重试（engine/cook.py），从来不是不提交的理由
    const server = !status?.outcome ? status?.error : undefined;
    if (!own.length && server) own.push(fromServer(server));
    out.push(...own.map((message) => ({ node: id, message })));
  }
  return out;
}

/** 同一个原因报在好几个节点上（服务器在真正出错的节点下游的每一个上都点名它）：只说一次，有指向参数的就用那一条。 */
export function dedupeBlockers(found: Blocker[]): Blocker[] {
  const byText = new Map<string, Blocker>();
  for (const b of found) {
    const cur = byText.get(b.message.text);
    if (!cur || (!cur.message.param && b.message.param)) byText.set(b.message.text, b);
  }
  return [...byText.values()];
}

/** 计算 `targets` 涉及的节点上尚存的提示（用法检查与计算时说的，W 与 N：被拒的连线是阻碍，信息只留在日志里）：一次点击的
 * 汇总从它们开始，所以已算过的节点上的警告在再算时会重新列出，而不只是留在它的标记上。 */
export function standing(ctx: BlockContext, targets: string[]): Blocker[] {
  return nodesCooked(ctx.reply, targets).flatMap((id) => nodeMessages(ctx.reply.nodes, id).filter((m) => !m.refused && (m.level === "W" || m.level === "N")).map((m) => ({ node: id, message: fromServer(m) })));
}

/** 服务器对一个节点说的：用法检查，以及计算时说的（lab2shot/messages）。 */
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

