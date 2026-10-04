import { api, type StatusReply } from "../api";
import type { MessageJson } from "../api/applies";
import { handleNode } from "../state/handleView";
import type { CookCase } from "../api/status";
import { follow, syncJob } from "./follow";
import { loadOutputs } from "./outputs";
import { cookBlocked, cookNote, readyForPause } from "../state/pause";
import { readyForQuota } from "../state/quota";
import { exposedParams, splitTarget, targetsOf, useCookInputs } from "../state/cookInputs";
import { getNodeDefs } from "../state/catalog";
import { useLook } from "../state/look";
import { planAtNow, planHere, planOf, trustedReplyNow, useResults, type CookJob, type PlanAt } from "../state/results";
import { useViewer } from "../state/viewer";
import { pauseUpload, taskFor, uploadBlocker, uploadStopsClick, useUploads } from "../transfer/uploads";
import { abandonSubmit, pickedFor, sendPicked, submitAbandoned } from "./apply";
import { sayJudgedWires } from "./judged";
import { answerStatusWith } from "./asking";
import { type BlockContext, type Blocker, blockers, dedupeBlockers, deliverBlocked, deliveryNodes, cookSpan, frameLimitProblem, isLive, rangeProblem, standing } from "./nodes";
import { snapshotNow } from "./snapshot";
import { fromServer, msg, reasonOf, say, textOf, type Message } from "../state/say";
import { MessageError } from "../messages/message";
import { ApiError } from "../platform/http";
import { toJSON, stopStatusRefresh } from "./document";
import { useSession } from "../state/session";
import { focusFollows, focusSubmitted } from "../editor/AppMode";
import { findSubmitted, LineGaveUp, lineDown, newSubmitKey, withLine } from "./submitLine";
import { blocksOf, useItems, viewForAsk } from "../state/items";
import { BLOCKED_NOTE, outcomeView } from "../model/nodeOutcome";
import { wordIn } from "./naming";
import { t } from "../i18n/t";

export { fileJSON, loadGraph, redo, savePoint, toJSON, undo } from "./document";
export { addBox, addChain, arrangeGraph, addNode, addPortRow, connect, copySelection, deleteElements, duplicateSelection, deriveParams, insertNode, mergeToExr, moveWires, pasteCopied, pickFile, renameNode, setComment, setInterface, setParam, setParams, setParamsAcross, toggleBox, toggleExposed, toggleOnNode, togglePromoted } from "./edit";

/** 跨 store 的操作：「添加节点」「连线」这类使用者操作同时改 state/cookInputs.ts（数据）和 state/look.ts（位置），
 * 「计算」读另外三个、写 state/results.ts。这些不属于任何一个 store，所以放在这里，而不是硬挂在其中某一个上。
 * 下面每个函数直接读写 store（`.getState()` / actions）；组件调用这些函数，从不自己调 zustand 的 `set`。 */



// ------------------------------------------------------------------ status, cook, deliver

const upload = () => useUploads.getState().tasks;

function blockContext(reply: StatusReply): BlockContext {
  return {
    ...snapshotNow(),
    reply,
    cookRange: useCookInputs.getState().cookRange,
    plan: planHere(),
    // 素材仍在使用者本机上的情况不拦截（点击「计算」时先上传：`graph/apply.ts sendPicked`）；
    // 只拦截确实无法继续的情况（刷新后文件不再可用、服务器拒绝、正在上传）
    uploadBlocked: (id, param, label) => {
      const going = taskFor(upload(), id, param);
      return going && uploadStopsClick(going) ? uploadBlocker(going, label) : undefined;
    },
    applies: useSession.getState().state?.applies,
  };
}

/** 显示节点对上当前编辑的 plan（state/results.ts planOf）；还没对上为 null，范围与帧数都按「还不知道」不拦。 */
const shownPlan = planHere;

/** 用当前的计算范围和显示节点的 plan 调 rangeProblem()：顶栏不必自己凑参数。 */
export function rangeProblemNow(): Message | null {
  return rangeProblem(useCookInputs.getState().cookRange, shownPlan());
}

/** 本次计算的帧数及是否超出上限：计算范围为使用者填写的范围与素材的交集（graph/nodes.ts cookSpan），
 * 未填写时为素材全部。素材缩短、旧范围仅部分落在素材内时，按实际计算的范围计数，而非旧范围。 */
export function frameLimitProblemNow(): Message | null {
  const plan = shownPlan();
  if (!plan) return null; // 还没对上：不知道素材覆盖哪些帧，不拦（提交时等对上的回复，服务器也会再核）
  return frameLimitProblem(cookSpan(useCookInputs.getState().cookRange, plan) ?? (plan?.range as [number, number] | undefined) ?? null, useResults.getState().maxFrames);
}

export function setCookRange(r: [string, string] | null): void {
  const plan = shownPlan();
  const all = r && plan?.range && r[0].trim() === String(plan.range[0]) && r[1].trim() === String(plan.range[1]);
  useCookInputs.getState().setCookRange(all ? null : r);
}

/** 视图正在看的显示节点的口（空：它的主输出）：「计算」提交的与状态回复的预估（plan，cookHold 读它）按同一个判，
 * 服务器 readiness 看的是同一件事。 */
const shownPorts = (): string[] => {
  const port = useLook.getState().displayPort;
  return port ? [port] : [];
};

/** 本页面向服务器查询状态的次数，以及已采纳的回答序号（两个数同时变化，因此作为一份状态，而非两个独立变量） */
const status = { asked: 0, answered: 0 };

/** 按节点图现在的样子问服务器：每次编辑一次请求，连同显示节点的 plan。计算输入已经变了的回复、或比已采纳的更早的
 * 回复丢掉。它从不触发计算：视图只显示已算出的，计算要点「计算」。 */
let againTimer: ReturnType<typeof setTimeout> | null = null; // refreshStatus: asking again while a file is being read

export async function refreshStatus(o: { signal?: AbortSignal; rethrow?: boolean } = {}): Promise<void> {
  const ci = useCookInputs.getState();
  if (!ci.order.length) return;
  const version = ci.version;
  const serial = ++status.asked;
  let reply: StatusReply;
  // handle data only for the node something draws handles of (state/handleView.ts); the copy held is offered by its key
  // only when it is of that node
  const display = useLook.getState().displayId;
  const port = useLook.getState().displayPort; // the reply's plan is for this output (state/results.ts planOf)
  const about = handleNode();
  const held = useResults.getState().reply?.handle_data;
  const handles = { node: about, key: about && held && held.node === about ? held.key : "" };
  try {
    reply = await api.status(toJSON(), version, display, viewForAsk(blocksOf(useResults.getState().reply), useItems.getState().view), handles, port ? [port] : [], o.signal, cookButtonNodes());
  } catch (e) {
    if (o.rethrow && lineDown(e)) throw e; // 提交路上：连不上由它按规矩重试（graph/submitLine.ts），已有的结果先留着
    if (useCookInputs.getState().version !== version || serial < status.asked) return; // 更新的一次询问已在路上（它会答）：不清掉已有的结果
    const before = useResults.getState().refused;
    useResults.getState().clearResults();
    // 留下服务器自己说的答不了的原因（code 与 detail）：「计算」照它说，而不是猜「刚改过，或者网络断了」——回答一直以
    // 同一种方式拒绝时，原因从来不是这个。报错只在日志里说（顶栏不显示）：换了一个原因时写一次
    const refused = e instanceof ApiError && e.code ? fromServer({ code: e.code, text: e.message }) : null;
    useResults.getState().setRefused(refused);
    if (refused && (before?.code !== refused.code || before?.text !== refused.text)) say(refused);
    for (const id of ci.order) {
      const st = useResults.getState().byNode[id];
      if (!isLive(st?.status)) useResults.getState().setNodeStatus(id, { status: "idle" });
    }
    return;
  }
  if (useCookInputs.getState().version !== version || serial < status.answered) return; // 其间计算输入已经变了
  status.answered = serial;
  useResults.getState().setReply(reply, port);
  // a file is still being read in the background (an import's listing): ask again soon, so what it holds shows by
  // itself; one timer at a time, replaced by any newer answer
  if (againTimer) clearTimeout(againTimer);
  againTimer = reply.again_ms ? setTimeout(() => void refreshStatus(), reply.again_ms) : null;
  const results = reply.nodes;
  for (const id of ci.order) {
    const st = useResults.getState().byNode[id];
    // 点击被拦下的原因一直留到计算输入变化（results.ts blockedAt）：视图变化时也会来回复，每次回复都清标记的话，
    // 点击刚标上就被清掉。
    if (st && version !== useResults.getState().blockedAt) useResults.getState().setNodeStatus(id, { blocked: undefined });
    if (isLive(st?.status)) continue;
    // 因出错而没有结果：自己出错（出错）或上游出错（已跳过），被「阻断」关着（已跳过，底行「已跳过（被阻断）」），按服务器
    // 记下的（engine/cook.py）；规则在 model/nodeOutcome.ts
    const view = outcomeView(results[id]?.outcome, !!results[id]?.cached);
    if (view.note !== null && st?.note !== view.note) useResults.getState().setNodeStatus(id, { status: view.status, note: view.note });
    else if (view.note === null && st?.note === BLOCKED_NOTE) useResults.getState().setNodeStatus(id, { status: view.status, note: "" });
    else if (st?.status !== view.status) useResults.getState().setNodeStatus(id, { status: view.status });
  }
  sayJudgedWires(reply);
  // 计算范围不随素材改写：提交的是它与素材的交集（toJSON 的 frames = cookSpan），节点图里存的是使用者写的
}
answerStatusWith(refreshStatus); // document.ts / follow.ts 经 graph/asking.ts 叫它（不 import actions：环）

/** 当前条目变了（条目栏、条目列表）：视图停在逐项处理块的另一个条目上（`where` 是它的逐项开始）。带着新的 `view`
 * 再问一次服务器，服务器答回块内每个节点在该条目上的状态。这是视图设置：不触发计算，计算输入不变，结果也不会过期。
 *
 * 列表的结果不经过此处：视图会绘制列表中的每一条（view/plan.ts 的 `expand`），
 * 不存在「当前条目」，因此不会有非块的 `where` 进入此处。 */
export function showItem(where: string, key: string): void {
  if (useItems.getState().view[where] === key) return;
  useItems.getState().setItem(where, key);
  void refreshStatus();
}

/** 与当前计算输入、当前显示节点与口对应的状态回复（它的预估对得上：state/results.ts planOf）：手上有就用，否则现在去问
 * （编辑后、或点「计算」刚把显示换到这个节点、换了看的口时马上点击会等它，而不是拿着别处的预估提交）。显示节点没有
 * 预估（服务器回 plan: null）时问一次就用。null：服务器没答，或其间节点图变了。 */
async function currentReply(): Promise<StatusReply | null> {
  const now = trustedReplyNow();
  const at = planAtNow();
  if (now && (!at.node || planOf(useResults.getState(), at))) return now;
  stopStatusRefresh();

  // 提交路上的这一问有时限、连不上有限次重试（graph/submitLine.ts）；到上限抛 LineGaveUp，cook / deliverAll 收下
  await withLine((signal) => refreshStatus({ signal, rethrow: true }));
  return trustedReplyNow();
}

/** 将本次所需、仍在使用者本机上的素材先行上传，返回重新查询后的上下文（null 表示上传失败或服务器未答复，
 * 两种情况均已给出提示）。选择文件时不上传，点击「计算」时才上传实际需要的素材；开始上传前提示「正在上传素材…
 * 传完自动开始计算」，文件参数行与节点上均有进度显示（`ui/UploadState.tsx`、`transfer/uploadText.ts`）。
 *
 * 必须重新查询的原因：现有回复是在上传前获取的，当时服务器无法打开该素材，读取节点上带有
 * 「文件不在服务器上」的提示。若据此检查问题，刚上传完成的本次计算也会被拦截。 */
async function afterSending(ctx: BlockContext, wanted: CookCase): Promise<BlockContext | null> {
  if (!pickedFor(ctx, wanted).length) return ctx;
  const doc = useViewer.getState().docId; // 这一次打开的文档（每次 loadGraph 换一个；「另存为」不换）
  if (!(await sendPicked(ctx, wanted)) || submitAbandoned()) return null; // 没传完（已说原因），或顶栏取消了这次提交
  // 上传可能要几分钟，其间打开了另一份文档：这次点击是给原来那份的，不能拿新文档（toJSON 取的是当前文档）接着提交。
  // 素材照样传完，归原来那份文档（transfer/uploads.ts 按 graphId 归属）
  if (useViewer.getState().docId !== doc) {
    say(msg("N-COOK-DOCCHANGED"));
    return null;
  }
  const fresh = await currentReply();
  if (fresh) return blockContext(fresh);
  say(useResults.getState().refused ?? msg("B-COOK-NOSTATUS"));
  return null;
}

/** 现在点「计算」（或打包）会不会在提交前就被拦下、为什么——所有「计算」入口的置灰和文字（按钮参数 editor/buttonActions.tsx、
 * 节点右键菜单 editor/FlowParts.tsx）与 cook / deliverAll 的闩读的是这同一处，看到能点的就不会被这几条拦：
 * - "submitting"：点过「计算」、素材还在上传或正在提交（一次只提交一个，否则反复点会各自上传、提交好几个任务）；
 * - "busy"：这张节点图已经有一个任务；
 * - "paused"：计算任务关着，或存储配额满了（state/pause.ts cookBlocked；点了由 readyForPause / readyForQuota 说明哪一条）；
 * - "unplannable"：问的是节点 `at.node`、它正是显示节点，而服务器对它的预估（状态回复的 plan，farm/timings.py look）说
 *   算不了（例如唯一的读取节点没选文件：要算的全都会失败，提交会被 readiness 拒绝）；原因是 `planError`。只信对上当前编辑
 *   （`at.version`，计算输入的版本）的预估（state/results.ts planOf）：改过之后、新回复到之前不按旧原因拦，也不按旧预估放。
 *   别的节点没有预估，点了由服务器照常拒绝并说原因。 */
export type CookHold = "submitting" | "busy" | "paused" | "unplannable" | null;
export const cookHold = (s: ReturnType<typeof useResults.getState>, at?: PlanAt): CookHold =>
  s.submitting ? "submitting" : s.job ? "busy" : cookBlocked(s.queueSwitches, s.storage) ? "paused" : at && planError(s, at) ? "unplannable" : null;
/** 点了「计算」就会自己解决的原因：素材选了、字节还在本机（transfer/uploads.py not_sent_yet）——点「计算」先上传、再提交
 * （afterSending），所以它不是算不了，不置灰。 */
const RESOLVED_BY_COOK = new Set(["E-UPLOAD-SENDING"]);
/** 节点现在算不了的原因：服务器的预估说的（显示节点的 plan，别的按钮节点的 holds），或者这组计算输入下点它被拦下的
 * （state/results.ts stopped）；null：没说算不了。 */
export const planError = (s: ReturnType<typeof useResults.getState>, at: PlanAt): MessageJson | null => {
  const e = planOf(s, at)?.error ?? (at.node && s.forCookInputs === at.version && s.plan?.node !== at.node ? s.reply?.holds?.[at.node] ?? null : null);
  if (e && !RESOLVED_BY_COOK.has(e.code)) return e;
  return at.node && s.stopped?.node === at.node && s.stopped.version === at.version ? s.stopped.message : null;
};

/** 这次点击（节点 `who` 的「计算」）被拦下：原因记在它的按钮下，直到计算输入变化（planError）。 */
const stoppedAt = (who: string, message: Message): void =>
  useResults.setState({ stopped: { node: who, version: useCookInputs.getState().version, message } });

/** 页面上除了显示节点以外还给出「计算」的节点（参数界面里的计算按钮：卡片上的「计算并打包」、阶段按钮）：状态请求带上它们
 * （StatusRequest holds），每个按钮不看视图显示谁，都按服务器说的置灰、写原因（planError）。 */
function cookButtonNodes(): string[] {
  const ci = useCookInputs.getState();
  const defs = getNodeDefs();
  const out = new Set<string>();
  for (const x of exposedParams(ci.exposed))
    for (const [nid, pname] of targetsOf(x).map(splitTarget)) {
      const p = defs[ci.nodes[nid]?.typeId ?? ""]?.params.find((q) => q.name === pname);
      if (p?.widget === "button" && p.action === "cook") out.add(nid);
    }
  return [...out];
}
/** 「计算」点不了的完整原因（`hold` 是 cookHold 的结果）：所有入口说同一句——节点右键菜单的悬停提示、聚焦页顶栏「计算」
 * 的悬停提示。计算任务关着、存储占满说的是哪一条（state/pause.ts cookNote），服务器说算不了说的是服务器的原句。""：没拦。 */
export function cookHoldWhy(s: ReturnType<typeof useResults.getState>, hold: CookHold, at?: PlanAt): string {
  if (hold === "submitting") return t("ui.graph.hold_submitting");
  if (hold === "busy") return t("ui.graph.hold_busy");
  if (hold === "paused") return cookNote(s.queueSwitches, s.storage).trim();
  if (hold === "unplannable") {
    const e = at ? planError(s, at) : null;
    return e ? t("ui.graph.hold_unplannable", { reason: textOf(e) }) : "";
  }
  return "";
}

/** 打开另一张节点图（打开 json 文件、从模板新建、打开「我的模板」、从队列「加载」）之前：这张图有任务在算、或点了
 * 「计算」正在上传素材 / 提交时不许换图（换了图，这个任务的进度、结果和上传都归原来那张，页面上就对不上了），
 * 拦下并说一句。判定就是 cookHold 的 "submitting" / "busy"，与「计算」按钮置灰同一处。返回 true：不打开。 */
export function openHeld(): boolean {
  const hold = cookHold(useResults.getState());
  if (hold !== "submitting" && hold !== "busy") return false;
  say(msg("B-GRAPH-COOKING"));
  return true;
}

/** 顶栏「取消」（点了「计算」、素材还在上传或正在提交时）：停下这张图正在传的素材（transfer/uploads.ts pauseUpload：
 * 已传的部分和归属都保留，下次「计算」接着传），这次提交不再发出；闩随 cook / deliverAll 结束放开。 */
export function cancelSubmit(): void {
  if (!useResults.getState().submitting || submitAbandoned()) return;
  abandonSubmit(true);
  const graph = useCookInputs.getState().graphId;
  for (const t of Object.values(useUploads.getState().tasks))
    if (t.graphId === graph && (t.state === "sending" || t.state === "waiting" || t.state === "finishing")) pauseUpload(t.key);
  say(msg("N-COOK-SUBMITCANCELLED"));
}

/** cook / deliverAll 的闩：正在提交时只说一句；否则先占上「正在提交」（`who`：哪个节点，打包为 "*"），再看页面记着的
 * 任务——它先按服务器核对（graph/follow.ts syncJob，以服务器为准：事件流断过、服务器上它早已结束的记录就此解除），
 * 真还在算才说一句、放开闩；计算任务关着、存储配额满了也在这里说哪一条。返回 true：这次不提交。占闩在核对之前，
 * 核对期间再点也只说一句，不会并出两次提交。 */
async function latched(who: string): Promise<boolean> {
  const at = { ...planAtNow(), node: who };
  const hold = cookHold(useResults.getState(), at);
  if (hold === "submitting") {
    say(msg("B-COOK-SUBMITTING"));
    return true;
  }
  if (hold === "unplannable") { // 快捷键不看按钮置灰，也在这里说原因
    say(fromServer(planError(useResults.getState(), at)!));
    return true;
  }
  useResults.getState().setSubmitting(who);
  abandonSubmit(false);
  if (useResults.getState().job && (await syncJob())) {
    useResults.getState().setSubmitting(null);
    say(msg("B-COOK-BUSY"));
    return true;
  }
  // 计算任务关着、存储配额满了：在传素材之前就拦下并说哪一条——不能先传几 GB 再告诉交不了。快捷键（Ctrl+Enter /
  // Ctrl+Shift+Enter）不看按钮置灰，也走这里（cookHold 的 "paused" 与这两条是同一判定）
  if (!(await readyForPause(useResults.getState().queueSwitches, (m) => say(m))) || !readyForQuota(useResults.getState().storage, (m) => say(m))) {
    useResults.getState().setSubmitting(null);
    return true;
  }
  return false;
}

/** 点「计算」：该节点及它所需的（按服务器回复的 `policy`），作为一个任务进队列。 */
export async function cook(target: string): Promise<void> {
  if (await latched(target)) return;
  try {
    await cookNow(target);
  } catch (e) {
    if (!(e instanceof LineGaveUp)) throw e; // 连不上服务器、到上限停下：日志里已说
  } finally {
    useResults.getState().setSubmitting(null);
  }
}

async function cookNow(target: string): Promise<void> {
  const reply = await currentReply();
  const wanted = reply?.nodes[target]?.policy;
  if (!reply || !wanted) {
    say(useResults.getState().refused ?? msg("B-COOK-NOSTATUS"));
    return;
  }
  const { targets } = wanted;
  let ctx = blockContext(reply);
  // 仍在使用者本机上的素材先行上传，完成后自动继续计算（实现见 `graph/apply.ts sendPicked`；上传哪些素材、每份上传
  // 哪些通道均由服务器在本次回复中给出：`wanted.computes` 与各节点的 `channels`）。
  const after = await afterSending(ctx, wanted);
  if (after === null) return;
  ctx = after;
  const found = blockers(ctx, targets);
  if (found.length) {
    sayBlocked(found, target);
    return;
  }
  // 帧数超过服务器上限：在发送前说明，不发送给服务器（服务器同样会独立拒绝）
  const over = frameLimitProblemNow();
  if (over !== null) {
    say(over);
    return;
  }
  startSummary(standing(ctx, targets)); // 先将节点上尚存的提示输出一遍（写入日志）
  const busy = await jobInTheWay();
  if (busy) {
    say(busy);
    return;
  }
  if (submitAbandoned()) return; // 顶栏取消了这次提交
  try {
    // 视图显示的输出口（显示了某个口时）：服务器只算它需要的（`show`）
    const version = useCookInputs.getState().version;
    const follows = focusFollows(target);
    const graph = toJSON(), show = shownPorts(), submit = newSubmitKey(); // 重发的是同一次点击：同一张图、同一个提交键
    const { job } = await withLine((signal) => api.cook(graph, version, target, show, follows, { submit, signal }), { before: () => findSubmitted(submit) });
    focusSubmitted(job, version, follows);
    const label = wordIn(ctx.nodes, target);
    say(msg("I-JOB-COOK", { node: label, job }));
    await confirmed(job);
    follow(job, target);
  } catch (e) {
    if (e instanceof LineGaveUp) return;
    submitRefused(e, target);
  }
}

export function cookNode(id: string): void {
  useLook.getState().setDisplay(id);
  void cook(id);
}

export async function deliverAll(): Promise<void> {
  if (await latched("*")) return;
  try {
    await deliverNow();
  } catch (e) {
    if (!(e instanceof LineGaveUp)) throw e;
  } finally {
    useResults.getState().setSubmitting(null);
  }
}

async function deliverNow(): Promise<void> {
  const snap = snapshotNow();
  const missing = deliverBlocked(snap.nodes, snap.nodeDefs, deliveryNodes(snap.nodes, snap.nodeDefs));
  if (missing) {
    sayBlocked([missing], "*");
    return;
  }
  const reply = await currentReply();
  if (!reply?.deliver) {
    say(useResults.getState().refused ?? msg("B-COOK-NOSTATUS"));
    return;
  }
  const { targets } = reply.deliver; // 这次提交打包的「输出」节点，按服务器解析的
  let ctx = blockContext(reply);
  const after = await afterSending(ctx, reply.deliver);  // 仍在使用者本机上的素材先行上传（与「计算」相同）
  if (after === null) return;
  ctx = after;
  const found = blockers(ctx, targets);
  if (found.length) {
    sayBlocked(found, "*");
    return;
  }
  const over = frameLimitProblemNow();
  if (over !== null) {  // 帧数超过服务器上限：在发送前拦截
    say(over);
    return;
  }
  startSummary(standing(ctx, targets)); // 这次提交的汇总从这里开始
  const busy = await jobInTheWay();
  if (busy) {
    say(busy);
    return;
  }
  if (submitAbandoned()) return; // 顶栏取消了这次提交
  try {
    const version = useCookInputs.getState().version;
    const follows = focusFollows("*");
    const graph = toJSON(), submit = newSubmitKey();
    const { job } = await withLine((signal) => api.deliver(graph, version, follows, { submit, signal }), { before: () => findSubmitted(submit) });
    focusSubmitted(job, version, follows);
    say(msg("I-JOB-DELIVER", { job }));
    await confirmed(job);
    follow(job, targets[0]); // 在它的第一个「输出」上跟踪这个任务
  } catch (e) {
    if (e instanceof LineGaveUp) return;
    submitRefused(e, "*");
  }
}

/** 提交后的第一次确认：任务在服务器上了吗（有时限、连不上有限次重试）。确认不成不算提交失败——任务已经在服务器上，
 * 事件流连上以后照常跟踪（platform/events.ts 自己会重连）；只在日志里说一句。 */
async function confirmed(job: string): Promise<void> {
  await withLine((signal) => api.jobState(job, signal), { gaveUp: msg("W-SUBMIT-UNCONFIRMED", { job }) }).catch((e) => {
    if (!(e instanceof LineGaveUp) && !(e instanceof ApiError)) throw e; // 服务器答了别的（找不到它）：交给跟踪去说
  });
}

/** 服务器拒绝了一次提交（计算、提交）：写进日志（报错只在日志里说，顶栏不显示）；「计算」的提示照它说，直到服务器再次回答这张图。 */
function submitRefused(e: unknown, who: string): void {
  // 服务器以 B 级拒绝了这张图（它说的原句，应用模式按 .app）：写到按钮下；别的（连接、服务器故障）只说一次，下次照常能点
  if (e instanceof MessageError && e.said.level === "B") stoppedAt(who, e.said);
  const why = msg("E-JOB-SUBMITFAILED", { reason: reasonOf(e) });
  useResults.getState().setRefused(why);
  say(why);
}

/** 一次点击的计算（一个节点，或全部「输出」）把它涉及的节点上尚存的提示说一遍（写进日志）。 */
function startSummary(said: Blocker[]): void {
  for (const b of said) say(b.message, b.node);
}

/** 一次还不能提交的点击：挡路的节点标成错误色，原因写在各自节点上（点它跳到它所说的参数）并写进日志，不弹窗。
 *
 * 阻止本次提交的原因在三处呈现：节点自身变红并显示原因（`byNode[节点].blocked`，`blockedAt` 使其保持显示
 * 直到计算输入变化）、页面跳转到该节点与对应参数（见下方代码）、以及日志（由 `say` 写入，顶栏图标显示红色计数）。 */
function sayBlocked(found: Blocker[], who: string): void {
  useResults.setState({ blockedAt: useCookInputs.getState().version }); // 标记一直留到计算输入变化
  stoppedAt(who, found[0].message);
  for (const b of found) if (b.node) useResults.getState().setNodeStatus(b.node, { blocked: b.message.text });
  for (const b of dedupeBlockers(found)) say(b.message, b.node);
  // 并带使用者过去：节点图选中挡路的节点，参数面板跳到它所说的参数。没有向服务器发送任何东西。
  const first = found.find((b) => b.node);
  if (!first) return;
  const viewer = useViewer.getState();
  viewer.setSelectedNodes([first.node]);
  if (first.message.param) viewer.revealParam(first.node, first.message.param);
  else viewer.select(first.node);
}

/** 同一时间只运行一个任务：有任务进行中时给出提示，使用者可自行取消后再提交。
 *
 * 返回「有任务进行中」的提示，或 null（可以提交）。不依赖页面自身的记录：对于页面记录的任务，
 * 先向服务器确认其是否仍存在；若不存在，说明页面漏收了「完成」事件（长连接静默失效时会发生：
 * 不断线、不报错、也不再有数据），此时立即解除并放行，避免使用者被早已完成的任务阻塞。
 * 静默返回会导致点击提交后长时间无反应且没有任何提示。 */
// 上传素材期间可能又有了任务记录（例如刷新后接着跟踪的）：提交前再按服务器核对一次（graph/follow.ts syncJob，同一处）
async function jobInTheWay(): Promise<ReturnType<typeof msg> | null> {
  const job = useResults.getState().job;
  if (!job) return null;
  return (await syncJob()) ? msg("N-JOB-INPROGRESS", { job: job.id }) : null;
}

export async function cancelCook(): Promise<void> {
  const job = useResults.getState().job;
  if (!job) return;
  try {
    await api.cancelCook(job.id);
  } catch (e) {
    say(msg("E-JOB-CANCELFAILED", { reason: reasonOf(e) }));
  }
}

/** 页面打开（或刷新）时：这张图仍在排队或计算的任务接着跟踪（别的图的任务不必跟踪：它们「输出」打的包留在队列里，
 * 队列 → 下载），这张图的「输出」之前打好的包从服务器读回（graph/outputs.ts）。 */
export async function resumeJobs(): Promise<void> {
  const queue = await api.queue().catch(() => null);
  const ci = useCookInputs.getState();
  for (const j of queue?.jobs ?? []) {
    if (!j.mine || (j.state !== "queued" && j.state !== "running")) continue;
    const ours = j.graph === ci.graphId && !!j.nodes?.length && j.nodes.every((id) => !!ci.nodes[id]);
    if (ours && !useResults.getState().job) follow(j.id, j.nodes![0]);
  }
  void loadOutputs();
}

export type { CookJob };
