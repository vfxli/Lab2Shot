import { api, type StatusReply } from "../api";
import type { CookCase } from "../api/status";
import { follow } from "./follow";
import { canWrite, cannot } from "../files/handles";
import { readyToSave, rememberDestinations, watchJob } from "../files/deliver";
import { readyForPause } from "../state/pause";
import { readyForQuota } from "../state/quota";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { planFor, trustedReplyNow, useResults, type CookJob } from "../state/results";
import { browserCanCompute } from "./rules";
import { hasLocalFile } from "../transfer/local";
import { useViewer } from "../state/viewer";
import { taskFor, uploadBlocker, uploadStopsClick, useUploads } from "../transfer/uploads";
import { pickedFor, sendPicked } from "./apply";
import { type BlockContext, type Blocker, blockers, dedupeBlockers, deliverBlocked, deliveryNodes, cookSpan, frameLimitProblem, isLive, rangeProblem, standing } from "./nodes";
import { snapshotNow } from "./snapshot";
import { fromServer, logMessage, msg, reasonOf, say, type Message } from "../state/say";
import { ApiError } from "../platform/http";
import { toJSON, stopStatusRefresh } from "./document";
import { wireKey } from "./rules";
import { getNodeDefs } from "../state/catalog";
import { useSession } from "../state/session";
import { blocksOf, useItems, viewForAsk } from "../state/items";

export { fileJSON, loadGraph, markSaved, redo, toJSON, undo } from "./document";
export { addBox, addChain, addNode, addPortRow, connect, deleteElements, deriveParams, insertNode, mergeToExr, pickFile, setLabel, setParam, setSaveTo, toggleBox, toggleExposed, toggleOnNode, togglePromoted } from "./edit";

/** Cross-store operations: a user action like "add a node" or
 * "connect a wire" touches state/cookInputs.ts (its data) and state/look.ts (its position) together, and "cook"
 * touches state/results.ts while reading the other three. None of that belongs to any one store, so it lives here
 * instead of being bolted onto one of them. Every function below reads and writes stores directly (`.getState()` /
 * actions); components call these, never zustand's `set` themselves. */



/** Wires connected since the last status reply: the reply that judges them says what is wrong with one, once. */
export const justWired = new Set<string>();

export function sayJudgedWires(reply: StatusReply): void {
  for (const w of reply.wires) {
    const id = wireKey(w.from[0], w.from[1], w.to[0], w.to[1]);
    if (!justWired.delete(id) || w.state === "ok" || !w.problem) continue;
    const problem = fromServer(w.problem);
    const fix = getNodeDefs()[w.fix];
    const node = useCookInputs.getState().nodes[w.to[0]]?.label ?? w.to[0];
    say(fix ? msg("W-WIRE-WRONGFIX", { problem, node, via: fix.label }, { port: w.to[1] }) : msg("W-WIRE-WRONG", { problem }, { port: w.to[1] }), w.to[0]);
  }
}

// ------------------------------------------------------------------ status, cook, deliver

const upload = () => useUploads.getState().tasks;

function blockContext(reply: StatusReply): BlockContext {
  return {
    ...snapshotNow(),
    reply,
    cookRange: useCookInputs.getState().cookRange,
    plan: planFor(useLook.getState().displayId),
    // 素材仍在使用者本机上的情况不拦截（点击「计算」时先上传：`graph/apply.ts sendPicked`）；
    // 只拦截确实无法继续的情况（刷新后文件不再可用、服务器拒绝、正在上传）
    uploadBlocked: (id, param, label) => {
      const going = taskFor(upload(), id, param);
      return going && uploadStopsClick(going) ? uploadBlocker(going, label) : undefined;
    },
    cannotChoose: (need) => cannot(need),
    canWrite,
    applies: useSession.getState().state?.applies,
  };
}

/** rangeProblem() fed the current cook range and the shown node's plan, so the top bar need not
 * assemble the arguments itself. */
export function rangeProblemNow(): Message | null {
  return rangeProblem(useCookInputs.getState().cookRange, planFor(useLook.getState().displayId));
}

/** 本次计算的帧数及是否超出上限：计算范围为使用者填写的范围与素材的交集（graph/nodes.ts cookSpan），
 * 未填写时为素材全部。素材缩短、旧范围仅部分落在素材内时，按实际计算的范围计数，而非旧范围。 */
export function frameLimitProblemNow(): Message | null {
  const r = useResults.getState();
  return frameLimitProblem(cookSpan(useCookInputs.getState().cookRange, r.plan ?? null), r.maxFrames);
}

export function setCookRange(r: [string, string] | null): void {
  const plan = useResults.getState().plan;
  const all = r && plan?.range && r[0].trim() === String(plan.range[0]) && r[1].trim() === String(plan.range[1]);
  useCookInputs.getState().setCookRange(all ? null : r);
}

/** 本页面向服务器查询状态的次数，以及已采纳的回答序号（两个数同时变化，因此作为一份状态，而非两个独立变量） */
const status = { asked: 0, answered: 0 };

/** Ask the server about the graph as it is: one request per edit, the shown
 * node's plan with it. A reply for cook inputs that have moved on since, or older than one already taken, is dropped.
 * `autoCook`: once it is in, what the shown node shows cooks by itself when that is light work delivering nothing. */
export async function refreshStatus(autoCook = true): Promise<void> {
  const ci = useCookInputs.getState();
  if (!ci.order.length) return;
  const version = ci.version;
  const serial = ++status.asked;
  let reply: StatusReply;
  try {
    reply = await api.status(toJSON(), version, useLook.getState().displayId, viewForAsk(blocksOf(useResults.getState().reply), useItems.getState().view));
  } catch (e) {
    if (useCookInputs.getState().version !== version || serial < status.answered) return; // an even newer ask is on its way
    useResults.getState().clearResults();
    // The server's own words for why it could not answer (its code and detail) are kept: 计算 reports them instead of
    // guessing「刚改过，或者网络断了」, which is never the reason when the answer keeps refusing the same way.
    useResults.getState().setStatusProblem(e instanceof ApiError && e.code ? fromServer({ code: e.code, text: e.message }) : null);
    for (const id of ci.order) {
      const st = useResults.getState().byNode[id];
      if (!isLive(st?.status)) useResults.getState().setNodeStatus(id, { status: "idle" });
    }
    return;
  }
  if (useCookInputs.getState().version !== version || serial < status.answered) return; // the cook inputs moved on meanwhile
  status.answered = serial;
  useResults.getState().setReply(reply);
  const results = reply.nodes;
  for (const id of ci.order) {
    const st = useResults.getState().byNode[id];
    // Why a click was stopped stands until the cook inputs change (results.ts blockedAt): a reply also comes when the
    // view changes, so clearing the marks on every reply would remove them right after the click set them.
    if (st && version !== useResults.getState().blockedAt) useResults.getState().setNodeStatus(id, { blocked: undefined });
    if (isLive(st?.status)) continue;
    // no result because of an error: its own (出错) or one above it (已跳过), as the server keeps it (engine/cook.py)
    const outcome = results[id]?.outcome?.state;
    const status = outcome === "failed" ? "error" : outcome === "skipped" ? "skipped" : results[id]?.cached ? "cooked" : "idle";
    if (st?.status !== status) useResults.getState().setNodeStatus(id, { status });
  }
  sayJudgedWires(reply);
  // 素材更换后计算范围随素材调整：节点图中保存的旧范围（如 100–300）在素材重新上传为 200–300 后
  // 会持续阻止提交并提示缺帧。因此新素材到达后，范围自动收缩到素材之内（计算范围的两个数值立即更新）；
  // 仅在完全不重叠（填写的是另一段镜头的帧号）时保持不变，由 rangeProblem 提示使用者修改。
  const ranged = useCookInputs.getState().cookRange;
  const shownRange = (useResults.getState().plan?.range as [number, number] | undefined) ?? null;
  if (ranged && shownRange) {
    const now = cookSpan(ranged, useResults.getState().plan ?? null);
    if (now && (String(now[0]) !== ranged[0].trim() || String(now[1]) !== ranged[1].trim())) {
      setCookRange([String(now[0]), String(now[1])]);
    }
  }
  // Houdini-style: what the shown node shows cooks by itself, when that is light work delivering nothing
  const disp = useLook.getState().displayId;
  const shown = disp ? results[disp]?.policy.shown : undefined;
  if (!autoCook || !disp || !shown || useResults.getState().job) return;
  const pending = shown.computes.filter((id) => results[id]?.fingerprint);
  const retry = pending.some((id) => results[id]?.outcome); // failed or skipped: tried again on a click, never by showing
  // 浏览器可自行计算的步骤不交由服务器：在「选人」上更改选择不提交任何计算，遮罩图与合成图不在服务器生成，
  // 也不回传，画面立即更新。这些步骤（选人、拆成列表、人物框转遮罩、图像合成）由浏览器当场计算（view/evaluate.ts），
  // 交由服务器重算只会让使用者多等一次队列。点击「计算」或提交时仍使用服务器的结果，该结果具有权威性
  const mine = pending.length > 0 && pending.every((id) => browserCanCompute(results[id]));
  // 文件位于使用者本机的，一律不交由服务器：双击读取序列节点时，视图播放的是无损原图，节点上不应显示「计算中」。
  // 判据为 `transfer/local.ts hasLocalFile`（定义见 `view/origin.ts` 开头）：
  // 确实持有原件时才跳过，无法获取时照常计算（不显示空白画面）。
  const local = pending.length > 0 && pending.every((id) => hasLocalFile(id, results[id]?.fingerprint));
  if (pending.length && !retry && !mine && !local && shown.kind.by_itself) void cook(disp, true);
}

/** 当前条目 changed (the item bar, the item list): the view now stands on another item of a 逐项处理 block
 * (`where` is its 逐项开始). The server is asked again, with the new `view`, and answers with that item's state for
 * every node inside it. It is a view setting: nothing is cooked (`autoCook` false), the cook inputs do not move and no
 * result goes stale.
 *
 * 列表的结果不经过此处：视图会绘制列表中的每一条（view/plan.ts 的 `expand`），
 * 不存在「当前条目」，因此不会有非块的 `where` 进入此处。 */
export function showItem(where: string, key: string): void {
  if (useItems.getState().view[where] === key) return;
  useItems.getState().setItem(where, key);
  void refreshStatus(false);
}

/** The status reply for the cook inputs as they are: the one in hand, else asked now (a click right after an edit
 * waits for it rather than judging the graph itself). null: the server did not answer, or the graph changed meanwhile. */
async function currentReply(): Promise<StatusReply | null> {
  const now = trustedReplyNow();
  if (now) return now;
  stopStatusRefresh();

  await refreshStatus(false);
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
  if (!(await sendPicked(ctx, wanted))) return null;
  const fresh = await currentReply();
  if (fresh) return blockContext(fresh);
  say(useResults.getState().statusProblem ?? msg("B-COOK-NOSTATUS"));
  return null;
}

export async function cook(target: string, auto = false): Promise<void> {
  if (useResults.getState().job) {
    if (!auto) say(msg("B-COOK-BUSY"));
    return;
  }
  const reply = auto ? trustedReplyNow() : await currentReply();
  const policy = reply?.nodes[target]?.policy;
  if (!reply || !policy) {
    if (!auto) say(useResults.getState().statusProblem ?? msg("B-COOK-NOSTATUS"));
    return;
  }
  const wanted = auto ? policy.shown : policy.click;
  const { targets, kind } = wanted;
  if (auto && (!targets.length || !kind.by_itself)) return;
  let ctx = blockContext(reply);
  // 仍在使用者本机上的素材先行上传，完成后自动继续计算（实现见 `graph/apply.ts sendPicked`；上传哪些素材、每份上传
  // 哪些通道均由服务器在本次回复中给出：`wanted.computes` 与各节点的 `channels`）。
  // 由显示节点触发的轻量计算在素材未上传时不执行任何操作：既不应在后台上传数 GB 素材，也不应强行
  // 发送给服务器，因为服务器无法打开尚未上传的素材，只会在日志中多出一条「文件已经不在服务器上了」
  if (auto && pickedFor(ctx, wanted).length) return;
  const after = await afterSending(ctx, wanted);
  if (after === null) return;
  ctx = after;
  const found = blockers(ctx, targets);
  if (found.length) {
    if (auto) return;
    sayBlocked(found);
    return;
  }
  // 帧数超过服务器上限：在发送前说明，不发送给服务器（服务器同样会独立拒绝）
  const over = frameLimitProblemNow();
  if (!auto && over !== null) {
    say(over);
    return;
  }
  if (!auto) startSummary(standing(ctx, targets)); // 对于点击发起的计算，先将节点上尚存的提示输出一遍（写入日志）
  if (!auto && !(await readyForPause(useResults.getState().queueSwitches, kind, (m) => say(m)))) return;
  // 配额已满：点击发起的计算在此停止，不发送给服务器（服务器按同一规则再次拦截）
  if (!auto && !readyForQuota(useResults.getState().storage, (m) => say(m))) return;
  const busy = await jobInTheWay();
  if (busy) {
    if (!auto) say(busy);  // 由「显示该节点」触发的轻量计算不输出提示，点击发起的提交则始终输出
    return;
  }
  const packs = auto ? [] : targets.map((id) => ctx.nodes.find((n) => n.id === id)!).filter((n) => ctx.nodeDefs[n.data.typeId]?.params.some((p) => p.widget === "deliver"));
  if (!auto) void readyToSave(packs, ctx.nodeDefs);
  try {
    // the port the viewer shows, when it shows one: the server cooks only what it needs (`show`)
    const shownPort = useLook.getState().displayPort;
    const { job } = await api.cook(toJSON(), useCookInputs.getState().version, target, auto, shownPort ? [shownPort] : []);
    const label = ctx.nodes.find((n) => n.id === target)?.data.label ?? target;
    say(msg(auto ? "I-JOB-SHOW" : "I-JOB-COOK", { node: label, job }));
    await rememberDestinations(job, packs);
    follow(job, target);
  } catch (e) {
    if (auto) logMessage(msg("W-JOB-SHOWNOTCOOKED", { node: ctx.nodes.find((n) => n.id === target)?.data.label ?? target, reason: reasonOf(e) }));
    else say(msg("E-JOB-SUBMITFAILED", { reason: reasonOf(e) }));
  }
}

export function cookNode(id: string): void {
  useLook.getState().setDisplay(id);
  void cook(id);
}

export async function deliverAll(): Promise<void> {
  if (useResults.getState().job) {
    say(msg("B-COOK-BUSY"));
    return;
  }
  const snap = snapshotNow();
  const missing = deliverBlocked(snap.nodes, snap.nodeDefs, canWrite, deliveryNodes(snap.nodes, snap.nodeDefs));
  if (missing) {
    sayBlocked([missing]);
    return;
  }
  const reply = await currentReply();
  if (!reply?.deliver) {
    say(useResults.getState().statusProblem ?? msg("B-COOK-NOSTATUS"));
    return;
  }
  const { targets, kind } = reply.deliver; // the 「输出」 nodes this delivery writes, as the server resolved them
  let ctx = blockContext(reply);
  const after = await afterSending(ctx, reply.deliver);  // 仍在使用者本机上的素材先行上传（提交没有「自动」档）
  if (after === null) return;
  ctx = after;
  const found = blockers(ctx, targets);
  if (found.length) {
    sayBlocked(found);
    return;
  }
  const over = frameLimitProblemNow();
  if (over !== null) {  // 帧数超过服务器上限：在发送前拦截
    say(over);
    return;
  }
  startSummary(standing(ctx, targets)); // this delivery's summary starts
  if (!(await readyForPause(useResults.getState().queueSwitches, kind, (m) => say(m)))) return;
  if (!readyForQuota(useResults.getState().storage, (m) => say(m))) return;
  const busy = await jobInTheWay();
  if (busy) {
    say(busy);
    return;
  }
  const packs = targets.map((id) => ctx.nodes.find((n) => n.id === id)!).filter((n) => ctx.nodeDefs[n.data.typeId]?.params.some((p) => p.widget === "deliver"));
  void readyToSave(packs, ctx.nodeDefs);
  try {
    const { job } = await api.deliver(toJSON(), useCookInputs.getState().version);
    say(msg("I-JOB-DELIVER", { job }));
    await rememberDestinations(job, packs);
    follow(job, targets[0]); // the job is followed at its first 「输出」
  } catch (e) {
    say(msg("E-JOB-SUBMITFAILED", { reason: reasonOf(e) }));
  }
}

/** A click's cook or delivery says, once, what still stands on its nodes (into the log). */
function startSummary(said: Blocker[]): void {
  for (const b of said) say(b.message, b.node);
}

/** A click that cannot be submitted yet: the nodes in its way are marked in the error colour, with the reason for each
 * on its node (a click goes to the parameter it names) and in the log, instead of a popup.
 *
 * 阻止本次提交的原因在三处呈现：节点自身变红并显示原因（`byNode[节点].blocked`，`blockedAt` 使其保持显示
 * 直到计算输入变化）、页面跳转到该节点与对应参数（见下方代码）、以及日志（由 `say` 写入，顶栏图标显示红色计数）。 */
function sayBlocked(found: Blocker[]): void {
  useResults.setState({ blockedAt: useCookInputs.getState().version }); // the marks stand until the cook inputs change
  for (const b of found) if (b.node) useResults.getState().setNodeStatus(b.node, { blocked: b.message.text });
  for (const b of dedupeBlockers(found)) say(b.message, b.node);
  // And take the user there: the graph selects the node that stops it and the parameter panel goes to the setting it
  // names (an 「输出」 with no 保存到). Nothing was sent to the server.
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
async function jobInTheWay(): Promise<ReturnType<typeof msg> | null> {
  const job = useResults.getState().job;
  if (!job) return null;
  const queue = await api.queue(false).catch(() => null);
  if (queue) {  // 能够查询到服务器时才下结论；查询失败时视为任务仍在进行，宁可多给一次提示
    const live = queue.jobs?.find((j) => j.id === job.id && (j.state === "queued" || j.state === "running"));
    if (!live) {
      useResults.getState().setJob(null);  // 页面记录已过期：解除记录，本次照常提交
      return null;
    }
  }
  return msg("N-JOB-INPROGRESS", { job: job.id });
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

/** The page was opened (or reloaded): this browser's jobs still queued or running are followed again. */
export async function resumeJobs(): Promise<void> {
  const queue = await api.queue().catch(() => null);
  const ci = useCookInputs.getState();
  for (const j of queue?.jobs ?? []) {
    if (!j.mine || (j.state !== "queued" && j.state !== "running")) continue;
    const ours = j.title === ci.meta.name && !!j.nodes?.length && j.nodes.every((id) => !!ci.nodes[id]);
    if (ours && !useResults.getState().job) follow(j.id, j.nodes![0]);
    else watchJob(j.id);
  }
}

export type { CookJob };
