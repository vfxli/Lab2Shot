/** 跟踪一个任务：其事件流、各事件对应的状态更新、取消任务，以及刷新后继续跟踪仍在运行的任务。
 *
 * 同一时间只跟踪一个任务，因此「正在跟踪的任务」是一份状态（state/results.ts 的 job），提交前须先检查它
 * （actions.ts jobInTheWay）。事件流具备自动恢复能力（platform/events.ts：断开后重连，停滞也视为断开）。 */

import { api, type CookEvent, type JobProgress } from "../api";
import type { Output } from "../api/files";
import { noteOutput } from "./outputs";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";
import { isLive, waitText } from "./nodes";
import { fromServer, logMessage, msg, say } from "../state/say";
import { textOf } from "../messages/message";
import { onNodeDone } from "./streamDone";
import { askStatus } from "./asking";

/** 跟踪一个任务的事件：每条都写进日志，从不弹出任何东西。事件对某个节点说的，只在提交它的那张图正开着时画到节点上
 * （每条事件都带着它：`graph`，即 meta.id）；打开了别的图之后，任务在队列和日志里照常进行，它的节点不是这张图的。 */
export function follow(id: string, target: string): void {
  following?.stop(); // 同一时间只跟一个任务（state/results.ts job）：之前的记录留下的事件流在这里停掉
  useResults.getState().setJob({ id, target, position: null, stopping: false });
  const end = () => {
    if (following?.id === id) following = null;
    useResults.getState().setJob(null);
    useResults.setState({ running: {} });
    for (const id of useCookInputs.getState().order) {
      if (isLive(useResults.getState().byNode[id]?.status)) useResults.getState().setNodeStatus(id, { status: "idle", note: "" });
    }
    void askStatus();
  };
  const es = api.cookEvents(id); // 能自行恢复的事件流（platform/events.ts）：断线后从最后一条事件接上
  es.onmessage = (data) => {
    // 一条读不通的事件（半截、不是 JSON）只丢这一条，不让整个跟踪断掉
    let raw: unknown;
    try {
      raw = JSON.parse(data);
    } catch {
      return;
    }
    const e = raw as { type: string; node?: string; position?: number; waiting?: { text: string } | null; waiting_detail?: { text: string } | null; name?: string; note?: string; done?: number; total?: number; cached?: boolean; seconds?: number; message?: string; state?: string; reason?: string } & Record<string, unknown>;
    const here = (e.graph ?? "") === useCookInputs.getState().graphId;
    const node = here ? (e.node ?? "") : ""; // ""：只写进日志，不指向这张图的任何节点
    switch (e.type) {
      case "queued": {
        const job = { position: e.position ?? null };
        useResults.getState().patchJob(job);
        if (!here) break;
        // 节点在队列里等：与计算中同一套状态，注记是服务器说它在等什么。
        // 排队中即显示「排队中」：等待显卡、等待内存、前方排队等情况会自行解除，服务器不将其发送给使用者
        // （farm/queue.py _told）；`waiting` 中剩余的都是不会自行解除、需要联系管理员的情况
        // （管理员关闭了计算、该服务器上没有能够计算它的机器），这些情况照常提示。`waiting_detail` 仅供管理员查看，
        // 不显示在节点上：节点底部只有一行，容纳不下两句
        useResults.getState().setNodeStatus(target, { status: "queued", note: [waitText(job), textOf(e.waiting)].filter(Boolean).join(" · ") });
        break;
      }
      case "started":
        useResults.getState().patchJob({ position: null });
        if (here && useResults.getState().byNode[target]?.status === "queued") useResults.getState().setNodeStatus(target, { status: "idle", note: "" });
        say(msg("I-JOB-STARTED", { job: id }));
        break;
      case "stopping":
        useResults.getState().patchJob({ stopping: true });
        break;
      case "node_start":
        if (node) useResults.getState().setNodeStatus(node, { status: "cooking", note: "" });
        break;
      case "progress":
        // 计算进度只有这一种事件（api/progress.ts）：服务器将 stage / progress / phase 合并为同一份描述，
        // 存在它自己的节点下（几个节点同时算时互不覆盖）。节点底部一行显示解算器报告的当前步骤（如「检测人物」）
        useResults.getState().setProgress(e as unknown as JobProgress);
        if (node) useResults.getState().setNodeStatus(node, { note: e.note ?? "" });
        break;
      case "node_done":
        if (!node) break;
        useResults.getState().endProgress(node);
        useResults.getState().setNodeStatus(node, { status: "cooked", note: e.cached ? "" : `用时 ${e.seconds} 秒` });
        onNodeDone(e as CookEvent, node, () => void askStatus());
        break;
      case "message": {
        const m = fromServer(e as Parameters<typeof fromServer>[0]);
        const cur = node ? useResults.getState().results[node] : undefined;
        if (cur && !cur.messages.some((w) => w.code === m.code && w.text === m.text)) // 现在就显示在节点上；之后的状态回复会保留它
          useResults.setState((s) => ({ results: { ...s.results, [node]: { ...cur, messages: [...cur.messages, { ...m }] } } }));
        say(m, node);
        break;
      }
      case "skipped": // 必需的输入来自出错的节点：在该节点上说一次
        if (node) useResults.getState().endProgress(node);
        if (node) useResults.getState().setNodeStatus(node, { status: "skipped", note: "已跳过" });
        say(fromServer(e as Parameters<typeof fromServer>[0]), node);
        break;
      case "output": // 「输出」自己的计算完成：它的 zip 已在，节点上提供下载
        if (e.task && e.pkg && e.node) {
          noteOutput(e as unknown as Output);
          say(msg("I-OUTPUT-READY", { node: String(e.label ?? e.node), name: String(e.name ?? "") }), node);
        }
        break;
      case "error":
        if (node) useResults.getState().endProgress(node);
        if (node) useResults.getState().setNodeStatus(node, { status: "error", note: "出错" });
        say(fromServer(e as Parameters<typeof fromServer>[0]), node);
        break;
      case "cancelled":
        say(e.reason ? msg("N-JOB-CANCELLEDWHY", { reason: e.reason }) : msg("N-JOB-CANCELLED"));
        break;
      case "finished":
        logMessage(msg(finishedCode(e.state), { job: id }));
        useResults.getState().setJustDone({ target, at: Date.now(), state: e.state ?? "done" }); // 按钮上短暂显示「✓ 完成」
        es.close();
        end();
        break;
    }
  };
  es.ongone = (why) => {
    say(why === "forbidden" ? msg("W-JOB-FORBIDDEN") : msg("W-JOB-GONE", { job: id }));
    end();
  };
  // 连接断过又接上：断开期间发生的事件会补发，但服务器重启、任务被别处清掉这类情况没有事件说——按服务器的队列核对一次
  es.onresume = () => void syncJob();
  following = { id, stop: () => (es.close(), end()) };
  es.onlogin = () => {
    // 任务进行中登录过期：登录门在页面上重新要登录；任务本身还在（在页面上重新登录后，仍在排队或计算的接着跟踪，
    // editor/App.tsx）
    logMessage(msg("N-JOB-LOGINOUT", { job: id }));
    end();
  };
}

/** 正在跟踪的任务（与 state/results.ts 的 job 是同一个）：核对后发现服务器上它已结束时，据此关掉事件流、解除记录。 */
let following: { id: string; stop: () => void } | null = null;

/** 页面的任务记录按服务器核对，以服务器为准：服务器上它已不在排队 / 计算中，就把它当作已结束——日志里记它的结果
 * （按服务器的历史记录；查不到的记作不在服务器上了），关掉事件流、解除记录（节点状态、按钮随之恢复）。
 * 返回任务是否仍在进行；没有记录时为 false。查询失败（连不上服务器）时不下结论，当作仍在进行。
 * 调用处：事件流断开后重新接上（follow 的 onresume），以及提交前的检查（graph/actions.ts）。 */
export async function syncJob(): Promise<boolean> {
  const job = useResults.getState().job;
  if (!job) return false;
  const queue = await api.queue(false).catch(() => null);
  if (!queue) return true;
  if (useResults.getState().job?.id !== job.id) return !!useResults.getState().job; // 核对期间记录已换（任务正常结束或新提交了一个）
  if (queue.jobs?.some((j) => j.id === job.id && (j.state === "queued" || j.state === "running"))) return true;
  const done = queue.history?.find((j) => j.id === job.id);
  logMessage(done ? msg(finishedCode(done.state), { job: job.id }) : msg("W-JOB-GONE", { job: job.id }));
  if (following?.id === job.id) following.stop();
  else useResults.getState().setJob(null);
  return false;
}

/** 任务结束时日志里那一条：事件流里的 finished 和核对时查到的历史记录用同一个对照。 */
function finishedCode(state: string | undefined): string {
  return state === "failed" ? "E-JOB-FAILED" : state === "partial" ? "W-JOB-PARTIAL" : state === "cancelled" ? "I-JOB-CANCELLEDLOG" : "I-JOB-DONE";
}
