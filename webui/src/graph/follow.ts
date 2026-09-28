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
import { refreshStatus } from "./actions";

/** Follow a job's events: every one is said into the log, never opening anything. What an event says of a node is
 * drawn on the node only while the graph the job was submitted from is the one open (every event names it: `graph`,
 * its meta.id); after another graph was opened the job goes on in the queue and the log, and its nodes are not this
 * graph's. */
export function follow(id: string, target: string): void {
  useResults.getState().setJob({ id, target, position: null, stopping: false });
  const end = () => {
    useResults.getState().setJob(null);
    useResults.setState({ running: {} });
    for (const id of useCookInputs.getState().order) {
      if (isLive(useResults.getState().byNode[id]?.status)) useResults.getState().setNodeStatus(id, { status: "idle", note: "" });
    }
    void refreshStatus();
  };
  const es = api.cookEvents(id); // a stream that mends itself (platform/events.ts): a dropped line reconnects from the last event
  es.onmessage = (data) => {
    const e = JSON.parse(data) as { type: string; node?: string; position?: number; waiting?: { text: string } | null; waiting_detail?: { text: string } | null; name?: string; note?: string; done?: number; total?: number; cached?: boolean; seconds?: number; message?: string; state?: string; reason?: string } & Record<string, unknown>;
    const here = (e.graph ?? "") === useCookInputs.getState().graphId;
    const node = here ? (e.node ?? "") : ""; // "": said in the log only, no node of this graph named
    switch (e.type) {
      case "queued": {
        const job = { position: e.position ?? null };
        useResults.getState().patchJob(job);
        if (!here) break;
        // The node waits in the queue: the same status model as cooking, its note what the server says it waits for.
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
        onNodeDone(e as CookEvent, node, () => void refreshStatus());
        break;
      case "message": {
        const m = fromServer(e as Parameters<typeof fromServer>[0]);
        const cur = node ? useResults.getState().results[node] : undefined;
        if (cur && !(cur.messages ?? []).some((w) => w.code === m.code && w.text === m.text)) // on the node now; a later status keeps it
          useResults.setState((s) => ({ results: { ...s.results, [node]: { ...cur, messages: [...(cur.messages ?? []), { ...m }] } } }));
        say(m, node);
        break;
      }
      case "skipped": // a required input comes from a node that failed: said once, at the node
        if (node) useResults.getState().endProgress(node);
        if (node) useResults.getState().setNodeStatus(node, { status: "skipped", note: "已跳过" });
        say(fromServer(e as Parameters<typeof fromServer>[0]), node);
        break;
      case "output": // 「输出」's own cook is through: its zip is there, its node offers the download
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
        logMessage(msg(e.state === "failed" ? "E-JOB-FAILED" : e.state === "partial" ? "W-JOB-PARTIAL" : e.state === "cancelled" ? "I-JOB-CANCELLEDLOG" : "I-JOB-DONE", { job: id }));
        es.close();
        end();
        break;
    }
  };
  es.ongone = () => {
    say(msg("W-JOB-GONE", { job: id }));
    end();
  };
  es.onlogin = () => {
    // the login ran out mid-job: the gate asks for it again over the page; the job itself is not gone (a login made
    // over the page resumes what is still queued or running, editor/App.tsx)
    logMessage(msg("N-JOB-LOGINOUT", { job: id }));
    end();
  };
}
