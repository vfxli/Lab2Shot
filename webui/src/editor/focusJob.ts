/** 地址 `#job=<任务号>[&focus=<节点 id>]`：打开自己的这个任务（和队列窗口「打开」同一条路：GET /api/jobs/{id}，
 * 节点图作为一张新的、没存成文件的节点图打开），并在聚焦模式（editor/AppMode.tsx）下让视图显示、参数面板列出那个节点；
 * 聚焦模式的「计算」与「完成」也在这里。
 *
 * 一个标签页只打开一次：打开过就记在 sessionStorage（`lab2shot.jobOpened.<任务号>`），刷新这一页时恢复的是这一页自己的
 * 工作副本（editor/autosave.ts），改过的参数不丢，不再从任务重新打开。 */

import { useEffect } from "react";
import { api, type GraphJSON } from "../api";
import { cook, deliverAll, loadGraph } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { msg, reasonOf, say } from "../state/say";
import { useViewer } from "../state/viewer";
import { getNodeDefs } from "../state/catalog";
import { jobFromAddress, markFocus, useAppMode } from "./AppMode";
import { MARK_DONE, MARK_FOCUS } from "./dccSignals";

const openedKey = (job: string) => `lab2shot.jobOpened.${job}`;

function openedHere(job: string): boolean {
  try {
    return sessionStorage.getItem(openedKey(job)) === "1";
  } catch {
    return false;
  }
}

function noteOpened(job: string): void {
  try {
    sessionStorage.setItem(openedKey(job), "1");
  } catch {
    /* storage blocked: a reload opens the job again */
  }
}

/** 页面开始时要不要恢复这个账号在本机的工作副本：地址要打开一个这一页还没打开过的任务时不恢复（直接打开任务，不问
 * 「有没有存」：别的工作副本照样留在这个浏览器里）。 */
export const restoresWorkingCopy = (): boolean => {
  const asked = jobFromAddress();
  return !asked || openedHere(asked.job);
};

/** 打开地址里的任务（App.tsx 在目录、登录、工作副本都就绪后调一次）。`restored`：刚恢复的工作副本的节点图（刷新时
 * 它就是这个任务改到一半的那张）。 */
export async function openAddressedJob(restored: GraphJSON | null): Promise<void> {
  const asked = jobFromAddress();
  if (!asked) return;
  if (asked.focus) markFocus(MARK_FOCUS);
  try {
    const job = await api.job(asked.job);
    const defs = getNodeDefs();
    const types = Object.fromEntries((job.graph.nodes ?? []).map((n) => [n.id, n.type]));
    useAppMode.getState().setFocus({ targets: job.nodes, delivers: job.nodes.some((t) => !!defs[types[t] ?? ""]?.delivers) });
    const again = openedHere(asked.job) && !!restored && (!asked.focus || (restored.nodes ?? []).some((n) => n.id === asked.focus));
    if (!again) {
      loadGraph(job.graph, null, false, undefined, { freshId: true }); // 一张新的、没存成文件的节点图（不和原图共用 id）
      noteOpened(asked.job);
      say(msg(job.cache.mark === "all" ? "N-JOB-LOADED" : job.cache.mark === "some" ? "N-JOB-LOADEDSOME" : "N-JOB-LOADEDNONE", { title: job.title }));
    }
    if (asked.focus && !types[asked.focus]) say(msg("E-FOCUS-NONODE", { node: asked.focus, title: job.title }));
  } catch (e) {
    say(msg("E-JOB-LOADFAILED", { reason: reasonOf(e) }));
  }
}

/** 聚焦模式：每打开一份文档（docId 变），视图显示、参数面板选中聚焦的节点。 */
export function useFocusDocument(): void {
  const docId = useViewer((s) => s.docId);
  const node = useAppMode((s) => (s.mode === "focus" ? s.focus?.node ?? "" : ""));
  const present = useCookInputs((s) => !!node && !!s.nodes[node]);
  useEffect(() => {
    if (!node || !present) return;
    useLook.getState().setDisplay(node);
    useViewer.getState().select(node);
  }, [docId, node, present]);
}

/** 聚焦模式的「计算」：再算一次打开的任务当初算的（整理打包「输出」，或那个节点），带 follows——结果才是 DCC 取得回去的。
 * 视图照样显示聚焦的节点。 */
export function focusCompute(): void {
  const f = useAppMode.getState().focus;
  if (!f) return;
  if (f.delivers) void deliverAll();
  else void cook(f.targets[0] ?? f.node);
}

/** 有没有改了还没「计算」的：这一页改过（dirty），并且现在的计算输入不是最近一次提交时的。 */
export function focusUncomputed(): boolean {
  const f = useAppMode.getState().focus;
  return !!f && useViewer.getState().dirty && f.submitted !== useCookInputs.getState().version;
}

/** 「完成」：给插件的关窗信号（markFocus done：页面标题和 <html data-lab2shot>），再试 window.close()（内嵌窗口里插件
 * 也收得到这一下）；关不掉的（系统浏览器里打开的）页面画一层「可以回到 DCC 了」（FocusBar）。 */
export function focusFinish(): void {
  useAppMode.getState().setFocus({ done: true });
  markFocus(MARK_DONE);
  window.close();
}
