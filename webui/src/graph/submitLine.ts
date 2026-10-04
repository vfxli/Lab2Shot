/** 提交（点「计算」「提交」到任务进队列、第一次确认它在服务器上）路上连不上服务器时怎么办：唯一一处。普通编辑器和
 * 聚焦页（editor/AppMode.tsx）走同一条 graph/actions.ts，都用这里。
 *
 * - 每一步都有时限（STEP_MS）：隧道可能把死掉的连接一直挂着，fetch 自己永远不会放弃。
 * - 连不上（没有应答、超时、502 / 503 / 504，服务器说正在重启）算「线路断了」：等 WAITS_S 里的秒数再试，最多
 *   TRIES 次；等待时顶栏「队列」上写「连不上服务器（可能正在重启），8 秒后重连（第 3/5 次）」（useSubmitLine），
 *   「取消」随时停下。到上限停下，日志里说清楚：再点一次「计算」就行，已传上去的素材不重传（上传本身断点续传）。
 *   服务器答了、但答的是别的（4xx 拒绝、500 出错）不重试，照原样说。
 * - 不重复提交：一次点击一个提交键（newSubmitKey，POST /api/jobs 的 `submit`）。服务器按它认（同一账号、同一个键只建一
 *   个任务，lab2shot/server/farm.py queue_job）；重发之前页面也先按它在自己的任务列表里找（findSubmitted），找到了就
 *   是它，不再发。 */

import { create } from "zustand";
import { api } from "../api";
import { ApiError, Unreached } from "../platform/http";
import { textOf, type Message } from "../messages/message";
import { msg, say } from "../state/say";
import { submitAbandoned } from "./apply";

export const TRIES = 5; // 重试几次（不算第一次）
export const WAITS_S = [2, 4, 8, 15, 30]; // 第 n 次重试前等几秒
export const STEP_MS = 30_000; // 一步（状态检查、提交、确认）最多等这么久没有回答就算断了

/** 顶栏上提交那一格现在说的（null：没在等线路）。 */
export const useSubmitLine = create<{ text: string | null }>(() => ({ text: null }));
const show = (m: Message | null) => useSubmitLine.setState({ text: m ? textOf(m) : null });

/** 这个错误是线路的事（等一等再试可能就好），不是服务器拒绝：没有应答（Unreached，platform/http.ts 只给 fetch 自己的
 * 拒绝包这一层）、超时或中断、502 / 503 / 504。`run` 里自己代码的 TypeError 是错误，照原样抛出，不当断线重试。
 * 503 带着服务器自己的拦下（B 级消息：管理员关了计算任务 B-QUEUE-PAUSED、磁盘不够暂停新计算 B-QUEUE-DISKLOW）不是线路：
 * 服务器答了，答的是现在不收任务，照原样说，不重连。 */
export function lineDown(e: unknown): boolean {
  if (e instanceof ApiError) {
    if (e.status === 503 && e.said.level === "B") return false;
    return e.status === 0 || e.status === 502 || e.status === 503 || e.status === 504;
  }
  const name = (e as Error)?.name;
  return e instanceof Unreached || name === "AbortError" || name === "TimeoutError";
}

/** 一步的时限：`STEP_MS` 后自己中断的信号。 */
export const stepSignal = (ms = STEP_MS): AbortSignal => AbortSignal.timeout(ms);

/** 到上限停下了：已经在日志里说过。 */
export class LineGaveUp extends Error {
  constructor() {
    super("line gave up");
    this.name = "LineGaveUp";
  }
}

/** 等 `s` 秒（每秒更新顶栏的倒数）；顶栏「取消」了就提早回来。返回 false：取消了。 */
async function waitOut(s: number, attempt: number): Promise<boolean> {
  for (let left = s; left > 0; left--) {
    show(msg("N-SUBMIT-RETRY", { seconds: left, n: attempt, most: TRIES }));
    for (let i = 0; i < 4; i++) {
      if (submitAbandoned()) return false;
      await new Promise((r) => setTimeout(r, 250));
    }
  }
  show(msg("N-SUBMIT-RECONNECTING", { n: attempt, most: TRIES }));
  return !submitAbandoned();
}

/** 跑一步，线路断了按上面的规矩重试。`run` 拿到这一次的时限信号；`before` 在每次重试之前跑（提交：先按提交键找一找，
 * 找到了就是它，不再发）；`gaveUp` 到上限时日志里说的（默认：这次没提交成）。到上限抛 LineGaveUp（日志里已说）；
 * 取消了抛 LineGaveUp 但不说；服务器的拒绝原样抛出。 */
export async function withLine<T>(run: (signal: AbortSignal) => Promise<T>, o: { before?: () => Promise<T | null>; gaveUp?: Message } = {}): Promise<T> {
  const before = o.before;
  try {
    for (let attempt = 0; ; attempt++) {
      if (attempt > 0) {
        if (!(await waitOut(WAITS_S[Math.min(attempt, WAITS_S.length) - 1], attempt))) throw new LineGaveUp();
        if (before) {
          const found = await before().catch((e) => {
            if (lineDown(e)) return null; // 找不成也是线路的事：照样重发，服务器按提交键认
            throw e;
          });
          if (found !== null) return found;
        }
      }
      try {
        return await run(stepSignal());
      } catch (e) {
        if (!lineDown(e) || submitAbandoned()) throw submitAbandoned() ? new LineGaveUp() : e;
        if (attempt >= TRIES) {
          say(o.gaveUp ?? msg("E-SUBMIT-GAVEUP", { tries: TRIES + 1 }));
          throw new LineGaveUp();
        }
      }
    }
  } finally {
    show(null);
  }
}

/** 一次点击的提交键。 */
export const newSubmitKey = (): string => crypto.randomUUID().replace(/-/g, "");

/** 自己的任务里有没有带着这个提交键的（上一次 POST 其实到了，只是回答丢了）：有就是它的任务号。 */
export async function findSubmitted(key: string): Promise<{ job: string } | null> {
  const q = await api.queue(false);
  const hit = q.history?.find((j) => (j.client as { details?: { submit?: string } } | undefined)?.details?.submit === key);
  return hit ? { job: hit.id } : null;
}

/** 上传那一步的上限（上传自己断点续传、一直等线路，transfer/uploads.ts lineWait）：提交时它连续 TRIES 次都没连上就停下
 * （暂停上传，已传的留着），日志里说；顶栏同样写在等第几次。`tries`、`retryAt`：上传任务现在的。返回 true：该停了。 */
export function uploadGaveUp(state: string, tries: number, retryAt: number): boolean {
  if (state !== "waiting" || !tries) return false;
  if (tries > TRIES) {
    say(msg("E-SUBMIT-GAVEUP", { tries }));
    show(null);
    return true;
  }
  show(msg("N-SUBMIT-RETRY", { seconds: Math.max(0, Math.ceil((retryAt - Date.now()) / 1000)), n: tries, most: TRIES }));
  return false;
}

/** 上传那一步结束（传完、失败或停下）：顶栏不再说等线路。 */
export const uploadLineDone = (): void => show(null);
