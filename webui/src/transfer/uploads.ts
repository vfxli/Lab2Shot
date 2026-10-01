import { clientInfo } from "../platform/client";
import { run } from "../platform/db";
import type { Upload } from "../api/files";
import { ApiError, awaitingLogin, json, LOGGED_IN, STALL_MS, upload } from "../platform/http";
import { useUploads, type UploadState, type UploadTask } from "../state/uploads";
import { forgetLocalOf, forgetPicked, rehomeLocal, rememberLocal } from "./local";
import { useCookInputs } from "../state/cookInputs";
import { useLocalProxies } from "./localProxy";
import { onServerChange, serverNow } from "../state/server";
import { Rate } from "./rate";
import { CODE } from "../messages/format";
import { MessageError, fromServer, msg, type Message } from "../messages/message";
import { item } from "./uploadText";
import { randomId } from "../platform/randomId";
import { backoff } from "../platform/backoff";

export { howFar, uploadBlocker, uploadLine, uploadNote, uploadStopsClick } from "./uploadText";

export { useUploads, type UploadState, type UploadTask };

/** 使用者文件的上传：每个文件分段发送，服务器收到一段存一段（lab2shot/transfer/uploads.py），断线不丢任何东西。页面等
 * 线路恢复（1 秒、2 秒……30 秒，浏览器报告重新在线时立即），问服务器每个文件到了多少，从那个字节接着传。服务器已有的
 * 一概不重传：本浏览器记得哪个文件是什么内容（sha256）、进了哪个分段，一次向服务器查询多个。
 *
 * 一份上传是某个节点某个文件参数上的一个任务。进度只在这里（useUploads），不进节点图：所有文件都到齐后节点图才收到
 * 上传的引用（一次编辑、一步撤销）。任务本身存在浏览器的数据库里，刷新或崩溃都不丢：参数随后说明缺什么，再选同样的
 * 文件即从停下处续传。本浏览器的另一个标签页正在传它时，连同进度一起显示（BroadcastChannel；发送方标签页在
 * localStorage 里的心跳表明它还活着）。 */

// `touched`：最后一次写下这条记录的时刻（ms）：暂停的任务多久没动过，按它清理（purgeStale）
type Kept = Omit<UploadTask, "state" | "sent" | "done" | "rate" | "error" | "retryAt"> & { done?: number; sent?: number; touched?: number };

/** 当前打开的文档（graphId）：上传任务按它归属，另一份文档的同名节点不会拿到这份文档的任务。 */
const openGraph = () => useCookInputs.getState().graphId;

/** 文档 `graph`（默认为当前打开的那份）里某节点某文件参数上正在发送（或等待发送）的任务，没有为 undefined。 */
export const taskFor = (tasks: Record<string, UploadTask>, node: string, param: string, graph: string = openGraph()): UploadTask | undefined =>
  Object.values(tasks).find((t) => t.graphId === graph && t.node === node && t.param === param);

/** 当前打开的文档里某节点任一文件参数上的上传（显示在节点底部）。 */
export const useNodeUpload = (node: string, def: { params: { name: string; widget: string | null }[] } | undefined): UploadTask | undefined => {
  const graph = useCookInputs((s) => s.graphId);
  return useUploads((s) => Object.values(s.tasks).find((t) => t.graphId === graph && t.node === node && !!def?.params.some((p) => p.name === t.param)));
};

/** 「另存为」给文档换了新的 graphId（`graph/graphFile.ts`），文档本身没变：它的上传任务和本机文件跟着换过去；
 * 打开另一份文档（`graph/document.ts loadGraph`）不在此列，那时旧文档的任务留给旧文档。 */
export function rehomeUploads(from: string, to: string): void {
  const moved = Object.values(useUploads.getState().tasks).filter((t) => t.graphId === from);
  if (!moved.length) return;
  useUploads.setState((s) => ({ tasks: { ...s.tasks, ...Object.fromEntries(moved.map((t) => [t.key, { ...t, graphId: to }])) } }));
  for (const t of moved) void keepTask(useUploads.getState().tasks[t.key]);
  rehomeLocal(from, to);
}

let opening: string | null = null; // loadGraph 正在打开的文档（它引起的 graphId 变化不是「另存为」）

/** 打开另一份文档之前（`graph/document.ts loadGraph`）：不属于新文档 `next` 的本机状态不再显示——
 * 本机代理进度（键为 `节点|参数|graphId`，`transfer/declare.ts`）和上传完成后留作显示的本机文件（`transfer/local.ts`）
 * 清掉；上传任务本身不动（可能正在传，也可能停在 picked 等用户回到那份文档），靠 graphId 过滤不再显示在新文档里。 */
export function leaveGraph(next: string): void {
  opening = next === openGraph() ? null : next; // 重新打开同一份文档：它的 id 不变
  forgetLocalOf(next);
  const tail = `|${next}`;
  useLocalProxies.setState((s) => ({ tasks: Object.fromEntries(Object.entries(s.tasks).filter(([k]) => k.endsWith(tail))) }));
  void purgeStale(next);
}

/** graphId 的变化：loadGraph 打开的是另一份文档；否则是「另存为」换了 id，任务跟着文档走（`rehomeUploads`）。 */
export function followGraphId(): () => void {
  return useCookInputs.subscribe((s, prev) => {
    if (s.graphId === prev.graphId) return;
    if (s.graphId === opening) opening = null;
    else rehomeUploads(prev.graphId, s.graphId); // 包括页面刚打开、从未载入过的空白文档（graphId 为 ""）
  });
}

const PARALLEL = 4; // 同时发送的文件数（一段序列由许多小文件组成）
const CHUNK = 8 << 20; // 每个请求的字节数：小帧在开分段的那个请求里就发完
const MAX_WAIT_S = 30;
const BEAT_MS = 2000;
const LIVE = "lab2shot.uploading"; // localStorage：{任务键: 发送方标签页最近一次心跳}
const channel = typeof BroadcastChannel !== "undefined" ? new BroadcastChannel("lab2shot.uploads") : null;

// ------------------------------------------------------------------ 任务：本标签页与数据库所知的

// 以下几项供选择文件步骤（`transfer/declare.ts`）使用：同一层拆成的两个文件之间的内部接口，不对外公开。
/** 改一份任务的几项；这几项都没变（每秒的进度报告在卡住、等待时常是如此）就不写，订阅者不重算、不重绘。 */
export const patchTask = (key: string, p: Partial<UploadTask>) =>
  useUploads.setState((s) => {
    const t = s.tasks[key];
    if (!t || (Object.keys(p) as (keyof UploadTask)[]).every((k) => t[k] === p[k])) return s;
    return { tasks: { ...s.tasks, [key]: { ...t, ...p } } };
  });

const remove = (key: string) => {
  forgetPicked(key); // 任务不在了：它选的文件也不再留着（transfer/local.ts）
  useUploads.setState((s) => {
    const { [key]: _, ...rest } = s.tasks;
    return { tasks: rest };
  });
};

export const keepTask = (t: UploadTask) => {
  const { state: _s, rate: _r, error: _e, retryAt: _t, ...rest } = t;
  const kept: Kept = { ...rest, touched: Date.now() };
  return run("uploads", "readwrite", (s) => s.put(kept, `task:${t.key}`)).catch(() => undefined);
};
const forget = (key: string) => run("uploads", "readwrite", (s) => s.delete(`task:${key}`)).catch(() => undefined);

/** 暂停的上传多久没动就清掉：别的节点图里选了素材、没传完就再没回去，记录不能永远留在浏览器里。门槛就是管理员的
 * 「任务保留天数」（服务状态里的 tasks.keep_days，state/server.ts）：服务器上的任务和缓存也按它清，过了这个天数
 * 那边的未完成部分多半也不在了。还不知道（服务状态没到、旧服务没带这一项）时为 null：不清。 */
const staleMs = (): number | null => {
  const days = serverNow()?.tasks?.keep_days;
  return days ? days * 24 * 3600 * 1000 : null;
};

const readKept = () => run<Kept[]>("uploads", "readonly", (s) => s.getAll(IDBKeyRange.bound("task:", "task:￿"))).catch(() => [] as Kept[]);

/** 该清的记录：暂停着（不是这个标签页正在传、另一个标签页在传或刚选好等「计算」的）、超过「任务保留天数」没动、也不是 `keep`
 * 这份文档（正开着的，它的参数行上正写着「没传完，重选即续传」）。没有 `touched` 的旧记录从现在起算、先补写上。 */
function staleOf(kept: Kept[], keep: string, now = Date.now()): Kept[] {
  const alive = beats();
  const limit = staleMs();
  if (limit === null) return [];
  return kept.filter((k) => {
    if (k.touched === undefined) {
      void run("uploads", "readwrite", (s) => s.put({ ...k, touched: now }, `task:${k.key}`)).catch(() => undefined);
      return false;
    }
    if (k.graphId === keep || now - k.touched < limit || now - (alive[k.key] ?? 0) < BEAT_MS * 3) return false;
    const here = useUploads.getState().tasks[k.key];
    return !here || here.state === "paused";
  });
}

/** 清掉一条暂停的上传和它在本机留下的：任务记录、各文件「服务器上那段未完成的部分叫什么」（part:，续传用）、
 * 发送心跳；页面上的任务一并去掉。服务器上的未完成部分由服务器自己的清理处理；「内容已在服务器上」的指纹对照
 * （按文件存，不属于哪个任务，别的任务也会用）不动。 */
async function dropKept(k: Kept): Promise<void> {
  await forget(k.key);
  for (const f of k.files) await run("uploads", "readwrite", (s) => s.delete(`part:${fileKeyOf(k.origin, f.name, f.size, f.modified)}`)).catch(() => undefined);
  beat(k.key, false);
  remove(k.key);
}

/** 启动（restoreUploads）、每次打开节点图（leaveGraph）、以及「任务保留天数」第一次知道或改了时整理一次：清掉超过这个
 * 天数没动的暂停上传（`keep`：要打开 / 开着的文档）。 */
export async function purgeStale(keep: string): Promise<void> {
  for (const k of staleOf(await readKept(), keep)) await dropKept(k);
}

// 服务状态到了（页面刚打开时它常比 restoreUploads 晚）或管理员改了天数：按新的门槛整理一次
let keepDays: number | undefined;
onServerChange(() => {
  const days = serverNow()?.tasks?.keep_days;
  if (!days || days === keepDays) return;
  keepDays = days;
  void purgeStale(openGraph());
});

function beats(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(LIVE) ?? "{}") as Record<string, number>;
  } catch {
    return {};
  }
}

function beat(key: string, on: boolean): void {
  const b = beats();
  if (on) b[key] = Date.now();
  else delete b[key];
  try {
    localStorage.setItem(LIVE, JSON.stringify(b));
  } catch {
    /* 存不了：别的标签页把任务看作已暂停 */
  }
}

export const sentHere = new Map<string, { files: File[]; stop: () => void }>(); // 本标签页正在发送的任务

/** 浏览器留着的任务（页面重新打开）：已暂停的，或另一个标签页正在发送的。每个只在它自己的文档（它的 `graphId`）里恢复。
 * 没有 `graphId` 的记录无从知道属于哪份文档，丢弃（再选同样的文件即续传：分段在服务器上留着）。 */
export async function restoreUploads(): Promise<void> {
  const kept = await readKept();
  const alive = beats();
  const stale = new Set(staleOf(kept, openGraph()));  // 超过「任务保留天数」没动的暂停上传：清掉，不再恢复（purgeStale）
  const tasks: Record<string, UploadTask> = {};
  for (const k of kept) {
    if (!k.graphId) {
      void forget(k.key);
      continue;
    }
    if (stale.has(k)) {
      void dropKept(k);
      continue;
    }
    const elsewhere = Date.now() - (alive[k.key] ?? 0) < BEAT_MS * 3;
    tasks[k.key] = { ...k, state: elsewhere ? "elsewhere" : "paused", sent: k.sent ?? 0, done: k.done ?? 0, rate: 0, error: "", retryAt: 0 };
  }
  useUploads.setState((s) => ({ tasks: { ...tasks, ...s.tasks } }));
}

// 另一个标签页报来的、它正在发送的任务的状态
channel?.addEventListener("message", (e: MessageEvent) => {
  const m = e.data as { key: string; task?: UploadTask; done?: Upload & { folder: string }; gone?: boolean };
  if (sentHere.has(m.key)) return;
  if (m.task) useUploads.setState((s) => ({ tasks: { ...s.tasks, [m.key]: { ...m.task!, state: m.task!.state === "sending" || m.task!.state === "waiting" || m.task!.state === "finishing" ? "elsewhere" : m.task!.state } } }));
  if (m.done) finished?.(useUploads.getState().tasks[m.key], m.done, true);
  if (m.done || m.gone) remove(m.key);
});

// 发送方标签页走了：它的任务在这里记为暂停（标签页关闭或刷新时会立即报告）
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => sentHere.forEach((_, key) => beat(key, false)));
  window.setInterval(() => {
    const alive = beats();
    for (const t of Object.values(useUploads.getState().tasks))
      if (t.state === "elsewhere" && Date.now() - (alive[t.key] ?? 0) > BEAT_MS * 3) patchTask(t.key, { state: "paused", rate: 0 });
  }, BEAT_MS);
}

/** 上传完成后页面要做的事（graph/apply.ts：写进节点的参数）。`elsewhere`：由另一个标签页发送（它的文档不在时由那个标签页
 * 自己说明；本标签页只写进同一份文档）。 */
let finished: ((t: UploadTask | undefined, up: Upload & { folder: string }, elsewhere?: boolean) => void) | null = null;
export const onUploaded = (f: typeof finished) => void (finished = f);

// ------------------------------------------------------------------ 单个请求

class Refused extends MessageError {} // 服务器拒绝，或文件已不在：不值得自动重试
class Moved extends Error {} // 分段不在预期的位置（其间另一个标签页发了一部分）：查询它的位置
/** 没有可用的回复（线路、服务器忙或正在重启、需要重新登录）：会重试。 */
class Unanswered extends MessageError {
  readonly status: number;
  constructor(said: Message, status: number) {
    super(said);
    this.status = status;
  }
}

/** 上传各步（分段、申报、组装）的失败分法，唯一一处：等一等再来的——没到服务器（0）、等登录（401、协议未同意的 403）、
 * 服务器暂时答不了（408、429、5xx）；其它 4xx 是服务器不收（413 太大、E-UPLOAD-NOTSENT 缺文件……），判失败。 */
const retryable = (status: number, body?: Record<string, unknown> | null): boolean =>
  status === 0 || status === 401 || (status === 403 && !!body?.terms) || status === 408 || status === 429 || status >= 500;

/** 同上，按 fetch 那一路抛出的错误：ApiError 看状态码；没有应答的（断线：fetch 抛 TypeError）等一等再来；别的（页面
 * 自己的错）不来回重试。 */
const retryableError = (e: unknown): boolean => (e instanceof ApiError ? retryable(e.status, e.body) : e instanceof TypeError);

/** 跑一步（申报、组装），按上面的分法：该等的等线路 / 登录（lineWait）再来，不收的原样抛出；任务停了就不再来。 */
export async function persist<T>(key: string, run: () => Promise<T>, stopped: () => boolean, back: UploadState = "sending"): Promise<T> {
  for (let attempt = 0; ; ) {
    try {
      return await run();
    } catch (e) {
      if (stopped() || !retryableError(e)) throw e;
      await lineWait(key, attempt++, (e as Error).message, stopped, undefined, back);
    }
  }
}

/** 服务器拒收一个分段的原因：给了消息（代码与文字）用它的消息，只给了文字按 E-REQUEST-REFUSED 带上文字，
 * 否则是带状态码的本地消息。 */
function refusal(data: Record<string, unknown>, status: number, ours: string): Message {
  if (typeof data.code === "string" && CODE.test(data.code)) return fromServer({ code: data.code, text: String(data.detail ?? "") });
  return data.detail ? msg("E-REQUEST-REFUSED", { status, detail: String(data.detail) }) : msg(ours, { status });
}

interface PartState {
  id: string;
  offset: number;
  size: number;
  sha?: string;
}

/** 经页面的上传通道带着 `body` 发一个请求（platform/http.ts upload：与其它所有请求同一个登录闸门与 401 处理，协议未同意的
 * 403 同样请出协议页）；`progress` 接收已发送的字节数。卡住即丢弃（可重试）。 */
async function xhr(method: string, url: string, body: Blob | string | null, progress?: (n: number) => void, started?: (x: XMLHttpRequest) => void, signal?: AbortSignal): Promise<PartState> {
  const { status, text } = await upload(method, url, body, { progress, started, stallMs: STALL_MS, signal });
  let data: Record<string, unknown> = {};
  try {
    data = JSON.parse(text || "{}");
  } catch {
    /* 不是 JSON：隧道自己的错误页 */
  }
  if (status >= 200 && status < 300) return data as unknown as PartState;
  if (status === 0) throw new Unanswered(msg("E-UPLOAD-DISCONNECTED"), 0);
  if (status === 409) throw new Moved(String(data.detail ?? ""));
  // 按 retryable 等一等再续传（404：服务器丢了这个分段，调用方从头来）；其它 4xx 判失败
  if (retryable(status, data) || status === 404)
    throw new Unanswered(refusal(data, status, "E-UPLOAD-NOANSWER"), status);
  throw new Refused(refusal(data, status, "E-UPLOAD-REFUSED"));
}

// ------------------------------------------------------------------ 一个任务

export { onDeclared, sendUpload, startUpload } from "./declare";

/** 重新发送一个失败的任务（它的文件在本标签页里仍可读）。 */
export function retryUpload(key: string): void {
  const t = useUploads.getState().tasks[key];
  const here = sentHere.get(key);
  if (!t || !here) return;
  here.stop(); // 失败那一轮还在跑的先停下：同一时间只有一轮工作者
  sentHere.delete(key);
  patchTask(key, { state: "sending", error: "" });
  void sendWith(useUploads.getState().tasks[key], here.files);
}

/** 停下这个标签页正在传的一份（顶栏「取消」这次提交：graph/actions.ts cancelSubmit）：任务留着、回到「选好了，点计算时
 * 上传」（picked），文件仍在本页手上，已传的部分服务器留着（part: 记录照旧），下次「计算」接着传；归属不变（graphId）。 */
export function pauseUpload(key: string): void {
  const here = sentHere.get(key);
  if (!here) return;
  here.stop();
  sentHere.set(key, { files: here.files, stop: () => undefined });
  beat(key, false);
  patchTask(key, { state: "picked", rate: 0, error: "" });
  const t = useUploads.getState().tasks[key];
  if (t) void keepTask(t);
}

/** 停止发送（参数被清空，或选了别的文件）。服务器已有的分段留着（再选这些文件即续传），过一阵由服务器丢弃。 */
export function cancelUpload(key: string, say = true): void {
  sentHere.get(key)?.stop();
  sentHere.delete(key);
  beat(key, false);
  remove(key);
  void forget(key);
  if (say) channel?.postMessage({ key, gone: true });
}

/** 页面还能不能读所选的文件（其间它可能在使用者的磁盘上被移走或删除）。 */
async function readable(f: File): Promise<boolean> {
  try {
    await f.slice(0, 1).arrayBuffer();
    return true;
  } catch {
    return false;
  }
}

// 一个文件在本机数据库里的键（续传的 part:、内容指纹）：来源 + 名字、大小、修改时间
const fileKeyOf = (origin: string, name: string, size: number, modified: number) => `${origin}|${name}|${size}|${modified}`;
const fileKey = (origin: string, f: File) => fileKeyOf(origin, f.name, f.size, f.lastModified);

/** 一份素材正在这个标签页里传：两条上传路径（整份 sendWith、按通道 transfer/planes.ts sendPlanes）共用的簿记——
 * 心跳（别的标签页据此看出它还活着）、每秒量一次速度与进度（没有进度事件时也量：卡住时速度随时间窗口降下来）、
 * 每秒把进度广播给别的标签页、每 5 秒落盘一次（刷新后知道传到哪了）。`measure` 给出此刻的进度；`end` 在结束时调用。 */
export function sending(key: string, measure: () => { sent: number; done: number; bytes?: number }) {
  const speed = new Rate();
  let told = 0;
  let saved = Date.now();
  const report = () => {
    const m = measure();
    const now = Date.now();
    patchTask(key, { ...m, rate: speed.add(m.sent, now) });
    if (now - told > 1000) {
      told = now;
      const t = useUploads.getState().tasks[key];
      if (t) channel?.postMessage({ key, task: t });
      if (t && now - saved > 5000) {
        saved = now;
        void keepTask(t);
      }
    }
  };
  beat(key, true);
  const heart = window.setInterval(() => beat(key, true), BEAT_MS);
  const ticker = window.setInterval(report, 1000); // 上传进度事件只记数，任务的进度与速度由它每秒报一次
  return {
    report,
    /** 断线后重新量速度（之前的记录不再算进来）。 */
    reset: () => speed.reset(),
    /** 结束：停心跳与计时；失败的留着心跳记录（keepTask 落盘），其余撤掉心跳。 */
    end: () => {
      window.clearInterval(heart);
      window.clearInterval(ticker);
      const t = useUploads.getState().tasks[key];
      if (t?.state === "failed") void keepTask(t);
      else beat(key, false);
    },
  };
}

export async function sendWith(task: UploadTask, files: File[]): Promise<void> {
  const key = task.key;
  let stopped = false; // 已取消，或有一个文件失败：所有工作者都停下
  let cancelled = false; // cancelUpload：什么都不报告
  const inflight = new Set<XMLHttpRequest>();
  const wakes = new Set<() => void>(); // 等线路恢复的工作者（lineWait）
  const quit = new AbortController(); // also reaches parts still waiting out a 429 pause (platform/http.ts upload)
  const halt = () => ((stopped = true), quit.abort(), inflight.forEach((x) => x.abort()), wakes.forEach((w) => w()));
  sentHere.set(key, { files, stop: () => ((cancelled = true), halt()) });
  const confirmed = new Map<string, number>(); // 文件名 -> 服务器已有的字节数
  const flying = new Map<string, number>(); // 文件名 -> 在途请求的字节数
  const shas: Record<string, string> = {};
  // 进度：服务器已有的字节加在途的字节（簿记与 sendPlanes 共用：`sending`）
  const book = sending(key, () => ({
    sent: Math.min(task.bytes, [...confirmed.values(), ...flying.values()].reduce((a, b) => a + b, 0)),
    done: Object.keys(shas).length,
  }));
  const report = book.report;

  // 线路断了：等待（lineWait），恢复发送后速度重新量。`tries`：这个工作者连续失败的次数（各算各的：四个工作者看到的
  // 同一次断线是各自一次失败，不是连续四次）
  const waitForLine = async (why: string, tries: number) => {
    book.reset();
    report();
    await lineWait(key, tries, why, () => stopped, wakes);
  };

  const request = (method: string, url: string, body: Blob | null, name = "") => {
    let x: XMLHttpRequest | null = null;
    // 进度事件只记在途字节，由每秒一次的 ticker 统一报（sending）：不每个事件都整表改一次任务
    const p = xhr(method, url, body, name ? (n) => void flying.set(name, n) : undefined, (r) => inflight.add((x = r)), quit.signal);
    return p.finally(() => (flying.delete(name), x && inflight.delete(x)));
  };

  // 一个文件：用它最前面的字节开分段，之后不断追加直到完整
  const sendFile = async (f: File) => {
    const fk = fileKey(task.origin, f);
    let part = await run<string | undefined>("uploads", "readonly", (s) => s.get(`part:${fk}`)).catch(() => undefined);
    let at: PartState | null = null;
    let tries = 0;
    for (;;) {
      if (stopped) return;
      let opening = false; // 这一轮是在开新分段（POST）：它的 404 不是「服务器丢了分段」
      try {
        if (part && !at) at = await request("GET", `/api/uploads/parts/${part}`, null);
        if (!part) {
          opening = true;
          // 在发送第一个字节之前存下分段名：中途刷新仍能找到服务器已有的部分
          const id = randomId();
          await run("uploads", "readwrite", (s) => s.put(id, `part:${fk}`)).catch(() => undefined);
          part = id;
          at = await request("POST", `/api/uploads/parts?size=${f.size}&id=${id}`, f.slice(0, CHUNK), f.name);
        } else if (at && !at.sha) {
          at = await request("PATCH", `/api/uploads/parts/${part}?offset=${at.offset}`, f.slice(at.offset, at.offset + CHUNK), f.name);
        }
        confirmed.set(f.name, at!.offset);
        tries = 0;
        if (at!.sha) {
          shas[f.name] = at!.sha;
          await run("uploads", "readwrite", (s) => s.put(at!.sha, fk)).catch(() => undefined);
          await run("uploads", "readwrite", (s) => s.delete(`part:${fk}`)).catch(() => undefined);
          report();
          return;
        }
        report();
      } catch (e) {
        if (stopped) return;
        if (e instanceof Refused) throw e;
        const status = (e as { status?: number }).status;
        if (status === 404 && part && !opening) {
          part = undefined; // 服务器丢弃了这个分段（清理）：从头来（开分段时的 404 在下面等待，与 sendBytes 相同）
          at = null;
          await run("uploads", "readwrite", (s) => s.delete(`part:${fk}`)).catch(() => undefined);
          continue;
        }
        at = null; // 查询分段到了哪里（中断的请求已到达的部分服务器会留着）
        if (e instanceof Moved) continue;
        // 完全没有回复：是线路，或是文件本身（选择之后在使用者的磁盘上被移走或删除）
        if (!(await readable(f))) throw new Refused("E-UPLOAD-FILEGONE", { name: f.name });
        await waitForLine((e as Error).message, tries++);
      }
    }
  };

  try {
    // 服务器已有的（按本浏览器以前发过的内容）：一次查询全部
    const known = await Promise.all(files.map((f) => run<string | undefined>("uploads", "readonly", (s) => s.get(fileKey(task.origin, f))).catch(() => undefined)));
    const asked = known.filter((x): x is string => !!x);
    let have = new Set<string>();
    for (let tries = 0; ; ) {
      try {
        have = new Set(asked.length ? (await json<{ have: string[] }>("POST", "/api/uploads/have", { shas: asked })).have : []);
        break;
      } catch (e) {
        if (stopped) return;
        await waitForLine((e as Error).message, tries++);
      }
    }
    const queue: File[] = [];
    files.forEach((f, i) => {
      if (known[i] && have.has(known[i]!)) {
        shas[f.name] = known[i]!;
        confirmed.set(f.name, f.size);
      } else queue.push(f);
    });
    report();
    const worker = async () => {
      try {
        for (let f = queue.shift(); f && !stopped; f = queue.shift()) await sendFile(f);
      } catch (e) {
        halt(); // 一个文件失败即整个任务失败：其余工作者随之停下（「重试」开始新的一轮）
        throw e;
      }
    };
    await Promise.all(Array.from({ length: Math.min(PARALLEL, queue.length) }, worker));
    if (stopped) return;
    await completeSet(task, files, shas, () => stopped);
  } catch (e) {
    if (!cancelled) patchTask(key, { state: "failed", error: (e as Error).message, rate: 0 });
    channel?.postMessage({ key, task: useUploads.getState().tasks[key] });
  } finally {
    book.end();
  }
}

/** 连接中断（或服务器未响应）：等待后重试，间隔逐次加长直至 30 秒；浏览器报告恢复在线或重新登录后立即重试；
 * 登录已结束时只等重新登录（不按时间重试），
 * 任务被停下时（`wakes` 中的每一个都被唤醒）立即结束。每个等待者各自监听，同时等待的几个上传工作者一起被唤醒。
 * 整份上传（`sendWith`）、通道级上传（`sendBytes`）与最后一步（`completeSet`）共用这一种等待；结束时任务已停下则
 * 不再改回「上传中」。 */
export async function lineWait(key: string, attempt: number, why: string, stopped: () => boolean, wakes?: Set<() => void>, back: UploadState = "sending"): Promise<void> {
  // 登录已结束：不按时间重试（每次都会是一个 401，服务器按 401 计数封 IP），只等重新登录（或任务停下）
  const forLogin = awaitingLogin();
  const ms = backoff(attempt + 1, 1000, MAX_WAIT_S * 1000);
  patchTask(key, { state: "waiting", error: why, retryAt: forLogin ? 0 : Date.now() + ms, rate: 0 });
  await new Promise<void>((r) => {
    const t = forLogin ? undefined : window.setTimeout(done, ms);
    function done() {
      window.clearTimeout(t);
      window.removeEventListener("online", done);
      window.removeEventListener(LOGGED_IN, done);
      wakes?.delete(done);
      r();
    }
    window.addEventListener("online", done);
    window.addEventListener(LOGGED_IN, done);
    wakes?.add(done);
  });
  if (!stopped()) patchTask(key, { state: back, error: "" });
}

/** 字节全部到达后的最后一步，整份上传与通道级上传共用：组装该输入（`POST /api/uploads`，与申报得到的
 * 引用相同），将引用写入参数，在本标签页记录已持有这些文件，并结束任务。`shas`：各文件原文件的内容
 * 指纹；通道级上传的数据同样按原文件记录（服务器清单中记录的即为此值，`lab2shot/transfer/uploads.py make_set`）。 */
export async function completeSet(task: UploadTask, files: File[], shas: Record<string, string>, stopped: () => boolean = () => false): Promise<void> {
  const key = task.key;
  patchTask(key, { state: "finishing", rate: 0 });
  let up: Upload;
  try {
    up = await persist(key, () => json<Upload>("POST", "/api/uploads", { name: item(task), files: shas, origin: { path: task.origin, client: clientInfo() } }), stopped);
  } catch (e) {
    if (stopped()) return;
    throw e instanceof ApiError ? new Refused(e.said) : e; // 服务器不收（缺文件 E-UPLOAD-NOTSENT、太大……）：判失败
  }
  for (const f of files) rememberLocal(key, undefined, shas[f.name], f);
  // 组装回来之前使用者已取消（顶栏「取消」这次提交 / 清掉了参数）：不写参数、不结束任务。服务器已收下的字节和
  // 本页记下的内容指纹都留着，下次「计算」时直接再组装一次（同样的文件得到同样的引用）
  if (stopped()) return;
  const done = { ...up, folder: task.folder };
  channel?.postMessage({ key, done });
  finished?.(useUploads.getState().tasks[key], done);
  beat(key, false);
  remove(key);
  void forget(key);
  sentHere.delete(key);
}

/** 按同一套分段协议上传一段字节（并非用户文件，而是通道级上传中一帧 gzip 后的平面，`transfer/planes.ts`），
 * 返回服务器计算的 sha256。断线时同样等待后续传（`lineWait`）；`progress` 接收已上传的字节数。
 * 服务器拒绝（Refused）时原样抛出；`stopped()` 为真时放弃并返回 ""。 */
export async function sendBytes(key: string, blob: Blob, progress: (sent: number) => void, stopped: () => boolean, inflight?: Set<XMLHttpRequest>): Promise<string> {
  // 在途的请求登记进 `inflight`：调用方停下（取消）时能立刻中止它，不再把余下的分段发完
  const send = (method: string, url: string, body: Blob | null, p?: (n: number) => void) => {
    let x: XMLHttpRequest | null = null;
    return xhr(method, url, body, p, (r) => inflight?.add((x = r))).finally(() => x && inflight?.delete(x));
  };
  const fresh = () => randomId();
  let id = fresh();
  let at: PartState | null = null;
  for (let attempt = 0; ; ) {
    if (stopped()) return "";
    try {
      if (!at) at = await send("POST", `/api/uploads/parts?size=${blob.size}&id=${id}`, blob.slice(0, CHUNK), progress);
      else if (!at.sha) {
        const from = at.offset;
        at = await send("PATCH", `/api/uploads/parts/${id}?offset=${from}`, blob.slice(from, from + CHUNK), (n) => progress(from + n));
      }
      attempt = 0;
      if (at.sha) {
        progress(blob.size);
        return at.sha;
      }
    } catch (e) {
      if (stopped()) return "";
      if (e instanceof Refused) throw e;
      if (e instanceof Moved) {
        at = await send("GET", `/api/uploads/parts/${id}`, null).catch(() => null); // 查询已上传的位置，从该处续传
        continue;
      }
      if ((e as { status?: number }).status === 404 && at) {
        at = null; // 服务器已丢弃该数据（已清理）：换用新名称从头上传
        id = fresh();
      }
      await lineWait(key, attempt++, (e as Error).message, stopped);
    }
  }
}
