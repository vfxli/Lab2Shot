import { clientInfo } from "../platform/client";
import { run } from "../platform/db";
import type { Upload } from "../api/files";
import { ApiError, json, LOGGED_IN, refusedLogin } from "../platform/http";
import { useUploads, type UploadState, type UploadTask } from "../state/uploads";
import { rememberLocal } from "./local";
import { CODE } from "../messages/format";
import { MessageError, fromServer, msg, type Message } from "../messages/message";
import { item } from "./uploadText";
import { randomId } from "../platform/randomId";
import { backoff } from "../platform/backoff";

export { eta, howFar, uploadBlocker, uploadLine, uploadNote, uploadStopsClick } from "./uploadText";

export { useUploads, type UploadState, type UploadTask };

/** Uploads of the user's files: each file is sent in parts that the server keeps as they arrive
 * (lab2shot/transfer/uploads.py), so a dropped connection loses nothing. The page waits for the connection (1 s, 2 s …
 * 30 s, or immediately when the browser reports being online again), asks the server how much of each file arrived,
 * and resumes from that byte. Nothing the server already has is sent again: this browser remembers which file had
 * which content (sha256) and which part it went into, and queries the server for many at once.
 *
 * An upload is a task for one file parameter of one node. Its progress lives here (useUploads), never in the graph: the
 * graph receives the upload's reference once every file has arrived (one edit, one undo step). The task itself is kept
 * in the browser's database, so a reload or a crash does not lose it: the parameter then reports what is missing, and
 * picking the same files again resumes where it stopped. When another tab of this browser is sending it, that is
 * reported with its progress (a BroadcastChannel; the sending tab's heartbeat in localStorage shows it is still alive). */

type Kept = Omit<UploadTask, "state" | "sent" | "done" | "rate" | "error" | "retryAt"> & { done?: number; sent?: number };

/** The task sending (or waiting to send) a node's file parameter, if any. */
export const taskFor = (tasks: Record<string, UploadTask>, node: string, param: string): UploadTask | undefined =>
  Object.values(tasks).find((t) => t.node === node && t.param === param);

/** The upload of any of a node's file parameters (shown in the node's footer). */
export const useNodeUpload = (node: string, def: { params: { name: string; widget: string | null }[] } | undefined): UploadTask | undefined =>
  useUploads((s) => Object.values(s.tasks).find((t) => t.node === node && !!def?.params.some((p) => p.name === t.param)));

const PARALLEL = 4; // files sent concurrently (a sequence consists of many small files)
const CHUNK = 8 << 20; // bytes per request: a small frame is sent in the request that opens its part
const STALL_MS = 30_000; // no progress for this long: the request is dropped and resumed (a tunnel may keep a dead connection open)
const MAX_WAIT_S = 30;
const BEAT_MS = 2000;
const LIVE = "lab2shot.uploading"; // localStorage: {task key: the sending tab's last heartbeat}
const channel = typeof BroadcastChannel !== "undefined" ? new BroadcastChannel("lab2shot.uploads") : null;

// ------------------------------------------------------------------ the tasks, as this tab and the database know them

/** 供选择文件步骤（`transfer/declare.ts`）使用的内部接口：同一层拆分出的两部分，不对外公开。 */
export const patchTask = (key: string, p: Partial<UploadTask>) =>
  useUploads.setState((s) => (s.tasks[key] ? { tasks: { ...s.tasks, [key]: { ...s.tasks[key], ...p } } } : {}));

const remove = (key: string) =>
  useUploads.setState((s) => {
    const { [key]: _, ...rest } = s.tasks;
    return { tasks: rest };
  });

export const keepTask = (t: UploadTask) => {
  const { state: _s, rate: _r, error: _e, retryAt: _t, ...kept } = t;
  return run("uploads", "readwrite", (s) => s.put(kept, `task:${t.key}`)).catch(() => undefined);
};
const forget = (key: string) => run("uploads", "readwrite", (s) => s.delete(`task:${key}`)).catch(() => undefined);

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
    /* no storage: other tabs see the task as paused */
  }
}

export const sentHere = new Map<string, { files: File[]; stop: () => void }>(); // the tasks this tab is sending

/** The tasks the browser kept (the page was opened again): paused, or being sent by another tab. */
export async function restoreUploads(): Promise<void> {
  const kept = await run<Kept[]>("uploads", "readonly", (s) => s.getAll(IDBKeyRange.bound("task:", "task:￿"))).catch(() => [] as Kept[]);
  const alive = beats();
  const tasks: Record<string, UploadTask> = {};
  for (const k of kept) {
    const elsewhere = Date.now() - (alive[k.key] ?? 0) < BEAT_MS * 3;
    tasks[k.key] = { ...k, state: elsewhere ? "elsewhere" : "paused", sent: k.sent ?? 0, done: k.done ?? 0, rate: 0, error: "", retryAt: 0 };
  }
  useUploads.setState((s) => ({ tasks: { ...tasks, ...s.tasks } }));
}

// status reported by another tab about the tasks it is sending
channel?.addEventListener("message", (e: MessageEvent) => {
  const m = e.data as { key: string; task?: UploadTask; done?: Upload & { folder: string }; gone?: boolean };
  if (sentHere.has(m.key)) return;
  if (m.task) useUploads.setState((s) => ({ tasks: { ...s.tasks, [m.key]: { ...m.task!, state: m.task!.state === "sending" || m.task!.state === "waiting" || m.task!.state === "finishing" ? "elsewhere" : m.task!.state } } }));
  if (m.done) finished?.(useUploads.getState().tasks[m.key], m.done);
  if (m.done || m.gone) remove(m.key);
});

// a sending tab that went away: its tasks are paused here (a tab that closes or reloads reports this immediately)
if (typeof window !== "undefined") {
  window.addEventListener("pagehide", () => sentHere.forEach((_, key) => beat(key, false)));
  window.setInterval(() => {
    const alive = beats();
    for (const t of Object.values(useUploads.getState().tasks))
      if (t.state === "elsewhere" && Date.now() - (alive[t.key] ?? 0) > BEAT_MS * 3) patchTask(t.key, { state: "paused", rate: 0 });
  }, BEAT_MS);
}

/** The page's action for a finished upload (graph/apply.ts: writes it into the node's parameter). */
let finished: ((t: UploadTask | undefined, up: Upload & { folder: string }) => void) | null = null;
export const onUploaded = (f: typeof finished) => void (finished = f);

// ------------------------------------------------------------------ one request

class Refused extends MessageError {} // the server refused, or the file is gone: not worth retrying automatically
class Moved extends Error {} // the part is not where expected (another tab sent some of it meanwhile): query its position
/** No usable response (the connection, a busy or restarting server, a required re-login): retried. */
class Unanswered extends MessageError {
  readonly status: number;
  constructor(said: Message, status: number) {
    super(said);
    this.status = status;
  }
}

/** The server's reason for a refused part: its message (code and text) when provided, its text as
 * E-REQUEST-REFUSED when only text was provided, otherwise a local message with the status. */
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

/** One request carrying `body`; `progress` receives the number of bytes sent so far. A stall drops it (a retryable error). */
function xhr(method: string, url: string, body: Blob | string | null, progress?: (n: number) => void, started?: (x: XMLHttpRequest) => void): Promise<PartState> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    started?.(x);
    let last = Date.now();
    const watch = window.setInterval(() => Date.now() - last > STALL_MS && x.abort(), 5000);
    const end = () => window.clearInterval(watch);
    x.open(method, url);
    if (typeof body === "string") x.setRequestHeader("Content-Type", "application/json");
    else if (body) x.setRequestHeader("Content-Type", "application/octet-stream");
    x.upload.onprogress = (e) => ((last = Date.now()), progress?.(e.loaded));
    x.onprogress = () => (last = Date.now());
    x.onload = () => {
      end();
      let data: Record<string, unknown> = {};
      try {
        data = JSON.parse(x.responseText || "{}");
      } catch {
        /* not JSON: a tunnel's own error page */
      }
      if (x.status >= 200 && x.status < 300) return resolve(data as unknown as PartState);
      if (x.status === 409) return reject(new Moved(String(data.detail ?? "")));
      if (x.status === 401) refusedLogin(data); // asks the user to log in, then continues
      if (x.status === 401 || x.status === 404 || x.status === 408 || x.status === 429 || x.status >= 500 || x.status === 0)
        return reject(new Unanswered(refusal(data, x.status, "E-UPLOAD-NOANSWER"), x.status));
      reject(new Refused(refusal(data, x.status, "E-UPLOAD-REFUSED")));
    };
    x.onerror = x.onabort = x.ontimeout = () => (end(), reject(new Unanswered(msg("E-UPLOAD-DISCONNECTED"), 0)));
    x.send(body);
  });
}

// ------------------------------------------------------------------ a task

export { onDeclared, sendUpload, startUpload } from "./declare";

/** Sends a failed task again (its files are still available in this tab). */
export function retryUpload(key: string): void {
  const t = useUploads.getState().tasks[key];
  const here = sentHere.get(key);
  if (!t || !here) return;
  here.stop(); // whatever of the failed round is still going stops first: one round of workers at a time
  sentHere.delete(key);
  patchTask(key, { state: "sending", error: "" });
  void sendWith(useUploads.getState().tasks[key], here.files);
}

/** Stops sending (the parameter was cleared, or other files were picked). The parts the server has are kept (picking
 * the files again resumes); the server discards them after a while. */
export function cancelUpload(key: string, say = true): void {
  sentHere.get(key)?.stop();
  sentHere.delete(key);
  beat(key, false);
  remove(key);
  void forget(key);
  if (say) channel?.postMessage({ key, gone: true });
}

/** Whether the page can still read the picked file (it may have been moved or deleted on the user's disk meanwhile). */
async function readable(f: File): Promise<boolean> {
  try {
    await f.slice(0, 1).arrayBuffer();
    return true;
  } catch {
    return false;
  }
}

const fileKey = (origin: string, f: File) => `${origin}|${f.name}|${f.size}|${f.lastModified}`;

export async function sendWith(task: UploadTask, files: File[]): Promise<void> {
  const key = task.key;
  let stopped = false; // cancelled, or one file failed: every worker stops
  let cancelled = false; // cancelUpload: nothing is reported
  const inflight = new Set<XMLHttpRequest>();
  const wakes = new Set<() => void>(); // the workers waiting for the line (lineWait)
  const halt = () => ((stopped = true), inflight.forEach((x) => x.abort()), wakes.forEach((w) => w()));
  sentHere.set(key, { files, stop: () => ((cancelled = true), halt()) });
  const confirmed = new Map<string, number>(); // file name -> bytes the server has
  const flying = new Map<string, number>(); // file name -> bytes of the request in transit
  const shas: Record<string, string> = {};
  let attempt = 0;
  const heart = window.setInterval(() => beat(key, true), BEAT_MS);
  beat(key, true);

  // progress: bytes the server has plus bytes in transit; the rate over the last few seconds
  const samples: [number, number][] = [];
  let told = 0;
  let saved = Date.now();
  const report = () => {
    const sent = Math.min(task.bytes, [...confirmed.values(), ...flying.values()].reduce((a, b) => a + b, 0));
    const now = Date.now();
    samples.push([now, sent]);
    while (samples.length > 2 && now - samples[0][0] > 8000) samples.shift();
    const [t0, s0] = samples[0];
    const rate = now - t0 > 1500 ? ((sent - s0) / (now - t0)) * 1000 : useUploads.getState().tasks[key]?.rate ?? 0;
    patchTask(key, { sent, done: Object.keys(shas).length, rate: Math.max(0, rate) });
    if (now - told > 1000) {
      told = now;
      const t = useUploads.getState().tasks[key];
      if (t) channel?.postMessage({ key, task: t });
      if (t && now - saved > 5000) {
        saved = now;
        void keepTask(t); // persisted so that a reload knows how far it got
      }
    }
  };

  // the connection dropped: wait (lineWait), with the rate measured afresh once sending resumes
  const waitForLine = async (why: string) => {
    samples.length = 0;
    report();
    await lineWait(key, attempt++, why, () => stopped, wakes);
  };

  const request = (method: string, url: string, body: Blob | null, name = "") => {
    let x: XMLHttpRequest | null = null;
    const p = xhr(method, url, body, name ? (n) => (flying.set(name, n), report()) : undefined, (r) => inflight.add((x = r)));
    return p.finally(() => (flying.delete(name), x && inflight.delete(x)));
  };

  // one file: its part is opened with its first bytes, then appended to until complete
  const sendFile = async (f: File) => {
    const fk = fileKey(task.origin, f);
    let part = await run<string | undefined>("uploads", "readonly", (s) => s.get(`part:${fk}`)).catch(() => undefined);
    let at: PartState | null = null;
    for (;;) {
      if (stopped) return;
      try {
        if (part && !at) at = await request("GET", `/api/uploads/parts/${part}`, null);
        if (!part) {
          // its name is saved before the first byte is sent: a reload midway still finds what the server has
          const id = randomId();
          await run("uploads", "readwrite", (s) => s.put(id, `part:${fk}`)).catch(() => undefined);
          part = id;
          at = await request("POST", `/api/uploads/parts?size=${f.size}&id=${id}`, f.slice(0, CHUNK), f.name);
        } else if (at && !at.sha) {
          at = await request("PATCH", `/api/uploads/parts/${part}?offset=${at.offset}`, f.slice(at.offset, at.offset + CHUNK), f.name);
        }
        confirmed.set(f.name, at!.offset);
        attempt = 0;
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
        if (status === 404 && part) {
          part = undefined; // the server discarded the part (cleanup): start over
          at = null;
          await run("uploads", "readwrite", (s) => s.delete(`part:${fk}`)).catch(() => undefined);
          continue;
        }
        at = null; // query how far the part got (the server keeps whatever arrived of an interrupted request)
        if (e instanceof Moved) continue;
        // no response at all: the connection, or the file itself (moved or deleted on the user's disk since it was picked)
        if (!(await readable(f))) throw new Refused("E-UPLOAD-FILEGONE", { name: f.name });
        await waitForLine((e as Error).message);
      }
    }
  };

  try {
    // what the server already has (by content this browser sent before): queried all at once
    const known = await Promise.all(files.map((f) => run<string | undefined>("uploads", "readonly", (s) => s.get(fileKey(task.origin, f))).catch(() => undefined)));
    const asked = known.filter((x): x is string => !!x);
    let have = new Set<string>();
    for (;;) {
      try {
        have = new Set(asked.length ? (await json<{ have: string[] }>("POST", "/api/uploads/have", { shas: asked })).have : []);
        break;
      } catch (e) {
        if (stopped) return;
        await waitForLine((e as Error).message);
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
        halt(); // one file failed, so the task has: the other workers stop with it (重试 starts a new round)
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
    window.clearInterval(heart);
    if (useUploads.getState().tasks[key]?.state !== "failed") beat(key, false);
    else void keepTask(useUploads.getState().tasks[key]);
  }
}

/** 连接中断（或服务器未响应）：等待后重试，间隔逐次加长直至 30 秒；浏览器报告恢复在线或重新登录后立即重试，
 * 任务被停下时（`wakes` 中的每一个都被唤醒）立即结束。每个等待者各自监听，同时等待的几个上传工作者一起被唤醒。
 * 整份上传（`sendWith`）、通道级上传（`sendBytes`）与最后一步（`completeSet`）共用这一种等待；结束时任务已停下则
 * 不再改回「上传中」。 */
async function lineWait(key: string, attempt: number, why: string, stopped: () => boolean, wakes?: Set<() => void>): Promise<void> {
  const ms = backoff(attempt + 1, 1000, MAX_WAIT_S * 1000);
  patchTask(key, { state: "waiting", error: why, retryAt: Date.now() + ms, rate: 0 });
  await new Promise<void>((r) => {
    const t = window.setTimeout(done, ms);
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
  if (!stopped()) patchTask(key, { state: "sending", error: "" });
}

/** 字节全部到达后的最后一步，整份上传与通道级上传共用：组装该输入（`POST /api/uploads`，与申报得到的
 * 引用相同），将引用写入参数，在本标签页记录已持有这些文件，并结束任务。`shas`：各文件原文件的内容
 * 指纹；通道级上传的数据同样按原文件记录（服务器清单中记录的即为此值，`lab2shot/transfer/uploads.py make_set`）。 */
export async function completeSet(task: UploadTask, files: File[], shas: Record<string, string>, stopped: () => boolean = () => false): Promise<void> {
  const key = task.key;
  patchTask(key, { state: "finishing", rate: 0 });
  let up: Upload;
  for (let attempt = 0; ; ) {
    try {
      up = await json<Upload>("POST", "/api/uploads", { name: item(task), files: shas, origin: { path: task.origin, client: clientInfo() } });
      break;
    } catch (e) {
      if (stopped()) return;
      if (e instanceof ApiError && e.code === "E-UPLOAD-NOTSENT") throw new Refused(e.said); // files the server does not have: not a dropped connection
      await lineWait(key, attempt++, (e as Error).message, stopped);
    }
  }
  for (const f of files) rememberLocal(key, undefined, shas[f.name], f);
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
export async function sendBytes(key: string, blob: Blob, progress: (sent: number) => void, stopped: () => boolean): Promise<string> {
  const fresh = () => randomId();
  let id = fresh();
  let at: PartState | null = null;
  for (let attempt = 0; ; ) {
    if (stopped()) return "";
    try {
      if (!at) at = await xhr("POST", `/api/uploads/parts?size=${blob.size}&id=${id}`, blob.slice(0, CHUNK), progress);
      else if (!at.sha) {
        const from = at.offset;
        at = await xhr("PATCH", `/api/uploads/parts/${id}?offset=${from}`, blob.slice(from, from + CHUNK), (n) => progress(from + n));
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
        at = await xhr("GET", `/api/uploads/parts/${id}`, null).catch(() => null); // 查询已上传的位置，从该处续传
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
