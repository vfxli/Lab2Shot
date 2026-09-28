/** 本机代理：读取本机 EXR 时在后台生成一份比服务器代理（默认 512）更清晰的缓存，默认边长 1024、磁盘上限 10 GB，
 * 生成进度显示在节点右上角。
 *
 * 用户选择序列后（或视图首次需要绘制用户机器上的某一帧时），本模块在后台将每一帧的每一层缩放到管理员设定的档位并压缩，
 * 写入浏览器私有文件系统（`store.ts`）；视图从该缓存绘制（`transfer/sources.ts fromFile`）。
 * 只生成各颜色层的显示图（以空间换流畅度，但不为无人查看的层占用磁盘）；数值通道（深度、遮罩、法线等）不生成：
 * 两个入口（`prepareLocalProxies`、`localPicture`）的 `planes` 均为空。当前查看的帧优先，其余按顺序处理（`worker.ts`）。
 *
 * 内存中只保存已显示的帧：本模块的产出均在磁盘上，读入内存的只有视图实际需要绘制的帧（由取帧账本的预算管理）。 */
import { create } from "zustand";
import { workerAsks } from "../../platform/work";
import { onServerChange, serverNow } from "../../state/server";
import { lutOfFile } from "../lut";
import type { Lut } from "../lookup";
import type { LayerSpec, ProxyAnswer, ProxyAsk } from "./worker";
import { madeKeys, onDropped, proxiesMade, proxyStoreAvailable, readProxy, writeProxy } from "./store";
import { cache } from "../cache";
import { shortHash } from "../../platform/digest";

// 已在磁盘上生成本机代理的文件：登记到页面唯一的缓存（不另建表），账本据此判断某帧是否已持有
// （`transfer/sources.ts localFramesOf`），时间线的绿色随生成进度推进（与 Nuke 一样可见缓存进度）
// 就绪标记包含档位：管理员更改本机代理尺寸后，旧档位生成的代理不计为已持有；
// 登记层（transfer/cache.ts registry）不计入预算、不会被淘汰
const READY = (fileKey: string, tier = localTier()) => `proxy:${fileKey}|${tier}`;
export const proxyReady = (fileKey: string): boolean => cache.registered(READY(fileKey)) === true;
const markReady = (fileKey: string, tier = localTier()) => { if (cache.registered(READY(fileKey, tier)) !== true) cache.register(READY(fileKey, tier), true); };
// a proxy trimmed off the disk (over the local cache's size) is not ready any more: made again when it is wanted
onDropped((fileKey, tierKey) => cache.unregister(READY(fileKey, Number(tierKey.split("-")[0]))));

export type { LayerSpec };

const DEFAULT_PX = 1024;
const DEFAULT_CAP_GB = 10;
const QUALITY = 0.9;
const PLANE = ".l2c1.gz";

/** 管理员设定的两个数值（随 `/api/server` 返回；服务器第一次回答之前使用默认值）。 */
const localTier = (): number => serverNow()?.view.local_px ?? DEFAULT_PX;
const capBytes = (): number => (serverNow()?.view.local_cache_gb ?? DEFAULT_CAP_GB) * (1 << 30);

/** 文件键：名称、大小、修改时间（`File` 均具备，刷新后或从授权目录重新获取时保持一致）。
 * 不使用内容指纹：指纹在申报步骤中异步计算，视图首次绘制时尚不可用，使用它会导致同一文件被处理两次。 */
export const fileKeyOf = (file: File): string => `n_${shortHash(`${file.name}|${file.size}|${file.lastModified}`)}`;

/** 显示变换的标识（表相同则代理相同）：模式、网格、范围及数据的短散列。 */
function lutKey(lut: Lut | null): string {
  if (!lut) return "raw";
  return `${lut.mode}${lut.size}_${shortHash(lut.data, 97)}`;
}

const isExr = (name: string) => name.toLowerCase().endsWith(".exr");

// ---------------------------------------------------------------- 进度（供节点右上角状态格读取）

export interface ProxyProgress { done: number; total: number; failed: number }
export const useLocalProxies = create<{ tasks: Record<string, ProxyProgress> }>(() => ({ tasks: {} }));

const progress = (key: string, patch: Partial<ProxyProgress>) =>
  useLocalProxies.setState((s) => ({ tasks: { ...s.tasks, [key]: { ...(s.tasks[key] ?? { done: 0, total: 0, failed: 0 }), ...patch } } }));

// ---------------------------------------------------------------- 队列和 worker 池

interface Job {
  fileKey: string;
  file: File;
  space: string;
  rules: string; // 整段按同一文件名请求显示变换表
  layers: LayerSpec[];
  planes: string[]; // 本次需生成平面的通道（两个入口均传空：只生成显示图）
  task: string; // 进度记录的键（节点|参数）
  order: number; // 值越小越优先
  waiters: ((ok: boolean) => void)[];
}

// 并行 worker 数：为页面和解码窗口保留两个核心；上限 8 个（更多时瓶颈在磁盘和内存）
const POOL = Math.max(1, Math.min(8, (typeof navigator !== "undefined" ? navigator.hardwareConcurrency ?? 4 : 4) - 2));
const asks = Array.from({ length: POOL }, () =>
  workerAsks<ProxyAnswer & { id: number }>(() => new Worker(new URL("./worker.ts", import.meta.url), { type: "module" })));

// `running`：正在处理的任务及其所属任务（等待其完成的调用方需挂接，见 `enqueue`）；`scannedTier`：磁盘上就绪标记已按哪一档位扫描
const own: { queue: Map<string, Job>; running: Map<string, Job>; clock: number; free: number[]; scannedTier: number | null } =
  { queue: new Map(), running: new Map(), clock: 0, free: Array.from({ length: POOL }, (_, i) => i), scannedTier: null };

// 刷新后，磁盘上当前档位已有的代理立即显示为绿色（madeKeys 返回 `文件键/档位-显示表`）。
// 档位须待服务器返回后才能得知：页面打开时 `serverNow()` 仍为空，若只在模块加载时按默认值 1024 扫描一次，
// 管理员设定其他档位时，已生成的代理刷新后将全部不被识别并整段重新生成。因此服务器每次响应时检查档位，变化时按新档位重新扫描
// （档位相同时不重复扫描；旧档位的标记可保留，`proxyReady` 只检查当前档位的键）
function scanReady(): void {
  if (!proxyStoreAvailable()) return;
  const tier = localTier();
  if (own.scannedTier === tier) return;
  own.scannedTier = tier;
  void madeKeys().then((keys) => keys.forEach((k) => { const [fk, tk] = k.split("/"); if (tk?.startsWith(`${tier}-`)) markReady(fk, tier); })).catch(() => undefined);
}
scanReady();
onServerChange(scanReady);

const tierKeyOf = (lut: Lut | null) => `${localTier()}-${lutKey(lut)}`;

async function lutFor(job: Job): Promise<Lut | null> {
  if (!isExr(job.file.name)) return null;
  return lutOfFile(job.space, job.rules || job.file.name).catch(() => null);
}

async function run(job: Job, slot: number): Promise<void> {
  let ok = false;
  try {
    const lut = await lutFor(job);
    const tierKey = tierKeyOf(lut);
    // 已存在的不重复生成：显示图生成一次即全部就绪；平面按名称逐条检查
    const made = new Set(await proxiesMade(job.fileKey, tierKey));
    const pictures = ![...made].some((n) => n.endsWith(".webp"));
    const planes = job.planes.filter((n) => !made.has(`${n}${PLANE}`));
    if (!pictures) markReady(job.fileKey);
    if (!pictures && !planes.length) { ok = true; return; }
    const bytes = new Uint8Array(await job.file.arrayBuffer());
    const ask: ProxyAsk = { kind: isExr(job.file.name) ? "exr" : "image", bytes, tier: localTier(), quality: QUALITY, lut, layers: job.layers,
                            pictures, planes };
    const got = await asks[slot](ask, [bytes]);
    const cap = capBytes();
    for (const p of got.pictures) await writeProxy(job.fileKey, tierKey, `${p.name}.webp`, p.webp, cap);
    for (const p of got.planes) await writeProxy(job.fileKey, tierKey, `${p.name}${PLANE}`, p.gz, cap);
    if (got.pictures.length) markReady(job.fileKey);
    ok = true;
  } catch (e) {
    console.error("local proxy failed", job.file.name, e); // 面向开发者的日志，非用户消息
  } finally {
    own.running.delete(job.fileKey);
    own.free.push(slot);
    const t = useLocalProxies.getState().tasks[job.task];
    if (t) progress(job.task, ok ? { done: t.done + 1 } : { failed: t.failed + 1 });
    job.waiters.forEach((f) => f(ok));
    pump();
  }
}

function pump(): void {
  while (own.free.length) {
    const next = [...own.queue.values()].sort((a, b) => a.order - b.order)[0];
    if (!next) return;
    own.queue.delete(next.fileKey);
    own.running.set(next.fileKey, next);
    void run(next, own.free.pop()!);
  }
}

function enqueue(job: Omit<Job, "order" | "waiters">, first: boolean): Promise<boolean> {
  const had = own.queue.get(job.fileKey);
  if (had) {
    if (first) had.order = -(++own.clock);
    for (const n of job.planes) if (!had.planes.includes(n)) had.planes.push(n);
    return new Promise((f) => had.waiters.push(f));
  }
  // 正在生成：等待其完成后再返回。若立即返回 true，调用方（`localPicture`）随即读取磁盘，而文件尚未写入，
  // 读到 null 后将整帧实时解码，恰好是本机代理要省去的步骤。本次所需的通道（planes）不并入正在运行的任务（其输出通道在开始时已确定），
  // 读取不到时仍实时解码一次，下次需要时再生成
  const going = own.running.get(job.fileKey);
  if (going) return new Promise((f) => going.waiters.push(f));
  const made: Job = { ...job, order: first ? -(++own.clock) : ++own.clock, waiters: [] };
  own.queue.set(job.fileKey, made);
  const p = new Promise<boolean>((f) => made.waiters.push(f));
  pump();
  return p;
}

// ---------------------------------------------------------------- 对外接口

/** 选择序列后：整段排入后台队列（当前帧优先）。`task`：进度记录的键（节点|参数）。 */
export function prepareLocalProxies(task: string, files: File[], space: string, layers: LayerSpec[],
                                    currentIndex = 0): void {
  if (!proxyStoreAvailable() || !files.length) return;
  progress(task, { total: files.length, done: 0, failed: 0 });
  const rules = files[0].name;
  const order = [...files.keys()].sort((a, b) => Math.abs(a - currentIndex) - Math.abs(b - currentIndex));
  for (const i of order) {
    const f = files[i];
    void enqueue({ fileKey: fileKeyOf(f), file: f, space, rules, layers, planes: [], task }, false)
      .catch(() => undefined);
  }
}

/** 视图需要绘制某帧的某一层：磁盘上已有则返回，否则优先生成后返回（`null`：无法生成，调用方照常整帧实时解码）。 */
export async function localPicture(file: File, layer = "rgba", space = "", rules = ""): Promise<Blob | null> {
  if (!proxyStoreAvailable()) return null;
  const key = fileKeyOf(file);
  const lut = isExr(file.name) ? await lutOfFile(space, rules || file.name).catch(() => null) : null;
  const tierKey = tierKeyOf(lut);
  const had = await readProxy(key, tierKey, `${layer}.webp`);
  if (had) return had;
  const ok = await enqueue({ fileKey: key, file, space, rules, layers: [], planes: [], task: `on-demand` }, true);
  return ok ? readProxy(key, tierKey, `${layer}.webp`) : null;
}
