/** 取视图数据字节的唯一入口：二维帧（图片路径、通道路径）、三维块、包自带的整份数据（包说明、人物框、跟踪点、查找表），
 * 舞台绘制、整段取回、跨包预取都经这里。
 * 调用方只决定要哪些、什么顺序；这里负责：
 *   - 已在缓存就直接给；
 *   - 同一个键已在路上就等那一次（在途去重）；
 *   - 调用方放弃（舞台换源）只是它不再等，所有等的都放弃了才中止传输，不连累整段取回和预取；
 *   - 连接配额：同一服务器的连接（HTTP/1.1 每个域名 6 条）在这里排队，正在画的先走（`Lane`）；
 *   - 失败记忆：页面上「这一帧 / 这一块暂时别再要」只记在这里（`mayAsk` / `whenAskable` / `failed` / `succeeded`）。
 *     失败分三类，只在 `kindOf` 一处分：
 *       等登录（401、协议没同意的 403，或登录已结束、根本没发：platform/http.ts awaitsLogin）→ 不是这一帧的事：
 *         不记失败，等重新登录后重放（`NeedLogin`）；
 *       没有 / 不给看（404 / 410、别的 403，或本机文件解不开）→ 记下，半分钟内不再要（`Gone`），原因留在失败记录里；
 *       其它（断线、卡住超过 STALL_MS 没有回音（`Stalled`）、5xx、408、429，以及服务器还在生成 E-VIEW-PREPARING）→ 按连续次数退避（1 秒起翻倍，最多半分钟）。
 *     「还在生成」与「这次没拿到」（`Later`）不是错误：调用方用 `notAnError` 判断，不说给使用者。
 *     重新登录后整张失败记忆清一次（登录前的失败多半是登录造成的），等登录的一起重放。
 *     取帧账本、整段取回、三维场景都问这里，不各自再记一份。
 * 地址和键都从 transfer/frameKey.ts 来；存哪、怎么取由各自的「运法」（carry）声明。
 * 正在显示的不被淘汰由缓存的 pin 负责（platform/cache.ts）。 */

import { ApiError, LOGGED_IN, STALL_MS, awaitsLogin, blob, bytes as fetchBytes, loginEnded, request } from "../platform/http";
import { cache, type Slot } from "../platform/cache";
import { backoff } from "../platform/backoff";
import { MessageError } from "../messages/message";
import { sizeText } from "../platform/format";
import { squeeze, unsqueeze } from "./plane";
import { bytesSlot, frameKey, partSlot, sceneDescKey } from "./frameKey";

/** 服务器无法提供这一帧（不存在，或确定画不出来）：记下，半分钟内不再要。 */
export class Gone extends Error {}

/** The network part of a fetch took longer than STALL_MS: a dead connection a tunnel holds open. Recorded as a failure
 * to retry (backoff), like any other that did not arrive. */
export class Stalled extends Error {}

/** `run` (the network part, once it has a connection), given up after STALL_MS (platform/http.ts, the same limit as an
 * upload's): it is aborted and the caller gets Stalled. Otherwise one connection that never answers holds its lane, and
 * the ledger's slots (transfer/frames.ts) with it, for good: the viewer would wait on 「加载中」 for ever. */
function unstalled<T>(run: () => Promise<T>, ctrl: AbortController): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const t = setTimeout(() => (reject(new Stalled("stalled")), ctrl.abort()), STALL_MS);
    run().then(resolve, reject).finally(() => clearTimeout(t));
  });
}
/** 这个键刚失败过、还在退避期：这次不去要。`key`：在退避的是哪个键（取字节的键；调用方自己的键可能不同，
 * 例如取帧账本的键带着「来源」段）：要等就等它（whenAskable(key)）。 */
export class Later extends Error {
  constructor(readonly key: string) {
    super("later");
  }
}
/** 这个键在等重新登录（认证 / 权限被拒，或登录已结束）：不算失败，登录后重放。 */
export class NeedLogin extends Error {}

/** 连接配额：取视图数据的请求在这里排队。`now`（正在画的这一帧、当前帧附近的三维块、包说明）可用满 `NET` 条、
 * 先走；`later`（整段取回、跨包预取、三维整段下载）同时最多 `LATER_MAX` 条，且只在有空位时走——当前帧前面至多压着
 * 这几个后台请求。同一个键已由后台在排队，又有正在画的要它时升为 `now`。 */
export type Lane = "now" | "later";
const NET = 6;
const LATER_MAX = 3;

interface Waiting {
  wants: { lane: Lane };
  go: (lane: Lane) => void;
}

class Lanes {
  readonly busy = { now: 0, later: 0 };
  readonly queue: Waiting[] = [];

  grant(): void {
    for (;;) {
      if (this.busy.now + this.busy.later >= NET) return;
      let i = this.queue.findIndex((w) => w.wants.lane === "now");
      if (i < 0 && this.busy.later < LATER_MAX) i = this.queue.findIndex((w) => w.wants.lane === "later");
      if (i < 0) return;
      const [w] = this.queue.splice(i, 1);
      this.busy[w.wants.lane]++;
      w.go(w.wants.lane);
    }
  }

  /** 排到一条连接再跑 `run`（只有走网络的那一段）；调用方放弃（signal）就不再排。 */
  async through<T>(wants: { lane: Lane }, signal: AbortSignal, run: () => Promise<T>): Promise<T> {
    const lane = await new Promise<Lane>((go, stop) => {
      const w: Waiting = { wants, go: (l) => (signal.removeEventListener("abort", quit), go(l)) };
      const quit = () => {
        const i = this.queue.indexOf(w);
        if (i >= 0) this.queue.splice(i, 1);
        stop(new DOMException("aborted", "AbortError"));
      };
      if (signal.aborted) return quit();
      signal.addEventListener("abort", quit);
      this.queue.push(w);
      this.grant();
    });
    try {
      return await run();
    } finally {
      this.busy[lane]--;
      this.grant();
    }
  }
}
const lanes = new Lanes();

/** 一种字节怎么取、存在哪。`net`：走网络的那一段交给它排队（读本机硬盘的不排）。 */
interface Carry<V> {
  cached: () => V | undefined;
  fetch: (signal: AbortSignal, net: <T>(run: () => Promise<T>) => Promise<T>) => Promise<V>;
  store: (value: V) => void;
}

const FAILED_MAX = 5000;
const GONE_TTL = 30_000; // 半分钟：服务器删了又按同一指纹重算的包，之后还能再取
const RETRY_FIRST = 1000;

interface Failure {
  at: number;
  n: number; // 连续失败次数
  gone: boolean;
  last: unknown; // 记下的那个错误：同一次失败被两层各报一次时不重复计数
}

class Store {
  readonly asking = new Map<string, { done: Promise<unknown>; ctrl: AbortController; waiters: number; wants: { lane: Lane } }>();
  readonly failed = new Map<string, Failure>(); // 键 -> 最近的失败（成功即删；有上限，先删最早的）
  readonly held = new Set<string>(); // 等重新登录的键
  readonly replay = new Set<() => void>(); // 重新登录后要重放的
}
const own = new Store();

const holdOf = (f: Failure): number => (f.gone ? GONE_TTL : backoff(f.n, RETRY_FIRST, GONE_TTL));

/** 这个键还要等多久才能再要（毫秒，0：现在就可以）。 */
export function waitLeft(key: string): number {
  const f = own.failed.get(key);
  return f ? Math.max(0, f.at + holdOf(f) - Date.now()) : 0;
}

/** 现在能不能去要这个键：不在等登录，也不在退避期。 */
export const mayAsk = (key: string): boolean => !own.held.has(key) && waitLeft(key) === 0;

/** 这个键能再要时叫 `fn` 一次：在等登录的，登录后；在退避期的，到期后。返回撤销。 */
export function whenAskable(key: string, fn: () => void): () => void {
  if (own.held.has(key)) return afterLogin(fn);
  const t = setTimeout(fn, waitLeft(key) + 50);
  return () => clearTimeout(t);
}

if (typeof addEventListener === "function")
  addEventListener(LOGGED_IN, () => {
    own.failed.clear();
    own.held.clear();
    [...own.replay].forEach((f) => f());
  });

const isAbort = (e: unknown): boolean => e instanceof DOMException && e.name === "AbortError";

/** 失败的三类（见文件头）：唯一的分法。 */
function kindOf(e: unknown): "login" | "gone" | "retry" | "none" {
  if (isAbort(e) || e instanceof Later) return "none";
  if (e instanceof NeedLogin || loginEnded()) return "login";
  if (awaitsLogin(e)) return "login";
  // 没有（404 / 410），或不给看（不带 terms 的 403：权限、归属，重新登录也不变）：记下，半分钟后可再试
  if (e instanceof ApiError) return e.status === 404 || e.status === 410 || e.status === 403 ? "gone" : "retry";
  return e instanceof Gone ? "gone" : "retry";
}

/** 记下一次失败（调用方放弃的不算；等登录的只记「在等」）。取字节的失败在 `get` 里已记；取到之后才出的错（解码、解析）
 * 由调用方按同一个键记。 */
export function failed(key: string, e: unknown): void {
  const kind = kindOf(e);
  if (kind === "none") return;
  if (kind === "login") return void own.held.add(key);
  const was = own.failed.get(key);
  if (was?.last === e) return;
  if (!was && own.failed.size >= FAILED_MAX) own.failed.delete(own.failed.keys().next().value as string);
  own.failed.set(key, { at: Date.now(), n: (was?.n ?? 0) + 1, gone: kind === "gone", last: e });
}

/** 这个键最近一次失败的错误（没有失败记录为 undefined）：要把原因说给使用者的一方读。 */
export const lastFailure = (key: string): unknown => own.failed.get(key)?.last;

/** 取到了（并且用上了）：清掉失败记录。 */
export const succeeded = (key: string): void => void own.failed.delete(key);

/** 抛给调用方的错误：按类换成 NeedLogin / Gone（调用方只看类型），其它原样。 */
function thrown(e: unknown): unknown {
  const kind = kindOf(e);
  if (kind === "login") return e instanceof NeedLogin ? e : new NeedLogin(e instanceof Error ? e.message : String(e));
  // 服务器说的原因带着（不给看的 403 要说清为什么）
  if (kind === "gone" && !(e instanceof Gone)) return new Gone(e instanceof ApiError ? e.message || String(e.status) : String(e));
  return e;
}

function get<V>(key: string, carry: Carry<V>, signal?: AbortSignal, lane: Lane = "now"): Promise<V> {
  const kept = carry.cached();
  if (kept !== undefined) return Promise.resolve(kept);
  if (own.held.has(key) || loginEnded()) {
    own.held.add(key); // 登录已结束：不发（answer 也不会发），不记失败，登录后重放
    return Promise.reject(new NeedLogin("login"));
  }
  if (waitLeft(key) > 0) return Promise.reject(own.failed.get(key)?.gone ? new Gone("gone") : new Later(key));
  let entry = own.asking.get(key);
  if (entry && lane === "now" && entry.wants.lane === "later") {
    entry.wants.lane = "now"; // 还在排队的升为正在画的（已在传的不受影响）
    lanes.grant();
  }
  if (!entry) {
    const ctrl = new AbortController();
    const wants = { lane };
    const done = carry.fetch(ctrl.signal, (run) => lanes.through(wants, ctrl.signal, () => unstalled(run, ctrl)))
      .then((v) => (carry.store(v), succeeded(key), v))
      .catch((e) => {
        const why = thrown(e);
        failed(key, why);
        throw why;
      })
      .finally(() => { if (own.asking.get(key) === entry) own.asking.delete(key); });
    entry = { done, ctrl, waiters: 0, wants };
    own.asking.set(key, entry);
  }
  const shared = entry;
  shared.waiters++;
  const done = shared.done as Promise<V>;
  if (!signal) return done.finally(() => void shared.waiters--);
  return new Promise<V>((resolve, reject) => {
    let gone = false; // 这一位已经不等了（放弃，或拿到了结果）：只减一次
    const out = () => {
      if (gone) return false;
      gone = true;
      signal.removeEventListener("abort", leave);
      shared.waiters--;
      return true;
    };
    const leave = () => {
      if (!out()) return;
      // 最后一个等的人走了：中止传输，并当场把它从在途表里拿掉——之后再要同一个键的另开一次，不接上这次已中止的
      if (shared.waiters <= 0 && own.asking.get(key) === shared) {
        own.asking.delete(key);
        shared.ctrl.abort();
      }
      reject(new DOMException("aborted", "AbortError"));
    };
    if (signal.aborted) return leave();
    signal.addEventListener("abort", leave);
    done.then((v) => out() && resolve(v), (e) => out() && reject(e));
  });
}

// ------------------------------------------------------------------ 二维帧的压缩字节

/** 通道路径里没法压缩（浏览器没有 CompressionStream）时原样存的字节：读的时候不解压。 */
const RAW = "application/x-lab2shot-raw";

/** 一帧的压缩字节（字节层 `small`，整段保留：位图被淘汰后从它重新解码，数毫秒，不走网络）。
 * `picture`：服务器给的 WebP / PNG 原样存；`channel`：gzip 已被浏览器解开，自行无损压缩后存（plane.ts squeeze）。 */
export function frameBytes(of: { id: string; frame: number }, url: string, carry: "picture" | "channel", signal?: AbortSignal, lane: Lane = "now"): Promise<Blob> {
  const key = frameKey(of.id, of.frame);
  const at = bytesSlot(of.id, of.frame);
  return get<Blob>(key, {
    cached: () => cache.get<Blob>(at.key),
    fetch: async (s, net) => {
      if (carry === "picture") return net(() => blob(url, { signal: s, headers: { Accept: "image/webp,image/png" } }));
      const buf = await net(() => fetchBytes(url, { signal: s }));
      return (await squeeze(buf)) ?? new Blob([buf], { type: RAW });
    },
    store: (b) => cache.keep(at, b, b.size, "small"),
  }, signal, lane);
}

/** 通道路径存下的字节还原成数据。 */
export const planeBytes = async (b: Blob): Promise<ArrayBuffer> => (b.type === RAW ? b.arrayBuffer() : unsqueeze(b));

// ------------------------------------------------------------------ 三维块

/** 单个三维块的上限：超过时浏览器装不下，说清大小，不静默截断。 */
const MAX_PART = 1.5e9;

async function loadPart(url: string, signal: AbortSignal): Promise<Uint8Array> {
  const r = await request(url, { signal }); // 没登录被拒的交给登录门
  const size = Number(r.headers.get("content-length") ?? 0);
  if (size > MAX_PART) throw new MessageError("E-VIEW-PARTTOOBIG", { size: sizeText(size), max: sizeText(MAX_PART) });
  try {
    return new Uint8Array(await r.arrayBuffer());
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    throw size ? new MessageError("E-VIEW-PARTNOROOM", { size: sizeText(size), reason }) : new MessageError("E-VIEW-NOROOM", { reason });
  }
}

/** 一个三维块（地址由服务器在场景描述里给，一个地址一串字节）。
 * `disk`：本机硬盘上的整段缓存（view/sceneStore.ts）——先从盘上读；能存盘的块由取它的一方写盘，不在内存另留一份
 * （整段可达数百 MB）；不能存盘的（边算边看）留在页面缓存的 `fetched` 档。
 * 给的是一份拷贝：接收方会把字节整体转交解析线程，共享的原件转交后就空了。 */
export function partBytes(url: string, disk: { durable: (url: string) => boolean; read: (url: string) => Promise<Uint8Array | null> }, lane: Lane): Promise<Uint8Array> {
  const at = partSlot(url);
  return get<Uint8Array>(at.key, {
    cached: () => cache.get<Uint8Array>(at.key),
    fetch: async (s, net) => (await disk.read(url).catch(() => null)) ?? net(() => loadPart(url, s)),
    store: (bytes) => { if (!disk.durable(url)) cache.keep(at, bytes, bytes.byteLength, "fetched"); },
  }, undefined, lane).then((bytes) => bytes.slice());
}

// ------------------------------------------------------------------ 整份 JSON（包说明、人物框、跟踪点、查找表）

/** 解析后的 JSON 在内存里约是文本的几倍（大量小对象：每个对象、每个数都有自己的开销）。 */
const PARSED = 3;

/** 一份 JSON 数据（`read` 把回复转成要存的样子），存在字节层 `small`。记账：`sizeOf` 给了就按它（转成紧凑结构的，
 * 如查找表的 Uint16Array），否则按回复文本长度 × PARSED 估。
 * 键带代次的（transfer/frameKey.ts describedKey），重算后换键重取；失败按同一套退避。 */
export function described<V>(at: Slot, url: string, read: (raw: unknown) => V = (raw) => raw as V, signal?: AbortSignal, sizeOf?: (v: V) => number): Promise<V> {
  let size = 0;
  return get<V>(at.key, {
    cached: () => cache.get<V>(at.key),
    fetch: async (s, net) => {
      const text = await net(async () => (await request(url, { signal: s })).text());
      const v = read(JSON.parse(text));
      size = sizeOf ? sizeOf(v) : text.length * PARSED;
      return v;
    },
    store: (v) => cache.keep(at, v, size, "small"),
  }, signal);
}

/** 同上，已在缓存里的（同步，不发请求）。 */
export const describedNow = <V>(key: string): V | undefined => cache.get<V>(key);

/** 三维场景的描述（view/sceneData.ts）：与别的视图数据同一个入口（在途去重、调用方都放弃才中止、失败记忆、连接配额），
 * 但不进缓存——留着的是由它建出的场景（边算边看的描述地址会变，也不能按地址缓存）。服务器还在生成
 * （E-VIEW-PREPARING）按退避再问（不当错误），那时走 `later` 道，不占正在画的连接。 */
export function sceneDescription(key: string, url: string, signal?: AbortSignal, lane: Lane = "now"): Promise<unknown> {
  return get<unknown>(sceneDescKey(key), {
    cached: () => undefined,
    fetch: (s, net) => net(async () => (await request(url, { signal: s })).json()),
    store: () => undefined,
  }, signal, lane);
}

/** 这次失败不必说给使用者：服务器还在生成（过一会儿自然有），或只是在退避期没去要。 */
export const notAnError = (e: unknown): boolean =>
  e instanceof Later || e instanceof NeedLogin || (e instanceof ApiError && e.code === "E-VIEW-PREPARING");

/** 登录后叫 `fn` 一次（等登录的一方用：得到 NeedLogin 的）。返回撤销。 */
export function afterLogin(fn: () => void): () => void {
  const once = () => (own.replay.delete(once), fn());
  own.replay.add(once);
  return () => void own.replay.delete(once);
}
