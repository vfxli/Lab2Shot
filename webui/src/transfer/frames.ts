import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { api, type Manifest } from "../api";
import { aheadFor, order } from "./frameWindow";
import { PIXELS_BUDGET, cache } from "./cache";
import { freePixels, sizeOf, type Pixels, type Plane } from "./plane";
import { Gone, abortUnder, bytesKey, localFramesOf, packetPart } from "./sources";
import type { FrameSource } from "./sources";
import { fillWhole, stopFilling } from "./fill";
import { backoff } from "../platform/backoff";

// 单帧的获取方式位于 transfer/sources.ts（下一层）：本模块只负责账本，即当前应取哪些帧、
// 取到何时、先后顺序及何时取消。取数层的导出在此原样转出，调用方无需了解两层的分界。
export * from "./sources";
export type { Pixels, Plane };

/** 二维视图所绘制帧的账本：决定当前应取哪些帧、先后顺序以及何时绘制。
 *
 * 两项职责，两条路径：
 * - 本模块：当前需要绘制的帧，解码后保留；每个源一个窗口，共用解码层的字节预算（frameWindow.ts）；
 * - `transfer/fill.ts`：整段的压缩字节，查看某个源时即取回其整段，不等待播放、不解码。
 *
 * 源上的每一帧都是代理（服务器计算后按管理员设定的档位缩放、压缩的版本），地址随内容固定，
 * 因此浏览器可长期保留；用户在本标签页中选择的文件直接从磁盘绘制，不经过网络传输。
 * 某帧尚未到达时保留该源上一张已到达的图（标记「加载中」），从不绘制黑帧；
 * 两个播放器互不干扰，且窗口移动时不会中止正在传输的帧（见下方 `release`）。
 *
 * 图像存放于页面唯一的缓存（transfer/cache.ts），键为 `${源}:${帧}`；本模块只记录在途请求，不存储像素。 */

const PARALLEL = 6; // frames fetched concurrently (the browser has 6 connections per server over HTTP/1.1)

/** What is being fetched and who is waiting for it; no pixels (those belong to the cache). */
interface Asked {
  key: string;
  load: () => Promise<Pixels>;
  failedAt: number; // time of the last failed fetch (0: never); not requested again while `holdOff` applies
  fails: number; // consecutive failure count (0: none); the hold-off grows with it
  gone: boolean; // the last failure was the server reporting that the frame does not exist (Gone): held off for a flat FAIL_TTL
  waiters: Set<() => void>;
}

/** 「该帧不存在」的记录保留半分钟。不能永久保留：服务器删除包后又重算（指纹相同）时，本标签页仍需能够重新获取该帧。 */
const FAIL_TTL = 30_000;
/** 一次失败后的重试间隔（毫秒）。非 Gone 的失败同样需要退避：服务器 5xx、断线、本机文件无法解码时，
 * 若不退避，`finally` 中的 `pump()` 会立即再次请求同一个键；6 路并发且每次失败都触发重绘，
 * 一个坏帧即可使页面陷入死循环。按连续失败次数翻倍：1 秒、2 秒、4 秒……最多 FAIL_TTL；到期后重试一次。 */
const holdOff = (e: Asked): number => (e.gone ? FAIL_TTL : backoff(e.fails, 1000, FAIL_TTL));
const stillFailed = (e: Asked | undefined): boolean => !!e && e.failedAt > 0 && Date.now() - e.failedAt < holdOff(e);

/** What is being fetched now, for whom, and what is pending from a worker: the module's state in one place.
 * `windows` / `loading` / `manifests` follow the open windows and clear as each response arrives; `asked` keeps a
 * frame's failure record after its window has moved on, so it is bounded by `trimAsked` instead. The pictures belong to the cache. */
class Fetching {
  readonly asked = new Map<string, Asked>(); // the fetcher's work, by key (bounded: ASKED_MAX)
  readonly bigOf = new Map<string, number>(); // source id -> the byte size of one of its decoded frames (only while it has a window)
  readonly windows = new Map<string, { keys: string[]; at: number }>(); // source id -> its window, most wanted first
  readonly loading = new Set<string>(); // the keys in flight
  readonly manifests = new Map<string, Promise<Manifest>>(); // the requested descriptions; a response is stored in the cache
  clock = 0;
}

const fetching = new Fetching();
const asked = fetching.asked;
const windows = fetching.windows;
const loading = fetching.loading;

const wantedKeys = (): Set<string> => new Set([...windows.values()].flatMap((w) => w.keys));
cache.pin(wantedKeys); // what a window wants is being drawn: the cache never releases it

const picture = (key: string): Pixels | undefined => cache.get<Pixels>(key);
const hold = (key: string, px: Pixels, near: boolean) => {
  cache.keep(key, px, sizeOf(px), near ? "near" : "viewing", { free: freePixels });
  fetching.bigOf.set(key.slice(0, key.lastIndexOf(":")), sizeOf(px)); // 计算窗口宽度时需要（见下方 windowFrames），避免每次渲染都扫描缓存
};

/** 账本中键的数量上限：超过时，从最早的条目开始移除既不在任何窗口中、也不在传输中的键。
 * 若不修剪，`asked` 只增不减（切换节点、通道、代理档位都会产生一组新键），长时间使用可积累数万条。
 * 移除一条只丢失其失败记录：再次需要时重新登记、重新获取。 */
const ASKED_MAX = 5000;
function trimAsked(): void {
  if (asked.size <= ASKED_MAX) return;
  const wanted = wantedKeys();
  for (const key of asked.keys()) {
    if (asked.size <= ASKED_MAX) return;
    if (!wanted.has(key) && !loading.has(key)) asked.delete(key);
  }
}

function pump(): void {
  // the most recently requested window first; within it, its own order
  const queue = [...windows.values()].sort((a, b) => b.at - a.at).flatMap((w) => w.keys);
  for (const key of queue) {
    if (loading.size >= PARALLEL) return;
    const e = asked.get(key);
    if (!e || cache.has(key) || stillFailed(e) || loading.has(key)) continue;
    loading.add(key);
    void e
      .load()
      .then((img) => {
        hold(key, img, false); // only frames that arrived are kept; a failed fetch is not
        e.failedAt = 0;
        e.fails = 0;
        e.gone = false;
      })
      .catch((err) => {
        // 任何失败都记录时间并退避（见上方 `holdOff`）；在此期间 `stillFailed` 阻止下方 pump() 再次请求同一个键。
        // 到期时若窗口无变化则不会有人调用 pump（播放器停住等待），因此到期后主动 pump 一次；若该键已不在账本中则不处理
        e.failedAt = Date.now();
        e.fails += 1;
        e.gone = err instanceof Gone;
        setTimeout(() => {
          if (asked.get(key) === e && !cache.has(key)) pump();
        }, holdOff(e) + 50);
      })
      .finally(() => {
        loading.delete(key);
        e.waiters.forEach((f) => f());
        pump();
      });
  }
}

/** The frames a source wants now: `order` from the current frame outward; frames no window wants any more stop loading.
 * A frame the prefetcher has already fetched is in the cache under this same key: it is drawn as is, promoted from the
 * prefetch class, and never fetched again. */
function want(id: string, list: { key: string; load: () => Promise<Pixels> }[]): void {
  windows.set(id, { keys: list.map((w) => w.key), at: ++fetching.clock });
  list.forEach((w, i) => {
    if (cache.has(w.key)) cache.reclass(w.key, i === 0 ? "viewing" : "near");
    else if (!asked.has(w.key)) asked.set(w.key, { ...w, failedAt: 0, fails: 0, gone: false, waiters: new Set() });
  });
  trimAsked();
  pump();
}

/** A source that no longer wants frames (its view closed): its window is removed after this turn (a view that
 * re-requests in the same render keeps its loads). */
function release(id: string, keys: string[]): void {
  const w = windows.get(id);
  if (w && w.keys === keys) {
    windows.delete(id);
    fetching.bigOf.delete(id);
    stopFilling(id);
    // 切换离开的源，其已解码的帧保留并降一档：预算未超时一直保留，切回时可直接使用；
    // 超出预算时先于当前查看的帧被淘汰（`near` 排在 `viewing` 之前，同档内最久未使用的先淘汰）
    for (const key of keys) if (cache.has(key)) cache.reclass(key, "near");
  }
}

// ------------------------------------------------------------------ a packet's description

/** 该包的说明当前是否已持有（同步，不发请求）：取数层据此同步判断本次查看的路径
 * （`transfer/prefetchLive.ts keyOf`）。`manifestOf` 取回后即存放于此缓存。 */
export const manifestNow = (fp: string): Manifest | null => cache.get<Manifest>(`manifest:${fp}`) ?? null;

/** A packet's description (its meta: frames, size, rate), fetched once and kept by fingerprint; the 2D stage, the
 * timeline and the prefetcher read the same copy. A refusal is not kept (requested again when needed again). */
export function manifestOf(fp: string): Promise<Manifest> {
  const key = `manifest:${fp}`;
  const got = cache.get<Manifest>(key);
  if (got) return Promise.resolve(got);
  let asking = fetching.manifests.get(fp);
  if (!asking) {
    asking = api
      .manifest(fp)
      .then((m) => {
        cache.keep(key, m, JSON.stringify(m).length, "small");
        fetching.manifests.delete(fp);
        return m;
      })
      .catch((e) => {
        fetching.manifests.delete(fp);
        throw e;
      });
    fetching.manifests.set(fp, asking);
  }
  return asking;
}

/** 一个源中无需网络即可获得的帧：本机文件路径始终全部计入，服务器路径取决于字节是否在缓存中。
 *
 * 本机原件（本标签页中刚选择的、已授权的本机目录中的文件，由 `transfer/originals.ts`
 * 判断）直接从磁盘读取，不经过服务器。若判断某帧是否可用时只看缓存中是否有已解码的位图，位图被预算
 * 淘汰后即判为不可用，播放将停下来等待一个本就在磁盘上的帧，循环回到第一帧时表现为卡死。 */
function loadedFrames(id: string): number[] {
  // 「在浏览器中」指字节存在，而非已解码的位图尚未被淘汰。位图体积大、会被预算淘汰；字节体积小、整段保留；
  // 字节存在即无需走网络，数毫秒即可解码绘制。时间线上的「已载入视图」表示的也是这一含义。
  const out = new Set<number>();
  const add = (key: string, from: number) => {
    const n = Number(key.slice(from));
    if (Number.isInteger(n)) out.add(n);
  };
  // 压缩字节按 id 的前半段存储（`transfer/sources.ts packetPart`），即「数据包 · 代理档位」的字节，
  // 与该帧是否从磁盘读取无关。缺少此项时，一个包只要有本机原件，此行报告的帧就会不完整
  const bytesAt = `bytes:${packetPart(id)}:`;
  for (const key of cache.keysUnder(id + ":", bytesAt)) add(key, key.startsWith(bytesAt) ? bytesAt.length : id.length + 1);
  for (const f of localFramesOf(id)) out.add(f);  // 本机文件：位于磁盘上，随时可读
  return [...out].sort((a, b) => a - b);
}

/** 一个源中已解码、可立即绘制的帧（位图或通道平面在缓存中）：播放器等待的是此项，而非 `loadedFrames`
 * （后者还包括字节及磁盘上的原件，它们须先解码才能绘制）。 */
function decodedFrames(id: string): number[] {
  const out: number[] = [];
  for (const key of cache.keysUnder(id + ":")) {
    const n = Number(key.slice(id.length + 1));
    if (Number.isInteger(n)) out.push(n);
  }
  return out.sort((a, b) => a - b);
}

/** 某个源的帧清单：缓存每次变化时重读，但只在清单内容变了时交给 React 一个新值（才重绘）。其他源的帧到达、
 * 别处的条目被释放，都不会重绘读它的视图（Stage2D 每次缓存变化都重绘时，播放中每一帧都要重画整个舞台）。 */
function useCachedFrames(id: string | null, read: (id: string) => number[]): number[] | null {
  const last = useRef<{ id: string | null; version: number; text: string; value: number[] | null } | null>(null);
  return useSyncExternalStore(
    (f) => cache.onChange(f),
    () => {
      const version = cache.changes();
      const was = last.current;
      if (was && was.id === id && was.version === version) return was.value;
      const value = id ? read(id) : null;
      const text = value ? value.join() : "";
      last.current = was && was.id === id && was.text === text ? { ...was, version } : { id, version, text, value };
      return last.current.value;
    },
  );
}

export const useDecodedFrames = (id: string | null): number[] | null => useCachedFrames(id, decodedFrames);

/** The frames of a source present in the browser, kept up to date as they arrive (the timeline's 已载入视图 row). */
export const useLoadedFrames = (id: string | null): number[] | null => useCachedFrames(id, loadedFrames);

/** 播放即将到达的帧当前是否有请求在读取（任何源均计入：视图自身的窗口即为取帧清单）。
 *
 * 时间线据此决定等待还是前进：按 DCC 的规则，下一帧尚未到达浏览器时停在原处等待（第一遍因此慢于
 * 实时，但连续且完整）。但等待一个无人读取的帧即为死等（例如循环播放到最后一帧时停住）。
 * 取帧顺序本身会绕回开头（frameWindow.ts order），此处为第二道保护：无人读取时不等待，照常前进。 */
export function onItsWay(frame: number): boolean {
  // 获取失败、正在退避的帧不算在途：退避期间无人读取，等待即为死等；播放器跳过该帧，
  // 到期后会自动重试（见上方 pump 的 catch）
  for (const w of windows.values()) if (w.keys.some((k) => k.endsWith(`:${frame}`) && !stillFailed(asked.get(k)))) return true;
  return false;
}

/** The frames a source keeps around `frame`, most wanted first: the current one, then frames ahead in the direction of
 * play as far as its share of the byte budget allows, and a few behind (frameWindow.ts). A single computation, so the
 * frames the view fetches and the frames it draws are the same. */
export function windowFrames(source: FrameSource, frame: number, dir: number): number[] {
  // 单帧解码后的大小：取到达时记录的值（`hold`）；尚无记录时（切回时所有帧都已在缓存中，无需再取）才扫描缓存找一张并记录
  let big = fetching.bigOf.get(source.id);
  if (big === undefined) {
    const one = picture(cache.keysUnder(source.id + ":")[0] ?? "");
    big = one ? sizeOf(one) : 0;
    if (one) fetching.bigOf.set(source.id, big);
  }
  const others = [...windows.keys()].filter((id) => id !== source.id).length;
  return order(source.frames, frame, dir || 1, aheadFor(big, PIXELS_BUDGET, others + 1));
}

/** Starts loading a source's window from `frame` outward; the returned function removes the window again (frames no
 * window wants any more stop loading). The single way a view or a player requests frames (useFrames). */
function watchFrames(source: FrameSource, frame: number, dir: number): () => void {
  const list = windowFrames(source, frame, dir).flatMap((f) => {
    const load = source.load(f);
    return load ? [{ key: `${source.id}:${f}`, load }] : [];
  });
  want(source.id, list);
  // 查看即取回整段：选择某个通道即开始取回，不等待用户播放。
  // 上一句负责当前需要绘制的帧（解码，有预算），此句负责取回字节（不解码）
  fillWhole(source, frame, dir);
  const keys = windows.get(source.id)!.keys;
  return () => release(source.id, keys);
}

interface Shown {
  image: Pixels | null; // what to draw: the frame, or the last drawn frame while it loads
  frame: number | null; // the frame that `image` shows
  loading: boolean; // the requested frame has not arrived yet
  failed: boolean;
  window: number[]; // the frames this source keeps around the one shown, most wanted first: the frames the view draws
  // next (the look is computed every frame; no processed picture is stored)
}

/** 同时跟踪多个源：一侧要绘制的内容可能是一张显示图（图片路径），也可能是一到三条通道加一条
 * 「有效像素」通道（按通道取数路径，transfer/plane.ts）。它们各自是一个源，但必须共用同一个窗口、
 * 同一个预算、同一次取消，否则播放时部分在传输、部分被丢弃，各帧画面将不一致。
 *
 * 数组长度每次可以不同（以 `null` 占位表示本轮不需要该格）。返回顺序与传入顺序一一对应。
 * 取帧规则与单个源相同：未到达时保留该源上一张已到达的图（标记 loading），从不绘制黑帧。 */
export function useFrames(sources: (FrameSource | null)[], frame: number, dir = 1, hold = false): Shown[] {
  const [, redraw] = useState(0);
  const last = useRef(new Map<string, { image: Pixels; frame: number }>());
  const ids = sources.map((x) => x?.id ?? "").join("|");
  const keys = sources.map((x) => (x ? `${x.id}:${frame}` : ""));
  // 源对象更换时即使 id 完全相同也须重新挂载：查找原件的请求以「没有」结束时（`view/useOriginals.ts`），
  // `plan.sourceKey` 改变、源被重建，但本机没有该文件，id 中的 local 段仍为空；id 不变则下方 effect 不会重新执行，
  // `fillWhole` 不会再次调用，被「查找中」阻止的整段取回将无法再启动（播放头停止时没有其他因素会触发它）。因此需统计源对象的更换次数
  const seen = useRef<{ list: (FrameSource | null)[]; gen: number }>({ list: [], gen: 0 });
  if (seen.current.list.length !== sources.length || sources.some((s, i) => s !== seen.current.list[i])) seen.current = { list: sources, gen: seen.current.gen + 1 };
  const gen = seen.current.gen;
  useEffect(() => {
    const stops: (() => void)[] = [];
    const f = () => redraw((n) => n + 1);
    let came = false;
    for (const source of sources) {
      if (!source) continue;
      // `hold`（拖动时间线期间）：不拉取任何帧，拖过的帧大多只是经过，逐帧拉取会浪费流量
      // （三维的 Stage3D 做法相同）。已解码的帧照常绘制；松开后 hold 变为 false，此 effect 再次执行，拉取停止处的帧
      if (!hold) stops.push(watchFrames(source, frame, dir));
      const key = `${source.id}:${frame}`;
      const e = asked.get(key);
      if (!e) continue;
      if (cache.has(key) || stillFailed(e)) came = true; // it arrived between the render and now: drawn immediately
      else {
        e.waiters.add(f);
        stops.push(() => e.waiters.delete(f));
      }
    }
    if (came) f();
    return () => stops.forEach((stop) => stop());
  }, [ids, gen, keys.join("|"), frame, dir, hold]); // eslint-disable-line react-hooks/exhaustive-deps
  return sources.map((source, i) => {
    if (!source) return { image: null, frame: null, loading: false, failed: false, window: [] };
    const key = keys[i];
    const keeping = windowFrames(source, frame, dir);
    const failed = stillFailed(asked.get(key));
    const image = picture(key);
    if (image) {
      last.current.set(source.id, { image, frame });
      return { image, frame, loading: false, failed: false, window: keeping };
    }
    const was = last.current.get(source.id);
    const prev = was && picture(`${source.id}:${was.frame}`) === was.image ? was : null;
    return { image: prev?.image ?? null, frame: prev?.frame ?? null, loading: !failed && !!source.load(frame), failed, window: keeping };
  });
}

/** The picture of `frame` from `source`, with the surrounding frames loading (as far ahead as the source's share of
 * the budget allows). While it loads, the last picture drawn from the same source stays (marked
 * loading). 单个源的版本，即只跟踪一个源的 `useFrames`。 */
export function useFrame(source: FrameSource | null, frame: number, dir = 1): Shown {
  return useFrames([source], frame, dir)[0];
}

/** 「加载中 1005」 over the picture's corner (or its centre when nothing is drawn yet): the requested frame is in
 * transit; the picture beneath it is the last one that arrived. */
export function drawLoading(ctx: CanvasRenderingContext2D, at: { x: number; y: number; s: number }, width: number, height: number, frame: number, nothing: boolean): void {
  const text = `加载中 ${frame}`;
  ctx.save();
  ctx.font = "12px system-ui, sans-serif";
  const w = ctx.measureText(text).width + 20;
  const x = nothing ? at.x + (width * at.s - w) / 2 : at.x + width * at.s - w - 10;
  const y = nothing ? at.y + (height * at.s) / 2 - 12 : at.y + 10;
  ctx.fillStyle = "rgba(20,22,26,0.78)";
  ctx.beginPath();
  ctx.roundRect(x, y, w, 24, 12);
  ctx.fill();
  ctx.fillStyle = "#cfd3da";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 10, y + 12);
  ctx.restore();
}

/** 该指纹已重算（生成号改变，transfer/gens.ts）：其包说明失效，下次重新向服务器请求；旧帧的键含旧生成号，因此不会再命中。 */
export function forgetManifest(fp: string): void {
  cache.forget(`manifest:${fp}`);
  fetching.manifests.delete(fp);
}

/** Forgets everything about a source (a running node's provisional frames once it is done, view/partial.ts): its
 * pictures, its compressed bytes (kept under bytes:<id>:, transfer/sources.ts bytesKey) and what was asked. */
export function dropSource(id: string): void {
  cache.forgetAll(id + ":");
  cache.forgetAll(bytesKey(id + ":"));
  for (const key of [...asked.keys()]) if (key.startsWith(id + ":")) asked.delete(key);
  abortUnder(id + ":");
}
