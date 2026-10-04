import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { Manifest } from "../api";
import { aheadFor, order } from "./frameWindow";
import { PIXELS_BUDGET, cache } from "../platform/cache";
import { freePixels, sizeOf, type Pixels, type Plane } from "./plane";
import { abortUnder, localFramesOf } from "./sources";
import { failed, Later, mayAsk, succeeded, waitLeft, whenAskable } from "./frameStore";
import { bytesGroup, bytesKey, frameIn, frameKey } from "./frameKey";
import { describedHeld, describedOf } from "./described";
import type { FrameSource } from "./sources";
import { fillWhole } from "./fill";
import { readyAcross } from "./readiness";
import { t } from "../i18n/t";

// 单帧的获取方式位于 transfer/sources.ts（下一层）：本模块只负责账本，即当前应取哪些帧、
// 取到何时、先后顺序及何时取消。取数层的导出在此原样转出，调用方无需了解两层的分界。
export * from "./sources";
export { channelId, tierOf, type Tier } from "./frameKey";
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
 * 图像存放于页面唯一的缓存（platform/cache.ts），键为 `${源}:${帧}`；本模块只记录在途请求，不存储像素。 */

const PARALLEL = 6; // 同时加载的帧数（走网络的在 transfer/frameStore.ts 排队等连接，正在画的先走）

/** 正在取什么、谁在等它；不含像素（像素归缓存）。取失败由 transfer/frameStore.ts（`mayAsk` / `waitLeft` / `whenAskable`）
 * 记住，那是页面唯一的失败记忆：账本这里不另记，只用 `askable` / `waitOf`（下面）按这一帧涉及的键去问它。 */
interface Asked {
  key: string;
  id: string; // 它所属的源：它的图存在这个组下
  load: () => Promise<Pixels>;
  waiters: Set<() => void>;
}

/** 此刻在取什么、为谁取、还在等 worker 回什么：本模块的状态集中在这一处。
 * `windows`（每个 watch 一个）/ `loading` 跟着打开的窗口走，每个回复到达即清；`asked` 在窗口挪走后仍留着一帧的失败记录，
 * 所以改由 `trimAsked` 限量。图归缓存。 */
class Fetching {
  readonly asked = new Map<string, Asked>(); // 取帧的工作，按键记（有上限：ASKED_MAX）
  readonly stills = new Map<string, readonly number[]>(); // 静止图的源 id -> 它代表的帧（只存一份，在第 0 槽：slotOf）
  readonly bigOf = new Map<string, number>(); // 源 id -> 它一帧解码后的字节数（只在它有窗口时）
  readonly windowOf = new WeakMap<FrameSource, { frame: number; dir: number; others: number; big: number | undefined; frames: number[] }>(); // windowFrames 的上一次结果
  // 每个 watch（watchFrames）一个窗口：两个视图看同一个源时各留各的（互不覆盖对方的当前帧）；所有窗口的帧都要
  readonly windows = new Map<object, { id: string; keys: string[]; at: number }>();
  readonly loading = new Set<string>(); // 在途的键
  clock = 0;
}

const fetching = new Fetching();
const asked = fetching.asked;
const windows = fetching.windows;
const loading = fetching.loading;

const wantedKeys = (): Set<string> => new Set([...windows.values()].flatMap((w) => w.keys));
cache.pin(wantedKeys); // 窗口要的就是正在画的：缓存永不释放它们

const picture = (key: string): Pixels | undefined => cache.get<Pixels>(key);
/** 账本键 -> 取它的字节用的那把键，两者不同时（带本机原件的源、合并源：账本键是 `源id:帧`，字节键是包的）。取字节的
 * 一层回 Later 时得知。字节键只是这一帧的另一个名字：它的退避就是这一帧的退避。 */
const bytesOf = new Map<string, string>();
const namesOf = (key: string): string[] => { const b = bytesOf.get(key); return b && b !== key ? [key, b] : [key]; };
/** 这一帧现在能不能取：它自己（解码等取到之后的失败按账本键记）与它的字节键都不在退避、不在等登录。退避、在途、缺帧
 * 只按这一处判。 */
const askable = (key: string): boolean => namesOf(key).every(mayAsk);
/** 这一帧还要等多久才能再取（毫秒）：同上，两个名字里久的那个。 */
const waitOf = (key: string): number => Math.max(...namesOf(key).map(waitLeft));
/** 这一帧真的在路上：正在取，或登记了、可以取（不在退避里）。「加载中」只看这一处。 */
const coming = (key: string): boolean => loading.has(key) || (asked.has(key) && askable(key));
const hold = (e: Asked, px: Pixels, near: boolean) => {
  cache.keep({ key: e.key, group: e.id }, px, sizeOf(px), near ? "near" : "viewing", { free: freePixels });
  fetching.bigOf.set(e.id, sizeOf(px)); // 计算窗口宽度时需要（见下方 windowFrames），避免每次渲染都扫描缓存
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
    if (!wanted.has(key) && !loading.has(key)) (asked.delete(key), bytesOf.delete(key));
  }
}

function pump(): void {
  // 最近请求的窗口先走；窗口内按它自己的顺序
  const queue = [...windows.values()].sort((a, b) => b.at - a.at).flatMap((w) => w.keys);
  for (const key of queue) {
    if (loading.size >= PARALLEL) return;
    const e = asked.get(key);
    if (!e || cache.has(key) || !askable(key) || loading.has(key)) continue;
    loading.add(key);
    void e
      .load()
      .then((img) => {
        // 正在画的这一帧（某个窗口的第一格）记 viewing，窗口里其余的记 near：超预算时先淘汰 near
        hold(e, img, ![...windows.values()].some((w) => w.keys[0] === key)); // 只存真正到达的帧
        succeeded(key);
      })
      .catch((err) => {
        // 取字节的那一层说这一帧的字节键还在退避（Later）：记下这一帧的字节键（bytesOf），此后 askable 按它判，在它能再要
        // 之前不再取（否则每次 pump 都立刻又被拒，主线程空转到退避期满），舞台按 waitOf 标出这一帧取不到；到期时再 pump
        if (err instanceof Later) {
          bytesOf.set(key, err.key);
          whenAskable(err.key, () => {
            if (asked.get(key) === e && !cache.has(key)) pump();
          });
          return;
        }
        // 记下失败（frameStore 取字节时已记的不重复计数）；退避期间 pump 跳过它。到期时窗口若没变就没人调 pump
        // （播放器停住等它），所以到期后主动 pump 一次；那时它已不在账本里就不管
        failed(key, err);
        whenAskable(key, () => {
          if (asked.get(key) === e && !cache.has(key)) pump();
        });
      })
      .finally(() => {
        loading.delete(key);
        e.waiters.forEach((f) => f());
        // 接着取下一帧放到宏任务里：一连串立刻失败的也不会在微任务里连着转、饿死事件循环
        setTimeout(pump, 0);
      });
  }
}

/** 一个源此刻要的帧：`order`，从当前帧向外；不再有窗口要的帧停止加载。
 * 预取已取到的帧就在缓存里同一个键下：原样拿来画，从预取档升上来，不再重取。 */
function want(watch: object, id: string, list: { key: string; load: () => Promise<Pixels> }[]): void {
  windows.set(watch, { id, keys: list.map((w) => w.key), at: ++fetching.clock });
  cache.repin();
  list.forEach((w, i) => {
    if (cache.has(w.key)) cache.reclass(w.key, i === 0 ? "viewing" : "near");
    else if (!asked.has(w.key)) asked.set(w.key, { ...w, id, waiters: new Set() });
  });
  trimAsked();
  pump();
}

/** 一个 watch 不再要帧（它的视图关了，或挪到了别的帧）：它的窗口立即撤掉，在途的帧不中止（到达后照常存下）。 */
function release(watch: object): void {
  const w = windows.get(watch);
  if (!w) return;
  windows.delete(watch);
  cache.repin();
  const still = [...windows.values()].some((o) => o.id === w.id); // 别的视图还在看同一个源
  if (!still) {
    fetching.bigOf.delete(w.id);
    fetching.stills.delete(w.id);
  }
  // 切换离开的帧保留并降一档（别的窗口还要的不降）：预算未超时一直保留，切回时可直接使用；
  // 超出预算时先于当前查看的帧被淘汰（`near` 排在 `viewing` 之前，同档内最久未使用的先淘汰）
  const wanted = wantedKeys();
  for (const key of w.keys) if (!wanted.has(key) && cache.has(key)) cache.reclass(key, "near");
}

// ------------------------------------------------------------------ 包说明

/** 该包的说明当前是否已持有（同步，不发请求）：取数层据此同步判断本次查看的路径（`transfer/prefetchLive.ts keyOf`）。 */
export const manifestNow = (fp: string): Manifest | null => describedHeld<Manifest>("manifest", fp);

/** 一个包当前代次的说明（它的 meta：帧、尺寸、帧率；transfer/described.ts）：二维舞台、时间线和预取读的是同一份。
 * 组件用 `useDescribed("manifest", …)` 读。 */
export const manifestOf = (fp: string): Promise<Manifest> => describedOf<Manifest>("manifest", fp);

/** 一个源中无需网络即可获得的帧：本机文件路径始终全部计入，服务器路径取决于字节是否在缓存中。
 *
 * 本机原件（本标签页中刚选择的、已授权的本机目录中的文件，由 `transfer/originals.ts`
 * 判断）直接从磁盘读取，不经过服务器。若判断某帧是否可用时只看缓存中是否有已解码的位图，位图被预算
 * 淘汰后即判为不可用，播放将停下来等待一个本就在磁盘上的帧，循环回到第一帧时表现为卡死。 */
function loadedFrames(id: string): number[] {
  // 「在浏览器中」指字节存在，而非已解码的位图尚未被淘汰。位图体积大、会被预算淘汰；字节体积小、整段保留；
  // 字节存在即无需走网络，数毫秒即可解码绘制。时间线上的「已载入视图」表示的也是这一含义。
  const still = fetching.stills.get(id);
  if (still) return [...still]; // 静止图是本机文件：随时可读
  const out = new Set<number>();
  const add = (key: string, group: string) => {
    const n = frameIn(group, key);
    if (Number.isInteger(n)) out.add(n);
  };
  // 压缩字节按 id 的前半段存储（`transfer/frameKey.ts bytesGroup`），即「数据包 · 代理档位」的字节，
  // 与该帧是否从磁盘读取无关。缺少此项时，一个包只要有本机原件，此行报告的帧就会不完整
  const bytes = bytesGroup(id);
  for (const key of cache.keysIn(id)) add(key, id);
  for (const key of cache.keysIn(bytes)) add(key, bytes);
  for (const f of localFramesOf(id)) out.add(f);  // 本机文件：位于磁盘上，随时可读
  return [...out].sort((a, b) => a - b);
}

/** 一个源中已解码、可立即绘制的帧（位图或通道平面在缓存中）：播放器等待的是此项，而非 `loadedFrames`
 * （后者还包括字节及磁盘上的原件，它们须先解码才能绘制）。 */
function decodedFrames(id: string): number[] {
  const still = fetching.stills.get(id);
  if (still) return cache.has(frameKey(id, 0)) ? [...still] : [];
  const out: number[] = [];
  for (const key of cache.keysIn(id)) {
    const n = frameIn(id, key);
    if (Number.isInteger(n)) out.push(n);
  }
  return out.sort((a, b) => a - b);
}

/** 某个源的帧清单：缓存每次变化时重读，但只在清单内容变了时交给 React 一个新值（才重绘）。其他源的帧到达、
 * 别处的条目被释放，都不会重绘读它的视图（Stage2D 每次缓存变化都重绘时，播放中每一帧都要重画整个舞台）。 */
function useCachedFrames(id: string | null, read: (id: string) => number[], of: readonly string[], withBytes: boolean, also: readonly string[] = []): number[] | null {
  const last = useRef<{ id: string | null; version: number; text: string; value: number[] | null } | null>(null);
  // 只订阅这些源自己的组（解码后的帧；读字节层的还有它们的压缩字节）与 `also`：别的源的帧到了、别处的条目被释放，不叫它
  const groups = [...of.flatMap((one) => (withBytes ? [one, bytesGroup(one)] : [one])), ...(of.length ? also : [])];
  const joined = groups.join("\n");
  return useSyncExternalStore(
    useCallback((f: () => void) => cache.watch(joined ? joined.split("\n") : [], f), [joined]),
    () => {
      const version = cache.changesIn(groups);
      const was = last.current;
      if (was && was.id === id && was.version === version) return was.value;
      const value = id ? read(id) : null;
      const text = value ? value.join() : "";
      last.current = was && was.id === id && was.text === text ? { ...was, version } : { id, version, text, value };
      return last.current.value;
    },
  );
}

/** 这几个源都已解码、可立即画的帧（交集）：一侧用了几格（图片、各条通道），播放器等的是每一格都到了。 */
export function useDecodedFrames(sources: readonly FrameSource[]): number[] | null {
  const ids = sources.map((s) => s.id);
  // 每格按自己的帧（transfer/readiness.ts readyAcross）：某格没有的帧不因它而永远「不可画」
  const has = new Map(sources.map((s) => [s.id, s.still ? null : s.frames]));
  return useCachedFrames(ids.length ? ids.join("\n") : null,
    (joined) => readyAcross(joined.split("\n").map((id) => ({ has: has.get(id) ?? null, ready: decodedFrames(id) }))) ?? [], ids, false);
}

/** 一个源已在浏览器里的帧，随到达随时更新（时间线的「已载入视图」一行）。
 * 本机帧是否就绪看本机代理的登记（组 proxy）和 EXR 的显示变换表（组 lut）：它们变了也要重读。 */
export const useLoadedFrames = (id: string | null): number[] | null => useCachedFrames(id, loadedFrames, id ? [id] : [], true, ["proxy", "lut"]);

/** 播放即将到达的帧当前是否有请求在读取（任何源均计入：视图自身的窗口即为取帧清单）。
 *
 * 时间线据此决定等待还是前进：按 DCC 的规则，下一帧尚未到达浏览器时停在原处等待（第一遍因此慢于
 * 实时，但连续且完整）。但等待一个无人读取的帧即为死等（例如循环播放到最后一帧时停住）。
 * 取帧顺序本身会绕回开头（frameWindow.ts order），此处为第二道保护：无人读取时不等待，照常前进。 */
export function onItsWay(frame: number): boolean {
  // 获取失败、正在退避的帧不算在途：退避期间无人读取，等待即为死等；播放器跳过该帧，
  // 到期后会自动重试（见上方 pump 的 catch）
  // 已到的（缓存里有）不算在途：在途只是真的还没到、有人会取的（与 useFrames 的 coming 同一个意思）
  for (const w of windows.values()) if (w.keys.some((k) => frameIn(w.id, k) === frame && !cache.has(k) && askable(k))) return true;
  return false;
}

/** 一帧在账本里占的槽：静止图（FrameSource.still）任何一帧都是同一个槽 0，只取、只存、只解码一次；序列就是帧号。 */
const slotOf = (source: FrameSource, frame: number): number => (source.still ? 0 : frame);

/** 这张图是否仍是缓存里这个源这一帧的那一张（被淘汰的位图已 close，不能再画）。 */
export const stillHeld = (source: FrameSource, frame: number, image: Pixels): boolean =>
  picture(frameKey(source.id, slotOf(source, frame))) === image;

/** 一个源在 `frame` 周围保留的帧，最想要的在前：当前帧，然后沿播放方向往前、直到它分得的字节预算用完，再加身后几帧
 * （frameWindow.ts）。只算这一处，所以视图取的帧与它画的帧是同一组。 */
export function windowFrames(source: FrameSource, frame: number, dir: number): number[] {
  const others = new Set([...windows.values()].map((w) => w.id).filter((id) => id !== source.id)).size;
  const was = fetching.windowOf.get(source);
  if (was && was.frame === frame && was.dir === dir && was.others === others && was.big === fetching.bigOf.get(source.id)) return was.frames;
  // 单帧解码后的大小：取到达时记录的值（`hold`）；尚无记录时（切回时所有帧都已在缓存中，无需再取）才扫描缓存找一张并记录
  let big = fetching.bigOf.get(source.id);
  if (big === undefined) {
    const one = picture(cache.keysIn(source.id)[0] ?? "");
    big = one ? sizeOf(one) : 0;
    if (one) fetching.bigOf.set(source.id, big);
  }
  const frames = order(source.frames, frame, dir || 1, aheadFor(big, PIXELS_BUDGET, others + 1));
  // 每次渲染、每格都要问一次：同一源同一帧同一方向（且预算条件没变）直接给上次的
  fetching.windowOf.set(source, { frame, dir, others, big: fetching.bigOf.get(source.id), frames });
  return frames;
}

/** 从 `frame` 向外开始加载一个源的窗口；返回的函数再把窗口撤掉（不再有窗口要的帧停止加载）。视图或播放器要帧只有这一条路
 * （useFrames）。 */
function watchFrames(source: FrameSource, frame: number, dir: number): () => void {
  if (source.still) fetching.stills.set(source.id, source.frames);
  const list = source.still
    ? (() => { const load = source.load(frame); return load ? [{ key: frameKey(source.id, slotOf(source, frame)), load }] : []; })()
    : windowFrames(source, frame, dir).flatMap((f) => {
      const load = source.load(f);
      return load ? [{ key: frameKey(source.id, f), load }] : [];
    });
  const watch = {};
  want(watch, source.id, list);
  // 查看即取回整段：选择某个通道即开始取回，不等待用户播放。
  // 上一句负责当前需要绘制的帧（解码，有预算），此句负责取回字节（不解码）
  const stopFill = fillWhole(source, frame, dir);
  return () => (release(watch), stopFill());
}

interface Shown {
  image: Pixels | null; // 要画的：这一帧，或它加载期间上一张画过的
  frame: number | null; // `image` 画的是哪一帧
  loading: boolean; // 要的这一帧在路上（取帧账本里在途或排着队：coming）
  absent: boolean; // 这个源本来就没有这一帧（结果只算了部分帧）：不是「还没到」
  failed: boolean;
  window: number[]; // 这个源在显示帧周围保留的帧，最想要的在前：视图接下来要画的帧（外观每帧现算，不存处理后的图）
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
  const keys = sources.map((x) => (x ? frameKey(x.id, slotOf(x, frame)) : ""));
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
      // `hold`（拖动时间线期间）：不拉取任何帧，拖过的帧大多只是经过，逐帧拉取会浪费流量（二维各格、三维背板同用）。
      // 已解码的帧照常绘制；松开后 hold 变为 false，此 effect 再次执行，拉取停止处的帧
      if (!hold) stops.push(watchFrames(source, frame, dir));
      const key = frameKey(source.id, slotOf(source, frame));
      const e = asked.get(key);
      if (!e) continue;
      if (cache.has(key) || !askable(key)) came = true; // 在渲染与此刻之间到了（或取不到了）：立即画
      else {
        e.waiters.add(f);
        stops.push(() => e.waiters.delete(f));
      }
    }
    if (came) f();
    return () => stops.forEach((stop) => stop());
  }, [ids, gen, keys.join("|"), frame, dir, hold]); // eslint-disable-line react-hooks/exhaustive-deps
  return sources.map((source, i) => {
    if (!source) return { image: null, frame: null, loading: false, absent: false, failed: false, window: [] };
    const key = keys[i];
    const keeping = windowFrames(source, frame, dir);
    const failed = waitOf(key) > 0;
    const image = picture(key);
    if (image) {
      last.current.set(source.id, { image, frame });
      return { image, frame, loading: false, absent: false, failed: false, window: keeping };
    }
    const was = last.current.get(source.id);
    const prev = was && stillHeld(source, was.frame, was.image) ? was : null;
    const has = !!source.load(frame);
    return { image: prev?.image ?? null, frame: prev?.frame ?? null, loading: !failed && has && coming(key), absent: !has, failed, window: keeping };
  });
}

/** `source` 里 `frame` 这一帧的图，周围的帧同时加载（往前加载到该源分得的预算为止）。加载期间留着同一个源上一张画过的图
 * （标记 loading）。单个源的版本，即只跟踪一个源的 `useFrames`。 */
export function useFrame(source: FrameSource | null, frame: number, dir = 1, hold = false): Shown {
  return useFrames([source], frame, dir, hold)[0];
}

/** 画面角上（还什么都没画时在中央）的「加载中 1005」：要的这一帧还在路上，底下的图是上一张到达的。 */
export function drawLoading(ctx: CanvasRenderingContext2D, at: { x: number; y: number; s: number }, width: number, height: number, frame: number, nothing: boolean): void {
  const text = t("ui.upload.loading_frame", { frame });
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

/** 忘掉一个源的一切（运行中节点的临时帧，在它算完时，view/partial.ts）：它的图、它的压缩字节（存在 `bytesKey(id)` 下，
 * transfer/frameKey.ts）以及账本里的请求。 */
export function dropSource(id: string): void {
  cache.forgetGroup(id);
  cache.forgetGroup(bytesKey(id));
  for (const key of [...asked.keys()]) if (key.startsWith(id + ":")) (asked.delete(key), bytesOf.delete(key));
  abortUnder(id + ":");
}
