import { useEffect, useMemo, useState } from "react";
import { NeedLogin, afterLogin, notAnError, sceneDescription, whenAskable, type Lane } from "../transfer/frameStore";
import { backoff } from "../platform/backoff";
import { useGens } from "../transfer/gens";
import { partialBase, partCacheKey, partialPointsKey, partialPointsUrl, pointsKey, pointsUrl, sceneCacheKey, sceneDescKey, sceneKey, sceneSlot, sceneUrl } from "../transfer/frameKey";
import { readDescription } from "../model/viewFormat";
import { cache } from "../platform/cache";
import { fetchPart, store } from "./sceneStore";

export type { CameraData, CharacterData, CharacterMeshData, CloudData, CloudSample, CurveData, CurveSample, GridSample, ModelData } from "./sceneTypes";

/** 三维舞台所画内容的载入与保留，数据按服务器所发（lab2shot/server/view_data.py，由 viewFormat.ts 解码）：按 DCC 的方式
 * 组织的三维数据（蒙皮角色以其蒙皮表示，静止物体只发一份），且精确（每个点，float32）。一份视图先载入基础数据
 * （所有不逐帧变化的内容），再载入逐帧样本的块（点缓存、变化的点云）：先载当前帧所在的块，再向两侧展开，只保留放得下的
 * 数量（离当前帧最远的块被释放，回到那一帧时重新请求）。 */

import { Scene } from "./scene";
// `Scene` 本身定义于 view/scene.ts，此处原样转出
export { Scene } from "./scene";

/** 已查看的场景保存在页面唯一的缓存中（platform/cache.ts），键为其地址，大小按基础数据计算。
 *
 * 不另建按个数淘汰的缓存：按个数保留时，两份大场景即占用两份场景份额（SCENE_SHARE）的内存且不受约束；
 * 在多个节点间来回对比时又会丢弃整份场景，描述和基础数据都需重新下载。
 * 它与已解码的画面帧共用解码层的预算和同一套 LRU：小场景可保留十余份，大场景在需要时才被释放。
 *
 * 重新获取一份场景需要一次网络往返加一次服务器重建（视图工作进程需重新读取 USD），是此处开销最大的操作。 */
// 服务器仍在生成的视图，放弃之前最多请求的次数，间隔按 frameStore 的退避（1 秒起翻倍，最多 30 秒：合计约四分钟；
// 服务器自身的生成上限是两分钟，view_worker.py BUILD_S）
const PREPARING_TRIES = 12;

/** 正在显示的场景（它们在缓存里的键 → 显示它的舞台数）：登记进缓存的 pin，显示期间不被淘汰、也不被切后台释放。
 * 不登记的话，显示中的场景在最低保留档（fetched），二维帧一进来它最先被淘汰：舞台还拿着它（样本仍占内存），
 * 预算却不再计它；再次载入同一份又会新建第二个 Scene。 */
const shown = new Map<string, number>();
cache.pin(() => new Set(shown.keys()));

/** 这些场景（kept 的键）正在显示：挂载期间登记，卸载或换了键时撤销。 */
export function useShownScenes(keys: string[]): void {
  const joined = keys.join("\n"); // 键里本身有「|」（点云、边算边看），用换行分隔
  useEffect(() => {
    const at = joined ? joined.split("\n").map(sceneCacheKey) : [];
    for (const k of at) shown.set(k, (shown.get(k) ?? 0) + 1);
    cache.repin();
    return () => {
      for (const k of at) {
        const n = (shown.get(k) ?? 1) - 1;
        if (n > 0) shown.set(k, n);
        else shown.delete(k);
      }
      cache.repin();
    };
  }, [joined]);
}

/** 谁在等一份还没建好的场景（kept 的键 → 中止用的控制器与在等的数）：都不等了（舞台换了、卸了）就中止它的描述请求。
 * 不带 signal 来要的算一直在等。 */
const waiting = new Map<string, { ctrl: AbortController; n: number; loading?: Promise<Scene> }>();

function kept(key: string, load: (signal: AbortSignal) => Promise<Scene>, signal?: AbortSignal): Promise<Scene> {
  const slot = sceneSlot(key);
  const at = slot.key;
  const held = cache.get<Promise<Scene>>(at);
  if (held) return held;
  // 还在建的在在等表里（不进可淘汰的缓存：占位被淘汰后这份场景就再也不计入预算，再要还会建第二份）
  const w = waiting.get(at);
  if (w?.loading) {
    wait(at, w, signal);
    return w.loading;
  }
  const ctrl = new AbortController();
  const mine: { ctrl: AbortController; n: number; loading?: Promise<Scene> } = { ctrl, n: 0 };
  waiting.set(at, mine);
  wait(at, mine, signal);
  const loading = load(ctrl.signal);
  mine.loading = loading;
  // 建好后记在 pixels 层（"fetched"：与画面帧同层、共用较大的预算）：一份场景解码后的样本可达数百 MB，若记录在 "small"
  // （压缩字节层，预算小得多），一份场景即会将二维的整段压缩字节全部挤出
  let kept = false;
  const size = (s: Scene) => {
    if (kept && cache.get<Promise<Scene>>(at) !== loading) return; // 已被淘汰或换了一份：不再记
    kept = true;
    cache.keep(slot, loading, s.bytes(), "fetched");
  };
  const settled = () => waiting.get(at) === mine && waiting.delete(at);
  loading.then(
    (s) => {
      settled();
      size(s);
      // 解码或释放一块时占用的字节随之变化：同步报告给该预算，保留的场景数量才能按字节决定。
      // 若不报告，此处只按基础数据计算大小，便可保留任意多份场景，且每份各自持有整份场景份额（SCENE_SHARE）
      // 的逐帧样本，预算约束随之失效。不作为订阅者登记：订阅者是正在绘制它的视图，失败的块只在有视图时重试
      s.onBytes = () => size(s);
    },
    () => settled(), // 获取失败的不记录：再次查看时重新尝试
  );
  return loading;
}

function wait(at: string, w: { ctrl: AbortController; n: number }, signal?: AbortSignal): void {
  if (!signal) return void (w.n = Infinity);
  w.n++;
  signal.addEventListener("abort", () => {
    if (--w.n > 0 || waiting.get(at) !== w) return;
    // 都不等了：中止，并当场把它从在等表里拿掉——同一轮里马上又来要的（键集合变了、同一份仍在显示）另开一次，
    // 不接上这份注定被中止的
    waiting.delete(at);
    w.ctrl.abort();
  }, { once: true });
}

async function view(key: string, url: string, signal: AbortSignal, lane: Lane): Promise<Scene> {
  // 描述不设 no-store：服务器为其提供 ETag（lab2shot/server/view_data.py respond_description），
  // 因此再次请求同一描述得到的是零字节的 304 响应，而非重新下载数百 KB。
  // 同一标签页内来回切换节点不会执行到此处：场景已在上述缓存中。
  const desc = readDescription(await sceneDescription(key, url, signal, lane));
  const names = Object.keys(desc.parts).filter((n) => !/^c\d+$/.test(n) && n !== "uv");
  const parts = Object.fromEntries(await Promise.all(names.map(async (n) => {
    const bytes = await fetchPart(desc.parts[n].url, "now");
    // 基础数据（固定点云、静止模型、相机……整段只有这一份）同样存到硬盘（view/sceneStore.ts），重看不再下载
    void store(desc.parts[n].url, bytes).catch(() => false);
    return [n, bytes] as const;
  })));
  return new Scene(key, desc, parts);
}

/** 一份三维场景（键与地址都带代次：transfer/frameKey.ts sceneKey / sceneUrl；重算后换键，不拿旧场景）。 */
export const loadScene = (fp: string, signal?: AbortSignal, lane: Lane = "now"): Promise<Scene> => {
  const key = sceneKey(fp);
  return kept(key, (s) => view(key, sceneUrl(fp), s, lane), signal);
};

/** 这些包的三维场景，按指纹给出（Stage3D、kinds3d 用）。键带各包的代次（订阅 useGens）：重算后自动换成新场景。 */
export function useScenes(fps: string[]): [Map<string, Scene>, string[]] {
  const gens = useGens((s) => fps.map((fp) => s.gens[fp] ?? "").join("|"));
  // 键随各包的代次变（gens 是订阅来的值：代次一变这里就重算）
  const keys = useMemo(() => fps.map(sceneKey), [fps.join("|"), gens]); // eslint-disable-line react-hooks/exhaustive-deps
  const [got, errors] = useLoaded(keys.map((key, i) => ({ key, of: fps[i] })), loadScene);
  useShownScenes(keys);
  const byFp = useMemo(() => new Map(keys.flatMap((k, i) => (got.has(k) ? [[fps[i], got.get(k)!] as const] : []))), [got, keys]); // eslint-disable-line react-hooks/exhaustive-deps
  return [byFp, errors];
}

/** 正在计算的节点已写出的帧，作为点云查看（边算边看，server/farm.py partial_points）。
 *
 * 与计算完成的结果使用同一套机制：同一个 Scene、同一条取块路径、同一个缓存、同一套删点策略。
 * 两处差异均在服务器端（view_data.partial_points_view）：有文件的帧为实时扫描，块为每帧一块。 */
const loadPartialPoints = (job: string, node: string, port: string, camera: string | null) => {
  const key = partialPointsKey(job, node, port, camera);
  return kept(key, (s) => view(key, partialPointsUrl(job, node, port, camera), s, "now"));
};

/** 该节点已计算完成：整体释放临时数据（描述、基础数据、已取得的每一帧）。
 *
 * 这些地址包含任务号和节点名而非内容地址，因此不能像计算完成的结果那样长期保留：
 * 同一节点再次计算时，地址相同而数据已更新。与二维的 `dropSource` 为同一操作、同一时机。 */
function forgetPartialPoints(job: string, node: string, port: string, camera: string | null): void {
  cache.forget(sceneCacheKey(partialPointsKey(job, node, port, camera)));
  cache.forgetAll(partCacheKey(`${partialBase(job, node, port)}/points/`));
}

/** 边算边看的点云：该节点仍在计算时已写出的帧。计算完成（或切换节点）后整体释放。 */
export function usePartialPoints(at: { job: string; node: string; port: string; done: number[] } | null, camera: string | null): Scene | null {
  const key = at ? partialPointsKey(at.job, at.node, at.port, camera) : "";
  useShownScenes(key ? [key] : []);
  const [scene, setScene] = useState<Scene | null>(null);
  const job = at?.job ?? "";
  const node = at?.node ?? "";
  const port = at?.port ?? "";
  useEffect(() => {
    if (!key) {
      setScene(null);
      return;
    }
    let alive = true;
    loadPartialPoints(job, node, port, camera).then(
      (s) => alive && setScene(s),
      () => alive && setScene(null), // 尚无可查看的内容（尚未写出任何帧）：三维舞台照常提示尚无结果
    );
    return () => {
      alive = false;
      setScene(null);
      forgetPartialPoints(job, node, port, camera);
    };
  }, [key, job, node, port, camera]);
  // 每次轮询都告知已写出的帧：新写出的帧立即开始获取，未写出的帧不请求
  const done = at?.done ?? null;
  const joined = done?.join() ?? "";
  useEffect(() => scene?.setReady(done), [scene, joined]); // eslint-disable-line react-hooks/exhaustive-deps
  return scene;
}

/** 深度图 / 位置图的点云预览（连同摆放它的相机）；键与地址都带两个包的代次（transfer/frameKey.ts pointsKey / pointsUrl）。 */
export const loadPoints = (fp: string, camera: string | null, signal?: AbortSignal, lane: Lane = "now") => {
  const key = pointsKey(fp, camera);
  return kept(key, (s) => view(key, pointsUrl(fp, camera), s, lane), signal);
};

/** 按一组键载入的值，键不变期间保留；以及失败的那些各自的原因。
 *
 * 失败怎么办按 transfer/frameStore.ts 的三类：等登录的（NeedLogin）不报错，重新登录后重来；服务器还在生成
 * （E-VIEW-PREPARING）不算失败，过几秒再问（走 `later` 道）；其它的报出来，并按连续次数退避后再试（断线恢复、
 * 服务器重启后舞台自己回来，不必换节点）。键变了、舞台卸了就不再等（中止还没回来的描述请求）。
 * 每一项带着键和载入它要的东西（`of`，结构化）：键只用来比对与存放，从不拆回去当参数用。 */
export function useLoaded<A, T>(asks: { key: string; of: A }[], load: (of: A, signal: AbortSignal, lane: Lane) => Promise<T>): [Map<string, T>, string[]] {
  const [got, setGot] = useState<[Map<string, T>, string[]]>([new Map(), []]);
  const keys = asks.map((a) => a.key);
  const joined = keys.join("|");
  useEffect(() => {
    let alive = true;
    const ctrl = new AbortController();
    // 每一份到了就用上（一份仍在生成的大结果不挡其他的）；期间仍在键表中的已有值保留
    setGot(([was]) => [new Map([...was].filter(([k]) => keys.includes(k))), []]);
    const errors = new Map<string, string>();
    const stops: (() => void)[] = [];
    const later = (ms: number, f: () => void) => { const t = setTimeout(() => alive && f(), ms); stops.push(() => clearTimeout(t)); };
    const ofKey = new Map(asks.map((a) => [a.key, a.of]));
    const ask = (k: string, tries: number, lane: Lane) =>
      load(ofKey.get(k)!, ctrl.signal, lane).then(
        (v) => {
          if (!alive) return;
          errors.delete(k);
          setGot(([was]) => [new Map(was).set(k, v), [...errors.values()]]);
        },
        (e) => {
          if (!alive) return;
          // 自己没放弃却收到中止（别的等待方都走了、刚好撞上）：重新要一次
          if (e instanceof DOMException && e.name === "AbortError") return void ask(k, tries, lane);
          if (e instanceof NeedLogin) return void stops.push(afterLogin(() => alive && void ask(k, 0, "now")));
          // 服务器还在生成（server/view_worker.py：单个请求最多等 10 秒，生成的结果会保留），或描述还在退避期：
          // 不是错误，按 frameStore 的退避到期再问（走 later 道）；生成不了时服务器另有提示（E-VIEW-TIMEOUT），那时才是错误
          if (notAnError(e) && tries < PREPARING_TRIES)
            return void stops.push(whenAskable(sceneDescKey(k), () => alive && void ask(k, tries + 1, "later")));
          errors.set(k, e instanceof Error ? e.message : String(e));
          setGot(([was]) => [new Map([...was].filter(([key]) => key !== k)), [...errors.values()]]);
          later(backoff(tries + 1, 1000, 30_000), () => void ask(k, tries + 1, "now"));
        },
      );
    for (const k of keys) void ask(k, 0, "now");
    return () => {
      alive = false;
      ctrl.abort();
      stops.forEach((stop) => stop());
    };
  }, [joined]); // eslint-disable-line react-hooks/exhaustive-deps
  return got;
}

/** 这些视图中任一份收到新样本时重绘。 */
export function useSceneVersions(scenes: Scene[]): number {
  const [, setTick] = useState(0);
  const joined = scenes.map((s) => s.key).join("|");
  useEffect(() => {
    const offs = scenes.map((s) => s.subscribe(() => setTick((t) => t + 1)));
    return () => offs.forEach((off) => off());
  }, [joined]); // eslint-disable-line react-hooks/exhaustive-deps
  return scenes.reduce((a, s) => a + s.version, 0);
}
