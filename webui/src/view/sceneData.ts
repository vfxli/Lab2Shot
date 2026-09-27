import { useEffect, useState } from "react";
import { ApiError, request } from "../platform/http";
import { api } from "../api";
import { genOf } from "../transfer/gens";
import { readDescription } from "../model/viewFormat";
import { cache } from "../transfer/cache";
import { fetchPart, PART_KEY } from "./scenePart";

export type { CameraData, CharacterData, CharacterMeshData, CloudData, CloudSample, CurveData, CurveSample, GridSample, ModelData } from "./sceneTypes";

/** What the 3D stage draws, as sent by the server (lab2shot/server/view_data.py, decoded by viewFormat.ts): 3D data
 * in the DCC manner (a skinned character as its skinning, a static object once), exact (every point, float32). A view
 * loads its base first (everything that is not per frame), then the chunks of per-frame samples (point caches, changing
 * clouds): the chunk of the current frame first, then the rest outward, keeping only as many as fit (the chunks
 * furthest from the current frame are released and requested again when that frame is revisited). */

import { Scene } from "./scene";
// `memoryBytes`（一份场景可保留的逐帧数据字节上限）与 `Scene` 本身定义于 view/scene.ts，
// 此处原样转出：三维画面的显示选项与之属于同一范畴
export { memoryBytes, Scene } from "./scene";

/** 已查看的场景保存在页面唯一的缓存中（transfer/cache.ts），键为其地址，大小按基础数据计算。
 *
 * 不另建按个数淘汰的缓存：按个数保留时，两份大场景即占用两份 `memoryBytes` 的内存且不受约束；
 * 在多个节点间来回对比时又会丢弃整份场景，描述和基础数据都需重新下载。
 * 它与画面帧、包描述共用同一份预算和同一套 LRU：小场景可保留十余份，大场景在需要时才被释放。
 *
 * 重新获取一份场景需要一次网络往返加一次服务器重建（视图工作进程需重新读取 USD），是此处开销最大的操作。 */
const SCENE_KEY = "3d:scene:";
const PREPARING_WAIT_MS = 3000; // interval between requests for a view the server is still building
const PREPARING_TRIES = 60; // number of requests before giving up (the server's own build limit is two minutes: view_worker.py BUILD_S)

function kept(key: string, load: () => Promise<Scene>): Promise<Scene> {
  const at = SCENE_KEY + key;
  const held = cache.get<Promise<Scene>>(at);
  if (held) return held;
  const loading = load();
  // 记录在 pixels 层（"fetched"：与画面帧同层、共用较大的预算）：一份场景解码后的样本可达数百 MB，若记录在 "small"
  // （压缩字节层，预算小得多），一份场景即会将二维的整段压缩字节全部挤出
  const size = (s: Scene) => cache.get<Promise<Scene>>(at) === loading && cache.keep(at, loading, s.bytes(), "fetched");
  cache.keep(at, loading, 0, "small"); // 仍在传输中：同时发起的第二次请求复用同一次传输
  loading.then(
    (s) => {
      size(s);
      // 解码或释放一块时占用的字节随之变化：同步报告给该预算，保留的场景数量才能按字节决定。
      // 若不报告，此处只按基础数据计算大小，便可保留任意多份场景，且每份各自持有整份内存预算（memoryBytes）
      // 的逐帧样本，预算约束随之失效
      s.subscribe(() => size(s));
    },
    () => cache.get<Promise<Scene>>(at) === loading && cache.forget(at), // 获取失败的不记录：再次查看时重新尝试
  );
  return loading;
}

async function view(key: string, url: string): Promise<Scene> {
  // 描述不设 no-store：服务器为其提供 ETag（lab2shot/server/view_data.py respond_description），
  // 因此再次请求同一描述得到的是零字节的 304 响应，而非重新下载数百 KB。
  // 同一标签页内来回切换节点不会执行到此处：场景已在上述缓存中。
  const r = await request(url);
  const desc = readDescription(await r.json());
  const names = Object.keys(desc.parts).filter((n) => !/^c\d+$/.test(n) && n !== "uv");
  const parts = Object.fromEntries(await Promise.all(names.map(async (n) => [n, await fetchPart(desc.parts[n].url)] as const)));
  return new Scene(key, desc, parts);
}

export const loadScene = (fp: string) => kept(fp, () => view(fp, api.sceneUrl(fp, genOf(fp))));

/** 正在计算的节点已写出的帧，作为点云查看（边算边看，server/farm.py partial_points）。
 *
 * 与计算完成的结果使用同一套机制：同一个 Scene、同一条取块路径、同一个缓存、同一套删点策略。
 * 两处差异均在服务器端（view_data.partial_points_view）：有文件的帧为实时扫描，块为每帧一块。 */
export const partialPointsKey = (job: string, node: string, port: string, camera: string | null) =>
  `partial|${job}|${node}|${port}|${camera ?? ""}`;

const partialPointsUrl = (job: string, node: string, port: string, camera: string | null) =>
  `/api/jobs/${job}/partial/${encodeURIComponent(node)}/${encodeURIComponent(port)}/points${camera ? `?camera=${camera}` : ""}`;

export const loadPartialPoints = (job: string, node: string, port: string, camera: string | null) => {
  const key = partialPointsKey(job, node, port, camera);
  return kept(key, () => view(key, partialPointsUrl(job, node, port, camera)));
};

/** 该节点已计算完成：整体释放临时数据（描述、基础数据、已取得的每一帧）。
 *
 * 这些地址包含任务号和节点名而非内容地址，因此不能像计算完成的结果那样长期保留：
 * 同一节点再次计算时，地址相同而数据已更新。与二维的 `dropSource` 为同一操作、同一时机。 */
export function forgetPartialPoints(job: string, node: string, port: string, camera: string | null): void {
  cache.forget(SCENE_KEY + partialPointsKey(job, node, port, camera));
  cache.forgetAll(PART_KEY + `/api/jobs/${job}/partial/${encodeURIComponent(node)}/${encodeURIComponent(port)}/points/`);
}

/** 边算边看的点云：该节点仍在计算时已写出的帧。计算完成（或切换节点）后整体释放。 */
export function usePartialPoints(at: { job: string; node: string; port: string; done: number[] } | null, camera: string | null): Scene | null {
  const key = at ? partialPointsKey(at.job, at.node, at.port, camera) : "";
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

/** A depth / position map's point preview (with the camera that places it). */
export const loadPoints = (fp: string, camera: string | null) =>
  kept(`${fp}|${camera ?? ""}`, () => view(`${fp}|${camera ?? ""}`, api.pointsUrl(fp, camera ?? null, genOf(fp))));

/** Values loaded for a list of keys, kept while the keys stay the same; and why the ones that failed did. */
export function useLoaded<T>(keys: string[], load: (key: string) => Promise<T>): [Map<string, T>, string[]] {
  const [got, setGot] = useState<[Map<string, T>, string[]]>([new Map(), []]);
  const joined = keys.join("|");
  useEffect(() => {
    let alive = true;
    // each is applied as it arrives (a large result still being built blocks nothing else); pending entries remain meanwhile
    setGot(([was]) => [new Map([...was].filter(([k]) => keys.includes(k))), []]);
    const errors: string[] = [];
    const timers: ReturnType<typeof setTimeout>[] = [];
    const ask = (k: string, tries: number) =>
      load(k).then(
        (v) => alive && setGot(([was, errs]) => [new Map(was).set(k, v), errs]),
        (e) => {
          // 服务器仍在生成该视图（E-VIEW-PREPARING，server/view_worker.py：单个请求最多等待 10 秒，生成的结果会保留）：数秒后重试，
          // 不视为失败。无法完成时服务器另有提示（E-VIEW-TIMEOUT），届时才视为错误
          if (alive && e instanceof ApiError && e.code === "E-VIEW-PREPARING" && tries < PREPARING_TRIES) {
            timers.push(setTimeout(() => alive && void ask(k, tries + 1), PREPARING_WAIT_MS));
            return;
          }
          errors.push(e instanceof Error ? e.message : String(e));
          if (alive) setGot(([was]) => [new Map([...was].filter(([key]) => key !== k)), [...errors]]);
        },
      );
    for (const k of keys) void ask(k, 0);
    return () => {
      alive = false;
      timers.forEach(clearTimeout);
    };
  }, [joined]); // eslint-disable-line react-hooks/exhaustive-deps
  return got;
}

/** Redraws when any of these views receives new samples. */
export function useSceneVersions(scenes: Scene[]): number {
  const [, setTick] = useState(0);
  const joined = scenes.map((s) => s.key).join("|");
  useEffect(() => {
    const offs = scenes.map((s) => s.subscribe(() => setTick((t) => t + 1)));
    return () => offs.forEach((off) => off());
  }, [joined]); // eslint-disable-line react-hooks/exhaustive-deps
  return scenes.reduce((a, s) => a + s.version, 0);
}
