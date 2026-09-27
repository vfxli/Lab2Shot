import { useEffect, useMemo, useState } from "react";
import type { Manifest } from "../api";
import type { FrameSource } from "../transfer/frames";
import { CHANNEL_NAMES, type Plane } from "../transfer/plane";
import { channelsOf } from "../model/view2d";
import { workerAsks } from "../platform/work";
import { useGraphSnapshot, type Snapshot } from "../graph/snapshot";
import { elementOf } from "../state/items";
import { evaluate, localBoxes, recipeFor, type Local } from "./evaluate";
import { keepLocal } from "../transfer/computed";
import type { Boxes } from "../ops/recipe";

/** 视图如何使用浏览器计算的结果。
 *
 * 在「选人」中更改所选人物、在画面上拖动框：不提交任何计算，遮罩图与合成图既不在服务器生成，也不回传，
 * 画面即时更新。输出的数据与服务器结果形式相同，舞台上只增加「服务器没有时使用它」这一规则：
 *
 * - 人物框：整段一份 JSON，计算完成后即在本地。
 * - 通道路径：每条通道一个帧源（数值图、遮罩、alpha）。输出为 `Plane`，与服务器路径完全相同，
 *   因此范围映射、黑白点、着色、合成仍在 GPU 上实时计算（view/look.ts），保持全精度，无需任何网络请求。
 * - 图片路径：多条颜色通道一起查看时使用。服务器上没有该图，因此显示变换由浏览器查询预先烘焙的表
 *   （transfer/lut.ts，与本机预览 EXR 使用同一张表、同一段代码）。
 *
 * 服务器结果仍为权威：交付与下游计算始终使用服务器结果，此处计算的仅是当前画面，不产生包、不写入缓存、不写任何数据。 */

const ask = workerAsks<{ w: number; h: number; c: number; values: Float32Array; pixels: Uint8ClampedArray }>(
  () => new Worker(new URL("./evalWorker.ts", import.meta.url), { type: "module" }),
);

type Made = { w: number; h: number; c: number; values: Float32Array; pixels: Uint8ClampedArray };

/** 在途的帧：同一帧被同时请求两次（显示图与三条通道请求的是同一结果）时只计算一次，
 * 完成后立即删除（成功与失败均删除），表中只包含正在计算的条目，在构造上即有上限。
 * 计算出的帧由取帧账本按其自身的键保存（`transfer/frames.ts`），不在此处长期存储。 */
const asked = new Map<string, Promise<Made>>();

/** 当前正在计算的帧数（供 `webui/tests/registries.test.ts` 读取）。 */
export const framesInFlight = (): number => asked.size;

function frameOf(id: string, local: Local, frame: number): Promise<Made> {
  const key = `${id}:${frame}`;
  let one = asked.get(key);
  if (!one) {
    // 计算失败不得静默：接线错误、上游某帧缺少通道都会执行到此处，
    // 而取帧账本只会将该帧记为未到达，画面停留在黑色，开发者无从排查
    one = recipeFor(local, frame).then((recipe) => ask({ recipe })).catch((e: unknown) => {
      console.error(`browser evaluation failed (${id} frame ${frame})`, e);  // 面向开发者的日志，非用户消息
      throw e;
    });
    asked.set(key, one);
    void one.finally(() => asked.delete(key));
  }
  return one;
}

/** 图片路径：该帧已绘制的显示图（多条颜色通道一起查看时舞台所需）。 */
function pictureSource(id: string, local: Local): FrameSource {
  return {
    id,
    frames: local.frames,
    load: (frame) =>
      local.frames.includes(frame)
        ? async () => {
            const made = await frameOf(id, local, frame);
            // 不预乘：GPU 路径需要原始值（与 transfer/frames.ts BITMAP 一段的说明相同）
            return createImageBitmap(new ImageData(made.pixels as Uint8ClampedArray<ArrayBuffer>, made.w, made.h),
                                     { premultiplyAlpha: "none", colorSpaceConversion: "none" });
          }
        : undefined,
  };
}

/** 供测试使用的入口（`webui/tests/registries.test.ts`）：不经过 React，直接获取一条链的图片路径帧源。 */
export const sourceForTest = (id: string, local: Local): FrameSource => pictureSource(id, local);

/** 通道路径：每条通道一个源，输出的 `Plane` 与服务器路径完全相同（format 3 = float32）。 */
function planeSource(id: string, local: Local, name: string, index: number): FrameSource {
  return {
    id: `${id}|${name}`,
    frames: local.frames,
    load: (frame) =>
      local.frames.includes(frame)
        ? async (): Promise<Plane> => {
            const made = await frameOf(id, local, frame);
            const n = made.w * made.h;
            let data = made.values;
            if (made.c > 1) {
              data = new Float32Array(n);
              for (let p = 0; p < n; p += 1) data[p] = made.values[p * made.c + index];
            }
            return { kind: "plane", format: 3, width: made.w, height: made.h, data };
          }
        : undefined,
  };
}

/** 浏览器计算的结果，整理为视图可识别的形式。 */
export interface LocalResult {
  manifest: Manifest; // 作为包说明使用（尺寸、帧、范围、是否为数值图、包含的通道）
  boxes: Boxes | null; // 人物框分支（一个承载人物框的列表：其中每一条都包含在这一份中，视图全部绘制）
}

function resultOf(id: string, local: Local): LocalResult {
  const holdsBoxes = elementOf(local.type) === "boxes"; // 人物框，或承载人物框的列表
  const count = channelsOf(local.type);
  const names = CHANNEL_NAMES.slice(0, count);
  const manifest: Manifest = { type: local.type, fingerprint: id, meta: local.meta, summary: {},
                               ...(count ? { channels: { names: [...names] } } : {}) };
  // 取帧、读取包说明、获取通道三条路径均按地址工作：在此登记后，它们遇到 `local:` 即从本地读取，
  // 不发出任何请求（transfer/computed.ts：不在服务器生成，也不回传）
  keepLocal(id, {
    manifest,
    source: count ? pictureSource(id, local) : null,
    planes: Object.fromEntries(names.map((n, i) => [n, planeSource(id, local, n, i)])),
  });
  return { manifest, boxes: holdsBoxes ? localBoxes(local) : null };
}

/** 将字符串转换为短标识（帧缓存按源 id 存储，因此该链的身份须计入 id）。 */
function short(text: string): string {
  let h = 2166136261;
  for (let i = 0; i < text.length; i += 1) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return (h >>> 0).toString(36);
}

/** 该链的身份：服务器为该端口计算的指纹（`outputs[port]`，尚未计算时也存在）。
 *
 * 指纹本身即表示「该端口内容的决定因素」：该节点的参数、其上游每个节点的参数与连线均包含在内
 * （`engine/evaluation.py` 的 plan）。因此上游任何修改都会改变指纹，浏览器随之重新计算。
 *
 * 只使用所显示节点自身的 `ops` 与连线是不够的：在「选人」中更改所选人物时，「图像合成」自身的 `ops`
 * 与连线均未改变，结果不会重算，屏幕将停留在修改前的画面。身份必须覆盖整条链，而不能只覆盖最后一个节点。 */
const chainKey = (s: Snapshot, nodeId: string, port: string): string =>
  JSON.stringify([nodeId, port, s.results[nodeId]?.outputs?.[port] ?? null, s.results[nodeId]?.present ?? null]);

/** 服务器尚未计算、而浏览器可以计算的端口（`视图项的 key` → 计算结果）。
 *
 * `want`：当前画面需要查看的端口（视图项：节点 + 端口 + 其 key）。只有服务器声明「该节点可由浏览器计算」
 * （状态回复中带有 `ops`）的端口才会进入此处，不做任何多余计算。 */
export function useLocalResults(want: readonly { key: string; nodeId: string; port: string; fp: string | null }[]): Record<string, LocalResult> {
  const snap = useGraphSnapshot();
  // 需要计算的链及其各自的身份（见上方 chainKey）：身份相同则不重算
  const chains = useMemo(
    () => want.filter((it) => !it.fp && snap.results[it.nodeId]?.ops !== undefined)
               .map((it) => ({ key: it.key, nodeId: it.nodeId, port: it.port, id: chainKey(snap, it.nodeId, it.port) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [want.map((it) => `${it.key}:${it.fp ?? ""}`).join("|"), snap.results, snap.edges],
  );
  const [made, setMade] = useState<Record<string, { id: string; result: LocalResult }>>({});
  const ids = chains.map((c) => c.id).join("|");
  // 上游刚计算完成时需要重新计算。上述身份为指纹：
  // 它表示「该端口内容的决定因素」，不会因上游计算完成而改变。显示积木节点
  // （选人 / 拆成列表 / 人物框转遮罩 / 图像合成）时，浏览器可能在上游尚无包时已计算过一次但未得到结果；
  // 上游计算完成后身份不变，若只依据身份则不会重算，须切换离开再切回才会显示。
  // 因此另外跟踪哪些包已计算完成（`present`）。它仅作为重算的触发条件，不计入身份：
  // 身份不变时计算出的帧仍相同（`sameKeys` 比较即可确认，不会产生多余的重绘）。
  const cooked = useMemo(() => Object.entries(snap.results).map(([id, r]) => `${id}:${(r.present ?? []).join("+")}`).join("|"),
                         [snap.results]);
  useEffect(() => {
    let alive = true;
    void Promise.all(chains.map(async (c) => {
      const local = await evaluate(snap, c.nodeId, c.port).catch(() => null);
      // 标识中含该链的身份：修改参数后即为另一个标识，旧的帧不会被当作新帧使用
      // （帧缓存按源 id 存储，transfer/frames.ts）
      return local ? { key: c.key, id: c.id, result: resultOf(`local:${c.key}:${short(c.id)}`, local) } : null;
    })).then((all) => {
      if (!alive) return;
      const next: Record<string, { id: string; result: LocalResult }> = {};
      for (const one of all) if (one) next[one.key] = { id: one.id, result: one.result };
      setMade((had) => (sameKeys(had, next) ? had : next));
    });
    return () => {
      alive = false;
    };
  }, [ids, cooked]); // eslint-disable-line react-hooks/exhaustive-deps
  return useMemo(() => {
    const out: Record<string, LocalResult> = {};
    for (const c of chains) if (made[c.key]?.id === c.id) out[c.key] = made[c.key].result;
    return out;
  }, [made, ids]); // eslint-disable-line react-hooks/exhaustive-deps
}

const sameKeys = (a: Record<string, { id: string }>, b: Record<string, { id: string }>): boolean => {
  const ka = Object.keys(a);
  return ka.length === Object.keys(b).length && ka.every((k) => a[k].id === b[k]?.id);
};
