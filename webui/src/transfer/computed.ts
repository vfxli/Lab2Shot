import type { Manifest } from "../api";
import type { FrameSource } from "./frames";
import { cache } from "./cache";

/** 将浏览器端计算的结果登记到取数层可识别的位置。
 *
 * 取帧与读取包说明（`transfer/frames.ts` 的 `serverFrames` / `manifestOf`）均按地址工作。
 * 浏览器端计算的结果没有服务器地址，因此为其分配以 `local:` 开头的标识并登记于此：上述两条路径遇到
 * 该前缀时直接从此处读取，不发出任何请求（遮罩图与合成图既不在服务器生成，也不回传）。
 *
 * 该结果不是数据包：`ViewItem.fp` 仍为 null（视图项另有 `local` 字段），因此「数据信息」「交付」「下游计算」
 * 等需要服务器数据的功能不会误用它。此处登记的仅是画面的绘制方式。 */

export const LOCAL = "local:";

/** 判断该标识是否指向浏览器端计算的结果。 */
export const isLocal = (id: string | null | undefined): boolean => !!id && id.startsWith(LOCAL);

interface Held {
  manifest: Manifest;
  source: FrameSource | null; // 图片路径：该帧已完成显示变换的图像（由浏览器执行显示变换）
  planes: Record<string, FrameSource>; // 通道路径：每条通道一个源（数值图与 alpha 使用此路径，在 GPU 上实时计算）
}

// 不单独建表：登记到页面统一缓存（`transfer/cache.ts`），与包说明（`manifestOf`）使用同一机制；
// 模块级 Map 不做长期存储
const key = (id: string) => `computed:${id}`;

/** 登记一份结果（对同一标识重复登记即替换，对应该链路的计算方式发生变化）。 */
export function keepLocal(id: string, one: Held): void {
  cache.register(key(id), one);  // 登记层：不计入预算、不被淘汰（见 cache.ts registry）
}

const heldOf = (id: string): Held | undefined => cache.registered<Held>(key(id));

export const localManifest = (id: string): Manifest | undefined => heldOf(id)?.manifest;
export const localSource = (id: string): FrameSource | null => heldOf(id)?.source ?? null;
export const localPlane = (id: string, name: string): FrameSource | null => heldOf(id)?.planes[name] ?? null;
