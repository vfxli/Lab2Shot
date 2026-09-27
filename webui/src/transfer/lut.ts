import { ApiError, json } from "../platform/http";
import { cache } from "./cache";
import { MessageError } from "../messages/message";
import type { Lut } from "../ops/lut";
/** 获取显示变换查找表（`GET /api/packet/{fp}/lut` 或 `/api/view/lut`）。
 *
 * 表的结构与查表算法见 `ops/lut.ts`（纯算术，最底层）。本模块只负责获取，每个地址只请求一次并长期保留
 * （表仅与色彩空间、显示、视图有关，与具体数据包无关）。
 *
 * 使用方：本机 EXR 预览（`transfer/exr.ts`）与浏览器端计算的画面（`view/evaluate.ts`）。
 * 二维舞台不使用该表，其图片与通道两条路径均不经过查找表，见 `transfer/route.ts`。 */

export type { Lut };


/** 进行中的请求：同一地址的并发请求合并为一次，完成后立即移除（无论成功或失败）。
 * 取回的表存入页面统一缓存（`transfer/cache.ts`，按字节计入同一预算），不在此处长期保存。 */
const asked = new Map<string, Promise<Lut>>();

/** 当前进行中的查找表请求数（供 `webui/tests/registries.test.ts` 读取）。 */
export const lutsInFlight = (): number => asked.size;

/** 按地址获取查找表，每个地址只请求一次（同一地址的内容恒定）。 */
export function lutAt(url: string): Promise<Lut> {
  const key = `lut:${url}`;
  const had = cache.get<Lut>(key);
  if (had) return Promise.resolve(had);
  let l = asked.get(url);
  if (!l) {
    l = (async () => {
      const j = await json<Omit<Lut, "data"> & { data: string }>("GET", url).catch((e) => {
        throw e instanceof ApiError ? new MessageError("E-VIEW-NOLUT", { status: e.status }) : e;
      });
      const bytes = Uint8Array.from(atob(j.data), (c) => c.charCodeAt(0));
      const made = { ...j, data: new Uint16Array(bytes.buffer) };
      cache.keep(key, made, made.data.byteLength, "small");
      return made;
    })();
    asked.set(url, l);
    void l.finally(() => asked.delete(url));  // 完成即移除：表中只保留进行中的请求
  }
  return l;
}

/** 该数据包的显示变换（数值图及本身已处于显示空间的像素返回 `mode: "raw"`，按原值绘制）。 */
export const lutOfPacket = (fp: string): Promise<Lut> => lutAt(`/api/packet/${fp}/lut`);

/** 用户所选 EXR 文件使用的查找表（`space`：节点的色彩空间，为空时按文件名规则判定）。 */
export const lutOfFile = (space: string, file: string): Promise<Lut> =>
  lutAt(`/api/view/lut?space=${encodeURIComponent(space)}&file=${encodeURIComponent(file)}`);

