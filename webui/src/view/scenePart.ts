/** 三维视图数据的网络获取：本模块只负责获取，所取字节的解析由 sceneData.ts 负责。
 *
 * sceneData.ts 负责场景在浏览器中的表示（模型、点云、曲线、相机，按帧的分块），本模块负责
 * 数据块的请求方式、大小上限以及获取失败时的提示，二者职责分离。
 *
 * 浏览器已持有的数据不再重复请求：
 * 取回的字节存入页面唯一的缓存（transfer/cache.ts），键为该数据的地址。
 * 基础块的地址含内容的 sha（`?h=…`），分块的地址含结果指纹、格式、帧号及「点云上限」
 * （`?v=…&f=lo-hi&mb=…`：该字节序列已按上限删减点，管理员修改上限后即为另一序列，因此上限也计入地址），
 * 因此同一地址始终对应同一字节序列，已持有时无需再请求服务器。
 *
 * 不依赖浏览器自身的 HTTP 缓存（尽管服务器发送 `Cache-Control: immutable`）：缓存容量、单个资源能否缓存、
 * 何时被清除均取决于浏览器的启发式策略，重传量无法稳定为 0。由本模块自行记录，才能保证为 0。 */

import { MessageError } from "../messages/message";
import { request } from "../platform/http";
import { sizeText } from "../platform/format";
import { cache } from "../transfer/cache";

/** 单个数据块的大小上限；超过时浏览器无法容纳，须说明实际大小与上限，不得静默截断。 */
export const MAX_PART = 1.5e9;

/** 三维数据在页面唯一缓存中的前缀（节点图关闭时据此整体释放）。 */
export const PART_KEY = "3d:part:";

/* 地址中不含画质：删点始终在服务器端进行（按后台的「点云上限」），一个地址即对应一个字节序列，
 * 页面端不拼接任何内容。 */

/** 在途请求：同一地址被同时请求两次（两个场景共用一份基础块，或已释放的分块被再次请求）时只传输一次。
 *
 * 在构造上即有上限：条目只在传输期间存在，到达（成功或失败）后立即删除。
 * 模块级 `Map` 默认不允许使用，此类在途登记表为明确豁免的例外
 * （`tools/rule_counts.py INFLIGHT_REGISTRIES`），条件是必须有测试验证：
 * 即下方供 `webui/tests/registries.test.ts` 使用的 `partsInFlight`。 */
const asking = new Map<string, Promise<Uint8Array>>();

/** 当前在途的请求数（仅供测试使用：验证完成后该表为空）。 */
export const partsInFlight = (): number => asking.size;

async function load(at: string): Promise<Uint8Array> {
  const r = await request(at); // 因未登录被拒的请求交由登录门处理
  const size = Number(r.headers.get("content-length") ?? 0);
  if (size > MAX_PART) throw new MessageError("E-VIEW-PARTTOOBIG", { size: sizeText(size), max: sizeText(MAX_PART) });
  try {
    return new Uint8Array(await r.arrayBuffer());
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    throw size ? new MessageError("E-VIEW-PARTNOROOM", { size: sizeText(size), reason }) : new MessageError("E-VIEW-NOROOM", { reason });
  }
}

/** 一个三维数据块：已持有则直接返回，否则向服务器请求，取回后记录在页面唯一的缓存中。
 *
 * 记录档位为 `fetched`（已下载、当前未被绘制的数据），与预取的画面帧同档：
 * 实际绘制的是 sceneData.ts 解码出的样本，字节本身并不在屏幕上。预算紧张时优先释放，
 * 释放后只需下次重新请求，不会移除正在查看的内容。 */
export function fetchPart(url: string): Promise<Uint8Array> {
  const at = url;
  const key = PART_KEY + at;
  // 返回的是一份拷贝：接收方会将字节整体 transfer 给解析线程（sceneWork.ts parseChunk），若缓存中是同一个
  // 对象，将变为空壳，被预算淘汰后重新取回时将无法解析，「在途」通知会一直显示
  const held = cache.get<Uint8Array>(key);
  if (held) return Promise.resolve(held.slice());
  const already = asking.get(key);
  if (already) return already.then((bytes) => bytes.slice());
  const going = load(at).then(
    (bytes) => {
      asking.delete(key);
      cache.keep(key, bytes, bytes.byteLength, "fetched");
      return bytes.slice();
    },
    (e) => {
      asking.delete(key); // 失败的不记录：再次请求时重新尝试
      throw e;
    },
  );
  asking.set(key, going);
  return going;
}
