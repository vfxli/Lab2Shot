import { ApiError } from "../platform/http";
import { described, describedNow } from "./frameStore";
import { LUTS, lutKey, lutSlot, lutUrl } from "./frameKey";
import { cache } from "../platform/cache";
import { onServerChange, serverNow } from "../state/server";
import { MessageError } from "../messages/message";
import type { Lut } from "./lookup";
/** 获取显示变换查找表（`GET /api/view/lut`）。
 *
 * 表的结构与查表算法见 `transfer/lookup.ts`（纯算术，最底层）。本模块只说怎么把回复转成表；取（在途去重、失败退避、
 * 存入页面缓存）经 transfer/frameStore.ts described，键和地址在 transfer/frameKey.ts（表只与色彩空间和文件名规则有关，与包无关）。
 *
 * 使用方：本机 EXR 预览（`transfer/exr.ts`）与本机代理（`transfer/localProxy`）。
 * 二维舞台不使用该表，其图片与通道两条路径均不经过查找表，见 `transfer/route.ts`。 */

export type { Lut };


/** 表的回复（数据为 base64）转成查表用的样子。 */
const readLut = (raw: unknown): Lut => {
  const j = raw as Omit<Lut, "data"> & { data: string };
  const bytes = Uint8Array.from(atob(j.data), (c) => c.charCodeAt(0));
  return { ...j, data: new Uint16Array(bytes.buffer) };
};

/** 用户所选 EXR 文件使用的查找表（`space`：节点的色彩空间，为空时按文件名规则判定）。 */
export const lutOfFile = (space: string, file: string): Promise<Lut> =>
  described(lutSlot(space, file), lutUrl(space, file), readLut, undefined, (l) => l.data.byteLength).catch((e) => {
    throw e instanceof ApiError ? new MessageError("E-VIEW-NOLUT", { status: e.status }) : e;
  });

/** 同上，已经取到的（同步，不发请求）；还没取到为 undefined。 */
export const lutNow = (space: string, file: string): Lut | undefined => describedNow<Lut>(lutKey(space, file));


// 表跟着服务器的 OCIO 配置（工作空间）走，而换配置要重启服务（lab2shot/config.py color.config）：服务器换了进程
// （boot 变了）就把已取的表全扔掉，下次按新配置取；本机代理按表的内容散列认，也跟着换
let boot: string | null = null;
onServerChange(() => {
  const now = serverNow()?.boot ?? null;
  if (boot !== null && now !== null && now !== boot) cache.forgetGroup(LUTS);
  if (now !== null) boot = now;
});
