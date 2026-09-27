import type { Manifest } from "../api";
import { CHANNEL_NAMES, VALID } from "./plane";

/** 决定一侧所需通道的传输方式，是全项目唯一的判定点。无论解算器输出多少通道，只传输使用者正在查看的通道。
 *
 * 仅有两条路径：
 * - 图片路径（`/frame/{f}.png`）：同时查看三条颜色通道时使用。该图是三条通道经 OCIO 显示变换后
 *   编码的无损图像，是其最小载体（一张实拍帧约 1.3 MB，三条 float32 通道约 3.8 MB）；
 *   单独查看某一颜色通道（R / G / B）时也使用此路径，因为显示变换是三维查找表，三条通道须一起查表，且该图通常已在本地。
 * - 通道路径（`/frame/{f}/channel/{name}`）：其余所有情况，包括 alpha、遮罩、深度、编号、视频的单条通道，
 *   以及数值图的整体查看（数值图不经过色彩管理，每条通道独立着色，不需要三维查找表）。
 *
 * 单独成文件的原因：显示界面（`view/Stage2D.tsx`）与后台整段预取（`transfer/prefetchLive.ts`）
 * 必须使用同一判定规则，否则舞台只需一条通道时，后台预取仍会逐帧下载整张显示图，造成流量浪费。 */

/** 判断该数据包是否不经过色彩管理（与服务器 `server/packets.py packet_lut` 的判定一致）。 */
export const rawOf = (m: Manifest | null): boolean =>
  m?.type === "video" || !!m?.meta.values || !m?.meta.colorspace;

/** 该侧使用通道路径时需要的通道列表（空数组表示使用图片路径）。
 * `index`：要查看的通道序号（null 表示同时查看多条）。 */
export function channelsFor(m: Manifest | null, index: number | null): string[] {
  const names = m?.channels?.names ?? [];
  if (!names.length) return [];                       // 服务器尚无该数据包，或该包不是二维像素数据
  if (index === null) {
    // 同时查看多条通道时，仅数值图使用通道路径：数值图不经过色彩管理（各通道独立着色），
    // 且黑白点直接作用于数值，8 位精度会产生断层，因此需要完整精度。
    // 画面类数据（照片、渲染、HDRI，以及视频这类已处于显示空间的 8 位像素）一律使用图片路径，
    // 因为该图是三条通道的最小载体。视频尤其如此：其三条通道本身来自一张已解码的 PNG，
    // 拆分发送只会更大，因此不能仅以是否经过色彩管理作为判定依据。
    return m?.meta.values ? names.filter((n) => n !== VALID).slice(0, 3) : [];
  }
  if (index < 3 && !rawOf(m)) return [];              // 单独查看颜色通道时同样需要经过三维显示变换表
  const one = CHANNEL_NAMES[index];
  return one && names.includes(one) ? [one] : [];
}

/** 返回该数据包中的有效像素通道（如有）；使用通道路径时需一并获取（无值处绘制为黑色）。 */
export const validFor = (m: Manifest | null, names: string[]): string | null =>
  (names.length && (m?.channels?.names ?? []).includes(VALID) ? VALID : null);
