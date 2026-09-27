import type { Manifest } from "../api";
import { cache } from "./cache";
import { localFile } from "./local";
import { fileKeyOf } from "./localProxy";

/** 本机原件：统一回答「该数据在用户本机上是否有原件」。
 * 视图数据的来源按每条输入分别判定：本地有原件即使用原件，无需传输且保持全精度；读取类节点和输出节点的文件
 * 本就位于用户本机，不使用压缩预览。
 *
 * 查询单位是「一份数据」而非「一个节点」：键即包的指纹（`fp`），
 * 与取帧、读取包说明两条路径使用同一个键。因此当合成节点的 A 端口连接本机读取的画面、
 * B 端口连接解算器输出的遮罩时，两者分别查询：A 获得本机原件（无传输、全精度），
 * B 无原件则照常使用服务器代理，合成在 GPU 上完成（`view/look.ts`）。
 * 不得因为节点有一条输入来自服务器，就将本地那一条也改为向服务器请求。
 *
 * 两份数据尺寸不同无需处理：在 `view/look.ts` 中两侧各为一张纹理，各自按其尺寸采样到舞台的像素网格上；
 * 代理为等比缩放，因此覆盖的画面范围相同，UV 一致。不缩小本地数据迁就代理，也不放大代理迁就本地数据。
 *
 * 本模块只改变帧字节的来源，其余均不变：
 * 包说明、画面尺寸、`data_window`、时间线、下游计算、交付均以服务器数据为准。
 * 特别是画面在屏幕上的尺寸由包说明中的 `meta.width/height` 决定，而非由图像的像素数决定；
 * 否则节点 A 使用原件、节点 B 使用代理时，切换节点会导致画面大小变化，无法对比效果。
 *
 * 混合来源无需标注：视图始终是预览。不加角标、提示或开关。 */

/** 一份数据在用户机器上的原件：包含哪些帧，以及每一帧对应的文件。
 *
 * `fileOf` 按需获取（需打开系统文件句柄并在包中定位）：获取失败返回 `null`，
 * 该帧照常使用服务器代理。找不到不属于错误（用户可能更换了下载目录，或该数据从未下载）。 */
export interface Originals {
  frames: number[];
  fileOf(frame: number): Promise<File | null>;
  space: string; // EXR 解码所用的色彩空间（节点的 colorspace，"" 表示按文件自身的规则）
  where: string; // 来源：仅用于开发者日志，不作为用户消息
  /** 整段用于查询色彩空间的文件名（`transfer/sources.ts fromFile` 的 `rules`）：一段序列只有一个
   * 色彩空间，读取节点本身也按第一个文件推断（`lab2shot/nodes/core/input.py`）。
   * 未提供时按每一帧自身的文件名查询，即每帧一张显示变换表，一段 48 帧的 EXR 将多发 48 次请求。 */
  name?: string;
  /** 每一帧对应的文件（`transfer/localProxy fileKeyOf`）：账本据此检查本机代理是否在磁盘上。在查找时一并列出 */
  keys?: Map<number, string>;
}

// 不另建表：登记到页面唯一的缓存（`transfer/cache.ts`），与浏览器计算的结果
// （`transfer/computed.ts`）、包说明（`manifestOf`）使用同一机制；模块级 Map 不做长期存储
const key = (fp: string) => `orig:${fp}`;

interface Held {
  originals: Originals | null;
  looking: boolean; // 仍在查找（打开文件句柄、列目录、识别包均为异步操作）
}

/** 开始查找该数据的原件：在结果确定之前，不取回整段代理。
 *
 * 刷新后查找原件需访问 IndexedDB、请求权限、列出目录，耗时两三百毫秒。若无此项，后台取回路径
 * （`fillWhole`）会在此期间开始拉取整段代理，待原件登记完成时带宽已被占用；
 * 素材位于用户本机磁盘、本无需传输，却先传输了整段。
 *
 * 当前帧仍立即从服务器获取（不出现空白帧），只阻止整段取回。
 *
 * 不会卡住：`useOriginals` 查找结束后必定调用一次 `keepOriginals`（找到则传入该原件，
 * 未找到则传入 null），即使组件已卸载也会调用，因此该状态不会停留在「一直查找」。 */
export function lookingFor(fp: string): void {
  cache.register(key(fp), { originals: null, looking: true } satisfies Held);
}

/** 查找结束（`one` 为 null 表示该数据没有原件，照常使用服务器代理）。 */
export function keepOriginals(fp: string, one: Originals | null): void {
  if (one) cache.register(key(fp), { originals: one, looking: false } satisfies Held);
  else cache.unregister(key(fp));
}

/** 该数据是否仍在查找原件（查找期间不取回整段代理）。 */
export const stillLooking = (fp: string): boolean => cache.registered<Held>(key(fp))?.looking === true;

/** 该包在用户机器上是否有原件。
 *
 * 按以下顺序查询两个来源：
 * 1. 已授权的本机目录或「输出」自身保存位置中找到的文件（由 `files/localDirs.ts` 查找、
 *    `view/useOriginals.ts` 登记）：句柄存于 IndexedDB，刷新后仍保留。
 * 2. 本标签页中刚选择的文件（`transfer/local.ts`）：服务器在包说明中写入每一帧的 sha256
 *    （`blobs`），页面持有相同内容的文件时即使用该文件，内容逐字节一致。刷新后失效
 *    （浏览器只在用户亲自选择时将文件交给页面）。
 *
 * 第 1 项需打开文件句柄，为异步操作，因此由上方的登记表提供；第 2 项可同步得知，在此处实现。 */
export function originalsFor(fp: string, manifest: Manifest | null, space = ""): Originals | null {
  const held = cache.registered<Held>(key(fp));
  if (held?.originals) return held.originals;
  const blobs = (manifest as (Manifest & { blobs?: Record<string, string> }) | null)?.blobs;
  if (!blobs) return null;
  const frames = Object.keys(blobs)
    .map(Number)
    .filter((f) => !!localFile(blobs[String(f)]));
  if (!frames.length) return null;
  return {
    frames,
    // 同时登记每一帧的文件键：取帧路径将其拼入源 id（`sources.ts filesDigest`），更换文件即对应另一组缓存键
    keys: new Map(frames.map((f) => [f, fileKeyOf(localFile(blobs[String(f)])!)])),
    fileOf: async (frame) => localFile(blobs[String(frame)]) ?? null,
    space,
    where: "picked in this tab",
    name: localFile(blobs[String(frames[0])])?.name,
  };
}
