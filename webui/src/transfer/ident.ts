/** 帧源的身份：缓存键必须由结构体生成，不得手工拼接。
 *
 * 一帧的呈现结果取决于：数据包（`fp`）、该包的计算批次（`gen`）、代理档位（`tier`）、用于去畸变的相机
 * （`through`），以及本机原件对应的文件与解码所用的色彩空间（`local`）。这些字段全部包含在结构体中，
 * 缺少任何一项都无法通过 TypeScript 检查；本文件的函数是将这些字段拼接为字符串的唯一位置。手工拼接时遗漏文件、
 * 色彩空间或重算信息中的任何一项，都会导致命中旧帧。 */
import type { Through } from "../api";
import { genOf } from "./gens";

interface LocalPart {
  digest: string; // 各帧文件键（名称、大小、修改时间）拼接后的散列（sources.ts filesDigest）
  where: string; // 来源（本机目录名，或在本标签页中选择的文件）
  frames: number; // 本机帧数
  space: string; // EXR 解码所用的色彩空间
}

interface SourceIdent {
  fp: string;
  gen: string; // 生成号（transfer/gens.ts），"" 表示未知
  tier: string; // `@512` 形式的档位片段（sources.ts tierTag），包说明未到时为 ""
  through?: Through | null;
  local?: LocalPart | null;
}

/** 本机部分的起始标记：`packetPart` 据此截取「数据包 · 批次 · 档位」前缀（压缩字节存放在该前缀下）。 */
export const FROM = "#from:";

/** 「数据包 · 批次 · 档位 · 去畸变相机」：服务器端字节的键前缀。 */
export const packetKey = (i: Pick<SourceIdent, "fp" | "gen" | "tier" | "through">): string =>
  `${i.fp}#${i.gen}${i.tier}${i.through ? `~${i.through.fp}|${i.through.at}` : ""}`;

/** 帧源 id：服务器部分，加上本机原件部分（如有）。 */
export const identOf = (i: SourceIdent): string =>
  packetKey(i) + (i.local ? `${FROM}本机 ${i.local.frames} 帧 @ ${i.local.where} #${i.local.digest} ${i.local.space}` : "");

/** 用户在本标签页中选择的文件构成的帧源 id（sources.ts pickedFrames）：节点、文件组成（各文件键的散列：更换文件即另一组
 * 缓存键）、EXR 解码所用的色彩空间（同一文件换一个色彩空间即另一张图）。 */
export const pickedIdent = (i: { node: string; digest: string; space: string }): string => `picked:${i.node}#${i.digest}#${i.space}`;

/** 服务器帧源与本机选择的帧源合并后的 id（sources.ts localFirst）：两者的 id，以及以哪一方为准（`exact`：本机文件
 * 优先）。`exact` 翻转后同一帧绘制的是另一份字节，因此它也在 id 中。 */
export const mergedIdent = (i: { server: string; local: string; exact: boolean }): string =>
  `${i.server}${FROM}${i.exact ? "本机优先" : "服务器优先"} ${i.local}`;

/** 单条通道的帧源 id（用于按通道取数的路径）。 */
export const channelIdent = (fp: string, name: string, tier: string): string => `ch:${packetKey({ fp, gen: genOf(fp), tier })}|${name}`;

/** 帧键：源 id + 帧号。解码后的位图存放在该键下，压缩字节存放在 `bytes:` + 该键下（sources.ts bytesKey）。 */
export const frameKey = (id: string, frame: number): string => `${id}:${frame}`;
