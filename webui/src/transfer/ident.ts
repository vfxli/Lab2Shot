/** 帧源的身份：缓存键必须由结构体生成，不得手工拼接。
 *
 * 一帧的呈现结果取决于：数据包（`fp`）、该包的计算批次（`gen`）、代理档位（`tier`）、用于去畸变的相机
 * （`through`），以及本机原件对应的文件与解码所用的色彩空间（`local`）。这些字段全部包含在结构体中，
 * 缺少任何一项都无法通过 TypeScript 检查；`identOf` 是全页面唯一将其拼接为字符串的位置。手工拼接时遗漏文件、
 * 色彩空间或重算信息中的任何一项，都会导致命中旧帧。 */
import type { Through } from "../api";
import { genOf } from "./gens";

export interface LocalPart {
  digest: string; // 各帧文件键（名称、大小、修改时间）拼接后的散列（sources.ts filesDigest）
  where: string; // 来源（本机目录名，或在本标签页中选择的文件）
  frames: number; // 本机帧数
  space: string; // EXR 解码所用的色彩空间
}

export interface SourceIdent {
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

/** 单条通道的帧源 id（用于按通道取数的路径）。 */
export const channelIdent = (fp: string, name: string, tier: string): string => `ch:${packetKey({ fp, gen: genOf(fp), tier })}|${name}`;

/** 帧键：源 id + 帧号。解码后的位图存放在该键下，压缩字节存放在 `bytes:` + 该键下（sources.ts bytesKey）。 */
export const frameKey = (id: string, frame: number): string => `${id}:${frame}`;
