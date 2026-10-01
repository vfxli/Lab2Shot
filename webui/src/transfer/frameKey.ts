/** 视图数据的地址与缓存键——唯一来源：二维帧（图片路径、通道路径、视频）、三维场景、点云预览、边算边看、三维块、
 * 包说明 / 人物框 / 跟踪点、查找表的地址，以及页面缓存里的键（含本机代理与原件的登记）都只在这里拼。
 * 别处拿结构体来要，不手工拼字符串。
 *
 * 一帧的样子取决于：数据包（fp）、它的代次（g：同一指纹重算后变）、显示档位（px 与代理做法 pf）、透过的相机（它的
 * 指纹、路径和代次 cg），本机原件（文件组成与解码用的色彩空间）。键里缺任何一项都会拿到旧帧；地址里带同样的几项，
 * 服务器才标 immutable，浏览器的 HTTP 缓存与页面缓存认的是同一份标识。 */

import type { Manifest } from "../api";
import type { Slot } from "../platform/cache";
import { genOf } from "./gens";

// ------------------------------------------------------------------ 显示档位（结构化，不拼字符串再解析）

/** 包说明里的显示档位：代理的长边像素（px）与代理做法（form，编号图为 "ids"：最近邻、值不压）。包说明没到时为 null。 */
export interface Tier {
  px: number;
  form: string;
}

export const tierOf = (m: Manifest | null): Tier | null => (m?.proxy ? { px: m.proxy.px, form: m.proxy.form ?? "" } : null);

/** 档位在键里的一段（`@512`、`@512.ids`；没有为 ""）：只在本文件里用。 */
const tierPart = (t: Tier | null): string => (t ? `@${t.px}${t.form ? `.${t.form}` : ""}` : "");

/** 透过相机看一帧底图（lab2shot/server/packets.py frame 的 through / at / cg）：相机包的指纹、它在场景里的路径、它的代次。
 * 带畸变时服务器按它的镜头去畸变再发，画面同时取决于两个包，所以相机的代次也进地址和键。 */
export interface Through {
  fp: string;
  at: string;
  gen: string;
}

/** 地址里区分字节内容的几项（lab2shot/server/wire.py versioned）：代次、档位、代理做法。 */
interface Version {
  g: string;
  tier: Tier | null;
}

/** 代次按调用方给的那一份（帧源建起时取的快照：键和地址出自同一份，见 `sources.ts serverFrames`）；没给才读此刻的表。 */
const versionOf = (fp: string, tier: Tier | null, g?: string): Version => ({ g: g ?? genOf(fp), tier });

const query = (parts: string[]): string => (parts.length ? `?${parts.join("&")}` : "");
const versionQuery = (v: Version): string[] =>
  [...(v.g ? [`g=${v.g}`] : []), ...(v.tier?.px ? [`px=${v.tier.px}`] : []), ...(v.tier?.form ? [`pf=${encodeURIComponent(v.tier.form)}`] : [])];

// ------------------------------------------------------------------ 二维帧：地址

/** 一帧的显示图（图片路径）：序列图与视频两条路由，透过带畸变的相机时带 through / at / cg。 */
export function pictureUrl(it: { fp: string; type: string; g?: string }, frame: number, tier: Tier | null, through: Through | null = null): string {
  const v = versionQuery(versionOf(it.fp, tier, it.g));
  if (it.type === "video") return `/api/view/${it.fp}/video/${frame}.png` + query(v);
  const t = through ? [`through=${encodeURIComponent(through.fp)}`, `at=${encodeURIComponent(through.at)}`, ...(through.gen ? [`cg=${through.gen}`] : [])] : [];
  return `/api/packet/${it.fp}/frame/${frame}.png` + query([...t, ...v]);
}

/** 一帧里单条通道的数据（通道路径：R / G / B / A 或 valid，lab2shot/server/packets.py frame_channel）。 */
export const channelUrl = (fp: string, frame: number, name: string, tier: Tier | null, g?: string): string =>
  `/api/packet/${fp}/frame/${frame}/channel/${name}` + query(versionQuery(versionOf(fp, tier, g)));

// ------------------------------------------------------------------ 二维帧：键

interface LocalPart {
  digest: string; // 各帧文件键（名称、大小、修改时间）拼接后的散列
  where: string; // 来源（本机目录名，或在本标签页里选的文件）
  frames: number; // 本机帧数
  space: string; // EXR 解码用的色彩空间
}

/** 源 id 里「来源」一段的开头：前半段是服务器那份字节的身份（压缩字节按它存），后半段说从哪儿读。 */
export const FROM = "#from:";

/** 服务器那份字节的身份：「数据包 · 代次 · 档位 · 去畸变相机」。压缩字节按它存（`bytesKey`）。 */
export const packetKey = (i: { fp: string; g?: string; tier: Tier | null; through?: Through | null }): string =>
  `${i.fp}#${i.g ?? genOf(i.fp)}${tierPart(i.tier)}${i.through ? `~${i.through.fp}#${i.through.gen}|${i.through.at}` : ""}`;

/** 帧源 id：服务器那一段，加上本机原件那一段（有的话）。本机帧数、文件组成、色彩空间都在里面：换了文件就是另一组键。 */
export const sourceId = (i: { fp: string; g?: string; tier: Tier | null; through?: Through | null; local?: LocalPart | null }): string =>
  packetKey(i) + (i.local ? `${FROM}本机 ${i.local.frames} 帧 @ ${i.local.where} #${i.local.digest} ${i.local.space}` : "");

/** 本标签页里选的文件构成的帧源：节点、文件组成、EXR 解码用的色彩空间。 */
export const pickedId = (i: { node: string; digest: string; space: string }): string => `picked:${i.node}#${i.digest}#${i.space}`;

/** 服务器帧源与本机选的帧源合并：两者的 id 与以谁为准（本机优先时同一帧画的是另一份字节）。 */
export const mergedId = (i: { server: string; local: string; exact: boolean }): string =>
  `${i.server}${FROM}${i.exact ? "本机优先" : "服务器优先"} ${i.local}`;

/** 单条通道的帧源。 */
export const channelId = (fp: string, name: string, tier: Tier | null, g?: string): string => `ch:${packetKey({ fp, g, tier })}|${name}`;

/** 一帧的键：源 id + 帧号。解码后的位图 / 通道数据存在它下面（组 = 源 id），压缩字节存在 `bytesKey(它)` 下
 * （组 = `bytesKey(源 id)`）。组都在这里给明（`Slot`），页面缓存不从字符串猜：代次是带冒号的时刻。 */
export const frameKey = (id: string, frame: number): string => `${id}:${frame}`;
/** 一个组里一帧的键对应的帧号（键都是 `frameKey(组, 帧)` 拼的）。 */
export const frameIn = (id: string, key: string): number => Number(key.slice(id.length + 1));

/** 一帧压缩字节的键（字节层，整段保留）。 */
export const bytesKey = (key: string): string => `bytes:${key}`;
export const bytesSlot = (id: string, frame: number): Slot => ({ key: bytesKey(frameKey(id, frame)), group: bytesKey(id) });

/** 一帧压缩字节在页面缓存里的键：只与服务器那份字节（源 id 的服务器那一段）有关。 */
export const frameBytesKey = (id: string, frame: number): string => bytesKey(frameKey(packetPart(id), frame));

/** 源 id 的服务器那一段（去掉「来源」）：压缩字节只与它有关，与这一帧是不是从本机读无关。 */
export const packetPart = (id: string): string => {
  const at = id.indexOf(FROM);
  return at < 0 ? id : id.slice(0, at);
};

/** 一个源全部压缩字节所在的组（`cache.keysIn` / `cache.watch` 用）：源 id 的服务器那一段。 */
export const bytesGroup = (id: string): string => bytesKey(packetPart(id));

// ------------------------------------------------------------------ 包自带的整份数据（一份 JSON）

/** 包说明（meta：帧表、尺寸、代理档位）、人物框、跟踪点、曲线。都随包的代次变：键和地址都带代次，重算后换键，
 * 重算前发出、重算后才回来的旧回复写在旧键下，不会被新代次读到。 */
export type Described = "manifest" | "boxes" | "tracks" | "curves";

export const describedKey = (kind: Described, fp: string): string => `${kind}:${fp}#${genOf(fp)}`;
export const describedSlot = (kind: Described, fp: string): Slot => ({ key: describedKey(kind, fp), group: kind });
export const describedUrl = (kind: Described, fp: string): string =>
  `/api/packet/${fp}${kind === "manifest" ? "" : `/${kind}`}` + query(genOf(fp) ? [`g=${genOf(fp)}`] : []);

/** 显示变换查找表（与包无关：只看色彩空间和查规则用的文件名）。 */
export const lutUrl = (space: string, file: string): string => `/api/view/lut?space=${encodeURIComponent(space)}&file=${encodeURIComponent(file)}`;
export const LUTS = "lut";
export const lutKey = (space: string, file: string): string => `lut:${encodeURIComponent(space)}|${encodeURIComponent(file)}`;
export const lutSlot = (space: string, file: string): Slot => ({ key: lutKey(space, file), group: LUTS });

// ------------------------------------------------------------------ 本机

/** 本机代理已在磁盘上的登记：文件 × 「档位-显示变换」（transfer/localProxy）。组为 "proxy"。 */
export const proxyReadyKey = (fileKey: string, tierKey: string): string => `proxy:${fileKey}|${tierKey}`;
export const proxyReadySlot = (fileKey: string, tierKey: string): Slot => ({ key: proxyReadyKey(fileKey, tierKey), group: "proxy" });

/** 一个包在本机的原件的登记（transfer/originals.ts）。组为 "orig"。 */
export const originalsKey = (fp: string): string => `orig:${fp}`;
export const originalsSlot = (fp: string): Slot => ({ key: originalsKey(fp), group: "orig" });

// ------------------------------------------------------------------ 三维

/** 一个包给 DCC 剪贴板的文字（「复制到 Nuke」）：不进缓存，只有地址。 */
export const clipboardUrl = (fp: string): string => `/api/packet/${fp}/clipboard`;

/** 三维场景：键为指纹 + 代次（重算后换键，不拿旧场景）；地址带同一个代次。 */
export const sceneKey = (fp: string): string => `${fp}#${genOf(fp)}`;
export const sceneUrl = (fp: string): string => `/api/packet/${fp}/scene` + query(genOf(fp) ? [`g=${genOf(fp)}`] : []);

/** 深度 / 位置图的点云预览：两个包（图与放置它的相机）的代次都在键和地址里。 */
export const pointsKey = (fp: string, camera: string | null): string =>
  `${fp}#${genOf(fp)}|${camera ?? ""}#${camera ? genOf(camera) : ""}`;
export const pointsUrl = (fp: string, camera: string | null): string =>
  `/api/view/${fp}/points` + query([...(camera ? [`camera=${camera}`, ...(genOf(camera) ? [`cg=${genOf(camera)}`] : [])] : []), ...(genOf(fp) ? [`g=${genOf(fp)}`] : [])]);

/** 边算边看的点云（地址按任务与节点，不是内容地址：算完整体丢掉）。 */
export const partialPointsKey = (job: string, node: string, port: string, camera: string | null): string =>
  `partial|${job}|${node}|${port}|${camera ?? ""}`;
export const partialBase = (job: string, node: string, port: string): string =>
  `/api/jobs/${job}/partial/${encodeURIComponent(node)}/${encodeURIComponent(port)}`;
/** 边算边看的二维帧：帧源 id（不是包指纹：它指向算完的结果）与一帧的地址。 */
export const partialFrameId = (job: string, node: string, port: string): string => `partial:${job}:${node}:${port}`;
export const partialFrameUrl = (job: string, node: string, port: string, frame: number): string => `${partialBase(job, node, port)}/frame/${frame}.png`;
export const partialPointsUrl = (job: string, node: string, port: string, camera: string | null): string =>
  `${partialBase(job, node, port)}/points${camera ? `?camera=${camera}` : ""}`;

/** 页面缓存里三维场景 / 三维块的键（前缀 + 上面的键或块地址）。 */
export const SCENES = "3d:scene";
export const sceneCacheKey = (key: string): string => `3d:scene:${key}`;
export const sceneSlot = (key: string): Slot => ({ key: sceneCacheKey(key), group: SCENES });
/** 三维场景描述在失败记忆 / 在途表里的键（描述本身不进缓存：留着的是建出的场景）。 */
export const sceneDescKey = (key: string): string => `3d:desc:${key}`;
export const partCacheKey = (url: string): string => `3d:part:${url}`;
export const partSlot = (url: string): Slot => ({ key: partCacheKey(url), group: "3d:part" });
