import { ApiError, blob, bytes as fetchBytes } from "../platform/http";
import { api, type Manifest, type Through, type Version } from "../api";
import { cache } from "./cache";
import { originalsFor, stillLooking, type Originals } from "./originals";
import { readPlane, squeeze, unsqueeze, type Pixels, type Plane } from "./plane";
import { isLocal, localPlane, localSource } from "./computed";
import { fileKeyOf, localPicture, proxyReady } from "./localProxy";
import { FROM, channelIdent, identOf, packetKey } from "./ident";
import { genOf } from "./gens";

/** 单帧的来源与解码方式，位于取帧账本（`transfer/frames.ts`）之下一层。
 *
 * 分工：本模块回答「第 1003 帧如何获取」（走哪条路径、何种格式、字节存放位置），
 * 账本回答「当前应取哪些帧、取到何时、先后顺序、何时取消」。
 * 依赖为单向：账本导入本模块，本模块不依赖账本。
 *
 * 一帧可以是两种数据（`Pixels`，见 `transfer/plane.ts`）：服务器生成的显示图，或一条通道的数据本身。
 * 本次查看需要哪些通道、走哪条路径，全项目只由 `transfer/route.ts` 一处决定。 */

/** 帧源：帧号 → 获取方式（undefined 表示该源没有此帧）。
 *
 * 两项职责，两个接口（存储压缩态，不存储解码态）：
 * - `load`：立即需要绘制。已持有字节则直接解码，否则走网络。解码后的位图进入解码层（有预算，会被淘汰）。
 * - `fill`：取回压缩字节但不解码。选中后即开始在后台取回整段。
 *   已持有则不做任何操作。没有 `fill` 的源（用户自选的文件、浏览器计算的结果）本身不经过网络。 */
export type FrameSource = {
  id: string;
  frames: number[];
  load: (frame: number) => (() => Promise<Pixels>) | undefined;
  fill?: (frame: number) => (() => Promise<void>) | undefined;
  /** 本机文件路径：该帧对应的文件（`transfer/localProxy fileKeyOf`），账本据此检查代理是否在磁盘上 */
  keyOf?: (frame: number) => string | undefined;
};

/** 服务器无法提供该帧（不存在，或暂时无法绘制）：账本记录后不再重复请求。 */
export class Gone extends Error {}

/** 本模块唯一的可变状态（与 `transfer/frames.ts` 的 `Fetching` 做法相同）：
 * 在途请求（按键记录，账本取消时按键查找；到达后立即删除），
 * 以及各源中来自本机文件的帧（有上限，见下方 `markLocalSource`）。 */
class Fetches {
  readonly aborters = new Map<string, AbortController>();
  readonly localFrames = new Map<string, { frames: number[]; keyOf?: (frame: number) => string | undefined }>();
  /** 当前仅取字节的在途请求（键 -> 该次请求）：同一帧被舞台、后台填充、跨包预取同时请求时只发一次。
   * 在构造上即有上限：条目只在传输期间存在，到达（成功或失败）后立即删除
   * （与 view/scenePart.ts 的 `asking` 同类，登记于 `tools/rule_counts.py INFLIGHT_REGISTRIES`）。 */
  readonly filling = new Map<string, Promise<void>>();
  /** 服务器无法提供的帧（`Gone`：该帧不存在或该结果已失效）：后台取回不再重复请求。
   *
   * 此项不可缺少：舞台每切换一帧都会重新挂载整段取回的记录（`transfer/frames.ts watchFrames`），
   * 「已取回」表随之重置；若无此项，一个 404 的帧每秒会被请求二十余次。
   * 与取帧账本中的 `Asked.failed` 含义相同，只是那里随窗口变化、此处随源变化。
   * 有上限（满时从最早的条目开始移除：移除后最多再请求一次）。 */
  readonly gone = new Map<string, number>(); // 键 -> 被拒时间：超过 GONE_TTL 后可再次请求（同一指纹重算后不能永久不取）
}

const GONE_MAX = 512;
const GONE_TTL = 30_000; // 「该帧不存在」记录 30 秒：服务器删除后又重算的包（指纹相同）在半分钟后即可重新获取
const isGone = (key: string): boolean => {
  const at = own.gone.get(key);
  if (at === undefined) return false;
  if (Date.now() - at < GONE_TTL) return true;
  own.gone.delete(key);
  return false;
};

const own = new Fetches();
const aborters = own.aborters;

/** 取消该前缀下的全部请求（整个源不再需要：切换了节点图，或该结果已作废）。
 *
 * 不提供取消单帧的操作：窗口移动时不中止滑出窗口的帧（下载到 80% 再丢弃会浪费带宽）。 */
export function abortUnder(prefix: string): void {
  for (const [key, ctrl] of [...aborters]) if (key.startsWith(prefix)) { ctrl.abort(); aborters.delete(key); }
}

/** 解码后的位图一律不预乘 alpha（`premultiplyAlpha: "none"`）。
 *
 * 浏览器默认会对带 alpha 的图像做预乘（颜色先乘以透明度）。绘制到 2D 画布时看不出差异（画布会自行换算），
 * 但 GPU 路径直接得到的是预乘后的数值：同一个像素，2D 路径读到 121，GPU 读到 121×121/255 = 57。
 * 显示计算位于 GPU 上（view/look.ts），预乘会导致计算错误：右侧一路整体相差一个 alpha 倍数
 * （[57,15,24] vs [121,33,51]）。
 * 在创建位图的这一处统一固定，两条路径即可得到相同的数值。 */
const BITMAP: ImageBitmapOptions = { premultiplyAlpha: "none", colorSpaceConversion: "none" };


/** 源 id 中表示「该帧字节来源」一段的起始标记。
 *
 * 源 id 分为两段：前半段为「数据包 · 代理档位」（`fp` + `@512`），
 * 后半段为「来源」（本机原件或服务器代理）。
 * 后半段不可缺少：解码后的帧按完整 id 存储，来源改变而 id 不变时会绘制旧图
 * （见下方 `serverFrames` 的说明）。
 *
 * 压缩字节只与前半段有关：它们是服务器该档代理的字节，与该帧是否从磁盘读取无关。
 * 因此时间线的「已载入视图」按前半段查找字节（`transfer/frames.ts loadedFrames`），
 * 否则一个包只要有本机原件，该行报告的帧就会偏少（不会报错，只是数值偏小）。
 *
 * 使用 `#` 开头是因为它不会出现在 id 的其他位置：包指纹为十六进制，代理档位为 `@512`，
 * 通道名为 R G B A / valid；通道路径的 id 中已含有 `|`（`ch:<fp>|<通道>`），
 * 因此分隔符不能使用 `|`。 */
/** 源 id 的前半段：「数据包 · 计算批次 · 代理档位」，即压缩字节存放的前缀（transfer/ident.ts packetKey）。 */
export const packetPart = (id: string): string => {
  const at = id.indexOf(FROM);
  return at < 0 ? id : id.slice(0, at);
};

/** 一帧已取回压缩字节的键：解码后的位图存放在 `key` 下，字节存放在此键下。
 *
 * 分开存储的原因（缓存会被淘汰，同时不能耗尽浏览器内存）：
 * 解码后的位图与代理大小无关，解码后为宽×高×4（512×288 的代理也有 0.6 MB），
 * 而同一帧的压缩字节仅数十 KB，相差两个数量级。
 * 因此字节整段保留（`small` 档，属于另一个预算，位图再多也不会挤占），位图只保留当前窗口；
 * 位图被淘汰时直接从字节重新解码（数毫秒），不再走网络。
 * 这正是 Nuke / Houdini「第一遍缓存、第二遍实时」得以成立的原因。 */
export const bytesKey = (key: string) => `bytes:${key}`;

/** 区分服务器的响应是「该帧不存在」还是「暂时无法获取」：前者记录后不再请求（Gone），后者稍后重试。 */
const refusal = (e: unknown): unknown =>
  e instanceof ApiError ? (e.status < 500 && e.status !== 429 && e.status !== 408 ? new Gone(String(e.status)) : new Error(String(e.status))) : e;

/** 只取回该帧的压缩字节，不解码。
 *
 * 用户选择某个通道后，视图所有帧的代理即开始取回，不等待播放，同时不能耗尽浏览器内存。
 * 整段取回的是压缩态（一帧数十 KB），存放时不解码；解码后的位图（512×288 也有 0.6 MB，1080p 源
 * 为 8 MB）只保留播放头附近的少量帧，由 `transfer/cache.ts` 的两个预算分别管理。
 * 因此整段取回经由本函数，而非 `fromServer` / `fromChannel`（后两者用于立即绘制）。
 *
 * 已持有该帧（字节或已解码的位图）时不做任何操作，不发出任何请求，
 * 因此用户切回时不会重复取回。 */
export function fillBytes(url: string, key: string, squeezed: boolean): Promise<void> {
  const already = own.filling.get(key);
  if (already) return already;
  if (cache.has(bytesKey(key)) || cache.has(key) || isGone(key)) return Promise.resolve();
  const run = (async () => {
    if (squeezed) {
      // 通道路径：gzip 由浏览器在原生代码中解压，页面只持有解压后的数组，因此自行压缩后再存储
      const buf = await fetchBytes(url).catch((e) => { throw refusal(e); });
      const packed = await squeeze(buf);
      if (packed) cache.keep(bytesKey(key), packed, packed.size, "small");
    } else {
      const got = await blob(url, { headers: { Accept: "image/webp,image/png" } }).catch((e) => { throw refusal(e); });
      cache.keep(bytesKey(key), got, got.size, "small");
    }
  })()
    .catch((e) => {
      if (e instanceof Gone) {   // 服务器无法提供该帧：记录（含时间），后台取回在半分钟内不再请求
        if (own.gone.size >= GONE_MAX) own.gone.delete(own.gone.keys().next().value as string);
        own.gone.set(key, Date.now());
      }
      throw e;
    })
    .finally(() => own.filling.delete(key));
  own.filling.set(key, run);
  return run;
}

/** 当前仅取字节的在途请求数（仅供测试使用：验证完成后该表为空）。 */
export const fillsInFlight = (): number => own.filling.size;

/** A picture fetched from the server (the proxy: lossy WebP or PNG), decoded off the page's thread. */
/** `under`：在途请求所记录的键（默认即 `key`）。图片路径的字节键为「数据包 · 代理档位」前半段（`server_key`），两个源共用一个包
 * 时会产生相同的键：一个源的取消会中止另一个源的在途帧，`abortUnder(源 id)` 也无法找到任何条目；因此按完整源 id 的键记录。 */
function fromServer(url: string, key: string, under = key): () => Promise<ImageBitmap> {
  return async () => {
    const kept = cache.get<Blob>(bytesKey(key));
    if (kept) return await createImageBitmap(kept, BITMAP); // 字节仍在：不走网络，直接解码
    const ctrl = new AbortController();
    aborters.set(under, ctrl);
    try {
      // the server has no such frame (or cannot show it): not requested again; busy or down: requested again when needed
      const bytes = await blob(url, { signal: ctrl.signal, headers: { Accept: "image/webp,image/png" } }).catch((e) => { throw refusal(e); });
      cache.keep(bytesKey(key), bytes, bytes.size, "small"); // 整段保留：一帧数十 KB
      return await createImageBitmap(bytes, BITMAP);
    } finally {
      aborters.delete(under);
    }
  };
}

/** 一条通道的一帧（transfer/plane.ts：数据本身，而非已生成的图）。
 *
 * 与图片路径相同，分为两层：解压后的数据（体积大，随窗口变化，被预算淘汰属正常）与压缩数据
 * （体积小，整段保留，`small` 档，优先级高于解压数据）。被淘汰后从压缩数据解压恢复，不走网络，
 * 「第一遍缓存、第二遍实时」即依赖于此。
 * 唯一区别：图片路径保留的是服务器发送的压缩字节；本路径的 gzip 由浏览器自动解压，
 * 页面无法获得压缩数据，因此在此处自行压缩（无损，plane.ts squeeze）。 */
function fromChannel(url: string, key: string): () => Promise<Plane> {
  return async () => {
    const kept = cache.get<Blob>(bytesKey(key));
    if (kept) return readPlane(await unsqueeze(kept)); // 压缩数据仍在：不走网络
    const ctrl = new AbortController();
    aborters.set(key, ctrl);
    try {
      // the server has no such frame (or cannot show it): not requested again; busy or down: requested again when needed
      const buf = await fetchBytes(url, { signal: ctrl.signal }).catch((e) => { throw refusal(e); });
      const plane = readPlane(buf);
      void squeeze(buf).then((packed) => packed && cache.keep(bytesKey(key), packed, packed.size, "small"));
      return plane;
    } finally {
      aborters.delete(key);
    }
  };
}

/** A picture from the user's own file (PNG / JPG; EXR through exr.ts, `space` being the node's colour space).
 *
 * `rules`：整段序列查询色彩空间时使用的文件名（默认为该帧自身的文件名）。
 * 一段序列只有一个色彩空间，读取节点本身也按第一个文件推断
 * （`lab2shot/nodes/core/input.py:260` `colorspace_for_file(str(first))`），
 * 因此整段使用同一文件名查询，一张表即可。不传入时每帧使用各自的文件名，即每帧一张表（见 `transfer/exr.ts`）。 */
export function fromFile(file: File, space = "", rules = ""): () => Promise<ImageBitmap> {
  return async () => {
    try {
      // 优先使用本机代理（`transfer/localProxy`：本机显示同样不是无损，使用一份本地压缩缓存）：
      // 磁盘上已有则数毫秒即可解码；没有则优先生成。无法生成（浏览器不支持私有文件系统）时才整帧实时解码
      const proxy = await localPicture(file, "rgba", space, rules).catch(() => null);
      if (proxy) return await createImageBitmap(proxy, BITMAP);
      if (file.name.toLowerCase().endsWith(".exr")) return await (await import("./exr")).decodeExr(file, space, rules || file.name);
      return await createImageBitmap(file, BITMAP);
    } catch (e) {
      // 本机文件无法解码（文件损坏、格式不受浏览器支持）是确定性的，重试结果相同。若不记为 Gone，
      // 账本会立即再次请求（transfer/frames.ts pump），一个坏文件即可使页面陷入死循环。记为 Gone：半分钟内不再请求
      throw e instanceof Gone ? e : new Gone(e instanceof Error ? e.message : String(e));
    }
  };
}


const LOCAL_MAX = 64;

/** 该源中来自本机文件且可立即播放的帧：本机代理已在磁盘上的帧（`transfer/localProxy proxyReady`）。
 * 不能将本机的每一帧都视为已持有：否则选择文件后时间线会整条变色，无法反映缓存进度；
 * 尚未生成代理的帧需整帧实时解码，不能实时播放。不支持私有文件系统的浏览器：只计已解码的帧（由账本另行统计）。 */
export const localFramesOf = (id: string): number[] => {
  const mine = own.localFrames.get(id);
  if (!mine) return [];
  return mine.frames.filter((f) => { const k = mine.keyOf?.(f); return !!k && proxyReady(k); });
};

export function markLocalSource(id: string, frames: number[], keyOf?: (frame: number) => string | undefined): void {
  const mine = own.localFrames;
  if (mine.size >= LOCAL_MAX && !mine.has(id)) mine.delete(mine.keys().next().value as string);
  mine.set(id, { frames, keyOf });
}

/** The frames of a 2D result on the server (a packet's picture, a video's frames).
 *
 * 每一帧先查询用户本机是否有该数据的原件（`transfer/originals.ts`）：
 * 有则从用户的原件绘制，无传输、全精度；没有才使用服务器的视图代理。
 * 查询单位是这一份数据（包指纹），因此一个节点的两条输入分别查询：
 * 一条获得原件、另一条没有时，两者各自处理，合成在 GPU 上完成（`view/look.ts`）。
 *
 * 此处只改变该帧字节的来源：帧号、画面尺寸、`data_window`、时间线均以包说明为准，
 * 因此同一帧使用原件或代理时在屏幕上占据相同的矩形，只有清晰度不同
 * （切换节点时画面尺寸不得变化）。 */
export function serverFrames(it: { fp: string; type: string; through?: Through | null }, manifest: Manifest | null): FrameSource | null {
  // 浏览器计算的结果：每一帧由其实时计算（view/useLocal.ts），不经过网络传输
  // （遮罩图与合成图既不在服务器生成，也不回传）
  if (isLocal(it.fp)) return localSource(it.fp);
  if (!manifest) return null;
  const frames = Array.isArray(manifest.meta.frames) ? (manifest.meta.frames as number[]) : [];
  // 代理档位计入 id（缓存键须包含「数据包 · 帧 · 通道 · 代理档位」四项）。管理员更改档位后，服务器发送的是另一份字节，键也随之改变，
  // 不会将旧档位的字节当作新档位使用。用户本机文件不受影响：它是用户磁盘上的原件，不经过网络传输
  const t = tierTag(manifest);
  // 透过带畸变的相机查看时，不得使用本机原件：原件是带畸变的实拍，此时需要服务器按该相机镜头去畸变后的结果。
  // 「透过的相机」同样计入 id 和每一帧的键：键由结构体生成（transfer/ident.ts），不再手工拼接
  const through = it.through ?? null;
  const own = through ? null : originalsFor(it.fp, manifest);
  const head = { fp: it.fp, gen: genOf(it.fp), tier: t, through };
  const server_key = packetKey(head);
  const here = new Set(own?.frames ?? []);
  // 「来源」同样计入 id，与「代理档位」计入 id 的规则相同。
  //
  // 解码后的帧存放在 `${源 id}:${帧}` 下（`transfer/frames.ts`）。查找原件是异步的：画面先用服务器数据
  // 绘制，原件稍后才登记；此时源已更换、`load` 已改为读取本机，但若 id 不变，
  // 账本发现缓存中已有该帧即直接使用旧图，屏幕上将一直是模糊的代理图，既不报错，测试也无法发现。
  // 因此本机帧数及对应的原件均写入 id；更换原件即对应另一组键，不会将旧数据当作新数据使用。
  // 文件组成及解码所用的色彩空间均包含在结构体中（transfer/ident.ts）：两段序列帧数相同、目录名相同、
  // 节点未重算因而 `fp` 也相同时，只能依靠这些字段区分，否则已查看过的数十帧会从缓存中原样返回上一段的画面
  const id = identOf({ ...head, local: own ? { digest: filesDigest(own), where: own.where, frames: own.frames.length, space: own.space } : null });
  // 位于本机的帧（用户选择的文件，或交付写入用户磁盘的文件）：它们在磁盘上，播放时无需等待
  const mine = frames.filter((f) => here.has(f));
  const keyOf = (f: number) => own?.keys?.get(f);
  if (mine.length) markLocalSource(id, mine, keyOf);
  const v = versionOf(it.fp, manifest);
  const server = (f: number) => fromServer(pictureUrl(it, f, through, v), `${server_key}:${f}`, `${id}:${f}`);
  return {
    id,
    frames,
    keyOf,
    load: (f) => {
      if (!frames.includes(f)) return undefined;
      if (!own || !here.has(f)) return server(f);
      // 暂时无法获取原件（文件被移动、权限失效、包中无此条目）时照常使用服务器数据；
      // 找不到不属于错误，不报错、不提示
      return async () => {
        const file = await own.fileOf(f).catch(() => null);
        // `own.name`：整段按同一文件名查询色彩空间（一张显示变换表，而非每帧一张，见 `fromFile`）
        return file ? await fromFile(file, own.space, own.name)() : await server(f)();
      };
    },
    fill: (f) => {
      // 本机文件在磁盘上，无需取回；仍在查找原件时也暂不取回：查找原件需两三百毫秒，
      // 在此期间整段代理可能已取回一半，而该数据本无需传输任何字节（`transfer/originals.ts lookingFor`）
      if (!frames.includes(f) || here.has(f) || (!through && stillLooking(it.fp))) return undefined;
      return () => fillBytes(pictureUrl(it, f, through), `${server_key}:${f}`, false);
    },
  };
}

/** 一份本机原件的文件组成：各帧文件键（`transfer/localProxy fileKeyOf`：名称、大小、修改时间）按帧序拼接后的
 * 短散列（FNV-1a）。更换文件即得到不同的值，即使帧数、目录名、包指纹均未改变。没有文件键时（未登记 `keys`）返回空串。 */
function filesDigest(own: Originals): string {
  if (!own.keys?.size) return "";
  return digestOf(own.frames.map((f) => `${f}=${own.keys!.get(f) ?? ""}`));
}

/** 若干字符串的短散列（FNV-1a），用作缓存键的一部分。 */
function digestOf(parts: readonly string[]): string {
  let h = 2166136261;
  for (const p of parts) for (const ch of `${p};`) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return (h >>> 0).toString(36);
}

/** 该帧走图片路径的地址（视频与序列图为两条路由，其余完全相同）。 */
const pictureUrl = (it: { fp: string; type: string }, frame: number, through: Through | null = null, v?: Version): string =>
  it.type === "video" ? api.videoFrameUrl(it.fp, frame, v) : api.packetFrameUrl(it.fp, frame, through ?? undefined, v);

/** 地址中附带的版本信息（api.Version）：与缓存键（ident.ts）使用同一份批次和档位，地址的标识即缓存键的标识。 */
export const versionOf = (fp: string, manifest: Manifest | null): Version => ({ g: genOf(fp), px: manifest?.proxy?.px });
const pxOfTag = (tier: string): number | undefined => (tier.startsWith("@") ? Number(tier.slice(1)) || undefined : undefined);

/** 该包各帧在视图中的代理档位，以键片段形式返回（`@512`）。包说明未到时不附带，
 * 到达后成为另一个 id 并重新获取一次，这优于用错误档位的字节绘制。 */
export const tierTag = (manifest: Manifest | null): string => (manifest?.proxy ? `@${manifest.proxy.px}` : "");

/** 一条通道的所有帧（按通道取数路径）：`name` 为通道名（R G B A，或 `valid`）。
 *
 * 与 `serverFrames` 属于同一账本中的同类数据，窗口、预算、取消、时间线均不作区分
 * （缓存键即「包指纹 + 帧 + 通道」）。
 * 代理档位计入 id，因此管理员更改档位即对应另一组键，不会将旧档位的字节当作新档位使用；切换离开再切回时使用缓存中
 * 的数据，无需再走网络。 */
/** 单条通道帧源的 id：「数据包 · 通道 · 代理档位」（帧号由账本附加在后）。
 * 后台整段填充与跨包预取按同一 id 存储（transfer/frames.ts、transfer/prefetchLive.ts），
 * 因此切换离开再切回时命中缓存，无需再传输任何字节。 */
export const channelId = channelIdent;

export function channelFrames(fp: string, name: string, frames: number[], tier: string): FrameSource {
  // 浏览器计算的结果：该通道由其实时计算，不经过网络传输（transfer/computed.ts）
  const mine = isLocal(fp) ? localPlane(fp, name) : null;
  if (mine) return mine;
  const id = channelId(fp, name, tier);
  const have = new Set(frames);
  const v: Version = { g: genOf(fp), px: pxOfTag(tier) };
  return {
    id,
    frames,
    load: (f) => (have.has(f) ? fromChannel(api.channelUrl(fp, f, name, v), `${id}:${f}`) : undefined),
    fill: (f) => (have.has(f) ? () => fillBytes(api.channelUrl(fp, f, name, v), `${id}:${f}`, true) : undefined),
  };
}

/** 用户在本标签页中选择的文件本身即构成一个帧源：
 * 帧到文件的对应关系由浏览器自行建立，无需等待服务器。`space`：EXR 解码所用的色彩空间。 */
export function pickedFrames(given: string, item: { kind: string; frames: number[]; files: File[] }, space = "", still = 1001): FrameSource {
  const frames = item.kind === "sequence" ? item.frames : [still];
  // 文件组成同样须计入 id：若 id 只有 `picked:<节点>`，更换文件后 id 完全相同，解码后的帧按 `${id}:${帧}`
  // 从缓存原样取出，显示的将是上一段的画面（读取另一个序列或重新选择均无法更新）。
  // 将文件键（名称、大小、修改时间）拼接后的短散列计入 id，更换文件即对应另一组缓存键
  const id = `${given}#${digestOf(item.files.map(fileKeyOf))}#${space}`;  // 色彩空间也计入键：同一文件换用不同色彩空间即为另一张图
  // 这些帧位于用户本机磁盘上，无需等待（规则见 `transfer/frames.ts loadedFrames` 上方说明）。
  // 若遗漏此句：位图被预算淘汰后即判为「该帧不在浏览器中」，播放器（`editor/Timeline.tsx`
  // 的 `keepingUp`）将停下来等待一个本就在磁盘上的帧，且不报错。
  const keyOf = (f: number) => { const i = item.kind === "sequence" ? item.frames.indexOf(f) : 0; const file = item.files[i]; return file ? fileKeyOf(file) : undefined; };
  markLocalSource(id, frames, keyOf);
  // 整段按同一文件名查询色彩空间：一段序列只有一个色彩空间（见 `fromFile` 的 `rules`）
  const rules = item.files[0]?.name ?? "";
  return {
    id,
    frames,
    keyOf,
    load: (f) => {
      const i = item.kind === "sequence" ? item.frames.indexOf(f) : 0;
      return i < 0 || !item.files[i] ? undefined : fromFile(item.files[i], space, rules);
    },
    // 此处有意不提供 `fill`：`fill` 用于取回压缩字节，而这些文件本身就在用户机器上，
    // 无需经过网络。预先准备的步骤位于账本的窗口中（`transfer/frames.ts watchFrames` →
    // `want` → 一次解码 6 帧，向前解码直至超出该档预算），不在此处。
  };
}

/** 将服务器数据与本机数据合并为一个帧源：已持有的数据不再重复获取。
 *
 * - 选择的是 PNG / JPEG：服务器提供的显示图即为该文件本身（与 server/packets.py `_shown_blobs` 的说明一致），
 *   因此始终使用本机文件，无需经过网络传输；
 * - 选择的是 EXR：服务器按 OCIO 执行显示变换，与浏览器自行解码的结果略有差异，因此只在服务器数据到达之前
 *   使用本机文件，到达后改用服务器数据，以保证所见即所得。
 *
 * 显示读取序列节点时不能等待上传到服务器再传回：若须等服务器的包说明（manifest 中每帧的 sha）返回后
 * 才能确定能否使用本机文件，在限速 30 Mbps 时切换过去的瞬间画面会出现空白。 */
export function localFirst(server: FrameSource | null, local: FrameSource | null, exact: boolean): FrameSource | null {
  if (!local) return server;
  if (!server) return local;
  const id = `${server.id}${FROM}${exact ? "本机优先" : "服务器优先"} ${local.id}`;
  const here = new Set(local.frames);
  // 合并后是一个新源，因此「哪些帧在本机」「整段是否需要取回」须在新 id 上重新声明。
  // 遗漏任何一项都不会报错：① `markLocalSource` 记录在 `server.id` / `local.id` 上，合并后的 id
  // 与二者均不同（`FROM` 段），不重新记录则 `loadedFrames` 不认任何本机帧，播放又会等待磁盘；
  // ② 只提供 `load` 而丢弃 `server.fill`，则「选中即整段」在该路径上完全失效
  //    （`transfer/fill.ts fillWhole` 首句即为 `if (!source.fill) return`）。
  // 标识改变后，挂在旧标识上的记录须一并迁移。
  // `exact` 决定哪一半属于本机：PNG / JPEG 以本机文件为准，因此其持有的帧都计为本机帧且无需取回；
  // EXR 以服务器数据为准，本机文件只是替代，这些帧不计为本机帧（实际绘制时走服务器路径）。
  if (exact) markLocalSource(id, server.frames.filter((f) => here.has(f)), local.keyOf);
  return {
    // 「来源」同样计入 id（与 `serverFrames` 规则相同，见上方 `FROM` 的说明）：
    // 该路径以哪一方为准由 `exact` 决定：PNG / JPEG 始终使用本机文件，EXR 只在服务器数据到达前使用本机文件。
    // `exact` 翻转后同一帧绘制的是另一份字节；若 id 不随之改变，账本会将上一份已解码的位图原样返回，
    // 且不会报错（与 `serverFrames` 的问题相同）
    id,
    frames: server.frames,
    load: (f) => (exact ? local.load(f) ?? server.load(f) : server.load(f) ?? local.load(f)),
    // 以本机文件为准的帧不取回（磁盘上已有）；其余照常由服务器路径整段取回
    fill: server.fill && ((f) => (exact && here.has(f) ? undefined : server.fill!(f))),
  };
}

/** Frames fetched by address (a benchmark sample's frames: the files' own bytes on the server). */
export function urlFrames(id: string, frames: number[], url: (frame: number) => string): FrameSource {
  const have = new Set(frames);
  return { id, frames, load: (f) => (have.has(f) ? fromServer(url(f), `${id}:${f}`) : undefined) };
}

/** The frames of a source currently in the browser, whether fetched by the view or by the prefetcher (one cache, one key
 * for both): what a ruler marks as loaded and what the timeline's 已载入视图 row reads, so the row fills as prefetch proceeds. */
