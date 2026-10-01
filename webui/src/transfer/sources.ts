import type { Manifest } from "../api";
import { cache } from "../platform/cache";
import { originalsFor, stillLooking, type Originals } from "./originals";
import { genOf } from "./gens";
import { readPlane, type Pixels, type Plane } from "./plane";
import { fileKeyOf, localPicture, proxyReady } from "./localProxy";
import { channelId, channelUrl, frameKey, mergedId, packetKey, pickedId, pictureUrl, sourceId, tierOf, type Through, type Tier } from "./frameKey";
import { Gone, frameBytes, planeBytes } from "./frameStore";
import { shortHash } from "../platform/digest";

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
 *   已持有则不做任何操作。没有 `fill` 的源（用户自选的文件）本身不经过网络。 */
export type FrameSource = {
  id: string;
  frames: number[];
  load: (frame: number) => (() => Promise<Pixels>) | undefined;
  fill?: (frame: number) => (() => Promise<void>) | undefined;
  /** 本机文件路径：该帧对应的文件（`transfer/localProxy fileKeyOf`），账本据此检查代理是否在磁盘上 */
  keyOf?: (frame: number) => string | undefined;
  /** 本机文件的显示变换由什么定（节点的色彩空间与查规则用的文件名）：本机代理按显示变换分份存，就绪也按它认 */
  display?: Display;
  /** 一张静止图（不是序列）：任何一帧都画它，账本按一个固定的槽取、存、解码一次 */
  still?: boolean;
};

type Display = { space: string; rules: string };

export { Gone };

/** 本模块唯一的可变状态（与 `transfer/frames.ts` 的 `Fetching` 做法相同）：
 * 在途请求（按键记录，账本取消时按键查找；到达后立即删除），
 * 以及各源中来自本机文件的帧（有上限，见下方 `markLocalSource`）。 */
class Fetches {
  readonly aborters = new Map<string, AbortController>();
  readonly localFrames = new Map<string, { frames: number[]; keyOf?: (frame: number) => string | undefined; display?: Display }>();
}

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


/** 从服务器取来的一张图（代理：有损 WebP 或 PNG），在页面主线程之外解码。
 *
 * `under`：在途请求所记录的键（默认即 `key`）。图片路径的字节键为「数据包 · 代理档位」前半段（`packetKey`），两个源共用一个包
 * 时会产生相同的键：一个源的取消会中止另一个源的在途帧，`abortUnder(源 id)` 也无法找到任何条目；因此按完整源 id 的键记录。 */
function fromServer(url: string, at: { id: string; frame: number }, under: string): () => Promise<ImageBitmap> {
  return async () => {
    const ctrl = new AbortController();
    aborters.set(under, ctrl);
    try {
      return await createImageBitmap(await frameBytes(at, url, "picture", ctrl.signal), BITMAP);
    } finally {
      aborters.delete(under);
    }
  };
}

/** 一条通道的一帧（transfer/plane.ts：数据本身，而非已生成的图）。与图片路径同一条取字节的路（transfer/frameStore.ts frameBytes），
 * 区别只在字节的样子：通道的 gzip 由浏览器解开，存的是自行压缩的（被淘汰后从它解压恢复，不走网络）。 */
function fromChannel(url: string, at: { id: string; frame: number }): () => Promise<Plane> {
  const key = frameKey(at.id, at.frame);
  return async () => {
    const ctrl = new AbortController();
    aborters.set(key, ctrl);
    try {
      return readPlane(await planeBytes(await frameBytes(at, url, "channel", ctrl.signal)));
    } finally {
      aborters.delete(key);
    }
  };
}

/** 整段取回的一帧：只要字节，不解码（已持有字节或位图时什么都不发）。 */
function fillBytes(url: string, at: { id: string; frame: number }, carry: "picture" | "channel"): Promise<void> {
  if (cache.has(frameKey(at.id, at.frame))) return Promise.resolve();
  return frameBytes(at, url, carry, undefined, "later").then(() => undefined);
}

/** 使用者自己文件里的一张图（PNG / JPG；EXR 经 exr.ts 解码，`space` 为节点的色彩空间）。
 *
 * `rules`：整段序列查询色彩空间时使用的文件名（默认为该帧自身的文件名）。
 * 一段序列只有一个色彩空间，读取节点本身也按第一个文件推断
 * （`lab2shot/nodes/core/input.py` 中的 `colorspace_for_file(str(first))`），
 * 因此整段使用同一文件名查询，一张表即可。不传入时每帧使用各自的文件名，即每帧一张表（见 `transfer/exr.ts`）。 */
function fromFile(file: File, space = "", rules = ""): () => Promise<ImageBitmap> {
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
  return mine.frames.filter((f) => { const k = mine.keyOf?.(f); return !!k && !!mine.display && proxyReady(k, mine.display.space, mine.display.rules); });
};

function markLocalSource(id: string, frames: number[], keyOf?: (frame: number) => string | undefined, display?: Display): void {
  const mine = own.localFrames;
  if (mine.size >= LOCAL_MAX && !mine.has(id)) mine.delete(mine.keys().next().value as string);
  mine.set(id, { frames, keyOf, display });
}

/** 服务器上一个二维结果的各帧（数据包的显示图、视频的各帧）。
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
  if (!manifest) return null;
  const frames = Array.isArray(manifest.meta.frames) ? (manifest.meta.frames as number[]) : [];
  // 代理档位计入 id（缓存键须包含「数据包 · 帧 · 通道 · 代理档位」四项）。管理员更改档位后，服务器发送的是另一份字节，键也随之改变，
  // 不会将旧档位的字节当作新档位使用。用户本机文件不受影响：它是用户磁盘上的原件，不经过网络传输
  const tier = tierOf(manifest);
  // 透过带畸变的相机查看时，不得使用本机原件：原件是带畸变的实拍，此时需要服务器按该相机镜头去畸变后的结果。
  // 「透过的相机」同样计入 id 和每一帧的键：键由结构体生成（transfer/frameKey.ts），不手工拼接
  const through = it.through ?? null;
  const own = through ? null : originalsFor(it.fp, manifest);
  // 代次在建源时取一份快照：键（id）和每一帧的地址都用它，不会一个是建源时的、一个是发请求时的
  const g = genOf(it.fp);
  const head = { fp: it.fp, g, tier, through };
  const packetAt = packetKey(head);
  const here = new Set(own?.frames ?? []);
  // 「来源」同样计入 id，与「代理档位」计入 id 的规则相同。
  //
  // 解码后的帧存放在 `${源 id}:${帧}` 下（`transfer/frames.ts`）。查找原件是异步的：画面先用服务器数据
  // 绘制，原件稍后才登记；此时源已更换、`load` 已改为读取本机，但若 id 不变，
  // 账本发现缓存中已有该帧即直接使用旧图，屏幕上将一直是模糊的代理图，且不报错。
  // 因此本机帧数及对应的原件均写入 id；更换原件即对应另一组键，不会将旧数据当作新数据使用。
  // 文件组成及解码所用的色彩空间均包含在结构体中（transfer/frameKey.ts）：两段序列帧数相同、目录名相同、
  // 节点未重算因而 `fp` 也相同时，只能依靠这些字段区分，否则已查看过的数十帧会从缓存中原样返回上一段的画面
  const id = sourceId({ ...head, local: own ? { digest: filesDigest(own), where: own.where, frames: own.frames.length, space: own.space } : null });
  // 位于本机的帧（用户选择的素材文件）：它们在磁盘上，播放时无需等待
  const mine = frames.filter((f) => here.has(f));
  const keyOf = (f: number) => own?.keys?.get(f);
  const display = own ? { space: own.space, rules: own.name ?? "" } : undefined;
  if (mine.length) markLocalSource(id, mine, keyOf, display);
  const server = (f: number) => fromServer(pictureUrl({ ...it, g }, f, tier, through), { id: packetAt, frame: f }, frameKey(id, f));
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
      return () => fillBytes(pictureUrl({ ...it, g }, f, tier, through), { id: packetAt, frame: f }, "picture");
    },
  };
}

/** 一份本机原件的文件组成：各帧文件键（`transfer/localProxy fileKeyOf`：名称、大小、修改时间）按帧序拼接后的
 * 短散列（FNV-1a）。更换文件即得到不同的值，即使帧数、目录名、包指纹均未改变。没有文件键时（未登记 `keys`）返回空串。 */
function filesDigest(own: Originals): string {
  if (!own.keys?.size) return "";
  return digestOf(own.frames.map((f) => `${f}=${own.keys!.get(f) ?? ""}`));
}

/** 若干字符串的短散列（platform/digest.ts），用作缓存键的一部分。 */
const digestOf = (parts: readonly string[]): string => shortHash(parts.map((p) => `${p};`).join(""));

/** 一条通道的所有帧（按通道取数路径）：`name` 为通道名（R G B A，或 `valid`）。
 *
 * 与 `serverFrames` 属于同一账本中的同类数据，窗口、预算、取消、时间线均不作区分
 * （缓存键即「包指纹 + 帧 + 通道」）。
 * 代理档位计入 id，因此管理员更改档位即对应另一组键，不会将旧档位的字节当作新档位使用；切换离开再切回时使用缓存中
 * 的数据，无需再走网络。 */
export function channelFrames(fp: string, name: string, frames: number[], tier: Tier | null): FrameSource {
  const g = genOf(fp); // 同 serverFrames：键和地址出自同一份代次
  const id = channelId(fp, name, tier, g);
  const have = new Set(frames);
  return {
    id,
    frames,
    load: (f) => (have.has(f) ? fromChannel(channelUrl(fp, f, name, tier, g), { id, frame: f }) : undefined),
    fill: (f) => (have.has(f) ? () => fillBytes(channelUrl(fp, f, name, tier, g), { id, frame: f }, "channel") : undefined),
  };
}

/** 用户在本标签页中选择的文件本身即构成一个帧源：
 * 帧到文件的对应关系由浏览器自行建立，无需等待服务器。`space`：EXR 解码所用的色彩空间。 */
export function pickedFrames(node: string, item: { kind: string; frames: number[]; files: File[] }, space: string, timeline: readonly number[]): FrameSource {
  // 一张静止图在时间线的每一帧都是它：帧表是时间线的整段（不随当前帧变，源不必每走一帧重建）
  const frames = item.kind === "sequence" ? item.frames : [...timeline];
  // 文件组成同样须计入 id：若 id 只有节点，更换文件后 id 完全相同，解码后的帧按 `${id}:${帧}` 从缓存原样取出，
  // 显示的将是上一段的画面（读取另一个序列或重新选择均无法更新）
  const id = pickedId({ node, digest: digestOf(item.files.map(fileKeyOf)), space });
  // 这些帧位于用户本机磁盘上，无需等待（规则见 `transfer/frames.ts loadedFrames` 上方说明）。
  // 若遗漏此句：位图被预算淘汰后即判为「该帧不在浏览器中」，播放器（`editor/Timeline.tsx`
  // 的 `keepingUp`）将停下来等待一个本就在磁盘上的帧，且不报错。
  const keyOf = (f: number) => { const i = item.kind === "sequence" ? item.frames.indexOf(f) : 0; const file = item.files[i]; return file ? fileKeyOf(file) : undefined; };
  // 整段按同一文件名查询色彩空间：一段序列只有一个色彩空间（见 `fromFile` 的 `rules`）
  const rules = item.files[0]?.name ?? "";
  markLocalSource(id, frames, keyOf, { space, rules });
  return {
    id,
    frames,
    keyOf,
    display: { space, rules },
    // 一张静止图：时间线上任何一帧都是它，取帧账本只解码、只存一份（transfer/frames.ts slotOf）
    still: item.kind !== "sequence",
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
  const id = mergedId({ server: server.id, local: local.id, exact });
  const here = new Set(local.frames);
  // 合并后是一个新源，因此「哪些帧在本机」「整段是否需要取回」须在新 id 上重新声明。
  // 遗漏任何一项都不会报错：① `markLocalSource` 记录在 `server.id` / `local.id` 上，合并后的 id
  // 与二者均不同（`FROM` 段），不重新记录则 `loadedFrames` 不认任何本机帧，播放又会等待磁盘；
  // ② 只提供 `load` 而丢弃 `server.fill`，则「选中即整段」在该路径上完全失效
  //    （`transfer/fill.ts fillWhole` 首句即为 `if (!source.fill) return`）。
  // 标识改变后，挂在旧标识上的记录须一并迁移。
  // `exact` 决定哪一半属于本机：PNG / JPEG 以本机文件为准，因此其持有的帧都计为本机帧且无需取回；
  // EXR 以服务器数据为准，本机文件只是替代，这些帧不计为本机帧（实际绘制时走服务器路径）。
  if (exact) markLocalSource(id, server.frames.filter((f) => here.has(f)), local.keyOf, local.display);
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

/** 按地址取的帧（运行中节点已写出的帧，view/partial.ts）。 */
export function urlFrames(id: string, frames: number[], url: (frame: number) => string): FrameSource {
  const have = new Set(frames);
  return { id, frames, load: (f) => (have.has(f) ? fromServer(url(f), { id, frame: f }, frameKey(id, f)) : undefined) };
}
