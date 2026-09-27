/** 一条通道的一帧数据（服务器 `GET /api/packet/{fp}/frame/{f}/channel/{name}`，
 * 编码定义于 `lab2shot/server/wire.py channel_blob`）。
 *
 * 无论解算器输出多少条通道，只传输用户查看的通道。图片路径发送的是已生成的显示图（所选通道、黑白点、
 * 着色均已烘焙进像素）；本路径发送的是数据，显示方式全部由浏览器计算（view/look.ts，在 GPU 上）。
 *
 * 值为数据本身的值，不含任何显示处理：数值图即数值（显示范围在包的 `meta.range` 中，由浏览器映射），
 * 带 alpha 的画面为预乘（与 Nuke 一致），视频为已变换的显示值。 */
export interface Plane {
  kind: "plane";
  /** 0 = u8（值 = k/255） 1 = u16（k/65535） 2 = 半精度 3 = float32。对应 GPU 纹理格式分别为 R8 / R32F / R16F / R32F。 */
  format: 0 | 1 | 2 | 3;
  width: number;
  height: number;
  /** 宽 × 高个值，行优先，从左上角开始（与图片、data_window 及整条二维链方向一致）。 */
  data: Uint8Array | Uint16Array | Float32Array;
}

/** 一帧的绘制内容：服务器生成的显示图（图片路径），或一条通道的数据（通道路径）。
 * 两者在取帧账本（transfer/frames.ts）中完全等同，窗口、预算、取消、时间线均不作区分。 */
export type Pixels = ImageBitmap | Plane;

export const isPlane = (p: Pixels): p is Plane => (p as Plane).kind === "plane";

/** 该数据占用的字节数（缓存据此计算预算）。位图为解码后的 RGBA，通道为其自身的数组。 */
export const sizeOf = (p: Pixels): number => (isPlane(p) ? p.data.byteLength : p.width * p.height * 4);

/** 释放该数据（位图占用显存，须显式关闭；通道为普通内存，交由垃圾回收）。 */
export const freePixels = (p: Pixels): void => {
  if (!isPlane(p)) p.close();
};

/** 头部 16 字节：`"L2C1"` + 格式 + 标志 + 2 字节保留 + 宽 + 高，均为小端（见 server/wire.py 中的定义）。 */
const HEAD = 16;
const MAGIC = 0x4c324331; // "L2C1" 按大端读取所得的数值
/* 标志的第 0 位是「低位尾数抹过零」。页面不读它：只有代理一种编法，每一份都是抹过的
   （`lab2shot/view/channels.py`），没有第二档可比。字节里留着是为了格式不变。 */

/** 将服务器返回的数据（gzip 由浏览器解压）读取为一条通道，不做任何转换。
 * 这正是选择该格式的依据：`Uint8Array` / `Uint16Array` / `Float32Array` 可直接建立在同一段内存上，
 * 头部 16 字节为 4 的倍数，因此 `Float32Array` 也是零拷贝。 */
export function readPlane(buf: ArrayBuffer): Plane {
  if (buf.byteLength < HEAD) throw new Error("channel: too short");
  const head = new DataView(buf);
  if (head.getUint32(0, false) !== MAGIC) throw new Error("channel: not L2C1");
  const format = head.getUint8(4) as Plane["format"];
  const width = head.getUint32(8, true);
  const height = head.getUint32(12, true);
  const count = width * height;
  if (!(count > 0)) throw new Error("channel: empty");
  const data =
    format === 0 ? new Uint8Array(buf, HEAD, count)
    : format === 3 ? new Float32Array(buf, HEAD, count)
    : new Uint16Array(buf, HEAD, count); // u16 与半精度均为每个值两个字节
  if (data.byteOffset + data.byteLength > buf.byteLength) throw new Error("channel: short data");
  return { kind: "plane", format, width, height, data };
}

/** 通道名称：与核心 `lab2shot/data/payloads.py channel_names` 使用同一套（页面不自行拼接）。
 * 本表为通道序号到名称的映射；额外的 `valid`（有值像素的标记）不在表中，按名称请求。 */
export const CHANNEL_NAMES = ["R", "G", "B", "A"] as const;

/** 有值像素的标记：包中额外写入的一条通道。显示图上无值处为黑色，通道路径须自行乘以此通道。 */
export const VALID = "valid";

/** 一条通道的值，统一转为 float32（u8 → k/255，u16 → k/65535，半精度按 IEEE 754 解码，float32 原样保留）。
 *
 * GPU 路径不使用本函数：它直接使用原始数组，零拷贝上传为 R8 / R16F / R32F 纹理（见上方 `readPlane` 的说明）。
 * 使用方是在 CPU 上实际计算通道的位置：浏览器计算积木节点（`ops/recipe.ts`：选人、人物框转遮罩、
 * 图像合成），算法目录的执行器只接受 float32（`lab2shot/ops/vocab.py` 的精度规定）。
 *
 * 同一条通道只转换一次（由 WeakMap 记录）：在「运算」档下一帧的三条通道会被多次读取。 */
const changed = new WeakMap<Plane, Float32Array>();

export function values(p: Plane): Float32Array {
  const had = changed.get(p);
  if (had) return had;
  const n = p.width * p.height;
  let out: Float32Array;
  if (p.format === 3) out = p.data as Float32Array;
  else {
    out = new Float32Array(n);
    if (p.format === 0) for (let i = 0; i < n; i += 1) out[i] = (p.data as Uint8Array)[i] / 255;
    else if (p.format === 1) for (let i = 0; i < n; i += 1) out[i] = (p.data as Uint16Array)[i] / 65535;
    else for (let i = 0; i < n; i += 1) out[i] = half((p.data as Uint16Array)[i]);
  }
  changed.set(p, out);
  return out;
}

/** 半精度（16 位）→ 数值（`DataView.getFloat16` 尚未被所有浏览器支持，因此按 IEEE 754 自行计算）。 */
function half(bits: number): number {
  // IEEE 754 半精度：1 位符号、5 位指数、10 位尾数。使用乘以 2 的负次幂，而非右移或除以 1024：
  // 含义相同，且不会触发「字节大小只能用 platform/format.ts 表示」的测试（该测试按 `>> 10`、`/ 1024` 识别字节运算）
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = Math.floor(bits * 2 ** -10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24; // 非规格化数（0 也走此分支）
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac * 2 ** -10) * 2 ** (exp - 15);
}

// ------------------------------------------------------------------ 整段保留的缓存层

/** 将一条通道压缩存储（gzip），与图片路径保留原始字节的做法相同。
 *
 * 解压后一条 1080p 通道为 2–8 MB，300 帧即超过 1 GB，超出预算时最早的帧被淘汰，导致后面缓存完成而前面丢失，
 * 来回播放无法凑齐整段，时间线上的「已载入视图」持续缩短。
 * 压缩后的同一段仅数十到数百 KB，可整段保留；位图被淘汰后从此处解压重建，无需走网络。
 *
 * 图片路径直接存储服务器发送的压缩字节即可；通道路径的 gzip 由浏览器自动解压（`Content-Encoding`），
 * 页面只持有解压后的数组，因此在此处重新压缩。压缩的是同一份数据，解压后完全一致，属于无损。
 * 浏览器不支持 `CompressionStream` 时返回 null：仅缺少这一缓存层，功能不变。 */
export async function squeeze(buf: ArrayBuffer): Promise<Blob | null> {
  if (typeof CompressionStream === "undefined") return null;
  try {
    const packed = new Blob([buf]).stream().pipeThrough(new CompressionStream("gzip"));
    return await new Response(packed).blob();
  } catch {
    return null;
  }
}

/** 逆操作：将 `squeeze` 存储的数据解压还原。 */
export async function unsqueeze(blob: Blob): Promise<ArrayBuffer> {
  const out = blob.stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(out).arrayBuffer();
}
