// @ts-expect-error 改编自 three.js 的 EXRLoader（JS，无类型声明），见 exrCore.js 顶部说明
import { parseExr } from "./exrCore.js";

/** 页面唯一的 EXR 解码器：解出 EXR 中任意若干通道的原始平面。
 * half 保持为 `Uint16Array`（不转为 float，与文件逐位一致），float 为 `Float32Array`，uint 为 `Uint32Array`。
 * 行序自上而下（与 three 的纹理相反），按 `data[y * width + x]` 索引。
 *
 * 使用方：本机代理（`transfer/localProxy/worker.ts`）与 EXR worker（`transfer/exrWorker.ts`：本机代理不可用时的显示解码，
 * 以及通道级上传 `transfer/planes.ts` 所需的平面）。 */
export interface ExrPlane {
  name: string;
  type: "half" | "float" | "uint";
  width: number;
  height: number;
  data: Uint16Array | Float32Array | Uint32Array;
}

export interface ExrDecoded {
  width: number; // 数据窗口的宽高（即平面尺寸）
  height: number;
  compression: string; // OpenEXR 压缩方式名称（小写）：none、rle、zips、zip、piz、pxr24、b44、b44a、dwaa、dwab
  dataWindow: [number, number, number, number]; // [x, y, w, h]，取文件自身记录的位置（序列中各帧可以不同）
  displayWindow: [number, number, number, number];
  allChannels: { name: string; type: ExrPlane["type"] }[]; // 文件头中的全部通道（无论是否被请求）
  channels: ExrPlane[]; // 已解出的请求通道
}

const COMPRESSION: Record<string, string> = {
  NO_COMPRESSION: "none", RLE_COMPRESSION: "rle", ZIPS_COMPRESSION: "zips", ZIP_COMPRESSION: "zip", PIZ_COMPRESSION: "piz",
  PXR24_COMPRESSION: "pxr24", B44_COMPRESSION: "b44", B44A_COMPRESSION: "b44a", DWAA_COMPRESSION: "dwaa", DWAB_COMPRESSION: "dwab",
};

interface Box { xMin: number; yMin: number; xMax: number; yMax: number }
const asRect = (b: Box): [number, number, number, number] => [b.xMin, b.yMin, b.xMax - b.xMin + 1, b.yMax - b.yMin + 1];

interface Raw { width: number; height: number; compression: string; dataWindow: Box; displayWindow: Box;
                allChannels: ExrDecoded["allChannels"]; channels: ExrPlane[] }

const shaped = (r: Raw): ExrDecoded => ({
  width: r.width, height: r.height, compression: COMPRESSION[r.compression] ?? r.compression.toLowerCase(),
  dataWindow: asRect(r.dataWindow), displayWindow: asRect(r.displayWindow), allChannels: r.allChannels, channels: r.channels,
});

const bufferOf = (bytes: Uint8Array): ArrayBuffer =>
  (bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
    ? bytes.buffer : bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)) as ArrayBuffer;

/** 同步解码一个 EXR（在 worker 中调用）：`wanted` 为空时返回全部通道，否则仅返回指定通道
 * （解压仍按整块进行，因为 ZIP / PIZ 的一个块内混有全部通道；节省的只是转换开销与内存）。
 * 多部分文件按 `part` 选择其中一部分。遇到不支持的压缩方式或损坏的文件时抛出异常，由调用方负责提示。 */
export function decodeExrPlanes(bytes: Uint8Array, wanted?: string[], part = 0): ExrDecoded {
  return shaped(parseExr(bufferOf(bytes), wanted && wanted.length ? wanted : undefined, part) as Raw);
}

/** 仅读取文件头：通道名、类型、尺寸、压缩方式、数据窗口（只需文件开头的数十 KB）。 */
export function exrHeader(bytes: Uint8Array): Omit<ExrDecoded, "channels"> {
  return shaped(parseExr(bufferOf(bytes), [], 0, true) as Raw);
}

/** half → float32（查表法，表含 2^16 项，仅构建一次）。 */
let HALF: Float32Array | null = null;
export function halfToFloat(h: Uint16Array): Float32Array {
  if (!HALF) {
    HALF = new Float32Array(65536);
    const b = new ArrayBuffer(4), f = new Float32Array(b), u = new Uint32Array(b);
    for (let i = 0; i < 65536; i++) {
      const s = (i >> 15) & 1, e = (i >> 10) & 0x1f, m = i & 0x3ff;
      if (e === 0) f[0] = (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
      else if (e === 31) { u[0] = (s << 31) | 0x7f800000 | (m << 13); }
      else f[0] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
      HALF[i] = f[0];
    }
  }
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) out[i] = HALF[h[i]];
  return out;
}
