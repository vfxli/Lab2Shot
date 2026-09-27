/// <reference lib="webworker" />
/** 生成一帧的本机代理：解码用户自己的文件，将每一层缩放到指定档位并压缩，
 * 返回字节；页面一侧将其写入私有文件系统（`store.ts`）。
 *
 * - EXR：`transfer/exr/decode.ts` 一次解出全部通道（ZIP / PIZ 的一个块内混有所有通道，解压是主要开销，
 *   多一条通道只增加缩放与编码的数毫秒）。颜色层（含 R、G、B）经显示变换生成一张 8 位 WebP；
 *   其余每条通道生成一张半精度平面，采用 `L2C1` 格式（与服务器发送的通道完全相同，`transfer/plane.ts readPlane` 可直接读取）后再 gzip。
 * - PNG / JPG：由浏览器解码、缩放后生成 WebP；其像素本身处于显示空间，不经过显示变换。 */
import { workerAnswers } from "../../platform/work";
import type { Lut } from "../../ops/lut";
import { decodeExrPlanes, exrHeader, halfToFloat, type ExrPlane } from "../exr/decode";

export interface LayerSpec { name: string; channels: string[] }

export interface ProxyAsk {
  kind: "exr" | "image";
  bytes: Uint8Array;
  tier: number; // 长边的最大像素数
  quality: number; // WebP 质量 0..1
  lut: Lut | null; // EXR 的显示变换（PNG / JPG 不用）
  layers: LayerSpec[]; // EXR 的层（申报时由服务器读出；为空时按通道名自行分层）
  pictures: boolean; // 是否生成颜色层的显示图
  planes: string[]; // 需生成半精度平面的通道（第一遍不生成：只生成显示图，数值通道在用户查看时按需生成；该分支目前尚无调用方，见 `index.ts localPlane`）
  // 2K 22 通道全部生成时一帧 12 MB、耗时 0.7 秒，200 帧需 2.4 GB、两分多钟；只生成显示图快十倍、体积小五十倍
}

export interface ProxyAnswer {
  width: number; // 缩放后的尺寸
  height: number;
  ms: { decode: number; prep: number; encode: number }; // 各步骤耗时（供开发者计时）
  pictures: { name: string; webp: Uint8Array }[]; // 颜色层：`<层>.webp`
  planes: { name: string; gz: Uint8Array }[]; // 其余通道：`<通道>.l2c1.gz`
}

const WEBP = "image/webp";

/** 缩放至长边 `tier`：按整数倍做面积平均（不放大）。 */
function shrink(src: Float32Array, w: number, h: number, ow: number, oh: number): Float32Array {
  if (ow === w && oh === h) return src;
  const out = new Float32Array(ow * oh);
  for (let y = 0; y < oh; y++) {
    const y0 = Math.floor((y * h) / oh), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * h) / oh));
    for (let x = 0; x < ow; x++) {
      const x0 = Math.floor((x * w) / ow), x1 = Math.max(x0 + 1, Math.floor(((x + 1) * w) / ow));
      let s = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) s += src[yy * w + xx];
      out[y * ow + x] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

const fitted = (w: number, h: number, tier: number): [number, number] => {
  const f = Math.min(1, tier / Math.max(w, h));
  return [Math.max(1, Math.round(w * f)), Math.max(1, Math.round(h * f))];
};

/** float32 → 半精度（IEEE 754，就近舍入到偶数）。 */
function floatToHalf(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  const b = new ArrayBuffer(4), f = new Float32Array(b), u = new Uint32Array(b);
  for (let i = 0; i < src.length; i++) {
    f[0] = src[i];
    const x = u[0];
    const sign = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 112;
    let m = x & 0x7fffff;
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      m = (m | 0x800000) >> (1 - e);
      if (m & 0x1000) m += 0x2000;
      out[i] = sign | (m >> 13);
    } else if (e === 0x8f) {
      out[i] = sign | 0x7c00 | (m ? 0x200 : 0); // inf / nan
    } else {
      if (m & 0x1000) { m += 0x2000; if (m & 0x800000) { m = 0; e += 1; } }
      out[i] = e >= 31 ? sign | 0x7c00 : sign | (e << 10) | (m >> 13);
    }
  }
  return out;
}

/** `L2C1` 头 + 半精度数据（即 `transfer/plane.ts readPlane` 读取的格式，格式号 2 表示半精度）。 */
function l2c1Half(half: Uint16Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(16 + half.byteLength);
  const head = new DataView(out.buffer);
  head.setUint32(0, 0x4c324331, false);
  head.setUint8(4, 2);
  head.setUint32(8, w, true);
  head.setUint32(12, h, true);
  out.set(new Uint8Array(half.buffer, half.byteOffset, half.byteLength), 16);
  return out;
}

async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new CompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function webpOf(rgba: Uint8ClampedArray, w: number, h: number, quality: number): Promise<Uint8Array> {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d")!;
  ctx.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
  const blob = await canvas.convertToBlob({ type: WEBP, quality });
  return new Uint8Array(await blob.arrayBuffer());
}

/** 对整张图查显示变换表（与 `ops/lut.ts lookup` 使用同一张表、同一套三线性插值，只是批量执行，不逐像素创建闭包：
 * 逐像素调用 `lookup` 处理一张 1024² 图需一秒以上，批量执行只需数十毫秒）。结果写入 `rgba` 的 R G B（0..255）。 */
function lookupAll(lut: Lut, r: Float32Array, g: Float32Array, b: Float32Array, rgba: Uint8ClampedArray): void {
  const n = r.length;
  if (lut.mode === "raw") {
    for (let i = 0; i < n; i++) { rgba[i * 4] = r[i] * 255; rgba[i * 4 + 1] = g[i] * 255; rgba[i * 4 + 2] = b[i] * 255; }
    return;
  }
  const N = lut.size - 1, s1 = lut.size, s2 = s1 * s1, L = lut.data, lo = lut.lo, span = lut.hi - lut.lo, floor = 2 ** lo;
  const k = 255 / 65535;
  for (let i = 0; i < n; i++) {
    let fr = ((Math.log2(r[i] > floor ? r[i] : floor) - lo) / span) * N; fr = fr < 0 ? 0 : fr > N ? N : fr;
    let fg = ((Math.log2(g[i] > floor ? g[i] : floor) - lo) / span) * N; fg = fg < 0 ? 0 : fg > N ? N : fg;
    let fb = ((Math.log2(b[i] > floor ? b[i] : floor) - lo) / span) * N; fb = fb < 0 ? 0 : fb > N ? N : fb;
    let r0 = fr | 0, g0 = fg | 0, b0 = fb | 0;
    if (r0 > N - 1) r0 = N - 1; if (g0 > N - 1) g0 = N - 1; if (b0 > N - 1) b0 = N - 1;
    const dr = fr - r0, dg = fg - g0, db = fb - b0;
    const base = (r0 + g0 * s1 + b0 * s2) * 3;
    const o = i * 4;
    for (let c = 0; c < 3; c++) {
      const p000 = L[base + c], p100 = L[base + 3 + c];
      const p010 = L[base + s1 * 3 + c], p110 = L[base + s1 * 3 + 3 + c];
      const p001 = L[base + s2 * 3 + c], p101 = L[base + s2 * 3 + 3 + c];
      const p011 = L[base + (s1 + s2) * 3 + c], p111 = L[base + (s1 + s2) * 3 + 3 + c];
      const c00 = p000 * (1 - dr) + p100 * dr, c10 = p010 * (1 - dr) + p110 * dr;
      const c01 = p001 * (1 - dr) + p101 * dr, c11 = p011 * (1 - dr) + p111 * dr;
      rgba[o + c] = ((c00 * (1 - dg) + c10 * dg) * (1 - db) + (c01 * (1 - dg) + c11 * dg) * db) * k;
    }
  }
}

const asFloat = (p: ExrPlane): Float32Array =>
  p.type === "half" ? halfToFloat(p.data as Uint16Array) : p.type === "float" ? (p.data as Float32Array) : Float32Array.from(p.data as Uint32Array);

/** 无申报信息时按通道名分层：`left.R` 属于 `left`，`R` 属于 `rgba`（与服务器 `data/layers.py group` 约定一致的简化版）。 */
function groupByName(names: string[]): LayerSpec[] {
  const by = new Map<string, string[]>();
  for (const n of names) {
    const dot = n.lastIndexOf(".");
    const layer = dot < 0 ? "rgba" : n.slice(0, dot);
    (by.get(layer) ?? by.set(layer, []).get(layer)!).push(n);
  }
  return [...by].map(([name, channels]) => ({ name, channels }));
}

const short = (n: string) => n.slice(n.lastIndexOf(".") + 1).toUpperCase();
const isColour = (l: LayerSpec) => ["R", "G", "B"].every((c) => l.channels.some((n) => short(n) === c));

async function exr(ask: ProxyAsk): Promise<ProxyAnswer> {
  // 只读取本次需要的通道：颜色层的 R G B A（需生成显示图时）及需生成平面的通道。解压仍按整块进行，
  // 但逐像素读取和内存只用于所需通道（2K 22 通道全部读取时一帧需 3.9 秒，主要耗在逐像素读取上）
  const wanted = new Set<string>(ask.planes);
  if (ask.pictures) {
    const head = exrHeader(ask.bytes).allChannels.map((c) => c.name);
    const layers = ask.layers.length ? ask.layers : groupByName(head);
    for (const l of layers) if (isColour(l)) for (const n of l.channels) if (["R", "G", "B", "A"].includes(short(n))) wanted.add(n);
  }
  const ms = { decode: 0, prep: 0, encode: 0 };
  const t0 = performance.now();
  const d = decodeExrPlanes(ask.bytes, [...wanted]);
  ms.decode = performance.now() - t0;
  const [ow, oh] = fitted(d.width, d.height, ask.tier);
  const byName = new Map(d.channels.map((c) => [c.name, c]));
  const layers = ask.layers.length ? ask.layers : groupByName(d.channels.map((c) => c.name));
  const pictures: ProxyAnswer["pictures"] = [];
  const planes: ProxyAnswer["planes"] = [];
  const shrunk = new Map<string, Float32Array>();
  const small = (name: string): Float32Array | null => {
    const had = shrunk.get(name);
    if (had) return had;
    const p = byName.get(name);
    if (!p) return null;
    const s = shrink(asFloat(p), d.width, d.height, ow, oh);
    shrunk.set(name, s);
    return s;
  };
  const wantPlane = new Set(ask.planes);
  for (const layer of layers) {
    const colour = isColour(layer);
    if (colour && ask.pictures) {
      const find = (c: string) => layer.channels.find((n) => short(n) === c);
      const r = small(find("R")!), g = small(find("G")!), b = small(find("B")!);
      const aName = find("A");
      const a = aName ? small(aName) : null;
      if (r && g && b) {
        const rgba = new Uint8ClampedArray(ow * oh * 4);
        if (ask.lut) lookupAll(ask.lut, r, g, b, rgba);
        else for (let i = 0; i < ow * oh; i++) { rgba[i * 4] = r[i] * 255; rgba[i * 4 + 1] = g[i] * 255; rgba[i * 4 + 2] = b[i] * 255; }
        for (let i = 0; i < ow * oh; i++) rgba[i * 4 + 3] = a ? Math.max(0, Math.min(1, a[i])) * 255 : 255;
        const t2 = performance.now();
        pictures.push({ name: layer.name, webp: await webpOf(rgba, ow, oh, ask.quality) });
        ms.encode += performance.now() - t2;
      }
    }
    // 颜色层中 R、G、B 以外的通道（如 A）及非颜色层的每条通道：生成半精度平面。
    // 无需缩放（源尺寸不超过该档位）且本身为半精度的通道原样写入，省去 half → float → half 两次转换
    for (const n of layer.channels) {
      if (!wantPlane.has(n) || (colour && ["R", "G", "B"].includes(short(n)))) continue;
      const p = byName.get(n);
      if (!p) continue;
      const half = ow === d.width && oh === d.height && p.type === "half" ? (p.data as Uint16Array) : floatToHalf(small(n)!);
      const t3 = performance.now();
      planes.push({ name: n, gz: await gzip(l2c1Half(half, ow, oh)) });
      ms.encode += performance.now() - t3;
    }
  }
  ms.prep = performance.now() - t0 - ms.decode - ms.encode;
  return { width: ow, height: oh, ms, pictures, planes };
}

async function image(ask: ProxyAsk): Promise<ProxyAnswer> {
  const bitmap = await createImageBitmap(new Blob([ask.bytes as BlobPart]), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const [ow, oh] = fitted(bitmap.width, bitmap.height, ask.tier);
  const canvas = new OffscreenCanvas(ow, oh);
  const ctx = canvas.getContext("2d")!;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bitmap, 0, 0, ow, oh);
  bitmap.close();
  const blob = await canvas.convertToBlob({ type: WEBP, quality: ask.quality });
  return { width: ow, height: oh, ms: { decode: 0, prep: 0, encode: 0 }, pictures: [{ name: "rgba", webp: new Uint8Array(await blob.arrayBuffer()) }], planes: [] };
}

workerAnswers<ProxyAsk>((ask) => (ask.kind === "exr" ? exr(ask) : image(ask)));
