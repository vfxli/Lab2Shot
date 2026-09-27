/// <reference lib="webworker" />
import { workerAnswers } from "../platform/work";
import { lookup, type Lut } from "../ops/lut";
import { decodeExrPlanes, exrHeader, halfToFloat, type ExrDecoded } from "./exr/decode";

/** The page's EXR worker. It answers two kinds of request:
 *
 * 1. `{ file, lut }` (display): decodes a user-selected EXR with the page's single decoder (`exr/decode.ts`,
 *    adapted from three's EXRLoader; supports PIZ, ZIP, RLE, PXR24, B44, DWA) and applies the display transform
 *    supplied by the server (exr.ts). Decoding runs off the main thread so that a 4K frame does not block the editor.
 *    The local proxy (`transfer/localProxy`) is the primary path; this request is used only when the browser has no
 *    origin private file system. The table lookup itself is shared with browser-computed images (view/evaluate.ts)
 *    via ops/lut.ts `lookup`.
 * 2. `{ file, planes }` (upload): returns the raw planes of the requested channels only (`planes`: the file's own
 *    channel names, taken from the status reply's `channels.take`), as half / float / uint as stored, with no display
 *    transform and no resizing. Channel-level upload sends these instead of the whole file (transfer/planes.ts).
 *    The decoder throws for unsupported input; the caller then falls back to sending the whole file and reports it. */

export type { Lut };

export type ExrAsk = { file: Uint8Array; lut: Lut; planes?: undefined } | { file: Uint8Array; planes: string[]; lut?: undefined };

/** Answer to request 2: `exr/decode.ts` ExrDecoded as is (planes are typed arrays, transferred per platform/work.ts). */
export type PlanesAnswer = ExrDecoded;

workerAnswers<ExrAsk>((ask) => {
  if (ask.planes) return decodeExrPlanes(ask.file, ask.planes) as unknown as object;
  // Request 1: decode the colour channels with exr/decode.ts and apply the display transform. Only R, G, B (and A)
  // are decoded; other layers are skipped. Files without R, G, B are drawn as greyscale from Y or the first channel.
  const names = exrHeader(ask.file).allChannels.map((c) => c.name);
  const pick = (c: string) => names.find((n) => n === c) ?? names.find((n) => n.toUpperCase().endsWith("." + c));
  const r = pick("R"), g = pick("G"), b = pick("B"), a = pick("A"), y = pick("Y");
  const wanted = [r, g, b, a].filter((n): n is string => !!n);
  const d = decodeExrPlanes(ask.file, wanted.length ? wanted : y ? [y] : [names[0]]);
  const plane = (n: string | undefined): Float32Array | null => {
    const p = n ? d.channels.find((c) => c.name === n) : undefined;
    return p ? (p.type === "half" ? halfToFloat(p.data as Uint16Array) : p.type === "float" ? (p.data as Float32Array) : Float32Array.from(p.data as Uint32Array)) : null;
  };
  const R = plane(r) ?? plane(y) ?? plane(names[0])!;
  const G = plane(g) ?? R, B = plane(b) ?? R;
  const { width: w, height: h } = d;
  const out = new Uint8ClampedArray(w * h * 4);
  const lut = ask.lut;
  for (let i = 0; i < w * h; i++) {
    lookup(lut, R[i], G[i], B[i], out, i * 4, 255);
    out[i * 4 + 3] = 255;
  }
  return { width: w, height: h, pixels: out };
});
