import { merged, mixOf, tintAlpha, tintLut } from "../model/view2d";
import { hasPixels, type LookAsk, type SideSource } from "./look";
import type { Plane } from "../transfer/plane";

/** 视图合成（look）在 CPU 上的参考实现，仅供测试（`webui/tests/look.check.py`）使用，产品代码不调用。
 *
 * GPU 实现（`view/look.ts` 的 shader）无法在单元测试中运行，因此同一算式在此另写一份，以相同输入分别计算后逐像素比对。
 * 判据为屏幕值相差不超过 1/255，不要求逐位一致：视图用于预览，GPU 运算顺序不同造成的微小误差可以接受。
 *
 * 本文件与 shader 必须逐行同步修改，测试即用于保证二者一致。 */

/** 在 CPU 上计算单路输入在某像素处的值，与 shader 中的 `side` 逐行对应。`pixels`：图片输入时为该图的 RGBA 字节。 */
function sideAt(s: SideSource, pixels: Uint8ClampedArray | null, x: number, y: number): [number, number, number, number] {
  const { box, side } = s;
  const u = (x - box[0]) / Math.max(box[2], 1);
  const v = (y - box[1]) / Math.max(box[3], 1);
  if (u < 0 || v < 0 || u >= 1 || v >= 1) return [0, 0, 0, 0];
  // 每一步都用 `Math.fround` 截断为 32 位浮点，与 GPU 精度一致。否则 64 位尾数的差异可能在
  // `round(t * 255)` 处落到相邻色标，而相邻色标相差 2/255，会被判为错误（半精度数据叠加范围映射和黑白点时容易出现）。
  const f = Math.fround;
  const grade = (t: number) => f(f(t - side.black) / (side.white - side.black || 1e-6));
  const planes = (s.planes ?? []).filter(Boolean) as Plane[];
  const tint = (t0: number): [number, number, number, number] => {
    const t = Math.min(1, Math.max(0, grade(t0)));
    const [r, g, b] = tintLut(side.tint, side.top ?? 0)[Math.round(f(t * 255))];
    return [r / 255, g / 255, b / 255, side.alpha === "weight" ? t : tintAlpha(side.tint, t)];
  };
  if (!planes.length) {
    const px = pixels!;
    const i = (Math.floor(v * box[3]) * Math.round(box[2]) + Math.floor(u * box[2])) * 4;
    if (side.index === null) return [grade(px[i] / 255), grade(px[i + 1] / 255), grade(px[i + 2] / 255), px[i + 3] / 255];
    return tint(px[i + side.index] / 255);
  }
  // 将通道在该点的值映射到该数据自身范围内的 0..1，与 shader 中的 `mapped` 逐行对应。
  const [lo, hi] = s.range ?? [0, 1];
  const span = Math.abs(hi - lo) > 1e-6 ? hi - lo : 1e-6;
  const raw = (p: Plane): number => {
    const i = Math.min(p.height - 1, Math.floor(v * p.height)) * p.width + Math.min(p.width - 1, Math.floor(u * p.width));
    return p.format === 0 ? p.data[i] / 255 : p.format === 1 ? p.data[i] / 65535 : p.data[i];
  };
  const at = (p: Plane) => Math.min(1, Math.max(0, f(f(raw(p) - lo) / span)));
  const ok = s.valid && raw(s.valid) <= 0 ? 0 : 1;   // 无效像素绘制为黑色
  if (planes.length > 1) {
    return [grade(at(planes[0]) * ok), grade(at(planes[1]) * ok),
            planes.length > 2 ? grade(at(planes[2]) * ok) : grade(0), 1];
  }
  return tint(at(planes[0]) * ok);
}

/** 在 CPU 上计算整幅视图合成结果，供测试与 GPU 实现比对（判据为像素差不超过 1/255）。产品代码不调用。 */
export function lookOnCpu(ask: LookAsk & { leftPixels?: Uint8ClampedArray; rightPixels?: Uint8ClampedArray }): Uint8ClampedArray {
  const { width: w, height: h } = ask;
  const out = new Uint8ClampedArray(w * h * 4);
  const r = ask.right && ask.merge && hasPixels(ask.right) ? ask.right : null;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const l = sideAt(ask.left, ask.leftPixels ?? null, x, y);
      const got = r
        ? merged(l, sideAt(r, ask.rightPixels ?? null, x, y), ask.merge!.op, mixOf(ask.merge!.op, ask.merge!.mix))
        : l;
      const i = (y * w + x) * 4;
      out[i] = got[0] * 255;
      out[i + 1] = got[1] * 255;
      out[i + 2] = got[2] * 255;
      out[i + 3] = got[3] * 255;
    }
  }
  return out;
}
