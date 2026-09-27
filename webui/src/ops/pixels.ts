import type { Pix } from "./run";
import { lookup, type Lut } from "./lut";

/** 将浏览器计算得到的结果绘制为屏幕像素（`ops/recipe.ts` 负责计算数值，本模块将数值转为图像）。
 *
 * 与 `recipe.ts` 分为两个文件：`recipe.ts` 是纯算术（只依赖 `ops/run.ts` 和 `transfer/plane.ts`），
 * 跨端对照测试可直接用 node 运行（`webui/tests/evalCross.run.ts`）；本模块需要查显示变换表，
 * 依赖 http 与消息目录，无法在该环境中运行。
 *
 * 仅图片路径使用本模块：多条颜色通道一起查看时舞台需要一张显示图，而服务器上没有该包，
 * 因此显示变换须由浏览器执行（`transfer/lut.ts`，与本机预览 EXR 使用同一张表和同一段代码）。
 * 走通道路径的数据（数值图、遮罩、alpha）不使用本模块，其显示方式在 GPU 上实时计算（`view/look.ts`）。 */

/** 计算结果转为屏幕像素的方式，与服务器生成显示图的算法一一对应
 * （`lab2shot/view/frames.py` 的 `_map_rgb` / `display_frame`）。 */
export interface Show {
  values: boolean; // 数值图：按自身范围映射；画面：经过显示变换
  range: [number, number];
  alpha: boolean; // 画面带 alpha（数据为预乘，与 Nuke 一致）
  lut: Lut | null; // 画面的显示变换（数值图为 null）
}

// ---------------------------------------------------------------- 计算结果 → 屏幕像素

/** 将一张二维数据绘制为 8 位 RGBA，与服务器生成显示图的算法一一对应：
 * - 数值图：按自身范围映射（`lab2shot/view/frames.py _map_rgb`：一条通道绘为灰度，两条绘为红绿，三到四条取前三条）；
 * - 画面：查显示变换表（`transfer/lut.ts`）；带 alpha 时先反预乘，查表后再乘回（与服务器
 *   `io/color.py apply_premultiplied` 规则相同，与 Nuke 的做法一致）。 */
export function toPixels(pix: Pix, show: Show): { width: number; height: number; pixels: Uint8ClampedArray } {
  const { w, h, c, data } = pix;
  const out = new Uint8ClampedArray(w * h * 4);
  if (show.values) {
    const [lo, hi] = show.range;
    const span = Math.abs(hi - lo) > 1e-6 ? hi - lo : 1e-6;
    for (let p = 0; p < w * h; p += 1) {
      const at = p * 4;
      const v = (i: number) => Math.min(Math.max((data[p * c + i] - lo) / span, 0), 1) * 255;
      if (c === 1) out[at] = out[at + 1] = out[at + 2] = v(0);
      else if (c === 2) {
        out[at] = v(0);
        out[at + 1] = v(1);
        out[at + 2] = 0;
      } else {
        out[at] = v(0);
        out[at + 1] = v(1);
        out[at + 2] = v(2);
      }
      out[at + 3] = 255;
    }
    return { width: w, height: h, pixels: out };
  }
  const lut = show.lut ?? { mode: "raw" as const, size: 2, lo: 0, hi: 1, data: new Uint16Array(0) };
  for (let p = 0; p < w * h; p += 1) {
    const at = p * 4;
    const a = show.alpha && c >= 4 ? data[p * c + 3] : 1;
    const by = a > 0 ? 1 / a : 0; // 反预乘：变换始终作用于未被覆盖率压暗的颜色
    const r = data[p * c] * by;
    const g = c >= 3 ? data[p * c + 1] * by : r;
    const b = c >= 3 ? data[p * c + 2] * by : r;
    lookup(lut, r, g, b, out, at, 255 * (show.alpha ? a : 1)); // 查表后乘回预乘的 alpha
    out[at + 3] = show.alpha ? a * 255 : 255;
  }
  return { width: w, height: h, pixels: out };
}
