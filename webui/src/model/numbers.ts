/** 一个数落进它声明的取值范围：页面上所有写数的地方（ui/controls.tsx Num 与各个滑块）只在这里规范化，与服务端对参数
 * 的校验同一口径（lab2shot/nodes/params.py：minimum / maximum，open_minimum / open_maximum 为 gt / lt，multiple_of）。
 * 纯函数，不 import 别的。 */

export interface NumSpec {
  min?: number | null;
  max?: number | null;
  openMin?: boolean; // 下界本身不可取（gt）：写到界上或界外不收
  openMax?: boolean; // 上界本身不可取（lt）
  integer?: boolean;
  multipleOf?: number | null; // 对齐到它的整数倍
}

/** 规范化 `n`：先对齐步长（整数按 1；正负一样，半个步长都离开 0：2.5 → 3、-2.5 → -3）并去掉浮点尾差（0.1 × 3 写成 0.3；
 * 只动第 15 位以后，整数只取整），再按范围判：
 * - `n` 本身在范围外：闭区间夹到范围内对齐的端点（夹完仍对齐），开区间不收；
 * - `n` 在范围内、只是对齐把它推到了界上或界外（开区间 [2, 3) 里的 2.5 取整成 3）：取界内最近的对齐值（2），不拒收；
 * 范围里没有对齐的值、或不是有限数，都是 null（不写，调用方退回原值）。界的判断在去尾差之后，去尾差不会把值推到开区间
 * 的界上。 */
export function coerce(n: number, s: NumSpec): number | null {
  if (!Number.isFinite(n)) return null;
  const step = s.multipleOf || (s.integer ? 1 : 0);
  // 15 位有效数字：去掉 0.1 × 3 的尾差，又不动 13、14 位的整数部分（1234567890123.5 照原样）
  const clean = (v: number) => (Number.isInteger(v) ? v : Number(v.toPrecision(15)));
  const align = (v: number, how: (x: number) => number) => (step ? clean(how(clean(v / step)) * step) : clean(v));
  const half = (x: number) => Math.sign(x) * Math.round(Math.abs(x));
  const below = (x: number) => s.min != null && (s.openMin ? x <= s.min : x < s.min);
  const above = (x: number) => s.max != null && (s.openMax ? x >= s.max : x > s.max);
  let v = step ? align(n, half) : clean(n);
  if (below(clean(n))) {
    if (s.openMin) return null;
    v = align(s.min!, Math.ceil);
  } else if (below(v)) {
    v = align(s.min!, Math.ceil);
    if (below(v) && step) v = clean(v + step);
  }
  if (above(clean(n))) {
    if (s.openMax) return null;
    v = align(s.max!, Math.floor);
  } else if (above(v)) {
    v = align(s.max!, Math.floor);
    if (above(v) && step) v = clean(v - step);
  }
  if (below(v) || above(v)) return null; // 范围里没有对齐的值
  return clean(v);
}
