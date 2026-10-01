/** 显示变换查找表的数据结构与查表算法。纯算术，不依赖网络和缓存。
 *
 * 服务器将 OCIO 显示变换在对数网格上烘焙为 `size³` 个采样点（`lab2shot/server/view.py lut_for`），
 * 浏览器逐像素查表。获取表的部分位于 `transfer/lut.ts`（需要发送请求），本模块只定义表的结构与查法。
 *
 * 适用于服务器上没有对应图像、只能由浏览器执行显示变换的场合：本机预览 EXR（`transfer/exrWorker.ts`，
 * 以及本机代理 `transfer/localProxy/worker.ts`）：用户刚选择的文件，服务器尚未读取。
 *
 * 二维舞台本身不查此表：多条颜色通道一起查看时走图片路径（服务器已完成显示变换），
 * 走通道路径的数据（alpha、数值图、视频单通道）本身不经过色彩管理（`transfer/route.ts rawOf`）。 */

export interface Lut {
  mode: "lut" | "raw"; // raw：按数值原样显示并截断（显示空间的画面或数据，与服务器做法相同）
  size: number; // 三维表每边的采样点数
  lo: number; // 表的输入：线性值的 log2，从 lo 到 hi 档
  hi: number;
  data: Uint16Array; // 每点 RGB，0..65535 对应显示值 0..1，红色变化最快
}

/** 查表一次：线性 `r g b` 转为 0..1 的显示值，写入 `out` 的 `at`、`at+1`、`at+2`。
 *
 * 使用三线性插值，网格与服务器烘焙时一致（`lab2shot/server/view.py lut_for`）。
 * 逐像素的实现只此一份（本机代理按整张图批量查表，`transfer/localProxy/worker.ts`）。 */
export function lookup(lut: Lut, r: number, g: number, b: number, out: Float32Array | Uint8ClampedArray, at: number, scale = 1): void {
  if (lut.mode === "raw") {
    out[at] = r * scale;
    out[at + 1] = g * scale;
    out[at + 2] = b * scale;
    return;
  }
  const n = lut.size - 1;
  const span = lut.hi - lut.lo;
  const grid = (v: number) => {
    const u = ((Math.log2(Math.max(v, 2 ** lut.lo)) - lut.lo) / span) * n;
    return u < 0 ? 0 : u > n ? n : u;
  };
  const L = lut.data;
  const s1 = lut.size;
  const s2 = s1 * s1;
  const fr = grid(r);
  const fg = grid(g);
  const fb = grid(b);
  const r0 = Math.min(fr | 0, n - 1);
  const g0 = Math.min(fg | 0, n - 1);
  const b0 = Math.min(fb | 0, n - 1);
  const dr = fr - r0;
  const dg = fg - g0;
  const db = fb - b0;
  for (let c = 0; c < 3; c += 1) {
    const p = (ri: number, gi: number, bi: number) => L[(ri + gi * s1 + bi * s2) * 3 + c];
    const c00 = p(r0, g0, b0) * (1 - dr) + p(r0 + 1, g0, b0) * dr;
    const c10 = p(r0, g0 + 1, b0) * (1 - dr) + p(r0 + 1, g0 + 1, b0) * dr;
    const c01 = p(r0, g0, b0 + 1) * (1 - dr) + p(r0 + 1, g0, b0 + 1) * dr;
    const c11 = p(r0, g0 + 1, b0 + 1) * (1 - dr) + p(r0 + 1, g0 + 1, b0 + 1) * dr;
    const v = (c00 * (1 - dg) + c10 * dg) * (1 - db) + (c01 * (1 - dg) + c11 * dg) * db;
    out[at + c] = (v / 65535) * scale;
  }
}
