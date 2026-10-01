/** 二维舞台的视图变换（Nuke 式平移 / 缩放：整个查看器一份，不按节点分；页面打开期间保留，不存入节点图，
 * 不产生撤销步骤；其状态与控件在 state/view2d.ts）与二维舞台唯一的一条预览链：
 *
 *     左（原图）：取通道
 *     中（运算）：加 / 乘 · mix · 只乘 Alpha 或 RGBA 一起乘
 *     右（结果）：取通道 → 黑点/白点 → 着色
 *     背景：棋盘格 / 纯色（和运算完全解耦，三个模式下都在）
 *
 * 纯模块：不引用任何东西。画布 CSS 像素 = x + 图像像素 × s（`s` 即 Stage2D 与 overlays.ts 中的 `at.s`）。 */

export interface Transform2D {
  x: number;
  y: number;
  s: number; // 每个图像像素对应的画布 CSS 像素数
}

const MIN_ZOOM = 0.05; // 5%
const MAX_ZOOM = 32; // 3200%

const clampZoom = (s: number): number => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, s));

/** 以画布上的固定点（px, py，CSS 像素）为中心按 `factor` 缩放：该点下方的图像点缩放后仍在该点下方。 */
export function zoomAt(t: Transform2D, factor: number, px: number, py: number): Transform2D {
  const s = clampZoom(t.s * factor);
  const k = t.s === 0 ? 1 : s / t.s;
  return { s, x: px - (px - t.x) * k, y: py - (py - t.y) * k };
}

/** 按屏幕空间的位移（画布 CSS 像素）平移：中键拖动，或 Alt + 左键拖动。 */
export const panBy = (t: Transform2D, dx: number, dy: number): Transform2D => ({ ...t, x: t.x + dx, y: t.y + dy });

/** 缩放到精确的百分比（在时间线的缩放框中输入），画布中心保持不动。 */
export function toPercent(t: Transform2D, pct: number, cw: number, ch: number): Transform2D {
  const s = clampZoom(pct / 100);
  return zoomAt(t, t.s === 0 ? 1 : s / t.s, cw / 2, ch / 2);
}

/** 返回画面在舞台上的摆放尺寸。摆放依据是该帧的画面范围，而非解码图像的像素数，以保证切换节点对比时画面大小不变。
 *
 * 视图可以显示本地原件，因此同一张图可能有两种像素尺寸（如本机原件 1920×1080、服务器代理 512×288）。
 * 若按解码图像的像素数摆放，切换节点时画面会缩放，且不会报错；因此摆放不得直接使用 `image.width`。
 * 画布是布局中的固定区域，图像像素尺寸与工具栏行数、通知条数、参数数量同属内容属性，不得反过来决定画布。
 *
 * 按可信度依次取值：
 * 1. 边算边看时服务器报告的尺寸（服务器正在写入该帧，尺寸已知）；
 * 2. 包说明中的 `meta.width/height`：代理与原件都按它摆放，二者在屏幕上占同一矩形，仅清晰度不同（代理为等比缩小）；
 * 3. 仅有人物框、跟踪点等叠加物时，使用叠加物自带的画面尺寸；
 * 4. 解码图像的像素尺寸，仅在以上均缺失时使用，即使用者刚选定文件、服务器尚未读出的短暂时段，此时不存在第二个来源。
 *
 * 验收条件：依次切换 A → B → A，画面的屏幕位置与缩放完全一致，包括 A 显示本地原件、B 显示服务器代理且像素尺寸不同的情况。 */
export function pictureSize(said: {
  partial?: { width: number; height: number } | null;
  meta?: { width?: unknown; height?: unknown } | null; // 包说明中的 meta
  overlays?: readonly ({ width: number; height: number } | null | undefined)[];
  decoded?: { w: number; h: number } | null; // 解码图像的像素尺寸，优先级最低
}): { w: number; h: number } {
  const one = (v: unknown): number => (typeof v === "number" && v > 0 ? v : 0);
  const from: [number, number][] = [
    [one(said.partial?.width), one(said.partial?.height)],
    [one(said.meta?.width), one(said.meta?.height)],
    ...(said.overlays ?? []).map((o): [number, number] => [one(o?.width), one(o?.height)]),
    [one(said.decoded?.w), one(said.decoded?.h)],
  ];
  for (const [w, h] of from) if (w && h) return { w, h };
  return { w: 0, h: 0 };
}

/** 适应 (F)：整幅画面留边后居中放入 cw × ch 的画布。 */
export function fitTransform(width: number, height: number, cw: number, ch: number): Transform2D {
  if (width <= 0 || height <= 0 || cw <= 0 || ch <= 0) return { x: cw / 2, y: ch / 2, s: 1 };
  const pad = 24;
  const s = clampZoom(Math.min((cw - pad * 2) / width, (ch - pad * 2 - 30) / height));
  return { x: (cw - width * s) / 2, y: (ch - height * s) / 2 + 8, s };
}

/** 1:1：一个图像像素对应一个设备像素（计入 devicePixelRatio），在 cw × ch 的画布中居中，左上角落在整数设备像素上
 * （每个图像像素恰好落在一个屏幕像素上，不会跨在两个之间）。 */
export function oneToOneTransform(width: number, height: number, cw: number, ch: number, dpr: number): Transform2D {
  const d = dpr || 1;
  const s = clampZoom(1 / d);
  return { x: Math.round(((cw - width * s) / 2) * d) / d, y: Math.round(((ch - height * s) / 2) * d) / d, s };
}

// ------------------------------------------------------------------ 二维像素数据：仅依据通道数与包自带信息

/** 返回图像的通道数（0 表示不是二维像素数据）。类型 id 即通道数（lab2shot/data/types.py PIXELS）：
 * image.1 Mask、image.2 UV、image.3 RGB、image.4 RGBA。
 *
 * 视图不区分「深度图」「遮罩」「置信度」等含义（通道含义由使用处决定）：绘制方式只取决于通道数，其余含义（数值还是画面、
 * 显示范围、坐标系）由包自带的 meta 给出（服务器按 data/payloads.py is_data 出图，网页只负责叠加）。 */
export function channelsOf(type: string): number {
  const one = type.split("|")[0].replace("[]", "");
  const dot = one.lastIndexOf(".");
  if (dot < 0 || one.slice(0, dot) !== "image") return 0;
  const n = Number(one.slice(dot + 1));
  return Number.isInteger(n) && n >= 1 && n <= 4 ? n : 0;
}

// ------------------------------------------------------------------ 一条预览链
//
//     [ 仅原图 │ 运算 │ 仅结果 ]
//     仅原图： 左：原图 通道▾                                                   │ 背景▾
//     运算：   左：通道▾ │ 加/乘 · mix · 只乘Alpha或RGBA一起乘 │ 右：通道▾ 着色▾ 黑点 白点 │ 背景▾
//     仅结果：                                              右：通道▾ 着色▾ 黑点 白点 │ 背景▾
//
// 左侧始终是原图（上游画面），右侧始终是本节点的计算结果，中间为一次运算。
// 不设「衬底」「第二通道」「自动半透明叠加」等开关，这些是同一功能的拆分，会使使用者难以理解。
//
// 核心区分是显示颜色还是数值。查看单通道时，屏幕颜色由映射产生（黑白点决定映射，着色决定颜色）；
// 三或四条通道一起查看时，颜色即数据本身，视图不做着色（所见即所得）。
// 一律按通道数和包自带的 meta 计算，不依据类型名，也不在各组件中分别判断
// （控件可用性集中声明于 model/viewControls.ts）。

/** 键盘切换通道（与 Nuke 相同，按 R G B A）：切换当前一侧的通道，再次按下同一键回到「整体」。 */
export const CHANNEL_KEYS: Record<string, number> = { r: 0, g: 1, b: 2, a: 3 };

/** 是否绘制单条通道：选中了某一条，或数据本身只有一条通道（此时「整体」即该通道）。
 * 「着色」仅在此情况下适用；三或四条通道一起查看时，屏幕颜色即数据本身（所见即所得）。 */
export const singleChannel = (index: number | null, channels: number): boolean => index !== null || channels === 1;

/** 返回当前显示的通道索引（null 表示多条通道一起绘制）。单通道数据的「整体」即该通道。 */
export const lookIndex = (index: number | null, channels: number): number | null => (index !== null ? index : channels === 1 ? 0 : null);

// ------------------------------------------------------------------ 黑点 / 白点

/** `out = (v − 黑) ÷ (白 − 黑)`，不做裁切（与 Nuke Grade 的默认行为一致：超出 0 和 1 的值沿同一直线延伸）。
 * 黑白点以数据自身范围的 0 到 1 表示，而非绝对值，因此默认值为 0 和 1，切换显示节点后仍然有效。 */
export const graded = (v: number, black: number, white: number): number => (v - black) / (white - black || 1e-6);

// ------------------------------------------------------------------ 着色：固定的色标与纯色，不支持自定义

/** 右侧通道的着色方式，三类共用一个下拉：
 *
 *   - 色标（`grey`、`warm`）：颜色表示数值大小，用于深度、置信度等连续值；
 *   - 纯色（`red`…`white`）：颜色固定，数值只决定浓淡，以「加」叠加即为常规的半透明叠加；
 *   - 编号（`id`）：每个整数对应一种颜色，不插值。 */
export type Tint = "grey" | "warm" | "id" | "red" | "green" | "blue" | "yellow" | "cyan" | "magenta" | "white";

/** 纯色各档的颜色（出现在 `SOLID` 中的为纯色，其余为色标）。 */
const SOLID: Partial<Record<Tint, readonly [number, number, number]>> = {
  red: [255, 69, 108],
  green: [61, 220, 132],
  blue: [76, 141, 255],
  yellow: [255, 214, 10],
  cyan: [100, 210, 255],
  magenta: [191, 90, 242],
  white: [255, 255, 255],
};

// 查表只认自有键：着色值来自存储，原型链上的名字（constructor 等）不是一档着色
const isSolid = (tint: Tint): boolean => Object.hasOwn(SOLID, tint);
const rampOf = (tint: Tint) => (Object.hasOwn(RAMPS, tint) ? RAMPS[tint] : undefined) ?? RAMPS.grey;

/** 单通道显示时该着色档的不透明度（0..1），同一规则适用于两种模式：
 *
 *   - 纯色：颜色固定，浓淡即数值。「仅结果」中遮罩显示为红色浓淡，「运算」中叠加即为常规的半透明叠加。
 *   - 色标（灰度、冷到暖、编号）：颜色已表示数值大小，再乘透明度会重复表达，因此不透明。 */
export const tintAlpha = (tint: Tint, v: number): number => (isSolid(tint) ? v : 1);

/** 「冷到暖」即 matplotlib 的 viridis：感知均匀，色觉障碍者可区分，暗紫为最小值，黄为最大值。
 *
 * 色标属于数据而非样式：每档写成 [r, g, b] 0..255，供逐像素循环直接查表，不属于界面颜色字面量。 */
const VIRIDIS: readonly (readonly [number, number, number])[] = [
  [68, 1, 84], [72, 36, 117], [65, 68, 135], [53, 95, 141], [42, 120, 142], [33, 145, 140],
  [34, 168, 132], [68, 191, 112], [122, 209, 81], [189, 223, 38], [253, 231, 37],
];
const BLACK: readonly [number, number, number] = [0, 0, 0];

/** 「编号」档的调色盘：按编号取色，不插值。分割图、物体编号、人物编号均为整数，插值得到的颜色不对应任何类别。
 * 0 保留给背景（黑）。相邻编号的颜色差异较大，以便区分相邻物体。
 * 分割图若按灰度绘制几乎全黑（0、1、2、3 在 0..1 范围内均为黑色），无法辨认。 */
const ID_COLOURS: readonly (readonly [number, number, number])[] = [
  [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
  [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
  [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195], [128, 128, 0], [255, 215, 180],
  [0, 0, 128], [128, 128, 128],
];

/** 返回编号 `i` 的颜色（0 为背景，黑色）。超出调色盘长度时循环取色。
 *
 * 必须先取整再判断是否为背景：`i` 可能是小数（调用方传入的是换算出来的值）。
 * 若先判断 `i <= 0`，0.2 会进入后一分支，`Math.round(0.2) - 1 = -1`，`ID_COLOURS[-1]` 为 `undefined`，
 * 调用处的 `[...idColour(...)]` 随即抛错，整个视图被错误边界接管（值域上界为 1 时，v/255 在 1..127 上均落入 (0, 0.5)）。 */
const idColour = (i: number): readonly [number, number, number] => {
  const n = Math.round(i);
  return n <= 0 ? BLACK : ID_COLOURS[(n - 1) % ID_COLOURS.length]!;
};

/** 各档绘制为小色条时的色标站点（纯色为单一颜色）。实际查色使用 `tintLut`。 */
const RAMPS: Record<Tint, readonly (readonly [number, number, number])[]> = {
  grey: [BLACK, [255, 255, 255]],
  warm: VIRIDIS,
  id: ID_COLOURS,
  red: [SOLID.red!, SOLID.red!],
  green: [SOLID.green!, SOLID.green!],
  blue: [SOLID.blue!, SOLID.blue!],
  yellow: [SOLID.yellow!, SOLID.yellow!],
  cyan: [SOLID.cyan!, SOLID.cyan!],
  magenta: [SOLID.magenta!, SOLID.magenta!],
  white: [SOLID.white!, SOLID.white!],
};

const said = (c: readonly [number, number, number]) => `rgb(${c.join(",")})`;

/** 将着色转换为 CSS 渐变，用于控件上的小色条（纯色为单色色条）。 */
export const rampGradient = (tint: Tint): string => `linear-gradient(to right, ${rampOf(tint).map(said).join(", ")})`;

/** 返回 0..1 的数值在该色标上的颜色，[r, g, b] 0..255（超出范围时取端点值；纯色档始终返回该纯色）。 */
function rampColor(tint: Tint, t: number): [number, number, number] {
  const solid = isSolid(tint) ? SOLID[tint] : undefined;
  if (solid) return [...solid] as [number, number, number];
  const stops = rampOf(tint);
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const [a, b] = [stops[i], stops[i + 1]];
  return [0, 1, 2].map((k) => Math.round(a[k] + (b[k] - a[k]) * f)) as [number, number, number];
}

/** 返回一条色标的查色表，供逐像素查表，避免对每个像素插值。第 i 格对应 t = i / (格数 − 1)。
 *
 * 色标与纯色为 256 档（8 位屏幕的一级）。「编号」档例外：每个编号一格（0..top 共 top + 1 格，`top` 为最大编号，
 * 即包的 range 上界），第 i 格就是编号 i 的颜色，不插值。不能也压成 256 档：最大编号超过 255 时几个编号会落进
 * 同一格，相邻编号画成同一种颜色，分不出两个物体。编号图的值原样到达浏览器（服务器不压尾数，
 * lab2shot/view/channels.py channel_blob 的 exact），着色器按 round(t × top) 取格（view/look.ts），得到的就是编号本身。 */
export const tintLut = (tint: Tint, top = 0): [number, number, number][] =>
  (tint === "id" && top > 0
    ? Array.from({ length: Math.round(top) + 1 }, (_, i) => [...idColour(i)] as [number, number, number])
    : Array.from({ length: 256 }, (_, v) => rampColor(tint, v / 255)));

/** 查色表的最大下标（格数 − 1）：着色器按 round(t × 它) 取格（view/look.ts），与 `tintLut` 的格数一一对应。 */
export const lutTop = (tint: Tint, top = 0): number => (tint === "id" && top > 0 ? Math.round(top) : 255);

// ------------------------------------------------------------------ 三种模式与中间运算

/** 视图模式：`plate` 仅原图（仅左侧）、`over` 运算（左 ⊕ 右）、`result` 仅结果（仅右侧）。
 * 每个节点带有预览标签，决定默认模式（NodeTypeDef.preview，由服务器计算，同一标签适用于 2D 与 3D；换算见 view/plan.ts）。 */
export type Mode = "plate" | "over" | "result";

/** 中间运算（对应 Nuke 的 merge），共四档、一个下拉：加、只乘 Alpha、RGBA 一起乘、盖上（over）。
 *
 * 「只乘 Alpha / RGBA 一起乘」的区分只对「乘」有意义：「加」和「盖上」对两者的像素结果相同，所以不拆档。
 *
 * 「乘」的两档：`mulAlpha` 不改变颜色、仅改变透明度，边缘不变暗；
 * `mulRgba` 同时乘 RGB 与 Alpha，抠出的边缘颜色也会变暗。 */
export type Op = "add" | "mulAlpha" | "mulRgba" | "over";

/** 用强度（mix）的档：「加」「盖上」要调浓淡，「乘」固定为 1（mixOf）。 */
export const blends = (op: Op): boolean => op === "add" || op === "over";

/** 返回本次运算实际使用的强度。「乘」固定为 1（此时 mix 控件不可用）：
 * 「乘」用于按遮罩预览抠像，抠一半的结果既非原图也非抠像结果，没有参考价值。
 * 不能通过将默认 mix 改为 1 实现，否则「加」档的半透明叠加也会失效。
 * 强度仅对「加」有意义：叠加需要调节浓淡（默认 0.5），抠像不需要。
 *
 * 使用者保存的 mix 值不变，仅在「乘」时忽略，切回「加」后恢复原值。 */
export const mixOf = (op: Op, mix: number): number => (blends(op) ? mix : 1);

/** 背景：透明区域的填充方式。与运算完全独立，三种模式下均生效；导入的图像自带 alpha 时，
 * 「仅原图」模式下也须显示背景。 */
export type Bg = "checker" | "solid";

/** 单个像素的合成。`l`、`r` 均为 [R, G, B, A]，取值 0..1。
 *
 * 两侧 alpha 的含义不同，这是本算法的关键约定：
 *   - `l[3]` 是原图自身的第四通道（缺失时为 1）；
 *   - `r[3]` 是右侧所选通道的值，即运算强度（选「整体」时取第四通道，缺失时为 1）。
 *     右侧 RGB 是「着色」生成的显示颜色，不参与「乘」的计算，因此更换着色不会改变抠像结果。
 *
 * `mix` 与 Nuke merge 相同，将运算结果向左侧回拉：mix = 0 为左侧本身，mix = 1 为完整运算结果。 */
export function merged(l: readonly number[], r: readonly number[], op: Op, mix: number): [number, number, number, number] {
  if (op === "add") {
    // 加：右侧按自身强度覆盖（强度为 0 处不覆盖），透明度不变，叠加不会使画面变透明
    const k = r[3] * mix;
    return [l[0] + r[0] * k, l[1] + r[1] * k, l[2] + r[2] * k, l[3]];
  }
  if (op === "over") {
    // 盖上（Nuke 的 A over B，这里 A 是右侧的结果、B 是原图）：右侧按自身 alpha 盖住原图，透明处露出原图；
    // 颜色按未预乘计：结果 = 原图 × (1 − k) + 右侧 × k，alpha 同理（与「图像合成」的「盖上」同一件事，视图里现算）
    const k = r[3] * mix;
    return [l[0] * (1 - k) + r[0] * k, l[1] * (1 - k) + r[1] * k, l[2] * (1 - k) + r[2] * k, l[3] * (1 - k) + k];
  }
  const k = 1 - mix + mix * r[3]; // 乘：mix 将乘数向 1 回拉
  return op === "mulAlpha" ? [l[0], l[1], l[2], l[3] * k] : [l[0] * k, l[1] * k, l[2] * k, l[3] * k];
}

// ------------------------------------------------------------------ 通道下拉：层与通道，与 Nuke 一致

/** 下拉中选中的层与通道。`index` 为 null 表示「整体」（多条通道一起显示）。
 * 与 Nuke 一致（uv uv.r uv.g N N.r N.g N.b）：不分两步选择结果与通道，
 * 层与通道列在同一下拉中，同时提供整体与各单通道。 */
export interface Pick {
  port: string; // 本节点的输出口（左侧只有一层，为 ""）
  index: number | null;
}

export const pickValue = (p: Pick): string => `${p.port}|${p.index === null ? "all" : p.index}`;

/** 解析下拉中的值（无法解析时返回 null）。 */
export function pickOf(value: string): Pick | null {
  const cut = value.lastIndexOf("|");
  if (cut < 0) return null;
  const tail = value.slice(cut + 1);
  if (tail === "all") return { port: value.slice(0, cut), index: null };
  const index = Number(tail);
  return Number.isInteger(index) && index >= 0 && index <= 3 ? { port: value.slice(0, cut), index } : null;
}

/** 一层：一个输出口的一份二维结果。 */
export interface Layer {
  port: string;
  label: string; // 单行显示的名称（如「遮罩」「原图」）
  channels: number;
  // 该层当前无法绘制的原因（如「还没算」），显示在每一项末尾。
  // 不得并入 `label`，否则展开为通道后显示为「forward · 还没算.R」，
  // 看似存在一条名为「还没算.R」的通道
  note?: string;
}

/** 将一层展开为下拉中的若干项：整体与每条通道（`遮罩`、`遮罩.R`…，与 Nuke 一致）。
 * 单通道的层不单列 `.R`，其「整体」即该通道。 */
export function channelOptions(layers: readonly Layer[]): { value: string; label: string; port: string; index: number | null }[] {
  const out: { value: string; label: string; port: string; index: number | null }[] = [];
  for (const layer of layers) {
    const tail = layer.note ? ` · ${layer.note}` : "";
    out.push({ value: pickValue({ port: layer.port, index: null }), label: layer.label + tail, port: layer.port, index: null });
    if (layer.channels > 1)
      for (let i = 0; i < layer.channels; i++)
        out.push({ value: pickValue({ port: layer.port, index: i }), label: `${layer.label}.${"RGBA"[i]}${tail}`, port: layer.port, index: i });
  }
  return out;
}

/** 若选中项已不在这些层中（例如切换显示节点后原输出口不存在），回到第一项。 */
export function pickedIn(value: string, options: readonly { value: string }[]): string {
  return options.some((o) => o.value === value) ? value : options[0]?.value ?? "";
}
