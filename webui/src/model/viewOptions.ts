/** The viewer's display options (Houdini's display options, in one panel): the 2D stage's preview chain
 * (左 取通道 · 中 运算（加 / 乘 Alpha / 乘 RGBA）· mix · 右 取通道 → 黑白点 → 着色 · 背景), and in 3D how points, lines and meshes are drawn, anti-aliasing, the camera's clipping and the scene aids.
 * Kept per browser in localStorage.
 *
 * 点始终绘制为圆片，仅有大小一个参数。大小和线宽的唯一单位是屏幕像素，二维、三维的线和三维曲线共用
 * `lineWidth`。localStorage 中不认识的键或取值均予忽略（`sanitize` 只按 `DEFAULTS` 的键读取，取值不在 `CHOICES` 中时
 * 恢复默认值）。
 *
 * Pure: only a type import (erased at run time). */

type PointColor = "color" | "constant";
type Shading = "smooth" | "flat" | "wire" | "wireShaded";
type Antialias = "off" | "msaa" | "fxaa" | "smaa";
export type Background = "solid" | "gradient";
type Lighting = "headlight" | "rig";
type CurveColor = "color" | "constant";
// 二维预览链的取值及其算法定义在 model/view2d.ts；本模块只保存用户的选择
import type { Bg, Op, Tint } from "./view2d";
export type { Bg, Op, Tint } from "./view2d";

export interface ViewOptions {
  // 2D 预览链：左右各取一条通道，中间做一次运算，右侧提供调色工具，背景单独设置。
  // 所选的层不在此保存：它是当前节点图中的一个端口，随本次编辑保存（state/view2d.ts left / right）
  op: Op; // 中：加、乘 Alpha、乘 RGBA（同一个下拉框的三档，model/view2d.ts Op）
  mix: number; // 中：运算强度（与 Nuke 的 merge 相同，0 表示左侧原样）
  black: number; // 右：黑点，取值相对于该数据自身范围的 0 到 1，而非绝对值
  white: number;
  tint: Tint; // 右：查看单通道时的着色（若干色标和纯色；画面本身不着色，颜色即数据）
  bg: Bg; // 背景：棋盘格或纯色（与运算完全解耦，三种模式下均可用）
  // 背景为纯色时的颜色。此处不写颜色字面量：空值表示色板第一档，颜色值只在
  // platform/palette.ts 中定义（本文件须保持纯净，不 import 页面模块）
  bgColor: string;
  // points
  pointPx: number; // size in screen pixels: near and far alike
  pointColor: PointColor;
  pointTint: string; // the constant colour
  // lines
  lineWidth: number; // screen pixels, 2D and 3D alike: boxes, tracks, handles, frustums, camera paths, bones, 3D curves, grid
  cameraColor: string;
  boneColor: string;
  // 三维曲线（发丝、毛发导向线、运动轨迹）：线宽取 lineWidth，着色使用曲线自身颜色或单色
  curveColor: CurveColor;
  curveTint: string;
  // meshes
  shading: Shading;
  uvChecker: boolean;
  overlayOpacity: number; // looking through a camera over the plate: how much of the models and characters shows
  overlayTint: string; // their colour there (a clay model over the plate)
  overlayByPerson: boolean; // "按人物": each person has its own colour by id (boxes on the 2D stage, characters in 3D)
  // picture
  antialias: Antialias;
  // camera (left tumbles, middle dollies, right pans; view/camera3d.tsx)
  near: number; // clipping, cm
  far: number;
  // scene aids
  grid: boolean;
  gridSpacing: number; // cm between grid lines (every tenth one stronger)
  gridSize: number; // cm across; 0: without end
  axes: boolean;
  axesSize: number; // the corner axes' size, pixels
  background: Background;
  backgroundColor: string;
  lighting: Lighting;
  exposure: number; // stops, on the lights
}

/** The overlay palette: the tint to choose from; its first colour is the plain white model (the default tint). */
export const OVERLAY_SWATCHES = ["#e8e8ec", "#FFD60A", "#FF9F0A", "#FF375F", "#64D2FF", "#30D158"];

/** "按人物": the colours assigned to people in turn. This is the overlay palette without its white, so that no person
 * (including id 0) looks like an uncoloured model. */
const PEOPLE_SWATCHES = OVERLAY_SWATCHES.slice(1);

/** "按人物": one person's colour, keyed by the person's id. Boxes (2D) and characters (lab2shot:person_id, 3D) carry
 * the same id, so a person has one colour on both stages. */
export const personTint = (id: number): string => PEOPLE_SWATCHES[Math.abs(Math.trunc(id)) % PEOPLE_SWATCHES.length];

export const DEFAULTS: ViewOptions = {
  op: "add",
  // 0.5：节点打开时默认为「加 · 红 · mix 0.5」，即抠像完成后最常用的半透明叠加视图。
  // 「乘」的强度固定为 1（model/view2d.ts mixOf），不受此默认值影响
  mix: 0.5,
  black: 0,
  white: 1,
  // 红（纯色）：抠像完成后，遮罩以红色半透明叠加在画面上。
  // 纯色的浓淡随数值变化（model/view2d.ts tintAlpha），因此在「仅结果」中显示为红色的浓淡，而非整片红色
  tint: "red",
  bg: "checker",
  bgColor: "", // 未选择：取色板第一档（platform/palette.ts bgColourOf）
  pointPx: 3,
  pointColor: "color",
  pointTint: "#d8d8dc",
  lineWidth: 1.5,
  cameraColor: "#FFD60A",
  boneColor: "#FF375F",
  curveColor: "color",
  // 橙色，取自共用色板，不另写颜色字面量（颜色值只在一处定义）
  curveTint: OVERLAY_SWATCHES[2],
  shading: "smooth",
  uvChecker: false,
  overlayOpacity: 0.6,
  overlayTint: "#e8e8ec",
  overlayByPerson: false,
  antialias: "msaa",
  near: 1,
  far: 1_000_000,
  grid: true,
  gridSpacing: 10,
  gridSize: 0,
  axes: true,
  axesSize: 40,
  background: "gradient",
  backgroundColor: "#0c0c0e",
  lighting: "rig",
  exposure: 0,
};

/** The choices of each option with a word each: the panel's labels. */
export const CHOICES = {
  op: { add: "加", mulAlpha: "乘 Alpha", mulRgba: "乘 RGBA" },
  tint: { grey: "灰度", warm: "冷到暖", id: "编号", red: "红", green: "绿", blue: "蓝", yellow: "黄", cyan: "青", magenta: "品红", white: "白" },
  bg: { checker: "棋盘格", solid: "纯色" },
  pointColor: { color: "自带", constant: "单色" },
  shading: { smooth: "平滑", flat: "平面", wire: "线框", wireShaded: "带线框" },
  antialias: { off: "关", msaa: "MSAA", fxaa: "FXAA", smaa: "SMAA" },
  background: { solid: "纯色", gradient: "渐变" },
  lighting: { headlight: "头灯", rig: "三点灯" },
  curveColor: { color: "自带", constant: "单色" },
} as const satisfies { [K in keyof ViewOptions]?: Record<string, string> };

/** 着色各档的含义：色标表示数值大小，纯色只表示浓淡（以「加」叠加时即为半透明叠加）。 */
const TINT_TIPS = {
  grey: "黑到白的灰度：数值本来的样子",
  warm: "暗紫到黄，感知均匀、色盲也分得清：看深度、置信度这类连续值最清楚",
  id: "按编号给每一类一个颜色，不在中间插值（分割图、物体编号、人物编号是整数，插出来的颜色不属于任何一类）；0 是背景，黑",
  red: "一整片红，浓淡按值：叠在画面上看遮罩、分割最常用",
  green: "一整片绿，浓淡按值",
  blue: "一整片蓝，浓淡按值",
  yellow: "一整片黄，浓淡按值",
  cyan: "一整片青，浓淡按值",
  magenta: "一整片品红，浓淡按值",
  white: "一整片白，浓淡按值",
} as const;

/** What each choice does, for its tip. */
export const CHOICE_TIPS: { [K in keyof typeof CHOICES]: Record<keyof (typeof CHOICES)[K], string> } = {
  op: {
    add: "右边的结果加到原图上：半透明叠加就是这一档（红色遮罩盖在画面上）",
    mulAlpha: "原图按遮罩扣一下，只乘 Alpha：颜色一点不动，只改透明度，边缘不变暗",
    mulRgba: "原图按遮罩扣一下，RGB 和 Alpha 一起乘：扣出来的边缘会连颜色一起变暗",
  },
  tint: TINT_TIPS,
  bg: { checker: "透出来的地方画棋盘格：看抠像边缘最清楚", solid: "透出来的地方铺一种颜色" },
  pointColor: { color: "用点自己带的颜色", constant: "全部一种颜色" },
  shading: { smooth: "平滑的明暗", flat: "每个面一种明暗，看得出面", wire: "只画线框", wireShaded: "明暗上再叠线框" },
  antialias: { off: "不抗锯齿：最快，有锯齿", msaa: "多重采样：边缘最干净", fxaa: "最快的抗锯齿，稍糊", smaa: "比 FXAA 清楚" },
  background: { solid: "纯色背景", gradient: "上下渐变的背景" },
  lighting: { headlight: "一盏跟着镜头的灯", rig: "固定的三点布光" },
  curveColor: { color: "用曲线自己带的颜色", constant: "全部一种颜色" },
};

/** Numeric limits: [min, max]. */
export const RANGES = {
  black: [-2, 2],
  white: [-2, 2],
  mix: [0, 1],
  pointPx: [1, 64],
  lineWidth: [0.5, 12],
  near: [0.001, 1e6],
  far: [1, 1e9],
  gridSpacing: [0.1, 100_000],
  gridSize: [0, 1e7],
  axesSize: [20, 240],
  overlayOpacity: [0.05, 1],
  exposure: [-8, 8],
} as const satisfies { [K in keyof ViewOptions]?: readonly [number, number] };

const HEX = /^#[0-9a-fA-F]{6}$/;

/** Options as stored (anything: old, hand-edited, broken) made whole and valid: unknown keys dropped (keys no longer
 * in use are ignored this way), missing or wrong ones back to their defaults, numbers clamped to their range, near
 * kept below far. */
export function sanitize(raw: unknown): ViewOptions {
  const src = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const out = { ...DEFAULTS } as Record<string, unknown>;
  for (const key of Object.keys(DEFAULTS) as (keyof ViewOptions)[]) {
    const v = src[key];
    const d = DEFAULTS[key];
    if (key in CHOICES) {
      if (typeof v === "string" && v in CHOICES[key as keyof typeof CHOICES]) out[key] = v;
    } else if (typeof d === "number") {
      if (typeof v === "number" && Number.isFinite(v)) {
        const r = (RANGES as Record<string, readonly [number, number]>)[key];
        out[key] = r ? Math.min(r[1], Math.max(r[0], v)) : v;
      }
    } else if (typeof d === "boolean") {
      if (typeof v === "boolean") out[key] = v;
    } else if (typeof d === "string") {
      if (typeof v === "string" && HEX.test(v)) out[key] = v;
    }
  }
  const o = out as unknown as ViewOptions;
  if (o.near >= o.far) {
    o.near = DEFAULTS.near;
    o.far = DEFAULTS.far;
  }
  return o;
}

export const STORAGE_KEY = "lab2shot.view3d";

type Store = Pick<Storage, "getItem" | "setItem">;

/** The options this browser keeps; the defaults when there are none or they can't be read (private window,
 * storage blocked, broken JSON). */
export function loadOptions(storage: Store | null | undefined): ViewOptions {
  try {
    const text = storage?.getItem(STORAGE_KEY);
    const raw = text ? (JSON.parse(text) as Record<string, unknown>) : null;
    return sanitize(raw);
  } catch {
    return { ...DEFAULTS };
  }
}

/** Keeps the options in this browser: only those that differ from the defaults, so a later default reaches
 * everyone who never changed it. False when the browser would not keep them. */
export function saveOptions(storage: Store | null | undefined, o: ViewOptions): boolean {
  const changed: Partial<Record<keyof ViewOptions, unknown>> = {};
  for (const key of Object.keys(DEFAULTS) as (keyof ViewOptions)[]) if (o[key] !== DEFAULTS[key]) changed[key] = o[key];
  try {
    storage?.setItem(STORAGE_KEY, JSON.stringify(changed));
    return !!storage;
  } catch {
    return false;
  }
}
