/** 查看器的显示选项（对应 Houdini 的 display options，集中在一个面板）：二维舞台的预览链
 * （左 取通道 · 中 运算（加 / 乘 Alpha / 乘 RGBA）· mix · 右 取通道 → 黑白点 → 着色 · 背景），以及三维中点、线、模型的画法、
 * 抗锯齿、相机的裁剪距离与场景辅助。每个浏览器各自保存在 localStorage 中。
 *
 * 点始终绘制为圆片，仅有大小一个参数。大小和线宽的唯一单位是屏幕像素，二维、三维的线和三维曲线共用
 * `lineWidth`。localStorage 中不认识的键或取值均予忽略（`sanitize` 只按 `DEFAULTS` 的键读取，取值不在 `CHOICES` 中时
 * 恢复默认值）。
 *
 * 纯模块：只有类型引用（运行时被擦除）。 */

type PointColor = "color" | "constant";
type Shading = "smooth" | "flat" | "wire" | "wireShaded";
type Antialias = "off" | "msaa" | "fxaa" | "smaa";
export type Background = "solid" | "gradient";
type Lighting = "headlight" | "rig";
type CurveColor = "color" | "constant";
type BoneStyle = "solid" | "wire";
// 二维预览链的取值及其算法定义在 model/view2d.ts；本模块只保存用户的选择
import type { Bg, Op, Tint } from "./view2d";
export type { Bg, Op, Tint } from "./view2d";

export interface ViewOptions {
  // 2D 预览链：左右各取一条通道，中间做一次运算，右侧提供调色工具，背景单独设置。
  // 所选的层与通道不在此保存：右侧的层是显示的端口（state/look.ts displayPort，随节点图保存），两侧各取的通道在 state/view2d.ts left / right
  op: Op; // 中：加、乘 Alpha、乘 RGBA（同一个下拉框的三档，model/view2d.ts Op）
  mix: number; // 中：运算强度（与 Nuke 的 merge 相同，0 表示左侧原样）
  black: number; // 右：黑点，取值相对于该数据自身范围的 0 到 1，而非绝对值
  white: number;
  tint: Tint; // 右：查看单通道时的着色（若干色标和纯色；画面本身不着色，颜色即数据）
  bg: Bg; // 背景：棋盘格或纯色（与运算完全解耦，三种模式下均可用）
  // 背景为纯色时的颜色。此处不写颜色字面量：空值表示色板第一档，颜色值只在
  // platform/palette.ts 中定义（本文件须保持纯净，不 import 页面模块）
  bgColor: string;
  // 点
  pointPx: number; // 大小以屏幕像素计：远近一样大
  pointColor: PointColor;
  pointTint: string; // 单色时的颜色
  // 线
  lineWidth: number; // 屏幕像素，二维与三维共用：人物框、跟踪线、手柄、视锥、相机路线、骨骼、三维曲线、网格
  cameraColor: string;
  boneColor: string;
  // 骨架的画法（view/elements3d.tsx BoneFigure）：实体（八面体 + 球）或线框（棱 + 三环骨点）、八面体骨的粗细倍率、
  // 骨点大小倍率、骨点是否按相连骨长自适应、骨点旁写不写关节名与字号（屏幕像素）
  boneStyle: BoneStyle;
  boneWidth: number;
  jointSize: number;
  jointAdaptive: boolean;
  jointNames: boolean;
  jointNamePx: number;
  // 双骨架编辑（手柄 rig_pair，view/rigPair.tsx）：两副骨架沿世界 Z 一前一后拉开多远（cm；0 = 完全重叠）。只影响显示
  rigPairGap: number;
  // 三维曲线（发丝、毛发导向线、运动轨迹）：线宽取 lineWidth，着色使用曲线自身颜色或单色
  curveColor: CurveColor;
  curveTint: string;
  // 模型
  shading: Shading;
  uvChecker: boolean;
  backFaces: boolean; // 画不画背面：关掉后只画朝向相机的面，叠在背板上调低透明度就是柔和的一层
  overlayOpacity: number; // 透过相机叠在背板上时：模型与角色的不透明度
  overlayTint: string; // 此时它们的颜色（叠在背板上的素模）
  overlayByPerson: boolean; // 「按人物」：每个人按编号各有一种颜色（二维舞台上的人物框、三维中的角色）
  // 画面
  antialias: Antialias;
  // 相机（左键旋转、中键平移、右键左右拖动推拉；view/camera3d.tsx）
  near: number; // 裁剪距离，cm
  far: number;
  cameraPath: boolean; // 有多帧相机时画它走过的路线（相机本身由工具栏的相机按钮管）
  // 场景辅助
  grid: boolean;
  gridSpacing: number; // 网格线间距，cm（每第十条加粗）
  gridSize: number; // 网格总宽，cm；0 表示无边
  axes: boolean;
  axesSize: number; // 角落坐标轴的大小，像素
  background: Background;
  backgroundColor: string;
  lighting: Lighting;
  exposure: number; // 档（stop），作用于灯光
}

/** 叠加色板：可选的颜色；第一档是白色素模（默认颜色）。 */
export const OVERLAY_SWATCHES = ["#e8e8ec", "#FFD60A", "#FF9F0A", "#FF375F", "#64D2FF", "#30D158"];

/** 「按人物」：依次分给各人的颜色。即叠加色板去掉白色，这样任何人（包括编号 0）都不会看起来像未着色的模型。 */
const PEOPLE_SWATCHES = OVERLAY_SWATCHES.slice(1);

/** 「按人物」：按人物编号取一个人的颜色。人物框（二维）与角色（lab2shot:person_id，三维）带同一个编号，
 * 因此同一个人在两个舞台上颜色相同。 */
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
  boneStyle: "solid",
  boneWidth: 1,
  jointSize: 1,
  jointAdaptive: true,
  jointNames: false,
  jointNamePx: 11,
  rigPairGap: 100,
  curveColor: "color",
  // 橙色，取自共用色板，不另写颜色字面量（颜色值只在一处定义）
  curveTint: OVERLAY_SWATCHES[2],
  shading: "smooth",
  uvChecker: false,
  backFaces: true,
  overlayOpacity: 0.6,
  overlayTint: "#e8e8ec",
  overlayByPerson: false,
  antialias: "msaa",
  near: 1,
  far: 1_000_000,
  cameraPath: true,
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

/** 各选项的取值及各自名称的键（面板上的标签，渲染时 t() 取当前语言）。 */
export const CHOICES = {
  op: { add: "ui.display.choice.op.add", over: "ui.display.choice.op.over", mulAlpha: "ui.display.choice.op.mul_alpha", mulRgba: "ui.display.choice.op.mul_rgba" },
  tint: { grey: "ui.display.choice.tint.grey", warm: "ui.display.choice.tint.warm", id: "ui.display.choice.tint.id", red: "ui.display.choice.tint.red", green: "ui.display.choice.tint.green", blue: "ui.display.choice.tint.blue", yellow: "ui.display.choice.tint.yellow", cyan: "ui.display.choice.tint.cyan", magenta: "ui.display.choice.tint.magenta", white: "ui.display.choice.tint.white" },
  bg: { checker: "ui.display.choice.bg.checker", solid: "ui.display.choice.bg.solid" },
  pointColor: { color: "ui.display.own_color", constant: "ui.display.constant_color" },
  shading: { smooth: "ui.display.choice.shading.smooth", flat: "ui.display.choice.shading.flat", wire: "ui.display.choice.shading.wire", wireShaded: "ui.display.choice.shading.wire_shaded" },
  antialias: { off: "ui.display.choice.antialias.off", msaa: "ui.display.choice.antialias.msaa", fxaa: "ui.display.choice.antialias.fxaa", smaa: "ui.display.choice.antialias.smaa" },
  background: { solid: "ui.display.choice.background.solid", gradient: "ui.display.choice.background.gradient" },
  lighting: { headlight: "ui.display.choice.lighting.headlight", rig: "ui.display.choice.lighting.rig" },
  curveColor: { color: "ui.display.own_color", constant: "ui.display.constant_color" },
  boneStyle: { solid: "ui.display.choice.bone_style.solid", wire: "ui.display.choice.bone_style.wire" },
} as const satisfies { [K in keyof ViewOptions]?: Record<string, string> };

/** 数值范围：[最小, 最大]。 */
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
  boneWidth: [0.2, 3],
  jointSize: [0.2, 5],
  jointNamePx: [6, 32],
  rigPairGap: [0, 1000],
} as const satisfies { [K in keyof ViewOptions]?: readonly [number, number] };

/** 只取整数的数值项（滑块、输入框、读存储都按它取整）。 */
export const INTEGERS: ReadonlySet<keyof ViewOptions> = new Set<keyof ViewOptions>(["axesSize", "jointNamePx"]);

const HEX = /^#[0-9a-fA-F]{6}$/;

/** 将存储中的选项（任何内容：旧版、手改、损坏）补全并校正：不认识的键丢弃（已停用的键也由此被忽略），
 * 缺失或取值不对的恢复默认值，数值夹到各自范围内，并保证 near 小于 far。 */
export function sanitize(raw: unknown): ViewOptions {
  const src = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
  const out = { ...DEFAULTS } as Record<string, unknown>;
  for (const key of Object.keys(DEFAULTS) as (keyof ViewOptions)[]) {
    const v = src[key];
    const d = DEFAULTS[key];
    // 合法取值只看 CHOICES 表的自有键（「toString」「constructor」这类原型链上的名字不算取值）
    if (Object.hasOwn(CHOICES, key)) {
      if (typeof v === "string" && Object.hasOwn(CHOICES[key as keyof typeof CHOICES], v)) out[key] = v;
    } else if (typeof d === "number") {
      if (typeof v === "number" && Number.isFinite(v)) {
        const r = (RANGES as Record<string, readonly [number, number]>)[key];
        const n = INTEGERS.has(key) ? Math.round(v) : v;
        out[key] = r ? Math.min(r[1], Math.max(r[0], n)) : n;
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

/** 本浏览器保存的选项；没有或读不出（隐私窗口、存储被禁用、JSON 损坏）时返回默认值。 */
export function loadOptions(storage: Store | null | undefined): ViewOptions {
  try {
    const text = storage?.getItem(STORAGE_KEY);
    const raw = text ? (JSON.parse(text) as Record<string, unknown>) : null;
    return sanitize(raw);
  } catch {
    return { ...DEFAULTS };
  }
}

/** 把选项保存在本浏览器中：只存与默认值不同的项，这样默认值调整后，没改过该项的使用者都会用上新默认值。
 * 浏览器不肯保存时返回 false。 */
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
