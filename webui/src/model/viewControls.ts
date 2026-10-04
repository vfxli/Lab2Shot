import { blends, type Mode } from "./view2d";
import type { ViewOptions } from "./viewOptions";

/** 查看器各控件是否适用的唯一一张表：显示选项与视图自带的小部件属于浏览器状态，因此在这里集中声明一次，
 * 由 view/available.ts（applies.ts resolveLocal）并入统一的可用性答案；DisplayOptions.tsx、Viewer.tsx、Stage3D.tsx
 * 与其他对象一样用 `usable` 读取。控件不会消失：
 * 不适用的控件置灰并说明原因（`WHY_OFF`），不随条件显示或隐藏。各面板包含哪些分页按二维 / 三维划分（属于结构），
 * 而非依靠控件消失。节点参数仍由服务器计算。
 *
 * 纯模块：除 model/view2d.ts 的纯函数外只引用类型；
 * 本文件不含中文文本：`WHY_OFF` 只写消息编号，模板位于 lab2shot/i18n/<lang>/messages/web.toml。 */

export interface ViewFacts {
  stage: "2d" | "3d";
  // 2D 预览链：当前模式、右侧数据的通道数、右侧是否只取单通道
  mode: Mode; // 仅原图 / 运算 / 仅结果
  channels: number; // 右侧数据的通道数（0 表示尚无图像）
  single: boolean; // 右侧只取单通道：屏幕颜色由映射产生，着色才有意义
  ready: boolean; // 该帧已到达
  // 舞台画出的、受控件影响的内容：三维的种类（points、model、character、skeleton、camera、curves），
  // 或二维舞台上的叠加层（boxes、tracks2d）与节点的手柄。
  shows: ReadonlySet<string>;
  options: ViewOptions;
}

type ViewWidget = "axesGizmo" | "fitRange";
export type ViewControl = keyof ViewOptions | ViewWidget;
type Applies = (f: ViewFacts) => boolean;

const in3d: Applies = (f) => f.stage === "3d";
const points: Applies = (f) => in3d(f) && f.shows.has("points");
const meshes: Applies = (f) => in3d(f) && (f.shows.has("model") || f.shows.has("character"));
const curves: Applies = (f) => in3d(f) && f.shows.has("curves");

// 二维舞台上必须存在画面，这一组控件才适用：当二维舞台只显示三维曲线或骨架等结果时，
// 整条预览链（黑白点、着色、运算、背景）均不起作用。
// 控件可点击却无反应比置灰更难理解，因此置灰并说明原因。
const in2d: Applies = (f) => f.stage === "2d" && f.channels > 0;
// 右侧部分（取通道 → 黑白点 → 着色）在「运算」与「仅结果」两种模式中均存在；中间运算仅在「运算」模式中存在。
// 三种模式属于结构划分，而非控件消失（与 2D / 3D 两个舞台、面板分页遵循同一规则）：切换模式改变的是这一行包含的部分，
// 而非同一行中控件的显示与隐藏。
const hasRight: Applies = (f) => in2d(f) && f.mode !== "plate";
const merging: Applies = (f) => in2d(f) && f.mode === "over";

export const VIEW_CONTROLS: Record<ViewControl, Applies> = {
  // 2D 预览链：集中声明，各处不单独判断
  op: merging,
  // 强度仅对「加」有意义：「乘」用于按遮罩预览抠像，固定为 1（model/view2d.ts mixOf）
  mix: (f) => merging(f) && blends(f.options.op),
  black: hasRight,
  white: hasRight,
  fitRange: hasRight, // 「拉满」按钮：按数据的真实最小值与最大值设置
  // 画面（三或四条通道一起显示）不着色：颜色即数据本身。「乘」时同样不着色，参与乘法的是该通道的数值，
  // 颜色不参与计算
  tint: (f) => hasRight(f) && f.single && !(f.mode === "over" && !blends(f.options.op)),
  bg: in2d,
  bgColor: in2d, // 背景为单个下拉：棋盘格与各颜色是同一选项的不同取值，不存在「颜色不适用」的情况
  // 点（点大小只按屏幕像素计，因此 pointPx 只取决于有没有点云）
  pointPx: points,
  pointColor: points,
  pointTint: (f) => points(f) && f.options.pointColor === "constant",
  // 线：三维中选中物的轮廓线始终存在；三维曲线也共用线宽
  lineWidth: (f) => (f.stage === "3d" ? true : f.shows.has("boxes") || f.shows.has("tracks2d") || f.shows.has("handle")),
  cameraColor: (f) => in3d(f) && f.shows.has("camera"),
  boneColor: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")),
  boneStyle: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")),
  boneWidth: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")),
  jointSize: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")),
  jointAdaptive: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")),
  jointNames: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")),
  jointNamePx: (f) => in3d(f) && (f.shows.has("skeleton") || f.shows.has("character")) && f.options.jointNames,
  rigPairGap: in3d, // 只在双骨架编辑的功能条上（那时舞台必在三维）
  cameraPath: (f) => in3d(f) && f.shows.has("cameraPath"),
  curveColor: curves,
  curveTint: (f) => curves(f) && f.options.curveColor === "constant",
  // 模型
  shading: meshes,
  uvChecker: meshes,
  backFaces: meshes,
  overlayOpacity: meshes,
  overlayTint: meshes,
  overlayByPerson: (f) => (f.stage === "2d" ? f.shows.has("boxes") : meshes(f) && f.shows.has("character")),
  // 画面、相机与场景辅助：三维舞台始终具备
  antialias: in3d,
  near: in3d,
  far: in3d,
  grid: in3d,
  gridSpacing: (f) => in3d(f) && f.options.grid,
  gridSize: (f) => in3d(f) && f.options.grid,
  axes: in3d,
  axesSize: (f) => in3d(f) && f.options.axes,
  axesGizmo: (f) => in3d(f) && f.options.axes,
  background: in3d,
  backgroundColor: in3d,
  lighting: in3d,
  exposure: in3d,
};

/** 各控件不适用时悬停提示原因的消息编号（参照 Houdini 的 disable_when），集中定义，工具条与显示选项面板共用。
 * 中文模板位于消息目录（`lab2shot/i18n/<lang>/messages/web.toml` 的 `I-VIEW-OFF*`，代码中只写编号）；
 * 文字经由 `view/available.ts` 获取。
 *
 * 每个控件都必须在此有对应条目（`Record<ViewControl, string>`，缺一项即类型检查不过），以保证置灰的同时说明原因；
 * 遗漏条目会导致控件置灰却无说明。编号按条件划分，而非每个控件一条：
 * 条件相同的控件共用一条（没有点云、没有模型、属于另一视图的选项等）。 */
export const WHY_OFF: Record<ViewControl, string> = {
  // 二维预览链
  op: "I-VIEW-OFFONLY2D",
  mix: "I-VIEW-OFFMIXLOCKED",
  black: "I-VIEW-OFFONLY2D",
  white: "I-VIEW-OFFONLY2D",
  fitRange: "I-VIEW-OFFONLY2D",
  tint: "I-VIEW-OFFNOTINT",
  bg: "I-VIEW-OFFONLY2D",
  bgColor: "I-VIEW-OFFONLY2D",
  // 点
  pointPx: "I-VIEW-OFFNOPOINTS",
  pointColor: "I-VIEW-OFFNOPOINTS",
  pointTint: "I-VIEW-OFFNOTCONSTANT",
  // 线
  lineWidth: "I-VIEW-OFFNOLINES",
  cameraColor: "I-VIEW-OFFNOCAMERA",
  boneColor: "I-VIEW-OFFNOSKELETON",
  boneStyle: "I-VIEW-OFFNOSKELETON",
  boneWidth: "I-VIEW-OFFNOSKELETON",
  jointSize: "I-VIEW-OFFNOSKELETON",
  jointAdaptive: "I-VIEW-OFFNOSKELETON",
  jointNames: "I-VIEW-OFFNOSKELETON",
  jointNamePx: "I-VIEW-OFFNOJOINTNAMES",
  rigPairGap: "I-VIEW-OFFONLY3D",
  cameraPath: "I-VIEW-OFFNOCAMERAPATH",
  curveColor: "I-VIEW-OFFNOCURVES",
  curveTint: "I-VIEW-OFFCURVENOTCONSTANT",
  overlayByPerson: "I-VIEW-OFFNOPERSON",
  // 模型
  shading: "I-VIEW-OFFNOMESH",
  uvChecker: "I-VIEW-OFFNOMESH",
  backFaces: "I-VIEW-OFFNOMESH",
  overlayOpacity: "I-VIEW-OFFNOMESH",
  overlayTint: "I-VIEW-OFFNOMESH",
  // 相机、场景、渲染：三维视图始终具备，在二维画面上属于另一视图的选项
  antialias: "I-VIEW-OFFONLY3D",
  near: "I-VIEW-OFFONLY3D",
  far: "I-VIEW-OFFONLY3D",
  grid: "I-VIEW-OFFONLY3D",
  gridSpacing: "I-VIEW-OFFGRIDOFF",
  gridSize: "I-VIEW-OFFGRIDOFF",
  axes: "I-VIEW-OFFONLY3D",
  axesSize: "I-VIEW-OFFAXESOFF",
  axesGizmo: "I-VIEW-OFFGIZMO",
  background: "I-VIEW-OFFONLY3D",
  backgroundColor: "I-VIEW-OFFONLY3D",
  lighting: "I-VIEW-OFFONLY3D",
  exposure: "I-VIEW-OFFONLY3D",
};
