import { useEffect, useMemo, useState } from "react";
import { packetOf } from "../state/results";
import type { DataType, HandleDef, Manifest, NodeTypeDef } from "../api";
import { useLook } from "../state/look";
import { elementOf, isList, typeOf } from "../state/items";
import { useDescribed } from "../transfer/described";
import { useGraphSnapshot, type Snapshot } from "../graph/snapshot";
import { upstream } from "../graph/nodes";
import type { GNode } from "../state/graph";
import { PARAM, handlesOf, inputsOf, mainOutput, outputPort, outputsOf } from "../graph/rules";
import { channelsOf, type Mode } from "../model/view2d";
import { useOriginals } from "./useOriginals";
import { useLocalPicture, type LocalPicture } from "./localPick";
import { sourceOf, type Source } from "./origin";
import { cookedWith, staleFp } from "../state/stale";

/** 根据数据类型的视图角色推导当前节点的显示内容，节点本身不编写任何视图代码。节点的所有结果同时显示；输出设置节点
 * 显示其写出的内容；图像空间数据叠加在其来源画面上；三维结果在场景中查看，或透过其相机叠加在衬底上查看。 */

export type Stage = "2d" | "3d";

export interface ViewItem {
  key: string; // 在显示计划中唯一（node.port）
  nodeId: string;
  port: string;
  label: string;
  type: string;
  fp: string | null; // 已计算的包（null 表示尚未计算）
  // 该输出口当前是否置灰（由服务器计算的 `Port.applies`，例如拟合模型选择「无畸变」时的「畸变系数」）。
  // 此时为空属于设计行为而非缺失，因此不参与「该节点是否有内容可显示」的任何判断。
  inactive: boolean;
  // 显示的是上一次的结果（`state/stale.ts`：参数已修改但结构未变）：`fp` 为上一次的包；时间线显示为土黄色，右上角标注「已过期」。
  stale: boolean;
  context: boolean; // 手柄所作用的输入（绘制时比节点自身结果更淡）
  // 数据来源（`view/origin.ts`，完整定义见该文件开头）。
  // 「是否有内容可显示」「属于该输出口自身还是上游衬底」「本机文件能否代表该输出口」均以此为准，
  // 不得在其他位置另写布尔判断，否则各处定义难以保持一致。
  from: Source;
}

export interface DisplayPlan {
  node: GNode;
  def: NodeTypeDef;
  mainPort: string; // 节点的主结果（graph/rules.ts mainOutput）：未作选择时显示的内容
  // 使用者本机的画面（在当前标签页中选择，`view/localPick.ts`；null 表示没有）。
  // 可能由该节点自身读取，也可能由上游读取，区分方式见 `view/origin.ts`。
  file: LocalPicture | null;
  filePort: string; // 该文件可代表的输出口（`view/origin.ts fileStandsFor`；"" 表示不能代表任何输出口）
  items: ViewItem[];
  /** 该节点显示的是上一次的结果（自身输出口中有过期数据，`state/stale.ts`）：无论 2D 还是 3D，时间线整体显示为土黄色。
   * 色带跟随当前显示的节点，与右上角「已过期」标注含义一致。 */
  stale: boolean;
  pictures: ViewItem[]; // 2D：每次显示一张
  // 二维手柄作用的画面（Handle.source，如「分割转遮罩」的分割图）：节点自身的画面还没有数据时，右侧显示它，
  // 手柄才有东西可点（editor/Viewer.tsx）。它是输入，不是节点的一层，不列入 `pictures`
  handleInput: ViewItem | null;
  overlays: ViewItem[]; // 2D：叠加在画面上
  // 叠加物的默认开关：主结果本身是可查看的画面时默认关闭，优先显示交付物本身。
  // 默认叠加的遮罩（例如按面部分割得到的遮罩）其硬边会落在画面上，容易被误认为交付物自身的缺陷。
  // 基础色、ST-map、法线图、深度图同理，它们本身即为查看对象，不应默认被遮盖。
  overlaysOn: boolean;
  elements: ViewItem[]; // 3D：场景中的对象（透过相机查看时也显示在 2D 中）
  pointMaps: ViewItem[]; // 3D：以点云预览的深度图或位置图
  strips: ViewItem[]; // 曲线，显示在两种舞台下方
  values: ViewItem[]; // 基本数值：按原值显示（逐帧数值显示为曲线），显示在舞台或下方条带中
  plate: string | null; // 左侧的原图：上游最近的画面，若无则为该节点自身输出的画面
  plateUpstream: boolean; // 上游是否有输出画面的输出口（无论是否已选文件）：只输出数值的节点据此决定数值显示在舞台还是画面下方条带
  // 同一衬底，附带所属节点与输出口信息：本机原件按节点查找（view/useOriginals.ts readAsk）。
  plateItem: ViewItem | null;
  plateLabel: string; // 在通道下拉中的名称（「原图」，需在一行内显示）
  preview: Mode; // 该节点的预览标签对应的二维档位（三维节点按「仅原图」处理，不使用该值）
  camera: string | null; // 用于透视查看和放置深度的相机：优先使用节点自身的相机，否则取上游
  handles: HandleDef[]; // 在当前参数下生效的节点手柄
  defaultStage: Stage;
  why: Record<Stage, string | null>; // null 表示该舞台有可显示的内容，否则为完整的原因说明
  shortly: Record<Stage, string | null>; // 上述原因的简短版本，显示在视图标签上（`why` 为其悬停提示）
  frames: string | null; // 时间线所跟随的帧所属的包
  // 结果为空列表：没有可绘制的内容，由舞台显示说明（空结果不是错误）。非空列表在此之前已拆分为各个元素，
  // 各元素按自身类型归类，因此除此字段外没有其他位置需要感知「列表」（见 useDisplayPlan）。
  emptyList: boolean;
  // 已在使用者本机找到原件的数据集合（`view/useOriginals.ts`），以字符串形式表示，内容本身无意义。
  // 其变化会触发取数层重新计算（`view/stageSources.ts` 的备忘依赖）。
  // 查找原件是异步的（需要打开系统文件句柄、列出目录、识别包），登记完成后需要借此通知视图重新查询，
  // 否则画面仍停留在服务器数据上，直到切换离开再切回。
  originals: string;
  // 视图中所有数据来源合成的身份（答案所依赖的每一项都必须包含在内），用作取数层的备忘依赖（`view/stageSources.ts`）。
  // 其中已包含 `originals`，因此来源变化、找到原件或更换本机文件都会使其变化。
  sourceKey: string;
}

/** 视图的身份：各数据的来源（`view/origin.ts`）加上本机原件的查找结果。
 * 该身份只在此处拼接。下方「仅 originals 变化」的快速路径若只替换 `originals` 字段而保留旧身份，
 * 找到原件后取数层不会重新计算，画面将停留在代理数据上，且不会报错。 */
const sourceKeyOf = (items: readonly ViewItem[], plateItem: ViewItem | null, originals: string): string =>
  `${items.map((it) => it.from.key).join("/")}|${plateItem?.from.key ?? ""}|${originals}`;

type S = Snapshot;

/** 画面依据的状态：可信的回复；参数刚改、新回复还没到时（`graph/document.ts` 改动后 250 ms 才问服务器），沿用上一次回复
 * （`graph/snapshot.ts` 的 `reply`：下一次回复到达之前照它画）。否则每改一次参数（每松开一次手柄），视图里的上游画面和
 * 场景都会消失一下再出现。只用于画：要不要计算、能不能计算只看可信的回复（state/results.ts useTrustedResults）。 */
const drawnResults = (s: S) => (s.resultsAreTrusted ? s.results : s.reply?.nodes ?? {});

// 只有状态回复标记为 `present` 的输出才真正有包（state/results.ts packetOf）
const cookedFp = (s: S, nodeId: string, port: string) => packetOf(drawnResults(s)[nodeId], port);

/** 该输出口当前显示的数据：当前指纹对应的包；若无，则使用上一次的结果（过期，规则由 `state/stale.ts` 定义）。 */
function resultOf(s: S, nodeId: string, port: string): { fp: string | null; stale: boolean } {
  const fp = cookedFp(s, nodeId, port);
  if (fp) return { fp, stale: false };
  const old = staleFp(s.graphId, (n) => { const g = s.nodes.find((x) => x.id === n); return g ? { typeId: g.data.typeId, params: g.data.params } : undefined; },
                      s.edges, nodeId, port, drawnResults(s)[nodeId]?.fingerprint);
  return { fp: old, stale: !!old };
}

/** 不显示这一项自身的数据（只显示手柄、上游画面）。 */
const withoutData = (it: ViewItem): ViewItem => ({ ...it, fp: null, stale: false });

/** 带手柄的节点只做显示：拖动、点选、画的时候，视图按手柄当前的参数显示已经有的结果，
 * 不计算、不产生任务；松手只记下参数。计算只有右键「计算」。节点自身的结果只在它与手柄当前的参数相符时才显示，
 * 按手柄的种类（每种一条规则，用到它的节点都一样）：
 *
 * - 二维手柄（points、box、corners、person、canvas、figure）：手柄把当前参数直接画在画面上（上游画面，以及手柄声明的
 *   输入 `Handle.source`：「选人」的人物框按点选高亮，「分割转遮罩」的分割图）。节点自身的结果只在它就是按当前参数算出时显示
 *   （可信的回复里有、且未过期）；过期的结果与画出的手柄对不上，参数刚改、新回复还没到时也不知道它还对不对，都不显示。
 * - 变换手柄（transform）：内容按当前参数摆放（view/Stage3D.tsx TransformHandle，用与服务器计算相同的矩阵）。
 *   节点自身的结果按算出它时的参数（state/stale.ts cookedWith）换算到当前参数，任何时候都摆得对，不知道那组参数时不显示；
 *   结果还没算或已过期、而手柄声明了输入时（「3D 变换」的场景），改为摆放输入本身：上游的结果按当前矩阵移动，正是计算会给出的；
 *   没有输入可摆（「创建相机」）时摆放节点上一次的结果。节点自己的结果就是摆好的输入，二者不同时显示。
 *
 * 返回节点自身要显示的条目，以及是否显示手柄声明的输入。 */
function underHandles(s: S, id: string, handles: HandleDef[], own: ViewItem[]): { own: ViewItem[]; sources: boolean } {
  if (!handles.length) return { own, sources: true };
  const placing = handles.find((h) => h.kind === "transform");
  if (!placing) return { own: own.map((it) => (s.resultsAreTrusted && it.fp && !it.stale ? it : withoutData(it))), sources: true };
  const placed = cookedWith(s.graphId, id) !== undefined ? own : own.map(withoutData);
  const fresh = placed.some((it) => it.fp && !it.stale);
  if (fresh || !placing.source) return { own: placed, sources: false };
  return { own: own.map(withoutData), sources: true };
}

/** 上游最近一个类型满足 `wants` 的已计算输出（衬底画面、相机），返回完整条目而非仅包指纹。
 *
 * 需要完整条目的原因：本机原件按「节点及其文件参数」查找（`view/useOriginals.ts readAsk` 的第一个参数为节点 id），
 * 仅凭包指纹无法查询。若只返回 `fp`，下游节点的二维舞台和三维背板将始终显示服务器代理数据，即使序列位于使用者本机磁盘。 */
function nearestUpstream(id: string, s: S, wants: (type: string) => boolean): ViewItem | null {
  for (const up of upstream(id, s.edges).slice(1)) {
    for (const p of outputsOf(s, up)) {
      if (wants(p.type)) {
        const { fp } = resultOf(s, up, p.name);  // 上游过期的结果同样可用作衬底或相机
        if (fp) return item(s, up, p.name, p.label, p.type, true);
      }
    }
  }
  return null;
}

/** 可用作衬底画面的数据：三或四通道的像素数据（RGB / RGBA）。一或两通道的数据（遮罩、UV）叠加在画面上，不作为画面。
 * 视图不区分照片与法线图等语义，该信息属于数据自身的 meta，而非类型。 */
const isPlate = (type: string) => channelsOf(type) >= 3;

/** 相机类型（含各子类型）。 */
const isCamera = (type: string) => type === "scene.camera" || type.startsWith("scene.camera.");

// 来源字段初始为空：它依赖 `file` / `filePort`，二者由视图决定（在 `displayPlan` 中由 `sourced` 填充）。
// 空值即「无内容」，因此中间状态同样有效，不会读到不完整的数据。
const NOWHERE: Source = { where: "none", stale: false, own: false, any: false, file: null, key: "" };

function item(s: S, nodeId: string, port: string, label: string, type: string, context = false, inactive = false): ViewItem {
  const { fp, stale } = resultOf(s, nodeId, port);
  return { key: `${nodeId}.${port}`, nodeId, port, label, type, fp, stale, inactive, context, from: NOWHERE };
}

/** 连接到节点某个输入的数据（多输入时每条连线各一项）。 */
function inputItems(s: S, node: GNode, port: string, context: boolean): ViewItem[] {
  return s.edges
    .filter((e) => e.target === node.id && e.targetHandle === port)
    .flatMap((e) => {
      const src = s.nodes.find((n) => n.id === e.source);
      const p = outputPort(s, e.source, e.sourceHandle);
      return src && p ? [item(s, src.id, p.name, `${src.data.label} · ${p.label}`, p.type, context)] : [];
    });
}

/** 手柄所作用的那个输入（HandleDef.source）上现在的数据包：接进来的第一条线的结果（与视图画手柄输入的是同一个，
 * 沿用 resultOf 的过期规则）。参数面板和节点上把「点选」说成「N 号」时用（editor/pickedPeople.tsx）。 */
export function handleSourceFp(s: S, nodeId: string, port: string): { fp: string | null; list: boolean } {
  const node = s.nodes.find((n) => n.id === nodeId);
  const it = node ? inputItems(s, node, port, true)[0] : undefined;
  return { fp: it?.fp ?? null, list: !!it && isList(it.type) };
}

/** 节点接收数据的输入（不含提升为输入的参数），即其结果的来源。 */
const dataInputs = (s: S, id: string) => inputsOf(s, id).filter((p) => !p.name.startsWith(PARAM));

/** 显示节点 → 画什么（导出只为离线用例：它是纯函数，读快照不读仓库）。 */
export function displayPlan(s: S, nodeId: string | null, expand?: (it: ViewItem) => ViewItem[], emptyList = false,
                            spaceOf: (fp: string) => string | null = () => null, originals = "",
                            file: LocalPicture | null = null): DisplayPlan | null {
  const node = s.nodes.find((n) => n.id === nodeId);
  const def = node ? s.nodeDefs[node.data.typeId] : undefined;
  if (!node || !def) return null;
  const types = s.types;
  const handles = handlesOf(s, node.id); // 按服务器判定当前生效的手柄（Handle.when）

  // 节点自身的结果；输出设置节点显示其写出的内容（其文件按来源数据显示，即 "inputs" 角色）；手柄的源输入也一并显示，
  // 供手柄操作。没有别的穿透：激活哪个节点就显示哪个节点自己的输出口，切换也按它自己的 out 口
  const madeFrom = (items: ViewItem[]): ViewItem[] =>
    items.flatMap((it) => {
      const src = s.nodes.find((n) => n.id === it.nodeId);
      if (typeOf(types, it.type)?.in_2d !== "inputs") return [it];   // 联合类型同样需要识别
      if (!src) return [];
      return madeFrom(dataInputs(s, src.id).flatMap((p) => inputItems(s, src, p.name, false)));
    });
  const outputs = outputsOf(s, node.id); // 由节点参数和输入决定（读取序列：每层一个）
  // 节点的主结果（`graph/rules.ts mainOutput`）：未作选择时显示的内容。
  const mainPort = mainOutput(s, node.id)?.name ?? "";
  // 本机文件可代表的层：该节点的主输出。浏览器解码本机 EXR 时只取文件自身的颜色通道，即主画面所在的层，
  // 无法取得文件中的其他层（详见 `view/origin.ts fileStandsFor`）。
  const filePort = mainPort;
  // 「骨架姿势」手柄只画它那一侧输入的骨架（舞台的手柄层，数据来自状态回复 handle_data），不改显示什么：与没有手柄同一规则
  const placing = handles.filter((h) => h.kind !== "skeleton_pose");
  const shown = underHandles(s, node.id, placing, madeFrom(outputs.map((p) => item(s, node.id, p.name, p.label, p.type, false, !!p.inactive))));
  const own = shown.own;
  // 没有输出口的节点（「输出」：收文件、打包）没有自己的结果，就不画结果：显示的永远是激活节点自己的输出
  const sources = shown.sources ? placing.flatMap((h) => (h.source ? inputItems(s, node, h.source, true) : [])) : [];
  // 结果为列表时在此拆分为各个元素（useDisplayPlan 读取其包说明之后）。每个元素使用自己的包和类型，
  // 后续处理与节点输出多份并列结果没有区别。不为列表单独提供「选择第几项」的控件：拆分列表的节点是普通运算节点，
  // 显示内容仍为原图加上各个结果。
  const raw = expand ? [...sources, ...own].flatMap(expand) : [...sources, ...own];
  // 来源在此统一计算一次（`view/origin.ts`）：后续各行以及视图的其他文件都只读取 `it.from`。
  const sourced = (it: ViewItem): ViewItem => ({ ...it, from: sourceOf(it, file, filePort) });
  const items = raw.map(sourced);

  // 输出口类型可能是联合类型（`scene.skeleton|scene.character`），而目录中只有单一类型：
  // 统一经 `state/items.ts typeOf` 处理（由其拆分 `|`），不得在此另行实现。
  const role = (it: ViewItem): DataType | undefined => typeOf(types, it.type);
  const pictures = items.filter((it) => !it.context && role(it)?.in_2d === "picture");
  const handleInput = items.find((it) => it.context && it.fp && role(it)?.in_2d === "picture") ?? null;
  const overlays = items.filter((it) => role(it)?.in_2d === "overlay");
  const elements = items.filter((it) => role(it)?.in_3d === "element");
  const pointMaps = items.filter((it) => role(it)?.in_3d === "points");
  const strips = items.filter((it) => role(it)?.in_2d === "strip");
  const values = items.filter((it) => role(it)?.in_2d === "value");

  // 主结果本身是否为画面（`in_2d === "picture"` 且为 `main` 输出口）
  const overlaysOn = !pictures.some((it) => it.port === mainPort);
  const ownImage = pictures.find((it) => isPlate(it.type) && it.fp) ?? null;
  // 衬底所属的节点与输出口（`plateItem`）：二维舞台和三维背板均显示它，
  // 而在使用者本机查找原件时需要知道所属节点（见上方 `nearestUpstream` 的注释）。
  const upstreamPlate = nearestUpstream(node.id, s, isPlate);
  const plateItem = upstreamPlate ? sourced(upstreamPlate) : ownImage;
  const plate = plateItem?.fp ?? null;
  // 通道数已知的画面输出口（`image.3`），或尚无法确定通道数的输出口（`image`：「读取序列」在选择文件并读取后才确定其层）
  const plateUpstream = upstream(node.id, s.edges).slice(1).some((up) => outputsOf(s, up).some((p) => p.type === "image" || isPlate(p.type)));
  const plateLabel = "原图"; // 左侧始终为原图，名称固定为「原图」
  const cameraItem = elements.find((it) => it.type === "scene.camera" && it.fp);
  const camera = cameraItem?.fp ?? nearestUpstream(node.id, s, isCamera)?.fp ?? null;

  // 是否需要相机才能放入三维场景：单通道数据为逐像素距离，需沿镜头方向反投影，必须有相机；三通道数据本身即为坐标，
  // 仅当包声明其位于相机空间（meta.space）时才需要相机。判断依据为通道数和包自身的信息，而非类型名。
  // 可作为点云显示的数据：单通道（深度）需配相机；三通道仅当包声明了 `space`（位置图）时才算，
  // 普通 RGB 画面同样是三通道，但不是坐标。尚未计算、没有包的画面不得视为世界坐标
  // （`spaceOf("") !== "camera"` 恒为真），否则双击「读取序列」时 3D 档位也会被判为有内容，视图停留在 3D 而不切换到 2D。
  const asPoints = pointMaps.filter((it) => !!it.fp && (channelsOf(it.type) < 3 || spaceOf(it.fp) !== null));
  const worldPositions = asPoints.some((it) => channelsOf(it.type) >= 3 && spaceOf(it.fp!) !== "camera");
  // 舞台无内容的原因：简短版本显示在视图标签上（在 1280 宽度下须在一行内显示），完整说明（缺少什么、如何处理）
  // 作为其悬停提示和切换按钮的提示文字。
  // 在画面上绘制的节点（手柄 stage 为 2d：手绘遮罩、平面四角、火柴人）的 2D 舞台始终有内容，即显示所需的画面，
  // 即使该节点自身的结果是三维的（「Sketch2Anim 动作生成」输出骨架动画），或上游画面尚未计算完成
  // （尚未计算完成属于等待状态，而非无内容；完全未连接时，视图在选择舞台之前已显示「还没有结果」页面）。
  // 缺少该判断时，服务器声明的画布手柄将因无法进入 2D 舞台而不可操作；此外舞台选择在节点间保持，
  // 中途一次误判为「2D 无内容」会使舞台一直停留在 3D。
  const draws2D = handles.some((h) => h.stage === "2d");
  const missing: Record<Stage, [string, string] | null> = {
    "2d":
      // 浏览器已持有该节点的画面时，2D 即有内容（`DisplayPlan.file`，view/localPick.ts）。
      // 下一行统计的是节点是否有输出口，而输出口来自最后一次状态回复
      // （`graph/rules.ts pendingPorts`：`s.reply?.nodes[id]?.ports ?? 类型在默认参数下的输出口`）。
      // 选择文件后发出的 `/api/status` 返回时文件仍在上传，服务器无法读取，`outputs` 为空数组；
      // 空数组不是 nullish，`??` 不会回退到默认的「图像」输出口。若只依据该值，整个上传过程都会判定为「没有 2D 结果」
      // 并显示灰屏，而本机文件实际一直可用；上行带宽较低时该过程会持续很久。
      //
      // 「该档位是否有内容」仅在此处声明，各组件不得另行判断。
      // 只输出三维结果的节点，其 2D 显示上游画面原样播放：节点自身在 2D 中无内容，有衬底即视为有内容。
      // 透过节点自身相机查看由 3D 舞台的视角下拉负责，2D 舞台不处理。
      !!file || pictures.length || overlays.length || strips.length || values.length || draws2D || plate
        ? null
        : ["没有 2D 结果", "这个节点没有能在 2D 里看的结果，上游也没有画面"],
    "3d":
      elements.length || (asPoints.length && (camera || worldPositions))
        ? null
        : asPoints.length
          ? ["还没配相机", "这张图要配上相机才能放进三维场景（每个像素的距离要沿镜头方向反投影）：接一台相机，或看它的解算节点"]
          : ["没有三维结果", "这个节点没有三维结果"],
  };
  const why: Record<Stage, string | null> = { "2d": missing["2d"]?.[1] ?? null, "3d": missing["3d"]?.[1] ?? null };
  const shortly: Record<Stage, string | null> = { "2d": missing["2d"]?.[0] ?? null, "3d": missing["3d"]?.[0] ?? null };
  // 节点的预览标签（NodeTypeDef.preview，由服务器统一计算：lab2shot/nodes/base.py default_preview）：
  // 一个标签同时决定两种舞台，`scene` 对应三维，其余三个对应二维的三个档位（仅原图 / 运算 / 仅结果）。
  // 显示的是「由什么做成的」（输出设置写出的、「输出」收来的：上面 madeFrom 换成的来源节点的结果）时，按那个来源节点
  // 自己的预览标签：这类节点自己的标签是「仅原图」（服务器 default_preview：没有画面输出），照它就只看得到原图、看不到结果
  const lead = items.find((it) => !it.context && it.nodeId !== node.id);
  const tag = (lead && s.nodeDefs[s.nodes.find((n) => n.id === lead.nodeId)?.data.typeId ?? ""]?.preview) || def.preview;
  const preview: Mode = tag === "compute" ? "over" : tag === "scene" ? "plate" : tag;
  // 默认舞台跟着主结果走：主结果是三维数据（in_3d 为 element：场景、点云、相机……）且已经有结果时进三维，即使节点
  // 带二维手柄（「3D 跟踪点」在画面上点要跟的点）——手柄的舞台只决定「二维也可用」（上面 draws2D），不决定默认。
  // 还没有结果时（点还没点、没算过）三维里什么都没有，照标签进二维点点
  const mainIn3d = items.some((it) => !it.context && !!it.fp && it.port === mainPort && it.nodeId === node.id && role(it)?.in_3d === "element");
  const preferred: Stage = tag === "scene" || mainIn3d ? "3d" : "2d";
  const defaultStage: Stage = why[preferred] === null ? preferred : why[preferred === "2d" ? "3d" : "2d"] === null ? (preferred === "2d" ? "3d" : "2d") : preferred;
  const frames = items.find((it) => it.fp && !it.context)?.fp ?? items.find((it) => it.fp)?.fp ?? plate;
  const sourceKey = sourceKeyOf(items, plateItem, originals);
  const stale = items.some((it) => !it.context && it.stale);
  return { node, def, mainPort, file, filePort, items, stale, pictures, handleInput, overlays, elements, pointMaps, strips, values, overlaysOn, plate, plateUpstream, plateItem, plateLabel, camera, handles, preview, defaultStage, why, shortly, frames, emptyList, originals, sourceKey };
}

/** 当前显示节点的显示计划，在图、结果或目录变化时重新计算。
 *
 * 结果为列表时，视图显示其中每个元素：列表自身的包说明记录了所含元素及其对应的包（`meta.items`，不复制任何数据字节），
 * 读取后将该条目拆分为每个元素一项，再次经过 `displayPlan`，使每个元素与从未放入列表的结果一样被归类、绘制和度量。
 * 包说明读取完成之前使用未拆分的版本（列表本身没有视图角色，不绘制任何内容）。
 * 拆分列表的节点是普通运算节点，不另外提供「选择第几项」的控件。 */
export function useDisplayPlan(): DisplayPlan | null {
  const snap = useGraphSnapshot();
  const displayId = useLook((s) => s.displayId);
  // 本机是否持有该节点的画面（使用者在当前标签页中选择的文件）：它决定 2D 档位是否有内容，
  // 因此需要纳入显示计划（原因见上方 `missing["2d"]` 部分）。
  const localPicture = useLocalPicture(displayId);
  // 各点云数据声明的坐标系（meta.space）：只用来判断要不要相机，不影响哪些数据算点云——所以按上一次算出的 pointMaps
  // 去读，读到了再算一遍（不为拿 pointMaps 另算一遍整张计划）
  const [pointFps, setPointFps] = useState<string[]>([]);
  const spaces = usePointSpaces(pointFps);
  const plain = useMemo(
    () => displayPlan(snap, displayId, undefined, false, (fp) => spaces[fp] ?? null, "", localPicture),
    [snap, displayId, spaces, localPicture],
  );
  const fpsNow = (plain?.pointMaps ?? []).flatMap((it) => (it.fp ? [it.fp] : []));
  useEffect(() => {
    setPointFps((was) => (was.join() === fpsNow.join() ? was : fpsNow));
  }, [fpsNow.join()]); // eslint-disable-line react-hooks/exhaustive-deps
  // 视图中属于列表的条目（服务器已计算的列表：有包，包中记录其所含元素）
  const listFps = useMemo(() => (plain?.items ?? []).flatMap((it) => (!it.context && it.fp && !it.inactive && isList(it.type) ? [it.fp] : [])), [plain]);
  const parts = useListParts(listFps);
  // 在使用者本机为视图所需的每份数据查找原件（`view/useOriginals.ts`）：找到后登记，取帧路径会自行查询。
  // 返回的字符串参与下方的备忘依赖；登记是异步的，缺少该依赖时登记完成后视图不会重新计算。
  // 衬底也需要查询：`plain.items` 只包含该节点自身的输出口和手柄相关条目（`displayPlan` 的 `sources + own`），
  // 不包含上游衬底画面，而二维舞台左侧和三维背板显示的正是衬底。若不单独查询，下游节点的衬底将始终是服务器代理数据；
  // 读取节点位于本机时，下游使用其作为衬底也应从本机读取，无需传输且保持完整精度。
  // 同一个包只查询一次（`useOriginals asksFor` 按 `fp` 去重），因此节点自身输出该画面时不会重复查询。
  const originals = useOriginals(displayId, useMemo(
    () => (plain ? (plain.plateItem ? [...plain.items, plain.plateItem] : plain.items) : []), [plain]));
  return useMemo(() => {
    if (!plain) return plain;
    const read = listFps.filter((fp) => parts[fp]); // 包说明已读取的列表
    // 空列表：不绘制任何内容，由舞台说明原因（空结果不是错误）。
    // 仅当所有列表的包说明均已读取且均为空时才视为空（尚未读取不算）。
    // 同时要求视图中其他输出口均无内容（`it.from.any`，定义见 `view/origin.ts`）：例如 COLMAP 输出相机、点云和
    // 一个「畸变系数」列表，列表为空（无畸变）并不意味着该节点没有可显示的内容。
    const emptyList = listFps.length > 0 && read.length === listFps.length && listFps.every((fp) => !parts[fp].length)
      && !plain.items.some((it) => !it.context && !isList(it.type) && it.from.any);
    // 仅本机原件查找结果变化的快速路径：身份必须同步更新（原因见上方 `sourceKeyOf` 的注释）。
    if (!read.length)
      return originals === plain.originals
        ? plain
        : { ...plain, originals, sourceKey: sourceKeyOf(plain.items, plain.plateItem, originals) };
    // 重新经过 displayPlan 而非在外部修改数组：类型变化的条目需要按新类型重新归类（列表没有视图角色，
    // 单个人物框才能绘制），归类逻辑仅在 displayPlan 中实现。
    const expand = (it: ViewItem): ViewItem[] => {
      const inside = it.fp && !it.context && isList(it.type) ? parts[it.fp] : undefined;
      if (!inside) return [it];
      // 服务器已计算的列表：每个元素有独立的包，在视图中作为并列的多份结果显示。
      // 名称仍使用该输出口的名称：叠加显示开关和画面下拉均按输出口计一项，各元素的区分已在画面上标注（1 号、2 号），不依赖名称。
      return inside.map((one) => ({ ...it, key: `${it.key}#${one.name}`, type: elementOf(it.type), fp: one.packet }));
    };
    return displayPlan(snap, displayId, expand, emptyList, (fp) => spaces[fp] ?? null, originals, localPicture);
  }, [plain, listFps, parts, originals, snap, displayId, spaces, localPicture]);  // 依赖列表必须包含结果所依赖的全部输入
}

/** 每个列表自身的包说明（`meta.items`）：所含元素及其对应的包，不传输任何数据字节。
 * 读到了才算数（服务器说是空列表才是空，由舞台说明）；读失败的由 transfer/described.ts 退避后再读。 */
function useListParts(fps: string[]): Record<string, { name: string; packet: string }[]> {
  const got = useDescribed<Manifest>("manifest", fps);
  return useMemo(() => Object.fromEntries(fps.flatMap((fp, i) => {
    const m = got[i];
    return m ? [[fp, (m.meta.items as { name: string; packet: string }[] | undefined) ?? []]] : [];
  })), [got]); // eslint-disable-line react-hooks/exhaustive-deps -- got 与 fps 一一对应
}

/** 各点云预览包声明的坐标系（`meta.space`：world / camera / canonical；未声明时视为 world）。
 * 含义来自包自身携带的信息，而非类型名（2D 数据只有通道数）：视图只区分单通道和三通道，是否需要相机放入场景由包自身声明。
 * 包说明经 transfer/described.ts 读（页面缓存里按代次的一份）。 */
function usePointSpaces(fps: string[]): Record<string, string> {
  const got = useDescribed<Manifest>("manifest", fps);
  // 读不到的视为未声明（不列出），三维舞台仍会说明缺少的内容
  return useMemo(() => Object.fromEntries(fps.flatMap((fp, i) => (got[i] ? [[fp, String(got[i]!.meta.space ?? "world")]] : []))),
    [got]); // eslint-disable-line react-hooks/exhaustive-deps -- got 与 fps 一一对应
}
