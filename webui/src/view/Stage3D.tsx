import { usable } from "../api/applies";
import { inverse } from "../model/math3d";
import { useAppMode } from "../editor/AppMode";
import { viewAvailable } from "./available";
import { msg, textOf } from "../messages/message";
import { t } from "../i18n/t";
import { useLang } from "../i18n/lang";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame as useEachDraw, useThree } from "@react-three/fiber";
import { KeepContext, Redraw } from "./canvasLife";
import * as THREE from "three";
import { setParams } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useHandleView } from "../state/handleView";
import { Picker, ViewCamera, type Lens } from "./camera3d";
import { cameraAt } from "./elements3d";
import { boxSegments, FatLines } from "./lines3d";
import type { DisplayPlan, ViewItem } from "./plan";
import { Cloud } from "./points3d";
import { useManifest } from "./Stage2D";
import { useGens } from "../transfer/gens";
import { pointsKey } from "../transfer/frameKey";
import { clearLoads, readyAcross, reportLoads } from "../transfer/readiness";
import { useDevicePixelRatio } from "../platform/size";
import { serverFrames, useDecodedFrames, useFrame, useLoadedFrames } from "../transfer/frames";
import { isPlane } from "../transfer/plane";
import { Axes, Background, GroundGrid, ImagePlane, Lights, Pipeline } from "./render3d";
import { loadPoints, useLoaded, usePartialPoints, useScenes, useSceneVersions, useShownScenes, type CameraData } from "./sceneData";
import type { Partial as PartialResult } from "./partial";
import { StageContext, StageState } from "./stageState";
import { DragGizmo, type DragMode } from "./dragGizmo";
import { SceneElement } from "./sceneElement";
import { PoseLayers, ReferenceLayer, SkinnedPose, poseDataOf, type PoseHandle } from "./stageLayers";
import { SkeletonPosePanel, usePoseSelection } from "./skeletonPose";
import { rowsOf as poseRowsOf } from "../model/skeletonPose";
import { RigPairLayer, RigPairPanels, RigPairWaiting } from "./rigPair";
import { useRigPairView } from "../state/rigPairView";
import type { RigPairData } from "../api";
import { role3d } from "./handleEditing";

import { VIEWER_SLOT, useStageNotes, useViewCamera, useViewer, useViewerNote, useViewOptions, useView2D, useView2DNav } from "../state/viewer";
import { useShortcut } from "../platform/keys";
import { useResults } from "../state/results";
import { getNodeDefs } from "../state/catalog";
import { draggedPlace, placeMatrix } from "../model/places";
import { cookedWith, useLastGood } from "../state/stale";
import { Preparing } from "./StageHud";
import { tipOf } from "../platform/tips";

/** 三维舞台：节点的全部三维结果放在一个场景里（相机及其路径、模型与蒙皮角色、骨架、点云、三维曲线），深度图 / 位置图
 * 作为点云预览，以及变换手柄。场景中任何一台相机都可以透过去看（视角菜单，与 Houdini 相同）：每一帧取它的视角，
 * 显示它的片门并压暗片门以外，有背板时背板作为图像平面放在后面。画法遵循显示选项（model/viewOptions.ts）。
 * 渲染由浏览器完成（WebGL），服务器只准备数据。 */



/** 变换手柄：按节点声明的摆放方式（它的参数、参数顺序与旋转顺序，model/places.ts）摆放节点产出内容的操纵器。
 * 它只做显示、不做计算：显示的内容用与计算相同的矩阵，放在当前参数（拖动中则为这次拖动会写的参数）所摆的位置；松手只记下参数
 * （一步撤销；不计算，「计算」在右键里）。显示什么由 view/plan.ts underHandles 决定：
 * - `input`：手柄的输入（节点还没有当前结果时，「3D 变换」的上游场景），节点尚未摆放过它：直接放在该摆放位置上；
 * - `own`：节点自身的结果，它已按计算时的参数摆放过（state/stale.ts cookedWith）：再乘以当前摆放与那次摆放之逆的积，
 *   因此过期的结果、或下一次状态回复到达前的旧结果，都待在当前参数所摆的位置，不会跳回去。 */
function TransformHandle({ input, own, mode }: { input: React.ReactNode; own: React.ReactNode; mode: DragMode }) {
  const displayId = useLook((s) => s.displayId);
  const graphId = useCookInputs((s) => s.graphId);
  const node = useCookInputs((s) => (displayId ? s.nodes[displayId] : undefined));
  const statusPlaces = useResults((s) => (displayId ? s.reply?.nodes[displayId]?.places : undefined));
  const places = statusPlaces ?? (node ? getNodeDefs()[node.typeId]?.places : null) ?? null;
  // 记下结果时重新读取：它计算时所用的参数随结果一起记录
  useLastGood((s) => s.byGraph[graphId]?.[displayId ?? ""]);
  const [live, setLive] = useState<Record<string, unknown> | null>(null); // 拖动中：这次拖动会写的参数
  const p = node?.params ?? {};
  const placedBy = displayId ? cookedWith(graphId, displayId) : undefined;
  const key = (q: Record<string, unknown>) => (places ? JSON.stringify([q[places.translate], q[places.rotate], places.scale ? q[places.scale] : 1]) : "");
  const current = useMemo(() => (places ? placeMatrix(places, p) : null), [places, key(p)]); // eslint-disable-line react-hooks/exhaustive-deps
  // 计算时那次摆放的逆（math3d.inverse：判不可逆只有那一处）；不可逆（缩放为 0）时不画自己的结果，并说明
  const placedInverse = useMemo(() => (places && placedBy ? inverse(placeMatrix(places, placedBy)) : null), [places, key(placedBy ?? {})]); // eslint-disable-line react-hooks/exhaustive-deps
  const before = placedInverse ? new THREE.Matrix4().fromArray(placedInverse) : null;
  const singular = !!places && !!placedBy && !placedInverse;
  const note = useStageNotes((n) => n.put);
  const lang = useLang((s) => s.lang); // 通知里存的是文字：换语言时重写
  useEffect(() => {
    note("absent", singular ? { text: textOf(msg("N-VIEW-PLACEDSINGULAR")), tip: tipOf("error", textOf(msg("I-VIEW-PLACEDSINGULARWHY"))) } : null);
    return () => note("absent", null);
  }, [singular, lang]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!places || !current) return <>{input}{own}</>;
  const at = new THREE.Matrix4().fromArray(live ? placeMatrix(places, { ...p, ...live }) : current); // 显示的位置：拖动中取会写的参数，否则取当前参数
  return (
    <>
      <group matrix={at} matrixAutoUpdate={false}>
        {input}
      </group>
      {/* 只有 `before` 已知时 view/plan.ts 才显示节点自身的结果 */}
      {before && (
        <group matrix={at.clone().multiply(before)} matrixAutoUpdate={false}>
          {own}
        </group>
      )}
      {/* 写的是这次拖动的增量套在按下时的参数上（model/places.ts draggedPlace）：只点不拖不写；一步撤销；不计算，
          「计算」在右键里 */}
      <DragGizmo at={current} mode={mode} size={0.8} uniform={!!places.scale}
        onDrag={(d) => setLive(draggedPlace(places, p, d))}
        onEnd={(d) => {
          setLive(null);
          const next = d && node ? draggedPlace(places, p, d) : null;
          if (next) setParams(displayId!, next);
        }} />
    </>
  );
}

/** 选中物体的包围框，用强调色画。包围框在每次绘制之前读（useFrame）：那时各部件已套上这一帧的姿势（它们的 effect 已跑过）、
 * 这次提交里才登记的部件也已在 stage.pickables 里；在渲染中读拿到的是上一帧的位置。只在框变了时重画。 */
function Picked({ stage, keyOf, width }: { stage: StageState; keyOf: string | null; width: number }) {
  const [box, setBox] = useState<THREE.Box3 | null>(null);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => invalidate(), [keyOf, invalidate]); // 换了选中：画一次，好读新选中的框
  useEachDraw(() => {
    const b = keyOf ? stage.pickables.get(keyOf)?.bounds() ?? null : null;
    const now = b && !b.isEmpty() ? b : null;
    setBox((was) => (was === now || (was && now && was.equals(now)) ? was : now));
  });
  if (!box) return null;
  return <FatLines segments={boxSegments(box)} color="#0a84ff" width={Math.max(1, width)} overlay opacity={0.9} />;
}

/** 场景中可以透过去看的一台相机。 */
interface SceneCamera {
  key: string; // 这一份的键（包 + 层级路径），只用来比对
  fp: string; // 它所在的包
  cam: CameraData;
  label: string; // 它在层级中的位置
}

/** 相机在这一帧的状态：位姿、片门的竖直视场角、焦距。 */
function lensOf(c: SceneCamera, frame: number): Lens & { focalMm: number } {
  const at = cameraAt(c.cam, frame);
  return { pose: at.matrix, fovV: THREE.MathUtils.radToDeg(2 * Math.atan(at.tanY)), aspect: c.cam.ref.width / c.cam.ref.height, focalMm: at.focalMm, shift: at.shift };
}

interface Props {
  plan: DisplayPlan;
  hidden: Set<string>; // 关掉的种类（view/kinds3d.ts KINDS）
  transformMode: DragMode;
  hint?: string | null; // 节点手柄的用法：显示在左侧底栏，那里不会有东西与它重叠
  // 边算边看（view/partial.ts）：该节点仍在计算时已写出的帧。可作为点云查看的数据（深度图、位置图）
  // 每写出一帧即可查看一帧，无需等待整段计算完成。
  // 与二维舞台使用同一份轮询结果（editor/Viewer.tsx 中的同一处），不另行获取
  partial?: { info: PartialResult; job: string; node: string; port: string } | null;
}

const NOTHING: ReadonlySet<string> = new Set();

/** 应用模式舞台左下角的出处说明两行的高度（含边距）：坐标轴往上让这么多 */
const NOTICE_LIFT = 48;

export function Stage3D({ plan, hidden, transformMode, hint, partial }: Props) {
  const dpr = useDevicePixelRatio(); // 与二维舞台同一个：拖到 DPR 不同的屏上画布跟着换比例
  const frame = useViewer((s) => s.frame);
  // 节点声明的「骨架姿势」手柄：主视图默认不画，只画参数面板上「在视图里改」打开的那一个（state/handleView.ts editing；
  // 数据在状态回复 handle_data 里）
  const editing = useHandleView((s) => s.editing);
  // 双骨架编辑（手柄 rig_pair，view/rigPair.tsx）：同样只在参数面板上「在视图里编辑」打开、且它的节点是显示节点时
  const rigDef = plan.handles.find((h) => role3d(h) === "pair");
  const rigIndex = rigDef ? plan.def?.handles.indexOf(rigDef) ?? -1 : -1;
  const rigEditing = !!rigDef && rigIndex >= 0 && editing?.node === plan.node.id && editing.handle === rigIndex;
  const rigOf = `${plan.node.id}/${rigIndex}`;
  useEffect(() => { if (rigEditing) useRigPairView.getState().at(rigOf); }, [rigEditing, rigOf]);
  const poseHandles = useMemo((): PoseHandle[] => plan.handles.flatMap((h) => (role3d(h) === "pose" ? [{ index: plan.def?.handles.indexOf(h) ?? -1, def: h, operable: !h.readonly }] : []))
    .filter((h) => h.index >= 0 && editing?.node === plan.node.id && editing.handle === h.index),
  [plan.handles, plan.def, plan.node.id, editing]);
  // 编辑骨架姿势手柄时，场景里的骨架（节点结果的 element）与手柄画的同一副重叠（透传节点的输出与输入完全重合）：
  // 隐去场景骨架，只画手柄那一副，网格照常。双骨架编辑把两副拉开，蒙皮角色由它跟着各自的骨架画：场景里的角色也隐去
  const sceneHidden = useMemo(() => {
    if (!editing || editing.node !== plan.node.id) return hidden;
    const s = new Set(hidden);
    s.add("skeleton");
    if (rigEditing) s.add("character");
    return s;
  }, [hidden, editing, plan.node.id, rigEditing]);
  const handleData = useResults((s) => s.reply?.handle_data);
  const poseData = handleData?.node === plan.node.id ? handleData.handles ?? {} : {};
  const rigData = rigEditing ? (poseData[String(rigIndex)] as RigPairData | undefined) : undefined;
  const nodeParams = useCookInputs((s) => s.nodes[plan.node.id]?.params);
  const reference = useHandleView((s) => s.reference);
  const o = useViewOptions((s) => s.o);
  const appMode = useAppMode((m) => m.mode === "app"); // 舞台左下角放着出处说明时坐标轴往上让
  const [stageEl, setStageEl] = useState<HTMLDivElement | null>(null); // （载入期间不存在；出现后开始观察）
  const [selected, setSelected] = useState<string | null>(null);
  const stage = useMemo(() => new StageState(), []);
  const elements = plan.elements.filter((it): it is ViewItem & { fp: string } => !!it.fp);
  const cameraFp = plan.camera;
  const [loaded, sceneErrors] = useScenes([...elements.map((e) => e.fp), ...(cameraFp ? [cameraFp] : [])]);
  // 相机来自上游（`plan.camera` 不是所显示内容之一）时，所显示的场景里若有同一台相机（层级中的路径相同），用显示内容里的那份：
  // 它是下游、变换之后的（「合成场景」→「3D 变换」的输出类型是宽类型 `scene`，view/plan.ts 只认 `scene.camera`，
  // 于是往上游找到的是变换之前那台）。上游那份只在显示内容里没有同路径相机时才用。
  const ownCamera = !!cameraFp && elements.some((e) => e.fp === cameraFp);
  const upPath = cameraFp && !ownCamera ? loaded.get(cameraFp)?.cameras[0]?.ref.path : undefined;
  // 深度图 / 位置图转点云同样用这台：服务器按该包里唯一的一台相机反投影（lab2shot/data/scene.py the_camera），
  // 因此只在那份场景里恰好一台相机时替换，有几台时仍用上游那台（替换了服务器会报「有几台相机」）。
  const sameCamera = upPath === undefined ? undefined : elements.find((e) => {
    const cams = loaded.get(e.fp)?.cameras;
    return cams?.length === 1 && cams[0].ref.path === upPath;
  });
  // 显示内容里有场景、相机来自上游，而两边还在载入时先不取点云：否则先按变换前的相机取一遍、载入后再换，画面会先错后对。
  // 只显示深度图、没有场景的节点（最常见）不等，照旧和相机一起取；载入出错时也不等（照旧用上游那台）
  const scenesPending = !!cameraFp && !ownCamera && elements.length > 0 && !sceneErrors.length
    && (!loaded.has(cameraFp) || elements.some((e) => !loaded.has(e.fp)));
  const pointsCamera = sameCamera?.fp ?? cameraFp;
  const maps = scenesPending ? [] : plan.pointMaps.filter((it): it is ViewItem & { fp: string } => !!it.fp);
  // 正在计算的端口本身是一张可作为点云查看的图时，先绘制已写出的帧
  const streaming = !scenesPending && partial && plan.pointMaps.some((it) => it.nodeId === partial.node && it.port === partial.port) ? partial : null;
  const partialScene = usePartialPoints(
    streaming ? { job: streaming.job, node: streaming.node, port: streaming.port, done: streaming.info.frames_done } : null,
    pointsCamera,
  );
  // 键带两个包的代次（sceneData.ts pointsKey）：深度图或相机按同一指纹重算后重新取
  const pointGens = useGens((s) => maps.map((m) => s.gens[m.fp] ?? "").join() + "|" + (pointsCamera ? s.gens[pointsCamera] ?? "" : ""));
  const pointKeys = useMemo(() => maps.map((m) => pointsKey(m.fp, pointsCamera ?? null)),
    [maps.map((m) => m.fp).join(), pointsCamera, pointGens]); // eslint-disable-line react-hooks/exhaustive-deps
  useShownScenes(pointKeys);
  const [points, pointErrors] = useLoaded(maps.map((m, i) => ({ key: pointKeys[i], of: m.fp })),
    (fp, signal, lane) => loadPoints(fp, pointsCamera ?? null, signal, lane));
  // 逐帧数据的块：先当前帧，播放时再取播放头前方的帧，数量以内存预算为限（platform/cache.ts SCENE_SHARE）；
  // 到达即重绘；并告知时间线哪些帧已在视图中
  const scenes = [...loaded.values(), ...points.values(), ...(partialScene ? [partialScene] : [])];
  const scenesKey = scenes.map((s) => s.key).join("|");
  useSceneVersions(scenes);
  const playDir = useViewer((s) => (s.playing ? s.playDir : 0)) as -1 | 0 | 1;
  const scrubbing = useViewer((s) => s.scrubbing);
  useEffect(() => {
    // 拖动时间线期间不移动「当前块」：拖过的帧大多只是经过，不值得为它们插队解码。整段下载不受影响，照常在后台
    // 进行（view/scene.ts fetchAll，总是整段、没有开关）。松开后（scrubbing 变为 false）此 effect 再次执行
    if (scrubbing) return;
    for (const s of scenes) s.setFrame(frame, playDir);
  }, [frame, playDir, scenesKey, scrubbing]); // eslint-disable-line react-hooks/exhaustive-deps
  // 不另行报告「实际绘制的是第几帧」：时间线上「已载入视图」的浅绿色即表示此信息。
  useEffect(() => clearLoads, []);
  useEffect(() => {
    if (o.uvChecker) for (const s of scenes) void s.loadUv();
  }, [o.uvChecker, scenesKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const chunkErrors = scenes.flatMap((s) => (s.error ? [s.error] : []));
  const chosen = useViewCamera((s) => s.look);
  const setLook = useViewCamera((s) => s.setLook);

  // H 框显全部，F 框显选中，Esc 清除选中：指针在舞台上且使用者没有在打字时生效（由按键登记处自己检查指针是否在该元素上，
  // platform/keys.ts `under`）。透过相机看时 Esc 同样有效，因为选中属于舞台，而非视角。
  const frameView = useViewCamera((s) => s.frame);
  useShortcut(
    {
      keys: ["h", "f", "escape"],
      run: (e) => {
        const k = e.key.toLowerCase();
        if (k === "escape") {
          if (!selected) return false; // 没有选中：Esc 交给舞台上方打开的东西
          setSelected(null);
          return;
        }
        if (k === "f" && useViewCamera.getState().look) return false; // 透过相机看时：F 让片门适应画布（与二维视图的行为相同）
        if (k === "h") frameView("all");
        else frameView(selected ? "selected" : "all");
      },
    },
    { over: () => stageEl },
  );

  // 显示内容中的每一台相机，按其在层级中的位置。
  // 同一台相机在菜单中只出现一次：按其在层级中的位置（`cam.ref.path`）去重，而不按所属数据。
  // 同一台相机可能来自两份数据：视图的相机（`plan.camera`）和所显示节点场景中的相机；
  // 若按 `${fp}|${path}` 去重，会出现两行名称完全相同的条目。保留第一份：
  // - `plan.camera` 就是所显示内容之一（节点自身输出相机）时它在最前；
  // - 它来自上游时放在最后：显示内容里同路径的那台是下游、变换之后的（「合成场景」→「3D 变换」），应当透过它看；
  //   上游那台只在显示内容里没有同路径相机时才出现在菜单里。
  const cameras: SceneCamera[] = [];
  const cameraOrder = ownCamera
    ? [cameraFp!, ...elements.map((e) => e.fp)]
    : [...elements.map((e) => e.fp), ...(cameraFp ? [cameraFp] : [])];
  for (const fp of cameraOrder) {
    for (const cam of loaded.get(fp)?.cameras ?? [])
      if (cam.ref.frames.length && !cameras.some((c) => c.cam.ref.path === cam.ref.path)) cameras.push({ key: `${fp}|${cam.ref.path}`, fp, cam, label: cam.ref.path || plan.items.find((it) => it.fp === fp)?.label || t("ui.view.camera") });
  }
  // 舞台透过视角菜单中选中的相机看（没有选中：自由视角）
  const looked = cameras.find((c) => c.key === chosen?.key);
  const lens = looked ? lensOf(looked, frame) : null;
  // 透过相机看时，片门就是二维视图中的一幅画面（state/view2d.ts，与画面同一套平移 / 缩放）：滚轮、中键拖动（二维舞台上
  // 还有 Alt + 左键拖动）、F 以及 适应 / 1:1 / % 在画布上移动和缩放片门及其背板。这是在相机投影之后施加的缩放与偏移，
  // 从不移动相机或改变镜头；1:1 时一个片门像素等于一个相机（背板）像素。二维舞台的视图即查看器的二维视图；三维舞台的
  // 则按透过的相机各存一份（与 Maya 的 Pan/Zoom 相同），离开该相机时丢弃
  const [gw, gh] = looked ? [looked.cam.ref.width, looked.cam.ref.height] : [0, 0];
  const nav = useView2DNav(looked ? stageEl : null, gw, gh, { slot: "look", key: looked?.key ?? "", altLeft: false });
  const gate = lens && nav.at ? { x: nav.at.x, y: nav.at.y, w: gw * nav.at.s, h: gh * nav.at.s } : null;
  const lookedKey = looked?.key ?? null;
  // 背板使用与二维舞台相同的取帧路径（transfer/frames.ts useFrame → 窗口 → 预算 + 取消）。
  // 若将帧地址直接交给 ImagePlane 自行加载，它将不进入取帧账本：播放时询问「下一帧是否已到达」总是得到肯定答复，
  // 按帧率推进而背板跟不上，且没有窗口、不会取消、不受预算约束，开始播放即请求整段。
  // `serverFrames` 会先查询用户本机是否有该数据的原件（`transfer/sources.ts` → `transfer/originals.ts`）：
  // 有则从用户磁盘上的原件绘制，无传输、全精度；没有才请求服务器对应档位的视图代理。
  // 下方的 memo 必须将原件查找结果计入依赖（`plan.originals`，`view/useOriginals.ts`）：查找原件是异步的，
  // 画面先用服务器数据绘制，原件稍后才登记；若不随之重算，源将一直是服务器数据，既不报错也难以察觉。
  // 背板属于相机自身的属性，而非节点图中的位置：透过哪台相机查看，背板即为该相机记录的画面
  // （`CameraRef.plate`，由解算节点在 lab2shot/nodes/kit/cameras.py solved_camera 中记录，或为「创建相机」所接的画面），与相机来源无关；
  // 未记录背板的相机（导入的 USD 相机）没有背板。
  // 不使用 `plan.plate`（「上游最近的画面」）：它是二维舞台的底图，与「该相机由哪段素材解算得到」
  // 是两个概念，只在最简单的节点图中恰好一致。
  const plateFp = looked?.cam.ref.plate || null;
  const plateManifest = useManifest(plateFp);
  const plateGen = useGens((s) => (plateFp ? s.gens[plateFp] ?? "" : ""));  // 重算后重建帧源（transfer/gens.ts）
  // 相机带畸变时背板不是原图：此处的相机为针孔模型（camera3d.tsx 按焦距和片门投影），针孔看到的是去畸变后的画面，
  // 因此背板由服务器按该相机自带的镜头去畸变后再发送（`transfer/frameKey.ts pictureUrl` 的 `through`，
  // `lab2shot/view/proxy.py through_picture_file`）；不带畸变时字节完全不变。否则
  // 鱼眼等强畸变相机的点云将与底图无法对齐
  const distortion = looked?.cam.ref.distortion || "";
  const throughFp = plateFp && looked && distortion ? looked.fp : null;
  // 相机包的代次也进地址（cg=）和缓存键：相机按同一指纹重算、镜头变了，去畸变的背板要重新取（transfer/gens.ts）
  const throughGen = useGens((s) => (throughFp ? s.gens[throughFp] ?? "" : ""));
  const through = throughFp && looked ? { fp: throughFp, at: looked.cam.ref.path, gen: throughGen } : null;
  const plateSource = useMemo(
    () => (plateFp && plateManifest ? serverFrames({ fp: plateFp, type: "", through }, plateManifest) : null),
    [plateFp, plateManifest, plan.sourceKey, through?.fp, through?.at, through?.gen, plateGen],  // 来源改变时重算（view/origin.ts；其中包含原件查找结果）
  );
  const plate = useFrame(plateSource, frame, playDir || 1, scrubbing); // 拖时间线期间不逐帧拉背板，松开再取停下处的
  // 背板的帧同样计入「该帧是否在视图中」：三维数据块很小、很快全部到达，若只看数据块，播放将永远不会等待背板
  const plateLoaded = useLoadedFrames(plateSource?.id ?? null);
  const plateReady = useDecodedFrames(plateSource ? [plateSource] : []); // 背板「可立即画」与二维同一定义：已解码
  const loadKey = scenes.map((s) => `${s.key}:${s.version}`).join("|");
  useEffect(() => {
    // 每一份按自己的帧（transfer/readiness.ts readyAcross）：背板常比算出的场景长，场景没有的帧只看背板，不因
    // 场景「没有」而永远不可画
    const cells = scenes.flatMap((s) => { const got = s.loaded(); return got === null ? [] : [{ has: s.frames, ready: got }]; });
    const each = [...cells];
    const ready = [...cells];
    if (plateSource && plateLoaded) each.push({ has: plateSource.frames, ready: plateLoaded });  // 背板同样计入（见上方 plateSource 的注释）
    if (plateSource) ready.push({ has: plateSource.frames, ready: plateReady ?? [] });
    // 静止的场景（一份点云、一条相机轨迹整段一次到齐）整条均可实时播放：色带全绿（或全土黄），而非无颜色；
    // 色带表示能否播放，静止内容的每一帧均可播放
    const all = useViewer.getState().frames;
    if (!each.length && scenes.length) each.push({ has: all, ready: all });
    if (!ready.length && scenes.length) ready.push({ has: all, ready: all });
    // 三维绘制的是否为上一次的结果（元素或底图中有一份已过期，state/stale.ts）：时间线色带显示为土黄色
    // 尚未到达的数据块（等待到达，不跳过）：任一份数据的块未到达，该帧即视为未到达
    const waiting = scenes.map((s) => s.pending()).filter((l): l is number[] => l !== null);
    // 整段缓存进度（视图通知区「已缓存 N / M 帧」）：几份逐帧数据时取最慢的那份
    const progress = scenes.map((s) => s.cached()).filter((c): c is [number, number] => c !== null);
    // 「可立即画」（transfer/readiness.ts）：块已解开或在本机，背板已解码；「无须再下载」：背板的字节到了即可
    reportLoads({ loaded: readyAcross(each), ready: readyAcross(ready), waiting: waiting.length ? [...new Set(waiting.flat())] : null, stale: plan.stale,
                  cached: progress.length ? progress.reduce((a, c) => (c[0] / Math.max(c[1], 1) < a[0] / Math.max(a[1], 1) ? c : a)) : null });
  }, [loadKey, plateLoaded, plateReady, scenes.length, plan.stale]); // eslint-disable-line react-hooks/exhaustive-deps

  const lastLooked = useRef(lookedKey);
  useEffect(() => {
    if (lastLooked.current && lastLooked.current !== lookedKey) useView2D.getState().dropView("look"); // 离开了这台相机
    lastLooked.current = lookedKey;
  }, [lookedKey]);
  // 正在透过的相机不属于当前显示的节点：就地退回透视，并给出通知
  const say = useViewerNote((s) => s.say);
  const allLoaded = elements.every((e) => loaded.has(e.fp)) && (!cameraFp || loaded.has(cameraFp));
  const cameraKeys = cameras.map((c) => c.key).join("\n");
  // 正在透过的相机是在哪个显示节点上选的：下方的按路径回退只在同一个节点里做。
  // 只在所选相机（`chosen`）本身变化时记下当时的显示节点（使用者从视角菜单选、或下方回退改选）；
  // 切换显示节点不改变 `chosen`，因此不会把记录改成新节点。依赖里不能放 `plan.node.id`：
  // 否则切换节点的那次提交里此处先把记录改成新节点，回退判定恒为真，跨节点也会按路径回退
  // 舞台刚挂载时（例如从 2D 切回 3D，其间可能换过显示节点）不知道那台相机是在哪个节点上选的：不记，回退不生效，照旧退回透视
  // （按挂载时的值判断而不用「是否首次执行」：开发模式的 StrictMode 会把挂载时的 effect 执行两次）
  const lookedOn = useRef<string | null>(null);
  const atMount = useRef<typeof chosen | undefined>(chosen); // undefined：所选相机已在挂载后变过
  useEffect(() => {
    if (atMount.current !== undefined && chosen === atMount.current) {
      lookedOn.current = null;
      return;
    }
    atMount.current = undefined;
    lookedOn.current = chosen ? plan.node.id : null;
  }, [chosen]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!chosen || !allLoaded || cameraKeys.split("\n").includes(chosen.key)) return;
    // 键随包变（LookAt.key）：显示的还是同一个节点、同一台相机换了一份数据（重算后指纹变了，例如透过相机看时调「3D 变换」的参数）
    // 时键会变，按层级路径（LookAt.path）回退匹配，继续透过这台相机看，不退回透视。换了显示节点的照旧：退回透视并提示
    const same = lookedOn.current === plan.node.id ? cameras.find((c) => c.cam.ref.path === chosen.path) : undefined;
    if (same) {
      setLook({ key: same.key, path: same.cam.ref.path });
      return;
    }
    setLook(null);
    say(msg("N-VIEW-NOSUCHCAMERA"), msg("I-VIEW-NOSUCHCAMERAWHY", { camera: chosen.path || chosen.key }));
  }, [chosen, allLoaded, cameraKeys]); // eslint-disable-line react-hooks/exhaustive-deps

  const first = elements.map((e) => loaded.get(e.fp)).find(Boolean);
  // 点中别的东西或空处：不选关节（点中「骨架姿势」的关节时，那副骨架随后自己选上它：stageState.ts Pickable.choose）；
  // 双骨架编辑点在空处才清掉先点的骨点（点在骨点、连线上由它们自己处理：第二下就是配对）
  const onPick = useCallback((key: string | null, kept: boolean) => {
    usePoseSelection.getState().set(VIEWER_SLOT, null);
    if (!kept) useRigPairView.getState().clearPick();
    setSelected(key);
  }, []);
  const onLeave = useCallback(() => setLook(null), [setLook]);
  const errors = [...sceneErrors, ...pointErrors, ...chunkErrors];
  // 画面上需要显示的文字全部进入统一通知区（viewTools.ts useStageNotes）
  const note = useStageNotes((s) => s.put);
  const lang = useLang((s) => s.lang); // 通知里存的是文字：换语言时重写
  useEffect(() => {
    note("hint", hint ? { text: hint } : null);
  }, [hint, note]);
  useEffect(() => {
    note("error", errors.length ? { text: errors[0], tip: tipOf("error", errors.join("\n")) } : null);
  }, [errors.join("\n"), note]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    note("camera", lens && looked
      ? { text: `${looked.label.split("/").pop() || looked.label} · ${Number(lens.focalMm.toFixed(1))} mm${!plateFp ? t("ui.view.camera_no_plate") : through ? t("ui.view.camera_plate_undistorted") : ""}`,
          tip: tipOf("value", t("ui.view.camera_tip", { camera: looked.label, width: looked.cam.ref.width, height: looked.cam.ref.height })
            + (through ? t("ui.view.camera_tip_undistorted", { distortion }) : "")) }
      : null);
  }, [looked?.key, lens?.focalMm, plateFp, through?.fp, note, lang]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { note("hint", null); note("error", null); note("camera", null); }, [note]);
  // 视角控件绘制在上方工具栏中，其所需的两项数据由此处提供
  const setCameras = useViewCamera((s) => s.setCameras);
  const setSelectedName = useViewCamera((s) => s.setSelected);
  useEffect(() => {
    setCameras(cameras.map((c) => ({ key: c.key, path: c.cam.ref.path, label: c.label, width: c.cam.ref.width, height: c.cam.ref.height })));
  }, [cameras.map((c) => c.key).join("\n"), setCameras, lang]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setSelectedName(selected ? stage.pickables.get(selected)?.label ?? null : null);
  }, [selected, stage, setSelectedName]);
  // 上方所有 hook 必须位于此 return 之前
  // （React 的 hook 不得在提前 return 之后调用，否则画面内容变化时 hook 顺序会错乱）
  // 双骨架编辑不等节点自己的结果：两副骨架来自它的输入（手柄数据），没算过也能编辑
  if (!elements.length && !maps.length && !partialScene && !rigEditing) return <div className="empty">{t("ui.view.no_3d_result")}</div>;


  const waiting = elements.filter((e) => !loaded.get(e.fp)).length > sceneErrors.length; // 有些仍在生成中
  if (elements.length && !first && !waiting && errors.length) return <div className="empty">{errors[0]}</div>;
  if (elements.length && !first) return <Preparing />;
  const transform = plan.handles.find((h) => role3d(h) === "place");
  const frameKey = `${plan.node.id}|${elements.map((e) => e.fp).join()}|${maps.map((m) => m.fp).join()}`;
  const lookKey = looked?.key ?? null;

  const sceneOf = (e: ViewItem & { fp: string }) =>
    loaded.get(e.fp) && <SceneElement key={e.key} item={e} d={loaded.get(e.fp)!} frame={frame} hidden={sceneHidden} through={!!lens} look={lookKey} o={o} />;
  // 手柄的输入（view/plan.ts：节点还没有当前结果时随变换手柄一起显示）与节点自身的产出：变换手柄对二者的摆放方式不同
  // （TransformHandle）
  const input = <>{elements.filter((e) => e.context).map(sceneOf)}</>;
  const own = (
    <>
      {elements.filter((e) => !e.context).map(sceneOf)}
      {!hidden.has("points") && partialScene?.clouds.map((c) => (
        <Cloud key={c.key} src={c} frame={frame} o={o} pickKey={`${streaming!.node}.${streaming!.port}/points/${c.name}`} version={partialScene.version} />
      ))}
      {!hidden.has("points") &&
        maps.map((m, i) => {
          const d = points.get(pointKeys[i]);
          return d?.clouds.map((c) => <Cloud key={c.key} src={c} frame={frame} o={o} pickKey={`${m.key}/points/${c.name}`} version={d.version} />);
        })}
    </>
  );

  return (
    <div
      ref={setStageEl}
      className="stage3d"
      style={lens && nav.cursor ? { cursor: nav.cursor } : undefined}
      onMouseLeave={lens ? nav.onMouseLeave : undefined}
      onMouseMove={lens ? nav.onMouseMove : undefined}
      onMouseDown={lens ? nav.onMouseDown : undefined}
      onContextMenu={lens ? nav.onContextMenu : undefined}
    >
      {/* `preserveDrawingBuffer`：若不开启此项，页面截图中的三维部分将始终为空。
          页面截图使用 `ui/snapshot.ts pictureOf` → `canvas.toDataURL()`，而 WebGL 画布
          在合成后会清除绘制缓冲，除非开启此项。二维部分（view/look.ts）同样开启。
          代价是每帧多保留一份缓冲；frameloop="demand" 在无变化时不绘制，该代价可以接受。 */}
      <Canvas frameloop="demand" flat dpr={dpr}
        gl={{ antialias: false, powerPreference: "high-performance", preserveDrawingBuffer: true }}>
        <StageContext.Provider value={stage}>
          <KeepContext />
          <Redraw />
          <Pipeline o={o} />
          <Background o={o} />
          {/* 背板走图片路径（serverFrames），因此此处得到的必然是一张图；
              按通道取数的路径只有二维舞台使用（transfer/plane.ts） */}
          {lens && plate.image && !isPlane(plate.image) && <ImagePlane image={plate.image} pose={lens.pose} fovV={lens.fovV} aspect={lens.aspect} shift={lens.shift} exact />}
          <Lights o={o} />
          <GroundGrid o={o} through={!!lens} />
          {transform ? <TransformHandle mode={transformMode} input={input} own={own} /> : <>{input}{own}</>}
          <PoseLayers handles={poseHandles} data={poseData} values={nodeParams} mode={transformMode} slot={VIEWER_SLOT} o={o}
            write={(param, rows) => setParams(plan.node.id, { [param]: rows })} />
          {poseHandles.filter((h) => h.operable && poseDataOf(poseData, h.index)).map((h) => (
            <SkinnedPose key={`skin-${h.index}`} data={poseDataOf(poseData, h.index)!} value={nodeParams?.[h.def.params.pose]} o={o} />
          ))}
          {rigDef && rigData && <RigPairLayer node={plan.node.id} def={rigDef} data={rigData} params={nodeParams} mode={transformMode} o={o}
            write={(patch) => setParams(plan.node.id, patch)} />}
          {reference && reference.node !== plan.node.id && <ReferenceLayer reference={reference} frame={frame} o={o} />}
          <Picked stage={stage} keyOf={selected} width={o.lineWidth} />
          <ViewCamera o={o} frameKey={frameKey} selected={selected} lens={lens} gate={gate} onLeave={onLeave} />
          <Picker onPick={onPick} />
          {/* 角落中可点击的坐标轴：使用 `usable` 而非 `shown`。控件置灰由显示选项面板负责，
              舞台上的坐标轴是三维视图中的可点击对象，跟随相机时无法点击，因此不绘制（由 view/available.ts 统一计算） */}
          {usable(viewAvailable({ stage: "3d", shows: NOTHING, options: o, mode: "plate", channels: 0, single: false, ready: false }), "axesGizmo") && <Axes size={o.axesSize} lift={appMode ? NOTICE_LIFT : 0} />}
        </StageContext.Provider>
      </Canvas>
      {lens && gate && looked && <div className="gate" style={{ left: gate.x, top: gate.y, width: gate.w, height: gate.h }} />}
      {poseHandles.filter((h) => h.operable && h.def.params.pose && poseDataOf(poseData, h.index)).map((h) => (
        <SkeletonPosePanel key={h.index} data={poseDataOf(poseData, h.index)!} index={h.index} slot={VIEWER_SLOT} label={plan.def?.params.find((q) => q.name === h.def.params.pose)?.label ?? h.def.params.pose}
          rows={poseRowsOf(nodeParams?.[h.def.params.pose])} write={(rows) => setParams(plan.node.id, { [h.def.params.pose]: rows })} />
      ))}
      {rigDef && rigEditing && (rigData
        ? <RigPairPanels node={plan.node.id} def={rigDef} data={rigData} params={nodeParams} write={(patch) => setParams(plan.node.id, patch)}
            onInsets={(insets) => void (stage.insets = insets)} />
        : <RigPairWaiting />)}
      {/* 手柄提示、显示错误不画在舞台上：进入 state 的通知区（viewTools.ts useStageNotes），由 editor/ViewerFrame.tsx
          统一绘制在左上角；视角与框显属于工具，位于上方工具栏。舞台上的字只有属于内容的：骨点名（elements3d.tsx
          JointLabels）和选中关节的数值面板（上面的 SkeletonPosePanel）。 */}
    </div>
  );
}
