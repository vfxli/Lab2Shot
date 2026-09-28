import { usable } from "../api/applies";
import { viewAvailable } from "./available";
import { msg } from "../messages/message";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { KeepContext, Redraw } from "./canvasLife";
import { TransformControls } from "@react-three/drei";
import * as THREE from "three";
import { setParams } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { Picker, ViewCamera, type Lens } from "./camera3d";
import { CameraPath, cameraAt, Character, ModelMesh, ShotCamera } from "./elements3d";
import { boxSegments, FatLines } from "./lines3d";
import type { ViewOptions } from "../model/viewOptions";
import type { DisplayPlan, ViewItem } from "./plan";
import { Cloud } from "./points3d";
import { useManifest } from "./Stage2D";
import { useGens } from "../transfer/gens";
import { serverFrames, useFrame, useLoadedFrames } from "../transfer/frames";
import { isPlane } from "../transfer/plane";
import { Curves } from "./curves3d";
import { Axes, Background, GroundGrid, ImagePlane, Lights, Pipeline } from "./render3d";
import { loadPoints, loadScene, useLoaded, usePartialPoints, useSceneVersions, type CameraData, type Scene } from "./sceneData";
import type { Partial as PartialResult } from "./partial";
import type { Kind } from "./kinds3d";
import { StageContext, StageState } from "./stageState";
import { useStageNotes, useViewCamera, useViewer, useViewerNote, useViewLoads, useViewOptions, useView2D, useView2DNav } from "../state/viewer";
import { useShortcut } from "../platform/keys";
import { useResults } from "../state/results";
import { getNodeDefs } from "../state/catalog";
import { placeMatrix, threeOrder } from "../model/places";
import { cookedWith, useLastGood } from "../state/stale";
import { Preparing } from "./StageHud";

/** The 3D stage: every 3D result of the node in one scene (cameras with their paths, models and skinned characters,
 * skeletons, point clouds, 3D curves), depth / position maps as a point preview, and the
 * transform handle. Any camera in the scene can be looked through (the view menu, as in Houdini): its view on every
 * frame, its resolution gate with the outside dimmed, and its plate behind it as an image plane when it has one. Drawing
 * follows the display options (model/viewOptions.ts). Rendering is done by the browser (WebGL); the server only
 * prepares the data.
 * 二维舞台不透过三维结果的相机查看：只输出三维结果的节点切换到 2D 时播放的是上游序列原样。 */


function SceneElement({ item, d, frame, hidden, through, look, o }: { item: ViewItem & { fp: string }; d: Scene; frame: number; hidden: Set<string>; through: boolean; look: string | null; o: ViewOptions }) {
  const shown = (k: Kind) => !hidden.has(k);
  return (
    <>
      {shown("model") && d.models.map((m, i) => <ModelMesh key={i} m={m} frame={frame} o={o} pickKey={`${item.key}/model/${i}`} through={through} version={d.version} />)}
      {d.characters.map((c, i) => (
        <Character key={i} c={c} frame={frame} o={o} pickKey={`${item.key}/character/${i}`} through={through} meshes={shown("character")} bones={shown("skeleton")} person={c.ref.person} />
      ))}
      {shown("points") && d.clouds.map((c) => <Cloud key={c.key} src={c} frame={frame} o={o} pickKey={`${item.key}/points/${c.name}`} version={d.version} />)}
      {shown("curves") && d.curves.map((c) => <Curves key={c.key} src={c} frame={frame} o={o} pickKey={`${item.key}/curves/${c.name}`} version={d.version} />)}
      {shown("camera") &&
        d.cameras.map((cam) =>
          `${d.key}|${cam.ref.path}` === look ? null : ( // not the camera looked through: the view is inside it
            <group key={cam.ref.path}>
              <ShotCamera cam={cam} frame={frame} o={o} pickKey={`${item.key}/camera${cam.ref.path}`} />
              {cam.ref.frames.length > 1 && <CameraPath cam={cam} o={o} />}
            </group>
          ),
        )}
    </>
  );
}

/** The transform handle: a gizmo placing what the node gives, according to the placement the node declares (its
 * parameters, their order and rotation order, model/places.ts). It only displays, and computes nothing: what is shown is
 * put where the current parameters (or, while dragging, the gizmo) place it, with the matrix the cook applies, and a
 * release only stores the parameters (one undo step; nothing is cooked, 计算 is a right click). What it shows is chosen
 * by view/plan.ts underHandles:
 * - `input`: the handle's input (「3D 变换」's upstream scene, before the node has a current result), which the node has
 *   not placed yet: it is put at the placement itself;
 * - `own`: the node's own result, which was placed by the parameters it was cooked with (state/stale.ts cookedWith): it
 *   is moved by the current placement times the inverse of that one, so a stale result, or the old one until the next
 *   status reply, sits where the current parameters put it rather than jumping back. */
function TransformHandle({ input, own, mode }: { input: React.ReactNode; own: React.ReactNode; mode: "translate" | "rotate" | "scale" }) {
  const displayId = useLook((s) => s.displayId);
  const graphId = useCookInputs((s) => s.graphId);
  const node = useCookInputs((s) => (displayId ? s.nodes[displayId] : undefined));
  const statusPlaces = useResults((s) => (displayId ? s.reply?.nodes[displayId]?.places : undefined));
  const places = statusPlaces ?? (node ? getNodeDefs()[node.typeId]?.places : null) ?? null;
  // re-read when a result is recorded: the parameters it was cooked with come with it
  useLastGood((s) => s.byGraph[graphId]?.[displayId ?? ""]);
  // the pivot is kept as state: a ref read during render is null on the first pass, so the controls appeared only
  // once something else re-rendered the stage
  const [pivotObj, setPivotObj] = useState<THREE.Group | null>(null);
  const pivot = useRef<THREE.Group | null>(null);
  const bind = (g: THREE.Group | null) => {
    pivot.current = g;
    setPivotObj((was) => (was === g ? was : g));
  };
  const [live, setLive] = useState<THREE.Matrix4 | null>(null);
  const p = node?.params ?? {};
  const placedBy = displayId ? cookedWith(graphId, displayId) : undefined;
  const key = (q: Record<string, unknown>) => (places ? JSON.stringify([q[places.translate], q[places.rotate], places.scale ? q[places.scale] : 1]) : "");
  const current = useMemo(() => (places ? new THREE.Matrix4().fromArray(placeMatrix(places, p)) : null), [places, key(p)]); // eslint-disable-line react-hooks/exhaustive-deps
  const before = useMemo(() => (places && placedBy ? new THREE.Matrix4().fromArray(placeMatrix(places, placedBy)).invert() : null), [places, key(placedBy ?? {})]); // eslint-disable-line react-hooks/exhaustive-deps
  // the gizmo sits at the current parameters, set on the object when they change and never as props: props are applied
  // again on every render, and a render during a drag (each move sets `live`) would put the gizmo back where the drag
  // began, so the release would store the old place
  useLayoutEffect(() => {
    if (!pivotObj || !current) return;
    current.decompose(pivotObj.position, pivotObj.quaternion, pivotObj.scale);
    pivotObj.updateMatrix();
  }, [pivotObj, current]);
  if (!places || !current) return <>{input}{own}</>;
  const order = threeOrder(places.rotation);
  const scaled = new THREE.Vector3();
  current.decompose(new THREE.Vector3(), new THREE.Quaternion(), scaled);
  const at = live ?? current; // where it is shown: the gizmo while dragging, else the current parameters
  return (
    <>
      <group matrix={at} matrixAutoUpdate={false}>
        {input}
      </group>
      {/* view/plan.ts shows the node's own result only when `before` is known */}
      {before && (
        <group matrix={at.clone().multiply(before)} matrixAutoUpdate={false}>
          {own}
        </group>
      )}
      <group ref={bind} />
      {pivotObj && (
        <TransformControls
          object={pivotObj}
          mode={mode}
          size={0.8}
          // from the gizmo's own position, rotation and scale: its matrix is only brought up to date when the scene is drawn
          onObjectChange={() => { const g = pivot.current; if (g) setLive(new THREE.Matrix4().compose(g.position, g.quaternion, g.scale)); }}
          onMouseUp={() => {
            const g = pivot.current;
            if (!node || !g) return;
            const e = new THREE.Euler().setFromQuaternion(g.quaternion, order);
            const round = (v: number, k = 100) => Math.round(v * k) / k;
            // one undo step; the stage then shows it at these parameters (`current`), where the cook will put it
            setParams(displayId!, {
              [places.translate]: g.position.toArray().map((v) => round(v)),
              [places.rotate]: [e.x, e.y, e.z].map((v) => round(THREE.MathUtils.radToDeg(v))),
              ...(places.scale ? { [places.scale]: round(mode === "scale" ? (g.scale.x + g.scale.y + g.scale.z) / 3 : scaled.x, 1000) } : {}),
            });
            setLive(null);
          }}
        />
      )}
    </>
  );
}

/** The picked object's box, in the accent colour. */
function Picked({ stage, keyOf, width }: { stage: StageState; keyOf: string | null; width: number }) {
  const box = keyOf ? stage.pickables.get(keyOf)?.bounds() : null;
  if (!box || box.isEmpty()) return null;
  return <FatLines segments={boxSegments(box)} color="#0a84ff" width={Math.max(1, width)} overlay opacity={0.9} />;
}

/** A camera of the scene that can be looked through. */
interface SceneCamera {
  key: string; // "packet|camera path"
  cam: CameraData;
  label: string; // its position in the hierarchy
}

/** A camera on this frame: its pose, the vertical field of view of its resolution gate, its focal length. */
function lensOf(c: SceneCamera, frame: number): Lens & { focalMm: number } {
  const at = cameraAt(c.cam, frame);
  return { pose: at.matrix, fovV: THREE.MathUtils.radToDeg(2 * Math.atan(at.tanY)), aspect: c.cam.ref.width / c.cam.ref.height, focalMm: at.focalMm };
}

interface Props {
  plan: DisplayPlan;
  hidden: Set<string>; // kinds switched off (view/kinds3d.ts KINDS)
  transformMode: "translate" | "rotate" | "scale";
  hint?: string | null; // usage of the node's handle: shown in the footer on the left, so nothing there overlaps
  // 边算边看 (view/partial.ts)：该节点仍在计算时已写出的帧。可作为点云查看的数据（深度图、位置图）
  // 每写出一帧即可查看一帧，无需等待整段计算完成。
  // 与二维舞台使用同一份轮询结果（editor/Viewer.tsx 中的同一处），不另行获取
  partial?: { info: PartialResult; job: string; node: string; port: string } | null;
}

const NOTHING: ReadonlySet<string> = new Set();

export function Stage3D({ plan, hidden, transformMode, hint, partial }: Props) {
  const frame = useViewer((s) => s.frame);
  const o = useViewOptions((s) => s.o);
  const [stageEl, setStageEl] = useState<HTMLDivElement | null>(null); // (absent while loading; observed once present)
  const [selected, setSelected] = useState<string | null>(null);
  const stage = useMemo(() => new StageState(), []);
  const elements = plan.elements.filter((it): it is ViewItem & { fp: string } => !!it.fp);
  const cameraFp = plan.camera;
  const [loaded, sceneErrors] = useLoaded([...elements.map((e) => e.fp), ...(cameraFp ? [cameraFp] : [])], loadScene);
  const maps = plan.pointMaps.filter((it): it is ViewItem & { fp: string } => !!it.fp);
  // 正在计算的端口本身是一张可作为点云查看的图时，先绘制已写出的帧
  const streaming = partial && plan.pointMaps.some((it) => it.nodeId === partial.node && it.port === partial.port) ? partial : null;
  const partialScene = usePartialPoints(
    streaming ? { job: streaming.job, node: streaming.node, port: streaming.port, done: streaming.info.frames_done } : null,
    cameraFp,
  );
  const [points, pointErrors] = useLoaded(maps.map((m) => `${m.fp}|${cameraFp ?? ""}`), (key) => {
    const [fp, cam] = key.split("|");
    return loadPoints(fp, cam || null);
  });
  // chunks of per-frame data: the current frame first, then frames ahead of the playhead while playing, as many as the
  // memory budget allows (view/scene.ts memoryBytes); redrawn as they arrive; the timeline is informed
  // which frames are in the view
  const scenes = [...loaded.values(), ...points.values(), ...(partialScene ? [partialScene] : [])];
  const scenesKey = scenes.map((s) => s.key).join("|");
  useSceneVersions(scenes);
  const playDir = useViewer((s) => (s.playing ? s.playDir : 0)) as -1 | 0 | 1;
  const scrubbing = useViewer((s) => s.scrubbing);
  useEffect(() => {
    // 拖动时间线期间不拉取任何数据块：拖过的帧大多只是经过，
    // 逐帧拉取会浪费流量。松开后（scrubbing 变为 false）此 effect 再次执行，拉取停止处的帧
    if (scrubbing) return;
    for (const s of scenes) s.setFrame(frame, playDir);
  }, [frame, playDir, scenesKey, scrubbing]); // eslint-disable-line react-hooks/exhaustive-deps
  // 不另行报告「实际绘制的是第几帧」：时间线上「已载入视图」的浅绿色即表示此信息。
  useEffect(() => () => useViewLoads.setState({ loaded: null, pending: null, stale: false }), []);
  useEffect(() => {
    if (o.uvChecker) for (const s of scenes) void s.loadUv();
  }, [o.uvChecker, scenesKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const chunkErrors = scenes.flatMap((s) => (s.error ? [s.error] : []));
  const chosen = useViewCamera((s) => s.look);
  const setLook = useViewCamera((s) => s.setLook);

  // H frames everything, F frames the selection, Esc clears the selection: while the pointer is over the stage and the
  // user is not typing (the key registry itself tests the pointer against the element, platform/keys.ts `under`). Esc
  // also works while looking through a camera, since the selection belongs to the stage, not to the view.
  const frameView = useViewCamera((s) => s.frame);
  useShortcut(
    {
      keys: ["h", "f", "escape"],
      run: (e) => {
        const k = e.key.toLowerCase();
        if (k === "escape") {
          if (!selected) return false; // nothing selected: Esc belongs to whatever is open over the stage
          setSelected(null);
          return;
        }
        if (k === "f" && useViewCamera.getState().look) return false; // looking through a camera: F fits its gate (the 2D view's behaviour)
        if (k === "h") frameView("all");
        else frameView(selected ? "selected" : "all");
      },
    },
    { over: () => stageEl },
  );

  // every camera of the displayed content, by its position in the hierarchy; the node's own camera (or the upstream one) first.
  // 同一台相机在菜单中只出现一次：按其在层级中的位置（`cam.ref.path`）去重，而不按所属数据。
  // 同一台相机可能来自两份数据：视图自身的相机（`plan.camera`）和所显示节点场景中的相机；
  // 若按 `${fp}|${path}` 去重，会出现两行名称完全相同的条目。保留第一份，即视图自身的相机优先。
  const cameras: SceneCamera[] = [];
  for (const fp of [...(cameraFp ? [cameraFp] : []), ...elements.map((e) => e.fp)]) {
    for (const cam of loaded.get(fp)?.cameras ?? [])
      if (!cameras.some((c) => c.cam.ref.path === cam.ref.path)) cameras.push({ key: `${fp}|${cam.ref.path}`, cam, label: cam.ref.path || plan.items.find((it) => it.fp === fp)?.label || "相机" });
  }
  // the stage looks through the camera chosen in the view menu (none: the free view)
  const looked = cameras.find((c) => c.key === chosen);
  const lens = looked ? lensOf(looked, frame) : null;
  // looking through a camera, the gate is its picture in a 2D view (state/view2d.ts, the same pan/zoom as a picture's):
  // the wheel, a middle-drag (on the 2D stage also an Alt+left-drag), F and 适应 / 1:1 / % move and scale the gate and
  // its plate on the canvas. This is a scale and offset applied after the camera's projection, never a movement of the
  // camera or its lens; at 1:1 one gate pixel equals one camera (plate) pixel. The 2D stage's view is the viewer's 2D
  // view; the 3D stage's is kept per camera looked through (as Maya's Pan/Zoom) and discarded when the view leaves it
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
  // 因此背板由服务器按该相机自带的镜头去畸变后再发送（`api.packetFrameUrl` 的 `through`，
  // `lab2shot/view/proxy.py through_picture_file`）；不带畸变时字节完全不变。否则
  // 鱼眼等强畸变相机的点云将与底图无法对齐
  const distortion = looked?.cam.ref.distortion || "";
  const through = plateFp && looked && distortion ? { fp: looked.key.split("|")[0], at: looked.cam.ref.path } : null;
  const plateSource = useMemo(
    () => (plateFp && plateManifest ? serverFrames({ fp: plateFp, type: "", through }, plateManifest) : null),
    [plateFp, plateManifest, plan.sourceKey, through?.fp, through?.at, plateGen],  // 来源改变时重算（view/origin.ts；其中包含原件查找结果）
  );
  const plate = useFrame(plateSource, frame, playDir || 1);
  // 背板的帧同样计入「该帧是否在视图中」：三维数据块很小、很快全部到达，若只看数据块，播放将永远不会等待背板
  const plateLoaded = useLoadedFrames(plateSource?.id ?? null);
  const loadKey = scenes.map((s) => `${s.key}:${s.version}`).join("|");
  useEffect(() => {
    const each = scenes.map((s) => s.loaded()).filter((l): l is number[] => l !== null);
    if (plateLoaded) each.push(plateLoaded);  // 背板同样计入（见上方 plateSource 的注释）
    // 静止的场景（一份点云、一条相机轨迹整段一次到齐）整条均可实时播放：色带全绿（或全土黄），而非无颜色；
    // 色带表示能否播放，静止内容的每一帧均可播放
    if (!each.length && scenes.length) each.push(useViewer.getState().frames);
    // 三维绘制的是否为上一次的结果（元素或底图中有一份已过期，state/stale.ts）：时间线色带显示为土黄色
    // 尚未到达的数据块（等待到达，不跳过）：任一份数据的块未到达，该帧即视为未到达
    const waiting = scenes.map((s) => s.pending()).filter((l): l is number[] => l !== null);
    useViewLoads.setState({ loaded: each.length ? each.reduce((a, l) => a.filter((f) => l.includes(f))) : null,
                            pending: waiting.length ? [...new Set(waiting.flat())] : null, stale: plan.stale });
  }, [loadKey, plateLoaded, scenes.length, plan.stale]); // eslint-disable-line react-hooks/exhaustive-deps

  const lastLooked = useRef(lookedKey);
  useEffect(() => {
    if (lastLooked.current && lastLooked.current !== lookedKey) useView2D.getState().dropView("look"); // left the camera
    lastLooked.current = lookedKey;
  }, [lookedKey]);
  // a looked-through camera that the currently displayed node does not have: revert to 透视 in place, with a notice
  const say = useViewerNote((s) => s.say);
  const allLoaded = elements.every((e) => loaded.has(e.fp)) && (!cameraFp || loaded.has(cameraFp));
  const cameraKeys = cameras.map((c) => c.key).join("\n");
  useEffect(() => {
    if (!chosen || !allLoaded || cameraKeys.split("\n").includes(chosen)) return;
    setLook(null);
    say(msg("N-VIEW-NOSUCHCAMERA"), msg("I-VIEW-NOSUCHCAMERAWHY", { camera: chosen.split("|").slice(1).join("|") || chosen }));
  }, [chosen, allLoaded, cameraKeys]); // eslint-disable-line react-hooks/exhaustive-deps

  const first = elements.map((e) => loaded.get(e.fp)).find(Boolean);
  const onPick = useCallback((key: string | null) => setSelected(key), []);
  const onLeave = useCallback(() => setLook(null), [setLook]);
  const errors = [...sceneErrors, ...pointErrors, ...chunkErrors];
  // 画面上需要显示的文字全部进入统一通知区（viewTools.ts useStageNotes）
  const note = useStageNotes((s) => s.put);
  useEffect(() => {
    note("hint", hint ? { text: hint, tip: hint } : null);
  }, [hint, note]);
  useEffect(() => {
    note("error", errors.length ? { text: errors[0], tip: errors.join("\n") } : null);
  }, [errors.join("\n"), note]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    note("camera", lens && looked
      ? { text: `${looked.label.split("/").pop() || looked.label} · ${Number(lens.focalMm.toFixed(1))} mm${!plateFp ? " · 没有背板" : through ? " · 背板已去畸变" : ""}`,
          tip: `透过「${looked.label}」看：${looked.cam.ref.width} × ${looked.cam.ref.height}，框外变暗的部分不在画面里。转动视图就离开相机，回到透视（相机本身不动）`
            + (!plateFp ? "。这台相机没记背板：导入的相机没有"
              : through ? `。这台相机带畸变（${distortion}），背板已按它的镜头去畸变，和点云对得上；原图在二维视图里看` : "。背板是它解算时的那张画面") }
      : null);
  }, [looked?.key, lens?.focalMm, plateFp, through?.fp, note]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => { note("hint", null); note("error", null); note("camera", null); }, [note]);
  // 视角控件绘制在上方工具栏中，其所需的两项数据由此处提供
  const setCameras = useViewCamera((s) => s.setCameras);
  const setSelectedName = useViewCamera((s) => s.setSelected);
  useEffect(() => {
    setCameras(cameras.map((c) => ({ key: c.key, label: c.label, width: c.cam.ref.width, height: c.cam.ref.height })));
  }, [cameras.map((c) => c.key).join("\n"), setCameras]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    setSelectedName(selected ? stage.pickables.get(selected)?.label ?? null : null);
  }, [selected, stage, setSelectedName]);
  // 上方所有 hook 必须位于此 return 之前
  // （React 的 hook 不得在提前 return 之后调用，否则画面内容变化时 hook 顺序会错乱）
  if (!elements.length && !maps.length && !partialScene) return <div className="empty">没有三维结果</div>;


  const waiting = elements.filter((e) => !loaded.get(e.fp)).length > sceneErrors.length; // some are still being produced
  if (elements.length && !first && !waiting && errors.length) return <div className="empty">{errors[0]}</div>;
  if (elements.length && !first) return <Preparing />;
  const transform = plan.handles.find((h) => h.kind === "transform");
  const frameKey = `${plan.node.id}|${elements.map((e) => e.fp).join()}|${maps.map((m) => m.fp).join()}`;
  const lookKey = looked?.key ?? null;

  const sceneOf = (e: ViewItem & { fp: string }) =>
    loaded.get(e.fp) && <SceneElement key={e.key} item={e} d={loaded.get(e.fp)!} frame={frame} hidden={hidden} through={!!lens} look={lookKey} o={o} />;
  // the handle's input (view/plan.ts: shown with a transform handle before the node has a current result) and what the node
  // itself gives: a transform handle places the two differently (TransformHandle)
  const input = <>{elements.filter((e) => e.context).map(sceneOf)}</>;
  const own = (
    <>
      {elements.filter((e) => !e.context).map(sceneOf)}
      {!hidden.has("points") && partialScene?.clouds.map((c) => (
        <Cloud key={c.key} src={c} frame={frame} o={o} pickKey={`${streaming!.node}.${streaming!.port}/points/${c.name}`} version={partialScene.version} />
      ))}
      {!hidden.has("points") &&
        maps.map((m) => {
          const d = points.get(`${m.fp}|${cameraFp ?? ""}`);
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
    >
      {/* `preserveDrawingBuffer`：若不开启此项，页面截图中的三维部分将始终为空。
          页面截图使用 `ui/snapshot.ts pictureOf` → `canvas.toDataURL()`，而 WebGL 画布
          在合成后会清除绘制缓冲，除非开启此项。二维部分（view/look.ts）同样开启。
          代价是每帧多保留一份缓冲；frameloop="demand" 在无变化时不绘制，该代价可以接受。 */}
      <Canvas frameloop="demand" flat dpr={window.devicePixelRatio || 1}
        gl={{ antialias: false, powerPreference: "high-performance", preserveDrawingBuffer: true }}>
        <StageContext.Provider value={stage}>
          <KeepContext />
          <Redraw />
          <Pipeline o={o} />
          <Background o={o} />
          {/* 背板走图片路径（serverFrames），因此此处得到的必然是一张图；
              按通道取数的路径只有二维舞台使用（transfer/plane.ts） */}
          {lens && plate.image && !isPlane(plate.image) && <ImagePlane image={plate.image} pose={lens.pose} fovV={lens.fovV} aspect={lens.aspect} exact />}
          <Lights o={o} />
          <GroundGrid o={o} through={!!lens} />
          {transform ? <TransformHandle mode={transformMode} input={input} own={own} /> : <>{input}{own}</>}
          <Picked stage={stage} keyOf={selected} width={o.lineWidth} />
          <ViewCamera o={o} frameKey={frameKey} selected={selected} lens={lens} gate={gate} onLeave={onLeave} />
          <Picker onPick={onPick} />
          {/* 角落中可点击的坐标轴：使用 `usable` 而非 `shown`。控件置灰由显示选项面板负责，
              舞台上的坐标轴是三维视图中的可点击对象，跟随相机时无法点击，因此不绘制（由 view/available.ts 统一计算） */}
          {usable(viewAvailable({ stage: "3d", shows: NOTHING, options: o, mode: "plate", channels: 0, single: false, ready: false }), "axesGizmo") && <Axes size={o.axesSize} />}
        </StageContext.Provider>
      </Canvas>
      {lens && gate && looked && <div className="gate" style={{ left: gate.x, top: gate.y, width: gate.w, height: gate.h }} />}
      {/* 三维舞台不在画面上绘制任何文字：
          手柄提示、显示错误全部进入 state 的通知区（viewTools.ts useStageNotes），
          由 editor/ViewerFrame.tsx 统一绘制在左上角。视角与框显属于工具，位于上方工具栏。 */}
    </div>
  );
}
