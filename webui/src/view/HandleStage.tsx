/** 弹窗里用的三维舞台（「对应关系」弹窗左右各一个）：与主视图同一套舞台部件（相机、网格、灯、背景、拾取、canvasLife），
 * 内容只有舞台统一提供的几层（view/stageLayers.tsx）——某个节点的一个「骨架姿势」手柄（可操作或只看）、参考显示、
 * 按修正后的姿势蒙皮。弹窗自己不写任何三维代码；写回走调用方给的 `write`（弹窗草稿），选中、相机各用自己的槽，
 * 不碰主视图的。数据来自状态回复的 handle_data：调用方开着弹窗时把这个节点记为要手柄数据的那个（state/handleView.ts
 * dialog），不改显示节点。 */

import { useCallback, useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { rowsOf as poseRowsOf, type PoseRow } from "../model/skeletonPose";
import { useResults } from "../state/results";
import { useCookInputs } from "../state/cookInputs";
import { useViewOptions } from "../state/viewer";
import { getNodeDefs } from "../state/catalog";
import type { Reference } from "../state/handleView";
import { useDevicePixelRatio } from "../platform/size";
import { KeepContext, Redraw } from "./canvasLife";
import { Background, GroundGrid, Lights, Pipeline } from "./render3d";
import { Picker, ViewCamera } from "./camera3d";
import type { DragMode } from "./dragGizmo";
import { useViewCamera } from "../state/viewer";
import { StageContext, StageState } from "./stageState";
import { PoseLayers, ReferenceLayer, SkinnedPose, type PoseHandle } from "./stageLayers";
import { SkeletonPosePanel, usePoseSelection } from "./skeletonPose";
import { Button, Segmented } from "../ui/Button";

const MODES: { value: DragMode; label: string; tip: string }[] = [
  { value: "rotate", label: "旋转", tip: "选中关节的手柄转它（子关节跟着走）" },
  { value: "translate", label: "位移", tip: "选中关节的手柄移它" },
  { value: "scale", label: "缩放", tip: "选中关节的手柄按轴缩放它" },
];

export function HandleStage({ node, handle, reference, skin, values, write, height, slot }: {
  node: string;
  handle: number; // 这个舞台可操作的那个手柄（NodeDef.handles 的下标）
  reference: Reference | null; // 叠在上面、只看的另一样（例如另一侧的骨架：{node, handle}）
  skin: boolean; // 按修正后的姿势蒙皮画角色
  values: Record<string, unknown>; // 修正从哪读（弹窗草稿：参数名 → 值）
  write: (param: string, rows: PoseRow[]) => void; // 写到哪（弹窗草稿）
  height: number;
  slot: string; // 相机与选中的槽：每个舞台一个
}) {
  const dpr = useDevicePixelRatio();
  const o = useViewOptions((s) => s.o);
  const [mode, setDragMode] = useState<DragMode>("rotate");
  const stage = useMemo(() => new StageState(), []);
  const typeId = useCookInputs((s) => s.nodes[node]?.typeId);
  const def = typeId ? getNodeDefs()[typeId]?.handles[handle] : undefined;
  const hd = useResults((s) => s.reply?.handle_data);
  const data = hd?.node === node ? hd.handles ?? {} : {};
  const mine = data[String(handle)];
  const handles = useMemo((): PoseHandle[] => (def ? [{ index: handle, def, operable: !def.readonly }] : []), [def, handle]);
  // 舞台上只有骨架：点空处不选关节，点中关节由那副骨架自己选上（stageState.ts Pickable.choose）
  const onPick = useCallback(() => usePoseSelection.getState().set(slot, null), [slot]);
  const param = def?.params.pose;
  if (!def) return <div className="handle-stage" style={{ height }}><div className="stage3d-empty">这个节点没有这一侧的骨架手柄</div></div>;
  if (!mine) return <div className="handle-stage" style={{ height }}><div className="stage3d-empty">骨架还没读到：上游算过才有（正在问服务器）</div></div>;
  const editable = !def.readonly && !!param;
  return (
    <div className="handle-stage-wrap">
      <div className="handle-stage" style={{ height }}>
        <Canvas frameloop="demand" flat dpr={dpr} gl={{ antialias: false, powerPreference: "high-performance" }}>
          <StageContext.Provider value={stage}>
            <KeepContext />
            <Redraw />
            <Pipeline o={o} />
            <Background o={o} />
            <Lights o={o} />
            <GroundGrid o={o} through={false} />
            <PoseLayers handles={handles} data={data} values={values} mode={mode} slot={slot} o={o}
              write={write} />
            {reference && <ReferenceLayer reference={reference} frame={0} o={o} slot={slot} values={() => values} />}
            {skin && <SkinnedPose data={mine} value={param ? values[param] : undefined} o={o} />}
            <ViewCamera o={o} frameKey={`${node}|${handle}`} selected={null} lens={null} gate={null} onLeave={() => undefined} slot={slot} ephemeral />
            <Picker onPick={onPick} />
          </StageContext.Provider>
        </Canvas>
      </div>
      <div className="handle-stage-tools">
        {editable && <Segmented label="手柄" value={mode} onChange={setDragMode} options={MODES} />}
        <span style={{ flex: 1 }} />
        <Button size="sm" tip="框显这副骨架（相机只属于这块视图，不动主视图的）" onClick={() => useViewCamera.getState().frame("all", slot)}>框显</Button>
      </div>
      {editable && (
        <SkeletonPosePanel data={mine} index={handle} rows={poseRowsOf(values[param!])} write={(rows) => write(param!, rows)} slot={slot}
          label={mine.pose} />
      )}
    </div>
  );
}
