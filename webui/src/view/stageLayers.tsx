/** 三维舞台上与显示内容无关、由舞台统一提供的几层：
 * - 手柄「骨架姿势」（PoseLayers）：节点声明了这种手柄，舞台就按状态回复的 handle_data 画那一侧的骨架并让人改；
 * - 参考显示（ReferenceLayer）：另一个节点的结果，半透明、另一种颜色叠画，不可操作；
 * - 按修正后的姿势蒙皮（SkinnedPose）：用现有 GPU 蒙皮把角色摆成骨架现在的姿势。
 * 双骨架编辑（手柄「rig_pair」，view/rigPair.tsx）也用这里的蒙皮层。 */

import type { DragMode } from "./dragGizmo";
import { useMemo } from "react";
import type { HandleData, HandleDef, SkeletonPoseData } from "../api";
import type { Reference } from "../state/handleView";
import { packetOf, useResults } from "../state/results";
import { useCookInputs } from "../state/cookInputs";
import { getNodeDefs } from "../state/catalog";
import type { ViewOptions } from "../model/viewOptions";
import { forward, rowsOf, type PoseRow } from "../model/skeletonPose";
import { SkeletonPoseLayer } from "./skeletonPose";
import { SceneElement } from "./sceneElement";
import { Character } from "./elements3d";
import { useScenes } from "./sceneData";
import { skinMatrices, skinPoints } from "../model/skinning";
import type { CharacterData } from "./sceneTypes";

/** 一个「骨架姿势」手柄在这个舞台上：它在 NodeDef.handles 里的下标、声明、能不能改。 */
export interface PoseHandle {
  index: number;
  def: HandleDef;
  operable: boolean; // 能改（可操作的一侧）；否则只画、半透明另一色
}

/** 手柄写回：`rows` 为这一侧的修正，一步（拖动中的样子在层里自己画，不写）。 */
export type PoseWrite = (param: string, rows: PoseRow[]) => void;

/** handle_data 里第 `index` 个手柄的数据，按它的种类（HandleDef.kind）读：骨架姿势的那一份。 */
export const poseDataOf = (data: Record<string, HandleData>, index: number): SkeletonPoseData | undefined =>
  data[String(index)] as SkeletonPoseData | undefined;

/** 这些「骨架姿势」手柄画成骨架（数据没到的不画）。 */
export function PoseLayers({ handles, data, values, write, mode, slot, o }: {
  handles: PoseHandle[];
  data: Record<string, HandleData>;
  values: Record<string, unknown> | undefined; // 参数名 → 值（节点参数）：按值的引用读成行
  write: PoseWrite;
  mode: DragMode;
  slot: string;
  o: ViewOptions;
}) {
  return (
    <>
      {handles.map((h) => {
        const d = poseDataOf(data, h.index);
        const param = h.def.params.pose;
        if (!d) return null;
        const editable = h.operable && !h.def.readonly && !!param;
        return (
          <SkeletonPoseLayer key={h.index} data={d} index={h.index} handle={h.def} value={param ? values?.[param] : undefined}
            write={(rows) => param && write(param, rows)} ghost={!editable} mode={mode} slot={slot} o={o} />
        );
      })}
    </>
  );
}

/** 参考显示的画法：叠加色取「另一种颜色」、半透明（与透过相机看时的叠加同一套画法，所以各种包都照常能画）。 */
const REF = "#b98cff";
export const refOptions = (o: ViewOptions): ViewOptions => ({
  ...o, overlayOpacity: 0.35, overlayTint: REF, overlayByPerson: false, boneColor: REF, cameraColor: REF,
  pointColor: "constant", pointTint: REF, curveColor: "constant", curveTint: REF,
});

/** 参考显示：另一个节点的一个口（或它的主输出）的结果，半透明另一色叠在舞台上，不可操作。 */
export function ReferenceLayer({ reference, frame, o }: {
  reference: Reference;
  frame: number;
  o: ViewOptions;
}) {
  const reply = useResults((s) => s.reply);
  const typeId = useCookInputs((s) => s.nodes[reference.node]?.typeId);
  const ghost = useMemo(() => refOptions(o), [o]);
  // 这个口真有的包（state/results.ts packetOf）；没给口时是节点声明的主输出
  const port = reference.port ?? (typeId ? getNodeDefs()[typeId]?.main : undefined) ?? "";
  const fp = packetOf(reply?.nodes[reference.node], port);
  const [loaded] = useScenes(fp ? [fp] : []);
  const scene = fp ? loaded.get(fp) : undefined;
  if (!scene) return null;
  return <SceneElement item={{ key: `reference/${reference.node}/${reference.port ?? ""}` }} d={scene} frame={frame} hidden={NONE} through look={null} o={ghost} />;
}
const NONE = new Set<string>();

/** 一副骨架摆姿势要的几项（骨架姿势手柄的数据，或双骨架编辑里有位置的一侧）。 */
type PosedSkeleton = Pick<SkeletonPoseData, "packet" | "path" | "names" | "parents" | "before" | "rotation">;

/** 角色按骨架现在的姿势蒙皮：取手柄数据里的角色（它那一侧输入的包、骨架路径），把关节摆到基准局部 + 修正经 FK 算出的
 * 世界位置，交给现有的 GPU 蒙皮画网格（不画骨，骨由手柄层画）。`pickKey`：同一舞台上画两份时各用各的拾取键。 */
export function SkinnedPose({ data, value, o, pickKey, offset, scale }: {
  data: PosedSkeleton; value: unknown; o: ViewOptions; pickKey?: string;
  offset?: readonly [number, number, number]; // 整个角色在舞台上挪多少（只影响显示；双骨架编辑把两侧挪开），拾取与框显也随它
  scale?: { s: number; ground: number }; // 整个角色 p ↦ s·(p − (0, ground, 0))（双骨架编辑按「尺寸」画），在挪之前
}) {
  const rows = useMemo(() => rowsOf(value), [value]);
  const [loaded] = useScenes([data.packet]);
  const scene = loaded.get(data.packet);
  const character = scene?.characters.find((c) => c.ref.path === data.path) ?? scene?.characters[0];
  const posed = useMemo(() => (character ? posedCharacter(character, data, rows, offset, scale) : null), [character, data, rows, offset, scale]);
  if (!posed) return null;
  return <Character c={posed} frame={0} o={o} pickKey={pickKey ?? `pose-skin/${data.path}`} through={false} meshes bones={false} person={undefined} />;
}

/** 角色的一份只有一帧的动画：按名字把每个关节摆到修正后的世界变换上（手柄数据里没有的关节用它第一帧的样子），整体
 * p ↦ s·(p − (0, ground, 0))（`scale`：绕地面原点缩放、脚底落到 y = 0）、再平移 `offset`。 */
function posedCharacter(c: CharacterData, data: PosedSkeleton, rows: PoseRow[], offset: readonly number[] = [0, 0, 0],
  scale?: { s: number; ground: number }): CharacterData {
  const { world } = forward(data, rows);
  const at = new Map(data.names.map((n, i) => [n, i]));
  const joints = c.ref.joints.length;
  const anim = new Float32Array(joints * 12);
  for (let j = 0; j < joints; j++) {
    const k = at.get(c.ref.joints[j]);
    if (k === undefined) {
      anim.set(c.anim.subarray(j * 12, j * 12 + 12), j * 12); // 第一帧
      continue;
    }
    const m = world[k]; // 列主序 4x4 → 行主序 3x4
    for (let r = 0; r < 3; r++) for (let col = 0; col < 4; col++) anim[j * 12 + r * 4 + col] = m[col * 4 + r];
  }
  if (scale && scale.s !== 1) {
    const { s, ground } = scale;
    for (let j = 0; j < joints; j++) for (let r = 0; r < 3; r++) {
      for (let col = 0; col < 3; col++) anim[j * 12 + r * 4 + col] *= s;
      anim[j * 12 + r * 4 + 3] = s * (anim[j * 12 + r * 4 + 3] - (r === 1 ? ground : 0));
    }
  }
  for (let j = 0; j < joints; j++) for (let r = 0; r < 3; r++) anim[j * 12 + r * 4 + 3] += offset[r];
  // 每点多于四个关节影响的网格平时画服务器逐帧求好的点（文件自己的动画），这个姿势服务器没有：按全部影响在这里蒙皮一次，
  // 当作它唯一的一帧（model/skinning.ts）。服务器没给影响（旧的视图数据）时画不出这个姿势：不画，不画一个错的姿势
  const skin = c.meshes.some((m) => m.ref.per_frame) ? skinMatrices(anim, c.bind, joints) : null;
  const meshes = c.meshes.flatMap((m) => {
    if (!m.ref.per_frame) return [m];
    if (!skin || !m.jointIndices || !m.jointWeights) return [];
    return [{ ...m, jointIndices: null, jointWeights: null, samples: new Map([[0, skinPoints(withShapes(m), m.jointIndices, m.jointWeights, m.ref.influences, skin)]]) }];
  });
  return { ...c, anim, meshes, ref: { ...c.ref, frames: [0] } };
}

/** 网格绑定姿势的点加上第一帧的混合变形（与 SkinnedBody 一样，偏移量加在蒙皮之前）。 */
function withShapes(m: CharacterData["meshes"][number]): Float32Array {
  const b = m.ref.shapes.length;
  if (!b || !m.shapeOffsets || !m.shapeWeights) return m.points;
  const out = Float32Array.from(m.points);
  const v = m.points.length;
  for (let s = 0; s < b; s++) {
    const w = m.shapeWeights[s];
    if (!w) continue;
    for (let k = 0; k < v; k++) out[k] += w * m.shapeOffsets[s * v + k];
  }
  return out;
}
