/** 三维舞台上与显示内容无关、由舞台统一提供的几层：
 * - 手柄「骨架姿势」（PoseLayers）：节点声明了这种手柄，舞台就按状态回复的 handle_data 画那一侧的骨架并让人改；
 * - 参考显示（ReferenceLayer）：另一个节点的结果或它某个手柄的数据，半透明、另一种颜色叠画，不可操作；
 * - 按修正后的姿势蒙皮（SkinnedPose）：用现有 GPU 蒙皮把角色摆成骨架现在的姿势。
 * 主视图（view/Stage3D.tsx）与弹窗里的舞台（view/HandleStage.tsx）都只用这几层，谁都不另写三维代码。 */

import type { DragMode } from "./dragGizmo";
import { useMemo } from "react";
import type { HandleDef, SkeletonPoseData } from "../api";
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
import type { CharacterData } from "./sceneTypes";

/** 一个「骨架姿势」手柄在这个舞台上：它在 NodeDef.handles 里的下标、声明、能不能改。 */
export interface PoseHandle {
  index: number;
  def: HandleDef;
  operable: boolean; // 能改（可操作的一侧）；否则只画、半透明另一色
}

/** 手柄写回：`rows` 为这一侧的修正，一步（拖动中的样子在层里自己画，不写）。 */
export type PoseWrite = (param: string, rows: PoseRow[]) => void;

/** 这些「骨架姿势」手柄画成骨架（数据没到的不画）。 */
export function PoseLayers({ handles, data, values, write, mode, slot, o }: {
  handles: PoseHandle[];
  data: Record<string, SkeletonPoseData>;
  values: Record<string, unknown> | undefined; // 参数名 → 值（主视图是节点参数，弹窗里是草稿）：按值的引用读成行
  write: PoseWrite;
  mode: DragMode;
  slot: string;
  o: ViewOptions;
}) {
  return (
    <>
      {handles.map((h) => {
        const d = data[String(h.index)];
        const param = h.def.params.pose;
        if (!d) return null;
        const editable = h.operable && !h.def.readonly && !!param;
        return (
          <SkeletonPoseLayer key={h.index} data={d} index={h.index} handle={h.def} value={param ? values?.[param] : undefined}
            write={(rows) => param && write(param, rows)} ghost={!editable} mode={mode} slot={slot} layer="edit" o={o} />
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

/** 参考显示：另一个节点的一个口（或它的主输出）的结果，或它某个手柄的数据，半透明另一色叠在舞台上，不可操作。
 * `values(node)`：手柄数据按哪一版修正画（参数名 → 值；主视图是节点参数，弹窗里是草稿）。 */
export function ReferenceLayer({ reference, frame, o, values, slot }: {
  reference: Reference;
  frame: number;
  o: ViewOptions;
  values: (node: string) => Record<string, unknown> | undefined;
  slot: string;
}) {
  const reply = useResults((s) => s.reply);
  const typeId = useCookInputs((s) => s.nodes[reference.node]?.typeId);
  const ghost = useMemo(() => refOptions(o), [o]);
  // 这个口真有的包（state/results.ts packetOf）；没给口时是节点声明的主输出
  const port = reference.port ?? (typeId ? getNodeDefs()[typeId]?.main : undefined) ?? "";
  const fp = reference.handle == null ? packetOf(reply?.nodes[reference.node], port) : null;
  const [loaded] = useScenes(fp ? [fp] : []);
  if (reference.handle != null) {
    const hd = reply?.handle_data;
    const d = hd?.node === reference.node ? hd.handles?.[String(reference.handle)] : undefined;
    const def = typeId ? getNodeDefs()[typeId]?.handles[reference.handle] : undefined;
    if (!d || !def) return null;
    const param = def.params.pose;
    return <SkeletonPoseLayer data={d} index={reference.handle} handle={def} value={param ? values(reference.node)?.[param] : undefined}
      write={() => undefined} ghost mode="rotate" slot={slot} layer="reference" o={o} />;
  }
  const scene = fp ? loaded.get(fp) : undefined;
  if (!scene) return null;
  return <SceneElement item={{ key: `reference/${reference.node}/${reference.port ?? ""}` }} d={scene} frame={frame} hidden={NONE} through look={null} o={ghost} />;
}
const NONE = new Set<string>();

/** 角色按骨架现在的姿势蒙皮：取手柄数据里的角色（它那一侧输入的包、骨架路径），把关节摆到基准局部 + 修正经 FK 算出的
 * 世界位置，交给现有的 GPU 蒙皮画网格（不画骨，骨由手柄层画）。 */
export function SkinnedPose({ data, value, o }: { data: SkeletonPoseData; value: unknown; o: ViewOptions }) {
  const rows = useMemo(() => rowsOf(value), [value]);
  const [loaded] = useScenes([data.packet]);
  const scene = loaded.get(data.packet);
  const character = scene?.characters.find((c) => c.ref.path === data.path) ?? scene?.characters[0];
  const posed = useMemo(() => (character ? posedCharacter(character, data, rows) : null), [character, data, rows]);
  if (!posed) return null;
  return <Character c={posed} frame={0} o={o} pickKey={`pose-skin/${data.path}`} through={false} meshes bones={false} person={undefined} />;
}

/** 角色的一份只有一帧的动画：按名字把每个关节摆到修正后的世界变换上（手柄数据里没有的关节用它第一帧的样子）。 */
function posedCharacter(c: CharacterData, data: SkeletonPoseData, rows: PoseRow[]): CharacterData {
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
  return { ...c, anim, ref: { ...c.ref, frames: [0] } };
}
