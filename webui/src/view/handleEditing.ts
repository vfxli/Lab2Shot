import type { HandleDef } from "../api";
import { useRigPairView } from "../state/rigPairView";

/** The 3D handles (nodes/handles.py): one row per kind, the one table of what the viewer, the 3D stage, the display
 * plan and the parameter panel need to know of a kind, so none of them names a kind (the 2D handles' table is
 * view/handles2d.ts TOOLS_2D).
 * - `hint`: the key of the words for the pointer's action;
 * - `role`: what the handle does in the 3D stage. "place": it places content by the node's current parameters (the
 *   transform gizmo, view/Stage3D.tsx TransformHandle; view/plan.ts underHandles shows the content placed by it);
 *   "pose": it edits one skeleton's joint corrections (view/skeletonPose.tsx; the panel's PoseSummary, its param
 *   role "pose"); "pair": it edits two skeletons side by side (view/rigPair.tsx; the panel's summary rows). A "pose"
 *   or "pair" handle draws its own skeleton (the stage's handle layer) and does not change what the plan shows;
 * - `scales`: whether its gizmo offers scale;
 * - `edited`: it is edited in the view through 「在视图里编辑 / 改」 (state/handleView.ts editing): the stage it is edited
 *   on (the viewer shows that stage while it is edited, result or no result: the handle's data comes with the status
 *   reply, not from a cooked result) and whether it uses the move / rotate / scale tools right now (it can depend on the
 *   editor's own mode: the skeleton-pair editor uses them in 「编辑姿态」 only). Read inside a React render: it may use
 *   stores. A kind without it is not edited in the view. */
interface HandleTool3D {
  hint: string;
  role: "place" | "pose" | "pair";
  scales?: (h: HandleDef) => boolean;
  edited?: { stage: "3d" | "2d"; transforms: (h: HandleDef) => boolean };
}

export const TOOLS_3D: Partial<Record<HandleDef["kind"], HandleTool3D>> = {
  transform: { hint: "ui.view.hint.transform", role: "place", scales: (h) => !!h.params.scale },
  skeleton_pose: { hint: "ui.view.hint.skeleton_pose", role: "pose", scales: () => true, edited: { stage: "3d", transforms: (h) => !h.readonly } },
  rig_pair: { hint: "ui.view.hint.rig_pair", role: "pair", scales: () => true, edited: { stage: "3d", transforms: () => useRigPairView.getState().mode === "pose" } },
};

/** A handle's 3D role (null: not a 3D handle). */
export const role3d = (h: { kind: string } | null | undefined): HandleTool3D["role"] | null => (h && TOOLS_3D[h.kind as HandleDef["kind"]]?.role) || null;
/** It draws its own skeleton (a "pose" or "pair" handle) and leaves what the display plan shows as it is. */
export const drawsSkeleton = (h: HandleDef): boolean => role3d(h) === "pose" || role3d(h) === "pair";
/** Its gizmo offers scale. */
export const handleScales = (h: HandleDef): boolean => !!TOOLS_3D[h.kind]?.scales?.(h);

export const editedOn = (h: HandleDef | undefined): "3d" | "2d" | null => (h && TOOLS_3D[h.kind]?.edited?.stage) || null;
export const usesTransforms = (h: HandleDef): boolean => !!TOOLS_3D[h.kind]?.edited?.transforms(h);
