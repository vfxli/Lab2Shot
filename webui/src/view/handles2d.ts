import type { BoxesData, HandleDef, TracksData } from "../api";
import type { Frame } from "./overlays";
import type { Drag, HandleTool2D, Pt } from "./handleParts";
import { pointsTool } from "./points2d";
import { boxTool } from "./box2d";
import { cornersTool } from "./corners2d";
import { canvasTool } from "./canvas2d";
import { personTool } from "./person2d";
import { figureTool } from "./figure2d";
import { TOOLS_3D } from "./handleEditing";

export type { Drag, HandleTool2D, Pt };
export { addFigure, figureAdd, figureFrames } from "./figure2d";

/** The 2D handles (nodes/handles.py): one tool per kind, bound to a list parameter of the node, each in its own file
 * (view/<kind>2d.ts) and listed once here. A tool draws the parameter's entries on the current frame and turns clicks
 * and drags into new entries; a right click removes the entry under the pointer. Entries are "frame:x,y[,label]"
 * (points, person), "frame:x1,y1,x2,y2" (box), "frame:x1,y1,...,x4,y4" (a plane's corners, in order around it),
 * "frame:x1,y1,..." with as many points as were drawn (canvas: one closed outline each) and a figure's joints, image
 * pixels from the top-left corner.
 *
 * The stage (view/Stage2D.tsx), the viewer's toolbar (editor/Viewer.tsx) and the parameter panel ask a handle's tool
 * (`tool2d`) and never name a kind: a new 2D handle kind is one file with its HandleTool2D, one row here, its words
 * (ui.view.hint.<kind>) and its server side (nodes/handles.py HANDLE_KINDS). */
export const TOOLS_2D: Partial<Record<HandleDef["kind"], HandleTool2D>> = {
  points: pointsTool,
  box: boxTool,
  corners: cornersTool,
  canvas: canvasTool,
  person: personTool,
  figure: figureTool,
};

/** A handle's 2D tool (undefined: not a 2D handle). */
export const tool2d = (h: HandleDef | null | undefined): HandleTool2D | undefined => (h ? TOOLS_2D[h.kind] : undefined);

/** The handle of the node's that picks people with the parameter `param` (any 「选人」-like handle: its tool has
 * `people`), or undefined. */
export const peopleHandle = (handles: readonly HandleDef[] | undefined, param?: string): HandleDef | undefined =>
  handles?.find((h) => !!tool2d(h)?.people && (param === undefined || Object.values(h.params).includes(param)));

/** The pointer's action with this handle, for the cursor and the toolbar hint: the key of its words (t() when shown). */
export const handleHint = (h: HandleDef): string => tool2d(h)?.hint ?? TOOLS_3D[h.kind]?.hint ?? "";

/** `tracked`: this node's own result, when it is current (a planar track's outline per frame, overlays.ts drawTracks;
 * view/plan.ts underHandles leaves a result out while it does not match the handle). */
export function drawHandle(h: HandleDef, f: Frame, values: string[], drag: Drag | null, tracked: TracksData | null = null) {
  tool2d(h)?.draw?.(h, f, values, drag, tracked);
}

/** A click: the new entries, or null when it adds nothing. `label`: the points label chosen in the toolbar. */
export function clickHandle(h: HandleDef, values: string[], frame: number, p: Pt, label: number, source: BoxesData | null): string[] | null {
  return tool2d(h)?.click?.(h, values, frame, p, label, source) ?? null;
}

/** Where a drag of this handle starts; null where nothing can be dragged. */
export function dragStart(h: HandleDef, values: string[], frame: number, p: Pt & { inside: boolean }, scale: number): Drag | null {
  return tool2d(h)?.start?.(h, values, frame, p, scale) ?? null;
}

/** The pointer moved during a drag: the drag's current state. */
export function moveDrag(h: HandleDef, drag: Drag, p: Pt): Drag {
  const move = tool2d(h)?.move;
  return move ? move(drag, p) : { ...drag, to: p };
}

/** A drag finished: the new entries, or null. */
export function dragHandle(h: HandleDef, values: string[], frame: number, drag: Drag): string[] | null {
  return tool2d(h)?.finish?.(h, values, frame, drag) ?? null;
}

/** A right click: the entries without the one under the pointer on this frame (null: nothing there). */
export function removeAt(h: HandleDef, values: string[], frame: number, p: Pt, scale: number, source: BoxesData | null = null, tracked: TracksData | null = null): string[] | null {
  return tool2d(h)?.remove(h, values, frame, p, scale, source, tracked) ?? null;
}
