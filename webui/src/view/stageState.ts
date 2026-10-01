import { createContext, useContext, useEffect } from "react";
import * as THREE from "three";

/** What the parts of one 3D stage share outside React's state, read every frame: what can be picked and framed. */

export interface PickRay {
  raycaster: THREE.Raycaster;
  camera: THREE.Camera;
  px: { x: number; y: number }; // the click in the canvas, CSS pixels
  size: { width: number; height: number };
}

export interface Pickable {
  label: string;
  bounds: () => THREE.Box3 | null; // in the world, on the current frame
  // how far along the view the click hits it (null: missed): always `project(point).depth` of the point hit, the one
  // quantity the Picker compares (view/camera3d.tsx) — a surface's ray distance is longer off the view's centre, so a
  // wall in front could lose to a joint behind it. A surface uses `surfaceHit` below
  hit: (p: PickRay) => number | null;
  // the draw order it is drawn at (view/drawOrder.ts ORDER; none: 0, the scene): what is drawn over the rest is picked over
  // it, whatever its depth — a skeleton drawn over everything (overlay) stays clickable inside the mesh around it. The
  // Picker compares this first, depth only within one order
  order?: number;
  // something that keeps its own selection (a pose handle's skeleton: which joint): picking it calls `choose` with the
  // `part` under the click instead of making it the stage's selection (view/camera3d.tsx Picker)
  part?: (p: PickRay) => string | null;
  choose?: (part: string | null) => void;
}

export class StageState {
  pickables = new Map<string, Pickable>();
  // when a handle's gizmo last took a press (performance.now(); view/dragGizmo.tsx): a click that started no earlier is
  // the gizmo's, not a pick (view/camera3d.tsx Picker)
  claimedAt = -Infinity;
}

export const StageContext = createContext<StageState>(new StageState());

/** A stage slot's life (view/camera3d.tsx ViewCamera `slot`): the viewer's lasts as long as the page; a dialog stage's
 * ends when that stage goes (ViewCamera `ephemeral`, the last part of the stage to unmount), and everything kept per slot
 * (the camera, the joint picked) is dropped then, by the ones who keep it, registered here. */
const slotEnds: ((slot: string) => void)[] = [];
export const onSlotEnd = (drop: (slot: string) => void): void => void slotEnds.push(drop);
export const endSlot = (slot: string): void => slotEnds.forEach((drop) => drop(slot));
export const useStage = () => useContext(StageContext);

/** Registers something that can be picked and framed while it is shown. */
export function usePickable(key: string, p: Pickable | null): void {
  const stage = useStage();
  useEffect(() => {
    if (!p) return;
    stage.pickables.set(key, p);
    return () => {
      if (stage.pickables.get(key) === p) stage.pickables.delete(key);
    };
  }, [stage, key, p]);
}

const scratch = new THREE.Vector3();

/** Screen position (CSS pixels) of a world point, and its distance along the view; null behind the camera. The page's
 * one world-to-screen projection: picking (a PickRay is such a view) and the joint names (view/elements3d.tsx
 * JointLabels) use it. */
export function project(v: THREE.Vector3, p: { camera: THREE.Camera; size: { width: number; height: number } }): { x: number; y: number; depth: number } | null {
  const q = scratch.copy(v).applyMatrix4(p.camera.matrixWorldInverse);
  const depth = -q.z;
  q.applyMatrix4(p.camera.projectionMatrix);
  if (!(p.camera as THREE.OrthographicCamera).isOrthographicCamera && depth <= 0) return null;
  return { x: ((q.x + 1) / 2) * p.size.width, y: ((1 - q.y) / 2) * p.size.height, depth };
}

/** Where the click's ray first meets a surface, as a Pickable's `hit` gives it (the depth along the view, `project`). */
export function surfaceHit(p: PickRay, obj: THREE.Object3D | null | undefined): number | null {
  const first = obj ? p.raycaster.intersectObject(obj, false)[0] : undefined;
  return first ? project(first.point, p)?.depth ?? null : null;
}

/** What a click picks among `pickables`: what is drawn on top first (Pickable.order, the draw order's own numbers), then
 * the nearest along the view. The one rule (view/camera3d.tsx Picker). */
export function picked(pickables: ReadonlyMap<string, Pickable>, ray: PickRay): { key: string; p: Pickable } | null {
  let best: { key: string; p: Pickable; at: number; order: number } | null = null;
  for (const [key, p] of pickables) {
    const at = p.hit(ray);
    const order = p.order ?? 0;
    if (at !== null && (!best || order > best.order || (order === best.order && at < best.at))) best = { key, p, at, order };
  }
  return best;
}

/** Where a click lands on lines drawn as segments (`segments`: x y z of each segment's two ends, as three's
 * LineSegments; `matrix`: their placement in the world): the depth (`project`) at the point of the nearest segment
 * closest to the click on screen, when that is within `reach` CSS pixels anywhere along it, not only at its ends (a bone
 * 180 px long is clickable at its middle). null: missed. Bones and curves both use it. */
export function segmentsHit(segments: ArrayLike<number>, p: PickRay, matrix: THREE.Matrix4 | null = null, reach = 8): number | null {
  let best: { off: number; depth: number } | null = null;
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  for (let k = 0; k + 5 < segments.length; k += 6) {
    a.set(segments[k], segments[k + 1], segments[k + 2]);
    b.set(segments[k + 3], segments[k + 4], segments[k + 5]);
    if (matrix) (a.applyMatrix4(matrix), b.applyMatrix4(matrix));
    const sa = project(a, p);
    const sb = sa && project(b, p);
    if (!sa || !sb) continue;
    const dx = sb.x - sa.x, dy = sb.y - sa.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.min(1, Math.max(0, ((p.px.x - sa.x) * dx + (p.px.y - sa.y) * dy) / len2)) : 0;
    const off = Math.hypot(sa.x + t * dx - p.px.x, sa.y + t * dy - p.px.y);
    const depth = sa.depth + t * (sb.depth - sa.depth);
    if (off < reach && (!best || depth < best.depth)) best = { off, depth };
  }
  return best ? best.depth : null;
}
