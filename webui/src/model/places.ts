/** Where a node places what it gives (lab2shot/nodes/handles.py Places): the parameters that move (cm, Y up), turn
 * (degrees, in the declared order) and scale it (a factor; null none), declared once on the node type and repeated per
 * node in the status. The viewer shows what the handle places with the same matrix the cook applies (lab2shot/data/scene.py
 * trs_matrix), so a result lands where the display showed it: `lab2shot check places` runs placeMatrix and the cook's
 * matrix on the same parameters and compares them. */
import { compose, roundChanged, turnedAngles, type Increment } from "./math3d";

export { threeOrder } from "./math3d";

export interface Places {
  translate: string;
  rotate: string;
  scale: string | null;
  order: "scale-rotate-translate";
  rotation: string; // the axes in the order they are applied: "XYZ" is about X first, then Y, then Z
  units: { translate: string; rotate: string };
}

/** The node's placement as the cook applies it: scale, then rotate in the declared order, then translate. 4x4 for
 * column vectors, column-major (three.js Matrix4.fromArray order). The rotation arithmetic is model/math3d.ts's one. */
export function placeMatrix(places: Places, params: Record<string, unknown>): number[] {
  const t = (params[places.translate] as number[] | undefined) ?? [0, 0, 0];
  const deg = (params[places.rotate] as number[] | undefined) ?? [0, 0, 0];
  const s = places.scale ? Number(params[places.scale] ?? 1) : 1;
  return compose(t, deg, [s, s, s], places.rotation);
}

/** The parameters after one handle drag (view/dragGizmo.tsx): the translation moves by the pivot's travel, the rotation
 * turns about the pivot (R' = turn · R, written as the angles nearest the old ones), the one scale factor grows by the
 * mean of the drag's per-axis factors. Only the changed components are rounded (1/100 cm, 1/100 degree, 1/1000); null
 * when the drag takes the scale to 0 or below (nothing is written). */
export function draggedPlace(places: Places, params: Record<string, unknown>, d: Increment): Record<string, unknown> | null {
  const t = (params[places.translate] as number[] | undefined) ?? [0, 0, 0];
  const deg = (params[places.rotate] as number[] | undefined) ?? [0, 0, 0];
  const out: Record<string, unknown> = {
    [places.translate]: roundChanged(t.map((v, k) => v + d.move[k]), t, 100),
    [places.rotate]: roundChanged(turnedAngles(places.rotation, d.turn, deg), deg, 100),
  };
  if (places.scale) {
    const s = Number(params[places.scale] ?? 1);
    const next = roundChanged([(s * (d.grow[0] + d.grow[1] + d.grow[2])) / 3], [s], 1000)[0];
    if (!(next > 0)) return null;
    out[places.scale] = next;
  }
  return out;
}
