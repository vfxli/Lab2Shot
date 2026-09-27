/** Where a node places what it gives (lab2shot/nodes/handles.py Places, G17): the parameters that move (cm, Y up), turn
 * (degrees, in the declared order) and scale it (a factor; null none), declared once on the node type and repeated per
 * node in the status. The viewer previews with the same matrix the cook applies (lab2shot/data/scene.py trs_matrix),
 * so a dragged object lands where it was let go (webui/tests/placePreview.test.ts). */
export interface Places {
  translate: string;
  rotate: string;
  scale: string | null;
  order: "scale-rotate-translate";
  rotation: string; // the axes in the order they are applied: "XYZ" is about X first, then Y, then Z
  units: { translate: string; rotate: string };
}

/** three.js builds Euler order "ABC" as the matrix A·B·C, which applies C first: the declared order, reversed. */
export function threeOrder(rotation: string): "XYZ" | "XZY" | "YXZ" | "YZX" | "ZXY" | "ZYX" {
  const axes = rotation.toUpperCase();
  if (axes.length !== 3 || new Set(axes).size !== 3 || /[^XYZ]/.test(axes)) throw new Error(`not a rotation order: ${rotation}`);
  return [...axes].reverse().join("") as ReturnType<typeof threeOrder>;
}

type M3 = [number, number, number, number, number, number, number, number, number]; // row-major 3x3

const mul = (a: M3, b: M3): M3 => {
  const out = new Array(9).fill(0) as M3;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
  return out;
};

const axis = (name: string, deg: number): M3 => {
  const a = (deg * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  if (name === "X") return [1, 0, 0, 0, c, -s, 0, s, c];
  if (name === "Y") return [c, 0, s, 0, 1, 0, -s, 0, c];
  return [c, -s, 0, s, c, 0, 0, 0, 1];
};

/** The node's placement as the cook applies it: scale, then rotate in the declared order, then translate. 4x4 for
 * column vectors, column-major (three.js Matrix4.fromArray order). */
export function placeMatrix(places: Places, params: Record<string, unknown>): number[] {
  const t = (params[places.translate] as number[] | undefined) ?? [0, 0, 0];
  const deg = (params[places.rotate] as number[] | undefined) ?? [0, 0, 0];
  const s = places.scale ? Number(params[places.scale] ?? 1) : 1;
  const angle = { X: deg[0], Y: deg[1], Z: deg[2] } as Record<string, number>;
  // applied first stands rightmost: "XYZ" is Rz·Ry·Rx
  let r: M3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (const a of places.rotation.toUpperCase()) r = mul(axis(a, angle[a]), r);
  return [r[0] * s, r[3] * s, r[6] * s, 0, r[1] * s, r[4] * s, r[7] * s, 0, r[2] * s, r[5] * s, r[8] * s, 0, t[0], t[1], t[2], 1];
}
