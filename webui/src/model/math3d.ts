/** The 3D viewer's maths that needs no three.js: bounds and framing.
 *
 * Pure: no imports. The page's one set of 4x4 and rotation-order arithmetic lives here too (below). */

export interface Bounds {
  min: [number, number, number];
  max: [number, number, number];
}

/** Bounds of one frame's points, from at most `sample` of them (the spread order makes the first ones a fair
 * sample). null when there are none. */
export function spanBounds(data: Float32Array, start: number, count: number, stride: number, sample = 200_000): Bounds | null {
  const n = Math.min(count, sample);
  if (n <= 0) return null;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    const p = (start + i) * stride;
    for (let c = 0; c < 3; c++) {
      const v = data[p + c];
      if (v < min[c]) min[c] = v;
      if (v > max[c]) max[c] = v;
    }
  }
  return { min, max };
}

/** How far a perspective camera stands from a sphere's centre to see it whole with a margin (fov in degrees,
 * vertical; aspect = width / height). */
export function fitDistance(radius: number, fovDeg: number, aspect: number, margin = 1.15): number {
  const half = (fovDeg * Math.PI) / 360;
  const halfX = Math.atan(Math.tan(half) * aspect);
  return (Math.max(radius, 1e-6) * margin) / Math.sin(Math.min(half, halfX));
}

/** The orthographic zoom (pixels per scene unit) that shows a sphere whole with a margin in a view of w x h pixels. */
export const fitZoom = (radius: number, w: number, h: number, margin = 1.15) => Math.min(w, h) / (2 * Math.max(radius, 1e-6) * margin);

// ------------------------------------------------------------------ 4x4 matrices and rotation orders

/** A 4x4 matrix: 16 numbers, column-major (three.js Matrix4.fromArray order), column vectors. The page's one set of
 * matrix arithmetic outside three.js: skeleton poses (model/skeletonPose.ts), placements (model/places.ts) and the
 * viewer's skinning inverses (view/elements3d.tsx) all use it. */
export type M4 = number[];

export const IDENTITY: M4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

export function mul(a: M4, b: M4): M4 {
  const out = new Array<number>(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  return out;
}

/** The inverse, or null for a singular matrix (a zero scale, axes collapsed: judged against its own size): the one place that decides a matrix
 * cannot be inverted. Code that can say why it cannot go on (a pose handle on a joint under a zero scale) asks this;
 * `invert` is for matrices that must be invertible. */
export function inverse(m: M4): M4 | null {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = m;
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12, b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  // singular relative to its own size (the determinant over the product of the columns' lengths): a joint scaled to 1e-5
  // is small, not singular; a zero column or collapsed axes are
  const size = Math.hypot(a00, a01, a02) * Math.hypot(a10, a11, a12) * Math.hypot(a20, a21, a22);
  if (!Number.isFinite(det) || size === 0 || Math.abs(det) / size < 1e-9) return null;
  const d = 1 / det;
  return [
    (a11 * b11 - a12 * b10 + a13 * b09) * d, (a02 * b10 - a01 * b11 - a03 * b09) * d, (a31 * b05 - a32 * b04 + a33 * b03) * d, (a22 * b04 - a21 * b05 - a23 * b03) * d,
    (a12 * b08 - a10 * b11 - a13 * b07) * d, (a00 * b11 - a02 * b08 + a03 * b07) * d, (a32 * b02 - a30 * b05 - a33 * b01) * d, (a20 * b05 - a22 * b02 + a23 * b01) * d,
    (a10 * b10 - a11 * b08 + a13 * b06) * d, (a01 * b08 - a00 * b10 - a03 * b06) * d, (a30 * b04 - a31 * b02 + a33 * b00) * d, (a21 * b02 - a20 * b04 - a23 * b00) * d,
    (a11 * b07 - a10 * b09 - a12 * b06) * d, (a00 * b09 - a01 * b07 + a02 * b06) * d, (a31 * b01 - a30 * b03 - a32 * b00) * d, (a20 * b03 - a21 * b01 + a22 * b00) * d,
  ];
}

/** The inverse; a singular matrix throws rather than returning zeros (a zero matrix passed on silently turns later
 * results into NaN). */
export function invert(m: M4): M4 {
  const inv = inverse(m);
  if (!inv) throw new Error("singular matrix");
  return inv;
}

type Axis = "X" | "Y" | "Z";

/** The rotation order as declared by the server ("XYZ": about X first, then Y, then Z, i.e. Rz·Ry·Rx), checked. */
export function axesOf(order: string): [Axis, Axis, Axis] {
  const axes = order.toUpperCase();
  if (axes.length !== 3 || new Set(axes).size !== 3 || /[^XYZ]/.test(axes)) throw new Error(`not a rotation order: ${order}`);
  return [...axes] as [Axis, Axis, Axis];
}

/** three.js builds Euler order "ABC" as the matrix A·B·C, which applies C first: the declared order, reversed. */
export function threeOrder(order: string): "XYZ" | "XZY" | "YXZ" | "YZX" | "ZXY" | "ZYX" {
  return [...axesOf(order)].reverse().join("") as ReturnType<typeof threeOrder>;
}

const rad = (deg: number) => (deg * Math.PI) / 180;
const deg = (r: number) => (r * 180) / Math.PI;

/** A rotation about one axis, as a 4x4. */
function about(axis: Axis, degrees: number): M4 {
  const c = Math.cos(rad(degrees)), s = Math.sin(rad(degrees));
  if (axis === "X") return [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
  if (axis === "Y") return [c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1];
  return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

/** The rotation of angles `degrees` = [about X, about Y, about Z], applied in the declared order. */
export function rotation(order: string, degrees: readonly number[]): M4 {
  const angle = { X: degrees[0], Y: degrees[1], Z: degrees[2] };
  let r = IDENTITY;
  for (const a of axesOf(order)) r = mul(about(a, angle[a]), r); // applied first stands rightmost
  return r;
}

/** The angles [about X, about Y, about Z] (degrees) of a rotation, for the declared order (the inverse of `rotation`;
 * at gimbal lock the free angle goes to the first axis the formulas below keep). `m` must be a pure rotation. */
export function anglesOf(order: string, m: M4): [number, number, number] {
  const [m11, m21, m31, , m12, m22, m32, , m13, m23, m33] = m;
  const cl = (v: number) => Math.min(1, Math.max(-1, v));
  let x = 0, y = 0, z = 0;
  const locked = (v: number) => Math.abs(v) >= 0.9999999;
  switch (threeOrder(order)) {
    case "XYZ": y = Math.asin(cl(m13)); if (!locked(m13)) { x = Math.atan2(-m23, m33); z = Math.atan2(-m12, m11); } else { x = Math.atan2(m32, m22); } break;
    case "YXZ": x = Math.asin(-cl(m23)); if (!locked(m23)) { y = Math.atan2(m13, m33); z = Math.atan2(m21, m22); } else { y = Math.atan2(-m31, m11); } break;
    case "ZXY": x = Math.asin(cl(m32)); if (!locked(m32)) { y = Math.atan2(-m31, m33); z = Math.atan2(-m12, m22); } else { z = Math.atan2(m21, m11); } break;
    case "ZYX": y = Math.asin(-cl(m31)); if (!locked(m31)) { x = Math.atan2(m32, m33); z = Math.atan2(m21, m11); } else { z = Math.atan2(-m12, m22); } break;
    case "YZX": z = Math.asin(cl(m21)); if (!locked(m21)) { x = Math.atan2(-m23, m22); y = Math.atan2(-m31, m11); } else { y = Math.atan2(m13, m33); } break;
    case "XZY": z = Math.asin(-cl(m12)); if (!locked(m12)) { x = Math.atan2(m32, m22); y = Math.atan2(m13, m11); } else { x = Math.atan2(-m23, m33); } break;
  }
  return [deg(x), deg(y), deg(z)];
}

/** T · R · S: translate, rotation of `degrees` in the declared order, per-axis scale. */
export function compose(translate: readonly number[], degrees: readonly number[], scale: readonly number[], order: string): M4 {
  const r = rotation(order, degrees);
  const [sx, sy, sz] = scale;
  return [r[0] * sx, r[1] * sx, r[2] * sx, 0, r[4] * sy, r[5] * sy, r[6] * sy, 0, r[8] * sz, r[9] * sz, r[10] * sz, 0, translate[0], translate[1], translate[2], 1];
}

/** The translation, angles and per-axis scale of a matrix made by `compose` (no shear). A mirroring matrix (negative
 * determinant) keeps the sign on the X scale, so a scale dragged through zero is not read back as a 180° turn. */
export function decompose(m: M4, order: string): { translate: [number, number, number]; rotate: [number, number, number]; scale: [number, number, number] } {
  const len = (c: number) => Math.hypot(m[c * 4], m[c * 4 + 1], m[c * 4 + 2]);
  const s: [number, number, number] = [len(0), len(1), len(2)];
  const det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]);
  if (det < 0) s[0] = -s[0];
  const r: M4 = [m[0] / s[0], m[1] / s[0], m[2] / s[0], 0, m[4] / s[1], m[5] / s[1], m[6] / s[1], 0, m[8] / s[2], m[9] / s[2], m[10] / s[2], 0, 0, 0, 0, 1];
  return { translate: [m[12], m[13], m[14]], rotate: anglesOf(order, r), scale: s };
}

/** `m` as translate · rotation (declared order) · per-axis scale, only if it is one (a change carried through frames that
 * are not mirror images of each other can pick up a shear: then null, rather than a decomposition that does not
 * rebuild `m`). */
export function exactTRS(m: M4, order: string): ReturnType<typeof decompose> | null {
  const trs = decompose(m, order);
  const back = compose(trs.translate, trs.rotate, trs.scale, order);
  // each column against its own length, the translation against its own size: a large scale on one axis must not loosen
  // the test of another (a shear of 5e-4 hidden beside a scale of 1000), nor a large translation that of the 3x3
  const len = [0, 1, 2].map((c) => Math.max(1e-12, Math.hypot(m[c * 4], m[c * 4 + 1], m[c * 4 + 2])));
  const moved = Math.max(1, Math.abs(m[12]), Math.abs(m[13]), Math.abs(m[14]));
  const size = (k: number) => (k >= 12 && k < 15 ? moved : k % 4 < 3 && k < 12 ? len[k >> 2] : 1);
  return back.every((v, k) => Math.abs(v - m[k]) <= 1e-6 * size(k)) ? trs : null;
}

/** Whether `m`'s 3x3 is a similarity (a rotation or reflection times one uniform scale): its columns are orthogonal and of
 * one length. A change conjugated by a similarity keeps its shape (no shear appears). */
export function similarity(m: M4): boolean {
  const c = [0, 1, 2].map((k) => [m[k * 4], m[k * 4 + 1], m[k * 4 + 2]]);
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const l2 = c.map((v) => dot(v, v));
  const size = Math.max(...l2);
  if (!(size > 0)) return false;
  const tol = 1e-6 * size;
  return Math.abs(l2[0] - l2[1]) <= tol && Math.abs(l2[0] - l2[2]) <= tol && [[0, 1], [0, 2], [1, 2]].every(([a, b]) => Math.abs(dot(c[a], c[b])) <= tol);
}

/** The reflection about the plane through `point` with normal `normal`: M = [I − 2nnᵀ, 2(n·p)n]. */
export function reflectionAbout(normal: readonly number[], point: readonly number[]): M4 {
  const l = Math.hypot(normal[0], normal[1], normal[2]) || 1;
  const [x, y, z] = [normal[0] / l, normal[1] / l, normal[2] / l];
  const d = x * point[0] + y * point[1] + z * point[2];
  return [1 - 2 * x * x, -2 * x * y, -2 * x * z, 0, -2 * x * y, 1 - 2 * y * y, -2 * y * z, 0, -2 * x * z, -2 * y * z, 1 - 2 * z * z, 0, 2 * d * x, 2 * d * y, 2 * d * z, 1];
}

/** A change `d` made in frame `from`, mirrored by `mirror` into frame `to`: inv(to) · M · from · d · inv(from) · M · to,
 * the whole affine (scale included). The one mirror formula (model/skeletonPose.ts mirroredRow). null when a frame is
 * singular. */
export function mirroredChange(d: M4, from: M4, to: M4, mirror: M4): M4 | null {
  const world = outOfFrame(from, d);
  return world && inFrame(to, mul(mirror, mul(world, mirror)));
}

/** The rotation part of a matrix, made orthonormal (Gram–Schmidt on its columns): a frame to express directions in. */
export function frameOf(m: M4): M4 {
  const n = (v: number[]) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return v.map((x) => x / l); };
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const x = n([m[0], m[1], m[2]]);
  const y0 = [m[4], m[5], m[6]];
  const y = n(y0.map((v, i) => v - dot(y0, x) * x[i]));
  const z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
  return [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, 0, 0, 0, 1];
}

/** A change `m` given in the world, as the same change in the frame `a` (inv(a) · m · a), and back (a · m · inv(a)):
 * how a correction moves between a joint's own frame and the world, the mirror included. null when `a` is singular. */
export function inFrame(a: M4, m: M4): M4 | null {
  const inv = inverse(a);
  return inv && mul(inv, mul(m, a));
}
export function outOfFrame(a: M4, m: M4): M4 | null {
  const inv = inverse(a);
  return inv && mul(a, mul(m, inv));
}

/** A turn in the world, as the same turn in the frame of `m`'s linear part: inv(A) · turn · A (A = `m` without its
 * translation). This is the one change of frame for a handle's increment: `m`'s real axes are used, so a frame that is
 * a mirror (negative determinant) turns the same way the handle's ring does; where `m` shears (an unevenly scaled parent)
 * the result is not a pure rotation and is taken to the nearest frame (frameOf). */
export function turnIn(m: M4, turn: M4): M4 {
  const a: M4 = [m[0], m[1], m[2], 0, m[4], m[5], m[6], 0, m[8], m[9], m[10], 0, 0, 0, 0, 1];
  return frameOf(mul(invert(a), mul(turn, a)));
}

/** A direction (w = 0) through the linear part of `m`. */
export const turn = (m: M4, v: readonly number[]): [number, number, number] =>
  [m[0] * v[0] + m[4] * v[1] + m[8] * v[2], m[1] * v[0] + m[5] * v[1] + m[9] * v[2], m[2] * v[0] + m[6] * v[1] + m[10] * v[2]];

/** The angles `deg` (declared order) after `turn` (a world turn already in their frame): the angles nearest the old
 * ones for turn · R. A turn that is no turn (within 1e-12 of the identity: a drag that only moved) leaves `deg` exactly
 * as they were: read back through anglesOf near ±90° (gimbal lock) they would come out as another set for the same
 * rotation, and a value the user typed would be rewritten (and an undo step made) by a drag that never turned. The one
 * way a handle's turn is written back (model/places.ts draggedPlace, model/skeletonPose.ts draggedRow). */
export function turnedAngles(order: string, turn: M4, deg: readonly number[]): [number, number, number] {
  const still = [0, 1, 2, 4, 5, 6, 8, 9, 10].every((k) => Math.abs(turn[k] - IDENTITY[k]) <= 1e-12);
  return still ? [deg[0], deg[1], deg[2]] : nearestAngles(order, mul(turn, rotation(order, deg)), deg);
}

/** Of the angle triples that make rotation `m` in the declared order, the one nearest `near`: each angle may move by
 * whole turns, and every Tait–Bryan order has a second solution (first and last angle + 180°, middle 180° − it). A
 * handle writes this, so a 100° turn nudged by 1° reads back as 101°, not as the equal (180°, 79°, 180°). */
export function nearestAngles(order: string, m: M4, near: readonly number[]): [number, number, number] {
  const a = anglesOf(order, m);
  const middle = "XYZ".indexOf(axesOf(order)[1]);
  const other = a.map((v, k) => (k === middle ? 180 - v : v + 180));
  const wrap = (v: number, to: number) => v + 360 * Math.round((to - v) / 360);
  const fit = (c: number[]) => c.map((v, k) => wrap(v, near[k])) as [number, number, number];
  const off = (c: number[]) => c.reduce((s, v, k) => s + Math.abs(v - near[k]), 0);
  const one = fit(a), two = fit(other);
  return off(two) < off(one) ? two : one;
}

/** Rounds to 1/`per` only the components that differ from `before`; the others keep their exact value (a handle writes
 * what it changed and leaves a typed-in 1.2345 alone). */
export const roundChanged = (next: readonly number[], before: readonly number[], per: number): [number, number, number] =>
  next.map((v, k) => (Math.abs(v - before[k]) < 1e-9 ? before[k] : Math.round(v * per) / per + 0)) as [number, number, number]; // + 0: no -0

/** One handle drag's increment in the world (view/dragGizmo.tsx makes it): `move` the pivot's travel, `turn` the
 * rotation about the pivot, `grow` the scale factor along each of the gizmo's own axes. */
export interface Increment {
  move: [number, number, number];
  turn: M4;
  grow: [number, number, number];
}

/** Of a scale handle's per-axis factors, the one being dragged: the farthest from 1 (by ratio, so ×0.5 and ×2 weigh the
 * same). A scale written as one number takes this for all three axes (view/dragGizmo.tsx `uniform`). */
export const draggedFactor = (grow: readonly number[]): number =>
  grow.reduce((a, b) => (Math.abs(Math.log(Math.abs(b) || 1e-9)) > Math.abs(Math.log(Math.abs(a) || 1e-9)) ? b : a), 1);
