/** The 3D viewer's data as the server sends it (lab2shot/server/view_data.py): its description, how its arrays are
 * packed (losslessly: float32 bytes sorted by significance, point caches as bit differences from the frame before,
 * gzip on the wire), and the arithmetic the viewer does on it off the graphics card (depth clouds rebuilt, samples
 * picked, matrices, bones, UVs).
 *
 * Pure: imports only the pure frame arithmetic (model/timelineMath.ts). */

import { closer } from "./timelineMath";

type TypeName = "f32" | "u32" | "u16" | "u8";
export type Typed = Float32Array | Uint32Array | Uint16Array | Uint8Array;

/** A base array: in which part, where, how many elements, of what type, how packed. */
interface ArrayRef {
  part: string;
  o: number;
  n: number;
  t: TypeName;
  e: "" | "s" | "x";
}

interface PartRef {
  url: string;
  bytes?: number; // on the wire (base parts; chunks are read when asked for)
  raw?: number;
  frames?: [number, number]; // a chunk: the frames its samples are at
}

interface Item {
  name: string;
  path: string;
  frames: number[]; // the frames it has a sample at (one: it never changes)
  world?: ArrayRef; // local-to-world per sample, 4x4 row after row
}

export interface ModelRef extends Item {
  faces: ArrayRef;
  vertices: number;
  points?: ArrayRef; // its shape, once (not per_frame)
  per_frame: boolean; // a point cache: its points per frame come in the chunks
  uv: { values: ArrayRef; indices: ArrayRef } | null;
}

export interface CharacterMeshRef {
  name: string;
  faces: ArrayRef;
  vertices: number;
  points: ArrayRef; // bind pose, in the world
  joint_indices?: ArrayRef; // `influences` per point (four, padded, when it is drawn on the graphics card)
  joint_weights?: ArrayRef;
  influences: number; // joints per point; more than four: per_frame, and the indices / weights (when sent) are for posing on the CPU (model/skinning.ts)
  per_frame?: boolean; // more than four joints per point: the server's evaluation per frame instead
  shapes: string[];
  shape_offsets?: ArrayRef; // [B,V,3]
  shape_weights?: ArrayRef; // [T,B]
  uv: { values: ArrayRef; indices: ArrayRef } | null;
}

export interface CharacterRef extends Item {
  person?: number; // the person it is (lab2shot:person_id: the id of the boxes it was solved from); absent: not one person
  joints: string[];
  parents: number[];
  bind: ArrayRef; // joint-to-world in the bind pose [J] 4x4
  anim: ArrayRef; // joint-to-world per sample [T,J] 3x4
  meshes: CharacterMeshRef[];
}

/** A cloud made from a depth map (深度转点云): its frames come as the depth map of every `step`-th pixel (`gw` × `gh`,
 * NaN where a pixel was dropped) and colour bytes, rebuilt through the camera of each sample (gridPoints). */
export interface GridRef {
  width: number; // the depth map's picture, pixels
  height: number;
  step: number;
  gw: number;
  gh: number;
  focal: ArrayRef; // fx in pixels, per sample
  aspect?: number; // the camera's pixel aspect (a pixel's width over its height): fy = fx x aspect; 1 when absent
  cam: ArrayRef; // camera-to-world per sample, 4x4 row after row
  principal?: ArrayRef; // (cx, cy) pixels per sample: the camera's principal point (the picture's centre unless the solver wrote one)
  // Proxy display (server/view_data.py _proxy_step): one cell in every `proxy`.
  // 1 (or absent): sent in full. Above 1 the view's notice area must say so; the cloud is never thinned silently.
  proxy?: number;
  // A distance map seen directly as a cloud (server/view_data.py _distance_grid_view): depths only, coloured by a ramp over
  // distance; `ramp` is the [near, far] value range. The ramp itself comes from the server (ramp_colour = [NEAR, SLOPE],
  // colour = clamp(NEAR + v × SLOPE); view_data.py RAMP_NEAR / RAMP_SLOPE is the one definition) and the shader follows it
  // (pointShaders.tsx): the page keeps no formula of its own
  ramp?: [number, number];
  ramp_colour?: [[number, number, number], [number, number, number]];
}

export interface CloudRef extends Item {
  gaussian?: boolean;
  sh_coefficients?: number;
  covariance?: ArrayRef;
  opacity?: ArrayRef;
  sh?: ArrayRef;
  width: number | null;
  per_frame: boolean;
  grid?: GridRef;
  count?: number; // per-frame cloud (points or gaussian): the first frame's splat count (what 「显示了 N / 共 M 点」 reports)
  // One point in every `every` (server/view_data.py _point_step for points, _gaussian_step for 3D 高斯): applied whenever
  // the cloud exceeds the 「点云上限」 / 「高斯显示上限」 setting. Absent: nothing dropped. Above 1 the view's notice
  // area keeps showing 「显示了 N / 共 M 点」
  every?: number;
  speed?: boolean; // the same points on every frame: they have a speed (server/view_data.py), 着色 · 速度 applies
  points?: ArrayRef;
  // float32, or bytes (k/255 within float32 precision). A per-frame cloud: the first frame's colours, sent once; a
  // frame whose colours are the same comes without them (server/view_data.py encode)
  colors?: ArrayRef;
  widths?: ArrayRef;
}

/** A set of 三维曲线 (hair, guide curves, a motion trail): how many points each curve has and its points, one curve
 * after another. `strands` / `count` of the first sample, so the viewer can tell before it draws whether it is within
 * the display budget (view/curves3d.tsx: over it, the box alone with the reason; never thinned). */
export interface CurveRef extends Item {
  width: number | null; // one width throughout, when they do not differ per point
  per_frame: boolean;
  strands: number;
  count: number; // points
  vertex_counts?: ArrayRef; // [C] points per curve
  points?: ArrayRef;
  colors?: ArrayRef; // float32, or bytes (k/255: the same numbers)
  widths?: ArrayRef; // a width per point, when they differ
}

export interface CameraRef extends Item {
  width: number;
  height: number;
  /** The camera's distortion (a lens model id; empty or absent when it has none). Looking through the camera, the
   * backdrop is undistorted with its lens (view/Stage3D.tsx, `through` in `transfer/frameKey.ts pictureUrl`): when the
   * solve chose a distorting lens model, the 3D view never uses the raw picture as its backdrop. */
  distortion?: string;
  /** The camera's plate: the fingerprint of the packet holding the pictures it was solved from (or the pictures assigned
   * to it by hand) (`lab2shot/io/usd.py PLATE`). Looking through the camera, the 3D stage uses it as the backdrop; empty
   * or absent: no plate (an imported camera). */
  plate?: string;
  focal_mm: ArrayRef;
  h_aperture_mm: ArrayRef;
  v_aperture_mm: ArrayRef;
  /** The lens centre (principal point) off the picture's centre, mm, +x right +y up, (x, y) per sample or one pair
   * for all (`lab2shot/data/camera.py center_mm`): a camera solved inside a crop of the plate has one. Looking through
   * the camera, the picture sits off the lens axis by the opposite amount (view/camera3d.tsx frustum, ImagePlane). */
  center_mm?: ArrayRef;
}


export interface ViewDescription {
  format: number;
  frames: number[];
  parts: Record<string, PartRef>;
  models: ModelRef[];
  characters: CharacterRef[];
  clouds: CloudRef[];
  curves: CurveRef[];
  cameras: CameraRef[];
}

/** One piece of a chunk: an item's array for a run of its samples. */
export interface Piece {
  item: string; // "models/0", "clouds/1", "characters/0/meshes/2"
  array: string;
  samples: [number, number];
  sizes: number[];
  t: TypeName;
  e: "" | "s" | "x";
  o: number;
}

const SIZE: Record<TypeName, number> = { f32: 4, u32: 4, u16: 2, u8: 1 };

/** Packed bytes back to their array. `xorSize`: the elements of one sample ("x": each sample was sent as its bits'
 * difference from the one before). */
function unpack(bytes: Uint8Array, t: TypeName, e: "" | "s" | "x", xorSize = 0): Typed {
  const n = bytes.length / SIZE[t];
  if (e === "") {
    const copy = bytes.slice().buffer;
    return t === "f32" ? new Float32Array(copy) : t === "u32" ? new Uint32Array(copy) : t === "u16" ? new Uint16Array(copy) : new Uint8Array(copy);
  }
  // byte planes: all first bytes, then all second ... (little-endian words)
  const out = new Uint8Array(n * 4);
  for (let b = 0; b < 4; b++) {
    const plane = bytes.subarray(b * n, (b + 1) * n);
    for (let i = 0; i < n; i++) out[i * 4 + b] = plane[i];
  }
  const bits = new Uint32Array(out.buffer);
  if (e === "x" && xorSize > 0) for (let i = xorSize; i < n; i++) bits[i] ^= bits[i - xorSize];
  return new Float32Array(out.buffer);
}

/** A base array from its part's bytes. */
export function baseArray(parts: Record<string, Uint8Array>, r: ArrayRef): Typed {
  const part = parts[r.part];
  if (!part) throw new Error(`part ${r.part} is not here`);
  return unpack(part.subarray(r.o, r.o + r.n * SIZE[r.t]), r.t, r.e);
}

/** A chunk: its pieces, each with its samples as arrays (views of the piece's one flat array, `flat`). */
export function readChunk(bytes: Uint8Array): { piece: Piece; flat: Typed; samples: Typed[] }[] {
  const n = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + n))) as { pieces: Piece[] };
  const body = bytes.subarray(4 + n);
  return head.pieces.map((p) => {
    const total = p.sizes.reduce((a, b) => a + b, 0);
    const flat = unpack(body.subarray(p.o, p.o + total * SIZE[p.t]), p.t, p.e, p.sizes[0]);
    const samples: Typed[] = [];
    let at = 0;
    for (const s of p.sizes) {
      samples.push(flat.subarray(at, at + s));
      at += s;
    }
    return { piece: p, flat, samples };
  });
}

/** A depth cloud's points, rebuilt from its depths as the server made them (server/view_data.py grid_points) and as
 * the graphics card does it (points3d.tsx GRID): float32 in the same order of operations (a pinhole at the camera's
 * principal point, else the picture's centre; pixel centres at +0.5, OpenCV to GL axes, then the camera; `cam`: 4x4 rows). The kept pixels (not NaN) in
 * row-major order: the cloud's own order. */
export function gridPoints(depth: Float32Array, gw: number, step: number, width: number, height: number, focal: number, cam: ArrayLike<number>, principal: ArrayLike<number> | null = null, aspect = 1): Float32Array {
  const f = Math.fround;
  let n = 0;
  for (let k = 0; k < depth.length; k++) if (depth[k] === depth[k]) n++;
  const out = new Float32Array(n * 3);
  const fx = f(focal);
  const fy = f(fx * f(aspect)); // a pixel that is not square: fy = fx x pixel aspect (server/view_data.py grid_points)
  // through the camera's principal point when it has one (server/view_data.py grid_points does the same; the cook too)
  const cx = principal ? f(principal[0]) : f(f(width) * 0.5);
  const cy = principal ? f(principal[1]) : f(f(height) * 0.5);
  const m = Array.from({ length: 12 }, (_, i) => f(cam[i]));
  let at = 0;
  for (let k = 0; k < depth.length; k++) {
    const z = depth[k];
    if (z !== z) continue;
    const c = f(f(k % gw) * step);
    const r = f(f(Math.floor(k / gw)) * step);
    const x = f(f(f(f(c + 0.5) - cx) / fx) * z);
    const y = f(f(f(f(r + 0.5) - cy) / fy) * z);
    for (let row = 0; row < 3; row++)
      out[at++] = f(f(f(f(m[row * 4] * x) + f(m[row * 4 + 1] * -y)) + f(m[row * 4 + 2] * -z)) + m[row * 4 + 3]);
  }
  return out;
}

/** Frames as they come: runs of consecutive frames, [first, count] each (thousands of props need not spell out 250
 * frames each), spelt out. */
function expandRuns(runs: [number, number][]): number[] {
  const out: number[] = [];
  for (const [first, n] of runs) for (let k = 0; k < n; k++) out.push(first + k);
  return out;
}

/** A description as the server sends it (every item's frames as runs) as the viewer reads it. */
export function readDescription(wire: unknown): ViewDescription {
  // a list the server did not send counts as empty (filled in here only: everywhere else relies on it being there)
  const w = (wire ?? {}) as Partial<ViewDescription>;
  const d: ViewDescription = {
    ...w, format: w.format ?? 0, frames: w.frames ?? [], parts: w.parts ?? {},
    models: w.models ?? [], characters: w.characters ?? [], clouds: w.clouds ?? [], curves: w.curves ?? [], cameras: w.cameras ?? [],
  };
  for (const items of [d.models, d.characters, d.clouds, d.curves, d.cameras])
    for (const it of items as Item[]) it.frames = expandRuns((it.frames ?? []) as unknown as [number, number][]);
  return d;
}


/** The sample to draw while frame `i` has not arrived: the one in `samples` nearest to `i`; undefined when there is none.
 *
 * Drawing 0 points for a frame not yet arrived would make the picture flicker while the timeline is scrubbed, settling
 * only once everything is cached. Yet another frame held on screen is not the current frame either, so drawing and
 * telling are kept apart: this only keeps the picture from going empty, and the frame actually shown is stated by the
 * one view notice area (ui/ViewNotices.tsx).
 *
 * The nearest, not the previous one: scrubbing backwards, the previous sample can be much farther than the next. The
 * nearest by model/timelineMath.ts nearest's rule (`closer`), what arrived here being a picture's frames, not samples. */
export function heldSample<T>(samples: Map<number, T>, i: number): { sample: T; at: number } | undefined {
  const here = samples.get(i);
  if (here !== undefined) return { sample: here, at: i };
  let at: number | undefined;
  for (const k of samples.keys()) if (at === undefined || closer(k, at, i)) at = k; // one pass, nearest's own rule
  return at === undefined ? undefined : { sample: samples.get(at)!, at };
}

/** Row-major 4x4 (or 3x4 with 0 0 0 1 below) at `i` as column-major elements (three.js's Matrix4.elements). */
export function columnMajor(rows: Float32Array, i: number, height: 3 | 4 = 4): number[] {
  if (i < 0) return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]; // no sample (sampleAt -1): no transform, never a row read from the end
  const r = rows.subarray(i * height * 4, (i + 1) * height * 4);
  const at = (row: number, col: number) => (row < height ? r[row * 4 + col] : row === 3 && col === 3 ? 1 : 0);
  const out: number[] = [];
  for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) out.push(at(row, col));
  return out;
}

/** A skeleton's bones on one sample: from each joint to its parent, [x0 y0 z0 x1 y1 z1] per bone. */
export function boneSegments(anim: Float32Array, parents: number[], sample: number): Float32Array {
  if (sample < 0) return new Float32Array(0); // no sample (sampleAt -1): no bones
  const joints = parents.length;
  const bones = parents.filter((p) => p >= 0).length;
  const out = new Float32Array(bones * 6);
  const at = (j: number, c: number) => anim[(sample * joints + j) * 12 + c * 4 + 3]; // the translation column
  let k = 0;
  parents.forEach((p, j) => {
    if (p < 0) return;
    for (let c = 0; c < 3; c++) {
      out[k * 6 + c] = at(j, c);
      out[k * 6 + 3 + c] = at(p, c);
    }
    k++;
  });
  return out;
}

/** UVs per corner of the triangles (the server says which UV each triangle corner takes): [u v] per corner. */
export function cornerUvs(values: Float32Array, indices: Typed): Float32Array {
  const out = new Float32Array(indices.length * 2);
  for (let i = 0; i < indices.length; i++) {
    out[i * 2] = values[indices[i] * 2];
    out[i * 2 + 1] = values[indices[i] * 2 + 1];
  }
  return out;
}

/** Bytes a view needs held in the browser: the base parts' raw sizes. */
export const baseBytes = (d: ViewDescription) =>
  Object.entries(d.parts)
    .filter(([name]) => !name.startsWith("c") && name !== "uv")
    .reduce((a, [, p]) => a + (p.raw ?? 0), 0);

/** How far a skinned mesh's points reach from the joint that carries each (the heaviest of its four influences:
 * `skinIndex` / `skinWeight` packed four per point, in no promised order — a USD skin's influences come unsorted, and
 * padding is index 0 at weight 0), in the bind pose: the largest such distance. A point turns and moves with that joint,
 * keeping its distance, so the joints' box widened this much holds the mesh in any pose of rigid joints
 * (view/elements3d.tsx SkinnedBody bounds). */
export function skinReach(points: ArrayLike<number>, skinIndex: ArrayLike<number>, skinWeight: ArrayLike<number>, jointAt: readonly (readonly number[])[]): number {
  if (!jointAt.length) return 0;
  let reach = 0;
  for (let v = 0; v * 3 < points.length; v++) {
    let c = v * 4;
    for (let k = v * 4 + 1; k < v * 4 + 4; k++) if (skinWeight[k] > skinWeight[c]) c = k;
    const q = jointAt[skinIndex[c]] ?? jointAt[0];
    reach = Math.max(reach, Math.hypot(points[v * 3] - q[0], points[v * 3 + 1] - q[1], points[v * 3 + 2] - q[2]));
  }
  return reach;
}
