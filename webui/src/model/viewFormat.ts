/** The 3D viewer's data as the server sends it (lab2shot/server/view_data.py): its description, how its arrays are
 * packed (losslessly: float32 bytes sorted by significance, point caches as bit differences from the frame before,
 * gzip on the wire), and the skinning the graphics card does, written out here for the parts of the viewer that need
 * skinned points on the CPU (normals, bounds) and for the tests (webui/tests/tools/view_check.ts).
 *
 * Pure: no imports, so node's own test runner reads it as it is. */

export type TypeName = "f32" | "u32" | "u16" | "u8";
export type Typed = Float32Array | Uint32Array | Uint16Array | Uint8Array;

/** A base array: in which part, where, how many elements, of what type, how packed. */
export interface ArrayRef {
  part: string;
  o: number;
  n: number;
  t: TypeName;
  e: "" | "s" | "x";
}

export interface PartRef {
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
  joint_indices?: ArrayRef; // four per point
  joint_weights?: ArrayRef;
  influences: number;
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
  focal: ArrayRef; // pixels, per sample
  cam: ArrayRef; // camera-to-world per sample, 4x4 row after row
  principal?: ArrayRef; // (cx, cy) pixels per sample: the camera's principal point (the picture's centre unless the solver wrote one)
  // 代理显示（server/view_data.py _proxy_step）：每 proxy 个格子取一个。
  // 1（或缺失）表示完整发送。大于 1 时视图通知区必须明确提示，不得静默抽稀。
  proxy?: number;
}

export interface CloudRef extends Item {
  width: number | null;
  per_frame: boolean;
  grid?: GridRef;
  count?: number; // 逐帧点云：第一帧的点数（「显示了 N / 共 M 点」据此显示）
  // 每 every 个点取一个（server/view_data.py _point_step）：超过后台「点云上限」时删减，且始终生效。
  // 缺少该项表示不删减任何点。大于 1 时视图通知区持续显示「显示了 N / 共 M 点」
  every?: number;
  speed?: boolean; // the same points on every frame: they have a speed (server/view_data.py), 着色 · 速度 applies
  points?: ArrayRef;
  colors?: ArrayRef; // float32, or bytes (k/255: the same numbers)
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
  /** 该相机带有的畸变（镜头模型 id；不带畸变时为空或缺失）。透过该相机观看时，底图按其镜头去畸变（view/Stage3D.tsx、
   * `api.packetFrameUrl` 的 `through`）：解算节点选择了带畸变的镜头模型时，三维视图不使用原图作为底图。 */
  distortion?: string;
  /** 该相机的背板：解算该相机所用的画面（或手动为其指定的画面）对应数据包的指纹（`lab2shot/io/usd.py PLATE`）。
   * 透过该相机观看时，三维舞台以此为底图；为空或缺失表示没有背板（导入的相机、旧版本解算的相机）。 */
  plate?: string;
  focal_mm: ArrayRef;
  h_aperture_mm: ArrayRef;
  v_aperture_mm: ArrayRef;
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
export function unpack(bytes: Uint8Array, t: TypeName, e: "" | "s" | "x", xorSize = 0): Typed {
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
 * the graphics card does it (points3d.tsx GRID): float32 in the same order of operations (a pinhole at the picture's
 * centre, pixel centres at +0.5, OpenCV to GL axes, then the camera; `cam`: 4x4 rows). The kept pixels (not NaN) in
 * row-major order: the cloud's own order. */
export function gridPoints(depth: Float32Array, gw: number, step: number, width: number, height: number, focal: number, cam: ArrayLike<number>, principal: ArrayLike<number> | null = null): Float32Array {
  const f = Math.fround;
  let n = 0;
  for (let k = 0; k < depth.length; k++) if (depth[k] === depth[k]) n++;
  const out = new Float32Array(n * 3);
  const fx = f(focal);
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
    const y = f(f(f(f(r + 0.5) - cy) / fx) * z);
    for (let row = 0; row < 3; row++)
      out[at++] = f(f(f(f(m[row * 4] * x) + f(m[row * 4 + 1] * -y)) + f(m[row * 4 + 2] * -z)) + m[row * 4 + 3]);
  }
  return out;
}

/** Frames as they come: runs of consecutive frames, [first, count] each (thousands of props need not spell out 250
 * frames each), spelt out. */
export function expandRuns(runs: [number, number][]): number[] {
  const out: number[] = [];
  for (const [first, n] of runs) for (let k = 0; k < n; k++) out.push(first + k);
  return out;
}

/** A description as the server sends it (every item's frames as runs) as the viewer reads it. */
export function readDescription(wire: unknown): ViewDescription {
  const d = wire as ViewDescription;
  for (const items of [d.models, d.characters, d.clouds, d.curves, d.cameras])
    for (const it of items as Item[]) it.frames = expandRuns(it.frames as unknown as [number, number][]);
  return d;
}

/** The sample of `frame` among an item's `frames`: the frame itself, else the nearest end (a still thing holds
 * everywhere). */
export const sampleAt = (frames: number[], frame: number) => {
  const i = frames.indexOf(frame);
  if (i >= 0) return i;
  return frames.length ? (frame < frames[0] ? 0 : frames.length - 1) : 0;
};

/** 返回该帧尚未到达时应绘制的帧：`samples` 中与 `i` 最近的一份，没有任何一份时返回 undefined。
 *
 * 若该帧未到达时绘制 0 个点，拖动时间线时画面会持续闪烁，只有全部缓存后才稳定。但保持显示其他帧也并非当前帧，
 * 因此绘制与提示分开处理：此处只保证画面不为空，实际显示的帧号由统一的视图通知区说明
 * （ui/ViewNotices.tsx）。
 *
 * 取最近的一份而非前一份：向回拖动时，前一份可能比后一份远得多。 */
export function heldSample<T>(samples: Map<number, T>, i: number): { sample: T; at: number } | undefined {
  const here = samples.get(i);
  if (here !== undefined) return { sample: here, at: i };
  let best: { sample: T; at: number } | undefined;
  for (const [k, v] of samples) if (best === undefined || Math.abs(k - i) < Math.abs(best.at - i)) best = { sample: v, at: k };
  return best;
}

/** Row-major 4x4 (or 3x4 with 0 0 0 1 below) at `i` as column-major elements (three.js's Matrix4.elements). */
export function columnMajor(rows: Float32Array, i: number, height: 3 | 4 = 4): number[] {
  const r = rows.subarray(i * height * 4, (i + 1) * height * 4);
  const at = (row: number, col: number) => (row < height ? r[row * 4 + col] : row === 3 && col === 3 ? 1 : 0);
  const out: number[] = [];
  for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) out.push(at(row, col));
  return out;
}

/** 4x4 (column-major) product a · b. */
export function multiply(a: number[], b: number[]): number[] {
  const out = new Array<number>(16).fill(0);
  for (let col = 0; col < 4; col++)
    for (let row = 0; row < 4; row++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + row] * b[col * 4 + k];
      out[col * 4 + row] = s;
    }
  return out;
}

/** 4x4 (column-major) inverse, of an affine matrix. */
export function invertAffine(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[4], m[8], m[1], m[5], m[9], m[2], m[6], m[10]];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  const inv = [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det, (f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det, (d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det];
  const t = [m[12], m[13], m[14]];
  const out = [inv[0], inv[3], inv[6], 0, inv[1], inv[4], inv[7], 0, inv[2], inv[5], inv[8], 0, 0, 0, 0, 1];
  for (let row = 0; row < 3; row++) out[12 + row] = -(inv[row * 3] * t[0] + inv[row * 3 + 1] * t[1] + inv[row * 3 + 2] * t[2]);
  return out;
}

/** The skinning matrices of one sample, as the graphics card gets them (float32, column-major, joint after joint):
 * each joint's transform on the frame times the inverse of its bind transform. */
export function skinningMatrices(bind: Float32Array, anim: Float32Array, sample: number, joints: number): Float32Array {
  const out = new Float32Array(joints * 16);
  for (let j = 0; j < joints; j++) {
    const m = multiply(columnMajor(anim, sample * joints + j, 3), invertAffine(columnMajor(bind, j, 4)));
    out.set(m, j * 16);
  }
  return out;
}

const f = Math.fround;

/** Skinned points of one sample in float32 arithmetic, as the graphics card does it: blend shapes added to the bind
 * points (their weights on the sample), then the four joints' skinning matrices, weighted. */
export function skinPoints(
  points: Float32Array,
  jointIndices: Typed,
  jointWeights: Float32Array,
  matrices: Float32Array,
  shapeOffsets?: Float32Array,
  shapeWeights?: ArrayLike<number>,
): Float32Array {
  const v = points.length / 3;
  const out = new Float32Array(points.length);
  const shapes = shapeWeights ? shapeWeights.length : 0;
  for (let i = 0; i < v; i++) {
    let x = points[i * 3];
    let y = points[i * 3 + 1];
    let z = points[i * 3 + 2];
    for (let b = 0; b < shapes; b++) {
      const w = shapeWeights![b];
      if (!w) continue;
      const o = (b * v + i) * 3;
      x = f(x + f(shapeOffsets![o] * w));
      y = f(y + f(shapeOffsets![o + 1] * w));
      z = f(z + f(shapeOffsets![o + 2] * w));
    }
    let sx = 0;
    let sy = 0;
    let sz = 0;
    for (let k = 0; k < 4; k++) {
      const w = jointWeights[i * 4 + k];
      if (!w) continue;
      const m = jointIndices[i * 4 + k] * 16;
      const px = f(f(f(matrices[m] * x) + f(matrices[m + 4] * y)) + f(f(matrices[m + 8] * z) + matrices[m + 12]));
      const py = f(f(f(matrices[m + 1] * x) + f(matrices[m + 5] * y)) + f(f(matrices[m + 9] * z) + matrices[m + 13]));
      const pz = f(f(f(matrices[m + 2] * x) + f(matrices[m + 6] * y)) + f(f(matrices[m + 10] * z) + matrices[m + 14]));
      sx = f(sx + f(px * w));
      sy = f(sy + f(py * w));
      sz = f(sz + f(pz * w));
    }
    out[i * 3] = sx;
    out[i * 3 + 1] = sy;
    out[i * 3 + 2] = sz;
  }
  return out;
}

/** A skeleton's bones on one sample: from each joint to its parent, [x0 y0 z0 x1 y1 z1] per bone. */
export function boneSegments(anim: Float32Array, parents: number[], sample: number): Float32Array {
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
