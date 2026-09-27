// The 3D scene's data as the browser holds it: clouds, grids, models, characters, curves, cameras. Re-exported
// by view/sceneData.ts.

import type { Bounds } from "../model/math3d";
import type { GridRef, CameraRef, CharacterMeshRef, CharacterRef, CloudRef, CurveRef, ModelRef, Typed } from "../model/viewFormat";

export interface CloudSample {
  points: Float32Array | null; // for a depth cloud: null until requested (bounds, picking); the worker reconstructs them (sceneData)
  colors: Float32Array | Uint8Array | Uint16Array; // bytes: k/255, words: k/65535, the same numbers
  widths: Float32Array | null; // a size per point, when they differ
  count: number;
  grid: GridSample | null; // made from a depth map: drawn from its depths by the graphics card
  bounds: Bounds | null; // the extent of its points, computed by the worker: null until requested and answered (or when there are no points)
}

/** One frame of a depth cloud as received: the depths of every `step`-th pixel (NaN: dropped), their colours (or a
 * single colour), and the camera of that frame.
 *
 * `gw` / `gh` / `step` belong to this frame, not to the cloud: a frame over the 点云上限 has points dropped
 * (server/view_data.py drop_grid), so it is a smaller grid of the same picture. Frames of both qualities coexist in
 * the same cloud, so switching back redraws the full frames without fetching them again. */
export interface GridSample {
  ref: GridRef;
  gw: number;
  gh: number;
  step: number;
  depth: Float32Array;
  colors: Float32Array | Uint8Array | Uint16Array | null; // one per kept pixel, in row-major order
  tint: Float32Array | null;
  focal: number;
  cam: Float32Array; // 4x4 rows
  principal: Float32Array | null; // (cx, cy) pixels, or null: the picture's centre
}

export interface ModelData {
  ref: ModelRef;
  faces: Typed;
  world: Float32Array; // 4x4 rows per sample
  points: Float32Array | null; // its shape (still), in its own space
  samples: Map<number, Float32Array>; // a point cache's shapes as they arrive
  uv: Float32Array | null; // per triangle corner, once asked for
}

export interface CharacterMeshData {
  ref: CharacterMeshRef;
  faces: Typed;
  points: Float32Array; // bind pose, in the world
  jointIndices: Typed | null;
  jointWeights: Float32Array | null;
  shapeOffsets: Float32Array | null;
  shapeWeights: Float32Array | null;
  samples: Map<number, Float32Array>; // evaluated by the server (more than four joints a point), as they arrive
  uv: Float32Array | null;
}

export interface CharacterData {
  ref: CharacterRef;
  bind: Float32Array; // [J] 4x4 rows
  anim: Float32Array; // [T,J] 3x4 rows
  meshes: CharacterMeshData[];
}

export interface CloudData {
  ref: CloudRef;
  key: string;
  name: string;
  frames: number[];
  world: Float32Array;
  width: number | null;
  still: CloudSample | null;
  samples: Map<number, CloudSample>;
  focal: Float32Array | null; // a depth cloud's camera per sample
  cams: Float32Array | null;
  principals: Float32Array | null; // (cx, cy) per sample, when the camera wrote a principal point
  scene: Scene; // the view owning this cloud: the target for requests of worker-derived data (bounds, speeds, grid points)
}

/** One sample of a set of 三维曲线: how many points each curve has, its points one curve after another, their
 * colours and, when they differ, a width per point. `segments` is the drawing form (pairs of points), made once per
 * sample and kept with it. */
export interface CurveSample {
  vertexCounts: Typed;
  points: Float32Array;
  colors: Float32Array | Uint8Array | Uint16Array;
  widths: Float32Array | null;
  count: number; // points
  strands: number;
}

export interface CurveData {
  ref: CurveRef;
  key: string;
  name: string;
  frames: number[];
  world: Float32Array;
  width: number | null;
  still: CurveSample | null;
  samples: Map<number, CurveSample>;
}

export interface CameraData {
  ref: CameraRef;
  world: Float32Array;
  focalMm: Float32Array;
  hAperture: Float32Array;
  vAperture: Float32Array;
}

/** The view a cloud belongs to (view/sceneData.ts): the target for requests of worker-derived data. */
export interface Scene {
  wantBounds(sample: CloudSample): void;
}
