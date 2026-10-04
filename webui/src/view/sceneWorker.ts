/// <reference lib="webworker" />
import { readChunk, gridPoints, type Piece, type Typed } from "../model/viewFormat";
import { spanBounds, type Bounds } from "../model/math3d";
import { workerAnswers } from "../platform/work";

/** The 3D viewer's arithmetic, off the page's thread: parsing a chunk's bytes (readChunk — the byte-plane and
 * bit-difference unpacking, the biggest plain loops in the viewer), rebuilding a depth cloud's points (gridPoints,
 * exactly the arithmetic the graphics card does) and a cloud's bounds. It asks nothing and keeps nothing:
 * every ask carries its own arrays in, every answer carries its arrays out (platform/work.ts: each array exactly its
 * own bytes, whatever larger buffer it was a view of), so the page holds the one copy the graphics card draws. Called
 * only through view/sceneWork.ts, so these functions never run on the page's thread. */

type SceneAsk =
  | { chunk: Uint8Array } // a chunk's bytes as they came
  | { grid: { depth: Float32Array; gw: number; step: number; width: number; height: number; focal: number; cam: Float32Array; principal: Float32Array | null; aspect: number } }
  | { bounds: { points: Float32Array } }; // a still cloud's points

export interface ChunkAnswer {
  pieces: { piece: Piece; flat: Typed }[];
  bounds: Record<string, Bounds | null>;
}

workerAnswers<SceneAsk>((ask) => {
  if ("chunk" in ask) {
    const bounds: Record<string, Bounds | null> = {}; // an explicit cloud sample's points, spanned while they are here
    const pieces = readChunk(ask.chunk).map(({ piece, flat, samples }) => {
      const [kind, i] = piece.item.split("/");
      if (kind === "clouds" && piece.array === "points")
        samples.forEach((arr, n) => (bounds[`${i}|${piece.samples[0] + n}`] = spanBounds(arr as Float32Array, 0, arr.length / 3, 3)));
      return { piece: piece, flat };
    });
    return { pieces, bounds: bounds };
  }
  if ("grid" in ask) {
    const g = ask.grid;
    const points = gridPoints(g.depth, g.gw, g.step, g.width, g.height, g.focal, g.cam, g.principal, g.aspect);
    return { points, bounds: spanBounds(points, 0, points.length / 3, 3) };
  }
  const points = ask.bounds.points;
  return { bounds: spanBounds(points, 0, points.length / 3, 3) };
});
