import type { Bounds } from "../model/math3d";
import type { Piece, Typed } from "../model/viewFormat";
import { workerAsks } from "../platform/work";
import type { ChunkAnswer } from "./sceneWorker";

/** The scene worker's client (view/sceneWorker.ts: the 3D viewer's arithmetic off the page's thread). One worker
 * for the whole page, started by the first request (platform/work.ts). Every request carries its arrays as themselves,
 * i.e. exactly their own bytes even when they are views into a larger buffer (a sample of a chunk, one camera of a
 * track); arrays are transferred when the page no longer needs them and copied otherwise, and the response's arrays are
 * transferred back. `parseChunk` returns what model/viewFormat.ts readChunk returned (the pieces with their samples as
 * views of one flat array each), plus the bounds of every explicit cloud sample computed while the worker held them. A
 * worker that failed or crashed rejects (E-THREAD-FAILED, E-THREAD-CRASHED). */

const ask = workerAsks<object>(() => new Worker(new URL("./sceneWorker.ts", import.meta.url), { type: "module" }));

interface Parsed {
  pieces: { piece: Piece; samples: Typed[] }[];
  bounds: Record<string, Bounds | null>; // "<cloud index>|<sample>" -> the extent of its points
}

/** A chunk's bytes as its pieces with their samples, plus the bounds of every explicit cloud sample in it. The bytes
 * are transferred (the page no longer needs them). */
export async function parseChunk(bytes: Uint8Array): Promise<Parsed> {
  const r = (await ask({ chunk: bytes }, [bytes])) as ChunkAnswer;
  return {
    pieces: r.pieces.map(({ piece, flat }) => {
      const samples: Typed[] = [];
      let at = 0;
      for (const s of piece.sizes) {
        samples.push(flat.subarray(at, at + s));
        at += s;
      }
      return { piece, samples };
    }),
    bounds: r.bounds,
  };
}

/** A depth cloud's points reconstructed from its depths (gridPoints, exactly the arithmetic the GPU performs) and their
 * extent. The depths stay with the page (its texture) and the camera is one sample of the track: both are sent as
 * copies of exactly their bytes. */
export async function gridPointsOf(depth: Float32Array, gw: number, step: number, width: number, height: number, focal: number, cam: Float32Array, principal: Float32Array | null = null): Promise<{ points: Float32Array; bounds: Bounds | null }> {
  return (await ask({ grid: { depth, gw, step, width, height, focal, cam, principal } })) as { points: Float32Array; bounds: Bounds | null };
}

/** The extent of a static cloud's points (the samples of a chunk were measured while being parsed). */
export async function boundsInWorker(points: Float32Array): Promise<Bounds | null> {
  return ((await ask({ bounds: { points } })) as { bounds: Bounds | null }).bounds;
}
