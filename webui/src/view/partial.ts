import { useEffect, useState } from "react";
import { json } from "../platform/http";
import { startPolling } from "../platform/poll";
import { dropSource, urlFrames, type FrameSource } from "../transfer/frames";
import { useResults } from "../state/results";
import { blocksOf, chainOf } from "../state/items";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";

/** Live preview ("边算边看"): while a node is being cooked and the viewer shows it, the frames already written are
 * displayed and the view fills in as the cook progresses.
 *
 * Rules:
 * - Applies only to a node whose type declares its frames final as written (`streams`, nodes/families/base.py), that
 *   is not inside a "逐项处理" block (per-item results have no partial), while a job of this page is running and the
 *   node is the one being cooked (`node_start`/`node_done` maintain the per-node status in state/results.ts).
 *   Nothing is requested otherwise, nor before the node reports its first written frame (`progress`), so no request
 *   is sent for a partial that cannot exist yet.
 * - `GET /api/jobs/{job}/partial/{node}/{port}` is polled every 0.5 s, uncached (the server responds with no-store):
 *   the frames written so far, the expected total, the size and the type.
 * - Each frame is fetched from `…/frame/{n}.png`, also uncached by the browser, since the unfinished packet has no
 *   content address. A frame file is written only once, so the page may cache it under the job's key while the cook
 *   runs; all entries under that key are dropped as soon as the node is done.
 * - Once the node is done (`node_done`, or the status reply marks the output `present`), the view switches to the
 *   content address `/api/packet/{fp}/frame/{n}.png`, which the browser caches permanently. */

export const PARTIAL_EVERY = 500; // ms between asks while a node is being cooked

export interface Partial {
  frames_done: number[]; // the frames written so far, in order
  total: number; // how many there will be
  width: number;
  height: number;
  type: string;
}

const url = (job: string, node: string, port: string) => `/api/jobs/${job}/partial/${encodeURIComponent(node)}/${encodeURIComponent(port)}`;

/** Cache id for a running node's frames, built from the job, node and port. The packet fingerprint is not used, as
 * it addresses the finished result. */
export const partialId = (job: string, node: string, port: string) => `partial:${job}:${node}:${port}`;

/** The frames of a partial result, exposed as a regular frame source for the 2D stage. */
export function partialFrames(job: string, node: string, port: string, frames: number[]): FrameSource {
  return urlFrames(partialId(job, node, port), frames, (f) => `${url(job, node, port)}/frame/${f}.png`);
}

/** The partial result shown for `node`.`port` while it is being cooked, or null when not running, not this node,
 * nothing written yet, or the finished result is available (served from its content address). */
export function usePartial(nodeId: string | null, port: string, cooked: boolean): { info: Partial; source: FrameSource } | null {
  const job = useResults((s) => s.job);
  const status = useResults((s) => (nodeId ? s.byNode[nodeId]?.status : undefined));
  const reply = useResults((s) => s.reply);
  const typeId = useCookInputs((s) => (nodeId ? s.nodes[nodeId]?.typeId : undefined));
  // Progress has a single source (state/results.ts `now`, api/progress.ts). Only its `done` count is used here, as a
  // gate on whether the node has written its first frame; this avoids requesting a partial that cannot exist yet,
  // which the browser would log as an error. It is not used to draw progress (that is the role of `at`).
  const done = useResults((s) => (nodeId && s.now?.node === nodeId ? s.now.done : 0));
  const [found, setFound] = useState<{ id: string; info: Partial } | null>(null);
  // The node type writes final frames incrementally and the node is not inside a block (no per-item partial).
  const shows = !!typeId && !!getNodeDefs()[typeId]?.streams && !chainOf(blocksOf(reply), nodeId ?? "").length;
  const running = shows && !!job && !!nodeId && !!port && !cooked && status === "cooking" && done > 0;
  const id = running ? partialId(job!.id, nodeId!, port) : "";
  useEffect(() => {
    if (!running) {
      setFound(null);
      return;
    }
    const ask = async () => {
      // Uncached: the frame set changes while the node writes (the server also responds with no-store).
      const info = await json<Partial>("GET", url(job!.id, nodeId!, port), undefined, { cache: "no-store" }).catch(() => null);
      setFound(info && info.frames_done.length ? { id, info } : null);
    };
    const poll = startPolling({ read: ask, every: PARTIAL_EVERY });
    return () => {
      poll.stop();
      dropSource(id); // drop the provisional frames; the finished result is fetched by its own address
      setFound(null);
    };
  }, [running, id, job?.id, nodeId, port]);
  if (!running || !found || found.id !== id) return null;
  return { info: found.info, source: partialFrames(job!.id, nodeId!, port, found.info.frames_done) };
}
