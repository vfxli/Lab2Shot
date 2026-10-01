/** 边算边传: what a node_done event does on the page: write the fresh fingerprints into results (when the event
 * still applies to this graph and the cook-inputs version it was submitted at), else ask the server once, and put the
 * node's just-cooked outputs into the prefetch queue. */

import type { CookEvent } from "../api";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useResults } from "../state/results";
import { useViewer } from "../state/viewer";
import { wantPrefetch } from "../transfer/prefetchLive";
import { setGens } from "../transfer/gens";
import type { PrefetchKind } from "../transfer/prefetch";
import { upstream } from "./nodes";
import { channelsOf } from "../model/view2d";

/** Which output kind the prefetcher fetches (2D only; scene3d / points / curves are not prefetched): everything
 * that is frames of pixels, i.e. a video, and the one image family (image, image.1 … image.4: 二维数据只以通道数区分). */
const prefetchKind = (type: string): PrefetchKind | null =>
  type === "video" || type === "image" || channelsOf(type) > 0 ? "frames2d" : null;

/** A node finished cooking. If the event still answers this page (same graph, and the cook-inputs version the job was
 * submitted at is still the page's) and carries the
 * fresh fingerprints, they go straight into `results`; otherwise the server is asked once. What is
 * shown now switches by itself (the plan re-reads results); the other outputs go to the prefetch queue. */
export function onNodeDone(e: CookEvent, node: string, refresh: () => void): void {
  // the new packets' generations are noted first (the same source as the status reply: a packet's commit time), so the
  // keys and addresses that follow (description, whole-range fetch, prefetch) carry them from the start rather than after
  // the status reply at the end of the job — until then a recook of the same fingerprint would hit old frames under
  // the old keys, and the new packet would be fetched once for nothing under keys without a generation
  if (e.outputs && e.gens) setGens(Object.fromEntries(Object.entries(e.outputs).flatMap(([port, fp]) => (e.gens![port] ? [[fp, e.gens![port]]] : []))));
  const ci = useCookInputs.getState();
  const sameGraph = (e.graph ?? "") === ci.graphId;
  const trusted = e.version === ci.version; // the page is still the version it submitted: nothing was edited since
  if (sameGraph && trusted && e.outputs) {
    useResults.getState().applyNodeDone(node, e.outputs);
  } else {
    refresh(); // the next status reply picks up the fresh fingerprints
  }
  schedulePrefetch(node, e.outputs);
}

/** Prefetch the node's just-cooked output packets: the shown node (0) and its upstream (1) before the rest (2). */
function schedulePrefetch(node: string, outputs: Record<string, string> | undefined): void {
  if (!outputs) return;
  const displayId = useLook.getState().displayId;
  const chain = displayId ? upstream(displayId, useCookInputs.getState().edges) : [];
  const ports = useResults.getState().results[node]?.ports?.outputs ?? [];
  const around = useViewer.getState().frame;
  for (const [port, fp] of Object.entries(outputs)) {
    const kind = prefetchKind(ports.find((p) => p.name === port)?.type ?? "");
    if (!kind) continue;
    const priority = node === displayId ? 0 : chain.includes(node) ? 1 : 2;
    wantPrefetch({ node, port, fp, kind, priority, around });
  }
}
