/** What each 「输出」 of the open graph last packed (lab2shot/transfer/outputs.py): its own cook collected the files
 * wired into it into its task's folder and packed them into one zip; only then is there something to download
 * (computing the nodes before it only puts their results on the server). Kept in state/results.ts `outputs` by
 * (graph, node), so an output is only ever drawn on the graph it was cooked from. Two ways it arrives: the "output"
 * event of the job this page follows (graph/follow.ts), and, when a graph is opened or the page comes back, the
 * server's own list of the account's outputs (GET /api/outputs), the newest per node. */

import { api } from "../api";
import type { Output } from "../api/files";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";

/** A finished output arrived (its task's "output" event): drawn on its node when it is this graph's. */
export function noteOutput(o: Output): void {
  const graphId = o.graph ?? "";
  if (graphId && graphId === useCookInputs.getState().graphId) useResults.getState().setOutput(graphId, o.node, o);
}

/** The newest output of every 「输出」 of the open graph still kept on the server (its task not yet gone). */
export async function loadOutputs(): Promise<void> {
  const graphId = useCookInputs.getState().graphId;
  if (!graphId) return;
  const all = await api.outputs().catch(() => [] as Output[]);
  if (useCookInputs.getState().graphId !== graphId) return; // another graph was opened meanwhile
  const newest = new Map<string, Output>();
  for (const o of all) {
    if ((o.graph ?? "") !== graphId) continue;
    const had = newest.get(o.node);
    if (!had || (o.finished ?? 0) > (had.finished ?? 0)) newest.set(o.node, o);
  }
  for (const [node, o] of newest) useResults.getState().setOutput(graphId, node, o);
}
