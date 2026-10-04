/** Which nodes' shown answer may be out of date (state/results.ts Shown `pending`): the one rule, pure, so it is tested
 * on its own (model/pending.test.ts). */

import { same, type Json } from "./graphPatch.ts";

interface NodeLike { typeId: string; params: Record<string, unknown> }
interface WireLike { source: string; sourceHandle: string; target: string; targetHandle: string }

/** The cook inputs a trusted reply answered (state/cookInputs.ts's own objects: an unchanged node keeps its object). */
export interface Answered {
  nodes: Record<string, NodeLike>;
  edges: WireLike[];
  cookRange: [string, string] | null;
}

/** The current cook inputs, as much of state/cookInputs.ts as the rule reads. */
export interface PendingInputs extends Answered {
  version: number;
  order: string[];
}

const NOTHING: ReadonlySet<string> = new Set();

const wiresInto = (edges: readonly WireLike[], id: string): string =>
  edges.filter((e) => e.target === id).map((e) => `${e.source}.${e.sourceHandle}>${e.targetHandle}`).sort().join("|");

/** None while the reply answers the current cook inputs (`forCookInputs === version`), or before any trusted answer
 * (nothing is shown to be out of date); else each node changed since the last trusted answer (its type, parameters,
 * picked files, or the wires into it) and everything downstream of one. Another frame range: all of them. */
export function pendingNodes(ci: PendingInputs, forCookInputs: number, answered: Answered | null): ReadonlySet<string> {
  if (forCookInputs === ci.version || !answered) return NOTHING;
  if ((answered.cookRange ?? []).join("|") !== (ci.cookRange ?? []).join("|")) return new Set(ci.order);
  const changed: string[] = [];
  for (const id of ci.order) {
    const was = answered.nodes[id];
    const now = ci.nodes[id];
    if (!was || (was !== now && !same(was as unknown as Json, now as unknown as Json))
        || (answered.edges !== ci.edges && wiresInto(answered.edges, id) !== wiresInto(ci.edges, id))) changed.push(id);
  }
  const out = new Set(changed);
  for (let i = 0; i < changed.length; i++)
    for (const e of ci.edges) if (e.source === changed[i] && !out.has(e.target)) (out.add(e.target), changed.push(e.target));
  return out;
}
