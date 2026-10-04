import { useMemo, useRef } from "react";
import type { Node } from "@xyflow/react";
import type { PortDef, ResolvedCost } from "../api";
import { getNodeDefs, useCatalog } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useResults, useShownResults } from "../state/results";
import type { Availability } from "../api/applies";
import type { GraphState } from "../state/graph";
import { commercialOf, costOf, inputsOf, paramPortNames, pendingPorts, wiredFrom, type GraphView } from "./rules";
import type { Snapshot } from "./snapshot";
import { same, type Json } from "../model/graphPatch";

export * from "./snapshot";
export * from "./actions";

/** The node editor's node array: each node's prepared data composed once per
 * (state/cookInputs.ts's version, state/look.ts's onNode, state/results.ts's reply, deliberately excluding positions and
 * per-node cook status/progress) so a cooking node's frequent progress ticks do not rebuild every other node's prepared object;
 * GraphNode reads its own live status, progress and delivery through its own small selectors instead. Everything it
 * shows from the server (ports, cost, licence, values, sources, overrides, a wire's value, which rows are greyed) is
 * the last reply's, by the one rule of state/results.ts (Shown): kept while an edit waits for the next reply, with
 * `pending` set on the nodes that reply may no longer be right about (the state cell says 「待更新」, values are faint),
 * so nothing disappears and comes back and no node changes height. */

interface PreparedOutput {
  name: string;
  label: string;
  type: string;
  list: boolean; // it gives a list (a square socket, a double wire)
  ghost: boolean;
  waits: string;
  // whether this output is unavailable now and why (computed by the server, nodes/port.py Port.applies): greyed, it
  // cannot be wired out of, and a refused wire says why (once a solver has a camera wired in, its 「相机」 output only
  // passes that camera through)
  inactive?: PortDef["inactive"];
}

interface PreparedNode extends Record<string, unknown> {
  typeId: string;
  params: Record<string, unknown>;
  promoted?: string[];
  picked?: Record<string, unknown>;
  onNode?: string[]; // 在节点上显示: the rows the user chose for this node (undefined: its type's NodeDef.on_node)
  stored?: Record<string, unknown>;
  inputs: PortDef[];
  outputs: PreparedOutput[];
  wired: Record<string, { from: string; node: string; source: string; value: string; fallback: boolean } | null>;
  values?: Record<string, string>;
  sources?: Record<string, string>;
  overrides?: Record<string, string>;
  applies?: Availability; // its parameters' availability (applies.ts)
  pending: boolean; // what it shows is the last reply's, which may be out of date since an edit (state/results.ts Shown)
  cost?: ResolvedCost;
  commercial: boolean;
  carried: Record<string, string[]>; // scene kinds an output carries, by port name
}

export type PreparedGNode = Node<PreparedNode, "l2s">;

const ORIGIN = { x: 0, y: 0 }; // a node with no position yet: one object, so it does not count as moved on every render

// `results`: what is shown (state/results.ts Shown: the last reply's, trusted or not), `pending` its out-of-date nodes
type Shape = GraphState & GraphView & { results: Snapshot["shown"]; pending: Snapshot["pending"] };

/** Its outputs, and a ghost for each wire from one that is not there: waiting for its parameter (what brings it), or gone. */
function shownOutputs(s: Shape, id: string): PreparedOutput[] {
  const ports = pendingPorts(s, id);
  const shown = ports.outputs.map((p) => ({ name: p.name, label: p.label, type: p.type, list: !!p.list, ghost: false, waits: "", inactive: p.inactive }));
  for (const e of s.edges) {
    const name = e.sourceHandle ?? "";
    if (e.source !== id || shown.some((p) => p.name === name)) continue;
    const w = ports.waiting.find((p) => p.name === name);
    shown.push(w ? { name, label: w.label, type: w.type, list: !!w.list, ghost: true, waits: w.waits ?? "", inactive: undefined } : { name, label: name, type: "", list: false, ghost: true, waits: "", inactive: undefined });
  }
  return shown;
}

function buildPrepared(s: Shape, id: string): PreparedNode {
  const n = s.nodes.find((m) => m.id === id)!;
  const results = s.results[id];
  const ports = pendingPorts(s, id);
  const carried = Object.fromEntries(ports.outputs.filter((p) => p.type.startsWith("scene")).map((p) => [p.name, p.kinds ?? []]));
  const wired: PreparedNode["wired"] = {};
  // the standing parameter inputs (wired_ports) are looked up as well: they are not in the file's `promoted`, and taking
  // only `promoted` would leave 「← AnyCalib」 off the node
  for (const name of paramPortNames(s.nodeDefs[n.data.typeId], n.data.promoted)) wired[name] = wiredFrom(s, id, name);
  return {
    typeId: n.data.typeId, params: n.data.params, promoted: n.data.promoted, picked: n.data.picked,
    onNode: n.data.onNode, stored: n.data.stored,
    inputs: inputsOf(s, id), outputs: shownOutputs(s, id), wired, values: results?.values, sources: results?.sources, overrides: results?.overrides, applies: results?.applies,
    pending: s.pending.has(id),
    cost: costOf(s, id), commercial: commercialOf(s, id), carried,
  };
}

/** The composed node array the editor's canvas draws, plus the edges it was built from (for callers that also need
 * the wires at the same version: NodeEditor). Each node's prepared data is built from
 * the graph's content only; a move (every pointer step of a drag) changes positions alone, so the data objects stay the
 * same and GraphNode (memo) redraws none of the nodes, only xyflow moves them. */
export function useComposedNodes(): { nodes: PreparedGNode[]; edges: ReturnType<typeof useCookInputs.getState>["edges"] } {
  const version = useCookInputs((s) => s.version);
  const nodesById = useCookInputs((s) => s.nodes);
  const order = useCookInputs((s) => s.order);
  const edges = useCookInputs((s) => s.edges);
  const positions = useLook((s) => s.positions);
  const onNode = useLook((s) => s.onNode);
  const { results, pending } = useShownResults();
  const reply = useResults((s) => s.reply);
  const catalog = useCatalog();
  // the objects handed out last time: a node whose prepared data (or position) is the same as before keeps its very
  // object, so a status reply, an edit or a move redraws only the nodes it changed (GraphNode is memo; xyflow keeps a
  // node whose object is unchanged), not all of them — on a graph of a few hundred nodes that is the difference
  // between a fluid canvas and hundreds of node redraws per reply
  const keptData = useRef(new Map<string, PreparedNode>());
  const keptNodes = useRef(new Map<string, PreparedGNode>());
  const prepared = useMemo(() => {
    const shape: Shape = {
      nodes: order.map((id) => ({ id, type: "l2s" as const, position: { x: 0, y: 0 }, data: { typeId: nodesById[id].typeId, params: nodesById[id].params, promoted: nodesById[id].promoted, picked: nodesById[id].picked, stored: nodesById[id].stored, onNode: onNode[id], status: "idle" as const, note: "" } })),
      edges,
      nodeDefs: getNodeDefs(),
      results,
      pending,
      reply,
      catalog,
    };
    const before = keptData.current;
    const now = new Map<string, PreparedNode>();
    for (const id of order) {
      const fresh = buildPrepared(shape, id);
      const old = before.get(id);
      now.set(id, old && same(old as unknown as Json, fresh as unknown as Json) ? old : fresh);
    }
    keptData.current = now;
    return now;
  }, [version, nodesById, order, edges, onNode, results, pending, reply, catalog]); // eslint-disable-line react-hooks/exhaustive-deps
  return useMemo(() => {
    const before = keptNodes.current;
    const now = new Map<string, PreparedGNode>();
    const nodes = order.map((id) => {
      const position = positions[id] ?? ORIGIN;
      const data = prepared.get(id)!;
      const old = before.get(id);
      const node = old && old.position === position && old.data === data ? old : { id, type: "l2s" as const, position, data };
      now.set(id, node);
      return node;
    });
    keptNodes.current = now;
    return { nodes, edges };
  }, [prepared, positions, order, edges]);
}
