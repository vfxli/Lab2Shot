import { useMemo } from "react";
import type { Node } from "@xyflow/react";
import type { PortDef, ResolvedCost } from "../api";
import { getNodeDefs, useCatalog } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useResults } from "../state/results";
import type { Availability } from "../api/applies";
import type { GraphState } from "../state/graph";
import { commercialOf, costOf, inputsOf, paramPortNames, pendingPorts, wiredFrom, type GraphView } from "./rules";
import type { Snapshot } from "./snapshot";

export * from "./snapshot";
export * from "./actions";

/** The node editor's node array: composed once per
 * (state/cookInputs.ts's version, state/look.ts's version, state/results.ts's reply, deliberately excluding per-node
 * cook status/progress) so a cooking node's frequent progress ticks do not rebuild every other node's prepared object;
 * GraphNode reads its own live status, progress and delivery through its own small selectors instead. Its ports,
 * cost and licence are the last status reply's (graph/rules.ts: kept while an edit waits for the next one); its
 * values, sources and parameter availability only the trusted results'. */

export interface PreparedOutput {
  name: string;
  label: string;
  type: string;
  type_label: string; // the type in the server's words (PortDef.type_label); a gone port carries none
  unit: string;
  tip: string; // what the pointer says over it, written by the server (nodes/base.py Port.tip)
  list: boolean; // it gives a list (a square socket, a double wire)
  ghost: boolean;
  waits: string;
  // 该输出口当前是否不可用及其原因（由服务器计算，nodes/base.py Port.applies）：置灰、悬停说明原因、
  // 无法连出（解算器接入相机后，其「相机」输出仅原样透传）
  inactive?: PortDef["inactive"];
}

export interface PreparedNode extends Record<string, unknown> {
  typeId: string;
  label: string;
  params: Record<string, unknown>;
  promoted?: string[];
  picked?: Record<string, unknown>;
  saveTo?: { handle: string; name: string };
  onNode?: string[]; // 在节点上显示: the rows the user chose for this node (undefined: its type's NodeDef.on_node)
  stored?: Record<string, unknown>;
  inputs: PortDef[];
  outputs: PreparedOutput[];
  wired: Record<string, { from: string; node: string; source: string; value: string } | null>;
  values?: Record<string, string>;
  sources?: Record<string, string>;
  applies?: Availability; // its parameters' availability (applies.ts)
  cost?: ResolvedCost;
  commercial: boolean;
  carried: Record<string, string[]>; // scene kinds an output carries, by port name
}

export type PreparedGNode = Node<PreparedNode, "l2s">;

type Shape = GraphState & GraphView & { results: Snapshot["results"] };

/** Its outputs, and a ghost for each wire from one that is not there: waiting for its parameter (what brings it), or gone. */
function shownOutputs(s: Shape, id: string): PreparedOutput[] {
  const ports = pendingPorts(s, id);
  const shown = ports.outputs.map((p) => ({ name: p.name, label: p.label, type: p.type, type_label: p.type_label, unit: p.unit, tip: p.tip ?? "", list: !!p.list, ghost: false, waits: "", inactive: p.inactive }));
  for (const e of s.edges) {
    const name = e.sourceHandle ?? "";
    if (e.source !== id || shown.some((p) => p.name === name)) continue;
    const w = ports.waiting.find((p) => p.name === name);
    shown.push(w ? { name, label: w.label, type: w.type, type_label: w.type_label, unit: "", tip: w.tip ?? "", list: !!w.list, ghost: true, waits: w.waits ?? "", inactive: undefined } : { name, label: name, type: "", type_label: "", unit: "", tip: "", list: false, ghost: true, waits: "", inactive: undefined });
  }
  return shown;
}

function buildPrepared(s: Shape, id: string): PreparedNode {
  const n = s.nodes.find((m) => m.id === id)!;
  const results = s.results[id];
  const ports = pendingPorts(s, id);
  const carried = Object.fromEntries(ports.outputs.filter((p) => p.type.startsWith("scene")).map((p) => [p.name, p.kinds ?? []]));
  const wired: PreparedNode["wired"] = {};
  // 常驻参数口（wired_ports）同样需要查询：它们不在文件的 promoted 中，若只取 promoted，节点上将不会显示「← AnyCalib」
  for (const name of paramPortNames(s.nodeDefs[n.data.typeId], n.data.promoted)) wired[name] = wiredFrom(s, id, name);
  return {
    typeId: n.data.typeId, label: n.data.label, params: n.data.params, promoted: n.data.promoted, picked: n.data.picked,
    saveTo: n.data.saveTo, onNode: n.data.onNode, stored: n.data.stored,
    inputs: inputsOf(s, id), outputs: shownOutputs(s, id), wired, values: results?.values, sources: results?.sources, applies: results?.applies,
    cost: costOf(s, id), commercial: commercialOf(s, id), carried,
  };
}

/** The composed node array the editor's canvas draws, plus the plain graph-shape used to build it (for callers that
 * also need edges/nodeDefs at the same version, e.g. NodeEditor's TypedEdge). */
export function useComposedNodes(): { nodes: PreparedGNode[]; edges: ReturnType<typeof useCookInputs.getState>["edges"] } {
  const version = useCookInputs((s) => s.version);
  const nodesById = useCookInputs((s) => s.nodes);
  const order = useCookInputs((s) => s.order);
  const edges = useCookInputs((s) => s.edges);
  const lookVersion = useLook((s) => s.version);
  const positions = useLook((s) => s.positions);
  const onNode = useLook((s) => s.onNode);
  const results = useResults((s) => s.results);
  const reply = useResults((s) => s.reply);
  const forCookInputs = useResults((s) => s.forCookInputs);
  const catalog = useCatalog();
  const trusted = forCookInputs === version;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => {
    const shape: Shape = {
      nodes: order.map((id) => ({ id, type: "l2s" as const, position: positions[id] ?? { x: 0, y: 0 }, data: { typeId: nodesById[id].typeId, label: nodesById[id].label, params: nodesById[id].params, promoted: nodesById[id].promoted, picked: nodesById[id].picked, saveTo: nodesById[id].saveTo, stored: nodesById[id].stored, onNode: onNode[id], status: "idle" as const, note: "" } })),
      edges,
      nodeDefs: getNodeDefs(),
      results: trusted ? results : {},
      reply,
      catalog,
    };
    return {
      nodes: order.map((id) => ({ id, type: "l2s" as const, position: positions[id] ?? { x: 0, y: 0 }, data: buildPrepared(shape, id) })),
      edges,
    };
  }, [version, lookVersion, nodesById, order, edges, positions, onNode, results, reply, trusted, catalog]);
}
