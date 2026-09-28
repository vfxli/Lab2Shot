import { useMemo } from "react";
import type { Catalog, DataType, NodeStatus as Status, StatusReply } from "../api";
import { getCatalog, getNodeDefs, getTypes, useCatalog } from "../state/catalog";
import { useCookInputs, type CookNode } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useResults, type NodeCookStatus } from "../state/results";
import type { GraphState, NodeData } from "../state/graph";

/** The plain graph snapshot every pure helper in graph/rules.ts and view/plan.ts reads: assembled from
 * state/cookInputs.ts + state/look.ts + state/results.ts + the catalog, recomputed only when one of them actually
 * changed. `nodes` here carries what a node IS (its type, its parameters, its position) and
 * its live cook status (idle/cooked/cooking/error, its note, why it is blocked); it is not the richer, precomputed
 * object the node editor hands each GraphNode (graph/index.ts's useComposedNodes), since most readers of a snapshot (the
 * parameter panel, the viewer's display plan, the cook-policy blockers list) only need this much. */

export interface Snapshot extends GraphState {
  graphId: string; // 节点图文件自身的标识（state/cookInputs.ts）：「过期」记录以此为准（state/stale.ts）
  types: Record<string, DataType>;
  results: Record<string, Status>; // trusted: {} while the cook inputs moved on since the last reply
  reply: StatusReply | null; // the last reply, trusted or not: what is drawn until the next one (graph/rules.ts)
  resultsAreTrusted: boolean;
  catalog: Catalog | null;
}

const IDLE: NodeCookStatus = { status: "idle", note: "" };

function toNodeData(n: CookNode, onNode: string[] | undefined, status: NodeCookStatus | undefined): NodeData {
  return { typeId: n.typeId, label: n.label, params: n.params, promoted: n.promoted, picked: n.picked, stored: n.stored, onNode, ...(status ?? IDLE) };
}

/** The imperative equivalent of useGraphSnapshot(), for event handlers and other stores' actions. */
export function snapshotNow(): Snapshot {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const r = useResults.getState();
  const trusted = r.forCookInputs === ci.version;
  return {
    graphId: ci.graphId,
    nodes: ci.order.map((id) => ({ id, type: "l2s", position: look.positions[id] ?? { x: 0, y: 0 }, data: toNodeData(ci.nodes[id], look.onNode[id], r.byNode[id]) })),
    edges: ci.edges,
    nodeDefs: getNodeDefs(),
    types: getTypes(),
    results: trusted ? r.results : {},
    reply: r.reply,
    resultsAreTrusted: trusted,
    catalog: getCatalog(),
  };
}

/** The reactive graph snapshot: recomputed only when state/cookInputs.ts's or state/look.ts's version, or
 * state/results.ts's trusted results or per-node cook status, change. */
export function useGraphSnapshot(): Snapshot {
  const version = useCookInputs((s) => s.version);
  const graphId = useCookInputs((s) => s.graphId);
  const nodesById = useCookInputs((s) => s.nodes);
  const order = useCookInputs((s) => s.order);
  const edges = useCookInputs((s) => s.edges);
  const lookVersion = useLook((s) => s.version);
  const positions = useLook((s) => s.positions);
  const onNode = useLook((s) => s.onNode);
  const results = useResults((s) => s.results);
  const reply = useResults((s) => s.reply);
  const forCookInputs = useResults((s) => s.forCookInputs);
  const byNode = useResults((s) => s.byNode);
  const catalog = useCatalog();
  const trusted = forCookInputs === version;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(
    () => ({
      graphId,
      nodes: order.map((id) => ({ id, type: "l2s" as const, position: positions[id] ?? { x: 0, y: 0 }, data: toNodeData(nodesById[id], onNode[id], byNode[id]) })),
      edges,
      nodeDefs: getNodeDefs(),
      types: getTypes(),
      results: trusted ? results : {},
      reply,
      resultsAreTrusted: trusted,
      catalog,
    }),
    [version, graphId, lookVersion, nodesById, order, edges, positions, onNode, results, reply, byNode, trusted, catalog],
  );
}
