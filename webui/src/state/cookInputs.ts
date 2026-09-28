import { create } from "zustand";
import type { GraphJSON, PickedFrom } from "../api";

/** 计算输入: what the server's check of the graph (status: fingerprints, cached, errors) reads. Changing any of it
 * is a cook change: `version` bumps, and everything that trusts a result (state/results.ts's `useTrustedResults`)
 * stops trusting the old one. Nothing here is the display node, a position, a group box, what a node's body shows
 * (on_node) or the playback range — those are 文档外观 (state/look.ts): showing another node, moving one, or
 * scrubbing the timeline leaves every cached result standing. This holds by construction: a change to this store IS
 * a cook change, so no field-by-field comparison of snapshots is needed anywhere. */

export interface CookNode {
  typeId: string;
  label: string;
  params: Record<string, unknown>;
  promoted?: string[]; // parameters driven by a wire: each has an input "param:<name>" (NodeTypeDef.param_ports)
  picked?: Record<string, PickedFrom>; // input file parameters: what was picked, as the parameter shows it
  // choices the file has that this account may not use (a non-commercial model): the node shows and cooks the first
  // choice it may use, and the file keeps its own value until the parameter is changed (checkGraph, fileJSON)
  stored?: Record<string, unknown>;
}

export interface Wire {
  id: string;
  source: string;
  sourceHandle: string;
  target: string;
  targetHandle: string;
}

interface State {
  graphId: string; // the graph file's own identity; "" before the first load
  meta: Omit<GraphJSON["meta"], "id">;
  exposed: GraphJSON["exposed"];
  cookRange: [string, string] | null; // the frames to cook, first and last as typed (null: every frame of the inputs)
  nodes: Record<string, CookNode>;
  order: string[]; // node ids, file order (also the iteration order graph/index.ts composes in)
  edges: Wire[];
  kept: { nodes: GraphJSON["nodes"]; edges: GraphJSON["edges"] }; // 「未知节点」 of the graph loaded (checkGraph)
  version: number; // bumps on every change: what state/results.ts's useTrustedResults compares against

  load: (p: { graphId: string; meta: Omit<GraphJSON["meta"], "id">; exposed: GraphJSON["exposed"]; cookRange: [string, string] | null; nodes: Record<string, CookNode>; order: string[]; edges: Wire[]; kept: State["kept"] }) => void;
  // the graph's own identity: never a cook change on its own — 另存为 gives the copy a fresh id
  // (model/graphId.ts's newGraphId, set by graph/graphFile.ts) without disturbing any computed result
  setGraphId: (id: string) => void;
  setMeta: (patch: Partial<State["meta"]>) => void;
  setExposed: (exposed: GraphJSON["exposed"]) => void;
  setCookRange: (r: [string, string] | null) => void;
  setNode: (id: string, patch: Partial<CookNode>) => void;
  insertNode: (id: string, data: CookNode) => void; // appended to `order`
  removeNodes: (ids: string[]) => void; // drops them from nodes/order and any wire touching them
  setEdges: (edges: Wire[]) => void;
  setKept: (kept: State["kept"]) => void;
}

export const useCookInputs = create<State>((set) => ({
  graphId: "",
  meta: { name: "未命名" },
  exposed: [],
  cookRange: null,
  nodes: {},
  order: [],
  edges: [],
  kept: { nodes: [], edges: [] },
  version: 0,

  load: (p) => set((s) => ({ ...p, version: s.version + 1 })),
  setGraphId: (id) => set({ graphId: id }),
  setMeta: (patch) => set((s) => ({ meta: { ...s.meta, ...patch }, version: s.version + 1 })),
  setExposed: (exposed) => set((s) => ({ exposed, version: s.version + 1 })),
  setCookRange: (r) => set((s) => ({ cookRange: r, version: s.version + 1 })),
  setNode: (id, patch) =>
    set((s) => (s.nodes[id] ? { nodes: { ...s.nodes, [id]: { ...s.nodes[id], ...patch } }, version: s.version + 1 } : {})),
  insertNode: (id, data) => set((s) => ({ nodes: { ...s.nodes, [id]: data }, order: [...s.order, id], version: s.version + 1 })),
  removeNodes: (ids) =>
    set((s) => {
      const gone = new Set(ids);
      const nodes = { ...s.nodes };
      for (const id of ids) delete nodes[id];
      return {
        nodes,
        order: s.order.filter((id) => !gone.has(id)),
        edges: s.edges.filter((e) => !gone.has(e.source) && !gone.has(e.target)),
        version: s.version + 1,
      };
    }),
  setEdges: (edges) => set((s) => ({ edges, version: s.version + 1 })),
  setKept: (kept) => set((s) => ({ kept, version: s.version + 1 })),
}));
