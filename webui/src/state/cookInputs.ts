import { create } from "zustand";
import type { ExposedEntry, ExposedGroup, ExposedParam, GraphJSON, PickedFrom } from "../api";
import { useViewer } from "./viewer";
import { disableWhenOf } from "../platform/conditions";
import { same, type Json } from "../model/graphPatch";

/** 计算输入: what the server's check of the graph (status: fingerprints, cached, errors) reads, plus the document's
 * own words that travel with it. Two counters, each bumped by the setters themselves (no field-by-field comparison of
 * snapshots anywhere):
 * - `version`（影响计算的输入）: nodes (type, label, parameters, promoted, picked files), wires, the frame range, the
 *   kept unknown nodes — what the server's answer depends on (engine/evaluations.py content_key). It bumps, and
 *   everything that trusts a result (state/results.ts's `useTrustedResults`, graph/snapshot.ts) stops trusting the old
 *   one and the status is asked again (graph/document.ts noticed); a job carries it back (graph/streamDone.ts).
 * - `edits`（文档任何改动）: bumps on every change, those above and also `meta` (name, description…) and `exposed` (the
 *   parameter interface: its order, labels, groups, conditions). Those two are not cook inputs — the server's answer
 *   does not read them — so renaming the graph or laying out its parameter interface asks nothing and leaves every
 *   result trusted; the working copy (editor/autosave.ts) watches `edits`, so it still writes them.
 * Nothing here is the display node, a position, a group box, what a node's body shows (on_node) or the playback range —
 * those are 文档外观 (state/look.ts): showing another node, moving one, or scrubbing the timeline leaves every cached
 * result standing. */

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
  version: number; // 影响计算的输入 changed: what state/results.ts's useTrustedResults compares against
  edits: number; // any change of the document here (also meta and exposed): what editor/autosave.ts watches

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

/** In a read-only tab (editor/tabs.ts: this graph is being edited in another tab; state/viewer.ts role "viewer") nothing
 * of the document may change: every write here and in state/look.ts is refused. Disabled controls and greyed styles
 * are only appearance and stop neither the keyboard nor other entry points; this is the real gate. Reading in (load:
 * opening, undo / redo, a newer version saved by another tab) and changing the id (setGraphId) are not edits and go
 * through as usual. */
export const readOnly = (): boolean => useViewer.getState().role === "viewer";

/** The same, for rendering: what the page draws as not editable (greyed controls, hidden handle gizmos, a view-only
 * editor) reads this, never the role itself. `READ_ONLY_WHY` is the reason shown with it. */
export const useReadOnly = (): boolean => useViewer((s) => s.role === "viewer");
export const READ_ONLY_WHY = "这张节点图在另一个标签页里编辑：这里只能看";

/** Whether a write leaves the value as it was (JSON values, compared deeply): the one test every setter here makes
 * before counting a change, so a write of the same value never moves the version. */
const unchanged = (a: unknown, b: unknown): boolean => same(a as Json, b as Json);

export const useCookInputs = create<State>((rawSet) => {
  // writes that edit the document: refused in a read-only tab (readOnly)
  const edit: typeof rawSet = (...a) => (readOnly() ? undefined : rawSet(...(a as Parameters<typeof rawSet>)));
  return {
    graphId: "",
    meta: { name: "未命名" },
    exposed: [],
    cookRange: null,
    nodes: {},
    order: [],
    edges: [],
    kept: { nodes: [], edges: [] },
    version: 0,
    edits: 0,

    load: (p) => rawSet((s) => ({ ...p, ...cookChange(s) })),
    setGraphId: (id) => rawSet({ graphId: id }),
    // the graph's name, description and parameter interface: the document changes, the cook inputs do not (the server's
    // answer does not read them)
    setMeta: (patch) => edit((s) => (unchanged({ ...s.meta, ...patch }, s.meta) ? {} : { meta: { ...s.meta, ...patch }, edits: s.edits + 1 })),
    setExposed: (exposed) => edit((s) => (unchanged(exposed, s.exposed) ? {} : { exposed, edits: s.edits + 1 })),
    setCookRange: (r) => edit((s) => (unchanged(r, s.cookRange) ? {} : { cookRange: r, ...cookChange(s) })),
    // writing what the node already has (a handle trembling back to the same value, a field committed unchanged) is not
    // a change: the cook inputs' version stays, results stay trusted (`unchanged`, the same test for every write here)
    setNode: (id, patch) =>
      edit((s) => {
        const node = s.nodes[id];
        if (!node || Object.entries(patch).every(([k, v]) => unchanged(v, node[k as keyof CookNode]))) return {};
        return { nodes: { ...s.nodes, [id]: { ...node, ...patch } }, ...cookChange(s) };
      }),
    insertNode: (id, data) => edit((s) => ({ nodes: { ...s.nodes, [id]: data }, order: [...s.order, id], ...cookChange(s) })),
    removeNodes: (ids) =>
      edit((s) => {
        const gone = new Set(ids);
        const nodes = { ...s.nodes };
        for (const id of ids) delete nodes[id];
        // nothing points at a removed node any more: neither wires nor exposed parameters (an interface entry targeting it
        // would otherwise silently vanish in app mode and only be refused when the template is saved)
      return {
          nodes,
          order: s.order.filter((id) => !gone.has(id)),
          edges: s.edges.filter((e) => !gone.has(e.source) && !gone.has(e.target)),
          exposed: withoutNodes(s.exposed, gone),
          ...cookChange(s),
        };
      }),
    setEdges: (edges) => edit((s) => (unchanged(edges, s.edges) ? {} : { edges, ...cookChange(s) })),
    setKept: (kept) => edit((s) => (unchanged(kept, s.kept) ? {} : { kept, ...cookChange(s) })),
  };
});

/** The cook inputs changed: both counters bump (it is a document change as well). */
function cookChange(s: { version: number; edits: number }): { version: number; edits: number } {
  return { version: s.version + 1, edits: s.edits + 1 };
}

// ------------------------------------------------------------------ the parameter interface (the `exposed` tree, api/catalog.ts ExposedEntry)

export const isGroup = (x: ExposedEntry): x is ExposedGroup => (x as ExposedGroup).kind === "group";

/** Every parameter entry of the tree, in display order. */
export function exposedParams(tree: ExposedEntry[]): ExposedParam[] {
  const out: ExposedParam[] = [];
  const walk = (xs: ExposedEntry[]) => xs.forEach((x) => (isGroup(x) ? walk(x.children) : out.push(x)));
  walk(tree);
  return out;
}

/** Removes the entries targeting these nodes (when nodes are deleted, removeNodes); groups stay. */
export function withoutNodes(tree: ExposedEntry[], ids: ReadonlySet<string>): ExposedEntry[] {
  return tree.filter((x) => isGroup(x) || !ids.has(x.target.split(".")[0])).map((x) => (isGroup(x) ? { ...x, children: withoutNodes(x.children, ids) } : x));
}

/** Removes the entries targeting `target` (in whichever group); groups stay. */
export function withoutTarget(tree: ExposedEntry[], target: string): ExposedEntry[] {
  return tree.filter((x) => isGroup(x) || x.target !== target).map((x) => (isGroup(x) ? { ...x, children: withoutTarget(x.children, target) } : x));
}

/** Reads a file's `exposed` as a tree: a flat list of the older format already is a tree (entries under the root), and
 * the older `when` becomes `disable_when`; anything not an array counts as none, a group without children gets an
 * empty list, and an entry that is neither group nor parameter is dropped (the server refuses it too when saving a
 * template: engine/templates.py check_exposed). */
export function readExposed(raw: unknown): ExposedEntry[] {
  if (!Array.isArray(raw)) return [];
  const out: ExposedEntry[] = [];
  for (const x of raw) {
    if (!x || typeof x !== "object") continue;
    const o = x as Record<string, unknown>;
    if (o.kind === "group") out.push({ ...(o as unknown as ExposedGroup), label: String(o.label ?? ""), children: readExposed(o.children) });
    else if (typeof o.name === "string" && typeof o.target === "string") {
      // the older `when` becomes `disable_when` (platform/conditions.ts disableWhenOf, the same rule as the server); saving
      // writes the new field
      const { when: _old, ...rest } = o as unknown as ExposedParam & { when?: unknown };
      out.push({ ...rest, disable_when: disableWhenOf(o) ?? undefined, label: String(o.label ?? o.name) });
    }
  }
  return out;
}
