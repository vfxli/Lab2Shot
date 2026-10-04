import { create } from "zustand";
import type { ExposedEntry, ExposedGroup, ExposedParam, GraphJSON, PickedFrom, Words } from "../api";
import { useViewer } from "./viewer";
import { disableWhenOf } from "../platform/conditions";
import { same, type Json } from "../model/graphPatch";
import { nodeWord } from "../graph/naming";
import type { Said } from "../messages/format";
import { pick, t } from "../i18n/t";
import { getLang } from "../i18n/lang";
import { drives, splitTarget, targetsOf } from "../model/targets";

/** 计算输入: what the server's check of the graph (status: fingerprints, cached, errors) reads, plus the document's
 * own words that travel with it. Two counters, each bumped by the setters themselves (no field-by-field comparison of
 * snapshots anywhere):
 * - `version`（影响计算的输入）: nodes (type, parameters, promoted, picked files), wires, the frame range, the
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
  // a node renamed (its id: graph/naming.ts): every place of the cook inputs that names it follows — its key and place in
  // `order`, the wires (and their ids), the exposed parameters' targets, the kept 「未知节点」's wires
  renameNode: (from: string, to: string) => void;
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
 * editor) reads this, never the role itself. `readOnlyWhy()` is the reason shown with it. */
export const useReadOnly = (): boolean => useViewer((s) => s.role === "viewer");
export const readOnlyWhy = (): string => t("ui.params.read_only");

/** Whether a write leaves the value as it was (JSON values, compared deeply): the one test every setter here makes
 * before counting a change, so a write of the same value never moves the version. */
const unchanged = (a: unknown, b: unknown): boolean => same(a as Json, b as Json);

export const useCookInputs = create<State>((rawSet) => {
  // writes that edit the document: refused in a read-only tab (readOnly)
  const edit: typeof rawSet = (...a) => (readOnly() ? undefined : rawSet(...(a as Parameters<typeof rawSet>)));
  return {
    graphId: "",
    meta: { name: "" },
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
    renameNode: (from, to) =>
      edit((s) => {
        if (!s.nodes[from] || s.nodes[to] || from === to) return {};
        const id = (x: string) => (x === from ? to : x);
        const nodes: Record<string, CookNode> = {};
        for (const k of s.order) nodes[id(k)] = s.nodes[k];
        const wire = (e: Wire): Wire => (e.source === from || e.target === from ? { ...e, id: `${id(e.source)}.${e.sourceHandle}->${id(e.target)}.${e.targetHandle}`, source: id(e.source), target: id(e.target) } : e);
        const kept = { ...s.kept, edges: s.kept.edges.map((e) => (e.from[0] === from || e.to[0] === from ? { ...e, from: [id(e.from[0]), e.from[1]] as [string, string], to: [id(e.to[0]), e.to[1]] as [string, string] } : e)) };
        return { nodes, order: s.order.map(id), edges: s.edges.map(wire), exposed: renamedTargets(s.exposed, from, to), kept, ...cookChange(s) };
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

export { targetsOf, splitTarget, firstTarget, drives } from "../model/targets";

/** What an entry is called on the page: a card's last step (it names a shared button word, `word`) is one term in
 * every card whichever word it names (button.pack), else its own label (the server's engine/templates.py
 * exposed_label, the same rule). */
export const entryLabel = (x: ExposedParam): string => (x.word ? t("button.pack") : pick(x.label));

/** 卡片上一个节点叫什么（应用模式里用卡片的人看不到节点名：视图标题、结果视图里各项的名字）：卡片上目标在它上面的那一项——
 * 它的「计算」按钮（「后处理并预览」）优先，其次第一项（开关「CoTracker3（2D 点）」、参数「跟踪点」）；卡片上没有它的：""。 */
export function cardNameOf(tree: ExposedEntry[], node: string): string {
  const on = exposedParams(tree).filter((x) => targetsOf(x).some((k) => splitTarget(k)[0] === node));
  const x = on.find((y) => targetsOf(y).some((k) => k === `${node}.cook`)) ?? on[0];
  return x ? entryLabel(x) : "";
}

/** The entry with these targets (one written as a plain key, as a file has it); null when none is left. */
const withTargets = (x: ExposedParam, keys: string[]): ExposedParam | null =>
  !keys.length ? null : keys.length === 1 ? { ...x, target: keys[0] } : { ...x, target: keys };

/** Each entry with only the targets `keep` takes (an entry left with none goes); groups stay. */
function keepTargets(tree: ExposedEntry[], keep: (key: string) => string | null): ExposedEntry[] {
  return tree.flatMap((x): ExposedEntry[] => {
    if (isGroup(x)) return [{ ...x, children: keepTargets(x.children, keep) }];
    const keys = targetsOf(x).map(keep).filter((k): k is string => k !== null);
    const now = keys.length === targetsOf(x).length && keys.every((k, i) => k === targetsOf(x)[i]) ? x : withTargets(x, keys);
    return now ? [now] : [];
  });
}

/** Removes the targets on these nodes (when nodes are deleted, removeNodes): an entry left with none goes; groups stay. */
export const withoutNodes = (tree: ExposedEntry[], ids: ReadonlySet<string>): ExposedEntry[] =>
  keepTargets(tree, (k) => (ids.has(splitTarget(k)[0]) ? null : k));

/** The targets on node `from` are on node `to` instead (a renamed node: `from.param` → `to.param`). */
export const renamedTargets = (tree: ExposedEntry[], from: string, to: string): ExposedEntry[] =>
  keepTargets(tree, (k) => (splitTarget(k)[0] === from ? `${to}.${splitTarget(k)[1]}` : k));

/** Removes `target` from the entries driving it (in whichever group): an entry left with none goes; groups stay. */
export const withoutTarget = (tree: ExposedEntry[], target: string): ExposedEntry[] =>
  keepTargets(tree, (k) => (k === target ? null : k));

/** The node parameters tied to `key` through an entry driving several (graph/edit.ts writes a value into all of them,
 * so they always hold one value): [] when none is. */
export function linkedTargets(tree: ExposedEntry[], key: string): string[] {
  const x = exposedParams(tree).find((y) => drives(y, key) && targetsOf(y).length > 1);
  return x ? targetsOf(x).filter((k) => k !== key) : [];
}

/** A display field as the document keeps it: a user's own words (a string) as they are, a built-in template's
 * {"zh": …, "en": …} kept whole (both languages go back into the file on save); read it with i18n/t.ts pick() (its type,
 * api/catalog.ts Words, makes tsc find a reader that does not). */
function shown(v: unknown, fallback: string): Words {
  if (typeof v === "string") return v;
  if (v && typeof v === "object") return v as Words;
  return fallback;
}

/** The language a plain (one-language) display text is taken to be in: Chinese when it has a Chinese character, else
 * English (lab2shot/site/library.py language_of, the same rule). */
export const languageOf = (text: string): "zh" | "en" => (/[\u3400-\u9fff]/.test(text) ? "zh" : "en");

/** A display field after the user typed `text` into it (the parameter interface's names, tips, groups and options, a
 * table's text cell such as a 「切换」 way's name): what is typed goes into the page's language, the other language is
 * kept as it is ({"zh", "en"}, read with i18n/t.ts pick). An old plain string is first put where its language says
 * (languageOf); nothing before: {page's language: text}. One input box: the other language is edited by switching the
 * page to it. */
export function edited(old: unknown, text: string): Words {
  const base: Record<string, string> =
    old && typeof old === "object" ? { ...(old as Record<string, string>) } : typeof old === "string" && old.trim() ? { [languageOf(old)]: old } : {};
  return { ...base, [getLang()]: text };
}

/** A display field made new on the page (a parameter exposed, a group added): {page's language: text}. */
export const newWords = (text: string): Words => ({ [getLang()]: text });

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
    if (o.kind === "group") out.push({ ...(o as unknown as ExposedGroup), label: shown(o.label, ""), children: readExposed(o.children) });
    else if (typeof o.name === "string" && (typeof o.target === "string"
             || (Array.isArray(o.target) && o.target.length > 0 && o.target.every((k) => typeof k === "string")))) {
      // the older `when` becomes `disable_when` (platform/conditions.ts disableWhenOf, the same rule as the server); saving
      // writes the new field
      const { when: _old, ...rest } = o as unknown as ExposedParam & { when?: unknown };
      // one called by a shared button word (`word`) has no label of its own: none is made up for it (saving would
      // write both, which the server refuses)
      out.push({ ...rest, disable_when: disableWhenOf(o) ?? undefined, ...(rest.word ? {} : { label: shown(o.label, String(o.name)) }) });
    }
  }
  return out;
}

/** How a message points at node `id` of the open graph, as its parameter: `name（type）` kept by its key (graph/naming.ts nodeWord). */
export const nodeRefOf = (id: string): Said | string => nodeWord(id, useCookInputs.getState().nodes[id]?.typeId);
