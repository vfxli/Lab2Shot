/** The open graph as a document: undo and redo over state/cookInputs.ts and
 * state/look.ts, the unsaved mark, loading a graph file (checked and fixed up) and writing one. graph/actions.ts
 * re-exports what components call. */

import { useHandleView } from "../state/handleView";
import type { Edge } from "@xyflow/react";
import type { GraphJSON, NodeTypeDef } from "../api";
import { History, absorb, anchor, bindRecorder, restore, restoring, writing, type Doc, type RecordAs } from "./history";
import { same, type Json } from "../model/graphPatch";
import { exposedValues, hiddenChoices } from "./exposedTree";
import { condEqual } from "../platform/conditions";
import { getNodeDefs } from "../state/catalog";
import { exposedParams, readExposed, readOnly, useCookInputs, type CookNode, type Wire } from "../state/cookInputs";
import { enterPicking } from "../state/viewPicking";
import { graphIdForLoad } from "../model/graphId";
import { useLook, type Box } from "../state/look";
import { planHere, useResults } from "../state/results";
import { useViewer } from "../state/viewer";
import { licensedChoices, mainOutput, outputsOf, paramPortNames, tableRows } from "./rules";
import { cookSpan, parseSpan } from "./nodes";
import { snapshotNow } from "./snapshot";
import type { GNode } from "../state/graph";
import { msg, say, type Message } from "../state/say";
import { askStatus } from "./asking";
import { useItems } from "../state/items";
import { justWired } from "./judged";
import { loadOutputs } from "./outputs";
import { randomId } from "../platform/randomId";
import { leaveGraph } from "../transfer/uploads";
import { cache } from "../platform/cache";

// ------------------------------------------------------------------ history + dirty

const history = new History();
let statusTimer: ReturnType<typeof setTimeout> | null = null;

/** history.ts's flat Doc, built fresh from state/cookInputs.ts + state/look.ts. */
function docNow(): Doc {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const byNode = useResults.getState().byNode;
  const nodes: GNode[] = ci.order.map((id) => ({
    id, type: "l2s", position: look.positions[id] ?? { x: 0, y: 0 },
    data: { ...ci.nodes[id], onNode: look.onNode[id], status: byNode[id]?.status ?? "idle", note: byNode[id]?.note ?? "", blocked: byNode[id]?.blocked },
  }));
  return { meta: { ...ci.meta }, exposed: ci.exposed, nodes, edges: ci.edges.map(wireToEdge), boxes: look.boxes, displayId: look.displayId, displayPort: look.displayPort, cookRange: ci.cookRange };
}

function wireToEdge(w: Wire): Edge {
  return { id: w.id, source: w.source, sourceHandle: w.sourceHandle, target: w.target, targetHandle: w.targetHandle, type: "l2s" };
}

function scheduleStatusRefresh(): void {
  if (statusTimer) clearTimeout(statusTimer);
  statusTimer = setTimeout(() => void askStatus(), 250);
}

/** A status reply asked right now (graph/actions.ts currentReply) makes the scheduled one needless. */
export function stopStatusRefresh(): void {
  if (statusTimer) clearTimeout(statusTimer);
  statusTimer = null;
}

/** What the server's answer depends on: the cook inputs' version, the node shown and which of its outputs (the plan is judged for it). A move, a box, the rows shown on
 * a node change neither, so they ask nothing (a drag would otherwise ask again after every pause of the pointer). */
let askedFor = { version: -1, display: "" as string | null, port: null as string | null };

/** After any edit of state/cookInputs.ts or state/look.ts: record a history step (`as`: how, graph/history.ts
 * RecordAs; none for a plain edit), refresh dirty/undo/redo, and ask the server again (debounced) if the cook inputs or
 * the node shown moved. */
function noticed(as?: RecordAs): void {
  const doc = docNow();
  history.record(doc, getNodeDefs(), (id, port) => (port ? outputsOf(snapshotNow(), id).find((p) => p.name === port) : mainOutput(snapshotNow(), id)), as);
  // this write may hide the current value of an exposed pull-down (a parameter used in a condition changed): it falls to
  // the first listed item, joined into the step just recorded (absorb: one undo takes both back, and the server is asked
  // again as usual). What absorb itself writes is not checked again, so it cannot bounce back and forth
  if (!(typeof as === "object" && "into" in as)) settleHiddenChoices();
  // a value the user changed that is set to show its node (「修改后在视图里显示这个节点」): the display switches in the same
  // step (not a derived fill, not what absorb itself wrote)
  if (as === undefined || (typeof as === "object" && "label" in as)) showChanged();
  else watchShown();
  useViewer.getState().setSaveState(!history.isSaved(doc), history.labels.undoLabel, history.labels.redoLabel);
  const now = { version: useCookInputs.getState().version, display: useLook.getState().displayId, port: useLook.getState().displayPort };
  if (now.version === askedFor.version && now.display === askedFor.display && now.port === askedFor.port) return;
  askedFor = now;
  scheduleStatusRefresh();
}

/** 「修改后在视图里显示这个节点」 (an exposed parameter's show_on_change): its value (or, for a file parameter, the file
 * picked) as last seen, per exposed target. Whatever changes it (a field, a pull-down, a file or hierarchy pick, the rig
 * map's OK, a click in the view) is an edit, and the edit is where the display switches: here, joined into its step,
 * never an effect watching values afterwards (which would record a step of its own). Undo / redo and opening a document only
 * take the values in as seen (watchShown). */
let shownValues = new Map<string, Json>();
function watchedNow(): Map<string, Json> {
  const ci = useCookInputs.getState();
  const out = new Map<string, Json>();
  for (const x of exposedParams(ci.exposed)) {
    if (!x.show_on_change) continue;
    const [nid, pname] = x.target.split(".");
    const n = ci.nodes[nid];
    if (n) out.set(x.target, [n.params[pname] ?? null, n.picked?.[pname] ?? null] as Json);
  }
  return out;
}
const watchShown = (): void => void (shownValues = watchedNow());
function showChanged(): void {
  const now = watchedNow();
  const changed = [...now].find(([target, key]) => shownValues.has(target) && !same(shownValues.get(target), key));
  shownValues = now;
  if (changed) absorb(anchor(), () => enterPicking(changed[0].split(".")[0]));
}

/** An exposed pull-down whose current value is hidden by that item's Hide When gets the first listed item (the rule:
 * graph/exposedTree.ts hiddenChoices). */
function settleHiddenChoices(): void {
  const ci = useCookInputs.getState();
  const fixes = hiddenChoices(ci.exposed, exposedValues(ci.exposed, (id) => ci.nodes[id], getNodeDefs()));
  if (!fixes.length) return;
  absorb(anchor(), () => {
    for (const { x, now } of fixes) {
      const [nid, pname] = x.target.split(".");
      const n = useCookInputs.getState().nodes[nid];
      if (n) useCookInputs.getState().setNode(nid, { params: { ...n.params, [pname]: now } });
    }
  });
}

// applyDoc() (undo/redo) and loadGraph() write state/cookInputs.ts and state/look.ts in two `load()` calls: wrapped in
// `restoring` (graph/history.ts) so the half-written document in between is never recorded as a user edit (a spurious
// step would wipe the redo entry undo() just pushed); history.showing() / reset() follow.

// A plain user action is one step: it often writes both stores, or one store several times (addNode: the node, then
// its position; a deletion: its wires, the node, its position, a box), all in the same task. The step is recorded
// once, at the end of that task (a microtask), with everything the action wrote. `scheduled` only batches that one
// microtask; what a write IS (a named step, a derived fill, a restore) is never a flag here but the scope the writer
// wrapped it in (graph/history.ts transaction / absorb / derived / restoring), during which the stores' notices are
// not taken for plain edits.
let scheduled = false;
function noticeSoon(): void {
  if (writing() || scheduled) return;
  scheduled = true;
  queueMicrotask(() => {
    if (!scheduled) return; // already recorded by a scope that began in this task (flush)
    scheduled = false;
    noticed();
  });
}

bindRecorder({
  // a scope starts: whatever plain edits this task already made are their own step first, not swallowed by the scope
  flush: () => {
    if (!scheduled) return;
    scheduled = false;
    noticed();
  },
  record: (as) => noticed(as),
  top: () => history.top(),
  fillUndone: (into, node, values, was) => history.fillUndone(into, node, values, was),
});

let watching = false;
function watchStores(): void {
  if (watching) return;
  watching = true;
  useCookInputs.subscribe(noticeSoon);
  useLook.subscribe(noticeSoon);
}
watchStores();

/** Undo / redo write the document like any edit, so the read-only tab's gate is here, on the entry itself (the button,
 * the shortcut and anything later all come through): a tab another one took over never changes the document by its
 * history (applyDoc's load is a read-in and passes the stores' own gate). Taking editing back reads the newer copy in
 * (editor/autosave.ts takeOver), which starts a fresh history. */
export function undo(): void {
  if (readOnly()) return;
  const doc = history.undo();
  if (!doc) return;
  applyDoc(doc);
  history.showing(docNow());
  useViewer.getState().setSaveState(!history.isSaved(docNow()), history.labels.undoLabel, history.labels.redoLabel);
  scheduleStatusRefresh();
}

export function redo(): void {
  if (readOnly()) return;
  const doc = history.redo();
  if (!doc) return;
  applyDoc(doc);
  history.showing(docNow());
  useViewer.getState().setSaveState(!history.isSaved(docNow()), history.labels.undoLabel, history.labels.redoLabel);
  scheduleStatusRefresh();
}

function applyDoc(doc: Doc): void {
  const restored = restore({ ...docNow(), selectedId: useViewer.getState().selectedId }, doc);
  const nodes: Record<string, CookNode> = {};
  const positions: Record<string, { x: number; y: number }> = {};
  const onNode: Record<string, string[] | undefined> = {};
  for (const n of restored.nodes) {
    nodes[n.id] = { typeId: n.data.typeId, label: n.data.label, params: n.data.params, promoted: n.data.promoted, picked: n.data.picked, stored: n.data.stored };
    positions[n.id] = n.position;
    onNode[n.id] = n.data.onNode;
  }
  restoring(() => {
    useCookInputs.getState().load({ graphId: useCookInputs.getState().graphId, meta: doc.meta, exposed: doc.exposed ?? [], cookRange: doc.cookRange, nodes, order: restored.nodes.map((n) => n.id), edges: restored.edges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle ?? "", target: e.target, targetHandle: e.targetHandle ?? "" })), kept: useCookInputs.getState().kept });
    useLook.getState().load({ positions, onNode, boxes: doc.boxes, displayId: doc.displayId, displayPort: doc.displayPort, playback: useLook.getState().playback });
  });
  watchShown(); // values undo / redo brought back are not a change
  useViewer.setState({ selectedId: restored.selectedId });
}

/** The document as it is now, taken when it is serialized for its file (graph/graphFile.ts); calling the result once
 * the write has finished marks exactly that as the file's content. An edit made while the write was under way is not in
 * the file, so it stays unsaved. */
export function savePoint(): () => void {
  const doc = docNow();
  return () => {
    history.markSaved(doc);
    useViewer.getState().setSaveState(!history.isSaved(docNow()), history.labels.undoLabel, history.labels.redoLabel);
  };
}

// ------------------------------------------------------------------ loading, saving

/** What a graph file may lack, filled in at once (only `schema` is checked by graphFile.ts parse): a hand-written or old
 * graph without `nodes` / `edges` / `meta`, or a node without `ui` (position, picked files, the rows on its body), still
 * opens, instead of throwing when one item is read and the whole graph failing to open. A missing position is placed
 * at (0, 0) in loadGraph (NodeEditor's 「未知节点」 uses 0 as well). */
function filledIn(g: GraphJSON): GraphJSON {
  return { ...g, meta: { ...g.meta, name: g.meta?.name ?? "未命名" }, nodes: g.nodes ?? [], edges: g.edges ?? [] };
}

/** A graph as the current node definitions read it: parameters over their defaults, and whatever the definitions do
 * not have left out and listed. */
function checkGraph(g: GraphJSON, defs: Record<string, NodeTypeDef>) {
  const problems: Message[] = [];
  const label = (id: string) => g.nodes.find((n) => n.id === id)?.label || id;
  const gone = g.nodes.filter((n) => !defs[n.type]);
  const goneIds = new Set(gone.map((n) => n.id));
  const kept = { nodes: gone, edges: g.edges.filter((e) => goneIds.has(e.from[0]) || goneIds.has(e.to[0])) };
  g = { ...g, nodes: g.nodes.filter((n) => defs[n.type]) };
  const storedOf = new Map<string, Record<string, unknown>>();
  const loaded = new Map(
    g.nodes.map((n) => {
      const def = defs[n.type];
      // a graph file may have no `params` for a node (a hand-written graph, a template node using only defaults): taken
      // as an empty object, i.e. the defaults. Object.keys(n.params) on a missing item would throw and the whole
      // template card would fail to open.
      const own = n.params ?? {};
      const unknown = Object.keys(own).filter((k) => !(k in def.defaults));
      if (unknown.length) problems.push(msg("W-GRAPH-NOPARAMS", { node: label(n.id), names: unknown }));
      const given = { ...def.defaults, ...Object.fromEntries(Object.entries(own).filter(([k]) => k in def.defaults)) };
      const stored: Record<string, unknown> = {};
      for (const k of Object.keys(licensedChoices(def))) {
        const spec = def.params.find((p) => p.name === k);
        const allowed = spec?.options ? spec.options.some((o) => String(o) === String(given[k])) : given[k] === def.defaults[k];
        if (!allowed && k in own) (stored[k] = given[k]), (given[k] = def.defaults[k]);
      }
      if (Object.keys(stored).length) storedOf.set(n.id, stored);
      return [n.id, given];
    }),
  );
  const params = (n: GraphJSON["nodes"][number]) => loaded.get(n.id)!;
  const stored = (n: GraphJSON["nodes"][number]) => storedOf.get(n.id);
  const promotedOf = new Map(
    g.nodes.map((n) => {
      const names = n.promoted ?? [];
      const gone = names.filter((p) => !defs[n.type].param_ports[p]);
      if (gone.length) problems.push(msg("W-GRAPH-NOTWIRABLE", { node: label(n.id), names: gone }));
      return [n.id, names.filter((p) => defs[n.type].param_ports[p])];
    }),
  );
  const promoted = (n: GraphJSON["nodes"][number]) => promotedOf.get(n.id)!;
  // the list of parameter rows shown on a node is kept in the graph file's `ui.on_node` (the node's own data, like
  // `params`). A node without it uses its type's factory default (NodeDef.on_node, graph/nodes.ts chosenOnNode).
  const onNodeOf = new Map(
    g.nodes.flatMap((n) => {
      // `ui` may be missing altogether (the graph file records no position): then there is no list and the type's
      // declaration applies. A key missing from a file must not crash the page (template files are checked by
      // `lab2shot check templates`)
      if (!n.ui?.on_node) return [];
      const simple = new Set(defs[n.type].params.filter((p) => p.simple).map((p) => p.name));
      const gone = n.ui.on_node.filter((p) => !simple.has(p));
      if (gone.length) problems.push(msg("W-GRAPH-NOTONNODE", { node: label(n.id), names: gone }));
      return [[n.id, n.ui.on_node.filter((p) => simple.has(p))]];
    }),
  );
  const onNode = (n: GraphJSON["nodes"][number]) => onNodeOf.get(n.id);
  const hasPort = (id: string, port: string, side: "inputs" | "outputs") => {
    const n = g.nodes.find((m) => m.id === id);
    if (!n) return false;
    // a file's wires into inputs its node no longer has are taken out once, on opening it, so an old file still opens
    // (the server refuses such a graph whole: Graph._connect E-GRAPH-NOSUCHPORT). The names a file can wire into: the
    // declared inputs, a ports_from table's rows (「多层 EXR 输出设置」's 图层: a row's name is its port), the node type's
    // always-there parameter inputs (NodeDef.wired_ports) and the parameters this node promoted itself. Everything
    // else about ports is read from the status reply once the graph is in.
    if (side === "inputs") {
      const def = defs[n.type];
      const rows = tableRows(def, params(n)).map((r) => r.name);
      const fromParams = paramPortNames(def, promoted(n)).map((p) => def.param_ports[p].name);
      return [...def.inputs.map((p) => p.name), ...rows, ...fromParams].includes(port);
    }
    return true; // outputs vary with parameters: the status reply judges those wires once the graph is in the editor
  };
  const removed = new Set(gone.map((n) => n.id));
  const edges = g.edges.filter((e) => {
    if (removed.has(e.from[0]) || removed.has(e.to[0])) return false;
    const ok = hasPort(e.from[0], e.from[1], "outputs") && hasPort(e.to[0], e.to[1], "inputs");
    if (!ok) problems.push(msg("W-GRAPH-NOPORTS", { source: label(e.from[0]), node: label(e.to[0]), output: e.from[1], input: e.to[1] }));
    return ok;
  });
  return { nodes: g.nodes, params, stored, promoted, onNode, edges, problems, kept };
}


/** Another graph (a template, a file, a job's, a fresh document): the working state a new document starts with.
 * `docId` given only when resuming this browser's last working state (editor/autosave.ts). `opts.freshId`: this document is a
 * new copy of whatever `g` was (a template opened again and again); it never keeps `g`'s own id, even
 * when `g.meta.id` is set (a template file's id exists only so the file itself always has one; every graph made
 * from it gets its own, so two graphs from the same template never share one). */
export function loadGraph(g: GraphJSON, file: { name: string; handle?: string } | null, dirty = false, docId?: string, opts?: { freshId?: boolean }): void {
  g = filledIn(g);
  const defs = getNodeDefs();
  const checked = checkGraph(g, defs);
  const nodes: Record<string, CookNode> = {};
  const positions: Record<string, { x: number; y: number }> = {};
  const onNode: Record<string, string[] | undefined> = {};
  const order: string[] = [];
  for (const n of checked.nodes) {
    order.push(n.id);
    nodes[n.id] = { typeId: n.type, label: n.label || defs[n.type]?.label || n.type, params: checked.params(n), promoted: checked.promoted(n).length ? checked.promoted(n) : undefined, picked: n.ui?.picked, stored: checked.stored(n) };
    positions[n.id] = { x: n.ui?.x ?? 0, y: n.ui?.y ?? 0 };
    onNode[n.id] = checked.onNode(n);
  }
  const edges: Wire[] = checked.edges.map((e) => ({ id: `${e.from[0]}.${e.from[1]}->${e.to[0]}.${e.to[1]}`, source: e.from[0], sourceHandle: e.from[1], target: e.to[0], targetHandle: e.to[1] }));
  // said once the graph is in (the results are cleared on the way)
  const said: Message[] = [];
  // a file's exposed pull-down value is the very item hidden under the current conditions: it falls to the first listed
  // item, this is said, and the document counts as changed (saving writes it in)
  const exposed = readExposed(g.exposed);
  for (const { x, was, now } of hiddenChoices(exposed, exposedValues(exposed, (id) => nodes[id], defs))) {
    const [nid, pname] = x.target.split(".");
    nodes[nid] = { ...nodes[nid], params: { ...nodes[nid].params, [pname]: now } };
    const label = (v: unknown) => x.options?.find((o) => condEqual(o.value, v))?.label ?? JSON.stringify(v);
    said.push(msg("N-EXPOSED-HIDDENVALUE", { label: x.label, was: label(was), now: label(now) }));
    dirty = true;
  }
  if (checked.problems.length) {
    dirty = true;
    said.push(msg("W-GRAPH-MISMATCH", { graph: g.meta.name, count: checked.problems.length, problems: checked.problems }));
  }
  if (checked.kept.nodes.length) said.push(msg("W-GRAPH-UNKNOWNNODES", { graph: g.meta.name, count: checked.kept.nodes.length }));

  const view: Partial<GraphJSON["view"]> = g.view ?? {};
  const display = view.display && order.includes(view.display) ? view.display : (order.at(-1) ?? null);
  const boxes: Box[] = (g.boxes ?? []).map((b) => ({ ...b, members: b.members ?? [] }));
  const cookRange: [string, string] | null = g.frames ? [String(g.frames[0]), String(g.frames[1])] : null;

  // meta.id: generated for a graph whose file has none, or that must never share the one it was opened from
  // (opts.freshId: a template; 另存为 gives its copy a new id in graph/graphFile.ts). There is no compatibility layer:
  // it is fixed up once, here, and the document counts as changed so saving writes the id in.
  const { id: graphId, generated } = graphIdForLoad(g.meta.id, opts?.freshId);
  // Only a file that lacks its id has something to write back; a new copy (a template) has no file yet and its
  // first save writes the id anyway. It opens unchanged, so an edit that is undone brings it back to unchanged.
  if (generated && !opts?.freshId) dirty = true;
  const { id: _unusedId, ...meta } = g.meta;
  void _unusedId;

  useResults.getState().reset();
  // another graph: nothing draws the previous graph's decoded bitmaps now, so the decoded tier is released (the compressed
  // bytes stay: reopening decodes again in milliseconds)
  cache.releasePixels();
  useItems.getState().reset(); // another graph: no block of the old one is being shown any more (state/items.ts)
  justWired.clear();
  // upload tasks, local files and local proxy progress belong to a document (graphId): those not of this document are not
  // shown (transfer/uploads.ts). Called before load(): it is how a graphId change is told to be opening another
  // document rather than 另存为
  leaveGraph(graphId);
  restoring(() => { // two separate store writes below, one document
    // a graph file without the exposed-parameters item counts as empty: a missing one would blank the whole parameter
    // panel. An older file's flat list is a parameter interface tree without groups and is read in as it is; saving
    // writes the tree (state/cookInputs.ts readExposed)
    useCookInputs.getState().load({ graphId, meta, exposed, cookRange, nodes, order, edges, kept: checked.kept });
    useLook.getState().load({ positions, onNode, boxes, displayId: display, displayPort: view.port ?? null, playback: view.playback ?? null });
  });
  watchShown(); // the opened document's values are where watching starts
  useViewer.getState().reset();
  useHandleView.getState().reset();
  useViewer.setState({ file, docId: docId ?? randomId(), role: "editor", peerBanner: null });
  if (view.frame) useViewer.setState({ frame: view.frame });
  history.reset(docNow(), !dirty);
  // the unsaved mark is the history's answer from the start (a file fixed up on load shows it at once), never a
  // separate flag that a later edit would contradict
  useViewer.getState().setSaveState(!history.isSaved(docNow()), history.labels.undoLabel, history.labels.redoLabel);
  noticed();
  for (const m of said) say(m);
  void loadOutputs(); // what its 「输出」 packed before, still on the server
}

/** The graph as sent to the server (toJSON) or kept in its file (fileJSON: adds back the 「未知节点」 kept as they
 * were, and the file's own values for choices this account may not use). */
export function toJSON(): GraphJSON {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const viewer = useViewer.getState();
  return {
    schema: "lab2shot.graph/1",
    meta: { ...ci.meta, id: ci.graphId },
    exposed: ci.exposed,
    // the server gets the range actually submitted (intersected with the footage, graph/nodes.ts cookSpan); fileJSON,
    // for the file, puts back what the user typed
    frames: cookSpan(ci.cookRange, planHere()),
    nodes: ci.order.map((id) => {
      const n = ci.nodes[id];
      const pos = look.positions[id] ?? { x: 0, y: 0 };
      return {
        id, type: n.typeId, label: n.label, params: n.params,
        ...(n.promoted?.length ? { promoted: n.promoted } : {}),
        ui: { x: Math.round(pos.x), y: Math.round(pos.y), ...(n.picked && Object.keys(n.picked).length ? { picked: n.picked } : {}), ...(look.onNode[id] ? { on_node: look.onNode[id] } : {}) },
      };
    }),
    edges: ci.edges.map((e) => ({ from: [e.source, e.sourceHandle], to: [e.target, e.targetHandle] })),
    boxes: look.boxes.map((b) => ({ ...b, x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.w), h: Math.round(b.h) })),
    view: { display: look.displayId, frame: viewer.frame, port: look.displayPort, playback: look.playback },
  };
}

export function fileJSON(): GraphJSON {
  const ci = useCookInputs.getState();
  const g = toJSON();
  const own = Object.entries(ci.nodes).filter(([, n]) => n.stored).map(([id, n]) => [id, n.stored!] as const);
  const ownMap = new Map(own);
  g.nodes = g.nodes.map((n) => (ownMap.has(n.id) ? { ...n, params: { ...n.params, ...ownMap.get(n.id) } } : n));
  const nodes = [...g.nodes, ...ci.kept.nodes.filter((n) => !g.nodes.some((m) => m.id === n.id))];
  const there = new Set(nodes.map((n) => n.id));
  const typed = parseSpan(ci.cookRange);
  return { ...g, frames: typed && "span" in typed ? typed.span : null, nodes, edges: [...g.edges, ...ci.kept.edges.filter((e) => there.has(e.from[0]) && there.has(e.to[0]))] };
}


