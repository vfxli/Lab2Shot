/** The open graph as a document: undo and redo over state/cookInputs.ts and
 * state/look.ts, the unsaved mark, loading a graph file (checked and fixed up) and writing one. graph/actions.ts
 * re-exports what components call. */

import type { Edge } from "@xyflow/react";
import type { GraphJSON, NodeTypeDef } from "../api";
import { History, restore, type Doc } from "./history";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs, type CookNode, type Wire } from "../state/cookInputs";
import { graphIdForLoad } from "../model/graphId";
import { useLook, type Box } from "../state/look";
import { useResults } from "../state/results";
import { useViewer } from "../state/viewer";
import { mainOutput, noncommercialChoices, outputsOf, paramPortNames, tableRows } from "./rules";
import { snapshotNow } from "./snapshot";
import type { GNode } from "../state/graph";
import { msg, say, type Message } from "../state/say";
import { refreshStatus } from "./actions";
import { useItems } from "../state/items";
import { justWired } from "./actions";
import { loadOutputs } from "./outputs";
import { randomId } from "../platform/randomId";

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
  statusTimer = setTimeout(() => void refreshStatus(), 250);
}

/** A status reply asked right now (graph/actions.ts currentReply) makes the scheduled one needless. */
export function stopStatusRefresh(): void {
  if (statusTimer) clearTimeout(statusTimer);
  statusTimer = null;
}

/** What the server's answer depends on: the cook inputs' version and the node shown. A move, a box, the rows shown on
 * a node change neither, so they ask nothing (a drag would otherwise ask again after every pause of the pointer). */
let askedFor = { version: -1, display: "" as string | null };

/** After any edit of state/cookInputs.ts or state/look.ts: record a history step, refresh dirty/undo/redo, and ask
 * the server again (debounced) if the cook inputs or the node shown moved. */
function noticed(): void {
  const doc = docNow();
  if (history.record(doc, getNodeDefs(), (id, port) => (port ? outputsOf(snapshotNow(), id).find((p) => p.name === port) : mainOutput(snapshotNow(), id)))) {
    useViewer.getState().setSaveState(!history.isSaved(doc), history.labels.undoLabel, history.labels.redoLabel);
  }
  const now = { version: useCookInputs.getState().version, display: useLook.getState().displayId };
  if (now.version === askedFor.version && now.display === askedFor.display) return;
  askedFor = now;
  scheduleStatusRefresh();
}

// applyDoc() (undo/redo) and loadGraph() each restore state/cookInputs.ts and state/look.ts with two separate
// `load()` calls, not one atomic write. Between them, this store's own subscription would see a "document changed"
// that history.ts's `this.current` (only updated by the explicit history.showing()/reset() call that follows) does
// not yet know about, and record it as a new user edit, pushing a spurious undo step and wiping the redo
// entry undo() just pushed. Suppressing noticed() for the duration of that restore (both stores are consistent
// again before undo()/redo()/loadGraph() call history.showing()/reset() themselves) keeps "three undos"
// undoing exactly three steps, without corrupting the stack.
let restoring = false;

// One user action is one step: an action often writes both stores, or one store several times (addNode: the node, then
// its position; a deletion: its wires, the node, its position, a box), and each write would otherwise be recorded as
// a step of its own (「添加节点」 then 「移动节点」 for one Tab-menu pick). The writes of one action all happen in the same
// task, so the step is recorded once, at its end, with everything the action wrote.
let noticing = false;
function noticeSoon(): void {
  if (restoring || noticing) return;
  noticing = true;
  queueMicrotask(() => {
    noticing = false;
    noticed();
  });
}

let watching = false;
function watchStores(): void {
  if (watching) return;
  watching = true;
  useCookInputs.subscribe(noticeSoon);
  useLook.subscribe(noticeSoon);
}
watchStores();

export function undo(): void {
  const doc = history.undo();
  if (!doc) return;
  applyDoc(doc);
  history.showing(docNow());
  useViewer.getState().setSaveState(!history.isSaved(docNow()), history.labels.undoLabel, history.labels.redoLabel);
  scheduleStatusRefresh();
}

export function redo(): void {
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
  restoring = true;
  try {
    useCookInputs.getState().load({ graphId: useCookInputs.getState().graphId, meta: doc.meta, exposed: doc.exposed ?? [], cookRange: doc.cookRange, nodes, order: restored.nodes.map((n) => n.id), edges: restored.edges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle ?? "", target: e.target, targetHandle: e.targetHandle ?? "" })), kept: useCookInputs.getState().kept });
    useLook.getState().load({ positions, onNode, boxes: doc.boxes, displayId: doc.displayId, displayPort: doc.displayPort, playback: useLook.getState().playback });
  } finally {
    restoring = false;
  }
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
      // 节点图文件中可以没有 params 项（手写的图、模板中全部使用默认值的节点）：视为空对象，按默认值处理。
      // 直接调用 Object.keys(n.params) 会在缺项时抛错，导致整张模板卡无法打开。
      const own = n.params ?? {};
      const unknown = Object.keys(own).filter((k) => !(k in def.defaults));
      if (unknown.length) problems.push(msg("W-GRAPH-NOPARAMS", { node: label(n.id), names: unknown }));
      const given = { ...def.defaults, ...Object.fromEntries(Object.entries(own).filter(([k]) => k in def.defaults)) };
      const stored: Record<string, unknown> = {};
      for (const k of Object.keys(noncommercialChoices(def))) {
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
  // 节点上显示的参数行名单保存在节点图文件的 `ui.on_node` 中（与 `params` 同属该节点自身的数据）。
  // 未写该项的节点按节点类型的出厂默认处理（NodeDef.on_node，graph/nodes.ts chosenOnNode）。
  const onNodeOf = new Map(
    g.nodes.flatMap((n) => {
      // `ui` 整体可以缺失（节点图文件未记录位置）：此时没有该名单，按类型声明处理。
      // 文件缺少某个键不应导致页面崩溃（模板文件由 `lab2shot check templates` 检查）
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
  const defs = getNodeDefs();
  const checked = checkGraph(g, defs);
  const nodes: Record<string, CookNode> = {};
  const positions: Record<string, { x: number; y: number }> = {};
  const onNode: Record<string, string[] | undefined> = {};
  const order: string[] = [];
  for (const n of checked.nodes) {
    order.push(n.id);
    nodes[n.id] = { typeId: n.type, label: n.label || defs[n.type]?.label || n.type, params: checked.params(n), promoted: checked.promoted(n).length ? checked.promoted(n) : undefined, picked: n.ui.picked, stored: checked.stored(n) };
    positions[n.id] = { x: n.ui.x, y: n.ui.y };
    onNode[n.id] = checked.onNode(n);
  }
  const edges: Wire[] = checked.edges.map((e) => ({ id: `${e.from[0]}.${e.from[1]}->${e.to[0]}.${e.to[1]}`, source: e.from[0], sourceHandle: e.from[1], target: e.to[0], targetHandle: e.to[1] }));
  // said once the graph is in (the results are cleared on the way)
  const said: Message[] = [];
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
  useItems.getState().reset(); // another graph: no block of the old one is being shown any more (state/items.ts)
  justWired.clear();
  restoring = true; // two separate store writes below, one document (see the comment on `restoring`)
  try {
    // 节点图文件未写「对外参数」项时按空处理：缺失会导致参数面板整体空白
    useCookInputs.getState().load({ graphId, meta, exposed: g.exposed ?? [], cookRange, nodes, order, edges, kept: checked.kept });
    useLook.getState().load({ positions, onNode, boxes, displayId: display, displayPort: view.port ?? null, playback: view.playback ?? null });
  } finally {
    restoring = false;
  }
  useViewer.getState().reset();
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
    frames: rangeNumbers(ci.cookRange),
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
  return { ...g, nodes, edges: [...g.edges, ...ci.kept.edges.filter((e) => there.has(e.from[0]) && there.has(e.to[0]))] };
}

function wholeNumbers(r: [string, string]): [number, number] | null {
  return r.every((v) => /^-?\d+$/.test(v.trim())) ? [Number(r[0]), Number(r[1])] : null;
}
function rangeNumbers(r: [string, string] | null): [number, number] | null {
  const nums = r && wholeNumbers(r);
  return nums && nums[0] <= nums[1] ? nums : null;
}
