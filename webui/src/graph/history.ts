import type { Edge } from "@xyflow/react";
import type { ExposedParam, GraphJSON, NodeTypeDef } from "../api";
import { chosenOnNode } from "./nodes";
import { entryLabel, exposedParams, targetsOf } from "../state/cookInputs";
import type { GBox, GNode } from "../state/graph";
import { same as sameJson, type Json } from "../model/graphPatch";
import { t } from "../i18n/t";
import { boxLabel } from "./naming";

/** A step's name in words, said when it is shown (so it follows the page's language). */
export type Said = () => string;

/** One of a node's outputs as the editor has them now (graph/rules.ts): the named port, or, when `port` is null, the node's
 * main result (graph/rules.ts mainOutput). Names a shown port in a step's label. */
type OutputOf = (nodeId: string, port: string | null) => { name: string; label: string } | undefined;

/** Undo and redo of the node graph. What is undone is the document: what the graph file holds, apart
 * from the current frame: nodes with their parameters, names, positions, picked files, save folders, the parameters
 * they have 提升到节点 (each of those is an input of its own: `promoted`), the parameters their bodies show
 * (`onNode`), wires, group
 * boxes, exposed parameters, the display node, the frame range to cook and the graph's meta. Selection, the view, viewer
 * settings, panel sizes and cook results are not part of it.
 *
 * The history watches the document, not the actions that change it: whatever changes it is a step, named after what
 * changed. Changes made by one piece of code (one event) are one step, and the same thing changing again within one
 * gesture (a drag, a slider, typing into a field) grows the step it started. A gesture ends at the next pointer press
 * or keyboard focus change. A step is the document before and after it; the state is immutable, so steps share
 * everything they did not change. Opening a graph starts a fresh history.
 *
 * A write that is not a plain user edit says what it is by the scope it runs in (transaction / absorb / derived /
 * restoring, below): the one mechanism for named steps, derived values and restores. */

export interface Doc {
  meta: GraphJSON["meta"];
  exposed: GraphJSON["exposed"];
  nodes: GNode[];
  edges: Edge[];
  boxes: GBox[];
  displayId: string | null;
  displayPort: string | null;
  cookRange: [string, string] | null;
}

interface Step {
  before: Doc;
  after: Doc;
  label: Said; // what it did, for the undo / redo tooltips
  key: string; // what it changed
  gesture: number;
}

const LIMIT = 200;

// A gesture runs from a pointer press (with the focus change the press brings: a slider takes focus after its first
// change) to the next press, or to a focus change made from the keyboard.
let gesture = 0;
let pressed = false;
if (typeof window !== "undefined") {
  window.addEventListener("pointerdown", () => ((pressed = true), gesture++), true);
  for (const type of ["pointerup", "pointercancel"]) window.addEventListener(type, () => (pressed = false), true);
  for (const type of ["focusin", "focusout"]) window.addEventListener(type, () => pressed || gesture++, true);
}

const EMPTY_DOC: Doc = { meta: { name: "" }, exposed: [], nodes: [], edges: [], boxes: [], displayId: null, displayPort: null, cookRange: null };

// the document's values are JSON (what a graph file holds): compared as trees, never as strings of them
const same = (a: unknown, b: unknown) => sameJson(a as Json, b as Json);

const changedParams = (a: Record<string, unknown>, b: Record<string, unknown>) =>
  a === b ? [] : [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((k) => !same(a[k], b[k]));

const sameNode = (a: GNode, b: GNode | undefined) =>
  a === b ||
  (!!b &&
    a.data.typeId === b.data.typeId &&
    same(a.data.comment ?? null, b.data.comment ?? null) &&
    a.position.x === b.position.x &&
    a.position.y === b.position.y &&
    !changedParams(a.data.params, b.data.params).length &&
    same(a.data.picked, b.data.picked) &&
    same(a.data.stored ?? null, b.data.stored ?? null) && // the file's own values for choices this account may not use
    same(a.data.promoted ?? [], b.data.promoted ?? []) &&
    same(a.data.onNode ?? null, b.data.onNode ?? null));

const sameBox = (a: GBox, b: GBox | undefined) =>
  a === b ||
  (!!b && a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h && same(a.label, b.label) && a.color === b.color && a.collapsed === b.collapsed && same(a.members, b.members));

const byId = <T extends { id: string }>(list: T[]) => new Map(list.map((x) => [x.id, x]));

function sameList<T extends { id: string }>(a: T[], b: T[], eq: (x: T, y: T | undefined) => boolean): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  const other = byId(b);
  return a.every((x) => eq(x, other.get(x.id)));
}

/** One thing an absorb wrote: a node's parameter, or a document field (the node shown, the range, meta, exposed). That
 * is all an absorb writes (derived parameters, a hidden pull-down's fill, the view entering picking). */
type Fill = { node: string; key: string; v: unknown } | { field: (typeof FIELDS)[number]; v: unknown };
const FIELDS = ["meta", "exposed", "displayId", "displayPort", "cookRange"] as const;

/** What an absorb wrote, from the document before it to the one after. */
function fillsOf(from: Doc, to: Doc): Fill[] {
  const was = byId(from.nodes);
  const out: Fill[] = [];
  for (const n of to.nodes) {
    const a = was.get(n.id);
    if (a) for (const k of changedParams(a.data.params, n.data.params)) out.push({ node: n.id, key: k, v: n.data.params[k] });
  }
  for (const f of FIELDS) if (!same(from[f], to[f])) out.push({ field: f, v: to[f] });
  return out;
}

const valueIn = (d: Doc, f: Fill): unknown => ("field" in f ? d[f.field] : d.nodes.find((n) => n.id === f.node)?.data.params[f.key]);

/** Whether a step itself changed what `f` fills (the user set that derived parameter by hand in it). */
const touches = (x: Step, f: Fill): boolean => !same(valueIn(x.before, f), valueIn(x.after, f));

/** `fills` laid onto a document of the history; a node it does not have is left out. */
function lay(onto: Doc, fills: Fill[]): Doc {
  if (!fills.length) return onto;
  const byNode = new Map<string, [string, unknown][]>();
  const out: Doc = { ...onto };
  for (const f of fills) {
    if ("field" in f) (out as unknown as Record<string, unknown>)[f.field] = f.v;
    else byNode.set(f.node, [...(byNode.get(f.node) ?? []), [f.key, f.v]]);
  }
  if (byNode.size) out.nodes = onto.nodes.map((n) => {
    const f = byNode.get(n.id);
    if (!f) return n;
    const params = { ...n.data.params };
    for (const [k, v] of f) if (v === undefined) delete params[k]; else params[k] = v;
    return { ...n, data: { ...n.data, params } };
  });
  return out;
}

const sameDoc = (a: Doc, b: Doc) =>
  a.displayId === b.displayId &&
  a.displayPort === b.displayPort &&
  same(a.cookRange, b.cookRange) &&
  same(a.meta, b.meta) &&
  same(a.exposed, b.exposed) &&
  sameList(a.nodes, b.nodes, sameNode) &&
  sameList(a.edges, b.edges, (_, y) => !!y) && // a wire's id names its ends
  sameList(a.boxes, b.boxes, sameBox);

/** What changed from `a` to `b`: in words, and which things (`key`). */
function describe(a: Doc, b: Doc, defs: Record<string, NodeTypeDef>, outputOf: OutputOf): { label: Said; key: string } {
  const was = byId(a.nodes);
  const now = byId(b.nodes);
  const ids = (list: { id: string }[]) => list.map((x) => x.id).join(",");
  const count = <T,>(list: T[], one: (x: T) => string, several: (count: number) => string): Said => () => (list.length === 1 ? one(list[0]) : several(list.length));
  const step = (label: Said, key: string) => ({ label, key });
  const paramLabel = (n: GNode, k: string) => defs[n.data.typeId]?.params.find((p) => p.name === k)?.label ?? k;

  const added = b.nodes.filter((n) => !was.has(n.id));
  if (added.length) return step(count(added, (n) => t("ui.history.node_add", { node: n.id }), (count) => t("ui.history.nodes_add", { count })), `node+${ids(added)}`);
  const removed = a.nodes.filter((n) => !now.has(n.id));
  if (removed.length) return step(count(removed, (n) => t("ui.history.node_delete", { node: n.id }), (count) => t("ui.history.nodes_delete", { count })), `node-${ids(removed)}`);

  const boxWas = byId(a.boxes);
  const boxNow = byId(b.boxes);
  const newBoxes = b.boxes.filter((x) => !boxWas.has(x.id));
  if (newBoxes.length) return step(count(newBoxes, (x) => t("ui.history.box_add", { box: boxLabel(x) }), (count) => t("ui.history.boxes_add", { count })), `box+${ids(newBoxes)}`);
  const goneBoxes = a.boxes.filter((x) => !boxNow.has(x.id));
  if (goneBoxes.length) return step(count(goneBoxes, (x) => t("ui.history.box_delete", { box: boxLabel(x) }), (count) => t("ui.history.boxes_delete", { count })), `box-${ids(goneBoxes)}`);

  // 提升到节点: the parameter gets a row on the node with an input of its own, or loses both (with its wire: one step)
  for (const n of b.nodes) {
    const before = was.get(n.id)?.data.promoted ?? [];
    const now = n.data.promoted ?? [];
    const k = now.find((p) => !before.includes(p)) ?? before.find((p) => !now.includes(p));
    if (k) {
      const on = now.includes(k);
      return step(() => t(on ? "ui.history.promote" : "ui.history.unpromote", { node: n.id, param: paramLabel(n, k) }), `promote ${n.id}.${k}`);
    }
  }

  const edgesWas = new Set(a.edges.map((e) => e.id));
  const edgesNow = new Set(b.edges.map((e) => e.id));
  const joined = b.edges.filter((e) => !edgesWas.has(e.id));
  if (joined.length) return step(count(joined, (e) => t("ui.history.wire_add", { source: e.source, target: e.target }), (count) => t("ui.history.wires_add", { count })), `wire+${ids(joined)}`);
  const cut = a.edges.filter((e) => !edgesNow.has(e.id));
  if (cut.length) return step(count(cut, (e) => t("ui.history.wire_cut", { source: e.source, target: e.target }), (count) => t("ui.history.wires_cut", { count })), `wire-${ids(cut)}`);

  const edited = b.nodes.filter((n) => !sameNode(n, was.get(n.id)));
  const params = edited.flatMap((n) => changedParams(was.get(n.id)!.data.params, n.data.params).map((k) => [n, k] as const));
  if (params.length) {
    const [n, k] = params[0];
    const touched = new Set(params.map(([m]) => m.id));
    const label: Said = () =>
      params.length === 1
        ? t("ui.history.param", { node: n.id, param: paramLabel(n, k) })
        : touched.size === 1
          ? t("ui.history.params", { node: n.id })
          : t("ui.history.params_nodes", { count: touched.size });
    return step(label, `param ${params.map(([m, q]) => `${m.id}.${q}`).join(",")}`);
  }
  // a node's comment (renaming a node is a named step of its own: graph/edit.ts renameNode)
  const commented = edited.filter((n) => !same(n.data.comment ?? null, was.get(n.id)!.data.comment ?? null));
  if (commented.length) {
    const n = commented[0];
    const [p, q] = [was.get(n.id)!.data.comment, n.data.comment];
    const label: Said = () => (p && q && p.text === q.text ? t(q.show ? "ui.history.comment_show" : "ui.history.comment_hide", { node: n.id }) : t("ui.history.comment_edit", { node: n.id }));
    return step(label, `comment ${ids(commented)}`);
  }
  const picked = edited.filter((n) => !same(n.data.picked, was.get(n.id)!.data.picked)); // the same files from another folder
  if (picked.length) return step(count(picked, (n) => t("ui.history.repick", { node: n.id }), (count) => t("ui.history.repick_nodes", { count })), `picked ${ids(picked)}`);
  // 在节点上显示 on / off (a parameter row enters or leaves the node; 提升到节点 is its own step, above)
  for (const n of edited) {
    const def = defs[n.data.typeId];
    const before = def ? chosenOnNode(def, was.get(n.id)!.data.onNode) : [];
    const after = def ? chosenOnNode(def, n.data.onNode) : [];
    const k = after.find((p) => !before.includes(p)) ?? before.find((p) => !after.includes(p));
    if (k) {
      const on = after.includes(k);
      return step(() => t(on ? "ui.history.on_node" : "ui.history.off_node", { node: n.id, param: paramLabel(n, k) }), `onNode ${n.id}.${k}`);
    }
  }
  const boxes = b.boxes.filter((x) => !sameBox(x, boxWas.get(x.id))).map((x) => [boxWas.get(x.id)!, x] as const);
  for (const [p, x] of boxes) {
    if (!same(p.label, x.label)) return step(() => t("ui.history.box_rename", { box: boxLabel(x) }), `boxLabel ${x.id}`);
    if (p.color !== x.color) return step(() => t("ui.history.box_color", { box: boxLabel(x) }), `boxColor ${x.id}`);
    if (p.collapsed !== x.collapsed) return step(() => t(x.collapsed ? "ui.history.box_fold" : "ui.history.box_unfold", { box: boxLabel(x) }), `boxFold ${x.id}`);
    if (p.w !== x.w || p.h !== x.h) return step(() => t("ui.history.box_size", { box: boxLabel(x) }), `boxSize ${x.id}`);
  }
  const movedBoxes = boxes.filter(([p, x]) => p.x !== x.x || p.y !== x.y).map(([, x]) => x);
  const moved = edited.filter((n) => n.position.x !== was.get(n.id)!.position.x || n.position.y !== was.get(n.id)!.position.y);
  if (movedBoxes.length || moved.length) {
    const label = movedBoxes.length ? count(movedBoxes, (x) => t("ui.history.box_move", { box: boxLabel(x) }), (count) => t("ui.history.boxes_move", { count })) : count(moved, (n) => t("ui.history.node_move", { node: n.id }), (count) => t("ui.history.nodes_move", { count }));
    return step(label, `move ${ids(movedBoxes)} ${ids(moved)}`);
  }

  // the parameter interface is a tree (groups / parameter items): exposing and unexposing compare its parameter items;
  // every other change (grouping, order, display, conditions) is 「编辑参数界面」
  const [pinsWas, pinsNow] = [exposedParams(a.exposed), exposedParams(b.exposed)];
  const sameTargets = (x: ExposedParam, y: ExposedParam) => targetsOf(x).join() === targetsOf(y).join();
  const pinned = pinsNow.filter((x) => !pinsWas.some((y) => sameTargets(x, y)));
  if (pinned.length) return step(count(pinned, (x) => t("ui.history.expose", { param: entryLabel(x) }), (count) => t("ui.history.expose_many", { count })), `expose+${pinned.map((x) => targetsOf(x).join("+"))}`);
  const unpinned = pinsWas.filter((x) => !pinsNow.some((y) => sameTargets(x, y)));
  if (unpinned.length) return step(count(unpinned, (x) => t("ui.history.unexpose", { param: entryLabel(x) }), (count) => t("ui.history.unexpose_many", { count })), `expose-${unpinned.map((x) => targetsOf(x).join("+"))}`);
  if (!same(a.exposed, b.exposed)) return reordered(a.exposed, b.exposed) ? step(() => t("ui.history.interface_order"), "order") : step(() => t("ui.history.interface_edit"), "interface");

  const display = b.displayId;
  if (a.displayId !== display) return step(() => (display ? t("ui.history.display", { node: display }) : t("ui.history.display_none")), "display");
  if (a.displayPort !== b.displayPort && display) {
    const port = outputOf(display, b.displayPort);
    const shown = port?.label ?? b.displayPort ?? "";
    return step(() => t("ui.history.display_port", { node: display, port: shown }), "port");
  }
  const range = b.cookRange;
  if (!same(a.cookRange, range)) return step(() => (range ? t("ui.history.range", { first: range[0], last: range[1] }) : t("ui.history.range_all")), "range");
  if (!same(a.meta, b.meta)) return step(() => t("ui.history.meta"), "meta");
  return step(() => t("ui.history.graph"), "graph");
}

/** Only the order and the grouping of the parameter interface changed (a drag in the panel or the editor): the same
 * parameter items and the same groups (by name and settings), placed differently. */
function reordered(a: Doc["exposed"], b: Doc["exposed"]): boolean {
  const parts = (tree: Doc["exposed"]) => {
    const out: string[] = [];
    const walk = (xs: Doc["exposed"]) => xs.forEach((x) => ("kind" in x && x.kind === "group" ? (out.push(JSON.stringify({ ...x, children: null })), walk(x.children)) : out.push(JSON.stringify(x))));
    walk(tree);
    return out.sort().join("\n");
  };
  return parts(a) === parts(b);
}

/** The state that shows `d`: the document's own fields from it, everything else (selection, sizes, cook status) kept
 * from the state as it is. Nodes and boxes the step did not touch stay the very same objects. */
export function restore(s: Doc & { selectedId: string | null }, d: Doc): Doc & { selectedId: string | null } {
  const nodesNow = byId(s.nodes);
  const nodes = d.nodes.map((n): GNode => {
    const cur = nodesNow.get(n.id);
    if (cur && sameNode(cur, n)) return cur;
    return {
      ...n,
      selected: cur?.selected ?? false,
      dragging: false,
      measured: cur?.measured ?? n.measured,
      data: { ...n.data, status: cur?.data.status ?? "idle", note: cur?.data.note ?? "", blocked: undefined },
    };
  });
  const edgesNow = byId(s.edges);
  const boxesNow = byId(s.boxes);
  return {
    ...d,
    nodes,
    edges: d.edges.map((e) => edgesNow.get(e.id) ?? { ...e, selected: false }),
    boxes: d.boxes.map((x) => {
      const cur = boxesNow.get(x.id);
      return cur && sameBox(cur, x) ? cur : { ...x, selected: cur?.selected ?? false };
    }),
    selectedId: nodes.some((n) => n.id === s.selectedId) ? s.selectedId : null,
  };
}

// ------------------------------------------------------------------ how a write enters the document: explicit scopes

/** What a write counts as in the undo history. A plain user edit needs no wrapping: the writes of both stores within one
 * task make one step, named after what changed. Every other write binds "the write" and "what it counts as" together by
 * being wrapped in one of the functions below:
 * - `transaction(label, key, fn)`: what fn writes is one step under this name (「整组改接」, 「整理节点图」: one change
 *   of many things, which a name built from the content would not fit);
 * - `absorb(into, fn)`: what fn writes joins the current document and also the step `into` (taken with `anchor()` when
 *   the cause was written) — it is a consequence of that step (the distortion parameters the server derives after
 *   「镜头模型」 changes, answered after the user went on to other edits), and undo / redo carry it along. The steps after
 *   `into` get it too, so undoing a later one does not take it back;
 * - `derived(fn)`: what fn writes joins only the current document, belongs to no step and does not mark it unsaved
 *   (filling an empty 「色彩空间」 from the file format: recorded as a step, undo would empty it and it would be filled
 *   again, so undo would never work; an old graph would also show as unsaved the moment it opens);
 * - `restoring(fn)`: undo / redo / opening a document replace both stores whole and record nothing (the caller then
 *   calls showing / reset itself). */
export type RecordAs = { label: Said; key: string } | { into: Anchor } | "derived";
/** A step of the history, as `absorb` names it (opaque outside this file); null: none (nothing recorded yet). */
export type Anchor = object | null;
type Scope = RecordAs | "restore";

/** The side that records the document (graph/document.ts: it has the two stores and the History instance): first the
 * plain edits this task already wrote and not yet recorded become a step of their own (flush), then what fn wrote is
 * recorded by `as`. */
interface Recorder {
  flush(): void;
  top(): Anchor;
  fillUndone(into: Anchor, node: string, values: Record<string, unknown>, was: Record<string, unknown>): void;
  record(as: RecordAs): void;
}
let recorder: Recorder | null = null;
export function bindRecorder(r: Recorder): void {
  recorder = r;
}

let scope: Scope | null = null;
/** Whether a write scope is running (graph/document.ts: store notices within a scope are not recorded as plain edits). */
export const writing = (): boolean => scope !== null;

function within(as: Scope, fn: () => void): void {
  if (scope) return fn(); // nested in another scope: the outer one decides
  recorder?.flush();
  scope = as;
  try {
    fn();
  } finally {
    scope = null;
  }
  if (as !== "restore") recorder?.record(as);
}
export const transaction = (label: Said, key: string, fn: () => void): void => within({ label, key }, fn);
export const absorb = (into: Anchor, fn: () => void): void => within({ into }, fn);
/** A derived answer that arrives while its step is undone (in the redo list): it still belongs to that step, by its
 * anchor and nothing else. It goes into the step's "after" and the undone steps after it (stopping where one set the
 * same parameter itself), never into the document now shown, which is from before the step: redo brings it back. `was`:
 * the node's parameters when the answer was asked for (a value set by hand since is kept). */
export const fillUndone = (into: Anchor, node: string, values: Record<string, unknown>, was: Record<string, unknown>): void =>
  recorder?.fillUndone(into, node, values, was);

/** The step that what was just written belongs to: the plain edits of this task are recorded now (their own step, or the
 * gesture's step they grow), and that step is returned. Taken outside a write scope, when the cause is written (a
 * scope's own writes are recorded only when it ends). */
export function anchor(): Anchor {
  recorder?.flush();
  return recorder?.top() ?? null;
}
export const derived = (fn: () => void): void => within("derived", fn);
export const restoring = (fn: () => void): void => within("restore", fn);

export class History {
  private undos: Step[] = [];
  private redos: Step[] = [];
  private current = EMPTY_DOC;
  private saved: Doc | null = EMPTY_DOC; // the document as its file has it (null: not known, e.g. unsaved work brought back)

  /** Another graph: a fresh history. `saved`: the document is its file's. */
  reset(doc: Doc, saved: boolean): void {
    this.undos = [];
    this.redos = [];
    this.current = doc;
    this.saved = saved ? doc : null;
  }

  /** The document as it is now: a new step, or the last step grown. False when the document did not change. `as`: how
   * to record it (RecordAs above; none: a plain user edit, named after what changed). */
  record(doc: Doc, defs: Record<string, NodeTypeDef>, outputOf: OutputOf, as?: RecordAs): boolean {
    if (as === "derived" || (typeof as === "object" && "into" in as)) {
      if (this.saved && sameDoc(this.saved, this.current)) this.saved = doc; // an unchanged graph stays unchanged after the fill
      // absorb: joins its step's "after" and every later step (undone ones too), otherwise undo then redo brings back the
      // new model with the old derived values (the value did not change, so nothing derives again). A step no longer in
      // the undo list (undone: its cause is not in the document any more; or dropped off the end) is left alone
      const at = as !== "derived" ? this.undos.findIndex((x) => x === as.into) : -1;
      if (at >= 0) {
        // the later steps in the order they were made (the redo stack's top is the earliest undone): a fill goes on until a
        // step that set the same thing itself (the user typed that derived value by hand): from there on it is the user's
        let fills = fillsOf(this.current, doc);
        this.undos[at].after = lay(this.undos[at].after, fills);
        for (const x of [...this.undos.slice(at + 1), ...[...this.redos].reverse()]) {
          if (!fills.length) break;
          const kept = fills.filter((f) => !touches(x, f)); // judged on the step as it was made
          x.before = lay(x.before, fills);
          x.after = lay(x.after, kept);
          fills = kept;
        }
      }
      this.current = doc;
      return false;
    }
    const changed = !sameDoc(doc, this.current);
    if (changed) {
      const { label, key } = as ?? describe(this.current, doc, defs, outputOf);
      const last = this.undos.at(-1);
      if (last && last.key === key && last.gesture === gesture && !this.redos.length) {
        last.after = doc;
        last.label = label;
        if (sameDoc(last.before, doc)) this.undos.pop(); // back where the gesture started: nothing to undo
      } else {
        this.undos.push({ before: this.current, after: doc, label, key, gesture });
        if (this.undos.length > LIMIT) this.undos.shift();
      }
      this.redos = [];
    }
    this.current = doc;
    return changed;
  }

  /** The document to go back to (null: nothing to undo). The store shows it, then tells `showing`. */
  undo(): Doc | null {
    const step = this.undos.pop();
    if (!step) return null;
    this.redos.push(step);
    gesture++; // what comes next is a step of its own
    return step.before;
  }

  redo(): Doc | null {
    const step = this.redos.pop();
    if (!step) return null;
    this.undos.push(step);
    gesture++;
    return step.after;
  }

  /** The document the store shows after undo / redo. */
  showing(doc: Doc): void {
    this.current = doc;
  }

  markSaved(doc: Doc): void {
    this.saved = doc;
  }

  isSaved(doc: Doc): boolean {
    return !!this.saved && sameDoc(doc, this.saved);
  }

  /** See the exported fillUndone above; nothing when `into` is not undone. */
  fillUndone(into: Anchor, node: string, values: Record<string, unknown>, was: Record<string, unknown>): void {
    const at = this.redos.findIndex((x) => x === into);
    if (at < 0) return;
    const step = this.redos[at];
    let fills: Fill[] = Object.entries(values)
      .filter(([k]) => same(valueIn(step.after, { node, key: k, v: null }), was[k]))
      .map(([k, v]) => ({ node, key: k, v }));
    step.after = lay(step.after, fills);
    // the undone steps made after it: the redo list's top is the earliest, so these sit below it, latest first
    for (const x of this.redos.slice(0, at).reverse()) {
      if (!fills.length) break;
      const kept = fills.filter((f) => !touches(x, f));
      x.before = lay(x.before, fills);
      x.after = lay(x.after, kept);
      fills = kept;
    }
  }

  /** The last step (Anchor above). */
  top(): Anchor {
    return this.undos.at(-1) ?? null;
  }

  get labels(): { undoLabel: Said | null; redoLabel: Said | null } {
    return { undoLabel: this.undos.at(-1)?.label ?? null, redoLabel: this.redos.at(-1)?.label ?? null };
  }
}
