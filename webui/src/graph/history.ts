import type { Edge } from "@xyflow/react";
import type { GraphJSON, NodeTypeDef } from "../api";
import { chosenOnNode } from "./nodes";
import type { GBox, GNode } from "../state/graph";
import { same as sameJson, type Json } from "../model/graphPatch";

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
 * everything they did not change. Opening a graph starts a fresh history. */

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
  label: string; // what it did, for the undo / redo tooltips
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

const EMPTY_DOC: Doc = { meta: { name: "未命名" }, exposed: [], nodes: [], edges: [], boxes: [], displayId: null, displayPort: null, cookRange: null };

// the document's values are JSON (what a graph file holds): compared as trees, never as strings of them
const same = (a: unknown, b: unknown) => sameJson(a as Json, b as Json);

const changedParams = (a: Record<string, unknown>, b: Record<string, unknown>) =>
  a === b ? [] : [...new Set([...Object.keys(a), ...Object.keys(b)])].filter((k) => !same(a[k], b[k]));

const sameNode = (a: GNode, b: GNode | undefined) =>
  a === b ||
  (!!b &&
    a.data.typeId === b.data.typeId &&
    a.data.label === b.data.label &&
    a.position.x === b.position.x &&
    a.position.y === b.position.y &&
    !changedParams(a.data.params, b.data.params).length &&
    same(a.data.picked, b.data.picked) &&
    same(a.data.promoted ?? [], b.data.promoted ?? []) &&
    same(a.data.onNode ?? null, b.data.onNode ?? null));

const sameBox = (a: GBox, b: GBox | undefined) =>
  a === b ||
  (!!b && a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h && a.label === b.label && a.color === b.color && a.collapsed === b.collapsed && same(a.members, b.members));

const byId = <T extends { id: string }>(list: T[]) => new Map(list.map((x) => [x.id, x]));

function sameList<T extends { id: string }>(a: T[], b: T[], eq: (x: T, y: T | undefined) => boolean): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  const other = byId(b);
  return a.every((x) => eq(x, other.get(x.id)));
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
function describe(a: Doc, b: Doc, defs: Record<string, NodeTypeDef>, outputOf: OutputOf): { label: string; key: string } {
  const was = byId(a.nodes);
  const now = byId(b.nodes);
  const name = (id: string) => `「${(now.get(id) ?? was.get(id))?.data.label ?? id}」`;
  const ids = (list: { id: string }[]) => list.map((x) => x.id).join(",");
  const count = <T,>(list: T[], one: (x: T) => string, several: string) => (list.length === 1 ? one(list[0]) : several.replace("#", String(list.length)));
  const step = (label: string, key: string) => ({ label, key });

  const added = b.nodes.filter((n) => !was.has(n.id));
  if (added.length) return step(count(added, (n) => `添加节点${name(n.id)}`, "添加 # 个节点"), `node+${ids(added)}`);
  const removed = a.nodes.filter((n) => !now.has(n.id));
  if (removed.length) return step(count(removed, (n) => `删除节点${name(n.id)}`, "删除 # 个节点"), `node-${ids(removed)}`);

  const boxWas = byId(a.boxes);
  const boxNow = byId(b.boxes);
  const newBoxes = b.boxes.filter((x) => !boxWas.has(x.id));
  if (newBoxes.length) return step(count(newBoxes, (x) => `添加分组「${x.label}」`, "添加 # 个分组"), `box+${ids(newBoxes)}`);
  const goneBoxes = a.boxes.filter((x) => !boxNow.has(x.id));
  if (goneBoxes.length) return step(count(goneBoxes, (x) => `删除分组「${x.label}」`, "删除 # 个分组"), `box-${ids(goneBoxes)}`);

  // 提升到节点: the parameter gets a row on the node with an input of its own, or loses both (with its wire: one step)
  for (const n of b.nodes) {
    const before = was.get(n.id)?.data.promoted ?? [];
    const now = n.data.promoted ?? [];
    const k = now.find((p) => !before.includes(p)) ?? before.find((p) => !now.includes(p));
    if (k) {
      const what = defs[n.data.typeId]?.params.find((p) => p.name === k)?.label ?? k;
      return step(`${now.includes(k) ? "提升" : "取消提升"}${name(n.id)}的「${what}」`, `promote ${n.id}.${k}`);
    }
  }

  const wire = (e: Edge) => `${name(e.source)}→${name(e.target)}`;
  const edgesWas = new Set(a.edges.map((e) => e.id));
  const edgesNow = new Set(b.edges.map((e) => e.id));
  const joined = b.edges.filter((e) => !edgesWas.has(e.id));
  if (joined.length) return step(count(joined, (e) => `连接${wire(e)}`, "连接 # 条线"), `wire+${ids(joined)}`);
  const cut = a.edges.filter((e) => !edgesNow.has(e.id));
  if (cut.length) return step(count(cut, (e) => `断开${wire(e)}`, "断开 # 条线"), `wire-${ids(cut)}`);

  const edited = b.nodes.filter((n) => !sameNode(n, was.get(n.id)));
  const params = edited.flatMap((n) => changedParams(was.get(n.id)!.data.params, n.data.params).map((k) => [n, k] as const));
  if (params.length) {
    const [n, k] = params[0];
    const touched = new Set(params.map(([m]) => m.id));
    const label =
      params.length === 1
        ? `修改${name(n.id)}的「${defs[n.data.typeId]?.params.find((p) => p.name === k)?.label ?? k}」`
        : touched.size === 1
          ? `修改${name(n.id)}的参数`
          : `修改 ${touched.size} 个节点的参数`;
    return step(label, `param ${params.map(([m, q]) => `${m.id}.${q}`).join(",")}`);
  }
  const renamed = edited.filter((n) => n.data.label !== was.get(n.id)!.data.label);
  if (renamed.length) return step(count(renamed, (n) => `重命名节点${name(n.id)}`, "重命名 # 个节点"), `label ${ids(renamed)}`);
  const picked = edited.filter((n) => !same(n.data.picked, was.get(n.id)!.data.picked)); // the same files from another folder
  if (picked.length) return step(count(picked, (n) => `重新选择${name(n.id)}的文件`, "重新选择 # 个节点的文件"), `picked ${ids(picked)}`);
  // 在节点上显示 / 不再显示（一行参数进出节点；提升到节点是上方的独立一步）
  for (const n of edited) {
    const def = defs[n.data.typeId];
    const before = def ? chosenOnNode(def, was.get(n.id)!.data.onNode) : [];
    const after = def ? chosenOnNode(def, n.data.onNode) : [];
    const k = after.find((p) => !before.includes(p)) ?? before.find((p) => !after.includes(p));
    if (k) {
      const what = def?.params.find((p) => p.name === k)?.label ?? k;
      return step(`${after.includes(k) ? "在节点上显示" : "不在节点上显示"}${name(n.id)}的「${what}」`, `onNode ${n.id}.${k}`);
    }
  }
  const boxes = b.boxes.filter((x) => !sameBox(x, boxWas.get(x.id))).map((x) => [boxWas.get(x.id)!, x] as const);
  for (const [p, x] of boxes) {
    if (p.label !== x.label) return step(`重命名分组「${x.label}」`, `boxLabel ${x.id}`);
    if (p.color !== x.color) return step(`更改分组「${x.label}」的颜色`, `boxColor ${x.id}`);
    if (p.collapsed !== x.collapsed) return step(`${x.collapsed ? "折叠" : "展开"}分组「${x.label}」`, `boxFold ${x.id}`);
    if (p.w !== x.w || p.h !== x.h) return step(`调整分组「${x.label}」的大小`, `boxSize ${x.id}`);
  }
  const movedBoxes = boxes.filter(([p, x]) => p.x !== x.x || p.y !== x.y).map(([, x]) => x);
  const moved = edited.filter((n) => n.position.x !== was.get(n.id)!.position.x || n.position.y !== was.get(n.id)!.position.y);
  if (movedBoxes.length || moved.length) {
    const label = movedBoxes.length ? count(movedBoxes, (x) => `移动分组「${x.label}」`, "移动 # 个分组") : count(moved, (n) => `移动节点${name(n.id)}`, "移动 # 个节点");
    return step(label, `move ${ids(movedBoxes)} ${ids(moved)}`);
  }

  const pinned = b.exposed.filter((x) => !a.exposed.some((y) => y.target === x.target));
  if (pinned.length) return step(count(pinned, (x) => `设为对外参数「${x.label}」`, "设 # 个对外参数"), `expose+${pinned.map((x) => x.target)}`);
  const unpinned = a.exposed.filter((x) => !b.exposed.some((y) => y.target === x.target));
  if (unpinned.length) return step(count(unpinned, (x) => `取消对外参数「${x.label}」`, "取消 # 个对外参数"), `expose-${unpinned.map((x) => x.target)}`);

  if (a.displayId !== b.displayId) return step(b.displayId ? `显示节点${name(b.displayId)}` : "不显示节点", "display");
  if (a.displayPort !== b.displayPort && b.displayId) {
    const port = outputOf(b.displayId, b.displayPort);
    return step(`显示${name(b.displayId)}的「${port?.label ?? b.displayPort}」`, "port");
  }
  if (!same(a.cookRange, b.cookRange)) return step(b.cookRange ? `计算范围改成 ${b.cookRange[0]}–${b.cookRange[1]}` : "计算范围改回全部", "range");
  if (!same(a.meta, b.meta)) return step("修改节点图名字", "meta");
  return step("修改节点图", "graph");
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

// 推导出的修改不计为一步，它们不是使用者的操作：editor/ColorspaceFill.tsx 按文件格式填写空缺的「色彩空间」、
// 服务器推导出的参数（graph/edit.ts editParams）、随素材收缩的计算范围（graph/actions.ts refreshStatus）。
// 以色彩空间为例，若记为一步：撤销后参数变空，随即又被填写，撤销永远无效且重做栈被清空；打开色彩空间为空的旧图时
// 也会立即被标记为已修改。因此填写前设置此标记，紧随其后的 record（document.ts 在同一微任务中调用）将变化并入
// current 而不记为一步；若文件原本与 current 一致，saved 同步移动，节点图不被标记为已修改。
let absorbing = false;
export function absorbNextChange(): void {
  absorbing = true;
}

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
    absorbing = false; // 切换了节点图：上一张图未消耗的标记不得吞掉新图的第一步
  }

  /** The document as it is now: a new step, or the last step grown. False when the document did not change. */
  record(doc: Doc, defs: Record<string, NodeTypeDef>, outputOf: OutputOf): boolean {
    if (absorbing) {
      absorbing = false;
      if (this.saved && sameDoc(this.saved, this.current)) this.saved = doc; // 未修改的图填写后仍为未修改状态
      this.current = doc;
      return false;
    }
    const changed = !sameDoc(doc, this.current);
    if (changed) {
      const { label, key } = describe(this.current, doc, defs, outputOf);
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

  get labels(): { undoLabel: string | null; redoLabel: string | null } {
    return { undoLabel: this.undos.at(-1)?.label ?? null, redoLabel: this.redos.at(-1)?.label ?? null };
  }
}
