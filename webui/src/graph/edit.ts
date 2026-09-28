/** Editing the graph: nodes, chains, wires, table rows, parameters, labels, copy / paste / duplicate,
 * exposed values, deleting, group boxes; each edit touches state/cookInputs.ts and state/look.ts together and is one
 * undo step (document.ts noticed()). graph/actions.ts re-exports what components call. */

import { api, type NodeTypeDef, type PickedFrom } from "../api";
import { getCatalog, getLayerPorts, getNodeDefs, getTypes } from "../state/catalog";
import { useCookInputs, type CookNode, type Wire } from "../state/cookInputs";
import { useLook, type Box, type Pos } from "../state/look";
import { useResults } from "../state/results";
import { useViewer, type LooseWire } from "../state/viewer";
import { BOX_COLORS, BOX_HEAD, boxContents, chosenOnNode, nodeSize, wouldCycle } from "./nodes";
import { PARAM, converter, inputPort, loosePort, looseType, outputPort, outputType, portAccepts, tableRows, wireKey } from "./rules";
import { snapshotNow } from "./snapshot";
import { pastedWires, takeCopy, uniqueCopyValue, type Copied } from "./clipboard";
import { absorbNextChange } from "./history";
import type { GNode } from "../state/graph";
import { msg, reasonOf, say } from "../state/say";
import { justWired } from "./actions";

let nodeSerial = 0;
// ------------------------------------------------------------------ editing

function uniqueValues(order: string[], nodes: Record<string, CookNode>, def: NodeTypeDef): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const p of def.params) {
    if (!p.unique) continue;
    const taken = new Set(order.map((id) => String(nodes[id].params[p.name] ?? "").toLowerCase()));
    const base = String(def.defaults[p.name] ?? p.name);
    let value = base;
    for (let i = 2; taken.has(value.toLowerCase()); i++) value = `${base}${i}`;
    out[p.name] = value;
  }
  return out;
}

function givenValue(nodes: Record<string, CookNode>, wire: LooseWire, def: NodeTypeDef): Record<string, unknown> {
  if (wire.side !== "target" || !wire.port.startsWith(PARAM) || !("value" in def.defaults)) return {};
  const node = nodes[wire.node];
  const spec = getNodeDefs()[node?.typeId ?? ""]?.params.find((p) => p.name === wire.port.slice(PARAM.length));
  const v = node?.params[spec?.name ?? ""];
  if (!spec || v === null || v === undefined) return {};
  const unit = "unit" in def.defaults && def.params.find((p) => p.name === "unit")?.options?.includes(spec.unit) ? { unit: spec.unit } : {};
  return { value: v, ...unit };
}

export function addNode(typeId: string, x: number, y: number, wire?: LooseWire): void {
  const def = getNodeDefs()[typeId];
  if (!def) return;
  const ci = useCookInputs.getState();
  const base = typeId.split(".").pop() ?? "node";
  let id = `${base}${++nodeSerial}`;
  while (ci.nodes[id] || ci.kept.nodes.some((n) => n.id === id)) id = `${base}${++nodeSerial}`;
  const snap = snapshotNow();
  const port = wire && loosePort(snap, wire, typeId);
  const promoted = wire?.side === "source" && port?.startsWith(PARAM) ? [port.slice(PARAM.length)] : undefined;
  useCookInputs.getState().insertNode(id, { typeId, label: def.label, params: { ...def.defaults, ...uniqueValues(ci.order, ci.nodes, def), ...(wire ? givenValue(ci.nodes, wire, def) : {}) }, promoted });
  useLook.getState().setPosition(id, x, y);
  // 节点上显示的参数行与参数值同源：类型声明的是出厂默认（NodeDef.on_node），在新建时写入该节点
  // 自身的名单（保存在节点图 json 的 `ui.on_node` 中），此后仅以 json 为准（上一行的参数 `...def.defaults` 采用相同做法）
  if (def.on_node.length) useLook.getState().setOnNode(id, [...def.on_node]);
  useViewer.getState().setSelectedNodes([id]);
  useViewer.setState({ selectedId: id, menu: null });
  if (!wire || !port) return;
  connect(wire.side === "source" ? { source: wire.node, sourceHandle: wire.port, target: id, targetHandle: port } : { source: id, sourceHandle: port, target: wire.node, targetHandle: wire.port });
}

export function addChain(typeIds: string[], x: number, y: number, wire: LooseWire): void {
  let into = wire;
  [...typeIds].reverse().forEach((typeId, i) => {
    const t = looseType(snapshotNow(), into);
    addNode(typeId, x - i * 280, y, into);
    const selected = useViewer.getState().selectedId!;
    const port = t && getNodeDefs()[typeId]?.at_defaults.ports.inputs.find((p) => portAccepts(getCatalog(), p.type, t))?.name;
    if (port) into = { node: selected, port, side: "target" };
  });
}

export function insertNode(target: string, port: string, typeId: string): void {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const wire = ci.edges.find((e) => e.target === target && e.targetHandle === port);
  if (!wire) return;
  const src = ci.nodes[wire.source];
  const dst = ci.nodes[target];
  if (!src || !dst) return;
  const srcNode: GNode = { id: wire.source, type: "l2s", position: look.positions[wire.source] ?? { x: 0, y: 0 }, data: src as never };
  const dstNode: GNode = { id: target, type: "l2s", position: look.positions[target] ?? { x: 0, y: 0 }, data: dst as never };
  const x = (srcNode.position.x + dstNode.position.x) / 2;
  const y = Math.max(srcNode.position.y + nodeSize(srcNode).h, dstNode.position.y + nodeSize(dstNode).h) + 40;
  addNode(typeId, x, y, { node: target, port, side: "target" });
  const addedId = useViewer.getState().selectedId!;
  const addedPos = useLook.getState().positions[addedId] ?? { x, y };
  useViewer.getState().requestPan(addedPos.x + nodeSize({ measured: undefined }).w / 2, addedPos.y + nodeSize({ measured: undefined }).h / 2);
  const from: LooseWire = { node: wire.source, port: wire.sourceHandle, side: "source" };
  const into = loosePort(snapshotNow(), from, typeId);
  if (into) connect({ source: from.node, sourceHandle: from.port, target: addedId, targetHandle: into });
  // The refused wire that the converter replaces is removed: a multi input keeps every wire, so the old one is
  // removed here. The converter stands between the two ends; it does not join them.
  deleteElements([], [], [wire.id]);
}

/** A default row label for a ports_from-input table (「多层 EXR 输出设置」's 图层), derived from which port is wired in
 * (data/layers.py LAYER_FOR_PORT, sent in the catalogue as layer_ports: image → rgba, alpha / mask → mask, normal → N,
 * position → P, flow → motion; anything else keeps its own name, such as depth, stmap, confidence), numbered on when it is already one
 * of `taken` (mirrors data/layers.py layer_names). */
function defaultRowLabel(fromPort: string, dataType: string, taken: string[]): string {
  const base = getLayerPorts()[fromPort] || fromPort || getTypes()[dataType.split("|")[0]]?.layer_default || "layer";
  if (!taken.includes(base)) return base;
  let n = 2;
  while (taken.includes(`${base}${n}`)) n += 1;
  return `${base}${n}`;
}

/** A fresh, stable row id (the port name) for a ports_from-input table: not already one of `taken`. */
function freshRowName(taken: string[]): string {
  let n = 1;
  while (taken.includes(`row${n}`)) n += 1;
  return `row${n}`;
}

/** A wire dropped on a ports_from-input node's body (not on an existing port): a row added to its table (「多层 EXR
 * 输出设置」's 图层), named after what is wired in, then connected to the new row's port; one step. Not such a node, or
 * the source type is not known yet: does nothing (returns false). */
export function addPortRow(targetId: string, fromNode: string, fromPort: string): boolean {
  const ci = useCookInputs.getState();
  const target = ci.nodes[targetId];
  const def = target && getNodeDefs()[target.typeId];
  if (!target || !def || def.ports_from_side !== "inputs") return false;
  const t = outputType(snapshotNow(), fromNode, fromPort);
  if (!t) return false;
  const rows = tableRows(def, target.params);
  const name = freshRowName(rows.map((r) => r.name));
  const label = defaultRowLabel(fromPort, t, rows.map((r) => r.label));
  setParam(targetId, def.ports_from, [...rows, { name, label }]);
  connect({ source: fromNode, sourceHandle: fromPort, target: targetId, targetHandle: name });
  return true;
}

/** Several 「序列图输出设置」 nodes replaced by one 「多层 EXR 输出设置」, one row per former node (its 名字 the row's
 * 图层名), rewired to the same sources and the same 「输出」; one step. */
export function mergeToExr(ids: string[]): void {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const exrDef = getNodeDefs()["core.output_exr"];
  const targetIds = ids.filter((id) => ci.nodes[id]?.typeId === "core.output_images");
  if (!exrDef || targetIds.length < 1) return;
  const taken: string[] = [];
  const rows = targetIds.map((tid, i) => {
    let label = String(ci.nodes[tid].params.name ?? `layer${i + 1}`);
    if (taken.includes(label)) {
      let k = 2;
      while (taken.includes(`${label}${k}`)) k += 1;
      label = `${label}${k}`;
    }
    taken.push(label);
    return { name: `row${i + 1}`, label };
  });
  const positions = targetIds.map((tid) => look.positions[tid] ?? { x: 0, y: 0 });
  const x = positions.reduce((a, p) => a + p.x, 0) / positions.length;
  const y = positions.reduce((a, p) => a + p.y, 0) / positions.length;
  let id = `output_exr${++nodeSerial}`;
  while (ci.nodes[id] || ci.kept.nodes.some((n) => n.id === id)) id = `output_exr${++nodeSerial}`;
  const idsSet = new Set(targetIds);
  const incoming = targetIds.map((tid) => ci.edges.find((e) => e.target === tid && e.targetHandle === "image"));
  const outgoing = targetIds.map((tid) => ci.edges.find((e) => e.source === tid && e.sourceHandle === "files")).find((e) => e);
  const kept = ci.edges.filter((e) => !idsSet.has(e.source) && !idsSet.has(e.target));
  const added: Wire[] = [];
  incoming.forEach((e, i) => {
    if (e) added.push({ id: `${e.source}.${e.sourceHandle}->${id}.${rows[i].name}`, source: e.source, sourceHandle: e.sourceHandle, target: id, targetHandle: rows[i].name });
  });
  if (outgoing) added.push({ id: `${id}.files->${outgoing.target}.${outgoing.targetHandle}`, source: id, sourceHandle: "files", target: outgoing.target, targetHandle: outgoing.targetHandle });
  const oldDisplay = look.displayId;
  const boxesToFix = look.boxes.filter((b) => b.members.some((m) => idsSet.has(m)));
  // The merged-away nodes are removed like any other deleted node (deleteElements clears their position, canvas selection,
  // results and, if one of them was shown, the display node). Showing or planning one of them by its old id would return a
  // 500 error (the node no longer exists) rather than an empty display.
  deleteElements(targetIds, [], []);
  useCookInputs.getState().insertNode(id, { typeId: "core.output_exr", label: exrDef.label, params: { ...exrDef.defaults, layers: rows } });
  useLook.getState().setPosition(id, x, y);
  useCookInputs.getState().setEdges([...kept, ...added]);
  for (const b of boxesToFix) useLook.getState().setBox(b.id, { members: [...b.members.filter((m) => !idsSet.has(m)), id] });
  if (oldDisplay && idsSet.has(oldDisplay)) useLook.getState().setDisplay(id);
  useViewer.getState().setSelectedNodes([id]);
  useViewer.setState({ selectedId: id });
}

export function connect(c: { source: string | null; sourceHandle?: string | null; target: string | null; targetHandle?: string | null }): void {
  if (!c.source || !c.target) return;
  const snap = snapshotNow();
  const dst = snap.nodes.find((n) => n.id === c.target);
  const srcType = outputType(snap, c.source, c.sourceHandle ?? null);
  const port = dst && inputPort(snap, dst.id, c.targetHandle ?? null);
  if (!srcType || !port) return;
  const converterOk = port.name.startsWith(PARAM) ? false : !!converter(snap.catalog, srcType, port.type);
  const at = { source: snap.nodes.find((n) => n.id === c.source)?.data.label ?? c.source, node: dst.data.label, input: port.label };
  // 该端口当前不可用（由服务器计算，nodes/port.py Port.applies）：任何途径都无法连接。
  // 在此处统一拦截，而非在拖线处拦截：拖线、从端口拖出后在菜单中选择节点、检查中的一键插入均经由此 connect()。
  // 源输出口同样可能不可用（解算器接入相机后，其「相机」输出仅原样透传），在同一处拦截。
  const out = outputPort(snap, c.source, c.sourceHandle ?? null);
  if (out?.inactive) {
    say(msg("B-WIRE-OUTINACTIVE", { source: at.source, output: out.label, why: out.inactive.text }, { port: out.name }), c.source);
    return;
  }
  if (port.inactive) {
    say(msg("B-WIRE-INACTIVE", { node: at.node, input: port.label, why: port.inactive.text }, { port: port.name }), dst.id);
    return;
  }
  if (!portAccepts(snap.catalog, port.type, srcType) && !converterOk) {
    say(msg("B-WIRE-MISMATCH", { ...at, got: srcType, want: port.type }, { port: port.name }), dst.id);
    return;
  }
  const ci = useCookInputs.getState();
  if (wouldCycle(ci.edges, c.source, c.target)) {
    say(msg("B-WIRE-CYCLE", { source: at.source, node: at.node }), dst.id);
    return;
  }
  const targetHandle = c.targetHandle ?? "";
  const sourceHandle = c.sourceHandle ?? "";
  const old = port.multi ? undefined : ci.edges.find((e) => e.target === c.target && e.targetHandle === targetHandle);
  const kept = port.multi ? ci.edges : ci.edges.filter((e) => e !== old);
  const id = wireKey(c.source, sourceHandle, c.target, targetHandle);
  useCookInputs.getState().setEdges([...kept, { id, source: c.source, sourceHandle, target: c.target, targetHandle }]);
  if (old) {
    say(msg("I-WIRE-REPLACED", { ...at, was: ci.nodes[old.source]?.label ?? old.source }), dst.id);
  }
  justWired.add(id); // what is wrong with it, if anything, is said once the server has judged it (sayJudgedWires)
}

export function setParam(id: string, name: string, value: unknown): void {
  editParams(id, { [name]: value });
}

/** 写入推导出的参数值（editor/ColorspaceFill.tsx：按文件格式填写空缺的「色彩空间」），
 * 但不计为使用者的一步操作：不进入撤销历史，不将节点图标记为已修改（graph/history.ts absorbNextChange）。若经由 setParam，
 * 撤销后参数变空并被再次填写，导致撤销无效；打开旧图时也会立即被标记为已修改。值未变化时不写入。 */
export function setDerivedParam(id: string, name: string, value: unknown): void {
  const node = useCookInputs.getState().nodes[id];
  if (!node || node.params[name] === value) return;
  absorbNextChange();
  useCookInputs.getState().setNode(id, { params: { ...node.params, [name]: value } });
}

/** 一次修改一组参数，计为一步撤销（粘贴的镜头包含十余个数值，撤销一次即全部恢复）。
 * 与 setParam 经由同一路径，因此 derived_from 参数（随「镜头模型」变化的畸变参数表）同样会重新计算。 */
export function setParams(id: string, changes: Record<string, unknown>): void {
  editParams(id, changes);
}

/** A node's parameters change (with whatever else of its data goes with them, `extra`: e.g. pickFile's own `picked`
 * record): the parameters derived from them (P(derived_from): an import node's singleton-kind auto-selection,
 * 「LensDistortion」's 畸变参数 from its 镜头模型) are requested from the server first, so the whole change is one edit.
 * `extra` is computed against the node as it was before `changes`. */
function editParams(id: string, changes: Record<string, unknown>, extra: (node: CookNode) => Partial<CookNode> = () => ({})): void {
  const ci = useCookInputs.getState();
  const node = ci.nodes[id];
  const def = getNodeDefs()[node?.typeId ?? ""];
  if (!node || !def) return;
  const unstored = node.stored && Object.keys(changes).some((k) => k in node.stored!) ? { stored: Object.fromEntries(Object.entries(node.stored).filter(([k]) => !(k in changes))) } : {};
  const derived = def.params.filter((p) => p.derived_from.some((d) => d in changes));
  const asked = { ...node.params, ...changes };
  // 使用者的修改立即生效（一步撤销）。派生参数等待服务器答复后再写入当时的节点，不计入撤销
  // （absorbNextChange：它不是使用者的操作）。不得在答复返回后用发送请求时的快照整体写回，否则会覆盖使用者在此期间修改的其他参数
  useCookInputs.getState().setNode(id, { ...extra(node), ...unstored, params: asked });
  if (!derived.length) return;
  const sources = new Set(derived.flatMap((p) => p.derived_from));
  const land = (more: Record<string, unknown>) => {
    const now = useCookInputs.getState().nodes[id];
    if (!now) return; // the node is gone
    if ([...sources].some((k) => now.params[k] !== asked[k])) return; // asked about values that have changed since: a later edit's answer applies
    absorbNextChange();
    useCookInputs.getState().setNode(id, { params: { ...now.params, ...more } });
  };
  void api.derive(def.id, asked).then(
    (got) => land(got),
    (e) => {
      const names = def.params.filter((p) => p.derived_from.length).map((p) => `「${p.label}」`).join("");
      say(msg("E-PARAM-DERIVE", { node: node.label, params: names, reason: reasonOf(e) }), id);
      land(Object.fromEntries(derived.map((p) => [p.name, def.defaults[p.name]])));
    },
  );
}

/** 提升到节点 / 取消提升（参数面板中的按钮）：
 * 提升后该参数在节点上增加输入口 `param:<name>`，同一行上也可直接编辑（graph/nodes.ts nodeRows）。
 * 取消提升时，接在该端口上的连线一并断开。 */
export function togglePromoted(id: string, name: string): void {
  const ci = useCookInputs.getState();
  const node = ci.nodes[id];
  if (!node) return;
  const on = node.promoted?.includes(name);
  const promoted = on ? node.promoted!.filter((p) => p !== name) : [...(node.promoted ?? []), name];
  useCookInputs.getState().setNode(id, { promoted: promoted.length ? promoted : undefined });
  if (on) useCookInputs.getState().setEdges(ci.edges.filter((e) => !(e.target === id && e.targetHandle === PARAM + name)));
}

/** 在节点上显示 / 不在节点上显示（参数面板中的标记）：
 * 仅控制该行是否显示在节点上，不提供参数输入口；输入口由相邻的「提升到节点」控制（togglePromoted）。
 *
 * 使用者可以移除作者默认显示的行：名单初始为类型声明各行的副本（chosenOnNode），
 * 点击即从副本中移除。因此每一行上的标记均可操作；否则作者默认的行（通常正是占用空间的行）
 * 点击后没有反应。 */
export function toggleOnNode(id: string, name: string): void {
  const def = getNodeDefs()[useCookInputs.getState().nodes[id]?.typeId ?? ""];
  if (!def) return;
  const was = chosenOnNode(def, useLook.getState().onNode[id]);
  // 点击后该节点的名单一律显式保存（即使恢复为与出厂默认完全一致）：以 json 为准，类型上的名单仅作为新建时的
  // 初值与 json 未写入时的兜底。保存为 undefined 会使该节点此后随出厂默认变化，形成两处数据来源
  useLook.getState().setOnNode(id, def.params.filter((p) => p.simple && (p.name === name ? !was.includes(name) : was.includes(p.name))).map((p) => p.name));
}

// Picking a file is a parameter change like any other (editParams): an import node's singleton-kind selection (the
// one camera, the one skeleton animation) is derived by the server from the file. A bare setNode here would silently
// drop that derivation, leaving every port that depends on it blank until something else touches the node's params.
// 「读取序列」没有图层表：端口直接由文件中的图层生成（nodes/core/input.py made_ports），
// 因此选定文件后，只要参数变化并返回一次状态，端口即可生成。
export function pickFile(id: string, name: string, picked: PickedFrom | null): void {
  editParams(id, { [name]: picked?.ref ?? "" }, (node) => {
    const { [name]: _drop, ...others } = node.picked ?? {};
    void _drop;
    return { picked: picked ? { ...others, [name]: picked } : others };
  });
}

export async function deriveParams(id: string, params?: Record<string, unknown>): Promise<Record<string, unknown> | null> {
  const node = useCookInputs.getState().nodes[id];
  const def = getNodeDefs()[node?.typeId ?? ""];
  if (!node || !def) return null;
  try {
    return await api.derive(def.id, params ?? node.params);
  } catch (e) {
    const names = def.params.filter((p) => p.derived_from.length).map((p) => `「${p.label}」`).join("");
    say(msg("E-PARAM-DERIVE", { node: node.label, params: names, reason: reasonOf(e) }), id);
    return null;
  }
}

export function setLabel(id: string, label: string): void {
  useCookInputs.getState().setNode(id, { label });
}

export function toggleExposed(nodeId: string, param: string, label: string): void {
  const ci = useCookInputs.getState();
  const target = `${nodeId}.${param}`;
  if (ci.exposed.some((x) => x.target === target)) {
    useCookInputs.getState().setExposed(ci.exposed.filter((x) => x.target !== target));
    return;
  }
  const name = ci.exposed.some((x) => x.name === param) ? `${nodeId}_${param}` : param;
  useCookInputs.getState().setExposed([...ci.exposed, { name, label, target }]);
}

/** What the user deleted in one go (Delete / Backspace on the canvas: React Flow's onDelete, which hands over the
 * nodes, boxes and wires together, where its change events would report them one by one): one undo step, as every
 * action is (noticeSoon). */
export function deleteElements(nodes: string[], boxes: string[], wireIds: string[]): void {
  if (boxes.length) useLook.getState().removeBoxes(boxes);
  if (nodes.length) {
    useCookInputs.getState().removeNodes(nodes); // with every wire touching them
    useLook.getState().removeNodes(nodes);
    useViewer.getState().removeCanvasNodes(nodes);
    useResults.getState().removeNodeStatus(nodes);
    if (nodes.includes(useViewer.getState().selectedId ?? "")) useViewer.setState({ selectedId: null });
    if (nodes.includes(useLook.getState().displayId ?? "")) useLook.getState().setDisplay(null);
  }
  const ci = useCookInputs.getState();
  if (wireIds.some((w) => ci.edges.some((e) => e.id === w))) ci.setEdges(ci.edges.filter((e) => !wireIds.includes(e.id)));
}

// ------------------------------------------------------------------ copy, paste, duplicate

// what Ctrl+C took (graph/clipboard.ts); kept by the page, so it pastes into another graph opened here as well
let clipboard: Copied | null = null;
let pastes = 0; // pastes of this copy not placed at the pointer: each lands one step further, never on the last one

const PASTE_GAP = 40;

/** The selected nodes (and group boxes, with the nodes inside them, as dragging one moves them) as a copy; null when
 * nothing is selected. 「未知节点」 are not copied: they are the file's, never the editor's. */
function selectionCopy(): Copied | null {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const view = useViewer.getState();
  const boxes = look.boxes.filter((b) => view.selectedBoxIds.includes(b.id));
  const nodes = snapshotNow().nodes;
  const ids = new Set(ci.order.filter((id) => view.canvas[id]?.selected));
  for (const b of boxes) for (const id of boxContents(b, nodes)) if (ci.nodes[id]) ids.add(id);
  if (!ids.size && !boxes.length) return null;
  const order = ci.order.filter((id) => ids.has(id)); // file order: the copies keep it
  return takeCopy(ci.graphId, order, boxes, ci.nodes, look.positions, look.onNode, ci.edges);
}

/** Ctrl+C: the selection is copied (false: nothing selected, the key is the browser's). */
export function copySelection(): boolean {
  const copied = selectionCopy();
  if (!copied) return false;
  clipboard = copied;
  pastes = 0;
  return true;
}

/** Ctrl+V: what was copied, added to this graph as one step and selected. `at` (flow coordinates, the pointer over the
 * node graph): the copy's top left lands there; without it the copy lands below where the copied nodes stood, one
 * step further for every paste. False: nothing copied. */
export function pasteCopied(at?: Pos): boolean {
  if (!clipboard) return false;
  pasteInto(clipboard, at ?? ++pastes);
  return true;
}

/** Ctrl+D: the selection duplicated right below itself, as one step (what Ctrl+C holds stays as it is). */
export function duplicateSelection(): boolean {
  const copied = selectionCopy();
  if (!copied) return false;
  pasteInto(copied, 1);
  return true;
}

/** The copy's extent in the graph: its nodes (at their measured size) and boxes. */
function extent(copied: Copied): { x: number; y: number; h: number } {
  const canvas = useViewer.getState().canvas;
  const rects = [
    ...copied.nodes.map((n) => ({ ...n.pos, ...nodeSize({ measured: canvas[n.id]?.measured as GNode["measured"] }) })),
    ...copied.boxes.map((b) => ({ x: b.x, y: b.y, w: b.w, h: b.collapsed ? BOX_HEAD : b.h })),
  ];
  const y = Math.min(...rects.map((r) => r.y));
  return { x: Math.min(...rects.map((r) => r.x)), y, h: Math.max(...rects.map((r) => r.y + r.h)) - y };
}

/** Adds a copy: new ids named like any new node's (graph/edit.ts addNode), labels, parameters, promoted parameters and
 * body rows as copied, unique parameters numbered on, the wires graph/clipboard.ts pastedWires keeps; the copies
 * become the selection. `place`: where its top left goes, or how many steps below the copied nodes. */
function pasteInto(copied: Copied, place: Pos | number): void {
  const box = extent(copied);
  const d = typeof place === "number" ? { x: 0, y: place * (box.h + PASTE_GAP) } : { x: place.x - box.x, y: place.y - box.y };
  const ci = useCookInputs.getState();
  const renamed = new Map<string, string>();
  for (const n of copied.nodes) {
    const def = getNodeDefs()[n.data.typeId];
    const now = useCookInputs.getState();
    const base = n.data.typeId.split(".").pop() ?? "node";
    let id = `${base}${++nodeSerial}`;
    while (now.nodes[id] || now.kept.nodes.some((k) => k.id === id)) id = `${base}${++nodeSerial}`;
    const params = { ...n.data.params };
    for (const p of def?.params ?? []) {
      if (!p.unique) continue;
      const taken = new Set(now.order.map((o) => String(now.nodes[o].params[p.name] ?? "").toLowerCase()));
      params[p.name] = uniqueCopyValue(String(params[p.name] ?? def!.defaults[p.name] ?? p.name), taken);
    }
    now.insertNode(id, { ...structuredClone(n.data), params });
    useLook.getState().setPosition(id, n.pos.x + d.x, n.pos.y + d.y);
    if (n.onNode) useLook.getState().setOnNode(id, [...n.onNode]);
    renamed.set(n.id, id);
  }
  const wires = pastedWires(copied, renamed, copied.graphId === ci.graphId, (id) => !!useCookInputs.getState().nodes[id]);
  if (wires.length) useCookInputs.getState().setEdges([...useCookInputs.getState().edges, ...wires]);
  const boxIds: string[] = [];
  for (const b of copied.boxes) {
    const taken = useLook.getState().boxes;
    let k = taken.length + 1;
    while (taken.some((x) => x.id === `box:${k}`)) k++;
    boxIds.push(`box:${k}`);
    useLook.getState().addBox({ ...b, id: `box:${k}`, x: b.x + d.x, y: b.y + d.y, members: b.members.map((m) => renamed.get(m)!).filter(Boolean) });
  }
  const view = useViewer.getState();
  const ids = [...renamed.values()];
  view.setSelectedNodes(ids);
  view.setSelectedBoxes(boxIds);
  view.setSelectedEdges([]);
  // the parameter panel follows: the copy of the node it showed, else the first copy
  useViewer.setState({ selectedId: renamed.get(view.selectedId ?? "") ?? ids[0] ?? null, menu: null });
}

// ------------------------------------------------------------------ group boxes

export function addBox(at?: { x: number; y: number }): void {
  const look = useLook.getState();
  const snap = snapshotNow();
  const picked = snap.nodes.filter((n) => useViewer.getState().canvas[n.id]?.selected);
  let box: Box;
  let n = look.boxes.length + 1;
  while (look.boxes.some((b) => b.id === `box:${n}`)) n++;
  const base = { id: `box:${n}`, label: `分组 ${n}`, color: BOX_COLORS[(n - 1) % BOX_COLORS.length], collapsed: false, members: [] };
  if (picked.length) {
    const pad = 26;
    const x0 = Math.min(...picked.map((p) => p.position.x)) - pad;
    const y0 = Math.min(...picked.map((p) => p.position.y)) - pad - BOX_HEAD;
    const x1 = Math.max(...picked.map((p) => p.position.x + nodeSize(p).w)) + pad;
    const y1 = Math.max(...picked.map((p) => p.position.y + nodeSize(p).h)) + pad;
    box = { ...base, x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  } else {
    const p = at ?? { x: 0, y: 0 };
    box = { ...base, x: p.x, y: p.y, w: 520, h: 260 };
  }
  useLook.getState().addBox(box);
  useViewer.getState().setSelectedBoxes([box.id]);
}

export function toggleBox(id: string): void {
  const look = useLook.getState();
  const box = look.boxes.find((b) => b.id === id);
  if (!box) return;
  const members = box.collapsed ? [] : boxContents(box, snapshotNow().nodes);
  useLook.getState().setBox(id, { collapsed: !box.collapsed, members });
}
