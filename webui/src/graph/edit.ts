/** Editing the graph: nodes, chains, wires, table rows, parameters, labels, copy / paste / duplicate,
 * exposed values, deleting, group boxes; each edit touches state/cookInputs.ts and state/look.ts together and is one
 * undo step (document.ts noticed()). graph/actions.ts re-exports what components call. */

import { role3d } from "../view/handleEditing";
import { api, type ExposedEntry, type NodeTypeDef, type PickedFrom, type Words } from "../api";
import { getCatalog, getLayerPorts, getNodeDefs, getTypes } from "../state/catalog";
import { wiredRule } from "../model/rigPair";
import { drives, edited, exposedParams, linkedTargets, newWords, nodeRefOf, splitTarget, targetsOf, useCookInputs, withoutTarget, type CookNode, type Wire } from "../state/cookInputs";
import { useLook, type Box, type Pos } from "../state/look";
import { useResults } from "../state/results";
import { useViewer, type LooseWire } from "../state/viewer";
import { BOX_COLORS, BOX_FOLD_W, BOX_HEAD, boxContents, chosenOnNode, nodeSize } from "./nodes";
import { wireProblem } from "./wireRule";
import { ADD_ROW, PARAM, inputPort, loosePort, looseType, outputType, portAccepts, rowsLeft, tableRows, wireKey } from "./rules";
import { snapshotNow } from "./snapshot";
import { pastedWires, takeCopy, uniqueCopyValue, type Copied } from "./clipboard";
import { absorb, anchor, derived, fillUndone, transaction } from "./history";
import { same, type Json } from "../model/graphPatch";
import { rewire, type End as RewireEnd, type Held } from "./rewire";
import { eachBlocks, inputSlot, layoutGraph } from "./layout";
import type { GNode } from "../state/graph";
import { msg, reasonOf, say } from "../state/say";
import { justWired } from "./judged";
import { defaultName, nameProblem, wordIn } from "./naming";
import { pick, t } from "../i18n/t";

/** Whether a node of the open graph (or a kept 「未知节点」) already has `id`: what a new name must not be. */
const idTaken = (id: string): boolean => {
  const ci = useCookInputs.getState();
  return !!ci.nodes[id] || ci.kept.nodes.some((n) => n.id === id);
};
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
  const id = defaultName(typeId, idTaken);
  const snap = snapshotNow();
  const port = wire && loosePort(snap, wire, typeId);
  const promoted = wire?.side === "source" && port?.startsWith(PARAM) ? [port.slice(PARAM.length)] : undefined;
  useCookInputs.getState().insertNode(id, { typeId, params: { ...def.defaults, ...uniqueValues(ci.order, ci.nodes, def), ...(wire ? givenValue(ci.nodes, wire, def) : {}) }, promoted });
  useLook.getState().setPosition(id, x, y);
  // the rows shown on a node come from the same place as the parameter values: the type declares the factory default
  // (NodeDef.on_node), written into the node's own list when it is created (kept in the graph json's `ui.on_node`), and
  // from then on only the json counts (the same as the parameters' `...def.defaults` on the line above)
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

/** `base`, or `base2`, `base3`… when it is already one of `taken` (mirrors data/layers.py layer_names). */
function numbered(base: string, taken: string[]): string {
  if (!taken.includes(base)) return base;
  let n = 2;
  while (taken.includes(`${base}${n}`)) n += 1;
  return `${base}${n}`;
}

/** A default row label for a ports_from-input table. A node that names its rows itself (`ports_from_names`: 「切换」's
 * ways) gets none ("": read as 第一路 / Input 1 … of its place, in the language when shown). Otherwise (「多层 EXR 输出设置」's 图层) it is
 * derived from which port is wired in (data/layers.py LAYER_FOR_PORT, sent in the catalogue as layer_ports: image → rgba,
 * alpha / mask → mask, normal → N, position → P, flow → motion; anything else keeps its own name, such as depth, stmap,
 * confidence), numbered on when it is already one of `taken`. Nothing wired in (「＋」 clicked): layer, layer2… */
function defaultRowLabel(def: NodeTypeDef, fromPort: string, dataType: string, taken: string[]): string {
  // a row the node names itself keeps no name of its own: it reads as the node's word for its place, said when shown
  // in the language then (editor/ParamTable.tsx rowDefault; the server's made_ports)
  if (def.ports_from_names?.length) return "";
  return numbered(getLayerPorts()[fromPort] || fromPort || getTypes()[dataType.split("|")[0]]?.layer_default || "layer", taken);
}

/** A fresh, stable row id (the port name) for a ports_from-input table, not already one of `taken`: the first of the
 * names the node gives its rows (`ports_from_names`, 「切换」's a…j; "" none left), else row1, row2… */
function freshRowName(def: NodeTypeDef, taken: string[]): string {
  if (def.ports_from_names) return def.ports_from_names.find((n) => !taken.includes(n)) ?? "";
  let n = 1;
  while (taken.includes(`row${n}`)) n += 1;
  return `row${n}`;
}

/** The node's input table as it is (every field of every row kept: 通道 too), when its type makes inputs from one. */
function inputTable(nodeId: string): { def: NodeTypeDef; rows: Record<string, unknown>[] } | null {
  const node = useCookInputs.getState().nodes[nodeId];
  const def = node && getNodeDefs()[node.typeId];
  if (!def || def.ports_from_side !== "inputs" || !def.ports_from) return null;
  return { def, rows: tableRows(def, node.params) as unknown as Record<string, unknown>[] };
}

/** One row added at the end of a ports_from-input table: its port name fresh, its label the node's own for that place
 * (「切换」: 第三路…) or from what is wired in (`fromPort`, `dataType`), layer, layer2… with nothing wired. Returns the new
 * row's port name ("": not such a node, or its table is full: rules.ts rowsLeft). */
function appendRow(nodeId: string, fromPort = "", dataType = ""): string {
  const table = inputTable(nodeId);
  if (!table) return "";
  const { def, rows } = table;
  const name = freshRowName(def, rows.map((r) => String(r.name)));
  if (!name) return "";
  const label = defaultRowLabel(def, fromPort, dataType, rows.map((r) => pick(r.label)));
  setParam(nodeId, def.ports_from, [...rows, { name, label }]);
  return name;
}

/** A click on 「＋」 (on the node) or on the table's 「添加」 (parameter panel): an empty row at the end of the table, and so
 * one more input, layer name layer, layer2… (「切换」: 第三路, 第四路…, ports c, d…). A row not wired is skipped when
 * cooking and said once (the rows of nodes/base.py made_ports are all optional ports); a full table (「切换」's ten ways)
 * gets nothing (then neither 「＋」 nor 「添加」 is shown). */
export function addEmptyRow(nodeId: string): void {
  appendRow(nodeId);
}

/** Renames a row on the node (its layer name): only that row's `label` changes, its port name `name` does not, so no
 * wire breaks. An empty name changes nothing. */
export function renameRow(nodeId: string, rowName: string, label: string): void {
  const table = inputTable(nodeId);
  const text = label.trim();
  if (!table || !text) return;
  const { def, rows } = table;
  if (!rows.some((r) => r.name === rowName && pick(r.label) !== text)) return;
  // a row the node names itself (「切换」's ways: ports_from_names) carries a display name: what is typed goes into the
  // page's language, the other kept, an old plain one first put where its language says (state/cookInputs.ts edited,
  // the rule of every display field). A row named after what is wired in (「多层 EXR 输出」's 图层) is data, the name the
  // file's layer gets: it stays a plain string
  const display = !!def.ports_from_names?.length;
  // left as shown, the node's own word for an unnamed row (第二路 / Input 2): it keeps no name, so it still follows the language
  const place = def.ports_from_labels?.[def.ports_from_names?.indexOf(rowName) ?? -1];
  if (display && text === place && !pick(rows.find((r) => r.name === rowName)?.label)) return;
  setParam(nodeId, def.ports_from, rows.map((r) => (r.name === rowName ? { ...r, label: display ? edited(r.label, text) : text } : r)));
}

/** A wire dropped on a ports_from-input node's body (not on one of its ports): the same as dropping it on the node's
 * 「＋」 (graph/rules.ts ADD_ROW): connect() adds a row named after what is wired in and connects there, or says why
 * the wire does not fit (then no row is added). Not such a node, or the source type is not known yet: does nothing
 * (returns false). */
export function addPortRow(targetId: string, fromNode: string, fromPort: string): boolean {
  const table = inputTable(targetId);
  if (!table || !outputType(snapshotNow(), fromNode, fromPort)) return false;
  const node = useCookInputs.getState().nodes[targetId];
  if (!rowsLeft(table.def, node.params)) {  // full (「切换」's ten ways): no row to add, said rather than dropped quietly
    const count = table.def.ports_from_names?.length ?? table.rows.length;
    say(msg("B-WIRE-ROWSFULL", { node: nodeRefOf(targetId), count, word: table.def.ports_from_word }), targetId);
    return true;
  }
  connect({ source: fromNode, sourceHandle: fromPort, target: targetId, targetHandle: ADD_ROW });
  return true;
}

/** Several 「序列图输出设置」 nodes replaced by one 「多层 EXR 输出设置」, one row per former node (its 名字 the row's
 * 图层名), rewired to the same sources and the same 「输出」; one step. */
export function mergeToExr(ids: string[]): void {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const from = ci.nodes[ids[0]]?.typeId ?? "";
  const exrDef = getNodeDefs()[getNodeDefs()[from]?.merges_into ?? ""];
  const targetIds = ids.filter((id) => ci.nodes[id]?.typeId === from);
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
  const id = defaultName(exrDef.id, idTaken);
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
  useCookInputs.getState().insertNode(id, { typeId: exrDef.id, params: { ...exrDef.defaults, layers: rows } });
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
  const at = { source: wordIn(snap.nodes, c.source), node: wordIn(snap.nodes, dst.id), input: port.label };
  // whether it may be wired is asked of graph/wireRule.ts only (the same place as the green ports while dragging and
  // moving a bundle): dragging a wire, picking a node from the menu after dragging out of a port, and one-click insert
  // all come through here
  const ci = useCookInputs.getState();
  const problem = wireProblem(snap, { node: c.source, port: c.sourceHandle ?? "" }, { node: c.target, port: c.targetHandle ?? "" }, ci.edges);
  if (problem) {
    say(problem, problem.code === "B-WIRE-OUTINACTIVE" ? c.source : dst.id);
    return;
  }
  const sourceHandle = c.sourceHandle ?? "";
  // dropped on 「＋」 (graph/rules.ts ADD_ROW): only once the checks above passed is a row added at the end of the table
  // (named after the source port), and the wire goes to the new row's port
  const targetHandle = port.name === ADD_ROW ? appendRow(c.target, sourceHandle, srcType) : c.targetHandle ?? "";
  if (!targetHandle && port.name === ADD_ROW) return;
  const edges = useCookInputs.getState().edges; // read after the row was added
  const old = port.multi ? undefined : edges.find((e) => e.target === c.target && e.targetHandle === targetHandle);
  const kept = port.multi ? edges : edges.filter((e) => e !== old);
  const id = wireKey(c.source, sourceHandle, c.target, targetHandle);
  useCookInputs.getState().setEdges([...kept, { id, source: c.source, sourceHandle, target: c.target, targetHandle }]);
  preselectRule(c.source, c.target);
  if (old) {
    say(msg("I-WIRE-REPLACED", { ...at, was: nodeRefOf(old.source) }), dst.id);
  }
  justWired.add(id); // what is wrong with it, if anything, is said once the server has judged it (sayJudgedWires)
}

/** A wire from a node with a two-skeleton handle (rig_pair, its 「忽略规则」 role) straight into a solver whose catalogue
 * entry lists the ignore rules it accepts (NodeTypeDef.retarget_rules): while the rule is still at its default, it
 * becomes that solver's first rule — in the same undo step as the wire (written in the same task; model/rigPair.ts
 * wiredRule). A rule the user picked is never changed. */
function preselectRule(source: string, target: string): void {
  const ci = useCookInputs.getState();
  const src = ci.nodes[source], dst = ci.nodes[target];
  const srcDef = getNodeDefs()[src?.typeId ?? ""];
  const role = srcDef?.handles.find((h) => role3d(h) === "pair")?.params.ignore_rule;
  if (!src || !srcDef || !role) return;
  const rule = wiredRule(getNodeDefs()[dst?.typeId ?? ""]?.retarget_rules, src.params[role], srcDef.defaults[role]);
  if (rule) editParams(source, { [role]: rule });
}

/** A Ctrl-carried bundle let go (editor/graphPointer.ts): the carried wires all move to `to` (the rule: graph/rewire.ts).
 * When they cannot, says why and returns false (the wires are still carried); not a landing place (the port they came
 * from, the other side): null; moved: true. One write of the wires = one undo step 「整组改接」. */
export function moveWires(held: Held, to: RewireEnd, side: "output" | "input"): boolean | null {
  const snap = snapshotNow();
  const got = rewire(snap, useCookInputs.getState().edges, held, to, side);
  if (!got) return null;
  if ("refused" in got) {
    say(got.refused, to.node);
    return false;
  }
  transaction(() => t("ui.history.rewire"), "rewire", () => useCookInputs.getState().setEdges(got.edges));
  for (const w of got.edges) if (!held.wires.some((h) => h.id === w.id)) justWired.add(w.id); // what is wrong with it, if anything, is said once the server has judged it (as in connect)
  if (got.replaced) {
    const port = inputPort(snap, to.node, to.port);
    say(msg("I-WIRE-REPLACED", { source: wordIn(snap.nodes, held.wires[0].source), node: wordIn(snap.nodes, to.node), input: port?.label ?? to.port,
      was: wordIn(snap.nodes, got.replaced.source) }), to.node);
  }
  return true;
}

export function setParam(id: string, name: string, value: unknown): void {
  editParams(id, { [name]: value });
}

/** Writes a derived parameter value (editor/ColorspaceFill.tsx: fills an empty 「色彩空间」 from the file format), not a
 * user step (graph/history.ts derived: no step, not marked unsaved; as a step, undo would empty it and it would be
 * filled again, so undo would never work). An unchanged value is not written. */
export function setDerivedParam(id: string, name: string, value: unknown): void {
  const node = useCookInputs.getState().nodes[id];
  if (!node || node.params[name] === value) return;
  derived(() => useCookInputs.getState().setNode(id, { params: { ...node.params, [name]: value } }));
}

/** Changes a set of parameters at once, as one undo step (a pasted lens holds a dozen values; one undo restores them
 * all). The same path as setParam, so derived_from parameters (the distortion table that follows 「镜头模型」) are
 * recomputed as well. */
export function setParams(id: string, changes: Record<string, unknown>): void {
  editParams(id, changes);
}

/** Parameters of several nodes at once, as one undo step named `label` (the parameter interface's 「全开 / 全关」 of a
 * group of switches: model/groupSwitches.ts). Each node's share goes through the same path as setParams. */
export function setParamsAcross(label: string | (() => string), changes: { target: string; value: unknown }[]): void {
  const byNode = new Map<string, Record<string, unknown>>();
  for (const c of changes) {
    const [id, name] = c.target.split(".");
    byNode.set(id, { ...(byNode.get(id) ?? {}), [name]: c.value });
  }
  if (!byNode.size) return;
  transaction(typeof label === "string" ? () => label : label, "params-across", () => {
    for (const [id, values] of byNode) editParams(id, values);
  });
}

/** A node's parameters change (with whatever else of its data goes with them, `extra`: e.g. pickFile's own `picked`
 * record): the parameters derived from them (P(derived_from): an import node's singleton-kind auto-selection,
 * 「LensDistortion」's 畸变参数 from its 镜头模型) are requested from the server first, so the whole change is one edit.
 * `extra` is computed against the node as it was before `changes`. */
function editParams(id: string, changes: Record<string, unknown>, extra: (node: CookNode) => Partial<CookNode> = () => ({}), linked = true): void {
  const ci = useCookInputs.getState();
  const node = ci.nodes[id];
  const def = getNodeDefs()[node?.typeId ?? ""];
  if (!node || !def) return;
  const unstored = node.stored && Object.keys(changes).some((k) => k in node.stored!) ? { stored: Object.fromEntries(Object.entries(node.stored).filter(([k]) => !(k in changes))) } : {};
  const derived = def.params.filter((p) => p.derived_from.some((d) => d in changes));
  const asked = { ...node.params, ...changes };
  // the user's change applies at once (one undo step). The derived parameters are written into the node as it is when
  // the server answers, joined into the user's step and the ones after it (graph/history.ts absorb: undo / redo carry them). Writing back the
  // whole snapshot taken when the request was sent would overwrite other parameters the user changed meanwhile
  useCookInputs.getState().setNode(id, { ...extra(node), ...unstored, params: asked });
  // a row removed from a node whose inputs come from a table (「去掉」 in the parameter panel's table): the wires into
  // that row's port go too, in the same undo step. Otherwise a wire would point at a port the node no longer has, and
  // the server could not read the graph at all (engine/graph.py _connect E-GRAPH-NOSUCHPORT)
  if (def.ports_from_side === "inputs" && def.ports_from && def.ports_from in changes) {
    const rows = new Set(tableRows(def, asked).map((r) => r.name));
    const before = tableRows(def, node.params).map((r) => r.name).filter((n) => !rows.has(n));
    if (before.length) {
      const edges = useCookInputs.getState().edges;
      useCookInputs.getState().setEdges(edges.filter((e) => !(e.target === id && before.includes(e.targetHandle))));
    }
  }
  // a parameter tied to others through an entry of the parameter interface driving several (state/cookInputs.ts
  // linkedTargets: two trackers sharing one set of picks) takes the same value in every one of them, whichever way it
  // was set (the panel, app mode, a handle in the view), in the same step; a picked file's record goes along
  if (linked) {
    const tree = useCookInputs.getState().exposed;
    const picked = useCookInputs.getState().nodes[id]?.picked;
    const others = new Map<string, Record<string, unknown>>();
    for (const [name, v] of Object.entries(changes))
      for (const key of linkedTargets(tree, `${id}.${name}`)) {
        const [nid, pname] = splitTarget(key);
        others.set(nid, { ...(others.get(nid) ?? {}), [pname]: v });
      }
    for (const [nid, values] of others)
      editParams(nid, values, (n) => {
        const names = Object.keys(values);
        const from = Object.fromEntries(Object.keys(changes).filter((k) => picked?.[k]).map((k) => [k, picked![k]]));
        const keep = Object.fromEntries(Object.entries(n.picked ?? {}).filter(([k]) => !names.includes(k)));
        const take = Object.fromEntries(names.filter((k) => from[k]).map((k) => [k, from[k]]));
        return n.picked || Object.keys(take).length ? { picked: { ...keep, ...take } } : {};
      }, false);
  }
  if (!derived.length) return;
  const into = anchor(); // the step this change is recorded in: the answer joins it, whatever the user did meanwhile
  const sources = new Set(derived.flatMap((p) => p.derived_from));
  const land = (more: Record<string, unknown>) => {
    const now = useCookInputs.getState().nodes[id];
    if (!now) return; // the node is gone
    // asked about values that are not the node's now: its step was undone (the answer still belongs to it: fillUndone,
    // redo brings it back), or a later edit changed them (that edit's answer applies; fillUndone does nothing)
    if ([...sources].some((k) => now.params[k] !== asked[k])) return fillUndone(into, id, more, node.params);
    // only what the user has not set since asking: a derived value typed by hand meanwhile stays (graph/history.ts absorb
    // stops at that step the same way)
    const fill = Object.fromEntries(Object.entries(more).filter(([k]) => same(now.params[k] as Json, node.params[k] as Json)));
    if (Object.keys(fill).length) absorb(into, () => useCookInputs.getState().setNode(id, { params: { ...now.params, ...fill } }));
  };
  void api.derive(def.id, asked).then(
    (got) => land(got),
    (e) => {
      const names = def.params.filter((p) => p.derived_from.length).map((p) => p.label);
      say(msg("E-PARAM-DERIVE", { node: nodeRefOf(id), params: names, reason: reasonOf(e) }), id);
      land(Object.fromEntries(derived.map((p) => [p.name, def.defaults[p.name]])));
    },
  );
}

/** 提升到节点 / 取消提升 (the parameter panel's button): a promoted parameter gets an input `param:<name>` on the node
 * and can be edited right on that row (graph/nodes.ts nodeRows). Unpromoting also disconnects the wire into that port. */
export function togglePromoted(id: string, name: string): void {
  const ci = useCookInputs.getState();
  const node = ci.nodes[id];
  if (!node) return;
  const on = node.promoted?.includes(name);
  const promoted = on ? node.promoted!.filter((p) => p !== name) : [...(node.promoted ?? []), name];
  useCookInputs.getState().setNode(id, { promoted: promoted.length ? promoted : undefined });
  if (on) useCookInputs.getState().setEdges(ci.edges.filter((e) => !(e.target === id && e.targetHandle === PARAM + name)));
}

/** 在节点上显示 on / off (the parameter panel's mark): only whether the row shows on the node, with no input of its
 * own; the input is the neighbouring 「提升到节点」's (togglePromoted).
 *
 * The user may remove rows the author shows by default: the list starts as a copy of the type's declared rows
 * (chosenOnNode), and a click removes from that copy. So the mark on every row works; otherwise the author's default
 * rows (often exactly the ones taking up space) would not react to a click. */
export function toggleOnNode(id: string, name: string): void {
  const def = getNodeDefs()[useCookInputs.getState().nodes[id]?.typeId ?? ""];
  if (!def) return;
  const was = chosenOnNode(def, useLook.getState().onNode[id]);
  // after a click the node's list is always saved explicitly (even when it equals the factory default again): the json
  // counts, the type's list is only the initial value on creation and the fallback when the json has none. Saving
  // undefined would make the node follow the factory default from then on: two sources for one piece of data
  useLook.getState().setOnNode(id, def.params.filter((p) => p.simple && (p.name === name ? !was.includes(name) : was.includes(p.name))).map((p) => p.name));
}

// Picking a file is a parameter change like any other (editParams): an import node's singleton-kind selection (the
// one camera, the one skeleton animation) is derived by the server from the file. A bare setNode here would silently
// drop that derivation, leaving every port that depends on it blank until something else touches the node's params.
// 「读取序列」 has no layer table: its ports are made straight from the file's layers (nodes/core/input.py made_ports),
// so once a file is picked, one parameter change and one status reply bring the ports.
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
    const names = def.params.filter((p) => p.derived_from.length).map((p) => p.label);
    say(msg("E-PARAM-DERIVE", { node: nodeRefOf(id), params: names, reason: reasonOf(e) }), id);
    return null;
  }
}

/** Renames a node (its id, Houdini's node name: graph/naming.ts): refused with the reason (returned, nothing changes)
 * when the name breaks the rule or another node has it. Everything that names the node follows in one undo step: the
 * cook inputs (wires, exposed targets: state/cookInputs.ts renameNode), the document's look (position, rows, comment,
 * boxes, the node shown: state/look.ts renameNode) and, outside the document, the selection, the expanded nodes, the
 * canvas's per-node state and the last answers by node, so nothing blinks while the server is asked again. */
export function renameNode(from: string, name: string): string | null {
  const to = name.trim();
  const node = useCookInputs.getState().nodes[from];
  if (!node || to === from) return null;
  const problem = nameProblem(to, (id) => id !== from && idTaken(id));
  if (problem) return problem;
  transaction(() => t("ui.history.rename", { from, to }), `rename ${from}`, () => {
    useCookInputs.getState().renameNode(from, to);
    useLook.getState().renameNode(from, to);
  });
  const id = (x: string) => (x === from ? to : x);
  const move = <T,>(r: Record<string, T>): Record<string, T> => {
    if (!(from in r)) return r;
    const { [from]: v, ...rest } = r;
    return { ...rest, [to]: v };
  };
  useViewer.setState((s) => ({ selectedId: s.selectedId && id(s.selectedId), expanded: s.expanded.map(id), canvas: move(s.canvas) }));
  useResults.setState((s) => ({ results: move(s.results), byNode: move(s.byNode), running: move(s.running) }));
  return null;
}

/** A node's comment (Houdini's node comment: any text, several lines) and whether it shows beside the node; an empty
 * text removes it. One undo step, like any edit of the document's look. */
export function setComment(id: string, text: string, show?: boolean): void {
  const was = useLook.getState().comments[id];
  useLook.getState().setComment(id, text.trim() ? { text, show: show ?? was?.show ?? true } : undefined);
}

/** Exposes / unexposes a parameter (the pin beside it). The parameter interface is a tree (api/catalog.ts
 * ExposedEntry): unexposing removes it from its group, which stays; a newly exposed one goes to the end of the root,
 * and from there is dragged in the parameter panel (the exposed-parameter tree shown with no node selected) or into a
 * group in the 「编辑参数界面」 dialog (the small button at the far right of the parameter panel's title row). */
export function toggleExposed(nodeId: string, param: string, label: Words): void {
  const ci = useCookInputs.getState();
  const target = `${nodeId}.${param}`;
  const all = exposedParams(ci.exposed);
  if (all.some((x) => drives(x, target))) {
    useCookInputs.getState().setExposed(withoutTarget(ci.exposed, target));
    return;
  }
  const name = all.some((x) => x.name === param) ? `${nodeId}_${param}` : param;
  // a new display field: in the page's language ({lang: text}; the parameter's own name as the page shows it)
  useCookInputs.getState().setExposed([...ci.exposed, { name, label: typeof label === "string" ? newWords(label) : label, target }]);
}

/** The parameter interface as the 「编辑参数界面」 dialog hands it back (its 「确定」), one undo step. An entry driving
 * several node parameters holds one value in all of them (the server refuses a template where they differ:
 * E-EXPOSED-TARGETS): one just made by 「合并」 takes its first target's value (and picked file) into the others. */
export function setInterface(tree: ExposedEntry[]): void {
  transaction(() => t("ui.interface.title"), "interface", () => {
    useCookInputs.getState().setExposed(tree);
    const defs = getNodeDefs();
    for (const x of exposedParams(tree)) {
      const keys = targetsOf(x);
      if (keys.length < 2) continue;
      const nodes = useCookInputs.getState().nodes;
      const [nid, pname] = splitTarget(keys[0]);
      const n = nodes[nid];
      const def = n && defs[n.typeId];
      if (!n || !def || def.params.find((p) => p.name === pname)?.widget === "button") continue;
      const value = n.params[pname] ?? def.defaults[pname] ?? null;
      const picked = n.picked?.[pname];
      for (const key of keys.slice(1)) {
        const [oid, oname] = splitTarget(key);
        const o = nodes[oid];
        if (!o || same((o.params[oname] ?? defs[o.typeId]?.defaults[oname] ?? null) as Json, value as Json)) continue;
        editParams(oid, { [oname]: value }, (m) => (picked ? { picked: { ...(m.picked ?? {}), [oname]: picked } } : {}), false);
      }
    }
  });
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
  return takeCopy(ci.graphId, order, boxes, ci.nodes, look.positions, look.onNode, look.comments, ci.edges);
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
    const id = defaultName(n.data.typeId, idTaken);
    const params = { ...n.data.params };
    for (const p of def?.params ?? []) {
      if (!p.unique) continue;
      const taken = new Set(now.order.map((o) => String(now.nodes[o].params[p.name] ?? "").toLowerCase()));
      params[p.name] = uniqueCopyValue(String(params[p.name] ?? def!.defaults[p.name] ?? p.name), taken);
    }
    now.insertNode(id, { ...structuredClone(n.data), params });
    useLook.getState().setPosition(id, n.pos.x + d.x, n.pos.y + d.y);
    if (n.onNode) useLook.getState().setOnNode(id, [...n.onNode]);
    if (n.comment) useLook.getState().setComment(id, { ...n.comment });
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
  const base = { id: `box:${n}`, label: "", color: BOX_COLORS[(n - 1) % BOX_COLORS.length], collapsed: false, members: [] };
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

/** 「整理节点图」 (bottom right of the canvas, editor/NodeEditor.tsx): layered layout (graph/layout.ts, the same code as
 * tools/layout_graph.mjs), writing back node positions and group boxes as one undo step 「整理节点图」. With nodes
 * selected, only those are arranged (with the wires between them) and put back at the top left of their former bounding
 * box; otherwise the whole graph, its top left at (0, 0). Node sizes are the canvas's measured ones (graph/nodes.ts
 * nodeSize, as the canvas). */
export function arrangeGraph(): void {
  const snap = snapshotNow();
  const canvas = useViewer.getState().canvas;
  const picked = snap.nodes.filter((n) => canvas[n.id]?.selected).map((n) => n.id);
  const only = picked.length > 1 ? new Set(picked) : null;
  const nodes = snap.nodes.filter((n) => !only || only.has(n.id)).map((n) => {
    const { w, h } = nodeSize({ measured: canvas[n.id]?.measured });
    return { id: n.id, x: n.position.x, y: n.position.y, w, h, delivers: !!snap.nodeDefs[n.data.typeId]?.delivers };
  });
  if (!nodes.length) return;
  const ids = new Set(nodes.map((n) => n.id));
  const ci = useCookInputs.getState();
  const edges = snap.edges.filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((e) => {
      const to = snap.nodes.find((n) => n.id === e.target);
      return { from: e.source, to: e.target, param: (e.targetHandle ?? "").startsWith(PARAM),
               slot: to && inputSlot(snap.nodeDefs[to.data.typeId], ci.nodes[e.target]?.params, e.targetHandle ?? "") };
    });
  const blocks = eachBlocks(ci.order.filter((id) => ids.has(id)).map((id) => ({ id, type: ci.nodes[id].typeId, block: ci.nodes[id].params.block })), snap.nodeDefs);
  const look = useLook.getState();
  // boxes: their members are taken before arranging (an open box by node centres inside it, a collapsed one by its own
  // list); when only the selection is arranged, only boxes whose members are all selected move
  // a collapsed box is laid out at its collapsed size, as one node (graph/layout.ts, 9)
  const boxes = look.boxes.map((b) => ({ id: b.id, x: b.x, y: b.y, w: b.w, h: b.h, members: boxContents(b, snap.nodes),
    ...(b.collapsed ? { folded: { w: Math.min(b.w, BOX_FOLD_W), h: BOX_HEAD } } : {}) }))
    .filter((b) => b.members.length && b.members.every((m) => ids.has(m)));
  const anchor = only ? { x: Math.min(...nodes.map((n) => n.x)), y: Math.min(...nodes.map((n) => n.y)) } : { x: 0, y: 0 };
  const out = layoutGraph({ nodes, edges, blocks, boxes, anchor });
  transaction(() => t("ui.history.arrange"), "arrange", () => useLook.getState().place(out.positions, out.boxes));
}
