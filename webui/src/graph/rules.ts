import type { Catalog, CookCase, HandleDef, NodePorts, NodeStatus, NodeTypeDef, ParamDef, PortDef, ResolvedCost, StatusReply, WireStatus } from "../api";
import type { LooseWire } from "../state/viewer";

/** What the editor knows of the graph's rules: nothing here works a rule out.
 * Every answer is read from the server: the status reply (a node's ports, handles, cost, licence, what a click and
 * showing it cook; every wire's state) or the catalogue's lookup tables (which types a port takes, which node converts
 * one type into another, what a node type refuses, its ports and handles at its defaults). The tables exist only for
 * what has to be answered before a status reply can: a wire still being drawn, a node just added, the drawing between
 * an edit and its reply (the last reply is kept meanwhile, so nothing flashes: pendingPorts).
 *
 * These read what the server resolves; no copy of a server rule is kept here.
 * Pure: types only. */

/** The input a promoted parameter gets: "param:<name>" (nodes/port.py PARAM). */
export const PARAM = "param:";

/** The graph as the lookups read it: a snapshot (graph/snapshot.ts) has all of it. */
export interface GraphView {
  nodes: { id: string; data: { typeId: string; label: string; promoted?: string[]; params?: Record<string, unknown> } }[];
  edges: { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }[];
  nodeDefs: Record<string, NodeTypeDef>;
  reply: StatusReply | null; // the last status reply, trusted or not
  results?: Record<string, NodeStatus>; // trusted
  catalog?: Catalog | null;
}

// ------------------------------------------------------------------ the catalogue's tables

const ACCEPTS = new WeakMap<Catalog, Map<string, Set<string>>>();

/** Does an input declared `portType` take what output `out` carries? The server's rule (engine/graph.py Graph.takes), the one
 * every way of wiring asks (dragging a wire: editor/graphPointer.ts canWire; dropping it: graph/edit.ts connect; moving a
 * bundle: graph/rewire.ts): its type fits, or it is one of several ways (「切换」: `ways`) and every way fits (「动作重定向」's
 * 动作 takes 骨架动画 or 蒙皮角色, so a switch between the two, whose common kind is 场景, goes straight in). */
export function takes(catalog: Catalog | null | undefined, portType: string, out: { type: string; ways?: string[] }): boolean {
  return portAccepts(catalog, portType, out.type) || (!!out.ways?.length && out.ways.every((w) => portAccepts(catalog, portType, w)));
}

/** Does a port declared as `portType` take data of `dataType`? (catalogue `accepts`, nodes/registry.py type_tables) */
export function portAccepts(catalog: Catalog | null | undefined, portType: string, dataType: string): boolean {
  if (!catalog) return false;
  let table = ACCEPTS.get(catalog);
  if (!table) ACCEPTS.set(catalog, (table = new Map(Object.entries(catalog.accepts).map(([p, ds]) => [p, new Set(ds)]))));
  return !!table.get(portType)?.has(dataType);
}

/** The node type that turns data of `dataType` into what a port of `portType` takes, "" none (catalogue `converters`). */
export const converter = (catalog: Catalog | null | undefined, dataType: string, portType: string): string => catalog?.converters[dataType]?.[portType] ?? "";

/** Why a node type can't take data of this type ("" it can): its catalogue entry's `refuses`. */
const refusedBy = (def: NodeTypeDef | undefined, dataType: string): string => def?.refuses[dataType] ?? "";

/** Can a value in unit `have` drive a parameter in unit `want`? The catalogue's units: those of one kind convert. */
function unitFits(catalog: Catalog | null | undefined, have: string, want: string): boolean {
  if (!have || !want || have === want) return true;
  const a = catalog?.units[have];
  return !!a && a.kind === catalog?.units[want]?.kind;
}

/** Parameter -> String(value) -> the licence class that value switches the node to (catalogue `option_traits`:
 * lab2shot/nodes/tags.py noncommercial / research), for the values that switch it at all. */
export function licensedChoices(def: NodeTypeDef): Record<string, Record<string, string>> {
  const kept = LICENSED.get(def);
  if (kept) return kept;
  const made: Record<string, Record<string, string>> = {};
  for (const [name, rows] of Object.entries(def.option_traits)) {
    const values = Object.fromEntries(Object.entries(rows).filter(([, r]) => r.licence).map(([v, r]) => [v, r.licence]));
    if (Object.keys(values).length) made[name] = values;
  }
  LICENSED.set(def, made);
  return made;
}
const LICENSED = new WeakMap<NodeTypeDef, Record<string, Record<string, string>>>();
const NO_VALUES: Record<string, string> = {};
export const licensedValues = (def: NodeTypeDef | undefined, name: string): Record<string, string> =>
  (def ? licensedChoices(def)[name] ?? NO_VALUES : NO_VALUES);

/** An option's name as the node type declares it (`option_labels`), else the value itself: the one place the name is
 * read, for a pull-down (ui/controls.tsx optionView adds the licence word and why it is off) and a message alike. */
export const optionName = (p: Pick<ParamDef, "option_labels">, o: unknown): string => p.option_labels?.[String(o)] ?? String(o);

// ------------------------------------------------------------------ a node's ports, cost, licence, handles

const EMPTY_PORTS: NodePorts = { inputs: [], outputs: [], waiting: [] };

const typeOf = (s: GraphView, id: string) => s.nodes.find((n) => n.id === id)?.data.typeId ?? "";

/** A node's ports: as the last status reply says them, else (a node added since) its type's at its defaults. */
export function pendingPorts(s: GraphView, id: string): NodePorts {
  return s.reply?.nodes[id]?.ports ?? s.nodeDefs[typeOf(s, id)]?.at_defaults.ports ?? EMPTY_PORTS;
}

export const outputsOf = (s: GraphView, id: string): PortDef[] => pendingPorts(s, id).outputs;

/** A node's main result, which the viewer shows and a label names when nothing else was chosen: the output its type
 * names (NodeTypeDef.main) when the node has that output now, else its first output — the server's rule
 * (nodes/base.py main_output). The fallback is needed: 读取序列's outputs are the file's layers, so its type names
 * none, and an importer's named output (the camera) is there only when the file holds one. The page asks here and
 * nowhere else. */
export const mainOutput = (s: GraphView, id: string): PortDef | undefined => {
  const outputs = outputsOf(s, id);
  const main = s.nodeDefs[typeOf(s, id)]?.main;
  return outputs.find((p) => p.name === main) ?? outputs.at(0);
};

/** The rows of a node's input table (a ports_from table landing on inputs: 「多层 EXR 输出设置」's 图层), each a port
 * named by its row (nodes/base.py made_ports); [] when the node has no such table. The rows are the document's
 * own (its parameters), so this is what the ports ARE, not a guess: reading a graph file and drawing the node's inputs
 * both ask here. */
export function tableRows(def: NodeTypeDef | undefined, params: Record<string, unknown> | undefined): { name: string; label: string }[] {
  if (!def || def.ports_from_side !== "inputs" || !def.ports_from) return [];
  return ((params?.[def.ports_from] as { name: string; label: string }[] | undefined) ?? []).filter((r) => r && typeof r.name === "string");
}

/** Whether one more row fits in a node's input table: a node that names its rows itself (`ports_from_names`,
 * 「切换」's a…j) has as many as it has names; any other takes as many as wanted. False for a node without one. */
export function rowsLeft(def: NodeTypeDef | undefined, params: Record<string, unknown> | undefined): boolean {
  if (!def || def.ports_from_side !== "inputs" || !def.ports_from) return false;
  const names = def.ports_from_names;
  return !names || tableRows(def, params).length < names.length;
}

/** The parameters that have an input on the node, in the order they sit there: the type's standing ones
 * (NodeDef.wired_ports, e.g. AnyCalib's six numbers into 「LensDistortion」) and then the ones this node promoted.
 * The one place that list is built: the ports drawn, the wires a graph file may name, and where a parameter's value
 * comes from all read it here. */
export function paramPortNames(def: NodeTypeDef | undefined, promoted: string[] | undefined): string[] {
  const own = (promoted ?? []).filter((n) => !def?.wired_ports?.includes(n));
  return [...(def?.wired_ports ?? []), ...own].filter((n) => def?.param_ports[n]);
}

/** A node's inputs: its declared ones (as the last reply resolved them), one per row of its input table in the
 * document's row order, then the input of each parameter that has one (paramPortNames: the type's standing ones,
 * then those promoted; the catalogue's `param_ports`). The rows and the promoted list are the document's own, so a
 * row just added (「＋」 clicked, a wire dropped on 「＋」 or on the node's body: graph/edit.ts addEmptyRow / connect) or a parameter just promoted takes a wire before the next reply comes; a
 * row the reply already has keeps the reply's port. With no parameters known (a bare view) the reply's rows stand. */
export function inputsOf(s: GraphView, id: string): PortDef[] {
  const node = s.nodes.find((n) => n.id === id);
  const def = s.nodeDefs[node?.data.typeId ?? ""];
  const own = pendingPorts(s, id).inputs.filter((p) => !p.name.startsWith(PARAM));
  // the parameters' inputs: the standing ones the node type declares (NodeDef.wired_ports, e.g. where AnyCalib's six
  // numbers go into 「LensDistortion」) and the parameters this node promoted itself. The server's reply has both, but
  // the line above filtered out every param: port, so they are added back here by name; taking only `promoted` would
  // miss the standing ones
  const promoted = paramPortNames(def, node?.data.promoted).map((name) => def!.param_ports[name]);
  if (!def || def.ports_from_side !== "inputs" || !node?.data.params) return [...own, ...promoted];
  const declared = new Set(def.inputs.map((p) => p.name));
  const replied = new Map(own.map((p) => [p.name, p]));
  // a row's name (the layer name) is the document's: a rename on the node shows at once, without waiting for the next reply
  const rows = tableRows(def, node.data.params).map(
    (r): PortDef => ({ ...(replied.get(r.name) ?? { ...ROW_PORT, name: r.name, type: def.ports_from_type, type_label: def.ports_from_type_label }), label: r.label }),
  );
  return [...own.filter((p) => declared.has(p.name)), ...rows, ...promoted];
}

// inputs made from a table are all optional: a row added but not wired yet is skipped by the node, said once
// (nodes/base.py made_ports)
const ROW_PORT: PortDef = { name: "", type: "", type_label: "", label: "", optional: true, multi: false, list: false, type_from: "", inserts: "", unit: "" };

/** The port name of 「＋」: the one port that always stands at the end of the input list of a node whose inputs are
 * made from a table (`ports_from_side === "inputs"`, e.g. 「多层 EXR 输出设置」; editor/GraphNode.tsx). It is not a row
 * of the table, is not in `inputsOf` and never appears in a graph file's wires: a wire dropped on it makes `connect()`
 * (graph/edit.ts) add a row first and connect to the new row's port. A click on it adds an empty row.
 * Judged by the declaration, so every such node has it, not one particular node. A row's port name is `row1`… (the
 * node's own names where it names its rows, 「切换」's a…j) or an older file's name, never "+". A full table
 * (`rowsLeft` false) has no 「＋」. */
export const ADD_ROW = "+";

/** 「＋」 as a port: its type is the type of every row of the table (`ports_from_type`), so whether a wire being drawn
 * or carried may land on it is the same rule as for an existing row (the type checks of canWire and connect apply as
 * usual). undefined for a node that is not of this kind. */
function addRowPort(s: GraphView, id: string): PortDef | undefined {
  const def = s.nodeDefs[typeOf(s, id)];
  // a table that is full (「切换」 with its ten ways) has no 「＋」: nothing lands there
  if (!def || !rowsLeft(def, s.nodes.find((n) => n.id === id)?.data.params)) return undefined;
  return { ...ROW_PORT, name: ADD_ROW, label: "＋", type: def.ports_from_type, type_label: def.ports_from_type_label };
}

export const outputPort = (s: GraphView, id: string, port: string | null | undefined): PortDef | undefined =>
  port == null ? undefined : outputsOf(s, id).find((p) => p.name === port);

export const inputPort = (s: GraphView, id: string, port: string | null | undefined): PortDef | undefined =>
  port == null ? undefined : port === ADD_ROW ? addRowPort(s, id) : inputsOf(s, id).find((p) => p.name === port);

export const outputType = (s: GraphView, id: string, port: string | null | undefined): string | undefined => outputPort(s, id, port)?.type;

/** What running the node costs with its parameters: the last reply's, else its type's at its defaults. */
export function costOf(s: GraphView, id: string): ResolvedCost | undefined {
  return s.reply?.nodes[id]?.cost ?? s.nodeDefs[typeOf(s, id)]?.at_defaults.cost;
}

/** Commercial use allowed with its parameters: the last reply's word, else its type's at its defaults. */
export function commercialOf(s: GraphView, id: string): boolean {
  return s.reply?.nodes[id]?.licence?.commercial ?? s.nodeDefs[typeOf(s, id)]?.at_defaults.licence.commercial ?? true;
}

/** The node's viewer handles that apply now (the reply's, else its type's at its defaults). */
export function handlesOf(s: GraphView, id: string): HandleDef[] {
  const def = s.nodeDefs[typeOf(s, id)];
  if (!def) return [];
  const on = s.reply?.nodes[id]?.handles ?? def.at_defaults.handles;
  return def.handles.filter((_, i) => on.includes(i));
}

// ------------------------------------------------------------------ wires

/** A wire's id in the editor (graph/edit.ts connect): its two ends. */
export const wireKey = (source: string, sourceHandle: string, target: string, targetHandle: string): string => `${source}.${sourceHandle}->${target}.${targetHandle}`;

const WIRES = new WeakMap<StatusReply, Map<string, WireStatus>>();

/** A wire as the last reply judged it (undefined: not in it, e.g. connected since). */
export function wireState(s: GraphView, e: { source: string; sourceHandle: string | null; target: string; targetHandle: string | null }): WireStatus | undefined {
  if (!s.reply) return undefined;
  let table = WIRES.get(s.reply);
  if (!table) WIRES.set(s.reply, (table = new Map(s.reply.wires.map((w) => [wireKey(w.from[0], w.from[1], w.to[0], w.to[1]), w]))));
  return table.get(wireKey(e.source, e.sourceHandle ?? "", e.target, e.targetHandle ?? ""));
}

/** Where a parameter driven by a wire gets its value, as the parameter shows it. */
export function wiredFrom(s: GraphView & { results: Record<string, NodeStatus> }, id: string, name: string): { from: string; node: string; source: string; value: string; fallback: boolean } | null {
  const e = s.edges.find((x) => x.target === id && x.targetHandle === PARAM + name);
  if (!e) return null;
  const src = s.nodes.find((n) => n.id === e.source);
  const def = s.nodeDefs[src?.data.typeId ?? ""];
  const out = outputPort(s, e.source, e.sourceHandle);
  const label = src?.data.label ?? e.source;
  return {
    from: `${label} · ${out?.label ?? e.sourceHandle}`,
    node: label,
    source: def && def.runtime !== "core" && label === def.label ? def.project : label,
    value: s.results[e.source]?.values?.[e.sourceHandle ?? ""] ?? "",
    // the output may give nothing (Port.may_be_empty, as the server says with the port): the parameter keeps its own
    // value for that case and stays editable — the same rule apply_values keeps (engine/templates.py)
    fallback: !!out?.may_be_empty,
  };
}

// ------------------------------------------------------------------ a wire let go on empty canvas (the node menu)

/** The type a loose wire stands for: its output's, or its input's. */
export function looseType(s: GraphView, w: LooseWire): string | undefined {
  return w.side === "source" ? outputType(s, w.node, w.port) : inputPort(s, w.node, w.port)?.type;
}

/** The same type in the words the artist knows, as the port itself carries them (the page spells no type name). */
export function looseTypeLabel(s: GraphView, w: LooseWire): string {
  const port = w.side === "source" ? outputPort(s, w.node, w.port) : inputPort(s, w.node, w.port);
  return port?.type_label ?? "";
}

/** Does a node type make data of `dataType` itself? An output that carries the type of whatever is wired into the node
 * (`type_from`: 「逐项开始」, 「切换」, 「命名」 and the other nodes that work on any data at all) passes data through
 * instead of making it, so it is never what the menu offers as the node that BRINGS this data. */
export function makesType(catalog: Catalog | null | undefined, def: NodeTypeDef, dataType: string): boolean {
  return def.at_defaults.ports.outputs.some((p) => !p.type_from && portAccepts(catalog, p.type, dataType));
}

/** The port of a new node of `typeId` the loose wire would go to (undefined: none), from its ports at its defaults. */
export function loosePort(s: GraphView, w: LooseWire, typeId: string): string | undefined {
  const t = looseType(s, w);
  const def = s.nodeDefs[typeId];
  if (!t || !def) return undefined;
  if (w.side === "source") {
    const unit = outputPort(s, w.node, w.port)?.unit ?? "";
    return (refusedBy(def, t) ? undefined : def.at_defaults.ports.inputs.find((p) => !p.name.startsWith(PARAM) && portAccepts(s.catalog, p.type, t))?.name)
      ?? Object.values(def.param_ports).find((p) => portAccepts(s.catalog, p.type, t) && unitFits(s.catalog, unit, p.unit))?.name;
  }
  const into = s.nodeDefs[typeOf(s, w.node)];
  return def.at_defaults.ports.outputs.find((p) => portAccepts(s.catalog, t, p.type) && !refusedBy(into, p.type))?.name;
}

/** The node type an input offers for a wire drawn out of it: the one it recommends, or the one a per-wire usage
 * check puts in front of it ("" none). */
export function looseFix(s: GraphView, w: LooseWire): string {
  return w.side === "target" ? inputPort(s, w.node, w.port)?.inserts ?? "" : "";
}

// ------------------------------------------------------------------ cooking

/** What a click on 计算 for the node is (the last reply's policy): whether it delivers, and whether it computes
 * nothing (everything is cached). */
export function clickKind(s: GraphView, id: string): { delivers: boolean; nothing: boolean } {
  return caseWords(s.reply?.nodes[id]?.policy);
}

const caseWords = (c: CookCase | null | undefined): { delivers: boolean; nothing: boolean } =>
  ({ delivers: !!c?.delivers, nothing: !!c && !c.computes.length });

// how every cook runs, said once (lab2shot/farm/queue.py, farm/scheduler/pools.py)
const QUEUED = "进队列排队，每个节点轮到了就在空着的显卡或 CPU 名额上算，互不依赖的节点同时算";

/** How the page says a cook: the 计算 button's tooltip and its short word. Only words: what the cook is comes
 * from the server. */
export function cookWords(delivers: boolean, nothing: boolean): { short: string; tip: string } {
  if (delivers)
    return { short: "打包", tip: `整理打包：${nothing ? "上游都已缓存，" : ""}${QUEUED}，算完把接进「输出」的结果整理成一个文件夹、打包成 zip，好了在节点上「下载」` };
  return nothing
    ? { short: "已缓存", tip: "要的结果都已缓存，不用算" }
    : { short: "计算", tip: `计算：${QUEUED}` };
}
