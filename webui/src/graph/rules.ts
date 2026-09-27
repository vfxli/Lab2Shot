import { kindOf } from "../ops/run";
import type { Catalog, CookCase, CookKind, HandleDef, NodePorts, NodeStatus, NodeTypeDef, PortDef, ResolvedCost, StatusReply, WireStatus } from "../api";
import type { LooseWire } from "../state/viewer";

/** What the editor knows of the graph's rules: nothing here works a rule out.
 * Every answer is read from the server: the status reply (a node's ports, handles, cost, licence, what a click and
 * showing it cook; every wire's state) or the catalogue's lookup tables (which types a port takes, which node converts
 * one type into another, what a node type refuses, its ports and handles at its defaults). The tables exist only for
 * what has to be answered before a status reply can: a wire still being drawn, a node just added, the drawing between
 * an edit and its reply (the last reply is kept meanwhile, so nothing flashes: pendingPorts).
 *
 * webui/tests/graphRules.test.ts checks these read what the server resolves for every template
 * (tests/test_applies.py test_the_editor_reads_what_the_server_resolves), and that no copy of a server rule comes back.
 * Pure: types only, so node's test runner loads it as it is. */

/** The input a promoted parameter gets: "param:<name>" (nodes/base.py PARAM). */
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
export const refusedBy = (def: NodeTypeDef | undefined, dataType: string): string => def?.refuses[dataType] ?? "";

/** Can a value in unit `have` drive a parameter in unit `want`? The catalogue's units: those of one kind convert. */
export function unitFits(catalog: Catalog | null | undefined, have: string, want: string): boolean {
  if (!have || !want || have === want) return true;
  const a = catalog?.units[have];
  return !!a && a.kind === catalog?.units[want]?.kind;
}

/** Parameter -> the values (as String) that switch the node type to non-commercial parts (catalogue `option_traits`). */
export function noncommercialChoices(def: NodeTypeDef): Record<string, string[]> {
  const kept = NC_CACHE.get(def);
  if (kept) return kept;
  const made: Record<string, string[]> = {};
  for (const [name, rows] of Object.entries(def.option_traits)) {
    const values = Object.entries(rows).filter(([, r]) => r.noncommercial).map(([v]) => v);
    if (values.length) made[name] = values;
  }
  NC_CACHE.set(def, made);
  return made;
}
const NC_CACHE = new WeakMap<NodeTypeDef, Record<string, string[]>>();
const NO_VALUES: string[] = [];
export const noncommercialValues = (def: NodeTypeDef | undefined, name: string): string[] => (def ? noncommercialChoices(def)[name] ?? NO_VALUES : NO_VALUES);

// ------------------------------------------------------------------ a node's ports, cost, licence, handles

const EMPTY_PORTS: NodePorts = { inputs: [], outputs: [], waiting: [] };

const typeOf = (s: GraphView, id: string) => s.nodes.find((n) => n.id === id)?.data.typeId ?? "";

/** A node's ports: as the last status reply says them, else (a node added since) its type's at its defaults. */
export function pendingPorts(s: GraphView, id: string): NodePorts {
  return s.reply?.nodes[id]?.ports ?? s.nodeDefs[typeOf(s, id)]?.at_defaults.ports ?? EMPTY_PORTS;
}

export const outputsOf = (s: GraphView, id: string): PortDef[] => pendingPorts(s, id).outputs;

/** A node's main result: the output the node declares as it (NodeTypeDef.main), which the viewer shows and a label
 * names when nothing else was chosen. Never the first output by position: the ports are ordered by data type,
 * so what comes first is whatever type sorts first. */
export const mainOutput = (s: GraphView, id: string): PortDef | undefined => {
  const outputs = outputsOf(s, id);
  const main = s.nodeDefs[s.nodes.find((n) => n.id === id)?.data.typeId ?? ""]?.main;
  return outputs.find((p) => p.name === main) ?? outputs.at(0); // the fallback goes when every node declares `main`
};

/** The rows of a node's input table (a ports_from table landing on inputs: 「多层 EXR 输出设置」's 图层), each a port
 * named by its row (nodes/base.py made_ports); [] when the node has no such table. The rows are the document's
 * own (its parameters), so this is what the ports ARE, not a guess: reading a graph file and drawing the node's inputs
 * both ask here. */
export function tableRows(def: NodeTypeDef | undefined, params: Record<string, unknown> | undefined): { name: string; label: string }[] {
  if (!def || def.ports_from_side !== "inputs" || !def.ports_from) return [];
  return ((params?.[def.ports_from] as { name: string; label: string }[] | undefined) ?? []).filter((r) => r && typeof r.name === "string");
}

/** A node's inputs: its declared ones (as the last reply resolved them), one per row of its input table in the
 * document's row order, then the input of each parameter it has promoted now, in the order promoted (the catalogue's
 * `param_ports`). The rows and the promoted list are the document's own, so a row just added (a wire dropped on the
 * node's body: graph/actions.ts addPortRow) or a parameter just promoted takes a wire before the next reply comes; a
 * row the reply already has keeps the reply's port. With no parameters known (a bare view) the reply's rows stand. */
/** The parameters that have an input on the node, in the order they sit there: the type's standing ones
 * (NodeDef.wired_ports, e.g. AnyCalib's six numbers into 「LensDistortion」) and then the ones this node promoted.
 * The one place that list is built: the ports drawn, the wires a graph file may name, and where a parameter's value
 * comes from all read it here. */
export function paramPortNames(def: NodeTypeDef | undefined, promoted: string[] | undefined): string[] {
  const own = (promoted ?? []).filter((n) => !def?.wired_ports?.includes(n));
  return [...(def?.wired_ports ?? []), ...own].filter((n) => def?.param_ports[n]);
}

export function inputsOf(s: GraphView, id: string): PortDef[] {
  const node = s.nodes.find((n) => n.id === id);
  const def = s.nodeDefs[node?.data.typeId ?? ""];
  const own = pendingPorts(s, id).inputs.filter((p) => !p.name.startsWith(PARAM));
  // 参数的输入口：节点类型声明的常驻口（NodeDef.wired_ports，即 AnyCalib 的六个数值接入「LensDistortion」处），
  // 以及该节点自身「提升到节点」的参数。服务器回复中二者都包含，但上一行已过滤掉所有 param: 口，因此在此按名称补回；
  // 若只取 promoted，常驻口将被遗漏
  const promoted = paramPortNames(def, node?.data.promoted).map((name) => def!.param_ports[name]);
  if (!def || def.ports_from_side !== "inputs" || !node?.data.params) return [...own, ...promoted];
  const declared = new Set(def.inputs.map((p) => p.name));
  const replied = new Map(own.map((p) => [p.name, p]));
  const rows = tableRows(def, node.data.params).map(
    (r): PortDef => replied.get(r.name) ?? { ...ROW_PORT, name: r.name, label: r.label, type: def.ports_from_type, type_label: def.ports_from_type_label },
  );
  return [...own.filter((p) => declared.has(p.name)), ...rows, ...promoted];
}

const ROW_PORT: PortDef = { name: "", type: "", type_label: "", label: "", optional: false, multi: false, list: false, type_from: "", inserts: "", unit: "", help: "" };

export const outputPort = (s: GraphView, id: string, port: string | null | undefined): PortDef | undefined =>
  port == null ? undefined : outputsOf(s, id).find((p) => p.name === port);

export const inputPort = (s: GraphView, id: string, port: string | null | undefined): PortDef | undefined =>
  port == null ? undefined : inputsOf(s, id).find((p) => p.name === port);

export const outputType = (s: GraphView, id: string, port: string | null | undefined): string | undefined => outputPort(s, id, port)?.type;

/** The declared output `port` absent until a parameter has a value (what brings it: `waits`). */
export const waitingPort = (s: GraphView, id: string, port: string | null | undefined): PortDef | undefined =>
  pendingPorts(s, id).waiting.find((p) => p.name === port);

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

/** A wire's id in the editor (graph/actions.ts connect): its two ends. */
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
export function wiredFrom(s: GraphView & { results: Record<string, NodeStatus> }, id: string, name: string): { from: string; node: string; source: string; value: string } | null {
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

export const INTERACTIVE: CookKind = { lane: "light", delivers: false, queues: false, by_itself: true, case: "interactive" };

/** What a click on 计算 for the node is (the last reply's policy; before any, at once), and whether it computes nothing. */
export function clickKind(s: GraphView, id: string): { kind: CookKind; nothing: boolean } {
  return caseWords(s.reply?.nodes[id]?.policy.click);
}

export const caseWords = (c: CookCase | null | undefined): { kind: CookKind; nothing: boolean } => ({ kind: c?.kind ?? INTERACTIVE, nothing: !!c && !c.computes.length });

/** How the page says a cook: the 计算 button's tooltip, the estimate's first word. `cpuJobs`: how many heavy jobs the
 * server runs at once (the queue says; 0 not known). Only words: what the cook is comes from the server. */
export function cookWords(kind: CookKind, nothing: boolean, cpuJobs = 0): { short: string; tip: string } {
  const lane = kind.lane === "gpu" ? "要显卡" : "CPU";
  const queued =
    kind.lane === "gpu"
      ? "进队列排队，轮到了在接任务的显卡上算"
      : `进 CPU 队列排队${cpuJobs ? `，同时最多算 ${cpuJobs} 个` : ""}`;
  if (kind.delivers)
    return kind.queues
      ? { short: `提交 · 排队（${lane}）`, tip: `提交：上游还有要${kind.lane === "gpu" ? "显卡" : "算很久"}的节点没算过，整个任务${queued}，算完把文件交出来（存到「保存到」选的位置）` }
      : { short: "提交", tip: `提交：${nothing ? "上游都已缓存" : "上游要算的都是轻量节点"}，马上把文件交出来（存到「保存到」选的位置），不用排队` };
  if (kind.queues)
    return { short: `排队（${lane}）`, tip: `排队（${lane}）：要算的节点里有${kind.lane === "gpu" ? "要显卡" : "要算很久"}的，任务${queued}` };
  return nothing
    ? { short: "已缓存", tip: "立即计算：要的结果都已缓存，不用算" }
    : { short: "立即计算", tip: "立即计算：要算的都是轻量节点（读文件、数值、遮罩调整……），马上在服务器上算，不用排队；显示它时也会自己算" };
}


/** 判断浏览器能否自行计算该节点：只依据声明，不依据节点类型名。需同时满足两点：
 * 1. 服务器确认可以计算：状态回复中带有 `ops`（`NodeDef.browser_ops` 的结果：所用算法及由参数确定的参数）；
 * 2. 所用算法均属于浏览器可执行的运算（词汇表 `lab2shot/ops/vocab.py` 的 `KINDS`，
 *    实现位于 `ops/recipe.ts runStep`，每种一段）。空序列同样视为可计算（「拆成列表」只拆分条目，不做运算）。
 * 不按节点名维护名单：否则每新增一个同类节点都需修改网页。
 *
 * 使用方有两处，判据仅此一份：向上游求值的链（`view/evaluate.ts`），以及是否交由服务器计算
 * （`graph/actions.ts`：显示节点时会顺带计算该节点，但这些步骤浏览器可当场完成，交由服务器重算
 * 只会多等一次队列；修改选人不应提交任何计算）。 */
export const browserCanCompute = (status: { ops?: { op: string }[] } | undefined): boolean =>
  !!status?.ops && status.ops.every((o) => BROWSER_KINDS.includes(kindOf(o.op)));

/** 浏览器可执行的运算种类（`ops/recipe.ts runStep`，每种一段）。
 * 词汇表是封闭的：新增运算时两个执行器都须修改（`lab2shot/ops/vocab.py`），此处同步增加一行。 */
const BROWSER_KINDS = ["pixel", "boxes.paint", "items.pick"];
