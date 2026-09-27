// The node catalogue and the graph file as the server describes them (lab2shot/nodes/base.py describe(), catalog.py):
// data types, 3D kinds, ports, parameters, node types, tags, templates, OCIO. Re-exported by api/index.ts.

import type { Upload } from "./deliveries";
import type { Places } from "../model/places";
import { NEUTRAL_COLOR } from "../platform/palette";
import type { StandingMark } from "./status";
import type { MessageJson } from "./applies";

export interface DataType {
  id: string;
  label: string;
  color: string;
  description: string;
  in_2d: "picture" | "overlay" | "strip" | "inputs" | "value" | null; // its role in the viewer's 2D stage (nodes/types.py; inputs: shown as what it was made from; value: a basic value, shown as itself)
  in_3d: "element" | "backplate" | "points" | "strip" | "inputs" | "value" | null; // and in the 3D stage
  end: string;
  // data that holds several things (人物框 several 人物, 场景 several 组): what one of them is called (data/types.py
  // DataType.items; "" the type cannot be worked on item by item). What 「逐个：人物」 reads.
  items: string;
  // the layer name a multi-layer EXR gives this kind by default ("" never one), suggested for a new row of 「多层 EXR 输出设置」's 图层 table (data/layers.py DEFAULT_NAMES)
  layer_default: string;
}

/** A kind of 3D data as DCCs tell them apart (nodes/types.py SCENE_KINDS: 模型, 相机, 点云, 骨架动画, 蒙皮角色, 灯光),
 * in the order the editor lists them. */
export interface SceneKind {
  id: string;
  label: string;
  type: string; // the data type it travels as on its own: its color
  description: string;
}

/** A 模型 whose points change every frame (a point cache): carried beside "model" (nodes/types.py DEFORMING). */
export const DEFORMING = "model.deforming";

/** What a 3D output-settings node's format holds of one kind (OutputSettings.writes): all of it, only still ones (模型),
 * or none; `reason` says how or why not. */
export interface WritesPart {
  how: "full" | "static" | "no";
  reason?: string; // 仅在有说明时下发（控制首屏数据量）
  // 该数据可以写出，但格式无法保留其中一部分（FBX 可写出模型，但无法保留网格分区）：
  // 显示在「支持的数据」中，不构成拒绝的理由（nodes/output.py Writes.lost）
  lost?: string;
}

/** Why a writer refuses a kind: on a wire of the kind's own type, on a packed 场景 holding it; the node that converts it
 * into what the writer takes ("" none), offered as one click. */
export interface Refusal {
  wire: string;
  packed: string;
  via: string;
}

/** One subcategory of a tree the administrator keeps as a data file (lab2shot/categories.py). */
export interface TreeSub {
  id: string;
  label: string;
  tip: string;
  rank: number;
}

/** One first-level category of such a tree: the templates panel's (templates/_categories.json) or the node menu's
 * (menu/categories.json, where `section` says which band). Nothing about a category is written in code: the
 * administrator makes, renames, orders and removes them in the panel itself. */
export interface TreeCategory {
  id: string;
  label: string;
  color: string;
  tip: string;
  rank: number;
  section: string; // the node menu's band ("tools" | "deliver"); "" in the templates tree
  subs: TreeSub[];
}

/** The node menu as the catalogue carries it: its bands, its tree, where every node type sits (type id -> a category
 * or subcategory id; one missing is 未分类) and, when a data file cannot be read, the sentence to show. */
export interface MenuInfo {
  sections: { id: string; label: string; tip: string }[];
  categories: TreeCategory[];
  placed: Record<string, string>;
  problem: string;
}

/** Where `where` (a category or subcategory id) is in `tree`: the category and, when it names one, the subcategory. */
export function placeIn(tree: TreeCategory[], where: string): { id: string; label: string; color: string; sub: string; subLabel: string } {
  const nowhere = { id: "", label: "", color: NEUTRAL_COLOR, sub: "", subLabel: "" };
  if (!where) return nowhere;
  const own = tree.find((c) => c.id === where);
  if (own) return { id: own.id, label: own.label, color: own.color, sub: "", subLabel: "" };
  const c = tree.find((x) => x.subs.some((s) => s.id === where));
  const s = c?.subs.find((x) => x.id === where);
  return c && s ? { id: c.id, label: c.label, color: c.color, sub: s.id, subLabel: s.label } : nowhere;
}


/** Where a node type sits in the node menu: the administrator's placing (Catalog.menu.placed), read against the
 * menu's tree. A node placed nowhere (a new one) is 未分类: an empty id and the neutral colour. */
export function nodeCategory(catalog: Catalog | null | undefined, def: { id: string } | null | undefined):
  { id: string; label: string; color: string; sub: string; subLabel: string } {
  if (!catalog || !def) return { id: "", label: "", color: NEUTRAL_COLOR, sub: "", subLabel: "" };
  return placeIn(catalog.menu.categories, catalog.menu.placed[def.id] ?? "");
}

/** A viewer handle bound to the node's parameters (nodes/handles.py). */
export interface HandleDef {
  kind: "points" | "box" | "corners" | "person" | "canvas" | "figure" | "transform";
  stage: "2d" | "3d";
  params: Record<string, string>; // role -> the parameter it edits
  source: string | null; // the input it works on
  labels: string[]; // points: what a click means (the first by default); figure: the joints, in order
}

export interface PortDef {
  name: string;
  type: string;
  // the type in the words the artist knows, said by the server (data/types.py type_label: 「场景或场景列表」): the page never spells a type's name itself, so a new type or a list form never shows up as a raw id
  type_label: string;
  label: string;
  optional: boolean;
  multi: boolean;
  list: boolean; // it takes or gives a list of its type (X[]): a square socket and a double wire
  type_from: string; // "input:<port>": the output carries the type of what is wired into that input
  inserts: string; // an input: the node type offered first for a wire drawn out of it (inserted in front of it, or a parameter's constant) ("" none); a check's own one click comes with the check, not from here
  unit: string; // a value's unit: what a value output gives ("param:<name>": its node's parameter says) or a parameter's input takes
  help: string; // an optional input: what connecting it does ("" none)
  // 该端口当前是否不可用及其原因（由服务器计算，nodes/base.py Port.applies）：节点上置灰、位置不变，
  // 悬停说明原因，且不允许连线（例如接入「图像」后，下方的 rgba 口不可用）。
  // 输入口与输出口都可能置灰：解算器接入相机后，其「相机」输出仅原样透传该相机，
  // 若从该处再连线，将无法区分交付中的相机来源，因此不允许从该端口连出。
  // 仅存在于图中的端口上（catalog 中没有：它描述的是当前这张图的状态）
  inactive?: MessageJson;
  // what a benchmark found about connecting it (nodes/base.py NodeDef.measured): sent only to an account that runs the benchmarks, so the page shows it when it is there
  finding?: string;
  // what the pointer says over the port, written by the server (nodes/base.py Port.tip): the type's name and unit,
  // what that type is, what this port means, and what it does with a picture's alpha. Only on the ports of a graph
  // (the status reply): the catalogue describes types, and would repeat every description on every port.
  tip?: string;
  when?: string; // an output there only once this parameter has a value: its name (a link for labels; the status reply says whether it is there)
  waits?: string; // what brings such an output ("选一台相机")
  kinds?: string[]; // declared (catalogue): an output's 3D data beyond its type's kind; resolved (status): all it carries here
}

/** A node's ports as they are in its graph (engine/graph.py Graph.ports, the status reply): outputs with the type, unit
 * and 3D kinds they carry there; `waiting`: declared outputs absent until a parameter has a value. */
export interface NodePorts {
  inputs: PortDef[];
  outputs: PortDef[];
  waiting: PortDef[];
}

export type SimpleKind = "" | "toggle" | "options" | "number" | "vector" | "text";

export interface ParamDef {
  name: string;
  label: string;
  description: string;
  finding?: string; // what a benchmark found about setting it (as PortDef.finding: there only for the accounts that run benchmarks)
  type: "string" | "number" | "integer" | "boolean" | "array";
  nullable: boolean;
  minimum: number | null;
  maximum: number | null;
  multiple_of: number | null; // a number must be a multiple of it
  options: string[] | null;
  option_labels: Record<string, string> | null;
  widget: string | null;
  parts: string[];  // widget "vec3" 三个输入格的名称（默认 X Y Z；颜色为 R G B）
  // 文字参数的行数：1 = 单行输入框，2 及以上 = 自动换行的多行文本框，高度为该行数（如提示词）。
  // 这不是另一种控件（nodes/base.py P(lines=...)）：`simple` 仍为 "text"，`wire` 仍为文字连线，
  // 只是 ui/controls.tsx 的 TextField 绘制为 <textarea>，参数面板与节点上由同一份声明绘制同一个控件
  lines: number;
  group: string;
  affects_result: boolean;
  placeholder: string;
  accept: string[]; // a file parameter: the suffixes it takes (.exr, .mov ...)
  // 文件参数：选定文件、服务器只取得文件头时，该节点能否给出结果
  // （服务器端声明 `nodes/base.py ReadsFile.head_is_enough`，该处说明了默认值为「不能」的原因）。
  // 真 → 选定后不传输任何字节（先申报，点击「计算」时才传输）；假 → 选定后立即传输（transfer/declare.ts）
  head_enough?: boolean;
  unit: string; // shown inside a number field (mm, °, px ...)
  derived_from: string[]; // parameters the node works this one out from (asked from the server when they change)
  unique: boolean; // a node added in the editor gets a value no other node has (its default, numbered on)
  option_needs: Record<string, string>; // choice -> what this browser must be able to do for it ("folders")
  choices_from: string[]; // parameters and input ports its options come from (widget "choice": asked from the server, NodeDef.choices)
  wire: string; // the value type a wire into it carries once it is 提升到节点 ("" it can't be driven by a wire)
  simple: SimpleKind; // how it can show on the node's body ("" panel only: files, tables, pickers ...), nodes/base.py simple_kind
  per_frame: boolean; // it takes one value per frame from a wire (a zoom's focal length)
  panel: boolean; // false: not in the parameter panel; only a wire drives it (its input port is there as always)
  overrides: string[]; // input ports whose own say it overrides when it is set (a focal length over the camera's)
  items: ParamDef[] | null; // a list of entries (a table): the fields of one entry
}

export interface NodeTypeDef {
  id: string;
  label: string;
  category: string; // the tool subcategory its author suggests (a word: where it sits is Catalog.menu.placed)
  description: string;
  inputs: PortDef[];
  outputs: PortDef[];
  param_ports: Record<string, PortDef>; // the input each parameter a wire can drive gets once 提升到节点 ("param:<name>")
  ports_from: string; // a parameter whose entries ({name, label, type, ...}) are more ports, one each, in row order
  ports_from_side: "outputs" | "inputs"; // which side ports_from's entries land on (读取序列: outputs; 多层 EXR 输出设置: inputs)
  ports_from_type: string; // the port type every row gets when ports_from_side is "inputs" (a row has no `type` of its own then)
  ports_from_type_label: string; // and how that type is said (the server's words)
  main: string; // its main result's port: what the viewer shows and a label names by default (ports go by data type)
  lens: "" | "pinhole" | "any" | "given"; // what it assumes of the plate's lens (nodes/applies.py LENSES)
  marks: StandingMark[]; // the standing marks its declaration gives it (a pinhole node: N-LENS-NEEDSUNDISTORTED)
  params: ParamDef[];
  defaults: Record<string, unknown>;
  runtime: string;
  project: string;
  handles: HandleDef[];
  places: Places | null; // where it places what it gives, when its effect on the viewer is a transform (G17)
  // 预览标签（由 nodes/base.py default_preview 集中计算，同一取值适用于 2D、3D 两个舞台）
  preview: "plate" | "compute" | "result" | "scene";
  // what it declares running it costs (nodes/applies.py Cost): the lane off a GPU, whether it always runs on one, the
  // rating its own measured numbers give (a choice may change them: option_traits)
  cost: { lane: "light" | "heavy"; gpu: boolean; rating: ComputeRating | null };
  // resolved at its default parameters (a node not in a graph yet); a node in a graph reads its status (NodeStatus)
  // and its ports and handles on its own (Graph.at_defaults): what a node just added shows until its status arrives
  at_defaults: { cost: ResolvedCost; licence: ResolvedLicence; ports: NodePorts; handles: number[] };
  // the choices that change something, as a lookup table: parameter -> String(value) -> what choosing it does
  option_traits: Record<string, Record<string, { gpu: boolean; noncommercial: boolean; rating: ComputeRating | null }>>;
  delivers: boolean; // it hands files to the user (「输出」): a click only, never by showing
  streams?: boolean; // its frames are final as they are written (nodes/families/base.py WorkerNode.streams): while it
  // is being cooked the viewer may show what is there already (边算边看, view/partial.ts). Only a worker node says it.
  tags: string[]; // lab2shot/nodes/tags.py: its licence class, 需注册 ...
  // 仅三方节点具有：项目的代码仓库与主页（标题行的 GitHub 标签）、许可证（面板底部，仅作信息展示）
  links?: { repo: string; homepage: string; licence: { name: string; url: string; summary: string } };
  word: string; // the one word to show for them (tags.strictest, on the server): 仅限研究 / 非商用 / 需注册 / …
  on_node: string[]; // its key parameters, shown on its body (the node author's default; a node instance may choose others: its ui.on_node)
  strip?: string[]; // 视图值条上显示的参数（NodeDef.strip；值由状态回复的 `strip` 提供，不是输出口）
  writes?: Record<string, WritesPart>; // a 3D output-settings node: what its format holds of each kind (empty: 2D data)
  // nodes/clipboard.py Pasteable: a node whose result carries a snippet another application reads from
  // its clipboard, such as 「2D 跟踪点输出设置」 at Tracker / CornerPin, 「LensDistortion」 and 「AnyCalib 镜头标定」 (the lens
  // as Nuke's own nodes). Its body offers 复制到 Nuke once it has cooked
  clipboard?: string;
  // which output port's result holds that snippet; missing: the node's `main` result
  clipboard_port?: string;
  // 反向路径（Pasteable.paste）：该节点可将哪个软件的一段文本解析为自身的一组参数（"nuke"）。
  // 声明后才绘制「从 Nuke 粘贴」按钮（editor/PasteFrom.tsx）
  paste?: string;
  // 节点类型声明的常驻参数接线口（NodeDef.wired_ports）：端口始终存在于节点上，面板中的「提升到节点」针脚固定开启且不可取消
  wired_ports?: string[];
  // 这些端口共同构成一项要求（nodes/base.py NodeDef.needs_any）：它们不标注「可选」，
  // 因为标注不正确（全部不接不可行），而标注「至少一个」又是冗余信息。一个都未接入时在提交前拦截
  needs_any?: string[];
  refuses: Record<string, string>; // data types it can't take, and why: a wire of one is wrong
  converts: string[]; // [from type, to type]: a generic conversion, the one click on a wire of the one into an input of the other
  refuses_kinds: Record<string, Refusal>; // kinds of 3D data it can't take whatever the wire's type (a 场景 holding one)
}

/** What a node costs with its parameters, as the server resolved it (nodes/applies.py ResolvedCost). */
export interface ResolvedCost {
  lane: "light" | "heavy" | "gpu";
  gpu: boolean;
  vram_gb: number; // the measured peak (RTX 4090) with these parameters; 0 off a GPU
  seconds_per_frame: number | null;
  ram_gb: number;
  rating: ComputeRating | null;
}

/** Whose licence a node's result is under with its parameters (nodes/applies.py ResolvedLicence). */
export interface ResolvedLicence {
  tags: string[];
  commercial: boolean;
  note: string; // why it is what it is
  word: string; // 服务器计算出的许可词（nodes/tags.py strictest）：仅限研究 / 非商用 / 需注册 / 可商用
}

/** 低/中/高/超高：GPU 节点的计算量档位（lab2shot/nodes/compute.py），附带包含实测显存与秒/帧数据的提示语。 */
export interface ComputeRating {
  tier: "低" | "中" | "高" | "超高";
  tip: string;
}

/** A tag (lab2shot/nodes/tags.py): what a node, a choice or a template is (基础, 可商用, 非商用, 仅限研究, 需注册). */
export interface TagInfo {
  label: string;
  tip: string;
  implied: boolean; // every user has it
}

export interface Catalog {
  version: string;
  types: DataType[];
  // 输出口名称 → 交付为 EXR 时的图层（data/layers.py LAYER_FOR_PORT）：向「多层 EXR 输出设置」拖入连线时据此填写图层名
  layer_ports: Record<string, string>;
  scene_kinds: SceneKind[];
  menu: MenuInfo; // the node menu: its bands, its tree and where every node sits (lab2shot/categories.py: data files)
  units: Record<string, { kind: string; kind_label: string; factor: number }>; // a value's units: those of one kind convert
  tags: Record<string, TagInfo>;
  templates: number; // how many built-in templates this account may use (the top bar's count, without the list)
  nodes: NodeTypeDef[]; // only those this account may use (server/access.py)
  // looked up while a wire is still drawn (nodes/registry.py type_tables): port type -> the types it takes; data type ->
  // port type -> the node type that converts one into the other
  accepts: Record<string, string[]>;
  converters: Record<string, Record<string, string>>;
}

/** What was picked for an input file parameter, the way the parameter shows it: the upload (its reference, how many
 * files, their size, a sequence's frames) and the folder of the user's machine it came from ("" unless a folder was
 * picked or dropped: a browser never tells a file's full path). Kept with the graph, so the parameter still says what
 * it was when the server no longer has the upload. */
export type PickedFrom = Omit<Upload, "name"> & { folder: string };

/** An 「输出」 node's folder on the user's machine: its key in this browser (files/handles.ts), and its name. */
export interface SaveTo {
  handle: string;
  name: string;
}

interface GraphNodeJSON {
  id: string;
  type: string;
  label: string;
  params: Record<string, unknown>;
  promoted?: string[]; // 提升到节点: each gets an input "param:<name>" and a row on the node's body
  // editor only: where the node's files came from (by parameter) and go to; the parameters its body shows when the
  // user chose others than its type's (NodeTypeDef.on_node, 「在节点上显示」, ParamPanel.tsx OnNodePin)
  ui: { x: number; y: number; picked?: Record<string, PickedFrom>; save_to?: SaveTo; on_node?: string[] };
}

/** A group box around nodes (Houdini network box). Members are listed only while it is collapsed. */
export interface BoxJSON {
  id: string;
  label: string;
  color: string;
  x: number;
  y: number;
  w: number;
  h: number;
  collapsed: boolean;
  members: string[];
}

export interface GraphJSON {
  schema: "lab2shot.graph/1";
  // id: the graph's own identity, 128 random bits as 32 hex
  // characters (graph/actions.ts's newGraphId), never an integer or anything else that could collide. Generated the
  // moment a document is created (a template opened, a 另存为 copy) or, for a file saved before this existed, the
  // moment it is opened (and then marked changed, so saving writes it in); a plain open, save or reopen keeps it.
  // Jobs and deliveries carry it, so the page only ever draws a delivery on the graph it belongs to. Two graphs from
  // the same template never share one; the id inside a template file (templates/*.json) exists only so the file
  // itself always has one, and is never inherited by a graph created from it.
  // `template`: the file stem of the built-in template this graph was made from (the templates panel writes it when
  // a card is opened). It rides with the graph a job is submitted with, which is where the administrator's 模板 page
  // counts 本月使用次数 from (lab2shot/server/templates.py _uses_this_month). A graph of one's own has none.
  meta: { id?: string; name: string; description?: string; author?: string; created?: string; category?: string; tags?: string[]; template?: string };
  exposed: { name: string; label: string; target: string }[];
  frames?: [number, number] | null; // the frames to cook, first and last (null: every frame of the inputs)
  nodes: GraphNodeJSON[];
  edges: { from: [string, string]; to: [string, string] }[];
  boxes?: BoxJSON[];
  view?: { display: string | null; frame: number | null; port?: string | null; playback?: [number, number] | null }; // playback: in / out; absent in a graph written by a script
}

export interface TemplateInfo {
  id: string;
  name: string;
  intro: string; // 简介：所结合的技术及其用途
  deliverable: string; // where its card sits (its file's meta.deliverable): a subcategory or first-level category id, "" 未分类
  category: string; // the first-level category it sits in, "" 未分类 (also when its place is no longer in the tree)
  graph: GraphJSON;
  projects: { name: string; title: string }[]; // third-party projects the template uses
  // 卡片右下角的小标签：核心项目的论文或发布年份（由服务器按声明选出核心项目，页面不自行选择）
  year: number | null;
  licence: string[]; // its tags (lab2shot/nodes/tags.py): every node's and choice's in it
  // the one word the card shows for all of them: the strictest (nodes/tags.py strictest). The page prints it and
  // never picks among the tags itself, nor reads it: whether that word means 可商用 is the boolean beside it.
  licence_word: string;
  commercial: boolean;
  owner: string; // where its file lives (lab2shot/library.py): "admin" (templates/, the project's presets; an administrator may delete it) or "adapter" (adapters/<name>/templates/, read-only: turn it off or drag it elsewhere)
  adapter: string; // the adapter it came with ("" for a project preset)
  // An administrator may take a template out of use. It is then absent from an ordinary account's list; the
  // answer of a login that manages templates carries `enabled: false`, and the card says so. The page never checks who is
  // looking: the route decides what is in the answer.
  enabled?: boolean;
  path: string; // its file on the server
  author: string; // who saved it ("" when the file does not say: an older file, or an adapter's)
  created: string; // when it was saved ("" when the file does not say)
  updated: number; // the file's last change (seconds)
  bytes: number;
}

/** GET /api/templates: the cards this account may use, the templates panel's tree they sit in, and the sentence to
 * show when the tree's file cannot be read (every card is 未分类 then). */
export interface TemplatesPage {
  templates: TemplateInfo[];
  categories: TreeCategory[];
  problem: string;
}

/** GET /api/ocio：「色彩空间」下拉的选项及工作空间名称（lab2shot/io/color.py：读入时一律转换到工作空间，输出时再转换出去；
 * 工作空间即屏幕显示所用的 sRGB，因此不设显示设备与视图层级）。 */
export interface OcioInfo {
  name: string;
  colorspaces: string[];
  working: string;
}
