// The node catalogue and the graph file as the server describes them (lab2shot/nodes/base.py describe(), catalog.py):
// data types, 3D kinds, ports, parameters, node types, tags, templates, OCIO. Re-exported by api/index.ts.

import type { Upload } from "./files";
import type { Places } from "../model/places";
import { NEUTRAL_COLOR } from "../platform/palette";
import type { StandingMark } from "./status";
import type { MessageJson } from "./applies";

export interface DataType {
  id: string;
  label: string;
  color: string;
  description: string;
  in_2d: "picture" | "overlay" | "strip" | "inputs" | "value" | null; // its role in the viewer's 2D stage (lab2shot/data/types.py; inputs: shown as what it was made from; value: a basic value, shown as itself)
  in_3d: "element" | "backplate" | "points" | "strip" | "inputs" | "value" | null; // and in the 3D stage
  end: string;
  // data that holds several things (人物框 several 人物, 场景 several 组): what one of them is called (data/types.py
  // DataType.items; "" the type cannot be worked on item by item). What 「逐个：人物」 reads.
  items: string;
  // the layer name a multi-layer EXR gives this kind by default ("" never one), suggested for a new row of 「多层 EXR 输出设置」's 图层 table (lab2shot/data/types.py layer_default)
  layer_default: string;
}

/** A kind of 3D data as DCCs tell them apart (lab2shot/data/types.py SCENE_KINDS: 模型, 相机, 点云, 骨架动画, 蒙皮角色, 灯光),
 * in the order the editor lists them. */
export interface SceneKind {
  id: string;
  label: string;
  type: string; // the data type it travels as on its own: its color
}

/** What a 3D output-settings node's format holds of one kind (OutputSettings.writes): all of it, only still ones (模型),
 * or none; `reason` says how or why not. */
export interface WritesPart {
  how: "full" | "static" | "no";
  // the data can be written but the format cannot keep part of it (FBX writes a model but not its mesh partitions):
  // marked 「部分」 in 「支持的数据」, never a reason to refuse (nodes/output.py Writes.lost); sent only when true
  lost?: boolean;
}

/** Why a writer refuses a kind: on a wire of the kind's own type, on a packed 场景 holding it; the node that converts it
 * into what the writer takes ("" none), offered as one click. */
interface Refusal {
  wire: string;
  packed: string;
  via: string;
}

/** One subcategory of a tree the administrator keeps as a data file (lab2shot/categories.py). */
interface TreeSub {
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
  kind: "points" | "box" | "corners" | "person" | "canvas" | "figure" | "transform" | "skeleton_pose";
  stage: "2d" | "3d";
  params: Record<string, string>; // role -> the parameter it edits (skeleton_pose: "pose" -> the list of joint corrections)
  source: string | null; // the input it works on
  labels: string[]; // points: what a click means (the first by default); figure: the joints, in order
  skeleton?: string; // skeleton_pose: the parameter naming the skeleton path on `source`
  readonly?: boolean; // skeleton_pose: drawn only (no parameter to edit: no gizmo, mirror or reset)
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
  // an output that may give nothing (nodes/port.py Port.may_be_empty: 「读取序列」's 帧率 on a PNG sequence): a parameter it
  // drives stays editable, its value used when the wire gives none (graph/rules.ts wiredFrom `fallback`)
  may_be_empty?: boolean;
  // an output that is one of several ways (「切换」, "#common"), in a status reply: the type each way carries (engine/graph.py
  // ways; only when there are two or more). `type` is then their common kind (场景 over 骨架动画 and 蒙皮角色), and an input
  // that takes every one of the ways takes it too (graph/rules.ts takes, the same rule as Graph.takes)
  ways?: string[];
  inserts: string; // an input: the node type offered first for a wire drawn out of it (inserted in front of it, or a parameter's constant) ("" none); a check's own one click comes with the check, not from here
  unit: string; // a value's unit: what a value output gives ("param:<name>": its node's parameter says) or a parameter's input takes
  // whether the port is unusable now, and why (computed by the server, nodes/base.py Port.applies): greyed on the node
  // in its place, the reason on hover, and no wire may be drawn to it (e.g. with 「图像」 wired in, the rgba port below
  // is unusable). Inputs and outputs alike may be greyed: with a camera wired into a solver, its 「相机」 output only
  // passes that camera through, and a wire from there would leave the delivered camera's source ambiguous, so none may
  // be drawn out of it. Only on the ports of a graph (not in the catalogue: it describes the state of this graph)
  inactive?: MessageJson;
  // what the pointer says over the port, written by the server (nodes/port.py Port.tip): the type's name and unit,
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

type SimpleKind = "" | "toggle" | "options" | "number" | "vector" | "text" | "button";

export interface ParamDef {
  name: string;
  label: string;
  type: "string" | "number" | "integer" | "boolean" | "array" | "button"; // "button": a button parameter (widget "button"), no value
  nullable: boolean;
  minimum: number | null;
  maximum: number | null;
  open_minimum: boolean; // the minimum itself is not allowed (gt): a value must be greater
  open_maximum: boolean; // the maximum itself is not allowed (lt)
  multiple_of: number | null; // a number must be a multiple of it
  options: string[] | null;
  option_labels: Record<string, string> | null;
  widget: string | null;
  parts: string[];  // widget "vec3": the names of its three fields (X Y Z by default; R G B for a colour)
  // a text parameter's line count: 1 = a one-line field, 2 or more = a wrapping multi-line box that many lines high (a
  // prompt, say). It is not another control (nodes/base.py P(lines=...)): `simple` stays "text" and `wire` a text wire;
  // only ui/controls.tsx's TextField draws a <textarea>, the same control from the same declaration in the parameter
  // panel and on the node
  lines: number;
  group: string;
  affects_result: boolean;
  placeholder: string;
  accept: string[]; // a file parameter: the suffixes it takes (.exr, .mov ...)
  // a file parameter: whether the node can give its result once a file is picked and the server has only its head
  // (declared on the server, `nodes/base.py ReadsFile.head_is_enough`, which says why the default is "no").
  // true -> picking transfers no bytes (declared first, sent when 「计算」 is clicked); false -> picking transfers at
  // once (transfer/declare.ts)
  head_enough?: boolean;
  unit: string; // shown inside a number field (mm, °, px ...)
  derived_from: string[]; // parameters the node works this one out from (asked from the server when they change)
  unique: boolean; // a node added in the editor gets a value no other node has (its default, numbered on)
  choices_from: string[]; // parameters and input ports its options come from (widget "choice": asked from the server, NodeDef.choices)
  wire: string; // the value type a wire into it carries once it is 提升到节点 ("" it can't be driven by a wire)
  simple: SimpleKind; // how it can show on the node's body ("" panel only: files, tables, pickers ...), nodes/base.py simple_kind
  per_frame: boolean; // it takes one value per frame from a wire (a zoom's focal length)
  panel: boolean; // false: not in the parameter panel; only a wire drives it (its input port is there as always)
  overrides: string[]; // input ports whose own say it overrides when it is set (a focal length over the camera's)
  items: ParamDef[] | null; // a list of entries (a table): the fields of one entry
  // a button parameter (widget "button", lab2shot/nodes/params.py Button): the page's action it runs (editor/buttonActions.tsx)
  action?: string;
  // a button that works on another parameter (「在视图里点选」, action pick_in_view: the picks / canvas parameter it picks for)
  target?: string;
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
  // an input table whose rows the node names itself (「切换」's ways): the port names its rows take, a new row the first
  // one not yet used, and the n-th row's default label; as many rows as names, no more. Absent: rows named here
  // (row1, row2…) and labelled after what is wired in (「多层 EXR 输出设置」's 图层)
  ports_from_names?: string[];
  ports_from_labels?: string[];
  ports_from_word: string; // what one row of an input table is called (「加一层」, 「加一路」); "" for any other node
  main: string; // its main result's port: what the viewer shows and a label names by default (ports go by data type)
  marks: StandingMark[]; // the standing marks its declaration gives it (a crop that decides its own size: I-SHAPE-CROP)
  params: ParamDef[];
  defaults: Record<string, unknown>;
  runtime: string;
  project: string;
  handles: HandleDef[];
  places: Places | null; // where it places what it gives, when its effect on the viewer is a transform
  // the preview tag (worked out in one place, nodes/base.py default_preview; one value for both the 2D and 3D stages)
  preview: "plate" | "compute" | "result" | "scene";
  // what it declares running it costs (nodes/applies.py Cost): whether it always runs on a GPU, the rating its own
  // measured numbers give (a choice may change them: option_traits)
  cost: { gpu: boolean; rating: ComputeRating | null };
  // resolved at its default parameters (a node not in a graph yet); a node in a graph reads its status (NodeStatus)
  // and its ports and handles on its own (Graph.at_defaults): what a node just added shows until its status arrives
  at_defaults: { cost: ResolvedCost; licence: ResolvedLicence; ports: NodePorts; handles: number[] };
  // the choices that change something, as a lookup table: parameter -> String(value) -> what choosing it does
  // licence: the class the choice switches the node to (lab2shot/nodes/tags.py: noncommercial / research), "" none
  option_traits: Record<string, Record<string, { gpu: boolean; licence: string; rating: ComputeRating | null }>>;
  delivers: boolean; // it hands files to the user (「输出」): a click only, never by showing
  streams?: boolean; // its frames are final as they are written (nodes/families/base.py WorkerNode.streams): while it
  // is being cooked the viewer may show what is there already (边算边看, view/partial.ts). Only a worker node says it.
  tags: string[]; // lab2shot/nodes/tags.py: its licence class, 需注册 ...
  // third-party nodes only: the project's repository and homepage (the GitHub chip in the title row) and its licence (at the panel's foot, information only)
  links?: { repo: string; homepage: string; licence: { name: string; url: string } };
  word: string; // the one word to show for them (tags.strictest, on the server): 仅限研究 / 非商用 / 需注册 / …
  on_node: string[]; // its key parameters, shown on its body (the node author's default; a node instance may choose others: its ui.on_node)
  strip?: string[]; // the parameters shown on the viewer's value strip (NodeDef.strip; their values come in the status reply's `strip`, not from output ports)
  writes?: Record<string, WritesPart>; // a 3D output-settings node: what its format holds of each kind (empty: 2D data)
  // nodes/clipboard.py Pasteable: a node whose result carries a snippet another application reads from
  // its clipboard, such as 「2D 跟踪点输出设置」 at Tracker / CornerPin, 「LensDistortion」 and 「AnyCalib 镜头标定」 (the lens
  // as Nuke's own nodes). Its body offers 复制到 Nuke once it has cooked
  clipboard?: string;
  // which output port's result holds that snippet; missing: the node's `main` result
  clipboard_port?: string;
  // the reverse path (Pasteable.paste): which application's text this node can parse into a set of its own parameters
  // ("nuke"). The 「从 Nuke 粘贴」 button is drawn only when it is declared (editor/PasteFrom.tsx)
  paste?: string;
  // parameter inputs the node type declares as permanent (NodeDef.wired_ports): the port is always on the node, and the panel's 「提升到节点」 pin is fixed on and cannot be unset
  wired_ports?: string[];
  // these ports together make one requirement (nodes/base.py NodeDef.needs_any): none is marked 「可选」, which would be
  // wrong (leaving all unwired is not possible), and 「至少一个」 would be redundant. With none wired, submission is
  // stopped before it is sent
  needs_any?: string[];
  refuses: Record<string, string>; // data types it can't take, and why: a wire of one is wrong
  converts: string[]; // [from type, to type]: a generic conversion, the one click on a wire of the one into an input of the other
  refuses_kinds: Record<string, Refusal>; // kinds of 3D data it can't take whatever the wire's type (a 场景 holding one)
}

/** What a node costs with its parameters, as the server resolved it (nodes/applies.py ResolvedCost). */
export interface ResolvedCost {
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
  word: string; // the licence word the server worked out (nodes/tags.py strictest): 仅限研究 / 非商用 / 需注册 / 可商用
}

/** 低/中/高/超高: a GPU node's compute rating (lab2shot/nodes/compute.py). */
interface ComputeRating {
  tier: "低" | "中" | "高" | "超高";
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
  // output port name -> its layer when delivered as EXR (data/layers.py LAYER_FOR_PORT): fills in the layer name when a wire is dragged into 「多层 EXR 输出设置」
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

interface GraphNodeJSON {
  id: string;
  type: string;
  label: string;
  params: Record<string, unknown>;
  promoted?: string[]; // 提升到节点: each gets an input "param:<name>" and a row on the node's body
  // editor only: where the node's files came from (by parameter); the parameters its body shows when the
  // user chose others than its type's (NodeTypeDef.on_node, 「在节点上显示」, ParamPanel.tsx OnNodePin)
  // may be absent from the file (a hand-written or older graph): every reader treats absence as the default (graph/document.ts loadGraph / checkGraph)
  ui?: { x: number; y: number; picked?: Record<string, PickedFrom>; on_node?: string[] };
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

/** A template's parameter interface (the graph's `exposed`, in the manner of Houdini's Edit Parameter Interface): a
 * tree, each array item a parameter or a group, and a group again such an array (a subgroup is a group in a group). An
 * older file's flat list is a tree without groups and is used as read (lab2shot/engine/templates.py module docstring).
 *
 * A parameter: `name` its outside name (lab2shot cook --set and the DCC plugins pass values by it:
 * engine/templates.py apply_values), `label` its display name, `target` "<node id>.<parameter name>"; `widget`
 * overrides the control: `menu` a dropdown (`options` values and labels filled in by the template's author, values
 * of the target parameter's type), `checkbox` a checkbox (the target is a boolean, or an integer whose options are
 * exactly 0 / 1: ticked = 1); two condition expressions (platform/conditions.ts), Houdini's Hide When / Disable When:
 * while `hide_when` is true the item is hidden (a group whose items are all hidden is hidden too), while
 * `disable_when` is true it is greyed and cannot be changed. The `when` that older files write (true: may be changed)
 * is read as disable_when = not (when) (state/cookInputs.ts readExposed). `target` may also be a button parameter
 * ("<node id>.cook" to cook, "<output id>.download" to download): a button takes neither menu nor checkbox. */
export interface ExposedParam {
  name: string;
  label: string;
  target: string;
  widget?: "menu" | "checkbox";
  options?: ExposedOption[];
  hide_when?: string;
  disable_when?: string;
  // show the node in the viewer on change: once it is changed in the parameter panel's exposed tree or in app mode (any
  // control; every click of a pick counts), the viewer switches to the target node (as a double click does). A pick or
  // drawing control shows it as soon as its row is touched. Defaults to false (editor/ParamPanel.tsx)
  show_on_change?: boolean;
}

export interface ExposedOption {
  value: unknown;
  label: string;
  // this option's own Hide When (the same syntax as an item's, platform/conditions.ts): while true the dropdown leaves it out; a parameter whose value is this option keeps it, and it stays listed
  hide_when?: string;
}

/** A group of the parameter interface: collapsible (`collapsed` is its default state on opening), and may hold further groups. */
export interface ExposedGroup {
  kind: "group";
  label: string;
  collapsed?: boolean;
  children: ExposedEntry[];
}

export type ExposedEntry = ExposedParam | ExposedGroup;

export interface GraphJSON {
  schema: "lab2shot.graph/1";
  // id: the graph's own identity, 128 random bits as 32 hex
  // characters (model/graphId.ts newGraphId), never an integer or anything else that could collide. Generated the
  // moment a document is created (a template opened, a 另存为 copy) or, for a file without one, the
  // moment it is opened (and then marked changed, so saving writes it in); a plain open, save or reopen keeps it.
  // Jobs and outputs carry it, so the page only ever draws an output on the graph it belongs to. Two graphs from
  // the same template never share one; the id inside a template file (templates/*.json) exists only so the file
  // itself always has one, and is never inherited by a graph created from it.
  // A graph never records which template it came from: saved from a template, it is an ordinary graph. `intro` and
  // `deliverable` are a template file's own words (lab2shot/library.py), kept like any key of meta; the save dialogs
  // prefill from them (editor/MyTemplates.tsx).
  meta: { id?: string; name: string; description?: string; author?: string; created?: string; category?: string; tags?: string[]; intro?: string; deliverable?: string };
  exposed: ExposedEntry[]; // the template's parameter interface, a tree (ExposedEntry); an older file's flat list is one with no groups
  frames?: [number, number] | null; // the frames to cook, first and last (null: every frame of the inputs)
  nodes: GraphNodeJSON[];
  edges: { from: [string, string]; to: [string, string] }[];
  boxes?: BoxJSON[];
  view?: { display: string | null; frame: number | null; port?: string | null; playback?: [number, number] | null }; // playback: in / out; absent in a graph written by a script
}

export interface TemplateInfo {
  id: string;
  name: string;
  intro: string; // its introduction: the techniques it combines and what it is for
  deliverable: string; // where its card sits (its file's meta.deliverable): a subcategory or first-level category id, "" 未分类
  category: string; // the first-level category it sits in, "" 未分类 (also when its place is no longer in the tree)
  graph: GraphJSON;
  projects: { name: string; title: string }[]; // third-party projects the template uses
  // the small tag at the card's lower right: the core project's paper or release year (the server picks the core project from the declarations; the page never picks)
  year: number | null;
  licence: string[]; // its tags (lab2shot/nodes/tags.py): every node's and choice's in it
  // the one word the card shows for all of them: the strictest (nodes/tags.py strictest). The page prints it and
  // never picks among the tags itself, nor reads it: whether that word means 可商用 is the boolean beside it.
  licence_word: string;
  commercial: boolean;
  // the best route another choice of its menus gives (the server's least strict route): its word and whether it means
  // commercial use; the same as licence_word / commercial when no choice does better
  best_licence_word: string;
  best_commercial: boolean;
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

/** GET /api/ocio: the options of the 「色彩空间」 dropdown and the working space's name (lab2shot/io/color.py: everything is
 * converted into the working space on read and out of it on write; the working space is the sRGB the screen shows, so
 * there is no display device or view level). */
export interface OcioInfo {
  name: string;
  colorspaces: string[];
  working: string;
}
