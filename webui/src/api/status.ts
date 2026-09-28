// What the server says about a graph (POST /api/status, engine/evaluation.py): messages, node and wire states, cook
// kinds and plans, choices, and the data packets the views read. Re-exported by api/index.ts.

import type { Availability, MessageJson } from "./applies";
import type { Level } from "../messages/format";
import type { Places } from "../model/places";
import type { NodePorts, ResolvedCost, ResolvedLicence } from "./catalog";
import type { JobState } from "./queue";
import type { Phase } from "./progress";

/** A standing mark (src/ui/nodeMarks.ts): the short word of a notice a node's declaration always gives it, derived on
 * the server (nodes/applies.py standing_marks). Its level decides its colour: red only for P. */
export interface StandingMark extends ServerMessage {
  mark: string; // the message's short words, one line on the node's bottom row; `text` is its tooltip
}

/** A message the server says about a node (lab2shot/messages): a usage check that failed (engine/lint.py: the node
 * connects and cooks, but probably not as meant; never blocks), or what the node said while it was cooked, kept with
 * its result. */
export interface ServerMessage {
  code: string; // 类型字母-模块-含义
  level: Level;
  text: string;
  // the few words for a place with one line (a node's footer): the message's own short form when it declares one
  // (lab2shot/messages 「<CODE>.short」, at most 12 full-width characters), else the whole sentence. The page never
  // cuts or abbreviates text itself.
  short?: string;
  params?: Record<string, unknown>;
  port?: string; // the input it is about
  param?: string; // the parameter it is about
  fix?: { insert: string; label: string }; // a node type inserted between the input and what feeds it puts it right
  refused?: boolean; // not a usage check: a refused wire (the node's error) the fix makes right
}

export interface NodeStatus {
  fingerprint: string | null;
  cached: boolean;
  // what the engine says of it now: cached it has a result; todo it would be computed; pending it waits for
  // something; failed / skipped / error say why there is none; unused it is not on the way to anything asked for
  state?: "cached" | "todo" | "pending" | "failed" | "skipped" | "unused" | "error";
  present?: string[]; // its outputs that already have a data packet
  /** 每个已落盘端口的生成号（包提交的时刻）：页面的缓存键包含该值（transfer/gens.ts、transfer/ident.ts） */
  gens?: Record<string, string>;
  // 该节点被需要的输出口：有连线引出的口；没有任何连线时为全部输出口。
  // 由服务器集中计算（`lab2shot/engine/evaluation.py needed_outputs`），网页不重复推算。
  // 用于决定本次计算需要上传哪些通道的字节（根据连线使用的通道启用上传）。
  needed?: string[];
  // 本次需要上传的通道（仅读取文件的节点具有；`lab2shot/nodes/core/input.py ReadSequence.upload_channels`）：
  // `take` 是文件中的通道名（worker 据此解码），`write` 是子集 EXR 中的名称（服务器据此写入）。
  // 缺少该项表示上传整个文件（PNG / JPG 不区分通道，需要全部通道）。端口与通道名的对应关系仅在服务器定义，
  // 网页不重复实现（`transfer/planes.ts` 依此执行）
  channels?: { take: string[]; write: string[] };
  applies: Availability; // its parameters that declare a condition: available, greyed with why (I-APPLIES-*), pending a cook (read only through applies.ts)
  cost?: ResolvedCost; // what it costs with its parameters
  licence?: ResolvedLicence; // whose licence its result is under with them
  outcome?: { state: "failed" | "skipped"; root: string }; // no result because of an error: its own, or the node `root`'s
  skipped?: ServerMessage; // skipped: why (N-COOK-SKIPPED)
  messages: ServerMessage[];
  sources: Record<string, string>; // where its parameters get their values, for those that say so ("Focal Length 38.6 mm · 来自 AnyCalib（覆盖相机的 Focal Length）")
  values?: Record<string, string>; // its value outputs, once known ("38.6 mm")
  strip?: { label: string; text: string }[]; // 该节点声明在值条上显示的参数的当前值（NodeDef.strip；不是输出口）
  outputs?: Record<string, string>;
  // a node inside a 逐项处理 block (engine/scopes.py): the fields above are the item the view is on, `item` says which
  // one it is, `summary` how all of its items stand together (「4/5 条已算 · 1 条失败」). Item by item is a call of
  // its own: GET /api/status/{graph}/node/{node}/items
  item?: { path: string[]; names: string[] };
  summary?: ItemSummary;
  ports: NodePorts; // its ports in this graph (what its 3D outputs carry: ports.outputs[].kinds)
  handles: number[]; // its viewer handles that apply now, by index into its type's handles
  places?: Places | null; // its placement (its type's, repeated per node)
  policy: CookCase; // what a click on 计算 cooks
  error?: ServerMessage; // why it can't be planned yet (no file chosen ...), or a wired value it can't take
}

/** One item of a node inside a block, as `GET /api/status/{graph}/node/{node}/items` answers it: the same fields the
 * status reply gives for the item the view is on, plus which item it is. */
export interface ItemStatus extends Partial<NodeStatus> {
  item: { path: string[]; names: string[] };
}

/** That call's answer: a page of items (offset from the first, at most 200 in one go). */
export interface ItemsPage {
  graph: string;
  node: string;
  total: number;
  offset: number;
  items: ItemStatus[];
}

/** How all the items of a node inside a block stand (engine/scopes.py summary): how many instances, and per state
 * (only the states that have any) how many. */
interface ItemSummary {
  total: number;
  cached?: number;
  todo?: number;
  pending?: number;
  failed?: number;
  skipped?: number;
  unused?: number;
  error?: number;
}

/** One 逐项处理 block as the server works it out (engine/scopes.py: the page never finds a block's members itself).
 * `lists`: the items the block has, one entry per path of the blocks around it (`pending`: not known until something
 * above is cooked). */
export interface ScopeItem {
  key: string; // what an item path carries
  name: string; // the item's own name (user data: sh010, person_01)
}

export interface ScopeList {
  path: string[]; // the item keys of the enclosing blocks
  items?: ScopeItem[];
  summary?: ItemSummary; // how the block's items stand together (engine/evaluation.py _scope_summary): 「3 条 · 2/3 已算」
  pending?: boolean;
}

export interface Scope {
  kind: string; // "each": the only kind of block
  name: string; // the block name that pairs its begin and ends
  begin: string;
  ends: string[];
  members: string[];
  parent: string | null; // the begin of the block around it
  lists: ScopeList[];
}

/** Cooking some targets (engine/evaluation.py _case): what they are, the nodes it computes (not cached), and whether it
 * collects and packs files for download (a 「输出」 among them). */
export interface CookCase {
  targets: string[];
  computes: string[];
  delivers: boolean;
}

/** A wire as the status reply says it (Graph.wire_states). */
export interface WireStatus {
  from: [string, string];
  to: [string, string];
  type: string; // what it carries ("" its output is not there)
  state: "ok" | "waiting" | "wrong";
  problem: ServerMessage | null;
  fix: string; // the node type put in between makes it right ("" none)
}

/** POST /api/status: one request per edit. */
export interface StatusReply {
  graph: string; // the graph version's key
  cook_inputs: number; // the page's cook-inputs version this answers, said back
  nodes: Record<string, NodeStatus>;
  wires: WireStatus[];
  deliver: CookCase | null; // 提交: every 「输出」 together (null: none)
  plan: Plan | null; // the shown node's (null: nothing shown)
  scopes?: Scope[]; // the graph's 逐项处理 blocks, their members and their items (engine/scopes.py; read by state/items.ts)
}

/** A node a cook will compute, with its estimate (lab2shot/farm/timings.py). */
interface PlanNode {
  node: string;
  label: string;
  frames: number;
  width: number;
  height: number;
  seconds: number | null; // null: no record of this node yet
  records: number; // how many earlier runs the estimate rests on
  device?: string; // the card (or CPU) those runs were on: card information, sent only to whoever may see the cards (farm.cards)
}

/** A look at a cook before submitting it (farm/timings.py look): the shown node's, with every status reply. */
export interface Plan {
  node: string; // what it is for
  range: [number, number] | null; // the frames the inputs cover (null: no sequence input)
  frames: [number, number] | null; // the frames this cook takes
  nodes: PlanNode[]; // what it computes, in order
  cached: number; // nodes it takes from the cache
  seconds: number; // the sum of the estimates there are
  unknown: number; // nodes without records
  error: MessageJson | null; // not plannable yet: why, as a catalogue message (the blockers say more)
}

/** Options of a parameter that come from a file or from what is wired in (NodeDef.choices): the list, per option a
 * name to show and other names that choose it (a segmentation class: its Chinese name, its number), what the empty
 * value comes to (自动; for a table, per entry name; "" nothing found), when "" means "none" its name (不映射), and
 * what the empty value shows when it has no 自动 (先选择相机文件, 选一台). */
export interface Choice {
  options: string[];
  labels?: Record<string, string>;
  details?: Record<string, string>; // what tells an option apart, shown next to it (an import node's entry: frames, focal, counts)
  aliases?: Record<string, string[]>;
  auto?: string | Record<string, string>;
  default?: string; // 参数为空时应填写的值（按格式确定的色彩空间）：网页将其写入参数，不显示为「自动」（editor/ParamControls.tsx）
  none?: string;
  empty?: string;
}

/** 下拉中「还没选」一行的文字，集中计算（参数面板与节点均读取）：节点声明会自动选择某项时显示「自动 · 某某」；
 * 声明了空值含义（Choice.empty：「选一台」「这个网格没有分区」）时使用该说明；选项尚未知时（上游尚未计算，
 * 无法查询 NodeDef.choices）使用参数自身声明的 placeholder（P(placeholder="先接上模型")）；均缺失时才显示「自动」。
 * 只有确实会自动选择的参数才可显示「自动」：不会自动选择却显示「自动」，使用者会等待一个不会出现的结果
 * （例如「按分区取出」的「分区」在接入模型之前，实际会停下等待使用者选择）。 */
export function emptyChoiceLabel(choice: Choice | null, placeholder: string, name: (option: string) => string = (o) => o): string {
  if (choice === null) return placeholder || "自动";
  if (typeof choice.auto === "string" && choice.auto) return `自动 · ${name(choice.auto)}`;
  if (choice.auto === "") return "自动 · 没找到";
  return choice.empty || placeholder || "自动";
}

export interface Manifest {
  type: string;
  fingerprint: string;
  created?: string; // 生成号（同一指纹重新计算后改变）
  meta: Record<string, unknown> & { frames?: number[]; width?: number; height?: number; values?: boolean; colorspace?: string; range?: number[]; classes?: unknown[]; data_window?: number[] };
  /** 该包包含的通道名称列表（核心声明：lab2shot/data/payloads.py channel_list）。按通道获取的接口
   * （api.channelUrl）只接受该列表中的名称；非二维像素的包（相机、点云、曲线、数值）没有此项。 */
  channels?: { names: string[] };
  /** 该包在视图中的尺寸：视图代理档位（管理员设置「视图 · 视图代理尺寸」，
   * `lab2shot/view/proxy.py`）以及该包按此缩放后的宽高。非二维像素的包没有此项。
   *
   * 页面将 `px` 纳入缓存键：包、帧、通道、代理档位四项齐全，
   * 因此切走再切回时命中缓存，无须传输任何字节；管理员更换档位后键随之改变，不会将旧档位的字节误用为新档位的数据。 */
  proxy?: { px: number; width: number; height: number };
  summary: DataSummary; // what this result is, said once by the server (lab2shot/data/summary.py describe)
}

/** One line of a result's summary: its value (for 「取信息」 and a page that lays it out in columns), the name of the
 * line and what it says. Made by lab2shot/data/summary.py from the data type's own declaration: the port tooltip,
 * the 数据信息 panel and 「取信息」 all read this one answer and the page writes no sentence of its own. */
interface SummaryLine {
  id: string;
  code: string; // the message code that said it
  label: string; // the line's name ("" when the sentence stands on its own)
  value: unknown;
  text: string; // what it says, without the name
  said: string; // the whole sentence, name included
}

interface DataSummary {
  type?: string;
  label?: string; // the data type's name
  known?: "cooked" | "planned" | "empty";
  items?: SummaryLine[];
}

/** The text an output-settings node wrote for another application, ready to put on the clipboard. */
export interface ClipboardText {
  app: string; // the application that reads it ("nuke")
  file: string; // the file it was written as, inside the result
  text: string;
  // what the person has to be told at the moment they copy: what that application cannot hold of this result (a lens
  // whose model has no node there travels as its intrinsics only). Said after the copy, never at cook time
  said?: { code: string; level: string; text: string };
}

export interface BoxesData {
  people: { id: number; boxes: Record<string, number[]>; prominence?: number }[];
  width: number;
  height: number;
  frames: number[];
}

export interface TracksData {
  points: ([number, number] | null)[][]; // per point, frame-aligned with `frames`
  outline?: { corners: [number, number][]; seen: boolean }[]; // a planar track: its quad on every frame (estimated where not seen)
  confidence?: number[][]; // a tracker that scores its points: per point, frame-aligned, 0..1
  names: string[];
  frames: number[];
  width: number;
  height: number;
  // 该跟踪点数据是一个火柴人（`lab2shot/nodes/core/sketch.py` 输出的草图，每个姿势 18 个关节）。
  // 该标记由包自带（`tracks_packet(…, figure=True)` → meta），服务器 `packets.py` 的 `{**p.meta, …}` 原样转发。
  // 视图据此决定绘制方式：火柴人按人体绘制，而非按「跟踪点 + 轨迹」绘制（view/overlays.ts drawTracks）
  figure?: boolean;
}

export interface CurvesData {
  names: string[];
  values: number[][]; // per curve, frame-aligned with `frames`
  frames: number[];
  range: [number, number];
  /** 前后对比: {后通道名: 前通道名}. A node that changes a curve (「相机去抖」, 「锁定 Focal Length」,
   * 「重定时」) hands out both, paired here; the curve editor draws the pair and knows nothing of the node. */
  before?: Record<string, string>;
}

export interface CookEvent {
  type:
    | "queued" // waiting in the queue (again whenever its place or why it waits changes): position, waiting
    | "started" // its first node got its place
    // 计算进度只有一种事件（api/progress.ts、lab2shot/progress.py）：服务器将 stage / progress / phase
    // 合并为同一份描述后发送，队列面板与节点读取同一数据
    | "node_start" | "node_done" | "progress" | "message" | "error" | "skipped" | "output" | "done"
    | "stopping" | "cancelled" | "finished";
  position?: number;
  waiting?: MessageJson | null; // queued: why it waits, the server's words
  waiting_detail?: MessageJson | null; // queued: the reason about the cards, when this session may see them
  reason?: string; // cancelled: why, when not by the one who started it
  state?: JobState;
  node?: string | null;
  label?: string;
  name?: string;
  // node_done: the graph the job was submitted from, the cook inputs'
  // version the page submitted (POST /api/jobs `version`, carried back on every event) and the node's fresh output
  // fingerprints, written straight into results only when they still apply to this page (same graph, same version).
  graph?: string;
  version?: number;
  outputs?: Record<string, string>;
  phase?: Phase; // progress: 排队中 / 加载模型 / 计算 / 取回结果
  note?: string; // progress: 解算器报告的当前步骤（如「检测人物」），仅为一句文字
  at?: number | null; // progress: 整个任务的完成量，0–1，单调不减；null 表示无法估计（绘制不确定进度条）
  message?: string; // progress: what it is counting
  code?: string; // message / error: its code, level and words (lab2shot/messages)
  level?: ServerMessage["level"];
  text?: string;
  port?: string;
  param?: string;
  done?: number;
  total?: number;
  cached?: boolean;
  seconds?: number;
  log?: string | null;
  // output: what the 「输出」 packed (api/files.ts Output: its task, pkg, name, bytes, count)
  task?: string;
  pkg?: string;
  bytes?: number;
  count?: number;
}
