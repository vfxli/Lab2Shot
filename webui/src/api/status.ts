// What the server says about a graph (POST /api/status, engine/evaluation.py): messages, node and wire states, cook
// kinds and plans, choices, and the data packets the views read. Re-exported by api/index.ts.

import type { Availability, MessageJson } from "./applies";
import type { Level } from "../messages/format";
import type { Places } from "../model/places";
import type { NodePorts, ResolvedCost, ResolvedLicence } from "./catalog";
import type { JobState } from "./queue";
import { t } from "../i18n/t.ts";
import type { Phase } from "./progress";

/** A standing mark (src/ui/nodeMarks.ts): the short word of a notice a node's declaration always gives it, derived on
 * the server (nodes/applies.py standing_marks). Its level decides its colour: red only for P. */
export interface StandingMark extends ServerMessage {
  mark: string; // the message's short words, one line on the node's bottom row; `text` is the whole sentence (its 数据信息 card)
}

/** A message the server says about a node (lab2shot/messages): a usage check that failed (engine/lint.py: the node
 * connects and cooks, but probably not as meant; never blocks), or what the node said while it was cooked, kept with
 * its result. */
export interface ServerMessage {
  code: string; // level letter-module-meaning
  level: Level;
  text: string;
  // the few words for a place with one line (a node's footer): the message's own short form when it declares one
  // (lab2shot/messages 「<CODE>.short」, at most 12 full-width characters), else the whole sentence. The page never
  // cuts or abbreviates text itself.
  short?: string;
  // its words for whoever uses a card (lab2shot/messages 「<CODE>.app」: no node names, no wires), when it has them;
  // messages/message.ts textOf says them in app mode
  app?: string;
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
  /** The generation of each port written to disk (when its packet was committed): part of the page's cache keys (transfer/gens.ts, transfer/frameKey.ts) */
  gens?: Record<string, string>;
  // the node's outputs that are needed: those with a wire out of them; every output when it has no wire at all.
  // Computed by the server alone (`lab2shot/engine/evaluation.py needed_outputs`); the page never works it out again.
  // Decides which channels' bytes this cook must upload (upload follows the channels the wires use).
  needed?: string[];
  // the channels this cook must upload (only on a node that reads a file; `lab2shot/nodes/core/input.py
  // ReadSequence.upload_channels`): `take` are the channel names in the file (the worker decodes by them), `write` the
  // names in the subset EXR (the server writes by them). Absent: the whole file is uploaded (PNG / JPG have no separate
  // channels and need all of them). Which port maps to which channel names is defined on the server only; the page
  // never repeats it (`transfer/planes.ts` just follows this)
  channels?: { take: string[]; write: string[] };
  applies: Availability; // its parameters that declare a condition: available, greyed with why (I-APPLIES-*), pending a cook (read only through applies.ts)
  cost?: ResolvedCost; // what it costs with its parameters
  licence?: ResolvedLicence; // whose licence its result is under with them
  // no result because of an error: its own, or the node `root`'s; `blocked`: skipped behind a 「阻断」 set to block
  // (engine/records.py Outcome.blocked) — not an error, shown 「已跳过（被阻断）」 (model/nodeOutcome.ts)
  outcome?: { state: "failed" | "skipped"; root: string; blocked?: boolean };
  skipped?: ServerMessage; // skipped: why (N-COOK-SKIPPED)
  messages: ServerMessage[];
  sources: Record<string, string>; // where its parameters get their values, for those that say so ("Focal Length 38.6 mm · 来自 AnyCalib（覆盖相机的 Focal Length）")
  // of those set over a connected input, that clause on its own ("覆盖相机的 Focal Length"): the panel shows it on its own line
  overrides: Record<string, string>;
  values?: Record<string, string>; // its value outputs, once known ("38.6 mm")
  curves?: string[]; // of those, the ones holding a number per frame that changes (data/values.py varies): worth a curve
  strip?: { label: string; text: string }[]; // the current values of the parameters the node declares for the value strip (NodeDef.strip; not output ports)
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
  // for the input ports its parameters' options come from (P(choices_from)): the packet that stands for what is wired
  // in (port -> fingerprint), before the node itself can cook (engine/evaluation.py stand_ins; ui/choices.ts)
  stand_ins?: Record<string, string>;
  // a switch whose route is known: the input ports it takes, its condition among them (engine/evaluation.py
  // taken_ports, the one answer to 「走哪一路」); the view shows what the taken way brings (view/plan.ts madeFrom)
  taken?: string[];
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
  // every node the cook takes something from, cached or not, along the routes switches take (engine/evaluation.py
  // needed), dependencies first: what the page checks before submitting (graph/nodes.ts blockers / standing)
  uses?: string[];
  delivers: boolean;
}

/** A wire as the status reply says it (Graph.wire_states). */
export interface WireStatus {
  from: [string, string];
  to: [string, string];
  type: string; // what it carries ("" its output is not there)
  state: "ok" | "waiting" | "unused" | "wrong"; // unused: into a switch on a way not taken; problems there do not count (server: Evaluation.unchosen_wires)
  problem: ServerMessage | null;
  fix: string; // the node type put in between makes it right ("" none)
}

/** POST /api/status: one request per edit. */
/** A file this account's page is having read in the background for an import's listing (no 「计算」 was pressed, so no
 * job exists and the queue cannot cancel it): the top bar says so with 「停止」 (engine/external.py ask_worker). */
export interface Reading {
  label: string; // the node type, 「导入 FBX」
  file: string;
  seconds: number;
  limit: number; // it is stopped by itself after this many seconds
}

export interface StatusReply {
  again_ms?: number; // a file is being read in the background: ask again this soon (what it holds shows by itself)
  readings?: Reading[]; // this account's readings in progress
  graph: string; // the graph version's key
  cook_inputs: number; // the page's cook-inputs version this answers, said back
  nodes: Record<string, NodeStatus>;
  wires: WireStatus[];
  deliver: CookCase | null; // 提交: every 「输出」 together (null: none)
  plan: Plan | null; // the shown node's (null: nothing shown)
  // why the other nodes whose 「计算」 the page offers (StatusRequest holds: a card's buttons) can't be cooked now, by node;
  // those that can are not listed
  holds?: Record<string, MessageJson>;
  scopes?: Scope[]; // the graph's 逐项处理 blocks, their members and their items (engine/scopes.py; read by state/items.ts)
  // what the displayed node's handles need to draw (only when a node is displayed and declares a handle that needs data:
  // skeleton_pose, rig_pair). `handles` is keyed by the handle's index in its NodeDef.handles, as a string; which shape a
  // handle's data has follows its kind (HandleDef.kind)
  // `key` changes with the parameters and input packets: a request sending the key it holds (StatusRequest handle_key)
  // gets back only {node, key} while it still matches, and keeps using its copy (state/results.ts)
  handle_data?: { node: string; key: string; handles?: Record<string, HandleData> };
}

/** One handle's data in handle_data: a skeleton_pose handle's skeleton, or a rig_pair handle's two skeletons. */
export type HandleData = SkeletonPoseData | RigPairData;

/** A skeleton_pose handle's skeleton (lab2shot retarget handle_data): its joints and their base pose before the
 * corrections, as locals (matrices of 16, column-major, column vectors; cm; local = inv(parent world) @ world, the root's
 * local its world). The page computes every world it draws or needs from these by FK (model/skeletonPose.ts forward):
 * the base pose with no rows, the corrected one with the current rows. */
export interface SkeletonPoseData {
  packet: string; // the packet standing for the handle's input (the skinning preview loads its character)
  path: string; // the skeleton's prim path in it
  names: string[];
  parents: number[]; // -1: a root; a parent comes before its children
  pose: string; // which base pose `before` is, in words (「第一帧（摆成 T 姿）」)
  before: { local: number[][] };
  mirror: [number, number][]; // joint index pairs, left and right
  mirror_plane: { normal: [number, number, number]; point: [number, number, number] } | null; // the body's left–right plane (world); null: no mirroring
  unknown: string[]; // joint names in the parameter the skeleton does not have
  units: { translate: string; rotate: string };
  rotation: string; // the rotation order of the rows' angles ("XYZ": about X first; model/math3d.ts rotation)
  order: string; // "local @ T·R·S"
}

/** Fills in the defaults of a status reply as it is read (the one place that does): when the server leaves out a list
 * or a table (an older version, a reply cut short by an error), the rest of the page reads it all the same, with no
 * checks of its own and no throw while iterating. Only fills in "nothing"; never changes a value the server gave. */
export function normalizeStatus(r: StatusReply): StatusReply {
  const nodes: Record<string, NodeStatus> = {};
  for (const [id, n] of Object.entries(r.nodes ?? {}))
    nodes[id] = {
      ...n,
      messages: n.messages ?? [],
      sources: n.sources ?? {},
      overrides: n.overrides ?? {},
      applies: n.applies ?? { available: [], inactive: {} },
      handles: n.handles ?? [],
      ports: n.ports ?? { inputs: [], outputs: [], waiting: [] },
    };
  return { ...r, nodes, wires: r.wires ?? [], deliver: r.deliver ?? null, plan: r.plan ?? null };
}

/** A node a cook will compute (lab2shot/farm/timings.py). */
interface PlanNode {
  node: string;
  label: string;
  frames: number;
  width: number;
  height: number;
}

/** A look at a cook before submitting it (farm/timings.py look): the shown node's, with every status reply. */
export interface Plan {
  node: string; // what it is for
  range: [number, number] | null; // the frames the inputs cover (null: no sequence input)
  frames: [number, number] | null; // the frames this cook takes
  nodes: PlanNode[]; // what it computes, in order
  cached: number; // nodes it takes from the cache
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
  default?: string; // the value an empty parameter should take (the colour space the format implies): the page writes it into the parameter rather than showing 「自动」 (editor/ParamControls.tsx)
  none?: string;
  empty?: string;
  // the 「表情重定向」 mapping table (widget expression_map, editor/ExpressionMap.tsx; lab2shot/nodes/kit/rig_map.py
  // expression_choice): one row per expression slot, the target's blendshapes, where both sides are (only shown)
  rows?: ExpressionSlot[];
  shapes?: string[];
  source?: string;
  target?: string;
}

/** One row of the expression table: a slot (data/expressions.py slot_rows: id, label, region), the source curves that
 * can fill it, and the guessed pair (a row of the parameter's format; null: nothing guessed). */
export interface ExpressionSlot extends RigPart {
  curves: string[];
  auto: RigRow | null;
}

/** A body part (lab2shot/data/joints.py part_rows). */
export interface RigPart {
  id: string;
  label: string;
  region: string;
  chain: boolean; // spine, neck, fingers: may span several joints
  required: boolean; // this node needs it mapped on both sides
  end?: string; // the part a chain part ends at (spine → chest, neck → head; server: data/joints.py CHAIN_ENDS); fingers have none
}

/** One row of the parameter's value (lab2shot/nodes/kit/rig_map.py PartMap): src drives, dst is driven; joint names. */
export interface RigRow {
  part: string;
  src: string[];
  dst: string[];
}

/** One side of a rig_pair handle (the server's kit/rig_map.py skeleton_handle's fields, plus the side's parts, the
 * suggested pose rows and whether it is a model's fixed skeleton). A fixed side has only names, parents and parts: no
 * positions, so it is drawn as a tree only. */
export interface RigPairSide extends Partial<Omit<SkeletonPoseData, "names" | "parents">> {
  names: string[];
  parents: number[];
  parts: Record<string, number[]>; // part -> joint indices (the parameter's rows over the guess, as the cook reads them)
  auto_pose?: { joint: string; translate: number[]; rotate: number[]; scale: number[] }[]; // 「自动姿态」's rows (nodes with a pose role)
  // 「自动尺寸」 (nodes with size roles): the factor it suggests, the side's leg length at its own size (cm), and the
  // world height of its ground (a factor s ≠ 1 maps the world p ↦ s·(p − (0, ground, 0)): scaled and stood on y = 0);
  // absent when no legs are paired
  size?: { auto: number; leg_cm: number; ground: number };
  fixed: boolean;
  recognition?: RigRecognition; // the skeleton recognition engine's judgement of this side (absent on a fixed side)
}

/** What the skeleton recognition engine said of one skeleton (lab2shot/data/skeleton_recognition.py Recognition.report):
 * per body part its confidence (0–1) and evidence (short sentences), `assigned: false` for a part it found a candidate
 * for but did not give (below `threshold`, or on a joint that drives no vertex). */
export interface RigRecognition {
  parts: Record<string, { confidence: number; evidence: string[]; assigned?: boolean }>;
  threshold: number;
}

/** An ignore rule as the 「自动忽略」 dropdown lists it (lab2shot/nodes/kit/retarget_needs.py): its id, its name, and the
 * joints it suggests ignoring on each side (the full list, descendants included). */
export interface RigPairRule {
  id: string;
  label: string;
  ignore: { src?: string[]; dst?: string[] };
  slots?: Record<string, number>; // the model's fixed skeleton: how many joints each part takes (nodes/kit/retarget_needs.py counts)
  required?: string[]; // the parts it cannot do without
}

/** A rig_pair handle's data (nodes/handles.py RigPair): the two skeletons, the part table, the guessed mapping, and for a node with
 * ignore roles the rules with their suggestions and the rule to preselect (the solver it is wired to; only a
 * preselection — the result reads the ignore_rule parameter alone). */
export interface RigPairData {
  src: RigPairSide;
  dst: RigPairSide;
  parts_table: RigPart[];
  auto_mapping: RigRow[];
  rules?: RigPairRule[];
  default_rule?: string;
}

/** The text of a dropdown's 「还没选」 row, worked out in this one place (the parameter panel and the node both read it):
 * 「自动 · X」 when the node declares it picks one automatically; the declared meaning of empty when there is one
 * (Choice.empty: 「选一台」, 「这个网格没有分区」); the parameter's own declared placeholder while the options are not
 * known yet (upstream not cooked, NodeDef.choices cannot be asked; P(placeholder="先接上模型")); 「自动」 only when
 * none of these exists. Only a parameter that really picks automatically may show 「自动」: one that does not would
 * leave the user waiting for a result that never comes (e.g. 「按分区取出」's 「分区」 before a model is wired in
 * stops and waits for the user's choice). */
export function emptyChoiceLabel(choice: Choice | null, placeholder: string, name: (option: string) => string = (o) => o): string {
  if (choice === null) return placeholder || t("ui.misc.choice_auto");
  if (typeof choice.auto === "string" && choice.auto) return t("ui.misc.choice_auto_found", { option: name(choice.auto) });
  if (choice.auto === "") return t("ui.misc.choice_auto_none");
  return choice.empty || placeholder || t("ui.misc.choice_auto");
}

export interface Manifest {
  type: string;
  fingerprint: string;
  created?: string; // the generation (changes when the same fingerprint is computed again)
  meta: Record<string, unknown> & { frames?: number[]; width?: number; height?: number; values?: boolean; colorspace?: string; range?: number[]; classes?: unknown[]; data_window?: number[] };
  /** The channel names the packet holds (declared by the core: lab2shot/data/payloads.py channel_list). The per-channel
   * route (api.channelUrl) accepts only names from this list; packets that are not 2D pixels (cameras, point clouds,
   * curves, values) have none. */
  channels?: { names: string[] };
  /** The packet's size in the viewer: the view proxy tier (the administrator's 「视图 · 视图代理尺寸」,
   * `lab2shot/view/proxy.py`) and the packet's width and height scaled to it. Packets that are not 2D pixels have none.
   *
   * The page puts `px` into the cache key, which then holds packet, frame, channel and proxy tier, so switching away
   * and back hits the cache without transferring a byte, and when the administrator changes the tier the key changes
   * with it, so bytes of one tier are never taken for another's. */
  // form: how this packet's proxies are made when not the usual way ("ids": an id map, nearest and exact values;
  // lab2shot/view/proxy.py form_of). It goes into the address (`pf=`) and the cache keys beside the tier (transfer/frameKey.ts Tier)
  proxy?: { px: number; width: number; height: number; form?: string };
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
  // this track data is a stick figure (the sketch `lab2shot/nodes/core/sketch.py` outputs, 18 joints per pose). The
  // packet carries the flag itself (`tracks_packet(…, figure=True)` → meta) and the server's `packets.py` passes it on
  // unchanged in `{**p.meta, …}`. The viewer draws by it: a stick figure as a body, not as track points with trails
  // (view/overlays.ts drawTracks)
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
    // cook progress has one event only (api/progress.ts, lab2shot/progress.py): the server sends stage / progress /
    // phase merged into one description, and the queue panel and the node read the same data
    | "node_start" | "node_done" | "progress" | "message" | "error" | "skipped" | "output" | "done"
    | "stopping" | "cancelled" | "finished";
  position?: number;
  waiting?: MessageJson | null; // queued: why it waits, the server's words
  waiting_detail?: MessageJson | null; // queued: the reason about the cards, when this session may see them
  reason?: string | MessageJson; // cancelled: why, when not by the one who started it
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
  gens?: Record<string, string>; // node_done: each output port's packet generation (lab2shot/server/wire.py generations)
  phase?: Phase; // progress: 排队中 / 加载模型 / 计算 / 取回结果
  note?: string; // progress: the solver's current step (e.g. 「检测人物」), a sentence only
  at?: number | null; // progress: how much of the whole task is done, 0–1, never decreasing; null: cannot be estimated (an indeterminate bar)
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
