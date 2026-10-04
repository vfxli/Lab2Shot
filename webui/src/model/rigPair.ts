/** 双骨架视图编辑（手柄种类「rig_pair」，服务端 nodes/handles.py RigPair）的纯算术：对应关系的合并与改写、骨点的状态
 * （已配对 / 未配对 / 忽略）、配对连线、忽略名单的切换、眼睛隐藏的遮罩、三个「自动」功能写什么参数，以及状态行的判定
 * （autoState）。舞台与树（view/rigPair.tsx、view/rigTree.tsx）只画、只收点击，规则都在这里。
 *
 * 数据是状态回复 handle_data 里这个手柄的那一份（api/status.ts RigPairData）。页面不自己推测对应、
 * 不自己算姿态、忽略或尺寸：四个「自动」只把服务端给的建议（auto_mapping、side.auto_pose、rules[].ignore、side.size.auto）写进参数，并在
 * auto_record 里记下这次写了什么。
 *
 * 对应关系参数的约定（同服务端 data/joints.py merged_rows）：参数里写了的部位以参数为准（「手动」），没写的部位按推测
 * （auto_mapping）；写回时只写手动的部位，一个也没有为 null（= 全部自动）。模型固定骨架的一侧（side.fixed）由节点
 * 定死：那一栏永远是模型这个部位的关节，改不了。
 *
 * 本文件只 import 类型（运行时被擦除），可以直接交给 Node 跑测试（model/rigPair.test.ts）。 */

import type { RigPairData, RigPairSide, RigPart, RigRecognition, RigRow } from "../api/status.ts";
import { t } from "../i18n/t.ts";
import { listSep } from "../i18n/words.ts";

export type Col = "src" | "dst";
export const COLS: Col[] = ["src", "dst"];
export const other = (c: Col): Col => (c === "src" ? "dst" : "src");

/** 手柄的角色 → 参数名（HandleDef.params）。mapping 必有，其余看节点声明了哪些。 */
export interface RigRoles {
  mapping?: string;
  src_pose?: string;
  dst_pose?: string;
  src_ignore?: string;
  dst_ignore?: string;
  ignore_rule?: string;
  auto_record?: string;
  src_scale?: string;
  dst_scale?: string;
}
export const poseRole = (c: Col): "src_pose" | "dst_pose" => (c === "src" ? "src_pose" : "dst_pose");
export const ignoreRole = (c: Col): "src_ignore" | "dst_ignore" => (c === "src" ? "src_ignore" : "dst_ignore");
export const scaleRole = (c: Col): "src_scale" | "dst_scale" => (c === "src" ? "src_scale" : "dst_scale");

// ------------------------------------------------------------------ 对应关系：参数行 + 推测

export interface Merged {
  rows: Record<string, RigRow>;
  manual: ReadonlySet<string>;
  stale?: readonly string[]; // 参数里点了现在骨架上没有的关节的部位（换过骨架）：按推测，同服务端 kit/retarget.py drop_stale
}

const copy = (r: RigRow): RigRow => ({ part: r.part, src: [...(r.src ?? [])], dst: [...(r.dst ?? [])] });

/** 固定一侧这个部位的关节名（side.parts 是关节序号）。 */
const fixedJoints = (side: RigPairSide, part: string): string[] => (side.parts?.[part] ?? []).map((i) => side.names[i]).filter((n): n is string => !!n);

/** 固定的一侧那一栏换成模型自己的关节。 */
function fixRow(row: RigRow, data: RigPairData): RigRow {
  let out = row;
  for (const c of COLS) if (data[c].fixed) out = { ...out, [c]: fixedJoints(data[c], row.part) };
  return out;
}

/** 参数值读成行（不认得的形状略去）。 */
export function mapRowsOf(value: unknown): RigRow[] {
  if (!Array.isArray(value)) return [];
  const names = (x: unknown) => (Array.isArray(x) ? x.filter((n): n is string => typeof n === "string") : []);
  return value.flatMap((r) => (r && typeof r === "object" && typeof (r as RigRow).part === "string"
    ? [{ part: (r as RigRow).part, src: names((r as RigRow).src), dst: names((r as RigRow).dst) }]
    : []));
}

/** 某个部位「自动」时是什么（推测；推测不到是两栏空）。 */
export function autoRow(part: string, data: RigPairData): RigRow {
  const found = data.auto_mapping.find((r) => r.part === part);
  return fixRow(found ? copy(found) : { part, src: [], dst: [] }, data);
}

/** 参数值 → 合并后的行：参数里写了的部位是手动的，其余用推测。 */
export function mergedOf(value: unknown, data: RigPairData): Merged {
  const rows: Record<string, RigRow> = {};
  for (const p of data.parts_table) rows[p.id] = autoRow(p.id, data);
  for (const r of data.auto_mapping) rows[r.part] ??= autoRow(r.part, data);
  const manual = new Set<string>();
  const stale: string[] = [];
  // 换过骨架：点了这副骨架上没有的关节的行不算手动，这个部位按推测（服务端 drop_stale 同一条规则，计算和视图一致）；
  // 下次一改，参数里就不再有它
  const known = (col: Col, n: string) => data[col].fixed || data[col].names.includes(n);
  for (const r of mapRowsOf(value)) {
    if (COLS.some((c) => (r[c] ?? []).some((n) => !known(c, n)))) {
      stale.push(r.part);
      continue;
    }
    rows[r.part] = fixRow(copy(r), data);
    manual.add(r.part);
  }
  return stale.length ? { rows, manual, stale } : { rows, manual };
}

/** 合并后的行 → 参数值：手动的部位按部位表的顺序（表外的跟在后面）；一个也没有 = null（全部自动）。 */
export function valueOf(m: Merged, data: RigPairData): RigRow[] | null {
  const order = [...data.parts_table.map((p) => p.id), ...Object.keys(m.rows).filter((k) => !data.parts_table.some((p) => p.id === k))];
  const out = order.filter((p) => m.manual.has(p) && m.rows[p]).map((p) => copy(m.rows[p]));
  return out.length ? out : null;
}

/** 这一侧：关节名 → 它现在属于哪个部位。 */
export function partOf(m: Merged, col: Col): Map<string, string> {
  const out = new Map<string, string>();
  for (const r of Object.values(m.rows)) for (const n of r[col]) if (!out.has(n)) out.set(n, r.part);
  return out;
}

// ------------------------------------------------------------------ 识别置信度（服务端识别引擎的判断，只读）

/** 置信度的三档：高（不提示）、中（提示色）、低（警告色）。阈值以下的部位服务端不分配（recognition.threshold）。 */
export type ConfLevel = "high" | "mid" | "low";
export const CONF_MID = 0.8; // 低于它为「中」
export const CONF_LOW = 0.65; // 低于它为「低」
export const confLevel = (c: number): ConfLevel => (c >= CONF_MID ? "high" : c >= CONF_LOW ? "mid" : "low");

export interface PartConfidence {
  part: string;
  value: number;
  level: ConfLevel;
  tip: string; // 悬停：置信度的数值与依据（界面上看不到的）
}

const confTip = (label: string, value: number, evidence: readonly string[]): string =>
  [t("ui.rig.confidence_tip", { part: label, value: value.toFixed(2) }), ...evidence].join("\n");

const sameJoints = (a: readonly string[], b: readonly string[]): boolean => a.length === b.length && a.every((n, i) => n === b[i]);

/** 这一侧按识别结果配的部位的置信度：关节名 → 它所属部位的置信度。这一栏的关节与推测（auto_mapping）相同才标——
 * 默认自动的、点「自动对应」写进参数的都标，手动改过的不标（那是使用者自己选的）。链只标第一节（一个部位一个标记）；
 * 固定一侧、服务端没给识别结果的不标。 */
export function confidenceOf(m: Merged, data: RigPairData, col: Col): Map<string, PartConfidence> {
  const out = new Map<string, PartConfidence>();
  const rec = data[col].recognition;
  if (!rec || data[col].fixed) return out;
  for (const r of Object.values(m.rows)) {
    if (!sameJoints(r[col], autoRow(r.part, data)[col])) continue;
    const said = rec.parts[r.part];
    const first = r[col][0];
    if (!said || said.assigned === false || !first || out.has(first)) continue;
    const label = partInfo(data, r.part)?.label ?? r.part;
    out.set(first, { part: r.part, value: said.confidence, level: confLevel(said.confidence), tip: confTip(label, said.confidence, said.evidence) });
  }
  return out;
}

/** 识别引擎找到了候选、却没分配的部位（置信度不够，或落在不驱动顶点的关节上）：给树头的一行，悬停看每个为什么。
 * 用户已手动配了的部位不算。 */
export function unsureParts(rec: RigRecognition | undefined, data: RigPairData, m: Merged, col: Col): { count: number; tip: string } {
  if (!rec || data[col].fixed) return { count: 0, tip: "" };
  const lines: string[] = [];
  for (const [part, said] of Object.entries(rec.parts)) {
    if (said.assigned !== false || (m.manual.has(part) && (m.rows[part]?.[col]?.length ?? 0) > 0)) continue;
    const label = partInfo(data, part)?.label ?? part;
    lines.push(t("ui.rig.unsure_line", { part: label, why: said.evidence[said.evidence.length - 1] ?? t("ui.rig.confidence", { value: said.confidence.toFixed(2) }) }));
  }
  return { count: lines.length, tip: lines.length ? [t("ui.rig.unsure_head"), ...lines].join("\n") : "" };
}

// ------------------------------------------------------------------ 层级

/** 每个关节的层级深度（根为 0）。 */
export function depths(parents: number[]): number[] {
  return parents.map((_, i) => {
    let d = 0;
    for (let j = parents[i] ?? -1; j >= 0 && d < 10000; j = parents[j] ?? -1) d++;
    return d;
  });
}

/** 关节 `i` 和它的全部子孙（序号，按序号排）。 */
export function subtree(parents: number[], i: number): number[] {
  const out: number[] = [];
  for (let k = 0; k < parents.length; k++) {
    let j = k;
    for (let n = 0; j >= 0 && j !== i && n < 10000; n++) j = parents[j] ?? -1;
    if (j === i) out.push(k);
  }
  return out;
}

/** 眼睛：`hidden` 里的关节和它们的子孙不画。返回每个关节画不画。 */
export function shownMask(side: Pick<RigPairSide, "names" | "parents">, hidden: readonly string[]): boolean[] {
  const roots = new Set(hidden.map((n) => side.names.indexOf(n)).filter((i) => i >= 0));
  return side.names.map((_, i) => {
    for (let j = i, n = 0; j >= 0 && n < 10000; j = side.parents[j] ?? -1, n++) if (roots.has(j)) return false;
    return true;
  });
}

/** 按层级从上到下放进一个链里（链的关节从根到梢）。 */
function byDepth(names: string[], side: RigPairSide): string[] {
  const d = depths(side.parents);
  const at = (n: string) => d[side.names.indexOf(n)] ?? 0;
  return [...names].sort((a, b) => at(a) - at(b));
}

// ------------------------------------------------------------------ 舞台上的摆放（只影响显示）

export type Vec3 = [number, number, number];

/** 一侧的锚点关节：髋部位配了就是它的第一个关节，否则第一个根关节。 */
export function anchorJoint(side: Pick<RigPairSide, "parents" | "parts">): number {
  const hips = side.parts?.hips?.[0];
  if (hips !== undefined && hips >= 0 && hips < side.parents.length) return hips;
  const root = side.parents.findIndex((p) => p < 0);
  return root >= 0 ? root : 0;
}

/** 关节 `i` 在基准姿势（不带修正）里的世界位置：基准局部（列主序 4x4）沿父子链累乘。 */
export function basePosition(side: Pick<RigPairSide, "parents" | "before">, i: number): Vec3 {
  const local = side.before?.local;
  if (!local || !local[i]) return [0, 0, 0];
  const chain: number[] = [];
  for (let j = i, n = 0; j >= 0 && n < 10000; j = side.parents[j] ?? -1, n++) chain.push(j);
  // 从根往下：p = M_root · M_1 · … · M_i · (0,0,0,1)，从叶子往上逐个左乘即可
  let p: number[] = [0, 0, 0, 1];
  for (const j of chain) {
    const m = local[j];
    p = [0, 1, 2, 3].map((r) => m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r] * p[3]);
  }
  return [p[0], p[1], p[2]];
}

/** 两侧在舞台上的平移（只影响显示，不写参数、不进计算）：每侧的锚点（anchorJoint）在基准姿势里的水平位置挪到世界原点
 * （减去 X、Z，Y 不动：脚还在地上），再沿世界 Z 拉开（源在后 −gap/2、目标在前 +gap/2）。动作捕捉的源常常站在离原点
 * 很远的场地上：不挪回来，两副骨架隔得很远，框显后都只剩一点。固定的模型一侧没有位置：不挪；有一侧固定时不拉开。
 * 用基准姿势而不是修正后的：拖髋时骨架要跟着动，不能被挪回原点。 */
export function sideOffsets(data: RigPairData, gap: number, scales?: Record<Col, number>): Record<Col, Vec3> {
  const placed = (c: Col) => !data[c].fixed && !!data[c].before;
  const g = placed("src") && placed("dst") ? gap : 0;
  const out = {} as Record<Col, Vec3>;
  for (const c of COLS) {
    if (!placed(c)) { out[c] = [0, 0, 0]; continue; }
    const [x, , z] = scaledPoint(basePosition(data[c], anchorJoint(data[c])), data[c], scales?.[c] ?? 1);
    out[c] = [-x, 0, (c === "src" ? -g / 2 : g / 2) - z];
  }
  return out;
}

/** 尺寸（「动作尺寸」「目标尺寸」，effective 的 scales）的显示：一侧按实际用的系数画（所见即所算）。服务端
 * （kit/retarget_prepare.py sizing_for）在世界里做的是 p ↦ s·(p − (0, ground, 0))：绕地面原点缩放、脚底落到 y = 0；
 * 系数 1 什么都不做。 */
export function scaledPoint(p: Vec3, side: Pick<RigPairSide, "size">, s: number): Vec3 {
  if (s === 1) return p;
  const g = side.size?.ground ?? 0;
  return [s * p[0], s * (p[1] - g), s * p[2]];
}

/** 一侧的世界矩阵（列主序 4x4）按系数变换：轴一起乘 s（蒙皮的网格也跟着缩放），位置同 scaledPoint。 */
export function scaledWorld(world: number[][], side: Pick<RigPairSide, "size">, s: number): number[][] {
  if (s === 1) return world;
  const g = side.size?.ground ?? 0;
  return world.map((m) => {
    const out = m.slice();
    for (let k = 0; k < 12; k++) if (k % 4 !== 3) out[k] = m[k] * s;
    out[12] = s * m[12]; out[13] = s * (m[13] - g); out[14] = s * m[14];
    return out;
  });
}

// ------------------------------------------------------------------ 骨点的状态与连线

export type JointState = "paired" | "unpaired" | "ignored";

/** 每个关节的状态：在忽略名单里 = 忽略；在某个部位那一栏里、且那个部位另一栏也有关节 = 已配对；否则未配对。 */
export function jointStates(m: Merged, data: RigPairData, col: Col, ignored: readonly string[]): JointState[] {
  const skip = new Set(ignored);
  const owner = partOf(m, col);
  return data[col].names.map((n) => {
    if (skip.has(n)) return "ignored";
    const part = owner.get(n);
    return part && m.rows[part]?.[other(col)].length ? "paired" : "unpaired";
  });
}

/** 链内的均分（服务端 data/joints.py spread 的同一规则，只用来画连线）：`count` 个被驱动关节按在链上的位置落到
 * 驱动一侧的 `joints` 上，首对首、尾对尾；驱动一侧更短时多出来的为 null。 */
export function spread<T>(count: number, joints: T[]): (T | null)[] {
  if (!joints.length) return Array.from({ length: count }, () => null);
  const picks = count === 1 ? [0] : Array.from({ length: count }, (_, k) => Math.round((k * (joints.length - 1)) / (count - 1)));
  const used = new Set<number>();
  return picks.map((i) => {
    const out = used.has(i) ? null : joints[i];
    used.add(i);
    return out;
  });
}

export interface Link {
  part: string;
  src: string;
  dst: string;
}

/** 配对连线：每个两栏都有关节的部位，被驱动一侧的每个关节连到驱动它的那个（链按 spread）。 */
export function linksOf(m: Merged, data: RigPairData): Link[] {
  const out: Link[] = [];
  for (const p of [...data.parts_table.map((q) => q.id), ...Object.keys(m.rows).filter((k) => !data.parts_table.some((q) => q.id === k))]) {
    const r = m.rows[p];
    if (!r || !r.src.length || !r.dst.length) continue;
    spread(r.dst.length, r.src).forEach((s, k) => { if (s) out.push({ part: p, src: s, dst: r.dst[k] }); });
  }
  return out;
}

/** 关节的配对对象（树里固定一侧不画 3D 时写在行上）：同一部位另一栏的关节。 */
export function partnersOf(m: Merged, col: Col, joint: string): string[] {
  const part = partOf(m, col).get(joint);
  return part ? m.rows[part]?.[other(col)] ?? [] : [];
}

// ------------------------------------------------------------------ 改对应关系

const partInfo = (data: RigPairData, id: string): RigPart | undefined => data.parts_table.find((p) => p.id === id);

/** 先点了 `first`、又点了另一侧的 `second`：配成哪个部位。固定一侧的关节决定部位（那一侧改不了）；否则取先点的那个
 * 关节现在（参数行 + 推测）所属的部位。null：先点的关节没有部位，要让使用者指定（部位小菜单）。 */
export function partFor(m: Merged, data: RigPairData, first: { col: Col; joint: string }, second: { col: Col; joint: string }): string | null {
  for (const at of [first, second]) if (data[at.col].fixed) return partOf(m, at.col).get(at.joint) ?? null;
  return partOf(m, first.col).get(first.joint) ?? null;
}

/** 把一个关节放进部位 `part` 的 `col` 那一栏（链按层级深度插入，非链就是它一个），并从这一栏的别的部位里拿掉它
 * （一个关节只能在一处）。动过的部位都算手动。固定的一侧不动。 */
function put(m: Merged, data: RigPairData, part: string, col: Col, joint: string): Merged {
  if (data[col].fixed) return m;
  const rows = { ...m.rows };
  const manual = new Set(m.manual);
  for (const [id, r] of Object.entries(rows)) {
    if (id !== part && r[col].includes(joint)) {
      rows[id] = { ...r, [col]: r[col].filter((n) => n !== joint) };
      manual.add(id);
    }
  }
  const row = rows[part] ?? { part, src: [], dst: [] };
  const chain = partInfo(data, part)?.chain ?? false;
  const next = chain ? byDepth([...row[col].filter((n) => n !== joint), joint], data[col]) : [joint];
  rows[part] = { ...row, [col]: next };
  manual.add(part);
  return { rows, manual };
}

/** 配对：两个关节都放进部位 `part`（各自那一栏）。 */
export function link(m: Merged, data: RigPairData, part: string, a: { col: Col; joint: string }, b: { col: Col; joint: string }): Merged {
  return put(put(m, data, part, a.col, a.joint), data, part, b.col, b.joint);
}

/** 解除一个已配对的关节：从它所属部位的那一栏去掉它；它在固定一侧时，去掉的是这个部位另一栏的全部关节（固定一栏改不了）。 */
export function unlinkJoint(m: Merged, data: RigPairData, col: Col, joint: string): Merged {
  const part = partOf(m, col).get(joint);
  if (!part) return m;
  const row = m.rows[part];
  const target: Col = data[col].fixed ? other(col) : col;
  if (data[target].fixed) return m;
  const next = data[col].fixed ? [] : row[col].filter((n) => n !== joint);
  return { rows: { ...m.rows, [part]: { ...row, [target]: next } }, manual: new Set(m.manual).add(part) };
}

/** 解除一条连线：两端都从这个部位的行里去掉（固定一侧那端不动）。 */
export function unlinkLink(m: Merged, data: RigPairData, l: Link): Merged {
  const row = m.rows[l.part];
  if (!row) return m;
  const next = { ...row };
  if (!data.src.fixed) next.src = row.src.filter((n) => n !== l.src);
  if (!data.dst.fixed) next.dst = row.dst.filter((n) => n !== l.dst);
  return { rows: { ...m.rows, [l.part]: next }, manual: new Set(m.manual).add(l.part) };
}

// ------------------------------------------------------------------ 显示用校验（服务端 check_mapping 里按行本身就能判的几条）

/** 每个部位的第一个问题（没问题的部位不在里面）：不存在的关节、非链部位多于一个关节、链不连成一串、一个关节在两个
 * 部位里、必需部位缺一侧。权威在服务端（计算时 data/joints.py check_mapping 报正式消息）；这里只为摘要行与树上的提示。 */
export function problems(m: Merged, data: RigPairData): Record<string, string> {
  const out: Record<string, string> = {};
  const say = (part: string, text: string) => (out[part] ??= text);
  const label = (id: string) => partInfo(data, id)?.label ?? id;
  const word: Record<Col, string> = { src: t("ui.rig.side_src"), dst: t("ui.rig.side_dst") };
  for (const col of COLS) {
    const side = data[col];
    if (side.fixed) continue;
    const index = new Map(side.names.map((n, i) => [n, i] as [string, number]));
    const taken = new Map<string, string>();
    for (const p of data.parts_table) {
      const joints = m.rows[p.id]?.[col] ?? [];
      for (const n of joints) if (!index.has(n)) say(p.id, t("ui.rig.problem_absent", { joint: n, side: word[col] }));
      if (!p.chain && joints.length > 1) say(p.id, t("ui.rig.problem_one", { side: word[col], count: joints.length }));
      for (let k = 1; k < joints.length; k++) {
        const a = index.get(joints[k - 1]), b = index.get(joints[k]);
        if (a !== undefined && b !== undefined && !subtree(side.parents, a).includes(b))
          say(p.id, t("ui.rig.problem_chain", { joint: joints[k], above: joints[k - 1], side: word[col] }));
      }
      for (const n of joints) {
        const was = taken.get(n);
        if (was && was !== p.id) {
          say(p.id, t("ui.rig.problem_twice", { side: word[col], joint: n, part: label(was) }));
          say(was, t("ui.rig.problem_twice", { side: word[col], joint: n, part: label(p.id) }));
        } else taken.set(n, p.id);
      }
    }
  }
  for (const p of data.parts_table) {
    if (!p.required) continue;
    const lack = COLS.filter((c) => !m.rows[p.id]?.[c]?.length).map((c) => word[c]);
    if (lack.length) say(p.id, t("ui.rig.problem_required", { sides: lack.join(listSep()) }));
  }
  return out;
}

/** 按选中的忽略规则（解算器的固定骨架，RigPairRule slots / required）检查对应关系：去掉忽略的关节之后，必需的部位
 * 两侧都要有、每个部位的节数不多于模型的。和解算器计算时的检查（nodes/kit/retarget_needs.py check）同一套数，
 * 只是在编辑时就说，而不是等算到解算器才报错。模型固定骨架的一侧（fixed）不查。 */
export function ruleProblems(m: Merged, data: RigPairData, ruleId: string, ignored: Record<Col, string[]>): string[] {
  const rule = (data.rules ?? []).find((r) => r.id === ruleId);
  if (!rule?.slots) return [];
  const label = (id: string) => partInfo(data, id)?.label ?? id;
  const word: Record<Col, string> = { src: t("ui.rig.side_src"), dst: t("ui.rig.side_dst") };
  const out: string[] = [];
  for (const col of COLS) {
    if (data[col].fixed) continue;
    const skip = new Set(ignored[col]);
    const sent = (part: string) => (m.rows[part]?.[col] ?? []).filter((n) => !skip.has(n));
    const missing = (rule.required ?? []).filter((part) => !sent(part).length).map(label);
    if (missing.length) out.push(t("ui.rig.rule_missing", { side: word[col], parts: missing.join(listSep()) }));
    for (const [part, want] of Object.entries(rule.slots)) {
      const got = sent(part).length;
      if (got > want) out.push(t("ui.rig.rule_too_many", { side: word[col], part: label(part), got, want }));
    }
  }
  return out;
}

/** 参数面板的摘要行：「手动 N 个部位，其余自动 · 有问题：…」（没有手柄数据时只有前半句）。 */
export function mappingSummary(value: unknown, data: RigPairData | null): { text: string; bad: boolean } {
  const merged = data ? mergedOf(value, data) : null;
  const manual = merged ? merged.manual.size : mapRowsOf(value).length;
  const how = manual ? t("ui.rig.summary_manual", { count: manual }) : t("ui.rig.summary_auto");
  if (!data || !merged) return { text: how, bad: false };
  const stale = merged.stale?.length ? t("ui.rig.summary_stale", { count: merged.stale.length }) : "";
  const bad = problems(merged, data);
  const wrong = data.parts_table.filter((q) => bad[q.id]).map((q) => q.label);
  return { text: (wrong.length ? t(wrong.length > 4 ? "ui.rig.summary_wrong_more" : "ui.rig.summary_wrong", { how, parts: wrong.slice(0, 4).join(listSep()) }) : how) + stale, bad: wrong.length > 0 };
}

// ------------------------------------------------------------------ 忽略

/** 参数值读成关节名单。 */
export const namesOf = (value: unknown): string[] => (Array.isArray(value) ? value.filter((n): n is string => typeof n === "string") : []);

/** 切换忽略：`single` 只切换这一个关节（Alt 点）；否则连同它的子孙一起，按这个关节现在是否被忽略决定全加或全去。
 * 结果按关节顺序排，不认得的名字（骨架里没有）原样留着。 */
export function toggleIgnored(list: readonly string[], side: Pick<RigPairSide, "names" | "parents">, joint: string, single: boolean): string[] {
  const i = side.names.indexOf(joint);
  if (i < 0) return [...list];
  const on = list.includes(joint);
  const touched = new Set((single ? [i] : subtree(side.parents, i)).map((k) => side.names[k]));
  const keep = new Set(list.filter((n) => !touched.has(n)));
  if (!on) for (const n of touched) keep.add(n);
  const order = new Map(side.names.map((n, k) => [n, k] as [string, number]));
  return [...keep].sort((a, b) => (order.get(a) ?? Infinity) - (order.get(b) ?? Infinity));
}

// ------------------------------------------------------------------ 三个「自动」与状态行

export interface PoseRowLike {
  joint: string;
  translate: number[];
  rotate: number[];
  scale: number[];
}

/** auto_record 参数（JSON 文字）：几个「自动」最近一次各写了什么（尺寸：两侧的系数）。姿态与忽略按节点的两个输入记（motion / target，
 * 即 src / dst）。`manual`：这一项不是点「自动」写的，而是第一次手动改姿态 / 忽略时记下的当时的默认自动结果
 * （withTouched）：有了这一项，服务端与页面就不再按「默认自动」补，空着就是空。 */
export interface AutoRecord {
  mapping?: RigRow[];
  pose?: { motion?: PoseRowLike[]; target?: PoseRowLike[]; manual?: boolean };
  ignore?: { rule: string; motion?: string[]; target?: string[]; manual?: boolean };
  size?: { motion?: number | null; target?: number | null };
}
export type AutoKind = "mapping" | "pose" | "ignore" | "size";
export const RECORD_SIDE: Record<Col, "motion" | "target"> = { src: "motion", dst: "target" };

/** auto_record 的值读成记录：空、读不出、不是对象时为 null（= 都还没执行）。 */
export function recordOf(value: unknown): AutoRecord | null {
  let v = value;
  if (typeof v === "string") {
    if (!v.trim()) return null;
    try { v = JSON.parse(v); } catch { return null; }
  }
  return v && typeof v === "object" && !Array.isArray(v) ? (v as AutoRecord) : null;
}

const poseRowsOf = (value: unknown): PoseRowLike[] =>
  Array.isArray(value) ? value.filter((r): r is PoseRowLike => !!r && typeof r === "object" && typeof (r as PoseRowLike).joint === "string") : [];

/** 现在的参数按记录的形状读出来（与 auto_record 逐项比）。 */
export function currentRecord(roles: RigRoles, params: Record<string, unknown> | undefined): AutoRecord {
  const p = params ?? {};
  const out: AutoRecord = { mapping: roles.mapping ? mapRowsOf(p[roles.mapping]) : [] };
  out.pose = {};
  out.ignore = { rule: roles.ignore_rule ? String(p[roles.ignore_rule] ?? "") : "" };
  for (const c of COLS) {
    const pose = roles[poseRole(c)], ignore = roles[ignoreRole(c)];
    if (pose) out.pose[RECORD_SIDE[c]] = poseRowsOf(p[pose]);
    if (ignore) out.ignore[RECORD_SIDE[c]] = namesOf(p[ignore]);
  }
  out.size = {};
  for (const c of COLS) { const r = roles[scaleRole(c)]; if (r) out.size[RECORD_SIDE[c]] = scaleOf(p[r]); }
  return out;
}

/** 尺寸参数的值：正数是填了的系数，其余（空、0、不是数）是 null = 自动。 */
export const scaleOf = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null);

const EPS = 1e-4;
const sameNums = (a: number[] | undefined, b: number[] | undefined, d: number) =>
  [0, 1, 2].every((k) => Math.abs((a?.[k] ?? d) - (b?.[k] ?? d)) < EPS);
const sameNames = (a: string[] | undefined, b: string[] | undefined) =>
  (a ?? []).length === (b ?? []).length && (a ?? []).every((n, k) => n === (b ?? [])[k]);

/** 两份记录之间不同的个数：对应关系按部位数行，姿态按关节数（两侧合计；没有一行等于没改），忽略按关节数（两侧名单的
 * 对称差合计）。 */
export function differences(kind: AutoKind, a: AutoRecord, b: AutoRecord): number {
  if (kind === "mapping") {
    const by = (rows: RigRow[] | undefined) => new Map((rows ?? []).map((r) => [r.part, r] as [string, RigRow]));
    const x = by(a.mapping), y = by(b.mapping);
    let n = 0;
    for (const part of new Set([...x.keys(), ...y.keys()])) {
      const r = x.get(part), s = y.get(part);
      if (!r || !s || !sameNames(r.src, s.src) || !sameNames(r.dst, s.dst)) n++;
    }
    return n;
  }
  if (kind === "pose") {
    let n = 0;
    for (const side of ["motion", "target"] as const) {
      const by = (rows: PoseRowLike[] | undefined) => new Map((rows ?? []).map((r) => [r.joint, r] as [string, PoseRowLike]));
      const x = by(a.pose?.[side]), y = by(b.pose?.[side]);
      for (const joint of new Set([...x.keys(), ...y.keys()])) {
        const r = x.get(joint), s = y.get(joint);
        if (!sameNums(r?.translate, s?.translate, 0) || !sameNums(r?.rotate, s?.rotate, 0) || !sameNums(r?.scale, s?.scale, 1)) n++;
      }
    }
    return n;
  }
  if (kind === "size") {
    let n = 0;
    for (const side of ["motion", "target"] as const) {
      const x = a.size?.[side] ?? null, y = b.size?.[side] ?? null;
      if (x === null || y === null ? x !== y : Math.abs(x - y) >= EPS) n++;
    }
    return n;
  }
  let n = 0;
  for (const side of ["motion", "target"] as const) {
    const x = new Set(a.ignore?.[side] ?? []), y = new Set(b.ignore?.[side] ?? []);
    for (const j of x) if (!y.has(j)) n++;
    for (const j of y) if (!x.has(j)) n++;
  }
  return n;
}

export type AutoTone = "idle" | "auto" | "done" | "edited" | "stale";

/** 功能条上每个「自动」旁的状态行：记录里没有这一项 → 「还没执行」；（忽略）现在的规则不是记录里那条 → 「规则已换成 X，还没按它
 * 执行」；现在的参数与记录相同 → 「已自动」；否则数出不同的行 / 关节 → 「已自动 · 之后手动改了 N 个」。`rule`：现在选的
 * 忽略规则（id 与它的名字），只对 ignore 有意义。 */
/** 服务端这一次给的建议，按记录的形状（与 auto_record 比：记录是点「自动」那一刻的建议，建议后来变了——换了骨架、
 * 改了对应关系让部位变了、软件更新改了自动的算法（如脚骨只转向正前）——状态行要说出来，而不是一直写「已自动」）。
 * 忽略不在这里：它的建议随规则走，有自己的「规则已换」。 */
export function suggestedRecord(data: RigPairData, roles: RigRoles): AutoRecord {
  const out: AutoRecord = { mapping: data.auto_mapping.map(copy), pose: {}, size: {} };
  for (const c of COLS) {
    if (data[c].fixed) continue;
    if (roles[poseRole(c)] && data[c].auto_pose) out.pose![RECORD_SIDE[c]] = data[c].auto_pose!;
    if (roles[scaleRole(c)] && data[c].size) out.size![RECORD_SIDE[c]] = data[c].size!.auto;
  }
  return out;
}

const AUTO_KEY: Record<AutoKind, string> = { mapping: "ui.rig.auto_mapping", pose: "ui.rig.auto_pose", ignore: "ui.rig.auto_ignore", size: "ui.rig.auto_size" };
/** 「自动对应」等按钮的名字（当前语言）。 */
export const autoName = (kind: AutoKind): string => t(AUTO_KEY[kind]);

export function autoState(kind: AutoKind, record: AutoRecord | null, current: AutoRecord, rule?: { id: string; label: string },
  suggested?: AutoRecord): { text: string; tone: AutoTone } {
  const done = record?.[kind];
  if (!record || !done) {
    // 默认自动（effective）：从没执行过、参数也空着的姿态 / 忽略，计算按服务端的建议走
    const empty = kind === "pose" ? (["motion", "target"] as const).every((k) => !current.pose?.[k]?.length)
      : kind === "ignore" ? (["motion", "target"] as const).every((k) => !current.ignore?.[k]?.length)
        : kind === "size" ? (["motion", "target"] as const).every((k) => (current.size?.[k] ?? null) === null) : false;
    return empty ? { text: t("ui.rig.state_auto_default"), tone: "auto" } : { text: t("ui.rig.state_not_run"), tone: "idle" };
  }
  const manual = (kind === "pose" || kind === "ignore") && !!(done as { manual?: boolean }).manual;
  if (kind === "ignore" && rule && !manual && rule.id !== record.ignore?.rule) return { text: t("ui.rig.state_rule_changed", { rule: rule.label }), tone: "stale" };
  const n = differences(kind, record, current);
  // 记录是第一次手动改时记下的：没点过「自动」，只说改了几个（与当时的默认自动比）
  if (manual) return { text: n ? t("ui.rig.state_manual_n", { count: n }) : t("ui.rig.state_manual"), tone: "edited" };
  // 点「自动」之后建议变了：参数还是旧的那一份（计算照它算），提示重新执行；只比记录里有的那几侧
  if (suggested && kind !== "ignore") {
    const sides = kind === "mapping" ? null : Object.keys((done as object) ?? {}).filter((k) => k === "motion" || k === "target");
    const pick = (r: AutoRecord): AutoRecord => (sides === null ? r : { [kind]: Object.fromEntries(sides.map((k) => [k, (r[kind] as Record<string, unknown> | undefined)?.[k]])) });
    const moved = differences(kind, pick(record), pick(suggested));
    if (moved) return { text: t("ui.rig.state_suggestion_moved", { count: moved, button: autoName(kind) }) + (n ? t("ui.rig.state_then_manual", { count: n }) : ""), tone: "stale" };
  }
  return n ? { text: t("ui.rig.state_done") + t("ui.rig.state_then_manual", { count: n }), tone: "edited" } : { text: t("ui.rig.state_done"), tone: "done" };
}

/** 姿态、忽略实际用的那一份（「默认自动」：服务端计算时同一条规则，页面画的、改的都按它，从不画出与计算不同的姿态）：
 * - 姿态，逐侧：auto_record 里没有 pose、且这一侧的行是空的 → 这一侧的建议行（side.auto_pose）；否则就是参数里的行；
 * - 忽略：auto_record 里没有 ignore、且两侧名单都空 → 现在选的规则的建议名单（rules[ignore_rule].ignore）；否则就是参数里的名单。
 * 对应关系不在这里：空着本来就是「全部按推测」（mergedOf）。编辑从这一份开始：第一次手动改写下的是这一份加上那一处改动，
 * 动一个关节不会丢掉自动的结果。`auto`：哪些是按默认自动得来的（还没写进参数）。 */
export interface Effective {
  poses: Record<Col, PoseRowLike[]>;
  ignored: Record<Col, string[]>;
  scales: Record<Col, number>; // 尺寸：填了的系数；空着 = 自动（side.size.auto，服务端 effective_scale 同一规则，不看记录）
  auto: { pose: Record<Col, boolean>; ignore: boolean; size: Record<Col, boolean> };
}

export function effective(data: RigPairData, roles: RigRoles, params: Record<string, unknown> | undefined): Effective {
  const p = params ?? {};
  const record = roles.auto_record ? recordOf(p[roles.auto_record]) : null;
  const out: Effective = {
    poses: { src: [], dst: [] }, ignored: { src: [], dst: [] }, scales: { src: 1, dst: 1 },
    auto: { pose: { src: false, dst: false }, ignore: false, size: { src: false, dst: false } },
  };
  for (const c of COLS) {
    const role = roles[scaleRole(c)];
    if (!role || data[c].fixed) continue;
    const v = scaleOf(p[role]);
    out.scales[c] = v ?? data[c].size?.auto ?? 1;
    out.auto.size[c] = v === null;
  }
  for (const c of COLS) {
    const role = roles[poseRole(c)];
    if (!role) continue;
    const rows = poseRowsOf(p[role]);
    const auto = !record?.pose && !rows.length && !data[c].fixed && !!data[c].auto_pose?.length;
    out.poses[c] = auto ? data[c].auto_pose!.map((r) => ({ ...r })) : rows;
    out.auto.pose[c] = auto;
  }
  for (const c of COLS) { const role = roles[ignoreRole(c)]; if (role) out.ignored[c] = namesOf(p[role]); }
  const roleSides = COLS.filter((c) => roles[ignoreRole(c)] && !data[c].fixed);
  if (!record?.ignore && roleSides.length && roleSides.every((c) => !out.ignored[c].length)) {
    const rule = (data.rules ?? []).find((r) => r.id === ruleNow(data, roles.ignore_rule ? p[roles.ignore_rule] : undefined));
    if (rule) {
      for (const c of roleSides) out.ignored[c] = [...(rule.ignore[c] ?? [])];
      out.auto.ignore = true;
    }
  }
  return out;
}

/** 手动写姿态 / 忽略时（拖手柄、填数、复位、清空、点骨点切换忽略）补上的东西：auto_record 里还没有这一项时，记下
 * 写之前的默认自动结果（effective，形状同点「自动」写的，带 manual），并把同一项里这次没写到的另一侧按 effective
 * 写实（否则记录一出现，那一侧的默认自动就没了）。这样「清空」之后就是空的，不会回到自动；状态行据此写「手动改过」。
 * `data` 为 null（手柄数据不在手）时补不了另一侧的自动结果，返回 null：调用方不写（说明要先打开视图编辑）。 */
export function withTouched(patch: Record<string, unknown>, data: RigPairData | null, roles: RigRoles, params: Record<string, unknown> | undefined): Record<string, unknown> | null {
  if (!roles.auto_record) return patch;
  const record: AutoRecord = { ...(recordOf(params?.[roles.auto_record]) ?? {}) };
  const touches = (kind: "pose" | "ignore") => COLS.some((c) => { const r = roles[kind === "pose" ? poseRole(c) : ignoreRole(c)]; return !!r && r in patch; });
  const poseNow = touches("pose") && !record.pose, ignoreNow = touches("ignore") && !record.ignore;
  if (!poseNow && !ignoreNow) return patch;
  if (!data) return null;
  const eff = effective(data, roles, params);
  const out = { ...patch };
  if (poseNow) {
    const pose: NonNullable<AutoRecord["pose"]> = { manual: true };
    for (const c of COLS) {
      const role = roles[poseRole(c)];
      if (!role) continue;
      pose[RECORD_SIDE[c]] = eff.poses[c].map((r) => ({ ...r }));
      if (!(role in out)) out[role] = eff.poses[c].map((r) => ({ ...r }));
    }
    record.pose = pose;
  }
  if (ignoreNow) {
    const ignore: NonNullable<AutoRecord["ignore"]> = { rule: ruleNow(data, roles.ignore_rule ? params?.[roles.ignore_rule] : undefined), manual: true };
    for (const c of COLS) {
      const role = roles[ignoreRole(c)];
      if (!role) continue;
      ignore[RECORD_SIDE[c]] = [...eff.ignored[c]];
      if (!(role in out)) out[role] = [...eff.ignored[c]];
    }
    record.ignore = ignore;
  }
  out[roles.auto_record] = JSON.stringify(record);
  return out;
}

/** 连线时的规则预选：要接上的节点在目录里声明了它接受的忽略规则（`retarget_rules`，第一条是它的默认），而这个节点的
 * 「忽略规则」还是默认值（或空着）时，返回要写的那条；不需要写时为 null。 */
export function wiredRule(accepts: readonly string[] | undefined, current: unknown, dflt: unknown): string | null {
  const first = accepts?.[0];
  if (!first) return null;
  const now = typeof current === "string" ? current : "";
  if (now && now !== dflt) return null;
  return first !== now ? first : null;
}

/** 现在选的忽略规则：参数里的；参数空着时是服务端给的预选（default_rule：节点接着的解算器的规则，否则通用），
 * 再没有就是第一条。 */
export function ruleNow(data: RigPairData, value: unknown): string {
  const rules = data.rules ?? [];
  const v = typeof value === "string" ? value : "";
  if (v) return v;
  if (data.default_rule && rules.some((r) => r.id === data.default_rule)) return data.default_rule;
  return rules[0]?.id ?? "";
}

/** 点一个「自动」要写的参数（一步撤销）：服务端给的建议原样写进参数，auto_record 里同一项换成这次写的。
 * 节点没有这一项的角色、或服务端没给建议时为 null（按钮置灰）。 */
/** 「全部自动」：几项「自动」依次叠成一份补丁（一次写入、一步撤销）。每一项都在前一项写过之后的参数上算，
 * 所以共用的 auto_record 记着全部执行过的项，不会被后一项盖掉。 */
export function autoAllPatch(kinds: AutoKind[], data: RigPairData, roles: RigRoles, params: Record<string, unknown> | undefined, rule = ""): Record<string, unknown> | null {
  let now: Record<string, unknown> = { ...(params ?? {}) };
  let all: Record<string, unknown> | null = null;
  for (const kind of kinds) {
    const patch = autoPatch(kind, data, roles, now, rule);
    if (!patch) continue;
    all = { ...(all ?? {}), ...patch };
    now = { ...now, ...patch };
  }
  return all;
}

export function autoPatch(kind: AutoKind, data: RigPairData, roles: RigRoles, params: Record<string, unknown> | undefined, rule = ""): Record<string, unknown> | null {
  const record: AutoRecord = { ...(roles.auto_record ? recordOf(params?.[roles.auto_record]) ?? {} : {}) };
  const patch: Record<string, unknown> = {};
  if (kind === "mapping") {
    if (!roles.mapping) return null;
    const rows = data.auto_mapping.map(copy);
    patch[roles.mapping] = rows.length ? rows : null;
    record.mapping = rows.map(copy);
  } else if (kind === "pose") {
    const pose: NonNullable<AutoRecord["pose"]> = {};
    for (const c of COLS) {
      const role = roles[poseRole(c)];
      const rows = data[c].auto_pose;
      if (!role || data[c].fixed || !rows) continue;
      patch[role] = rows.map((r) => ({ ...r }));
      pose[RECORD_SIDE[c]] = rows.map((r) => ({ ...r }));
    }
    if (!Object.keys(patch).length) return null;
    record.pose = pose;
  } else if (kind === "size") {
    const size: NonNullable<AutoRecord["size"]> = {};
    for (const c of COLS) {
      const role = roles[scaleRole(c)];
      const auto = data[c].size?.auto;
      if (!role || data[c].fixed || auto === undefined) continue;
      patch[role] = auto;
      size[RECORD_SIDE[c]] = auto;
    }
    if (!Object.keys(patch).length) return null;
    record.size = size;
  } else {
    const found = (data.rules ?? []).find((r) => r.id === rule);
    if (!found) return null;
    const ignore: NonNullable<AutoRecord["ignore"]> = { rule };
    for (const c of COLS) {
      const role = roles[ignoreRole(c)];
      if (!role || data[c].fixed) continue;
      patch[role] = [...(found.ignore[c] ?? [])];
      ignore[RECORD_SIDE[c]] = [...(found.ignore[c] ?? [])];
    }
    if (!Object.keys(patch).length) return null;
    if (roles.ignore_rule) patch[roles.ignore_rule] = rule;
    record.ignore = ignore;
  }
  if (roles.auto_record) patch[roles.auto_record] = JSON.stringify(record);
  return patch;
}
