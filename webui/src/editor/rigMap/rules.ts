/** 「对应关系」编辑器的草稿和它的显示用校验。
 *
 * 草稿 = 每个部位现在两栏各是哪些关节，以及哪些部位是手动定的（其余「自动」：每次计算按关节名和层级重新推测，
 * 服务端 lab2shot/data/joints.py guess()）。写回参数时只写手动的部位，一个也没有就写空（= 全部自动）。
 *
 * 槽标成错误色（紫）的规则是服务端校验里按草稿本身就能判的那几条：lab2shot/data/joints.py check_mapping 的不存在的关节、
 * 非链部位多于一个关节、链不连成一串、一个关节在两个部位里，以及 nodes/kit/retarget.py resolve_mapping 与
 * nodes/kit/rig.py RigModel.mapping 的必需部位。check_mapping 另外两条这里不判：E-MAP-HIPS（髋下面要有两条大腿与躯干，
 * 按 lab2shot_shared/motion.py hips_joint 的层级规则）与 E-MAP-DUPNAME（一个名字几个关节共用）——照抄一份就是同一条规则
 * 的第二种实现，所以只由服务端判：槽照常显示，计算时报正式消息。**权威在服务端**：这里只按 NodeDef.choices 给的数据做显示；
 * 页面自身不对数据做判断（editor/ParamControls.tsx 的原则）以外的只有这一处，因为每点一下都问服务端代价太大。 */

import type { RigChoice, RigPart, RigRow, RigSide } from "../../api";

export type Col = "src" | "dst";
export const COLS: Col[] = ["src", "dst"];

export interface Draft {
  rows: Record<string, RigRow>;
  manual: ReadonlySet<string>;
}

const copy = (r: RigRow): RigRow => ({ part: r.part, src: [...(r.src ?? [])], dst: [...(r.dst ?? [])] });

/** 模型类节点的模型一侧是固定的：dst 总是模型这个部位的关节。 */
function fixDst(row: RigRow, rig: RigChoice): RigRow {
  return rig.dst.fixed ? { ...row, dst: [...(rig.dst.parts?.[row.part] ?? [])] } : row;
}

/** 某个部位「自动」时是什么（推测结果；推测不到就是空的两栏）。 */
export function autoRow(part: string, rig: RigChoice): RigRow {
  const found = rig.auto.find((r) => r.part === part);
  return fixDst(found ? copy(found) : { part, src: [], dst: [] }, rig);
}

/** 参数值 → 草稿：参数里写了的部位是手动的，其余用推测。 */
export function draftOf(value: unknown, rig: RigChoice): Draft {
  const rows: Record<string, RigRow> = {};
  for (const p of rig.parts) rows[p.id] = autoRow(p.id, rig);
  const manual = new Set<string>();
  for (const r of Array.isArray(value) ? (value as RigRow[]) : []) {
    if (!r || typeof r.part !== "string") continue;
    rows[r.part] = fixDst(copy(r), rig);
    manual.add(r.part);
  }
  return { rows, manual };
}

/** 草稿 → 参数值：手动的部位按部位表的顺序；一个也没有 = null（全部自动）。 */
export function valueOf(d: Draft, rig: RigChoice): RigRow[] | null {
  const order = [...rig.parts.map((p) => p.id), ...Object.keys(d.rows).filter((k) => !rig.parts.some((p) => p.id === k))];
  const out = order.filter((p) => d.manual.has(p) && d.rows[p]).map((p) => copy(d.rows[p]));
  return out.length ? out : null;
}

export interface Tree {
  index: Map<string, number>;
  depth: number[];
}

export function treeOf(side: RigSide): Tree {
  const index = new Map(side.names.map((n, i) => [n, i] as [string, number]));
  const parents = side.parents ?? [];
  const depth = side.names.map((_, i) => {
    let d = 0;
    for (let j = parents[i] ?? -1; j >= 0 && d < 10000; j = parents[j] ?? -1) d++;
    return d;
  });
  return { index, depth };
}

/** 关节 `j` 在 `top` 下面（不是它自己）。 */
export function below(j: number, top: number, parents: number[]): boolean {
  for (let k = parents[j] ?? -1, n = 0; k >= 0 && n < 10000; k = parents[k] ?? -1, n++) if (k === top) return true;
  return false;
}

/** 两个关节之间的整段父子路径（含两端，从上到下）；不在一条线上时为 null。Shift+点选链用它
 * （Houdini KineFX 的 Auto Map Inline Points：只给首尾，中间自动补）。 */
export function pathBetween(a: string, b: string, side: RigSide, tree: Tree): string[] | null {
  const ia = tree.index.get(a), ib = tree.index.get(b);
  const parents = side.parents ?? [];
  if (ia === undefined || ib === undefined) return null;
  if (ia === ib) return [a];
  const [top, bottom] = below(ib, ia, parents) ? [ia, ib] : below(ia, ib, parents) ? [ib, ia] : [-1, -1];
  if (top < 0) return null;
  const out: number[] = [];
  for (let k = bottom; k !== top && k >= 0; k = parents[k] ?? -1) out.push(k);
  out.push(top);
  return out.reverse().map((i) => side.names[i]);
}

/** 按层级从上到下排（Ctrl+点加进来的关节）。 */
export const byDepth = (names: string[], tree: Tree) =>
  [...names].sort((x, y) => (tree.depth[tree.index.get(x) ?? 0] ?? 0) - (tree.depth[tree.index.get(y) ?? 0] ?? 0));

/** 每个部位的第一个问题（没有问题的部位不在里面）。规则见文件头，与服务端逐条对应。 */
export function problems(d: Draft, rig: RigChoice): Record<string, string> {
  const out: Record<string, string> = {};
  const say = (part: string, text: string) => (out[part] ??= text);
  const label = (id: string) => rig.parts.find((p) => p.id === id)?.label ?? id;
  for (const col of COLS) {
    const side = rig[col];
    if (side.fixed) continue; // 模型一侧由节点固定
    const tree = treeOf(side);
    const where = `${side.label}${side.noun ?? "骨架"}`; // 动作骨架 / 表情曲线 / 目标形变
    const parents = side.parents ?? [];
    const taken = new Map<string, string>();
    for (const p of rig.parts) {
      const joints = d.rows[p.id]?.[col] ?? [];
      for (const n of joints) if (!tree.index.has(n)) say(p.id, `${n} 在${where}里没有（换过${side.noun ?? "骨架"}？）`);
      if (!p.chain && joints.length > 1) say(p.id, `只能配一个，${side.label}这一栏选了 ${joints.length} 个`);
      for (let k = 1; k < joints.length; k++) {
        const a = tree.index.get(joints[k - 1]), b = tree.index.get(joints[k]);
        if (a !== undefined && b !== undefined && !below(b, a, parents))
          say(p.id, `${joints[k]} 不在 ${joints[k - 1]} 下面（${side.label}骨架）：一个部位里的关节要从上到下连成一串`);
      }
      for (const n of joints) {
        const other = taken.get(n);
        if (other && other !== p.id) {
          say(p.id, `${where}的 ${n} 也配在「${label(other)}」上：一个只能配一处`);
          say(other, `${where}的 ${n} 也配在「${label(p.id)}」上：一个只能配一处`);
        } else taken.set(n, p.id);
      }
    }
  }
  for (const p of rig.parts) {
    const r = d.rows[p.id];
    if (!p.required) continue;
    const lack = COLS.filter((c) => !(r?.[c]?.length)).map((c) => rig[c].label);
    if (lack.length) say(p.id, `必需的部位：${lack.join("、")}这边还没配`);
  }
  return out;
}

export type State = "ok" | "half" | "none" | "bad";

/** 槽的颜色（HumanIK 的约定）：绿 = 两栏都配了；黄 = 只有一栏（另一边没有这个部位，不参与、不算错）；
 * 灰 = 两边都没有；紫（项目的错误色，tokens.css --error）= 有问题。 */
export function stateOf(part: RigPart, d: Draft, bad: Record<string, string>): State {
  if (bad[part.id]) return "bad";
  const r = d.rows[part.id];
  const n = (r?.src.length ? 1 : 0) + (r?.dst.length ? 1 : 0);
  return n === 2 ? "ok" : n === 1 ? "half" : "none";
}

/** 每一侧：关节名 → 它现在属于哪个部位（树上、3D 里标部位用）。 */
export function partOf(d: Draft, col: Col): Map<string, string> {
  const out = new Map<string, string>();
  for (const r of Object.values(d.rows)) for (const n of r[col]) if (!out.has(n)) out.set(n, r.part);
  return out;
}
