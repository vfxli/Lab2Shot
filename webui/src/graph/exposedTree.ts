/** 参数界面树（节点图的 exposed，api/catalog.ts ExposedEntry）的挪动：「编辑参数界面」弹窗（editor/ParamInterfaceEditor.tsx）
 * 和参数面板里直接拖动（editor/ParamPanel.tsx）共用这一套。都返回新的树，不改原来的。路径 Path 是一层层的下标。 */

import type { ExposedEntry, ExposedParam, NodeTypeDef } from "../api";
import { conditionHolds, renamedInCondition, settledMenus } from "../platform/conditions";
import { exposedParams, firstTarget, isGroup, targetsOf } from "../state/cookInputs";

/** 每个公开参数现在的值（按对外名字；节点没写的取节点类型的默认值，目标不在了为 null）——条件表达式里的名字取的就是它，
 * 只在这里算（参数面板、下拉的落位、服务端 engine/templates.py exposed_values 同一口径）。按钮没有值，不影响。 */
export function exposedValues(tree: ExposedEntry[], nodeOf: (id: string) => { typeId: string; params: Record<string, unknown> } | undefined,
                              defs: Record<string, NodeTypeDef>): Record<string, unknown> {
  const values: Record<string, unknown> = {};
  for (const x of exposedParams(tree)) {
    const [nid, pname] = firstTarget(x); // an entry driving several holds one value (graph/edit.ts writes them all)
    const n = nodeOf(nid);
    values[x.name] = n ? n.params[pname] ?? defs[n.typeId]?.defaults[pname] ?? null : null;
  }
  return values;
}

/** 一个公开下拉的当前值被它自己那一项的 Hide When 藏起、要落到第一个列出的选项上。 */
export interface HiddenChoice {
  x: ExposedParam;
  was: unknown;
  now: unknown;
}

/** 整棵树里当前值被藏起的下拉（该落到哪儿）：规则只在 platform/conditions.ts settledMenus（服务端 conditions.py settled
 * 同一规则，命令行 / 插件设值后也照它落位）。条件一变就问它：graph/document.ts 在使用者的修改之后把它们作为同一步写回，
 * 打开文档时写回并提示。 */
export function hiddenChoices(tree: ExposedEntry[], values: Record<string, unknown>): HiddenChoice[] {
  const menus = exposedParams(tree).filter((x) => x.widget === "menu" && x.options?.length);
  const of = new Map(menus.map((x) => [x.name, x]));
  return settledMenus(menus.map((x) => ({ name: x.name, options: x.options ?? [] })), values)
    .map(([name, was, now]) => ({ x: of.get(name)!, was, now }));
}

/** 参数界面里一项该不该画：参数项 Hide When 成立就不画；组里有东西、但全都不画时组也不画。空组照画（刚新建的组要看得见，
 * 面板里才能把参数拖进去；模板作者建了空组就是他要的样子）。`values`：每个公开参数的
 * 对外名字 → 它现在的值（ParamPanel.tsx InterfaceTree 按 target 从节点参数取，没写的取节点类型的默认值）。条件没写、
 * 写错、用到不存在的名字都算不成立（platform/conditions.ts conditionHolds）：照常画。 */
export function entryHidden(x: ExposedEntry, values: Record<string, unknown>): boolean {
  return isGroup(x) ? x.children.length > 0 && x.children.every((c) => entryHidden(c, values)) : conditionHolds(x.hide_when, values);
}

/** 参数项该不该置灰：Disable When 成立（规则同上）。 */
export function entryDisabled(x: ExposedEntry, values: Record<string, unknown>): boolean {
  return !isGroup(x) && conditionHolds(x.disable_when, values);
}

export type Path = number[];

export function at(tree: ExposedEntry[], path: Path): ExposedEntry | undefined {
  let list = tree;
  let x: ExposedEntry | undefined;
  for (const i of path) {
    x = list[i];
    if (!x) return undefined;
    list = isGroup(x) ? x.children : [];
  }
  return x;
}

/** 把 `path` 所在的那一层换成 `edit(那一层)`。 */
function editList(tree: ExposedEntry[], parent: Path, edit: (list: ExposedEntry[]) => ExposedEntry[]): ExposedEntry[] {
  if (!parent.length) return edit(tree);
  const [i, ...rest] = parent;
  return tree.map((x, j) => (j === i && isGroup(x) ? { ...x, children: editList(x.children, rest, edit) } : x));
}

export function replaceAt(tree: ExposedEntry[], path: Path, entry: ExposedEntry): ExposedEntry[] {
  return editList(tree, path.slice(0, -1), (list) => list.map((x, j) => (j === path[path.length - 1] ? entry : x)));
}

export function removeAt(tree: ExposedEntry[], path: Path): ExposedEntry[] {
  return editList(tree, path.slice(0, -1), (list) => list.filter((_, j) => j !== path[path.length - 1]));
}

export function insertAt(tree: ExposedEntry[], parent: Path, index: number, entries: ExposedEntry[]): ExposedEntry[] {
  return editList(tree, parent, (list) => [...list.slice(0, index), ...entries, ...list.slice(index)]);
}

export const inside = (path: Path, of: Path) => path.length > of.length && of.every((v, i) => path[i] === v);
export const samePath = (a: Path | null, b: Path | null) => !!a && !!b && a.length === b.length && a.every((v, i) => b[i] === v);

/** 把 `from` 那一项挪到 `parent` 这一层的第 `index` 个位置；返回新的树和它挪到哪儿了。 */
export function move(tree: ExposedEntry[], from: Path, parent: Path, index: number): [ExposedEntry[], Path] {
  const entry = at(tree, from)!;
  // 先拿走：同一层里从前面挪到后面，位置少一个；拿走的若在 parent 的上一级的前面，parent 自己的位置也要少一个
  const fParent = from.slice(0, -1);
  const fi = from[from.length - 1];
  const p = [...parent];
  if (p.length > fParent.length && fParent.every((v, i) => p[i] === v) && p[fParent.length] > fi) p[fParent.length]--;
  let idx = index;
  if (samePath(fParent, parent) && fi < idx) idx--;
  const took = removeAt(tree, from);
  return [insertAt(took, p, idx, [entry]), [...p, idx]];
}

/** 把 `from` 那一项合并进 `into`（「编辑参数界面」的「合并」：两项变一项，一个值写进两边的参数，同 Houdini 一个参数
 * 引用到几处）：`into` 留下，目标接上 `from` 的（去重，按先后），`from` 去掉；别的项的 Hide When / Disable When（含下拉
 * 各项自己的）里用到 `from` 名字的改成 `into` 的。两项都得是参数项。返回新的树和 `into` 现在的位置。 */
export function mergeInto(tree: ExposedEntry[], into: Path, from: Path): [ExposedEntry[], Path] {
  const a = at(tree, into) as ExposedParam;
  const b = at(tree, from) as ExposedParam;
  const keys = [...new Set([...targetsOf(a), ...targetsOf(b)])];
  const merged: ExposedParam = { ...a, target: keys.length === 1 ? keys[0] : keys };
  const renamed = (e: ExposedEntry): ExposedEntry => {
    if (isGroup(e)) return { ...e, children: e.children.map(renamed) };
    const r = (s: string | undefined) => (s === undefined ? s : renamedInCondition(s, b.name, a.name));
    const out: ExposedParam = { ...e };
    if (e.hide_when !== undefined) out.hide_when = r(e.hide_when);
    if (e.disable_when !== undefined) out.disable_when = r(e.disable_when);
    if (e.options) out.options = e.options.map((o) => ({ ...o, ...(o.hide_when !== undefined ? { hide_when: r(o.hide_when) } : {}),
      ...(o.disable_when !== undefined ? { disable_when: r(o.disable_when) } : {}) }));
    return out;
  };
  const next = removeAt(replaceAt(tree, into, merged), from).map(renamed);
  const now = rowsOf(next).find((row) => !isGroup(row.entry) && row.entry.name === a.name)?.path ?? into;
  return [next, now];
}

/** 拖动排序的落点规则（参数面板里直接拖、「编辑参数界面」弹窗里拖，同一套）：`rows` 是 rowsOf 的一行行，行 `to` 等于
 * rows.length 是最下面那条「放到最外层末尾」。拖到参数上 = 放到它前面，拖到组上 = 放进这个组的末尾。返回新的树和挪到了哪儿。 */
export function dropAt(tree: ExposedEntry[], rows: Row[], from: number, to: number): [ExposedEntry[], Path] {
  const src = rows[from].path;
  if (to === rows.length) return move(tree, src, [], tree.length);
  const dst = rows[to];
  return isGroup(dst.entry) ? move(tree, src, dst.path, dst.entry.children.length) : move(tree, src, dst.path.slice(0, -1), dst.path[dst.path.length - 1]);
}

/** 能不能放在那儿：不能放进自己或自己的子组里。 */
export const canDropAt = (rows: Row[], from: number, to: number): boolean =>
  to === rows.length || (!inside(rows[to].path, rows[from].path) && !samePath(rows[to].path, rows[from].path));

export interface Row {
  path: Path;
  entry: ExposedEntry;
  depth: number;
}

export function rowsOf(tree: ExposedEntry[], parent: Path = [], depth = 0): Row[] {
  return tree.flatMap((entry, i) => {
    const path = [...parent, i];
    return [{ path, entry, depth }, ...(isGroup(entry) ? rowsOf(entry.children, path, depth + 1) : [])];
  });
}

