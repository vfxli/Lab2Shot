/** 参数界面树（应用模式 / 参数面板）里一组开关的「全开 / 全关」：判定只在这里，ParamPanel.tsx 的组头只读它。
 *
 * 规则：组里直接的子项全是开关（目标是布尔参数，或公开成复选框的整数 0 / 1 参数），没有子组、按钮或别的参数，且至少
 * 两个——这时组头有「全开」「全关」。点一下把组里现在显示、没置灰的开关一起设成开或关（一步撤销）；被 Hide When 藏起、
 * Disable When 置灰的不动（它们此刻不由使用者决定）。纯函数，不碰页面：Node 直接跑测试（groupSwitches.test.ts）。 */

import type { ExposedEntry, ExposedGroup, ExposedParam } from "../api/catalog.ts";
import { targetsOf } from "./targets.ts";

/** 一项公开参数作为开关的样子：布尔参数（真 / 假），或公开成复选框的整数参数（1 / 0）；null：不是开关。 */
export type SwitchKind = "bool" | "int01" | null;

const isGroupEntry = (x: ExposedEntry): x is ExposedGroup => (x as ExposedGroup).kind === "group";

/** 组里的开关，只有整组都是开关（且至少两个）时才给；否则 null（组头不出「全开 / 全关」）。`kindOf`：一项参数是不是开关
 * （看它目标参数的类型和公开的控件，ParamPanel 从节点类型的参数说明里查）。 */
export function groupSwitches(g: ExposedGroup, kindOf: (x: ExposedParam) => SwitchKind): ExposedParam[] | null {
  if (g.children.length < 2) return null;
  const out: ExposedParam[] = [];
  for (const c of g.children) {
    if (isGroupEntry(c) || !kindOf(c)) return null;
    out.push(c);
  }
  return out;
}

/** 「全开」（on = true）/「全关」要写的值：每个现在可改的开关（没藏起、没置灰）一个，值已经是它的就不写。
 * `value`：这项现在的值；`shown`、`enabled`：Hide When / Disable When 的结果。 */
export function switchesTo(on: boolean, switches: ExposedParam[], kindOf: (x: ExposedParam) => SwitchKind,
                           value: (x: ExposedParam) => unknown, shown: (x: ExposedParam) => boolean,
                           enabled: (x: ExposedParam) => boolean): { target: string; value: boolean | number }[] {
  const out: { target: string; value: boolean | number }[] = [];
  for (const x of switches) {
    if (!shown(x) || !enabled(x)) continue;
    const want = kindOf(x) === "int01" ? (on ? 1 : 0) : on;
    const now = value(x);
    if (kindOf(x) === "int01" ? now === want : !!now === want) continue;
    for (const target of targetsOf(x)) out.push({ target, value: want }); // an entry driving several: each of them
  }
  return out;
}

/** 组里的开关现在是不是全开 / 全关（按钮据此置灰：已经全开时「全开」没有事可做）。 */
export function switchesState(switches: ExposedParam[], kindOf: (x: ExposedParam) => SwitchKind, value: (x: ExposedParam) => unknown,
                              shown: (x: ExposedParam) => boolean, enabled: (x: ExposedParam) => boolean): { allOn: boolean; allOff: boolean } {
  const live = switches.filter((x) => shown(x) && enabled(x));
  const on = (x: ExposedParam) => (kindOf(x) === "int01" ? value(x) === 1 : !!value(x));
  return { allOn: live.every(on), allOff: live.every((x) => !on(x)) };
}
