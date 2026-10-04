/** model/groupSwitches.ts 的测试：Node 直接跑（node src/model/groupSwitches.test.ts）。全部通过时打印一行，有一条不对就抛错。 */

import type { ExposedGroup, ExposedParam } from "../api/catalog.ts";
import { groupSwitches, switchesState, switchesTo, type SwitchKind } from "./groupSwitches.ts";

let count = 0;
function eq(got: unknown, want: unknown, what: string): void {
  count++;
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) throw new Error(`${what}\n  得到 ${g}\n  应为 ${w}`);
}

const P = (name: string, extra: Partial<ExposedParam> = {}): ExposedParam => ({ name, label: name, target: `${name}.v`, ...extra });
// 名字里带 b 的是布尔参数，带 i 的是公开成复选框的整数，带 t 的是文字（不是开关）
const kind = (x: ExposedParam): SwitchKind => (x.name.startsWith("b") ? "bool" : x.name.startsWith("i") && x.widget === "checkbox" ? "int01" : null);
const group = (...children: ExposedGroup["children"]): ExposedGroup => ({ kind: "group", label: "深度", children });

// 哪些组有「全开 / 全关」
eq(groupSwitches(group(P("b1"), P("b2"), P("b3")), kind)?.map((x) => x.name), ["b1", "b2", "b3"], "全是布尔：有");
eq(groupSwitches(group(P("b1"), P("i1", { widget: "checkbox" })), kind)?.map((x) => x.name), ["b1", "i1"], "布尔 + 复选框整数：有");
eq(groupSwitches(group(P("b1")), kind), null, "只有一个开关：没有");
eq(groupSwitches(group(), kind), null, "空组：没有");
eq(groupSwitches(group(P("b1"), P("t1")), kind), null, "混了别的参数：没有");
eq(groupSwitches(group(P("b1"), P("i1")), kind), null, "整数没公开成复选框：不是开关，没有");
eq(groupSwitches(group(P("b1"), P("b2"), group(P("b3"), P("b4"))), kind), null, "有子组：没有");

// 写什么
const sw = [P("b1"), P("b2"), P("i1", { widget: "checkbox" }), P("b3")];
const values: Record<string, unknown> = { b1: true, b2: false, i1: 0, b3: false };
const val = (x: ExposedParam) => values[x.name];
const all = () => true;
eq(switchesTo(true, sw, kind, val, all, all), [{ target: "b2.v", value: true }, { target: "i1.v", value: 1 }, { target: "b3.v", value: true }], "全开：只写没开的，整数写 1");
eq(switchesTo(false, sw, kind, val, all, all), [{ target: "b1.v", value: false }], "全关：只写开着的");
eq(switchesTo(true, sw, kind, val, (x) => x.name !== "b2", all).map((c) => c.target), ["i1.v", "b3.v"], "藏起的不动");
eq(switchesTo(true, sw, kind, val, all, (x) => x.name !== "b3").map((c) => c.target), ["b2.v", "i1.v"], "置灰的不动");
eq(switchesTo(true, sw, kind, () => undefined, all, all).length, 4, "没写的值（默认）按关算");

// 按钮置灰
eq(switchesState(sw, kind, val, all, all), { allOn: false, allOff: false }, "有开有关：两个都能点");
eq(switchesState(sw, kind, () => true, all, all).allOn, false, "整数开 = 1，不是 true");
eq(switchesState(sw, kind, (x) => (x.name === "i1" ? 1 : true), all, all), { allOn: true, allOff: false }, "全开时「全开」置灰");
eq(switchesState(sw, kind, (x) => (x.name === "i1" ? 0 : false), all, all), { allOn: false, allOff: true }, "全关时「全关」置灰");
eq(switchesState(sw, kind, (x) => x.name !== "b2", (x) => x.name !== "b2", all), { allOn: false, allOff: false }, "藏起的那个不算（其余里 i1 = true 不是 1）");

console.log(`groupSwitches：${count} 项都对`);
