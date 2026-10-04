/** graph/naming.ts 的测试：Node 直接跑（node --experimental-strip-types src/graph/naming.test.ts）。 */

import { defaultName, isDefaultName, nameProblem, nameStem, nodeRef, nodeWord } from "./naming.ts";
import { namesNode } from "../messages/shorten.ts";
import { addWords } from "../i18n/words.ts";
import { t } from "../i18n/t.ts";
import { WORDS } from "../messages/generatedCatalogue.ts";

addWords(WORDS);

let count = 0;
function eq(got: unknown, want: unknown, what: string): void {
  count++;
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) throw new Error(`${what}\n  得到 ${g}\n  应为 ${w}`);
}

const taken = (ids: string[]) => (id: string) => ids.includes(id);
eq(nameStem("fbx.import"), "fbx_import", "点换下划线");
eq(defaultName("fbx.import", taken([])), "fbx_import1", "第一个从 1 起");
eq(defaultName("retarget", taken(["retarget1", "retarget3"])), "retarget2", "取未用的最小编号");
eq(defaultName("retarget", taken(["retarget1", "retarget2"])), "retarget3", "连续占用接着编");
eq(isDefaultName("fbx_import12", "fbx.import"), true, "默认名");
eq(isDefaultName("char_fbx", "fbx.import"), false, "改过名");
eq(isDefaultName("fbx_import", "fbx.import"), false, "没编号不是默认名");
eq(nameProblem("char_fbx", taken([])), null, "合规");
eq(nameProblem("Char", taken([])) !== null, true, "大写不行");
eq(nameProblem("1abc", taken([])) !== null, true, "数字开头不行");
eq(nameProblem("a.b", taken([])) !== null, true, "点不行");
eq(nameProblem("角色", taken([])) !== null, true, "中文不行");
eq(nameProblem("a".repeat(33), taken([])) !== null, true, "超过 32 不行");
eq(nameProblem("a".repeat(32), taken([])), null, "32 可以");
eq(nameProblem("", taken([])) !== null, true, "空不行");
eq(nameProblem("cam", taken(["cam"])) !== null, true, "重名不行");
eq(nodeRef("fbx_import1", "fbx.import"), t("ui.node.ref", { name: "fbx_import1", type: "fbx.import" }), "消息里指向节点");
eq(t("ui.node.ref", { name: "fbx_import1", type: "fbx.import" }, "zh"), "fbx_import1（fbx.import）", "中文的括号");
eq(t("ui.node.ref", { name: "fbx_import1", type: "fbx.import" }, "en"), "fbx_import1 (fbx.import)", "英文的括号");
eq(nodeRef("x", undefined), "x", "没有类型只写名字");
// 消息参数里的节点按键保存（nodeWord）：同一条日志换语言时括号跟着语言
const w = nodeWord("fbx_import1", "fbx.import");
eq(t("ui.state.log_node", { node: w, text: "x" }, "zh"), "「fbx_import1（fbx.import）」x", "按键保存的节点：中文");
eq(t("ui.state.log_node", { node: w, text: "x" }, "en"), "fbx_import1 (fbx.import): x", "按键保存的节点：英文");
eq(JSON.parse(JSON.stringify(w)), w, "存进浏览器再读出来不变");
eq(namesNode({ node: w }, "fbx_import1"), true, "参数里按键保存的节点认得出");
eq(namesNode({ node: { said: "engine.node_ref", params: { name: "fbx_import1", type: "fbx.import" } } }, "fbx_import1"), true, "服务器的节点引用也认得出");
eq(namesNode({ node: w }, "fbx_import2"), false, "别的节点不算");
console.log(`naming：${count} 项都对`);
