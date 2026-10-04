/** model/nodeOutcome.ts 的测试：Node 直接跑（node src/model/nodeOutcome.test.ts）。 */

import { BLOCKED_NOTE, isBlocked, noteSaid, notesToSay, noteText, noteWord, outcomeView, SKIPPED_NOTE, skippedBranches, skippedEvent, switchedOff, switchedOffNodes } from "./nodeOutcome.ts";
import { shownTarget } from "./targets.ts";
import { addWords } from "../i18n/words.ts";
import { WORDS } from "../messages/generatedCatalogue.ts";
import { t } from "../i18n/t.ts";
import { setLang } from "../i18n/lang.ts";

addWords(WORDS);
setLang("zh");

let count = 0;
function eq(got: unknown, want: unknown, what: string): void {
  count++;
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) throw new Error(`${what}\n  得到 ${g}\n  应为 ${w}`);
}

eq(outcomeView({ state: "skipped", blocked: true }, false), { status: "skipped", note: BLOCKED_NOTE }, "被阻断：已跳过（被阻断）");
eq(outcomeView({ state: "skipped" }, false), { status: "skipped", note: null }, "上游出错：已跳过，底行不写被阻断");
eq(outcomeView({ state: "failed" }, true), { status: "error", note: null }, "自己出错：出错");
eq(outcomeView(undefined, true), { status: "cooked", note: null }, "有结果：已缓存");
eq(outcomeView(undefined, false), { status: "idle", note: null }, "没结果：未计算");
eq(skippedEvent({ blocked: true }), { note: BLOCKED_NOTE, say: false }, "计算事件：被阻断不在消息栏说");
eq(skippedEvent({}), { note: SKIPPED_NOTE, say: true }, "计算事件：上游出错照常说");
eq(noteText(BLOCKED_NOTE), "已跳过（被阻断）", "文字");
eq(noteText("算到 3 / 9"), "算到 3 / 9", "别的底行字原样");

// 应用模式：按钮所在节点的分支里哪些因开关关着跳过了
eq(isBlocked({ status: "skipped", note: BLOCKED_NOTE }), true, "被阻断");
eq(isBlocked({ status: "skipped", note: SKIPPED_NOTE }), false, "上游出错的跳过不算");
eq(isBlocked(undefined), false, "没有状态");
eq(skippedBranches("out", (id) => id === "a" || id === "b"), "", "接进来的分支关着：不说（开关就在卡片上）");
eq(skippedBranches("out", () => false), "", "都没关：不说");
eq(skippedBranches("a", (id) => id === "a"), switchedOff(), "自己被阻断：说一句");

// 底行的字存的是怎么说：显示时按当前语言取（切换语言后跟着变）
const took = noteWord("ui.graph.took", { seconds: 3 });
eq(noteText(took), t("ui.graph.took", { seconds: 3 }), "页面的词：中文");
const waiting = [noteWord("ui.graph.waiting"), noteSaid({ code: "B-QUEUE-PAUSED", text: "服务器暂停接任务" })];
eq(noteText(waiting), `${t("ui.graph.waiting")} · 服务器暂停接任务`, "几段用 · 连起来");
setLang("en");
eq(noteText(took), t("ui.graph.took", { seconds: 3 }), "页面的词：切到英文后是英文");
eq(noteText(BLOCKED_NOTE), t(BLOCKED_NOTE), "被阻断：英文");
eq(notesToSay([waiting, took, ""], "en").map((n) => n.code), ["B-QUEUE-PAUSED"], "服务器的消息还没有英文的字：要重说");
eq(noteText(waiting).endsWith("服务器暂停接任务"), true, "还没重说好：用它说过的字");
(waiting[1] as { texts: Record<string, string> }).texts.en = "The server has paused taking jobs";
eq(noteText(waiting), `${t("ui.graph.waiting")} · The server has paused taking jobs`, "重说好了：英文");
eq(notesToSay([waiting], "en").length, 0, "已经有英文的字：不再重说");
setLang("zh");

// 一项公开参数驱动两个跟踪节点（workflow_2d_track：CoTracker3、TAPNext++ 共用跟踪点）：只开 TAPNext++ 时替 TAPNext++ 说话
{
  const edges = [
    { source: "read", target: "cotracker" }, { source: "read", target: "tapnext" },
    { source: "cotracker", target: "track_cotracker" }, { source: "tapnext", target: "track_tapnext" },
    { source: "track_cotracker", target: "out_c" }, { source: "track_tapnext", target: "out_t" },
    { source: "out_c", target: "deliver" }, { source: "out_t", target: "deliver" },
  ];
  const picks = { target: ["cotracker.picks", "tapnext.picks"] };
  const offWith = (closed: string[]) => switchedOffNodes(edges, (id) => closed.includes(id) || closed.some((g) => id === `out_${g.slice(6, 7)}`));
  eq(offWith(["track_cotracker"])("cotracker"), true, "开关关了：开关上游只供这一路的节点也关着");
  eq(offWith(["track_cotracker"])("read"), false, "还供着开着的一路：开着");
  eq(offWith(["track_cotracker"])("deliver"), false, "没有接出去的节点只看自己");
  eq(shownTarget(picks, offWith(["track_cotracker"])), ["tapnext", "picks"], "只开 TAPNext++：落到 TAPNext++");
  eq(shownTarget(picks, offWith([])), ["cotracker", "picks"], "都开着：第一个");
  eq(shownTarget(picks, offWith(["track_cotracker", "track_tapnext"])), ["cotracker", "picks"], "都关着：回落到第一个");
  eq(shownTarget({ target: "a.b" }, () => true), ["a", "b"], "只有一个目标：就是它");
}

console.log(`nodeOutcome：${count} 项都对`);
