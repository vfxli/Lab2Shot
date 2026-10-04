/** 服务器说一个节点「为什么没有结果」（状态回复的 outcome、计算事件 skipped）在节点上怎么显示：只在这里判定，graph/actions.ts
 * （状态回复）和 graph/follow.ts（计算事件）都读它。
 *
 * - 自己出错 → 出错；上游出错 → 已跳过；
 * - 被「阻断」（gate）关着（outcome.blocked，引擎 engine/evaluation.py Outcome.blocked）→ 已跳过，节点底行写
 *   「已跳过（被阻断）」：这是开关关着，不是出错，也不在消息栏里说一遍。
 * 纯函数，Node 直接跑测试（nodeOutcome.test.ts）。 */

import type { NodeStatus } from "../state/graph.ts";
import { t } from "../i18n/t.ts";
import { getLang, type Lang } from "../i18n/lang.ts";
import type { Params } from "../messages/format.ts";

/** 节点底行的字：存的是怎么说，显示时（noteText）按当前语言取，切换语言后底行跟着变。
 * - 字符串：下面两个键（被阻断、上游出错），或服务器给的进度说明（计算中的一步，事件流在切换语言后重接，下一条就是新语言）；
 * - NoteWord：页面自己的词（「用时 x 秒」「出错」「排队中」）；
 * - NoteSaid：服务器的一条消息（排队在等什么）：按它说过的语言存下文字，切换语言时由服务器重说（state/language.ts）；
 * - 数组：几段，用「 · 」连起来。 */
export interface NoteWord { key: string; params?: Params }
export interface NoteSaid { code: string; params: Record<string, unknown>; texts: Partial<Record<Lang, string>> }
export type Note = string | NoteWord | NoteSaid | readonly Note[];

/** 页面自己的一个词，留到显示时再说。 */
export const noteWord = (key: string, params?: Params): NoteWord => (params ? { key, params } : { key });

/** 服务器的一条消息（code、参数、它说的字），留到显示时按当前语言取（没有当前语言的字时用它说过的）。 */
export const noteSaid = (m: { code: string; text: string; params?: Record<string, unknown> }): NoteSaid =>
  ({ code: m.code, params: m.params ?? {}, texts: { [getLang()]: m.text } });

const isWord = (n: Note): n is NoteWord => typeof n === "object" && !Array.isArray(n) && "key" in n;
const isSaid = (n: Note): n is NoteSaid => typeof n === "object" && !Array.isArray(n) && "texts" in n;

/** 底行里服务器说过、却还没有 `lang` 的字的那些消息：切换语言时请服务器重说（POST /api/said）。 */
export function notesToSay(notes: readonly Note[], lang: Lang): NoteSaid[] {
  const out: NoteSaid[] = [];
  const walk = (n: Note): void => {
    if (typeof n === "string") return;
    if (Array.isArray(n)) return n.forEach(walk);
    if (isSaid(n as Note) && !(n as NoteSaid).texts[lang]) out.push(n as NoteSaid);
  };
  notes.forEach(walk);
  return out;
}

/** 节点底行在被阻断时写的字（状态格只容 5 个字，写「已跳过」；底行写全）。存进节点状态的是这个键（不是文字），
 * 显示时经 noteText 按当前语言取字：切换语言后底行跟着变，isBlocked 也不受语言影响。 */
export const BLOCKED_NOTE = "ui.model.skipped_blocked";
/** 计算事件 skipped（上游出错）在底行写的字，同样存键。 */
export const SKIPPED_NOTE = "ui.model.skipped";
const NOTE_KEYS: ReadonlySet<string> = new Set([BLOCKED_NOTE, SKIPPED_NOTE]);

/** 节点状态里存的底行字 → 显示的字（当前语言）：上面两个键、页面的词按当前语言取，服务器的消息取当前语言的字
 * （还没重说好时用它说过的），其余（服务器给的进度说明）原样。 */
export function noteText(note: Note): string {
  if (typeof note === "string") return NOTE_KEYS.has(note) ? t(note) : note;
  if (Array.isArray(note)) return note.map(noteText).filter(Boolean).join(" · ");
  if (isWord(note as Note)) return t((note as NoteWord).key, (note as NoteWord).params);
  const texts = (note as NoteSaid).texts;
  return texts[getLang()] ?? Object.values(texts).find(Boolean) ?? "";
}

interface Outcome {
  state: "failed" | "skipped";
  blocked?: boolean;
}

/** 状态回复里一个节点的样子 → 状态格的状态和底行的字（"" 不改底行，`note` 为 null 时表示要清掉之前写的「被阻断」）。 */
export function outcomeView(outcome: Outcome | undefined, cached: boolean): { status: NodeStatus; note: string | null } {
  if (outcome?.state === "skipped" && outcome.blocked) return { status: "skipped", note: BLOCKED_NOTE };
  const status: NodeStatus = outcome?.state === "failed" ? "error" : outcome?.state === "skipped" ? "skipped" : cached ? "cooked" : "idle";
  return { status, note: null };
}

/** 计算事件 skipped（engine/cook.py：带 blocked 的是被阻断）：底行的字，以及要不要在消息栏里说（被阻断的不说）。 */
export function skippedEvent(e: { blocked?: boolean }): { note: string; say: boolean } {
  return e.blocked ? { note: BLOCKED_NOTE, say: false } : { note: SKIPPED_NOTE, say: true };
}

/** 这个节点现在是不是被「阻断」关着：状态回复或计算事件写下的（上面两个函数），节点底行（editor/NodeFoot.tsx）和应用模式
 * （skippedBranches）都按这一条认。 */
export const isBlocked = (st: { status: NodeStatus; note: Note } | undefined): boolean => st?.status === "skipped" && st.note === BLOCKED_NOTE;

/** 现在「关着」的节点：自己被阻断（isBlocked），或它接出去的每一路都关着——开关关了，开关上游只供这一路的节点
 * （如只开 TAPNext++ 时的 CoTracker3）也不会算。没有接出去的节点只看自己。`blocked` 按 isBlocked 判；model/targets.ts
 * shownTarget 按它挑一项公开参数替哪个节点说话。 */
export function switchedOffNodes(edges: readonly { source: string; target: string }[], blocked: (id: string) => boolean): (id: string) => boolean {
  const memo = new Map<string, boolean>();
  const off = (id: string): boolean => {
    const known = memo.get(id);
    if (known !== undefined) return known;
    memo.set(id, false); // a loop (none in a graph that cooks) counts as on
    const down = edges.filter((e) => e.source === id).map((e) => e.target);
    const v = blocked(id) || (down.length > 0 && down.every(off));
    memo.set(id, v);
    return v;
  };
  return off;
}

/** 应用模式里按钮下面的安静提示（不是出错、不是警告）：这一步被卡片上的开关关着，按了也不会算。渲染时取（当前语言）。 */
export const switchedOff = (): string => t("ui.model.skipped_switched_off");

/** 应用模式（editor/ParamPanel.tsx ExposedRow）：一个计算 / 打包按钮下要不要安静说一句。只在按钮所在的节点自己被阻断时说
 * （按了不会算）；接进它的分支被关着不说——那是卡片上看得见的开关的结果（关着的开关就在卡片上），而按节点逐个列出会把
 * 「每层另存」这类输出节点列成「MatAnyone 已跳过」、把重定目标关着时的结果输出列成「主结果跳过」，名字也是内部节点名，
 * 对用卡片的人只是误导。`blocked`：按 isBlocked 判。 */
export function skippedBranches(node: string, blocked: (id: string) => boolean): string {
  return blocked(node) ? switchedOff() : "";
}
