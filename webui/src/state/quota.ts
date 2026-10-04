import type { StorageGate } from "../api/library";
import { msg, type Message } from "../messages/message";
import { gbText, sizeText } from "../platform/format";
import { t } from "../i18n/t";

/** 存储配额已满（B-STORAGE-FULL，lab2shot/i18n/<lang>/messages/web.toml）。结构与 state/pause.ts 的「计算任务已暂停」相同：
 * 一条消息、一条「本次点击将遇到的限制」的说明、一个「当前是否可点击」的判断。所有「计算」入口的置灰（graph/actions.ts
 * cookHold：按钮参数、节点右键菜单）与提交前的闩（latched，快捷键也走它）均读取此处，页面中不得另写一套判断。
 *
 * 规则与服务器完全一致（lab2shot/server/farm.py submit → quota.refuse_if_full）：「计算」与「提交」一律拦下；
 * 查看已有的结果不受影响，用户仍能看到数据，据此判断应清理哪一部分。 */
const fullMessage = (s: StorageGate): Message =>
  msg("B-STORAGE-FULL", { used: sizeText(s.total), limit: s.limit ? gbText(s.limit / 2 ** 30) : t("ui.state.no_limit") });

/** 配额是否已满（尚未查询过队列时按未满处理：服务器端仍会拦截）。 */
export const blockedByQuota = (s: StorageGate | null): boolean => !!s?.over;

/** 「计算」悬停提示中的补充说明（""：未满，不显示）。 */
export const quotaNote = (s: StorageGate | null): string => (blockedByQuota(s) ? `\n\n${fullMessage(s!).text}` : "");

/** 提交前检查：配额已满时说明已用量、上限及清理位置，不向服务器发送任何请求。 */
export function readyForQuota(s: StorageGate | null, blocked: (m: Message) => void): boolean {
  if (!blockedByQuota(s)) return true;
  blocked(fullMessage(s!));
  return false;
}
