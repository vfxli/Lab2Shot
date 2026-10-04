import { nodeRefOf } from "./cookInputs";
import { LEVEL_TABLE } from "../messages/levels";
import type { Message } from "../messages/message";
import { log, nameOf } from "./log";
import { namesNode } from "../messages/shorten";
import type { Said } from "../messages/format";
import { t } from "../i18n/t";

/** Writes a message into the page's log, with its code (log.ts).
 *
 * 节点名只写一次：许多服务器消息的模板本身以「{node}」开头（W-INPUT-UNUSED、B-COOK-* 等），
 * 若此处无条件再加前缀，将出现「「序列图输出设置」「序列图输出设置」没有用上「图像」」。 */
export function logMessage(m: Message, node: Said | string = ""): void {
  // 消息自己的参数里已经写了这个节点（messages/shorten.ts namesNode：按参数判断，不看文字）就照原样记，
  // 否则在前面加一次节点（名字（类型），按键保存，换语言时照样说对），名称不会重复出现
  const text = m.text;
  // kept as the message it is too, so a reader who switches language reads it in theirs (log.ts entryText)
  const params = m.params;
  const named = nameOf(node);
  log(LEVEL_TABLE[m.level].log, named && !namesNode(params, named) ? t("ui.state.log_node", { node, text }) : text, m.code,
      { ...(params ? { params } : {}), body: m.text, ...(named ? { node } : {}) });
}

/** Writes a message about a node into the page's log, prefixed with that node's name.
 *
 * 所有消息统一写入日志，没有其他落点：本函数不向任何面板写入内容，也不打开任何面板。
 * 与 `logMessage` 的唯一区别：本函数接收节点 id，由其在 `cookInputs` 中查找名称；
 * `logMessage` 接收已查好的名称（任务结束等消息没有节点 id）。
 *
 * 节点计算完成后的角标及「数据信息」中的「消息」组走另一条路径：服务器随结果一并保存的
 * `results[节点].messages`（`state/results.ts`），与此处互不依赖。 */
export function say(m: Message, node = ""): void {
  logMessage(m, node ? nodeRefOf(node) : "");
}

export { fromServer, messageOf, msg, reasonOf, textOf, type Message } from "../messages/message";
