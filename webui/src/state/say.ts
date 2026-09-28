import { log } from "./log";
import { useCookInputs } from "./cookInputs";
import { LEVEL_TABLE } from "../messages/levels";
import type { Message } from "../messages/message";
import { shorten } from "../messages/shorten";

/** Writes a message into the page's log, with its code (log.ts).
 *
 * 节点名只写一次：许多服务器消息的模板本身以「{node}」开头（W-INPUT-UNUSED、B-COOK-* 等），
 * 若此处无条件再加前缀，将出现「「序列图输出设置」「序列图输出设置」没有用上「图像」」。 */
export function logMessage(m: Message, nodeLabel = ""): void {
  // 先去掉正文中已有的名称（messages/shorten.ts：整段「节点名」或项目名），再统一添加一次前缀，
  // 使日志中始终为「「节点名」消息内容」的形式，名称不会重复出现
  const text = shorten(m.text, nodeLabel);
  log(LEVEL_TABLE[m.level].log, nodeLabel ? `「${nodeLabel}」${text}` : text, m.code);
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
  logMessage(m, node ? (useCookInputs.getState().nodes[node]?.label ?? node) : "");
}

export { fromServer, messageOf, msg, reasonOf, type Message } from "../messages/message";
