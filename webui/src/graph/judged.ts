/** 刚连上、服务器还没判过的线：判它的那次状态回复说一次它哪里不对（sayJudgedWires），之后不再说。connect / moveWires
 * 记进来（graph/edit.ts），换文档时清（graph/document.ts loadGraph），状态回复到了读（graph/actions.ts refreshStatus）。
 * 单独一个模块：这三处都要它，放在 actions 里会让 document / edit 反过来 import actions（环）。 */

import type { StatusReply } from "../api";
import { getNodeDefs } from "../state/catalog";
import { nodeRefOf } from "../state/cookInputs";
import { fromServer, msg, say } from "../state/say";
import { wireKey } from "./rules";

export const justWired = new Set<string>();

export function sayJudgedWires(reply: StatusReply): void {
  for (const w of reply.wires) {
    const id = wireKey(w.from[0], w.from[1], w.to[0], w.to[1]);
    if (!justWired.delete(id) || w.state === "ok" || !w.problem) continue;
    const problem = fromServer(w.problem);
    const fix = getNodeDefs()[w.fix];
    const node = nodeRefOf(w.to[0]);
    say(fix ? msg("W-WIRE-WRONGFIX", { problem, node, via: fix.subtitle }, { port: w.to[1] }) : msg("W-WIRE-WRONG", { problem }, { port: w.to[1] }), w.to[0]);
  }
}
