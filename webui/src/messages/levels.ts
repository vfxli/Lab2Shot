import type { Level } from "./format";

/** What the page does with each message level — one table for every place that shows a level (the page's log, a node's
 * mark and its 数据信息, the one message row ui/MessageRow.tsx). The letters are the server's (generatedGateCatalogue.ts
 * LEVELS, from lab2shot/messages): a letter the server adds and this table lacks does not compile.
 * `order`: which level is shown first when a node has several (errors, then production risk, then the rest);
 * `log`: the page log's level; `listed`: 要给用户看的那些（`I` 信息只进日志，不上节点、不算进「最响的那一条」），
 * read only by `editor/NodeInfoCard.tsx worstLevel`; `risk`: drawn in the production-risk red — the one level that is. */
export const LEVEL_TABLE: Readonly<Record<Level, { word: string; tip: string; log: "error" | "warn" | "info"; listed: boolean; risk: boolean; order: number }>> = {
  P: { word: "生产风险", tip: "生产风险：可能造成生产事故，交付前需处理。红色只用于此级别。", log: "error", listed: true, risk: true, order: 1 },
  E: { word: "错误", tip: "错误：节点未能完成计算。", log: "error", listed: true, risk: false, order: 0 },
  B: { word: "拦下", tip: "拦下：提交前已拦下，处理后重新提交。", log: "error", listed: true, risk: false, order: 2 },
  W: { word: "警告", tip: "警告：已完成计算，结果可能不符合预期。", log: "warn", listed: true, risk: false, order: 3 },
  N: { word: "提醒", tip: "提醒：正常情况，供参考。", log: "info", listed: true, risk: false, order: 4 },
  I: { word: "信息", tip: "信息：只记录在日志中。", log: "info", listed: false, risk: false, order: 5 },
};

/** The levels in the order they are listed. */
export const LEVELS_BY_SEVERITY = (Object.keys(LEVEL_TABLE) as Level[]).sort((a, b) => LEVEL_TABLE[a].order - LEVEL_TABLE[b].order);
