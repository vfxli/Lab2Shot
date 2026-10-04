import type { Level } from "./format";

/** What the page does with each message level — one table for every place that shows a level (the page's log, a node's
 * mark and its 数据信息, the one message row ui/MessageRow.tsx). The letters are the server's (generatedGateCatalogue.ts
 * LEVELS, from lab2shot/messages): a letter the server adds and this table lacks does not compile.
 * `order`: which level is shown first when a node has several (errors, then production risk, then the rest);
 * `log`: the page log's level; `listed`: the levels shown to the user (`I` information goes only to the log: never on a node, never counted as
 * the loudest one),
 * read only by `editor/NodeInfoCard.tsx worstLevel`; `risk`: drawn in the production-risk red — the one level that is.
 * `word`, `tip`: the keys of the level's name and its explanation (lab2shot/i18n/<lang>/ui/misc.toml), read with t(). */
export const LEVEL_TABLE: Readonly<Record<Level, { word: string; tip: string; log: "error" | "warn" | "info"; listed: boolean; risk: boolean; order: number }>> = {
  P: { word: "ui.misc.level.p", tip: "ui.misc.level.p_tip", log: "error", listed: true, risk: true, order: 1 },
  E: { word: "ui.misc.level.e", tip: "ui.misc.level.e_tip", log: "error", listed: true, risk: false, order: 0 },
  B: { word: "ui.misc.level.b", tip: "ui.misc.level.b_tip", log: "error", listed: true, risk: false, order: 2 },
  W: { word: "ui.misc.level.w", tip: "ui.misc.level.w_tip", log: "warn", listed: true, risk: false, order: 3 },
  N: { word: "ui.misc.level.n", tip: "ui.misc.level.n_tip", log: "info", listed: true, risk: false, order: 4 },
  I: { word: "ui.misc.level.i", tip: "ui.misc.level.i_tip", log: "info", listed: false, risk: false, order: 5 },
};

/** The levels in the order they are listed. */
export const LEVELS_BY_SEVERITY = (Object.keys(LEVEL_TABLE) as Level[]).sort((a, b) => LEVEL_TABLE[a].order - LEVEL_TABLE[b].order);
