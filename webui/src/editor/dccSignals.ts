/** 页面给 DCC 插件的约定信号（B4 聚焦模式），只在这里定义一次：
 * - `<html data-lab2shot="focus|done">`：聚焦页打开时是 FOCUS，按了「完成」是 DONE；
 * - `<html data-lab2shot-job="…">`：最近一次「计算」提交的任务号；
 * - 按了「完成」时页面标题变成 DONE_TITLE。
 * 插件侧读同样的名字：clients/common/lab2shot_dcc/ui/web.py（DONE_TITLE、MARK_ATTRIBUTE）。改这里就要同时改那边，
 * 两边任何一处变了，内嵌窗口就等不到「完成」。写这些标记的只有 editor/AppMode.tsx markFocus。 */
export const DONE_TITLE = "lab2shot:done";
/** document.documentElement.dataset 上的键：`lab2shot` 即 `data-lab2shot`，`lab2shotJob` 即 `data-lab2shot-job`。 */
export const MARK_KEY = "lab2shot";
export const MARK_JOB_KEY = "lab2shotJob";
export const MARK_FOCUS = "focus";
export const MARK_DONE = "done";
export type FocusMark = typeof MARK_FOCUS | typeof MARK_DONE;
