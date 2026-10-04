/** 编辑器模式（节点模式 / 应用模式 / 聚焦模式）的唯一来源：当前模式、按账号记住的选择，以及切换它的按钮
 * （顶栏右边、「模板」左边：下面的 ModeSwitch，放在 editor/Chrome.tsx 顶栏里）。
 *
 * 节点模式是完整的编辑器：节点图、节点参数、做模板的一切操作。应用模式给只用模板、不做模板的人：节点图画布和节点
 * 面板都不显示，参数面板就是模板公开的参数界面树（模板作者在「编辑参数界面」和面板拖动里配的：分组、顺序、下拉 / 复选框、
 * Hide When / Disable When）。「计算」「下载」也是树里的按钮参数（editor/buttonActions.tsx），模板作者公开进来、放在想放的
 * 位置；没公开的就没有，这里不写死任何块。视图、时间线、队列照常。
 * 同一份文档，切回节点模式一切照旧：应用模式只是换一种看法，不改文档。
 *
 * 聚焦模式是应用模式的一个变体（同一套布局：节点图收起，左栏只有视图），给 DCC 插件的内嵌窗口：地址
 * `#job=<任务号>&focus=<节点 id>` 打开自己的这个任务（editor/focusJob.ts），只显示视图（显示这个节点）、这个节点的参数
 * 面板（它的「在视图里编辑」等手柄照常），顶栏只剩「计算」「完成」（editor/Chrome.tsx FocusBar）。它不记进偏好、
 * 不能从按钮切进切出：只由地址进入，这一页一直是它。
 *
 * 默认进应用模式；节点 / 应用两种按账号记在这个浏览器里（localStorage：platform/storage.ts，键 `mode.<账号 id>`），不上服务器。 */

import { useEffect } from "react";
import { create } from "zustand";
import { readPref, writePref } from "../platform/storage";
import { useSession } from "../state/session";
import { Button } from "../ui/Button";
import { DONE_TITLE, MARK_DONE, MARK_FOCUS, MARK_JOB_KEY, MARK_KEY, type FocusMark } from "./dccSignals";
import { t } from "../i18n/t";
import { setPhrasing } from "../i18n/lang";

export type EditorMode = "template" | "app" | "focus";

/** 聚焦模式打开的是哪个任务的哪个节点；`targets`、`delivers`：那个任务当初算的节点、是不是整理打包「输出」
 * （GET /api/jobs/{id}：聚焦页的「计算」照样再算一次，结果才是 DCC 取得回去的）。打开之前两者为空。 */
export interface Focus {
  job: string;
  node: string;
  targets: string[];
  delivers: boolean;
  submitted: number | null; // 聚焦页最近一次「计算」提交时文档的计算输入版本（state/cookInputs.ts version）；null：还没算过
  done: boolean; // 点过「完成」
}

/** 地址里的 `#job=<id>&focus=<node>`（只有 job：打开任务但不聚焦）。读一次，页面刷新时照样读。 */
export function jobFromAddress(hash = window.location.hash): { job: string; focus: string } | null {
  const q = new URLSearchParams(hash.replace(/^#/, ""));
  const job = q.get("job") ?? "";
  return /^[A-Za-z0-9_-]{1,64}$/.test(job) ? { job, focus: q.get("focus") ?? "" } : null;
}

const prefKey = (account: number | null) => `mode.${account ?? 0}`;
const asked = jobFromAddress();

interface ModeState {
  mode: EditorMode;
  account: number | null; // 记的是哪个账号的选择
  focus: Focus | null; // 聚焦模式时有
  follow: (account: number | null) => void; // 登录的账号（换了账号就读那个账号上次的选择）
  setMode: (mode: EditorMode) => void;
  setFocus: (patch: Partial<Focus>) => void;
}

export const useAppMode = create<ModeState>((set, get) => ({
  // 所有人默认进应用模式；切过节点模式的按账号记住。地址要聚焦的，一开始就是聚焦布局（不先闪一下完整编辑器）
  mode: asked?.focus ? "focus" : "app",
  account: null,
  focus: asked?.focus ? { job: asked.job, node: asked.focus, targets: [], delivers: false, submitted: null, done: false } : null,
  follow: (account) => {
    if (account === get().account) return;
    if (get().mode === "focus") return set({ account }); // 聚焦页一直是聚焦模式
    set({ account, mode: readPref<{ mode?: unknown }>(prefKey(account), {}).mode === "template" ? "template" : "app" });
  },
  setMode: (mode) => {
    if (get().mode === "focus" || mode === "focus") return; // 只由地址进入
    writePref(prefKey(get().account), { mode });
    set({ mode });
  },
  setFocus: (patch) => {
    const f = get().focus;
    if (f) set({ focus: { ...f, ...patch } });
  },
}));

/** 节点图收起（应用模式和聚焦模式：左栏只有视图）。 */
export const graphHidden = (mode: EditorMode): boolean => mode !== "template";

// 页面对谁说话跟着模式走（i18n/lang.ts phrasing，唯一的开关）：节点图收起时是用卡片的人——有「.app」说法的词和消息
// 按它说（不出现节点名、连线、右键节点这类节点模式的操作）；节点模式照原话
setPhrasing(graphHidden(useAppMode.getState().mode) ? "app" : "graph");
useAppMode.subscribe((s) => setPhrasing(graphHidden(s.mode) ? "app" : "graph"));

/** 一次提交带不带 follows（POST /api/jobs）：在聚焦模式里、并且算的正是打开的任务当初算的（整理打包「输出」：`what`
 * 为 "*"；或那个节点）时，是打开的那个任务号——DCC 取回它的结果；别的（参数面板里节点自己的「计算」，只为在视图里看）
 * 不带，免得插件把一个没有交付的任务当成新版本。不在聚焦模式时为 undefined。 */
export const focusFollows = (what: string): string | undefined => {
  const s = useAppMode.getState();
  const f = s.mode === "focus" ? s.focus : null;
  if (!f) return undefined;
  return (what === "*" ? f.delivers : !f.delivers && (f.targets[0] ?? f.node) === what) ? f.job : undefined;
};

/** 聚焦模式里「计算」提交成功（graph/actions.ts）：记下提交时的计算输入版本（「完成」据此看有没有没算的修改），并把
 * 新任务号写到页面的约定标记上（markFocus）。不在聚焦模式时什么也不做。 */
export function focusSubmitted(job: string, version: number, follows: string | undefined): void {
  if (!follows) return;
  useAppMode.getState().setFocus({ submitted: version });
  markFocus(MARK_FOCUS, job);
}

/** 写给 DCC 插件看的约定标记（名字只在 editor/dccSignals.ts 定义，插件侧读同样的名字）：`<html data-lab2shot="focus|done">`，
 * 最近一次「计算」的任务号在 `data-lab2shot-job`；「完成」时页面标题变成 DONE_TITLE。 */
export function markFocus(state: FocusMark, job?: string): void {
  const root = document.documentElement;
  root.dataset[MARK_KEY] = state;
  if (job) root.dataset[MARK_JOB_KEY] = job;
  if (state === MARK_DONE) document.title = DONE_TITLE;
}

/** 「模板」左边的「节点模式 / 应用模式」切换：两个按钮和「模板」同一样式，拼成一个整体、由同一圈彩虹亮边（entry-rim，
 * 和「模板」的亮边同一套动画）框住——看得出是一组二选一；当前模式高亮（.on）。 */
export function ModeSwitch() {
  const account = useSession((s) => s.state?.user?.id ?? null);
  const mode = useAppMode((s) => s.mode);
  const setMode = useAppMode((s) => s.setMode);
  useEffect(() => useAppMode.getState().follow(account), [account]);
  return (
    <span className="mode-switch entry-rim" role="group" aria-label={t("ui.app.mode")}>
      <Button on={mode === "template"} onClick={() => setMode("template")}>
        {t("ui.app.node_mode")}
      </Button>
      <Button on={mode === "app"} onClick={() => setMode("app")}>
        {t("ui.app.app_mode")}
      </Button>
    </span>
  );
}
