/** 编辑器模式（节点模式 / 应用模式）的唯一来源：当前模式、按账号记住的选择，以及切换它的按钮
 * （顶栏右边、「模板」左边：下面的 ModeSwitch，放在 editor/Chrome.tsx 顶栏里）。
 *
 * 节点模式是完整的编辑器：节点图、节点参数、做模板的一切操作。应用模式给只用模板、不做模板的人：节点图画布和节点
 * 面板都不显示，参数面板就是模板公开的参数界面树（模板作者在「编辑参数界面」和面板拖动里配的：分组、顺序、下拉 / 复选框、
 * Hide When / Disable When）。「计算」「下载」也是树里的按钮参数（editor/buttonActions.tsx），模板作者公开进来、放在想放的
 * 位置；没公开的就没有，这里不写死任何块。视图、时间线、队列照常。
 * 同一份文档，切回节点模式一切照旧：应用模式只是换一种看法，不改文档。
 *
 * 默认进应用模式；选哪种模式按账号记在这个浏览器里（localStorage：platform/storage.ts，键 `mode.<账号 id>`），不上服务器。 */

import { useEffect } from "react";
import { create } from "zustand";
import { readPref, writePref } from "../platform/storage";
import { useSession } from "../state/session";
import { Button } from "../ui/Button";

export type EditorMode = "template" | "app";

const prefKey = (account: number | null) => `mode.${account ?? 0}`;

interface ModeState {
  mode: EditorMode;
  account: number | null; // 记的是哪个账号的选择
  follow: (account: number | null) => void; // 登录的账号（换了账号就读那个账号上次的选择）
  setMode: (mode: EditorMode) => void;
}

export const useAppMode = create<ModeState>((set, get) => ({
  mode: "app", // 所有人默认进应用模式；切过节点模式的按账号记住
  account: null,
  follow: (account) => {
    if (account === get().account) return;
    set({ account, mode: readPref<{ mode?: unknown }>(prefKey(account), {}).mode === "template" ? "template" : "app" });
  },
  setMode: (mode) => {
    writePref(prefKey(get().account), { mode });
    set({ mode });
  },
}));

/** 「模板」左边的「节点模式 / 应用模式」切换：两个按钮和「模板」同一样式，拼成一个整体、由同一圈彩虹亮边（entry-rim，
 * 和「模板」的亮边同一套动画）框住——看得出是一组二选一；当前模式高亮（.on）。 */
export function ModeSwitch() {
  const account = useSession((s) => s.state?.user?.id ?? null);
  const mode = useAppMode((s) => s.mode);
  const setMode = useAppMode((s) => s.setMode);
  useEffect(() => useAppMode.getState().follow(account), [account]);
  return (
    <span className="mode-switch entry-rim" role="group" aria-label="模式">
      <Button on={mode === "template"} tip="节点模式：看得到节点图，能改任何节点的参数、做模板" onClick={() => setMode("template")}>
        节点模式
      </Button>
      <Button on={mode === "app"} tip="应用模式：只看模板公开的参数，选好素材点「计算」；节点图收起来。按账号记在这个浏览器里" onClick={() => setMode("app")}>
        应用模式
      </Button>
    </span>
  );
}
