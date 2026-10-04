// 三维视图的显示选项（存在本浏览器里）、它的相机、视图的简短通知、逐帧就绪状态，以及时间尺的窗口（视图状态）。
// 由 state/viewer.ts 转出。

import { create } from "zustand";
import { same, type Json } from "../model/graphPatch";
import type { NoticeKind } from "../model/viewNotices";
import type { Message } from "../messages/message";
import { DEFAULTS, loadOptions, sanitize, saveOptions, STORAGE_KEY, type ViewOptions } from "../model/viewOptions";
import { t } from "../i18n/t";
import type { Tip } from "../platform/tips";

// ------------------------------------------------------------------ 显示选项、三维相机及相关状态

interface OptionsState {
  o: ViewOptions;
  set: (patch: Partial<ViewOptions>) => void;
  reset: (keys?: (keyof ViewOptions)[]) => void;
}

const storage = (): Storage | null => {
  try {
    return window.localStorage;
  } catch {
    return null; // 存储被禁用：选项只在页面存续期间有效
  }
};

export const useViewOptions = create<OptionsState>((set, get) => ({
  o: loadOptions(storage()),
  set: (patch) => {
    const o = sanitize({ ...get().o, ...patch });
    saveOptions(storage(), o);
    set({ o });
  },
  reset: (keys) => {
    const o = keys ? sanitize({ ...get().o, ...Object.fromEntries(keys.map((k) => [k, DEFAULTS[k]])) }) : { ...DEFAULTS };
    saveOptions(storage(), o);
    set({ o });
  },
}));

export type ViewName = "persp" | "top" | "front" | "side";
/** The views' names, each said when read (so they follow the page's language). */
export const VIEW_NAMES: Readonly<Record<ViewName, string>> = {
  get persp() { return t("ui.state.view_persp"); },
  get top() { return t("ui.state.view_top"); },
  get front() { return t("ui.state.view_front"); },
  get side() { return t("ui.state.view_side"); },
};

/** One 3D stage's camera choices. Each stage has its own camera slot: the viewer's ("viewer", driven by the toolbar) and
 * any stage in a dialog, so a view or projection set in one never changes the other's. */
export interface SlotCamera {
  view: ViewName;
  ortho: boolean;
  frameAsk: { what: "all" | "selected"; n: number };
  viewAsk: number;
}
export const VIEWER_SLOT = "viewer";
const FREE: SlotCamera = { view: "persp", ortho: false, frameAsk: { what: "all", n: 0 }, viewAsk: 0 };
/** A slot's choices (the default free perspective view until something is set). */
export const slotCamera = (s: Pick<CameraState, "slots">, slot: string): SlotCamera => s.slots[slot] ?? FREE;

/** A scene camera looked through: `key` tells this copy of it apart (its packet and path), `path` is where it sits in
 * the hierarchy (the same camera in a recooked packet has the same path). */
export interface LookAt {
  key: string;
  path: string;
}

interface CameraState {
  slots: Record<string, SlotCamera>;
  look: LookAt | null; // the viewer's only: which scene camera it looks through
  /** 场景中的相机列表及当前选中项：由三维舞台计算后存放于此，工具栏据此绘制「视角」控件。
   * 视角与框显属于视图工具，与 2D / 3D 等位于同一排工具栏，不浮在画面上。 */
  cameras: (LookAt & { label: string; width: number; height: number })[];
  selected: string | null;  // 视图中选中物体的名称（「框显选中」据此启用或置灰）
  setView: (v: ViewName, slot?: string) => void;
  setOrtho: (on: boolean, slot?: string) => void;
  setLook: (camera: LookAt | null) => void;
  frame: (what: "all" | "selected", slot?: string) => void;
  forget: (slot: string) => void; // a stage that is gone for good (a closed dialog): its next opening starts afresh
  setCameras: (list: CameraState["cameras"]) => void;
  setSelected: (label: string | null) => void;
}

export const useViewCamera = create<CameraState>((set) => {
  const patch = (slot: string, p: (c: SlotCamera) => Partial<SlotCamera>, extra: Partial<CameraState> = {}) =>
    set((s) => { const c = slotCamera(s, slot); return { ...extra, slots: { ...s.slots, [slot]: { ...c, ...p(c) } } }; });
  // looking through a scene camera is the viewer's: a view or framing asked of the viewer leaves it
  const leave = (slot: string) => (slot === VIEWER_SLOT ? { look: null } : {});
  return {
  slots: {},
  look: null,
  setView: (view, slot = VIEWER_SLOT) => patch(slot, (c) => ({ view, ortho: view !== "persp", viewAsk: c.viewAsk + 1 }), leave(slot)),
  setOrtho: (ortho, slot = VIEWER_SLOT) => patch(slot, () => ({ ortho })),
  setLook: (look) => (look ? patch(VIEWER_SLOT, () => ({ view: "persp", ortho: false }), { look: { key: look.key, path: look.path } }) : set({ look: null })),
  frame: (what, slot = VIEWER_SLOT) => patch(slot, (c) => ({ frameAsk: { what, n: c.frameAsk.n + 1 } }), leave(slot)),
  forget: (slot) => set((s) => { const slots = { ...s.slots }; delete slots[slot]; return { slots }; }),
  cameras: [],
  selected: null,
  setCameras: (cameras) => set((s) => (same(s.cameras as unknown as Json, cameras as unknown as Json) ? s : { cameras })),
  setSelected: (selected) => set((s) => (s.selected === selected ? s : { selected })),
  };
});

/** 视图自行改动了什么之后短暂显示的简短通知（新节点在用户当前的舞台上没有可显示的内容而换了舞台，或透过的相机在该节点上
 * 不存在）：胶囊里几个字，完整原因在它的悬停提示里。 */
export const useViewerNote = create<{ note: Message | null; why: Message | string; n: number; say: (note: Message, why?: Message | string) => void }>((set, get) => ({
  note: null,
  why: "", // 完整原因，显示在胶囊的悬停提示里；胶囊本身只占短短一行
  n: 0,
  say: (note, why = "") => set({ note, why, n: get().n + 1 }),
}));

/** 视图的逐帧就绪情况，由 transfer/readiness.ts 写（reportLoads / clearLoads），播放器经 readiness.playable 读：
 * `loaded` 无须再下载的帧（时间线「已载入视图」的颜色）；`ready` 可立即画的帧（二维：所有在用的格都已解码——
 * 「在本机磁盘上」不等于「已解码」，否则播放器把本机序列当就绪，解码跟不上时一直显示上一帧）；`waiting` 在途的帧；
 * `stale` 画的是上一次的结果；`cached` 三维整段缓存进度 [已在本机的帧数, 总帧数]；`catchingUp` 本轮放弃实时、逐帧等待。 */
export const useViewLoads = create<{ loaded: number[] | null; ready: number[] | null; waiting: number[] | null; stale: boolean; catchingUp: boolean;
                                     cached: [number, number] | null }>(
  () => ({ loaded: null, ready: null, waiting: null, stale: false, catchingUp: false, cached: null }),
);
// `stale`：二维主源绘制的是上一次的结果（参数已修改，state/stale.ts），时间线色带据此显示为土黄色

/** 二维舞台当前绘制的画面尺寸（图像像素，而非屏幕像素；null 表示当前没有二维舞台）。
 *
 * 用途：部分控件不在舞台内，却需要在画面像素坐标中放置内容。例如参数面板的「添加帧」
 * 按下后需在画面正中放置一个火柴人，因此需要知道画面尺寸（`view/figure2d.ts tposeIn`）。
 * 二维舞台（`view/Stage2D.tsx`，按 `model/view2d.ts pictureSize`）算出的宽高是唯一来源，
 * 不在其他位置重复计算。其性质与上方「已载入视图」相同：
 * 舞台绘制时发布该值，舞台卸载时清除。 */
export const useStagePicture = create<{ size: { width: number; height: number } | null }>(() => ({ size: null }));

/** 镜头 `key`（它的首帧与末帧）在时间线时间尺上显示的帧范围（滚轮缩放、拖动平移）：曲线编辑器的帧轴用同一个窗口，
 * 所以同一帧在两处位置相同。 */
export interface FrameWindow {
  start: number;
  end: number;
}
export const useRulerView = create<{ zoom: { key: string; view: FrameWindow } | null; setZoom: (z: { key: string; view: FrameWindow } | null) => void }>(
  (set) => ({ zoom: null, setZoom: (zoom) => set({ zoom }) }),
);

/** 显示原始：曲线编辑器是否在每条改过的通道下面画出收到时的那条通道（包的 `before`）。视图状态：属于浏览器而不属于
 * 文档；不可撤销，不发送。 */
export const useCurveView = create<{ original: boolean; setOriginal: (on: boolean) => void }>((set) => ({
  original: true,
  setOriginal: (original) => set({ original }),
}));

try {
  window.addEventListener("storage", (e) => {
    if (e.key === STORAGE_KEY) useViewOptions.setState({ o: loadOptions(storage()) });
  });
} catch {
  // 没有 window 事件（不在浏览器里）：无需跟随
}

/** 舞台需要在画面上显示的文字均放在此处（统一通知区，所有此类通知都经由它）。二维舞台、三维舞台、本机画面各自放入
 * 自己的一条，`editor/ViewerFrame.tsx` 统一绘制在舞台左上角（ui/ViewNotices.tsx）。其他位置不得在画面上绘制文字。
 * 按 kind 存储：同一种类只保留一条，内容变化时替换该条，位置保持不变。 */
export const useStageNotes = create<{
  notes: Partial<Record<NoticeKind, { text: string; tip?: Tip | null }>>;
  put: (kind: NoticeKind, note: { text: string; tip?: Tip | null } | null) => void;
}>((set) => ({
  notes: {},
  put: (kind, note) =>
    set((s) => {
      const now = s.notes[kind];
      if (!note) return now === undefined ? s : { notes: { ...s.notes, [kind]: undefined } };
      return now && now.text === note.text && now.tip?.text === note.tip?.text && now.tip?.why === note.tip?.why ? s : { notes: { ...s.notes, [kind]: note } };
    }),
}));
