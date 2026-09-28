// The 3D view's display options (kept in this browser), its camera, the viewer's short notices, the load state, and
// the ruler's window (视图 state). Re-exported by state/viewer.ts.

import { create } from "zustand";
import type { NoticeKind } from "../model/viewNotices";
import type { Message } from "../messages/message";
import { DEFAULTS, loadOptions, sanitize, saveOptions, STORAGE_KEY, type ViewOptions } from "../model/viewOptions";

// ------------------------------------------------------------------ display options, the 3D camera and related state

interface OptionsState {
  o: ViewOptions;
  set: (patch: Partial<ViewOptions>) => void;
  reset: (keys?: (keyof ViewOptions)[]) => void;
}

const storage = (): Storage | null => {
  try {
    return window.localStorage;
  } catch {
    return null; // storage blocked: the options last only for the page's lifetime
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
export const VIEW_NAMES: Record<ViewName, string> = { persp: "透视", top: "顶", front: "前", side: "侧" };

interface CameraState {
  view: ViewName;
  ortho: boolean;
  look: string | null;
  frameAsk: { what: "all" | "selected"; n: number };
  viewAsk: number;
  /** 场景中的相机列表及当前选中项：由三维舞台计算后存放于此，工具栏据此绘制「视角」控件。
   * 视角与框显属于视图工具，与 2D / 3D 等位于同一排工具栏，不浮在画面上。 */
  cameras: { key: string; label: string; width: number; height: number }[];
  selected: string | null;  // 视图中选中物体的名称（「框显选中」据此启用或置灰）
  setView: (v: ViewName) => void;
  setOrtho: (on: boolean) => void;
  setLook: (camera: string | null) => void;
  frame: (what: "all" | "selected") => void;
  setCameras: (list: CameraState["cameras"]) => void;
  setSelected: (label: string | null) => void;
}

export const useViewCamera = create<CameraState>((set, get) => ({
  view: "persp",
  ortho: false,
  look: null,
  frameAsk: { what: "all", n: 0 },
  viewAsk: 0,
  setView: (view) => set({ view, ortho: view !== "persp", look: null, viewAsk: get().viewAsk + 1 }),
  setOrtho: (ortho) => set({ ortho }),
  setLook: (look) => set(look ? { look, view: "persp", ortho: false } : { look: null }),
  frame: (what) => set({ look: null, frameAsk: { what, n: get().frameAsk.n + 1 } }),
  cameras: [],
  selected: null,
  setCameras: (cameras) => set((s) => (JSON.stringify(s.cameras) === JSON.stringify(cameras) ? s : { cameras })),
  setSelected: (selected) => set((s) => (s.selected === selected ? s : { selected })),
}));

/** A short notice the viewer shows briefly after changing something by itself (the stage switched because the new node
 * has nothing to show on the user's current stage, or a looked-through camera does not exist on the node): a few words
 * in the pill, with the full reason in its tooltip. */
export const useViewerNote = create<{ note: Message | null; why: Message | string; n: number; say: (note: Message, why?: Message | string) => void }>((set, get) => ({
  note: null,
  why: "", // the full reason, shown in the pill's tooltip; the pill itself stays one short line
  n: 0,
  say: (note, why = "") => set({ note, why, n: get().n + 1 }),
}));

/** The frames of the displayed 3D view whose data is currently in the browser (the timeline's 已载入视图 row); null:
 * nothing shown is loaded frame by frame (it arrives together with the rest). */
/** `catchingUp`：本轮放弃实时播放：待播放的帧尚未传到，因此逐帧等待拉取而不跳帧（与 DCC 相同：
 * 缓存未就绪的第一遍不实时，第二遍才实时）。此状态在统一的视图通知区显示，不在视图上另绘控件。 */
/** `loaded`：该源中无需网络即可绘制的帧（字节在内存中，或原件在本机磁盘上），对应时间线上「已载入视图」的颜色。
 * `decoded`：二维主源中已解码、可立即绘制的帧，播放器只依据此项。「在磁盘上」与「已解码」必须区分：
 * 若合并为一项，播放器会将本机文件的每一帧视为已就绪，按 24 fps 强行推进，解码跟不上时持续显示上一帧，
 * 而不是暂停等待。三维舞台不设置此项（null）。 */
/** `pending`：三维舞台中逐帧数据块尚未到达的帧（view/scene.ts pending）。播放器播放到这些帧时需等待，与二维取帧账本中
 * 「正在读取」含义相同（三维数据块不经过该账本，因此单独报告）。二维舞台不设置此项（null）。 */
export const useViewLoads = create<{ loaded: number[] | null; decoded: number[] | null; pending: number[] | null; stale: boolean; catchingUp: boolean }>(
  () => ({ loaded: null, decoded: null, pending: null, stale: false, catchingUp: false }),
);
// 供探针读取（开发用途，非页面功能）：`window.__l2s_loads()` 返回此状态
if (typeof window !== "undefined") (window as unknown as { __l2s_loads?: () => unknown }).__l2s_loads = () => useViewLoads.getState();
// `stale`：二维主源绘制的是上一次的结果（参数已修改，state/stale.ts），时间线色带据此显示为土黄色

/** 二维舞台当前绘制的画面尺寸（图像像素，而非屏幕像素；null 表示当前没有二维舞台）。
 *
 * 用途：部分控件不在舞台内，却需要在画面像素坐标中放置内容。例如参数面板的「添加帧」
 * 按下后需在画面正中放置一个火柴人，因此需要知道画面尺寸（`view/figure2d.ts tposeIn`）。
 * 二维舞台（`view/Stage2D.tsx`，按 `model/view2d.ts pictureSize`）算出的宽高是唯一来源，
 * 不在其他位置重复计算。其性质与上方「已载入视图」相同：
 * 舞台绘制时发布该值，舞台卸载时清除。 */
export const useStagePicture = create<{ size: { width: number; height: number } | null }>(() => ({ size: null }));

/** The frames shown by the timeline's ruler (zoomed with the wheel, moved by dragging) for the shot `key` (its first
 * and last frame): the curve editor's frame axis uses the same window, so a frame is at the same position in both. */
export interface FrameWindow {
  start: number;
  end: number;
}
export const useRulerView = create<{ zoom: { key: string; view: FrameWindow } | null; setZoom: (z: { key: string; view: FrameWindow } | null) => void }>(
  (set) => ({ zoom: null, setZoom: (zoom) => set({ zoom }) }),
);

/** 显示原始: whether the curve editor draws, under each changed channel, the channel as received (a packet's
 * `before`). 视图 state: belongs to the browser, not the document; never undoable, never sent. */
export const useCurveView = create<{ original: boolean; setOriginal: (on: boolean) => void }>((set) => ({
  original: true,
  setOriginal: (original) => set({ original }),
}));

try {
  window.addEventListener("storage", (e) => {
    if (e.key === STORAGE_KEY) useViewOptions.setState({ o: loadOptions(storage()) });
  });
} catch {
  // no window events (not a browser): nothing to follow
}

/** 舞台需要在画面上显示的文字均放在此处（统一通知区，所有此类通知都经由它）。二维舞台、三维舞台、本机画面各自放入
 * 自己的一条，`editor/ViewerFrame.tsx` 统一绘制在舞台左上角（ui/ViewNotices.tsx）。其他位置不得在画面上绘制文字。
 * 按 kind 存储：同一种类只保留一条，内容变化时替换该条，位置保持不变。 */
export const useStageNotes = create<{
  notes: Partial<Record<NoticeKind, { text: string; tip: string }>>;
  put: (kind: NoticeKind, note: { text: string; tip: string } | null) => void;
}>((set) => ({
  notes: {},
  put: (kind, note) =>
    set((s) => {
      const now = s.notes[kind];
      if (!note) return now === undefined ? s : { notes: { ...s.notes, [kind]: undefined } };
      return now && now.text === note.text && now.tip === note.tip ? s : { notes: { ...s.notes, [kind]: note } };
    }),
}));
