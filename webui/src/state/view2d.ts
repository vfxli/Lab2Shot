// The 2D view (视图 state): its zoom and pan per slot, and the navigation hook that drives them. Re-exported by
// state/viewer.ts.

import { useEffect, useRef, useState } from "react";
import { create } from "zustand";
import { fitTransform, oneToOneTransform, panBy, toPercent, zoomAt, type Mode, type Transform2D } from "../model/view2d";
import { useElementSize } from "../platform/size";
import { useShortcut } from "../platform/keys";
import { followDrag } from "../platform/drag";

// ------------------------------------------------------------------ the 2D view

export type ViewSlot = "2d" | "look";

/** 视图的缩放与平移：每个视图一份，仅由用户修改（拖动、滚轮、适应、1:1、百分比）。
 *
 * 变换保持独立：切换显示的节点、画面尺寸或容器尺寸变化时均不修改变换，不做任何补偿性重算。
 * 若视图随数据和容器变化，切换节点或工具栏换行都会导致图像在屏幕上移位，无法在两个节点间来回对比。
 * 仅在切换镜头（`key` 改变）时重新适应一次。 */
export interface View2D {
  at: Transform2D;
  key: string; // 镜头标识（切换镜头时才重新适应一次）
}

interface View2DState {
  // 视图当前的档位：仅原图 / 运算 / 仅结果（model/view2d.ts Mode）。
  // 双击显示某个节点时，按该节点自带的预览标签切换一次（NodeTypeDef.preview，由服务器计算，一条标签
  // 同时适用于 2D 和 3D）；之后若用户手动切换，则以用户选择为准，直到切换到下一个节点
  mode: Mode;
  setMode: (mode: Mode) => void;
  // 左右两侧各自选取的通道（null：整体，多条通道一起查看）。层另行管理：右侧所看节点的层为
  // `state/look.ts` 的 `displayPort`（需随节点图保存），左侧只有一层（原图），因此此处只保存通道。
  // 通道属于查看方式，随视图保存，切换节点后仍保留
  left: number | null;
  setLeft: (index: number | null) => void;
  right: number | null;
  setRight: (index: number | null) => void;
  views: Record<ViewSlot, View2D | null>;
  dropView: (slot: ViewSlot) => void;
  zoomPercent: number;
  navigable: boolean;
  fitAsk: number;
  oneAsk: number;
  goAsk: { pct: number; n: number };
  fit: () => void;
  one: () => void;
  goTo: (pct: number) => void;
}

export const useView2D = create<View2DState>((set) => ({
  mode: "plate",
  setMode: (mode) => set({ mode }),
  left: null,
  setLeft: (left) => set({ left }),
  right: null,
  setRight: (right) => set({ right }),
  views: { "2d": null, look: null },
  dropView: (slot) => set((s) => ({ views: { ...s.views, [slot]: null } })),
  zoomPercent: 100,
  navigable: false,
  fitAsk: 0,
  oneAsk: 0,
  goAsk: { pct: 100, n: 0 },
  fit: () => set((s) => ({ fitAsk: s.fitAsk + 1 })),
  one: () => set((s) => ({ oneAsk: s.oneAsk + 1 })),
  goTo: (pct) => set((s) => ({ goAsk: { pct, n: s.goAsk.n + 1 } })),
}));

export interface Nav2D {
  at: Transform2D | null;
  box: { w: number; h: number };
  cursor: "grab" | "grabbing" | null;
  onMouseLeave: () => void;
  onMouseMove: (e: React.MouseEvent) => void;
  onMouseDown: (e: React.MouseEvent) => boolean;
}

interface NavOptions {
  slot?: ViewSlot;
  key?: string;
  keys?: (key: string) => boolean;
  altLeft?: boolean;
}

/** The 2D view's navigation, shared by every way the view is drawn: the wheel zooms around the cursor, a middle-drag
 * (or an Alt+left-drag) pans, and F fits (plus `keys` for the stage's own keys) while the pointer is over `el`; the zoom
 * bar's requests are applied here, where the picture's size (`width` x `height` image pixels) and the stage's size are known. */
export function useView2DNav(el: HTMLElement | null, width: number, height: number, { slot = "2d", key = "", keys, altLeft = true }: NavOptions = {}): Nav2D {
  const slotView = useView2D((s) => s.views[slot]);
  const stored = slotView && slotView.key === key ? slotView : null;
  const fitAsk = useView2D((s) => s.fitAsk);
  const oneAsk = useView2D((s) => s.oneAsk);
  const goAsk = useView2D((s) => s.goAsk);
  const box = useElementSize(el);
  const [cursor, setCursor] = useState<"grab" | "grabbing" | null>(null);
  const keysRef = useRef(keys);
  keysRef.current = keys;


  const ready = !!el && width > 0 && height > 0 && box.w > 0 && box.h > 0;
  // 已保存的变换原样使用：切换节点、切换档位、工具栏换行、面板拖宽时一律不重算（参见上方 View2D 的注释）
  const at = ready ? (stored?.at ?? fitTransform(width, height, box.w, box.h)) : null;
  const store = (s: ViewSlot, v: View2D) => useView2D.setState((st) => ({ views: { ...st.views, [s]: v } }));
  const put = (t: Transform2D) => store(slot, { at: t, key });
  // listeners bound once modify the view as it currently is (two mouse moves may arrive before a render)
  const now = useRef({ width, height, at, slot, key, box });
  now.current = { width, height, at, slot, key, box };
  const change = (f: (t: Transform2D) => Transform2D) => {
    const { width: w, height: h, at: shown, slot: sl, key: k } = now.current;
    const v = useView2D.getState().views[sl];
    const t = v && v.key === k ? v.at : shown;
    if (t && w > 0 && h > 0) store(sl, { at: f(t), key: k });
  };

  useEffect(() => {
    if (ready && !stored) put(at!); // the first picture shown is fitted; afterwards the view belongs to the user
    if (at) useView2D.setState((s) => (s.zoomPercent === Math.round(at.s * 100) ? s : { zoomPercent: Math.round(at.s * 100) }));
  });
  useEffect(() => {
    useView2D.setState({ navigable: ready });
    return () => useView2D.setState({ navigable: false });
  }, [ready]);

  // the zoom bar's requests, applied once each (a request made before this stage was shown is not replayed)
  const seen = useRef({ fit: fitAsk, one: oneAsk, go: goAsk.n });
  useEffect(() => {
    const s = seen.current;
    seen.current = { fit: fitAsk, one: oneAsk, go: goAsk.n };
    if (!at) return;
    if (fitAsk !== s.fit) put(fitTransform(width, height, box.w, box.h));
    else if (oneAsk !== s.one) put(oneToOneTransform(width, height, box.w, box.h, window.devicePixelRatio || 1));
    else if (goAsk.n !== s.go) put(toPercent(at, goAsk.pct, box.w, box.h));
  }, [fitAsk, oneAsk, goAsk]); // eslint-disable-line react-hooks/exhaustive-deps

  // the wheel: a dedicated non-passive listener, so it can prevent page scrolling
  useEffect(() => {
    if (!el || !ready) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      change((t) => zoomAt(t, Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [el, ready]);

  // F fits, and the stage's own keys, while the pointer is over the stage (platform/keys.ts)
  useShortcut(
    {
      keys: ["*"],
      run: (e) => {
        const k = e.key.toLowerCase();
        if (k === "f") return void useView2D.getState().fit();
        return keysRef.current?.(k) ?? false;
      },
    },
    { over: () => el },
  );

  return {
    at,
    box,
    cursor,
    onMouseLeave: () => setCursor(null),
    onMouseMove: (e) => {
      if (altLeft && e.altKey !== (cursor === "grab") && cursor !== "grabbing") setCursor(e.altKey ? "grab" : null);
    },
    onMouseDown: (e) => {
      if (!at || !(e.button === 1 || (altLeft && e.button === 0 && e.altKey))) return false;
      e.preventDefault(); // no page autoscroll on the middle button, no text selection on Alt+drag
      setCursor("grabbing");
      let last = { x: e.clientX, y: e.clientY };
      followDrag(
        (ev) => {
          const [dx, dy] = [ev.clientX - last.x, ev.clientY - last.y];
          change((t) => panBy(t, dx, dy));
          last = { x: ev.clientX, y: ev.clientY };
        },
        (ev) => setCursor(ev?.altKey ? "grab" : null),
      );
      return true;
    },
  };
}
