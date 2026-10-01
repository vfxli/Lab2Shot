// The 2D view (视图 state): its zoom and pan per slot, and the navigation hook that drives them. Re-exported by
// state/viewer.ts.

import { useEffect, useRef, useState } from "react";
import { create } from "zustand";
import { fitTransform, oneToOneTransform, panBy, toPercent, zoomAt, type Mode, type Transform2D } from "../model/view2d";
import { useElementSize } from "../platform/size";
import { useShortcut } from "../platform/keys";
import { followDrag, followPress } from "../platform/drag";

// ------------------------------------------------------------------ the 2D view

export type ViewSlot = "2d" | "look";

/** A view's zoom and pan: one per view, changed only by the user (drag, wheel, fit, 1:1, a percentage).
 *
 * The transform stands on its own: switching the displayed node, or a change of picture size or container size, never
 * alters it, and nothing is recomputed to compensate. A view that followed the data and the container would shift the
 * image on screen whenever the node changes or the toolbar wraps, making it impossible to compare two nodes back and
 * forth. It is fitted again only when the shot changes (`key` changes). */
export interface View2D {
  at: Transform2D;
  key: string; // the shot's identity (the view is fitted again only when it changes)
}

interface View2DState {
  // the view's current mode: plate only / operation / result only (model/view2d.ts Mode). Double-clicking a node to
  // display it switches once to that node's own preview tag (NodeTypeDef.preview, computed by the server; one tag
  // serves both 2D and 3D); after that a manual switch by the user holds until the next node is displayed
  mode: Mode;
  setMode: (mode: Mode) => void;
  // the channel picked on each side (null: the whole, several channels viewed together). Layers are kept elsewhere: the
  // right side's layer is `state/look.ts` `displayPort` (saved with the graph), and the left side has only one layer
  // (the plate), so only channels are kept here. A channel is a way of viewing: it is kept with the view and survives
  // switching nodes
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
  cursor: "grab" | "grabbing" | "ew-resize" | null;
  onMouseLeave: () => void;
  onMouseMove: (e: React.MouseEvent) => void;
  onMouseDown: (e: React.MouseEvent) => boolean;
  /** Keeps the browser's menu off the view. It is never the stage's right click: macOS and Linux send it on the press,
   * before anyone knows whether a zoom drag follows (NavOptions.onRightPress gives the click). */
  onContextMenu: (e: React.MouseEvent) => void;
}

interface NavOptions {
  slot?: ViewSlot;
  key?: string;
  keys?: (key: string) => boolean;
  altLeft?: boolean;
  // asked when the right button goes down: what a right click (platform/drag.ts followPress: released without dragging)
  // then does, with where it was pressed (the stage removes the point there). Asked at the press, so what it returns is
  // bound to the stage as it was then (its frame and entries), not as it is at the release (playback may move on)
  onRightPress?: () => ((at: { clientX: number; clientY: number }) => void) | null;
}

/** The 2D view's navigation, shared by every way the view is drawn, with Houdini's buttons: the wheel zooms around the
 * cursor, a middle-drag (or an Alt+left-drag) pans, a right-drag zooms around the point pressed (to the right zooms in,
 * to the left out), and F fits (plus `keys` for the stage's own keys) while the pointer is over `el`; the zoom bar's
 * requests are applied here, where the picture's size (`width` x `height` image pixels) and the stage's size are known. */

/** Right-drag zoom: the scale changes by e^(pixels × this); 200 px to the right is about 2.7×, as the wheel's feel. */
const ZOOM_PER_PX = 0.005;
export function useView2DNav(el: HTMLElement | null, width: number, height: number, { slot = "2d", key = "", keys, altLeft = true, onRightPress }: NavOptions = {}): Nav2D {
  const slotView = useView2D((s) => s.views[slot]);
  const stored = slotView && slotView.key === key ? slotView : null;
  const fitAsk = useView2D((s) => s.fitAsk);
  const oneAsk = useView2D((s) => s.oneAsk);
  const goAsk = useView2D((s) => s.goAsk);
  const box = useElementSize(el);
  const [cursor, setCursor] = useState<"grab" | "grabbing" | "ew-resize" | null>(null);
  const keysRef = useRef(keys);
  keysRef.current = keys;
  const rightPress = useRef(onRightPress);
  rightPress.current = onRightPress;


  const ready = !!el && width > 0 && height > 0 && box.w > 0 && box.h > 0;
  // a stored transform is used as is: never recomputed on switching node or mode, a toolbar wrap or a panel resize (see
  // View2D above)
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
      if (!at) return false;
      if (e.button === 2) {
        e.preventDefault();
        const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
        const [px, py] = [e.clientX - r.left, e.clientY - r.top];
        // a click or a zoom drag, as platform/drag.ts followPress decides (on release, the same on every platform); the
        // click acts where the button was pressed
        const at0 = { clientX: e.clientX, clientY: e.clientY };
        // the click acts on the stage as it was when pressed (NavOptions.onRightPress)
        const click = rightPress.current?.() ?? null;
        let last = e.clientX;
        followPress(
          at0,
          (ev) => {
            setCursor("ew-resize");
            const dx = ev.clientX - last;
            last = ev.clientX;
            if (dx) change((t) => zoomAt(t, Math.exp(dx * ZOOM_PER_PX), px, py));
          },
          (how, ev) => {
            setCursor(ev?.altKey ? "grab" : null);
            if (how === "click") click?.(at0);
          },
        );
        return true;
      }
      if (!(e.button === 1 || (altLeft && e.button === 0 && e.altKey))) return false;
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
    onContextMenu: (e) => e.preventDefault(),
  };
}
