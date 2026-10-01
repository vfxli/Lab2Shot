/** Follows a drag the page started on a press (a splitter, a strip's grip, a pan): `move` on every pointer move, then
 * `end` once, when the button is released, the pointer is cancelled (a touch taken over by the system) or the window
 * loses focus (switching away mid-drag, where no release ever arrives). Returns what ends it early (an unmount). */
export function followDrag(move: (e: PointerEvent) => void, end: (e: PointerEvent | null) => void): () => void {
  const up = (e: PointerEvent) => stop(e);
  const lost = () => stop(null);
  let ended = false; // `end` once: an unmount's cleanup calling the returned stop after the release does nothing
  function stop(e: PointerEvent | null): void {
    if (ended) return;
    ended = true;
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", up);
    window.removeEventListener("pointercancel", up);
    window.removeEventListener("blur", lost);
    end(e);
  }
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", up);
  window.addEventListener("pointercancel", up);
  window.addEventListener("blur", lost);
  return () => stop(null);
}

/** How far (screen pixels, in any direction) a press may wander and still be a click. */
export const CLICK_SLOP = 4;

/** How a press ended: "click" released within CLICK_SLOP of where it was pressed, "drag" released after moving further,
 * "lost" never released (cancelled, the window lost focus): neither is acted on. */
export type PressEnd = "click" | "drag" | "lost";

/** One press, as far as the page's one rule sees it: `see` each pointer position; it becomes a drag once it is further
 * than CLICK_SLOP (straight-line distance, in any direction) from where it was pressed, and stays one. followPress below and the 3D gizmo
 * (view/dragGizmo.tsx, which follows the pointer its own way: three's TransformControls) both decide by it. */
export function pressAt(from: { clientX: number; clientY: number }): { see: (e: { clientX: number; clientY: number }) => boolean; readonly dragged: boolean } {
  let dragged = false;
  return {
    see: (e) => (dragged ||= Math.hypot(e.clientX - from.clientX, e.clientY - from.clientY) > CLICK_SLOP),
    get dragged() {
      return dragged;
    },
  };
}

/** The page's one rule for a press that may be a click or a drag (the 2D handles, the right-button zoom, the 3D stage's
 * pick; the 3D gizmo uses pressAt directly): `move` runs only once the press is a drag, on every pointer move after
 * that; `end` runs once, saying which it was. */
export function followPress(from: { clientX: number; clientY: number }, move: (e: PointerEvent) => void,
                            end: (how: PressEnd, e: PointerEvent | null) => void): () => void {
  const press = pressAt(from);
  return followDrag(
    (e) => {
      if (press.see(e)) move(e);
    },
    (e) => end(!e || e.type !== "pointerup" ? "lost" : press.dragged ? "drag" : "click", e),
  );
}

/** A press watched only to read, at its release, whether it was a drag (the node graph, whose gestures themselves are
 * @xyflow/react's): the same pressAt, fed until the button is let go. */
export function watchPress(from: { clientX: number; clientY: number }): { readonly dragged: boolean } {
  const press = pressAt(from);
  followDrag((e) => void press.see(e), () => undefined);
  return press;
}
