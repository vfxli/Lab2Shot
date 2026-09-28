/** Follows a drag the page started on a press (a splitter, a strip's grip, a pan): `move` on every pointer move, then
 * `end` once, when the button is released, the pointer is cancelled (a touch taken over by the system) or the window
 * loses focus (switching away mid-drag, where no release ever arrives). Returns what ends it early (an unmount). */
export function followDrag(move: (e: PointerEvent) => void, end: (e: PointerEvent | null) => void): () => void {
  const up = (e: PointerEvent) => stop(e);
  const lost = () => stop(null);
  function stop(e: PointerEvent | null): void {
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
