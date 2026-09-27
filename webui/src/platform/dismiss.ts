import { useEffect, useRef, type RefObject } from "react";
import { useKeyLayer, useShortcut } from "./keys";

/** A popup's one way of closing: while `open`, a press anywhere outside `inside` (its
 * element, or a CSS selector the pressed element must be inside) or Escape calls `close`. The press is caught on its
 * way down, so the click that closes a menu does not also reach what is under it first. */
export function useDismiss(open: boolean, inside: RefObject<HTMLElement | null> | string, close: () => void): void {
  const closing = useRef(close);
  closing.current = close;
  useEffect(() => {
    if (!open) return;
    const within = (target: EventTarget | null) =>
      typeof inside === "string" ? !!(target as Element | null)?.closest?.(inside) : !!inside.current?.contains(target as Node);
    const away = (e: PointerEvent) => !within(e.target) && closing.current();
    window.addEventListener("pointerdown", away, true);
    return () => window.removeEventListener("pointerdown", away, true);
  }, [open, inside]);
  // Esc through the keyboard registry: an open popup is a layer (nothing under it hears keys while it is open)
  const layer = useKeyLayer(open);
  useShortcut({ keys: ["escape"], inText: true, run: () => closing.current() }, { layer, enabled: open });
}
