import { useEffect, useState } from "react";

/** Element sizes on the page: the single place that observes element sizes.
 *
 * One ResizeObserver for the whole page. Its callback only notes which elements changed; the sizes are read and handed
 * out once per animation frame, rounded to whole pixels, and only to watchers whose rounded size really changed. So a
 * size that makes a component re-render and resize what is watched is answered in the next frame, never inside the
 * observer's own delivery (the browser's "ResizeObserver loop completed with undelivered notifications"), and dragging a
 * splitter costs one update per frame per element however many notifications the browser sends.
 * No other file creates a ResizeObserver. */

export interface Size {
  w: number;
  h: number;
}

const NO_SIZE: Size = Object.freeze({ w: 0, h: 0 });

interface Watched {
  size: Size;
  listeners: Set<(s: Size) => void>;
}

/** The browser facilities the watcher needs, handed in rather than reached for (`browser()` below supplies them). */
interface SizeHost {
  observe: (onChange: (elements: Element[]) => void) => { observe: (el: Element) => void; unobserve: (el: Element) => void };
  measure: (el: Element) => Size;
  frame: (run: () => void) => void;
}

const browser = (): SizeHost => ({
  observe: (onChange) => {
    const ro = new ResizeObserver((entries) => onChange(entries.map((e) => e.target)));
    return { observe: (el) => ro.observe(el), unobserve: (el) => ro.unobserve(el) };
  },
  measure: (el) => ({ w: Math.round((el as HTMLElement).clientWidth), h: Math.round((el as HTMLElement).clientHeight) }),
  frame: (run) => void requestAnimationFrame(run),
});

/** A page's size watcher: `watch(el, f)` calls `f` with the element's size now and whenever its rounded size changes,
 * at most once per frame; returns what stops it. */
function sizeWatcher(host: SizeHost) {
  const watched = new Map<Element, Watched>();
  const dirty = new Set<Element>();
  let scheduled = false;
  let observer: ReturnType<SizeHost["observe"]> | null = null;

  const flush = () => {
    scheduled = false;
    const now = [...dirty];
    dirty.clear();
    for (const el of now) {
      const w = watched.get(el);
      if (!w) continue;
      const size = host.measure(el);
      if (size.w === w.size.w && size.h === w.size.h) continue;
      w.size = size;
      w.listeners.forEach((f) => f(size));
    }
  };
  const changed = (elements: Element[]) => {
    for (const el of elements) if (watched.has(el)) dirty.add(el);
    if (dirty.size && !scheduled) {
      scheduled = true;
      host.frame(flush);
    }
  };

  return function watch(el: Element, listener: (s: Size) => void): () => void {
    observer ??= host.observe(changed);
    let w = watched.get(el);
    if (!w) {
      w = { size: host.measure(el), listeners: new Set() };
      watched.set(el, w);
      observer.observe(el);
    }
    w.listeners.add(listener);
    listener(w.size);
    return () => {
      const at = watched.get(el);
      if (!at) return;
      at.listeners.delete(listener);
      if (!at.listeners.size) {
        watched.delete(el);
        dirty.delete(el);
        observer?.unobserve(el);
      }
    };
  };
}

let pageWatch: ReturnType<typeof sizeWatcher> | null = null;
const watchSize = (el: Element, f: (s: Size) => void) => (pageWatch ??= sizeWatcher(browser()))(el, f);

/** The size of `el` (0×0 until there is one), re-rendering when its rounded size changes, at most once per frame. */
export function useElementSize(el: Element | null | undefined): Size {
  const [size, setSize] = useState<Size>(NO_SIZE);
  useEffect(() => {
    if (!el) return setSize(NO_SIZE);
    return watchSize(el, (s) => setSize((was) => (was.w === s.w && was.h === s.h ? was : s)));
  }, [el]);
  return size;
}

/** The same for an element held in a ref that is set once on mount (a ref's element is read after the first render). */
export function useRefSize(ref: { current: Element | null }): Size {
  const [el, setEl] = useState<Element | null>(null);
  useEffect(() => setEl(ref.current), [ref]);
  return useElementSize(el);
}

/** The screen's device pixel ratio (it changes when the window moves to a screen of another DPR, or the page is
 * zoomed): a change re-renders, and canvases rebuild their buffers at the new ratio. */
export function useDevicePixelRatio(): number {
  const [dpr, setDpr] = useState(() => (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1));
  useEffect(() => {
    // matchMedia can only ask "is it exactly this ratio": after a change, listen again at the new ratio
    const q = window.matchMedia?.(`(resolution: ${dpr}dppx)`);
    if (!q) return;
    const changed = () => setDpr(window.devicePixelRatio || 1);
    q.addEventListener("change", changed);
    return () => q.removeEventListener("change", changed);
  }, [dpr]);
  return dpr;
}
