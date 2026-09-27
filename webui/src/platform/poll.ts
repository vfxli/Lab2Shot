import { useCallback, useEffect, useRef, useState } from "react";

/** Repeated polling of the server, the single mechanism used by every page. The first request is sent immediately, even
 * in a hidden tab (it was requested: a page or a section was opened); afterwards no request is sent while the tab is
 * hidden, and becoming visible triggers an immediate request. A response identical to the previous one lengthens the
 * interval, up to `slowest`; a failed request waits `afterError`. `until` ends polling (for example, a finished job). */
export interface Polling<T> {
  read: () => Promise<T>;
  every: number | ((last: T | undefined, failed: boolean) => number); // ms until the next request
  slowest?: number | ((last: T | undefined) => number); // an unchanged response doubles the interval up to this (default: `every`)
  afterError?: number; // ms after a failed request (default: `every`)
  until?: (value: T) => boolean; // stop polling once this returns true
  // 「响应是否变化」由调用方判定（退避依据此判定）。默认比较整份响应；
  // 但部分响应中含有装饰性的实时数值，如 /api/load 中的 CPU、内存、显存百分比每秒变化，
  // 比较整份响应将无法退避。此时由调用方提供真正需要比较的字段，变动的数值不参与判断
  identity?: (value: T) => unknown;
  onValue?: (value: T) => void;
  onError?: (error: unknown) => void;
}

/** Starts polling; returns a function that stops it. */
export function startPolling<T>(p: Polling<T>): { stop: () => void; now: () => void } {
  let alive = true;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let lastText: string | undefined;
  let last: T | undefined;
  let wait = 0;
  let asking = false;
  let first = true;
  const base = (failed: boolean) => (typeof p.every === "function" ? p.every(last, failed) : p.every);
  const visible = () => typeof document === "undefined" || document.visibilityState === "visible";
  const schedule = (ms: number) => {
    clearTimeout(timer);
    if (alive) timer = setTimeout(tick, ms);
  };
  async function tick(): Promise<void> {
    if (!alive || asking) return;
    if (!first && !visible()) return schedule(base(false)); // hidden: request again once visible (see `back`)
    first = false;
    asking = true;
    try {
      const value = await p.read();
      if (!alive) return;
      const text = JSON.stringify(p.identity ? p.identity(value) : value);
      const b = base(false);
      last = value;
      const slowest = typeof p.slowest === "function" ? p.slowest(last) : p.slowest ?? b;
      wait = text === lastText ? Math.min(Math.max(slowest, b), Math.max(b, wait * 2)) : b;
      lastText = text;
      p.onValue?.(value);
      if (p.until?.(value)) return stop();
      schedule(wait);
    } catch (e) {
      if (!alive) return;
      p.onError?.(e);
      schedule(p.afterError ?? base(true));
    } finally {
      asking = false;
    }
  }
  const back = () => visible() && alive && ((wait = 0), void tick());
  function stop(): void {
    alive = false;
    clearTimeout(timer);
    if (typeof document !== "undefined") document.removeEventListener("visibilitychange", back);
  }
  if (typeof document !== "undefined") document.addEventListener("visibilitychange", back);
  void tick();
  return { stop, now: () => ((first = true), (wait = 0), void tick()) };
}

/** A value read now and again every `every` ms while the tab is visible (null: only on demand). A change of `key`
 * triggers a new request; `reload` requests immediately. */
export function usePoll<T>(
  read: () => Promise<T>,
  every: number | null,
  options: { slowest?: number; key?: unknown; identity?: (value: T) => unknown; until?: (value: T) => boolean; onError?: (error: unknown) => void } = {},
): { data: T | null; error: unknown; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [asked, setAsked] = useState(0);
  const until = useRef(options.until);
  until.current = options.until;
  const onError = useRef(options.onError);
  onError.current = options.onError;
  useEffect(() => {
    const p = startPolling<T>({
      read,
      every: every ?? 0,
      slowest: options.slowest,
      identity: options.identity,
      until: (v) => every === null || !!until.current?.(v),
      onValue: (v) => (setData(v), setError(null)),
      onError: (e) => (setError(e), onError.current?.(e)),
    });
    return p.stop;
  }, [read, every, options.slowest, options.key, asked]);
  return { data, error, reload: useCallback(() => setAsked((n) => n + 1), []) };
}
