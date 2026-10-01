import { useCallback, useEffect, useRef, useState } from "react";
import { LOGGED_IN, awaitsLogin } from "./http";
import { backoff } from "./backoff";

/** Repeated polling of the server, the single mechanism used by every page. The first request is sent immediately, even
 * in a hidden tab (it was requested: a page or a section was opened); afterwards no request is sent while the tab is
 * hidden, and becoming visible triggers an immediate request. A response identical to the previous one lengthens the
 * interval, up to `slowest`. A failed request waits `afterError` when the caller sets one (a watch that must notice
 * the server coming back at once); otherwise the interval, at least ERROR_WAIT_MIN, doubled with every further failure
 * in a row up to ERROR_WAIT_MAX. A refusal that only a new login changes (401, or a 403 for terms not agreed:
 * http.ts awaitsLogin) is not asked again: the server counts every one toward blocking the session
 * (lab2shot/server/access.py), so polling rests until the next login over the page (LOGGED_IN). Another 403 (a right
 * this login lacks) is an error like the rest: backed off, reported through onError. `every: null` reads once: neither repeated nor retried.
 * `until` ends polling (for example, a finished job). */
interface Polling<T> {
  read: () => Promise<T>;
  every: number | null | ((last: T | undefined) => number); // ms until the next request (null: once)
  slowest?: number | ((last: T | undefined) => number); // an unchanged response doubles the interval up to this (default: `every`)
  afterError?: number; // ms after every failed request (default: backing off from `every`)
  until?: (value: T) => boolean; // stop polling once this returns true
  // whether a response changed (what the backing off rests on) is the caller's to decide. By default the whole response
  // is compared, but some carry decorative live numbers (the CPU, memory and video-memory percentages of /api/load
  // change every second), which would never let the interval lengthen: the caller then gives the fields that really
  // matter, leaving the changing numbers out
  identity?: (value: T) => unknown;
  onValue?: (value: T) => void;
  onError?: (error: unknown) => void;
}

const ERROR_WAIT_MIN = 1000;
const ERROR_WAIT_MAX = 60_000;

/** A refusal that asking again cannot change (see Polling). */
const refused = awaitsLogin;

/** Starts polling; returns a function that stops it. */
export function startPolling<T>(p: Polling<T>): { stop: () => void; now: () => void } {
  let alive = true;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let lastText: string | undefined;
  let last: T | undefined;
  let wait = 0;
  let asking = false;
  let first = true;
  let failures = 0; // failed requests in a row
  let resting = false; // refused until a new login (awaitsLogin): waiting for LOGGED_IN
  const once = p.every === null;
  const base = () => (typeof p.every === "function" ? p.every(last) : p.every ?? 0);
  const visible = () => typeof document === "undefined" || document.visibilityState === "visible";
  const schedule = (ms: number) => {
    clearTimeout(timer);
    if (alive) timer = setTimeout(tick, ms);
  };
  async function tick(): Promise<void> {
    if (!alive || asking || resting) return;
    if (!first && !visible()) return schedule(base()); // hidden: request again once visible (see `back`)
    first = false;
    asking = true;
    try {
      const value = await p.read();
      if (!alive) return;
      const text = JSON.stringify(p.identity ? p.identity(value) : value);
      const b = base();
      last = value;
      const slowest = typeof p.slowest === "function" ? p.slowest(last) : p.slowest ?? b;
      wait = text === lastText ? Math.min(Math.max(slowest, b), Math.max(b, wait * 2)) : b;
      lastText = text;
      failures = 0;
      p.onValue?.(value);
      if (once || p.until?.(value)) return stop();
      schedule(wait);
    } catch (e) {
      if (!alive) return;
      p.onError?.(e);
      if (once) return stop();
      if (refused(e)) return rest();
      schedule(p.afterError ?? backoff(++failures, Math.max(base(), ERROR_WAIT_MIN), ERROR_WAIT_MAX));
    } finally {
      asking = false;
    }
  }
  const back = () => visible() && alive && ((wait = 0), void tick());
  const again = () => ((resting = false), (failures = 0), (wait = 0), (first = true), void tick()); // asked now, or logged in again
  function rest(): void {
    resting = true;
    window.addEventListener(LOGGED_IN, again, { once: true });
  }
  function stop(): void {
    alive = false;
    clearTimeout(timer);
    window.removeEventListener(LOGGED_IN, again);
    if (typeof document !== "undefined") document.removeEventListener("visibilitychange", back);
  }
  if (typeof document !== "undefined") document.addEventListener("visibilitychange", back);
  void tick();
  return { stop, now: () => (window.removeEventListener(LOGGED_IN, again), again()) };
}

/** A value read now and again every `every` ms while the tab is visible (null: only on demand). A change of `key`
 * clears the value and triggers a new request (the old value belongs to another key); `reload` requests immediately. */
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
  const key = useRef(options.key);
  useEffect(() => {
    if (key.current !== options.key) (setData(null), setError(null));
    key.current = options.key;
    const p = startPolling<T>({
      read,
      every,
      slowest: options.slowest,
      identity: options.identity,
      until: (v) => !!until.current?.(v),
      onValue: (v) => (setData(v), setError(null)),
      onError: (e) => (setError(e), onError.current?.(e)),
    });
    return p.stop;
  }, [read, every, options.slowest, options.key, asked]);
  return { data, error, reload: useCallback(() => setAsked((n) => n + 1), []) };
}
