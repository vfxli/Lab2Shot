/** The page's single handler for its own errors, installed by the entry point before anything else runs (main.tsx), so
 * errors on the login page are caught as well. Each distinct error is kept once with an occurrence count (the same
 * kind, message and location increments it) for the feedback report (state/diagnostics.ts), and passed to listeners
 * (the page's log, once the page behind the gate has loaded; errors caught earlier are delivered when it starts
 * listening). Imports nothing: it is part of the gate's small bundle. */

export interface PageError {
  t: number; // first occurrence
  count: number;
  last: number;
  kind: "error" | "rejection" | "panel"; // panel: a part of the page caught by an ErrorBoundary
  message: string;
  where: string;
  stack: string;
}

const KEEP = 50;
const TEXT = 2000; // maximum characters of one message or stack

const errors: PageError[] = [];
const listeners = new Set<(e: PageError, detail: string) => void>();
const unheard: [PageError, string][] = [];
const cut = (s: string, n = TEXT) => (s.length > n ? `${s.slice(0, n)}…` : s);
let installed = false;

/** Records one page error; `detail` is the log's description of it. */
export function notePageError(kind: PageError["kind"], message: string, where: string, stack: string, detail = message): void {
  const now = Date.now();
  message = cut(message);
  let e = errors.find((x) => x.kind === kind && x.message === message && x.where === where);
  if (e) {
    e.count += 1;
    e.last = now;
  } else {
    e = { t: now, count: 1, last: now, kind, message, where, stack: cut(stack) };
    errors.push(e);
    if (errors.length > KEEP) errors.shift();
  }
  if (listeners.size) listeners.forEach((f) => f(e, detail));
  else if (unheard.length < KEEP) unheard.push([e, detail]);
}

/** Starts catching (called once, by the page's entry point). */
export function catchPageErrors(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("error", (e) => {
    const where = e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : "";
    const message = e.message || String(e.error);
    notePageError("error", message, where, (e.error as Error)?.stack ?? "", `${cut(message)}${e.filename ? ` (${e.filename}:${e.lineno})` : ""}`);
  });
  window.addEventListener("unhandledrejection", (e) => {
    const r = e.reason as Error | undefined;
    notePageError("rejection", r?.message ?? String(e.reason), "", r?.stack ?? "");
  });
}

/** Receives every error from now on, plus those caught before any listener existed. */
export function onPageError(f: (e: PageError, detail: string) => void): () => void {
  listeners.add(f);
  unheard.splice(0).forEach(([e, detail]) => f(e, detail));
  return () => void listeners.delete(f);
}

/** The errors kept since the page opened, oldest first. */
export const pageErrors = (): PageError[] => errors.map((e) => ({ ...e }));
