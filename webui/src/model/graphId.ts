/** A graph's own identity: 128 random bits as 32 hex characters — never an integer, never a counter or a
 * timestamp, so it can't collide with another document's. state/results.ts keys `deliveries` by "graphId:node";
 * two different documents (two templates opened side by side, a file and its 另存为 copy) must never share one, or
 * a delivery notification could land on the wrong graph.
 *
 * Kept in its own module, separate from graph/actions.ts, so it can be unit-tested under node's native test runner
 * without pulling in api.ts (which imports gate.tsx, a JSX file the runner can't load) — see
 * webui/tests/graphId.test.ts. */

export function newGraphId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
  let s = ""; // no crypto.randomUUID (very old browser): still 32 hex-ish characters, still not an integer
  while (s.length < 32) s += Math.random().toString(16).slice(2);
  return s.slice(0, 32);
}

/** The id a freshly loaded document should carry, given the file's own `meta.id` (if any) and whether this load is
 * a new copy of another document (`freshId`: a template, or 另存为 — both get a fresh id). Returns the id and whether it had to be generated (the caller marks the
 * document changed when it did, so saving actually writes the id in). A plain open, save or reopen (`freshId`
 * false, `existing` set) always keeps the file's own id. */
export function graphIdForLoad(existing: string | undefined, freshId: boolean | undefined): { id: string; generated: boolean } {
  if (!existing || freshId) return { id: newGraphId(), generated: true };
  return { id: existing, generated: false };
}
