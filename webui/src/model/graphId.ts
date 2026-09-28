import { randomId } from "../platform/randomId";

/** A graph's own identity: 128 random bits as 32 hex characters — never an integer, never a counter or a
 * timestamp, so it can't collide with another document's. state/results.ts keys `outputs` by "graphId:node";
 * two different documents (two templates opened side by side, a file and its 另存为 copy) must never share one, or
 * an output could be drawn on the wrong graph.
 *
 * Kept in its own module, which imports nothing of the page but platform/randomId.ts. */

export const newGraphId = (): string => randomId();

/** The id a freshly loaded document should carry, given the file's own `meta.id` (if any) and whether this load is
 * a new copy of another document (`freshId`: a template, or 另存为 — both get a fresh id). Returns the id and whether it had to be generated (the caller marks the
 * document changed when it did, so saving actually writes the id in). A plain open, save or reopen (`freshId`
 * false, `existing` set) always keeps the file's own id. */
export function graphIdForLoad(existing: string | undefined, freshId: boolean | undefined): { id: string; generated: boolean } {
  if (!existing || freshId) return { id: newGraphId(), generated: true };
  return { id: existing, generated: false };
}
