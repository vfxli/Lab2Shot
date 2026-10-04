/** A graph as a tree to diff: nodes keyed by their id (their order kept apart), so an edit of one parameter is one
 * small change, never the whole list. The server applies the same (lab2shot/server/graphs.py): both sides must agree
 * on every step here. */

export type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
type Obj = { [k: string]: Json };

/** One change: set the value at a path, or (no "v") take the key away. */
export interface Op {
  p: string[];
  v?: Json;
}

const isObj = (x: Json | undefined): x is Obj => typeof x === "object" && x !== null && !Array.isArray(x);

/** The graph as the diff sees it: `nodes` by id, their order in `node_order`. */
export function tree(graph: Obj): Obj {
  const nodes = Array.isArray(graph.nodes) ? (graph.nodes as Obj[]) : [];
  return { ...graph, nodes: Object.fromEntries(nodes.map((n) => [String(n.id), n])), node_order: nodes.map((n) => String(n.id)) };
}

/** Whether two values are the same as JSON: what a graph file would hold for them (a key whose value is undefined is not
 * there at all, an undefined in a list is null, -0 is 0, and a NaN written twice is the same). The page's one deep
 * comparison of document values (the history, the "same value, no write" test of state/cookInputs.ts). */
export function same(a: Json | undefined, b: Json | undefined): boolean {
  if (a === b || (a !== a && b !== b)) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => same(x ?? null, b[i] ?? null));
  if (isObj(a) && isObj(b)) {
    const ka = Object.keys(a).filter((k) => a[k] !== undefined);
    const kb = Object.keys(b).filter((k) => b[k] !== undefined);
    return ka.length === kb.length && ka.every((k) => b[k] !== undefined && same(a[k], b[k]));
  }
  return false;
}

/** `next`, with every part of it that is the same as JSON as the matching part of `prev` taken from `prev` (the very
 * object): what a component selects from a new answer that did not change is the object it had, so it does not draw
 * again. `prev` itself when nothing changed. */
export function shared<T>(prev: T, next: T): T {
  const a = prev as Json | undefined, b = next as Json | undefined;
  if (a === b) return prev;
  if (Array.isArray(a) && Array.isArray(b)) {
    const out = b.map((x, i) => shared(a[i], x));
    return (out.length === a.length && out.every((x, i) => x === a[i]) ? a : out) as T;
  }
  if (isObj(a) && isObj(b)) {
    const out: Obj = {};
    let all = Object.keys(a).length === Object.keys(b).length;
    for (const k of Object.keys(b)) {
      out[k] = shared(a[k], b[k]) as Json;
      all &&= k in a && out[k] === a[k];
    }
    return (all ? a : out) as T;
  }
  return same(a, b) ? prev : next;
}

/** How big a value is, in leaves: a number, a string, a flag or null counts one, an empty list or object one, the
 * rest the sum of what they hold. */
export function size(x: Json | undefined): number {
  if (Array.isArray(x)) return x.length ? x.reduce((n: number, y) => n + size(y), 0) : 1;
  if (isObj(x)) {
    const keys = Object.keys(x);
    return keys.length ? keys.reduce((n, k) => n + size(x[k]), 0) : 1;
  }
  return 1;
}

/** How big a patch is, in the same leaves: each change's path and the value it sets. */
export const opsSize = (ops: Op[]): number => ops.reduce((n, o) => n + o.p.length + (o.v === undefined ? 0 : size(o.v)), 0);

/** What turns `a` into `b`: objects key by key, anything else (lists, values) whole. */
export function diff(a: Json, b: Json, path: string[] = [], out: Op[] = []): Op[] {
  if (isObj(a) && isObj(b)) {
    for (const k of Object.keys(b)) {
      if (!(k in a)) out.push({ p: [...path, k], v: b[k] });
      else if (!same(a[k], b[k])) diff(a[k], b[k], [...path, k], out);
    }
    for (const k of Object.keys(a)) if (!(k in b)) out.push({ p: [...path, k] });
  } else if (!same(a, b)) out.push({ p: path, v: b });
  return out;
}

