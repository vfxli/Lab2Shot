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

export function same(a: Json | undefined, b: Json | undefined): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => same(x, b[i]));
  if (isObj(a) && isObj(b)) {
    const ka = Object.keys(a);
    return ka.length === Object.keys(b).length && ka.every((k) => k in b && same(a[k], b[k]));
  }
  return false;
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

