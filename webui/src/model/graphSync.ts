import { diff, opsSize, same, size, tree, type Json, type Op } from "./graphPatch";

/** The graph goes to the server once per version. Each version the page asks about
 * gets a name (a key); the first ask of it carries a patch against the version the server confirmed last (or the
 * whole graph when a patch would not be much smaller, or the server has none); every later ask names the key alone.
 * The server keeps the last few versions per browser (lab2shot/server/graphs.py) and answers 409 for one it does not
 * have (restarted, or forgotten): then the whole graph goes once more. */

type Obj = { [k: string]: Json };

export interface GraphRef {
  key: string;
  graph?: Obj;
  base?: string;
  patch?: Op[];
}

const SESSION = Array.from(crypto.getRandomValues(new Uint8Array(6)), (b) => b.toString(16).padStart(2, "0")).join("");
let serial = 0;
let confirmed: { key: string; tree: Obj } | null = null; // the version the server said it has
let current: { key: string; tree: Obj } | null = null; // the version being asked about
const asked = new Map<string, Obj>(); // the last versions asked about, by key (an answer may come after the next edit)

/** What names `graph` in an ask (`whole`: the whole graph, after the server did not know the version). */
export function graphRef(graph: object, whole = false): GraphRef {
  const g = graph as Obj;
  const t = tree(g);
  if (!whole && confirmed && same(confirmed.tree, t)) return { key: confirmed.key };
  if (!current || !same(current.tree, t)) {
    current = { key: `${SESSION}.${++serial}`, tree: t };
    asked.set(current.key, t);
    if (asked.size > 4) asked.delete(asked.keys().next().value!);
  }
  const key = current.key;
  if (whole || !confirmed) return { key, graph: g };
  const patch = diff(confirmed.tree, t);
  return opsSize(patch) * 2 < size(t) ? { key, base: confirmed.key, patch } : { key, graph: g }; // much smaller, or the whole
}

/** The server answered an ask naming `ref`: it has that version now. */
export function graphKept(ref: GraphRef): void {
  const t = asked.get(ref.key);
  if (t && Number(ref.key.split(".")[1]) >= Number(confirmed?.key.split(".")[1] ?? -1)) confirmed = { key: ref.key, tree: t };
}

/** The server does not have the version an ask named (409 with graph: "unknown"). */
export const unknownGraph = (status: number, body: { graph?: string } | null) => status === 409 && body?.graph === "unknown";
