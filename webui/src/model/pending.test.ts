/** model/pending.ts 的测试：Node 直接跑（node src/model/pending.test.ts）。 */

import { pendingNodes, type Answered, type PendingInputs } from "./pending.ts";

let count = 0;
function eq(got: ReadonlySet<string>, want: string[], what: string): void {
  count++;
  const g = [...got].sort().join(",");
  if (g !== [...want].sort().join(",")) throw new Error(`FAIL ${what}: got [${g}] want [${want.join(",")}]`);
}

const n = (v: number) => ({ typeId: "x", params: { v } });
const w = (a: string, b: string) => ({ source: a, sourceHandle: "out", target: b, targetHandle: "in" });
// a -> b -> c, d alone
const nodes = { a: n(1), b: n(1), c: n(1), d: n(1) };
const edges = [w("a", "b"), w("b", "c")];
const answered: Answered = { nodes, edges, cookRange: null };
const at = (patch: Partial<PendingInputs>): PendingInputs => ({ version: 2, order: ["a", "b", "c", "d"], nodes, edges, cookRange: null, ...patch });

eq(pendingNodes(at({}), 2, answered), [], "reply answers the current version: nothing pending");
eq(pendingNodes(at({}), 1, null), [], "no trusted answer yet: nothing pending");
eq(pendingNodes(at({ nodes: { ...nodes, b: n(2) } }), 1, answered), ["b", "c"], "an edited node and its downstream");
eq(pendingNodes(at({ nodes: { ...nodes, a: n(2) } }), 1, answered), ["a", "b", "c"], "an upstream edit reaches all below");
eq(pendingNodes(at({ nodes: { ...nodes, d: n(2) } }), 1, answered), ["d"], "a node alone");
eq(pendingNodes(at({ nodes: { ...nodes, b: { typeId: "x", params: { v: 1 } } } }), 1, answered), [], "a new object with the same content (undo / redo) is not a change");
eq(pendingNodes(at({ edges: [w("a", "b"), w("b", "c"), w("d", "c")] }), 1, answered), ["c"], "a wire added: the node it goes into");
eq(pendingNodes(at({ cookRange: ["1", "10"] }), 1, answered), ["a", "b", "c", "d"], "another frame range: all");
eq(pendingNodes(at({ order: ["a", "b", "c", "d", "e"], nodes: { ...nodes, e: n(1) } }), 1, answered), ["e"], "a new node");

console.log(`model/pending.test.ts: ${count} 项都对`);
