/** model/graphPatch.ts 的 shared 的测试：Node 直接跑（node src/model/graphPatch.test.ts）。 */

import { shared } from "./graphPatch.ts";

let count = 0;
function ok(cond: boolean, what: string): void {
  count++;
  if (!cond) throw new Error(`FAIL ${what}`);
}

// the same as JSON: the old object itself
const a = { n1: { messages: [{ code: "W-X", text: "t" }], values: { out: "1" } }, n2: { messages: [] } };
const b = JSON.parse(JSON.stringify(a));
ok(shared(a, b) === a, "equal answer keeps the old object");

// one node changed: that node new, the other the old object, the changed node's unchanged parts old as well
const c = JSON.parse(JSON.stringify(a));
c.n1.values.out = "2";
const s = shared(a, c);
ok(s !== a, "changed answer is a new object");
ok(s.n2 === a.n2, "unchanged node keeps its object");
ok(s.n1 !== a.n1 && s.n1.messages === a.n1.messages, "changed node keeps its unchanged parts");
ok(s.n1.values.out === "2", "the new value is taken");

// a node gone, a node new
const d = { n1: a.n1, n3: { messages: [] } } as Record<string, unknown>;
const t = shared(a as Record<string, unknown>, d);
ok(!("n2" in t) && t.n1 === a.n1 && t.n3 === d.n3, "keys follow the new answer");

// lists: a longer or shorter list is new; equal items keep theirs
const l1 = [{ x: 1 }, { x: 2 }];
const l2 = [{ x: 1 }, { x: 2 }, { x: 3 }];
const l = shared(l1, l2);
ok(l !== l1 && l[0] === l1[0] && l[1] === l1[1] && l[2] === l2[2], "list items shared where equal");
ok(shared(l2, [{ x: 1 }]) !== l2, "shorter list is new");

// undefined keys count as absent (as JSON)
ok(shared({ a: 1, b: undefined } as Record<string, unknown>, { a: 1 }) !== undefined, "undefined key handled");
ok(shared(1, 1) === 1 && shared("a", "b") === "b" && shared(null, null) === null, "leaves");

console.log(`graphPatch shared: ${count} ok`);
