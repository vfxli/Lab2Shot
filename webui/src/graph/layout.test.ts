/** graph/layout.ts 的测试：Node 直接跑（node src/graph/layout.test.ts）。全部通过时打印一行，有一条不对就抛错。 */

import type { NodeTypeDef } from "../api/catalog.ts";
import { countCrossings, eachBlocks, estimateSize, layoutGraph, overlaps, type LayoutEdge, type LayoutNode } from "./layout.ts";

let count = 0;
function ok(cond: boolean, what: string): void {
  count++;
  if (!cond) throw new Error(what);
}

// 可重复的随机数
let seed = 12345;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const pick = (n: number) => Math.floor(rand() * n);

/** 随机有向无环图：节点按编号只连向编号更大的，带几个输入口次序、几根参数线。 */
function graph(n: number, extra: number): { nodes: LayoutNode[]; edges: LayoutEdge[] } {
  const nodes: LayoutNode[] = Array.from({ length: n }, (_, i) => ({
    id: `n${i}`, x: pick(2000), y: pick(2000), w: 240 + pick(120), h: 80 + pick(100), delivers: i === n - 1,
  }));
  const edges: LayoutEdge[] = [];
  for (let i = 1; i < n; i++) edges.push({ from: `n${pick(i)}`, to: `n${i}`, slot: 0 });
  for (let k = 0; k < extra; k++) {
    const a = pick(n - 1), b = a + 1 + pick(n - 1 - a);
    edges.push({ from: `n${a}`, to: `n${b}`, slot: 1 + pick(3), param: rand() < 0.15 });
  }
  return { nodes, edges };
}

const same = (a: Record<string, { x: number; y: number }>, b: Record<string, { x: number; y: number }>) =>
  Object.keys(a).every((id) => a[id].x === b[id].x && a[id].y === b[id].y);

// 1. 再整理一次不变、不重叠、连线全向右（随机图，分区 / strict 两种）
for (let t = 0; t < 24; t++) {
  const { nodes, edges } = graph(6 + pick(50), pick(30));
  for (const strict of [false, true]) {
    const out = layoutGraph({ nodes, edges, strict });
    const again = layoutGraph({ nodes: nodes.map((n) => ({ ...n, ...out.positions[n.id] })), edges, strict });
    ok(same(out.positions, again.positions), `第 ${t} 张图（strict ${strict}）再整理一次变了`);
    ok(!overlaps(out.positions, nodes).length, `第 ${t} 张图有重叠`);
    const size = new Map(nodes.map((n) => [n.id, n]));
    ok(edges.every((e) => out.positions[e.from].x + size.get(e.from)!.w <= out.positions[e.to].x), `第 ${t} 张图有向左的线`);
  }
}

// 2. 交叉数：与两两全比的结果相同（包括向左的线、竖直的线）
for (let t = 0; t < 20; t++) {
  const { nodes, edges } = graph(5 + pick(30), pick(20));
  if (t % 3 === 0) for (const n of nodes) n.x = 100 * pick(4); // 很多线横向范围重合、甚至竖直
  const pos = Object.fromEntries(nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
  const size = new Map(nodes.map((n) => [n.id, n]));
  const segs = edges.map((e) => ({ e, x1: pos[e.from].x + size.get(e.from)!.w, y1: pos[e.from].y + size.get(e.from)!.h / 2, x2: pos[e.to].x, y2: pos[e.to].y + size.get(e.to)!.h / 2 }));
  const d = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  let brute = 0;
  for (let i = 0; i < segs.length; i++)
    for (let j = i + 1; j < segs.length; j++) {
      const p = segs[i], q = segs[j], a = p.e, b = q.e;
      if (a.from === b.from || a.to === b.to || a.from === b.to || a.to === b.from) continue;
      if (d(p.x1, p.y1, p.x2, p.y2, q.x1, q.y1) * d(p.x1, p.y1, p.x2, p.y2, q.x2, q.y2) < 0
          && d(q.x1, q.y1, q.x2, q.y2, p.x1, p.y1) * d(q.x1, q.y1, q.x2, q.y2, p.x2, p.y2) < 0) brute++;
    }
  ok(countCrossings(pos, nodes, edges) === brute, `第 ${t} 张图交叉数 ${countCrossings(pos, nodes, edges)}，两两全比 ${brute}`);
}

// 3. 大图要快（167 个节点的模板在 node 里约 0.3 秒；这里给宽一点的上限）
{
  const { nodes, edges } = graph(170, 130);
  const t0 = performance.now();
  layoutGraph({ nodes, edges });
  const ms = performance.now() - t0;
  ok(ms < 3000, `170 个节点整理用了 ${ms.toFixed(0)} ms`);
}

// 4. 节点尺寸：宽按副标题、节点名（id）与类型名三行取最宽的；高按三行头
{
  const def = { id: "fbx.import", subtitle: "导入 FBX", inputs: [], outputs: [{ label: "场景" }], params: [] };
  const short = estimateSize(def, { id: "a" });
  ok(short.w === 240, `短名字按最小宽 240，得到 ${short.w}`);
  const long = estimateSize(def, { id: "a_very_long_node_name_for_testing_width" });
  ok(long.w > 300, `长节点名撑宽，得到 ${long.w}`);
  const longType = estimateSize({ ...def, id: "some.really_long_type_name_here.x" }, { id: "a" });
  ok(longType.w >= 33 * 7 + 116, `长类型名撑宽，得到 ${longType.w}`);
  ok(estimateSize(undefined, { id: "x" }).w === 240, "没有定义：最小宽");
  const longSub = estimateSize({ ...def, subtitle: "很长很长很长很长很长很长很长很长的中文副标题" }, { id: "a" });
  ok(longSub.w >= 22 * 12 + 116, `长副标题撑宽，得到 ${longSub.w}`);
  ok(short.h === 53 + 20 + 8 + 24 + 6, `三行头的高，得到 ${short.h}`);
}

// 5. 块：按角色配对（查询函数 / 定义里的 scope_role / 不传按已知类型），同名块才配
{
  const nodes = [
    { id: "b1", type: "loop_open", block: "A" }, { id: "e1", type: "loop_close", block: "A" },
    { id: "b2", type: "loop_open", block: "B" }, { id: "e2", type: "loop_close", block: "B" }, { id: "e3", type: "loop_close" },
  ];
  const role = (t: string) => (t === "loop_open" ? "begin" : t === "loop_close" ? "end" : undefined);
  const want = JSON.stringify([{ begin: "b1", ends: ["e1", "e3"] }, { begin: "b2", ends: ["e2"] }]);
  ok(JSON.stringify(eachBlocks(nodes, role)) === want, "按查询函数配对");
  ok(JSON.stringify(eachBlocks(nodes, { loop_open: { scope_role: "begin" }, loop_close: { scope_role: "end" } })) === want, "按定义的 scope_role 配对");
  ok(JSON.stringify(eachBlocks([{ id: "b", type: "foreach_begin" }, { id: "e", type: "foreach_end" }], {})) === "[]", "定义里没有 scope_role：不按类型名猜");
}

// 6. 按分组框排（文件开头第 8 条）：随机图分几个框，组内不重叠、连线全向右、框互不重叠、框外节点不压进框、输出在最右、
//    再整理一次不变；没有框（或框里没有节点）时与不传框完全一样
{
  const hit = (a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) =>
    a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  for (let t = 0; t < 16; t++) {
    const { nodes, edges } = graph(12 + pick(50), pick(30));
    const k = 2 + pick(5);
    // 留几个节点不进框（含最后的输出），其余按编号随机分到 k 个框
    const boxes = Array.from({ length: k }, (_, i) => ({ id: `box:${i}`, x: 0, y: 0, w: 0, h: 0, members: [] as string[] }));
    for (const n of nodes) if (!n.delivers && rand() < 0.85) boxes[pick(k)].members.push(n.id);
    const out = layoutGraph({ nodes, edges, boxes });
    const again = layoutGraph({ nodes: nodes.map((n) => ({ ...n, ...out.positions[n.id] })), edges, boxes });
    ok(same(out.positions, again.positions), `分组第 ${t} 张图再整理一次变了`);
    ok(!overlaps(out.positions, nodes).length, `分组第 ${t} 张图有重叠`);
    const size = new Map(nodes.map((n) => [n.id, n]));
    ok(edges.every((e) => out.positions[e.from].x + size.get(e.from)!.w <= out.positions[e.to].x), `分组第 ${t} 张图有向左的线`);
    const rects = Object.entries(out.boxes);
    for (let a = 0; a < rects.length; a++)
      for (let b = a + 1; b < rects.length; b++) ok(!hit(rects[a][1], rects[b][1]), `分组第 ${t} 张图 ${rects[a][0]} 与 ${rects[b][0]} 重叠`);
    for (const [id, r] of rects) {
      const members = new Set(boxes.find((b) => b.id === id)!.members);
      for (const n of nodes) if (!members.has(n.id)) ok(!hit(r, { ...out.positions[n.id], w: n.w, h: n.h }), `分组第 ${t} 张图 ${n.id} 压进了 ${id}`);
    }
    const last = nodes.find((n) => n.delivers)!;
    ok(nodes.every((n) => out.positions[n.id].x <= out.positions[last.id].x), `分组第 ${t} 张图输出不在最右`);
  }
  const { nodes, edges } = graph(30, 15);
  const none = layoutGraph({ nodes, edges });
  ok(same(none.positions, layoutGraph({ nodes, edges, boxes: [{ id: "box:1", x: 0, y: 0, w: 10, h: 10, members: [] }] }).positions), "空框：与不传框一样");
  // 输入 → [框 A：a1 a2] → [框 B：b1 b2] → 输出：两条带上下排开，框 A 在框 B 左边的列；输出在两框右边、整图中间高度
  const n = (id: string, delivers = false): LayoutNode => ({ id, x: 0, y: 0, w: 240, h: 100, delivers });
  const small = [n("in"), n("a1"), n("a2"), n("b1"), n("b2"), n("out", true)];
  const wires: LayoutEdge[] = [["in", "a1"], ["a1", "a2"], ["a2", "b1"], ["b1", "b2"], ["b2", "out"], ["a2", "out"]].map(([from, to]) => ({ from, to }));
  const r = layoutGraph({ nodes: small, edges: wires, boxes: [{ id: "A", x: 0, y: 0, w: 0, h: 0, members: ["a1", "a2"] }, { id: "B", x: 0, y: 0, w: 0, h: 0, members: ["b1", "b2"] }] });
  ok(!hit(r.boxes.A, r.boxes.B), "两个框不重叠");
  ok(r.positions.out.x > r.boxes.A.x + r.boxes.A.w && r.positions.out.x > r.boxes.B.x + r.boxes.B.w, "输出在两框右边");
  ok(r.boxes.A.y + r.boxes.A.h <= r.boxes.B.y || r.boxes.B.y + r.boxes.B.h <= r.boxes.A.y, "两框上下排开");
  ok(r.positions.a1.y === r.positions.a2.y && r.positions.b1.y === r.positions.b2.y, "组内一条链排成一行");
  ok(r.positions.in.x < r.boxes.A.x, "框外的输入挂在框 A 左边、不进框");
  // 第 9 条：两个框都折叠着（原来展开时摆得很远）：整理后按标题栏大小收拢，框里的节点随框平移、相对位置不变
  const far = small.map((q) => ({ ...q, x: q.id.startsWith("b") ? 4000 + (q.id === "b2" ? 300 : 0) : q.id === "a2" ? 300 : 0, y: q.id.startsWith("b") ? 3000 : 0 }));
  const foldA = { id: "A", x: -24, y: -58, w: 588, h: 182, members: ["a1", "a2"], folded: { w: 280, h: 34 } };
  const foldB = { id: "B", x: 3976, y: 2942, w: 588, h: 182, members: ["b1", "b2"], folded: { w: 280, h: 34 } };
  const f = layoutGraph({ nodes: far, edges: wires, boxes: [foldA, foldB] });
  const span = Math.max(...["in", "out"].map((id) => f.positions[id].y), f.boxes.A.y, f.boxes.B.y) - Math.min(...["in", "out"].map((id) => f.positions[id].y), f.boxes.A.y, f.boxes.B.y);
  ok(span < 400, `折叠的框收拢（竖向跨度 ${span}）`);
  ok(f.boxes.A.w === 588 && f.boxes.A.h === 182, "折叠的框宽高照旧（展开时原样）");
  ok(f.positions.a2.x - f.positions.a1.x === 300 && f.positions.a2.y === f.positions.a1.y, "框里的节点随框平移");
  ok(f.positions.a1.x - f.boxes.A.x === 24 && f.positions.b1.y - f.boxes.B.y === 58, "成员相对框的位置不变");
  ok(f.boxes.A.x + 280 < f.boxes.B.x, "折叠的框 A 在 B 左边");
  const again = layoutGraph({ nodes: far.map((q) => ({ ...q, ...f.positions[q.id] })), edges: wires, boxes: [{ ...foldA, ...f.boxes.A }, { ...foldB, ...f.boxes.B }] });
  ok(same(again.positions, f.positions), "折叠的框：再整理一次不变");
}

// 画布传的节点定义（state/catalog getNodeDefs）能直接给 eachBlocks（只做类型检查）
const typeCheckOnly = (defs: Record<string, NodeTypeDef>) => eachBlocks([], defs);
void typeCheckOnly;

console.log(`graph/layout：${count} 条全部通过`);
