/** 整理节点图：分层有向图排版（Sugiyama 的思路，Houdini 的 Layout / Nuke 的 Autoplace 那种从左到右的样子）。纯函数：
 * 输入节点（尺寸、现在的位置）、连线、逐项块、分组框，输出每个节点的新位置和框的新矩形。画布右下角「整理节点图」
 * （editor/NodeEditor.tsx → graph/edit.ts arrangeGraph）和给模板用的 tools/layout_graph.mjs 跑的是同一份。本文件不 import
 * 任何东西（node 里直接载入）。
 *
 * 1. 分层：按依赖深度（最长路径）从左到右分列，连线永远从左列到右列，一眼看出前后。没有输入的节点放在它最早的下游的
 *    前一列（主输入自然在最左；只喂给后面某个节点的常量 / 开关贴着它的下游，不从最左拉一根长线过来，Houdini、Nuke 的
 *    参数节点都这样放）。「输出」类（没有下游的交付节点）放在最右一列。跨列的连线在经过的每一列里占一个小位置
 *    （虚拟点），给线留出通道、少压节点。
 * 2. 列内排序：按上游邻居的重心排、再按下游邻居的重心反向扫，各扫 4 遍；交叉只在相邻列之间数，允许有、但取最少的一次；
 *    同分保持原有顺序（第一轮的顺序就是现在画布上从上到下的顺序），所以反复整理不乱跳。
 * 3. 坐标：列的 x = 前面各列最宽的节点累加 + 列距；列内按顺序从上到下堆、行距之外，做几轮「对齐拉直」：每个节点往它的
 *    主上游（第一根接进来的线）看齐、反向时往主下游看齐（最小二乘，保持顺序、不重叠），主线拉成一条直线、支线挂在旁边
 *    （Houdini / Nuke 的样子）。逐项块的「结束」对齐它的「开始」，尽量同一行高。
 * 4. 框：按框里的节点重算矩形（内边距、标题高度），不删框；没有节点的框不动。
 * 5. 整张图左上角对齐到 (0, 0)；只整理一部分（选中的）时放回它们原来包围盒的左上角。
 *
 * 结果若比整理前的交叉更多（按画出来的样子数：每根线从源节点右边中点到目标节点左边中点的线段），就用交叉最少的那种排序；
 * 再整理一次结果不变（内部迭代到不动为止）。 */

export interface LayoutNode {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  delivers?: boolean; // 交付类（「输出」）：没有下游时放最右一列
}

export interface LayoutEdge {
  from: string;
  to: string;
  param?: boolean; // 接到提升出来的参数口（param:…）：是参数的连线，不是数据主线（找主链时不走它）
}

export interface LayoutBlock {
  begin: string;
  ends: string[];
}

export interface LayoutBox {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  members: string[]; // 框里的节点（整理前算好：展开的框按中心在框内，折叠的框按它自己的名单）
}

export interface LayoutResult {
  positions: Record<string, { x: number; y: number }>;
  boxes: Record<string, { x: number; y: number; w: number; h: number }>;
}

export const LAYOUT = { colGap: 120, rowGap: 40, laneH: 16, boxPad: 24, boxHead: 34, mainGap: 160, zoneGap: 200, blockGap: 120 };

interface Item {
  id: string; // 节点 id，或虚拟点 "~边序号~列"
  w: number;
  h: number;
  real: boolean;
}

/** 按画出来的样子数交叉：每根线是源节点右边中点到目标节点左边中点的线段，共用节点的两根不算。 */
export function countCrossings(pos: Record<string, { x: number; y: number }>, nodes: LayoutNode[], edges: LayoutEdge[]): number {
  const size = new Map(nodes.map((n) => [n.id, n]));
  const segs = edges.flatMap((e) => {
    const a = pos[e.from], b = pos[e.to], sa = size.get(e.from), sb = size.get(e.to);
    return a && b && sa && sb ? [{ e, x1: a.x + sa.w, y1: a.y + sa.h / 2, x2: b.x, y2: b.y + sb.h / 2 }] : [];
  });
  const cross = (p: (typeof segs)[number], q: (typeof segs)[number]) => {
    const d = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const d1 = d(p.x1, p.y1, p.x2, p.y2, q.x1, q.y1), d2 = d(p.x1, p.y1, p.x2, p.y2, q.x2, q.y2);
    const d3 = d(q.x1, q.y1, q.x2, q.y2, p.x1, p.y1), d4 = d(q.x1, q.y1, q.x2, q.y2, p.x2, p.y2);
    return d1 * d2 < 0 && d3 * d4 < 0;
  };
  let n = 0;
  for (let i = 0; i < segs.length; i++)
    for (let j = i + 1; j < segs.length; j++) {
      const a = segs[i].e, b = segs[j].e;
      if (a.from === b.from || a.to === b.to || a.from === b.to || a.to === b.from) continue;
      if (cross(segs[i], segs[j])) n++;
    }
  return n;
}

/** 两个节点的矩形是否重叠（留 1 px）。 */
export function overlaps(pos: Record<string, { x: number; y: number }>, nodes: LayoutNode[]): [string, string][] {
  const out: [string, string][] = [];
  for (let i = 0; i < nodes.length; i++)
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j], pa = pos[a.id], pb = pos[b.id];
      if (pa.x < pb.x + b.w - 1 && pb.x < pa.x + a.w - 1 && pa.y < pb.y + b.h - 1 && pb.y < pa.y + a.h - 1) out.push([a.id, b.id]);
    }
  return out;
}

/** 分列：最长路径；没有输入的贴着最早的下游；没有下游的交付节点在最右。 */
function columns(nodes: LayoutNode[], edges: LayoutEdge[]): Map<string, number> {
  const preds = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  const succs = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  for (const e of edges) {
    if (!preds.has(e.to) || !succs.has(e.from) || e.from === e.to) continue;
    preds.get(e.to)!.push(e.from);
    succs.get(e.from)!.push(e.to);
  }
  const col = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (id: string): number => {
    if (col.has(id)) return col.get(id)!;
    if (visiting.has(id)) return 0; // 环（节点图不该有）：断开
    visiting.add(id);
    const c = Math.max(-1, ...preds.get(id)!.map(depth)) + 1;
    visiting.delete(id);
    col.set(id, c);
    return c;
  };
  nodes.forEach((n) => depth(n.id));
  // 没有输入的：放到最早的下游的前一列
  for (const n of nodes) {
    const s = succs.get(n.id)!;
    if (preds.get(n.id)!.length || !s.length) continue;
    col.set(n.id, Math.max(0, Math.min(...s.map((t) => col.get(t)!)) - 1));
  }
  const last = Math.max(0, ...col.values());
  for (const n of nodes) if (n.delivers && !succs.get(n.id)!.length) col.set(n.id, last);
  return col;
}

/** 一轮完整排版：给定每列的顺序，算坐标。 */
function place(order: Item[][], links: [string, string][], pull: [string, string][], mode: "primary" | "mean" = "primary", gap: number = LAYOUT.colGap): Record<string, { x: number; y: number }> {
  const pos: Record<string, { x: number; y: number }> = {};
  const size = new Map<string, Item>();
  let x = 0;
  for (const col of order) {
    let y = 0;
    for (const it of col) {
      size.set(it.id, it);
      pos[it.id] = { x, y };
      y += it.h + (it.real ? LAYOUT.rowGap : LAYOUT.rowGap / 2);
    }
    x += Math.max(0, ...col.filter((it) => it.real).map((it) => it.w)) + gap;
  }
  const up = new Map<string, string[]>(), down = new Map<string, string[]>();
  for (const [a, b] of links) {
    (down.get(a) ?? down.set(a, []).get(a)!).push(b);
    (up.get(b) ?? up.set(b, []).get(b)!).push(a);
  }
  const extra = new Map<string, string[]>();
  for (const [a, b] of pull) {
    (extra.get(a) ?? extra.set(a, []).get(a)!).push(b);
    (extra.get(b) ?? extra.set(b, []).get(b)!).push(a);
  }
  const centre = (id: string) => pos[id].y + size.get(id)!.h / 2;
  const settle = (col: Item[], want: (it: Item) => number | null) => {
    // 最小二乘地让每个节点的中心靠近它想去的地方，同时保持顺序、不重叠（相邻违例的块合并，取平均）
    const gaps = col.map((it, i) => (i === 0 ? 0 : col[i - 1].h + (col[i - 1].real && it.real ? LAYOUT.rowGap : LAYOUT.rowGap / 2)));
    const offs: number[] = [];
    gaps.reduce((acc, g, i) => ((offs[i] = acc + g), acc + g), 0);
    const target = col.map((it, i) => {
      const w = want(it);
      return (w === null ? centre(it.id) : w) - it.h / 2 - offs[i];
    });
    const blocks: { start: number; end: number; sum: number; n: number }[] = [];
    target.forEach((t, i) => {
      blocks.push({ start: i, end: i, sum: t, n: 1 });
      while (blocks.length > 1) {
        const b = blocks[blocks.length - 1], a = blocks[blocks.length - 2];
        if (a.sum / a.n <= b.sum / b.n) break;
        blocks.splice(blocks.length - 2, 2, { start: a.start, end: b.end, sum: a.sum + b.sum, n: a.n + b.n });
      }
    });
    for (const b of blocks) for (let i = b.start; i <= b.end; i++) pos[col[i].id].y = b.sum / b.n + offs[i];
  };
  // 对齐：每个节点往它的「主上游」（第一根接进来的线的源）看齐，像 Houdini / Nuke 那样主线拉成一条直线、支线挂在旁边；
  // 反向扫时往「主下游」看齐（只在那个下游的主上游正是它时，免得一个节点被好几个下游拉扯）。逐项结束对齐它的逐项开始。
  const primaryUp = (id: string) => up.get(id)?.[0];
  const primaryDown = (id: string) => down.get(id)?.find((d) => primaryUp(d) === id);
  if (mode === "mean") {
    // 另一种：往全部上游 / 下游邻居的平均中心靠（主线不一定直，但多进多出的地方交叉可能更少）；只在它交叉更少时才用
    const mean = (ids: string[]) => (ids.length ? ids.reduce((s, id) => s + centre(id), 0) / ids.length : null);
    for (let round = 0; round < 6; round++) {
      const both = round >= 4;
      const cols = round % 2 === 0 ? order : [...order].reverse();
      for (const col of cols)
        settle(col, (it) => mean([...(both || round % 2 === 0 ? up.get(it.id) ?? [] : []), ...(both || round % 2 === 1 ? down.get(it.id) ?? [] : []),
                                  ...(extra.get(it.id) ?? []), ...(extra.get(it.id) ?? [])]));
    }
    return pos;
  }
  for (let round = 0; round < 6; round++) {
    const backward = round % 2 === 1 && round < 5;
    const cols = backward ? [...order].reverse() : order;
    for (const col of cols) {
      settle(col, (it) => {
        const pair = extra.get(it.id)?.find((b) => pos[b] && pos[b].x < pos[it.id].x); // 逐项结束：对齐它的逐项开始
        const to = backward ? primaryDown(it.id) : pair ?? primaryUp(it.id);
        return to ? centre(to) : null;
      });
    }
  }
  return pos;
}

function orderings(cols: Item[][], links: [string, string][]): Item[][][] {
  const up = new Map<string, string[]>(), down = new Map<string, string[]>();
  for (const [a, b] of links) {
    (down.get(a) ?? down.set(a, []).get(a)!).push(b);
    (up.get(b) ?? up.set(b, []).get(b)!).push(a);
  }
  const out: Item[][][] = [cols.map((c) => [...c])];
  let cur = cols.map((c) => [...c]);
  const index = () => new Map(cur.flatMap((c) => c.map((it, i) => [it.id, i] as [string, number])));
  const sweep = (range: number[], near: Map<string, string[]>) => {
    for (const ci of range) {
      const at = index();
      const keyed = cur[ci].map((it, i) => {
        const ns = (near.get(it.id) ?? []).map((n) => at.get(n)).filter((v): v is number => v !== undefined);
        return { it, i, k: ns.length ? ns.reduce((s, v) => s + v, 0) / ns.length : i };
      });
      keyed.sort((a, b) => a.k - b.k || a.i - b.i); // 同分保持原有顺序
      cur[ci] = keyed.map((k) => k.it);
    }
  };
  const idx = cur.map((_, i) => i);
  for (let r = 0; r < 4; r++) {
    sweep(idx.slice(1), up);
    out.push(cur.map((c) => [...c]));
    sweep([...idx].reverse().slice(1), down);
    out.push(cur.map((c) => [...c]));
  }
  return out;
}

/** 相邻列之间的交叉数（排序时比较用）。 */
function layerCrossings(order: Item[][], links: [string, string][]): number {
  const at = new Map(order.flatMap((c, ci) => c.map((it, i) => [it.id, [ci, i]] as [string, [number, number]])));
  const byCol = new Map<number, [number, number][]>();
  for (const [a, b] of links) {
    const pa = at.get(a), pb = at.get(b);
    if (!pa || !pb || pb[0] !== pa[0] + 1) continue;
    (byCol.get(pa[0]) ?? byCol.set(pa[0], []).get(pa[0])!).push([pa[1], pb[1]]);
  }
  let n = 0;
  for (const segs of byCol.values())
    for (let i = 0; i < segs.length; i++)
      for (let j = i + 1; j < segs.length; j++)
        if ((segs[i][0] - segs[j][0]) * (segs[i][1] - segs[j][1]) < 0) n++;
  return n;
}

function once(nodes: LayoutNode[], edges: LayoutEdge[], blocks: LayoutBlock[], gap: number = LAYOUT.colGap): Record<string, { x: number; y: number }> {
  const ids = new Set(nodes.map((n) => n.id));
  const es = edges.filter((e) => ids.has(e.from) && ids.has(e.to) && e.from !== e.to);
  const col = columns(nodes, es);
  const ncol = Math.max(0, ...col.values()) + 1;
  const cols: Item[][] = Array.from({ length: ncol }, () => []);
  // 起始顺序 = 现在画布上从上到下（同高按 x、再按 id），这样同分时保持原样
  const sorted = [...nodes].sort((a, b) => a.y + a.h / 2 - (b.y + b.h / 2) || a.x - b.x || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  for (const n of sorted) cols[col.get(n.id)!].push({ id: n.id, w: n.w, h: n.h, real: true });
  // 跨列的线：经过的每一列放一个虚拟点，按两端的中点插进去
  const links: [string, string][] = [];
  const centreOf = new Map(nodes.map((n) => [n.id, n.y + n.h / 2]));
  es.forEach((e, k) => {
    const a = col.get(e.from)!, b = col.get(e.to)!;
    if (b - a <= 1) {
      links.push([e.from, e.to]);
      return;
    }
    let prev = e.from;
    for (let c = a + 1; c < b; c++) {
      const id = `~${k}~${c}`;
      const y = ((centreOf.get(e.from) ?? 0) + (centreOf.get(e.to) ?? 0)) / 2;
      const list = cols[c];
      const at = list.findIndex((it) => (it.real ? centreOf.get(it.id) ?? 0 : Infinity) > y);
      list.splice(at < 0 ? list.length : at, 0, { id, w: 0, h: LAYOUT.laneH, real: false });
      links.push([prev, id]);
      prev = id;
    }
    links.push([prev, e.to]);
  });
  const pull: [string, string][] = blocks.flatMap((b) => b.ends.filter((e) => ids.has(e) && ids.has(b.begin)).map((e) => [b.begin, e] as [string, string]));
  // 候选排序：原样、以及每一次扫之后；先按相邻列交叉数挑，再按画出来的交叉数、最后按先后（原样优先）
  const cands = orderings(cols, links);
  const measure = (o: Item[][]) => {
    // 两种拉直都算，取画出来交叉少的（同分用主线拉直）
    const layer = layerCrossings(o, links);
    const [a, b] = (["primary", "mean"] as const).map((mode) => {
      const all = place(o, links, pull, mode, gap);
      const pos = Object.fromEntries(nodes.map((n) => [n.id, all[n.id]]));
      return { o, pos, layer, drawn: countCrossings(pos, nodes, es) };
    });
    return b.drawn < a.drawn ? b : a;
  };
  // 比较：先比画出来的交叉（看到的就是它），再比相邻列的交叉，同分取先出现的（原样优先）
  const better = (a: ReturnType<typeof measure>, b: ReturnType<typeof measure>) => a.drawn < b.drawn || (a.drawn === b.drawn && a.layer < b.layer);
  let best = measure(cands[0]);
  for (const o of cands.slice(1)) {
    const m = measure(o);
    if (better(m, best)) best = m;
  }
  // 再做几轮相邻交换（列里相邻两个换位置，画出来的交叉少了才换）：扫重心之后常还剩一两处能解开的
  for (let pass = 0; pass < 4; pass++) {
    let improved = false;
    for (let ci = 0; ci < best.o.length; ci++)
      for (let i = 0; i + 1 < best.o[ci].length; i++) {
        const o = best.o.map((c) => [...c]);
        [o[ci][i], o[ci][i + 1]] = [o[ci][i + 1], o[ci][i]];
        const m = measure(o);
        if (better(m, best)) {
          best = m;
          improved = true;
        }
      }
    if (!improved) break;
  }
  return best.pos;
}

// ------------------------------------------------------------------ 功能区：主链居中，支区成团放在上下，块前后留空

/** 按功能区排：
 * - 主链：只沿数据线（接到参数口 param:… 的线不算，那是参数的旁支），从主输入（没有输入、往下游能到的节点最多的那个）
 *   到「输出」（交付类；没有就取最远的末端）的最长路径；加上主链上
 *   逐项块里的节点，一起当「主带」，用同一套分层算法排（列距 LAYOUT.mainGap），主链拉成中间一条线；逐项块的「开始」之前、
 *   「结束」之后各加 LAYOUT.blockGap。
 * - 支区：其余节点沿下游第一次碰到的主带节点（取最靠左的那个）= 挂接点；挂接点相同、去掉主带后彼此相连的归一区（例如
 *   相机来源一支、角色一支）。往下游碰不到主带的，按上游碰到的最右的主带节点挂在它后面；两头都碰不到的孤立节点各一区。
 * - 每个支区先自己排紧凑（列距 LAYOUT.colGap、行距照旧），再整体放到挂接点所在列的上方或下方（画出来交叉少的一边；一样
 *   就离主带近的一边，再一样按它原来在主链之上还是之下），
 *   横向让它最后一个节点落在挂接点前一列；与主带、与已放的支区上下左右至少 LAYOUT.zoneGap。连线放不下（支区的输入来自
 *   挂接点之后）时把挂接点及其右边的整体右移让出位置。
 * 不看节点类型的语义（除逐项块）。结果有连线向左或重叠时返回 null（用不分区的排法）。 */
export function zoned(nodes: LayoutNode[], edges: LayoutEdge[], blocks: LayoutBlock[], start?: string): Record<string, { x: number; y: number }> | null {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const es = edges.filter((e) => byId.has(e.from) && byId.has(e.to) && e.from !== e.to);
  const succ = new Map<string, string[]>(nodes.map((n) => [n.id, []])), pred = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  // 数据线（不含接到参数口的线）：主链只沿数据走——相机焦距、切换的「走哪一路」这类接参数的线是旁支
  const dsucc = new Map<string, string[]>(nodes.map((n) => [n.id, []])), dpred = new Map<string, string[]>(nodes.map((n) => [n.id, []]));
  for (const e of es) {
    succ.get(e.from)!.push(e.to);
    pred.get(e.to)!.push(e.from);
    if (!e.param) (dsucc.get(e.from)!.push(e.to), dpred.get(e.to)!.push(e.from));
  }
  const ids = nodes.map((n) => n.id).sort();
  const reach = (from: string, next: Map<string, string[]>) => {
    const seen = new Set([from]);
    const stack = [from];
    while (stack.length) for (const n of next.get(stack.pop()!)!) if (!seen.has(n)) (seen.add(n), stack.push(n));
    return seen;
  };
  const sources = ids.filter((id) => !dpred.get(id)!.length);
  if (!sources.length) return null;
  const input = start ?? sources.map((id) => ({ id, n: reach(id, dsucc).size })).sort((a, b) => b.n - a.n || (a.id < b.id ? -1 : 1))[0].id;
  if (!dsucc.has(input)) return null;
  const down = reach(input, dsucc);
  // 最长路径（按节点数）：拓扑序上从主输入往下推
  const order: string[] = [];
  const mark = new Set<string>();
  const visit = (id: string) => {
    if (mark.has(id)) return;
    mark.add(id);
    for (const p of dpred.get(id)!) if (down.has(p)) visit(p);
    order.push(id);
  };
  [...down].sort().forEach(visit);
  const dist = new Map<string, number>([[input, 1]]), from = new Map<string, string>();
  for (const id of order) {
    if (id === input) continue;
    for (const p of [...dpred.get(id)!].sort()) {
      const d = dist.get(p);
      if (d !== undefined && d + 1 > (dist.get(id) ?? 0)) (dist.set(id, d + 1), from.set(id, p));
    }
  }
  const sinks = [...down].filter((id) => !dsucc.get(id)!.length);
  const ends = sinks.filter((id) => byId.get(id)!.delivers);
  const target = (ends.length ? ends : sinks).sort((a, b) => (dist.get(b) ?? 0) - (dist.get(a) ?? 0) || (a < b ? -1 : 1))[0];
  if (!target) return null;
  const chain: string[] = [];
  for (let at: string | undefined = target; at; at = from.get(at)) chain.unshift(at);
  if (chain.length < 3) return null;
  // 主带 = 主链 + 主链上的逐项块里的节点（从开始往下、能到结束的）
  const band = new Set(chain);
  for (const b of blocks) {
    if (!band.has(b.begin)) continue;
    const after = reach(b.begin, succ);
    for (const end of b.ends) {
      if (!byId.has(end)) continue;
      const before = reach(end, pred);
      for (const id of after) if (before.has(id)) band.add(id);
    }
  }
  const bandNodes = nodes.filter((n) => band.has(n.id));
  const pos: Record<string, { x: number; y: number }> = { ...once(bandNodes, es, blocks, LAYOUT.mainGap) };
  // 逐项块前后留空：开始那一列（含）往右、结束那一列之后往右各移 blockGap
  for (const b of blocks) {
    if (!pos[b.begin]) continue;
    const bx = pos[b.begin].x;
    for (const id of Object.keys(pos)) if (pos[id].x >= bx) pos[id].x += LAYOUT.blockGap;
    const ex = Math.max(...b.ends.filter((e) => pos[e]).map((e) => pos[e].x));
    if (Number.isFinite(ex)) for (const id of Object.keys(pos)) if (pos[id].x > ex) pos[id].x += LAYOUT.blockGap;
  }
  // 支区：挂接点
  const hits = (id: string, next: Map<string, string[]>) => {
    const seen = new Set([id]), found: string[] = [];
    const stack = [id];
    while (stack.length)
      for (const n of next.get(stack.pop()!)!) {
        if (seen.has(n)) continue;
        seen.add(n);
        if (band.has(n)) found.push(n);
        else stack.push(n);
      }
    return found;
  };
  const key = new Map<string, string>();
  for (const id of ids) {
    if (band.has(id)) continue;
    const dn = hits(id, succ);
    if (dn.length) {
      key.set(id, `down:${dn.sort((a, b) => pos[a].x - pos[b].x || (a < b ? -1 : 1))[0]}`);
      continue;
    }
    const upHits = hits(id, pred);
    key.set(id, upHits.length ? `up:${upHits.sort((a, b) => pos[b].x - pos[a].x || (a < b ? -1 : 1))[0]}` : `alone:${id}`);
  }
  // 挂接点相同且去掉主带后相连的合一区
  const root = new Map<string, string>([...key.keys()].map((id) => [id, id]));
  const find = (id: string): string => (root.get(id) === id ? id : (root.set(id, find(root.get(id)!)), root.get(id)!));
  for (const e of es)
    if (key.has(e.from) && key.has(e.to) && key.get(e.from) === key.get(e.to)) {
      const a = find(e.from), b = find(e.to);
      if (a !== b) root.set(a < b ? b : a, a < b ? a : b);
    }
  const zones = new Map<string, string[]>();
  for (const id of key.keys()) (zones.get(find(id)) ?? zones.set(find(id), []).get(find(id))!).push(id);
  const chainY = chain.reduce((s, id) => s + byId.get(id)!.y + byId.get(id)!.h / 2, 0) / chain.length;
  const placed: { x1: number; y1: number; x2: number; y2: number }[] = [];
  const rectOf = (list: string[]) => ({
    x1: Math.min(...list.map((id) => pos[id].x)), y1: Math.min(...list.map((id) => pos[id].y)),
    x2: Math.max(...list.map((id) => pos[id].x + byId.get(id)!.w)), y2: Math.max(...list.map((id) => pos[id].y + byId.get(id)!.h)),
  });
  const bandRect = rectOf([...band]);
  const zoneList = [...zones.values()].map((list) => {
    const k = key.get(list[0])!;
    const at = k.split(":").slice(1).join(":");
    return { list: list.sort(), k, at, ax: pos[at]?.x ?? Infinity };
  }).sort((a, b) => a.ax - b.ax || (a.k < b.k ? -1 : a.k > b.k ? 1 : a.list[0] < b.list[0] ? -1 : 1));
  for (const z of zoneList) {
    const zn = z.list.map((id) => byId.get(id)!);
    const zp = once(zn, es, blocks);
    const w = (id: string) => byId.get(id)!.w;
    // 横向：出去的线要求 dx ≤ …，进来的线要求 dx ≥ …（只看已经放好的一端）
    let ub = Infinity, lb = -Infinity;
    const outs: string[] = [];
    for (const e of es) {
      if (zp[e.from] && !zp[e.to] && pos[e.to]) (ub = Math.min(ub, pos[e.to].x - LAYOUT.colGap - (zp[e.from].x + w(e.from)))), outs.push(e.to);
      if (zp[e.to] && !zp[e.from] && pos[e.from]) lb = Math.max(lb, pos[e.from].x + w(e.from) + LAYOUT.colGap - zp[e.to].x);
    }
    if (Number.isFinite(ub) && Number.isFinite(lb) && lb > ub) {
      const need = lb - ub, threshold = Math.min(...outs.map((id) => pos[id].x));
      for (const id of Object.keys(pos)) if (pos[id].x >= threshold) pos[id].x += need;
      for (const r of placed) if (r.x1 >= threshold) (r.x1 += need, r.x2 += need);
      ub += need;
    }
    const dx = Number.isFinite(ub) ? ub : Number.isFinite(lb) ? lb : bandRect.x1 - Math.min(...zn.map((n) => zp[n.id].x));
    const zr = { x1: Math.min(...zn.map((n) => zp[n.id].x)) + dx, x2: Math.max(...zn.map((n) => zp[n.id].x + n.w)) + dx,
                 y1: Math.min(...zn.map((n) => zp[n.id].y)), y2: Math.max(...zn.map((n) => zp[n.id].y + n.h)) };
    const h = zr.y2 - zr.y1;
    const band2 = rectOf([...band]);
    const topFor = (above: boolean) => {
      let top = above ? band2.y1 - LAYOUT.zoneGap - h : band2.y2 + LAYOUT.zoneGap;
      for (let guard = 0; guard < 100; guard++) {
        const hit = placed.find((r) => zr.x1 < r.x2 + LAYOUT.zoneGap && r.x1 < zr.x2 + LAYOUT.zoneGap
          && top < r.y2 + LAYOUT.zoneGap && r.y1 < top + h + LAYOUT.zoneGap);
        if (!hit) break;
        top = above ? hit.y1 - LAYOUT.zoneGap - h : hit.y2 + LAYOUT.zoneGap;
      }
      return top;
    };
    // 放上面还是下面：先看画出来的交叉（与已放好的部分），再看离主带近的，再按它原来在主链之上还是之下
    const wasAbove = zn.reduce((s, n) => s + n.y + n.h / 2, 0) / zn.length < chainY;
    const tries = [wasAbove, !wasAbove].map((above) => {
      const top = topFor(above);
      const trial = { ...pos };
      for (const n of zn) trial[n.id] = { x: zp[n.id].x + dx, y: zp[n.id].y - zr.y1 + top };
      const dist = above ? band2.y1 - (top + h) : top - band2.y2;
      return { above, top, trial, n: countCrossings(trial, nodes, es), dist };
    });
    const pick = tries[1].n < tries[0].n || (tries[1].n === tries[0].n && tries[1].dist < tries[0].dist) ? tries[1] : tries[0];
    const top = pick.top;
    for (const n of zn) pos[n.id] = { x: zp[n.id].x + dx, y: zp[n.id].y - zr.y1 + top };
    placed.push({ x1: zr.x1, x2: zr.x2, y1: top, y2: top + h });
  }
  // 最后在每一区（主带、各支区）的每一列里试着交换上下相邻的两个节点（两者占的总高度不变），画出来的交叉少了才换
  const region = new Map<string, string>([...band].map((id) => [id, "band"]));
  for (const [k, list] of zones) for (const id of list) region.set(id, k);
  for (let pass = 0; pass < 4; pass++) {
    let improved = false;
    const groups = new Map<string, string[]>();
    for (const id of ids) {
      const g = `${region.get(id)}|${pos[id].x}`;
      (groups.get(g) ?? groups.set(g, []).get(g)!).push(id);
    }
    let now = countCrossings(pos, nodes, es);
    for (const list of groups.values()) {
      list.sort((a, b) => pos[a].y - pos[b].y);
      for (let i = 0; i + 1 < list.length; i++) {
        const a = list[i], b = list[i + 1];
        const pa = pos[a], pb = pos[b];
        const gapAB = pb.y - (pa.y + byId.get(a)!.h);
        const trial = { ...pos, [b]: { x: pb.x, y: pa.y }, [a]: { x: pa.x, y: pa.y + byId.get(b)!.h + gapAB } };
        const n = countCrossings(trial, nodes, es);
        if (n < now) {
          pos[a] = trial[a];
          pos[b] = trial[b];
          list[i] = b;
          list[i + 1] = a;
          now = n;
          improved = true;
        }
      }
    }
    if (!improved) break;
  }
  // 检查：连线全向右、不重叠，否则不用分区的排法
  if (es.some((e) => pos[e.from].x + byId.get(e.from)!.w > pos[e.to].x) || overlaps(pos, nodes).length) return null;
  return pos;
}

/** 一次排版：按功能区排（zoned，主链从主输入起）；排不出来（有线向左、重叠）时用不分区的排法（once）。
 * `strict`：分区的交叉比不分区的多时也用不分区的（分区把支线拉开，跨区的长线常会多交叉几处）。 */
function arrange(nodes: LayoutNode[], edges: LayoutEdge[], blocks: LayoutBlock[], strict = false): Record<string, { x: number; y: number }> {
  const flat = once(nodes, edges, blocks);
  const zone = zoned(nodes, edges, blocks);
  return zone && (!strict || countCrossings(zone, nodes, edges) <= countCrossings(flat, nodes, edges)) ? zone : flat;
}

const shift = (pos: Record<string, { x: number; y: number }>, dx: number, dy: number) =>
  Object.fromEntries(Object.entries(pos).map(([id, p]) => [id, { x: Math.round(p.x + dx), y: Math.round(p.y + dy) }]));

/** 整理：`nodes` 是要整理的节点（全图，或选中的那些），`edges` / `blocks` 只算两端都在其中的。`anchor`：结果的左上角
 * （全图整理为 (0, 0)；只整理选中的为它们原来包围盒的左上角）。框按里面的节点重算。 */
export function layoutGraph(input: { nodes: LayoutNode[]; edges: LayoutEdge[]; blocks?: LayoutBlock[]; boxes?: LayoutBox[]; anchor?: { x: number; y: number }; strict?: boolean }): LayoutResult {
  const { edges, blocks = [], boxes = [], strict = false } = input;
  if (!input.nodes.length) return { positions: {}, boxes: {} };
  const anchor = input.anchor ?? { x: 0, y: 0 };
  const before = Object.fromEntries(input.nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
  const crossBefore = countCrossings(before, input.nodes, edges);
  // 排到不动为止（第二次整理结果不变）
  let nodes = input.nodes;
  let pos: Record<string, { x: number; y: number }> = before;
  for (let k = 0; k < 6; k++) {
    let next = arrange(nodes, edges, blocks, strict);
    const minX = Math.min(...Object.values(next).map((p) => p.x)), minY = Math.min(...Object.values(next).map((p) => p.y));
    next = shift(next, anchor.x - minX, anchor.y - minY);
    const same = Object.keys(next).every((id) => pos[id] && pos[id].x === next[id].x && pos[id].y === next[id].y);
    pos = next;
    if (same) break;
    nodes = nodes.map((n) => ({ ...n, ...next[n.id] }));
  }
  // 迭代几轮后交叉比整理前多：退回第一轮（按整理前的位置排出来的那一次）的结果，若它交叉更少
  if (countCrossings(pos, input.nodes, edges) > crossBefore) {
    const kept = arrange(input.nodes.map((n) => ({ ...n })), edges, blocks, strict);
    const minX = Math.min(...Object.values(kept).map((p) => p.x)), minY = Math.min(...Object.values(kept).map((p) => p.y));
    const alt = shift(kept, anchor.x - minX, anchor.y - minY);
    if (countCrossings(alt, input.nodes, edges) < countCrossings(pos, input.nodes, edges)) pos = alt;
  }
  const size = new Map(input.nodes.map((n) => [n.id, n]));
  const outBoxes: LayoutResult["boxes"] = {};
  for (const b of boxes) {
    const ms = b.members.filter((m) => pos[m]);
    if (!ms.length) continue;
    const x1 = Math.min(...ms.map((m) => pos[m].x)), y1 = Math.min(...ms.map((m) => pos[m].y));
    const x2 = Math.max(...ms.map((m) => pos[m].x + size.get(m)!.w)), y2 = Math.max(...ms.map((m) => pos[m].y + size.get(m)!.h));
    outBoxes[b.id] = { x: x1 - LAYOUT.boxPad, y: y1 - LAYOUT.boxPad - LAYOUT.boxHead, w: x2 - x1 + 2 * LAYOUT.boxPad, h: y2 - y1 + 2 * LAYOUT.boxPad + LAYOUT.boxHead };
  }
  return { positions: pos, boxes: outBoxes };
}

/** 逐项块：「逐项开始」与同名「逐项结束」（参数「块名」相同）配成一对。 */
export function eachBlocks(nodes: { id: string; type: string; block?: unknown }[], begin = "core.each_begin", end = "core.each_end"): LayoutBlock[] {
  return nodes.filter((n) => n.type === begin).map((b) => ({
    begin: b.id,
    ends: nodes.filter((n) => n.type === end && String(n.block ?? "A") === String(b.block ?? "A")).map((n) => n.id),
  }));
}

/** 画布还没量过的节点（给模板离线整理时，tools/layout_graph.mjs）按类型估一个尺寸：宽按标题、端口名、节点上的参数名估
 * （至少 240，同 .gnode 的 min-width），高 = 标题 + 端口行（输入、输出并排，取多的一边）+ 节点上的参数行 + 底行。
 * 画布上用量出来的尺寸（graph/nodes.ts nodeSize），两者只差几个像素，行距留得出。 */
export function estimateSize(def: { label: string; inputs: { label: string }[]; outputs: { label: string }[]; params: { name: string; label: string }[]; on_node?: string[] } | undefined,
                             node: { label?: string; ui?: { on_node?: string[] }; promoted?: string[] }): { w: number; h: number } {
  if (!def) return { w: 240, h: 96 };
  const chars = (s: string) => [...s].reduce((n, c) => n + (c.charCodeAt(0) > 255 ? 1 : 0.55), 0);
  const rows = [...new Set([...(node.ui?.on_node ?? def.on_node ?? []), ...(node.promoted ?? [])])];
  const rowLabels = rows.map((r) => def.params.find((p) => p.name === r)?.label ?? r);
  const ports = Math.max(def.inputs.length, def.outputs.length);
  const pair = Array.from({ length: ports }, (_, i) => chars(def.inputs[i]?.label ?? "") + chars(def.outputs[i]?.label ?? ""));
  const w = Math.max(240, 13 * chars(node.label || def.label) + 70, ...pair.map((c) => 12 * c + 70), ...rowLabels.map((l) => 12 * chars(l) + 160));
  const h = 34 + ports * 20 + 8 + rows.length * 28 + 24 + 6;
  return { w: Math.round(w), h: Math.round(h) };
}
