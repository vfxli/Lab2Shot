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
 *    同分保持原有顺序（第一轮的顺序就是现在画布上从上到下的顺序）。再做相邻交换：先按相邻列的交叉（局部算），再按画出来的
 *    交叉试有限次（TRY_SWAPS，按次数不按时间，结果确定）。
 * 3. 坐标：列的 x = 前面各列最宽的节点累加 + 列距；列内按顺序从上到下堆、行距之外，做几轮「对齐拉直」：每个节点往它的
 *    主上游（第一根接进来的线）看齐、反向时往主下游看齐（最小二乘，保持顺序、不重叠），主线拉成一条直线、支线挂在旁边
 *    （Houdini / Nuke 的样子）。逐项块的「结束」对齐它的「开始」，尽量同一行高。
 * 4. 框：按框里的节点重算矩形（内边距、标题高度），不删框；没有节点的框不动。
 * 5. 整张图左上角对齐到 (0, 0)；只整理一部分（选中的）时放回它们原来包围盒的左上角。
 *
 * 6. 输入顺序：排好之后，一个节点的几个上游在同一列时，按它们接进来的输入口在节点上从上到下的顺序重新堆（「选择目标
 *    角色」前面的三个节点），各带着只喂给它的那一串上游。只在交叉不增加、不重叠、再排一次也不变时才采用，大排版不动
 *    （inputOrder）。
 *
 * 7. 再整理一次结果不变：见 layoutGraph（取一轮排版的不动点）。
 *
 * 8. 有分组框时按分组排（grouped，上面 1–6 不用）：每个框一条横带（框里的节点；一个节点在几个框里时归最小的那个），
 *    带与带上下排开、中间留 LAYOUT.zoneGap，框随内容伸缩、互不重叠。列仍是全图的依赖深度（连线永远从左到右），各带共用
 *    列宽，组内每列从上往下堆、按组内上游的重心排序。框外的「输出」及只通向输出的节点放在最右几列、整图中间高度；框里
 *    只通向输出的节点（各层的写出）推到本框最右。框外的其他节点：能挂在相邻的框带上又不落进框的列范围的就挂上去（在框
 *    左右，不进框），没有输入的单个节点（开关、常量）放在带与带之间的空当里，其余各自成一条不带框的带。带的上下顺序取
 *    跨带连线竖向总长最短的（同长保持现在的上下顺序）。组内各列先按组内上游重心排，再左右来回扫几遍取组内交叉最少的；
 *    「各列从带顶堆起」与「拉直（和组内上游同一行）」两种整图各排一次，取画出来交叉少的。没有分组框的图照旧用 1–6。
 *
 * 9. 折叠的框（LayoutBox.folded）按折叠后的大小当一个节点排：框里的节点不单独排，接进 / 接出它们的线改接到这个节点
 *    （框内互连的线去掉），包着它的展开框把它当成员；排好后框的左上角放到这个节点的位置（宽高照旧，展开时还是原样），
 *    框里的节点随框平移同样的距离。折叠的框因此收拢，不在原来展开的位置空出一大块。
 *
 * 速度（167 个节点的图约 0.3–0.4 秒）：交叉按横向范围排序后只比横向重叠的线；交换只数碰到那两个节点的线；place 用编号数组；
 * 不分区的排法只在分区排不出来、或 strict 要比较时才算。 */

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
  slot?: number; // 接到的输入口在节点上从上到下排第几（inputSlot）；没有就不参与「输入顺序」
}

/** 输入口在节点上从上到下的位置（同 lab2shot/nodes/base.py input_ports）：声明的输入口，然后常驻的参数口（wired_ports：
 * 「切换」的「走哪一路」），然后输入表的各行（「切换」的各路，NodeDef.ports_from），然后提升到节点的参数口（param:…，画在
 * 下面的参数行上，按参数顺序）。画布（graph/edit.ts）和模板整理（tools/layout_graph.mjs）用同一份。 */
export function inputSlot(def: { inputs: { name: string }[]; params: { name: string }[]; wired_ports?: string[]; ports_from?: string; ports_from_side?: string } | undefined,
                          params: Record<string, unknown> | undefined, port: string): number | undefined {
  if (!def) return undefined;
  const declared = def.inputs.findIndex((p) => p.name === port);
  if (declared >= 0) return declared;
  const standing = def.wired_ports ?? [];
  const name = port.startsWith("param:") ? port.slice(6) : "";
  if (name && standing.includes(name)) return def.inputs.length + standing.indexOf(name);
  const base = def.inputs.length + standing.length;
  const rows = def.ports_from && def.ports_from_side === "inputs" && Array.isArray(params?.[def.ports_from]) ? params![def.ports_from] as { name?: unknown }[] : [];
  const row = rows.findIndex((r) => r && r.name === port);
  if (row >= 0) return base + row;
  const param = name ? def.params.findIndex((p) => p.name === name) : -1;
  return param >= 0 ? base + rows.length + param : undefined;
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
  folded?: { w: number; h: number }; // 折叠着：画布上只剩这么大的标题栏（第 9 条）
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

/** 画出来的一根线：源节点右边中点到目标节点左边中点。lo / hi 是它横向占的范围（按 lo 排好序后只比横向重叠的）。 */
interface Seg {
  from: string;
  to: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  lo: number;
  hi: number;
}

function segOf(e: { from: string; to: string }, a: { x: number; y: number }, b: { x: number; y: number }, sa: { w: number; h: number }, sb: { w: number; h: number }): Seg {
  const x1 = a.x + sa.w, y1 = a.y + sa.h / 2, x2 = b.x, y2 = b.y + sb.h / 2;
  return { from: e.from, to: e.to, x1, y1, x2, y2, lo: Math.min(x1, x2), hi: Math.max(x1, x2) };
}

function segments(pos: Record<string, { x: number; y: number }>, size: Map<string, { w: number; h: number }>, edges: { from: string; to: string }[]): Seg[] {
  const out: Seg[] = [];
  for (const e of edges) {
    const a = pos[e.from], b = pos[e.to], sa = size.get(e.from), sb = size.get(e.to);
    if (a && b && sa && sb) out.push(segOf(e, a, b, sa, sb));
  }
  return out;
}

/** 两根线是否交叉（共用节点的不算）。 */
function crosses(p: Seg, q: Seg): boolean {
  if (q.lo >= p.hi || p.lo >= q.hi) return false;
  if (Math.max(p.y1, p.y2) < Math.min(q.y1, q.y2) || Math.max(q.y1, q.y2) < Math.min(p.y1, p.y2)) return false;
  if (p.from === q.from || p.to === q.to || p.from === q.to || p.to === q.from) return false;
  const d = (ax: number, ay: number, bx: number, by: number, cx: number, cy: number) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const d1 = d(p.x1, p.y1, p.x2, p.y2, q.x1, q.y1), d2 = d(p.x1, p.y1, p.x2, p.y2, q.x2, q.y2);
  if (!(d1 * d2 < 0)) return false;
  const d3 = d(q.x1, q.y1, q.x2, q.y2, p.x1, p.y1), d4 = d(q.x1, q.y1, q.x2, q.y2, p.x2, p.y2);
  return d3 * d4 < 0;
}

function crossingsOfSegs(segs: Seg[]): number {
  const s = [...segs].sort((a, b) => a.lo - b.lo);
  let n = 0;
  for (let i = 0; i < s.length; i++) {
    const p = s[i];
    for (let j = i + 1; j < s.length && s[j].lo < p.hi; j++) if (crosses(p, s[j])) n++;
  }
  return n;
}

/** 按画出来的样子数交叉：每根线是源节点右边中点到目标节点左边中点的线段，共用节点的两根不算。
 * 按横向范围排序后只比横向重叠的两根（结果与两两全比相同）。 */
export function countCrossings(pos: Record<string, { x: number; y: number }>, nodes: LayoutNode[], edges: LayoutEdge[]): number {
  return crossingsOfSegs(segments(pos, new Map(nodes.map((n) => [n.id, n])), edges));
}

/** 输入顺序不对的对数（文件开头第 6 条）：接进同一个节点、在同一列的两个上游，上下顺序和输入口的上下顺序相反的算一对。 */
export function outOfOrder(pos: Record<string, { x: number; y: number }>, edges: LayoutEdge[]): number {
  const slots = new Map<string, Map<string, number>>();
  for (const e of edges) {
    if (e.slot === undefined || !pos[e.from] || !pos[e.to] || e.from === e.to) continue;
    const m = slots.get(e.to) ?? slots.set(e.to, new Map()).get(e.to)!;
    m.set(e.from, Math.min(m.get(e.from) ?? Infinity, e.slot));
  }
  let n = 0;
  for (const m of slots.values()) {
    const list = [...m];
    for (let i = 0; i < list.length; i++)
      for (let j = i + 1; j < list.length; j++) {
        const [a, sa] = list[i], [b, sb] = list[j];
        if (pos[a].x === pos[b].x && sa !== sb && (pos[a].y - pos[b].y) * (sa - sb) < 0) n++;
      }
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

/** 一次排版里不变的东西（place 反复用）：节点 / 虚拟点编号、上下游（按连线顺序）、主上游 / 主下游、逐项块配对。 */
interface Net {
  idx: Map<string, number>;
  ids: string[];
  h: Float64Array;
  w: Float64Array;
  real: Uint8Array;
  up: number[][];
  down: number[][];
  extra: number[][];
  primaryUp: Int32Array;
  primaryDown: Int32Array;
}

function netOf(cols: Item[][], links: [string, string][], pull: [string, string][]): Net {
  const items = cols.flat();
  const idx = new Map(items.map((it, i) => [it.id, i]));
  const m = items.length;
  const up: number[][] = Array.from({ length: m }, () => []), down: number[][] = Array.from({ length: m }, () => []);
  const extra: number[][] = Array.from({ length: m }, () => []);
  for (const [a, b] of links) {
    const i = idx.get(a), j = idx.get(b);
    if (i === undefined || j === undefined) continue;
    down[i].push(j);
    up[j].push(i);
  }
  for (const [a, b] of pull) {
    const i = idx.get(a), j = idx.get(b);
    if (i === undefined || j === undefined) continue;
    extra[i].push(j);
    extra[j].push(i);
  }
  // 主上游：第一根接进来的线的源；主下游：只在那个下游的主上游正是它时（免得一个节点被好几个下游拉扯）
  const primaryUp = new Int32Array(m).map((_, i) => (up[i].length ? up[i][0] : -1));
  const primaryDown = new Int32Array(m).map((_, i) => down[i].find((d) => primaryUp[d] === i) ?? -1);
  return {
    idx, ids: items.map((it) => it.id), up, down, extra, primaryUp, primaryDown,
    h: Float64Array.from(items, (it) => it.h), w: Float64Array.from(items, (it) => it.w), real: Uint8Array.from(items, (it) => (it.real ? 1 : 0)),
  };
}

/** 一轮完整排版：给定每列的顺序，算坐标（只返回真节点）。 */
function place(order: Item[][], net: Net, mode: "primary" | "mean" = "primary", gap: number = LAYOUT.colGap): Record<string, { x: number; y: number }> {
  const { h, w, real, up, down, extra, primaryUp, primaryDown } = net;
  const m = net.ids.length;
  const X = new Float64Array(m), Y = new Float64Array(m);
  const cols = order.map((c) => c.map((it) => net.idx.get(it.id)!));
  let x = 0;
  for (const col of cols) {
    let y = 0, wide = 0;
    for (const i of col) {
      X[i] = x;
      Y[i] = y;
      y += h[i] + (real[i] ? LAYOUT.rowGap : LAYOUT.rowGap / 2);
      if (real[i]) wide = Math.max(wide, w[i]);
    }
    x += wide + gap;
  }
  const centre = (i: number) => Y[i] + h[i] / 2;
  const offs = new Float64Array(m), target = new Float64Array(m);
  const bStart = new Int32Array(m), bEnd = new Int32Array(m), bSum = new Float64Array(m), bN = new Int32Array(m);
  const settle = (col: number[], want: (i: number) => number | null) => {
    // 最小二乘地让每个节点的中心靠近它想去的地方，同时保持顺序、不重叠（相邻违例的块合并，取平均）
    let acc = 0;
    for (let k = 0; k < col.length; k++) {
      if (k > 0) acc += h[col[k - 1]] + (real[col[k - 1]] && real[col[k]] ? LAYOUT.rowGap : LAYOUT.rowGap / 2);
      offs[k] = acc;
    }
    for (let k = 0; k < col.length; k++) {
      const i = col[k], wt = want(i);
      target[k] = (wt === null ? centre(i) : wt) - h[i] / 2 - offs[k];
    }
    let nb = 0;
    for (let k = 0; k < col.length; k++) {
      bStart[nb] = k; bEnd[nb] = k; bSum[nb] = target[k]; bN[nb] = 1; nb++;
      while (nb > 1 && !(bSum[nb - 2] / bN[nb - 2] <= bSum[nb - 1] / bN[nb - 1])) {
        bEnd[nb - 2] = bEnd[nb - 1]; bSum[nb - 2] += bSum[nb - 1]; bN[nb - 2] += bN[nb - 1]; nb--;
      }
    }
    for (let b = 0; b < nb; b++) for (let k = bStart[b]; k <= bEnd[b]; k++) Y[col[k]] = bSum[b] / bN[b] + offs[k];
  };
  const reversed = [...cols].reverse();
  if (mode === "mean") {
    // 另一种：往全部上游 / 下游邻居的平均中心靠（主线不一定直，但多进多出的地方交叉可能更少）；只在它交叉更少时才用
    for (let round = 0; round < 6; round++) {
      const both = round >= 4;
      const withUp = both || round % 2 === 0, withDown = both || round % 2 === 1;
      for (const col of round % 2 === 0 ? cols : reversed)
        settle(col, (i) => {
          let sum = 0, n = 0;
          if (withUp) for (const j of up[i]) (sum += centre(j), n++);
          if (withDown) for (const j of down[i]) (sum += centre(j), n++);
          for (const j of extra[i]) (sum += 2 * centre(j), n += 2);
          return n ? sum / n : null;
        });
    }
  } else {
    // 对齐：每个节点往它的「主上游」看齐，像 Houdini / Nuke 那样主线拉成一条直线、支线挂在旁边；反向扫时往「主下游」
    // 看齐。逐项结束对齐它的逐项开始。
    for (let round = 0; round < 6; round++) {
      const backward = round % 2 === 1 && round < 5;
      for (const col of backward ? reversed : cols)
        settle(col, (i) => {
          if (backward) return primaryDown[i] >= 0 ? centre(primaryDown[i]) : null;
          const pair = extra[i].find((b) => X[b] < X[i]); // 逐项结束：对齐它的逐项开始
          const to = pair ?? (primaryUp[i] >= 0 ? primaryUp[i] : undefined);
          return to !== undefined ? centre(to) : null;
        });
    }
  }
  const pos: Record<string, { x: number; y: number }> = {};
  for (let i = 0; i < m; i++) if (real[i]) pos[net.ids[i]] = { x: X[i], y: Y[i] };
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
  const at = new Map(cur.flatMap((c) => c.map((it, i) => [it.id, i] as [string, number])));
  const sweep = (range: number[], near: Map<string, string[]>) => {
    for (const ci of range) {
      const keyed = cur[ci].map((it, i) => {
        const ns = (near.get(it.id) ?? []).map((n) => at.get(n)).filter((v): v is number => v !== undefined);
        return { it, i, k: ns.length ? ns.reduce((s, v) => s + v, 0) / ns.length : i };
      });
      keyed.sort((a, b) => a.k - b.k || a.i - b.i); // 同分保持原有顺序
      cur[ci] = keyed.map((k) => k.it);
      cur[ci].forEach((it, i) => at.set(it.id, i));
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

/** 相邻交换：列里上下相邻的两个，换过之后与左右两列之间的交叉少了就换，直到换不动（最多几轮）。只数这两个节点的线
 * （每对线的上下关系），一次比较是 O(两者度数之积)。没有换过返回 null。 */
function exchange(order: Item[][], links: [string, string][]): Item[][] | null {
  const up = new Map<string, string[]>(), down = new Map<string, string[]>();
  for (const [a, b] of links) {
    (down.get(a) ?? down.set(a, []).get(a)!).push(b);
    (up.get(b) ?? up.set(b, []).get(b)!).push(a);
  }
  const o = order.map((c) => [...c]);
  const at = new Map<string, number>();
  o.forEach((c) => c.forEach((it, i) => at.set(it.id, i)));
  // u 在 v 之上时，u、v 各自连到同一侧的邻居之间的交叉数
  const pairs = (u: string, v: string, near: Map<string, string[]>) => {
    let n = 0;
    for (const a of near.get(u) ?? []) for (const b of near.get(v) ?? []) if (at.get(a)! > at.get(b)!) n++;
    return n;
  };
  const cost = (u: string, v: string) => pairs(u, v, up) + pairs(u, v, down);
  let any = false;
  for (let pass = 0; pass < 8; pass++) {
    let improved = false;
    for (const col of o)
      for (let i = 0; i + 1 < col.length; i++) {
        const u = col[i].id, v = col[i + 1].id;
        if (cost(v, u) < cost(u, v)) {
          [col[i], col[i + 1]] = [col[i + 1], col[i]];
          at.set(u, i + 1);
          at.set(v, i);
          improved = any = true;
        }
      }
    if (!improved) break;
  }
  return any ? o : null;
}

/** 按画出来的交叉试换的最多次数（每次一整套 place）。 */
const TRY_SWAPS = 30;

/** u 在 v 之上时，两者连到左右相邻列的线之间的交叉数（相邻交换用）。 */
function localCost(u: string, v: string, at: Map<string, number>, net: Net): number {
  const iu = net.idx.get(u)!, iv = net.idx.get(v)!;
  let n = 0;
  for (const near of [net.up, net.down])
    for (const a of near[iu]) for (const b of near[iv]) if (at.get(net.ids[a])! > at.get(net.ids[b])!) n++;
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
  const seen = new Set<string>();
  const cands = orderings(cols, links).filter((o) => {
    const key = o.map((c) => c.map((it) => it.id).join("|")).join("\n");
    return !seen.has(key) && !!seen.add(key);
  });
  const net = netOf(cols, links, pull);
  const measure = (o: Item[][]) => {
    // 两种拉直都算，取画出来交叉少的（同分用主线拉直）
    const layer = layerCrossings(o, links);
    const [a, b] = (["primary", "mean"] as const).map((mode) => {
      const pos = place(o, net, mode, gap);
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
  // 再做几轮相邻交换（列里相邻两个换位置）：扫重心之后常还剩一两处能解开的。是否换按相邻列之间的交叉（只数这两个
  // 节点的线，局部算，很快）；换完的整套排序再按画出来的交叉和原来的比，少了才用
  const swapped = exchange(best.o, links);
  if (swapped) {
    const m = measure(swapped);
    if (better(m, best)) best = m;
  }
  // 再按画出来的交叉试换相邻两个（换过要重排坐标，一次要整套 place，贵）：只试换了以后相邻列交叉不变多的，且总共只试
  // 有限次（按次数，不按时间：结果要确定）
  let budget = TRY_SWAPS;
  for (let pass = 0; pass < 4 && budget > 0; pass++) {
    let improved = false;
    const at = new Map<string, number>();
    best.o.forEach((c) => c.forEach((it, i) => at.set(it.id, i)));
    for (let ci = 0; ci < best.o.length && budget > 0; ci++)
      for (let i = 0; i + 1 < best.o[ci].length && budget > 0; i++) {
        const u = best.o[ci][i].id, v = best.o[ci][i + 1].id;
        if (localCost(v, u, at, net) > localCost(u, v, at, net)) continue;
        budget--;
        const o = best.o.map((c) => [...c]);
        [o[ci][i], o[ci][i + 1]] = [o[ci][i + 1], o[ci][i]];
        const m = measure(o);
        if (better(m, best)) {
          best = m;
          at.set(u, i + 1);
          at.set(v, i);
          improved = true;
        }
      }
    if (!improved) break;
  }
  return best.pos;
}

// ------------------------------------------------------------------ 功能区：主链居中，支区成团放在上下，块前后留空

/** 每一组（同一区、同一列）里试着交换上下相邻的两个节点（两者占的总高度不变），画出来的交叉少了才换（原地改 pos）。
 * 只有碰到这两个节点的线会变：只数它们参与的交叉，换前换后比；交换只改上下位置，线的横向范围不变，按横向分格只和同格
 * 的线比。 */
function swapPass(pos: Record<string, { x: number; y: number }>, nodes: LayoutNode[], es: LayoutEdge[], groupOf: (id: string) => string): void {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const ids = nodes.map((n) => n.id).sort();
  const incident = new Map<string, number[]>(nodes.map((n) => [n.id, []]));
  es.forEach((e, k) => (incident.get(e.from)!.push(k), incident.get(e.to)!.push(k)));
  const segAt = (k: number, p: (id: string) => { x: number; y: number }) => segOf(es[k], p(es[k].from), p(es[k].to), byId.get(es[k].from)!, byId.get(es[k].to)!);
  const segs = es.map((_, k) => segAt(k, (id) => pos[id]));
  const CELL = 256;
  const span = (sg: Seg) => [Math.floor(sg.lo / CELL), Math.floor(sg.hi / CELL)];
  const grid = new Map<number, number[]>();
  const stamp = new Int32Array(segs.length), isMine = new Int32Array(segs.length);
  let tick = 0;
  const touching = (mine: number[], own: Seg[]) => {
    let n = 0;
    const skip = ++tick;
    for (const k of mine) isMine[k] = skip; // 自己这几根之间单独数
    for (let i = 0; i < own.length; i++) {
      for (let j = i + 1; j < own.length; j++) if (crosses(own[i], own[j])) n++;
      const mark = ++tick;
      const [c1, c2] = span(own[i]);
      for (let c = c1; c <= c2; c++)
        for (const k of grid.get(c) ?? []) {
          if (stamp[k] === mark || isMine[k] === skip) continue;
          stamp[k] = mark;
          if (crosses(own[i], segs[k])) n++;
        }
    }
    return n;
  };
  for (let pass = 0; pass < 4; pass++) {
    let improved = false;
    grid.clear();
    segs.forEach((sg, k) => {
      const [c1, c2] = span(sg);
      for (let c = c1; c <= c2; c++) (grid.get(c) ?? grid.set(c, []).get(c)!).push(k);
    });
    const groups = new Map<string, string[]>();
    for (const id of ids) {
      const g = `${groupOf(id)}|${pos[id].x}`;
      (groups.get(g) ?? groups.set(g, []).get(g)!).push(id);
    }
    for (const list of groups.values()) {
      list.sort((a, b) => pos[a].y - pos[b].y);
      for (let i = 0; i + 1 < list.length; i++) {
        const a = list[i], b = list[i + 1];
        const pa = pos[a], pb = pos[b];
        const gapAB = pb.y - (pa.y + byId.get(a)!.h);
        const na = { x: pa.x, y: pa.y + byId.get(b)!.h + gapAB }, nb = { x: pb.x, y: pa.y };
        const mine = [...new Set([...incident.get(a)!, ...incident.get(b)!])];
        if (!mine.length) continue;
        const moved = mine.map((k) => segAt(k, (id) => (id === a ? na : id === b ? nb : pos[id])));
        if (touching(mine, moved) < touching(mine, mine.map((k) => segs[k]))) {
          pos[a] = na;
          pos[b] = nb;
          mine.forEach((k, i) => (segs[k] = moved[i]));
          list[i] = b;
          list[i + 1] = a;
          improved = true;
        }
      }
    }
    if (!improved) break;
  }
}

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
  // 最后在每一区（主带、各支区）的每一列里试着交换上下相邻的两个节点（两者占的总高度不变），画出来的交叉少了才换。
  // 只有碰到这两个节点的线会变：只数它们参与的交叉，换前换后比
  const region = new Map<string, string>([...band].map((id) => [id, "band"]));
  for (const [k, list] of zones) for (const id of list) region.set(id, k);
  swapPass(pos, nodes, es, (id) => region.get(id)!);
  // 检查：连线全向右、不重叠，否则不用分区的排法
  if (es.some((e) => pos[e.from].x + byId.get(e.from)!.w > pos[e.to].x) || overlaps(pos, nodes).length) return null;
  return pos;
}

/** 一次排版：按功能区排（zoned，主链从主输入起）；排不出来（有线向左、重叠）时用不分区的排法（once）。
 * `strict`：分区的交叉比不分区的多时也用不分区的（分区把支线拉开，跨区的长线常会多交叉几处）。 */
function arrange(nodes: LayoutNode[], edges: LayoutEdge[], blocks: LayoutBlock[], strict = false): Record<string, { x: number; y: number }> {
  const flat = () => {
    const pos = once(nodes, edges, blocks);
    const ids = new Set(nodes.map((n) => n.id));
    swapPass(pos, nodes, edges.filter((e) => ids.has(e.from) && ids.has(e.to) && e.from !== e.to), () => "");
    return pos;
  };
  const zone = zoned(nodes, edges, blocks);
  if (!zone) return flat();
  if (!strict) return zone;
  const other = flat();
  return countCrossings(zone, nodes, edges) <= countCrossings(other, nodes, edges) ? zone : other;
}

/** 带的上下顺序（文件开头第 8 条）：跨带连线竖向总长最短。`links` 每根跨带的线一项：[带 a, 线在带 a 里的高度（相对带顶）,
 * 带 b, 在带 b 里的高度]；带 b 为 -1 时是连到最右「输出」的线，按到整图中间的距离算。带不多于 8 条时全试（按现在的上下
 * 顺序的字典序，同长取先出现的），多了从现在的顺序做插入式局部搜索。 */
function bandOrder(heights: number[], links: [number, number, number, number][], gap: number): number[] {
  const n = heights.length;
  const total = heights.reduce((s, h) => s + h, 0) + gap * Math.max(0, n - 1);
  const top = new Float64Array(n);
  const cost = (perm: number[]) => {
    let y = 0;
    for (const b of perm) (top[b] = y, y += heights[b] + gap);
    let c = 0;
    for (const [a, ya, b, yb] of links) c += Math.abs(top[a] + ya - (b < 0 ? total / 2 : top[b] + yb));
    return c;
  };
  let best = [...Array(n).keys()], bestCost = cost(best);
  if (n <= 8) {
    const perm: number[] = [], used = new Array(n).fill(false);
    const walk = () => {
      if (perm.length === n) {
        const c = cost(perm);
        if (c < bestCost - 1e-6) (bestCost = c, best = [...perm]);
        return;
      }
      for (let i = 0; i < n; i++) if (!used[i]) (used[i] = true, perm.push(i), walk(), perm.pop(), used[i] = false);
    };
    walk();
    return best;
  }
  for (let round = 0, improved = true; improved && round < 50; round++) {
    improved = false;
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const trial = [...best];
        trial.splice(j, 0, trial.splice(i, 1)[0]);
        const c = cost(trial);
        if (c < bestCost - 1e-6) (bestCost = c, best = trial, improved = true);
      }
  }
  return best;
}

/** 按分组框排（文件开头第 8 条）。`boxes` 的 members 是整理前算好的成员。排不出来（有向左的线、重叠）时返回 null。 */
export function grouped(nodes: LayoutNode[], edges: LayoutEdge[], boxes: LayoutBox[], straight = false): Record<string, { x: number; y: number }> | null {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const ids = nodes.map((n) => n.id);
  const index = new Map(ids.map((id, i) => [id, i]));
  const es = edges.filter((e) => byId.has(e.from) && byId.has(e.to) && e.from !== e.to);
  const succ = new Map<string, string[]>(ids.map((i) => [i, []])), pred = new Map<string, string[]>(ids.map((i) => [i, []]));
  for (const e of es) (succ.get(e.from)!.push(e.to), pred.get(e.to)!.push(e.from));
  // 组：节点归到含它的最小的框
  const band = new Map<string, string>();
  const bySize = boxes.map((b, i) => ({ b, i, n: b.members.filter((m) => byId.has(m)).length })).filter((o) => o.n)
    .sort((a, b) => a.n - b.n || a.i - b.i);
  if (!bySize.length) return null;
  for (const { b } of bySize) for (const m of b.members) if (byId.has(m) && !band.has(m)) band.set(m, b.id);
  const boxed = new Set(band.keys());
  // 输出：框外的交付节点，以及框外、下游全是输出的节点
  const deliver = new Set(ids.filter((i) => !boxed.has(i) && byId.get(i)!.delivers));
  for (let grew = true; grew;) {
    grew = false;
    for (const i of ids)
      if (!boxed.has(i) && !deliver.has(i) && succ.get(i)!.length && succ.get(i)!.every((s) => deliver.has(s))) (deliver.add(i), grew = true);
  }
  // 列：最长路径；没有输入的贴着最早的下游；输出在主体之后按它们自己的深度
  const col = new Map<string, number>();
  const visiting = new Set<string>();
  const depth = (id: string): number => {
    if (col.has(id)) return col.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const c = Math.max(-1, ...pred.get(id)!.map(depth)) + 1;
    visiting.delete(id);
    col.set(id, c);
    return c;
  };
  ids.forEach(depth);
  const body = ids.filter((i) => !deliver.has(i));
  const pullSources = () => {
    for (const i of body) if (!pred.get(i)!.length && succ.get(i)!.length) col.set(i, Math.max(0, Math.min(...succ.get(i)!.map((s) => col.get(s)!)) - 1));
  };
  for (let k = 0; k < 3; k++) pullSources();
  const maxBody = Math.max(-1, ...body.map((i) => col.get(i)!));
  const inner = new Map<string, number>();
  const innerDepth = (id: string, seen = new Set<string>()): number => {
    if (inner.has(id)) return inner.get(id)!;
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = pred.get(id)!.filter((p) => deliver.has(p));
    const d = ps.length ? Math.max(...ps.map((p) => innerDepth(p, seen) + 1)) : 0;
    inner.set(id, d);
    return d;
  };
  for (const i of deliver) col.set(i, maxBody + 1 + innerDepth(i));
  // 只通向输出的节点（下游全是输出，或只有一个下游且它只通向输出）推到紧挨着下游的前一列
  const late = new Set<string>();
  for (let grew = true; grew;) {
    grew = false;
    for (const i of body) {
      const s = succ.get(i)!;
      if (late.has(i) || !s.length) continue;
      if (s.every((t) => deliver.has(t)) || (s.length === 1 && late.has(s[0]))) (late.add(i), grew = true);
    }
  }
  for (let k = 0; k < 6; k++) {
    for (const i of [...late].sort((a, b) => col.get(b)! - col.get(a)! || index.get(a)! - index.get(b)!))
      col.set(i, Math.min(...succ.get(i)!.map((t) => (deliver.has(t) ? maxBody + 1 : col.get(t)!))) - 1);
    pullSources();
  }
  // 去掉空列
  const used = [...new Set(col.values())].sort((a, b) => a - b);
  const renum = new Map(used.map((c, k) => [c, k]));
  for (const i of ids) col.set(i, renum.get(col.get(i)!)!);
  const colW = new Array(used.length).fill(0);
  for (const i of ids) colW[col.get(i)!] = Math.max(colW[col.get(i)!], byId.get(i)!.w);
  const colX: number[] = [];
  for (let c = 0, x = 0; c < used.length; c++) (colX.push(x), (x += colW[c] + LAYOUT.colGap));
  // 框外的其他节点：按彼此的连线连成团
  const free = body.filter((i) => !boxed.has(i));
  const root = new Map(free.map((i) => [i, i]));
  const find = (i: string): string => (root.get(i) === i ? i : (root.set(i, find(root.get(i)!)), root.get(i)!));
  for (const e of es)
    if (root.has(e.from) && root.has(e.to)) {
      const a = find(e.from), b = find(e.to);
      if (a !== b) root.set(index.get(a)! < index.get(b)! ? b : a, index.get(a)! < index.get(b)! ? a : b);
    }
  const clumps = new Map<string, string[]>();
  for (const i of free) (clumps.get(find(i)) ?? clumps.set(find(i), []).get(find(i))!).push(i);
  const range = new Map<string, [number, number]>();
  for (const i of boxed) {
    const r = range.get(band.get(i)!), c = col.get(i)!;
    range.set(band.get(i)!, r ? [Math.min(r[0], c), Math.max(r[1], c)] : [c, c]);
  }
  const boxOrder = boxes.map((b) => b.id).filter((id) => range.has(id));
  const gapNodes: string[] = [];
  const bands = [...boxOrder];
  for (const list of clumps.values()) {
    // 挂到连线最多的相邻框带上：只要团里的节点都不落进那个框的列范围（在框的左右，不进框）
    const count = new Map<string, number>();
    for (const i of list) for (const j of [...pred.get(i)!, ...succ.get(i)!]) if (boxed.has(j)) count.set(band.get(j)!, (count.get(band.get(j)!) ?? 0) + 1);
    const pickBand = [...count].sort((a, b) => b[1] - a[1] || boxOrder.indexOf(a[0]) - boxOrder.indexOf(b[0]))
      .map(([b]) => b).find((b) => list.every((i) => col.get(i)! < range.get(b)![0] || col.get(i)! > range.get(b)![1]));
    if (pickBand) list.forEach((i) => band.set(i, pickBand));
    else if (list.length === 1 && !pred.get(list[0])!.length) gapNodes.push(list[0]);
    else {
      const id = `~free~${list[0]}`;
      list.forEach((i) => band.set(i, id));
      bands.push(id);
    }
  }
  // 每条带自己从上往下排（相对带顶）：带框的留出标题和内边距
  const ROW = LAYOUT.rowGap, PAD = LAYOUT.boxPad, HEAD = LAYOUT.boxHead;
  const rel = new Map<string, number>();
  const heights = bands.map((b) => {
    const framed = range.has(b);
    const start = framed ? HEAD + PAD : 0;
    const members = ids.filter((i) => band.get(i) === b);
    const cols = [...new Set(members.map((i) => col.get(i)!))].sort((p, q) => p - q);
    const byCol = new Map<number, string[]>(cols.map((c) => [c, members.filter((i) => col.get(i) === c)]));
    const inBand = es.filter((e) => band.get(e.from) === b && band.get(e.to) === b);
    const mine = members.map((i) => byId.get(i)!);
    // 按各列现在的顺序从上往下堆（拉直时尽量和组内上游同一行，只往下让）
    const stack = () => {
      let bottom = start;
      for (const c of cols) {
        let y = start;
        for (const i of byCol.get(c)!) {
          const ups = straight ? pred.get(i)!.filter((p) => band.get(p) === b && col.get(p)! < c).map((p) => rel.get(p)!) : [];
          if (ups.length) y = Math.max(y, Math.min(...ups));
          rel.set(i, y);
          y += byId.get(i)!.h + ROW;
        }
        bottom = Math.max(bottom, y - ROW);
      }
      return bottom;
    };
    const centreOf = (i: string) => rel.get(i)! + byId.get(i)!.h / 2;
    // 一列按组内邻居（上游或下游）的重心排；没有这类邻居的按它现在的位置
    const sortBy = (c: number, near: Map<string, string[]>, first: boolean) => {
      const key = new Map(byCol.get(c)!.map((i, k) => {
        const ys = near.get(i)!.filter((j) => band.get(j) === b && rel.has(j)).map(centreOf);
        return [i, ys.length ? ys.reduce((s0, v) => s0 + v, 0) / ys.length : first ? -1e9 + k : rel.has(i) ? centreOf(i) : k];
      }));
      byCol.get(c)!.sort((p, q) => key.get(p)! - key.get(q)! || index.get(p)! - index.get(q)!);
    };
    // 第一遍：从左往右按组内上游排；再左右来回扫几遍，取组内交叉最少的一次（同分取先出现的）
    for (const c of cols) (sortBy(c, pred, true), stack());
    const snap = () => new Map([...byCol].map(([c, l]) => [c, [...l]]));
    const crossing = () => countCrossings(Object.fromEntries(members.map((i) => [i, { x: colX[col.get(i)!], y: rel.get(i)! }])), mine, inBand);
    let best = snap(), bestN = crossing();
    for (let round = 0; round < 4 && bestN > 0; round++) {
      for (const c of [...cols].reverse()) sortBy(c, succ, false);
      stack();
      for (const c of cols) (sortBy(c, pred, false), stack());
      const n = crossing();
      if (n < bestN) (best = snap(), bestN = n);
    }
    for (const [c, l] of best) byCol.set(c, l);
    return stack() + (framed ? PAD : 0);
  });
  // 带的上下顺序：从现在的上下顺序出发，跨带连线最短
  const bandIdx = new Map(bands.map((b, k) => [b, k]));
  const nowY = bands.map((b) => {
    const ms = ids.filter((i) => band.get(i) === b);
    return ms.reduce((s, i) => s + byId.get(i)!.y + byId.get(i)!.h / 2, 0) / ms.length;
  });
  const start = [...bands.keys()].sort((a, b) => nowY[a] - nowY[b] || a - b);
  const startAt = new Map(start.map((b, k) => [b, k]));
  const links: [number, number, number, number][] = [];
  const mid = (i: string) => rel.get(i)! + byId.get(i)!.h / 2;
  for (const e of es) {
    const a = bandIdx.get(band.get(e.from) ?? ""), b = bandIdx.get(band.get(e.to) ?? "");
    if (a !== undefined && b !== undefined && a !== b) links.push([startAt.get(a)!, mid(e.from), startAt.get(b)!, mid(e.to)]);
    else if (a !== undefined && b === undefined && deliver.has(e.to)) links.push([startAt.get(a)!, mid(e.from), -1, 0]);
    else if (b !== undefined && a === undefined && deliver.has(e.from)) links.push([startAt.get(b)!, mid(e.to), -1, 0]);
  }
  const perm = bandOrder(start.map((b) => heights[b]), links, LAYOUT.zoneGap).map((k) => start[k]);
  const pos: Record<string, { x: number; y: number }> = {};
  const spans: [number, number][] = [];
  let top = 0;
  for (const b of perm) {
    for (const i of ids) if (band.get(i) === bands[b]) pos[i] = { x: colX[col.get(i)!], y: top + rel.get(i)! };
    spans.push([top, top + heights[b]]);
    top += heights[b] + LAYOUT.zoneGap;
  }
  const total = Math.max(0, top - LAYOUT.zoneGap);
  // 输出：每列按上游的重心排，整列竖着居中在整图中间
  const centre = (i: string) => pos[i].y + byId.get(i)!.h / 2;
  const outCols = new Map<number, string[]>();
  for (const i of ids) if (deliver.has(i)) (outCols.get(col.get(i)!) ?? outCols.set(col.get(i)!, []).get(col.get(i)!)!).push(i);
  for (const c of [...outCols.keys()].sort((a, b) => a - b)) {
    const want = (i: string) => {
      const ys = pred.get(i)!.filter((p) => pos[p]).map(centre);
      return ys.length ? ys.reduce((s, v) => s + v, 0) / ys.length : total / 2;
    };
    const list = outCols.get(c)!.sort((a, b) => want(a) - want(b) || index.get(a)! - index.get(b)!);
    const h = list.reduce((s, i) => s + byId.get(i)!.h, 0) + ROW * (list.length - 1);
    let y = Math.round(total / 2 - h / 2);
    for (const i of list) (pos[i] = { x: colX[col.get(i)!], y }, (y += byId.get(i)!.h + ROW));
  }
  // 开关、常量：放进带与带之间离下游最近、放得下的空当（居中）；都放不下就放到最下面
  let below = total + LAYOUT.zoneGap;
  for (const i of gapNodes) {
    const n = byId.get(i)!, x = colX[col.get(i)!];
    const ys = succ.get(i)!.filter((t) => pos[t]).map(centre);
    const want = ys.length ? ys.reduce((s, v) => s + v, 0) / ys.length : total / 2;
    const fits = spans.slice(1).map((s, k) => [spans[k][1], s[0]]).filter(([a, b]) => b - a >= n.h + 20)
      .map(([a, b]) => Math.round((a + b) / 2 - n.h / 2)).sort((a, b) => Math.abs(a + n.h / 2 - want) - Math.abs(b + n.h / 2 - want) || a - b);
    const y = fits.find((y0) => !gapNodes.some((o) => o !== i && pos[o] && pos[o].x < x + n.w && x < pos[o].x + byId.get(o)!.w
      && pos[o].y < y0 + n.h && y0 < pos[o].y + byId.get(o)!.h));
    if (y !== undefined) pos[i] = { x, y };
    else (pos[i] = { x, y: below }, (below += n.h + ROW));
  }
  swapPass(pos, nodes, es, (i) => band.get(i) ?? (deliver.has(i) ? "~deliver" : `~gap~${i}`));
  if (es.some((e) => pos[e.from].x + byId.get(e.from)!.w > pos[e.to].x) || overlaps(pos, nodes).length) return null;
  return pos;
}

/** 分组排两种（组内各列从带顶堆起 / 拉直），取画出来交叉少的（同分取前一种）。 */
function groupedBest(nodes: LayoutNode[], edges: LayoutEdge[], boxes: LayoutBox[]): Record<string, { x: number; y: number }> | null {
  const tries = [grouped(nodes, edges, boxes), grouped(nodes, edges, boxes, true)].filter((p): p is Record<string, { x: number; y: number }> => !!p);
  if (!tries.length) return null;
  const n = tries.map((p) => countCrossings(p, nodes, edges));
  return tries[n.length > 1 && n[1] < n[0] ? 1 : 0];
}

const shift = (pos: Record<string, { x: number; y: number }>, dx: number, dy: number) =>
  Object.fromEntries(Object.entries(pos).map(([id, p]) => [id, { x: Math.round(p.x + dx), y: Math.round(p.y + dy) }]));

type Pos = Record<string, { x: number; y: number }>;

const samePos = (a: Pos, b: Pos) => Object.keys(a).every((id) => b[id] && b[id].x === a[id].x && b[id].y === a[id].y);

/** 整理：`nodes` 是要整理的节点（全图，或选中的那些），`edges` / `blocks` 只算两端都在其中的。`anchor`：结果的左上角
 * （全图整理为 (0, 0)；只整理选中的为它们原来包围盒的左上角）。框按里面的节点重算。
 *
 * 不动点（再整理一次结果不变）：一轮排版 `run`（arrange，再平移取整到 anchor）只看节点现在的位置，结果记作 run(位置)。
 * 1. 从现在的位置反复 run，直到 run(P) = P（P 是 run 的不动点）；
 * 2. 再从一个与现在位置无关的起点（全部节点在原点：起始顺序只按尺寸、id）反复 run，得到另一个不动点 H；
 * 3. 两者取画出来交叉少的，一样多取 1（离现在的样子近）；
 * 4. 输入顺序（inputOrder）只接受仍是 run 的不动点、交叉不多的结果，并做到不能再做为止。
 * 对结果再整理：1 立刻停在它自己（它是 run 的不动点）；2 与起点无关，还是 H；3 它的交叉不比 H 多，同分取它；4 上次
 * 最后一遍已经一个都不能做，这次也一样——所以结果不变。两种都转圈（run 的迭代不收敛，几乎不会）时用 H 那条迭代的最后
 * 一次结果 R，且输入正是 R 时原样返回 R。 */
export function layoutGraph(input: { nodes: LayoutNode[]; edges: LayoutEdge[]; blocks?: LayoutBlock[]; boxes?: LayoutBox[]; anchor?: { x: number; y: number }; strict?: boolean }): LayoutResult {
  const folded = (input.boxes ?? []).filter((b) => b.folded && b.members.length);
  if (!folded.length) return layoutOpen(input);
  // 第 9 条：每个折叠的框换成一个它折叠后大小的节点（id 用框的 id），成员不参与排版
  const owner = new Map<string, LayoutBox>();
  for (const b of folded) for (const m of b.members) if (!owner.has(m)) owner.set(m, b);
  const as = (id: string) => owner.get(id)?.id ?? id;
  const nodes: LayoutNode[] = [
    ...input.nodes.filter((n) => !owner.has(n.id)),
    ...folded.filter((b) => input.nodes.some((n) => owner.get(n.id) === b)).map((b) => ({ id: b.id, x: b.x, y: b.y, w: b.folded!.w, h: b.folded!.h })),
  ];
  const edges = input.edges.filter((e) => as(e.from) !== as(e.to))
    .map((e) => (owner.has(e.from) || owner.has(e.to) ? { from: as(e.from), to: as(e.to), param: e.param } : e));
  const shown = new Set(nodes.map((n) => n.id));
  const blocks = (input.blocks ?? []).filter((k) => shown.has(k.begin) && k.ends.every((x) => shown.has(x)));
  const boxes = (input.boxes ?? []).filter((b) => !b.folded).map((b) => ({ ...b, members: [...new Set(b.members.map(as))] }));
  const out = layoutOpen({ ...input, nodes, edges, blocks, boxes });
  const positions: Record<string, { x: number; y: number }> = {};
  const outBoxes: LayoutResult["boxes"] = { ...out.boxes };
  for (const n of input.nodes) {
    const b = owner.get(n.id);
    const p = b ? out.positions[b.id] : out.positions[n.id];
    if (p) positions[n.id] = b ? { x: Math.round(n.x + p.x - b.x), y: Math.round(n.y + p.y - b.y) } : p;
  }
  for (const b of folded) {
    const p = out.positions[b.id];
    if (p) outBoxes[b.id] = { x: p.x, y: p.y, w: b.w, h: b.h };
  }
  return { positions, boxes: outBoxes };
}

/** 整理（没有折叠的框）：见 layoutGraph。 */
function layoutOpen(input: { nodes: LayoutNode[]; edges: LayoutEdge[]; blocks?: LayoutBlock[]; boxes?: LayoutBox[]; anchor?: { x: number; y: number }; strict?: boolean }): LayoutResult {
  const { edges, blocks = [], boxes = [], strict = false } = input;
  if (!input.nodes.length) return { positions: {}, boxes: {} };
  const anchor = input.anchor ?? { x: 0, y: 0 };
  const inGraph = new Set(input.nodes.map((n) => n.id));
  const hasGroups = boxes.some((b) => b.members.some((m) => inGraph.has(m)));
  const run = (p: Pos): Pos => {
    const ns = input.nodes.map((n) => ({ ...n, ...p[n.id] }));
    const next = (hasGroups && groupedBest(ns, edges, boxes)) || arrange(ns, edges, blocks, strict);
    let minX = Infinity, minY = Infinity;
    for (const q of Object.values(next)) (minX = Math.min(minX, q.x), minY = Math.min(minY, q.y));
    return shift(next, anchor.x - minX, anchor.y - minY);
  };
  // 反复 run 到不动点；转圈返回 null（以及最后一次结果）
  const settle = (from: Pos): { fixed: Pos | null; last: Pos } => {
    let pos = run(from);
    if (samePos(pos, from)) return { fixed: from, last: from };
    for (let k = 0; k < 8; k++) {
      const next = run(pos);
      if (samePos(next, pos)) return { fixed: pos, last: pos };
      pos = next;
    }
    return { fixed: null, last: pos };
  };
  const now: Pos = Object.fromEntries(input.nodes.map((n) => [n.id, { x: n.x, y: n.y }]));
  const home = settle(Object.fromEntries(input.nodes.map((n) => [n.id, { x: 0, y: 0 }])));
  if (!home.fixed && samePos(home.last, now)) return finish(home.last, input.nodes, boxes);
  const here = settle(now);
  const cands = [here.fixed, home.fixed].filter((p): p is Pos => !!p);
  if (!cands.length) return finish(home.last, input.nodes, boxes);
  const cross = cands.map((p) => countCrossings(p, input.nodes, edges));
  let pos = cands[cross[1] !== undefined && cross[1] < cross[0] ? 1 : 0];
  if (!hasGroups) pos = inputOrder(pos, input.nodes, edges, run);
  return finish(pos, input.nodes, boxes);
}

/** 框按里面的节点重算矩形。 */
function finish(pos: Pos, nodes: LayoutNode[], boxes: LayoutBox[]): LayoutResult {
  const size = new Map(nodes.map((n) => [n.id, n]));
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

/** 输入顺序（文件开头第 6 条）：同一列里接进同一个节点的几个上游，按输入口的上下顺序重新堆（占的那段高度和间距照旧），
 * 每个上游带着只通向它的那一串上游一起上下挪，再让排版从这里走一遍（`again`：下游跟着对齐到新的主上游）。采用的条件：
 * 顺序不对的对数少了、画出来的交叉没多、不重叠，并且那个结果再排也不变——所以「再整理一次」不会挪回去。 */
function inputOrder(start: Record<string, { x: number; y: number }>, nodes: LayoutNode[], edges: LayoutEdge[],
                    again: (pos: Record<string, { x: number; y: number }>) => Record<string, { x: number; y: number }>): Record<string, { x: number; y: number }> {
  let pos = start;
  const size = new Map(nodes.map((n) => [n.id, n]));
  const es = edges.filter((e) => pos[e.from] && pos[e.to] && e.from !== e.to);
  const preds = new Map<string, string[]>(), succs = new Map<string, string[]>();
  for (const e of es) {
    (preds.get(e.to) ?? preds.set(e.to, []).get(e.to)!).push(e.from);
    (succs.get(e.from) ?? succs.set(e.from, []).get(e.from)!).push(e.to);
  }
  // 只通向 `id` 的上游：它们的下游全在这一串里
  const own = (id: string) => {
    const group = new Set([id]);
    for (let grew = true; grew;) {
      grew = false;
      for (const g of [...group])
        for (const p of preds.get(g) ?? [])
          if (!group.has(p) && (succs.get(p) ?? []).every((s) => group.has(s))) (group.add(p), grew = true);
    }
    return group;
  };
  const same = (a: Record<string, { x: number; y: number }>, b: Record<string, { x: number; y: number }>) =>
    Object.keys(a).every((id) => b[id] && Math.round(b[id].x) === Math.round(a[id].x) && Math.round(b[id].y) === Math.round(a[id].y));
  const targets = [...new Set(es.filter((e) => e.slot !== undefined).map((e) => e.to))].sort();
  // 做到一遍下来一个都不能做为止（每做一次顺序不对的对数严格变少，必然停）；这样对结果再做一次，第一遍就什么都不做
  for (let pass = 0; ; pass++) {
    if (pass > 64) return start; // 防御：不会到这里
    let moved = false;
    for (const t of targets) {
      const slot = new Map<string, number>();
      for (const e of es) if (e.to === t && e.slot !== undefined) slot.set(e.from, Math.min(slot.get(e.from) ?? Infinity, e.slot));
      const byCol = new Map<number, string[]>();
      for (const id of slot.keys()) (byCol.get(pos[id].x) ?? byCol.set(pos[id].x, []).get(pos[id].x)!).push(id);
      for (const list of byCol.values()) {
        if (list.length < 2) continue;
        const now = [...list].sort((a, b) => pos[a].y - pos[b].y);
        const want = [...list].sort((a, b) => slot.get(a)! - slot.get(b)! || pos[a].y - pos[b].y);
        if (now.every((id, i) => id === want[i])) continue;
        // 原来那段的间距照旧，按新顺序从最上面往下堆
        const gaps = now.slice(1).map((id, i) => pos[id].y - (pos[now[i]].y + size.get(now[i])!.h));
        const trial = { ...pos };
        let y = pos[now[0]].y;
        want.forEach((id, i) => {
          const dy = y - pos[id].y;
          for (const g of own(id)) trial[g] = { x: pos[g].x, y: Math.round(pos[g].y + dy) };
          y += size.get(id)!.h + (gaps[i] ?? 0);
        });
        // 挪过之后让排版再走一遍（下游跟着对齐到新的主上游），那个结果再排也不变才算数
        const settled = again(trial);
        if (outOfOrder(settled, es) < outOfOrder(pos, es) && countCrossings(settled, nodes, es) <= countCrossings(pos, nodes, es)
            && !overlaps(settled, nodes).length && same(settled, again(settled))) {
          pos = settled;
          moved = true;
        }
      }
    }
    if (!moved) return pos;
  }
}

/** 节点类型在块里的角色：块的开始 "begin"、块的结束 "end"，其他没有。 */
export type ScopeRole = (type: string) => string | undefined;

/** 按节点定义里的角色（`scope_role`，同 lab2shot/engine/scopes.py 的声明）查。目前服务器的 NodeDef.describe() 还没有
 * 输出这个字段；定义里一个都没有时退回到已知的「逐项开始 / 逐项结束」类型（foreach_begin / foreach_end），有了就只看字段。 */
export function scopeRoles(defs: Record<string, object | undefined>): ScopeRole {
  const roleOf = (d: object | undefined) => {
    const r = d && (d as { scope_role?: unknown }).scope_role;
    return typeof r === "string" && r ? r : undefined;
  };
  return (type) => roleOf(defs[type]);
}

/** 块：角色为「开始」的节点与同名（参数「块名」相同，没有按 "A"）的「结束」配成一对。`roles`：节点类型 → 角色，
 * 传节点定义（按 scopeRoles：服务器下发的 NodeTypeDef.scope_role）或一个查询函数；从不按类型名认。 */
export function eachBlocks(nodes: { id: string; type: string; block?: unknown }[],
                           roles: ScopeRole | Record<string, object | undefined>): LayoutBlock[] {
  const role = typeof roles === "function" ? roles : scopeRoles(roles);
  return nodes.filter((n) => role(n.type) === "begin").map((b) => ({
    begin: b.id,
    ends: nodes.filter((n) => role(n.type) === "end" && String(n.block ?? "A") === String(b.block ?? "A")).map((n) => n.id),
  }));
}

/** 节点头三行的字宽（px）：第一行副标题 .gnode-subtitle（Geist Medium 12px），第二行节点名 .gnode-title（Geist 10.5px，
 * 按 12px 的字宽乘 0.875），第三行类型名 .gnode-type（Geist Mono 10.5px，等宽 7）。按无头 Chromium 载入 webui/src/fonts 量出的每个字符的宽；表里没有的 ASCII 按 7，非 ASCII 按 12（汉字）。 */
const TITLE_W: Record<string, number> = {
  ...Object.fromEntries([..."abcdeghnopquvyz"].map((c) => [c, 7])), f: 4, i: 3, j: 3, k: 6, l: 3, m: 11, r: 5, s: 6, t: 4.5, w: 10, x: 6,
  ...Object.fromEntries([..."ABDKPRSUVX"].map((c) => [c, 8])), ...Object.fromEntries([..."CGHNOQ"].map((c) => [c, 9])),
  ...Object.fromEntries([..."EFJLTYZ"].map((c) => [c, 7])), I: 3, M: 11, W: 12,
  ...Object.fromEntries([..."02345"].map((c) => [c, 8])), 1: 5, 6: 7, 7: 6, 8: 7, 9: 7, _: 7, ".": 3, "-": 5, " ": 3,
};
const titleWidth = (s: string) => [...s].reduce((n, c) => n + (TITLE_W[c] ?? (c.charCodeAt(0) > 255 ? 12 : 7)), 0);
const typeWidth = (s: string) => [...s].reduce((n, c) => n + (c.charCodeAt(0) > 255 ? 12 : 7), 0);

/** 画布还没量过的节点（给模板离线整理时，tools/layout_graph.mjs）按类型估一个尺寸。
 * 宽：节点头三行——副标题（def.subtitle，当前语言）、节点名（node.id）与类型名（def.id，没有时用 node.type）取最宽的那个，加上头部其余部分（左右内边距 8+8、
 * 色条 3+4、间距 4×3、状态格 60、「在视图中显示」16、边框 2，共 113，取 116）；端口行、节点上的参数行照旧；至少 240
 * （同 .gnode 的 min-width）。高 = 三行头（约 53：副标题 15 + 节点名 13 + 类型名 13 + 间距 2 + 内边距 9 + 边线 1）+ 端口行（输入、输出并排，取多的一边）+ 节点上的参数行 + 底行。
 * 画布上用量出来的尺寸（graph/nodes.ts nodeSize），两者只差几个像素，行距留得出。 */
export function estimateSize(def: { id?: string; subtitle?: string; inputs: { label: string }[]; outputs: { label: string }[]; params: { name: string; label: string }[]; on_node?: string[] } | undefined,
                             node: { id: string; type?: string; ui?: { on_node?: string[] }; promoted?: string[] }): { w: number; h: number } {
  const head = Math.max(titleWidth(def?.subtitle ?? ""), titleWidth(node.id) * 0.875, typeWidth(def?.id ?? node.type ?? "")) + 116;
  if (!def) return { w: Math.round(Math.max(240, head)), h: 109 };
  const chars = (s: string) => [...s].reduce((n, c) => n + (c.charCodeAt(0) > 255 ? 1 : 0.55), 0);
  const rows = [...new Set([...(node.ui?.on_node ?? def.on_node ?? []), ...(node.promoted ?? [])])];
  const rowLabels = rows.map((r) => def.params.find((p) => p.name === r)?.label ?? r);
  const ports = Math.max(def.inputs.length, def.outputs.length);
  const pair = Array.from({ length: ports }, (_, i) => chars(def.inputs[i]?.label ?? "") + chars(def.outputs[i]?.label ?? ""));
  const w = Math.max(240, head, ...pair.map((c) => 12 * c + 70), ...rowLabels.map((l) => 12 * chars(l) + 160));
  const h = 53 + ports * 20 + 8 + rows.length * 28 + 24 + 6;
  return { w: Math.round(w), h: Math.round(h) };
}
