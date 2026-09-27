// 算法目录的浏览器执行器：按 lab2shot/ops/ops.toml 的描述运行。
//
// 服务器端执行器为 lab2shot/ops/run.py，按同一份描述运行。两端对照测试：tests/test_ops_crossend.py
// （判据：像素最大差 ≤ 1/255。视图用于预览，不要求逐位一致）。
//
// 词汇表与精度规定见 lab2shot/ops/vocab.py（三种运算：每像素算术、按框涂、挑条目）。
// 新增算法时无需修改本文件，只需在 ops.toml 中增加一条描述。
//
// 每像素算术使用普通 TypeScript 逐像素计算；改用 GPU 着色器时只需替换 runPixel，描述与调用方均不变。

// 此处的导入须带 `.ts` 扩展名（全项目唯一的例外）：文件名本身含有点号（`ops.gen.ts` 为生成文件），
// node 会将 `./ops.gen` 视为已带扩展名，测试的解析钩子（webui/tests/support/tsExtResolve.mjs）
// 因而不会再尝试 `.ts`。其他位置一律不带扩展名。
import { OPS } from "./ops.gen.ts";

export type Desc = {
  kind: string;
  desc: string;
  expr?: string;
  axis?: string;
  combine?: string;
  accumulate?: string;
  base?: number;
  rules?: readonly string[];
  count?: string;
};

export const CATALOG = OPS as unknown as Record<string, Desc>;

/** 一张二维数据：w × h 个像素，每个像素 c 条通道，按像素、通道顺序连续存放。 */
export type Pix = { w: number; h: number; c: number; data: Float32Array };

/** 一个条目（列表中的一条或一个人物）：名称、编号、各帧的框，均为可选。 */
export type Item = { name?: string; id?: number; boxes?: Record<string, number[]> };

export class BadOp extends Error {}

// ---------------------------------------------------------------- 公式文法（与 vocab.py 完全一致）

const FUNCTIONS: Record<string, number> = { min: 2, max: 2, clamp: 3 };

export type Tree =
  | { k: "num"; v: number }
  | { k: "var"; n: string }
  | { k: "neg"; x: Tree }
  | { k: "bin"; o: string; l: Tree; r: Tree }
  | { k: "call"; f: string; a: Tree[] };

type Tok = { k: "num" | "name" | "sym"; t: string };

const TOKEN = /\s*(?:(\d+\.?\d*(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z_0-9]*)|([-+*/(),]))/y;

function tokens(text: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < text.length) {
    TOKEN.lastIndex = i;
    const m = TOKEN.exec(text);
    if (!m) {
      if (text.slice(i).trim() === "") break;
      throw new BadOp(`unexpected character in ${text}: ${text.slice(i)}`);
    }
    i = TOKEN.lastIndex;
    out.push(m[1] ? { k: "num", t: m[1] } : m[2] ? { k: "name", t: m[2] } : { k: "sym", t: m[3] });
  }
  return out;
}

/** 将一条公式解析为语法树（结构与 vocab.py 的 parse 一一对应）。 */
export function parse(expr: string): Tree {
  // 不设缓存表：一条公式在一次 `run` 中只解析一次（逐像素部分使用已解析的树），
  // 公式本身很短，不值得维护一张无上限的模块级 Map
  const ts = tokens(expr);
  let pos = 0;
  const peek = (): Tok | undefined => ts[pos];
  const isSym = (s: string) => {
    const t = peek();
    return t !== undefined && t.k === "sym" && t.t === s;
  };
  const eat = (s: string) => {
    if (!isSym(s)) throw new BadOp(`${expr}: expected ${s}`);
    pos += 1;
  };

  const additive = (): Tree => {
    let node = multiplicative();
    while (isSym("+") || isSym("-")) {
      const o = ts[pos].t;
      pos += 1;
      node = { k: "bin", o, l: node, r: multiplicative() };
    }
    return node;
  };
  const multiplicative = (): Tree => {
    let node = unary();
    while (isSym("*") || isSym("/")) {
      const o = ts[pos].t;
      pos += 1;
      node = { k: "bin", o, l: node, r: unary() };
    }
    return node;
  };
  const unary = (): Tree => {
    if (isSym("-")) {
      pos += 1;
      return { k: "neg", x: unary() };
    }
    if (isSym("+")) {
      pos += 1;
      return unary();
    }
    return primary();
  };
  const primary = (): Tree => {
    const tok = peek();
    if (!tok) throw new BadOp(`${expr}: ends too early`);
    if (tok.k === "num") {
      pos += 1;
      return { k: "num", v: Number(tok.t) };
    }
    if (tok.k === "name") {
      pos += 1;
      if (isSym("(")) {
        const want = FUNCTIONS[tok.t];
        if (want === undefined) throw new BadOp(`${expr}: no such function ${tok.t}`);
        eat("(");
        const a = [additive()];
        while (isSym(",")) {
          pos += 1;
          a.push(additive());
        }
        eat(")");
        if (a.length !== want) throw new BadOp(`${expr}: ${tok.t} takes ${want} arguments, got ${a.length}`);
        return { k: "call", f: tok.t, a };
      }
      return { k: "var", n: tok.t };
    }
    if (tok.t === "(") {
      pos += 1;
      const node = additive();
      eat(")");
      return node;
    }
    throw new BadOp(`${expr}: ${tok.t} is out of place`);
  };

  const tree = additive();
  if (pos !== ts.length) throw new BadOp(`${expr}: trailing tokens`);
  return tree;
}

/** 语法树中用到的变量名。 */
export function variables(tree: Tree, into: Set<string> = new Set()): Set<string> {
  if (tree.k === "var") into.add(tree.n);
  else if (tree.k === "bin") {
    variables(tree.l, into);
    variables(tree.r, into);
  } else if (tree.k === "neg") variables(tree.x, into);
  else if (tree.k === "call") tree.a.forEach((a) => variables(a, into));
  return into;
}

function ev(tree: Tree, vars: Record<string, number>): number {
  switch (tree.k) {
    case "num":
      return tree.v;
    case "var": {
      const v = vars[tree.n];
      if (v === undefined) throw new BadOp(`the formula wants ${tree.n}, the call gave none`);
      return v;
    }
    case "neg":
      return -ev(tree.x, vars);
    case "bin": {
      const a = ev(tree.l, vars);
      const b = ev(tree.r, vars);
      return tree.o === "+" ? a + b : tree.o === "-" ? a - b : tree.o === "*" ? a * b : a / b;
    }
    default: {
      const a = tree.a.map((x) => ev(x, vars));
      if (tree.f === "min") return Math.min(a[0], a[1]);
      if (tree.f === "max") return Math.max(a[0], a[1]);
      return Math.min(Math.max(a[0], a[1]), a[2]); // clamp(x, lo, hi)
    }
  }
}

// ---------------------------------------------------------------- 每像素算术

function isPix(v: unknown): v is Pix {
  return typeof v === "object" && v !== null && "data" in (v as Pix);
}

function runPixel(desc: Desc, args: Record<string, Pix | number>): { value: Pix } {
  const tree = parse(desc.expr!);
  const names = [...variables(tree)];
  const pixes: Pix[] = [];
  for (const n of names) {
    const v = args[n];
    if (v === undefined) throw new BadOp(`the op wants ${n}, the call gave none`);
    if (isPix(v)) pixes.push(v);
  }
  // 形状与通道数取公式的第一个变量（词汇表约定：变量名即输入名；view/evaluate.ts pixelwise 也据此确定形状）。
  // 与服务器端一致（nodes/core/image.py cook：输出按 A 的通道数分配，B 只取第一条通道 `[..., :1]`，
  // 再由 numpy 广播到 A 的每条通道）。不能取通道最多的一份：A 为 RGBA、B 为 RGB 时按相同下标读取 B 的第 3 条通道
  // 会越界得到 undefined，公式随即报错「wants b」
  const base = pixes[0];
  if (!base) throw new BadOp("a pixel op wants at least one image");
  const w = base.w;
  const h = base.h;
  const c = Math.max(1, base.c);
  // 宽高不同的一份按比例取最近像素：页面显示的是缩小的代理（transfer 的档位），而按框涂出的遮罩
  // 为原画面尺寸（框坐标以原画面为准）。两者各自正确，合并计算时须将遮罩映射到代理的像素网格上。
  // 直接按下标取值会使较小的一份越界得到 undefined，公式随即报错「wants a」。
  const at = (v: Pix, px: number, py: number): number => {
    if (v.w === w && v.h === h) return py * w + px;
    return Math.min(v.h - 1, Math.floor((py * v.h) / h)) * v.w + Math.min(v.w - 1, Math.floor((px * v.w) / w));
  };
  const out = new Float32Array(w * h * c);
  const vars: Record<string, number> = {};
  for (const n of names) if (!isPix(args[n])) vars[n] = args[n] as number;
  for (let py = 0; py < h; py += 1) {
    for (let px = 0; px < w; px += 1) {
      const p = py * w + px;
      for (let ch = 0; ch < c; ch += 1) {
        for (const n of names) {
          const v = args[n];
          // 通道数与输出不同的一份只取第一条通道（对应服务器的 `[..., :1]`），相同的则逐通道对应
          if (isPix(v)) vars[n] = v.data[at(v, px, py) * v.c + (v.c === c ? ch : 0)];
        }
        out[p * c + ch] = ev(tree, vars);
      }
    }
  }
  return { value: { w, h, c, data: out } };
}

// ---------------------------------------------------------------- 按框涂

function runBoxesPaint(desc: Desc, args: { boxes: number[][]; width: number; height: number }): { mask: Pix } {
  const axis = parse(desc.axis!);
  const combine = parse(desc.combine!);
  const w = Math.trunc(args.width);
  const h = Math.trunc(args.height);
  const out = new Float32Array(w * h).fill(desc.base ?? 0);
  for (const box of args.boxes) {
    const [x1, y1, x2, y2] = box.map(Number);
    const cx = new Float64Array(w);
    const cy = new Float64Array(h);
    for (let p = 0; p < w; p += 1) cx[p] = ev(axis, { p, lo: x1, hi: x2 });
    for (let p = 0; p < h; p += 1) cy[p] = ev(axis, { p, lo: y1, hi: y2 });
    for (let row = 0; row < h; row += 1) {
      for (let col = 0; col < w; col += 1) {
        const one = ev(combine, { x: cx[col], y: cy[row] });
        const i = row * w + col;
        if (one > out[i]) out[i] = one; // desc.accumulate === "max"（词汇表目前仅此一种）
      }
    }
  }
  return { mask: { w, h, c: 1, data: out } };
}

// ---------------------------------------------------------------- 挑条目

const RULE_ARGS: Record<string, string[]> = {
  index: ["index"],
  name: ["name"],
  first: [],
  top: ["count"],
  all: [],
  ids: ["ids"],
  at_point: ["picks"],
};

export type Missed =
  | { rule: "index"; index: number; count: number }
  | { rule: "name"; name: string }
  | { rule: "at_point"; frame: number; x: number; y: number };

/** 当前帧的框；该人物在当前帧不存在时，取数值上最近的一帧（按帧号升序比较，距离相同时取较小者）。 */
function boxAt(boxes: Record<string, number[]>, frame: number): number[] | null {
  const keys = Object.keys(boxes);
  if (!keys.length) return null;
  let key = String(frame);
  if (!(key in boxes)) {
    let best: number | null = null;
    for (const k of keys.slice().sort((a, b) => Number(a) - Number(b))) {
      const d = Math.abs(Number(k) - frame);
      if (best === null || d < best) {
        best = d;
        key = k;
      }
    }
  }
  return boxes[key];
}

type PickArgs = { items: Item[]; rule: string; index?: number; name?: string; count?: number; ids?: number[]; picks?: number[][] };

function runItemsPick(desc: Desc, args: PickArgs): { indices: number[]; missed: Missed[] } {
  const rule = args.rule;
  if (!desc.rules!.includes(rule)) throw new BadOp(`this op takes ${desc.rules!.join()}, got ${rule}`);
  for (const need of RULE_ARGS[rule]) {
    if ((args as Record<string, unknown>)[need] === undefined) throw new BadOp(`rule ${rule} wants ${need}, the call gave none`);
  }
  const items = args.items;
  let picked: number[] = [];
  const missed: Missed[] = [];

  if (rule === "index") {
    const n = Math.trunc(args.index!);
    if (n >= 1 && n <= items.length) picked = [n - 1];
    else missed.push({ rule: "index", index: n, count: items.length });
  } else if (rule === "name") {
    const want = String(args.name).trim();
    const found = items.findIndex((it) => it.name === want);
    if (found < 0) missed.push({ rule: "name", name: want });
    else picked = [found];
  } else if (rule === "first") {
    picked = items.length ? [0] : [];
  } else if (rule === "top") {
    picked = items.slice(0, Math.max(Math.trunc(args.count!), 0)).map((_, i) => i);
  } else if (rule === "all") {
    picked = items.map((_, i) => i);
  } else if (rule === "ids") {
    const wanted = new Set(args.ids!.map((v) => Math.trunc(v)));
    picked = items.map((_, i) => i).filter((i) => items[i].id !== undefined && wanted.has(items[i].id!));
  } else {
    // at_point：在 2D 视图中点选的位置。取面积最小的框（人物重叠时点中的是前面的人）
    const keep = new Set<number>();
    for (const pick of args.picks!) {
      const frame = Math.trunc(pick[0]);
      const x = pick[1];
      const y = pick[2];
      const hits: [number, number, number][] = [];
      items.forEach((it, i) => {
        const box = boxAt(it.boxes ?? {}, frame);
        if (!box) return;
        const [x1, y1, x2, y2] = box;
        if (x1 <= x && x <= x2 && y1 <= y && y <= y2) hits.push([(x2 - x1) * (y2 - y1), it.id ?? i, i]);
      });
      if (hits.length) {
        hits.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
        keep.add(hits[0][2]);
      } else missed.push({ rule: "at_point", frame, x, y });
    }
    picked = [...keep].sort((a, b) => a - b); // 输出顺序始终为条目表的顺序，而非点选顺序
  }

  if (desc.count === "one") picked = picked.slice(0, 1);
  return { indices: picked, missed };
}

// ---------------------------------------------------------------- 入口

/** 该算法所属的运算种类（词汇表 `lab2shot/ops/vocab.py` 的 `KINDS`：每像素算术 / 按框涂 / 挑条目）。
 * 接线层按种类执行，而非按节点（`ops/recipe.ts`），因此新增节点时无需修改。
 * 目录中没有该算法时返回 `""`，表示浏览器无法执行，沿用服务器结果。 */
export const kindOf = (id: string): string => CATALOG[id]?.kind ?? "";

/** 该每像素算术公式所需的变量（`a`、`b`）。
 *
 * 这是端口与算法参数对应关系的声明：变量名即端口名
 * （词汇表 `lab2shot/ops/vocab.py` 的 `pixel`：「变量是输入的名字（a、b）」），
 * 因此接线层无需为每个节点单独编写对应关系。`tests/test_browser_wiring.py` 检查两端一致。 */
export const varsOf = (id: string): string[] => {
  const expr = CATALOG[id]?.expr;
  return expr ? [...variables(parse(expr))] : [];
};

/** 执行一条算法。`id` 为 ops.toml 中的 id，`args` 为该运算所需的输入。 */
export function run(id: string, args: Record<string, unknown>): Record<string, unknown> {
  const desc = CATALOG[id];
  if (!desc) throw new BadOp(`no such op ${id} (lab2shot/ops/ops.toml)`);
  if (desc.kind === "pixel") return runPixel(desc, args as Record<string, Pix | number>);
  if (desc.kind === "boxes.paint") return runBoxesPaint(desc, args as { boxes: number[][]; width: number; height: number });
  return runItemsPick(desc, args as unknown as PickArgs);
}
