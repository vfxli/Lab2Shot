/** 条件表达式：模板参数界面里一项参数什么时候藏起、什么时候置灰（节点图 exposed 参数项的 `hide_when` / `disable_when`，
 * Houdini 的 Hide When / Disable When；旧文件的 `when` 读入时换成 disable_when，state/cookInputs.ts readExposed）。
 *
 * 网页这一份与服务端的 lab2shot/engine/conditions.py 规则逐条相同
 * 两边都跑：`lab2shot check conditions`（它用 node 直接跑本文件，所以本文件不 import 任何东西）。改规则要两边一起改、
 * 再补用例。不做任何函数调用，不用 eval。
 *
 * 规则（与 conditions.py 顶部说明一致）：
 * - 操作数：公开参数的对外名字（参数项的 `name`，取它的当前值），或字面量：数字（`3`、`-1.5`）、`true` / `false`、带引号
 *   的字符串（`"abc"` 或 `'abc'`，反斜杠转义下一个字符）。名字由字母（含中文）、数字、下划线组成，不以数字开头；
 *   `and or not in true false` 是保留字。
 * - 运算：`==`、`!=`；`in [a, b]`、`not in [a, b]`；`<`、`<=`、`>`、`>=`（两边都是数字才可能为真）；`and`、`or`、`not`；
 *   括号。优先级从低到高：`or` < `and` < `not` < 比较。比较不连写（`a < b < c` 是写法错误）。
 * - 相等：布尔只等于布尔，数字按数值比，字符串逐字比，空值只等于空值，列表逐项比，其他都不相等。`true == 1` 为假。
 * - 单独一个操作数按「有没有」算：布尔取本身，数字非零，字符串非空，列表非空，空值为假。
 * - 用法（Houdini 的 Hide When / Disable When）：成立 = 藏起 / 置灰。判「成立」一律用 conditionHolds：空表达式（没有条件）、
 *   写法错误、用到不存在的参数名都不成立（照常显示、可改，编辑器标红：conditionProblem）。`judge` 是两边共用用例的口径，
 *   对空表达式答 true，不能直接拿来判 Hide / Disable。 */

import { pick, t } from "../i18n/t.ts";

export class ConditionError extends Error {
  why: string;
  at: number; // 出错的位置（第几个字，从 0 数）
  constructor(why: string, at: number) {
    super(why);
    this.why = why;
    this.at = at;
  }
}

type Tok = { kind: "num" | "str" | "name" | "kw" | "op" | "end"; value: string | number | null; at: number };

export type CondNode =
  | { k: "lit"; v: unknown }
  | { k: "name"; n: string }
  | { k: "not"; x: CondNode }
  | { k: "and" | "or"; xs: CondNode[] }
  | { k: "cmp"; op: string; a: CondNode; b: CondNode }
  | { k: "in"; a: CondNode; xs: CondNode[]; neg: boolean };

const KEYWORDS = new Set(["and", "or", "not", "in", "true", "false"]);
// 数字最多写多长，与服务端相同（conditions.py MOST_DIGITS）
const MOST_DIGITS = 30;
const COMPARE = ["==", "!=", "<=", ">=", "<", ">"];
const digit = (c: string | undefined) => c !== undefined && c >= "0" && c <= "9";
// what a name is made of, the same ranges as the server's (conditions.py NAME_LETTERS): ASCII letters and _, the CJK
// unified ideographs (with extension A), ASCII digits after the first; never \p{L}, whose Unicode version is the runtime's
const letter = (c: string) => /^[A-Za-z_\u3400-\u4DBF\u4E00-\u9FFF]$/.test(c);
const wordChar = (c: string) => letter(c) || (c >= "0" && c <= "9");
// 分隔记号的空白字符，与服务端逐个列出的相同（conditions.py SPACE）：\s 与 Python 的 str.isspace 范围不同
const SPACE = new Set([..."\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"]);
const space = (c: string) => SPACE.has(c);

/** No condition written: nothing but SPACE (conditions.py blank; never trim(), whose white space is not the server's). */
const blank = (text: unknown) => [...String(text ?? "")].every(space);

function tokens(text: string): Tok[] {
  const out: Tok[] = [];
  const chars = [...text]; // 按字符，与 Python 给 str 编号的方式相同
  const n = chars.length;
  let i = 0;
  while (i < n) {
    const c = chars[i];
    if (space(c)) {
      i++;
      continue;
    }
    const start = i;
    if (digit(c) || (c === "-" && digit(chars[i + 1]))) {
      i++;
      while (i < n && digit(chars[i])) i++;
      if (chars[i] === ".") {
        if (!digit(chars[i + 1])) throw new ConditionError(t("ui.conditions.decimal_digits"), i);
        i++;
        while (i < n && digit(chars[i])) i++;
      }
      if (i - start > MOST_DIGITS) throw new ConditionError(t("ui.conditions.number_too_long", { most: MOST_DIGITS }), start);
      out.push({ kind: "num", value: Number(chars.slice(start, i).join("")), at: start });
      continue;
    }
    if (c === '"' || c === "'") {
      i++;
      let s = "";
      while (i < n && chars[i] !== c) {
        if (chars[i] === "\\" && i + 1 < n) i++;
        s += chars[i];
        i++;
      }
      if (i >= n) throw new ConditionError(t("ui.conditions.unclosed_quote"), start);
      i++;
      out.push({ kind: "str", value: s, at: start });
      continue;
    }
    if (letter(c)) {
      while (i < n && wordChar(chars[i])) i++;
      const word = chars.slice(start, i).join("");
      out.push({ kind: KEYWORDS.has(word) ? "kw" : "name", value: word, at: start });
      continue;
    }
    const two = c + (chars[i + 1] ?? "");
    if (["==", "!=", "<=", ">="].includes(two)) {
      out.push({ kind: "op", value: two, at: start });
      i += 2;
      continue;
    }
    if ("<>()[],".includes(c)) {
      out.push({ kind: "op", value: c, at: start });
      i++;
      continue;
    }
    if (c === "=") throw new ConditionError(t("ui.conditions.double_equals"), start);
    throw new ConditionError(t("ui.conditions.unknown_char", { char: c }), start);
  }
  out.push({ kind: "end", value: null, at: n });
  return out;
}

const said = (tok: Tok) => (tok.kind === "str" ? `"${tok.value}"` : tok.kind === "end" ? t("ui.conditions.end") : String(tok.value));

// 括号与 not 最多嵌套几层（服务端 conditions.MOST_NESTED）：超过即为写法错误
const MOST_NESTED = 32;

class Parser {
  private i = 0;
  private depth = 0;
  private toks: Tok[];
  constructor(text: string) {
    this.toks = tokens(text);
  }
  private peek(k = 0): Tok {
    return this.toks[Math.min(this.i + k, this.toks.length - 1)];
  }
  private take(): Tok {
    return this.toks[this.i++];
  }
  private is(kind: Tok["kind"], value?: string, k = 0): boolean {
    const t = this.peek(k);
    return t.kind === kind && (value === undefined || t.value === value);
  }
  private deeper(at: number): void {
    if (++this.depth > MOST_NESTED) throw new ConditionError(t("ui.conditions.too_deep", { most: MOST_NESTED }), at);
  }
  private expect(kind: Tok["kind"], value: string, why: string): void {
    if (!this.is(kind, value)) throw new ConditionError(why, this.peek().at);
    this.take();
  }
  whole(): CondNode {
    const node = this.or();
    if (!this.is("end")) throw new ConditionError(t("ui.conditions.misplaced", { token: said(this.peek()) }), this.peek().at);
    return node;
  }
  private or(): CondNode {
    const xs = [this.and()];
    while (this.is("kw", "or")) {
      this.take();
      xs.push(this.and());
    }
    return xs.length === 1 ? xs[0] : { k: "or", xs };
  }
  private and(): CondNode {
    const xs = [this.not()];
    while (this.is("kw", "and")) {
      this.take();
      xs.push(this.not());
    }
    return xs.length === 1 ? xs[0] : { k: "and", xs };
  }
  private not(): CondNode {
    if (this.is("kw", "not")) {
      this.deeper(this.take().at);
      const node: CondNode = { k: "not", x: this.not() };
      this.depth--;
      return node;
    }
    return this.compare();
  }
  private compare(): CondNode {
    const a = this.operand();
    if (this.is("op") && COMPARE.includes(String(this.peek().value))) {
      const op = String(this.take().value);
      return { k: "cmp", op, a, b: this.operand() };
    }
    if (this.is("kw", "in")) {
      this.take();
      return { k: "in", a, xs: this.items(), neg: false };
    }
    if (this.is("kw", "not") && this.is("kw", "in", 1)) {
      this.take();
      this.take();
      return { k: "in", a, xs: this.items(), neg: true };
    }
    return a;
  }
  private items(): CondNode[] {
    this.expect("op", "[", t("ui.conditions.in_list"));
    const out: CondNode[] = [];
    if (this.is("op", "]")) {
      this.take();
      return out;
    }
    for (;;) {
      out.push(this.operand());
      if (this.is("op", ",")) {
        this.take();
        continue;
      }
      this.expect("op", "]", t("ui.conditions.list_unclosed"));
      return out;
    }
  }
  private operand(): CondNode {
    const tok = this.peek();
    if (tok.kind === "num" || tok.kind === "str") {
      this.take();
      return { k: "lit", v: tok.value };
    }
    if (tok.kind === "kw" && (tok.value === "true" || tok.value === "false")) {
      this.take();
      return { k: "lit", v: tok.value === "true" };
    }
    if (tok.kind === "name") {
      this.take();
      return { k: "name", n: String(tok.value) };
    }
    if (tok.kind === "op" && tok.value === "(") {
      this.deeper(this.take().at);
      const node = this.or();
      this.expect("op", ")", t("ui.conditions.paren_unclosed"));
      this.depth--;
      return node;
    }
    if (tok.kind === "end") throw new ConditionError(t("ui.conditions.unfinished"), tok.at);
    throw new ConditionError(t("ui.conditions.misplaced", { token: said(tok) }), tok.at);
  }
}

/** 解析；空表达式（或只有空白）为 null（没有条件）。写法错误抛 ConditionError。 */
export function parseCondition(text: string | null | undefined): CondNode | null {
  if (blank(text)) return null;
  return new Parser(String(text)).whole();
}

/** The expression with every use of the name `from` written as `to` (two parameter-interface entries merged into one:
 * the one that goes is called by the one that stays). Only names are touched, never a quoted text or a keyword; an
 * expression that does not read comes back as it is (the editor marks it). */
export function renamedInCondition(text: string, from: string, to: string): string {
  let toks: Tok[];
  try {
    toks = tokens(text);
  } catch {
    return text;
  }
  const chars = [...text];
  const width = [...from].length;
  for (const tk of [...toks].reverse()) if (tk.kind === "name" && tk.value === from) chars.splice(tk.at, width, to);
  return chars.join("");
}

/** 表达式用到的参数名（按出现先后，不重复）。 */
export function conditionNames(node: CondNode | null): string[] {
  const out: string[] = [];
  const walk = (x: CondNode): void => {
    if (x.k === "name") {
      if (!out.includes(x.n)) out.push(x.n);
    } else if (x.k === "not") walk(x.x);
    else if (x.k === "and" || x.k === "or") x.xs.forEach(walk);
    else if (x.k === "cmp") (walk(x.a), walk(x.b));
    else if (x.k === "in") (walk(x.a), x.xs.forEach(walk));
  };
  if (node) walk(node);
  return out;
}

/** 表达式里拿参数名和字面量比相等的地方（`==`、`!=`、`in [...]`、`not in [...]`，名字在哪一边都算）：[参数名, 字面量]，按
 * 出现先后。与 conditions.py compared 逐条相同。
 * 参数界面编辑器用它查字面量是不是这个参数可能取到的值（editor/ParamInterfaceEditor.tsx，规则与服务端保存模板时的
 * engine/templates.py check_exposed 一致：`cam_src != 7` 永远成立，多半是写错了）。 */
export function compared(node: CondNode | null): [string, unknown][] {
  const out: [string, unknown][] = [];
  const walk = (x: CondNode): void => {
    if (x.k === "cmp") {
      if (x.op === "==" || x.op === "!=") {
        if (x.a.k === "name" && x.b.k === "lit") out.push([x.a.n, x.b.v]);
        else if (x.a.k === "lit" && x.b.k === "name") out.push([x.b.n, x.a.v]);
      }
      walk(x.a);
      walk(x.b);
    } else if (x.k === "in") {
      if (x.a.k === "name") for (const y of x.xs) if (y.k === "lit") out.push([x.a.n, y.v]);
      walk(x.a);
      x.xs.forEach(walk);
    } else if (x.k === "not") walk(x.x);
    else if (x.k === "and" || x.k === "or") x.xs.forEach(walk);
  };
  if (node) walk(node);
  return out;
}

/** 用例口径（check 两边比）：一条表达式的 compared；写法错误为 "parse"。 */
export function comparedIn(text: string): [string, unknown][] | "parse" {
  try {
    return compared(parseCondition(text));
  } catch (e) {
    if (e instanceof ConditionError) return "parse";
    throw e;
  }
}

const isNumber = (v: unknown): v is number => typeof v === "number";

/** 条件里的相等（规则见文件顶部：只比标量与标量的列表，对象、NaN 一律不等）。与文档值的深比较 model/graphPatch.ts same
 * 不是一回事：只用在条件、以及按条件的口径找菜单项上。 */
export function condEqual(a: unknown, b: unknown): boolean {
  if (typeof a === "boolean" || typeof b === "boolean") return a === b;
  if (isNumber(a) && isNumber(b)) return a === b;
  if (typeof a === "string" && typeof b === "string") return a === b;
  if (a === null && b === null) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => condEqual(x, b[i]));
  return false;
}

export function truthy(v: unknown): boolean {
  if (typeof v === "boolean") return v;
  if (isNumber(v)) return v !== 0;
  if (typeof v === "string" || Array.isArray(v)) return v.length > 0;
  if (v !== null && typeof v === "object") return Object.keys(v).length > 0;
  return false;
}

function value(node: CondNode, values: Record<string, unknown>): unknown {
  if (node.k === "lit") return node.v;
  if (node.k === "name") return values[node.n] ?? null;
  return truth(node, values);
}

function truth(node: CondNode, values: Record<string, unknown>): boolean {
  switch (node.k) {
    case "or":
      return node.xs.some((x) => truth(x, values));
    case "and":
      return node.xs.every((x) => truth(x, values));
    case "not":
      return !truth(node.x, values);
    case "cmp": {
      const a = value(node.a, values);
      const b = value(node.b, values);
      if (node.op === "==") return condEqual(a, b);
      if (node.op === "!=") return !condEqual(a, b);
      if (!isNumber(a) || !isNumber(b)) return false;
      return node.op === "<" ? a < b : node.op === "<=" ? a <= b : node.op === ">" ? a > b : a >= b;
    }
    case "in": {
      const a = value(node.a, values);
      const found = node.xs.some((x) => condEqual(a, value(x, values)));
      return node.neg ? !found : found;
    }
    default:
      return truthy(value(node, values));
  }
}

/** 一条表达式在这些参数值下的结果：true / false；写法错误为 "parse"，用到了 `values` 里没有的名字为 "unknown"。
 * 两边比的就是这个。 */
export function judge(text: string, values: Record<string, unknown>): boolean | "parse" | "unknown" {
  let node: CondNode | null;
  try {
    node = parseCondition(text);
  } catch (e) {
    if (e instanceof ConditionError) return "parse";
    throw e;
  }
  if (node === null) return true;
  if (conditionNames(node).some((n) => !Object.prototype.hasOwnProperty.call(values, n))) return "unknown";
  return truth(node, values);
}

/** Hide When / Disable When 成立吗：只有表达式读得通、用到的名字都在、结果为真才算成立。空着（没有条件）、写错、
 * 用到了不存在的名字都不成立——这一项照常显示、照常可改（错的在编辑器里标红），不能因为求值出错把参数藏掉或锁住。
 * 注意与 `judge` 不同：`judge` 对空表达式答 true（它是「没有条件」的共用用例口径），直接拿它判 Hide / Disable 会把没写
 * 条件的项全当成「成立」。 */
export function conditionHolds(text: string | null | undefined, values: Record<string, unknown>): boolean {
  if (blank(text)) return false;
  return judge(String(text), values) === true;
}

/** 公开下拉里现在列出的选项：自己那一项的 hide_when 成立的不列（conditions.py shown_options 同一规则，用例
 *）。参数的值恰好是被藏起的那一项时落到第一个列出的（settledMenus）；命令行 /
 * 插件直接设成被藏起的那一项时服务端报错。 */
export function shownOptions<T extends { hide_when?: string }>(options: T[], values: Record<string, unknown>): T[] {
  return options.filter((o) => !conditionHolds(o.hide_when, values));
}

/** 公开下拉的一项现在是不是置灰：作者给它写的 disable_when 成立时给作者写的为什么（disable_why），否则 null。
 * 服务端 engine/templates.py hidden_values 同一规则（命令行 / 插件设不进这一项）。 */
export function optionDisabled(o: { disable_when?: string; disable_why?: unknown }, values: Record<string, unknown>): string | null {
  return o.disable_when !== undefined && conditionHolds(o.disable_when, values) ? (pick(o.disable_why) || o.disable_when) : null;
}

/** 公开下拉的当前值被它自己那一项的 hide_when 藏起时落到哪儿（conditions.py settled 同一规则
 * settled 两边都跑）：落到按当时的值第一个列出的选项；全被藏起的不动。一个落了位会让别的条件变，所以按新值再看，直到没有
 * 一个停在被藏起的项上（不动点）；回到见过的一组值时（循环），在落过位的下拉里按界面顺序、选项顺序找第一组谁都不停在
 * 被藏起项上的值（有的话）。答 [名字, 原来的值, 落到的值]（落回原值的不列），按第一次
 * 落位的先后。 */
export function settledMenus(menus: { name: string; options: { value: unknown; hide_when?: string }[] }[],
                             values: Record<string, unknown>): [string, unknown, unknown][] {
  const now: Record<string, unknown> = { ...values };
  const was = new Map<string, unknown>();
  const seen = new Set<string>();
  for (;;) {
    const state = JSON.stringify(menus.map(({ name }) => now[name] ?? null));
    if (seen.has(state)) break;
    seen.add(state);
    for (const { name, options } of menus) {
      const option = options.find((o) => condEqual(o.value, now[name] ?? null));
      const shown = option && conditionHolds(option.hide_when, now) ? shownOptions(options, now) : [];
      if (!shown.length) continue;
      if (!was.has(name)) was.set(name, now[name] ?? null);
      now[name] = shown[0].value;
    }
  }
  if (menus.some((m) => stuck(m, now))) {  // a cycle: the first values of the menus that moved that leave none stuck
    let picks: Record<string, unknown>[] = [{}];
    for (const m of menus.filter((m) => was.has(m.name))) {
      picks = picks.flatMap((p) => m.options.map((o) => ({ ...p, [m.name]: o.value }))).slice(0, 4096);
    }
    const pick = picks.find((p) => !menus.some((m) => stuck(m, { ...now, ...p })));
    if (pick) Object.assign(now, pick);
  }
  return [...was].filter(([name, old]) => !condEqual(old, now[name])).map(([name, old]) => [name, old, now[name]]);
}

/** Its value is an option its own hide_when hides, while another is listed (conditions.py _stuck). */
function stuck(menu: { name: string; options: { value: unknown; hide_when?: string }[] }, values: Record<string, unknown>): boolean {
  const option = menu.options.find((o) => condEqual(o.value, values[menu.name] ?? null));
  return !!option && conditionHolds(option.hide_when, values) && shownOptions(menu.options, values).length > 0;
}

// ------------------------------------------------------------------ 参数界面的校验（与服务端 engine/templates.py 同一规则）
// 「编辑参数界面」即时标红读这里（editor/ParamInterfaceEditor.tsx），保存模板时服务端 check_exposed 权威再判一遍。三条规则
// 两边逐字相同，改一边要改另一边。

/** 规则要看的目标参数定义（api/catalog.ts ParamDef 的这几项；本文件不 import 任何东西）。 */
export interface SpecLike {
  type: string;
  nullable: boolean;
  options: unknown[] | null;
  minimum: number | null;
  maximum: number | null;
  widget: string | null;
}

/** 数字、布尔、文字写成服务端 Python 的样子（`str()` / `:g`）：说明文字两边一字不差。 */
const pyText = (v: unknown): string => (typeof v === "boolean" ? (v ? "True" : "False") : String(v));

/** What a value of the wrong kind is told, by the target parameter's type. */
const WANT_KEY: Record<string, string> = {
  boolean: "ui.conditions.want_boolean",
  integer: "ui.conditions.want_integer",
  number: "ui.conditions.want_number",
  string: "ui.conditions.want_string",
};

/** 目标参数收不收这个值（null：收）——engine/templates.py _refuses。 */
export function valueRefused(p: SpecLike, v: unknown): string | null {
  if (v === null || v === undefined) return p.nullable ? null : t("ui.conditions.not_null");
  const fits = p.type === "boolean" ? typeof v === "boolean" : p.type === "integer" ? isNumber(v) && Number.isInteger(v)
    : p.type === "number" ? isNumber(v) : p.type === "string" ? typeof v === "string" : false;
  if (!fits) return t(WANT_KEY[p.type] ?? "ui.conditions.want_menu_unfit");
  if (p.options?.length && !p.options.some((o) => condEqual(o, v))) return t("ui.conditions.one_of", { options: p.options.map(pyText).join(" / ") });
  if (isNumber(v) && p.minimum != null && v < p.minimum) return t("ui.conditions.below_min", { min: pyText(p.minimum) });
  if (isNumber(v) && p.maximum != null && v > p.maximum) return t("ui.conditions.above_max", { max: pyText(p.maximum) });
  return null;
}

/** 这个参数能不能显示成下拉 / 复选框（null：能）——engine/templates.py _widget_refused。 */
export function widgetRefused(widget: "menu" | "checkbox", p: SpecLike, options: { value: unknown }[] | null | undefined): string | null {
  if (p.widget === "button") return t("ui.conditions.button_no_value");
  if (p.type === "array" || ["file", "sequence", "table", "hierarchy"].includes(p.widget ?? "")) return t("ui.conditions.not_single");
  if (widget === "menu") return options?.length ? null : t("ui.conditions.menu_empty");
  if (p.type === "boolean") return null;
  const vs = (options ?? []).map((o) => o.value);
  const zeroOne = vs.length === 2 && vs.every(isNumber) && [...(vs as number[])].sort().join() === "0,1";
  return p.type === "integer" && zeroOne ? null : t("ui.conditions.checkbox_target");
}

/** 公开参数 `name` 永远不会是 `value` 的原因（null：可能是）：有下拉选项的按选项的值，否则按目标参数收不收——
 * engine/templates.py _never。条件里 `cam_src != 7` 这种永远成立的，多半是写错了。 */
export function neverValue(name: string, value: unknown, options: { value: unknown }[] | null | undefined, p: SpecLike): string | null {
  if (p.widget === "button") return null;
  const said = (v: unknown) => JSON.stringify(v);
  if (options?.length) {
    if (options.some((o) => condEqual(value, o.value))) return null;
    return t("ui.conditions.never_option", { name, options: options.map((o) => said(o.value)).join(" / "), value: said(value) });
  }
  const why = valueRefused(p, value);
  return why ? t("ui.conditions.never_value", { name, value: said(value), why }) : null;
}

/** 一项公开参数的 Disable When。更早的文件写 `when`（意思相反：为真时可以改）：只有没有 `disable_when` 这个键时才换成
 * `not (when)`；有这个键（哪怕是 null）就以它为准。与服务端 engine/conditions.py disable_when_of 同一规则、同一组用例
 * 。 */
export function disableWhenOf(entry: Record<string, unknown>): string | null {
  if ("disable_when" in entry) return typeof entry.disable_when === "string" ? entry.disable_when : null;
  const old = entry.when;
  return typeof old === "string" && !blank(old) ? `not (${old})` : null;
}

/** 条件的问题（中文），没有问题为 null。`known`：所有公开参数的对外名字。 */
export function conditionProblem(text: string | null | undefined, known: string[]): string | null {
  let node: CondNode | null;
  try {
    node = parseCondition(text);
  } catch (e) {
    if (e instanceof ConditionError) return t("ui.conditions.syntax", { at: e.at + 1, why: e.why });
    throw e;
  }
  const missing = conditionNames(node).filter((n) => !known.includes(n));
  return missing.length ? t("ui.conditions.unknown_names", { names: missing }) : null;
}
