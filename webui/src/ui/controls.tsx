import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { coerce, type NumSpec } from "../model/numbers";
import type { ParamDef } from "../api";
import { Select } from "./Select";
import { Switch } from "./Button";
import { why as whyOf, type Availability } from "../api/applies";
import { getCatalog } from "../state/catalog";
import { optionName } from "../graph/rules";

/** 选项当前是否可选，不可选时给出原因：接入的数据不是该档所需的类型（由服务器按节点的
 * option_applies 计算，id 为 "<参数>=<选项>"，lab2shot/nodes/applies.py option_conditions）。
 * 统一在一处计算，节点上的下拉框、参数面板的下拉框、提交前的拦截三处读取同一结果，页面本身不对数据做任何判断。 */
export function optionOff(p: ParamDef, answer: Availability | null | undefined, o: unknown, at = p.name): string {
  return whyOf(answer, `${at}=${String(o)}`);
}

// 参数的输入控件，参数面板与节点上的行共用：提交方式只有一种（回车或离开输入框），数字保持在参数的范围和步长内，
// 并显示单位。

interface NumBase extends NumSpec {
  value: number | null;
  digits?: number; // 只管显示几位；写回的是输入的原值，不按它取整
  unit?: string;
  placeholder?: string;
  tip?: string;
  mini?: boolean;
  disabled?: boolean;
  label?: string;
  className?: string;
  autoFocus?: boolean;
  onDone?: () => void; // 离开输入框之后（提交与否）：用完即收的输入框（缩放百分比）据此收起
}

/** 页面上唯一的数字输入框（参数面板、节点上的行、表格格子、参数界面的选项值、显示选项与它们的滑块旁、骨架姿势面板、
 * 视图的黑点 / 白点、缩放百分比、磁盘天数都用它）。
 * - 回车或离开输入框时才提交，而且只在文字被改过、规范化出来的数和原值不同时提交：点进去再点出来什么都不写；Esc 放弃；
 * - 取值按声明规范化（model/numbers.ts coerce：步长、整数、闭区间夹到对齐的端点、开区间的界上不收），显示位数
 *   `digits` 只管显示；
 * - 清空：`nullable` 的写 null，否则退回原值；不是数、开区间界上的也退回原值；
 * - 框里显示的永远是存储里的值：调用方把写入夹回或不收时，框跟着存储走，不显示没写进去的数。 */
export function Num(props: NumBase & ({ nullable: true; onChange: (v: number | null) => void } | { nullable?: false; onChange: (v: number) => void })) {
  const { value, digits, unit, placeholder, tip, mini, disabled, label, className, autoFocus, onDone } = props;
  const show = (v: number | null) => (v == null ? "" : digits === undefined ? String(v) : String(Number(v.toFixed(digits))));
  const [text, setText] = useState(show(value));
  const shown = useRef(show(value)); // 框里现在显示的是这个值：文字没被改过就不提交
  useLayoutEffect(() => {
    shown.current = show(value);
    setText(shown.current);
  }, [value, digits]); // eslint-disable-line react-hooks/exhaustive-deps
  const revert = () => setText(shown.current);
  const commit = () => {
    if (text === shown.current) return;
    if (text.trim() === "") {
      if (!props.nullable || value === null) return revert();
      revert(); // 存储变了由上面的 effect 显示
      props.onChange(null);
      return;
    }
    const n = coerce(Number(text), props);
    revert();
    if (n !== null && n !== value) props.onChange(n);
  };
  const input = (
    <input
      className={`field num${mini ? " mini" : ""}${className ? ` ${className}` : ""}`}
      value={text}
      disabled={disabled}
      aria-label={label}
      placeholder={placeholder}
      autoFocus={autoFocus}
      onFocus={autoFocus ? (e) => e.currentTarget.select() : undefined}
      data-tip={tip ?? (props.min != null && props.max != null ? `在 ${props.min}–${props.max}${unit ? ` ${unit}` : ""} 之间输入，回车或点别处确定` : undefined)}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => (commit(), onDone?.())}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        else if (e.key === "Escape") {
          setText(shown.current);
          const el = e.currentTarget;
          requestAnimationFrame(() => el.blur());
        }
        e.stopPropagation(); // 输入数字不是视图的快捷键
      }}
    />
  );
  return unit ? (
    <span className={`num-unit${mini ? " mini" : ""}`} data-unit={unit}>
      {input}
    </span>
  ) : (
    input
  );
}

/** 三个数的参数（widget vec3：方向、颜色）：三个 Num，分量名按参数声明（vecLabels），单位与范围按参数（paramNum）。
 * 可为空的（「放平」的重力方向：空 = 自动估计）空着时三格都空着、写着它的「空」（自动），不冒充成 0 0 0；空着时填了
 * 一格先留在这里，三格都有了才整组写入（写一格就把另两格写成 0 会悄悄换掉「自动」）；有值时旁边的「×」把它清回空。
 * 参数面板（带分量名）与节点上的行（mini）都用它。空着时填了一半的分量是这一个节点的这一个参数的：调用方按它给 key
 * （参数面板换节点时不换组件，不给 key 就会把别的节点的半截草稿带过去、写进去）。 */
export function VecField({ p, value, set, mini = false }: { p: ParamDef; value: unknown; set: (v: unknown) => void; mini?: boolean }) {
  const v = Array.isArray(value) && value.length === 3 ? (value as number[]) : null;
  const [draft, setDraft] = useState<(number | null)[] | null>(null); // 空着时已填的分量
  useEffect(() => { if (v) setDraft(null); }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  const shown: (number | null)[] = v ?? draft ?? [null, null, null];
  const empty = valueText(p, null);
  const put = (i: number, n: number | null) => {
    if (v && n !== null) return set(v.map((x, j) => (j === i ? n : x)));
    const next = shown.map((x, j) => (j === i ? n : x));
    if (next.every((x) => x !== null)) {
      setDraft(null);
      set(next);
    } else setDraft(next);
  };
  const spec = paramNum(p);
  return (
    <span className={`vec3${mini ? " mini" : ""}${p.nullable && v ? " clearable" : ""}`}>
      {vecLabels(p).map(([axis, color], i) => (
        <label key={axis}>
          {!mini && <span className="vec3-axis" style={{ color }}>{axis}</span>}
          {v
            ? <Num mini={mini} value={shown[i]} onChange={(n) => put(i, n)} {...spec} />
            : <Num mini={mini} nullable value={shown[i]} placeholder={empty} onChange={(n) => put(i, n)} {...spec} />}
        </label>
      ))}
      {p.nullable && v && (
        <button type="button" className="vec3-clear" data-tip={`清空：回到「${empty}」`} aria-label="清空" onClick={() => set(null)}>×</button>
      )}
    </span>
  );
}

/** 一个数值参数的声明交给 Num 与滑块：范围（含开闭）、步长、整数、单位（与节点接受的值一致）。 */
export const paramNum = (p: ParamDef) => ({
  min: p.minimum, max: p.maximum, openMin: p.open_minimum, openMax: p.open_maximum, multipleOf: p.multiple_of,
  integer: p.type === "integer", unit: p.unit || undefined,
});

/** 判断该文本参数是否为多行：只依据声明（nodes/base.py `P(lines=…)`），不依据参数名称。
 * 统一在一处判断，参数面板的行、节点上的行、控件本身三处读取同一结果。 */
export const multiline = (p: { lines?: number }) => (p.lines ?? 1) > 1;

/** 自由文本值。声明 `lines` 为 2 及以上的参数（提示词一类：值为一整句英文描述）绘制为自动换行的多行文本框，
 * 高度等于声明的行数；其余仍为单行输入框。两者是同一个控件而非两种：参数面板和节点上均使用此控件，
 * `mini` 只决定其绘制尺寸较小。
 * 换行由文本框自身处理（`.field.area` 中的 `white-space: pre-wrap`），因此多行时回车为换行而非提交；
 * 单行时回车照常提交（失焦）。两种情况均在失焦时提交值。 */
export function TextField({ value, onChange, placeholder, mini, tip, lines }: {
  value: string; onChange: (v: string) => void; placeholder?: string; mini?: boolean; tip?: string; lines?: number;
}) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  const commit = () => text !== value && onChange(text);
  if (multiline({ lines })) {
    return (
      <textarea
        className={`field area${mini ? " snug" : ""}`}
        rows={lines}
        value={text}
        data-tip={tip}
        placeholder={placeholder}
        spellCheck={false}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
      />
    );
  }
  return (
    <input
      className={`field${mini ? " mini" : ""}`}
      value={text}
      data-tip={tip}
      placeholder={placeholder}
      spellCheck={false}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
    />
  );
}

export const AXIS = [
  ["X", "#ff6961"],
  ["Y", "#63e6a0"],
  ["Z", "#6cb4ff"],
] as const;

/** widget "vec3" 三个分量的名称：默认为 X Y Z（坐标、角度），颜色为 R G B。
 * 控件相同，仅分量名称不同，由参数自身声明（`P(..., parts=("R","G","B"))`，
 * 核心 nodes/base.py），不另设控件。颜色参数上显示 X Y Z 这类坐标轴名称，美术人员难以理解。 */
export const vecLabels = (p: { parts?: string[] | null }): readonly (readonly [string, string])[] =>
  p.parts?.length === 3 ? p.parts.map((name, i) => [name, AXIS[i][1]] as const) : AXIS;

/** 一个选项在页面上的样子，所有下拉、表格只读格、缩小后的值文字都只经这里：
 * - `label`：选项的名称（`name`：作者在参数界面给的显示名，只替换名称这一部分），许可受限时加上许可词（「非商用」「仅限
 *   研究」，服务器的 tags 表；`licensed`：String(值) -> 许可等级，graph/rules.ts licensedValues），作者覆盖不掉；
 * - `off`：现在不能选的原因（optionOff，服务器按声明判的；`at`：控件在节点里的位置，表格格子是 `表[i].列`，
 *   editor/paramPath.ts），"" 能选；`tip`：名称，不能选时连同原因。 */
export function optionView(p: ParamDef, o: unknown, answer: Availability | null | undefined, licensed: Record<string, string>, name?: string, at = p.name): { label: string; tip: string; off: string } {
  const tag = licensed[String(o)];
  const word = tag ? getCatalog()?.tags[tag]?.label ?? tag : "";
  const named = name ?? optionName(p, o);
  // 名称里已经写着这个许可词（作者手写的「（非商用）」）就不再加一遍：许可词只出现一次，作者也拿不掉它
  const label = word && !named.includes(word) ? `${named} · ${word}` : named;
  const off = optionOff(p, answer, o, at);
  return { label, tip: off ? `${label}：${off}` : label, off };
}

const short = (n: number) => String(Math.round(n * 1000) / 1000);

/** 简单参数的值的纯文本形式（缩小后节点上的行、面板的「信息」）：开 / 关、选项的名称、带单位的数字、向量的三个数、
 * 文本；值为空时，写参数对「空」的定义（自动）。 */
export function valueText(p: ParamDef, v: unknown, licensed: Record<string, string> = {}): string {
  if (p.widget === "button") return p.label; // 按钮参数没有值：缩小后写它的名字
  if (p.type === "boolean") return v ? "开" : "关";
  // 先判断空值，再判断选项：顺序颠倒时，留空的选项参数会被 `String(null)` 显示为字面的 `null`
  if (p.options && v !== null && v !== undefined && v !== "") return optionView(p, v, null, licensed).label;
  if (v === null || v === undefined || v === "") {
    // 「空」的含义，以输入框的占位文字为准（自动、第一帧）；占位文字为数字时即实际使用的值（Filmback 36 mm）
    if (!p.placeholder) return p.nullable ? "自动" : "空";
    return Number.isFinite(Number(p.placeholder)) ? `自动 · ${p.placeholder}${p.unit ? ` ${p.unit}` : ""}` : p.placeholder;
  }
  if (Array.isArray(v)) return v.map((x) => short(Number(x))).join(", ") + (p.unit ? ` ${p.unit}` : "");
  if (typeof v === "number") return `${short(v)}${p.unit ? ` ${p.unit}` : ""}`;
  return String(v);
}

/** 节点上简单参数的控件：小开关、选项下拉框、数字（向量为三个）或文本框；值与参数面板相同，提交方式也相同。 */
export function NodeControl({ p, value, set, nc, answer }: { p: ParamDef; value: unknown; set: (v: unknown) => void; nc: Record<string, string>; answer?: Availability | null }) {
  switch (p.simple) {
    case "toggle":
      return <Switch mini on={!!value} label={p.label} onChange={set} />;
    case "options": {
      // 节点上的多选项一律使用下拉框，不按 widget="segmented" 绘制分段控件：分段控件会撑破节点框。
      //
      // 可留空的参数将「空」也作为一档（与参数面板一致）：它不在 `p.options` 中，若遗漏此档，
      // 留空时 `String(null)` 会被显示为字面的 `null`。
      // 空档的名称统一由 `valueText` 决定（自动 / 自动 · 36 mm / 参数自身的说法）
      const empty = p.nullable ? valueText(p, null) : "";
      const now = value === null || value === undefined ? "" : String(value);
      return (
        <Select className="mini" value={now} label={p.label}
          options={[...(empty ? [{ value: "", label: empty, tip: empty }] : []),
                    ...p.options!.map((o) => {
                      const view = optionView(p, o, answer, nc); // 不可选的档位仍然保留，只是置灰，原因写在其悬停提示中
                      return { value: String(o), label: view.label, tip: view.tip, off: !!view.off };
                    })]}
          onPick={(v) => set(v === "" ? null : p.options!.find((o) => String(o) === v) ?? v)} />
      );
    }
    case "vector":
      return <VecField p={p} value={value} set={set} mini />;
    case "number":
      return p.nullable
        ? <Num mini nullable value={value as number | null} onChange={set} placeholder={p.placeholder || "自动"} {...paramNum(p)} />
        : <Num mini value={value as number | null} onChange={set} placeholder={p.placeholder} {...paramNum(p)} />;
    case "text":
      // 单行或多行由参数自身声明（`P(lines=…)`）：节点上与参数面板中使用同一个控件、同一份声明
      return <TextField mini lines={p.lines} value={(value as string) ?? ""} onChange={(v) => set(v === "" && p.nullable ? null : v)} placeholder={p.placeholder} />;
    default:
      return null;
  }
}
