import { useEffect, useState } from "react";
import type { ParamDef } from "../api";
import { Select } from "./Select";
import { Switch } from "./Button";
import { why as whyOf, type Availability } from "../api/applies";

/** 选项当前是否可选，不可选时给出原因：接入的数据不是该档所需的类型（由服务器按节点的
 * option_applies 计算，id 为 "<参数>=<选项>"，lab2shot/nodes/applies.py option_conditions）。
 * 统一在一处计算，节点上的下拉框、参数面板的下拉框、提交前的拦截三处读取同一结果，页面本身不对数据做任何判断。 */
export function optionOff(p: ParamDef, answer: Availability | null | undefined, o: unknown): string {
  return whyOf(answer, `${p.name}=${String(o)}`);
}

// The input fields for parameters, shared by the parameter panel and the rows on a node's body: a single commit method
// (Enter or leaving the field), keeping numbers within the parameter's range and step, and showing the unit.

/** A number; with `p`, clamped to the parameter's range and snapped to a multiple of its step (as the node accepts it). */
export function NumberField({ value, onChange, placeholder, integer, p, mini, tip }: {
  value: number | null; onChange: (v: number | null) => void; placeholder?: string; integer?: boolean; p?: ParamDef; mini?: boolean; tip?: string;
}) {
  const [text, setText] = useState(value == null ? "" : String(value));
  useEffect(() => setText(value == null ? "" : String(value)), [value]);
  const commit = () => {
    if (text.trim() === "") return onChange(null);
    let n = Number(text);
    if (!Number.isFinite(n)) return setText(value == null ? "" : String(value));
    if (p?.multiple_of) n = Math.round(n / p.multiple_of) * p.multiple_of;
    if (p?.minimum != null) n = Math.max(p.minimum, n);
    if (p?.maximum != null) n = Math.min(p.maximum, n);
    n = integer ? Math.round(n) : n;
    setText(String(n));
    onChange(n);
  };
  const input = (
    <input
      className={`field num${mini ? " mini" : ""}`}
      value={text}
      data-tip={tip}
      placeholder={placeholder}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
    />
  );
  return p?.unit ? (
    <span className={`num-unit${mini ? " mini" : ""}`} data-unit={p.unit}>
      {input}
    </span>
  ) : (
    input
  );
}

/** 判断该文本参数是否为多行：只依据声明（nodes/base.py `P(lines=…)`），不依据参数名称。
 * 统一在一处判断，参数面板的行、节点上的行、控件本身三处读取同一结果。 */
export const multiline = (p: { lines?: number }) => (p.lines ?? 1) > 1;

/** A free-text value. 声明 `lines` 为 2 及以上的参数（提示词一类：值为一整句英文描述）绘制为自动换行的多行文本框，
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

/** A choice as the user reads it: its label, and 非商用 when the node declares that choice non-commercial. */
export const optionText = (p: ParamDef, o: unknown, nc: unknown[]) => `${p.option_labels?.[String(o)] ?? String(o)}${nc.some((v) => String(v) === String(o)) ? " · 非商用" : ""}`;

const short = (n: number) => String(Math.round(n * 1000) / 1000);

/** A simple parameter's value as plain text (a node's row when zoomed out, the panel's 信息): 开 / 关, a choice's label,
 * a number with its unit, a vector's three numbers, a text; for an empty value, what the parameter defines empty as (自动). */
export function valueText(p: ParamDef, v: unknown): string {
  if (p.type === "boolean") return v ? "开" : "关";
  // 先判断空值，再判断选项：顺序颠倒时，留空的选项参数会被 `String(null)` 显示为字面的 `null`
  if (p.options && v !== null && v !== undefined && v !== "") return p.option_labels?.[String(v)] ?? String(v);
  if (v === null || v === undefined || v === "") {
    // the meaning of empty, as the field's placeholder states it (自动, 第一帧); a number there is the value used (Filmback 36 mm)
    if (!p.placeholder) return p.nullable ? "自动" : "空";
    return Number.isFinite(Number(p.placeholder)) ? `自动 · ${p.placeholder}${p.unit ? ` ${p.unit}` : ""}` : p.placeholder;
  }
  if (Array.isArray(v)) return v.map((x) => short(Number(x))).join(", ") + (p.unit ? ` ${p.unit}` : "");
  if (typeof v === "number") return `${short(v)}${p.unit ? ` ${p.unit}` : ""}`;
  return String(v);
}

/** A simple parameter's control on a node's body: a small switch, a pull-down of its choices, a number (three for a
 * vector) or a text field; the same value as in the panel, committed the same way. */
export function NodeControl({ p, value, set, nc, answer }: { p: ParamDef; value: unknown; set: (v: unknown) => void; nc: unknown[]; answer?: Availability | null }) {
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
                      const off = optionOff(p, answer, o); // 不可选的档位仍然保留，只是置灰，原因写在其悬停提示中
                      const label = optionText(p, o, nc);
                      return { value: String(o), label, tip: off ? `${label}：${off}` : label, off: !!off };
                    })]}
          onPick={(v) => set(v === "" ? null : p.options!.find((o) => String(o) === v) ?? v)} />
      );
    }
    case "vector": {
      const v = (value as number[]) ?? [0, 0, 0];
      return (
        <span className="vec3 mini">
          {vecLabels(p).map(([axis], i) => (
            <NumberField key={axis} mini value={v[i]} onChange={(n) => set(v.map((x, j) => (j === i ? n ?? 0 : x)))} />
          ))}
        </span>
      );
    }
    case "number":
      return <NumberField mini value={value as number | null} onChange={set} placeholder={p.placeholder || (p.nullable ? "自动" : "")} integer={p.type === "integer"} p={p} />;
    case "text":
      // 单行或多行由参数自身声明（`P(lines=…)`）：节点上与参数面板中使用同一个控件、同一份声明
      return <TextField mini lines={p.lines} value={(value as string) ?? ""} onChange={(v) => set(v === "" && p.nullable ? null : v)} placeholder={p.placeholder} />;
    default:
      return null;
  }
}
