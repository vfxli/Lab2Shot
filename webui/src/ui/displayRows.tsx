import { useEffect, useState } from "react";
import { CHOICES, CHOICE_TIPS, type ViewOptions } from "../model/viewOptions";
import type { ViewControl } from "../model/viewControls";
import { type SegmentOption } from "./Button";

/** 显示选项面板的行与小组件（ui/DisplayOptions.tsx 及其各页共用）：行布局、数字输入、滑块、色标。
 * 只使用组件库的组件和 ui/displayoptions.css 中的类，不新增样式。
 *
 * 控件不得时隐时现，只区分可用与不可用：每一行始终位于原位置，不适用的整行置灰，原因（`off`）写在行名的
 * 悬停提示中，中文模板位于消息目录（model/viewControls.ts WHY_OFF → view/available.ts）。 */

export type O = ViewOptions;
export type Patch = (patch: Partial<O>) => void;
/** 该控件当前不可用的原因（`""` 表示可用）：面板各页只调用此函数。 */
export type Off = (control: ViewControl) => string;
/** 视图当前绘制的内容（三维的种类；二维：人物框 boxes、跟踪点 tracks2d、节点手柄 handle）。 */
export type Has = (...what: string[]) => boolean;

/** 一项设置的各选项及其作用（model/viewOptions.ts CHOICES、CHOICE_TIPS）；`off` 非空时整组置灰。 */
export const choices = <K extends keyof typeof CHOICES>(key: K, off = ""): SegmentOption<keyof (typeof CHOICES)[K] & string>[] =>
  (Object.keys(CHOICES[key]) as (keyof (typeof CHOICES)[K] & string)[]).map((value) => ({
    value,
    label: CHOICES[key][value] as string,
    tip: CHOICE_TIPS[key][value],
    disabled: off || false,
  }));

/** 一行：名称 + 控件。`off` 为当前不可用的原因：整行置灰、位置不变，原因显示在该行的悬停提示中。 */
export function Row({ label, tip, off = "", children }: { label: string; tip: string; off?: string; children: React.ReactNode }) {
  return (
    <div className={off ? "vo-row off" : "vo-row"} aria-disabled={off ? true : undefined}>
      <span className="vo-label" data-tip={off || tip}>
        {label}
      </span>
      <div className="vo-ctl">{children}</div>
    </div>
  );
}

/** A numeric input, clamped to its range when the field is left (or on Enter). */
export function Num({ value, onChange, min, max, unit, digits = 2, disabled = false }:
  { value: number; onChange: (v: number) => void; min: number; max: number; unit?: string; digits?: number; disabled?: boolean }) {
  const show = (v: number) => String(Number(v.toFixed(digits)));
  const [text, setText] = useState(show(value));
  useEffect(() => setText(show(value)), [value]); // eslint-disable-line react-hooks/exhaustive-deps
  const commit = () => {
    const n = Number(text);
    if (text.trim() === "" || !Number.isFinite(n)) return setText(show(value));
    const v = Math.min(max, Math.max(min, n));
    setText(show(v));
    onChange(v);
  };
  const input = (
    <input
      className="field num"
      value={text}
      disabled={disabled}
      data-tip={`在 ${min}–${max}${unit ? ` ${unit}` : ""} 之间输入，回车或点别处确定`}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        e.stopPropagation(); // typing a number is not a viewer hotkey
      }}
    />
  );
  return unit ? (
    <span className="num-unit" data-unit={unit}>
      {input}
    </span>
  ) : (
    input
  );
}

/** A slider with its number; `log`: steps are uniform in ratio (sizes from 0.01 to 1000). */
export function Slider({ value, onChange, min, max, unit, log = false, digits = 2, disabled = false }:
  { value: number; onChange: (v: number) => void; min: number; max: number; unit?: string; log?: boolean; digits?: number; disabled?: boolean }) {
  const to = (v: number) => (log ? Math.log(v / min) / Math.log(max / min) : (v - min) / (max - min));
  const from = (t: number) => (log ? min * Math.pow(max / min, t) : min + t * (max - min));
  const t = Math.min(1, Math.max(0, to(value)));
  return (
    <div className="vo-slider">
      <input
        type="range"
        data-tip={`拖动调整；右边可以输入（${min}–${max}${unit ? ` ${unit}` : ""}）`}
        min={0}
        max={1000}
        value={Math.round(t * 1000)}
        disabled={disabled}
        style={{ ["--pct" as string]: `${t * 100}%` }}
        onChange={(e) => onChange(Number(from(Number(e.target.value) / 1000).toPrecision(3)))}
      />
      <Num value={value} onChange={onChange} min={min} max={max} unit={unit} digits={digits} disabled={disabled} />
    </div>
  );
}

