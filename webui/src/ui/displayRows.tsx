import { CHOICES, type ViewOptions } from "../model/viewOptions";
import { t } from "../i18n/t";
import type { ViewControl } from "../model/viewControls";
import { type SegmentOption } from "./Button";
import { Num } from "./controls";
import { coerce } from "../model/numbers";
import { tipOf } from "../platform/tips";
import { LabelRow } from "./LabelRow";

/** 显示选项面板的行与小组件（ui/DisplayOptions.tsx 及其各页共用）：行布局、滑块、色标；数字输入是页面唯一的那个（ui/controls.tsx Num）。
 * 只使用组件库的组件和 ui/displayoptions.css 中的类，不新增样式。
 *
 * 控件不得时隐时现，只区分可用与不可用：每一行始终位于原位置，不适用的整行置灰，原因（`off`）写在行名的
 * 悬停提示中，中文模板位于消息目录（model/viewControls.ts WHY_OFF → view/available.ts）。 */

export type O = ViewOptions;
export type Patch = (patch: Partial<O>) => void;
/** 该控件当前不可用的原因（`""` 表示可用）：面板各页只调用此函数。 */
export type Off = (control: ViewControl) => string;

/** 一项设置的各选项（model/viewOptions.ts CHOICES）；`off` 非空时整组置灰，原因是每一项的悬停提示。 */
export const choices = <K extends keyof typeof CHOICES>(key: K, off = ""): SegmentOption<keyof (typeof CHOICES)[K] & string>[] =>
  (Object.keys(CHOICES[key]) as (keyof (typeof CHOICES)[K] & string)[]).map((value) => ({
    value,
    label: t(CHOICES[key][value] as string),
    disabled: off || false,
  }));

/** 一行：名称 + 控件。`off` 为当前不可用的原因：整行置灰、位置不变，原因显示在该行的悬停提示中（可用时没有提示）。 */
export function Row({ label, off = "", children }: { label: string; off?: string; children: React.ReactNode }) {
  // the site's one 「标签 + 控件」 row (ui/LabelRow.tsx): the label column is the panel's longest label (.vo-body is a
  // LabelGrid), a label too long for it cut and whole on hover, never over the control
  return (
    <LabelRow className={off ? "vo-row off" : "vo-row"} aria-disabled={off ? true : undefined}
      label={label} labelClass="vo-label" labelTip={tipOf("disabled", off)} ctlClass="vo-ctl">
      {children}
    </LabelRow>
  );
}

/** 带数字的滑块；`log`：按比例均匀步进（尺寸从 0.01 到 1000）。滑块给出的数与旁边 Num 输入的数按同一规则落进范围
 * （model/numbers.ts coerce）；`integer`：这一项只取整数（model/viewOptions.ts INTEGERS）。 */
export function Slider({ value, onChange, min, max, unit, log = false, digits = 2, integer = false, disabled = false }:
  { value: number; onChange: (v: number) => void; min: number; max: number; unit?: string; log?: boolean; digits?: number; integer?: boolean; disabled?: boolean }) {
  const to = (v: number) => (log ? Math.log(v / min) / Math.log(max / min) : (v - min) / (max - min));
  const from = (t: number) => (log ? min * Math.pow(max / min, t) : min + t * (max - min));
  const t = Math.min(1, Math.max(0, to(value)));
  return (
    <div className="vo-slider">
      <input
        type="range"
        min={0}
        max={1000}
        value={Math.round(t * 1000)}
        disabled={disabled}
        style={{ ["--pct" as string]: `${t * 100}%` }}
        onChange={(e) => {
          const n = coerce(Number(from(Number(e.target.value) / 1000).toPrecision(3)), { min, max, integer });
          if (n !== null && n !== value) onChange(n);
        }}
      />
      <Num value={value} onChange={onChange} min={min} max={max} integer={integer} unit={unit} digits={digits} disabled={disabled} />
    </div>
  );
}

