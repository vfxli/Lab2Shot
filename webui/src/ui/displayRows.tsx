import { CHOICES, CHOICE_TIPS, type ViewOptions } from "../model/viewOptions";
import type { ViewControl } from "../model/viewControls";
import { type SegmentOption } from "./Button";
import { Num } from "./controls";
import { coerce } from "../model/numbers";

/** 显示选项面板的行与小组件（ui/DisplayOptions.tsx 及其各页共用）：行布局、滑块、色标；数字输入是页面唯一的那个（ui/controls.tsx Num）。
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
        data-tip={`拖动调整；右边可以输入（${min}–${max}${unit ? ` ${unit}` : ""}）`}
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

