import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";
import "./Button.css";

/** The page's buttons: every clickable control is a Button, an IconButton, a ButtonLink or a Segmented, styled only
 * here (ui/Button.css) and never restyled or resized by a page (webui/tests/uiKit.test.ts). A tip is required: `tip` is
 * the pointer's tooltip (data-tip). */

type Tone = "default" | "primary" | "ghost" | "link";
type Size = "lg" | "md" | "sm" | "xs" | "xxs"; // 38 (a page that is only a form), 28, 22, 20, 16 px

interface Look {
  tip: string;
  tone?: Tone;
  size?: Size;
  danger?: boolean; // an action that removes or ends something
  warn?: boolean; // clickable, but something is not ready: the tip and the click explain what
  on?: boolean; // a toggle that is on
  entry?: boolean; // the page's single entry point (「模板」): a rim in the wire colours, rotating slowly
  layout?: string; // a page's layout class only (placement, margins); never colours or sizes
  children?: ReactNode;
}

const cls = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(" ");

/** 视图上方工具栏的外壳（必须共用一份的原因见 ui/parts.css 中 .seg.sm 一段）。 */
export const HUD_SEG = "hud-seg glass static sm";

const classOf = ({ tone = "default", size = "md", danger, warn, on, entry, layout }: Look, shape?: string) =>
  cls("btn", tone !== "default" && tone, size !== "md" && size, shape, danger && "danger", warn && "warn", on && "on", entry && "entry", layout);

export interface ButtonProps extends Look, Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className" | "children"> {}

export function Button({ tip, tone, size, danger, warn, on, entry, layout, type = "button", ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={classOf({ tip, tone, size, danger, warn, on, entry, layout })}
      data-tip={tip}
      aria-pressed={on === undefined ? undefined : on}
      {...rest}
    />
  );
}

/** A square button holding only an icon (its children); `aria-label` names it for a screen reader. */
export function IconButton({ tip, tone, size, danger, warn, on, layout, type = "button", ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={classOf({ tip, tone, size, danger, warn, on, layout }, "icon-btn")}
      data-tip={tip}
      aria-pressed={on === undefined ? undefined : on}
      {...rest}
    />
  );
}

/** A link that looks like a button: another page, a new tab, a download. */
export function ButtonLink({ tip, tone, size, danger, warn, on, layout, ...rest }: Look & Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "className" | "children">) {
  return <a className={classOf({ tip, tone, size, danger, warn, on, layout })} data-tip={tip} {...rest} />;
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  tip: string;
  disabled?: string | false | null; // why it cannot be chosen now (shown as its tip)
  field?: string; // data-field: the identifier UI tests use to find it
}

export interface SegmentedProps<T extends string> {
  label: string; // what the group chooses (aria-label)
  value: T | ReadonlySet<T>; // a set: several may be on at once (shown layers)
  options: readonly SegmentOption<T>[];
  onChange: (value: T) => void;
  stretch?: boolean; // fills its row; the choices share it evenly
  size?: "sm" | "md"; // 22, 26 px
  hud?: boolean; // over a view: glass style
  tabs?: boolean; // a row of tabs (role tablist)
  tip?: string; // the group's own tip, in addition to each choice's
  layout?: string;
}

/** One choice among a few, side by side (or, with a set as value, several shown or hidden). */
export function Segmented<T extends string>({ label, value, options, onChange, stretch, size = "sm", hud, tabs, tip, layout }: SegmentedProps<T>) {
  const isOn = (v: T) => (typeof value === "string" ? value === v : (value as ReadonlySet<T>).has(v));
  return (
    <div className={cls("seg", stretch && "stretch", size === "md" && "md", hud && HUD_SEG, layout)} role={tabs ? "tablist" : "group"} aria-label={label} data-tip={tip}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={isOn(o.value) ? "on" : undefined}
          role={tabs ? "tab" : undefined}
          aria-selected={tabs ? isOn(o.value) : undefined}
          aria-pressed={tabs ? undefined : isOn(o.value)}
          disabled={!!o.disabled}
          data-tip={o.disabled || o.tip}
          data-field={o.field}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** A single on/off control that reads as a word (实时, 曲线, 显示): a segmented control with one option. */
export function Toggle({ on, onChange, tip, hud, layout, children }: { on: boolean; onChange: (on: boolean) => void; tip: string; hud?: boolean; layout?: string; children: ReactNode }) {
  return (
    <div className={cls("seg", hud && HUD_SEG, layout)}>
      <button type="button" className={on ? "on" : undefined} aria-pressed={on} data-tip={tip} onClick={() => onChange(!on)}>
        {children}
      </button>
    </div>
  );
}

/** A setting that is on or off: the sliding switch. `label` names it for screen readers when no text is next to it. */
export function Switch({ on, onChange, tip, label, mini, disabled }: { on: boolean; onChange: (on: boolean) => void; tip?: string; label?: string; mini?: boolean; disabled?: boolean }) {
  return (
    <button
      type="button"
      className={cls("switch", mini && "mini", on && "on")}
      role="switch"
      aria-checked={on}
      aria-label={label}
      data-tip={tip}
      disabled={disabled}
      onClick={() => onChange(!on)}
    />
  );
}

/** A pill that selects or filters: sm among values (a class to keep, a group to open), md in a filter row (with the
 * colour of what it filters and the count). */
export function Chip({ tip, on, size = "sm", color, count, layout, children, type = "button", style, ...rest }: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className"> & { tip: string; on?: boolean; size?: "sm" | "md"; color?: string; count?: number; layout?: string }) {
  return (
    <button
      type={type}
      className={cls(size === "md" ? "chip-md" : "chip", on && "on", layout)}
      aria-pressed={on === undefined ? undefined : on}
      data-tip={tip}
      style={color ? { ...style, ["--c" as string]: color } : style}
      {...rest}
    >
      {color && <i />}
      {children}
      {count !== undefined && <> <em>{count}</em></>}
    </button>
  );
}
