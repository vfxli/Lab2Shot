import type { AnchorHTMLAttributes, ButtonHTMLAttributes, ReactNode } from "react";
import "./Button.css";
import { tipAttrs, type Tip } from "../platform/tips";

/** The page's buttons: every clickable control is a Button, an IconButton, a ButtonLink or a Segmented, styled only
 * here (ui/Button.css) and never restyled or resized by a page. `tip` is the pointer's tooltip: a Tip
 * (platform/tips.ts), which says why it is there — why the control is unavailable now, the full text of something cut
 * short, an exact value, an error, a keyboard shortcut, or what an action that removes or cannot be undone loses. It
 * never repeats the words on the control, and an icon's name is not a tip: it is `aria-label` (screen readers). */

type Tone = "default" | "primary" | "ghost" | "link";
type Size = "lg" | "md" | "sm" | "xs" | "xxs"; // 38 (a page that is only a form), 28, 22, 20, 16 px

interface Look {
  tip?: Tip | null;
  tone?: Tone;
  size?: Size;
  danger?: boolean; // an action that removes or ends something
  warn?: boolean; // clickable, but something is not ready: the tip and the click explain what
  on?: boolean; // a toggle that is on
  entry?: boolean; // an entry the eye must find (「模板」, 「提交反馈」): a rim in the wire colours, rotating slowly
  layout?: string; // a page's layout class only (placement, margins); never colours or sizes
  children?: ReactNode;
}

const cls = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(" ");

/** The shell of the toolbar over the view (why it must be shared: the .seg.sm section of ui/parts.css). */
export const HUD_SEG = "hud-seg glass static sm";

const classOf = ({ tone = "default", size = "md", danger, warn, on, entry, layout }: Look, shape?: string) =>
  cls("btn", tone !== "default" && tone, size !== "md" && size, shape, danger && "danger", warn && "warn", on && "on", entry && "entry", layout);

interface ButtonProps extends Look, Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className" | "children"> {}

export function Button({ tip, tone, size, danger, warn, on, entry, layout, type = "button", ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={classOf({ tip, tone, size, danger, warn, on, entry, layout })}
      {...tipAttrs(tip)}
      aria-pressed={on === undefined ? undefined : on}
      {...rest}
    />
  );
}

/** A square button holding only an icon (its children); `aria-label` (required) names it for a screen reader. */
export function IconButton({ tip, tone, size, danger, warn, on, layout, type = "button", ...rest }: ButtonProps & { "aria-label": string }) {
  return (
    <button
      type={type}
      className={classOf({ tip, tone, size, danger, warn, on, layout }, "icon-btn")}
      {...tipAttrs(tip)}
      aria-pressed={on === undefined ? undefined : on}
      {...rest}
    />
  );
}

/** A link that looks like a button: another page, a new tab, a download. */
export function ButtonLink({ tip, tone, size, danger, warn, on, layout, ...rest }: Look & Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "className" | "children">) {
  return <a className={classOf({ tip, tone, size, danger, warn, on, layout })} {...tipAttrs(tip)} {...rest} />;
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  tip?: Tip | null;
  disabled?: string | false | null; // why it cannot be chosen now (shown as its tip: disabled)
  field?: string; // data-field: the identifier that finds it from outside the page
}

interface SegmentedProps<T extends string> {
  label: string; // what the group chooses (aria-label)
  value: T | ReadonlySet<T>; // a set: several may be on at once (shown layers)
  options: readonly SegmentOption<T>[];
  onChange: (value: T) => void;
  stretch?: boolean; // fills its row; the choices share it evenly
  size?: "sm" | "md"; // 22, 26 px
  hud?: boolean; // over a view: glass style
  tabs?: boolean; // a row of tabs (role tablist)
  tip?: Tip | null; // the group's own tip, in addition to each choice's
  layout?: string;
}

/** One choice among a few, side by side (or, with a set as value, several shown or hidden). */
export function Segmented<T extends string>({ label, value, options, onChange, stretch, size = "sm", hud, tabs, tip, layout }: SegmentedProps<T>) {
  const isOn = (v: T) => (typeof value === "string" ? value === v : (value as ReadonlySet<T>).has(v));
  return (
    <div className={cls("seg", stretch && "stretch", size === "md" && "md", hud && HUD_SEG, layout)} role={tabs ? "tablist" : "group"} aria-label={label} {...tipAttrs(tip)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={isOn(o.value) ? "on" : undefined}
          role={tabs ? "tab" : undefined}
          aria-selected={tabs ? isOn(o.value) : undefined}
          aria-pressed={tabs ? undefined : isOn(o.value)}
          disabled={!!o.disabled}
          {...tipAttrs(o.disabled ? { why: "disabled", text: o.disabled } : o.tip)}
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
export function Toggle({ on, onChange, tip, hud, layout, children }: { on: boolean; onChange: (on: boolean) => void; tip?: Tip | null; hud?: boolean; layout?: string; children: ReactNode }) {
  return (
    <div className={cls("seg", hud && HUD_SEG, layout)}>
      <button type="button" className={on ? "on" : undefined} aria-pressed={on} {...tipAttrs(tip)} onClick={() => onChange(!on)}>
        {children}
      </button>
    </div>
  );
}

/** A setting that is on or off: the sliding switch. `label` names it for screen readers when no text is next to it. */
export function Switch({ on, onChange, tip, label, mini, disabled }: { on: boolean; onChange: (on: boolean) => void; tip?: Tip | null; label?: string; mini?: boolean; disabled?: boolean }) {
  return (
    <button
      type="button"
      className={cls("switch", mini && "mini", on && "on")}
      role="switch"
      aria-checked={on}
      aria-label={label}
      {...tipAttrs(tip)}
      disabled={disabled}
      onClick={() => onChange(!on)}
    />
  );
}

/** A pill that selects or filters: sm among values (a class to keep, a group to open), md in a filter row (with the
 * colour of what it filters and the count). */
export function Chip({ tip, on, size = "sm", color, count, layout, children, type = "button", style, ...rest }: Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className"> & { tip?: Tip | null; on?: boolean; size?: "sm" | "md"; color?: string; count?: number; layout?: string }) {
  return (
    <button
      type={type}
      className={cls(size === "md" ? "chip-md" : "chip", on && "on", layout)}
      aria-pressed={on === undefined ? undefined : on}
      {...tipAttrs(tip)}
      style={color ? { ...style, ["--c" as string]: color } : style}
      {...rest}
    >
      {color && <i />}
      {children}
      {count !== undefined && <> <em>{count}</em></>}
    </button>
  );
}
