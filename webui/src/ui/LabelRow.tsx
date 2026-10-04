import { forwardRef, useCallback, useLayoutEffect, useRef, useState, type HTMLAttributes, type ReactNode } from "react";
import { joinTips, tipAttrs, tipOf, type Tip } from "../platform/tips";
import "./labelRow.css";

/** The site's one 「标签 + 控件」 row (打磨清单 13): every row that puts a label beside its control — the parameter panel,
 * 应用模式, the admin's settings, the dialogs' forms — is this, never a layout of its own.
 *
 * A grid of cells, each with its edges, in this order: marks (the parameter panel's pins; empty elsewhere), the label,
 * the control, the tail (the settings' 需重启 / 已改 marks); under it, its `below` lines and its `note`. Nothing is positioned over another cell: a label too long
 * for its cell ends in an ellipsis and shows its whole text on hover (a `truncated` tip), never reaching into the marks or
 * the control. Lines of the row's own under it (`below`: why it is greyed, where a value comes from) start at the
 * control's left edge.
 *
 * The label column is as wide as the longest label of its group, up to a limit (`--lrow-label-max`): rows inside a
 * LabelGrid share its columns (CSS subgrid), so the group's labels line up. A grid that is told every label it can
 * ever show (`labels`: the rows hidden now by a condition too) fixes the column to the longest of them, so showing or
 * hiding a row never moves the controls; without it the browser sizes the column to the labels shown. A row on its
 * own sizes its label to its own text, with the same limit. */
export interface LabelRowProps extends Omit<HTMLAttributes<HTMLDivElement>, "children"> {
  label: ReactNode;
  /** the label's own words when `label` is more than text: what the hover says when the label is cut short */
  labelText?: string;
  marks?: ReactNode;
  children?: ReactNode;
  tail?: ReactNode;
  below?: ReactNode;
  /** the row's note: what it means or what changing it leads to, written by whoever declared the row (a setting's
   * note, a template author's note on an exposed parameter), shown under it, always visible, faint, wrapping — never a
   * hover text (platform/tips.ts: a tip may not explain). The one way a row says such a thing */
  note?: ReactNode;
  labelClass?: string;
  ctlClass?: string;
  /** the label cell's element: "label" ties it to the control inside (a form) */
  labelAs?: "span" | "label";
  /** a tip the label has of its own (one the artist cannot see otherwise: platform/tips.ts), besides its whole text
   * when it is cut short */
  labelTip?: Tip | null;
  /** `labelAs="label"`: the control it names */
  htmlFor?: string;
}

export const LabelRow = forwardRef<HTMLDivElement, LabelRowProps>(function LabelRow(
  { label, labelText, marks, children, tail, below, note, labelClass, ctlClass, labelAs = "span", labelTip, htmlFor, className, ...rest }, ref) {
  const [cut, setCut] = useState(false);
  const own = useRef<HTMLElement | null>(null);
  // cut short or not is measured where it is shown, when the pointer comes to it: no tip for a label shown whole
  const measure = () => {
    const el = own.current;
    if (el) setCut(el.scrollWidth > el.clientWidth + 0.5);
  };
  const Label = labelAs;
  const text = labelText ?? (typeof label === "string" ? label : own.current?.textContent ?? "");
  return (
    <div ref={ref} className={`lrow${className ? ` ${className}` : ""}`} {...rest}>
      <span className="lrow-marks">{marks}</span>
      <Label ref={own as never} className={`lrow-label${labelClass ? ` ${labelClass}` : ""}`} onPointerEnter={measure} onFocus={measure}
        {...(htmlFor ? { htmlFor } : {})} {...(cut ? { "data-tips": "" } : {})} {...tipAttrs(joinTips(cut ? tipOf("truncated", text) : undefined, labelTip))}>
        {label}
      </Label>
      <div className={`lrow-ctl${ctlClass ? ` ${ctlClass}` : ""}`}>{children}</div>
      {tail !== undefined && <span className="lrow-tail">{tail}</span>}
      {below}
      {note && <div className="lrow-note lrow-under">{note}</div>}
    </div>
  );
});

export interface LabelGridProps extends HTMLAttributes<HTMLDivElement> {
  /** every label a row of this grid can show, those hidden now (a Hide When, a collapsed group) included: the label
   * column is fixed to the longest (up to `--lrow-label-max`) and does not move when rows come and go. Without it the
   * column follows the labels shown. */
  labels?: readonly string[];
  /** called once the column's width has been worked out from `labels` (after it is drawn with it): the parameter
   * panel sizes itself to its rows (editor/ParamControls.tsx fitWidth) */
  onColumn?: () => void;
}

/** A group of LabelRows sharing one label column (as wide as its longest label, up to the limit): a whole parameter
 * panel (all its groups line up), a settings card, a dialog's form. Anything in it that is not a row (a title, a note,
 * a button row) takes the whole width; `.lrow-under` starts at the control column. */
export const LabelGrid = forwardRef<HTMLDivElement, LabelGridProps>(function LabelGrid({ className, labels, onColumn, style, ...rest }, ref) {
  const own = useRef<HTMLDivElement | null>(null);
  const width = useLabelColumn(own, labels);
  const told = useRef(onColumn);
  told.current = onColumn;
  useLayoutEffect(() => {
    if (width !== null) told.current?.();
  }, [width]);
  const setRef = useCallback((el: HTMLDivElement | null) => {
    own.current = el;
    if (typeof ref === "function") ref(el);
    else if (ref) ref.current = el;
  }, [ref]);
  return (
    <div ref={setRef} className={`lgrid${className ? ` ${className}` : ""}`} data-label-w={width === null ? undefined : width}
      style={width === null ? style : { ...style, ["--lrow-label-w" as string]: `min(${width}px, var(--lrow-label-max))` }} {...rest} />
  );
});

/** The width (CSS px, its padding included) the longest of `labels` takes in a label cell of `grid`: each written into
 * a copy of a label cell shown now (its classes, so its font and padding), laid out unconstrained and unseen beside it,
 * then removed (null while no label is shown, or without `labels`). Measured again when the labels change and once the
 * page's fonts have loaded. */
function useLabelColumn(grid: React.RefObject<HTMLDivElement | null>, labels: readonly string[] | undefined): number | null {
  const [width, setWidth] = useState<number | null>(null);
  const key = labels ? labels.join("\n") : null;
  useLayoutEffect(() => {
    if (key === null) return setWidth(null);
    const measure = () => {
      const cell = grid.current?.querySelector<HTMLElement>(".lrow-label");
      if (!cell?.parentElement) return;
      const probe = cell.cloneNode(false) as HTMLElement;
      probe.removeAttribute("id");
      probe.setAttribute("aria-hidden", "true");
      probe.style.cssText = "position:absolute;visibility:hidden;left:0;top:0;width:auto;max-width:none;overflow:visible;white-space:nowrap;pointer-events:none";
      cell.parentElement.append(probe);
      let most = 0;
      for (const l of key.split("\n")) {
        if (!l) continue;
        probe.textContent = l;
        most = Math.max(most, probe.getBoundingClientRect().width);
      }
      probe.remove();
      setWidth(most > 0 ? Math.ceil(most + 1) : null);
    };
    measure();
    let live = true;
    void document.fonts?.ready.then(() => live && measure());
    return () => { live = false; };
  }, [key, grid]);
  return width;
}
