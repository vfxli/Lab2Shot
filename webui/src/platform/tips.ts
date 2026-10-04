/** The page's one kind of tooltip: an element given a Tip (tipAttrs: data-tip="…" with its data-tip-why) shows it in a
 * dark floating panel after the pointer rests on it (or it gets keyboard focus), and hides it when the pointer leaves,
 * clicks or scrolls. The browser's own title tooltips are not used: they are light, slow and look different everywhere.
 *
 * A tip only ever says what the artist cannot see (the user's rule, 打磨清单 14): every tip declares why it is there
 * (TipWhy), and there is no other way to make one — components take a `Tip`, never a bare text (tsc), and a raw
 * data-tip attribute anywhere but here is refused by `lab2shot check` (tips). A tip that would repeat what the
 * control already says, or explain the design at length, has no why to give and is not written. */

/** Why a tip is there: the only things a tip may say.
 * - truncated: the whole of a text cut short where it is shown (a long name, a path);
 * - disabled: why a control cannot be used now;
 * - value: an exact value the control shows rounded or not at all (a number, a time, a size, a version);
 * - error: what went wrong, in full;
 * - shortcut: the key that does it;
 * - consequence: what a destructive action loses or leaves, not said on the control. */
export type TipWhy = "truncated" | "disabled" | "value" | "error" | "shortcut" | "consequence";

export const TIP_WHYS: readonly TipWhy[] = ["truncated", "disabled", "value", "error", "shortcut", "consequence"];

/** A tip: its words and why it may be shown (TipWhy), made by `tipOf`. */
export interface Tip {
  readonly why: TipWhy;
  readonly text: string;
}

/** A tip for `why` saying `text`; none (undefined) when there is nothing to say. */
export function tipOf(why: TipWhy, text: string | null | undefined): Tip | undefined {
  const said = (text ?? "").trim();
  return said ? { why, text: said } : undefined;
}

/** The attributes an element carries for its tip (none when there is none): the one place data-tip is written. */
export function tipAttrs(t: Tip | null | undefined | false): { "data-tip"?: string; "data-tip-why"?: TipWhy } {
  return t ? { "data-tip": t.text, "data-tip-why": t.why } : {};
}

/** Two tips as one (a control's own and its group's): the first's why, both texts on their own lines. */
export function joinTips(a: Tip | null | undefined, b: Tip | null | undefined): Tip | undefined {
  if (!a || !b) return (a ?? b) || undefined;
  return { why: a.why, text: `${a.text}\n${b.text}` };
}

const DELAY_MS = 350;

/** The element whose tip a pointer or focus on `node` shows: the nearest with data-tip, unless it sits inside a
 * [data-no-tips] area, which shows none: the nodes on the graph canvas (editor/GraphNode.tsx, editor/UnknownNode.tsx),
 * the parameter panel (editor/ParamPanel.tsx), the viewer's toolbar (editor/ViewerFrame.tsx) and the timeline
 * (editor/Timeline.tsx). A menu is a menu wherever it opens: one inside such an area (the toolbar's 视角 menu) shows its
 * tips, so the nearer of the two decides; so does a part of such an area marked [data-tips] (a node's name: its name,
 * type and description). */
const tipTarget = (node: Element | null | undefined): HTMLElement | null => {
  const el = node?.closest<HTMLElement>("[data-tip][data-tip-why]") ?? null; // a tip without its why is never shown
  const area = el?.closest('[data-no-tips], [role="menu"], [data-tips]');
  return el && (!area || area.getAttribute("role") === "menu" || area.hasAttribute("data-tips")) ? el : null;
};

/** Whether the text of `el` (or of an element inside it) is cut short where it is shown: a "truncated" tip shows only
 * then, so a name that fits never repeats itself on hover. */
function cutShort(el: HTMLElement): boolean {
  const over = (e: Element) => e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1;
  return over(el) || [...el.querySelectorAll("*")].slice(0, 40).some(over);
}

export function installTips(): void {
  const tip = document.createElement("div");
  tip.className = "tip glass";
  tip.setAttribute("role", "tooltip");
  document.body.append(tip);
  let target: HTMLElement | null = null;
  let timer = 0;
  let pointer = { x: 0, y: 0 };

  const hide = () => {
    window.clearTimeout(timer);
    target = null;
    tip.classList.remove("on");
  };

  const show = (armed: HTMLElement) => {
    if (target !== armed) return;
    // the page may have drawn the element again meanwhile: take the one under the pointer now
    const el = armed.isConnected ? armed : tipTarget(document.elementFromPoint(pointer.x, pointer.y));
    const text = el?.dataset.tip;
    if (!el || !text) return;
    if (el.dataset.tipWhy === "truncated" && !cutShort(el)) return; // the whole text is in view: nothing to add
    tip.textContent = text;
    tip.classList.add("on");
    const r = el.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    const gap = 8;
    const below = r.bottom + gap + t.height < window.innerHeight;
    const top = below ? r.bottom + gap : Math.max(gap, r.top - gap - t.height);
    const left = Math.min(Math.max(gap, r.left + r.width / 2 - t.width / 2), window.innerWidth - t.width - gap);
    tip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  };

  const arm = (el: HTMLElement | null) => {
    if (el === target) return;
    hide();
    if (!el?.dataset.tip) return;
    target = el;
    timer = window.setTimeout(() => show(el), DELAY_MS);
  };

  // Scrolling hides the tip (it would float away from its element), but the pointer usually comes to rest over another
  // element with help: the page moved under it, so no pointerover follows. Once scrolling settles, arm whatever is
  // under the pointer now — otherwise a wheel scroll (or a scroll into view before a hover) leaves help dead until the
  // pointer is moved again.
  let inside = false;
  let settle = 0;
  const scrolled = () => {
    hide();
    window.clearTimeout(settle);
    if (!inside) return;
    settle = window.setTimeout(() => arm(tipTarget(document.elementFromPoint(pointer.x, pointer.y))), 120);
  };

  document.addEventListener("pointerover", (e) => arm(tipTarget(e.target as Element)));
  document.addEventListener("pointermove", (e) => ((pointer = { x: e.clientX, y: e.clientY }), (inside = true)), { passive: true });
  document.documentElement.addEventListener("pointerleave", () => (inside = false));
  // a tip on focus is for someone moving by keyboard: a text field is always :focus-visible, so a page that focuses
  // its first field as it opens (the login) would otherwise open a tip over its own form before anyone touched a key
  let byKey = false;
  document.addEventListener("focusin", (e) => {
    const el = tipTarget(e.target as Element);
    if (byKey && el?.matches(":focus-visible")) arm(el);
  });
  document.addEventListener("focusout", hide);
  document.addEventListener("pointerdown", () => ((byKey = false), hide()));
  document.addEventListener("keydown", (e) => ((byKey = e.key === "Tab"), hide()));
  window.addEventListener("scroll", scrolled, true);
  window.addEventListener("blur", hide);
}
