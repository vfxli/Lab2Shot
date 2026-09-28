/** The page's one kind of tooltip: any element with data-tip="…" shows it in a dark floating panel after the pointer
 * rests on it (or it gets keyboard focus), and hides it when the pointer leaves, clicks or scrolls. The browser's own
 * title tooltips are not used: they are light, slow and look different everywhere. */

const DELAY_MS = 350;

/** The element whose tip a pointer or focus on `node` shows: the nearest with data-tip, unless it sits inside a
 * [data-no-tips] area, which shows none: the nodes on the graph canvas (editor/GraphNode.tsx, editor/UnknownNode.tsx),
 * the parameter panel (editor/ParamPanel.tsx), the viewer's toolbar (editor/ViewerFrame.tsx) and the timeline
 * (editor/Timeline.tsx). A menu is a menu wherever it opens: one inside such an area (the toolbar's 视角 menu) shows its
 * tips, so the nearer of the two decides. */
const tipTarget = (node: Element | null | undefined): HTMLElement | null => {
  const el = node?.closest<HTMLElement>("[data-tip]") ?? null;
  const area = el?.closest('[data-no-tips], [role="menu"]');
  return el && (!area || area.getAttribute("role") === "menu") ? el : null;
};

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
