/** A bar of cells that never overlap or squeeze one another (the 「标签 + 控件」 row's rule, ui/LabelRow.tsx, for a
 * toolbar): every cell keeps its own width, and when the bar is too narrow for all of them the ones that only tell
 * something (marked `data-fit="<n>"`, the highest n first) step out whole, one after another, until the rest fits and
 * the bar's one stretching cell (`data-fit-keep`: the document's name) is shown whole. Nothing is cut in the middle of
 * a number; a cell that steps out comes back as soon as there is room for it again. No width is written anywhere: what
 * fits is measured in the language and the window as they are. */
import { useLayoutEffect, type RefObject } from "react";

const HIDDEN = "data-fit-out";

function fit(bar: HTMLElement): void {
  const optional = [...bar.querySelectorAll<HTMLElement>("[data-fit]")].sort((a, b) => Number(b.dataset.fit) - Number(a.dataset.fit));
  const keep = bar.querySelector<HTMLElement>("[data-fit-keep]");
  for (const el of optional) el.removeAttribute(HIDDEN);
  const tight = () => bar.scrollWidth > bar.clientWidth + 0.5 || (!!keep && keep.scrollWidth > keep.clientWidth + 0.5);
  for (const el of optional) {
    if (!tight()) break;
    el.setAttribute(HIDDEN, "");
  }
}

/** Keeps `ref`'s bar fitted (see above) as its size, its cells' sizes or `deps` (what its words are made of) change. */
export function useFitBar(ref: RefObject<HTMLElement | null>, deps: readonly unknown[]): void {
  useLayoutEffect(() => {
    const bar = ref.current;
    if (!bar) return;
    fit(bar);
    // the bar itself (the window), and each cell (a button whose words grow: 「队列 · 计算中」, another language)
    const seen = new ResizeObserver(() => fit(bar));
    seen.observe(bar);
    for (const cell of bar.querySelectorAll(":scope > *, :scope > * > *")) seen.observe(cell);
    return () => seen.disconnect();
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
}
