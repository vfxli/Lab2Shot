/** The curve editor's arithmetic, kept free of the page (no imports): which
 * channels are picked (a click, Ctrl, Shift, as in Maya's and Houdini's channel lists), which move most, the value
 * range a graph fits and its axis ticks. */

/** A curves packet's before/after pairing as indices: `pair` maps a channel to the channel
 * holding what it was before, `hidden` is every channel that is only another channel's before (they are never listed on
 * their own; one switch shows them all). A name the packet does not have is ignored. */
export function beforePairs(names: string[], before?: Record<string, string>): { pair: Map<number, number>; hidden: Set<number> } {
  const index = new Map(names.map((name, i) => [name, i]));
  const pair = new Map<number, number>();
  const hidden = new Set<number>();
  for (const [after, original] of Object.entries(before ?? {})) {
    const a = index.get(after);
    const b = index.get(original);
    if (a === undefined || b === undefined || a === b) continue;
    pair.set(a, b);
    hidden.add(b);
  }
  for (const a of pair.keys()) hidden.delete(a); // a channel that is both stays listed
  return { pair, hidden };
}

/** The channels (indices) that move most over the shot, the first `n`, in their own order; `skip` is left out. */
export function mostMoving(values: number[][], n: number, skip?: ReadonlySet<number>): number[] {
  const spread = values.map((v) => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const x of v) {
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    return v.length ? hi - lo : 0;
  });
  return values
    .map((_, i) => i)
    .filter((i) => !skip?.has(i))
    .sort((a, b) => spread[b] - spread[a] || a - b)
    .slice(0, n)
    .sort((a, b) => a - b);
}

export type PickMode = "only" | "toggle" | "range";

/** A click on channel `i` in the list as shown (`shown`, in its order): alone ("only"), added or taken away (Ctrl:
 * "toggle"), or every channel from the last one clicked (`anchor`) to it (Shift: "range", added to what is picked). */
export function pick(picked: ReadonlySet<number>, shown: number[], i: number, mode: PickMode, anchor: number | null): Set<number> {
  if (mode === "only") return new Set([i]);
  if (mode === "toggle") {
    const next = new Set(picked);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    return next;
  }
  const a = anchor === null ? -1 : shown.indexOf(anchor);
  const b = shown.indexOf(i);
  if (a < 0 || b < 0) return new Set([...picked, i]);
  const [lo, hi] = a < b ? [a, b] : [b, a];
  return new Set([...picked, ...shown.slice(lo, hi + 1)]);
}

/** Channel names matching what is typed in the search (case aside; every word must be there), as indices. */
export function search(names: string[], typed: string, skip?: ReadonlySet<number>): number[] {
  const words = typed.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return names.map((_, i) => i).filter((i) => !skip?.has(i) && words.every((w) => names[i].toLowerCase().includes(w)));
}

/** The value range the graph shows: every value of the picked channels at frames `from`–`to`, with a margin; a flat
 * curve gets room around it. */
export function fitRange(values: number[][], frames: number[], channels: Iterable<number>, from: number, to: number): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const c of channels)
    frames.forEach((f, k) => {
      if (f < from || f > to) return;
      const v = values[c][k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
  if (!Number.isFinite(lo)) return [-1, 1];
  const pad = hi > lo ? (hi - lo) * 0.08 : Math.max(Math.abs(lo) * 0.1, 0.5);
  return [lo - pad, hi + pad];
}

/** Round values for a value axis from `lo` to `hi`, about `count` of them (steps of 1, 2 or 5 × 10ⁿ). */
export function valueTicks(lo: number, hi: number, count: number): { step: number; values: number[] } {
  const span = hi - lo;
  if (!(span > 0) || !(count > 0)) return { step: 1, values: [] };
  const rough = span / count;
  const p = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= rough) ?? 10 * p;
  const values: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { step, values };
}

/** A value as the axis and the list write it: as many decimals as the step needs. */
export function formatValue(v: number, step = 0.01): string {
  const decimals = Math.min(6, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(decimals);
}
