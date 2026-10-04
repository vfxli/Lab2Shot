/** The timeline's arithmetic, kept free of the page (no imports): the frame
 * ruler's ticks, the visible window's zoom and pan, the playback range, runs and gaps of frames, and playback itself
 * (loop, once, ping-pong; in real time). Frames are the shot's own numbers (1001–1124), never indices. */

export type Range = [number, number]; // first and last frame, both included
export type LoopMode = "loop" | "once" | "bounce";
export type Dir = 1 | -1;

/** A named set of frames the timeline marks (a node's key frames, or any "marks" its data carries). */
export interface MarkLayer {
  name: string;
  frames: number[];
}

// ------------------------------------------------------------------ the ruler

const STEPS = [1, 5]; // ×10ⁿ: every 1, 5, 10, 50, 100, 500 … frames, as Houdini and Nuke count

function* steps(): Generator<number> {
  for (let p = 1; ; p *= 10) for (const s of STEPS) yield s * p;
}

/** The frames between numbered ticks: the smallest of 1, 5, 10, 50 … whose labels stay `minPx` apart. */
function tickStep(pxPerFrame: number, minPx = 44): number {
  if (!(pxPerFrame > 0)) return 1;
  for (const s of steps()) if (s * pxPerFrame >= minPx || s >= 1e9) return s;
  return 1;
}

/** The frames between the small ticks under the numbered ones (0: none): the smallest step that divides `major` and
 * stays `minPx` apart. */
function minorStep(major: number, pxPerFrame: number, minPx = 6): number {
  for (const s of steps()) {
    if (s >= major) return 0;
    if (major % s === 0 && s * pxPerFrame >= minPx) return s;
  }
  return 0;
}

interface Ticks {
  step: number; // frames between numbered ticks
  major: number[]; // numbered frames, multiples of step
  minor: number[]; // small ticks (not on a numbered one)
}

/** The ticks of a window `start`–`end` (frames, fractional) drawn `width` pixels wide. */
export function ticks(start: number, end: number, width: number): Ticks {
  const ppf = width / Math.max(end - start, 1e-9);
  const step = tickStep(ppf);
  const small = minorStep(step, ppf);
  const multiples = (s: number) => {
    const out: number[] = [];
    for (let f = Math.ceil(start / s) * s; f <= end; f += s) out.push(f);
    return out;
  };
  return { step, major: multiples(step), minor: small ? multiples(small).filter((f) => f % step !== 0) : [] };
}

// ------------------------------------------------------------------ the visible window

export interface View {
  start: number; // the ruler's left edge, in frames (a frame is the cell f-0.5 … f+0.5)
  end: number;
}

/** The whole shot in view: every frame's cell. */
export const fullView = (first: number, last: number): View => ({ start: first - 0.5, end: last + 0.5 });

export const frameX = (v: View, f: number, width: number) => ((f - v.start) / (v.end - v.start)) * width;
export const xFrame = (v: View, x: number, width: number) => v.start + (x / width) * (v.end - v.start);

/** The window kept inside `bounds`, as wide as it was (never wider than the bounds). */
function inside(v: View, bounds: View): View {
  const span = Math.min(v.end - v.start, bounds.end - bounds.start);
  const start = Math.min(Math.max(v.start, bounds.start), bounds.end - span);
  return { start, end: start + span };
}

/** Zoomed by `factor` (>1: closer) around frame `at`, which stays under the pointer; at least `minSpan` frames shown. */
export function zoomView(v: View, at: number, factor: number, bounds: View, minSpan = 8): View {
  const full = bounds.end - bounds.start;
  const span = Math.min(full, Math.max(Math.min(minSpan, full), (v.end - v.start) / factor));
  const k = (at - v.start) / (v.end - v.start);
  return inside({ start: at - k * span, end: at - k * span + span }, bounds);
}

/** Moved by `frames` (positive: later frames come into view). */
export const panView = (v: View, frames: number, bounds: View): View => inside({ start: v.start + frames, end: v.end + frames }, bounds);

/** The window showing frame `f`: unchanged when it is in view, else moved just enough (playing past the edge). */
export function follow(v: View, f: number, bounds: View): View {
  const span = v.end - v.start;
  if (f - 0.5 >= v.start && f + 0.5 <= v.end) return v;
  return inside(f + 0.5 > v.end ? { start: f + 0.5 - span, end: f + 0.5 } : { start: f - 0.5, end: f - 0.5 + span }, bounds);
}

// ------------------------------------------------------------------ frames and ranges

/** The page has two rules for "a frame that is not in the data", one for each kind of data, both here:
 * - `nearest`: the frames a picture has (2D sources, the decode window, the timeline's current frame, a point cache
 *   or cloud held while the frame is on its way): the existing frame nearest, ties to the earlier;
 * - `sampleAt`: the samples of an animated thing (a character, a camera, a model's placement): it holds where it last
 *   was, so the last sample at or before the frame.
 * Looking through a camera at a picture with gaps, the two can differ by design: the camera holds, the picture snaps. */

/** The sample of `frame` among an item's `frames` (ascending; runs may leave gaps): the frame itself, else the last
 * sample before it (inside a gap the thing holds where it last was, not where it ends up), else the first (before the
 * first sample; a still thing holds everywhere). -1 when the item has no frames at all: nothing per frame to draw
 * (view/sceneElement.tsx leaves out what needs a sample; a still thing draws its own shape). Binary search: called
 * every frame for every item. */
export const sampleAt = (frames: number[], frame: number) => {
  if (!frames.length) return -1;
  let lo = 0, hi = frames.length - 1, at = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid] <= frame) (at = mid), (lo = mid + 1);
    else hi = mid - 1;
  }
  return at;
};

/** Whether frame `a` is nearer to `f` than `b` by `nearest`'s rule (ties: the earlier). */
export const closer = (a: number, b: number, f: number): boolean => Math.abs(a - f) < Math.abs(b - f) || (Math.abs(a - f) === Math.abs(b - f) && a < b);

/** The existing frame nearest to `f` (frames sorted); ties go to the earlier one. */
export function nearest(frames: number[], f: number): number {
  if (!frames.length) return f;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid] < f) lo = mid + 1;
    else hi = mid;
  }
  return lo > 0 && f - frames[lo - 1] <= frames[lo] - f ? frames[lo - 1] : frames[lo];
}

/** Consecutive frames as runs: [1001, 1002, 1003, 1007] → [[1001, 1003], [1007, 1007]]. */
export function runs(frames: number[]): Range[] {
  const out: Range[] = [];
  for (const f of [...frames].sort((a, b) => a - b)) {
    const last = out.at(-1);
    if (last && f <= last[1] + 1) last[1] = Math.max(last[1], f);
    else out.push([f, f]);
  }
  return out;
}

/** The frames missing between the first and the last (a depth solved every other frame): as runs. */
export function gaps(frames: number[]): Range[] {
  const r = runs(frames);
  return r.slice(1).map(([a], i): Range => [r[i][1] + 1, a - 1]);
}

/** A typed frame number, or null (not a whole number). */
export function parseFrame(text: string): number | null {
  const t = text.trim();
  return /^-?\d+$/.test(t) ? Number(t) : null;
}

/** The playback range that applies to these frames: the one set, kept inside the data (the whole data when none is
 * set, or when the one set lies outside it, e.g. the range of another shot). */
export function playRange(frames: number[], playback: Range | null): Range | null {
  if (!frames.length) return null;
  const all: Range = [frames[0], frames.at(-1)!];
  if (!playback) return all;
  const a = Math.max(playback[0], all[0]);
  const b = Math.min(playback[1], all[1]);
  return a <= b && frames.some((f) => f >= a && f <= b) ? [a, b] : all;
}

/** A range typed or dragged, made valid for these frames: whole frames inside the data, first not after last. `moved`
 * says which end the user changed (the other one gives way). Null: the whole data (no range of its own). */
export function setRange(frames: number[], r: Range, moved: 0 | 1): Range | null {
  if (!frames.length) return null;
  const [lo, hi] = [frames[0], frames.at(-1)!];
  const clamp = (f: number) => Math.min(hi, Math.max(lo, Math.round(f)));
  let [a, b] = [clamp(r[0]), clamp(r[1])];
  if (a > b) [a, b] = moved === 0 ? [a, a] : [b, b];
  return a === lo && b === hi ? null : [a, b];
}

const within = (frames: number[], r: Range) => frames.filter((f) => f >= r[0] && f <= r[1]);

/** One step of `d` frames inside the range, wrapping around its ends (← → in Houdini and Nuke). From outside the
 * range, the first step lands on its nearest end. */
export function stepIn(frames: number[], range: Range, frame: number, d: number): number {
  const list = within(frames, range);
  if (!list.length) return frame;
  const i = list.indexOf(frame);
  // not one of the frames (between two of them): the next one the way asked, never one behind
  if (i < 0) return frame < range[0] ? list[0] : frame > range[1] ? list.at(-1)! : d > 0 ? list.find((f) => f > frame)! : [...list].reverse().find((f) => f < frame)!;
  const n = list.length;
  return list[(((i + d) % n) + n) % n];
}

/** Playback moved on by `n` frames in direction `dir`: looping, once (stops at the end) or ping-pong (turns at the
 * ends). Frames outside the range start again at its beginning (its end when playing backwards). */
export function advance(frames: number[], range: Range, frame: number, dir: Dir, mode: LoopMode, n = 1): { frame: number; dir: Dir; stopped: boolean } {
  const list = within(frames, range);
  if (!list.length) return { frame, dir, stopped: true };
  const i = list.indexOf(frame);
  if (i < 0) return { frame: dir > 0 ? list[0] : list[list.length - 1], dir, stopped: false };
  const to = moveAt(list.length, i, dir, mode, n);
  return { frame: list[to.i], dir: to.dir, stopped: to.stopped };
}

/** The move itself, on positions in the range's list of `L` frames (advance above, playOn below). */
function moveAt(L: number, i: number, dir: Dir, mode: LoopMode, n: number): { i: number; dir: Dir; stopped: boolean } {
  if (L === 1) return { i: 0, dir, stopped: mode === "once" };
  if (mode === "loop") return { i: (((i + dir * n) % L) + L) % L, dir, stopped: false };
  if (mode === "once") {
    const j = i + dir * n;
    const end = dir > 0 ? L - 1 : 0;
    const past = dir > 0 ? j >= end : j <= end;
    return { i: past ? end : j, dir, stopped: past };
  }
  const period = 2 * (L - 1); // ping-pong: unfold the back-and-forth into one line
  const u = ((((dir > 0 ? i : period - i) + n) % period) + period) % period;
  return u <= L - 1 ? { i: u, dir: 1, stopped: false } : { i: period - u, dir: -1, stopped: false };
}

/** Playback moved on by up to `n` frames one at a time, each one asked `drawable` first: it stops before the first
 * frame that cannot be drawn yet (`short`: fewer than `n` taken and not stopped at the end), never landing on it. The
 * range's frames are listed once; each step only moves a position in that list. */
export function playOn(frames: number[], range: Range, frame: number, dir: Dir, mode: LoopMode, n: number, drawable: (f: number) => boolean): { frame: number; dir: Dir; stopped: boolean; short: boolean } {
  const list = within(frames, range);
  let at = { frame, dir, stopped: !list.length };
  let i = list.indexOf(frame);
  let went = 0;
  while (went < n && !at.stopped) {
    const next = i < 0 ? { i: dir > 0 ? 0 : list.length - 1, dir: at.dir, stopped: false } : moveAt(list.length, i, at.dir, mode, 1);
    if (!drawable(list[next.i])) break;
    i = next.i;
    at = { frame: list[i], dir: next.dir, stopped: next.stopped };
    went++;
  }
  return { ...at, short: went < n && !at.stopped };
}

// ------------------------------------------------------------------ the playback clock

interface Clock {
  t0: number; // seconds: when the frame counted as step 0 was due
  done: number; // steps taken since
}

/** How many frames to move on at time `now` (seconds): real time, i.e. the shot's pace, with frames dropped when drawing
 * falls behind. Frames not in the cache yet are not skipped by this: the player stops the clock and waits for them
 * (editor/Timeline.tsx keepingUp; as in Nuke, without cache it waits frame by frame, and once cached it plays in real
 * time). There is no other mode. */
export function tick(c: Clock, now: number, fps: number): { steps: number; clock: Clock } {
  const due = Math.floor((now - c.t0) * fps + 1e-6) - c.done;
  if (due <= 0) return { steps: 0, clock: c };
  // more than a second behind: the page was not drawing at all (a tab in the background stops animation frames), not
  // drawing slowly. That time is not owed: one step, and the clock starts again from now
  if (due > Math.max(1, fps)) return { steps: 1, clock: { t0: now, done: 0 } };
  return { steps: due, clock: { t0: c.t0, done: c.done + due } };
}

// ------------------------------------------------------------------ marks carried by data

/** The marks a result's metadata carries: its key frames ("keys": an animation's keys, the in-betweening family) and
 * any named sets under "marks" ({"<name>": [...], …}): any node can give its data marks. `keysName`: what the key frames
 * are called (the page's word, in its language). */
export function markLayers(meta: Record<string, unknown>, keysName: string): MarkLayer[] {
  const frameList = (v: unknown): number[] | null =>
    Array.isArray(v) && v.every((f) => Number.isInteger(f)) ? [...new Set(v as number[])].sort((a, b) => a - b) : null;
  const out: MarkLayer[] = [];
  const keys = frameList(meta.keys);
  if (keys?.length) out.push({ name: keysName, frames: keys });
  const marks = meta.marks;
  if (marks && typeof marks === "object" && !Array.isArray(marks))
    for (const [name, v] of Object.entries(marks as Record<string, unknown>)) {
      const frames = frameList(v);
      if (frames?.length) out.push({ name, frames });
    }
  return out;
}

/** Mark layers of several results together: one layer per name, frames joined. */
export function joinLayers(layers: MarkLayer[]): MarkLayer[] {
  const by = new Map<string, Set<number>>();
  for (const l of layers) {
    const s = by.get(l.name) ?? new Set<number>();
    l.frames.forEach((f) => s.add(f));
    by.set(l.name, s);
  }
  return [...by].map(([name, s]) => ({ name, frames: [...s].sort((a, b) => a - b) }));
}
