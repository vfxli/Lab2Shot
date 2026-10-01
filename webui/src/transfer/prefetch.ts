/** Node-level prefetch: as soon as a node is done, its display data is fetched ahead of being shown. What is on screen
 * now (priority 0) goes first, then the shown node's upstream (1), then the rest of the job (2); switching nodes or
 * scrubbing aborts lower-priority work that is not near the new focus. The frames are stored in the page's single cache
 * (platform/cache.ts) under the key the view reads.
 *
 * Prefetch fetches compressed bytes, without decoding them, and fetches the whole range whatever the network speed (see
 * `windowFrames` below). The source being viewed is fetched whole by the stage itself (`transfer/fill.ts`); this module
 * fetches the other packets. */

import { cache } from "../platform/cache";
import { bytesKey } from "./frameKey";

export type PrefetchKind = "frames2d" | "scene3d" | "points" | "curves";

/** One packet worth prefetching. */
export interface Want {
  node: string;
  port: string;
  fp: string;
  kind: PrefetchKind;
  priority: 0 | 1 | 2; // 0 displayed; 1 display-chain upstream; 2 the rest of the same job
  around?: number; // the current frame: fetched first, then outward
}

/** How the prefetcher knows a frame is already on the page (bytes or a bitmap): what is there is not fetched again.
 * Storing is the fetching layer's own job (transfer/frameStore.ts frameBytes stores what it fetches); the prefetcher
 * stores nothing itself — compressed bytes, not decoded bitmaps: decoding happens when a frame is drawn
 * (`transfer/sources.ts fromServer` / `fromChannel`). */
interface PrefetchStore {
  isCached: (key: string) => boolean;
}

/** How the prefetcher obtains frames: passed in, so the queue itself knows nothing of the network. */
export interface PrefetchSource {
  /** The frames a want covers, most wanted first (async: the manifest may still be in transit). Throws Paused when they
   * cannot be had now (the want waits and goes on when it says), anything else when there are none. */
  frames(want: Want): Promise<number[]>;
  /** Fetches one frame's compressed bytes; it may be aborted while in flight (the promise should then reject). */
  load(want: Want, frame: number, signal: AbortSignal): Promise<void>;
  /** The key a fetched frame is stored under: the view reads exactly the same key (built by transfer/frameKey.ts: the
   * packet, its generation, the channel and the proxy tier). */
  keyOf(want: Want, frame: number): string;
}

/** A want whose current frame lies within this many frames of the focus is considered near it: a focus change does not abort it. */
const NEAR = 2;

/** The fetching layer says "not now" (waiting for a new login, or the whole page or this frame is backing off): the
 * frame does not count as fetched, the want pauses, and `resume` registers the function called when it may go on. */
export class Paused extends Error {
  constructor(readonly resume: (go: () => void) => void) {
    super("paused");
  }
}

interface Job {
  want: Want;
  waiting?: boolean; // paused by the source (Paused): skipped until it says to go on
  order: number; // newest first within a priority
  fetched: Set<string>; // the frame keys (`keyOf`) already requested for this packet: never fetched again (even if evicted)
  // this packet's fetch order (from the current frame, wrapping around) and how far it has got: ordered once and
  // advanced by a cursor, rather than reordering the whole range and recomputing every key for each frame fetched
  // (O(N²) for an N-frame packet). Reordered only when the current frame (around) changes
  plan?: { around: number | undefined; frames: number[]; cursor: number };
}

export class Prefetcher {
  private jobs = new Map<string, Job>(); // fp -> job (one per packet: same fp keeps the higher priority)
  private pending = new Set<string>(); // fp of a step still determining which frame to fetch
  private inflight = new Map<string, { ctrl: AbortController; want: Want }>();
  private paused = false;
  private lastFocus = ""; // the focus already applied ("${node}:${frame}"): a repeat changes nothing
  private seq = 0;
  private revision = 0; // incremented when a higher-priority want arrives: a pending lower-priority step yields
  private cache: PrefetchStore;
  private source: PrefetchSource;
  private parallel: number;

  constructor(cache: PrefetchStore, source: PrefetchSource, parallel = 4) {
    this.cache = cache;
    this.source = source;
    this.parallel = parallel;
  }

  want(w: Want): void {
    const prev = this.jobs.get(w.fp);
    if (prev && prev.want.priority <= w.priority) {
      // the same packet already wanted at this or a higher priority: only the current frame may move
      if (w.around !== undefined) prev.want.around = w.around;
      return;
    }
    this.jobs.set(w.fp, { want: { ...w }, order: ++this.seq, fetched: new Set() });
    this.revision++;
    this.pump();
  }

  /** The user switched nodes or scrubbed the timeline: aborts work of lower priority than the shown node that is not near
   * the new focus (a want whose current frame is at the focus stays), and drops those wants so they do not immediately
   * re-fetch behind the user's view. A repeated focus does nothing (the playhead advancing by itself is not a focus
   * change: staying ahead in the play direction is the prefetcher's own responsibility). */
  focus(node: string, frame: number): void {
    const at = `${node}:${frame}`;
    if (at === this.lastFocus) return;
    this.lastFocus = at;
    const near = (w: Want) => w.node === node || (w.around !== undefined && Math.abs(w.around - frame) <= NEAR);
    for (const it of this.inflight.values()) {
      if (it.want.priority > 0 && !near(it.want)) it.ctrl.abort();
    }
    for (const [fp, job] of this.jobs) {
      if (job.want.priority > 0 && !near(job.want)) this.jobs.delete(fp);
    }
    this.pump();
  }

  pause(): void {
    this.paused = true;
    for (const it of this.inflight.values()) it.ctrl.abort();
    this.inflight.clear();
  }

  resume(): void {
    this.paused = false;
    this.pump();
  }

  private pump(): void {
    if (this.paused) return;
    while (this.inflight.size + this.pending.size < this.parallel) {
      const job = this.nextJob();
      if (!job) return;
      this.pending.add(job.want.fp);
      void this.step(job);
    }
  }

  private nextJob(): Job | undefined {
    const jobs = [...this.jobs.values()].filter((j) => !this.pending.has(j.want.fp) && !j.waiting);
    if (!jobs.length) return undefined;
    jobs.sort((a, b) => a.want.priority - b.want.priority || b.order - a.order);
    return jobs[0];
  }

  private async step(job: Job): Promise<void> {
    const rev = this.revision;
    const frames = await this.framesOf(job.want);
    this.pending.delete(job.want.fp);
    if (frames === null) {
      // the source says wait (its description not to be had now): the want stays and goes on when it says
      job.waiting = true;
      return this.pump();
    }
    if (this.paused) return this.pump();
    if (rev !== this.revision) return this.pump(); // a higher-priority want arrived meanwhile: it goes first
    if (!job.plan || job.plan.around !== job.want.around) job.plan = { around: job.want.around, frames: this.windowFrames(job.want, frames), cursor: 0 };
    const plan = job.plan;
    let next: number | undefined;
    let k = "";
    for (; plan.cursor < plan.frames.length; plan.cursor++) {
      const f = plan.frames[plan.cursor];
      k = this.source.keyOf(job.want, f);
      // a frame already in the page's single cache (fetched by the view or by an earlier want) is never fetched again
      if (!job.fetched.has(k) && !this.inflight.has(k) && !this.cache.isCached(k)) {
        next = f;
        plan.cursor++;
        break;
      }
    }
    if (next === undefined) {
      this.jobs.delete(job.want.fp);
      return this.pump();
    }
    const ctrl = new AbortController();
    this.inflight.set(k, { ctrl, want: job.want });
    let paused = false;
    try {
      await this.source.load(job.want, next, ctrl.signal);
    } catch (e) {
      // the fetching layer says wait (a login, a backoff): the frame was not fetched; the want stops and later resumes
      // from this frame. Any other failure (the server has no such frame): this want does not ask for it again
      if (e instanceof Paused) {
        paused = true;
        job.waiting = true;
        if (job.plan) job.plan.cursor = Math.max(0, job.plan.cursor - 1);
        e.resume(() => {
          job.waiting = false;
          this.pump();
        });
      }
    } finally {
      // an aborted frame was never fetched: after a pause (the tab hidden) it is asked for again; a focus change that
      // aborted it has dropped the want itself
      if (!ctrl.signal.aborted && !paused) job.fetched.add(k);
      else if (ctrl.signal.aborted && job.plan) job.plan.cursor = 0; // an aborted frame was not fetched: go through again (frames already fetched are skipped by `fetched`)
      this.inflight.delete(k);
      this.pump();
    }
  }

  /** The frames a want covers (the source answers from the packet's description, which the single cache holds:
   * requesting again costs nothing and avoids a second cache here). */
  private framesOf(want: Want): Promise<number[] | null> {
    // Paused: not now (null, the want waits and is resumed by the source); any other failure: there are none
    return Promise.resolve(this.source.frames(want)).catch((e: unknown) => {
      if (!(e instanceof Paused)) return [];
      const job = this.jobs.get(want.fp);
      e.resume(() => {
        if (job) job.waiting = false;
        this.pump();
      });
      return null;
    });
  }

  /** The frames a want fetches: all of them, starting from the current frame. Not cut short by network speed: once a
   * channel is chosen, the proxies of every frame are fetched without waiting for playback; a single compressed
   * channel is small and a colour one only three times that, so extra traffic buys smoothness. */
  private windowFrames(want: Want, frames: number[]): number[] {
    if (!frames.length) return [];
    const around = want.around !== undefined && frames.includes(want.around) ? want.around : frames[0];
    const i = frames.indexOf(around);
    return [...frames.slice(i), ...frames.slice(0, i)];
  }
}

/** Whether a frame is already in the page's single cache (bytes or a bitmap, under the stage's own key): what is there
 * is not fetched again, nor fetched a second time on switching back. */
export const prefetchStore: PrefetchStore = {
  isCached: (key) => cache.has(key) || cache.has(bytesKey(key)),
};
