/** Node-level prefetch: as soon as a node is done, its display data is fetched ahead of being shown. What is on screen
 * now (priority 0) goes first, then the shown node's upstream (1), then the rest of the job (2); switching nodes or
 * scrubbing aborts lower-priority work that is not near the new focus. The frames are stored in the page's single cache
 * (transfer/cache.ts) under the key the view reads, so webui/tests/prefetch.test.ts drives the queue without a server.
 *
 * 预取取回的是压缩字节，不解码，且无论网络快慢都取回整段（见下方 `windowFrames`）。
 * 当前查看的源由舞台自行整段取回（`transfer/fill.ts`），本模块取回的是其他数据包。 */

import { cache } from "./cache";
import { bytesKey } from "./sources";

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

/** 一次取回的结果：压缩字节及其字节数。`value` 为 null 时，该帧的字节已由取数层
 * 自行存储（一次取多条通道时即如此），此处不再重复存储。
 *
 * 不是解码后的位图（存储压缩态而非解码态）：预取只负责取回字节，
 * 解码发生在需要绘制的时刻（`transfer/sources.ts fromServer` / `fromChannel`）。 */
export interface Fetched<V> {
  value: V | null;
  bytes: number;
}

/** Where a fetched frame is stored: the page's single cache (transfer/cache.ts), or a stand-in in the tests. */
export interface PrefetchStore<V> {
  isCached: (key: string) => boolean;
  keep: (key: string, value: V, bytes: number) => void;
}

/** How the prefetcher obtains frames, abstracted so that the tests can drive it. */
export interface PrefetchSource<V> {
  /** The frames a want covers, most wanted first (async: the manifest may still be in transit). */
  frames(want: Want): Promise<number[]>;
  /** Fetches one decoded frame; it may be aborted while in flight (the promise should then reject). */
  load(want: Want, frame: number, signal: AbortSignal): Promise<Fetched<V>>;
  /** The key a fetched frame is stored under, when it is not `${fp}:${frame}`: the view reads exactly the same key, so
   * everything the view puts in the id is included here too (transfer/sources.ts: the packet, the channel and the proxy tier). */
  keyOf?(want: Want, frame: number): string;
}

const key = (fp: string, frame: number) => `${fp}:${frame}`;

/** A want whose current frame lies within this many frames of the focus is considered near it: a focus change does not abort it. */
const NEAR = 2;

interface Job {
  want: Want;
  order: number; // newest first within a priority
  fetched: Set<string>; // `${fp}:${frame}` already requested for this packet: never fetched again (even if evicted)
}

export class Prefetcher<V> {
  private jobs = new Map<string, Job>(); // fp -> job (one per packet: same fp keeps the higher priority)
  private pending = new Set<string>(); // fp of a step still determining which frame to fetch
  private inflight = new Map<string, { ctrl: AbortController; want: Want }>();
  private paused = false;
  private lastFocus = ""; // the focus already applied ("${node}:${frame}"): a repeat changes nothing
  private seq = 0;
  private revision = 0; // incremented when a higher-priority want arrives: a pending lower-priority step yields
  private cache: PrefetchStore<V>;
  private source: PrefetchSource<V>;
  private parallel: number;

  constructor(cache: PrefetchStore<V>, source: PrefetchSource<V>, parallel = 4) {
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
    const jobs = [...this.jobs.values()].filter((j) => !this.pending.has(j.want.fp));
    if (!jobs.length) return undefined;
    jobs.sort((a, b) => a.want.priority - b.want.priority || b.order - a.order);
    return jobs[0];
  }

  /** The key this frame's picture is stored under: the source's key if it defines one (transfer/prefetchLive.ts), else
   * `${fp}:${frame}`, the same key transfer/frames.ts reads, so a prefetched frame is never fetched twice. */
  private keyOf(want: Want, frame: number): string {
    return this.source.keyOf ? this.source.keyOf(want, frame) : key(want.fp, frame);
  }

  private async step(job: Job): Promise<void> {
    const rev = this.revision;
    const frames = await this.framesOf(job.want);
    this.pending.delete(job.want.fp);
    if (this.paused) return this.pump();
    if (rev !== this.revision) return this.pump(); // a higher-priority want arrived meanwhile: it goes first
    const next = this.windowFrames(job.want, frames).find((f) => {
      const k = this.keyOf(job.want, f);
      // a frame already in the page's single cache (fetched by the view or by an earlier want) is never fetched again
      return !job.fetched.has(k) && !this.inflight.has(k) && !this.cache.isCached(k);
    });
    if (next === undefined) {
      this.jobs.delete(job.want.fp);
      return this.pump();
    }
    const k = this.keyOf(job.want, next);
    const ctrl = new AbortController();
    this.inflight.set(k, { ctrl, want: job.want });
    try {
      const got = await this.source.load(job.want, next, ctrl.signal);
      if (!ctrl.signal.aborted && got.value !== null) this.cache.keep(k, got.value, got.bytes);
    } catch {
      /* aborted, or the server has no such frame: neither is requested again for this want */
    } finally {
      job.fetched.add(k);
      this.inflight.delete(k);
      this.pump();
    }
  }

  /** The frames a want covers (the source answers from the packet's description, which the single cache holds:
   * requesting again costs nothing and avoids a second cache here). */
  private framesOf(want: Want): Promise<number[]> {
    return Promise.resolve(this.source.frames(want)).catch(() => []);
  }

  /** 一个 want 需取回的帧：全部帧，从当前帧开始。不按网络快慢截断：用户选择某个通道后即开始取回所有帧的代理，
   * 不等待播放；单个压缩通道体积不大，彩色通道也只是三倍，以额外流量换取流畅度。 */
  private windowFrames(want: Want, frames: number[]): number[] {
    if (!frames.length) return [];
    const around = want.around !== undefined && frames.includes(want.around) ? want.around : frames[0];
    const i = frames.indexOf(around);
    return [...frames.slice(i), ...frames.slice(0, i)];
  }
}

/** 预取写入页面唯一缓存的方式：存储压缩字节（`small` 层，与舞台自行取回的字节
 * 使用同一个键），不存储解码后的位图（一帧数十 KB 对比 0.6 MB，且解码层另有预算）。
 * 已持有（字节或位图）时不再获取，因此切回时不会重复取回。 */
export const prefetchStore: PrefetchStore<Blob> = {
  isCached: (key) => cache.has(key) || cache.has(bytesKey(key)),
  keep: (key, value, bytes) => cache.keep(bytesKey(key), value, bytes, "small"),
};
