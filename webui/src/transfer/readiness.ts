/** 帧就绪——唯一的判定：画面「这一帧能不能合成」与播放器「要不要等这一帧」都问这里，二维、三维同一个接口。
 *
 * 画面：一侧要用的几格（图片、各条通道）都拿到了**这一帧**才换画面；某格还停在上一帧（没到时取帧账本给的是它上一张图）
 * 就不拿来拼——否则 R 是第 N 帧、G 是第 N−1 帧，或左边第 N 帧、右边第 N−1 帧。没到齐时整侧沿用上一次完整合成的那一帧。
 *
 * 两侧（左 ⊕ 右）只在两侧都完整、且是同一帧时才换成新的一对（`PairHold`）；拼不成时整对沿用上一次完整的那一对，
 * 从没有过就是还没到（画「加载中」），从不只画一侧、静默丢掉另一侧。
 *
 * 播放器：各舞台报告三件事（`reportLoads`）——无须再下载的帧（时间线的颜色）、可立即画的帧、在途的帧；卸载时一起清掉
 * （`clearLoads`）。播放器只问 `playable(帧)`。「可立即画」只有一个定义：舞台在用的每一份内容这一帧都已在内存里可画——
 * 二维的图和通道、三维的背板是「已解码」（transfer/frames.ts useDecodedFrames），三维的块是「已解开或在本机」
 * （本机的块解码是毫秒级，view/scene.ts）；各份按自己的帧合起来（`readyAcross`：一份没有的帧不因它而不可画）。 */

import { useViewLoads } from "../state/viewer";
import { onItsWay } from "./frames";

export interface Cell<P> {
  image: P | null;
  frame: number | null;
  absent?: boolean; // 这一格的源本来就没有这一帧（不是还没到）
}

/** 这几格是否都已是 `frame` 这一帧。 */
export const complete = <P>(cells: readonly Cell<P>[], frame: number): boolean =>
  cells.length > 0 && cells.every((c) => c.image !== null && c.frame === frame);

/** 一侧的合成：到齐了用这一帧；没到齐沿用上一次完整的那一帧（同一组源才沿用：换了源就不沿用旧的）。
 * 沿用前核对那几张图仍在页面缓存里（`held(格, 帧, 图)`）：被淘汰的位图已经 close、平面也不再计入预算，不能再拿来画。 */
export class SideHold<P> {
  private last: { ids: string; frame: number; images: (P | null)[] } | null = null;

  take(ids: string, frame: number, cells: readonly Cell<P>[], need: readonly number[],
       held: (cell: number, frame: number, image: P) => boolean): SideTake<P> {
    // 要用的格本来就没有这一帧：这一侧这一帧缺席，不沿用旧的（沿用会让画面停在旧帧而看不出来）
    if (need.some((i) => cells[i].absent)) return { images: cells.map(() => null), frame: null, ready: false, absent: true };
    if (complete(need.map((i) => cells[i]), frame)) {
      const images = cells.map((c) => (c.frame === frame ? c.image : null));
      const was = this.last;
      // 同一组源、同一帧、每格还是那几张：给上一次的数组（舞台的绘制按它判断要不要重画，新数组会让它每次渲染都重画）
      if (was && was.ids === ids && was.frame === frame && was.images.length === images.length && was.images.every((p, i) => p === images[i]))
        return { images: was.images, frame, ready: true };
      this.last = { ids, frame, images };
      return { images, frame, ready: true };
    }
    const was = this.last;
    if (was && was.ids === ids && was.images.every((p, i) => p === null || held(i, was.frame, p)))
      return { images: was.images, frame: was.frame, ready: true };
    return { images: cells.map(() => null), frame: null, ready: false };
  }
}

/** 一侧合成的样子（SideHold.take 给的）。`absent`：这一侧本来就没有这一帧。 */
export interface SideTake<P> {
  images: (P | null)[];
  frame: number | null;
  ready: boolean;
  absent?: boolean;
}

/** 两侧合成的一对：两侧都完整、且是同一帧（不拿左边第 N 帧配右边第 N−1 帧）才换成这一对；否则整对沿用上一次完整的那
 * 一对（同一组源，且两侧的图仍在页面缓存里：`held`），没有就是还没到（ready false）。一侧本来就没有这一帧（`absent`）
 * 不是「还没到」：另一侧照它自己的画，缺的那一侧不画（舞台说明缺了哪一帧），不沿用旧的一对。 */
export class PairHold<P> {
  private last: { ids: string; a: SideTake<P>; b: SideTake<P> } | null = null;

  take(ids: string, a: SideTake<P>, b: SideTake<P>, held: (side: 0 | 1, s: SideTake<P>) => boolean): { a: SideTake<P>; b: SideTake<P>; ready: boolean } {
    if (a.absent || b.absent) return { a, b, ready: false };
    if (a.ready && b.ready && a.frame === b.frame) {
      this.last = { ids, a, b };
      return { a, b, ready: true };
    }
    const was = this.last;
    if (was && was.ids === ids && held(0, was.a) && held(1, was.b)) return { a: was.a, b: was.b, ready: true };
    return { a, b, ready: false };
  }
}

/** 几格合起来可以画的帧，按每格自己的帧：一帧可画 = 每个「有这一帧」的格这一帧都已可画；某格本来没有这一帧（它的
 * `has` 里没有：「运算」档帧数少的那一路、比场景长的背板之外的那一段）不参与这一帧的判断，由舞台按缺帧显示，
 * 不让这一帧永远不可画（播放器会在那里永远等）。`has` 为 null：静止的一格（任何一帧都是它，有过就都可画）。 */
export function readyAcross(cells: readonly { has: readonly number[] | null; ready: readonly number[] }[]): number[] | null {
  if (!cells.length) return null;
  const all = [...new Set(cells.flatMap((c) => c.has ?? c.ready))].sort((a, b) => a - b);
  const sets = cells.map((c) => ({ has: c.has && new Set(c.has), ready: new Set(c.ready) }));
  return all.filter((f) => sets.every((c) => (c.has ? !c.has.has(f) || c.ready.has(f) : c.ready.size > 0)));
}

/** 舞台报告：`loaded` 无须再下载的帧（时间线颜色）、`ready` 可立即画的帧、`waiting` 在途的帧、`stale` 画的是上一次的结果。
 * null：这个舞台没有逐帧的内容。 */
export function reportLoads(p: { loaded: number[] | null; ready: number[] | null; waiting: number[] | null; stale: boolean; cached?: [number, number] | null }): void {
  useViewLoads.setState({ loaded: p.loaded, ready: p.ready, waiting: p.waiting, stale: p.stale, cached: p.cached ?? null });
}

/** 舞台卸载：它报告过的一起清掉（两个舞台同一个清法）。 */
export function clearLoads(): void {
  useViewLoads.setState({ loaded: null, ready: null, waiting: null, stale: false, cached: null });
}

/** 播放器要不要走到 `frame`：已能立即画就走；在途（三维块、二维取帧账本里有人在取）就等；没人在取也不等（否则死等）。
 * 没有逐帧内容时照走。 */
export function playable(frame: number): boolean {
  const s = useViewLoads.getState();
  const ready = s.ready ?? s.loaded;
  if (!ready) return true;
  return ready.includes(frame) || !(onItsWay(frame) || !!s.waiting?.includes(frame));
}
