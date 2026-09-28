/** How long to wait before asking again after failures in a row: `first` ms after the first, doubled with each
 * further one, never more than `most`. The one retry schedule of the page: polling (platform/poll.ts), uploads waiting
 * for the line (transfer/uploads.ts), frames and 3D chunks that failed to arrive (transfer/frames.ts, view/scene.ts).
 * `fails`: failures in a row so far, 1 for the first. */
export const backoff = (fails: number, first: number, most: number): number => Math.min(most, first * 2 ** Math.max(0, fails - 1));
