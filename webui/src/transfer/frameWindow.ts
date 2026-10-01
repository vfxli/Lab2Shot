/** The arithmetic of fetching frames, independent of the page:
 *
 * - which frames a player wants now: the current one, then frames ahead in the direction of play and a few behind;
 * - how far ahead to decode: as many frames as the source's share of the decoded budget holds. There is no "seconds
 *   ahead while playing" cap: such a window fetches too little whenever the connection fluctuates. Compressed bytes are
 *   handled separately: as much of the range as the byte layer holds around the play head is fetched as soon as a
 *   source is selected (`transfer/fill.ts fillWhole`), without waiting for playback.
 * How playback uses them (it waits for a frame that is on its way, never skipping it, as a DCC does) is written once,
 * where it is done: model/timelineMath.ts tick / playOn and editor/Timeline.tsx usePlayback. */

import { nearest } from "../model/timelineMath";

const BEHIND = 4;

/** The wanted frames, most wanted first, wrapping around the ends because playback wraps around the ends.
 *
 * The fetch order must match the play order: after the last frame comes the first. If fetching stopped going forward at
 * the last frame, nobody would read the first frames, yet looping playback back to the first frame waits, as in a DCC,
 * for that frame to reach the browser; waiting for a frame nobody reads waits forever, and shows as playback stuck on
 * the last frame. (The timeline has a second guard: a frame due for playback that nobody is reading is not waited for,
 * playback moves on; see frames.ts onItsWay.) */
export function order(frames: number[], frame: number, dir: number, ahead: number, behind = BEHIND): number[] {
  const n = frames.length;
  if (!n) return [];
  // a frame this source does not have (a depth computed every other frame, a cell with fewer frames): its nearest frame,
  // the one the stage shows there (model/timelineMath.ts nearest, the rule for a picture's frames)
  const from = nearest(frames, frame);
  const i = frames.indexOf(from);
  const at = (k: number) => frames[(((i + k) % n) + n) % n];
  const out = [from];
  const seen = new Set([from]); // fewer frames than the window: keeps a frame from being queued twice
  const push = (f: number) => {
    if (seen.has(f)) return;
    seen.add(f);
    out.push(f);
  };
  // past every frame once nothing is new: a tiny frame's share is tens of thousands of frames, and this runs per render
  for (let k = 1, last = Math.min(Math.max(ahead, behind), n); k <= last; k++) {
    if (k <= ahead) push(at(dir * k));
    if (k <= behind) push(at(-dir * k));
  }
  return out;
}

/** How many frames a source may decode ahead and keep: its share of the byte budget (split evenly among `sources`
 * sources) divided by the decoded size of one frame, at least 2. No upper cap (see the top of the file): a 512×288 proxy
 * frame decodes to 0.6 MB, and the budget holds the whole range. */
export function aheadFor(frameBytes: number, budget: number, sources: number): number {
  const share = (budget * 0.8) / Math.max(1, sources);
  if (!(frameBytes > 0)) return 24;  // no frame decoded yet, so the frame size is unknown: assume one second's worth
  return Math.max(2, Math.floor(share / frameBytes));
}

/** All frames of a source, ordered outward from `frame` in the direction of play (wrapping around the ends): the order
 * in which the whole range of compressed bytes is fetched in the background (`transfer/fill.ts fillWhole`). The same
 * logic as `order`, without the window limit. */
export const ordered = (frames: number[], frame: number, dir: number): number[] => order(frames, frame, dir || 1, frames.length, frames.length);
