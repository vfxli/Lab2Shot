/** The arithmetic of fetching and playing frames, independent of the page (tests/frameWindow.test.ts runs it in Node):
 *
 * - which frames a player wants now: the current one, then frames ahead in the direction of play and a few behind;
 * - how far ahead to decode: as many frames as the source's share of the byte budget holds. There is no "seconds
 *   ahead while playing" cap: such a window fetches too little whenever the connection fluctuates, the clock does not
 *   wait, skipped frames are dropped and playback never catches up; a proxy frame is only 512 on its long edge, so the
 *   budget holds the whole range anyway. Compressed bytes are handled separately: the whole range is fetched as soon as
 *   a source is selected (`transfer/frames.ts fillWhole`), without waiting for playback;
 * - playing against a clock at the plate's rate: the frame due now is drawn once decoded; a frame not decoded in
 *   time is not waited for (the last one stays on screen and the tick counts as late); frames the clock passed without
 *   drawing them count as dropped. Nothing stalls: the clock never waits for the network. */

export const BEHIND = 4;

/** The wanted frames, most wanted first, wrapping around the ends because playback wraps around the ends.
 *
 * 取帧顺序必须与播放顺序一致：最后一帧之后即为第一帧。若到达最后一帧后不再向前取，开头几帧将无人读取，
 * 而循环播放回到第一帧时按 DCC 的规则须等待该帧到达浏览器；等待一个无人读取的帧即为死等，
 * 表现为停在最后一帧不动。（时间线另有第二道保护：待播放的帧无人读取时不等待，直接前进，
 * 见 frames.ts onItsWay。） */
export function order(frames: number[], frame: number, dir: number, ahead: number, behind = BEHIND): number[] {
  const n = frames.length;
  const i = frames.indexOf(frame);
  if (i < 0) return [];
  const at = (k: number) => frames[(((i + k) % n) + n) % n];
  const out = [frame];
  const seen = new Set([frame]); // 帧数少于窗口时避免同一帧重复排入
  const push = (f: number) => {
    if (seen.has(f)) return;
    seen.add(f);
    out.push(f);
  };
  for (let k = 1; k <= Math.max(ahead, behind); k++) {
    if (k <= ahead) push(at(dir * k));
    if (k <= behind) push(at(-dir * k));
  }
  return out;
}

/** 一个源可预先解码保留的帧数：其分得的字节预算（由 `sources` 个源均分）除以单帧解码后的大小，
 * 最少 2 帧。不设上限（见文件顶部说明）：一帧 512×288 的代理解码后为 0.6 MB，预算足以容纳整段。
 * 参数中保留 `fps` / `playing` 是为了不改动调用方，二者不影响结果。 */
export function aheadFor(frameBytes: number, budget: number, sources: number, _fps: number, _playing: boolean): number {
  const share = (budget * 0.8) / Math.max(1, sources);
  if (!(frameBytes > 0)) return 24;  // 尚未解码过任何帧，单帧大小未知：先按一秒计算
  return Math.max(2, Math.floor(share / frameBytes));
}

/** 一个源的全部帧，从 `frame` 沿播放方向向外排序（首尾循环）：即后台取回整段压缩字节的顺序
 * （`transfer/frames.ts fillWhole`）。与 `order` 逻辑相同，只是不受窗口限制。 */
export const ordered = (frames: number[], frame: number, dir: number): number[] =>
  (frames.includes(frame) ? order(frames, frame, dir || 1, frames.length, frames.length) : [...frames]);

export interface PlayStats {
  due: number; // frame moments the clock has passed since playing started
  drawn: number; // of these, the number drawn
  dropped: number; // frames the clock passed without ever drawing them
  late: number; // moments at which the due frame was not yet decoded (the previous one stayed)
}

export interface Playhead {
  t0: number; // seconds: when frame moment 0 was due
  shownAt: number; // the frame moment last drawn
  lateAt: number; // the last moment counted late (a moment counts once)
  frame: number; // the frame last drawn
  stats: PlayStats;
}

export const startPlay = (now: number, frame: number): Playhead => ({
  t0: now,
  shownAt: 0,
  lateAt: 0,
  frame,
  stats: { due: 0, drawn: 0, dropped: 0, late: 0 },
});

/** One animation tick at `now` (seconds): the frame to draw (null: keep what is on screen) and the resulting playhead.
 * `move(frame, steps)` applies the loop mode to a frame (timelineMath.advance); `ready(frame)` reports whether it is
 * decoded. */
export function stepPlay(p: Playhead, now: number, fps: number, move: (frame: number, steps: number) => { frame: number; stopped: boolean },
                         ready: (frame: number) => boolean): { draw: number | null; stopped: boolean; head: Playhead } {
  const moment = Math.floor((now - p.t0) * fps + 1e-6);
  if (moment <= p.shownAt || moment <= p.lateAt) return { draw: null, stopped: false, head: p };
  const target = move(p.frame, moment - p.shownAt);
  const stats = { ...p.stats, due: moment };
  if (ready(target.frame)) {
    stats.drawn += 1;
    stats.dropped += Math.max(0, moment - p.shownAt - 1);
    return { draw: target.frame, stopped: target.stopped, head: { ...p, shownAt: moment, lateAt: moment, frame: target.frame, stats } };
  }
  stats.late += 1;
  return { draw: null, stopped: false, head: { ...p, lateAt: moment, stats } };
}
