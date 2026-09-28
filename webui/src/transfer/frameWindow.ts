/** The arithmetic of fetching and playing frames, independent of the page:
 *
 * - which frames a player wants now: the current one, then frames ahead in the direction of play and a few behind;
 * - how far ahead to decode: as many frames as the source's share of the byte budget holds. There is no "seconds
 *   ahead while playing" cap: such a window fetches too little whenever the connection fluctuates, the clock does not
 *   wait, skipped frames are dropped and playback never catches up; a proxy frame is only 512 on its long edge, so the
 *   budget holds the whole range anyway. Compressed bytes are handled separately: the whole range is fetched as soon as
 *   a source is selected (`transfer/fill.ts fillWhole`), without waiting for playback;
 * - playing against a clock at the plate's rate: the frame due now is drawn once decoded; a frame not decoded in
 *   time is not waited for (the last one stays on screen and the tick counts as late); frames the clock passed without
 *   drawing them count as dropped. Nothing stalls: the clock never waits for the network. */

const BEHIND = 4;

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
 * 最少 2 帧。不设上限（见文件顶部说明）：一帧 512×288 的代理解码后为 0.6 MB，预算足以容纳整段。 */
export function aheadFor(frameBytes: number, budget: number, sources: number): number {
  const share = (budget * 0.8) / Math.max(1, sources);
  if (!(frameBytes > 0)) return 24;  // 尚未解码过任何帧，单帧大小未知：先按一秒计算
  return Math.max(2, Math.floor(share / frameBytes));
}

/** 一个源的全部帧，从 `frame` 沿播放方向向外排序（首尾循环）：即后台取回整段压缩字节的顺序
 * （`transfer/fill.ts fillWhole`）。与 `order` 逻辑相同，只是不受窗口限制。 */
export const ordered = (frames: number[], frame: number, dir: number): number[] =>
  (frames.includes(frame) ? order(frames, frame, dir || 1, frames.length, frames.length) : [...frames]);
