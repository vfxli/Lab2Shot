/** 播放：在播放范围内逐帧移动、跳到两端。
 * 当前帧属于视图（state/viewer.ts），播放范围属于文档外观（state/look.ts），两份状态分别管理。 */

import { playRange, stepIn } from "../model/timelineMath";
import { useLook } from "../state/look";
import { framesShown, useViewer } from "../state/viewer";

/** 在播放范围内走一步，到头后绕回。 */
export function step(d: number): void {
  const viewer = useViewer.getState();
  const range = playRange(framesShown(), useLook.getState().playback);
  if (range) useViewer.setState({ frame: stepIn(framesShown(), range, viewer.frame, d) });
}

/** 跳到播放范围的开头或结尾。 */
export function jump(end: 0 | 1): void {
  const range = playRange(framesShown(), useLook.getState().playback);
  if (!range) return;
  const at = framesShown().filter((f) => f >= range[0] && f <= range[1]).at(end ? -1 : 0);
  if (at !== undefined) useViewer.setState({ frame: at });
}
