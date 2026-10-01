/** 按住拖动换帧（时间线的标尺、曲线图）的唯一做法：按下即跳到那一帧，拖动中逐帧跟着走；期间 `scrubbing`（拖过的帧大多
 * 只是经过：视图不取帧，松开后再取停下处的，transfer/frames.ts useFrames 的 hold）。松开、被系统取消、切走窗口都由
 * platform/drag.ts followDrag 一处收尾，scrubbing 跟着收回。 */

import { followDrag } from "../platform/drag";
import { useViewer } from "../state/viewer";

/** `frameAt`：指针的横坐标（clientX）落在哪一帧（已吸附到有的帧上；null：没有帧）。 */
export function startScrub(e: { clientX: number }, frameAt: (clientX: number) => number | null, onFrame: (f: number) => void): void {
  const go = (x: number) => {
    const f = frameAt(x);
    if (f !== null) onFrame(f);
  };
  useViewer.setState({ scrubbing: true });
  go(e.clientX);
  followDrag((m) => go(m.clientX), () => useViewer.setState({ scrubbing: false }));
}
