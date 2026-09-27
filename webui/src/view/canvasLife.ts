import { useEffect } from "react";
import { useThree } from "@react-three/fiber";

/** 画布生命周期相关组件：何时重绘（`Redraw`）与何时释放 WebGL 上下文（`KeepContext`）。
 * 与场景内容无关，因此不放在 Stage3D 中。 */

/** 保证三维视图的 WebGL 上下文不丢失。
 *
 * 浏览器可同时持有的 WebGL 上下文数量有上限（Chrome 约十余个），超出时会静默丢弃最早的上下文，
 * 画面只剩底色（`--window`），且不抛出任何错误。若画布卸载时依赖垃圾回收释放上下文，
 * 在同一标签页内反复切换视图会使上下文数量累积到上限。因此需处理两种情况：
 *
 *   1. 卸载时立即释放：`dispose()` 仅清理 three 自身的资源，使浏览器回收上下文的是
 *      `forceContextLoss()`（WEBGL_lose_context）。不调用它，回收时机不确定。
 *   2. 丢失后自动恢复：three 会调用 `preventDefault()` 并在 `webglcontextrestored` 中重建资源
 *      （three.module.js onContextLost / onContextRestore），但不会重绘。
 *      本画布使用 `frameloop="demand"`，无请求时不绘制，因此恢复后必须调用一次 invalidate，
 *      否则上下文虽已恢复，画面仍为空。
 */
export function KeepContext() {
  const gl = useThree((s) => s.gl);
  const invalidate = useThree((s) => s.invalidate);
  useEffect(() => {
    const el = gl.domElement;
    const back = () => invalidate(); // demand 模式不会自动重绘，恢复后需主动请求一次
    el.addEventListener("webglcontextrestored", back);
    return () => {
      el.removeEventListener("webglcontextrestored", back);
      gl.dispose();
      gl.forceContextLoss(); // 将上下文交还浏览器
    };
  }, [gl, invalidate]);
  return null;
}

export function Redraw() {
  const invalidate = useThree((s) => s.invalidate);
  const size = useThree((s) => s.size);
  const dpr = useThree((s) => s.viewport.dpr);
  useEffect(() => invalidate());
  useEffect(() => invalidate(), [size, dpr, invalidate]);
  return null;
}
