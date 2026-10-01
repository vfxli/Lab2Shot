/** 三维舞台上的手柄拖动会话：舞台框架给的唯一一种「拖手柄改参数」。「3D 变换」手柄（view/Stage3D.tsx TransformHandle）与
 * 「骨架姿势」的关节手柄（view/skeletonPose.tsx）都只用它，谁都不自己接 TransformControls。
 *
 * 契约：
 * - 按下时记下操纵器的起点；指针离按下处超过页面统一的阈值（platform/drag.ts pressAt / CLICK_SLOP）才算拖动，之前操纵器
 *   原地不动、什么都不报。只点不拖（含按住轴抖几像素）：`onEnd(null)`，调用方什么都不写、不记撤销步。
 * - 报的是这次拖动的增量（model/math3d.ts Increment：世界里的位移、绕枢轴的转动、沿操纵器自己轴的缩放倍数），不是
 *   操纵器现在的 TRS：调用方把增量套到按下时的参数上（model/places.ts draggedPlace、model/skeletonPose.ts draggedRow），
 *   没动的分量原样保留。操纵器的 TRS 表示不了剪切，拿它当关节的世界矩阵换回参数会写出错值。
 * - 操纵器放在 `at` 的原点、朝 `at` 正交化后的轴，缩放为 1；拖动中不跟着 `at` 挪（否则每次重画都把它放回起点），
 *   松手后回到 `at`。
 * - 按下操纵器的那次点击不算舞台的拾取（StageState.claimedAt，view/camera3d.tsx Picker 读）：点一下轴不会清掉选中。
 * - 写不了（ui/writeLock.ts：只读标签页，或所在弹窗的参数不适用）不出操纵器：手柄只画、拖不动；拖到一半变成写不了
 *   （或操纵器卸下）时这次会话照样结束：`onEnd(null)`，调用方丢掉拖动中画的样子。 */

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as THREE from "three";
import { TransformControls } from "@react-three/drei";
import { draggedFactor, frameOf, type Increment, type M4 } from "../model/math3d";
import { useStage } from "./stageState";
import { pressAt } from "../platform/drag";
import { useWriteLock } from "../ui/writeLock";

export type DragMode = "translate" | "rotate" | "scale";

export function DragGizmo({ at, mode, size, onDrag, onEnd, uniform = false }: {
  at: M4; // 枢轴：位置与朝向（列主序 4x4）
  mode: DragMode;
  // 写回的缩放只有一个数（model/places.ts 的 scale）：拖哪个轴都是等比缩放，操纵器画出来的与写回的是同一个倍数
  uniform?: boolean;
  size: number;
  onDrag: (d: Increment) => void; // 拖动中：画出来的样子
  onEnd: (d: Increment | null) => void; // 松手：这次的增量；null 为没动过
}) {
  const stage = useStage();
  const locked = !!useWriteLock(); // 只读标签页，或弹窗里参数不适用（ui/writeLock.ts）
  const [obj, setObj] = useState<THREE.Group | null>(null);
  const press = useRef<{ p: THREE.Vector3; q: THREE.Quaternion; at: ReturnType<typeof pressAt> | null } | null>(null);
  // the pointer, seen before three's TransformControls sees it (window, capture): where a press began, where it is now
  const pointer = useRef<{ clientX: number; clientY: number } | null>(null);
  useEffect(() => {
    const down = (e: PointerEvent) => void (pointer.current = { clientX: e.clientX, clientY: e.clientY });
    const move = (e: PointerEvent) => void press.current?.at?.see(e);
    window.addEventListener("pointerdown", down, true);
    window.addEventListener("pointermove", move, true);
    return () => {
      window.removeEventListener("pointerdown", down, true);
      window.removeEventListener("pointermove", move, true);
    };
  }, []);
  // a drag cut off before its release (the page became unable to write mid-drag: the manipulator goes; or the stage
  // went): the session still ends, with nothing written, so the caller drops what it was drawing for the drag
  const endRef = useRef(onEnd);
  endRef.current = onEnd;
  const cut = () => {
    if (!press.current) return;
    press.current = null;
    endRef.current(null);
  };
  useEffect(() => { if (locked) cut(); }, [locked]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => cut, []); // eslint-disable-line react-hooks/exhaustive-deps
  const place = (g: THREE.Group) => {
    g.position.set(at[12], at[13], at[14]);
    g.quaternion.setFromRotationMatrix(new THREE.Matrix4().fromArray(frameOf(at)));
    g.scale.set(1, 1, 1);
    g.updateMatrix();
  };
  useLayoutEffect(() => {
    if (obj && !press.current) place(obj);
  }, [obj, at]); // eslint-disable-line react-hooks/exhaustive-deps
  const increment = (g: THREE.Group, from: { p: THREE.Vector3; q: THREE.Quaternion }): Increment | null => {
    const move = g.position.clone().sub(from.p);
    const dq = g.quaternion.clone().multiply(from.q.clone().invert());
    const grow = g.scale;
    const still = move.length() < 1e-9 && Math.abs(Math.abs(dq.w) - 1) < 1e-12 && Math.abs(grow.x - 1) + Math.abs(grow.y - 1) + Math.abs(grow.z - 1) < 1e-9;
    return still ? null : { move: move.toArray(), turn: new THREE.Matrix4().makeRotationFromQuaternion(dq).toArray(), grow: grow.toArray() };
  };
  if (locked) return null;
  return (
    <>
      <group ref={setObj} />
      {obj && (
        <TransformControls object={obj} mode={mode} size={size}
          onMouseDown={() => {
            stage.claimedAt = performance.now();
            press.current = { p: obj.position.clone(), q: obj.quaternion.clone(), at: pointer.current && pressAt(pointer.current) };
          }}
          onObjectChange={() => {
            const from = press.current;
            if (!from) return;
            // not a drag yet: the manipulator stays where it was pressed
            if (from.at && !from.at.dragged) return void (obj.position.copy(from.p), obj.quaternion.copy(from.q), obj.scale.set(1, 1, 1));
            if (uniform) {
              obj.scale.setScalar(draggedFactor(obj.scale.toArray()));
            }
            const d = increment(obj, from);
            if (d) onDrag(d);
          }}
          onMouseUp={() => {
            const from = press.current;
            press.current = null;
            const d = from && (!from.at || from.at.dragged) ? increment(obj, from) : null;
            place(obj);
            onEnd(d);
          }} />
      )}
    </>
  );
}
