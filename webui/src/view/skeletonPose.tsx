/** 手柄种类「骨架姿势」（HandleDef kind "skeleton_pose"，lab2shot core.retarget 的 motion_pose / target_pose）在三维舞台上的
 * 画法与操作：画这一侧骨架（线 + 关节点），点关节选中，选中的关节出变换手柄（与「3D 变换」同一个拖动会话），拖动
 * 时就地按 FK 重算整副骨架，松手把这次拖动的增量写成那个关节的一行（view/dragGizmo.tsx、model/skeletonPose.ts
 * draggedRow；只点不拖不写）；面板（SkeletonPosePanel）给数值框、镜像、单关节 / 全部
 * 复位。任何声明了这种手柄的节点都走这里；写回经舞台给的 `write`（view/stageLayers.tsx PoseWrite：主视图写节点参数、记一步撤销；
 * 弹窗里的舞台写弹窗草稿）。
 *
 * 数据：状态回复的 handle_data（api/status.ts SkeletonPoseData）给这一侧骨架的关节、父子、基准姿势的局部矩阵。
 * 显示的永远是「基准局部 + 当前参数（拖动中为拖出来的那一版）」经 FK 算出的世界，不等服务器。不可操作（`ghost`）的一侧半透明、另一种颜色、不能点。 */

import { useMemo, useState } from "react";
import { create } from "zustand";
import * as THREE from "three";
import type { HandleDef, SkeletonPoseData } from "../api";
import { draggedRow, forward, identityRow, mirroredRow, rowsOf, stuckWhy, tidy, withRow, type PoseRow } from "../model/skeletonPose";
import { DragGizmo, type DragMode } from "./dragGizmo";
import { Num } from "../ui/controls";
import { useWriteLock } from "../ui/writeLock";
import { BoneFigure } from "./elements3d";
import { onSlotEnd, project, usePickable, type PickRay, type Pickable } from "./stageState";
import type { ViewOptions } from "../model/viewOptions";
import { ORDER } from "./drawOrder";

/** 选中的关节（按舞台的相机槽分开：主视图一份，弹窗里每个舞台各一份）。 */
interface Selection {
  by: Record<string, { handle: number; joint: string } | null>;
  set: (slot: string, pick: { handle: number; joint: string } | null) => void;
}
export const usePoseSelection = create<Selection>((set) => ({
  by: {},
  set: (slot, pick) => set((s) => ({ by: { ...s.by, [slot]: pick } })),
}));
// the pick belongs to its slot's life: a dialog stage that goes takes its picked joint with it (view/stageState.ts endSlot)
onSlotEnd((slot) => usePoseSelection.setState((s) => { const by = { ...s.by }; delete by[slot]; return { by }; }));

const GHOST = "#b98cff"; // 不可操作的一侧：半透明、另一种颜色（与可操作一侧的骨骼色区分）
const PICKED = "#0a84ff";

/** 一副骨架的画法与操作。`value` 是参数值本身（弹窗里是草稿）：按它的引用读成行，播放时舞台每帧重画，姿势没变就什么都
 * 不重算。`write(rows)`：松手写一步（拖动中的样子只在层里画）。`layer` 分开同一个手柄在舞台上的两份（可操作的与参考显示的），
 * 各登记各的拾取物。 */
export function SkeletonPoseLayer({ data, index, handle, value, write, ghost, mode, slot, layer, o }: {
  data: SkeletonPoseData;
  index: number;
  handle: HandleDef;
  value: unknown;
  write: (rows: PoseRow[]) => void;
  ghost: boolean;
  mode: DragMode;
  slot: string;
  layer: "edit" | "reference";
  o: ViewOptions;
}) {
  const rows = useMemo(() => rowsOf(value), [value]);
  const [live, setLive] = useState<PoseRow[] | null>(null);
  const shown = live ?? rows;
  const { world } = useMemo(() => forward(data, shown), [data, shown]);
  const points = useMemo(() => {
    const out = new Float32Array(world.length * 3);
    world.forEach((m, i) => out.set([m[12], m[13], m[14]], i * 3));
    return out;
  }, [world]);
  const picked = usePoseSelection((s) => s.by[slot]);
  const sel = !ghost && picked?.handle === index ? data.names.indexOf(picked.joint) : -1;

  // 整副骨架是舞台的一个可拾取物（可框显）：点中时离点击最近的关节是选中的那个（不可操作的一侧只框显、点不中）
  const pickable = useMemo((): Pickable => {
    const nearest = (r: PickRay): { i: number; depth: number } | null => {
      let best: { i: number; depth: number; off: number } | null = null;
      const v = new THREE.Vector3();
      for (let i = 0; i < data.names.length; i++) {
        const s = project(v.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]), r);
        const off = s ? Math.hypot(s.x - r.px.x, s.y - r.px.y) : Infinity;
        if (s && off < 9 && (!best || off < best.off)) best = { i, depth: s.depth, off };
      }
      return best;
    };
    return {
      label: handle.params.pose,
      bounds: () => (points.length ? new THREE.Box3().setFromArray(points).expandByScalar(0.5) : null),
      hit: (r) => (ghost ? null : nearest(r)?.depth ?? null),
      order: ORDER.overlayBones, // the side that can be worked is drawn over everything (BoneFigure overlay below)
      part: (r) => { const n = nearest(r); return n ? data.names[n.i] : null; },
      choose: (joint) => usePoseSelection.getState().set(slot, joint ? { handle: index, joint } : null),
    };
  }, [data.names, points, ghost, index, handle.params.pose, slot]);
  usePickable(`${layer}-pose/${index}`, pickable);
  return (
    <>
      {/* 与场景里的骨架同一个画法（view/elements3d.tsx BoneFigure）：可操作的一侧叠在最上面，另一侧半透明、另一种颜色 */}
      <BoneFigure points={points} parents={data.parents} names={data.names} color={ghost ? GHOST : o.boneColor} o={o} opacity={ghost ? 0.45 : 0.95} overlay={!ghost} highlight={sel} highlightColor={PICKED} />
      {sel >= 0 && !stuckWhy(data, sel, rows) && (
        <DragGizmo at={world[sel]} mode={mode} size={0.7}
          onDrag={(d) => { const row = draggedRow(data, sel, rows, d); if (row) setLive(withRow(rows, row)); }}
          onEnd={(d) => {
            setLive(null);
            const row = d && draggedRow(data, sel, rows, d);
            if (row) write(tidy(withRow(rows, row)));
          }} />
      )}
    </>
  );
}

// ------------------------------------------------------------------ 面板（画布外：数值框、镜像、复位）

const AXES = ["X", "Y", "Z"] as const;
/** 数值框的范围；缩放须 > 0（model/skeletonPose.ts validRow），填 0 退回原值 */
const RANGE = {
  translate: { min: -1e6, max: 1e6 },
  rotate: { min: -36000, max: 36000 },
  scale: { min: 0, max: 1000, openMin: true },
};

/** 选中关节的数值：平移（cm）、转角（度，XYZ）、缩放；镜像到对侧、复位这个关节、全部复位。 */
export function SkeletonPosePanel({ data, index, rows, write, slot, label }: {
  data: SkeletonPoseData;
  index: number;
  rows: PoseRow[];
  write: (rows: PoseRow[]) => void;
  slot: string;
  label: string;
}) {
  const picked = usePoseSelection((s) => s.by[slot]);
  const setPick = usePoseSelection((s) => s.set);
  const why = useWriteLock(); // 写不了（只读标签页、弹窗里参数不适用）：数值能看，改动的按钮与数值框置灰，写明原因
  const locked = !!why;
  const joint = picked?.handle === index ? picked.joint : null;
  const row = joint ? rows.find((r) => r.joint === joint) ?? identityRow(joint) : null;
  const j = joint ? data.names.indexOf(joint) : -1;
  const pair = j >= 0 ? data.mirror.find(([a, b]) => a === j || b === j) : undefined;
  const otherAt = pair ? (pair[0] === j ? pair[1] : pair[0]) : -1;
  const other = otherAt >= 0 && data.mirror_plane ? data.names[otherAt] : null;
  // 这个关节或对侧关节的轴架退化（缩放为 0）：拖不了、镜像不了，说明原因（model/skeletonPose.ts stuckWhy）
  const stuck = j >= 0 ? stuckWhy(data, j, rows) || (otherAt >= 0 ? stuckWhy(data, otherAt, rows) : "") : "";
  const mirror = other && data.mirror_plane && !stuck ? mirroredRow(data, j, otherAt, data.mirror_plane, rows) : null;
  const mirrored = mirror && "row" in mirror ? mirror.row : null;
  const mirrorWhy = mirror && "why" in mirror ? mirror.why : "";
  const edit = (key: "translate" | "rotate" | "scale", k: number, v: number) => {
    if (!row) return;
    const next = { ...row, [key]: row[key].map((x, i) => (i === k ? v : x)) } as PoseRow;
    write(tidy(withRow(rows, next)));
  };
  return (
    <div className="pose-panel">
      <div className="pose-head">
        <b>{label}</b>
        <span className="pose-dim">{data.pose} · 改了 {rows.length} 个关节</span>
        <span style={{ flex: 1 }} />
        <button className="btn ghost" disabled={locked || !rows.length} data-tip="这一侧所有关节都回到基准姿势" onClick={() => write([])}>全部复位</button>
      </div>
      {row ? (
        <>
          <div className="pose-joint">
            <b data-user-data>{joint}</b>
            <span style={{ flex: 1 }} />
            <button className="btn ghost" disabled={locked || !mirrored} data-tip={stuck ? stuck : mirrorWhy ? mirrorWhy : other ? `把这个关节的修正照人物左右对称面镜像到「${other}」` : !data.mirror_plane ? "找不到人物的左右对称面（没有大腿或躯干）：不能镜像" : "没有对侧关节（不在成对的部位里）"}
              onClick={() => mirrored && write(tidy(withRow(rows, mirrored)))}>镜像到对侧</button>
            <button className="btn ghost" disabled={locked} data-tip="这个关节回到基准姿势" onClick={() => write(tidy(rows.filter((r) => r.joint !== joint)))}>复位</button>
            <button className="btn ghost" data-tip="不选关节" onClick={() => setPick(slot, null)}>取消选中</button>
          </div>
          {(["translate", "rotate", "scale"] as const).map((key) => (
            <div key={key} className="pose-row">
              <span className="pose-dim">{key === "translate" ? `平移（${data.units.translate}）` : key === "rotate" ? `旋转（${data.units.rotate}，${data.rotation}）` : "缩放"}</span>
              {AXES.map((a, k) => (
                <label key={a} className="pose-num">
                  {a}
                  <Num value={row[key][k]} digits={key === "scale" ? 4 : 3} disabled={locked} {...RANGE[key]} onChange={(v) => edit(key, k, v)} />
                </label>
              ))}
            </div>
          ))}
        </>
      ) : (
        <div className="pose-dim">在骨架上点一个关节：拖它的手柄，或在这里填数</div>
      )}
      {locked && <div className="pose-warn">{why}</div>}
      {!locked && stuck && <div className="pose-warn">{stuck}</div>}
      {data.unknown.length > 0 && <div className="pose-warn">参数里有骨架上没有的关节（计算时略过）：{data.unknown.join("、")}</div>}
    </div>
  );
}
