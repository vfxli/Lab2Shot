/** 手柄种类「骨架姿势」（HandleDef kind "skeleton_pose"，lab2shot retarget 的 motion_pose / target_pose）在三维舞台上的
 * 画法与操作：画这一侧骨架（线 + 关节点），点关节选中，选中的关节出变换手柄（与「3D 变换」同一个拖动会话），拖动
 * 时就地按 FK 重算整副骨架，松手把这次拖动的增量写成那个关节的一行（view/dragGizmo.tsx、model/skeletonPose.ts
 * draggedRow；只点不拖不写）；面板（SkeletonPosePanel）给数值框、镜像、单关节 / 全部
 * 复位。任何声明了这种手柄的节点都走这里；写回经舞台给的 `write`（view/stageLayers.tsx PoseWrite：写节点参数、记一步撤销）。
 * 手柄沿关节自己的局部轴（正交化后；DragGizmo space="local"）：与修正的约定「局部 @ T·R·S」同一套轴。
 * 双骨架编辑（view/rigPair.tsx）的「编辑姿态」用这里的数值编辑（PoseJointEditor），同一套行与镜像、复位。
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
import { t } from "../i18n/t";
import { listSep } from "../i18n/words";
import { tipAttrs, tipOf } from "../platform/tips";

/** 选中的关节（按舞台的相机槽分开：主视图一份，弹窗里每个舞台各一份）。 */
interface Selection {
  by: Record<string, { handle: number; joint: string } | null>;
  set: (slot: string, pick: { handle: number; joint: string } | null) => void;
}
export const usePoseSelection = create<Selection>((set) => ({
  by: {},
  set: (slot, pick) => set((s) => ({ by: { ...s.by, [slot]: pick } })),
}));
// 骨架树里悬停的关节：视图里临时高亮（变颜色）并出手柄，方便在重叠的骨点里选（不必先点中骨点）。瞬时，移开即清。
export const usePoseHover = create<{ joint: string | null; set: (j: string | null) => void }>((set) => ({
  joint: null,
  set: (joint) => set({ joint }),
}));
// the pick belongs to its slot's life: a dialog stage that goes takes its picked joint with it (view/stageState.ts endSlot)
onSlotEnd((slot) => usePoseSelection.setState((s) => { const by = { ...s.by }; delete by[slot]; return { by }; }));

const GHOST = "#b98cff"; // 不可操作的一侧：半透明、另一种颜色（与可操作一侧的骨骼色区分）
const PICKED = "#0a84ff";

/** 一副骨架的画法与操作。`value` 是参数值本身：按它的引用读成行，播放时舞台每帧重画，姿势没变就什么都
 * 不重算。`write(rows)`：松手写一步（拖动中的样子只在层里画）。 */
export function SkeletonPoseLayer({ data, index, handle, value, write, ghost, mode, slot, o }: {
  data: SkeletonPoseData;
  index: number;
  handle: HandleDef;
  value: unknown;
  write: (rows: PoseRow[]) => void;
  ghost: boolean;
  mode: DragMode;
  slot: string;
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
  // 骨架树悬停的关节：视图里临时高亮并出手柄，不必先点中骨点
  const hovered = usePoseHover((s) => s.joint);
  const hoverIdx = !ghost && hovered ? data.names.indexOf(hovered) : -1;

  // 整副骨架是舞台的一个可拾取物（可框显）：点中时离点击最近的关节是选中的那个（不可操作的一侧只框显、点不中）
  const pickable = useMemo((): Pickable => {
    const nearest = (r: PickRay): { i: number; depth: number } | null => {
      let best: { i: number; depth: number; off: number } | null = null;
      const v = new THREE.Vector3();
      for (let i = 0; i < data.names.length; i++) {
        const s = project(v.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]), r);
        const off = s ? Math.hypot(s.x - r.px.x, s.y - r.px.y) : Infinity;
        if (s && off < 15 && (!best || off < best.off)) best = { i, depth: s.depth, off };
      }
      return best;
    };
    return {
      label: handle.params.pose,
      bounds: () => (points.length ? new THREE.Box3().setFromArray(points).expandByScalar(0.5) : null),
      hit: (r) => (ghost ? null : nearest(r)?.depth ?? null),
      order: ORDER.overlayBalls, // 关节优先于场景骨架（overlayBones）：透传节点（rest_pose / fix_joints）的手柄骨架与场景骨架重叠，同层则点骨点被场景骨架抢走、选中整体
      part: (r) => { const n = nearest(r); return n ? data.names[n.i] : null; },
      choose: (joint) => usePoseSelection.getState().set(slot, joint ? { handle: index, joint } : null),
    };
  }, [data.names, points, ghost, index, handle.params.pose, slot]);
  usePickable(`pose/${index}`, pickable);
  const active = hoverIdx >= 0 ? hoverIdx : sel;
  return (
    <>
      {/* 与场景里的骨架同一个画法（view/elements3d.tsx BoneFigure）：可操作的一侧叠在最上面，另一侧半透明、另一种颜色 */}
      <BoneFigure points={points} parents={data.parents} names={data.names} color={ghost ? GHOST : o.boneColor} o={o} opacity={ghost ? 0.45 : 0.95} overlay={!ghost} highlight={active} highlightColor={PICKED} />
      {active >= 0 && !stuckWhy(data, active, rows) && (
        <DragGizmo at={world[active]} mode={mode} size={0.7} space="local"
          onDrag={(d) => { const row = draggedRow(data, active, rows, d); if (row) setLive(withRow(rows, row)); }}
          onEnd={(d) => {
            setLive(null);
            const row = d && draggedRow(data, active, rows, d);
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

/** 骨架树：所有关节按层级列出，悬停联动视图（高亮 + 出手柄），点击选中。解决骨点重叠时点不中的问题：从列表里选关节。 */
function PoseTree({ data, slot, index }: { data: SkeletonPoseData; slot: string; index: number }) {
  const setPick = usePoseSelection((s) => s.set);
  const picked = usePoseSelection((s) => s.by[slot]);
  const hovered = usePoseHover((s) => s.joint);
  const setHover = usePoseHover((s) => s.set);
  const [expanded, setExpanded] = useState<Set<number>>(() => new Set());
  const order = useMemo(() => {
    const kids = new Map<number, number[]>();
    data.parents.forEach((p, i) => { if (p >= 0) { if (!kids.has(p)) kids.set(p, []); kids.get(p)!.push(i); } });
    const out: { i: number; depth: number }[] = [];
    const visit = (i: number, depth: number) => { out.push({ i, depth }); (kids.get(i) ?? []).forEach((c) => visit(c, depth + 1)); };
    data.parents.forEach((p, i) => { if (p < 0) visit(i, 0); });
    return out;
  }, [data.parents]);
  const hasKids = useMemo(() => {
    const m = new Set<number>();
    data.parents.forEach((p) => { if (p >= 0) m.add(p); });
    return m;
  }, [data.parents]);
  // 默认全部折叠：只显示根关节，展开的关节才露出它的子关节
  const visible = order.filter(({ i, depth }) => {
    if (depth === 0) return true;
    let p = data.parents[i];
    while (p >= 0) { if (!expanded.has(p)) return false; p = data.parents[p]; }
    return true;
  });
  const toggle = (i: number) => setExpanded((prev) => { const s = new Set(prev); if (s.has(i)) s.delete(i); else s.add(i); return s; });
  return (
    <div className="pose-tree">
      {visible.map(({ i, depth }) => {
        const name = data.names[i];
        const on = picked?.handle === index && picked.joint === name;
        const hov = hovered === name;
        const open = expanded.has(i);
        return (
          <div key={name} className={`pose-tree-row${on ? " on" : ""}${hov ? " hov" : ""}`}
            style={{ paddingLeft: 6 + depth * 14, color: on ? PICKED : undefined, fontWeight: on ? 600 : undefined }}
            onMouseEnter={() => setHover(name)}
            onMouseLeave={() => setHover(null)}
            onClick={() => setPick(slot, { handle: index, joint: name })}>
            <span className={`pose-tree-caret${hasKids.has(i) ? "" : " leaf"}`}
              onClick={hasKids.has(i) ? (e) => { e.stopPropagation(); toggle(i); } : undefined}>{hasKids.has(i) ? (open ? "▾" : "▸") : ""}</span>
            {name}
          </div>
        );
      })}
    </div>
  );
}

/** 选中关节的数值：平移（cm）、转角（度，XYZ）、缩放；镜像到对侧、复位这个关节、全部复位；关节树。 */
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
  const why = useWriteLock(); // 写不了（只读标签页、参数不适用）：数值能看，改动的按钮与数值框置灰，写明原因
  const joint = picked?.handle === index ? picked.joint : null;
  return (
    <div className="pose-panel">
      <div className="pose-head">
        <b>{label}</b>
        <span className="pose-dim">{t("ui.rig.pose_changed", { pose: data.pose, count: rows.length })}</span>
        <span style={{ flex: 1 }} />
        <button className="btn ghost" disabled={!!why || !rows.length} onClick={() => write([])}>{t("ui.rig.reset_all")}</button>
      </div>
      <PoseTree data={data} slot={slot} index={index} />
      {joint ? (
        <PoseJointEditor data={data} rows={rows} write={write} joint={joint} onDeselect={() => setPick(slot, null)} />
      ) : (
        <div className="pose-dim">{t("ui.rig.pose_pick_hint")}</div>
      )}
      {why && <div className="pose-warn">{why}</div>}
      {data.unknown.length > 0 && <div className="pose-warn">{t("ui.rig.pose_unknown", { joints: data.unknown.join(listSep()) })}</div>}
    </div>
  );
}

/** 一个关节的修正：平移、转角、缩放三行数值框，镜像到对侧、复位、取消选中。「骨架姿势」面板与双骨架编辑的「编辑姿态」
 * 共用（同一种行、同一套镜像与复位，model/skeletonPose.ts）。 */
export function PoseJointEditor({ data, rows, write, joint, onDeselect }: {
  data: Pick<SkeletonPoseData, "names" | "parents" | "before" | "rotation" | "mirror" | "mirror_plane" | "units">;
  rows: PoseRow[];
  write: (rows: PoseRow[]) => void;
  joint: string;
  onDeselect: () => void;
}) {
  const why = useWriteLock();
  const locked = !!why;
  const row = rows.find((r) => r.joint === joint) ?? identityRow(joint);
  const j = data.names.indexOf(joint);
  const pair = j >= 0 ? data.mirror.find(([a, b]) => a === j || b === j) : undefined;
  const otherAt = pair ? (pair[0] === j ? pair[1] : pair[0]) : -1;
  const other = otherAt >= 0 && data.mirror_plane ? data.names[otherAt] : null;
  // 这个关节或对侧关节的轴架退化（缩放为 0）：拖不了、镜像不了，说明原因（model/skeletonPose.ts stuckWhy）
  const stuck = j >= 0 ? stuckWhy(data, j, rows) || (otherAt >= 0 ? stuckWhy(data, otherAt, rows) : "") : "";
  const mirror = other && data.mirror_plane && !stuck ? mirroredRow(data, j, otherAt, data.mirror_plane, rows) : null;
  const mirrored = mirror && "row" in mirror ? mirror.row : null;
  const mirrorWhy = mirror && "why" in mirror ? mirror.why : "";
  const edit = (key: "translate" | "rotate" | "scale", k: number, v: number) => {
    const next = { ...row, [key]: row[key].map((x, i) => (i === k ? v : x)) } as PoseRow;
    write(tidy(withRow(rows, next)));
  };
  return (
    <>
      <div className="pose-joint">
        <b data-user-data>{joint}</b>
        <span style={{ flex: 1 }} />
        <button className="btn ghost" disabled={locked || !mirrored} {...tipAttrs(tipOf("disabled", locked ? why : stuck ? stuck : mirrorWhy ? mirrorWhy : other ? undefined : !data.mirror_plane ? t("ui.rig.no_mirror_plane") : t("ui.rig.no_mirror_joint")))}
          onClick={() => mirrored && write(tidy(withRow(rows, mirrored)))}>{t("ui.rig.mirror")}</button>
        <button className="btn ghost" disabled={locked} {...tipAttrs(tipOf("disabled", locked ? why : undefined))} onClick={() => write(tidy(rows.filter((r) => r.joint !== joint)))}>{t("ui.common.reset")}</button>
        <button className="btn ghost" onClick={onDeselect}>{t("ui.rig.deselect")}</button>
      </div>
      {(["translate", "rotate", "scale"] as const).map((key) => (
        <div key={key} className="pose-row">
          <span className="pose-dim">{key === "translate" ? t("ui.rig.translate_unit", { unit: data.units.translate }) : key === "rotate" ? t("ui.rig.rotate_unit", { unit: data.units.rotate, order: data.rotation }) : t("ui.rig.scale")}</span>
          {AXES.map((a, k) => (
            <label key={a} className="pose-num">
              {a}
              <Num value={row[key][k]} digits={key === "scale" ? 4 : 3} disabled={locked} {...RANGE[key]} onChange={(v) => edit(key, k, v)} />
            </label>
          ))}
        </div>
      ))}
      {!locked && stuck && <div className="pose-warn">{stuck}</div>}
    </>
  );
}
