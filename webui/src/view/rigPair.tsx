/** 手柄种类「rig_pair」（双骨架视图编辑，服务端 nodes/handles.py RigPair）在三维舞台上的画法与操作。任何声明了这种手柄
 * 的节点都走这里（「重定向预处理」、「动作重定向」、骨骼动作家族的「模型骨骼对应」），节点自己不写页面代码。
 *
 * 参数面板上手柄的参数行点「在视图里编辑」（editor/ParamControls.tsx RigPairParam → state/handleView.ts editing），显示
 * 这个节点、视图在三维时进入编辑：
 * - 舞台（RigPairLayer）：两副骨架一前一后（源在后、目标在前，沿世界 Z，「前后距离」是显示选项 rigPairGap，0 = 重叠），
 *   按状态上色（已配对实色、未配对灰、忽略暗红半透明，view/rigTree.tsx RIG_COLORS），配对的骨点之间连线；有蒙皮的
 *   角色按现在的姿势蒙皮，跟着各自的骨架挪开。模型固定骨架的一侧没有位置，不画（只有树）。
 * - 左右两棵树与功能条（RigPairPanels）：模式（编辑关系 / 编辑姿态 / 选忽略）、三个「自动」与各自的状态行、忽略规则；
 *   节点有尺寸参数时（「重定向预处理」）再加一栏「尺寸」：两侧系数、「自动尺寸」与它的状态。舞台按实际用的系数画两侧
 *   （model/rigPair.ts effective.scales、scaledWorld：绕地面原点缩放、脚底落到 y = 0，所见即所算），姿态的行仍在原尺寸里（拖动的位移换回原尺寸再写）。
 *
 * 点骨点（树里或视图里）按模式（useRigPair act）：
 * - 编辑关系：先点一侧、再点另一侧 → 配成先点那个骨点所属的部位（没有部位时弹部位小菜单）；点已配对的骨点或连线 → 解除；
 * - 编辑姿态：选中，出手柄（关节自己的局部轴，view/dragGizmo.tsx space="local"），下方填数、镜像、复位
 *   （view/skeletonPose.tsx PoseJointEditor）；
 * - 选忽略：切换这个骨点连同子孙；Alt 点只切换这一个。
 * 写回都经节点参数（graph/actions.ts setParams：一步撤销），格式即参数自己的格式；规则在 model/rigPair.ts。 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { HandleDef, RigPairData, RigPairSide, SkeletonPoseData } from "../api";
import {
  COLS, autoAllPatch, autoPatch, ruleProblems, autoState, confidenceOf, currentRecord, effective, ignoreRole, jointStates, link, linksOf, mergedOf, partFor, partOf, unsureParts,
  autoName, partnersOf, poseRole, problems, recordOf, ruleNow, scaleOf, suggestedRecord, scaleRole, scaledWorld, shownMask, sideOffsets, toggleIgnored, unlinkJoint,
  unlinkLink, valueOf, withTouched,
  type AutoKind, type Col, type Vec3, type JointState, type Link, type Merged, type RigRoles,
} from "../model/rigPair";
import { draggedRow, forward, rowsOf as poseRowsOf, stuckWhy, tidy, withRow, type PoseRow, type PoseSkeleton } from "../model/skeletonPose";
import type { ViewOptions } from "../model/viewOptions";
import { RANGES } from "../model/viewOptions";
import { useRigPairView, type RigMode } from "../state/rigPairView";
import { usePreferences } from "../state/preferences";
import { useViewCamera, useViewOptions } from "../state/viewer";
import { useShortcut } from "../platform/keys";
import { useWriteLock } from "../ui/writeLock";
import { Button } from "../ui/Button";
import { Select } from "../ui/Select";
import { Num } from "../ui/controls";
import { Menu, type MenuRow } from "../ui/Menu";
import { Slider } from "../ui/displayRows";
import { BoneFigure } from "./elements3d";
import { FatLines } from "./lines3d";
import { DragGizmo, type DragMode } from "./dragGizmo";
import type { Increment } from "../model/math3d";
import { ORDER } from "./drawOrder";
import { project, usePickable, type Insets, type PickRay, type Pickable } from "./stageState";
import { SkinnedPose } from "./stageLayers";
import { PoseJointEditor } from "./skeletonPose";
import { RIG_COLORS, RigTree, stateColor, type ActHow } from "./rigTree";
import { t } from "../i18n/t";
import { useLang } from "../i18n/lang";
import { listSep } from "../i18n/words";
import { tipAttrs, tipOf } from "../platform/tips";

/** 两侧的名字（键；t() 取当前语言）。 */
const SIDE_KEY: Record<Col, string> = { src: "ui.rig.side_src_title", dst: "ui.rig.side_dst_title" };
const SKELETON_KEY: Record<Col, string> = { src: "ui.rig.skeleton_src", dst: "ui.rig.skeleton_dst" };

/** 有位置、能在舞台上画的一侧（模型固定骨架没有）。 */
const posable = (side: RigPairSide): side is RigPairSide & SkeletonPoseData => !side.fixed && !!side.before && !!side.rotation;

/** 一次编辑要的全部：手柄的数据、参数、合并后的对应关系、各侧的状态，以及点骨点 / 点连线 / 选部位的动作。舞台层与
 * 面板在画布内外两棵 React 树里，各按同一份参数建一份（useRigPair，都是轻量的 memo）；点到哪一步在共用的页面状态里
 * （state/rigPairView.ts）。 */
export interface RigPairCtl {
  node: string;
  data: RigPairData;
  roles: RigRoles;
  params: Record<string, unknown> | undefined;
  merged: Merged;
  ignored: Record<Col, string[]>;
  poses: Record<Col, PoseRow[]>;
  scales: Record<Col, number>; // 两侧实际用的尺寸系数（显示按它）
  states: Record<Col, JointState[]>;
  links: Link[];
  lock: string; // 写不了的原因（只读标签页）；"" 能写
  write: (patch: Record<string, unknown>) => void; // 手动改（姿态、忽略的第一次手动改补上 auto_record：model/rigPair.ts withTouched）
  writeAuto: (patch: Record<string, unknown>) => void; // 点「自动」的那一份补丁原样写（它自己带着 auto_record）
  act: (col: Col, joint: string, how: ActHow) => void;
  unlink: (l: Link) => void;
  choosePart: (part: string) => void;
}

/** 舞台给双骨架编辑的：节点、手柄声明、手柄数据、节点参数、写回（一步撤销）。 */
export interface RigPairProps {
  node: string;
  def: HandleDef;
  data: RigPairData;
  params: Record<string, unknown> | undefined;
  write: (patch: Record<string, unknown>) => void;
}

function useRigPair({ node, def, data, params, write: writeParams }: RigPairProps): RigPairCtl {
  const lock = useWriteLock(); // 只读标签页：能看、能选，不写
  const roles = def.params as RigRoles;
  const write = (patch: Record<string, unknown>) => { const out = withTouched(patch, data, roles, params); if (out) writeParams(out); };
  const merged = useMemo(() => mergedOf(roles.mapping ? params?.[roles.mapping] : null, data), [data, roles.mapping, params]);
  // 姿态与忽略按「默认自动」之后实际用的那一份（model/rigPair.ts effective）画、上色、改：与计算同一条规则
  const eff = useMemo(() => effective(data, roles, params), [data, roles, params]);
  const ignored = eff.ignored;
  const poses = useMemo(() => ({ src: poseRowsOf(eff.poses.src), dst: poseRowsOf(eff.poses.dst) }), [eff]);
  const states = useMemo(() => ({ src: jointStates(merged, data, "src", ignored.src), dst: jointStates(merged, data, "dst", ignored.dst) }), [merged, data, ignored]);
  const links = useMemo(() => linksOf(merged, data), [merged, data]);
  const writeMap = (m: Merged) => roles.mapping && write({ [roles.mapping]: valueOf(m, data) });
  const view = useRigPairView.getState;
  const act = (col: Col, joint: string, how: ActHow) => {
    const v = view();
    const side = data[col];
    if (v.mode === "pose") {
      if (posable(side) && roles[poseRole(col)]) v.set({ picked: { col, joint } });
      return;
    }
    if (lock) return;
    if (v.mode === "ignore") {
      const role = roles[ignoreRole(col)];
      if (!role || side.fixed) return;
      // 两侧一起写：默认自动的名单是两侧一套，只写一侧，另一侧的自动结果就丢了
      const patch: Record<string, unknown> = {};
      for (const c of COLS) { const r = roles[ignoreRole(c)]; if (r) patch[r] = c === col ? toggleIgnored(ignored[c], side, joint, how.alt) : [...ignored[c]]; }
      write(patch);
      return;
    }
    // 编辑关系。固定一侧没有部位的关节配不了（那一侧由节点定死）
    if (side.fixed && !partOf(merged, col).has(joint)) return;
    const first = v.first;
    if (first) {
      if (first.col === col) return v.set({ first: first.joint === joint ? null : { col, joint } });
      const second = { col, joint };
      const part = partFor(merged, data, first, second);
      if (!part) return v.set({ menu: { x: how.x, y: how.y, first, second } });
      writeMap(link(merged, data, part, first, second));
      return v.set({ first: null });
    }
    const i = side.names.indexOf(joint);
    if (states[col][i] === "paired") return void writeMap(unlinkJoint(merged, data, col, joint));
    v.set({ first: { col, joint } });
  };
  return {
    node, data, roles, params, merged, ignored, poses, scales: eff.scales, states, links, lock, write, writeAuto: writeParams, act,
    unlink: (l) => { if (!lock) writeMap(unlinkLink(merged, data, l)); },
    choosePart: (part) => {
      const m = view().menu;
      view().set({ menu: null, first: null });
      if (m && !lock) writeMap(link(merged, data, part, m.first, m.second));
    },
  };
}

// ------------------------------------------------------------------ 舞台（画布内）

/** 一侧的世界矩阵挪到它在舞台上的位置（sideOffsets 的平移）。 */
const shifted = (m: number[], t: Vec3): number[] => { const out = m.slice(); out[12] += t[0]; out[13] += t[1]; out[14] += t[2]; return out; };

export function RigPairLayer({ mode, o, ...props }: RigPairProps & { mode: DragMode; o: ViewOptions }) {
  const ctl = useRigPair(props);
  // 每侧的显示位置：锚点水平挪回原点、再前后拉开（model/rigPair.ts sideOffsets）。只影响显示：手柄的增量与平移无关，
  // 拖动换算（draggedRow）照旧在这一侧自己的空间里做
  const { src: ss, dst: ds } = ctl.scales;
  const off = useMemo(() => sideOffsets(ctl.data, o.rigPairGap, { src: ss, dst: ds }), [ctl.data, o.rigPairGap, ss, ds]);
  // 蒙皮按同一个变换画（model/rigPair.ts scaledWorld：绕地面原点缩放、脚底落到 y = 0）；1 不变
  const skinScale = useMemo(() => ({
    src: ss === 1 ? undefined : { s: ss, ground: ctl.data.src.size?.ground ?? 0 },
    dst: ds === 1 ? undefined : { s: ds, ground: ctl.data.dst.size?.ground ?? 0 },
  }), [ss, ds, ctl.data]);
  const editMode = useRigPairView((s) => s.mode);
  const picked = useRigPairView((s) => s.picked);
  // 拖动中：这一侧拖出来的那一版行（只在层里画，松手才写）
  const [live, setLive] = useState<{ col: Col; rows: PoseRow[] } | null>(null);
  const rowsOf = (c: Col) => (live?.col === c ? live.rows : ctl.poses[c]);
  const worlds = {
    src: useWorld(ctl.data.src, rowsOf("src"), ss),
    dst: useWorld(ctl.data.dst, rowsOf("dst"), ds),
  };
  const hidden = useRigPairView((s) => s.hidden);
  const shown = { src: useMemo(() => shownMask(ctl.data.src, hidden.src), [ctl.data.src, hidden.src]), dst: useMemo(() => shownMask(ctl.data.dst, hidden.dst), [ctl.data.dst, hidden.dst]) };
  // 选中的关节出手柄：这一侧有姿势参数、写得了、轴架不退化（model/skeletonPose.ts stuckWhy）
  const gizmo = (() => {
    if (editMode !== "pose" || !picked || ctl.lock) return null;
    const side = ctl.data[picked.col];
    const role = ctl.roles[poseRole(picked.col)];
    const world = worlds[picked.col];
    const j = side.names.indexOf(picked.joint);
    if (!posable(side) || !role || !world || j < 0) return null;
    const rows = ctl.poses[picked.col];
    if (stuckWhy(side, j, rows)) return null;
    const sk: PoseSkeleton = side;
    // 画的是缩放后的：手柄拖出的位移换回原尺寸（行在原尺寸里），转动、缩放倍数不变
    const s = ctl.scales[picked.col];
    const back = (d: Increment): Increment => (s === 1 ? d : { ...d, move: d.move.map((v) => v / s) as typeof d.move });
    return (
      <DragGizmo at={shifted(world[j], off[picked.col])} mode={mode} size={0.7} space="local"
        onDrag={(d) => { const row = draggedRow(sk, j, rows, back(d)); if (row) setLive({ col: picked.col, rows: withRow(rows, row) }); }}
        onEnd={(d) => {
          setLive(null);
          const row = d && draggedRow(sk, j, rows, back(d));
          if (row) ctl.write({ [role]: tidy(withRow(rows, row)) });
        }} />
    );
  })();
  return (
    <>
      {COLS.map((c) => {
        const side = ctl.data[c];
        const world = worlds[c];
        if (!posable(side) || !world) return null;
        return (
          <group key={c}>
            <SkinnedPose data={side} value={rowsOf(c)} o={o} pickKey={`rig-pair-skin/${c}`} offset={off[c]} scale={skinScale[c]} />
            <SideFigure ctl={ctl} col={c} world={world} offset={off[c]} shown={shown[c]} o={o} />
          </group>
        );
      })}
      <Links ctl={ctl} worlds={worlds} off={off} shown={shown} o={o} active={editMode === "map"} />
      {gizmo}
    </>
  );
}

function useWorld(side: RigPairSide, rows: PoseRow[], s: number): number[][] | null {
  return useMemo(() => (posable(side) ? scaledWorld(forward(side, rows).world, side, s) : null), [side, rows, s]);
}

/** 一侧骨架：按状态分两份画（不忽略的实色 / 灰，忽略的暗红半透明），眼睛关掉的不画；可拾取（点中最近的骨点）。 */
function SideFigure({ ctl, col, world, offset, shown, o }: { ctl: RigPairCtl; col: Col; world: number[][]; offset: Vec3; shown: boolean[]; o: ViewOptions }) {
  const side = ctl.data[col];
  const states = ctl.states[col];
  const points = useMemo(() => {
    const out = new Float32Array(world.length * 3);
    world.forEach((m, i) => out.set([m[12] + offset[0], m[13] + offset[1], m[14] + offset[2]], i * 3));
    return out;
  }, [world, offset]);
  const tints = useMemo(() => states.map((s) => stateColor(col, s)), [states, col]);
  const kept = useMemo(() => shown.map((v, i) => v && states[i] !== "ignored"), [shown, states]);
  const skipped = useMemo(() => shown.map((v, i) => v && states[i] === "ignored"), [shown, states]);
  const hover = useRigPairView((s) => (s.hover?.col === col ? s.hover.joint : null));
  const first = useRigPairView((s) => (s.first?.col === col ? s.first.joint : null));
  const picked = useRigPairView((s) => (s.mode === "pose" && s.picked?.col === col ? s.picked.joint : null));
  const active = side.names.indexOf(first ?? picked ?? hover ?? "");
  // 动作按最新的一份（参数一变 ctl 就换）；拾取物只随骨架的位置重建
  const act = useRef(ctl.act);
  act.current = ctl.act;
  const lang = useLang((s) => s.lang); // the pick label is a word
  const pickable = useMemo((): Pickable => {
    const nearest = (r: PickRay): { i: number; depth: number } | null => {
      let best: { i: number; depth: number; off: number } | null = null;
      const v = new THREE.Vector3();
      for (let i = 0; i < side.names.length; i++) {
        if (!shown[i]) continue;
        const s = project(v.set(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]), r);
        const off = s ? Math.hypot(s.x - r.px.x, s.y - r.px.y) : Infinity;
        if (s && off < 15 && (!best || off < best.off)) best = { i, depth: s.depth, off };
      }
      return best;
    };
    return {
      label: t(SKELETON_KEY[col]),
      bounds: () => (points.length ? new THREE.Box3().setFromArray(points).expandByScalar(0.5) : null),
      hit: (r) => nearest(r)?.depth ?? null,
      order: ORDER.overlayBalls,
      part: (r) => { const n = nearest(r); return n ? side.names[n.i] : null; },
      choose: (joint, how) => { if (joint) act.current(col, joint, { alt: how.alt, x: how.x, y: how.y }); },
    };
  }, [side.names, points, shown, col, lang]); // eslint-disable-line react-hooks/exhaustive-deps -- lang: the label
  usePickable(`rig-pair/${col}`, pickable);
  return (
    <>
      <BoneFigure points={points} parents={side.parents} names={side.names} color={RIG_COLORS.unpaired} o={o} opacity={0.95}
        tints={tints} shown={kept} highlight={kept[active] ? active : -1} highlightColor={RIG_COLORS.active} />
      <BoneFigure points={points} parents={side.parents} color={RIG_COLORS.ignored} o={o} opacity={0.35}
        shown={skipped} highlight={skipped[active] ? active : -1} highlightColor={RIG_COLORS.active} />
    </>
  );
}

/** 配对连线：两端都画着时才画；编辑关系时可点（点中最近的一条 → 解除）。 */
function Links({ ctl, worlds, off, shown, o, active }: {
  ctl: RigPairCtl; worlds: Record<Col, number[][] | null>; off: Record<Col, Vec3>; shown: Record<Col, boolean[]>; o: ViewOptions; active: boolean;
}) {
  const drawn = useMemo(() => {
    const src = worlds.src, dst = worlds.dst;
    if (!src || !dst) return { links: [] as Link[], segments: new Float32Array(0) };
    const at = { src: new Map(ctl.data.src.names.map((n, i) => [n, i] as [string, number])), dst: new Map(ctl.data.dst.names.map((n, i) => [n, i] as [string, number])) };
    const links = ctl.links.filter((l) => {
      const a = at.src.get(l.src), b = at.dst.get(l.dst);
      return a !== undefined && b !== undefined && shown.src[a] && shown.dst[b];
    });
    const segments = new Float32Array(links.length * 6);
    links.forEach((l, k) => {
      const a = src[at.src.get(l.src)!], b = dst[at.dst.get(l.dst)!];
      const p = off.src, q = off.dst;
      segments.set([a[12] + p[0], a[13] + p[1], a[14] + p[2], b[12] + q[0], b[13] + q[1], b[14] + q[2]], k * 6);
    });
    return { links, segments };
  }, [ctl.links, ctl.data, worlds.src, worlds.dst, off, shown.src, shown.dst]);
  const unlink = useRef(ctl.unlink);
  unlink.current = ctl.unlink;
  const lang = useLang((s) => s.lang); // the pick label is a word
  const pickable = useMemo((): Pickable | null => {
    if (!active || !drawn.links.length) return null;
    const nearest = (r: PickRay): { k: number; depth: number } | null => {
      let best: { k: number; depth: number; off: number } | null = null;
      const a = new THREE.Vector3(), b = new THREE.Vector3();
      for (let k = 0; k < drawn.links.length; k++) {
        const s = drawn.segments;
        const sa = project(a.set(s[k * 6], s[k * 6 + 1], s[k * 6 + 2]), r);
        const sb = sa && project(b.set(s[k * 6 + 3], s[k * 6 + 4], s[k * 6 + 5]), r);
        if (!sa || !sb) continue;
        const dx = sb.x - sa.x, dy = sb.y - sa.y, len2 = dx * dx + dy * dy;
        const t = len2 > 0 ? Math.min(1, Math.max(0, ((r.px.x - sa.x) * dx + (r.px.y - sa.y) * dy) / len2)) : 0;
        const off = Math.hypot(sa.x + t * dx - r.px.x, sa.y + t * dy - r.px.y);
        if (off < 6 && (!best || off < best.off)) best = { k, depth: sa.depth + t * (sb.depth - sa.depth), off };
      }
      return best;
    };
    return {
      label: t("ui.rig.links"),
      bounds: () => new THREE.Box3().setFromArray(drawn.segments),
      hit: (r) => nearest(r)?.depth ?? null,
      order: ORDER.lines,
      part: (r) => { const n = nearest(r); return n ? String(n.k) : null; },
      choose: (k) => { const l = k !== null ? drawn.links[Number(k)] : undefined; if (l) unlink.current(l); },
    };
  }, [active, drawn, lang]); // eslint-disable-line react-hooks/exhaustive-deps -- lang: the label
  usePickable("rig-pair/links", pickable);
  if (!drawn.links.length) return null;
  return <FatLines segments={drawn.segments} color={RIG_COLORS.link} width={Math.max(1, o.lineWidth)} opacity={0.85} overlay />;
}

// ------------------------------------------------------------------ 面板（画布外：两棵树与功能条）



const MODES: { value: RigMode; label: string }[] = [ // label: the key of its words
  { value: "map", label: "ui.rig.mode_map" },
  { value: "pose", label: "ui.rig.mode_pose" },
  { value: "ignore", label: "ui.rig.mode_ignore" },
];

export function RigPairPanels({ onInsets, ...props }: RigPairProps & { onInsets: (insets: Insets) => void }) {
  const ctl = useRigPair(props);
  const barRef = useRef<HTMLDivElement>(null);
  useCoveredEdges(barRef, onInsets, `${ctl.node}|${ctl.data.src.path ?? ""}|${ctl.data.dst.path ?? ""}`);
  const { data, roles, params, merged, lock } = ctl;
  const mode = useRigPairView((s) => s.mode);
  const first = useRigPairView((s) => s.first);
  const picked = useRigPairView((s) => s.picked);
  const menu = useRigPairView((s) => s.menu);
  const set = useRigPairView((s) => s.set);
  const clearPick = useRigPairView((s) => s.clearPick);
  const gap = useViewOptions((s) => s.o.rigPairGap);
  const setOption = useViewOptions((s) => s.set);
  const trees = usePreferences((s) => s.rigTreeWidth); // the bar sits between the two trees, however wide they are dragged
  useShortcut({ keys: ["escape"], run: () => { if (!first && !picked && !menu) return false; clearPick(); } }, { enabled: !!(first || picked || menu) });

  const hasPose = COLS.some((c) => roles[poseRole(c)] && posable(data[c]));
  const hasIgnore = COLS.some((c) => roles[ignoreRole(c)] && !data[c].fixed);
  const sizeSides = COLS.filter((c) => roles[scaleRole(c)] && !data[c].fixed);
  const hasSize = sizeSides.length > 0;
  const rules = data.rules ?? [];
  const rule = ruleNow(data, roles.ignore_rule ? params?.[roles.ignore_rule] : undefined);
  const ruleLabel = rules.find((r) => r.id === rule)?.label ?? rule;
  const record = roles.auto_record ? recordOf(params?.[roles.auto_record]) : null;
  const current = currentRecord(roles, params);
  const suggested = suggestedRecord(data, roles);
  const status = (kind: AutoKind) => (roles.auto_record ? autoState(kind, record, current, kind === "ignore" ? { id: rule, label: ruleLabel } : undefined, suggested) : null);
  const runAuto = (kind: AutoKind) => { const patch = autoPatch(kind, data, roles, params, rule); if (patch) ctl.writeAuto(patch); };
  const owners = { src: partOf(merged, "src"), dst: partOf(merged, "dst") };
  // 识别置信度（按推测的部位）与识别没认的部位：服务端识别引擎的判断，树上只读地标出来
  const lang = useLang((s) => s.lang); // the tips below are words
  const confidence = useMemo(() => ({ src: confidenceOf(merged, data, "src"), dst: confidenceOf(merged, data, "dst") }), [merged, data, lang]); // eslint-disable-line react-hooks/exhaustive-deps
  const unsure = useMemo(() => ({ src: unsureParts(data.src.recognition, data, merged, "src"), dst: unsureParts(data.dst.recognition, data, merged, "dst") }), [merged, data, lang]); // eslint-disable-line react-hooks/exhaustive-deps
  const label = (part: string | undefined) => (part ? data.parts_table.find((p) => p.id === part)?.label ?? part : undefined);
  const anyFixed = data.src.fixed || data.dst.fixed;
  const bad = problems(merged, data);
  const badParts = data.parts_table.filter((p) => bad[p.id]);
  // 对应关系和忽略合不合选中规则的模型骨架：编辑时就说（解算器计算时按同一套数报错）
  const ruleIssues = ruleProblems(merged, data, rule, ctl.ignored);

  const offWhy = {
    mapping: lock || (data.auto_mapping.length ? "" : t("ui.rig.off_mapping")),
    pose: lock || (!hasPose ? t("ui.rig.no_pose_param") : COLS.some((c) => roles[poseRole(c)] && data[c].auto_pose) ? "" : t("ui.rig.off_pose")),
    ignore: lock || (!hasIgnore ? t("ui.rig.no_ignore_param") : rules.some((r) => r.id === rule) ? "" : t("ui.rig.off_ignore")),
    size: lock || (!hasSize ? t("ui.rig.no_size_param") : sizeSides.some((c) => data[c].size) ? "" : t("ui.rig.off_size")),
  };
  // 「全部自动」：能做的几项按 对应 → 忽略 → 姿态 → 尺寸 一次做完（一步撤销）；一项都做不了时说为什么。尺寸放最后：
  // 它只看腿长（骨长），不受忽略和姿态影响，放在哪一步结果都一样，排在最后与功能条从左到右的顺序一致
  const allKinds = ([["mapping", offWhy.mapping], ["ignore", hasIgnore ? offWhy.ignore : "-"], ["pose", hasPose ? offWhy.pose : "-"],
    ["size", hasSize ? offWhy.size : "-"]] as [AutoKind, string][])
    .filter(([, why]) => !why).map(([k]) => k);
  const allWhy = lock || (allKinds.length ? "" : t("ui.rig.off_all"));
  const runAll = () => { const patch = autoAllPatch(allKinds, data, roles, params, rule); if (patch) ctl.writeAuto(patch); };
  const line = (kind: AutoKind) => {
    const s = status(kind);
    return s && <span className={`rpair-state ${s.tone}`}>{s.text}</span>;
  };
  const hint = mode === "map"
    ? first ? t("ui.rig.hint_map_first", { side: t(SIDE_KEY[first.col]), joint: first.joint }) : t("ui.rig.hint_map")
    : mode === "ignore" ? t("ui.rig.hint_ignore")
      : t("ui.rig.hint_pose");

  const menuRows: MenuRow[] = menu ? data.parts_table
    .filter((p) => p.chain || !merged.rows[p.id]?.[menu.first.col]?.length)
    .map((p) => ({ key: p.id, label: p.label, desc: p.region, run: () => ctl.choosePart(p.id) })) : [];

  const pickedSide = picked ? data[picked.col] : null;
  const pickedRole = picked ? roles[poseRole(picked.col)] : undefined;
  return (
    <>
      {COLS.map((c) => (
        <RigTree key={c} col={c} side={data[c]} title={t(SKELETON_KEY[c])} states={ctl.states[c]}
          partOf={(n) => label(owners[c].get(n))}
          partners={anyFixed ? (n) => partnersOf(merged, c, n) : null}
          confidence={confidence[c]} unsure={unsure[c]}
          onAct={(n, how) => ctl.act(c, n, how)} />
      ))}
      <div className="rpair-bar" ref={barRef} style={{ left: trees.src + 16, right: trees.dst + 16 }}>
        {/* 三栏，一栏一个模式：上面是模式（点它切换），下面是这个模式自己的「自动」和它的状态；当前模式那一栏高亮。
            「全部自动」单独一栏在右边。「前后距离」只是怎么看，放在底下提示那一行 */}
        <div className={`rpair-modes${hasSize ? " with-size" : ""}`}>
          {MODES.map((m) => {
            const kind: AutoKind = m.value === "map" ? "mapping" : m.value;
            const absent = m.value === "pose" && !hasPose ? t("ui.rig.no_pose_param_map_only") : m.value === "ignore" && !hasIgnore ? t("ui.rig.no_ignore_param") : "";
            return (
              <div key={m.value} className={`rpair-mode${mode === m.value ? " on" : ""}${absent ? " absent" : ""}`}>
                <button type="button" className="rpair-mode-head" disabled={!!absent} {...tipAttrs(tipOf("disabled", absent))}
                  aria-pressed={mode === m.value} onClick={() => set({ mode: m.value, first: null, picked: null, menu: null })}>
                  {t(m.label)}
                </button>
                {!absent && (
                  <div className="rpair-mode-auto">
                    {kind === "ignore" && (
                      <Select label={t("ui.rig.ignore_rule")} value={rule} disabled={!!lock || !rules.length || !roles.ignore_rule}
                        options={rules.map((r) => ({ value: r.id, label: r.label }))}
                        onPick={(v) => roles.ignore_rule && ctl.write({ [roles.ignore_rule]: v })} />
                    )}
                    <span className="rpair-mode-run">
                      <Button size="sm" disabled={!!offWhy[kind]} tip={tipOf("disabled", offWhy[kind])} onClick={() => runAuto(kind)}>{autoName(kind)}</Button>
                      {line(kind)}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
          {hasSize && (
            // 尺寸不是一种编辑模式：栏头只是名字。两侧的系数（空着 = 自动，框里灰字是自动的值）、「自动尺寸」与它的状态
            <div className="rpair-mode rpair-size">
              <div className="rpair-mode-head rpair-size-head">{t("ui.rig.size")}</div>
              <div className="rpair-mode-auto">
                <span className="rpair-size-row">
                  {sizeSides.map((c) => {
                    const role = roles[scaleRole(c)]!;
                    const auto = data[c].size?.auto;
                    return (
                      <span key={c} className="rpair-size-side">
                        <span className="rpair-dim">{t(SIDE_KEY[c])}</span>
                        <Num value={scaleOf(params?.[role])} nullable min={0} openMin unit={t("ui.rig.times")} mini disabled={!!lock}
                          label={t("ui.rig.size_of", { side: t(SIDE_KEY[c]) })} placeholder={auto === undefined ? t("ui.rig.auto") : t("ui.rig.auto_value", { value: auto })}
                          onChange={(v) => ctl.write({ [role]: v })} />
                      </span>
                    );
                  })}
                </span>
                <span className="rpair-mode-run">
                  <Button size="sm" disabled={!!offWhy.size} tip={tipOf("disabled", offWhy.size)} onClick={() => runAuto("size")}>{autoName("size")}</Button>
                  {line("size")}
                </span>
                <span className="rpair-dim rpair-size-legs">
                  {t("ui.rig.leg_length", { sides: sizeSides.map((c) => `${t(SIDE_KEY[c])} ${data[c].size ? `${Math.round(data[c].size!.leg_cm)} cm` : "—"}`).join(" · ") })}
                </span>
              </div>
            </div>
          )}
          <div className="rpair-all">
            <Button size="sm" tone="primary" disabled={!!allWhy} tip={tipOf("disabled", allWhy)} onClick={runAll}>{t("ui.rig.all_auto")}</Button>
            <span className="rpair-dim">{hasSize ? t("ui.rig.all_order_size") : t("ui.rig.all_order")}</span>
          </div>
        </div>
        <div className="rpair-bar-row rpair-hint">
          <span className="rpair-dim rpair-hint-text">{hint}</span>
          {first && <Button size="sm" tone="ghost" onClick={clearPick}>{t("ui.common.cancel")}</Button>}
          <span className="rpair-view">
            <span className="rpair-dim">{t("ui.rig.gap")}</span>
            <span className="rpair-gap">
              <Slider value={gap} onChange={(v) => setOption({ rigPairGap: v })} min={RANGES.rigPairGap[0]} max={RANGES.rigPairGap[1]} unit="cm" digits={0}
                disabled={anyFixed} />
            </span>
          </span>
        </div>
        {anyFixed && <div className="rpair-bar-row rpair-dim">{t("ui.rig.fixed_note")}</div>}
        {merged.stale?.length ? (
          <div className="rpair-bar-row rpair-bad">
            {t("ui.rig.stale", { parts: merged.stale.map((id) => data.parts_table.find((q) => q.id === id)?.label ?? id).join(listSep()), button: autoName("mapping") })}
          </div>
        ) : null}
        {ruleIssues.length > 0 && (
          <div className="rpair-bar-row rpair-bad">
            {t(ruleIssues.length > 3 ? "ui.rig.rule_fails_more" : "ui.rig.rule_fails", { rule: rules.find((r) => r.id === rule)?.label ?? rule, issues: ruleIssues.slice(0, 3).join(t("ui.rig.issue_sep")), count: ruleIssues.length })}
          </div>
        )}
        {badParts.length > 0 && (
          <div className="rpair-bar-row rpair-bad">
            {t(badParts.length > 3 ? "ui.rig.problems_more" : "ui.rig.problems", { parts: badParts.slice(0, 3).map((p) => t("ui.rig.problem_of", { part: p.label, why: bad[p.id] })).join(t("ui.rig.issue_sep")), count: badParts.length })}
          </div>
        )}
        {mode === "pose" && (
          <div className="rpair-pose">
            <div className="rpair-bar-row">
              {COLS.filter((c) => roles[poseRole(c)] && posable(data[c])).map((c) => (
                <span key={c} className="rpair-bar-row">
                  <span className="rpair-dim">{t("ui.rig.side_changed", { side: t(SIDE_KEY[c]), count: ctl.poses[c].length })}</span>
                  <Button size="sm" tone="ghost" disabled={!!lock || !ctl.poses[c].length} tip={tipOf("disabled", lock)}
                    onClick={() => ctl.write({ [roles[poseRole(c)]!]: [] })}>{t("ui.rig.reset_all")}</Button>
                </span>
              ))}
            </div>
            {picked && pickedSide && posable(pickedSide) && pickedRole && (
              <PoseJointEditor data={pickedSide} rows={ctl.poses[picked.col]} joint={picked.joint}
                write={(rows) => ctl.write({ [pickedRole]: rows })} onDeselect={() => set({ picked: null })} />
            )}
          </div>
        )}
        {lock && <div className="rpair-bar-row pose-warn">{lock}</div>}
      </div>
      {menu && (
        <Menu at={menu} rows={menuRows} label={t("ui.rig.menu_part")} onClose={() => set({ menu: null })} />
      )}
    </>
  );
}

/** 舞台被功能条和两棵树盖住的边（像素）：框显只用剩下的那块（view/stageState.ts StageState.insets）。随它们的大小量；
 * 进入编辑（或换了一对骨架）量好之后框显一次。 */
function useCoveredEdges(bar: React.RefObject<HTMLDivElement | null>, onInsets: (i: Insets) => void, pair: string) {
  const framed = useRef("");
  useLayoutEffect(() => {
    const el = bar.current;
    const stage = el?.parentElement;
    if (!el || !stage) return;
    const measure = () => {
      const box = stage.getBoundingClientRect();
      const trees = [...stage.querySelectorAll<HTMLElement>(":scope > .rpair-tree-panel")].map((t) => t.getBoundingClientRect());
      const left = Math.max(0, ...trees.filter((t) => t.left - box.left < box.width / 2).map((t) => t.right - box.left));
      const right = Math.max(0, ...trees.filter((t) => t.left - box.left >= box.width / 2).map((t) => box.right - t.left));
      const bottom = Math.max(0, box.bottom - el.getBoundingClientRect().top);
      onInsets({ left, right, top: 0, bottom });
      if (framed.current !== pair) {
        framed.current = pair;
        // 骨架在画布里登记拾取物要晚一拍：下一帧再框
        requestAnimationFrame(() => requestAnimationFrame(() => useViewCamera.getState().frame("all")));
      }
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    ro.observe(stage);
    return () => ro.disconnect();
  }, [pair]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => () => onInsets({ left: 0, right: 0, top: 0, bottom: 0 }), []); // eslint-disable-line react-hooks/exhaustive-deps
}

/** 编辑中的手柄数据还没到（上游没算过，或正在问服务器）。 */
export function RigPairWaiting() {
  return <div className="rpair-bar"><span className="rpair-dim">{t("ui.rig.waiting")}</span></div>;
}
