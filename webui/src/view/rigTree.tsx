/** 双骨架编辑（view/rigPair.tsx）左右两棵骨架树：左源右目标。按层级缩进、可收起、可按名字搜；每行一个眼睛（这个
 * 关节和它的子孙在视图里不画，只影响显示）、名字（按状态上色：已配对 / 未配对 / 忽略）、所属部位，按推测的部位
 * 再带识别置信度（model/rigPair.ts confidenceOf：中、低两档上提示色，悬停看数值与依据）；树头写识别没认的部位数。悬停与视图联动
 * 高亮，点击与在视图里点骨点同一个动作（按模式：配对 / 选中摆姿态 / 切换忽略；Alt 点只切换这一个）。
 * 模型固定骨架的一侧（没有位置、视图里不画）在行尾写出它配上的源关节，连线就看这里。 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { RigPairSide } from "../api";
import type { Col, JointState, PartConfidence } from "../model/rigPair";
import { useRigPairView } from "../state/rigPairView";
import { usePreferences } from "../state/preferences";
import { followDrag } from "../platform/drag";
import { IconButton } from "../ui/Button";
import { IconChevron, IconEye } from "../ui/icons";
import { OVERLAY_SWATCHES } from "../model/viewOptions";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 双骨架编辑的颜色（视图与树同一套）：已配对按侧（源青、目标黄，实色），未配对灰，忽略暗红（视图里半透明），
 * 连线绿，选中 / 先点的骨点与悬停用强调蓝。 */
export const RIG_COLORS = {
  paired: { src: OVERLAY_SWATCHES[4], dst: OVERLAY_SWATCHES[1] },
  unpaired: "#8e8e93",
  ignored: "#9b2c3a",
  link: OVERLAY_SWATCHES[5],
  active: "#0a84ff",
};
export const stateColor = (col: Col, s: JointState): string => (s === "paired" ? RIG_COLORS.paired[col] : s === "ignored" ? RIG_COLORS.ignored : RIG_COLORS.unpaired);

export interface ActHow {
  alt: boolean;
  x: number;
  y: number;
}

export function RigTree({ col, side, title, states, partOf, partners, confidence, unsure, onAct }: {
  col: Col;
  side: RigPairSide;
  title: string;
  states: JointState[];
  partOf: (joint: string) => string | undefined; // 部位的中文名
  partners: ((joint: string) => string[]) | null; // 固定一侧：每行写它配上的另一侧关节
  confidence?: Map<string, PartConfidence>; // 按推测的部位的识别置信度（标在部位的第一个关节上）
  unsure?: { count: number; tip: string }; // 识别没认的部位
  onAct: (joint: string, how: ActHow) => void;
}) {
  const width = usePreferences((s) => s.rigTreeWidth[col]);
  const setWidth = usePreferences((s) => s.setRigTreeWidth);
  const hidden = useRigPairView((s) => s.hidden[col]);
  const hover = useRigPairView((s) => (s.hover?.col === col ? s.hover.joint : null));
  const first = useRigPairView((s) => (s.first?.col === col ? s.first.joint : null));
  const picked = useRigPairView((s) => (s.picked?.col === col ? s.picked.joint : null));
  const set = useRigPairView((s) => s.set);
  const toggleHidden = useRigPairView((s) => s.toggleHidden);
  const { kids, roots, depth } = useMemo(() => {
    const kids: number[][] = side.names.map(() => []);
    const roots: number[] = [];
    side.parents.forEach((p, i) => (p >= 0 && kids[p] ? kids[p].push(i) : roots.push(i)));
    const depth: number[] = [];
    const walk = (i: number, d: number) => { depth[i] = d; kids[i].forEach((k) => walk(k, d + 1)); };
    roots.forEach((r) => walk(r, 0));
    return { kids, roots, depth };
  }, [side]);
  const [closed, setClosed] = useState<Set<number>>(() => new Set());
  const [query, setQuery] = useState("");
  const hiddenSet = useMemo(() => new Set(hidden), [hidden]);
  // 眼睛关着的关节的子孙在树里变暗（视图里不画它们）
  const dim = useMemo(() => side.names.map((_, i) => {
    for (let j = side.parents[i] ?? -1, n = 0; j >= 0 && n < 10000; j = side.parents[j] ?? -1, n++) if (hiddenSet.has(side.names[j])) return true;
    return false;
  }), [side, hiddenSet]);
  const rows = useMemo(() => {
    const out: number[] = [];
    const q = query.trim().toLowerCase();
    if (q) {
      const keep = new Set<number>();
      side.names.forEach((n, i) => {
        if (!n.toLowerCase().includes(q)) return;
        for (let k = i; k >= 0 && !keep.has(k); k = side.parents[k] ?? -1) keep.add(k);
      });
      const walk = (i: number) => { if (keep.has(i)) { out.push(i); kids[i].forEach(walk); } };
      roots.forEach(walk);
      return out;
    }
    const walk = (i: number) => { out.push(i); if (!closed.has(i)) kids[i].forEach(walk); };
    roots.forEach(walk);
    return out;
  }, [side, kids, roots, closed, query]);

  // 在视图里点中的骨点（先点的、选中的）：展开到它、滚到它
  const list = useRef<HTMLDivElement>(null);
  const focus = first ?? picked;
  useEffect(() => {
    if (!focus) return;
    const i = side.names.indexOf(focus);
    if (i < 0) return;
    setClosed((was) => {
      let next = was;
      for (let k = side.parents[i] ?? -1; k >= 0; k = side.parents[k] ?? -1)
        if (next.has(k)) { if (next === was) next = new Set(was); next.delete(k); }
      return next;
    });
    requestAnimationFrame(() => {
      const el = list.current?.querySelector<HTMLElement>(`[data-row="${i}"]`);
      el?.scrollIntoView({ block: "nearest" });
    });
  }, [focus, side]);

  const flip = (i: number) => setClosed((was) => { const next = new Set(was); if (next.has(i)) next.delete(i); else next.add(i); return next; });
  return (
    <div className={`rpair-tree-panel ${col}`} style={{ width }}>
      {/* the inner edge drags the width (kept in this browser, state/preferences.ts); at most half the stage */}
      <div className="rpair-tree-grip" onPointerDown={(e) => {
        e.preventDefault();
        const panel = e.currentTarget.parentElement!;
        const stage = panel.parentElement!.getBoundingClientRect();
        const from = e.clientX, was = panel.getBoundingClientRect().width;
        followDrag((m) => {
          const dx = (m.clientX - from) * (col === "src" ? 1 : -1);
          setWidth(col, Math.min(stage.width * 0.5, was + dx));
        }, () => {});
      }} />
      <div className="rpair-tree-head">
        <b>{title}</b>
        <span className="rpair-dim">{t("ui.rig.joint_count", { count: side.names.length })}{side.fixed ? t("ui.rig.model_fixed") : ""}</span>
        {!!unsure?.count && <span className="rpair-conf low" {...tipAttrs(tipOf("value", unsure.tip))}>{t("ui.rig.unsure_count", { count: unsure.count })}</span>}
      </div>
      <input className="field rpair-search" placeholder={t("ui.rig.search_joint")} value={query} onChange={(e) => setQuery(e.target.value)} />
      <div className="rpair-tree" ref={list} onMouseLeave={() => set({ hover: null })}>
        {rows.map((i) => {
          const name = side.names[i];
          const state = states[i] ?? "unpaired";
          const part = partOf(name);
          const off = hiddenSet.has(name);
          const on = name === first || name === picked;
          const open = !closed.has(i) || !!query;
          const with_ = partners?.(name) ?? [];
          const conf = confidence?.get(name);
          return (
            <div key={i} data-row={i} className={`rpair-row${on ? " on" : ""}${hover === name ? " hov" : ""}${off || dim[i] ? " dim" : ""}`}
              style={{ paddingLeft: 2 + (depth[i] ?? 0) * 12 }}
              onMouseEnter={() => set({ hover: { col, joint: name } })}
              onMouseDown={(e) => e.preventDefault()}
              onClick={(e) => onAct(name, { alt: e.altKey, x: e.clientX, y: e.clientY })}>
              <span className="rpair-twist" onClick={(e) => { e.stopPropagation(); if (kids[i].length && !query) flip(i); }}>
                {kids[i].length > 0 && <span style={{ display: "inline-flex", transform: open ? "rotate(180deg)" : "rotate(90deg)" }}><IconChevron size={9} up /></span>}
              </span>
              <IconButton size="xxs" tone="ghost" on={!off} aria-label={off ? t("ui.rig.show") : t("ui.rig.hide")}
                onClick={(e) => { e.stopPropagation(); toggleHidden(col, name); }}>
                <IconEye size={11} />
              </IconButton>
              <span className="rpair-dot" style={{ background: stateColor(col, state), opacity: state === "ignored" ? 0.6 : 1 }} />
              <span className="rpair-name" data-user-data {...tipAttrs(tipOf("truncated", name))}>{name}</span>
              {part && <span className="rpair-tag">{part}</span>}
              {part && conf && <span className={`rpair-conf ${conf.level}`} {...tipAttrs(tipOf("value", conf.tip))}>{Math.round(conf.value * 100)}%</span>}
              {with_.length > 0 && <span className="rpair-with" data-user-data {...tipAttrs(tipOf("truncated", with_.join(" → ")))}>{with_.join(" → ")}</span>}
            </div>
          );
        })}
        {rows.length === 0 && <div className="rpair-empty">{query ? t("ui.rig.no_joint_matching", { query }) : t("ui.rig.no_joints")}</div>}
      </div>
    </div>
  );
}
