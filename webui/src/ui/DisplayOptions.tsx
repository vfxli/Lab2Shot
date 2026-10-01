import "./displayoptions.css";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { DEFAULTS, type ViewOptions } from "../model/viewOptions";
import { useViewOptions } from "../state/viewer";
import { usable, why as whyOff } from "../api/applies";
import { type ViewFacts } from "../model/viewControls";
import { viewAvailable } from "../view/available";
import { useDismiss } from "../platform/dismiss";
import { usePreferences } from "../state/preferences";
import { Button, Segmented, Toggle } from "./Button";
import type { Has, Off } from "./displayRows";
import { BonesTab, BONE_KEYS, CameraTab, CAMERA_KEYS, LinesTab, LINE_KEYS, MeshesTab, MESH_KEYS, PointsTab, POINT_KEYS,
         QualityTab, QUALITY_KEYS, SceneTab, SCENE_KEYS } from "./displayTabs";

/** 视图的显示选项（与 Houdini 相同）：视图控制栏上的一个按钮，打开一个小面板。
 * 每项改动立即生效，并保存在本浏览器中。
 *
 * 各舞台包含哪些页由结构定义（`STAGE_TABS`）：二维画面只有「线」，三维视图有点、线、模型、骨骼、相机、场景、渲染。
 * 页中的每一行始终存在（不适用的置灰并注明原因），因此切换显示的节点时面板不跳动、不变形，位置保持不变。
 * 当前页由用户首选项记录，记录的页不属于当前舞台时回退到该舞台的第一页。
 *
 * 画面没有画质设置：二维始终经由视图代理。「渲染」页中的抗锯齿属于渲染质量，因此不称为「画质」，以免同名异义造成混淆。 */
type Tab = "points" | "lines" | "meshes" | "bones" | "camera" | "scene" | "quality";
const TABS: Record<Tab, string> = { points: "点", lines: "线", meshes: "模型", bones: "骨骼", camera: "相机", scene: "场景", quality: "渲染" };
const STAGE_TABS: Record<"2d" | "3d", Tab[]> = {
  "2d": ["lines"],
  "3d": ["points", "lines", "meshes", "bones", "camera", "scene", "quality"],
};
const TAB_KEYS: Record<Tab, (keyof ViewOptions)[]> = {
  points: POINT_KEYS,
  lines: LINE_KEYS,
  meshes: MESH_KEYS,
  bones: BONE_KEYS,
  camera: CAMERA_KEYS,
  scene: SCENE_KEYS,
  quality: QUALITY_KEYS,
};

/** 「视图设置」按钮及其弹出面板。
 *
 * 二维预览链（通道、黑白点、着色、运算、背景）不在此处，而位于控制栏上的一排控件中（editor/previewBar.tsx）。 */
export function DisplayOptions({ stage, shows, preview }:
  { stage: "2d" | "3d"; shows: ReadonlySet<string>; preview?: ViewFacts }) {
  const o = useViewOptions((s) => s.o);
  const set = useViewOptions((s) => s.set);
  const reset = useViewOptions((s) => s.reset);
  const [open, setOpen] = useState(false);
  const stored = usePreferences((s) => s.displayOptionsTab);
  const setStored = usePreferences((s) => s.setDisplayOptionsTab);
  const box = useRef<HTMLDivElement>(null);
  const only2d = stage === "2d";
  const has: Has = (...what) => what.some((w) => shows.has(w));
  const facts: ViewFacts = preview ?? { stage, shows, options: o, mode: "plate", channels: 0, single: false, ready: false };
  const view = viewAvailable({ ...facts, stage, shows, options: o });
  const off: Off = (control) => (usable(view, control) ? "" : whyOff(view, control));
  const tabs = STAGE_TABS[stage];
  const shownTab: Tab = tabs.includes(stored as Tab) ? (stored as Tab) : tabs[0];

  // 面板挂载在 body 上，因此判断「点击在外部」时需同时考虑按钮和面板两个元素（useDismiss 支持选择器）
  useDismiss(open, ".vo-opts, .vo-panel", () => setOpen(false));
  // 面板挂载在 body 上，因此需自行测量按钮当前的屏幕坐标（打开时测量一次，窗口变化时重新测量）
  const [at, setAt] = useState<{ left: number; top: number } | null>(null);
  useEffect(() => {
    if (!open) return setAt(null);
    const place = () => {
      const r = box.current?.getBoundingClientRect();
      if (r) setAt({ left: r.left, top: r.bottom + 8 });
    };
    place();
    window.addEventListener("resize", place);
    return () => window.removeEventListener("resize", place);
  }, [open]);

  const changed = TAB_KEYS[shownTab].some((k) => o[k] !== DEFAULTS[k]);

  return (
    <div className="vo-anchor vo-opts" ref={box}>
      {/* 按钮位于视图工具栏中，该工具栏不显示悬停提示；按钮打开的面板挂载在 body 上，保留其自身的提示 */}
      <Toggle hud on={open} onChange={setOpen}>
        视图设置
      </Toggle>
      {open && at && createPortal(
        /* 挂载在 body 上，按屏幕坐标定位（与 ui/Menu.tsx 做法相同）。
           若在按钮旁使用 `position: absolute`，工具栏将无法横向滚动，滚动时面板会被裁掉一半。
           而工具栏必须可滚动：该行只允许一行，放不下时滚动，不得换行将画布向下推。 */
        <div className="popover vo-panel glass" role="dialog" aria-label="视图设置"
             style={{ position: "fixed", left: at.left, top: at.top, right: "auto" }}>
          <Segmented label="显示选项" stretch tabs layout="vo-tabs" value={shownTab}
            options={tabs.map((t) => ({ value: t, label: TABS[t], tip: `「${TABS[t]}」这一页的显示选项` }))} onChange={setStored} />
          <div className="vo-body">
            {shownTab === "points" && <PointsTab o={o} set={set} off={off} />}
            {shownTab === "lines" && <LinesTab o={o} set={set} only2d={only2d} has={has} off={off} />}
            {shownTab === "meshes" && <MeshesTab o={o} set={set} off={off} />}
            {shownTab === "bones" && <BonesTab o={o} set={set} off={off} />}
            {shownTab === "camera" && <CameraTab o={o} set={set} off={off} />}
            {shownTab === "scene" && <SceneTab o={o} set={set} off={off} />}
            {shownTab === "quality" && (
              <QualityTab o={o} set={set} off={off} />
            )}
          </div>
          <div className="vo-foot">
            <Button tip="这一页的选项回到默认" tone="ghost" size="sm" disabled={!changed} onClick={() => reset(TAB_KEYS[shownTab])}>
              恢复默认
            </Button>
          </div>
        </div>,
        document.body,
      )}
    </div>
  );
}
