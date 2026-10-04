import { OVERLAY_SWATCHES, INTEGERS, RANGES } from "../model/viewOptions";
import { slotCamera, useViewCamera, VIEWER_SLOT } from "../state/viewer";
import { Segmented, Switch } from "./Button";
import { Swatches } from "./Swatches";
import { Num } from "./controls";
import { choices, Row, Slider, type Off, type O, type Patch } from "./displayRows";
import { msg, textOf } from "../messages/message";
import { t } from "../i18n/t";

/** 「视图设置」面板的几页：点、线、模型、骨骼、相机、场景、渲染。
 * 一页一个组件，每页只排自己的行；行的样子和变灰都在 ui/displayRows.tsx。
 * 大小和线宽只有屏幕像素一种单位；曲线和其它线共用一个线宽。
 *
 * 每一行永远在原来的位置：用不上的变灰并写原因，不按「没有东西可改就不渲染」处理，
 * 所以切结果的时候面板里的东西不会跳位置。 */

const SWATCHES = ["#FFD60A", "#FF9F0A", "#FF375F", "#BF5AF2", "#0A84FF", "#64D2FF", "#30D158", "#F2F2F7"];
const BACKGROUNDS = ["#0c0c0e", "#1c1c20", "#2a2d34", "#3a3d45", "#5b5f68", "#8e9199"];

export const POINT_KEYS: (keyof O)[] = ["pointPx", "pointColor", "pointTint"];
export const LINE_KEYS: (keyof O)[] = ["lineWidth", "cameraColor", "curveColor", "curveTint", "overlayByPerson"];
export const BONE_KEYS: (keyof O)[] = ["boneColor", "boneStyle", "boneWidth", "jointSize", "jointAdaptive", "jointNames", "jointNamePx"];
export const MESH_KEYS: (keyof O)[] = ["shading", "backFaces", "uvChecker", "overlayOpacity", "overlayTint"];
/** 消息表里的一段文字（面板上新加的几行走消息表，lab2shot/i18n/<lang>/messages/web.toml）。 */
const said = (code: string) => textOf(msg(code));

export const CAMERA_KEYS: (keyof O)[] = ["near", "far", "cameraPath"];
export const SCENE_KEYS: (keyof O)[] = ["grid", "gridSpacing", "gridSize", "axes", "axesSize", "background", "backgroundColor", "lighting", "exposure"];
export const QUALITY_KEYS: (keyof O)[] = ["antialias"];

export function PointsTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  const [pxLo, pxHi] = RANGES.pointPx;
  return (
    <>
      <Row label={t("ui.display.size")} off={off("pointPx")}>
        <Slider value={o.pointPx} onChange={(pointPx) => set({ pointPx })} min={pxLo} max={pxHi} unit="px" log digits={1} disabled={!!off("pointPx")} />
      </Row>
      <Row label={t("ui.display.color")} off={off("pointColor")}>
        <Segmented label={t("ui.display.color")} stretch value={o.pointColor} options={choices("pointColor", off("pointColor"))} onChange={(pointColor) => set({ pointColor })} />
      </Row>
      <Row label={t("ui.display.constant_color_label")} off={off("pointTint")}>
        <Swatches label={t("ui.display.constant_color_label")} value={o.pointTint} colors={["#d8d8dc", ...SWATCHES]} onChange={(pointTint) => set({ pointTint })} disabled={off("pointTint")} />
      </Row>
    </>
  );
}

/* 三维没有「压缩预览」这一档：点云超过后台的「视图 · 点云上限」就每 N 个点取一个，
 * 删了多少由视图通知区那一条「显示了 N / 共 M 点」说（不许偷偷抽稀）。 */

export function LinesTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  const [lo, hi] = RANGES.lineWidth;
  return (
    <>
      <Row label={t("ui.display.line_width")} off={off("lineWidth")}>
        <Slider value={o.lineWidth} onChange={(lineWidth) => set({ lineWidth })} min={lo} max={hi} unit="px" digits={1} disabled={!!off("lineWidth")} />
      </Row>
      <Row label={t("ui.display.camera_color")} off={off("cameraColor")}>
        <Swatches label={t("ui.display.camera_color")} value={o.cameraColor} colors={SWATCHES} onChange={(cameraColor) => set({ cameraColor })} disabled={off("cameraColor")} />
      </Row>
      <Row label={t("ui.display.curve_color")} off={off("curveColor")}>
        <Segmented label={t("ui.display.curve_color")} stretch value={o.curveColor} options={choices("curveColor", off("curveColor"))} onChange={(curveColor) => set({ curveColor })} />
      </Row>
      <Row label={t("ui.display.curve_tint")} off={off("curveTint")}>
        <Swatches label={t("ui.display.curve_tint")} value={o.curveTint} colors={SWATCHES} onChange={(curveTint) => set({ curveTint })} disabled={off("curveTint")} />
      </Row>
      <Row label={t("ui.display.by_person")} off={off("overlayByPerson")}>
        <Switch label={t("ui.display.by_person")} on={o.overlayByPerson} onChange={(overlayByPerson) => set({ overlayByPerson })} disabled={!!off("overlayByPerson")} />
      </Row>
    </>
  );
}

export function MeshesTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label={t("ui.display.shading")} off={off("shading")}>
        <Segmented label={t("ui.display.shading")} stretch value={o.shading} options={choices("shading", off("shading"))} onChange={(shading) => set({ shading })} />
      </Row>
      <Row label={t("ui.display.back_faces")} off={off("backFaces")}>
        <Switch on={o.backFaces} onChange={(backFaces) => set({ backFaces })} disabled={!!off("backFaces")} />
      </Row>
      <Row label={t("ui.display.uv_checker")} off={off("uvChecker")}>
        <Switch on={o.uvChecker} onChange={(uvChecker) => set({ uvChecker })} disabled={!!off("uvChecker")} />
      </Row>
      <Row label={t("ui.display.overlay_opacity")} off={off("overlayOpacity")}>
        <Slider value={o.overlayOpacity} onChange={(overlayOpacity) => set({ overlayOpacity })}
          min={RANGES.overlayOpacity[0]} max={RANGES.overlayOpacity[1]} digits={2} disabled={!!off("overlayOpacity")} />
      </Row>
      <Row label={t("ui.display.overlay_tint")} off={off("overlayTint")}>
        <Swatches label={t("ui.display.overlay_tint")} value={o.overlayTint} colors={OVERLAY_SWATCHES} onChange={(overlayTint) => set({ overlayTint })} disabled={off("overlayTint")} />
      </Row>
    </>
  );
}

export function CameraTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  const ortho = useViewCamera((s) => slotCamera(s, VIEWER_SLOT).ortho); // the toolbar's camera tab sets the viewer's
  const setOrtho = useViewCamera((s) => s.setOrtho);
  return (
    <>
      {/* 正交是相机自己的状态（state/viewer.ts），不是显示选项：三维视图里一直能用 */}
      <Row label={t("ui.display.ortho")}>
        <Switch on={ortho} onChange={setOrtho} />
      </Row>
      <Row label={t("ui.display.near")} off={off("near")}>
        <Num value={o.near} onChange={(near) => set({ near: Math.min(near, o.far * 0.5) })} min={RANGES.near[0]} max={RANGES.near[1]} unit="cm" digits={3} disabled={!!off("near")} />
      </Row>
      <Row label={t("ui.display.far")} off={off("far")}>
        <Num value={o.far} onChange={(far) => set({ far: Math.max(far, o.near * 2) })} min={RANGES.far[0]} max={RANGES.far[1]} unit="cm" digits={0} disabled={!!off("far")} />
      </Row>
      <Row label={said("I-VIEW-CAMERAPATH")} off={off("cameraPath")}>
        <Switch on={o.cameraPath} onChange={(cameraPath) => set({ cameraPath })} disabled={!!off("cameraPath")} />
      </Row>
    </>
  );
}

export function BonesTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label={t("ui.display.bone_color")} off={off("boneColor")}>
        <Swatches label={t("ui.display.bone_color")} value={o.boneColor} colors={SWATCHES} onChange={(boneColor) => set({ boneColor })} disabled={off("boneColor")} />
      </Row>
      <Row label={said("I-VIEW-BONESTYLE")} off={off("boneStyle")}>
        <Segmented label={said("I-VIEW-BONESTYLE")} stretch value={o.boneStyle} options={choices("boneStyle", off("boneStyle"))} onChange={(boneStyle) => set({ boneStyle })} />
      </Row>
      <Row label={said("I-VIEW-BONEWIDTH")} off={off("boneWidth")}>
        <Slider value={o.boneWidth} onChange={(boneWidth) => set({ boneWidth })} min={RANGES.boneWidth[0]} max={RANGES.boneWidth[1]} unit="×" log digits={2} disabled={!!off("boneWidth")} />
      </Row>
      <Row label={said("I-VIEW-JOINTSIZE")} off={off("jointSize")}>
        <Slider value={o.jointSize} onChange={(jointSize) => set({ jointSize })} min={RANGES.jointSize[0]} max={RANGES.jointSize[1]} unit="×" log digits={2} disabled={!!off("jointSize")} />
      </Row>
      <Row label={said("I-VIEW-JOINTADAPTIVE")} off={off("jointAdaptive")}>
        <Switch on={o.jointAdaptive} onChange={(jointAdaptive) => set({ jointAdaptive })} disabled={!!off("jointAdaptive")} />
      </Row>
      <Row label={said("I-VIEW-JOINTNAMES")} off={off("jointNames")}>
        <Switch on={o.jointNames} onChange={(jointNames) => set({ jointNames })} disabled={!!off("jointNames")} />
      </Row>
      <Row label={said("I-VIEW-JOINTNAMEPX")} off={off("jointNamePx")}>
        <Slider value={o.jointNamePx} onChange={(jointNamePx) => set({ jointNamePx })} min={RANGES.jointNamePx[0]} max={RANGES.jointNamePx[1]} integer={INTEGERS.has("jointNamePx")} unit="px" digits={0} disabled={!!off("jointNamePx")} />
      </Row>
    </>
  );
}

export function SceneTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label={t("ui.display.grid")} off={off("grid")}>
        <Switch on={o.grid} onChange={(grid) => set({ grid })} disabled={!!off("grid")} />
      </Row>
      <Row label={t("ui.display.grid_spacing")} off={off("gridSpacing")}>
        <Num value={o.gridSpacing} onChange={(gridSpacing) => set({ gridSpacing })} min={RANGES.gridSpacing[0]} max={RANGES.gridSpacing[1]} unit="cm" disabled={!!off("gridSpacing")} />
      </Row>
      <Row label={t("ui.display.grid_size")} off={off("gridSize")}>
        <Num value={o.gridSize} onChange={(gridSize) => set({ gridSize })} min={RANGES.gridSize[0]} max={RANGES.gridSize[1]} unit="cm" digits={0} disabled={!!off("gridSize")} />
      </Row>
      <Row label={t("ui.display.axes")} off={off("axes")}>
        <Switch on={o.axes} onChange={(axes) => set({ axes })} disabled={!!off("axes")} />
      </Row>
      <Row label={t("ui.display.axes_size")} off={off("axesSize")}>
        <Slider value={o.axesSize} onChange={(axesSize) => set({ axesSize })} min={RANGES.axesSize[0]} max={RANGES.axesSize[1]} integer={INTEGERS.has("axesSize")} unit="px" digits={0} disabled={!!off("axesSize")} />
      </Row>
      <Row label={t("ui.display.background")} off={off("background")}>
        <Segmented label={t("ui.display.background")} stretch value={o.background} options={choices("background", off("background"))} onChange={(background) => set({ background })} />
      </Row>
      <Row label={t("ui.display.background_color")} off={off("backgroundColor")}>
        <Swatches label={t("ui.display.background_color")} value={o.backgroundColor} colors={BACKGROUNDS} onChange={(backgroundColor) => set({ backgroundColor })} disabled={off("backgroundColor")} />
      </Row>
      <Row label={t("ui.display.lighting")} off={off("lighting")}>
        <Segmented label={t("ui.display.lighting")} stretch value={o.lighting} options={choices("lighting", off("lighting"))} onChange={(lighting) => set({ lighting })} />
      </Row>
      <Row label={t("ui.display.exposure")} off={off("exposure")}>
        <Slider value={o.exposure} onChange={(exposure) => set({ exposure })} min={RANGES.exposure[0]} max={RANGES.exposure[1]} unit={t("ui.display.stops")} digits={1} disabled={!!off("exposure")} />
      </Row>
    </>
  );
}

export function QualityTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label={t("ui.display.antialias")} off={off("antialias")}>
        <Segmented label={t("ui.display.antialias")} stretch value={o.antialias} options={choices("antialias", off("antialias"))} onChange={(antialias) => set({ antialias })} />
      </Row>
    </>
  );
}
