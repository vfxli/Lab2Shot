import { OVERLAY_SWATCHES, RANGES } from "../model/viewOptions";
import { useViewCamera } from "../state/viewer";
import { Segmented, Switch } from "./Button";
import { Swatches } from "./Swatches";
import { choices, Num, Row, Slider, type Has, type Off, type O, type Patch } from "./displayRows";

/** 「视图设置」面板的几页：点、线、模型、相机、场景、渲染。
 * 一页一个组件，每页只排自己的行；行的样子和变灰都在 ui/displayRows.tsx。
 * 大小和线宽只有屏幕像素一种单位；曲线和其它线共用一个线宽。
 *
 * 每一行永远在原来的位置：用不上的变灰并写原因，不按「没有东西可改就不渲染」处理，
 * 所以切结果的时候面板里的东西不会跳位置。 */

const SWATCHES = ["#FFD60A", "#FF9F0A", "#FF375F", "#BF5AF2", "#0A84FF", "#64D2FF", "#30D158", "#F2F2F7"];
const BACKGROUNDS = ["#0c0c0e", "#1c1c20", "#2a2d34", "#3a3d45", "#5b5f68", "#8e9199"];

export const POINT_KEYS: (keyof O)[] = ["pointPx", "pointColor", "pointTint"];
export const LINE_KEYS: (keyof O)[] = ["lineWidth", "cameraColor", "boneColor", "curveColor", "curveTint", "overlayByPerson"];
export const MESH_KEYS: (keyof O)[] = ["shading", "uvChecker", "overlayOpacity", "overlayTint"];
export const CAMERA_KEYS: (keyof O)[] = ["near", "far"];
export const SCENE_KEYS: (keyof O)[] = ["grid", "gridSpacing", "gridSize", "axes", "axesSize", "background", "backgroundColor", "lighting", "exposure"];
export const QUALITY_KEYS: (keyof O)[] = ["antialias"];

export function PointsTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  const [pxLo, pxHi] = RANGES.pointPx;
  return (
    <>
      <Row label="大小" tip="点在屏幕上多少像素，远近一样大" off={off("pointPx")}>
        <Slider value={o.pointPx} onChange={(pointPx) => set({ pointPx })} min={pxLo} max={pxHi} unit="px" log digits={1} disabled={!!off("pointPx")} />
      </Row>
      <Row label="颜色" tip="自带：用点自己带的颜色；单色：全部一种颜色，颜色在下面选" off={off("pointColor")}>
        <Segmented label="颜色" stretch value={o.pointColor} options={choices("pointColor", off("pointColor"))} onChange={(pointColor) => set({ pointColor })} />
      </Row>
      <Row label="单色" tip="单色时点的颜色" off={off("pointTint")}>
        <Swatches label="单色" value={o.pointTint} colors={["#d8d8dc", ...SWATCHES]} onChange={(pointTint) => set({ pointTint })} disabled={off("pointTint")} />
      </Row>
    </>
  );
}

/* 三维没有「压缩预览」这一档：点云超过后台的「视图 · 点云上限」就每 N 个点取一个，
 * 删了多少由视图通知区那一条「显示了 N / 共 M 点」说（不许偷偷抽稀）。 */

export function LinesTab({ o, set, only2d, has, off }: { o: O; set: Patch; only2d: boolean; has: Has; off: Off }) {
  const [lo, hi] = RANGES.lineWidth;
  const drawn = only2d
    ? [has("boxes") && "人物框", has("tracks2d") && "跟踪点轨迹", has("handle") && "节点的控制手柄"]
    : ["选中的框", has("camera") && "相机和相机路径", has("skeleton", "character") && "骨骼", has("curves") && "三维曲线"];
  const what = drawn.filter(Boolean).join("、");
  return (
    <>
      <Row label="线宽" tip={what ? `${what}的线宽，屏幕像素；二维三维共用` : "画面上线的宽度，屏幕像素；二维三维共用"} off={off("lineWidth")}>
        <Slider value={o.lineWidth} onChange={(lineWidth) => set({ lineWidth })} min={lo} max={hi} unit="px" digits={1} disabled={!!off("lineWidth")} />
      </Row>
      <Row label="相机颜色" tip="相机和它路径（实线）的颜色" off={off("cameraColor")}>
        <Swatches label="相机颜色" value={o.cameraColor} colors={SWATCHES} onChange={(cameraColor) => set({ cameraColor })} disabled={off("cameraColor")} />
      </Row>
      <Row label="骨骼颜色" tip="骨架的颜色" off={off("boneColor")}>
        <Swatches label="骨骼颜色" value={o.boneColor} colors={SWATCHES} onChange={(boneColor) => set({ boneColor })} disabled={off("boneColor")} />
      </Row>
      <Row label="曲线着色" tip="三维曲线（发丝、毛发导向线、运动轨迹）用自己带的颜色，还是全部一种颜色；宽度就是上面的线宽" off={off("curveColor")}>
        <Segmented label="曲线着色" stretch value={o.curveColor} options={choices("curveColor", off("curveColor"))} onChange={(curveColor) => set({ curveColor })} />
      </Row>
      <Row label="曲线单色" tip="单色时三维曲线的颜色" off={off("curveTint")}>
        <Swatches label="曲线单色" value={o.curveTint} colors={SWATCHES} onChange={(curveTint) => set({ curveTint })} disabled={off("curveTint")} />
      </Row>
      <Row label="按人物" tip="每人一种颜色，按人物编号取色：2D 是人物框，3D 是透过相机看到的蒙皮角色。在浏览器里立刻生效，不重新计算" off={off("overlayByPerson")}>
        <Switch label="按人物" on={o.overlayByPerson} onChange={(overlayByPerson) => set({ overlayByPerson })} disabled={!!off("overlayByPerson")} />
      </Row>
    </>
  );
}

export function MeshesTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label="着色" tip="平滑：光滑的明暗；平面：每个面一个明暗，看得清面；线框：只画边；带线框：明暗上画出边" off={off("shading")}>
        <Segmented label="着色" stretch value={o.shading} options={choices("shading", off("shading"))} onChange={(shading) => set({ shading })} />
      </Row>
      <Row label="UV 棋盘格" tip="用棋盘格贴图看 UV：格子均匀说明 UV 展得好，蓝色角是 UV 的原点。没有 UV 的网格照常显示" off={off("uvChecker")}>
        <Switch on={o.uvChecker} onChange={(uvChecker) => set({ uvChecker })} disabled={!!off("uvChecker")} />
      </Row>
      <Row label="叠加透明度" tip="透过相机看、背后是原始画面时，模型和角色有多实：1 是不透明。在浏览器里立刻改，不用重新计算" off={off("overlayOpacity")}>
        <Slider value={o.overlayOpacity} onChange={(overlayOpacity) => set({ overlayOpacity })}
          min={RANGES.overlayOpacity[0]} max={RANGES.overlayOpacity[1]} digits={2} disabled={!!off("overlayOpacity")} />
      </Row>
      <Row label="叠加颜色" tip="透过相机看、背后是原始画面时模型和角色的颜色（白模看贴合，亮色在暗画面上更清楚）" off={off("overlayTint")}>
        <Swatches label="叠加颜色" value={o.overlayTint} colors={OVERLAY_SWATCHES} onChange={(overlayTint) => set({ overlayTint })} disabled={off("overlayTint")} />
      </Row>
    </>
  );
}

export function CameraTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  const ortho = useViewCamera((s) => s.ortho);
  const setOrtho = useViewCamera((s) => s.setOrtho);
  return (
    <>
      {/* 正交是相机自己的状态（state/viewer.ts），不是显示选项：三维视图里一直能用 */}
      <Row label="正交" tip="正交投影：没有近大远小，量尺寸、对齐时用。顶、前、侧视图总是正交">
        <Switch on={ortho} onChange={setOrtho} />
      </Row>
      <Row label="近裁剪" tip="比这更近的内容不画，厘米。透过相机看时也用它" off={off("near")}>
        <Num value={o.near} onChange={(near) => set({ near: Math.min(near, o.far * 0.5) })} min={RANGES.near[0]} max={RANGES.near[1]} unit="cm" digits={3} disabled={!!off("near")} />
      </Row>
      <Row label="远裁剪" tip="比这更远的内容不画，厘米" off={off("far")}>
        <Num value={o.far} onChange={(far) => set({ far: Math.max(far, o.near * 2) })} min={RANGES.far[0]} max={RANGES.far[1]} unit="cm" digits={0} disabled={!!off("far")} />
      </Row>
    </>
  );
}

export function SceneTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label="地面网格" tip="地面上的网格，每十格一条粗线" off={off("grid")}>
        <Switch on={o.grid} onChange={(grid) => set({ grid })} disabled={!!off("grid")} />
      </Row>
      <Row label="网格间距" tip="相邻两条网格线相隔多少厘米" off={off("gridSpacing")}>
        <Num value={o.gridSpacing} onChange={(gridSpacing) => set({ gridSpacing })} min={RANGES.gridSpacing[0]} max={RANGES.gridSpacing[1]} unit="cm" disabled={!!off("gridSpacing")} />
      </Row>
      <Row label="网格大小" tip="网格一共多宽，厘米；0 是无边的，远处渐隐" off={off("gridSize")}>
        <Num value={o.gridSize} onChange={(gridSize) => set({ gridSize })} min={RANGES.gridSize[0]} max={RANGES.gridSize[1]} unit="cm" digits={0} disabled={!!off("gridSize")} />
      </Row>
      <Row label="坐标轴" tip="左下角的世界坐标轴：X 红、Y 绿、Z 蓝；点一个轴就沿它看过去" off={off("axes")}>
        <Switch on={o.axes} onChange={(axes) => set({ axes })} disabled={!!off("axes")} />
      </Row>
      <Row label="坐标轴大小" tip="左下角坐标轴有多大，像素" off={off("axesSize")}>
        <Slider value={o.axesSize} onChange={(axesSize) => set({ axesSize: Math.round(axesSize) })} min={RANGES.axesSize[0]} max={RANGES.axesSize[1]} unit="px" digits={0} disabled={!!off("axesSize")} />
      </Row>
      <Row label="背景" tip="背景是纯色还是上亮下暗的渐变。透过相机看时背景是原始画面" off={off("background")}>
        <Segmented label="背景" stretch value={o.background} options={choices("background", off("background"))} onChange={(background) => set({ background })} />
      </Row>
      <Row label="背景色" tip="背景的颜色（渐变时是下面的颜色）" off={off("backgroundColor")}>
        <Swatches label="背景色" value={o.backgroundColor} colors={BACKGROUNDS} onChange={(backgroundColor) => set({ backgroundColor })} disabled={off("backgroundColor")} />
      </Row>
      <Row label="灯光" tip="头灯：一盏跟着镜头的灯，转到哪都亮；三点灯：固定的主光、补光和天光，看得出形体" off={off("lighting")}>
        <Segmented label="灯光" stretch value={o.lighting} options={choices("lighting", off("lighting"))} onChange={(lighting) => set({ lighting })} />
      </Row>
      <Row label="曝光" tip="灯光亮度，档：+1 亮一倍，-1 暗一半。只影响受光的模型" off={off("exposure")}>
        <Slider value={o.exposure} onChange={(exposure) => set({ exposure })} min={-4} max={4} unit="档" digits={1} disabled={!!off("exposure")} />
      </Row>
    </>
  );
}

export function QualityTab({ o, set, off }: { o: O; set: Patch; off: Off }) {
  return (
    <>
      <Row label="抗锯齿" tip="去掉边缘的锯齿。MSAA：多重采样，边缘最干净；FXAA：最快，稍糊；SMAA：比 FXAA 清楚；关：最快，有锯齿" off={off("antialias")}>
        <Segmented label="抗锯齿" stretch value={o.antialias} options={choices("antialias", off("antialias"))} onChange={(antialias) => set({ antialias })} />
      </Row>
    </>
  );
}
