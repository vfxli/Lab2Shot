/** 一份三维场景在舞台上的画法：按种类分发给各自的绘制组件（elements3d / points3d / curves3d）。 */

import { CameraPath, Character, ModelMesh, ShotCamera } from "./elements3d";
import { Cloud } from "./points3d";
import { Gaussian } from "./gaussian3d";
import { Curves } from "./curves3d";
import type { Scene } from "./sceneData";
import type { Kind } from "./kinds3d";
import type { ViewOptions } from "../model/viewOptions";

/** 一份场景里的全部东西（模型、角色与骨架、点云、三维曲线、相机与路径），按显示 / 隐藏开关画。逐帧才有样子的东西
 * （角色、相机）一帧都没有时不画（model/timelineMath.ts sampleAt 为 -1）；模型、点云、曲线有静止形状或取最近的样本。`through`：透过相机看
 * 时的叠加画法（颜色与透明度取显示选项的叠加色，也是参考显示的画法：view/stageLayers.tsx ReferenceLayer）。 */
export function SceneElement({ item, d, frame, hidden, through, look, o }: { item: { key: string }; d: Scene; frame: number; hidden: Set<string>; through: boolean; look: string | null; o: ViewOptions }) {
  const shown = (k: Kind) => !hidden.has(k);
  return (
    <>
      {shown("model") && d.models.map((m, i) => <ModelMesh key={i} m={m} frame={frame} o={o} pickKey={`${item.key}/model/${i}`} through={through} version={d.version} />)}
      {d.characters.map((c, i) => c.ref.frames.length > 0 && (
        <Character key={i} c={c} frame={frame} o={o} pickKey={`${item.key}/character/${i}`} through={through} meshes={shown("character")} bones={shown("skeleton")} person={c.ref.person} />
      ))}
      {shown("points") && d.clouds.filter((c) => !c.ref.gaussian).map((c) => <Cloud key={c.key} src={c} frame={frame} o={o} pickKey={`${item.key}/points/${c.name}`} version={d.version} />)}
      {shown("gaussian") && d.clouds.filter((c) => c.ref.gaussian).map((c) => <Gaussian key={c.key} src={c} frame={frame} pickKey={`${item.key}/gaussian/${c.name}`} version={d.version} />)}
      {shown("curves") && d.curves.map((c) => <Curves key={c.key} src={c} frame={frame} o={o} pickKey={`${item.key}/curves/${c.name}`} version={d.version} />)}
      {shown("camera") &&
        d.cameras.map((cam) =>
          `${d.key}|${cam.ref.path}` === look || !cam.ref.frames.length ? null : ( // 不画正在透过它看的那台相机：视图就在它里面
            <group key={cam.ref.path}>
              <ShotCamera cam={cam} frame={frame} o={o} pickKey={`${item.key}/camera${cam.ref.path}`} />
              {cam.ref.frames.length > 1 && o.cameraPath && <CameraPath cam={cam} o={o} />}
            </group>
          ),
        )}
    </>
  );
}
