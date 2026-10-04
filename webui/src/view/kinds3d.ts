/** 三维数据种类的唯一定义处：有哪些种类、显示名称，某个显示计划画了哪些种类，以及点云的抽稀情况。 */

import { useScenes, type Scene } from "./sceneData";
import type { DisplayPlan } from "./plan";
import { drawsSkeleton } from "./handleEditing";

/** 三维数据的种类及其显示名称的键（t() 取当前语言），顺序与视图的显示 / 隐藏开关一致。 */
export const KINDS = { camera: "ui.view.kind.camera", model: "ui.view.kind.model", character: "ui.view.kind.character", skeleton: "ui.view.kind.skeleton", points: "ui.view.kind.points", gaussian: "ui.view.kind.gaussian", curves: "ui.view.kind.curves" } as const;
export type Kind = keyof typeof KINDS;

/** 场景中包含的数据种类。 */
function kindsOf(d: Scene): Set<Kind> {
  const out = new Set<Kind>();
  if (d.cameras.length) out.add("camera");
  if (d.models.length) out.add("model");
  if (d.characters.some((c) => c.meshes.length)) out.add("character");
  if (d.characters.length) out.add("skeleton");
  if (d.clouds.some((c) => !c.ref.gaussian)) out.add("points");
  if (d.clouds.some((c) => c.ref.gaussian)) out.add("gaussian");
  if (d.curves.length) out.add("curves");
  return out;
}


/** 这几个 hook 只看计划里画了什么：没有显示节点时，编辑器传一份三个字段都为空的计划，类型在这里约束，
 * 少一个字段就编译不过。 */
export type PlanShown = Pick<DisplayPlan, "elements" | "pointMaps" | "handles">;

/** 节点三维结果中包含的数据种类，供显示 / 隐藏开关使用。 */
export function useSceneKinds(plan: PlanShown): Kind[] {
  const fps = plan.elements.flatMap((e) => (e.fp ? [e.fp] : []));
  const [loaded] = useScenes(fps);
  const kinds = new Set<Kind>();
  for (const d of loaded.values()) for (const k of kindsOf(d)) kinds.add(k);
  if (plan.pointMaps.some((m) => m.fp)) kinds.add("points");
  return (Object.keys(KINDS) as Kind[]).filter((k) => kinds.has(k));
}


/** 显示选项面板判断适用性用的另外几件事（不是显示 / 隐藏开关的种类）：有没有多帧相机（相机路径）、有没有骨架姿势手柄
 * （它画的骨架也受骨骼粗细、骨点大小管）。 */
export function useSceneShows(plan: PlanShown): string[] {
  const fps = plan.elements.flatMap((e) => (e.fp ? [e.fp] : []));
  const [loaded] = useScenes(fps);
  const out: string[] = [];
  if ([...loaded.values()].some((d) => d.cameras.some((c) => c.ref.frames.length > 1))) out.push("cameraPath");
  if (plan.handles.some(drawsSkeleton)) out.push("skeleton");
  return out;
}

/** 视图中当前绘制的点云 / 3D 高斯的抽稀情况（见 server/view_data.py `_proxy_step` / `_point_step` / `_gaussian_step`）。
 *
 * 抽稀始终生效，不提供开关：单帧超过后台「点云上限」（设置项 view.points_max_mb，默认 5 MB）时点云每 N 个点取一个，
 * 单帧超过「高斯显示上限」（设置项 view.gaussian_max，默认 30 万个）时 3D 高斯每 N 个取一个。
 *
 * 抽稀仅作用于显示副本，坐标不做任何修改，计算与交付的点数不受影响。
 * `every` 大于 1 时视图通知区必须持续标示（「显示了 N / 共 M 点」），不得静默抽稀。
 * 本函数仅提供数值，提示文字由编辑器层生成（view 层不依赖 ui 层）。 */
interface CloudProxy {
  every: number;  // 抽样间隔（总点数 ÷ 绘制点数）；1 表示未抽稀
  shown: number;  // 视图中实际绘制的点数（所有点云合计）
  total: number;  // 数据中的总点数
}

export function useCloudProxy(plan: PlanShown): CloudProxy {
  const fps = plan.elements.flatMap((e) => (e.fp ? [e.fp] : []));
  const [loaded] = useScenes(fps);
  let shown = 0;
  let total = 0;
  for (const d of loaded.values())
    for (const c of d.clouds) {
      if (c.ref.grid) {
        // 格网点云：每格一个点。服务器发送的已是抽稀后的数据（gw×gh）
        const use = c.ref.grid;
        const step = use.proxy ?? 1;
        const here = use.gw * use.gh;
        shown += here;
        total += here * step * step;  // 抽样后为 gw×gh，原始点数为其乘以 step²（横纵两个方向各抽样一次）
      } else {
        // 普通点云与 3D 高斯都走这里：count 是完整点数，every 是抽样间隔（server/view_data.py _point_step / _gaussian_step）
        const here = c.ref.count ?? 0;
        const step = c.ref.every ?? 1;
        shown += Math.ceil(here / step);
        total += here;
      }
    }
  return { every: shown ? Math.max(1, Math.round(total / shown)) : 1, shown, total };
}
