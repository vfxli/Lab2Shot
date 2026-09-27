import { loadScene, useLoaded, type Scene } from "./sceneData";
import type { DisplayPlan } from "./plan";

/** 三维数据的种类及其显示名称，顺序与视图的显示 / 隐藏开关一致。 */
export const KINDS = { camera: "相机", model: "模型", character: "蒙皮角色", skeleton: "骨架", points: "点云", curves: "三维曲线", lights: "灯光" } as const;
export type Kind = keyof typeof KINDS;

/** 场景中包含的数据种类。 */
export function kindsOf(d: Scene): Set<Kind> {
  const out = new Set<Kind>();
  if (d.cameras.length) out.add("camera");
  if (d.models.length) out.add("model");
  if (d.characters.some((c) => c.meshes.length)) out.add("character");
  if (d.characters.length) out.add("skeleton");
  if (d.clouds.length) out.add("points");
  if (d.curves.length) out.add("curves");
  return out;
}


/** 节点三维结果中包含的数据种类，供显示 / 隐藏开关使用。 */
export function useSceneKinds(plan: DisplayPlan): Kind[] {
  const fps = plan.elements.flatMap((e) => (e.fp ? [e.fp] : []));
  const [loaded] = useLoaded(fps, loadScene);
  const kinds = new Set<Kind>();
  for (const d of loaded.values()) for (const k of kindsOf(d)) kinds.add(k);
  if (plan.pointMaps.some((m) => m.fp)) kinds.add("points");
  return (Object.keys(KINDS) as Kind[]).filter((k) => kinds.has(k));
}


/** 视图中当前绘制的点云的抽稀情况（见 server/view_data.py `_proxy_step` / `_point_step`）。
 *
 * 抽稀始终生效，不提供开关：单帧超过后台「点云上限」（设置项 view.points_max_mb，默认 5 MB）时每 N 个点取一个。
 *
 * 抽稀仅作用于显示副本，坐标不做任何修改，计算与交付的点数不受影响。
 * `every` 大于 1 时视图通知区必须持续标示（「显示了 N / 共 M 点」），不得静默抽稀。
 * 本函数仅提供数值，提示文字由编辑器层生成（view 层不依赖 ui 层）。 */
export interface CloudProxy {
  every: number;  // 抽样间隔（总点数 ÷ 绘制点数）；1 表示未抽稀
  shown: number;  // 视图中实际绘制的点数（所有点云合计）
  total: number;  // 数据中的总点数
}

export function useCloudProxy(plan: DisplayPlan): CloudProxy {
  const fps = plan.elements.flatMap((e) => (e.fp ? [e.fp] : []));
  const [loaded] = useLoaded(fps, loadScene);
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
        const here = c.ref.count ?? 0;
        const step = c.ref.every ?? 1;
        shown += Math.ceil(here / step);
        total += here;
      }
    }
  return { every: shown ? Math.max(1, Math.round(total / shown)) : 1, shown, total };
}
