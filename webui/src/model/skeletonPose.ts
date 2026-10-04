/** 骨架姿势修正（手柄种类「骨架姿势」，lab2shot retarget 的 motion_pose / target_pose）的纯算术：一行修正怎么变成
 * 矩阵、怎么经前向运动学（FK）传到子关节、一次拖动的增量怎么换成新的一行、左右镜像（在世界里关于人物的对称面）。舞台
 * （view/skeletonPose.tsx）只画、只收拖动，算术都在这里；矩阵与旋转顺序用 model/math3d.ts 那一份。服务端用同一个约定：
 *
 *   修正后局部 = 基准局部 @ D，D = T(translate) · R(rotate) · S(scale)
 *   R 按服务端声明的旋转顺序（handle_data.rotation；「XYZ」= Rz·Ry·Rx），S 为逐轴缩放；D 在关节自己的坐标系里（cm）。
 *   世界 = 父世界 @ 修正后局部；根的世界 = 修正后局部。
 *
 * 缩放向子关节传递（与服务端一致），所以父关节非等比缩放后子关节的世界带剪切：拖动只写增量（draggedRow），不把手柄
 * 的 TRS 当成关节的世界矩阵换回局部。合法域：各分量有限，缩放每轴 > 0（validRow）；越界的一行不写（withRow）。 */

import { IDENTITY, compose, decompose, exactTRS, inverse, mirroredChange, mul, reflectionAbout, similarity, nearestAngles, rotation, roundChanged, turn, turnedAngles, turnIn, type Increment, type M4 } from "./math3d";
import { t } from "../i18n/t";

export type { M4 } from "./math3d";

export interface PoseRow {
  joint: string;
  translate: [number, number, number]; // cm
  rotate: [number, number, number]; // 度，[绕 X, 绕 Y, 绕 Z]，按声明的旋转顺序
  scale: [number, number, number]; // 倍数，> 0
}

/** 一副骨架（服务端 handle_data 的那几项，api/status.ts SkeletonPoseData 满足它）。 */
export interface PoseSkeleton {
  names: string[];
  parents: number[]; // -1 为根，父在子前
  before: { local: number[][] }; // 基准局部（修正前）
  rotation: string; // 旋转顺序
}

/** 一行修正的 D = T·R·S。 */
export const deltaOf = (row: Pick<PoseRow, "translate" | "rotate" | "scale">, order: string): M4 => compose(row.translate, row.rotate, row.scale, order);

/** 把 D 拆回一行（D 由 T·R·S 来，没有剪切）。 */
export function rowOf(joint: string, d: M4, order: string): PoseRow {
  return { joint, ...decompose(d, order) };
}

/** 一行在合法域里：各分量有限，缩放每轴 > 0（缩放 0 让关节塌成一点、矩阵不可逆；负缩放是镜像，不是这种手柄的操作）。 */
export const validRow = (r: PoseRow): boolean =>
  [...r.translate, ...r.rotate, ...r.scale].every(Number.isFinite) && r.scale.every((v) => v > 0);

const EPS = 1e-4;
/** 这一行等于没改（全零平移与转角、全一缩放）：写回时略去。 */
export const isIdentity = (r: PoseRow): boolean =>
  r.translate.every((v) => Math.abs(v) < EPS) && r.rotate.every((v) => Math.abs(v) < EPS) && r.scale.every((v) => Math.abs(v - 1) < EPS);

/** 写回用：略去没改的行。数值不在这里取整：取整只落在一次操作改动的分量上（draggedRow、mirroredRow）。 */
export const tidy = (rows: PoseRow[]): PoseRow[] => rows.filter((row) => !isIdentity(row));

/** 前向运动学：各关节修正后的局部与世界。 */
export function forward(sk: PoseSkeleton, rows: PoseRow[]): { local: M4[]; world: M4[] } {
  const by = new Map(rows.map((r) => [r.joint, r]));
  const local: M4[] = [];
  const world: M4[] = [];
  sk.before.local.forEach((base, i) => {
    const row = by.get(sk.names[i]);
    const l = row ? mul(base, deltaOf(row, sk.rotation)) : base;
    local.push(l);
    world.push(sk.parents[i] >= 0 ? mul(world[sk.parents[i]], l) : l);
  });
  return { local, world };
}

/** 关节 i 的轴架：父世界 @ 基准局部（它的一行修正在这里面算）。 */
const frameAt = (sk: PoseSkeleton, i: number, world: M4[]): M4 =>
  mul(sk.parents[i] >= 0 ? world[sk.parents[i]] : IDENTITY, sk.before.local[i]);

/** 关节 i 现在能不能改（拖手柄、镜像到它）：它的轴架退化（上级关节或它自己的基准姿势缩放为 0，math3d.inverse 判
 * 不可逆）时方向定不下来，给出原因；"" 能改。手柄与面板都按它禁用并说明。 */
export function stuckWhy(sk: PoseSkeleton, i: number, rows: PoseRow[]): string {
  return inverse(frameAt(sk, i, forward(sk, rows).world)) ? "" : t("ui.rig.pose_stuck");
}

/** 放进 / 换掉一行（没改的删掉）；不在合法域里的一行不放，原样返回。 */
export function withRow(rows: PoseRow[], row: PoseRow): PoseRow[] {
  if (!validRow(row)) return rows;
  const rest = rows.filter((r) => r.joint !== row.joint);
  return isIdentity(row) ? rest : [...rest, row];
}

/** 一行的初值（没改）。 */
export const identityRow = (joint: string): PoseRow => ({ joint, translate: [0, 0, 0], rotate: [0, 0, 0], scale: [1, 1, 1] });

/** 关节 i 按一次拖动的增量（世界里，view/dragGizmo.tsx）改出的新一行；增量把缩放拖到 ≤ 0 时为 null（不写）。
 * 只动增量碰到的分量，取整也只落在它们上：
 *   平移：关节原点在世界里走了 move，局部 Δt = inv(父世界 @ 基准局部 的线性部分) · move（有剪切也准）；
 *   旋转：世界里绕关节转了 turn，换到 父世界 @ 基准局部 的真实轴里（math3d.turnIn，带镜像的轴也对）左乘到 R 上，
 *        按声明顺序取离原转角最近的一组角（上游没有非等比缩放时与世界转角完全一致）；
 *   缩放：手柄沿自己的轴（= 关节的轴）的倍数，逐轴乘到 S 上。
 * 轴架退化（stuckWhy）时也是 null。 */
export function draggedRow(sk: PoseSkeleton, i: number, rows: PoseRow[], d: Increment): PoseRow | null {
  const joint = sk.names[i];
  const row = rows.find((r) => r.joint === joint) ?? identityRow(joint);
  const f = frameAt(sk, i, forward(sk, rows).world);
  const inv = inverse(f);
  if (!inv) return null; // 轴架退化（stuckWhy）：不写
  const step = turn(inv, d.move);
  const translate = roundChanged(row.translate.map((v, k) => v + step[k]), row.translate, 1000);
  const rotate = roundChanged(turnedAngles(sk.rotation, turnIn(f, d.turn), row.rotate), row.rotate, 1000);
  const scale = roundChanged(row.scale.map((v, k) => v * d.grow[k]), row.scale, 10000);
  const next = { joint, translate, rotate, scale };
  return validRow(next) ? next : null;
}

/** 人物的左右对称面（世界里）：法向是人物的「左」，过两大腿根的中点（服务端 handle_data.mirror_plane）。 */
export interface MirrorPlane {
  normal: [number, number, number];
  point: [number, number, number];
}

/** 把关节 i 自己的修正镜像到对侧关节 j，在世界里做（各家骨架左右关节的坐标系一般不对称，按局部翻符号是错的）：
 * i 这一行的修正 D_i 在基准世界里反射过人物的对称面，再换进 j 的基准世界，就是 j 的一行（math3d.mirroredChange：完整
 * 仿射，含缩放）。只看 i 自己这一行、只用基准姿势：对称面是按基准姿势算的（服务端 handle_data.mirror_plane），上级
 * 关节的修正（人物整体转身）不会被一起镜像。
 * 镜过去不是 T·R·S 形状（两侧关节的轴不互为镜像，而修正里有非等比缩放：缩放轴落不到 j 自己的轴上，成了剪切），或
 * 某一边的基准世界不可逆（缩放为 0），就给出原因（`why`），不写一个近似的错行。 */
export function mirroredRow(sk: PoseSkeleton, i: number, j: number, plane: MirrorPlane, rows: PoseRow[]): { row: PoseRow } | { why: string } {
  const before = forward(sk, []).world;
  const own = rows.find((r) => r.joint === sk.names[i]) ?? identityRow(sk.names[i]);
  const d = mirroredChange(deltaOf(own, sk.rotation), before[i], before[j], reflectionAbout(plane.normal, plane.point));
  if (!d) return { why: t("ui.rig.pose_mirror_zero") };
  const trs = exactTRS(d, sk.rotation);
  if (!trs) {
    // 两种原因分开说：两侧的基准本身不互为镜像（i 的基准经对称面到 j 的基准不是相似变换：某一侧带非等比缩放或剪切），
    // 什么修正镜过去都会变形；基准互为镜像，只是两侧的轴向不对应，这一行的非等比缩放镜过去落不到 j 自己的轴上
    const back = inverse(before[j]);
    const across = back && mul(back, mul(reflectionAbout(plane.normal, plane.point), before[i]));
    return { why: across && similarity(across)
      ? t("ui.rig.pose_mirror_axes")
      : t("ui.rig.pose_mirror_base") };
  }
  const was = rows.find((r) => r.joint === sk.names[j]) ?? identityRow(sk.names[j]);
  return { row: {
    joint: sk.names[j],
    translate: roundChanged(trs.translate, was.translate, 1000),
    rotate: roundChanged(nearestAngles(sk.rotation, rotation(sk.rotation, trs.rotate), was.rotate), was.rotate, 1000),
    scale: roundChanged(trs.scale, was.scale, 10000),
  } };
}

/** 参数值读成行（不认得的形状略去；不在合法域里的一行，如缩放 0，也略去：与 withRow 同一个域）。 */
export function rowsOf(value: unknown): PoseRow[] {
  if (!Array.isArray(value)) return [];
  const v3 = (x: unknown, d: number): [number, number, number] =>
    Array.isArray(x) && x.length === 3 && x.every((n) => typeof n === "number") ? (x as [number, number, number]) : [d, d, d];
  return value.flatMap((r) => (r && typeof r === "object" && typeof (r as PoseRow).joint === "string"
    ? [{ joint: (r as PoseRow).joint, translate: v3((r as PoseRow).translate, 0), rotate: v3((r as PoseRow).rotate, 0), scale: v3((r as PoseRow).scale, 1) }]
    : [])).filter(validRow);
}
