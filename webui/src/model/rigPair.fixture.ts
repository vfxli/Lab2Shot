/** model/rigPair.test.ts 用的手柄数据：一份 rig_pair 的 handle_data（服务端给的形状），两副小骨架。
 * 源：Hips → Spine → Twist（辅助骨，没有部位）；Hips → LeftUpLeg → LeftLeg；Hips → RightUpLeg。
 * 目标：pelvis → spine_01 → spine_02；pelvis → thigh_l → calf_l；pelvis → thigh_r。
 * 另有一份目标是模型固定骨架（没有位置，只有名字、父子、部位）的。 */

import type { RigPairData, RigPairSide } from "../api/status.ts";

const at = (x: number, y: number, z: number): number[] => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1];

const posed = (names: string[], parents: number[], local: number[][], parts: Record<string, number[]>): RigPairSide => ({
  packet: "fp-test", path: "/Rig/Skeleton", names, parents, pose: "绑定姿势", before: { local },
  mirror: [], mirror_plane: null, unknown: [], units: { translate: "cm", rotate: "°" }, rotation: "XYZ", order: "local @ T·R·S",
  parts, fixed: false,
});

export const SRC: RigPairSide = {
  ...posed(["Hips", "Spine", "Twist", "LeftUpLeg", "LeftLeg", "RightUpLeg"], [-1, 0, 1, 0, 3, 0],
    [at(0, 100, 0), at(0, 10, 0), at(0, 5, 0), at(10, 0, 0), at(0, -45, 0), at(-10, 0, 0)],
    { hips: [0], spine: [1], "l.thigh": [3], "l.shin": [4], "r.thigh": [5] }),
  auto_pose: [{ joint: "LeftUpLeg", translate: [0, 0, 0], rotate: [0, 0, 5], scale: [1, 1, 1] }],
};

export const DST: RigPairSide = {
  ...posed(["pelvis", "spine_01", "spine_02", "thigh_l", "calf_l", "thigh_r"], [-1, 0, 1, 0, 3, 0],
    [at(0, 90, 0), at(0, 8, 0), at(0, 8, 0), at(9, 0, 0), at(0, -40, 0), at(-9, 0, 0)],
    { hips: [0], spine: [1, 2], "l.thigh": [3], "l.shin": [4], "r.thigh": [5] }),
  auto_pose: [],
};

export const PARTS = [
  { id: "hips", label: "髋", region: "躯干", chain: false, required: true },
  { id: "spine", label: "脊柱", region: "躯干", chain: true, required: false, end: "chest" },
  { id: "l.thigh", label: "左大腿", region: "左腿", chain: false, required: false },
  { id: "l.shin", label: "左小腿", region: "左腿", chain: false, required: false },
  { id: "r.thigh", label: "右大腿", region: "右腿", chain: false, required: false },
  { id: "r.shin", label: "右小腿", region: "右腿", chain: false, required: false },
];

export const DATA: RigPairData = {
  src: SRC,
  dst: DST,
  parts_table: PARTS,
  auto_mapping: [
    { part: "hips", src: ["Hips"], dst: ["pelvis"] },
    { part: "spine", src: ["Spine"], dst: ["spine_01", "spine_02"] },
    { part: "l.thigh", src: ["LeftUpLeg"], dst: ["thigh_l"] },
    { part: "l.shin", src: ["LeftLeg"], dst: ["calf_l"] },
    { part: "r.thigh", src: ["RightUpLeg"], dst: ["thigh_r"] },
  ],
  rules: [
    { id: "generic", label: "通用（只忽略辅助骨）", ignore: { src: ["Twist"], dst: [] } },
    { id: "star", label: "STaR（固定身体部位）", ignore: { src: ["Twist", "LeftLeg"], dst: ["spine_02"] } },
  ],
  default_rule: "generic",
};

/** 目标是模型固定骨架的一份（骨骼动作家族的「模型骨骼对应」）：没有 before / mirror，只有名字、父子、部位。 */
export const FIXED: RigPairData = {
  src: { ...SRC, auto_pose: undefined },
  dst: { names: ["root", "pelvis", "spine", "lhip", "rhip"], parents: [-1, 0, 1, 1, 1], parts: { hips: [1], spine: [2], "l.thigh": [3], "r.thigh": [4] }, fixed: true },
  parts_table: PARTS,
  auto_mapping: [
    { part: "hips", src: ["Hips"], dst: ["pelvis"] },
    { part: "spine", src: ["Spine"], dst: ["spine"] },
    { part: "l.thigh", src: ["LeftUpLeg"], dst: ["lhip"] },
    { part: "r.thigh", src: [], dst: ["rhip"] },
  ],
};
