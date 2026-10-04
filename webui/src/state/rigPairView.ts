import { create } from "zustand";
import type { Col } from "../model/rigPair";

/** 双骨架编辑（手柄「rig_pair」，view/rigPair.tsx）的页面状态：只影响怎么看、点到哪一步，不进文档、不记撤销。
 * 属于正在编辑的那个手柄（`of` = 节点 + 手柄序号）：换了手柄就回到初值（at）。「前后距离」是显示选项
 * （model/viewOptions.ts rigPairGap），不在这里。
 * - `mode`：编辑关系 / 编辑姿态 / 选忽略；
 * - `first`：编辑关系时先点的那个骨点（再点另一侧的骨点就配对）；
 * - `picked`：编辑姿态时选中的关节（出手柄、下方填数）；
 * - `hover`：树里或视图里悬停的骨点：视图与树互相高亮；
 * - `hidden`：每侧点了眼睛的关节（它和子孙不画、树里变暗），只影响显示；
 * - `menu`：先点的骨点没有部位时弹出的部位小菜单（屏幕位置与要配的两个骨点）。 */

export type RigMode = "map" | "pose" | "ignore";
export interface JointAt {
  col: Col;
  joint: string;
}

interface State {
  of: string;
  mode: RigMode;
  first: JointAt | null;
  picked: JointAt | null;
  hover: JointAt | null;
  hidden: Record<Col, string[]>;
  menu: { x: number; y: number; first: JointAt; second: JointAt } | null;
  at: (of: string) => void;
  set: (patch: Partial<Omit<State, "of" | "at" | "set" | "toggleHidden" | "clearPick">>) => void;
  toggleHidden: (col: Col, joint: string) => void;
  clearPick: () => void;
}

const FRESH = { mode: "map" as RigMode, first: null, picked: null, hover: null, hidden: { src: [], dst: [] }, menu: null };

export const useRigPairView = create<State>((set, get) => ({
  of: "",
  ...FRESH,
  at: (of) => { if (get().of !== of) set({ of, ...FRESH, mode: get().mode }); },
  set: (patch) => set(patch),
  toggleHidden: (col, joint) => set((s) => ({
    hidden: { ...s.hidden, [col]: s.hidden[col].includes(joint) ? s.hidden[col].filter((n) => n !== joint) : [...s.hidden[col], joint] },
  })),
  clearPick: () => set({ first: null, picked: null, menu: null }),
}));
