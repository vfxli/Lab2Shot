import { create } from "zustand";
import { askStatus } from "../graph/asking";

/** 三维舞台在显示节点的结果之外还画什么，属于「怎么看」：不进文档、不记撤销、换图时清空（graph/document.ts 打开图时
 * reset）。
 * - `editing`：主视图里正在改的那个手柄（参数面板上「在视图里改」「在视图里编辑」打开的）：「骨架姿势」手柄画那一副骨架，
 *   「rig_pair」手柄进入双骨架编辑（view/rigPair.tsx）。主视图默认不画这两种手柄，只画这一个，并且只在它的节点是显示
 *   节点时画（view/Stage3D.tsx）。
 * - `reference`：参考显示（view/stageLayers.tsx ReferenceLayer），查看器工具栏的「参考」菜单选。
 * 手柄数据（状态回复 handle_data）要谁的由 `handleNode` 定，状态请求带着它（StatusRequest handle_node，graph/actions.ts
 * refreshStatus）；它变了就再问一次服务器。 */

/** 参考显示：另一个节点的结果（它的某个口）半透明、另一种颜色叠在当前显示的内容上，不可操作。按包类型照常画（角色、
 * 骨架、模型、点云、相机、曲线）。 */
export interface Reference {
  node: string;
  port?: string | null; // 哪个口（空：它的主输出）
}

interface State {
  editing: { node: string; handle: number } | null;
  // 正在视图里点选的节点：只有它的二维手柄（点选、框、画遮罩……）画、接。打开卡片、显示一个节点都不进点选，使用者点
  // 「在视图里点选」才进（state/viewPicking.ts）；视图换显示别的节点、点「退出编辑」就结束
  picking: string | null;
  reference: Reference | null;
  setEditing: (e: State["editing"]) => void;
  setPicking: (node: string | null) => void;
  setReference: (r: Reference | null) => void;
  reset: () => void;
}

/** 手柄数据要谁的：主视图里在改的手柄的节点；没有为 ""，状态请求就不带 handle_node，服务器不发手柄数据。 */
export const handleNode = (s: Pick<State, "editing"> = useHandleView.getState()): string => s.editing?.node ?? "";

export const useHandleView = create<State>((set, get) => {
  // 手柄数据换了节点：再问一次（状态请求带上新的 handle_node）
  const put = (patch: Partial<State>) => {
    const was = handleNode(get());
    set(patch);
    if (handleNode(get()) !== was) void askStatus();
  };
  return {
    editing: null,
    picking: null,
    reference: null,
    setEditing: (editing) => put({ editing }),
    setPicking: (picking) => set({ picking }),
    setReference: (reference) => set({ reference }),
    reset: () => set({ editing: null, picking: null, reference: null }),
  };
});
