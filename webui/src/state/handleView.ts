import { create } from "zustand";
import { askStatus } from "../graph/asking";

/** 三维舞台在显示节点的结果之外还画什么，属于「怎么看」：不进文档、不记撤销、换图时清空（graph/document.ts 打开图时
 * reset）。
 * - `editing`：主视图里正在改的那个「骨架姿势」手柄（参数面板上姿势一行的「在视图里改」）。主视图默认不画姿势手柄，
 *   只画这一个，并且只在它的节点是显示节点时画（view/Stage3D.tsx）。
 * - `dialog`：打开着的「对应关系」弹窗的节点：弹窗的舞台画它的手柄，但不因此改显示节点。
 * - `reference`：参考显示（view/stageLayers.tsx ReferenceLayer），查看器工具栏的「参考」菜单选。
 * 手柄数据（状态回复 handle_data）要谁的由 `handleNode` 定，状态请求带着它（StatusRequest handle_node，graph/actions.ts
 * refreshStatus）；它变了就再问一次服务器。 */

/** 参考显示：另一个节点的结果（它的某个口；或它的某个手柄的数据，如「骨架姿势」的基准骨架）半透明、另一种颜色叠在当前
 * 显示的内容上，不可操作。按包类型照常画（角色、骨架、模型、点云、相机、曲线）。 */
export interface Reference {
  node: string;
  port?: string | null; // 哪个口（空：它的主输出）
  handle?: number | null; // 改看它的这个手柄的数据（NodeDef.handles 的下标）
}

interface State {
  editing: { node: string; handle: number } | null;
  dialog: string | null;
  reference: Reference | null;
  setEditing: (e: State["editing"]) => void;
  setDialog: (node: string | null) => void;
  setReference: (r: Reference | null) => void;
  reset: () => void;
}

/** 手柄数据要谁的：开着的弹窗优先，其次主视图里在改的手柄；都没有为 ""，状态请求就不带 handle_node，服务器不发手柄数据
 * （要手柄数据的只有这两处：弹窗的舞台与骨长比例条、主视图里在改的那一副）。 */
export const handleNode = (s: Pick<State, "editing" | "dialog"> = useHandleView.getState()): string => s.dialog ?? s.editing?.node ?? "";

export const useHandleView = create<State>((set, get) => {
  // 手柄数据换了节点：再问一次（状态请求带上新的 handle_node）
  const put = (patch: Partial<State>) => {
    const was = handleNode(get());
    set(patch);
    if (handleNode(get()) !== was) void askStatus();
  };
  return {
    editing: null,
    dialog: null,
    reference: null,
    setEditing: (editing) => put({ editing }),
    setDialog: (dialog) => put({ dialog }),
    setReference: (reference) => set({ reference }),
    reset: () => set({ editing: null, dialog: null, reference: null }),
  };
});
