/** 「在视图里点选」：带手柄的参数（widget picks / canvas，例如「选人」的点选）在 2D 视图里点，而手柄只画在视图显示的那个
 * 节点上（view/plan.ts handles）。打开卡片、显示一个节点（双击、点「计算」）都不进点选：使用者点「在视图里点选」
 * （editor/buttonActions.tsx pick_in_view；或碰公开参数树里 show_on_change 的那一行，ParamPanel.tsx PickRow）才进
 * （handleView picking），视图显示这个节点、它的二维手柄画、接（pickingOn）。视图换显示别的、点「退出编辑」就结束。
 *
 * 另记的是「进入前视图显示的是什么」（enterPicking 记下），给按钮的「点这里结束」用：回到进入前显示的节点和层；没有
 * 记录（进入时已经在显示它）就不动 */

import { useHandleView } from "./handleView";
import { useLook } from "./look";
import { useViewer } from "./viewer";

/** 正在视图里点选这个节点的这个参数吗：使用者点「在视图里点选」进了它（handleView picking）、视图显示的就是它，且它的手柄
 * 现在可用。 */
export const pickingOn = (displayId: string | null, nodeId: string, handleActive: boolean, picking: string | null = null): boolean =>
  displayId === nodeId && handleActive && picking === nodeId;

// 点选只对进入时显示的那个节点算数：视图一换显示别的，就结束（再点「在视图里点选」才回来）
useLook.subscribe((s, prev) => {
  const h = useHandleView.getState();
  if (s.displayId !== prev.displayId && h.picking && h.picking !== s.displayId) h.setPicking(null);
});

// `doc`：记下时打开的是哪份文档（state/viewer.ts docId）——换了文档，另一份里同 id 的节点不认这条记录
let before: { doc: string; node: string; displayId: string | null; displayPort: string | null } | null = null;

const recorded = (nodeId: string) => !!before && before.doc === useViewer.getState().docId && before.node === nodeId && before.displayId !== nodeId;

/** 进入点选：让视图显示这个节点（已经在显示它就不动、不改记录），记下之前显示的，它的二维手柄从此画、接。 */
export function enterPicking(nodeId: string): void {
  const look = useLook.getState();
  if (look.displayId !== nodeId) {
    before = { doc: useViewer.getState().docId, node: nodeId, displayId: look.displayId, displayPort: look.displayPort };
    look.setDisplay(nodeId);
  }
  useHandleView.getState().setPicking(nodeId);
}

/** 退出点选：手柄不再画、不再接；视图还在显示这个节点、且记得进入前显示的是别的，就回到那里，否则视图照常显示它。 */
export function leavePicking(nodeId: string): void {
  const look = useLook.getState();
  const was = before;
  const back = look.displayId === nodeId && recorded(nodeId);
  before = null;
  if (useHandleView.getState().picking === nodeId) useHandleView.getState().setPicking(null);
  if (!was || !back) return;
  look.setDisplay(was.displayId);
  if (was.displayPort) look.setDisplayPort(was.displayPort);
}

/** 退出视图里正在进行的操作（参数面板上的按钮打开的：骨架关系 / 姿势编辑，handleView editing；或显示节点的二维手柄：
 * 点选、框、画遮罩）。视图工具栏的「退出编辑」和变蓝的按钮再点一下，都走这一个。 */
export function exitViewOperation(displayId: string | null): void {
  const h = useHandleView.getState();
  if (h.editing) return h.setEditing(null);
  if (displayId) leavePicking(displayId);
}
