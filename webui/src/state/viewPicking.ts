/** 「在视图里点选」：带手柄的参数（widget picks / canvas，例如「选人」的点选）在 2D 视图里点，而手柄只画在视图显示的那个
 * 节点上（view/plan.ts handles）。所以「正在点选」不是另存的一个开关，而是一个事实：**视图现在显示的节点就是这个参数的
 * 节点，且它的手柄现在可用**（pickingOn）。谁让视图显示了它都一样：按钮参数「在视图里点选」（editor/buttonActions.tsx
 * pick_in_view）、「修改后在视图里显示这个节点」（ParamPanel.tsx show_on_change）、节点模式里双击节点；谁让视图显示了
 * 别的（点「计算」、双击别的节点、换模式后显示别的），它就自然不在点了。没有「结束点选」的专门代码。
 *
 * 唯一另记的是「进入前视图显示的是什么」（enterPicking 记下），给按钮的「点这里结束」用：回到进入前显示的节点和层；没有
 * 记录（例如是双击进来的）就不动。 */

import { useLook } from "./look";
import { useViewer } from "./viewer";

/** 正在视图里点选这个节点的这个参数吗：视图显示的就是它，且它的手柄现在可用。 */
export const pickingOn = (displayId: string | null, nodeId: string, handleActive: boolean): boolean =>
  displayId === nodeId && handleActive;

// `doc`：记下时打开的是哪份文档（state/viewer.ts docId）——换了文档，另一份里同 id 的节点不认这条记录
let before: { doc: string; node: string; displayId: string | null; displayPort: string | null } | null = null;

const recorded = (nodeId: string) => !!before && before.doc === useViewer.getState().docId && before.node === nodeId && before.displayId !== nodeId;

/** 让视图显示这个节点（同双击），并记下之前显示的（已经在显示它就不动、不改记录）。 */
export function enterPicking(nodeId: string): void {
  const look = useLook.getState();
  if (look.displayId === nodeId) return;
  before = { doc: useViewer.getState().docId, node: nodeId, displayId: look.displayId, displayPort: look.displayPort };
  look.setDisplay(nodeId);
}

/** 「点这里结束」能不能回去：视图在显示这个节点，且记得进入前显示的是别的节点（双击进来的没有记录：按钮只写「正在视图里点选…」）。 */
export function canLeave(displayId: string | null, nodeId: string): boolean {
  return displayId === nodeId && recorded(nodeId);
}

/** 「点这里结束」：视图还在显示这个节点、且记得进入前显示的是别的，就回到那里；否则不动。 */
export function leavePicking(nodeId: string): void {
  const look = useLook.getState();
  const was = before;
  const back = look.displayId === nodeId && recorded(nodeId);
  before = null;
  if (!was || !back) return;
  look.setDisplay(was.displayId);
  if (was.displayPort) look.setDisplayPort(was.displayPort);
}
