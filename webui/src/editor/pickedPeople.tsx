/** 「点选」说成「N 号」（规则在 model/pickChips.ts）：「选人」的点选参数存的是点击（"帧:x,y"，lab2shot/nodes/handles.py parse_picks），服务端按
 * 「点在谁的框里」选人（ops at_point）。网页有同一条规则（model/people.ts personAt，`lab2shot check people` 两边比对），
 * 这里用它对手柄输入上的人物框逐个点击算出点的是几号，参数面板（ParamControls.tsx DrawnList）和节点上（GraphNode.tsx）
 * 都这样说。只是显示：参数格式不变，计算仍在服务端按坐标算。
 *
 * 上游人物框还没算出来（或是列表、这里不拆）时说不出几号：退回「第 F 帧 (x, y)」并说明先检测人物。同一个人点了几次只
 * 列一次，标「×次数」。 */

import type { BoxesData } from "../api";
import { useDescribed } from "../transfer/described";
import { useGraphSnapshot } from "../graph/snapshot";
import { handlesOf } from "../graph/rules";
import { pickChips, pickSummary } from "../model/pickChips";
import { handleSourceFp } from "../view/plan";

/** 这个节点「点选」参数（`param`）的手柄输入上的人物框：null = 还没有（没接、没算、是列表）。 */
export function usePickBoxes(nodeId: string, param: string): BoxesData | null {
  const snap = useGraphSnapshot();
  const handle = handlesOf(snap, nodeId).find((h) => h.kind === "person" && Object.values(h.params).includes(param))
    ?? snap.nodeDefs[snap.nodes.find((n) => n.id === nodeId)?.data.typeId ?? ""]?.handles.find((h) => h.kind === "person" && Object.values(h.params).includes(param));
  const src = handle?.source ? handleSourceFp(snap, nodeId, handle.source) : null;
  const fp = src && !src.list ? src.fp : null;
  return useDescribed<BoxesData>("boxes", fp ? [fp] : [])[0] ?? null;
}

/** 节点上「选人」点选的摘要一行（GraphNode.tsx）：点人手柄现在生效（mode = 点选）且点过时画「点选：2 号、3 号」。 */
export function NodePickSummary({ nodeId }: { nodeId: string }) {
  const snap = useGraphSnapshot();
  const handle = handlesOf(snap, nodeId).find((h) => h.kind === "person");
  const param = handle ? Object.values(handle.params)[0] : "";
  const picks = (snap.nodes.find((n) => n.id === nodeId)?.data.params[param] as string[] | undefined) ?? [];
  const boxes = usePickBoxes(nodeId, param);
  const text = handle && picks.length ? pickSummary(pickChips(picks, boxes), !!boxes) : "";
  return text ? <div className="gnode-picks" data-user-data>{text}</div> : null;
}
