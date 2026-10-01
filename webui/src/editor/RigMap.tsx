// 本模块是「对应关系」编辑器（widget "rig_map"，lab2shot/nodes/kit/rig_map.py）：两副骨架之间按身体部位的对应，照绑定软件的
// 用法——Maya HumanIK 的 Character Definition（人形图上的部位槽：点槽变蓝，再在骨架里点骨头；配好变绿，必需部位
// 配齐时状态灯变绿；手有单独的一页）、Unreal IK Retargeter（脊柱、手指这类一串关节按链配，不管两边几节）、
// Houdini KineFX Map Points（Shift 只点首尾，中间自动补）。「动作重定向」两侧都能点；Kimodo / StableMotion /
// Two-stage Transformer / UnderPressure 的模型一侧是固定的（只读）。
//
// 走「弹窗编辑参数」框架（editor/ParamSheet.tsx）：数据是节点 NodeDef.choices 给的（两副骨架的关节、父子、绑定姿势
// 位置，部位表，推测结果），确定写回参数，取消 / Esc / 点遮罩不写回。写回的只有手动定过的部位（其余每次计算按名字
// 和层级重新推测，槽上标「自动」）；一个也没有就是空值 = 全部自动。

import { lazyRetry } from "../platform/lazyRetry";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import type { Choice, RigChoice, RigSide } from "../api";
import { COLS } from "./rigMap/rules";
import type { SheetEditor, SheetSummaryProps } from "./ParamSheet";
import { PLACES } from "./rigMap/figure";
import { draftOf, problems } from "./rigMap/rules";

export const rigOf = (p: { name: string }, choices: Record<string, Choice> | null): RigChoice | null => choices?.[p.name]?.rig ?? null;

/** 这个编辑器配的是什么：人体的部位 / 骨架 / 关节（默认），或者「表情重定向（ARKit52）」的表情 / 曲线 / 形变（服务端给的说法）。 */
export const slotWord = (rig: RigChoice) => rig.slot ?? "部位";
export const nounOf = (s: RigSide) => s.noun ?? "骨架";
export const itemOf = (s: RigSide) => s.item ?? "个关节";
/** 部位槽能摆在人形图上（身体、手指两页）；摆不上的（表情）按区域排成一格一格的槽。 */
export const onFigure = (rig: RigChoice) => rig.parts.some((q) => q.id in PLACES.body || q.id in PLACES.hands);

/** 面板上的摘要：自动还是手动、配上了几个部位、哪些部位有问题。 */
function RigMapSummary({ p, value, choices }: SheetSummaryProps) {
  const rig = rigOf(p, choices);
  const manual = Array.isArray(value) ? value.length : 0;
  const how = manual ? `手动 ${manual} 个${rig ? slotWord(rig) : "部位"}，其余自动` : "自动";
  if (!rig) {
    const why = choices?.[p.name]?.empty ?? choices?.[""]?.empty ?? "上游还没算：先算一次这个节点（右键「计算」，或模板里给它的计算按钮），再回来编辑";
    return <span className="psheet-none">{how} · {why}</span>;
  }
  const d = draftOf(value, rig);
  const bad = problems(d, rig);
  const paired = rig.parts.filter((q) => d.rows[q.id]?.src.length && d.rows[q.id]?.dst.length).length;
  const wrong = rig.parts.filter((q) => bad[q.id]).map((q) => q.label);
  return (
    <span className={wrong.length ? "rmap-sum bad" : "rmap-sum"}>
      {how} · 配上 {paired} 个{slotWord(rig)}{wrong.length ? ` · 有问题：${wrong.slice(0, 4).join("、")}${wrong.length > 4 ? " 等" : ""}` : ""}
    </span>
  );
}

/** widget「rig_map」：登记在 editor/ParamControls.tsx 的 SHEET_EDITORS。 */
export const RigMap: SheetEditor = {
  // 窗里内容按需加载（editor/RigMapEditor.tsx）：它带着三维舞台，打开编辑器不该把 three.js 拉进主包
  editor: lazyRetry(() => import("./RigMapEditor")),
  title: (p) => p.label,
  width: 1240,
  summary: RigMapSummary,
  ready: (x) => !!rigOf(x.p, x.choices),
  also: (x) => { const rig = rigOf(x.p, x.choices); return rig ? posesOf(x.nodeId, rig) : []; },
};

/** 这一侧的骨架在舞台上是节点的哪个「骨架姿势」手柄（NodeDef.handles 的下标）：服务端按手柄的输入口与这一侧的输入口
 * 配好写在 RigSide.handle 里；没有（模型节点固定的那一侧、表情重定向）就是 null，只有树视图。 */
export const handleOf = (side: RigSide): number | null => side.handle ?? null;

/** 窗连带写的姿势修正参数：两侧「骨架姿势」手柄各自的 pose 参数（编辑器的草稿与窗的基准都按这一份）。 */
export function posesOf(nodeId: string, rig: RigChoice): string[] {
  const typeId = useCookInputs.getState().nodes[nodeId]?.typeId;
  const defs = typeId ? getNodeDefs()[typeId]?.handles ?? [] : [];
  return [...new Set(COLS.flatMap((c) => { const h = handleOf(rig[c]); const q = h !== null ? defs[h]?.params.pose : undefined; return q ? [q] : []; }))];
}
