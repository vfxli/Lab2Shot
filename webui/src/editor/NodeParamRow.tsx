/** 节点上的一行参数。 */

import { Handle, Position } from "@xyflow/react";
import type { MouseEvent } from "react";
import type { ParamDef, PortDef } from "../api";
import { setParam } from "../graph/actions";
import { useTypes } from "../state/catalog";
import { useViewer } from "../state/viewer";
import { portColor } from "../graph/nodes";
import { multiline, NodeControl, valueText } from "../ui/controls";
import { useResults } from "../state/results";

/** One parameter on the node's body: its name, and its value, edited directly here (the same value as the panel's).
 * 提升到节点 的参数，其输入口位于该行（下方 `param:<name>` 的 Handle）。
 *
 * 接入连线后，该行的控件置灰，取值以连线传入的值为准。置灰的格中显示连线传入的值
 * （尚未计算时显示「待计算」），前面一小行灰字说明来源（「← AnyCalib」），完整说明在参数面板中。
 * 参数当前不可用时（由服务器 applies 判定）采用相同方式：控件保留但置灰，原因在参数面板中。
 *
 * 控件不隐藏，只区分可用与不可用：该行的第二格始终是控件形状的元素，可编辑时为实际控件，
 * 不可编辑时为同尺寸的灰色格，位置不变。
 *
 * Clicking the row finds the parameter in the panel; a double click on the value puts the keyboard in the panel's
 * field. Zoomed out, the whole value reads as plain text (NodeEditor's zoom classes: `.gctl` 收起、`.gprow-text` 出现). */
export function NodeParam({ nodeId, p, value, promoted, port, wired, why, nc }: {
  nodeId: string; p: ParamDef; value: unknown; promoted: boolean; port?: PortDef; wired: { source: string; value: string } | null;
  why?: string; nc: unknown[];
}) {
  const reveal = useViewer((s) => s.revealParam);
  const types = useTypes();
  // 选项是否可选与参数面板中查询的结果相同（由服务器按声明计算，ui/controls.tsx optionOff）
  const answer = useResults((s) => s.results[nodeId]?.applies);
  // 缩小后整行显示为一句文字（`.gprow-text`，仅在 zoom-text / zoom-far 下出现）
  const text = wired ? `← ${wired.source}${wired.value ? ` · ${wired.value}` : ""}` : valueText(p, value);
  // a double click on the value: the keyboard in the panel's field (not 在视图中显示, which the rest of the node does)
  const toPanel = (e: MouseEvent) => {
    e.stopPropagation();
    reveal(nodeId, p.name, true);
  };
  return (
    <div
      // 多行文本框行采用上下两格排列（标签在上，文本框占满整行宽度）：判据为参数自身的声明
      // （`P(lines=…)`，ui/controls.tsx multiline），而非参数名称
      className={`gprow${multiline(p) ? " multi" : ""}${promoted ? " promoted" : ""}${wired ? " wired" : ""}${why && !wired ? " inactive" : ""}`}
      data-param={p.name}
      onPointerDown={(e) => {
        if (!(e.target as HTMLElement).closest(".react-flow__handle")) reveal(nodeId, p.name);
      }}
    >
      {/* 端口形状只表达一项信息：方形表示列表。参数口与数据口使用同一套形状和颜色，以其所在的参数行表明它是参数 */}
      {promoted && <Handle type="target" position={Position.Left} id={`param:${p.name}`} className={`param${port?.list ? " list" : ""}`} style={{ ["--c" as string]: portColor(types, port?.type ?? "") }} />}
      <span className="gprow-label">{p.label}</span>
      <span className="gprow-text" data-user-data onDoubleClick={toPanel}>{text}</span>
      {wired ? (
        // 已接线：控件格变为同尺寸的灰色格，显示连线传入的值，前面一小行说明其来源
        <span className="gctl gwired" onDoubleClick={toPanel}>
          <small className="gwired-from" data-user-data>← {wired.source}</small>
          <span className="field mini gwired-val" data-user-data>
            {wired.value || "待计算"}
          </span>
        </span>
      ) : (
        <fieldset className="gctl nodrag" disabled={!!why} onDoubleClick={toPanel}>
          <NodeControl p={p} value={value} set={(v) => setParam(nodeId, p.name, v)} nc={nc} answer={answer} />
        </fieldset>
      )}
    </div>
  );
}
