/** 节点本体上的参数行：一个参数在节点上的名称、控件、接线后的来源说明与缩小后的文字形式，均由本模块绘制。 */

import { Handle, Position } from "@xyflow/react";
import type { MouseEvent } from "react";
import type { ParamDef, PortDef } from "../api";
import { setParam } from "../graph/actions";
import { useTypes } from "../state/catalog";
import { useViewer } from "../state/viewer";
import { useWriteLock } from "../ui/writeLock";
import { portColor } from "../graph/nodes";
import { multiline, NodeControl, valueText } from "../ui/controls";
import { useResults } from "../state/results";
import { ButtonParam } from "./buttonActions";

/** 节点本体上的一个参数：名称与取值，可在此直接编辑（与参数面板中的是同一个值）。
 * 「提升到节点」的参数，其输入口位于该行（下方 `param:<name>` 的 Handle）。
 *
 * 接入连线后，该行的控件置灰，取值以连线传入的值为准。置灰的格中显示连线传入的值
 * （尚未计算时显示「待计算」），前面一小行灰字说明来源（「← AnyCalib」），完整说明在参数面板中。
 * 参数当前不可用时（由服务器 applies 判定）采用相同方式：控件保留但置灰，原因在参数面板中。
 *
 * 控件不隐藏，只区分可用与不可用：该行的第二格始终是控件形状的元素，可编辑时为实际控件，
 * 不可编辑时为同尺寸的灰色格，位置不变。
 *
 * 点击该行在参数面板中定位该参数；双击取值则把键盘焦点交给面板中的对应输入框。
 * 缩小后整个取值显示为纯文字（NodeEditor 的缩放类：`.gctl` 收起、`.gprow-text` 出现）。 */
export function NodeParam({ nodeId, p, value, promoted, port, wired, why, nc }: {
  nodeId: string; p: ParamDef; value: unknown; promoted: boolean; port?: PortDef; wired: { source: string; value: string; fallback?: boolean } | null;
  why?: string; nc: Record<string, string>;
}) {
  const reveal = useViewer((s) => s.revealParam);
  const readOnly = !!useWriteLock(); // 写不了：节点上的控件真正禁用，不只是样式置灰
  const types = useTypes();
  // 选项是否可选与参数面板中查询的结果相同（由服务器按声明计算，ui/controls.tsx optionView）
  const answer = useResults((s) => s.results[nodeId]?.applies);
  // 缩小后整行显示为一句文字（`.gprow-text`，仅在 zoom-text / zoom-far 下出现）
  const text = wired ? `← ${wired.source}${wired.value ? ` · ${wired.value}` : ""}` : valueText(p, value, nc);
  // 双击取值：键盘焦点进入面板中的输入框（而非节点其余部分双击时的「在视图中显示」）
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
            {/* 可能没有值的口（Port.may_be_empty）：没有时用节点上填的 */}
            {wired.value || (wired.fallback ? valueText(p, value, nc) : "待计算")}
          </span>
        </span>
      ) : (
        <fieldset className="gctl nodrag" disabled={!!why || readOnly} onDoubleClick={toPanel}>
          {p.widget === "button" ? <ButtonParam nodeId={nodeId} p={p} mini />
            : <NodeControl p={p} value={value} set={(v) => setParam(nodeId, p.name, v)} nc={nc} answer={answer} />}
        </fieldset>
      )}
    </div>
  );
}
