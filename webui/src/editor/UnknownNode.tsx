import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import "./unknown.css";

/** A node of a type this server does not have for this account (an extension not installed, one the account may not
 * use, one removed): drawn as a plain 「未知节点」 box with its wires, saying nothing about what it is. It keeps its data
 * as the file had it (store.ts kept), is written back when the graph is saved, and is never sent to the server; it
 * can't be moved, changed, connected or cooked. */

export type UnknownNodeData = { inputs: string[]; outputs: string[] };
export type UnknownFlowNode = Node<UnknownNodeData, "unknown">;

const TIP = "这个服务器上没有这种节点：它的参数和连线原样留在节点图里，保存时一起写回文件，不会丢；在这里不能改，也不能计算";

export const UnknownNode = memo(function UnknownNode({ data }: NodeProps<UnknownFlowNode>) {
  const rows = Math.max(data.inputs.length, data.outputs.length, 1);
  return (
    <div className="unknown-node" data-tip={TIP} style={{ height: 34 + rows * 16 }}>
      <div className="unknown-head">未知节点</div>
      {data.inputs.map((p, i) => (
        <Handle key={`i-${p}`} type="target" id={p} position={Position.Left} isConnectable={false} className="unknown-port" style={{ top: 34 + i * 16 }} />
      ))}
      {data.outputs.map((p, i) => (
        <Handle key={`o-${p}`} type="source" id={p} position={Position.Right} isConnectable={false} className="unknown-port" style={{ top: 34 + i * 16 }} />
      ))}
    </div>
  );
});
