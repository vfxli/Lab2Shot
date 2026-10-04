import type { Node } from "@xyflow/react";
import type { BoxJSON, NodeTypeDef, PickedFrom } from "../api";
import type { NodeComment } from "./look";
import type { Note } from "../model/nodeOutcome";

/** The graph document as the page holds it for drawing and for the pure helpers (graph/nodes.ts, graph/rules.ts): a
 * node's type, parameters and live cook status, a group box, the whole graph. Types only. */

export type NodeStatus = "idle" | "queued" | "cooked" | "cooking" | "error" | "skipped";

export interface NodeData extends Record<string, unknown> {
  typeId: string;
  params: Record<string, unknown>;
  comment?: NodeComment; // 节点备注（state/look.ts）：文档的一部分，随撤销
  status: NodeStatus;
  note: Note; // stage / progress / error text shown on the node, said when shown (model/nodeOutcome.ts noteText)
  blocked?: string; // why it cannot be cooked yet (determined before anything is sent to the server)
  picked?: Record<string, PickedFrom>; // input file parameters: the picked source, as the parameter displays it
  promoted?: string[]; // 提升到节点: an input "param:<name>" (NodeTypeDef.param_ports) and a row on the node's body
  // 在节点上显示: the rows shown on the node body, when the user chose rows other than the type's default (NodeTypeDef.on_node).
  // Display only, with no input port; an input port is `promoted` above (two separate marks: ParamPanel.tsx OnNodePin / PromotePin)
  onNode?: string[];
  stored?: Record<string, unknown>;
}

export type GNode = Node<NodeData, "l2s">;

export type GBox = BoxJSON & { selected?: boolean };

/** A graph as these helpers read it: every node's type and parameters, every wire, and the node definitions. Built
 * fresh (graph/snapshot.ts) from state/cookInputs.ts + state/look.ts, never itself reactive. */
export interface GraphState {
  nodes: GNode[];
  edges: { id: string; source: string; sourceHandle: string | null; target: string; targetHandle: string | null }[];
  nodeDefs: Record<string, NodeTypeDef>;
}
