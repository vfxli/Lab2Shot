import type { CookNode, Wire } from "../state/cookInputs";
import type { Box, Pos } from "../state/look";
import { wireKey } from "./rules";

/** 复制节点（Ctrl+C / Ctrl+V / Ctrl+D）: what a copy holds and which wires a paste brings along. Pure: graph/edit.ts
 * reads the stores and writes the result back as one step.
 *
 * The wires: a copied node keeps every wire coming INTO it — from another copied node (the pasted wire then runs
 * between the two copies) or from a node outside the copy (the pasted wire comes from that same node, which is still
 * there: only while pasting into the graph it was copied from, and only while that node exists). Wires going OUT of the
 * copied nodes to nodes outside the copy are dropped: the copy is a branch of its own, it never feeds what the original
 * feeds. */

interface CopiedNode {
  id: string; // the original's id: how the copied wires name it
  data: CookNode;
  pos: Pos;
  onNode?: string[]; // the rows its body shows (state/look.ts onNode)
}

export interface Copied {
  graphId: string; // the graph copied from: wires from outside the copy are kept only when pasting into it
  nodes: CopiedNode[];
  wires: Wire[]; // every wire into a copied node, from inside the copy or outside it
  boxes: Box[]; // the group boxes copied with their contents (a collapsed box's `members` are copied ids)
}

/** The copy of these nodes (and boxes), their data taken as it is now. */
export function takeCopy(graphId: string, ids: string[], boxes: Box[], nodes: Record<string, CookNode>, positions: Record<string, Pos>, onNode: Record<string, string[] | undefined>, edges: Wire[]): Copied {
  const inCopy = new Set(ids);
  return {
    graphId,
    nodes: ids.map((id) => {
      const data = structuredClone(nodes[id]);
      const rows = onNode[id];
      return { id, data, pos: { ...(positions[id] ?? { x: 0, y: 0 }) }, ...(rows ? { onNode: [...rows] } : {}) };
    }),
    wires: edges.filter((e) => inCopy.has(e.target)).map((e) => ({ ...e })),
    boxes: boxes.map((b) => ({ ...b, members: b.members.filter((m) => inCopy.has(m)) })),
  };
}

/** The wires a paste adds: `renamed` maps each copied node's id to its copy's. A wire between two copied nodes joins
 * the two copies; a wire from outside the copy is kept only when pasting into the graph copied from (`sameGraph`) and
 * its source node is still there (`exists`). */
export function pastedWires(copied: Copied, renamed: Map<string, string>, sameGraph: boolean, exists: (id: string) => boolean): Wire[] {
  const out: Wire[] = [];
  for (const w of copied.wires) {
    const target = renamed.get(w.target);
    if (!target) continue;
    const inner = renamed.get(w.source);
    if (!inner && !(sameGraph && exists(w.source))) continue;
    const source = inner ?? w.source;
    out.push({ id: wireKey(source, w.sourceHandle, target, w.targetHandle), source, sourceHandle: w.sourceHandle, target, targetHandle: w.targetHandle });
  }
  return out;
}

/** A unique parameter's value for a copy (the output settings' 名字: no two alike): the copied value when no other
 * node has it, else numbered on from its stem ("beauty" → "beauty2", "beauty2" → "beauty3"). */
export function uniqueCopyValue(value: string, taken: Set<string>): string {
  if (!taken.has(value.toLowerCase())) return value;
  const stem = value.replace(/\d+$/, "") || value;
  let i = 2;
  while (taken.has(`${stem}${i}`.toLowerCase())) i++;
  return `${stem}${i}`;
}
