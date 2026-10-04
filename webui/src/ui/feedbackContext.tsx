// What a feedback carries about the editor it was sent from (the open graph, the shown node, the frame, the job):
// read when 提交反馈 opens.

import { graphForFeedback } from "../state/diagnostics";
import { toJSON } from "../graph/actions";
import { framesShown, useViewer } from "../state/viewer";
import { useCookInputs } from "../state/cookInputs";
import { useLook } from "../state/look";
import { useResults } from "../state/results";
import { nodeRef } from "../graph/naming";
import { noteText } from "../model/nodeOutcome";

export function editorContext(): Record<string, unknown> {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  const viewer = useViewer.getState();
  const r = useResults.getState();
  const frames = framesShown();
  const node = (id: string | null) => {
    const n = id ? ci.nodes[id] : undefined;
    if (!n || !id) return null;
    const status = r.byNode[id];
    return { id, type: n.typeId, label: nodeRef(id, n.typeId), status: status?.status ?? "idle", note: noteText(status?.note ?? ""), blocked: status?.blocked ?? "" };
  };
  return {
    editor: {
      graph_name: ci.meta.name,
      file: viewer.file?.name ?? null,
      unsaved: viewer.dirty,
      selected: node(viewer.selectedId),
      displayed: node(look.displayId),
      frame: viewer.frame,
      frames: frames.length ? { first: frames[0], last: frames[frames.length - 1], count: frames.length } : null,
      cook_range: ci.cookRange,
      plan: r.plan && { range: r.plan.range, frames: r.plan.frames, error: r.plan.error?.text ?? null, nodes: r.plan.nodes.length, cached: r.plan.cached },
      problems: ci.order.filter((id) => r.byNode[id]?.status === "error" || r.byNode[id]?.blocked).map((id) => node(id)),
    },
    graph: graphForFeedback(toJSON()),
    jobs: r.job ? [{ id: r.job.id, target: r.job.target }] : [],
  };
}
