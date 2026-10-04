import { memo, useMemo } from "react";
import { ViewportPortal } from "@xyflow/react";
import { render } from "../messages/format";
import { nodeSize } from "../graph/nodes";
import { useTypes } from "../state/catalog";
import { useItems, blocksOf, colourOf, countOf, frameOf, itemWord, nodesOf, viewPath, type BlockCount, type Rect, type Scope } from "../state/items";
import { useLook } from "../state/look";
import { useResults } from "../state/results";
import { useViewer } from "../state/viewer";
import "./styles/27-blocks.css";

/** Owns how 逐项处理 blocks are drawn on the node graph: behind the nodes, a dashed rounded frame around the nodes of
 * one block, coloured by the block's name, with one title pill (「逐项处理 · 逐个：人物 · 3 条 · 2/3 已算」). There is
 * no fold: with a block folded its wires would vanish too, and nothing would show what data enters or leaves it.
 *
 * Which nodes belong to a block, how many items it has and what they are called are the server's answer
 * (engine/scopes.py, the status reply's `scopes`, read through state/items.ts): this file draws it and works out
 * nothing of it.
 *
 * The frames are drawn in the canvas's own layer (xyflow's ViewportPortal), not as nodes: a frame's size follows the
 * sizes of the nodes inside it, and a node whose size is measured by the same observer that the frame would then
 * resize is exactly the loop the browser reports as 「ResizeObserver loop completed with undelivered notifications」.
 * Nothing measures the frame, nothing selects it, and every click goes through it to the canvas below. */

interface BlockData {
  scope: Scope;
  rect: Rect; // where it is, in the canvas's own coordinates
  kind: string; // what one item is called (逐个：人物)
  count: BlockCount;
  colour: number; // which of ui/tokens.css's --block-N its frame is drawn in (state/items.ts colourOf)
}

/** The title pill's words, made of the catalogue's lines (no sentence is written here). */
function blockWords(data: BlockData): { title: string; count: string; done: string; failed: string } {
  const { count } = data;
  return {
    title: render("I-EACH-TITLE", { kind: data.kind }),
    count: count.pending ? render("I-EACH-WAITING") : render("I-EACH-COUNT", { count: count.items }),
    done: count.pending || !count.items ? "" : render("I-EACH-DONE", { done: count.done, total: count.items }),
    failed: count.failed ? render("I-EACH-FAILED", { count: count.failed }) : "",
  };
}

const BlockBox = memo(function BlockBox({ data }: { data: BlockData }) {
  const words = blockWords(data);
  return (
    <div
      className="blockframe"
      style={{ transform: `translate(${data.rect.x}px, ${data.rect.y}px)`, width: data.rect.w, height: data.rect.h }}
      data-block={data.scope.name}
      data-colour={data.colour}
    >
      <div className="blockframe-head">
        <span className="blockframe-title">{words.title}</span>
        <span className="blockframe-count tnum">{words.count}</span>
        {words.done && <span className="blockframe-done tnum">{words.done}</span>}
        {words.failed && <span className="blockframe-failed tnum">{words.failed}</span>}
      </div>
    </div>
  );
});

/** Every block's frame, in the canvas's own layer (behind the nodes: the layer's z-index is under theirs). */
export function BlockFrames({ frames }: { frames: BlockData[] }) {
  if (!frames.length) return null;
  return (
    <ViewportPortal>
      <div className="blockframes">
        {frames.map((f) => (
          <BlockBox key={f.scope.begin} data={f} />
        ))}
      </div>
    </ViewportPortal>
  );
}

/** The frames to draw behind the graph's nodes (every node inside a block stays on the canvas: there is no fold
 * as its one bar between its 逐项开始 and 逐项结束). Everything here comes from the status reply's `scopes` and from
 * where the nodes are; no rule of a block is worked out on this side. */
export function useBlockFrames(): BlockData[] {
  const reply = useResults((s) => s.reply);
  const positions = useLook((s) => s.positions);
  const canvas = useViewer((s) => s.canvas);
  const view = useItems((s) => s.view);
  const types = useTypes();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => build(reply, positions, canvas, view, types), [reply, positions, canvas, view, types]);
}

function build(
  reply: ReturnType<typeof useResults.getState>["reply"],
  positions: ReturnType<typeof useLook.getState>["positions"],
  canvas: ReturnType<typeof useViewer.getState>["canvas"],
  view: Record<string, string>,
  types: ReturnType<typeof useTypes>,
): BlockData[] {
  const blocks = blocksOf(reply);
  const frames: BlockData[] = [];
  for (const scope of blocks) {
    const ids = nodesOf(scope);
    const boxes: Record<string, Rect | undefined> = {};
    for (const nid of ids) {
      const at = positions[nid];
      if (!at) continue;
      const { w, h } = nodeSize({ measured: canvas[nid]?.measured });
      boxes[nid] = { x: at.x, y: at.y, w, h };
    }
    const rect = frameOf(ids, boxes);
    if (!rect) continue;
    // what one item is: the begin's 「条目」 output type, as the server resolved it here
    const itemType = reply?.nodes[scope.begin]?.ports.outputs.find((p) => p.name === "item")?.type ?? "";
    frames.push({
      scope,
      rect,
      kind: itemWord(types, itemType),
      count: countOf(scope, viewPath(blocks, scope.begin, view).slice(0, -1)),
      colour: colourOf(scope.name),
    });
  }
  return frames;
}
