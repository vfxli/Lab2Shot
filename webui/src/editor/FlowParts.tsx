/** What the node graph draws besides nodes: typed wires, the wire being dragged, zoom classes, a node's right-click menu. */

import { portColor, UNKNOWN_COLOR } from "../graph/nodes";
import { useMemo } from "react";
import { BaseEdge, getBezierPath, type ConnectionLineComponentProps, type EdgeProps } from "@xyflow/react";
import { cookBlocked, cookNote } from "../state/pause";
import { NODE_MENU, type NodeMenuFacts } from "./nodeActions";
import { snapshotNow } from "../graph/snapshot";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";
import { useViewer } from "../state/viewer";
import { clickKind, cookWords, inputPort, outputType, wireState } from "../graph/rules";
import { ERROR_COLOR } from "../platform/palette";
import { getNodeDefs as nodeDefsCached, useTypes as useTypesCached } from "../state/catalog";
import { Menu, type MenuRow } from "../ui/Menu";
import { isList } from "../state/items";

const WRONG = ERROR_COLOR;
const WAITING = "rgba(235, 235, 245, 0.34)";

/** A wire in the color of what it carries now; dashed in the error colour once it is wrong (its output gone or of another type);
 * muted and finely dotted while it waits for its output (an import node's kind not selected yet: it connects by itself
 * once it is). A wire carrying a list is drawn as two lines, in the colour of what the list holds.
 * The wire is looked up by id: a collapsed group box redraws it to the box, the same wire. */
export function TypedEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, selected }: EdgeProps) {
  const [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, curvature: 0.35 });
  const edges = useCookInputs((s) => s.edges);
  const nodes = useCookInputs((s) => s.nodes);
  const reply = useResults((s) => s.reply);
  const types = useTypesCached();
  const { look, list } = useMemo(() => {
    const e = edges.find((x) => x.id === id);
    if (!e) return { look: UNKNOWN_COLOR, list: false };
    // as the last status reply judged it (graph/rules.ts); a wire connected since is drawn in its output's colour
    const snap = { nodes: Object.entries(nodes).map(([nid, n]) => ({ id: nid, data: n })), edges, nodeDefs: nodeDefsCached(), reply };
    const w = wireState(snap, e);
    const t = w?.type || outputType(snap, e.source, e.sourceHandle) || "";
    const carries = isList(t.split("|")[0]);
    if (w?.state === "waiting") return { look: WAITING, list: carries };
    if (w?.state === "wrong") return { look: WRONG, list: carries };
    // a list is drawn in the colour of what it holds: 图像序列[] is the colour of 图像序列
    // 列表绘制为双线，颜色也使用列表对应的档位（graph/nodes.ts portColor：同色相、降低饱和度）
    return { look: t ? portColor(types, t) : UNKNOWN_COLOR, list: carries };
  }, [edges, nodes, reply, types, id]);
  const dash = look === WRONG ? "6 5" : look === WAITING ? "2 5" : undefined;
  return (
    <>
      {/* a wide, near-invisible line under the wire: what the pointer actually hits */}
      <BaseEdge id={id} path={path} style={{ stroke: look, strokeOpacity: 0.18, strokeWidth: 7 }} />
      {/* a wire is thin; a list's is two lines: the colour with a hairline through it */}
      <BaseEdge path={path} className="edge-path" style={{ stroke: look, strokeWidth: selected ? (list ? 4.6 : 3) : list ? 3.4 : 1.8, strokeDasharray: dash }} />
      {list && <BaseEdge path={path} className="edge-list-core" style={{ strokeWidth: selected ? 1.4 : 1 }} />}
    </>
  );
}

export function ConnectionLine({ fromX, fromY, toX, toY, fromPosition, toPosition, fromNode, fromHandle }: ConnectionLineComponentProps) {
  const s = snapshotNow();
  const t = fromHandle?.type === "target" ? inputPort(s, fromNode.id, fromHandle.id)?.type : outputType(s, fromNode.id, fromHandle?.id);
  const types = s.types;
  const color = t ? portColor(types, t) : UNKNOWN_COLOR;
  const [path] = getBezierPath({ sourceX: fromX, sourceY: fromY, targetX: toX, targetY: toY, sourcePosition: fromPosition, targetPosition: toPosition });
  return <path d={path} fill="none" stroke={color} strokeWidth={1.8} strokeDasharray="5 4" />;
}

// Zoomed out, the parameters on the nodes' bodies read as plain text (below ZOOM_TEXT: nothing small to hit by
// accident), and further out they fade (below ZOOM_FAR); the nodes keep their size, so nothing moves.
const ZOOM_TEXT = 0.75;
const ZOOM_FAR = 0.45;
export const zoomClass = (zoom: number) => (zoom < ZOOM_FAR ? " zoom-far" : zoom < ZOOM_TEXT ? " zoom-text" : "");

/** A node's right-click menu, drawn from the one table of what it offers (editor/nodeActions.ts NODE_MENU): 计算 (cook
 * it and what it needs, and show it; on a node whose cook hands files over, i.e. an 「输出」, it packs that one
 * 「输出」 alone, not the whole graph), 显示, and 合并成多层 EXR on 序列图输出设置 nodes. Closes on any click,
 * Escape or scroll. */
export function NodeMenu({ at, onClose }: { at: { x: number; y: number; id: string }; onClose: () => void }) {
  const busy = useResults((s) => !!s.job);
  const queueSwitches = useResults((s) => s.queueSwitches);
  const storage = useResults((s) => s.storage); // 配额已满：此处的「计算」与顶栏的「提交」一样置灰并说明原因
  const typeId = useCookInputs((s) => s.nodes[at.id]?.typeId ?? "");
  const snap = useMemo(() => snapshotNow(), [at.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const kind = clickKind(snap, at.id);
  const delivers = kind.delivers;
  const short = cookWords(kind.delivers, kind.nothing).short;
  // the same rule and note (state/pause.ts) as the top bar's 提交, so a click here is not a surprise
  const tip = cookWords(kind.delivers, kind.nothing).tip + cookNote(queueSwitches, storage);
  const blocked = cookBlocked(queueSwitches, storage);
  // 序列图输出设置 nodes to merge: the ones selected together with the one right-clicked, or just that node if it is not
  // part of a multi-selection. It is always included, since a right click never clears the canvas selection on its own.
  // Computed once when the menu opens (like `snap` above): the menu closes on any pointerdown/wheel/Escape, so the
  // selection cannot change while it is open. No reactive selector is needed here (which also avoids a selector
  // returning a fresh array on every call, which useSyncExternalStore turns into "Maximum update depth exceeded",
  // React error #185, instead of a plain extra render).
  const mergeIds = useMemo(() => {
    const ci = useCookInputs.getState();
    const n = ci.nodes[at.id];
    if (!n || n.typeId !== "core.output_images") return [] as string[];
    const canvas = useViewer.getState().canvas;
    const selected = ci.order.filter((id) => canvas[id]?.selected && ci.nodes[id].typeId === "core.output_images");
    return selected.includes(at.id) ? selected : [at.id];
  }, [at.id]); // eslint-disable-line react-hooks/exhaustive-deps
  // inside a 逐项处理 block: how many items this node has and which one the view is on, both as the server answered
  // (engine/scopes.py; the page never counts a block's items itself)
  const status = snap.results[at.id];
  const facts: NodeMenuFacts = {
    id: at.id,
    typeId,
    items: status?.summary?.total ?? 0,
    itemName: status?.item?.names.at(-1) ?? "",
    delivers,
    busy,
    blocked,
    cookTip: tip,
    cookShort: short,
    mergeIds,
  };
  const rows: MenuRow[] = NODE_MENU.filter((item) => item.when(facts)).map((item) => ({
    key: item.key,
    label: item.label(facts),
    desc: item.desc(facts),
    tip: item.tip(facts),
    off: item.off?.(facts) ?? false,
    run: () => item.run(facts),
  }));
  return <Menu at={at} rows={rows} label="节点" layout="ctx-menu" onClose={onClose} />;
}
