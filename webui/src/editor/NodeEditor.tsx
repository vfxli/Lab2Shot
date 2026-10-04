/** The node graph canvas: owns how the graph document (state/cookInputs.ts, state/look.ts) is handed to xyflow and how
 * xyflow's changes come back as edits (moves, selection, deletion as one undo step), plus the canvas's own furniture
 * (group boxes, 逐项处理 frames, the carried wire, zoom and arrange buttons). Pointer gestures and wiring live in
 * editor/graphPointer.ts and editor/wiring.ts. */
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Background, BackgroundVariant, MiniMap, ReactFlow, SelectionMode, useReactFlow, useStore as useFlow, type Edge, type EdgeChange, type IsValidConnection, type Node, type NodeChange } from "@xyflow/react";
import { nodeCategory, type NodeTypeDef } from "../api";
import { addBox, arrangeGraph, connect, deleteElements } from "../graph/actions";
import { ProjectNotice } from "./ProjectNotice";
import { GraphHelp } from "./GraphHelp";
import { canWire, useGraphPointer } from "./graphPointer";
import { CLICK_SLOP } from "../platform/drag";
import { snapshotNow } from "../graph/snapshot";
import { useComposedNodes, type PreparedGNode } from "../graph";
import { useCookInputs } from "../state/cookInputs";
import { useWriteLock } from "../ui/writeLock";
import { useLook, type Box } from "../state/look";
import { usePreferences } from "../state/preferences";
import { useViewer } from "../state/viewer";
import { BOX_FOLD_W, BOX_HEAD, boxContents } from "../graph/nodes";
import { GraphNode } from "./GraphNode";
import { BlockFrames, useBlockFrames } from "./BlockFrame";
import { NetworkBox, type BoxNode } from "./NetworkBox";
import { UnknownNode, type UnknownFlowNode } from "./UnknownNode";
import { IconArrange, IconFit, IconGroup, IconMap, IconMinus, IconPlus } from "../ui/icons";
import { getNodeDefs as nodeDefsCached } from "../state/catalog";
import { IconButton } from "../ui/Button";
import { ConnectionLine, NodeMenu, TypedEdge, zoomClass } from "./FlowParts";
import { GRAPH_DOT_COLOR } from "../platform/palette";
import { useShortcut } from "../platform/keys";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

const nodeTypes = { l2s: GraphNode, box: NetworkBox, unknown: UnknownNode };

const isBox = (id: string) => id.startsWith("box:");
const edgeTypes = { l2s: TypedEdge };

/** The node menu at a point of the screen, a new node going where it is: Tab, a right-click or a double-click on the
 * canvas, and the empty graph's welcome. */
export function useMenuAt() {
  const flow = useReactFlow();
  const openMenu = useViewer((s) => s.openMenu);
  return useCallback(
    (clientX: number, clientY: number) => {
      const p = flow.screenToFlowPosition({ x: clientX, y: clientY });
      openMenu({ x: clientX, y: clientY, flowX: p.x - 120, flowY: p.y - 20 });
    },
    [flow, openMenu],
  );
}

export function NodeEditor() {
  const { nodes, edges: wireEdges } = useComposedNodes();
  // the 逐项处理 blocks the server worked out (editor/BlockFrame.tsx): their frames are drawn in the canvas's own
  // layer, never as nodes. Blocks never collapse, so the nodes inside a block are always on the canvas
  const blockFrames = useBlockFrames();
  const boxes = useLook((s) => s.boxes);
  const kept = useCookInputs((s) => s.kept);
  const selectedNodeIds = useViewer((s) => s.canvas);
  const selectedEdgeIds = useViewer((s) => s.selectedEdgeIds);
  const selectedBoxIds = useViewer((s) => s.selectedBoxIds);
  const setCanvasNode = useViewer((s) => s.setCanvasNode);
  const setEdgeSelected = useViewer((s) => s.setEdgeSelected);
  const setBoxSelected = useViewer((s) => s.setBoxSelected);
  const select = useViewer((s) => s.select);
  const setDisplay = useLook((s) => s.setDisplay);
  const loads = useViewer((s) => s.loads);
  const viewer = !!useWriteLock(); // nothing may be written (ui/writeLock.ts: this graph is being edited in another tab)
  const flow = useReactFlow();
  const zoomed = useFlow((s) => zoomClass(s.transform[2])); // changes only when a threshold is crossed
  const wrap = useRef<HTMLDivElement>(null);
  const minimap = usePreferences((s) => s.minimap);
  const toggleMinimap = usePreferences((s) => s.toggleMinimap);
  const dragMembers = useRef<Record<string, string[]>>({}); // box id -> nodes riding along while it is dragged
  const boxed = useRef<Node[]>([]); // what the selection box holds while it is drawn (onSelectionChange)
  const [nodeMenu, setNodeMenu] = useState<{ x: number; y: number; id: string } | null>(null);
  const closeNodeMenu = useCallback(() => setNodeMenu(null), []);

  // xyflow's own ephemera (selected/dragging/measured) merged onto the composed content. A node none of whose parts
  // changed keeps last render's object: xyflow then keeps it and draws none of its wrapper again (opening a big graph,
  // every node's size arrives one after another: without this each arrival redrew every node)
  const keptLive = useRef(new Map<string, PreparedGNode>());
  const liveNodes = useMemo(() => {
    const before = keptLive.current;
    const now = new Map<string, PreparedGNode>();
    const out = nodes.map((n) => {
      const c = selectedNodeIds[n.id];
      const selected = !!c?.selected, dragging = !!c?.dragging, measured = c?.measured;
      const old = before.get(n.id);
      const live = old && old.data === n.data && old.position === n.position && old.selected === selected && old.dragging === dragging && old.measured === measured
        ? old : { ...n, selected, dragging, measured };
      now.set(n.id, live);
      return live;
    });
    keptLive.current = now;
    return out;
  }, [nodes, selectedNodeIds]);
  const liveWireEdges = useMemo(
    () => wireEdges.map((e) => ({ id: e.id, source: e.source, sourceHandle: e.sourceHandle, target: e.target, targetHandle: e.targetHandle, type: "l2s" as const, selected: selectedEdgeIds.includes(e.id) })),
    [wireEdges, selectedEdgeIds],
  );

  // group boxes are drawn as nodes behind the real ones; collapsed boxes hide their nodes
  // and the wires to those nodes attach to the box instead
  const { shownNodes, shownEdges } = useMemo(() => {
    const owner = new Map<string, string>();
    for (const b of boxes) if (b.collapsed) for (const m of b.members) owner.set(m, b.id);
    const boxNodes: BoxNode[] = boxes.map((b) => ({
      id: b.id,
      type: "box",
      position: { x: b.x, y: b.y },
      data: { box: b },
      width: b.collapsed ? Math.min(b.w, BOX_FOLD_W) : b.w,
      height: b.collapsed ? BOX_HEAD : b.h,
      zIndex: -1,
      selected: selectedBoxIds.includes(b.id),
      dragHandle: ".netbox-head",
    }));
    const real = owner.size ? liveNodes.map((n) => (owner.has(n.id) ? { ...n, hidden: true } : n)) : liveNodes;
    let wires: Edge[] = liveWireEdges;
    if (owner.size) {
      wires = [];
      for (const e of liveWireEdges) {
        const sb = owner.get(e.source);
        const tb = owner.get(e.target);
        if (sb && sb === tb) continue;
        wires.push(
          sb || tb
            ? { ...e, source: sb ?? e.source, sourceHandle: sb ? "out" : e.sourceHandle, target: tb ?? e.target, targetHandle: tb ? "in" : e.targetHandle }
            : e,
        );
      }
    }
    // 「未知节点」: what the file has that this server does not, drawn plainly with its wires, never changed
    const unknown: UnknownFlowNode[] = kept.nodes.map((n) => ({
      id: n.id,
      type: "unknown",
      position: { x: n.ui?.x ?? 0, y: n.ui?.y ?? 0 },
      data: {
        inputs: [...new Set(kept.edges.filter((e) => e.to[0] === n.id).map((e) => e.to[1]))],
        outputs: [...new Set(kept.edges.filter((e) => e.from[0] === n.id).map((e) => e.from[1]))],
      },
      draggable: false,
      selectable: false,
      deletable: false,
      connectable: false,
    }));
    const present = new Set([...liveNodes.map((n) => n.id), ...kept.nodes.map((n) => n.id)]);
    const keptWires: Edge[] = kept.edges.filter((e) => present.has(e.from[0]) && present.has(e.to[0])).map((e) => ({
      id: `kept:${e.from.join(".")}->${e.to.join(".")}`,
      source: owner.get(e.from[0]) ?? e.from[0],
      sourceHandle: owner.has(e.from[0]) ? "out" : e.from[1],
      target: owner.get(e.to[0]) ?? e.to[0],
      targetHandle: owner.has(e.to[0]) ? "in" : e.to[1],
      selectable: false,
      deletable: false,
      className: "unknown-wire",
    }));
    return { shownNodes: [...boxNodes, ...real, ...unknown] as (PreparedGNode | BoxNode | UnknownFlowNode)[], shownEdges: [...wires, ...keptWires] };
  }, [liveNodes, liveWireEdges, boxes, kept, selectedBoxIds]);

  const onChanges = useCallback(
    (changes: NodeChange<PreparedGNode | BoxNode | UnknownFlowNode>[]) => {
      // removals are not handled here: React Flow reports one deletion as several changes (the wires first, then the
      // nodes), which would record several undo steps; onDelete below gets them together (deleteElements: one step)
      const unknownIds = new Set(kept.nodes.map((n) => n.id));
      for (const c of changes) {
        const id = "id" in c ? c.id : "";
        if (unknownIds.has(id)) continue; // 「未知节点」 never change
        if (!isBox(id)) {
          if (c.type === "position" && c.position) {
            useLook.getState().setPosition(id, c.position.x, c.position.y);
            setCanvasNode(id, { dragging: c.dragging });
          } else if (c.type === "select") setCanvasNode(id, { selected: c.selected });
          else if (c.type === "dimensions") setCanvasNode(id, { measured: c.dimensions });
          continue;
        }
        if (c.type === "position" && c.position) useLook.getState().moveBox(id, c.position.x, c.position.y, dragMembers.current[id] ?? []);
        else if (c.type === "select") setBoxSelected(id, !!c.selected);
      }
    },
    [kept, setCanvasNode, setBoxSelected],
  );
  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      for (const c of changes) {
        if ("id" in c && c.id.startsWith("kept:")) continue;
        if (c.type === "select") setEdgeSelected(c.id, c.selected);
      }
    },
    [setEdgeSelected],
  );
  const onDelete = useCallback(
    ({ nodes, edges }: { nodes: { id: string; type?: string }[]; edges: { id: string }[] }) => {
      const unknownIds = new Set(kept.nodes.map((n) => n.id));
      const real = nodes.filter((n) => !unknownIds.has(n.id));
      deleteElements(
        real.filter((n) => !isBox(n.id)).map((n) => n.id),
        real.filter((n) => isBox(n.id)).map((n) => n.id),
        edges.filter((e) => !e.id.startsWith("kept:")).map((e) => e.id),
      );
    },
    [kept],
  );
  // Delete / Backspace remove the selection, as one step. Through the page's single key registry (platform/keys.ts):
  // an open sheet, popover or menu takes the key first, whereas xyflow's own deleteKeyCode listens on the whole
  // document and would delete the nodes behind it
  useShortcut({
    keys: ["delete", "backspace"],
    run: () => {
      const v = useViewer.getState();
      const picked = Object.entries(v.canvas).filter(([, c]) => c.selected).map(([id]) => ({ id }));
      if (viewer || !(picked.length || v.selectedBoxIds.length || v.selectedEdgeIds.length)) return false;
      onDelete({ nodes: [...picked, ...v.selectedBoxIds.map((id) => ({ id }))], edges: v.selectedEdgeIds.map((id) => ({ id })) });
    },
  });
  const isValidConnection: IsValidConnection = useCallback(
    (c) => !!c.source && !!c.target && canWire(snapshotNow(), { node: c.source, port: c.sourceHandle ?? "" }, { node: c.target, port: c.targetHandle ?? "" }),
    [],
  );

  const menuAt = useMenuAt();

  // ReactFlow's handlers are the same functions from render to render: xyflow hands the node ones to every node's
  // wrapper, and a new function there redraws every node of the graph on each render of this component
  const onNodeDragStart = useCallback((_: unknown, _n: unknown, dragged: Node[]) => {
    const boxesNow = useLook.getState().boxes;
    const nodesNow = nodesForBoxContents();
    const moving = new Set(dragged.map((d) => d.id));
    dragMembers.current = {};
    for (const d of dragged) {
      if (!isBox(d.id)) continue;
      const box = boxesNow.find((b) => b.id === d.id);
      // nodes dragged at the same time already move themselves
      if (box) dragMembers.current[d.id] = boxContents(box, nodesNow).filter((m) => !moving.has(m));
    }
  }, []);
  const onNodeDragStop = useCallback(() => void (dragMembers.current = {}), []);
  const onNodeClick = useCallback((_: unknown, n: Node) => void (n.type !== "unknown" && !isBox(n.id) && select(n.id)), [select]);
  const onNodeDoubleClick = useCallback((_: unknown, n: Node) => void (n.type !== "unknown" && !isBox(n.id) && setDisplay(n.id)), [setDisplay]);
  const onPaneClick = useCallback(() => select(null), [select]);
  const onSelectionChange = useCallback(({ nodes: picked }: { nodes: Node[] }) => void (boxed.current = picked), []);
  const onSelectionEnd = useCallback(() => {
    const real = boxed.current.filter((n) => n.type !== "unknown" && !isBox(n.id));
    if (real.length === 1) select(real[0].id);
  }, [select]);
  const onDoubleClick = useCallback((e: React.MouseEvent) => {
    if (!viewer && (e.target as HTMLElement).classList.contains("react-flow__pane")) menuAt(e.clientX, e.clientY);
  }, [viewer, menuAt]);
  // while the view pans or zooms the canvas is one layer the compositor moves (data-moving: will-change in
  // styles/08-node-graph.css), not repainted and rasterised again every frame: a big graph stays fluid. When the
  // gesture ends the mark goes and the canvas is drawn sharp at the new zoom once. Set on the element, not through
  // React: no render for it
  const onMoveStart = useCallback(() => wrap.current?.setAttribute("data-moving", ""), []);
  const onMoveEnd = useCallback(() => wrap.current?.removeAttribute("data-moving"), []);

  // the pointer's division of labour and wiring by clicks: editor/graphPointer.ts
  const pointer = useGraphPointer({ wrap, viewer, flow, menuAt, unknownIds: kept.nodes.map((n) => n.id), onNodeMenu: setNodeMenu });
  // wires lifted with Ctrl are drawn faint (they follow the pointer and move as a group when dropped; graphPointer.ts)
  const heldWires = pointer.heldWires;
  const edgesDrawn = useMemo(
    () => (heldWires.size ? shownEdges.map((e) => (heldWires.has(e.id) ? { ...e, className: `${e.className ?? ""} held`.trim() } : e)) : shownEdges),
    [shownEdges, heldWires],
  );

  // fit the view whenever another graph is loaded
  useEffect(() => {
    const t = setTimeout(() => flow.fitView({ padding: 0.08, maxZoom: 1, duration: 250 }), 30);
    return () => clearTimeout(t);
  }, [loads, flow]);

  // a point that should be on screen (a node 插入 just placed, e.g. below the wire it fixed): pan there, but only if
  // it is not already inside the graph's own pane — an ordinary selection never moves the view, this is only for
  // something new that could otherwise land unseen
  const panTo = useViewer((s) => s.panTo);
  useEffect(() => {
    if (!panTo || !wrap.current) return;
    const r = wrap.current.getBoundingClientRect();
    const margin = 40;
    const p = flow.flowToScreenPosition({ x: panTo.x, y: panTo.y });
    if (p.x >= r.left + margin && p.x <= r.right - margin && p.y >= r.top + margin && p.y <= r.bottom - margin) return;
    flow.setCenter(panTo.x, panTo.y, { zoom: flow.getZoom(), duration: 300 });
  }, [panTo, flow]);

  return (
    <div className={`graph${zoomed}${pointer.carrying ? " wiring" : ""}`} ref={wrap} {...pointer.handlers}>
      <ReactFlow<PreparedGNode | BoxNode | UnknownFlowNode>
        nodes={shownNodes}
        edges={edgesDrawn}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onChanges}
        onNodeDragStart={onNodeDragStart}
        onNodeDragStop={onNodeDragStop}
        elevateNodesOnSelect={false}
        onEdgesChange={onEdgesChange}
        onDelete={onDelete}
        onConnect={viewer ? undefined : connect}
        onConnectEnd={viewer ? undefined : pointer.onConnectEnd}
        isValidConnection={isValidConnection}
        connectionLineComponent={ConnectionLine}
        onNodeClick={onNodeClick}
        onNodeDoubleClick={onNodeDoubleClick}
        onPaneClick={onPaneClick}
        // a selection box that ends on exactly one node opens it in the parameter panel, as a click does; several leave
        // the panel as it is
        onSelectionChange={onSelectionChange}
        onSelectionEnd={onSelectionEnd}
        connectOnClick={false} // a click on a port is the wiring machine's (editor/wiring.ts), not two machines at once
        connectionDragThreshold={CLICK_SLOP} // under it the gesture is a click (the page's one threshold): the wire is carried, not dragged; @xyflow/react starts a drag past it, so the two never both happen
        noPanClassName="l2s-nopan" // nothing has it: a middle-drag pans over a node as well as over the empty pane
        panOnDrag={[1]} // middle-drag pans; so does Space + left-drag (panActivationKeyCode, xyflow's own)
        selectionOnDrag // left-drag on the empty pane = box select
        // a node touched by the selection box is selected, it need not be enclosed (as in Nuke and Houdini)
        selectionMode={SelectionMode.Partial}
        selectNodesOnDrag={false}
        onDoubleClick={onDoubleClick}
        zoomOnDoubleClick={false}
        nodesDraggable={!viewer}
        nodesConnectable={!viewer}
        edgesReconnectable={!viewer}
        deleteKeyCode={null} // Delete / Backspace go through the page's key registry (the shortcut above)
        onMoveStart={onMoveStart}
        onMoveEnd={onMoveEnd}
        minZoom={0.02}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        fitView
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color={GRAPH_DOT_COLOR} />
        {/* the 逐项处理 frames: drawn with the canvas, behind the nodes, measured by nothing (editor/BlockFrame.tsx) */}
        <BlockFrames frames={blockFrames} />
        {minimap && (
          <MiniMap
            pannable
            zoomable
            position="top-right"
            className="graph-minimap"
            style={{ width: 220, height: 130 }}
            bgColor="rgba(22,22,26,0.92)"
            maskColor="rgba(0,0,0,0.45)"
            nodeBorderRadius={4}
            nodeColor={minimapColor}
          />
        )}
      </ReactFlow>
      {nodeMenu && <NodeMenu at={nodeMenu} onClose={closeNodeMenu} />}
      {/* the wire the pointer carries: a dashed line from the port it comes from; a group lifted with Ctrl draws one line per wire (<g>) */}
      <svg className="wire-line" aria-hidden={!pointer.carrying}>
        <path ref={pointer.line} />
        <g ref={pointer.bundle} />
      </svg>
      <ProjectNotice className="graph-notice" />
      <GraphHelp />
      <div className="flow-controls glass">
        <IconButton aria-label={t("ui.editor.minimap")} tone="ghost" on={minimap} onClick={toggleMinimap}>
          <IconMap size={13} />
        </IconButton>
        <IconButton aria-label={t("ui.editor.add_box")} tip={tipOf("shortcut", t("ui.editor.add_box"))} tone="ghost" disabled={viewer} onClick={() => addBox(flow.screenToFlowPosition({ x: (wrap.current?.getBoundingClientRect().left ?? 0) + 80, y: (wrap.current?.getBoundingClientRect().top ?? 0) + 60 }))}>
          <IconGroup size={13} />
        </IconButton>
        <IconButton aria-label={t("ui.editor.zoom_out")} tone="ghost" onClick={() => flow.zoomOut({ duration: 150 })}>
          <IconMinus size={13} />
        </IconButton>
        <IconButton aria-label={t("ui.editor.zoom_in")} tone="ghost" onClick={() => flow.zoomIn({ duration: 150 })}>
          <IconPlus size={13} />
        </IconButton>
        <IconButton aria-label={t("ui.editor.fit_all")} tone="ghost" onClick={() => flow.fitView({ padding: 0.18, duration: 250 })}>
          <IconFit size={13} />
        </IconButton>
        {/* arrange the graph (graph/layout.ts): columns left to right in order, few crossings within a column, straight chains
            straightened; with several nodes selected only those are arranged. Not offered in a read-only tab */}
        {!viewer && (
          <IconButton aria-label={t("ui.editor.arrange")} tone="ghost"
            onClick={() => (arrangeGraph(), requestAnimationFrame(() => flow.fitView({ padding: 0.18, duration: 250 })))}>
            <IconArrange size={13} />
          </IconButton>
        )}
      </div>
    </div>
  );
}

function minimapColor(n: Node): string {
  if (n.type === "box") return `${(n.data as { box: Box }).box.color}33`;
  const def = nodeDefsCached()[(n.data as PreparedGNode["data"]).typeId];
  return catalogCategoryColor(def);
}

function catalogCategoryColor(def: NodeTypeDef | undefined): string {
  return nodeCategory(snapshotNow().catalog, def).color;
}

/** The nodes as boxContents() needs them (id + position): built fresh from state/cookInputs.ts + state/look.ts,
 * imperative (a drag start is an event handler, not a render). */
function nodesForBoxContents() {
  const ci = useCookInputs.getState();
  const look = useLook.getState();
  return ci.order.map((id) => ({ id, position: look.positions[id] ?? { x: 0, y: 0 }, measured: undefined }) as never);
}
