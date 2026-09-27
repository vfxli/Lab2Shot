import { useCallback, useEffect, useRef, useState } from "react";
import { useOnViewportChange, type OnConnectEnd, type useReactFlow } from "@xyflow/react";
import { addBox, addPortRow, connect, deleteElements } from "../graph/actions";
import { snapshotNow } from "../graph/snapshot";
import { useCookInputs } from "../state/cookInputs";
import { useViewer } from "../state/viewer";
import { PARAM, converter, inputPort, outputType, portAccepts } from "../graph/rules";
import { heldKey, useShortcut } from "../platform/keys";
import { wiring, type End, type Wiring } from "./wiring";

/** The node graph's pointer: which button does what, and the wire the pointer carries (editor/wiring.ts).
 * Everything the graph pane does with a press lives here, so editor/NodeEditor.tsx stays limited to drawing: the hook
 * returns the handlers the pane spreads onto itself, the dashed line's path element, and whether a wire is being carried.
 *
 *   左键  point: select a node, or clear the selection; drag: a selection box on empty canvas (selectionOnDrag),
 *         or move a node (both handled by @xyflow/react); on a port: a wire (a drag is @xyflow/react's, a click is this module's)
 *   中键  drag: pan the canvas, over a node as well as over empty canvas (@xyflow/react's panOnDrag=[1]); the browser's
 *         own auto-scroll never starts. A middle click opens nothing.
 *   右键  one click in place only: the menu that adds a node on empty canvas, the node's own menu (计算 / 提交) on a
 *         node. A right drag does nothing.
 *   滚轮  zoom (@xyflow/react's own), and 空格 + 左键 pans like 中键 does
 *   数据信息  the mark at a node's bottom right (editor/GraphNode.tsx), never a key and never a click of the pointer's */

/** A press that moved no further than this is a click in place; @xyflow/react starts a drag past the
 * same distance (connectionDragThreshold), so the two never both happen. */
export const CLICK_PX = 4;

/** Which node an event landed on (null: empty canvas). 右键 compares this between press and release instead of a
 * distance: the right button has no drag of its own (中键 pans), so a hand that shifts a few pixels while clicking must
 * still open the menu. A press and a release on two different nodes is a real drag, and opens nothing. */
function nodeUnder(el: HTMLElement): string | null {
  return el.closest<HTMLElement>(".react-flow__node")?.dataset.id ?? null;
}

/** May a wire run from this output to this input: the server's rules, looked up (graph/rules.ts), never judged here.
 * A type the input does not take goes in only when a conversion node exists (the wire is kept dashed, one click). */
export function canWire(s: ReturnType<typeof snapshotNow>, from: End, to: End): boolean {
  const t = outputType(s, from.node, from.port);
  const port = inputPort(s, to.node, to.port);
  // 该端口当前不可用（由服务器计算，nodes/base.py Port.applies）：连线无法接入，而非接入后绘制为虚线
  if (port?.inactive) return false;
  return !!t && !!port && from.node !== to.node && (portAccepts(s.catalog, port.type, t) || (!port.name.startsWith(PARAM) && !!converter(s.catalog, t, port.type)));
}

/** The port an element belongs to, as the node graph draws it (xyflow's handle carries its node and port). */
function portAt(el: Element | null): { end: End; side: "output" | "input" } | null {
  const h = (el as HTMLElement | null)?.closest<HTMLElement>(".react-flow__handle");
  if (!h?.dataset.nodeid) return null;
  return { end: { node: h.dataset.nodeid, port: h.dataset.handleid ?? "" }, side: h.classList.contains("source") ? "output" : "input" };
}

const isBox = (id: string) => id.startsWith("box:");

const handleSelector = (end: End, side: "output" | "input") =>
  `.react-flow__handle.${side === "output" ? "source" : "target"}[data-nodeid="${CSS.escape(end.node)}"][data-handleid="${CSS.escape(end.port)}"]`;

/** The middle of an element, in the graph pane's own pixels. */
function centreIn(box: DOMRect, el: Element): { x: number; y: number } {
  const r = el.getBoundingClientRect();
  return { x: r.left + r.width / 2 - box.left, y: r.top + r.height / 2 - box.top };
}

export interface GraphPointer {
  /** A wire dragged out of a port and let go where no port took it (@xyflow/react's own gesture). */
  onConnectEnd: OnConnectEnd;
  handlers: {
    onMouseMove: React.MouseEventHandler<HTMLDivElement>;
    onPointerDownCapture: React.PointerEventHandler<HTMLDivElement>;
    onMouseDownCapture: React.MouseEventHandler<HTMLDivElement>;
    onPointerUp: React.PointerEventHandler<HTMLDivElement>;
    onAuxClick: React.MouseEventHandler<HTMLDivElement>;
    onContextMenu: React.MouseEventHandler<HTMLDivElement>;
  };
  line: React.RefObject<SVGPathElement | null>;
  carrying: boolean;
}

export function useGraphPointer({ wrap, viewer, flow, menuAt, unknownIds, onNodeMenu }: {
  wrap: React.RefObject<HTMLDivElement | null>;
  viewer: boolean;
  flow: ReturnType<typeof useReactFlow>;
  menuAt: (clientX: number, clientY: number) => void;
  unknownIds: string[];
  onNodeMenu: (at: { x: number; y: number; id: string }) => void;
}): GraphPointer {
  const select = useViewer((s) => s.select);
  const openMenu = useViewer((s) => s.openMenu);
  const press = useRef<{ x: number; y: number; button: number; node: string | null } | null>(null);
  const swallow = useRef(false); // this press was the wiring's or the data panel's: the graph below must not act on it
  const wire = useRef<Wiring>(null);
  const line = useRef<SVGPathElement>(null);
  const mouse = useRef({ x: 0, y: 0 });
  const pannedWithSpace = useRef(false); // something was dragged while 空格 was held: the key panned, it does not play
  const tookWire = useRef(false); // this gesture's press picked the wire up (a click leaves it carried, a drag carries it)
  const actedOnPress = useRef(false); // the press did this gesture's work: its release does nothing more

  const [carried, setCarried] = useState<Wiring>(null); // the wire the pointer carries, as the graph draws it
  const armed = useRef(0); // a right press over the graph: the browser's own menu is refused for this gesture

  // The browser's own menu never opens over the node graph. A handler on the pane alone is not enough: on Windows the
  // contextmenu event arrives when the button is released, by which time this module's menu is already drawn under the
  // pointer. That menu is not inside the graph's element, so the event never reaches the pane's handler and both menus
  // show at once. On Linux the event comes with the press, so the problem does not occur there. The press itself arms this
  // guard, and the next contextmenu, wherever it lands, is refused.
  useEffect(() => {
    const refuse = (e: MouseEvent) => {
      if (armed.current && performance.now() - armed.current < 2000) e.preventDefault();
      armed.current = 0;
    };
    document.addEventListener("contextmenu", refuse, true);
    return () => document.removeEventListener("contextmenu", refuse, true);
  }, []);

  /** The wire that ends at this input and starts nearest the pointer (a multi input holds several). */
  const wiredInto = useCallback(
    (end: End, at: { clientX: number; clientY: number }): End | null => {
      const into = useCookInputs.getState().edges.filter((e) => e.target === end.node && e.targetHandle === end.port);
      if (!into.length) return null;
      const near = into
        .map((e) => {
          const el = wrap.current?.querySelector(handleSelector({ node: e.source, port: e.sourceHandle }, "output"));
          const r = el?.getBoundingClientRect();
          return { end: { node: e.source, port: e.sourceHandle }, d: r ? Math.hypot(r.left + r.width / 2 - at.clientX, r.top + r.height / 2 - at.clientY) : Infinity };
        })
        .sort((a, b) => a.d - b.d);
      return near[0].end;
    },
    [],
  );

  /** The dashed line from the port the wire comes from to the pointer (drawn in the pane's own pixels). */
  const drawLine = useCallback((x: number, y: number) => {
    const path = line.current;
    const w = wire.current;
    if (!path || !w || !wrap.current) return;
    const box = wrap.current.getBoundingClientRect();
    const el = wrap.current.querySelector(handleSelector(w.from, w.side));
    const a = el ? centreIn(box, el) : { x: x - box.left, y: y - box.top };
    const b = { x: x - box.left, y: y - box.top };
    const out = w.side === "output" ? 1 : -1;
    const bend = Math.max(40, Math.abs(b.x - a.x) * 0.4);
    path.setAttribute("d", `M ${a.x} ${a.y} C ${a.x + out * bend} ${a.y}, ${b.x - out * bend} ${b.y}, ${b.x} ${b.y}`);
  }, []);
  // 平移与缩放时，连线须跟随鼠标与端口移动。有两种情况无法从 pane 获得鼠标位置：① 中键拖动平移时，@xyflow/react（d3-zoom）在 window 上以捕获方式
  // 监听 mousemove 并调用 stopImmediatePropagation，pane 上的 onMouseMove 不会收到任何事件；
  // ② 滚轮缩放时鼠标不动，不产生 mousemove，但端口在屏幕上的位置已改变。因此在 window 的捕获阶段记录鼠标位置
  // （本模块在挂载时即注册，先于 d3 的监听器，不会被拦截），视口变化时按记录的鼠标位置重绘一次
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      mouse.current = { x: e.clientX, y: e.clientY };
      if (wire.current) drawLine(e.clientX, e.clientY);
    };
    window.addEventListener("mousemove", onMove, true);
    return () => window.removeEventListener("mousemove", onMove, true);
  }, [drawLine]);
  // 在下一帧绘制：onChange 在视口数值变化时立即触发，而节点移动到屏幕上的新位置发生在随后的 React 提交阶段
  useOnViewportChange({ onChange: () => { if (wire.current) requestAnimationFrame(() => drawLine(mouse.current.x, mouse.current.y)); } });

  /** The ports the wire can end on are marked while it is carried (the node graph's own feedback, like a drag's). */
  const markPorts = useCallback((w: Wiring) => {
    const s = w ? snapshotNow() : null;
    for (const el of wrap.current?.querySelectorAll<HTMLElement>(".react-flow__handle") ?? []) {
      el.classList.remove("wire-fit", "wire-no");
      const port = portAt(el);
      if (!w || !s || !port || port.side === w.side) continue;
      const [from, to] = w.side === "output" ? [w.from, port.end] : [port.end, w.from];
      el.classList.add(canWire(s, from, to) ? "wire-fit" : "wire-no");
    }
  }, []);

  const carry = useCallback((w: Wiring) => {
    wire.current = w;
    setCarried(w);
  }, []);

  // the ports the carried wire can end on, marked after the graph has drawn (a render would wipe the marks)
  useEffect(() => markPorts(carried), [carried, markPorts]);

  /** Cut the wire from `from` into this input (a multi input keeps its others). */
  const cutWire = useCallback((from: End, input: End) => {
    const edge = useCookInputs.getState().edges.find((e) => e.target === input.node && e.targetHandle === input.port && e.source === from.node && e.sourceHandle === from.port);
    if (edge) deleteElements([], [], [edge.id]);
  }, []);

  /** One step of the machine: what happened goes in, what it answers is carried out. */
  const step = useCallback(
    (event: Parameters<typeof wiring>[1]) => {
      const s = snapshotNow();
      const from = wire.current?.from;
      const { state, action } = wiring(wire.current, event, (o, i) => canWire(s, o, i));
      carry(state);
      if (action.do === "connect") {
        connect({ source: action.from.node, sourceHandle: action.from.port, target: action.to.node, targetHandle: action.to.port });
        if (action.was && !(action.was.node === action.to.node && action.was.port === action.to.port)) cutWire(action.from, action.was);
      } else if (action.do === "cut") {
        if (from) cutWire(from, action.input);
      } else if (action.do === "offer") {
        const p = flow.screenToFlowPosition({ x: action.x, y: action.y });
        openMenu({
          x: action.x,
          y: action.y,
          flowX: action.side === "output" ? p.x + 12 : p.x - 252,
          flowY: p.y - 44,
          wire: { node: action.from.node, port: action.from.port, side: action.side === "output" ? "source" : "target" },
        });
      }
    },
    [carry, cutWire, flow, openMenu],
  );

  // Esc releases a carried wire (the node graph's own layer of the one key registry, platform/keys.ts); with nothing
  // carried the key is not taken, so any other handler receives it.
  useShortcut(
    { keys: ["escape"], run: () => (wire.current ? void step({ at: "cancel" }) : false) },
    { over: () => wrap.current },
  );
  // 空格: held with the left button it pans (@xyflow/react panActivationKeyCode arms it on key down), but a
  // press and release with no drag in between is still 播放 / 暂停, so the playhead keeps working while the pointer rests
  // over the node graph. The key is therefore taken while it is held and, if nothing was dragged, sent again on key up for
  // the page's own space shortcut (the one registry, editor/App.tsx) to handle.
  const space = useRef<ReturnType<typeof heldKey> | null>(null);
  useEffect(() => {
    const k = heldKey(" ", () => pannedWithSpace.current, () => window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true })));
    space.current = k;
    return k.stop;
  }, []);
  useShortcut(
    {
      keys: ["space"],
      run: () => {
        if (space.current?.down() === false) return false; // the key it sent on again: the page's 播放 takes it
        pannedWithSpace.current = false;
        return undefined;
      },
    },
    { over: () => wrap.current },
  );

  // a wire drawn out of a port and let go on empty canvas: the node menu there, with only the node types it can go to;
  // the new node sits with its first port row at the pointer, on the side the wire comes in
  const onConnectEnd: OnConnectEnd = useCallback(
    (e, c) => {
      // let go on a port it fits: connected there. xyflow names the nearest port within its connection radius as
      // `toHandle` even when the wire does not fit it (isValid false): let go on bare canvas near such a port, the menu
      // still opens; let go on the port itself, the check below (not the pane) keeps it shut
      if (!c.fromHandle?.id || (c.toHandle && c.isValid)) return;
      const at = "changedTouches" in e ? e.changedTouches[0] : e;
      const el = document.elementFromPoint(at.clientX, at.clientY);
      // let go over a node's body (not one of its ports): a ports_from-input node (「多层 EXR 输出设置」) adds a row
      // named after what is wired in and connects there, like dropping a new wire on it always does
      if (c.fromHandle.type === "source") {
        const onto = el?.closest<HTMLElement>(".react-flow__node")?.dataset.id;
        if (onto && onto !== c.fromHandle.nodeId && addPortRow(onto, c.fromHandle.nodeId, c.fromHandle.id)) return;
      }
      if (!el?.classList.contains("react-flow__pane")) return;
      const side = c.fromHandle.type;
      const p = flow.screenToFlowPosition({ x: at.clientX, y: at.clientY });
      openMenu({ x: at.clientX, y: at.clientY, flowX: side === "source" ? p.x + 12 : p.x - 252, flowY: p.y - 44, wire: { node: c.fromHandle.nodeId, port: c.fromHandle.id, side } });
    },
    [flow, openMenu],
  );

  // Tab opens the node menu at the mouse, like Houdini / Nuke, while the pointer is over the graph (platform/keys.ts
  // scopes it there, and a real text field keeps its own Tab); Shift+O / Ctrl+G group the selection into a box.
  useShortcut(
    { keys: ["shift+o", "mod+g"], run: () => addBox(flow.screenToFlowPosition(mouse.current)) },
    { enabled: !viewer }, // tabs.ts: editing here is off; adding a group box would be too
  );
  useShortcut(
    { keys: ["tab"], run: () => menuAt(mouse.current.x, mouse.current.y) },
    { over: () => wrap.current, enabled: !viewer }, // the pointer over another pane (the parameter panel): Tab is its own there
  );


  return {
    carrying: !!carried,
    line,
    onConnectEnd,
    handlers: {
      onMouseMove: (e) => {  // window 捕获阶段的监听（见上方）已完成记录与重绘；此处保留是因为 pane 上的其他手势需要同一份鼠标位置
        mouse.current = { x: e.clientX, y: e.clientY };
      },
      // A wire being carried takes the click first: the graph itself must not act on it (button roles: see the module header)
      onPointerDownCapture: (e) => {
        // a button pressed while 空格 is held is the pan gesture (@xyflow/react pans with it): the key did its work and
        // is not 播放 / 暂停 when it is let go
        if (space.current?.held() && e.button === 0) pannedWithSpace.current = true;
        press.current = { x: e.clientX, y: e.clientY, button: e.button, node: nodeUnder(e.target as HTMLElement) };
        tookWire.current = false;
        actedOnPress.current = false;
        const el = e.target as HTMLElement;
        // 中键: the canvas pans (@xyflow/react's panOnDrag = [1]), over a node as well. Nothing is done to the press
        // here, and in particular its default is NOT prevented: preventDefault() on a pointerdown tells the browser to
        // send no compatibility mouse events for it, and @xyflow/react pans from `mousedown` (d3-zoom), so the pan
        // would never start (a browser driven through CDP still gets its mousedown, so automated tests do not catch
        // this). The browser's own middle-click auto-scroll is stopped on the mouse event instead (onMouseDownCapture / onAuxClick below).
        if (e.button === 1) return;
        // 右键: the browser's own menu is refused for this gesture wherever the event lands (see the guard below)
        if (e.button === 2) armed.current = performance.now();
        if (!wire.current) {
          // 左键按在已接线的输入口上：拿起该连线，与点击效果相同，因此拖动时带走的是该连线的端点，
          // 而不会像 @xyflow/react 默认那样新建第二根连线；松开时决定其去向
          const from = e.button === 0 && !viewer ? portAt(el) : null;
          if (from?.side === "input") {
            const wired = wiredInto(from.end, e);
            if (wired) {
              e.preventDefault();
              e.stopPropagation();
              swallow.current = actedOnPress.current = tookWire.current = true;
              step({ at: "port", end: from.end, side: "input", wired });
              drawLine(e.clientX, e.clientY);
            }
          }
          return;
        }
        if (e.button === 2) {
          e.preventDefault();
          e.stopPropagation();
          swallow.current = actedOnPress.current = true;
          step({ at: "cancel" });
          return;
        }
        const port = portAt(el);
        if (port) {
          e.preventDefault();
          e.stopPropagation();
          swallow.current = actedOnPress.current = true;
          step({ at: "port", end: port.end, side: port.side, wired: port.side === "input" ? wiredInto(port.end, e) : null });
          return;
        }
        if (el.classList.contains("react-flow__pane")) {
          e.preventDefault();
          e.stopPropagation();
          swallow.current = actedOnPress.current = true;
          step({ at: "canvas", x: e.clientX, y: e.clientY });
        }
      },
      // a pointerdown this component took (a wire's click) must not reach the node graph as a mouse event either: the
      // two are separate events, and @xyflow/react listens for the mouse one. 中键 is the one press that is only
      // defused, not swallowed: Windows Chrome starts its auto-scroll on this event's default action, while
      // @xyflow/react's own listener (which pans) runs whether or not the default was prevented.
      onMouseDownCapture: (e) => {
        if (e.button === 1) e.preventDefault();
        if (!swallow.current) return;
        swallow.current = false;
        e.preventDefault();
        e.stopPropagation();
      },
      // the middle button let go: the other half of the browser's auto-scroll (and of its「open in a new tab」)
      onAuxClick: (e) => {
        if (e.button === 1) e.preventDefault();
      },
      onPointerUp: (e) => {
        const p = press.current;
        press.current = null;
        const inPlace = !!p && Math.hypot(e.clientX - p.x, e.clientY - p.y) <= CLICK_PX;
        const acted = actedOnPress.current; // the press did this gesture's work (a wire made, a wire picked up, a menu)
        actedOnPress.current = false;
        if (!viewer && e.button === 0) {
          const port = portAt(e.target as HTMLElement);
          // A wire picked up by this gesture's press and dragged away: the release decides where it goes. On a port it
          // is connected; on empty canvas it is cut (or the node menu is offered). A press that only picked it up in place
          // leaves it carried until the next click.
          if (wire.current && tookWire.current && !inPlace) {
            if (port) step({ at: "port", end: port.end, side: port.side, wired: port.side === "input" ? wiredInto(port.end, e) : null });
            else if ((e.target as HTMLElement).classList.contains("react-flow__pane")) step({ at: "canvas", x: e.clientX, y: e.clientY });
            return;
          }
          if (acted) return; // the press already acted: its release does nothing more
          // 左键在端口上原地点击：拿起该连线。从输出口拖出由 @xyflow/react 处理
          // (connectionDragThreshold)，从已接线的输入口拖出已在按下时接管
          if (!wire.current && inPlace && port) {
            step({ at: "port", end: port.end, side: port.side, wired: port.side === "input" ? wiredInto(port.end, e) : null });
            drawLine(e.clientX, e.clientY);
          }
          return;
        }
        // 右键 only opens a menu: the menu of the node under it, or the one that adds a node. It never opens after a press
        // that already acted (a right click that released a carried wire), and never when the press and the release are on two
        // different targets (that is a drag). No distance check here: the right button has nothing to drag, so a hand shift of
        // a few pixels still opens the menu.
        const node = nodeUnder(e.target as HTMLElement);
        if (viewer || acted || e.button !== 2 || !p || p.button !== 2 || node !== p.node) return;
        if (node && !isBox(node) && !unknownIds.includes(node)) (select(node), onNodeMenu({ x: e.clientX, y: e.clientY, id: node }));
        else if (!node) menuAt(e.clientX, e.clientY);
      },
      onContextMenu: (e) => e.preventDefault(), // the browser's menu never opens over the node graph
    },
  };
}
