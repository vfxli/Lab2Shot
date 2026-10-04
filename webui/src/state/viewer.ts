import { create } from "zustand";
import { nearest } from "../model/timelineMath";
import type { Dir } from "../model/timelineMath";
import { randomId } from "../platform/randomId";

export { useView2D, useView2DNav } from "./view2d";
export type { Nav2D, View2D, ViewSlot } from "./view2d";
export { VIEW_NAMES, VIEWER_SLOT, slotCamera, useCurveView, useRulerView, useStageNotes, useStagePicture, useViewCamera, useViewLoads, useViewOptions, useViewerNote } from "./viewTools";
export type { FrameWindow, SlotCamera, ViewName } from "./viewTools";

/** 视图: everything that belongs to the browser tab looking at the document, never the document itself — selecting a
 * node, panning the 2D view, tumbling the 3D camera, picking the 2D view's mode, opening a menu, the log. None of it is undoable, none of it is sent to the server, none of it makes state/results.ts stop
 * trusting a result, and nothing here has a `version`: nothing here decides whether anything else is stale. Only the
 * display options are kept (in localStorage); everything else resets when the tab closes.
 *
 * This file gathers the viewer's zustand stores (the 2D view and the display-option stores are re-exported from
 * view2d.ts and viewTools.ts) and the viewer fields: selection, menus, playback,
 * expanded nodes, the parameter panel's flash, where a new node should pan the view, the templates and log sheets,
 * and this tab's editing role. `canvas` is the node editor's own per-id ephemera (selected/dragging/measured): xyflow
 * needs them, but they are no more part of the undoable document than a mouse hovering over a node is, so they are
 * kept apart from position and params and history.ts does not count them. */

interface Reveal {
  node: string;
  param: string;
  focus: boolean;
  n: number;
}

export interface LooseWire {
  node: string;
  port: string;
  side: "source" | "target";
}

interface CanvasNode {
  selected?: boolean;
  dragging?: boolean;
  measured?: { width?: number; height?: number };
}


interface State {
  // ---- selection, panels, chrome ----
  selectedId: string | null;
  expanded: string[]; // nodes whose body shows every simple parameter for now (a view of the moment, not the graph's)
  reveal: Reveal | null;
  panTo: { x: number; y: number; n: number } | null;
  inspectorFit: number; // the width (px) the parameter panel needs to show the selected node's parameters in full
  menu: { x: number; y: number; flowX: number; flowY: number; wire?: LooseWire } | null;
  templatesOpen: boolean;
  logOpen: boolean;
  frames: number[]; // the frames the timeline follows: the displayed node's results (or the plate it works on)
  frame: number;
  playing: boolean;
  // the timeline is being dragged: no frames are fetched until release. Frames dragged across are mostly just passed
  // through, and fetching each would waste traffic (bandwidth is limited)
  scrubbing: boolean;
  playDir: Dir;
  // playback speed: a view setting, not data. A shot has no frame rate of its own (the rate is stated once, on the output
  // settings node), so this number comes from no packet, is not in the graph and does not affect written files
  fps: number;
  loads: number; // bumps whenever another graph is loaded (NodeEditor.tsx fits the view)

  // ---- this working copy's identity and save state in this tab (tabs.ts, history.ts) ----
  // As browser-local and reset-on-load as everything else here, and unrelated to the server's check of the graph:
  // `file` is which of the user's own files this tab is editing (never sent to the server), `dirty` and the
  // undo/redo labels are this tab's own read of graph/history.ts. None of them make anything else stale.
  docId: string;
  role: "editor" | "viewer";
  peerBanner: "same-open" | "demoted" | null;
  file: { name: string; handle?: string } | null;
  dirty: boolean;
  undoLabel: (() => string) | null; // said when shown (graph/history.ts Said: it follows the page's language)
  redoLabel: (() => string) | null;

  // ---- the node editor's own canvas ephemera (never saved, never undone) ----
  canvas: Record<string, CanvasNode>;
  selectedEdgeIds: string[];
  selectedBoxIds: string[];

  select: (id: string | null) => void;
  toggleExpanded: (id: string) => void;
  revealParam: (id: string, name: string, focus?: boolean) => void;
  requestPan: (x: number, y: number) => void;
  setInspectorFit: (px: number) => void;
  openMenu: (m: State["menu"]) => void;
  setTemplatesOpen: (o: boolean) => void;
  setLogOpen: (open: boolean) => void;
  setFrame: (f: number) => void;
  setFrames: (frames: number[], current: number) => void;
  togglePlay: () => void;
  play: (dir: Dir) => void;
  setPlaying: (playing: boolean) => void;
  setScrubbing: (scrubbing: boolean) => void;

  claimEditing: () => void;
  stayViewer: () => void;
  freshDoc: () => void; // a fresh graph is nobody else's until proven otherwise (tabs.ts's reset())
  setFile: (f: State["file"]) => void;
  setSaveState: (dirty: boolean, undoLabel: (() => string) | null, redoLabel: (() => string) | null) => void;

  setCanvasNode: (id: string, patch: CanvasNode) => void;
  removeCanvasNodes: (ids: string[]) => void;
  setSelectedNodes: (ids: string[]) => void; // marquee-select / delete: every id's `.selected` set at once
  setSelectedEdges: (ids: string[]) => void;
  setSelectedBoxes: (ids: string[]) => void;
  setEdgeSelected: (id: string, on: boolean) => void; // one wire at a time (xyflow's own per-id "select" change)
  setBoxSelected: (id: string, on: boolean) => void;

  reset: () => void; // another graph loaded: selection, panels and canvas ephemera do not carry over
}

export const useViewer = create<State>((set, get) => ({
  selectedId: null,
  expanded: [],
  reveal: null,
  panTo: null,
  inspectorFit: 0,
  menu: null,
  templatesOpen: false,
  logOpen: false,
  frames: [],
  frame: 1001,
  playing: false,
  scrubbing: false,
  playDir: 1,
  fps: 24,
  loads: 0,

  docId: randomId(),
  role: "editor",
  peerBanner: null,
  file: null,
  dirty: false,
  undoLabel: null,
  redoLabel: null,

  canvas: {},
  selectedEdgeIds: [],
  selectedBoxIds: [],

  select: (id) => set((s) => (s.selectedId === id ? {} : { selectedId: id })),
  toggleExpanded: (id) => set((s) => ({ expanded: s.expanded.includes(id) ? s.expanded.filter((x) => x !== id) : [...s.expanded, id] })),
  revealParam: (id, name, focus = false) => set((s) => ({ selectedId: id, reveal: { node: id, param: name, focus, n: (s.reveal?.n ?? 0) + 1 } })),
  requestPan: (x, y) => set((s) => ({ panTo: { x, y, n: (s.panTo?.n ?? 0) + 1 } })),
  setInspectorFit: (px) => get().inspectorFit !== px && set({ inspectorFit: px }),
  openMenu: (m) => set({ menu: m }),
  setTemplatesOpen: (o) => set({ templatesOpen: o }),
  setLogOpen: (open) => set({ logOpen: open }),
  // the current frame is one the timeline has (as setFrames keeps it): a frame asked for between them (a figure's frame
  // chip, a typed number) lands on the nearest, so stepping from it goes the way asked
  setFrame: (f) => set((s) => ({ frame: s.frames.length ? nearest(s.frames, f) : f })),
  // the frame shown stays as near as the new frames allow (model/timelineMath.ts nearest, the timeline's one rule for
  // landing on a frame: scrubbing, stepping and curves use it too), not thrown back to the first
  setFrames: (fr, current) => set({ frames: fr, frame: nearest(fr, current) }),
  setScrubbing: (scrubbing) => set({ scrubbing }),
  togglePlay: () => set((s) => ({ playing: !s.playing })),
  play: (dir) => set((s) => ({ playing: !(s.playing && s.playDir === dir), playDir: dir })),
  setPlaying: (playing) => set({ playing }),

  claimEditing: () => set({ role: "editor", peerBanner: null }),
  stayViewer: () => set({ peerBanner: null }),
  freshDoc: () => set({ role: "editor", peerBanner: null }),
  setFile: (f) => set({ file: f }),
  setSaveState: (dirty, undoLabel, redoLabel) => set({ dirty, undoLabel, redoLabel }),

  setCanvasNode: (id, patch) =>
    set((s) => {
      const before = s.canvas[id];
      const after = { ...before, ...patch };
      // xyflow reports a "dimensions" NodeChange (ResizeObserver) each time a node's box is (re)measured, sometimes
      // repeatedly for the same, unchanged size; without this equality check, an identical measurement (same
      // width/height, in a freshly allocated object) would still replace `canvas` with a new object, which
      // editor/NodeEditor.tsx's `liveNodes` would treat as new node content, needlessly rebuilding and
      // re-measuring. Compared field by field, `measured` by its width/height rather than object identity.
      const same =
        !!before &&
        before.selected === after.selected &&
        before.dragging === after.dragging &&
        before.measured?.width === after.measured?.width &&
        before.measured?.height === after.measured?.height;
      return same ? {} : { canvas: { ...s.canvas, [id]: after } };
    }),
  removeCanvasNodes: (ids) =>
    set((s) => {
      const canvas = { ...s.canvas };
      for (const id of ids) delete canvas[id];
      return { canvas };
    }),
  setSelectedNodes: (ids) =>
    set((s) => {
      const on = new Set(ids);
      const canvas = { ...s.canvas };
      for (const id of new Set([...Object.keys(canvas), ...ids])) canvas[id] = { ...canvas[id], selected: on.has(id) };
      return { canvas };
    }),
  setSelectedEdges: (ids) => set({ selectedEdgeIds: ids }),
  setSelectedBoxes: (ids) => set({ selectedBoxIds: ids }),
  setEdgeSelected: (id, on) => set((s) => ({ selectedEdgeIds: on ? [...new Set([...s.selectedEdgeIds, id])] : s.selectedEdgeIds.filter((x) => x !== id) })),
  setBoxSelected: (id, on) => set((s) => ({ selectedBoxIds: on ? [...new Set([...s.selectedBoxIds, id])] : s.selectedBoxIds.filter((x) => x !== id) })),

  reset: () =>
    set((s) => ({
      selectedId: null, expanded: [], reveal: null, menu: null, canvas: {}, selectedEdgeIds: [], selectedBoxIds: [],
      frames: [], frame: 1001, playing: false, scrubbing: false, playDir: 1, fps: 24, loads: s.loads + 1,
    })),
}));

export const framesShown = (): number[] => useViewer.getState().frames;
