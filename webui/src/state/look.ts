import { create } from "zustand";
import type { BoxJSON } from "../api";

/** 文档外观: saved with the graph file, undoable, but never part of the server's check of the graph
 * (state/cookInputs.ts) — moving a node, folding a group box, showing another node or scrubbing the playback range
 * leaves every cached result standing. `version` bumps on every change here too (state/results.ts never compares
 * against this one), so autosave (editor/autosave.ts) and the graph snapshot (graph/snapshot.ts) can tell "the
 * document changed" apart from "the cook inputs changed". Undo steps are recorded by graph/document.ts on any change
 * of either store. */

export interface Pos {
  x: number;
  y: number;
}

export type Box = BoxJSON;

interface State {
  positions: Record<string, Pos>;
  // 这个节点身上显示哪几行参数，当用户自己选过（没选过就是 undefined：按节点类型声明的 NodeDef.on_node 来）。
  onNode: Record<string, string[] | undefined>;
  boxes: Box[];
  displayId: string | null;
  displayPort: string | null; // which output of the displayed node the viewer shows (null = the first)
  playback: [number, number] | null; // in / out, kept with the graph file's view (not an undo step: set by hand)
  version: number;

  load: (p: { positions: Record<string, Pos>; onNode: Record<string, string[] | undefined>; boxes: Box[]; displayId: string | null; displayPort: string | null; playback: [number, number] | null }) => void;
  setPosition: (id: string, x: number, y: number) => void;
  removeNodes: (ids: string[]) => void;
  setOnNode: (id: string, rows: string[] | undefined) => void; // undefined：这个节点回到类型声明的那几行
  setDisplay: (id: string | null) => void;
  setDisplayPort: (port: string | null) => void;
  setPlayback: (r: [number, number] | null) => void; // does not bump version: not an undo step
  addBox: (box: Box) => void;
  setBox: (id: string, patch: Partial<Box>) => void;
  moveBox: (id: string, x: number, y: number, members: string[]) => void; // nodes riding along while it is dragged
  removeBoxes: (ids: string[]) => void;
}

export const useLook = create<State>((set) => ({
  positions: {},
  onNode: {},
  boxes: [],
  displayId: null,
  displayPort: null,
  playback: null,
  version: 0,

  load: (p) => set((s) => ({ ...p, version: s.version + 1 })),
  setPosition: (id, x, y) => set((s) => ({ positions: { ...s.positions, [id]: { x, y } }, version: s.version + 1 })),
  removeNodes: (ids) =>
    set((s) => {
      const positions = { ...s.positions };
      const onNode = { ...s.onNode };
      for (const id of ids) (delete positions[id], delete onNode[id]);
      return { positions, onNode, version: s.version + 1 };
    }),
  setOnNode: (id, rows) => set((s) => ({ onNode: { ...s.onNode, [id]: rows }, version: s.version + 1 })),
  setDisplay: (id) => set((s) => ({ displayId: id, displayPort: s.displayId === id ? s.displayPort : null, version: s.version + 1 })),
  setDisplayPort: (port) => set((s) => ({ displayPort: port, version: s.version + 1 })),
  setPlayback: (r) => set({ playback: r }),
  addBox: (box) => set((s) => ({ boxes: [...s.boxes, box], version: s.version + 1 })),
  setBox: (id, patch) => set((s) => ({ boxes: s.boxes.map((b) => (b.id === id ? { ...b, ...patch } : b)), version: s.version + 1 })),
  moveBox: (id, x, y, members) =>
    set((s) => {
      const box = s.boxes.find((b) => b.id === id);
      if (!box) return {};
      const dx = x - box.x;
      const dy = y - box.y;
      if (!dx && !dy) return {};
      const positions = { ...s.positions };
      for (const m of members) if (positions[m]) positions[m] = { x: positions[m].x + dx, y: positions[m].y + dy };
      return { boxes: s.boxes.map((b) => (b.id === id ? { ...b, x, y } : b)), positions, version: s.version + 1 };
    }),
  removeBoxes: (ids) => set((s) => ({ boxes: s.boxes.filter((b) => !ids.includes(b.id)), version: s.version + 1 })),
}));
