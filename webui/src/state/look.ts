import { create } from "zustand";
import type { BoxJSON } from "../api";
import { readOnly } from "./cookInputs";

/** 文档外观：随节点图文件保存、可撤销，但从不参与服务器对节点图的检查（state/cookInputs.ts）——挪节点、折叠框、
 * 显示另一个节点、拖动播放范围，缓存的结果都照样成立。这里每次改动也会让 `version` 加一（state/results.ts 从不拿它
 * 比对），自动保存（editor/autosave.ts）和节点图快照（graph/snapshot.ts）据此区分「文档改了」与「计算输入改了」。
 * 两个仓库任一有改动，graph/document.ts 都记一步撤销。 */

export interface Pos {
  x: number;
  y: number;
}

export type Box = BoxJSON;

/** 节点备注（Houdini 的 node comment）：任意文字，不参与任何引用、不进消息；`show`：「显示备注」开关，打开时淡色显示在节点旁边。 */
export interface NodeComment {
  text: string;
  show: boolean;
}

interface State {
  positions: Record<string, Pos>;
  // 这个节点身上显示哪几行参数，当用户自己选过（没选过就是 undefined：按节点类型声明的 NodeDef.on_node 来）。
  onNode: Record<string, string[] | undefined>;
  comments: Record<string, NodeComment | undefined>; // 节点备注（没有就是 undefined）
  boxes: Box[];
  displayId: string | null;
  displayPort: string | null; // 视图显示的是显示节点的哪个输出（null：第一个）
  playback: [number, number] | null; // 入点 / 出点，随节点图文件的视图保存（不是撤销步骤：手动设定）
  version: number;

  load: (p: { positions: Record<string, Pos>; onNode: Record<string, string[] | undefined>; comments: Record<string, NodeComment | undefined>; boxes: Box[]; displayId: string | null; displayPort: string | null; playback: [number, number] | null }) => void;
  setPosition: (id: string, x: number, y: number) => void;
  removeNodes: (ids: string[]) => void;
  setOnNode: (id: string, rows: string[] | undefined) => void; // undefined：这个节点回到类型声明的那几行
  setComment: (id: string, comment: NodeComment | undefined) => void; // undefined：没有备注
  renameNode: (from: string, to: string) => void; // 节点改名（graph/naming.ts）：位置、节点体上的行、备注、框的成员、显示节点跟着改
  setDisplay: (id: string | null) => void;
  setDisplayPort: (port: string | null) => void;
  setPlayback: (r: [number, number] | null) => void; // 不加 version：不是撤销步骤
  addBox: (box: Box) => void;
  setBox: (id: string, patch: Partial<Box>) => void;
  moveBox: (id: string, x: number, y: number, members: string[]) => void; // 拖动框时跟着一起挪的节点
  removeBoxes: (ids: string[]) => void;
  /** 一次挪很多节点和框（graph/edit.ts arrangeGraph「整理节点图」）：没给的照旧 */
  place: (positions: Record<string, Pos>, boxes: Record<string, Pick<Box, "x" | "y" | "w" | "h">>) => void;
}

export const useLook = create<State>((set) => {
  // 改文档外观的写入（位置、框、节点体上的名单）：只读标签页里拒绝，同 state/cookInputs.ts readOnly。显示哪个节点、
  // 播放范围是看的动作，照常
  const edit: typeof set = (...a) => (readOnly() ? undefined : set(...(a as Parameters<typeof set>)));
  return {
    positions: {},
    onNode: {},
    comments: {},
    boxes: [],
    displayId: null,
    displayPort: null,
    playback: null,
    version: 0,

    load: (p) => set((s) => ({ ...p, version: s.version + 1 })),
    setPosition: (id, x, y) => edit((s) => ({ positions: { ...s.positions, [id]: { x, y } }, version: s.version + 1 })),
    removeNodes: (ids) =>
      edit((s) => {
        const positions = { ...s.positions };
        const onNode = { ...s.onNode };
        const comments = { ...s.comments };
        for (const id of ids) (delete positions[id], delete onNode[id], delete comments[id]);
        // 折叠框的成员名单里也去掉（展开的框按位置算成员，没有名单要改）
        const boxes = s.boxes.map((b) => (b.members.some((m) => ids.includes(m)) ? { ...b, members: b.members.filter((m) => !ids.includes(m)) } : b));
        return { positions, onNode, comments, boxes, version: s.version + 1 };
      }),
    setOnNode: (id, rows) => edit((s) => ({ onNode: { ...s.onNode, [id]: rows }, version: s.version + 1 })),
    setComment: (id, comment) =>
      edit((s) => {
        const was = s.comments[id];
        if (was?.text === comment?.text && was?.show === comment?.show) return {};
        return { comments: { ...s.comments, [id]: comment }, version: s.version + 1 };
      }),
    renameNode: (from, to) =>
      edit((s) => {
        const move = <T,>(r: Record<string, T>): Record<string, T> => {
          if (!(from in r)) return r;
          const { [from]: v, ...rest } = r;
          return { ...rest, [to]: v };
        };
        const boxes = s.boxes.map((b) => (b.members.includes(from) ? { ...b, members: b.members.map((m) => (m === from ? to : m)) } : b));
        return { positions: move(s.positions), onNode: move(s.onNode), comments: move(s.comments), boxes, displayId: s.displayId === from ? to : s.displayId, version: s.version + 1 };
      }),
    setDisplay: (id) => set((s) => ({ displayId: id, displayPort: s.displayId === id ? s.displayPort : null, version: s.version + 1 })),
    setDisplayPort: (port) => set((s) => ({ displayPort: port, version: s.version + 1 })),
    setPlayback: (r) => set({ playback: r }),
    addBox: (box) => edit((s) => ({ boxes: [...s.boxes, box], version: s.version + 1 })),
    setBox: (id, patch) => edit((s) => ({ boxes: s.boxes.map((b) => (b.id === id ? { ...b, ...patch } : b)), version: s.version + 1 })),
    moveBox: (id, x, y, members) =>
      edit((s) => {
        const box = s.boxes.find((b) => b.id === id);
        if (!box) return {};
        const dx = x - box.x;
        const dy = y - box.y;
        if (!dx && !dy) return {};
        const positions = { ...s.positions };
        for (const m of members) if (positions[m]) positions[m] = { x: positions[m].x + dx, y: positions[m].y + dy };
        return { boxes: s.boxes.map((b) => (b.id === id ? { ...b, x, y } : b)), positions, version: s.version + 1 };
      }),
    removeBoxes: (ids) => edit((s) => ({ boxes: s.boxes.filter((b) => !ids.includes(b.id)), version: s.version + 1 })),
    place: (positions, boxes) =>
      edit((s) => ({ positions: { ...s.positions, ...positions }, boxes: s.boxes.map((b) => (boxes[b.id] ? { ...b, ...boxes[b.id] } : b)), version: s.version + 1 })),
  };
});
