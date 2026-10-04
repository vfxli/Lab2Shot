import { create } from "zustand";
import type { DataType, Scope, ScopeItem, ScopeList, StatusReply } from "../api";

/** 逐项处理 blocks as read by the page. Every rule of a block (its membership, nesting, item count and item names) is
 * computed once on the server (lab2shot/engine/scopes.py) and delivered in the status reply's `scopes`. Nothing here
 * traverses wires to find a block's members; `blocksOf` only reads that list.
 *
 * The page owns only which item each block is showing: 「当前条目」 is view state. Changing it does not change the cook
 * inputs, cooks nothing, and leaves every result intact. It is sent with the next status request (`view`,
 * graph/actions.ts refreshStatus) so that the server answers with that item's state for the nodes inside the block.
 */

export type { Scope, ScopeItem, ScopeList };

// ------------------------------------------------------------------ list types

const LIST = "[]";

/** Whether this is a list type (`image[]`, `scene.character[]`). Of candidate types (`a|b`): only when every candidate
 * is one, the same rule as lab2shot/data/types.py is_list. */
export const isList = (type: string): boolean => type.split("|").every((t) => t.endsWith(LIST));

/** The type of one item (`image[]` -> `image`); a non-list type maps to itself; candidate types one by one. */
export const elementOf = (type: string): string =>
  [...new Set(type.split("|").map((t) => (t.endsWith(LIST) ? t.slice(0, -LIST.length) : t)))].join("|");

/** The catalogue's data type for a port's declared type. A port type may be a union (`scene.skeleton|scene.character`:
 * the 「动画」 port of the skeleton-motion family, whose type depends on what is wired to it), while the catalogue has
 * single types only, so the first one is taken. The members of a union belong to the same class in the view (all are
 * 3D elements).
 *
 * Every lookup of the catalogue by port type goes through this function: `types[portType]` misses a union, and the view
 * would then judge the node's result to be neither a picture nor a 3D element. */
export const typeOf = (types: Record<string, DataType>, portType: string): DataType | undefined =>
  types[elementOf(portType).split("|")[0]];

/** The label 「逐个：…」 shows for a block: the name of one item of the begin node's list, i.e. the element type's own word
 * for one of the things it holds (人物框 -> 人物), else the element type's name. "" while the type is not yet known. */
export function itemWord(types: Record<string, DataType>, elementType: string): string {
  const t = typeOf(types, elementType);
  return t ? t.items || t.label : "";
}

// ------------------------------------------------------------------ the blocks of a graph (read only)

/** The blocks of the last status reply, in the order the server lists them. */
export const blocksOf = (reply: StatusReply | null): Scope[] => reply?.scopes ?? [];

/** Every node of a block, once each: its begin node (which the server counts among the members), the members and the
 * end nodes. The frame is drawn around these. */
export const nodesOf = (s: Scope): string[] => [...new Set([s.begin, ...s.members, ...s.ends])];

/** The blocks containing a node, outermost first ([] outside every block): its chain, derived from membership alone.
 * The server counts a block's 逐项开始 among its members but not its 逐项结束 (engine/scopes.py: an end node sits one
 * level out and gathers the members' results for every item, so it has one instance rather than one per item). */
export function chainOf(blocks: Scope[], nodeId: string): Scope[] {
  const holding = blocks.filter((s) => s.members.includes(nodeId));
  // outermost first: a block whose begin node is another block's member lies inside it
  const depth = (s: Scope): number => {
    let n = 0;
    for (let p = s.parent; p; ) {
      n += 1;
      p = blocks.find((x) => x.begin === p)?.parent ?? null;
    }
    return n;
  };
  return holding.sort((a, b) => depth(a) - depth(b));
}

/** The items of a block at the path its enclosing blocks are showing: the list for that path (null: not yet known to
 * the server, 等上游). */
export function itemsOf(s: Scope, parentPath: string[]): ScopeItem[] | null {
  const list = listAt(s, parentPath);
  return !list || list.pending ? null : list.items ?? [];
}

/** The block's entry for the path the blocks around it are showing. */
function listAt(s: Scope, parentPath: string[]): ScopeList | undefined {
  const same = (a: string[]) => a.length === parentPath.length && a.every((k, i) => k === parentPath[i]);
  return s.lists.find((l) => same(l.path)) ?? (s.lists.length === 1 && !parentPath.length ? s.lists[0] : undefined);
}

/** The item path the view is on for a node: one key per enclosing block, outermost first. A block whose items are not
 * yet known ends the path (the node has no instance to show). */
export function viewPath(blocks: Scope[], nodeId: string, view: Record<string, string>): string[] {
  const path: string[] = [];
  for (const s of chainOf(blocks, nodeId)) {
    const items = itemsOf(s, path);
    if (!items?.length) return path;
    const want = view[s.begin];
    path.push(items.some((i) => i.key === want) ? want : items[0].key);
  }
  return path;
}

/** The item each block is showing, in the form the status request sends (`view`: begin -> item key). Only blocks that
 * currently exist are included; a key left over from a removed block is never sent. */
export function viewForAsk(blocks: Scope[], view: Record<string, string>): Record<string, string> {
  const out: Record<string, string> = {};
  for (const s of blocks) if (view[s.begin]) out[s.begin] = view[s.begin];
  return out;
}

// ------------------------------------------------------------------ block and node summaries

export interface BlockCount {
  items: number; // the block's current item count (0 while pending)
  pending: boolean; // the item count is not yet known (等上游)
  done: number; // items for which every node of the block has a result
  failed: number; // items on which a node of the block failed
}

/** A block's status, exactly as reported by the server (engine/evaluation.py _scope_summary): what counts as done or
 * failed for a whole block is a rule, and rules are computed once, on the server. */
export function countOf(s: Scope, parentPath: string[] = []): BlockCount {
  const list = listAt(s, parentPath);
  if (!list || list.pending || !list.items) return { items: 0, pending: true, done: 0, failed: 0 };
  const said = list.summary;
  return { items: list.items.length, pending: false, done: said?.cached ?? 0, failed: said?.failed ?? 0 };
}

// ------------------------------------------------------------------ frame placement

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

const BLOCK_HEAD = 26; // the title pill's row above the nodes
const BLOCK_PAD = 18; // padding around the nodes inside the frame

/** The frame around a block's nodes: the bounding box of all of them, with padding and its title row on top. null when
 * none of them has a position yet. */
export function frameOf(ids: string[], boxes: Record<string, Rect | undefined>): Rect | null {
  const found = ids.map((id) => boxes[id]).filter((r): r is Rect => !!r);
  if (!found.length) return null;
  const x = Math.min(...found.map((r) => r.x)) - BLOCK_PAD;
  const y = Math.min(...found.map((r) => r.y)) - BLOCK_PAD - BLOCK_HEAD;
  const right = Math.max(...found.map((r) => r.x + r.w)) + BLOCK_PAD;
  const bottom = Math.max(...found.map((r) => r.y + r.h)) + BLOCK_PAD;
  return { x, y, w: right - x, h: bottom - y };
}

/** The number of colours a block frame is chosen from (ui/tokens.css --block-1..4: neutral and cool, none of them a
 * status colour or a data type colour, since a block name says nothing about state). */
const BLOCK_COLOURS = 4;

/** The colour a block receives: derived from its name, so the same name has the same colour in every graph and two
 * blocks in one graph rarely share one. The colour itself is defined in the style sheet (the frame carries the index). */
export function colourOf(name: string, colours = BLOCK_COLOURS): number {
  let n = 0;
  for (let i = 0; i < name.length; i++) n = (n * 31 + name.charCodeAt(i)) % 1000003;
  return (n % colours) + 1;
}

// ------------------------------------------------------------------ view state

interface Items {
  /** begin -> the item key the view is on (视图状态, not a cook input). */
  view: Record<string, string>;
  setItem: (begin: string, key: string) => void;
  reset: () => void;
}

/** 当前条目 per block: a 视图 setting (state/viewer.ts). Cleared when another graph is opened. */
export const useItems = create<Items>((set) => ({
  view: {},
  setItem: (begin, key) => set((s) => (s.view[begin] === key ? {} : { view: { ...s.view, [begin]: key } })),
  reset: () => set({ view: {} }),
}));
