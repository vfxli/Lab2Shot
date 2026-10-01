/** 「点选」说成「N 号」的纯规则（editor/pickedPeople.tsx 用；不 import 页面状态，node 里可直接跑）：每个点击（"帧:x,y"）
 * 按 model/people.ts personAt（与服务端「选人」at_point 同一规则，`lab2shot check people` 比对）命中人物框。点得到人的
 * 按号合并、标次数；说不出的（没有人物框、没点在框里）退回「第 F 帧 (x, y)」。 */

import type { BoxesData } from "../api";
import { personAt } from "./people";

export interface PickChip {
  key: string;
  text: string; // 「3 号」「3 号 ×2」，或说不出几号时「第 12 帧 (640, 360)」
  picks: string[]; // 这个 chip 代表的点击（去掉它 = 去掉这些点击）
  known: boolean; // 说得出是几号
}

function parse(text: string): { frame: number; x: number; y: number } | null {
  const [f, rest] = String(text).split(":");
  const [x, y] = (rest ?? "").split(",").map(Number);
  const frame = Number(f);
  return Number.isFinite(frame) && Number.isFinite(x) && Number.isFinite(y) ? { frame, x, y } : null;
}

/** 点击 → chip：点得到人的按号合并（按第一次点的先后），说不出的一次一个。 */
export function pickChips(picks: string[], boxes: BoxesData | null): PickChip[] {
  const byId = new Map<number, PickChip & { count: number }>();
  const out: (PickChip & { count?: number })[] = [];
  picks.forEach((text, i) => {
    const e = parse(text);
    const id = e && boxes ? personAt(boxes, e.frame, { x: e.x, y: e.y }) : null;
    if (id === null) {
      out.push({ key: `${i}:${text}`, text: e ? `第 ${e.frame} 帧 (${Math.round(e.x)}, ${Math.round(e.y)})` : text, picks: [text], known: false });
      return;
    }
    const had = byId.get(id);
    if (had) {
      had.picks.push(text);
      had.count++;
      had.text = `${id} 号 ×${had.count}`;
      return;
    }
    const chip = { key: `id:${id}`, text: `${id} 号`, picks: [text], known: true, count: 1 };
    byId.set(id, chip);
    out.push(chip);
  });
  return out.map(({ key, text, picks: ps, known }) => ({ key, text, picks: ps, known }));
}

/** 节点上的一行摘要：「点选：2 号、3 号」（说不出几号的记作「N 处」）。没有点击时为 ""。 */
export function pickSummary(chips: PickChip[], boxesKnown = false): string {
  if (!chips.length) return "";
  const known = chips.filter((c) => c.known).map((c) => c.text.replace(/ ×\d+$/, ""));
  const unknown = chips.length - known.length;
  return `点选：${[...known, ...(unknown ? [boxesKnown ? `${unknown} 处没点到人` : `${unknown} 处（先检测人物才知道是几号）`] : [])].join("、")}`;
}

