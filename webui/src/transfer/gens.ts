/** 各数据包的生成号（服务器状态回复中每个端口的 `gens`，即包的提交时刻，见 lab2shot/data/packet.py 的 created）。
 *
 * 页面缓存键由结构体生成（transfer/ident.ts），生成号是其组成部分之一：同一指纹重算后得到另一组键，
 * 旧帧因此不会被命中。服务器删除或重算数据包后，页面在下一次状态回复中即可得知，无需等待 LRU 淘汰旧帧。
 * 本模块维护一张表（fp -> created），由 state/results.ts 在每次状态回复后合并；空串表示尚未知晓（状态回复尚未带来该包的生成号）。
 * 本模块是一个 zustand 仓库，舞台据此重建帧源（view/stageSources.ts 的 memo 将其列为依赖）。 */
import { create } from "zustand";

interface Gens {
  gens: Record<string, string>;
  set: (next: Record<string, string>) => string[];
}

const listeners = new Set<(changed: string[]) => void>();

export const useGens = create<Gens>((set, get) => ({
  gens: {},
  set: (next) => {
    const was = get().gens;
    const changed = Object.keys(next).filter((fp) => was[fp] !== undefined && was[fp] !== next[fp]);
    if (changed.length || Object.keys(next).some((fp) => was[fp] === undefined)) set({ gens: { ...was, ...next } });
    if (changed.length) listeners.forEach((f) => f(changed));
    return changed;
  },
}));

/** 返回该指纹当前的生成号（"" 表示未知）。供 React 之外的代码使用；组件应使用 `useGens`。 */
export const genOf = (fp: string): string => useGens.getState().gens[fp] ?? "";

/** 合并状态回复带来的新表，并通知监听者哪些指纹的生成号发生了变化（即已重算）。 */
export const setGens = (next: Record<string, string>): void => void useGens.getState().set(next);

export function onGenChanged(listener: (changed: string[]) => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
