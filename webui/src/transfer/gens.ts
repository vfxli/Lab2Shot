/** 各数据包的生成号（服务器状态回复中每个端口的 `gens`，即包的提交时刻，见 lab2shot/data/packet.py 的 created）。
 *
 * 页面缓存键由结构体生成（transfer/frameKey.ts），生成号是其组成部分之一：同一指纹重算后得到另一组键，
 * 旧帧因此不会被命中。服务器重算数据包后，页面在下一次状态回复中即可得知（生成号变了），无需等待 LRU 淘汰旧帧。
 *
 * 表按回复合并、不按回复整张替换：回复里只有当前参数下的结果，而参数刚改、还没重算时视图仍画上一次的结果
 * （state/stale.ts），那些包不在回复里，却还要用它们的生成号认出缓存里的帧。回复里不再出现的包按最近出现的先后
 * 保留至多 KEEP 个，更早的删掉（删掉后再用到时按「未知」处理：键不带生成号，照样能取，只是重取一次）。
 * 包被服务器删掉而没有重算的，页面不需要知道：再要它时服务器答 404，由 transfer/frameStore.ts 记为「没有」。
 *
 * 空串表示尚未知晓（状态回复尚未带来该包的生成号）。本模块是一个 zustand 仓库，舞台据此重建帧源
 * （view/stageSources.ts 的 memo 将其列为依赖）。 */
import { create } from "zustand";

/** 回复里不再出现的包，最多留多少个的生成号。 */
const KEEP = 5000;

interface Gens {
  gens: Record<string, string>;
}

export const useGens = create<Gens>(() => ({ gens: {} }));

/** 返回该指纹当前的生成号（"" 表示未知）。供 React 之外的代码使用；组件应使用 `useGens`。 */
export const genOf = (fp: string): string => useGens.getState().gens[fp] ?? "";

/** 并入一次状态回复带来的表（见文件头：合并，回复里没有的按先后留至多 KEEP 个）。 */
export function setGens(next: Record<string, string>): void {
  const was = useGens.getState().gens;
  const same = Object.keys(next).every((fp) => was[fp] === next[fp]);
  const absent = Object.keys(was).filter((fp) => !(fp in next));
  if (same && absent.length <= KEEP) return;
  // 对象的键按插入先后：回复里的放到最后（最近出现），回复里没有的只留最近的 KEEP 个
  const kept = absent.slice(Math.max(0, absent.length - KEEP));
  useGens.setState({ gens: { ...Object.fromEntries(kept.map((fp) => [fp, was[fp]])), ...next } });
}
