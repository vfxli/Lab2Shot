/** 选中即取整段：视图一旦查看某个源，即取回该源整段每一帧的压缩字节，不等待用户播放。
 * 流量略大但使用流畅，且单个压缩通道体积不大，彩色通道也只是三倍；不得将其改回懒加载。
 *
 * 触发条件不是播放，而是舞台正在查看该源（`transfer/frames.ts watchFrames` 挂载时即调用此处）：
 * 从当前帧沿播放方向向外排序，覆盖整段所有帧。切换通道即切换到另一个源（id 中含通道名和代理档位），
 * 因此切换后自动开始取回新源的所有帧；切回原通道时其字节仍在缓存中，检查后直接跳过，
 * 不发出任何请求。
 *
 * 取回的是压缩字节，不解码（`FrameSource.fill`）：一帧数十 KB，整段数 MB；
 * 解码后的位图才是主要开销（与代理大小无关，解码后为宽×高×4），该层由 `transfer/cache.ts` 的
 * 另一个预算管理，只保留播放头附近的少量帧。
 *
 * 与取帧账本（`transfer/frames.ts`）的分工：账本负责当前需要绘制的帧（解码，有预算和窗口），
 * 本模块负责取回整段字节（不解码、不进入窗口）。两者使用同一批键，先到者有效。 */

import { cache } from "./cache";
import { ordered } from "./frameWindow";
import { bytesKey, type FrameSource } from "./sources";

/** 后台取回整段时的并发帧数：比舞台低一档，不应抢占当前待绘制帧的连接。
 * 三条用于后台取回、三条用于绘制，恰好为 HTTP/1.1 单个域名的六条连接。 */
const AT_ONCE = 3;

interface Filling {
  source: FrameSource;
  order: number[]; // 全部帧，从播放头沿播放方向向外排序
  done: Set<number>; // 已取回（或原本已持有）的帧
  at: number; // 挂载序号：最近挂载的源优先取回
}

/** 本模块唯一的可变状态（与 `transfer/frames.ts` 的 `Fetching` 做法相同）：
 * 每个源一条，与舞台的窗口同时挂载、同时移除（随 `watchFrames`），因此在构造上即有上限。 */
class Fills {
  readonly per = new Map<string, Filling>();
  readonly now = new Set<string>(); // 当前正在取回的键
  clock = 0;
}

const own = new Fills();

/** 舞台开始查看该源：取回其整段数据（`frame` / `dir`：起始帧与向外排序的方向）。 */
export function fillWhole(source: FrameSource, frame: number, dir: number): void {
  if (!source.fill) return; // 本机文件、浏览器计算的结果：本身不经过网络
  const had = own.per.get(source.id);
  // 同一 id 换了新的源对象：重置「已取回」表。查找原件的请求以「没有」结束时源会重建，id 完全相同
  // （本机没有该文件，id 中的 local 段仍为空）；但原有的 `done` 已包含每一帧：查找期间 `fill` 返回
  // undefined，下方 pump 仍将其记为已取回，若沿用则整段取回将无法再启动。已持有的帧由 `nextOne` 跳过，重置不会产生多余请求
  own.per.set(source.id, { source, order: ordered(source.frames, frame, dir), done: had && had.source === source ? had.done : new Set(), at: ++own.clock });
  pump();
}

/** 该源已无人查看：停止取回。已取回的字节保留在缓存中，切回时无需重新获取。 */
export function stopFilling(id: string): void {
  own.per.delete(id);
}

/** 当前正在取回整段的源数量与在途帧数（仅供测试使用：验证完成后该表为空）。 */
export const fillingNow = (): { sources: number; inFlight: number } => ({ sources: own.per.size, inFlight: own.now.size });

function pump(): void {
  while (own.now.size < AT_ONCE) {
    const next = nextOne();
    if (!next) return;
    const { entry, frame } = next;
    entry.done.add(frame);
    const start = entry.source.fill?.(frame);
    if (!start) continue;
    const key = `${entry.source.id}:${frame}`;
    own.now.add(key);
    void start()
      .catch(() => {
        /* 服务器没有该帧或暂时无法提供：后台取回不报错。实际需要绘制时经由 `load`，由其报告错误 */
      })
      .finally(() => {
        own.now.delete(key);
        pump();
      });
  }
}

/** 下一个待取回的帧：最近挂载的源优先，源内按从播放头向外的顺序；
 * 已持有的（字节或位图）跳过，切回时不重复取回。 */
function nextOne(): { entry: Filling; frame: number } | null {
  for (const entry of [...own.per.values()].sort((a, b) => b.at - a.at)) {
    for (const frame of entry.order) {
      if (entry.done.has(frame)) continue;
      const key = `${entry.source.id}:${frame}`;
      if (cache.has(key) || cache.has(bytesKey(key)) || own.now.has(key)) {
        entry.done.add(frame);
        continue;
      }
      return { entry, frame };
    }
  }
  return null;
}
