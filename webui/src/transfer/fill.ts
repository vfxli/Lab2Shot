/** 选中即取整段：视图一旦查看某个源，即取回该源整段每一帧的压缩字节，不等待用户播放。
 * 流量略大但使用流畅，且单个压缩通道体积不大，彩色通道也只是三倍；因此这里不做懒加载。
 *
 * 触发条件不是播放，而是舞台正在查看该源（`transfer/frames.ts watchFrames` 挂载时即调用此处）：
 * 从当前帧沿播放方向向外排序，覆盖整段所有帧。切换通道即切换到另一个源（id 中含通道名和代理档位），
 * 因此切换后自动开始取回新源的所有帧；切回原通道时其字节仍在缓存中，检查后直接跳过，
 * 不发出任何请求。
 *
 * 取回的是压缩字节，不解码（`FrameSource.fill`）：一帧数十 KB，整段数 MB；
 * 解码后的位图才是主要开销（与代理大小无关，解码后为宽×高×4），该层由 `platform/cache.ts` 的
 * 另一个预算管理，只保留播放头附近的少量帧。
 *
 * 与取帧账本（`transfer/frames.ts`）的分工：账本负责当前需要绘制的帧（解码，有预算和窗口），
 * 本模块负责取回整段字节（不解码、不进入窗口；长镜头装不下整段时只取播放头周围放得下的那段：reachOf）。两者使用同一批键，先到者有效。 */

import { BYTES_BUDGET, cache } from "../platform/cache";
import { aheadFor, ordered } from "./frameWindow";
import type { FrameSource } from "./sources";
import { bytesGroup, frameBytesKey, frameKey, packetPart } from "./frameKey";
import { mayAsk, whenAskable } from "./frameStore";

/** 后台取回整段时同时在排的帧数。真正占几条连接由 transfer/frameStore.ts 的连接配额定（后台合计至多 3 条，
 * 正在画的先走），这里只是不一次排太多。 */
const AT_ONCE = 3;

interface Filling {
  source: FrameSource;
  order: number[]; // 全部帧，从播放头沿播放方向向外排序
  from: number; // order 按哪一帧、哪个方向排的：播放时每次只走一两帧，沿用原顺序（已取的按 done 跳过），不每帧重排
  dir: number;
  cursor: number; // order 里还没看过的第一个位置：挑下一帧时从这里往后，不每次从头扫
  done: Set<number>; // 已取回（或原本已持有）的帧
  later: Set<number>; // 取失败的帧：退避到期后再补（失败记忆在 transfer/frameStore.ts，按字节键）
  at: number; // 挂载序号：最近挂载的源优先取回
  holders: number; // 正在看它的舞台数
}

/** 本模块唯一的可变状态：每个源一条，有人看就在，没人看就删（有上限：同时开着的舞台数）。 */
class Fills {
  readonly per = new Map<string, Filling>();
  readonly now = new Set<string>(); // 当前正在取回的键
  clock = 0;
}

const own = new Fills();

/** How far out from the play head a source's whole range is fetched: as many frames as its share of the byte layer holds
 * (the same split as the decoded window, frameWindow.ts aheadFor, by the size its frames have so far). A short shot is
 * all of it; a long one (3 channels × 3000 frames of 45 KB do not fit in 128 MB) is the part around the play head,
 * which then stays (pinned below) instead of the channels pushing each other's whole ranges out and fetching them again. */
function reachOf(entry: Filling): number {
  const group = bytesGroup(entry.source.id);
  const n = cache.keysIn(group).length;
  // no frame held yet, its size unknown: no cap until the first ones arrive and say how big they are
  return n ? Math.min(entry.order.length, aheadFor(cache.bytesIn(group) / n, BYTES_BUDGET, own.per.size)) : entry.order.length;
}

// the bytes within reach of a source being looked at are being used: the byte layer keeps them (platform/cache.ts pin)
cache.pin(() => new Set([...own.per.values()].flatMap((e) => e.order.slice(0, reachOf(e)).map((f) => frameBytesKey(e.source.id, f)))));

/** 舞台开始看这个源：取回它整段（`frame` / `dir`：从哪一帧、往哪个方向向外排）。返回「不看了」。
 *
 * 按源 id 计数而不是随帧重建：舞台每换一帧都会先撤后挂（React 的 effect），若随之重建，「已取回」表每帧清空、
 * 每帧都从头扫一遍整段。撤掉后同一轮同步代码里又挂回来的沿用原来那条（只换起点）；真没人看了才删。
 * 同一个 id 换了新的源对象（查找原件以「没有」结束时源会重建）就不沿用：查找期间 `fill` 返回 undefined 的帧已被记为取过，
 * 沿用会让整段取回再也不启动。 */
export function fillWhole(source: FrameSource, frame: number, dir: number): () => void {
  if (!source.fill) return () => undefined; // 本机文件：本身不经过网络
  const had = own.per.get(source.id);
  const entry: Filling = had && had.source === source ? had : { source, order: [], from: NaN, dir: 0, cursor: 0, done: new Set(), later: new Set(), at: 0, holders: 0 };
  // 同方向、只挪了一两帧（播放、单步）：沿用原顺序和游标；跳到别处或换了方向才重排、从头扫
  if (!(entry.dir === dir && Math.abs(frame - entry.from) <= 2)) {
    entry.order = ordered(source.frames, frame, dir);
    entry.cursor = 0;
    cache.repin();
  }
  entry.from = frame;
  entry.dir = dir;
  entry.at = ++own.clock;
  entry.holders++;
  if (!had) cache.repin();
  own.per.set(source.id, entry);
  pump();
  return () => {
    entry.holders--;
    queueMicrotask(() => {
      if (entry.holders <= 0 && own.per.get(source.id) === entry) (own.per.delete(source.id), cache.repin());
    });
  };
}

function pump(): void {
  while (own.now.size < AT_ONCE) {
    const next = nextOne();
    if (!next) return;
    const { entry, frame } = next;
    entry.done.add(frame);
    const start = entry.source.fill?.(frame);
    if (!start) continue;
    const key = frameKey(entry.source.id, frame);
    own.now.add(key);
    void start()
      .catch(() => {
        // 服务器没有该帧或暂时拿不到：后台取回不报错（真要画时经 `load` 报）。从「已取回」里拿掉，退避到期后再补
        entry.done.delete(frame);
        entry.later.add(frame);
        whenAskable(bytesAt(entry, frame), pump);
      })
      .finally(() => {
        own.now.delete(key);
        pump();
      });
  }
}

/** 一帧的压缩字节在 frameStore 里的键（失败记忆也按它记）：只与服务器那份字节有关，与是不是从本机读无关。 */
const bytesAt = (entry: Filling, frame: number): string => frameKey(packetPart(entry.source.id), frame);

/** 下一个待取回的帧：最近挂载的源优先，源内按从播放头向外的顺序；已持有的（字节或位图）跳过，切回时不重复取回。 */
function nextOne(): { entry: Filling; frame: number } | null {
  for (const entry of [...own.per.values()].sort((a, b) => b.at - a.at)) {
    for (const frame of entry.later) {
      if (!mayAsk(bytesAt(entry, frame))) continue;
      entry.later.delete(frame);
      return { entry, frame };
    }
    const reach = reachOf(entry);
    for (; entry.cursor < reach; entry.cursor++) {
      const frame = entry.order[entry.cursor];
      if (entry.done.has(frame)) continue;
      const key = frameKey(entry.source.id, frame);
      if (cache.has(key) || cache.has(frameBytesKey(entry.source.id, frame)) || own.now.has(key)) {
        entry.done.add(frame);
        continue;
      }
      return { entry, frame };
    }
  }
  return null;
}
