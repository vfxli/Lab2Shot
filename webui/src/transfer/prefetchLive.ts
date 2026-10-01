/** 实际运行的预取器：从服务器获取数据包帧的二维数据源（transfer/prefetch.ts 为纯逻辑；
 * 本模块是唯一将其接入 api 与浏览器的位置，当前显示的节点与帧、页面转入后台等事件均由此传给预取器）。
 *
 * 预取结果为压缩字节，不解码：每帧数十 KB，整段数 MB；解码后的位图占用更大，由
 * `platform/cache.ts` 的另一项预算管理。字节存放在视图读取的键下（transfer/frameKey.ts 的 frameKey + bytesKey），
 * 因此视图绘制时无需发出请求，直接就地解码。
 *
 * 分工：当前查看的源由舞台自行整段预取（`transfer/fill.ts fillWhole`：选中即整段）；
 * 本模块预取其他数据包，即显示链上游及同一任务中的其余结果，使使用者切换过去时数据已在本地。
 * 两侧使用同一组键，且均先检查本地是否已有，因此不会重复预取。 */

import { useLook } from "../state/look";
import { useViewer } from "../state/viewer";
import { Paused, Prefetcher, prefetchStore, type PrefetchSource, type Want } from "./prefetch";
import { manifestNow, manifestOf } from "./frames";
import { cache } from "../platform/cache";
import { Gone, Later, NeedLogin, frameBytes, whenAskable } from "./frameStore";
import { channelId, channelUrl, describedSlot, frameKey, packetKey, pictureUrl, tierOf as tierIn } from "./frameKey";
import { channelsFor } from "./route";
import { channelsOf, lookIndex } from "../model/view2d";

/** 该数据包在后台需要预取的内容：与显示界面使用同一判定规则（`transfer/route.ts channelsFor`），
 * 否则舞台只需遮罩通道时，此处仍会逐帧下载整张显示图，造成流量浪费。
 *
 * 预取以默认视图为准，即使用者切换到该节点时首先看到的内容（单通道数据包取该通道，彩色画面取显示图）。 */
const routeOf = (fp: string): string[] => {
  const m = manifestNow(fp);
  return channelsFor(m, lookIndex(null, channelsOf(String(m?.type ?? ""))));
};

/** 该数据包在视图中的显示档位（与舞台同一份：包说明里的 proxy）。 */
const tierOf = (fp: string) => tierIn(manifestNow(fp));

/** 图片路径一帧（与 transfer/sources.ts serverFrames 的字节同一个源：不透过相机）。 */
const pictureAt = (fp: string, frame: number) => ({ id: packetKey({ fp, tier: tierOf(fp) }), frame });

const source: PrefetchSource = {
  frames: async (want) => {
    if (want.kind !== "frames2d") return [];
    // 包说明这次没取到（退避、等登录、断线）不是「没有帧」：暂停这份 want，等它能再要时接着；只有服务器说没有（Gone）才是空的
    const m = await manifestOf(want.fp).catch((e: unknown): never => {
      if (e instanceof Gone) throw e;
      throw new Paused((go) => void whenAskable(e instanceof Later ? e.key : describedSlot("manifest", want.fp).key, go));
    });
    return Array.isArray(m.meta.frames) ? (m.meta.frames as number[]) : [];
  },
  load: async (want, frame, signal) => {
    // 取字节与舞台、整段取回同一条路（transfer/frameStore.ts frameBytes：同一帧只发一次、同一套存放与失败记忆），
    // 地址同一个构造（pictureUrl、channelUrl，带版本）；这里只决定预取哪些、什么顺序
    const names = routeOf(want.fp);
    const tier = tierOf(want.fp);
    if (!names.length) {
      // 类型取包说明的（视频走视频的路由）：与舞台同一个地址，浏览器缓存与服务器的代理文件都是同一份
      const type = String(manifestNow(want.fp)?.type ?? "");
      const at = pictureAt(want.fp, frame);
      await frameBytes(at, pictureUrl({ fp: want.fp, type }, frame, tier), "picture", signal, "later").catch(pausedAt(at));
      return;
    }
    // 通道路径：按当前视图所需的通道逐条取，每条存在视图读取的键下（transfer/sources.ts channelId；frameBytes 取到即存）
    for (const name of names) {
      const at = { id: channelId(want.fp, name, tier), frame };
      if (!prefetchStore.isCached(frameKey(at.id, frame)))
        await frameBytes(at, channelUrl(want.fp, frame, name, tier), "channel", signal, "later").catch(pausedAt(at));
    }
  },
  keyOf: (want, frame) => {
    const names = routeOf(want.fp);
    const tier = tierOf(want.fp);
    return names.length ? frameKey(channelId(want.fp, names[0], tier), frame) : frameKey(pictureAt(want.fp, frame).id, frame);
  },
};

/** 取数层说「现在别要」（等登录、退避期）：不算取过，暂停这份 want，这一帧能再要时接着（frameStore whenAskable）。 */
const pausedAt = (at: { id: string; frame: number }) => (e: unknown): never => {
  // 在退避的是取字节的那个键（Later.key），不一定是这里的帧键：等它
  if (e instanceof Later) throw new Paused((go) => void whenAskable(e.key, go));
  if (e instanceof NeedLogin) throw new Paused((go) => void whenAskable(frameKey(at.id, at.frame), go));
  throw e;
};

const prefetcher = new Prefetcher(prefetchStore, source);

let watching = false;

/** 预取器跟踪的状态，自第一次请求数据包起开始跟踪（加载时不跟踪）：当前显示的节点、使用者移动的播放头
 * （仅这些字段，而非仓库的任意变化），以及页面转入后台。 */
function watch(): void {
  if (watching) return;
  watching = true;
  // 使用者切换节点或拖动时间线时，当前显示的节点优先，较低优先级的预取被中止。
  // 播放过程中播放头自动前进不视为焦点变化，不予传递
  const refocus = (): void => {
    const { displayId } = useLook.getState();
    const { frame, playing } = useViewer.getState();
    if (displayId !== null && !playing) prefetcher.focus(displayId, frame);
  };
  useLook.subscribe((s, p) => s.displayId !== p.displayId && refocus());
  useViewer.subscribe((s, p) => (s.frame !== p.frame || s.playing !== p.playing) && refocus());
}

// 页面转入后台时暂停预取，并主动释放解码层（保留压缩字节，返回前台后数毫秒即可重新解码）。
// `ImageBitmap` 与 GPU 纹理不在 JS 堆中，不显式释放将持续占用。装在模块加载时：不论有没有预取过，切后台都释放
if (typeof document !== "undefined") {
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      prefetcher.pause();
      cache.releasePixels();
    } else {
      prefetcher.resume();
    }
  });
}

/** 登记一个需要在显示前预取的数据包（graph/streamDone.ts 在 `node_done` 时调用）。 */
export const wantPrefetch = (w: Want): void => {
  watch();
  prefetcher.want(w);
};
