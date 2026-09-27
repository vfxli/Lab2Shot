/** 实际运行的预取器：从服务器获取数据包帧的二维数据源（transfer/prefetch.ts 为纯逻辑；
 * 本模块是唯一将其接入 api 与浏览器的位置，当前显示的节点与帧、页面转入后台等事件均由此传给预取器）。
 *
 * 预取结果为压缩字节，不解码：每帧数十 KB，整段数 MB；解码后的位图占用更大，由
 * `transfer/cache.ts` 的另一项预算管理。字节存放在视图读取的键下（`bytes:${源 id}:${帧}`），
 * 因此视图绘制时无需发出请求，直接就地解码。
 *
 * 分工：当前查看的源由舞台自行整段预取（`transfer/frames.ts fillWhole`：选中即整段）；
 * 本模块预取其他数据包，即显示链上游及同一任务中的其余结果，使使用者切换过去时数据已在本地。
 * 两侧使用同一组键，且均先检查本地是否已有，因此不会重复预取。 */

import { api } from "../api";
import { blob, bytes as fetchBytes } from "../platform/http";
import { useLook } from "../state/look";
import { useViewer } from "../state/viewer";
import { Prefetcher, prefetchStore, type PrefetchSource, type Want } from "./prefetch";
import { channelId, manifestNow, manifestOf, tierTag } from "./frames";
import { cache } from "./cache";
import { bytesKey, versionOf } from "./sources";
import { packetKey } from "./ident";
import { genOf } from "./gens";
import { squeeze } from "./plane";
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

/** 该数据包在视图中使用的代理档位，以键片段形式返回（与 `transfer/sources.ts` 的规则一致）。 */
const tierOf = (fp: string): string => tierTag(manifestNow(fp));

const source: PrefetchSource<Blob> = {
  frames: async (want) => {
    if (want.kind !== "frames2d") return [];
    const m = await manifestOf(want.fp);
    return Array.isArray(m.meta.frames) ? (m.meta.frames as number[]) : [];
  },
  load: async (want, frame, signal) => {
    const names = routeOf(want.fp);
    if (!names.length) {
      // 图片路径：服务器返回的即为压缩字节，原样保存，不解码
      const bytes = await blob(api.packetFrameUrl(want.fp, frame, undefined, versionOf(want.fp, manifestNow(want.fp))), { signal, headers: { Accept: "image/webp,image/png" } });
      return { value: bytes, bytes: bytes.size };
    }
    // 通道路径：按当前视图所需的通道逐条获取，每条存放在视图读取的键下（transfer/sources.ts channelId）。
    // 传输层的 gzip 由浏览器在原生代码中解压，页面只能拿到解压后的数据，因此需自行无损压缩后再存储。
    const tier = tierOf(want.fp);
    let first: Blob | null = null;
    let total = 0;
    for (const name of names) {
      const key = `${channelId(want.fp, name, tier)}:${frame}`;
      if (prefetchStore.isCached(key)) continue;
      const buf = await fetchBytes(api.channelUrl(want.fp, frame, name, versionOf(want.fp, manifestNow(want.fp))), { signal });
      const packed = await squeeze(buf);
      if (!packed) continue;  // 浏览器不支持 CompressionStream 时跳过该层，功能不受影响
      total += packed.size;
      if (name === names[0]) first = packed;
      else cache.keep(bytesKey(key), packed, packed.size, "small");  // 第一条由返回值交回，此处不重复存储
    }
    return { value: first, bytes: total };
  },
  keyOf: (want, frame) => {
    const names = routeOf(want.fp);
    const tier = tierOf(want.fp);
    return names.length ? `${channelId(want.fp, names[0], tier)}:${frame}` : `${packetKey({ fp: want.fp, gen: genOf(want.fp), tier })}:${frame}`;
  },
};

export const prefetcher = new Prefetcher<Blob>(prefetchStore, source);

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
  // 页面转入后台时暂停预取，并主动释放解码层（保留压缩字节，返回前台后数毫秒即可重新解码）。
  // `ImageBitmap` 与 GPU 纹理不在 JS 堆中，不显式释放将持续占用。
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
}

/** 登记一个需要在显示前预取的数据包（graph/streamDone.ts 在 `node_done` 时调用）。 */
export const wantPrefetch = (w: Want): void => {
  watch();
  prefetcher.want(w);
};
