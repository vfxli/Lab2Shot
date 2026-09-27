/** 页面唯一的缓存：二维舞台、三维舞台、播放器、预取所持有的每一张图、每一条通道、
 * 每一份包说明都存放于此，按字节计量。其他位置不得另建缓存
 * （测试 `webui/tests/frames.test.ts` 统计实现数量，须为 1）。
 *
 * 分为两层，各有预算（见下方 `PIXELS_BUDGET` / `BYTES_BUDGET`）：
 * 压缩字节层可容纳整段数据（成本低），已解码位图层只保留播放头附近的少量帧。
 *
 * 同一层内，优先释放重新获取成本最低的条目，`Keep` 即此顺序：
 *
 *   fetched  预先取回、当前未被绘制的图（解码层）
 *   near     当前绘制帧附近的帧（解码层）
 *   viewing  当前绘制的帧（解码层）
 *   small    一帧的压缩字节、一份包说明（字节层，与上面三档不在同一层，互不挤占）
 *
 * 同一档内最久未使用的条目先被释放。视图当前正在绘制的帧永不释放（`pin`）。
 *
 * 持有 `ImageBitmap` 的条目记录其关闭方式（`free`），图像内存随条目一同释放；
 * 不关闭则一直占用，且该内存不在 JS 堆中，浏览器的垃圾回收无法感知。
 * 只存储实际获取到的数据：获取失败的帧不存储（下次需要时重新获取）。 */

/** 条目的保留价值，由低到高：
 *   `fetched` 按地址取回的图（遮罩预览图、灯光贴图）
 *   `near`    窗口内但非当前绘制的帧
 *   `small`   体积小且整段保留的数据（包说明、一帧的压缩字节、一条通道的压缩数据），体积很小，丢弃最不划算
 *   `viewing` 当前绘制的帧
 *
 * 缓存层只存储数据（键为包指纹 + 帧 + 通道），不存储任何显示方式的结果：显示方式每帧在 GPU 上实时计算
 * （view/look.ts）。 */
export type Keep = "fetched" | "small" | "near" | "viewing";

const ORDER: Record<Keep, number> = { fetched: 0, near: 1, small: 2, viewing: 3 };

/** 两层各有预算，避免耗尽浏览器内存。
 *
 * | 存储内容 | 单帧大小（1080p 源，代理档 512） | 100 帧 |
 * |---|---|---|
 * | 解码后的位图（`ImageBitmap` / 一条通道的数值） | 0.6 MB（与代理大小无关，解码后为宽×高×4） | 60 MB |
 * | 压缩字节（代理本身） | 数十 KB | 数 MB |
 *
 * 因此整段取回的是压缩字节，存放时不解码；只有播放头附近的帧才解码为位图，
 * 两层各自管理预算，一层满时不会挤占另一层。若共用一个预算，位图增多时会挤掉整段字节，
 * 再次查看时又需重新走网络。
 *
 * 预算不能凭经验设定一个固定值：
 * `ImageBitmap` 和 GPU 纹理不在 JS 堆中，浏览器的垃圾回收无法得知其大小，
 * 因此只能按本模块自行记录的字节数淘汰（本缓存本身即按字节计量），上限按本机内存
 * 计算（`navigator.deviceMemory`，Chrome 最大报告 8），无法获取时使用保守默认值。 */
const deviceGB = (): number => {
  const said = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  return typeof said === "number" && said > 0 ? said : 2;  // 无法获取（Safari / Firefox）：按保守的 2 GB 计算
};

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** 已解码位图的预算上限：本机内存的八分之一，范围 128 MB 至 1 GB。
 * （8 GB 内存 → 1 GB；4 GB → 512 MB；无法获取 → 256 MB。） */
export const PIXELS_BUDGET = clamp((deviceGB() / 8) * (1 << 30), 128 << 20, 1 << 30);

/** 压缩字节的预算上限：本机内存的十六分之一，范围 64 MB 至 512 MB。
 * 一帧代理为数十 KB，因此 512 MB 可容纳多条通道的整段数据；其成本低，故预算较大。 */
export const BYTES_BUDGET = clamp((deviceGB() / 16) * (1 << 30), 64 << 20, 512 << 20);

/** 条目所属的层：`small`（包说明、一帧的压缩字节）属于字节层，其余均属于解码层。 */
const layerOf = (keep: Keep): "bytes" | "pixels" => (keep === "small" ? "bytes" : "pixels");

interface Held {
  key: string;
  value: unknown;
  bytes: number;
  keep: Keep;
  used: number; // a counter incremented on every read and write: the least recently used entry within a class is evicted first
  free?: (value: never) => void;
}

class ByteCache {
  private held = new Map<string, Held>();
  private bytes = { bytes: 0, pixels: 0 };  // 两层各自的占用量
  private clock = 0;
  private version = 0;
  private listeners = new Set<() => void>();
  private pinned: () => ReadonlySet<string> = () => new Set();

  /** 登记层：小型登记条目（原件登记 `orig:`、浏览器计算结果 `computed:`、本机代理就绪 `proxy:`）不计入预算、不被淘汰。
   * 它们不能与帧字节同层：否则会被 LRU 淘汰，淘汰后视图将静默退回服务器代理，浏览器计算的结果也不再登记。
   * 此类条目总共数百个小对象，内存占用可忽略。 */
  private registry = new Map<string, unknown>();

  register<V>(key: string, value: V): void {
    this.registry.set(key, value);
    this.changed();
  }

  registered<V>(key: string): V | undefined {
    return this.registry.get(key) as V | undefined;
  }

  unregister(key: string): void {
    if (this.registry.delete(key)) this.changed();
  }

  /** Entries that must not be released while being drawn (transfer/frames.ts: the frames of every open window). */
  pin(what: () => ReadonlySet<string>): void {
    this.pinned = what;
  }

  /** The value under this key, marked as used now; undefined when absent. */
  get<V>(key: string): V | undefined {
    const e = this.held.get(key);
    if (!e) return undefined;
    e.used = ++this.clock;
    return e.value as V;
  }

  has(key: string): boolean {
    return this.held.has(key);
  }

  /** Stores a value of the given byte size under this class. An existing key is replaced (its previous value freed). */
  keep<V>(key: string, value: V, size: number, keep: Keep, o?: { free?: (value: V) => void }): void {
    this.forget(key);
    this.held.set(key, { key, value, bytes: size, keep, used: ++this.clock, free: o?.free as Held["free"] });
    this.bytes[layerOf(keep)] += size;
    this.evict();
    this.changed();
  }

  /** Moves an entry to another class (a prefetched frame that the view now draws). */
  reclass(key: string, keep: Keep): void {
    const e = this.held.get(key);
    if (!e || e.keep === keep) return;
    this.bytes[layerOf(e.keep)] -= e.bytes;
    this.bytes[layerOf(keep)] += e.bytes;
    e.keep = keep;
    e.used = ++this.clock;
  }

  /** Releases one entry (its value freed). */
  forget(key: string): void {
    const e = this.held.get(key);
    if (!e) return;
    this.held.delete(key);
    this.bytes[layerOf(e.keep)] -= e.bytes;
    e.free?.(e.value as never);
    this.changed();
  }

  /** Releases every entry under this prefix (a removed source, a closed graph). */
  forgetAll(prefix: string): void {
    for (const key of [...this.held.keys()]) if (key.startsWith(prefix)) this.forget(key);
  }

  /** The keys held under this prefix. */
  keysUnder(prefix: string): string[] {
    return [...this.held.keys()].filter((k) => k.startsWith(prefix));
  }

  /** Notifies listeners whenever the held entries change (a frame arrived, an entry was released): the timeline's
   * 已载入视图 row follows the prefetcher this way without polling on every render. */
  onChange(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }

  /** Incremented on every change: a React view reads it as its snapshot. */
  changes(): number {
    return this.version;
  }

  /** 页面切到后台或节点图关闭时，主动释放解码层（保留压缩字节，返回前台后数毫秒即可重新解码）。
   * `ImageBitmap` 和 GPU 纹理不在 JS 堆中，不显式释放会一直占用。
   * 当前绘制的帧（`pin`）不释放，页面回到前台时画面不会出现空白。 */
  releasePixels(): void {
    const keepAll = this.pinned();
    for (const e of [...this.held.values()]) {
      if (layerOf(e.keep) === "pixels" && !keepAll.has(e.key)) this.forget(e.key);
    }
  }

  /** The entries currently held, for the tests and the page's diagnostics. */
  stats(): { bytes: number; pixels: number; budget: number; pixelsBudget: number; count: number; byKeep: Record<Keep, number> } {
    const byKeep = { fetched: 0, small: 0, near: 0, viewing: 0 } as Record<Keep, number>;
    for (const e of this.held.values()) byKeep[e.keep] += e.bytes;
    return { bytes: this.bytes.bytes, pixels: this.bytes.pixels, budget: BYTES_BUDGET, pixelsBudget: PIXELS_BUDGET, count: this.held.size, byKeep };
  }

  /** Releases everything (used when a test restarts). */
  forgetEverything(): void {
    for (const key of [...this.held.keys()]) this.forget(key);
  }

  private changed(): void {
    this.version++;
    for (const l of this.listeners) l();
  }

  /** 两层分别淘汰：解码层满时只淘汰位图，字节层满时只淘汰字节。
   * 同一层内按重新获取的成本排序（`ORDER`），同一档内最久未使用的先淘汰；当前绘制的帧（`pin`）永不淘汰。 */
  private evict(): void {
    for (const layer of ["pixels", "bytes"] as const) {
      const budget = layer === "pixels" ? PIXELS_BUDGET : BYTES_BUDGET;
      if (this.bytes[layer] <= budget) continue;
      const keepAll = this.pinned();
      const order = [...this.held.values()]
        .filter((e) => layerOf(e.keep) === layer)
        .sort((a, b) => ORDER[a.keep] - ORDER[b.keep] || a.used - b.used);
      for (const e of order) {
        if (this.bytes[layer] <= budget) break;
        if (keepAll.has(e.key)) continue; // currently on screen: never evicted from under the view
        this.forget(e.key);
      }
    }
  }
}

/** The page's single cache: one instance, so its contents are instance state rather than module state. */
export const cache = new ByteCache();
