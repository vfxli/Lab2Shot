/** 页面唯一的缓存：二维舞台、三维舞台、播放器、预取持有的每一张图、每一条通道、每一份包说明、每一份三维场景都存在这里，
 * 按字节计量。别处不另建缓存。
 *
 * 页面上只有这一套内存预算：三维场景的逐帧样本也从解码层分（`SCENE_SHARE`），不另算一份。
 * 两层，各有预算（`PIXELS_BUDGET` / `BYTES_BUDGET`），一层满了不挤另一层：
 *   解码层  `fetched`（按地址取回、没在画的：三维块、场景）→ `near`（窗口里、不是这一帧）→ `viewing`（正在画的这一帧），
 *           按此顺序、同档内最久没用的先淘汰；
 *   字节层  `small`（一帧的压缩字节、一条通道的压缩数据、一份包说明）：体积小，整段保留，最久没用的先淘汰。
 * 正在画的（各舞台 `pin` 登记的）永不淘汰，切后台释放解码层时也留着；它们也计入预算，但没在画的至少还能留
 * 预算的四分之一（`FLOOR`）——显示中的一份大场景占满预算时，其余条目不会每存一条就被全部清空。
 *
 * 每一条存进来时都带明确的「组」（`Slot`，由 transfer/frameKey.ts 唯一生成：一帧的组是它的源，一份包说明的组是
 * 它的种类，三维场景是 SCENES……）：按组列键、按组订阅、按组计量都不扫全表。这里不从键的字符串猜组——键里的
 * 代次是带冒号的时刻，任何按分隔符的猜法都会错。
 * 变化通知合并：同一段同步代码里的所有增删只在它结束后通知一次，订阅者只收到它关心的组。
 *
 * 持有 `ImageBitmap` 的条目带释放函数（`free`）：那块内存不在 JS 堆里，垃圾回收看不见，不释放就一直占着。
 * 只存真正拿到的数据：取失败的不存。 */

type Keep = "fetched" | "near" | "viewing" | "small";

const deviceGB = (): number => {
  const said = (typeof navigator !== "undefined" ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory : undefined);
  return typeof said === "number" && said > 0 ? said : 2;  // 拿不到（Safari / Firefox）：按保守的 2 GB 算
};

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** 解码层的预算：本机内存的八分之一，128 MB 至 1 GB（位图与 GPU 纹理不在 JS 堆里，只能按这里记的字节数管）。 */
export const PIXELS_BUDGET = clamp((deviceGB() / 8) * (1 << 30), 128 << 20, 1 << 30);

/** 全部三维场景合计可常驻的字节（基础数据 + 已解码的逐帧样本，view/scene.ts）：解码层预算的一半，其余留给二维帧。
 * 几份场景同时显示时共用这一份（每份按「这一份 − 其它场景已占」算自己还能解多少），不是各拿一份。 */
export const SCENE_SHARE = PIXELS_BUDGET / 2;

/** 登记层（原件查找结果、本机代理就绪）最多留多少条。 */
const REGISTRY_MAX = 20_000;

/** 超预算时，没在画的条目至少还能留的份额（占该层预算的比例）。 */
const FLOOR = 1 / 4;

/** 字节层的预算：本机内存的十六分之一，64 MB 至 512 MB（一帧代理数十 KB，够放多条通道的整段）。 */
export const BYTES_BUDGET = clamp((deviceGB() / 16) * (1 << 30), 64 << 20, 512 << 20);

const LAYERS = { pixels: ["fetched", "near", "viewing"], bytes: ["small"] } as const;
const layerOf = (keep: Keep): "bytes" | "pixels" => (keep === "small" ? "bytes" : "pixels");
/** 一条缓存的地址：键（取、查、放都按它）和它属于的组（订阅、列键、计量按它）。只由 transfer/frameKey.ts 生成。 */
export interface Slot {
  key: string;
  group: string;
}

interface Held {
  key: string;
  group: string;
  value: unknown;
  bytes: number;
  keep: Keep;
  free?: (value: never) => void;
}

class PageCache {
  private held = new Map<string, Held>();
  // 每档一张按最近使用排序的表（最久没用的在前）：取用时挪到末尾，淘汰从头往后，不必每次全排序
  private lru: Record<Keep, Map<string, Held>> = { fetched: new Map(), near: new Map(), viewing: new Map(), small: new Map() };
  private groups = new Map<string, Set<string>>();
  private bytes = { bytes: 0, pixels: 0 };
  private pins = new Set<() => ReadonlySet<string>>();
  // 登记层：小型登记（原件登记、本机代理就绪）不计预算、不淘汰——淘汰了视图会静默退回服务器代理
  private registry = new Map<string, { group: string; value: unknown }>();

  // ---- 通知
  private version = 0;
  private groupVersion = new Map<string, number>();
  private listeners = new Set<() => void>();
  private watchers = new Map<string, Set<() => void>>();
  private dirty = new Set<string>();
  private scheduled = false;

  register<V>(at: Slot, value: V): void {
    this.registry.delete(at.key); // 重登的挪到最后：上限按最久没登记的先丢
    this.registry.set(at.key, { group: at.group, value });
    this.touch(at.group);
    // 有上限：每条很小，但按包 / 文件只增不减。丢掉的是最久没碰过的（早就不在当前文档里的包）：再用到时重新查一次
    while (this.registry.size > REGISTRY_MAX) {
      const [oldest, was] = this.registry.entries().next().value as [string, { group: string }];
      this.registry.delete(oldest);
      this.touch(was.group);
    }
  }

  registered<V>(key: string): V | undefined {
    return this.registry.get(key)?.value as V | undefined;
  }

  unregister(key: string): void {
    const was = this.registry.get(key);
    if (was && this.registry.delete(key)) this.touch(was.group);
  }

  /** 正在画的不许释放：每个画的一方登记自己正在画的键（transfer/frames.ts 各窗口的帧；view/sceneData.ts 显示中的场景）。
   * 返回撤销登记的函数。 */
  pin(what: () => ReadonlySet<string>): () => void {
    this.pins.add(what);
    this.repin();
    return () => (this.pins.delete(what), this.repin());
  }

  /** 登记方的「正在画」变了（窗口挪了、显示的场景换了）：下次淘汰重新问。 */
  repin(): void {
    this.pinnedNow = null;
  }

  // 正在画的键：登记方没说变（repin）就沿用，最多到下一拍（播放时每存一帧都要淘汰，每次都把所有窗口的键重建一遍是 O(n)）
  private pinnedNow: ReadonlySet<string> | null = null;

  private pinned(): ReadonlySet<string> {
    if (this.pinnedNow) return this.pinnedNow;
    const all = new Set<string>();
    for (const what of this.pins) for (const k of what()) all.add(k);
    this.pinnedNow = all;
    queueMicrotask(() => (this.pinnedNow = null));
    return all;
  }

  /** 这个键下的值（记为刚用过）；没有为 undefined。 */
  get<V>(key: string): V | undefined {
    const e = this.held.get(key);
    if (!e) return undefined;
    const tier = this.lru[e.keep];
    tier.delete(key);
    tier.set(key, e);
    return e.value as V;
  }

  has(key: string): boolean {
    return this.held.has(key);
  }

  /** 存一个值，记它的字节数和保留档；同键的旧值先释放。 */
  keep<V>(at: Slot, value: V, size: number, keep: Keep, o?: { free?: (value: V) => void }): void {
    const { key, group } = at;
    this.drop(key);
    const e: Held = { key, group, value, bytes: size, keep, free: o?.free as Held["free"] };
    this.held.set(key, e);
    this.lru[keep].set(key, e);
    let set = this.groups.get(group);
    if (!set) this.groups.set(group, (set = new Set()));
    set.add(key);
    this.bytes[layerOf(keep)] += size;
    this.touch(group);
    this.evict();
  }

  /** 换保留档（预取来的帧现在被画了；窗口挪走的帧降一档）。 */
  reclass(key: string, keep: Keep): void {
    const e = this.held.get(key);
    if (!e || e.keep === keep) return;
    this.lru[e.keep].delete(key);
    this.bytes[layerOf(e.keep)] -= e.bytes;
    e.keep = keep;
    this.lru[keep].set(key, e);
    this.bytes[layerOf(keep)] += e.bytes;
  }

  /** 释放一条。 */
  forget(key: string): void {
    this.drop(key);
  }

  /** 释放一个组的全部（一个源不要了）。 */
  forgetGroup(group: string): void {
    for (const key of [...(this.groups.get(group) ?? [])]) this.drop(key);
  }

  /** 释放某个前缀下的全部（边算边看的一整段临时数据）：少见，按前缀扫一遍。 */
  forgetAll(prefix: string): void {
    for (const key of [...this.held.keys()]) if (key.startsWith(prefix)) this.drop(key);
  }

  /** 一个组里现有的键。 */
  keysIn(group: string): string[] {
    return [...(this.groups.get(group) ?? [])];
  }

  /** 一个组里的条目合计记了多少字节（显示中的三维场景合计多大：view/scene.ts 按它分 SCENE_SHARE）。 */
  bytesIn(group: string): number {
    let n = 0;
    for (const key of this.groups.get(group) ?? []) n += this.held.get(key)?.bytes ?? 0;
    return n;
  }

  /** 一条记了多少字节（没有为 0）。 */
  sizeOf(key: string): number {
    return this.held.get(key)?.bytes ?? 0;
  }

  /** 有变化就叫（合并：一段同步代码里的所有变化只叫一次）。全局订阅只给少数需要看全部的（缓存记账）。 */
  onChange(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }

  /** 这几个组有变化才叫（合并同上）：时间线、舞台按自己的源订阅，别的源的帧到了不叫它。 */
  watch(groups: readonly string[], listener: () => void): () => void {
    for (const g of groups) {
      let set = this.watchers.get(g);
      if (!set) this.watchers.set(g, (set = new Set()));
      set.add(listener);
    }
    return () => {
      for (const g of groups) {
        const set = this.watchers.get(g);
        set?.delete(listener);
        if (set && !set.size) this.watchers.delete(g);
      }
    };
  }

  /** 全局的变化计数（全局订阅者的快照）。 */
  changes(): number {
    return this.version;
  }

  /** 这几个组的变化计数之和（按组订阅者的快照：别的组变了它不变）。 */
  changesIn(groups: readonly string[]): number {
    let n = 0;
    for (const g of groups) n += this.groupVersion.get(g) ?? 0;
    return n;
  }

  /** 切到后台时释放解码层（压缩字节留着，回来数毫秒就能重新解码）；正在画的留着，回来时画面不空。 */
  releasePixels(): void {
    const keepAll = this.pinned();
    // 边走边删：Map 的迭代器跳过已删的，不必先复制整张表
    for (const tier of LAYERS.pixels) for (const key of this.lru[tier].keys()) if (!keepAll.has(key)) this.drop(key);
  }

  private drop(key: string): void {
    const e = this.held.get(key);
    if (!e) return;
    this.held.delete(key);
    this.lru[e.keep].delete(key);
    const g = e.group;
    const set = this.groups.get(g);
    set?.delete(key);
    if (set && !set.size) this.groups.delete(g);
    this.bytes[layerOf(e.keep)] -= e.bytes;
    e.free?.(e.value as never);
    this.touch(g);
  }

  private touch(g: string): void {
    this.groupVersion.set(g, (this.groupVersion.get(g) ?? 0) + 1);
    this.version++;
    this.dirty.add(g);
    if (this.scheduled) return;
    this.scheduled = true;
    queueMicrotask(() => this.flush());
  }

  private flush(): void {
    this.scheduled = false;
    const groups = [...this.dirty];
    this.dirty.clear();
    const called = new Set<() => void>();
    for (const g of groups) for (const f of this.watchers.get(g) ?? []) if (!called.has(f)) (called.add(f), f());
    for (const f of this.listeners) f();
  }

  /** 两层分别淘汰：解码层满只淘汰位图，字节层满只淘汰字节；按档位、同档最久没用的先走；正在画的永不淘汰。
   * 正在画的计入预算：没在画的可留「预算 − 正在画的」，但至少 `FLOOR` 份。 */
  private evict(): void {
    for (const layer of ["pixels", "bytes"] as const) {
      const budget = layer === "pixels" ? PIXELS_BUDGET : BYTES_BUDGET;
      if (this.bytes[layer] <= budget) continue;
      const keepAll = this.pinned();
      let pinnedBytes = 0;
      for (const key of keepAll) {
        const e = this.held.get(key);
        if (e && layerOf(e.keep) === layer) pinnedBytes += e.bytes;
      }
      const room = Math.max(budget - pinnedBytes, budget * FLOOR); // 没在画的可留多少
      done: for (const tier of LAYERS[layer]) {
        for (const key of this.lru[tier].keys()) { // 边走边删（同上）
          if (this.bytes[layer] - pinnedBytes <= room) break done;
          if (!keepAll.has(key)) this.drop(key);
        }
      }
    }
  }
}

/** 页面唯一的缓存：一个实例。 */
export const cache = new PageCache();
