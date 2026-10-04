import { baseArray, baseBytes, cornerUvs, type GridRef, type ModelRef, type ViewDescription } from "../model/viewFormat";
import { boundsInWorker, gridPointsOf, parseChunk } from "./sceneWork";
import { durable, fetchPart, store, storedOf } from "./sceneStore";
import type { Bounds } from "../model/math3d";
import type { CameraData, CharacterData, CloudData, CloudSample, CurveData, CurveSample, GridSample, ModelData } from "./sceneTypes";
import { failed, mayAsk, notAnError, succeeded, whenAskable } from "../transfer/frameStore";
import { SCENES, partCacheKey, sceneCacheKey } from "../transfer/frameKey";

import { SCENE_SHARE, cache } from "../platform/cache";

/** 一份三维数据在浏览器中的表示：基础数据（静止网格、蒙皮权重、骨骼动画、相机曲线）加按帧到达的数据块。
 *
 * 本模块负责一份场景本身的组织方式以及数据块的获取与释放；`sceneData.ts` 负责获取来源、所用缓存
 * 以及组件的使用方式。 */

type Listener = () => void;

/** 整段缓存（总是整段，没有开关）与两层存放，职责不同，不得混淆：
 *
 * - 下载：一份视图打开后，后台把整段的块按顺序（当前块起、向后、回绕）并行 `PARALLEL` 路全部取回，与播放头无关；
 *   取回的字节写到使用者硬盘（view/sceneStore.ts，浏览器私有文件系统），刷新、重开浏览器后再看只从盘上读。
 *   时间线的绿色（loaded）= 已在本机、播放无须再下载；「已缓存 N / M 帧」由 `cached()` 报告。
 *   不能存盘的（边算边看：地址里没有代次；浏览器不支持私有文件系统）照旧只在页面缓存（platform/cache.ts）里。
 * - 解码：内存预算（页面缓存分给场景的一份，platform/cache.ts SCENE_SHARE）管理当前已解码的帧数。放得下就取到即解；放不下时只解
 *   当前块附近（停止时前后各一块，播放时向前 `AHEAD` 块），超出时释放距当前帧最远的样本块（drop），回到这些帧
 *   只需从硬盘重新读、重新解码（毫秒级），不重新下载。 */

// 播放时向前解码的块数（停止时为 1 块，见 Scene.reach）：块已在本机时足以覆盖解码
const AHEAD = 3;
// 整段下载的并行路数：服务器发的是预先生成的文件（view_data.store_chunk），并行才有收益；实际占几条连接由
// transfer/frameStore.ts 的连接配额定（后台道）
const PARALLEL = 3;


/** 一块的帧范围 [首, 末] 在整段帧表（升序）里对应的下标段 [起, 止)。 */
function spanOf(frames: number[], [first, last]: [number, number]): [number, number] {
  const at = (v: number) => { let lo = 0, hi = frames.length; while (lo < hi) { const m = (lo + hi) >> 1; if (frames[m] < v) lo = m + 1; else hi = m; } return lo; };
  return [at(first), at(last + 1)];
}

function explicit(points: Float32Array, colors: CloudSample["colors"], widths: Float32Array | null, bounds: Bounds | null = null): CloudSample {
  return { points, colors, widths, count: points.length / 3, grid: null, bounds };
}

/** 深度点云的一帧：点数当场数出，点本身在第一次有人要时（包围盒、拾取；Scene.wantBounds）由工作线程重建。
 *
 * 只有一份网格：服务器按「点云上限」抽稀后即发送该份，每个点均为数据自身的值。 */
function gridSample(ref: GridRef, depth: Float32Array, colors: GridSample["colors"], tint: Float32Array | null, focal: number, cam: Float32Array, principal: Float32Array | null): CloudSample {
  let count = 0;
  for (let k = 0; k < depth.length; k++) if (depth[k] === depth[k]) count++;
  const g = ref;
  return {
    points: null,
    colors: new Float32Array(0),
    widths: null,
    count,
    grid: { ref, gw: g.gw, gh: g.gh, step: g.step, depth, colors, tint, focal, cam, principal },
    bounds: null,
  };
}

/** 一份视图（一个场景包的，或一张深度图的点云预览）：它的数据，以及陆续到达的块。 */
export class Scene {
  readonly key: string;
  readonly desc: ViewDescription;
  readonly frames: number[];
  readonly models: ModelData[];
  readonly characters: CharacterData[];
  readonly clouds: CloudData[];
  readonly curves: CurveData[];
  readonly cameras: CameraData[];
  version = 0;
  error: string | null = null;
  private listeners = new Set<Listener>(); // 正在画它的视图：只有还有视图时，失败的块才会重新请求
  // 页面缓存的大小报告（sceneData.ts kept）：每次变化都通知它，但它不是视图，因此不会让重试继续下去
  onBytes: (() => void) | null = null;
  // local：字节已在使用者硬盘上（sceneStore）；fetched：本页打开后已下载过一次（不能存盘的块据此不重复下载）
  private chunks: { name: string; url: string; frames: [number, number]; state: "idle" | "loading" | "here"; bytes: number; span: [number, number];
                    local: boolean; fetched: boolean; fetching: boolean }[];
  private base: Record<string, Uint8Array>;
  private want = 0;
  private dir: -1 | 0 | 1 = 0;
  private loading = 0;
  private fetchFrom = 0; // 整段下载的游标：从当前块起这么多块都已下完（fetchAll）
  private downloading = 0;
  private diskKnown = false; // 已问过硬盘上有哪些块（构造时的 storedOf）
  private uvAsked: Promise<void> | null = null;
  private askedBounds = new WeakSet<CloudSample>(); // 只请求一次；请求失败则该样本没有包围盒
  // 正在计算的结果（边算边看）：已写出的帧。null 表示整段均已就绪（计算完成的结果）
  private ready: Set<number> | null = null;

  constructor(key: string, desc: ViewDescription, base: Record<string, Uint8Array>) {
    this.key = key;
    this.desc = desc;
    this.base = base;
    this.frames = desc.frames;
    const a = (r: Parameters<typeof baseArray>[1]) => baseArray(base, r);
    this.models = desc.models.map((m) => ({
      ref: m, faces: a(m.faces), world: a(m.world!) as Float32Array, points: m.points ? (a(m.points) as Float32Array) : null, samples: new Map(), uv: null,
    }));
    this.characters = desc.characters.map((c) => ({
      ref: c,
      bind: a(c.bind) as Float32Array,
      anim: a(c.anim) as Float32Array,
      meshes: c.meshes.map((m) => ({
        ref: m, faces: a(m.faces), points: a(m.points) as Float32Array,
        jointIndices: m.joint_indices ? a(m.joint_indices) : null, jointWeights: m.joint_weights ? (a(m.joint_weights) as Float32Array) : null,
        shapeOffsets: m.shape_offsets ? (a(m.shape_offsets) as Float32Array) : null, shapeWeights: m.shape_weights ? (a(m.shape_weights) as Float32Array) : null,
        samples: new Map(), uv: null,
      })),
    }));
    this.clouds = desc.clouds.map((c) => ({
      ref: c, key: `${key}.${c.name}`, name: c.name, frames: c.frames, world: a(c.world!) as Float32Array, width: c.width,
      still: c.points ? { ...explicit(a(c.points) as Float32Array, a(c.colors!) as CloudSample["colors"], c.widths ? (a(c.widths) as Float32Array) : null),
        covariance: c.covariance ? a(c.covariance) as Float32Array : undefined,
        opacity: c.opacity ? a(c.opacity) as Float32Array : undefined, sh: c.sh ? a(c.sh) as Float32Array : undefined } : null,
      samples: new Map(),
      focal: c.grid ? (a(c.grid.focal) as Float32Array) : null,
      cams: c.grid ? (a(c.grid.cam) as Float32Array) : null,
      principals: c.grid?.principal ? (a(c.grid.principal) as Float32Array) : null,
      scene: this,
    }));
    this.curves = desc.curves.map((c) => ({
      ref: c, key: `${key}.${c.name}`, name: c.name, frames: c.frames, world: a(c.world!) as Float32Array, width: c.width,
      still: c.points
        ? { vertexCounts: a(c.vertex_counts!), points: a(c.points) as Float32Array, colors: a(c.colors!) as CurveSample["colors"],
            widths: c.widths ? (a(c.widths) as Float32Array) : null, count: c.count, strands: c.strands }
        : null,
      samples: new Map(),
    }));
    this.cameras = desc.cameras.map((c) => ({
      ref: c, world: a(c.world!) as Float32Array, focalMm: a(c.focal_mm) as Float32Array, hAperture: a(c.h_aperture_mm) as Float32Array, vAperture: a(c.v_aperture_mm) as Float32Array,
      centerMm: c.center_mm ? (a(c.center_mm) as Float32Array) : null,
    }));
    this.chunks = Object.entries(desc.parts)
      .filter(([name]) => /^c\d+$/.test(name))
      .map(([name, p]) => ({ name, url: p.url, frames: p.frames!, state: "idle" as const, bytes: 0,
                             span: spanOf(this.frames, p.frames!), local: false, fetched: false, fetching: false }))
      .sort((x, y) => x.frames[0] - y.frames[0]);
    // 已在硬盘上的块（之前打开时已下载）：不重复下载，时间线直接标为已缓存。问到之前 `cached()` 不报进度
    // （否则整段早已在盘上的，打开时先闪一下「已缓存 0 / M 帧」）
    void storedOf(this.chunks.map((c) => c.url)).then(
      (kept) => {
        for (const c of this.chunks) if (kept.has(c.url)) c.local = c.fetched = true;
        this.diskKnown = true;
        this.changed();
        this.pump();
      },
      () => {
        this.diskKnown = true;
        this.pump();
      },
    );
  }

  /** 边算边看：解算器每写出一帧即传输一帧。
   *
   * 正在计算的结果为每帧一块（服务器 view_data.chunk_plan 的 one_frame_chunks），描述中一开始即列出
   * 整段的全部块，使块的地址不随已计算帧数变化。此处声明的是当前实际有内容的块：
   * 尚未写出的帧服务器返回 404，请求无意义。每次轮询时更新一次，新写出的帧立即开始获取。 */
  setReady(frames: number[] | null): void {
    this.ready = frames ? new Set(frames) : null;
    this.pump();
  }

  /** 该块当前是否可以请求（计算完成的结果始终可以）。每帧一块，因此只需检查首帧。 */
  private there(c: Scene["chunks"][number]): boolean {
    return !this.ready || this.ready.has(c.frames[0]);
  }

  /** 该块取失败且仍在退避期内：期间不取，也不让播放器等它（`pending`）。失败记忆在 transfer/frameStore.ts（按块的键）。 */
  private holding(c: Scene["chunks"][number]): boolean {
    return !mayAsk(partCacheKey(c.url));
  }

  /** 块取到并用上了 / 失败了：记进页面唯一的失败记忆；失败的到期后重试一次（该场景已无人看就不管）。 */
  private settled(c: Scene["chunks"][number], e?: unknown): void {
    const key = partCacheKey(c.url);
    if (e === undefined) {
      succeeded(key);
      this.error = null; // 后续的块已到达：清除上一次的失败通知
      return;
    }
    failed(key, e);
    // 等重新登录、服务器还在生成、退避期没去要的都不是这一块的错：不报，到时候再要
    if (!notAnError(e)) this.error = e instanceof Error ? e.message : String(e);
    whenAskable(key, () => {
      if (!this.listeners.size) return;
      this.pump();
      this.changed(); // 让 Stage3D 重新算 pending：该块重新进入在途
    });
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    return () => void this.listeners.delete(fn);
  }

  private changed() {
    this.version++;
    this.onBytes?.();
    for (const fn of this.listeners) fn();
  }

  /** 查看器所在的帧与播放方向（0：未播放）：先取该帧所在的块，再取播放方向前方的块，然后向两侧展开，
   * 数量以内存预算（SCENE_SHARE）为限。 */
  setFrame(frame: number, dir: -1 | 0 | 1 = 0): void {
    if (!this.chunks.length) return;
    const i = this.chunks.findIndex((c) => frame >= c.frames[0] && frame <= c.frames[1]);
    const was = this.want;
    this.want = i >= 0 ? i : frame < this.chunks[0].frames[0] ? 0 : this.chunks.length - 1;
    // 已切换到另一块：清除上一块的失败通知（再次失败时会重新提示）。若不清除 `error`，通知将一直显示，
    // 该场景也不会再拉取其他块
    if (this.want !== was) {
      this.error = null;
      this.fetchFrom = 0; // 整段下载改从新的当前块起
    }
    this.dir = dir;
    this.pump();
  }

  /** 这一份还能常驻多少逐帧样本：全部场景共用的 SCENE_SHARE，减去其它场景已占的、减去自己的基础数据
   * （各场景的字节由 sceneData.ts 按 `bytes()` 报给页面缓存，组 SCENES）。 */
  private room(): number {
    // 其它场景已占的（自己可能还没记进缓存：在建时是 0 字节占位）不会是负的
    const others = Math.max(0, cache.bytesIn(SCENES) - cache.sizeOf(sceneCacheKey(this.key)));
    return Math.max(0, SCENE_SHARE - others - baseBytes(this.desc));
  }

  /** 该场景当前占用的字节数：基础数据（始终存在）加已解码的逐帧样本块。
   *
   * 页面唯一的缓存据此计算该项的大小（kept()），因此保留的场景数量按字节而非按个数决定：
   * 小场景可保留十余份，一份大场景即可占满预算，需要时被释放。 */
  bytes(): number {
    return baseBytes(this.desc) + this.chunks.reduce((a, c) => a + (c.state === "here" ? c.bytes : 0), 0);
  }

  /** 播放无须再下载的帧（时间线的绿色）：已解码的，和字节已在本机硬盘上的；null：没有逐帧的内容。 */
  loaded(): number[] | null {
    if (!this.chunks.length) return null;
    const out: number[] = [];
    for (const c of this.chunks) if (c.state === "here" || c.local) this.push(out, c);
    return out;
  }

  /** 整段缓存的进度：[已在本机的帧数, 总帧数]；null：没有逐帧的内容（固定点云等整段一次到齐），或还没问到硬盘上有哪些。 */
  cached(): [number, number] | null {
    if (!this.diskKnown) return null;
    const got = this.loaded();
    return got === null ? null : [got.length, this.frames.length];
  }

  /** 逐帧数据块尚未到达的帧（正在拉取、排队待拉取、解算器尚未写出的均计入）：播放器到达这些帧时须等待
   * （editor/Timeline.tsx：「待播放的帧未到达时等待其到达」）。三维数据块不在二维取帧账本中（transfer/frames.ts onItsWay），
   * 若不由此处报告，播放器会认为无人读取而按帧率强行推进，导致画面不连续。null：该数据不是逐帧的。 */
  pending(): number[] | null {
    if (!this.chunks.length) return null;
    const out: number[] = [];
    // 拉取失败、正在退避的块不算在途：退避期间无人拉取，等待即为死等；播放器跳过这些帧，
    // 到期后会自动重新拉取（见 `load` 的 catch）
    // 字节已在本机的块也不算在途：缓存完成后播放只从本地取，解码是毫秒级，向前 AHEAD 块已先解好
    for (const c of this.chunks) if (c.state !== "here" && !c.local && !this.holding(c)) this.push(out, c);
    return out;
  }

  /** 一块覆盖的帧（构造时按下标算好的一段 `span`，不逐帧比较）。 */
  private push(out: number[], c: Scene["chunks"][number]): void {
    for (let i = c.span[0]; i < c.span[1]; i++) out.push(this.frames[i]);
  }

  /** 一块相对当前帧的载入距离：播放方向前方的块算近，后方的算远。 */
  private cost(i: number): number {
    const d = i - this.want;
    return this.dir !== 0 && Math.sign(d) === -this.dir ? Math.abs(d) * 4 : Math.abs(d);
  }

  /** 内存放不下整段时，解码到当前帧附近多远：停止时当前块及前后各一块（单帧步进不卡顿），播放时沿播放方向向前
   * `AHEAD` 块。只管解码；下载不受它限制，总是整段（`fetchAll`）。 */
  private reach(): number {
    return this.dir !== 0 ? AHEAD : 1;
  }

  private pump(): void {
    this.decodeNear();
    this.fetchAll();
  }

  /** 当前帧附近的块：取来（已在本机的从硬盘读）并解码，内存预算不够时先释放最远的。 */
  private decodeNear(): void {
    while (this.loading < (this.dir ? 3 : 2)) {
      const next = this.chunks
        .map((c, i) => ({ c, i, d: this.cost(i) }))
        .filter(({ c, d }) => c.state === "idle" && !this.holding(c) && d <= this.reach() && this.there(c))
        .sort((x, y) => x.d - y.d)[0];
      if (!next) return;
      const held = this.chunks.reduce((a, c) => a + (c.state === "here" ? c.bytes : 0), 0);
      if (held > this.room()) {
        // 为近处的块腾出空间：释放最远的已解码样本块。其字节仍在页面缓存或本机硬盘上（transfer/frameStore.ts partBytes），
        // 回到这些帧只需重新解码，无需重新下载

        const far = this.chunks.map((c, i) => ({ c, d: this.cost(i) })).filter(({ c }) => c.state === "here").sort((x, y) => y.d - x.d)[0];
        if (far && far.d > next.d) {
          this.drop(far.c.frames);
          far.c.state = "idle";
          far.c.bytes = 0;
          continue;
        }
        // 没有更远的块可释放（通常是只保留了当前块且其单独占满预算）：当前块与相邻的下一块仍然拉取，
        // 暂时超出一块预算，下一轮播放头移开后它即成为最远的块而被优先释放。若不拉取，`pending()` 会一直报告这些帧，
        // 播放器按 editor/Timeline.tsx 的规则将无限等待一个不会到达的块。更远的块不拉取，到达时再处理
        if (next.d > 1) return;
      }
      this.load(next.c);
    }
  }

  private load(chunk: Scene["chunks"][number]): void {
    chunk.state = "loading";
    this.loading++;
    fetchPart(chunk.url, "now")
      .then(async (bytes) => {
        chunk.bytes = bytes.byteLength;
        await this.keepOnDisk(chunk, bytes); // 先存盘：解析时字节整体转交给解析线程，之后就不在这里了
        return this.apply(bytes);
      })
      .then(() => {
        chunk.state = "here";
        this.settled(chunk);
      })
      .catch((e) => {
        // 失败不是永久性的：退避期间 `pump` 不选它、`pending` 不报它，到期后重试
        chunk.state = "idle";
        this.settled(chunk, e);
      })
      .finally(() => {
        this.loading--;
        this.changed();
        this.pump(); // 失败的块处于退避期，不会被再次选中，其余块照常拉取
      });
  }

  /** 字节写到硬盘（可存盘的地址才写，sceneStore.durable），成功后该块记为已在本机。同一块可能被 `load`（当前块附近）
   * 和 `fetchAll`（整段）同时取到、各写一次：只让成功的那次置位，后落地的失败不把已置的「在本机」改回去。 */
  private async keepOnDisk(chunk: Scene["chunks"][number], bytes: Uint8Array): Promise<void> {
    chunk.fetched = true;
    if (!chunk.local && durable(chunk.url) && (await store(chunk.url, bytes).catch(() => false))) chunk.local = true;
  }

  /** 整段后台缓存：还没下载过的块，从当前块起向后（回绕）按顺序，`PARALLEL` 路并行取回并存盘，与播放头远近无关。
   * 内存放得下就顺手解码（整段常驻，播放不必读盘）；放不下的只留在硬盘上，播到附近时由 `decodeNear` 读盘解码。
   * 只在有视图画它时下载（`listeners`，与失败重试同一条件）：换了显示的节点，旧的那份停下、不和正在看的那份抢带宽
   * （否则点过几个点云节点就有几份各 `PARALLEL` 路同时下载）；再显示它时（Stage3D 订阅后 setFrame → pump）从停下处接着下。 */
  private fetchAll(): void {
    if (!this.listeners.size) return;
    const n = this.chunks.length;
    while (this.downloading < PARALLEL) {
      // 从当前块起向后（回绕）第一块还没下的：游标 `fetchFrom`（相对当前块的偏移）之前的都已下完，不每次从头扫、不排序
      let chunk: Scene["chunks"][number] | null = null;
      for (let k = this.fetchFrom; k < n; k++) {
        const c = this.chunks[(this.want + k) % n];
        if (c.fetched) {
          if (k === this.fetchFrom) this.fetchFrom++;
          continue;
        }
        if (!c.fetching && c.state === "idle" && !this.holding(c) && this.there(c)) {
          chunk = c;
          break;
        }
      }
      if (!chunk) return;
      chunk.fetching = true;
      this.downloading++;
      fetchPart(chunk.url, "later")
        .then(async (bytes) => {
          await this.keepOnDisk(chunk, bytes);
          const held = this.chunks.reduce((a, c) => a + (c.state === "here" ? c.bytes : 0), 0);
          if (chunk.state === "idle" && held + bytes.byteLength <= this.room()) {
            chunk.state = "loading";
            chunk.bytes = bytes.byteLength;
            try {
              await this.apply(bytes);
              chunk.state = "here";
            } catch {
              chunk.state = "idle"; // 解不开：留待播到附近时 decodeNear 再试（字节仍在本机）
            }
          }
          this.settled(chunk);
        })
        .catch((e) => this.settled(chunk, e))
        .finally(() => {
          chunk.fetching = false;
          this.downloading--;
          this.changed();
          this.pump();
        });
    }
  }

  /** 逐帧点云在基础数据里的颜色（与首帧相同的帧不重复带颜色）；没有时 null。取一次、留着。 */
  private colours = new Map<number, CloudSample["colors"] | null>();
  private cloudColours(i: number): CloudSample["colors"] | null {
    if (!this.colours.has(i)) {
      const ref = this.clouds[i].ref;
      this.colours.set(i, ref.per_frame && ref.colors ? (baseArray(this.base, ref.colors) as CloudSample["colors"]) : null);
    }
    return this.colours.get(i)!;
  }

  /** 一块的字节：在页面线程之外解析（sceneWork → sceneWorker：readChunk，外加工作线程持有时顺带算出的每个显式点云样本的
   * 包围盒），这里只负责放进各样本。 */
  private apply(bytes: Uint8Array): Promise<void> {
    return parseChunk(bytes).then(({ pieces, bounds }) => {
      const clouds = new Map<string, Partial<CloudSample>>();
      const strands = new Map<string, Partial<CurveSample>>();
      for (const { piece, samples } of pieces) {
        const [kind, i, , k] = piece.item.split("/");
        samples.forEach((arr, n) => {
          const s = piece.samples[0] + n;
          if (kind === "models") this.models[+i].samples.set(s, arr as Float32Array);
          else if (kind === "characters") this.characters[+i].meshes[+k].samples.set(s, arr as Float32Array);
          else if (kind === "curves") {
            const key = `${i}|${s}`;
            const got = strands.get(key) ?? {};
            (got as Record<string, unknown>)[piece.array] = arr;
            strands.set(key, got);
          } else if (kind === "clouds") {
            const key = `${i}|${s}`;
            const got = clouds.get(key) ?? {};
            (got as Record<string, unknown>)[piece.array] = arr;
            clouds.set(key, got);
          }
        });
      }
      for (const [key, got] of strands) {
        const [i, s] = key.split("|").map(Number);
        const g = got as Record<string, Float32Array | undefined>;
        const points = g.points ?? new Float32Array(0);
        const counts = (g.vertex_counts ?? new Uint16Array(0)) as unknown as CurveSample["vertexCounts"];
        this.curves[i].samples.set(s, {
          vertexCounts: counts, points, colors: (g.colors?.length ? g.colors : new Float32Array(points.length).fill(0.7)) as CurveSample["colors"],
          widths: g.widths ?? null, count: points.length / 3, strands: counts.length,
        });
      }
      for (const [key, got] of clouds) {
        const [i, s] = key.split("|").map(Number);
        const c = this.clouds[i];
        const g = got as Record<string, Float32Array | undefined>;
        if (c.ref.grid && g.depth?.length) c.samples.set(s, gridSample(c.ref.grid, g.depth, g.grid_colors?.length ? (g.grid_colors as GridSample["colors"]) : null, g.grid_tint?.length ? g.grid_tint : null, c.focal![s], c.cams!.subarray(s * 16, s * 16 + 16), c.principals ? c.principals.subarray(s * 2, s * 2 + 2) : null));
        else {
          const points = g.points ?? new Float32Array(0);
          // 这一帧没带颜色：颜色与首帧相同，只在基础数据里发了一次（server/view_data.py encode，点缓存）
          const same = this.cloudColours(i);
          const colors = g.colors?.length ? g.colors : same && same.length === points.length ? same : new Float32Array(points.length).fill(0.7);
          c.samples.set(s, { ...explicit(points, colors as CloudSample["colors"], g.widths ?? null, bounds[key] ?? null),
            covariance: g.covariance, opacity: g.opacity, sh: g.sh });
        }
      }
    });
  }

  /** 在工作线程里由样本派生的数据：只请求一次，写回样本，回复到达后重绘（changed()）。深度点云的点与包围盒一起重建，
   * 与 GPU 绘制时的算术完全相同（gridPoints）。 */
  wantBounds(sample: CloudSample): void {
    if (sample.bounds || this.askedBounds.has(sample)) return;
    this.askedBounds.add(sample);
    if (sample.grid) {
      const { ref, depth, focal, cam, principal } = sample.grid;
      gridPointsOf(depth, sample.grid.gw, sample.grid.step, ref.width, ref.height, focal, cam, principal, ref.aspect ?? 1).then(
        ({ points, bounds }) => {
          sample.points = points;
          sample.bounds = bounds;
          this.changed();
        },
        () => undefined, // 包围盒保持未设：按高度 / 深度着色与拾取在没有它时照常进行
      );
    } else if (sample.points) boundsInWorker(sample.points).then((bounds) => ((sample.bounds = bounds), this.changed()), () => undefined);
  }

  private drop(range: [number, number]): void {
    const inside = (frames: number[], s: number) => frames[s] >= range[0] && frames[s] <= range[1];
    for (const m of this.models) for (const s of [...m.samples.keys()]) if (inside(m.ref.frames, s)) m.samples.delete(s);
    for (const c of this.characters) for (const m of c.meshes) for (const s of [...m.samples.keys()]) if (inside(c.ref.frames, s)) m.samples.delete(s);
    for (const c of this.clouds) for (const s of [...c.samples.keys()]) if (inside(c.frames, s)) c.samples.delete(s);
    for (const c of this.curves) for (const s of [...c.samples.keys()]) if (inside(c.frames, s)) c.samples.delete(s);
  }

  /** UV（只有 UV 棋盘格需要）：只取一次。 */
  loadUv(): Promise<void> {
    const part = this.desc.parts.uv;
    if (!part) return Promise.resolve();
    this.uvAsked ??= fetchPart(part.url, "now").then((bytes) => {
      const parts = { ...this.base, uv: bytes };
      const corners = (uv: ModelRef["uv"]) => (uv ? cornerUvs(baseArray(parts, uv.values) as Float32Array, baseArray(parts, uv.indices)) : null);
      for (const m of this.models) m.uv = corners(m.ref.uv);
      for (const c of this.characters) for (const m of c.meshes) m.uv = corners(m.ref.uv);
      this.changed();
    });
    return this.uvAsked;
  }
}
