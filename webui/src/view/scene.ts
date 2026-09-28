import { baseArray, baseBytes, cornerUvs, type GridRef, type ModelRef, type ViewDescription } from "../model/viewFormat";
import { boundsInWorker, gridPointsOf, parseChunk } from "./sceneWork";
import { fetchPart } from "./scenePart";
import type { Bounds } from "../model/math3d";
import type { CameraData, CharacterData, CloudData, CloudSample, CurveData, CurveSample, GridSample, ModelData } from "./sceneTypes";
import { backoff } from "../platform/backoff";

/** 一份三维数据在浏览器中的表示：基础数据（静止网格、蒙皮权重、骨骼动画、相机曲线）加按帧到达的数据块。
 *
 * 本模块负责一份场景本身的组织方式以及数据块的获取与释放；`sceneData.ts` 负责获取来源、所用缓存
 * 以及组件的使用方式。 */

type Listener = () => void;

/** 两层预算，职责不同，不得混淆：
 *
 * - 页面唯一的缓存（transfer/cache.ts）记录下载的字节：场景的描述、基础数据以及每个数据块。
 *   键即其地址，地址中含内容的 sha 或结果指纹 + 帧号，因此已持有的数据不会再向服务器请求。
 * - 内存预算（memoryBytes，按标签页的堆上限计算）管理当前已解码的帧数：
 *   超出时释放距当前帧最远的样本块（drop）。被释放的块其字节仍在上述缓存中，回到这些帧只需重新解码（毫秒级），
 *   不重新下载任何字节，也不依赖浏览器自身的 HTTP 缓存是否保留。 */

// 播放时向前预取的块数（停止时为 1 块，见 Scene.reach）：足以覆盖解码与网络往返，又不至于拉取整段
const AHEAD = 3;

/** 视图每份场景可保留的逐帧数据字节上限：按当前标签页的堆上限计算，而非按机器内存
 * （`navigator.deviceMemory` 按规范上限为 8，据此每份可达 4 GB，而一个标签页的堆约为 4 GB；堆接近上限时
 * 新建位图会静默失败，导致着色与运算失效、浏览器卡顿）。 */
export function memoryBytes(): number {
  const limit = (performance as { memory?: { jsHeapSizeLimit?: number } }).memory?.jsHeapSizeLimit;
  return Math.min(1e9, Math.max(0.25e9, (limit ?? 2e9) / 4));
}

function explicit(points: Float32Array, colors: CloudSample["colors"], widths: Float32Array | null, bounds: Bounds | null = null): CloudSample {
  return { points, colors, widths, count: points.length / 3, grid: null, bounds };
}

/** A depth cloud's frame: counted now, its points reconstructed (by the worker) the first time something requests
 * them (bounds, picking; Scene.wantBounds).
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

/** One view (a scene packet's, or a depth map's point preview): its data, and its chunks as they arrive. */
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
  private listeners = new Set<Listener>(); // the views drawing it: a failed chunk is asked again only while there is one
  // the page cache's size report (sceneData.ts kept): told of every change, but not a view, so it keeps no retry going
  onBytes: (() => void) | null = null;
  // failedAt / fails：该块上次拉取失败的时间及连续失败次数（0：未失败），退避时长见 `holdOff`
  private chunks: { name: string; url: string; frames: [number, number]; state: "idle" | "loading" | "here"; bytes: number; failedAt: number; fails: number }[];
  private base: Record<string, Uint8Array>;
  private want = 0;
  private dir: -1 | 0 | 1 = 0;
  private keep = memoryBytes();
  private loading = 0;
  private uvAsked: Promise<void> | null = null;
  private askedBounds = new WeakSet<CloudSample>(); // requested once; a failed request leaves the sample without bounds
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
      still: c.points ? explicit(a(c.points) as Float32Array, a(c.colors!) as CloudSample["colors"], c.widths ? (a(c.widths) as Float32Array) : null) : null,
      samples: new Map(),
      focal: c.grid ? (a(c.grid.focal) as Float32Array) : null,
      cams: c.grid ? (a(c.grid.cam) as Float32Array) : null,
      principals: c.grid?.principal ? (a(c.grid.principal) as Float32Array) : null,
      scene: this,
    }));
    this.curves = (desc.curves ?? []).map((c) => ({
      ref: c, key: `${key}.${c.name}`, name: c.name, frames: c.frames, world: a(c.world!) as Float32Array, width: c.width,
      still: c.points
        ? { vertexCounts: a(c.vertex_counts!), points: a(c.points) as Float32Array, colors: a(c.colors!) as CurveSample["colors"],
            widths: c.widths ? (a(c.widths) as Float32Array) : null, count: c.count, strands: c.strands }
        : null,
      samples: new Map(),
    }));
    this.cameras = desc.cameras.map((c) => ({
      ref: c, world: a(c.world!) as Float32Array, focalMm: a(c.focal_mm) as Float32Array, hAperture: a(c.h_aperture_mm) as Float32Array, vAperture: a(c.v_aperture_mm) as Float32Array,
    }));
    this.chunks = Object.entries(desc.parts)
      .filter(([name]) => /^c\d+$/.test(name))
      .map(([name, p]) => ({ name, url: p.url, frames: p.frames!, state: "idle" as const, bytes: 0, failedAt: 0, fails: 0 }))
      .sort((x, y) => x.frames[0] - y.frames[0]);
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

  /** 拉取失败后的重试间隔（毫秒）：按连续失败次数翻倍，1 秒、2 秒、4 秒……最多半分钟。 */
  private static holdOff(c: Scene["chunks"][number]): number {
    return backoff(c.fails, 1000, 30_000);
  }

  /** 该块拉取失败且仍在退避期内：期间不拉取，也不让播放器等待它（`pending`）。 */
  private holding(c: Scene["chunks"][number]): boolean {
    return c.fails > 0 && Date.now() - c.failedAt < Scene.holdOff(c);
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

  /** The frame the viewer is on and the play direction (0: not playing): the frame's chunk comes first, then the
   * chunks ahead in the play direction, then the rest outward, as many as the memory budget (memoryBytes) allows. */
  setFrame(frame: number, dir: -1 | 0 | 1 = 0): void {
    if (!this.chunks.length) return;
    const i = this.chunks.findIndex((c) => frame >= c.frames[0] && frame <= c.frames[1]);
    const was = this.want;
    this.want = i >= 0 ? i : frame < this.chunks[0].frames[0] ? 0 : this.chunks.length - 1;
    // 已切换到另一块：清除上一块的失败通知（再次失败时会重新提示）。若不清除 `error`，通知将一直显示，
    // 该场景也不会再拉取其他块
    if (this.want !== was) this.error = null;
    this.dir = dir;
    this.pump();
  }

  /** 该场景当前占用的字节数：基础数据（始终存在）加已解码的逐帧样本块。
   *
   * 页面唯一的缓存据此计算该项的大小（kept()），因此保留的场景数量按字节而非按个数决定：
   * 小场景可保留十余份，一份大场景即可占满预算，需要时被释放。 */
  bytes(): number {
    return baseBytes(this.desc) + this.chunks.reduce((a, c) => a + (c.state === "here" ? c.bytes : 0), 0);
  }

  /** The frames whose samples are currently in the view (the timeline's 已载入视图); null: nothing here is per frame. */
  loaded(): number[] | null {
    if (!this.chunks.length) return null;
    const out: number[] = [];
    for (const c of this.chunks) if (c.state === "here") for (const f of this.frames) if (f >= c.frames[0] && f <= c.frames[1]) out.push(f);
    return out;
  }

  /** 逐帧数据块尚未到达的帧（正在拉取、排队待拉取、解算器尚未写出的均计入）：播放器到达这些帧时须等待
   * （editor/Timeline.tsx：「待播放的帧未到达时等待其到达」）。三维数据块不在二维取帧账本中（transfer/frames.ts onItsWay），
   * 若不由此处报告，播放器会认为无人读取而按帧率强行推进，导致画面不连续。null：该数据不是逐帧的。 */
  pending(): number[] | null {
    if (!this.chunks.length) return null;
    const out: number[] = [];
    // 拉取失败、正在退避的块不算在途：退避期间无人拉取，等待即为死等；播放器跳过这些帧，
    // 到期后会自动重新拉取（见 `load` 的 catch）
    for (const c of this.chunks) if (c.state !== "here" && !this.holding(c)) for (const f of this.frames) if (f >= c.frames[0] && f <= c.frames[1]) out.push(f);
    return out;
  }

  /** A chunk's loading distance from the current frame: chunks ahead in the play direction are near, those behind are far. */
  private cost(i: number): number {
    const d = i - this.want;
    return this.dir !== 0 && Math.sign(d) === -this.dir ? Math.abs(d) * 4 : Math.abs(d);
  }

  /** 只拉取可见的帧。
   *
   * 若一直补充直至内存预算（memoryBytes）上限，最终会拉取整段：本机访问时影响不大，
   * 但在公网上随意拖动几下时间条即可产生数百 MB 流量（一段 150 帧 1080p 的深度点云整段约 136 MB，点云间隔为 1 时约 2.2 GB）。
   * 停止时只拉取当前块及相邻的块（前后各一块，保证单帧步进不卡顿）；
   * 播放时沿播放方向向前预取 `AHEAD` 块。其余块不拉取，拖动到该处时再拉取，同样优先拉取该块。
   * 内存预算上限仍决定保留多少，本规则决定拉取多少，二者相互独立。 */
  private reach(): number {
    return this.dir !== 0 ? AHEAD : 1;
  }

  private pump(): void {
    while (this.loading < (this.dir ? 3 : 2)) {
      const next = this.chunks
        .map((c, i) => ({ c, i, d: this.cost(i) }))
        .filter(({ c, d }) => c.state === "idle" && !this.holding(c) && d <= this.reach() && this.there(c))
        .sort((x, y) => x.d - y.d)[0];
      if (!next) return;
      const held = this.chunks.reduce((a, c) => a + (c.state === "here" ? c.bytes : 0), 0);
      if (held > this.keep) {
        // 为近处的块腾出空间：释放最远的已解码样本块。其字节仍在页面唯一的缓存中（scenePart.ts），
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
    fetchPart(chunk.url)
      .then((bytes) => ((chunk.bytes = bytes.byteLength), this.apply(bytes)))
      .then(() => {
        chunk.state = "here";
        chunk.fails = 0;
        chunk.failedAt = 0;
        this.error = null; // 后续的块已到达：清除上一次的失败通知
      })
      .catch((e) => {
        // 失败不是永久性的：记录时间并退避（`holdOff`），到期后重试。退避期间 `pump` 不选择它，`pending` 也不报告它
        chunk.state = "idle";
        chunk.fails += 1;
        chunk.failedAt = Date.now();
        this.error = e instanceof Error ? e.message : String(e);
        // 到期时若播放头未移动则不会有人调用 pump，因此主动重试一次；若该场景已无订阅者则不处理
        setTimeout(() => {
          if (!this.listeners.size) return;
          this.pump();
          this.changed(); // 通知 Stage3D 重新计算 pending：该块重新进入在途状态
        }, Scene.holdOff(chunk) + 50);
      })
      .finally(() => {
        this.loading--;
        this.changed();
        this.pump(); // 失败的块处于退避期，不会被再次选中，其余块照常拉取
      });
  }

  /** A chunk's bytes, parsed off the page's thread (sceneWork → sceneWorker: readChunk, plus the bounds of every
   * explicit cloud sample computed while the worker held them), then only placed into the samples here. */
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
          c.samples.set(s, explicit(points, (g.colors?.length ? g.colors : new Float32Array(points.length).fill(0.7)) as CloudSample["colors"], g.widths ?? null, bounds[key] ?? null));
        }
      }
    });
  }

  /** Data derived from a sample in the worker: requested once, written into the sample, and the redraw follows the
   * response (changed()). A depth cloud's points are reconstructed together with its bounds, exactly as the GPU
   * draws them (gridPoints). */
  wantBounds(sample: CloudSample): void {
    if (sample.bounds || this.askedBounds.has(sample)) return;
    this.askedBounds.add(sample);
    if (sample.grid) {
      const { ref, depth, focal, cam, principal } = sample.grid;
      gridPointsOf(depth, sample.grid.gw, sample.grid.step, ref.width, ref.height, focal, cam, principal).then(
        ({ points, bounds }) => {
          sample.points = points;
          sample.bounds = bounds;
          this.changed();
        },
        () => undefined, // bounds remain unset: colouring by height/depth and picking proceed without them
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

  /** The UVs (needed only by the UV checker): fetched once. */
  loadUv(): Promise<void> {
    const part = this.desc.parts.uv;
    if (!part) return Promise.resolve();
    this.uvAsked ??= fetchPart(part.url).then((bytes) => {
      const parts = { ...this.base, uv: bytes };
      const corners = (uv: ModelRef["uv"]) => (uv ? cornerUvs(baseArray(parts, uv.values) as Float32Array, baseArray(parts, uv.indices)) : null);
      for (const m of this.models) m.uv = corners(m.ref.uv);
      for (const c of this.characters) for (const m of c.meshes) m.uv = corners(m.ref.uv);
      this.changed();
    });
    return this.uvAsked;
  }
}
