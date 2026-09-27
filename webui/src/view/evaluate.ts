import { api } from "../api";
import { cache } from "../transfer/cache";
import { channelFrames, fetchNow, manifestOf } from "../transfer/frames";
import { lutOfPacket } from "../transfer/lut";
import { isPlane, values, type Plane } from "../transfer/plane";
import { manifestNow, tierTag } from "../transfer/frames";
import type { Snapshot } from "../graph/snapshot";
import { PARAM, inputsOf } from "../graph/rules";
import { elementOf } from "../state/items";
import { BadOp, kindOf, varsOf } from "../ops/run";
import { runRecipe, type Boxes, type Give, type Grid, type OpCall, type Recipe, type Step } from "../ops/recipe";
import type { Show } from "../ops/pixels";
import { browserCanCompute } from "../graph/rules";

/** 从显示的节点向上游遍历，判断当前画面能否由浏览器自行计算：选人、拆分列表、人物框转遮罩、图像合成
 * 等可在浏览器完成的计算均在浏览器中完成。
 *
 * 判断能否在浏览器计算的三个条件（任一不满足即回退到服务器结果）：
 * 1. 服务器声明该节点可由浏览器计算：状态回复中带有 `ops`（即 `NodeDef.browser_ops` 的结果，
 *    内容为所用算法及由参数确定的算法参数）。是否带有由服务器判定，泛型节点还按其实际承载的类型分别判定
 *    （「拆成列表」承载人物框时可拆分，承载场景时不可），页面不解释任何参数。
 * 2. 其所用算法可由浏览器执行：即词汇表中的三种运算（`lab2shot/ops/vocab.py` 的 `KINDS`，
 *    接线见 `ops/recipe.ts runStep`，按运算种类编写，而非按节点编写）。
 * 3. 向上游遍历，遇到第一个服务器已计算的包即停止；途中任一节点不满足条件 1 或 2，则整条路径不在浏览器计算。
 *
 * 无法在浏览器计算不属于静默降级：服务器结果本身即为权威（交付与下游计算始终使用它），浏览器结果只是让用户
 * 调整参数后能立即看到效果。因此无法计算时照常显示服务器结果，无需在通知区提示。
 *
 * 本模块不产生包、不写入缓存、不写任何数据：计算结果仅用于绘制当前帧。 */

/** 某一步输入的来源（`Give` 为执行时的形式，此处另有一种「按帧获取指定通道」）。 */
export type Need = Give | { got: "channels"; fp: string; names: string[]; frames: number[] };

export interface Planned {
  node: string;
  ops: OpCall[];
  give: Record<string, Need>;
  at: Record<string, unknown>;
}

/** 浏览器计算的结果：作为包说明使用的 `meta`、按帧实时计算的方式（`steps`）、转为像素的方式（`show`）。 */
export interface Local {
  type: string;
  meta: Record<string, unknown>; // width / height / frames / range / values / data_window：作为 manifest.meta 使用
  steps: Planned[];
  show: Show;
  frames: number[];
}

/** 路径上的某一份数据：来源、类型及其说明。 */
interface Src {
  give: Need;
  type: string;
  meta: Record<string, unknown>;
  boxes?: Boxes; // 人物框分支：整段数据量很小，到达此处时已在本地
}

const num = (v: unknown, or = 0): number => (typeof v === "number" ? v : or);
const frameList = (meta: Record<string, unknown>): number[] => (Array.isArray(meta.frames) ? (meta.frames as number[]) : []);

/** 一个包的人物框，整段获取一次后保留（视图绘制框与浏览器计算均需使用：一处获取，两处使用）。
 * 保留的是 Promise 而非结果，否则同一时刻的两次请求会发出两个网络请求。 */
export function boxesOf(fp: string): Promise<Boxes> {
  const key = `boxes:${fp}`;
  const had = cache.get<Promise<Boxes>>(key);
  if (had) return had;
  const asking = api.boxes(fp).then((b) => b as unknown as Boxes);
  cache.keep(key, asking, 1024, "small");
  asking.catch(() => cache.forget(key));
  return asking;
}

/** 服务器已计算的端口（由 `present` 给出），作为该路径的起点。
 *
 * 该路径的起点有意不查询本机原件：
 *
 * 显示路径（`transfer/sources.ts serverFrames`）会查询本机原件，因为它输出的是纹理，
 * 两侧尺寸不同时由 GPU 按各自的 UV 采样到舞台的像素网格上，`view/look.ts` 即如此处理。
 * 此处不同：该路径在 worker 中基于同一张网格进行算术运算（`ops/recipe.ts runRecipe`），
 * 「图像合成」一步要求 A 与 B 逐像素对齐。本机原件为全尺寸，服务器代理为等比缩放后的档位，
 * 混用须先重采样，而重采样属于需要读取邻近像素的运算，两个执行器均尚未支持
 * （见 `lab2shot/ops/vocab.py` 中的表）。
 *
 * 因此该路径仍使用代理的通道：这是能力限制，而非设计选择。
 * 将来在 `KINDS` 中加入采样类运算后，此处应改为先查询 `transfer/originals.ts`，
 * 再按数据类型的声明选择滤波器（遮罩和标签图不得插值）。 */
async function served(fp: string, type: string): Promise<Src | null> {
  const m = await manifestOf(fp);
  if (type === "boxes") return { give: { got: "boxes", data: await boxesOf(fp) }, type, meta: m.meta, boxes: await boxesOf(fp) };
  const names = ((m as { channels?: { names?: string[] } }).channels?.names ?? []).filter((n) => n !== "valid");
  if (!names.length) return null; // 非二维像素的包（相机、点云、曲线）：本层不处理
  return { give: { got: "channels", fp, names, frames: frameList(m.meta) }, type, meta: m.meta };
}

/** 该端口当前是否有服务器已计算的包（`present`：实际已写出的端口）。 */
const cookedFp = (s: Snapshot, nodeId: string, port: string): string | null =>
  s.results[nodeId]?.present?.includes(port) ? s.results[nodeId]?.outputs?.[port] ?? null : null;

/** 该端口在节点图中的类型（见服务器计算的 `ports`）。 */
const portType = (s: Snapshot, nodeId: string, port: string): string =>
  s.results[nodeId]?.ports?.outputs?.find((p) => p.name === port)?.type ?? "";

class Walk {
  readonly steps: Planned[] = [];
  private readonly seen = new Set<string>();
  private readonly s: Snapshot;

  // 参数属性（`constructor(private readonly s)`）在 node 的「仅剥离类型」模式下无法运行，
  // 而测试需要用 node 直接运行这些文件（webui/tests/*.test.ts），因此写成普通赋值
  constructor(s: Snapshot) {
    this.s = s;
  }

  /** 一个端口数据的来源（null：浏览器无法继续该路径）。 */
  async source(nodeId: string, port: string): Promise<Src | null> {
    const fp = cookedFp(this.s, nodeId, port);
    if (fp) return served(fp, portType(this.s, nodeId, port));
    if (this.seen.has(nodeId)) return null; // 节点图中存在环（引擎本身也会拦截）：不进入
    this.seen.add(nodeId);
    const node = this.s.nodes.find((n) => n.id === nodeId);
    const status = this.s.results[nodeId];
    const ops = status?.ops;
    if (!node || !ops || !browserCanCompute(status)) return null; // 服务器未声明其可计算，或其所用运算在浏览器端无法执行
    const ins: Record<string, Src> = {};
    for (const p of inputsOf(this.s, nodeId)) {
      if (p.name.startsWith(PARAM)) continue; // 由连线驱动的参数：其值已在参数中
      const wire = this.s.edges.find((e) => e.target === nodeId && e.targetHandle === p.name);
      if (!wire) return null; // 缺少连线：服务器会自行报告「必需输入未连接」
      const from = await this.source(wire.source, wire.sourceHandle ?? "");
      if (!from) return null;
      ins[p.name] = from;
    }
    return this.put(node.data.typeId, ops, ins, portType(this.s, nodeId, port));
  }

  /** 该步输出数据的说明（作为 `manifest.meta` 使用），以及浏览器能否继续计算。
   *
   * 按算法种类计算，而非按节点：网页中不写死节点类型，新增同类节点无需修改网页。
   *
   * 三种运算的处理（词汇表 `lab2shot/ops/vocab.py` 的 `KINDS`，与 `ops/recipe.ts runStep` 一一对应）：
   * - 挑条目（`items.pick`）及不使用任何算法的步骤（「拆成列表」：拆分条目而非运算）：
   *   形状不变，说明原样向下传递；
   * - 按框涂（`boxes.paint`）：输出为覆盖率，即一条通道的数值图，取值 0..1，画布尺寸为该条目数据自身的
   *   宽高（与服务器 cook 中 `ExrWriter(..., value_range=UNIT)` 的含义相同）。
   *   这是该运算种类的性质而非某个节点的性质，因此新节点使用它同样正确；
   * - 每像素算术（`pixel`）：公式第一个变量对应的数据决定形状（同名端口，词汇表约定：变量名即输入名），
   *   要求每个变量都有端口输入且宽高一致。
   *
   * 浏览器不计算数值图上的每像素算术（回退到服务器）：服务器写数值图时会扫描整段以确定显示范围
   * （`ExrWriter` 的 `value_range=None`），浏览器只能看到一帧，无法得到相同的范围，绘制结果将是另一张图，
   * 而非同一张图的预览。画面不存在此问题（显示变换表只与色彩空间有关，与具体的包无关）。
   * 这同样是该运算种类的限制，而非某个节点的限制。 */
  private put(typeId: string, ops: OpCall[], ins: Record<string, Src>, outType: string): Src | null {
    const give: Record<string, Need> = Object.fromEntries(Object.entries(ins).map(([k, v]) => [k, v.give]));
    const kinds = ops.map((o) => kindOf(o.op));
    const last = kinds[kinds.length - 1] ?? "";
    const shape = last === "boxes.paint" ? painted(ins) : last === "pixel" ? pixelwise(ops, ins) : passed(ins);
    if (!shape) return null;
    const type = outType || shape.type;
    this.steps.push({ node: typeId, ops, give, at: {} });
    const at = this.steps.length - 1;
    // 人物框分支整段数据量很小，到达此处时即已计算完成，视图用其绘制框。
    // 「拆成列表」输出的是整条列表，而视图需绘制列表中的每一条，合起来即为输入的那一份；
    // 这由 `runStep` 的「原样向下传递」完成，此处无需单独处理
    const boxes = elementOf(type) === "boxes" ? runRecipe(resolveKnown(this.steps)).boxes : undefined;
    return { give: { got: "step", at }, type, meta: shape.meta, ...(boxes ? { boxes } : {}) };
  }
}

/** 形状不变的步骤（挑条目、拆分条目）：输入中唯一一份数据的说明原样向下传递。
 * 挑条目另有一项要求：输入数据必须确实包含条目（浏览器端的条目只有人物框）。 */
function passed(ins: Record<string, Src>): { type: string; meta: Record<string, unknown> } | null {
  const got = Object.values(ins);
  if (got.length !== 1 || !got[0].boxes) return null;
  return { type: got[0].type, meta: got[0].meta };
}

/** 按框涂的输出：一条通道的数值图，取值 0..1，画布尺寸为该条目数据自身的宽高。 */
function painted(ins: Record<string, Src>): { type: string; meta: Record<string, unknown> } | null {
  const src = Object.values(ins).find((v) => v.boxes);
  if (!src?.boxes) return null;
  return { type: "image.1",
           meta: { width: src.boxes.width, height: src.boxes.height, frames: src.boxes.frames,
                   values: true, range: [0, 1] } };
}

/** 每像素算术的输出：公式第一个变量对应的数据决定形状。 */
function pixelwise(ops: OpCall[], ins: Record<string, Src>): { type: string; meta: Record<string, unknown> } | null {
  const names = varsOf(ops[ops.length - 1].op);
  const sources = names.map((n) => ins[n]);
  if (!sources.length || sources.some((v) => !v)) return null; // 公式所需的端口未连接：服务器会自行报告
  const first = sources[0];
  if (first.meta.values) return null; // 数值图回退到服务器计算（原因见上方注释）
  if (sources.some((v) => num(v.meta.width) !== num(first.meta.width) || num(v.meta.height) !== num(first.meta.height)))
    return null;
  // 通道数不同无需回退：执行器按服务器的规则处理，输出取第一个变量的通道数，另一份只取其第一条通道
  // （ops/run.ts runPixel）；因此此处输出的类型与说明仍取第一份
  return { type: first.type, meta: first.meta };
}

/** 一张二维数据转为屏幕像素的方式（与服务器生成显示图的算法一一对应，见 `recipe.ts toPixels`）。 */
async function showOf(meta: Record<string, unknown>, lutFrom: string | null): Promise<Show> {
  const range = (Array.isArray(meta.range) ? (meta.range as number[]) : [0, 1]) as [number, number];
  if (meta.values) return { values: true, range, alpha: false, lut: null };
  return { values: false, range, alpha: !!meta.alpha, lut: lutFrom ? await lutOfPacket(lutFrom) : null };
}

/** 路径上最上游的服务器包（画面分支的显示变换据此获取：该表只与色彩空间有关，与具体的包无关）。 */
function lutSource(steps: Planned[]): string | null {
  for (const s of steps) for (const give of Object.values(s.give)) if (give.got === "channels") return give.fp;
  return null;
}

/** 浏览器能否计算该节点的该端口（null：无法计算，显示服务器结果）。 */
export async function evaluate(s: Snapshot, nodeId: string, port: string): Promise<Local | null> {
  const walk = new Walk(s);
  const src = await walk.source(nodeId, port).catch(() => null);
  if (!src || walk.steps.length === 0) return null;
  const steps = walk.steps;
  return { type: src.type, meta: src.meta, steps,
           show: await showOf(src.meta, lutSource(steps)), frames: frameList(src.meta) };
}

/** 无需获取任何数据即可执行的步骤（人物框分支：整段 JSON 已在本地）。 */
const resolveKnown = (steps: Planned[]): Step[] =>
  steps.map((s) => ({ ...s, give: Object.fromEntries(Object.entries(s.give).map(([k, g]) => [k, g.got === "channels" ? { got: "none" as const } : g])) }));

/** 人物框分支的结果（整段一次计算完成，与帧无关）。 */
export const localBoxes = (local: Local): Boxes | null => runRecipe(resolveKnown(local.steps)).boxes ?? null;

/** 当前帧所需的通道。经由取帧账本获取（transfer/frames.ts channelFrames + fetchNow）：
 * 使用同一地址、同一套键、同一份缓存，视图为绘制该层已获取的通道不会被重复请求
 * （不得绕过账本另建取数路径）。 */
async function planesFor(fp: string, frame: number, names: string[], frames: number[]): Promise<Grid[]> {
  const tier = tierTag(manifestNow(fp));
  const got = await Promise.all(names.map((n) => fetchNow(channelFrames(fp, n, frames, tier), frame)));
  const planes = got.filter((p): p is Plane => !!p && isPlane(p));
  // 开发错误而非用户消息：使用英文（webui/tests/messages.test.ts）
  if (planes.length !== names.length) throw new BadOp(`channels missing on frame ${frame}: ${names.join()}`);
  // 传输格式随数据而定（u8 / u16 / 半精度 / float32，GPU 路径直接作为纹理上传）；
  // CPU 路径需要 float32，转换一次后保留（transfer/plane.ts values）
  return planes.map((p) => ({ w: p.width, h: p.height, values: values(p) }));
}

/** 当前帧的计算方式：获取所需的通道（按通道取数路径）后，配方即完整。 */
export async function recipeFor(local: Local, frame: number): Promise<Recipe> {
  const steps: Step[] = [];
  for (const s of local.steps) {
    const give: Record<string, Give> = {};
    for (const [port, need] of Object.entries(s.give)) {
      if (need.got !== "channels") give[port] = need;
      else give[port] = need.frames.includes(frame)
        ? { got: "planes", planes: await planesFor(need.fp, frame, need.names, need.frames) }
        : { got: "none" };
    }
    steps.push({ ...s, give, at: { ...s.at, frame } });
  }
  return { steps, show: local.show };
}
