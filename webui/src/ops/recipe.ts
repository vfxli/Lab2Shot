import { BadOp, kindOf, run, varsOf, type Pix } from "./run";
import type { Show } from "./pixels";

/** 由浏览器计算节点图的最后几步：选人、拆分列表、人物框转遮罩、图像合成等可在客户端完成的计算均在客户端完成。
 *
 * 本模块只负责接线，不包含算法：哪个端口对应算法的哪个参数，以及计算结果如何成为下一步的输入。
 * 接线按运算种类编写，而非按节点编写：词汇表共三种运算（`lab2shot/ops/vocab.py` 的 `KINDS`），
 * 下方 `runStep` 每种一段。新增节点无需修改本文件，只需在 `NodeDef.browser_ops` 中声明所用算法。
 * 仅当词汇表新增第四种运算时才需修改本文件（届时两个执行器都须修改）。
 * 算法本体位于算法目录（`lab2shot/ops/ops.toml`，执行器 `ops/run.py` 与 `ops/run.ts` 按同一份描述运行），
 * 节点参数到算法参数的转换在服务器端完成（`NodeDef.browser_ops`，状态回复中的 `ops` 即其结果），
 * 因此浏览器不解释任何参数，只按服务器给出的 `{op, args}` 执行。
 *
 * 一份配方（Recipe）描述一帧的计算方式，可整体发送给 worker（不含函数和 DOM）：
 * 逐像素部分计算量较大（1080p 三通道一次合成约六百万次求值），在页面主线程执行会造成卡顿，
 * 因此 `view/evalWorker.ts` 在另一线程运行同一个 `runRecipe`，两处共用同一份实现。
 *
 * 本模块不产生包、不写入缓存、不写任何数据：计算结果仅用于绘制当前帧。交付与下游计算始终以服务器结果为准。
 *
 * 此处抛出的 `BadOp` 表示开发错误，而非用户消息（与 `ops/run.ts`、`lab2shot/ops/vocab.py` 规则相同）：
 * 仅在接线错误时触发，因此使用英文且不进入消息目录（由 `webui/tests/messages.test.ts` 检查）。
 * 用户可见的消息一律由服务器端的节点给出。 */

/** 服务器指定的算法及由节点参数确定的算法参数（`NodeDef.browser_ops`）。 */
export interface OpCall {
  op: string;
  args: Record<string, unknown>;
}

/** 一个人物：编号及各帧的框（格式与 `/api/packet/{fp}/boxes` 相同，也是算法目录中「挑条目」接受的格式）。 */
export interface Person {
  id: number;
  boxes: Record<string, number[]>;
  prominence?: number;
}

/** 一份完整的人物框数据：每帧若干浮点数，体量较小。 */
export interface Boxes {
  people: Person[];
  width: number;
  height: number;
  frames: number[];
}

/** 一条通道的一帧，展平为 float32（仅描述形状：本层不涉及包和传输格式）。
 * 传输格式（`transfer/plane.ts` 的 `Plane`，格式随数据而定）由取数层转换为本类型（`values(p)`）。 */
export interface Grid {
  w: number;
  h: number;
  values: Float32Array;
}

/** 某一步的某个端口的数据来源。 */
export type Give =
  | { got: "step"; at: number } // 前面第 at 步的结果
  | { got: "boxes"; data: Boxes } // 服务器提供的人物框（整段一份 JSON）
  | { got: "planes"; planes: Grid[] } // 服务器提供的当前帧各通道，按顺序叠为一份多通道数据
  | { got: "none" }; // 该路当前帧无数据（B 缺帧：服务器端同样将 A 原样输出）

/** 一步：节点的接线、所用算法、各端口的数据来源，以及该步所需的附加数值。 */
export interface Step {
  node: string; // 节点类型 id
  ops: OpCall[];
  give: Record<string, Give>;
  at: Record<string, unknown>; // 该步所需的附加数值（目前仅有当前帧号）
}

export interface Recipe {
  steps: Step[];
  show: Show;
}

/** 一步的结果：人物框（列表中各条合并后也是此类型）或一张二维数据。 */
export interface Made {
  boxes?: Boxes;
  pix?: Pix;
}

// ---------------------------------------------------------------- 多条通道 → 一份多通道数据

function stack(planes: Grid[]): Pix {
  const { w, h } = planes[0];
  const c = planes.length;
  if (c === 1) return { w, h, c, data: planes[0].values }; // 单通道：直接使用原数组，不复制
  const data = new Float32Array(w * h * c);
  for (let i = 0; i < c; i += 1) {
    const v = planes[i].values;
    for (let p = 0; p < w * h; p += 1) data[p * c + i] = v[p];
  }
  return { w, h, c, data };
}

const asPix = (m: Made | undefined): Pix => {
  if (!m?.pix) throw new BadOp("a step wants an image, the step before gave none");
  return m.pix;
};
function fetchGive(give: Give, done: Made[]): Made | undefined {
  if (give.got === "step") return done[give.at];
  if (give.got === "boxes") return { boxes: give.data };
  if (give.got === "planes") return { pix: stack(give.planes) };
  return undefined;
}

// ---------------------------------------------------------------- 接线（仅接线，不含算法；按运算种类，而非按节点）

/** 该人物在当前帧的框；该帧不在画面中时返回 undefined（与服务器 cook 的判断一致：`str(f) in person["boxes"]`）。 */
const boxAt = (p: Person, frame: number): number[] | undefined => p.boxes[String(frame)];

/** 该步输入中唯一的一份（用于不使用任何算法的步骤：拆分条目，而非运算）。 */
function theOne(ins: Record<string, Made | undefined>): Made {
  const got = Object.values(ins).filter((m): m is Made => !!m);
  if (got.length !== 1) throw new BadOp(`a pass-through step wants exactly one input, got ${got.length}`);
  return got[0];
}

/** 该步输入中带条目的那一份（「挑条目」所需）。 */
function theItems(ins: Record<string, Made | undefined>): Boxes {
  const got = Object.values(ins).find((m) => m?.boxes);
  if (!got?.boxes) throw new BadOp("an items step wants items, no input has any");
  return got.boxes;
}

/** 执行一步：按算法种类分派，而非按节点。
 *
 * 三种运算各一段（词汇表 `lab2shot/ops/vocab.py` 的 `KINDS`）：
 * - 挑条目（`items.pick`）：从输入条目中挑出若干条，输出同一份数据且仅保留选中的条目
 *   （「选人」即此步；后续还有运算时，下一步使用挑出的条目）；
 * - 按框涂（`boxes.paint`）：将上一步挑出的框在当前帧的位置绘制为一张遮罩，
 *   画布尺寸取该条目数据自身的宽高；
 * - 每像素算术（`pixel`）：公式所需的变量从同名端口读取（词汇表约定：变量名即输入名）。
 *   若某个端口当前帧无数据（B 缺帧），则将第一个变量原样输出，与服务器 cook 的处理一致
 *   （`lacking`：这些帧原样输出并给出提示）。
 *
 * 不使用任何算法时（`ops` 为空，如「拆成列表」：拆分条目而非运算），输入原样向下传递。 */
function runStep(ops: OpCall[], ins: Record<string, Made | undefined>, at: Record<string, unknown>): Made {
  if (!ops.length) return theOne(ins);
  let made: Made | null = null;
  for (const call of ops) {
    const kind = kindOf(call.op);
    if (kind === "items.pick") {
      const src: Boxes = made?.boxes ?? theItems(ins);
      const got = run(call.op, { items: src.people, ...call.args }) as { indices: number[] };
      made = { boxes: { ...src, people: got.indices.map((i) => src.people[i]) } };
    } else if (kind === "boxes.paint") {
      const src: Boxes = made?.boxes ?? theItems(ins);
      const frame = Number(at.frame);
      const boxes = src.people.map((p) => boxAt(p, frame)).filter((b): b is number[] => b !== undefined);
      const got = run(call.op, { boxes, width: src.width, height: src.height, ...call.args }) as { mask: Pix };
      made = { pix: got.mask };
    } else if (kind === "pixel") {
      const names = varsOf(call.op);
      const args: Record<string, unknown> = { ...call.args };
      for (const n of names) if (ins[n]?.pix) args[n] = ins[n]!.pix;
      // 某个变量当前帧无数据：将第一个变量原样输出（与服务器端一致）
      if (names.some((n) => args[n] === undefined)) made = { pix: asPix(ins[names[0]]) };
      else made = { pix: (run(call.op, args) as { value: Pix }).value };
    } else throw new BadOp(`no browser wiring for op kind ${kind || "(not in the catalog)"}: ${call.op}`);
  }
  if (!made) throw new BadOp("a step gave nothing");
  return made;
}

/** 执行一份配方，返回最后一步的结果。纯函数：页面主线程与 worker 运行同一段代码。 */
export function runRecipe(steps: Step[]): Made {
  const done: Made[] = [];
  for (const step of steps) {
    const ins: Record<string, Made | undefined> = {};
    for (const [port, give] of Object.entries(step.give)) ins[port] = fetchGive(give, done);
    done.push(runStep(step.ops, ins, step.at));
  }
  return done[done.length - 1];
}
