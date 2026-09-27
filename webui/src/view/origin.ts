// 本文件只允许类型导入，不得 import 任何有运行时副作用的模块：`webui/tests/origin.test.ts` 直接导入本文件进行校验，
// 引入 react 或状态库会导致该测试无法运行。
import type { LocalPicture } from "./localPick";
import type { ViewItem } from "./plan";

/** # 来源：视图中某个输出口当前所显示数据的出处
 *
 * 「该输出口是否有内容可显示、内容来自何处」仅在本模块定义一次。若在多处分别实现，各处定义的细微差异会导致画面变暗、
 * 相机参数显示为「—」、通道只剩 rgba、已有结果却提示「还没有结果」、depth 层显示彩色图等问题。
 *
 * ## 定义
 *
 * 输出口的来源恰为以下五种之一，按优先级排列（`Where`）：
 *
 * | 来源 | 含义 | 判定依据 |
 * |---|---|---|
 * | `"server"` | 服务器计算得到的包 | `ViewItem.fp`（状态回复中该输出口为 `present`） |
 * | `"browser"` | 浏览器本地计算的结果（积木节点） | `ViewItem.local`（`view/useLocal.ts`） |
 * | `"file"` | 使用者本机的文件，且由该节点读取 | `useLocalPicture` 返回的文件，其 `node` 等于该输出口所属节点 |
 * | `"plate"` | 同样是本机文件，但由上游节点读取，仅作为衬底，不属于该输出口自身 | 同上，但 `node` 为某个上游节点 |
 * | `"none"` | 无内容 | 以上四种均不成立 |
 *
 * 调用方不得自行编写「是否有内容」的布尔判断，只使用以下两个字段：
 * - `own`：该输出口自身有内容可显示，即 `server` / `browser` / `file`。
 *   「结果通道」下拉中的「还没算」提示、2D/3D 舞台选择、叠加显示开关均以此为准。
 *   非自身内容也不压暗显示，因为使用者无法判断压暗的时机与原因。
 * - `any`：屏幕上该输出口有内容可显示，即 `own` 或上游提供的衬底（`plate`）。
 *   是否显示「还没有结果」页面以此为准。
 *
 * `file` 与 `plate` 必须区分：`useLocalPicture` 沿上游查找（`view/localPick.ts`），因此下游运算节点也会取得
 * 上游读入的文件，该文件只是衬底而非其自身结果。若二者合并，「还没算」提示将不再出现，
 * 边算边看（`view/partial.ts`）也会误判为已有结果而停止。
 *
 * ## 不在本模块范围内：帧字节从本机磁盘还是从服务器读取
 *
 * 该问题由 `transfer/sources.ts` 处理（`serverFrames` 对每一帧先查询 `originalsFor`，`localFirst` 按 PNG / EXR
 * 决定优先来源）。来源为 `"server"` 的输出口，其字节仍可能来自使用者本机磁盘（已计算的包加授权目录中的原件，无需传输）。
 * 两者不得混淆：包说明、画面尺寸、时间线、下游计算与交付始终以服务器为准，只有字节可能从本机读取。
 *
 * ## 身份：答案所依赖的每一项都必须包含在身份中
 *
 * `Source.key` 是该答案的身份，来源一旦变化即随之变化。它参与取数层的备忘依赖（`view/stageSources.ts`）
 * 以及 `DisplayPlan.sourceKey`。遗漏任何一项都会使画面停留在旧来源上，且既不报错也不会使测试失败。 */

export type Where = "server" | "browser" | "file" | "plate" | "none";

export interface Source {
  /** 内容的来源（见上表）。 */
  where: Where;
  /** 显示的是上一次的结果（参数已修改但结构未变，`state/stale.ts`）：时间线显示为土黄色，节点右上角标注「已过期」。 */
  stale: boolean;
  /** 该输出口自身有内容可显示（服务器的包、浏览器计算结果，或该节点自身读取的本机文件）。 */
  own: boolean;
  /** 屏幕上该输出口有内容可显示：`own`，或以上游读取的本机文件作为衬底。 */
  any: boolean;
  /** 本机文件，仅在 `where` 为 `"file"` 或 `"plate"` 时存在：包含帧列表、文件以及 EXR 的解码色彩空间。 */
  file: LocalPicture | null;
  /** 该答案的身份（见上文「身份」一节）。 */
  key: string;
}

const NOTHING: Source = { where: "none", stale: false, own: false, any: false, file: null, key: "" };

/** 本机文件的身份，作为 `Source.key` 的一部分：由读取节点和文件集合决定。
 * 是否为同一答案仅以该字符串判断。`view/localPick.ts` 在该字符串不变时返回同一对象，取数层的备忘才能生效，
 * 否则每次渲染都会产生新对象并重建帧源。
 *
 * 上传进度等信息不属于身份：它持续变化，但不影响显示的像素（该信息由 `useUploadHint` 和文件参数行显示，
 * 见 `ui/UploadState.tsx`）。 */
export const pictureKey = (p: LocalPicture): string =>
  `${p.node}|${p.item.kind}|${p.item.name}|${p.item.files.length}|${p.item.frames.length}|${p.item.size}|${p.space}`;

/** 判断使用者本机文件能否代表视图中的某个输出口。`filePort` 为本机文件可代表的输出口名，`""` 表示不能代表任何输出口。
 *
 * 浏览器使用 three 的 EXRLoader 解码本机 EXR（`transfer/exr.ts`），只能得到文件自身的颜色通道，即节点主画面所在的层；
 * 文件中的其他层（depth、法线、运动矢量、遮罩）无法取得。
 *
 * 若不加此限制，多层 EXR 的读取节点不会自动计算、各层的包尚不存在，无条件使用本机文件会在「depth」层下显示彩色图，
 * 且没有任何提示。
 *
 * 不硬编码 `"image"`：那是服务器端的输出口命名规则（`lab2shot/nodes/core/input.py _outputs_of`：`rgba` 层命名为 `image`），
 * 页面不重复服务器的规则（见 `graph/rules.ts`）。此处使用视图自身的主输出口（`graph/rules.ts mainOutput`：
 * 节点声明的主输出口，未声明时取第一个；「取第一个」的规则全项目仅在该处实现，由 `webui/tests/mainOutput.test.ts` 保证）。 */
export const fileStandsFor = (filePort: string, port: string): boolean => !!filePort && port === filePort;

/** 返回输出口当前显示内容的来源，定义见模块说明。
 *
 * `file` 为视图可从使用者本机取得的画面（`view/localPick.ts`，沿上游查找），
 * `filePort` 为该文件可代表的输出口（`fileStandsFor`）。 */
export function sourceOf(it: ViewItem, file: LocalPicture | null, filePort: string): Source {
  const mine = file && fileStandsFor(filePort, it.port) ? file : null;
  // 只有该节点自身读取的文件才是其自身结果；上游读取的文件仅作为衬底（见上文 file 与 plate 的区分）。
  const own = mine && mine.node === it.nodeId ? mine : null;
  const plate = mine && !own ? mine : null;
  const where: Where = it.fp ? "server" : it.local ? "browser" : own ? "file" : plate ? "plate" : "none";
  if (where === "none") return NOTHING;
  return {
    where,
    stale: where === "server" && it.stale,
    own: where !== "plate",
    any: true,
    file: mine,
    // 身份随来源变化，必须包含包、浏览器计算结果和本机文件三项。
    key: `${where}${it.stale ? "~" : ""}|${it.fp ?? ""}|${it.local?.manifest.fingerprint ?? ""}|${mine ? pictureKey(mine) : ""}`,
  };
}

/** 同一判断的另一个入口是 `transfer/local.ts hasLocalFile`，用于尚无 `ViewItem` 的场景
 * （判断节点是否需要自动计算，`graph/actions.ts`）。
 *
 * 该入口不放在本文件中，原因是网页分层约束（`webui/tests/layers.test.ts`）：`graph` 层位于 `view` 层之下，不得导入本模块。
 * `transfer/` 层负责管理使用者本机文件的可用性，因此放在该层。两处使用同一判据（`drawable`：浏览器能否解码该文件），
 * 本文件不重复实现。 */
