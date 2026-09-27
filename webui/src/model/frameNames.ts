/** 帧号与文件名之间的换算，均为纯函数，集中定义于此（`view/useOriginals.ts` 查找本机原件时使用）。
 *
 * 两种来源各有一种写法，均遵循服务器的规则（`lab2shot/io/sequence.py`，序列识别规则仅在该处定义）：
 *
 * 1. 读取节点：参数中保存的是图案（`plate.####.exr`，`#` 的个数即补零位数，
 *    `lab2shot/io/sequence.py` 中 `FrameSequence.name` 使用 `'#' * max(padding, 1)`）。
 *    第 1003 帧即将 `#` 串替换为补足位数的帧号（`_spell`：负号不计入补零位数）。
 * 2. 交付的文件：包中已列出每个文件名（`Delivery.files`），
 *    因此不推断图案，而是从文件名中读取帧号，即扩展名之前的最后一串数字。
 *    读取结果不一致（有重复或完全没有）时视为非序列：单个文件视为静帧，
 *    其他情况返回空表，这些帧仍使用服务器代理。
 *
 * 不向服务器请求此表的原因：换算发生在绘制帧的过程中，额外的往返会增加等待，且文件名已在本地。 */

const HASHES = /#+/;

/** 返回序列图案中第 `frame` 帧的文件名（`plate.####.exr` + 1003 → `plate.1003.exr`）。
 * 图案中没有 `#`（单个文件，非序列）时返回图案本身。 */
export function nameForFrame(pattern: string, frame: number): string {
  const run = HASHES.exec(pattern);
  if (!run) return pattern;
  const body = String(Math.abs(frame)).padStart(run[0].length, "0");
  return pattern.replace(HASHES, frame < 0 ? `-${body}` : body);
}

/** 返回扩展名之前的最后一串数字（`depth_ML_Lab2Shot_Pi3.1001.exr` → 1001），没有时返回 null。 */
export function frameInName(name: string): number | null {
  const base = name.slice(name.lastIndexOf("/") + 1);
  const dot = base.lastIndexOf(".");
  const stem = dot > 0 ? base.slice(0, dot) : base;
  const runs = stem.match(/\d+/g);
  if (!runs?.length) return null;
  return Number(runs[runs.length - 1]);
}

/** 一条路径的扩展名（小写，带点）。 */
const extOf = (name: string): string => {
  const base = name.slice(name.lastIndexOf("/") + 1);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot).toLowerCase() : "";
};

/** 从一个「输出设置」交付的文件中筛选出序列本身。
 *
 * 交付的子文件夹中不只有序列：`lab2shot/nodes/output.py` 还会在旁边写入一份说明文件
 * （`images/images.lab2shot.json`）。该文件必须排除，否则「images.lab2shot」中的 `2` 会被识别为第 2 帧，
 * 序列中会多出一帧，且画面上无法察觉。
 *
 * 判据为扩展名而非文件名：序列各帧的扩展名相同，取出现次数最多的扩展名。
 * 文件名形式多样（含 `_ML_Lab2Shot_<项目>`、图层名、视角），扩展名则保持一致。 */
const sameKind = (names: readonly string[]): string[] => {
  const byExt = new Map<string, string[]>();
  for (const n of names) byExt.set(extOf(n), [...(byExt.get(extOf(n)) ?? []), n]);
  let best: string[] = [];
  for (const group of byExt.values()) if (group.length > best.length) best = group;
  return best;
};

/** 文件名列表 → 帧号 → 文件名。单个文件视为静帧（帧号由调用方提供）；
 * 多个文件中存在重复帧号或无法识别帧号时返回空表（不做推断，改用服务器提供的数据）。
 * 扩展名与多数文件不同的文件（交付子文件夹中的说明文件）预先排除，见 `sameKind`。 */
export function framesOfNames(all: readonly string[], still?: number): Map<number, string> {
  const names = sameKind(all);
  const out = new Map<number, string>();
  if (names.length === 1) {
    const one = frameInName(names[0]);
    const at = one ?? still;
    if (at !== undefined) out.set(at, names[0]);
    return out;
  }
  for (const name of names) {
    const at = frameInName(name);
    if (at === null || out.has(at)) return new Map();
    out.set(at, name);
  }
  return out;
}

// ------------------------------------------------------------------ 查询的缓存键
//
// 规则：答案所依赖的每一项都必须包含在键中（`view/useOriginals.ts` 的 `Ask.key`）。
// 遗漏任何一项，该项变化时不会重新查询，答案将停留在旧值，且不会报错，测试也无法发现，
// 只能通过截图比对像素察觉。因此键以纯函数形式定义于此，
// 供测试直接验证（`webui/tests/originals.test.ts`）；视图层依赖浏览器环境，测试无法覆盖。

/** 输出侧查询的缓存键（纯函数，由 `webui/tests/originals.test.ts` 验证）。
 *
 * `state` 表示交付进度，它决定包是否已写入使用者磁盘，从而决定查询结果，
 * 因此必须包含在键中（原因见上方 `Ask.key` 一节）。 */
export const deliveredKey = (a: { fp: string; graphId: string; node: string; name: string;
                                  handle?: string; run?: string; state?: string; dirs: number }): string =>
  `out|${a.fp}|${a.graphId}|${a.node}|${a.name}|${a.handle ?? ""}|${a.run ?? ""}:${a.state ?? ""}|d${a.dirs}`;

/** 读取侧查询的缓存键（纯函数，由测试验证）。
 *
 * `dirs` 表示已授权目录的变更次数（`state/localDirs.ts`）：使用者新指定目录后，
 * 先前找不到的文件可能变得可用，因此需要重新查询。 */
export const readKey = (a: { fp: string; ref: string; folder: string; dirs: number; span?: string; space?: string }): string =>
  `read|${a.fp}|${a.ref}|${a.folder}|d${a.dirs}|${a.span ?? ""}|${a.space ?? ""}`;  // 色彩空间包含在键中：答案（EXR 的解码方式）依赖于它

/** 返回序列图案在文件名列表中匹配到的帧（`plate.####.exr` + 目录中的文件名 → [1001, 1002, …]）。
 *
 * 不由首帧与末帧推算的原因：序列可能存在跳号，
 * 例如使用者选择的 1001–1300 之间可能只有 200 张（见 `editor/FileParam.tsx` 中「中间跳 N 帧」的提示）。
 * 推算得到的帧表会包含不存在的帧，视图上这些帧将始终处于未到达状态，且不会给出任何提示。
 * 以目录中实际存在的文件为准，可正确处理跳号。
 *
 * `#` 的个数即补零位数（与 `nameForFrame` 规则相同，服务器实现位于
 * `lab2shot/io/sequence.py`）：位数不符的文件名不匹配（`plate.1001.exr` 的图案不匹配 `plate.12345.exr`）。
 * 图案中没有 `#`（单张图像，非序列）时返回空表：此时帧号只有服务器的包说明能确定，
 * 不得在此推断（推断错误会将第 1001 帧显示为其他帧）。 */
export function framesMatchingPattern(names: Iterable<string>, pattern: string): number[] {
  const run = HASHES.exec(pattern);
  if (!run) return [];
  const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const at = run.index;
  const re = new RegExp(`^${escape(pattern.slice(0, at))}(-?\\d{${run[0].length}})${escape(pattern.slice(at + run[0].length))}$`);
  const out: number[] = [];
  for (const name of names) {
    const m = re.exec(name);
    if (m) out.push(Number(m[1]));
  }
  return out.sort((a, b) => a - b);
}
