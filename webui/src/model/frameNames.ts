/** 帧号与文件名之间的换算，均为纯函数，集中定义于此（`view/useOriginals.ts` 查找本机原件时使用）。
 *
 * 遵循服务器的规则（`lab2shot/io/sequence.py`，序列识别规则仅在该处定义）：读取节点的参数中保存的是图案
 * （`plate.####.exr`，`#` 的个数即补零位数，`lab2shot/io/sequence.py` 中 `FrameSequence.name` 使用
 * `'#' * max(padding, 1)`）。第 1003 帧即将 `#` 串替换为补足位数的帧号（`_spell`：负号不计入补零位数）。
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

// ------------------------------------------------------------------ 查询的缓存键
//
// 规则：答案所依赖的每一项都必须包含在键中（`view/useOriginals.ts` 的 `Ask.key`）。
// 遗漏任何一项，该项变化时不会重新查询，答案将停留在旧值，且不会报错。
// 因此键以纯函数形式集中定义于此，不散落在视图层中。

/** 读取侧查询的缓存键（纯函数）。
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
