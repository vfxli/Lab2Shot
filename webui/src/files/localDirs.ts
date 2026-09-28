import { run } from "../platform/db";
import { canReadFolder, DIR_PREFIX, keptAll, readable, readFolder } from "./handles";
import { useLocalDirs } from "../state/localDirs";
import { randomId } from "../platform/randomId";

/** 用户授权的本机目录：视图从中查找读取节点所读素材的原件（`transfer/originals.ts` 查询的第一个来源）。
 *
 * 浏览器仅在用户选择文件时向页面提供文件，单纯在标签页中选取的文件刷新后即失效；因此由用户授权一次素材所在的
 * 目录（`FileSystemDirectoryHandle`，只读，句柄存于 IndexedDB），此后页面直接读取该目录，刷新后仍然有效。
 *
 * 约束：
 * 1. 权限并非永久有效：句柄可以持久化，但刷新后 `queryPermission()` 可能回到「需再次询问」，
 *    而再次询问必须由用户点击触发。因此不弹出对话框：无法读取时静默回落到服务器代理，
 *    需要原件的用户可通过「文件 · 授权本机素材文件夹…」授权（该菜单项始终存在，不可用时置灰并注明原因）。
 * 2. 该接口仅 Chromium 系浏览器提供（Firefox / Safari 没有 `showDirectoryPicker`）。
 *    不支持的浏览器上本功能静默不可用：回落到代理，不报错、不提示、不在视图上显示通知控件。
 * 3. 页面无法得知素材所在的目录（浏览器不暴露本机路径），因此由用户授权一次；
 *    之后按文件名在该目录中查找，找不到时回落到代理，不视为错误。
 *
 * IndexedDB 仅用于省去重复授权，不是可靠存储（私密窗口或清除站点数据后不可用）：
 * 所有读写均包在 try/catch 中，读取失败视为不存在。 */

interface Kept {
  id: string;
  handle: FileSystemHandle;
}

const idOf = () => `${DIR_PREFIX}${randomId()}`;

/** 用户授权过的全部目录，不检查权限（其中可能有当前不可读的，见 `readableDirs`）。 */
export async function authorisedDirs(): Promise<{ id: string; dir: FileSystemDirectoryHandle }[]> {
  const all = await keptAll();
  return all
    .filter((k): k is Kept => k.id.startsWith(DIR_PREFIX) && k.handle?.kind === "directory")
    .map((k) => ({ id: k.id, dir: k.handle as FileSystemDirectoryHandle }));
}

/** 当前可读的授权目录（`queryPermission` 返回 granted）；不可读的目录静默忽略。 */
async function readableDirs(): Promise<FileSystemDirectoryHandle[]> {
  const out: FileSystemDirectoryHandle[] = [];
  for (const { dir } of await authorisedDirs()) if (await readable(dir)) out.push(dir);
  return out;
}

/** 由用户选择一个目录。必须在点击事件中调用，系统对话框需要用户手势。
 * 返回目录名，取消时返回 null。同一目录重复选择只保留一条记录。 */
export async function addDir(): Promise<string | null> {
  if (!canReadFolder) return null;
  const dir = await readFolder("lab2shot-originals").catch(() => null);
  if (!dir) return null;
  try {
    // 同一目录重复选择只保留一条记录，但仍需发出下方的变更通知：
    // 重复选择通常意味着希望重新读取（文件夹内容可能已变化）。
    let had = false;
    for (const one of await authorisedDirs()) if (!had && (await one.dir.isSameEntry(dir))) had = true;
    if (!had) await run("handles", "readwrite", (s) => s.put({ id: idOf(), handle: dir }));
  } catch {
    // 无法持久化（私密窗口或已清除站点数据）：本次仍可使用，下次需重新选择。
  }
  // 选择后必须发出变更通知：视图查询「本机是否有原件」的输入之一是授权目录列表，
  // 不通知则该查询不会重新执行，画面会一直停留在服务器代理上，直到刷新。
  useLocalDirs.getState().bump();
  return dir.name;
}

/** 目录中的一项（文件或子目录），找不到时返回 undefined。每一步都捕获异常：
 * 目录已删除、权限失效、名称含系统不支持的字符，均视为不存在，而非错误。 */
async function entry(dir: FileSystemDirectoryHandle, name: string, kind: "file" | "directory") {
  try {
    return kind === "file" ? await dir.getFileHandle(name) : await dir.getDirectoryHandle(name);
  } catch {
    return undefined;
  }
}

/** 列出目录中的文件名，不读取文件内容。
 *
 * 一段序列可能有数百帧，逐帧查询文件是否存在需要数百次系统调用；
 * 列一次目录后在内存中比对名称即可。无法列出（权限失效、目录已删除）时返回空集合。 */
export async function listNames(dir: FileSystemDirectoryHandle): Promise<Set<string>> {
  const out = new Set<string>();
  try {
    for await (const h of (dir as FileSystemDirectoryHandle & { values(): AsyncIterable<FileSystemHandle> }).values())
      if (h.kind === "file") out.add(h.name);
  } catch {
    return out;
  }
  return out;
}

/** 返回包含 `probe` 文件的授权目录。素材可能直接位于授权的那一层，也可能位于其下的子文件夹 `sub` 中。
 * 找不到时返回 null。 */
export async function dirHolding(sub: string, probe: string): Promise<FileSystemDirectoryHandle | null> {
  for (const dir of await readableDirs()) {
    const at = sub ? ((await entry(dir, sub, "directory")) as FileSystemDirectoryHandle | undefined) : dir;
    if (at && (await entry(at, probe, "file"))) return at;
    if (sub && (await entry(dir, probe, "file"))) return dir; // 子文件夹名称不匹配，但文件位于当前层
  }
  return null;
}
