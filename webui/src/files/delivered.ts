import type { Delivery } from "../api/deliveries";
import { kept, readable } from "./handles";
import { fileAt, findFile, readableDirs } from "./localDirs";

/** 一次交付在用户本机上的副本：按包内相对路径（如 `深度/depth.1001.exr`）返回对应文件。
 * 计算结束后数据已在用户本机，预览「输出」节点时直接解析本地文件，不经服务器。
 *
 * 查找顺序为先交付目标位置、后授权目录：
 * 「输出」节点的「保存到」是用户已选定的位置，句柄存于 IndexedDB
 * （`editor/FileParam.tsx` 的 `DeliverParam` → `files/handles.ts keep`），
 * 交付也已写入该处（`files/deliver.ts`），因此通常无需新的授权。
 * 授权目录用于另一种情况：结果从队列下载到浏览器的下载目录，而浏览器不向页面暴露该目录的位置。
 *
 * 任一路径读取失败均视为「不存在」，回落到服务器代理，不作为错误处理：不报错、不提示、
 * 不在视图上显示通知控件。 */
export interface LocalPackage {
  fileAt(rel: string): Promise<File | null>;
  where: string; // 仅用于开发日志，不面向用户
}

/** 包内路径在用户本机上的对应路径：逐项处理中的每个交付各占一个子文件夹
 * （`files/transfer.ts saveDelivery` 的 `into`）。 */
const under = (d: Delivery, rel: string) => (d.item ? `${d.name}/${rel}` : rel);

/** 浏览器下载该交付时使用的包名，与 `files/transfer.ts downloadDelivery` 写入的名称一致。 */
const downloadedName = (d: Delivery) => (d.mode === "folder" ? `${d.name}.tar` : d.name);

const isTar = (name: string) => name.toLowerCase().endsWith(".tar");

/** 本机 tar 包：先索引一遍文件头，之后按字节区间切出各成员（`files/untar.ts indexTar`）。
 *
 * 解包模块按需动态加载（`import(...)`，原因同 `files/transfer.ts`）：打开编辑器时不加载，
 * 仅在需要从本机 tar 中读取帧时加载（首屏体积预算见 `tests/test_transfer.py`）。 */
async function fromTar(file: File, where: string): Promise<LocalPackage | null> {
  const { indexTar } = await import("./untar");
  const index = await indexTar(file).catch(() => new Map<string, { at: number; size: number }>());
  if (!index.size) return null;
  return {
    where,
    fileAt: async (rel) => {
      const one = index.get(rel);
      if (!one) return null;
      const part = file.slice(one.at, one.at + one.size);
      return new File([part], rel.split("/").pop() ?? rel);
    },
  };
}

/** 返回一次交付在用户本机上的副本；找不到时返回 null，由调用方继续使用服务器代理。
 * `saveHandle`：「输出」节点「保存到」句柄的键（`GNode.data.saveTo.handle`）。 */
export async function localPackage(d: Delivery, saveHandle?: string): Promise<LocalPackage | null> {
  // ① 交付目标位置：用户为该「输出」选定的文件夹或包。刷新后权限可能需要重新确认；
  //    无法确认时继续后续查找，不弹出对话框。
  const h = await kept<FileSystemHandle>(saveHandle).catch(() => undefined);
  if (h && (await readable(h))) {
    if (h.kind === "directory") {
      const dir = h as FileSystemDirectoryHandle;
      return { where: `保存到 ${dir.name}`, fileAt: (rel) => fileAt(dir, under(d, rel)) };
    }
    if (isTar(h.name)) {
      const file = await (h as FileSystemFileHandle).getFile().catch(() => null);
      const pack = file && (await fromTar(file, `保存到 ${h.name}`));
      if (pack) return pack;
    }
    // tar.gz 不走此路径：gzip 必须从头解压才能定位成员（见 `files/untar.ts indexTar` 的注释）。
  }
  // ② 已授权的本机目录：目录即交付文件夹本身，或其中包含下载的包。
  const dirs = await readableDirs();
  if (!dirs.length) return null;
  const name = downloadedName(d);
  if (isTar(name)) {
    const file = await findFile(name);
    const pack = file && (await fromTar(file, name));
    if (pack) return pack;
  }
  return { where: "本机目录", fileAt: (rel) => findFile(under(d, rel)) };
}
