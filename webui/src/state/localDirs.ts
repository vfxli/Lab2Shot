import { create } from "zustand";

/** 用户首选项：用户授权过哪些本机目录，改过几次（`files/localDirs.ts` 里存的是句柄本身）。
 *
 * 只存一个数：句柄存在 IndexedDB 里（跨标签页、跨刷新都在），那才是那份数据；这里存的是「它变过没有」。
 * 视图问「这一份在他机器上有没有原件」时，「他授权过哪些目录」是这一问的输入之一，所以它一变，
 * 那一问就要重新问一遍（`view/useOriginals.ts` 的 `Ask.key`）；不然用户在「文件」菜单里指了一个目录后，
 * 画面会一直停在服务器那份代理上，非得刷新才对。
 *
 * 不存名字、不存句柄：那会变成第二份真相。名字要显示的时候现去 IndexedDB 读。 */
interface LocalDirs {
  version: number; // 授权过的目录改过几次（只用来当「重新问一遍」的由头）
  bump: () => void;
}

export const useLocalDirs = create<LocalDirs>((set) => ({
  version: 0,
  bump: () => set((s) => ({ version: s.version + 1 })),
}));
