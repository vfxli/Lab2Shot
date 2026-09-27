import { useEffect, useState } from "react";
import { api, type TemplatesPage } from "../api";

// Requested once per page: the templates, with their graphs (the card grid), and the tree they sit in. One promise, so two
// readers never request the same list twice.
let cache: Promise<TemplatesPage> | null = null;

/** 保存、复制、删除、移动模板或修改分类树后：使该缓存失效，下一次读取时重新请求。 */
export function refreshTemplates(): void {
  cache = null;
}

/** `again`：值变化时重新读取（模板面板传入其是否打开，每次打开时重新读取，因此刚保存的卡片打开即可见，
 * 与「我的模板」规则相同）。 */
export function useTemplates(again: unknown = null): TemplatesPage | null {
  const [got, set] = useState<TemplatesPage | null>(null);
  useEffect(() => {
    let live = true;
    (cache ??= api.templates()).then(
      (page) => live && set(page),
      () => (cache = null),
    );
    return () => {
      live = false;
    };
  }, [again]);
  return got;
}
