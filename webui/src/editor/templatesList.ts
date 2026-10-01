import { useEffect, useState } from "react";
import { api, type TemplatesPage } from "../api";

// The page's one copy of the templates list: the templates, with their graphs (the card grid), and the tree they sit
// in, requested once per page. One promise, so two readers never request the same list twice.
let cache: Promise<TemplatesPage> | null = null;

/** After a template is saved, copied, deleted or moved, or the category tree changes: drops the cache, so the next read requests again. */
export function refreshTemplates(): void {
  cache = null;
}

/** `again`: read again whenever it changes (the templates sheet passes whether it is open, so every opening reads
 * again and a card just saved is there as soon as the sheet opens, the same rule as 「我的模板」). */
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
