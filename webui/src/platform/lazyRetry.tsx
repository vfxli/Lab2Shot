import { lazy, type ComponentProps, type ComponentType } from "react";

/** The page's one way to load a part on demand (the 3D stage, the rig-map editor, the editor and admin pages). React's
 * own lazy keeps a failed import forever: one network hiccup and the part stays broken until the page is reloaded. Here
 * a failed load is remembered, and retryLoads (the 重试 of the ErrorBoundary around it, ui/ErrorBoundary.tsx) makes the
 * next render import it again. A file of the page that cannot be fetched (most often: the page was built again while it
 * was open, so the file this page asks for is gone) fails as PageOutdated, which the ErrorBoundary answers with 刷新页面
 * instead of the browser's own words. */
const failed = new Set<() => void>();

/** A part of the page whose file could not be fetched: only a reload of the page (with its new files) helps. */
export class PageOutdated extends Error {}

// what the browsers say when a dynamic import cannot fetch its file (Chromium, Firefox, Safari) or Vite cannot preload it
const UNFETCHED = /dynamically imported module|Importing a module script failed|Unable to preload/i;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function lazyRetry<C extends ComponentType<any>>(load: () => Promise<{ default: C }>): ComponentType<ComponentProps<C>> {
  const make = () => lazy(() => load().catch((e) => {
    failed.add(reset);
    throw UNFETCHED.test(String((e as Error)?.message ?? e)) ? new PageOutdated(String((e as Error)?.message ?? e)) : e;
  }));
  let current = make();
  const reset = () => void (current = make());
  function Loaded(props: ComponentProps<C>) {
    const Part = current as ComponentType<ComponentProps<C>>;
    return <Part {...props} />;
  }
  return Loaded;
}

/** Every part whose load failed imports again on its next render. */
export function retryLoads(): void {
  for (const reset of failed) reset();
  failed.clear();
}
