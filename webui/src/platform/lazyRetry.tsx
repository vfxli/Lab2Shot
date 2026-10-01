import { lazy, type ComponentProps, type ComponentType } from "react";

/** The page's one way to load a part on demand (the 3D stage, the rig-map editor, the editor and admin pages). React's
 * own lazy keeps a failed import forever: one network hiccup and the part stays broken until the page is reloaded. Here
 * a failed load is remembered, and retryLoads (the 重试 of the ErrorBoundary around it, ui/ErrorBoundary.tsx) makes the
 * next render import it again. */
const failed = new Set<() => void>();

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function lazyRetry<C extends ComponentType<any>>(load: () => Promise<{ default: C }>): ComponentType<ComponentProps<C>> {
  const make = () => lazy(() => load().catch((e) => {
    failed.add(reset);
    throw e;
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
