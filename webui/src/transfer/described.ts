/** 包自带的整份数据（包说明、人物框、跟踪点、曲线）：组件读它们只有 `useDescribed` 这一个钩子，React 之外用 `describedOf`。
 *
 * 键和地址带包的代次（transfer/frameKey.ts describedKey）：重算后换键，组件跟着代次重读，不会拿旧代次的包说明
 * 去建新代次的帧源；重算前发出的请求后回来，写在旧键下，没人再读。取、在途去重、失败退避都在 transfer/frameStore.ts。
 * 正在读的键登记为 pin：显示中的包说明不被淘汰（淘汰了组件也不会知道要重取）。 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { cache } from "../platform/cache";
import { useGens } from "./gens";
import { describedKey, describedSlot, describedUrl, type Described } from "./frameKey";
import { described, describedNow, lastFailure, whenAskable } from "./frameStore";

/** 一份数据（Promise）：已在缓存直接给。 */
export const describedOf = <V>(kind: Described, fp: string): Promise<V> => described<V>(describedSlot(kind, fp), describedUrl(kind, fp));

/** 同上，已在缓存里的（同步）。 */
export const describedHeld = <V>(kind: Described, fp: string): V | null => describedNow<V>(describedKey(kind, fp)) ?? null;

// 正在被组件读的键（按次数）：登记为 pin
const reading = new Map<string, number>();
const readingKeys = (): ReadonlySet<string> => new Set(reading.keys());
cache.pin(readingKeys);

/** 这份数据最近一次读失败的原因（读到了或没读过为 null）：配合 `useDescribed`（失败时它会重绘）。 */
export function describedFailure(kind: Described, fp: string): string | null {
  const e = lastFailure(describedKey(kind, fp));
  return e === undefined ? null : e instanceof Error ? e.message : String(e);
}

/** 这几个包的这一种数据，与 `fps` 一一对应（还没到的为 null）。内容不变时返回同一个数组（可作依赖）。
 * 读失败的按 frameStore 的退避到期后再读，组件挂着就一直会补上。 */
export function useDescribed<V>(kind: Described, fps: readonly string[]): (V | null)[] {
  const gens = useGens((s) => fps.map((fp) => s.gens[fp] ?? "").join("|"));
  const keys = fps.map((fp) => describedKey(kind, fp));
  const joined = keys.join("\n");
  const [, failedOnce] = useState(0); // 读失败时重绘一次：要说原因的一方（describedFailure）读得到

  useEffect(() => {
    if (!joined) return;
    let alive = true;
    const stops: (() => void)[] = [];
    const want = fps.map((fp, i) => ({ fp, key: keys[i] }));
    for (const { key } of want) reading.set(key, (reading.get(key) ?? 0) + 1);
    cache.repin();
    const ask = (fp: string, key: string): void =>
      void described({ key, group: kind }, describedUrl(kind, fp)).catch(() => {
        if (!alive) return;
        failedOnce((n) => n + 1);
        stops.push(whenAskable(key, () => alive && ask(fp, key)));
      });
    want.forEach(({ fp, key }) => ask(fp, key));
    return () => {
      alive = false;
      stops.forEach((stop) => stop());
      for (const { key } of want) {
        const n = (reading.get(key) ?? 1) - 1;
        if (n > 0) reading.set(key, n);
        else reading.delete(key);
      }
    };
  }, [kind, joined, gens]); // eslint-disable-line react-hooks/exhaustive-deps -- keys 由 joined 表示

  const last = useRef<{ joined: string; values: (V | null)[] } | null>(null);
  return useSyncExternalStore(
    useCallback((f: () => void) => cache.watch([kind], f), [kind]),
    () => {
      const values = joined ? joined.split("\n").map((k) => describedNow<V>(k) ?? null) : [];
      const was = last.current;
      if (was && was.joined === joined && was.values.length === values.length && was.values.every((v, i) => v === values[i])) return was.values;
      last.current = { joined, values };
      return values;
    },
  );
}
