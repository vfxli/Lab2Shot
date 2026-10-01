import { useEffect, useState } from "react";

/** 一个值稳定了 `ms` 毫秒之后才采用它：顶栏的「队列 · 计算中」「取消」和任务数角标用它——结果全在缓存里时，
 * 任务提交后不到一秒就算完，原样显示会闪一下（计算中 → 又没了）。变化撑不到 `ms` 的看不见，撑过了照常显示。 */
export function useSettled<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    if (Object.is(value, settled)) return;
    const t = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(t);
  }, [value, settled, ms]);
  return settled;
}
