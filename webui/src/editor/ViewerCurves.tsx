import { useEffect, useRef, useState } from "react";
import { api, type CurvesData } from "../api";
import { CurvesView } from "./CurvesView";
import { TIMELINE_STRIP, usePreferences } from "../state/preferences";
import { Toggle } from "../ui/Button";
import type { ViewItem } from "../view/plan";
import { followDrag } from "../platform/drag";

/** 视图下方的曲线编辑器（editor/Viewer.tsx 的组成部分）：控制栏上的「曲线」开关、停靠在舞台下方的区域
 * （高度由使用者拖动决定，舞台让出该区域，不覆盖舞台），以及读取并绘制曲线结果。 */

/** 曲线: the curve editor under the stage shown or not (this browser remembers). */
export function CurveToggle() {
  const open = usePreferences((s) => s.timelineOpen);
  const setTimelineStrip = usePreferences((s) => s.setTimelineStrip);
  return (
    <Toggle hud on={open} onChange={(v) => setTimelineStrip({ open: v })}>
      曲线
    </Toggle>
  );
}

/** The curve editor docked under the stage, as tall as the user dragged it (the stage gives up that room: nothing is
 * drawn over it). */
export function CurveStrip({ item }: { item: ViewItem }) {
  const open = usePreferences((s) => s.timelineOpen);
  const height = usePreferences((s) => s.timelineHeight);
  const setTimelineStrip = usePreferences((s) => s.setTimelineStrip);
  const el = useRef<HTMLDivElement>(null);
  if (!open) return null;
  const grab = (e: React.PointerEvent) => {
    e.preventDefault();
    const y0 = e.clientY;
    const h0 = el.current!.offsetHeight;
    const most = () => Math.max(TIMELINE_STRIP.min, (el.current?.parentElement?.clientHeight ?? 600) * 0.7);
    followDrag((m) => setTimelineStrip({ height: Math.min(most(), h0 + y0 - m.clientY) }), () => undefined);
  };
  return (
    <div ref={el} className="curve-strip" style={{ height }}>
      <div className="curve-strip-grip" onPointerDown={grab} data-tip="上下拖动：改曲线编辑器的高度" />
      <Curves item={item} />
    </div>
  );
}

/** A curves result read and shown in the curve editor. */
export function Curves({ item }: { item: ViewItem }) {
  const [data, setData] = useState<{ fp: string; value: CurvesData } | { fp: string; error: string } | null>(null);
  useEffect(() => {
    let alive = true;
    api.curves(item.fp!).then(
      (value) => alive && setData({ fp: item.fp!, value }),
      (e) => alive && setData({ fp: item.fp!, error: e instanceof Error ? e.message : String(e) }),
    );
    return () => {
      alive = false;
    };
  }, [item.fp]);
  if (data?.fp !== item.fp) return <div className="empty">读取曲线…</div>;
  if ("error" in data) return <div className="empty">曲线读不出来：{data.error}</div>;
  return <CurvesView data={data.value} title={item.label} />;
}
