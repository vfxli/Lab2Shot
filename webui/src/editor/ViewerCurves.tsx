import { useRef } from "react";
import type { CurvesData } from "../api";
import { describedFailure, useDescribed } from "../transfer/described";
import { CurvesView } from "./CurvesView";
import { TIMELINE_STRIP, usePreferences } from "../state/preferences";
import { Toggle } from "../ui/Button";
import type { ViewItem } from "../view/plan";
import { followDrag } from "../platform/drag";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** The curve editor under the view (part of editor/Viewer.tsx) and the one owner of it: the 「曲线」 toggle on the
 * control bar, the area docked under the stage (as tall as the user drags it; the stage gives up that room and nothing
 * covers the stage), and reading and drawing a curves result. */

/** 曲线: the curve editor under the stage shown or not (this browser remembers). */
export function CurveToggle() {
  const open = usePreferences((s) => s.timelineOpen);
  const setTimelineStrip = usePreferences((s) => s.setTimelineStrip);
  return (
    <Toggle hud on={open} onChange={(v) => setTimelineStrip({ open: v })}>
      {t("ui.timeline.curves")}
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
      <div className="curve-strip-grip" onPointerDown={grab} {...tipAttrs(tipOf("shortcut", t("ui.timeline.strip_grip_tip")))} />
      <Curves item={item} />
    </div>
  );
}

/** A curves result read and shown in the curve editor. */
export function Curves({ item }: { item: ViewItem }) {
  const data = useDescribed<CurvesData>("curves", [item.fp!])[0];
  const error = data ? null : describedFailure("curves", item.fp!);
  if (error) return <div className="empty">{t("ui.timeline.curves_failed", { error })}</div>;
  if (!data) return <div className="empty">{t("ui.timeline.reading_curves")}</div>;
  return <CurvesView data={data} title={item.label} />;
}
