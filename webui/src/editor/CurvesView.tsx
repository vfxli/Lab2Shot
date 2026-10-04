/** Owns how curves are drawn in the editor: the full curve editor (CurvesView) and the small per-frame curve of a
 * wire-driven parameter in the parameter panel (ParamCurve). The curve maths lives in model/curvesMath.ts. */
import { useEffect, useMemo, useRef, useState } from "react";
import type { CurvesData } from "../api";
import { useDescribed } from "../transfer/described";
import { beforePairs, fitRange, formatValue, mostMoving, pick, search, valueTicks, type PickMode } from "../model/curvesMath";
import { fullView, nearest, ticks, xFrame, zoomView } from "../model/timelineMath";
import { useCurveView, useRulerView, useViewer } from "../state/viewer";
import { useRefSize } from "../platform/size";
import { Button } from "../ui/Button";
import { startScrub } from "./scrub";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

const FIRST = 6; // channels drawn at first: the ones that move most

/** A curve editor, as Maya's Graph Editor and Houdini's channel list with its graph: the channels on the left
 * (search, a click picks one, Ctrl adds or takes away, Shift picks a run; all, none), each with its value at the
 * current frame; the picked ones drawn on the right against a value axis and a frame axis. The frame axis is the
 * timeline's own window (its wheel zoom and drag); clicking or dragging on the graph moves the current frame, the
 * wheel zooms the timeline's window with it. `title`: what the curves are. */
export function CurvesView({ data, title }: { data: CurvesData; title?: string }) {
  const frame = useViewer((s) => s.frame);
  const setFrame = useViewer((s) => s.setFrame);
  const shotFrames = useViewer((s) => s.frames);
  const zoom = useRulerView((s) => s.zoom);
  const setZoom = useRulerView((s) => s.setZoom);
  const original = useCurveView((s) => s.original);
  const setOriginal = useCurveView((s) => s.setOriginal);
  const names = data.names.join("\n");
  const { pair, hidden } = useMemo(() => beforePairs(data.names, data.before), [data.names, data.before]);
  const [picked, setPicked] = useState<{ names: string; set: Set<number>; anchor: number | null }>(() => ({ names, set: new Set(mostMoving(data.values, FIRST, hidden)), anchor: null }));
  const chosen = picked.names === names ? picked : { names, set: new Set(mostMoving(data.values, FIRST, hidden)), anchor: null };
  const [typed, setTyped] = useState("");
  const [hover, setHover] = useState<number | null>(null);
  const shown = useMemo(() => search(data.names, typed, hidden), [data.names, typed, hidden]);
  const k = data.frames.indexOf(frame);

  const click = (i: number, e: React.MouseEvent) => {
    const mode: PickMode = e.shiftKey ? "range" : e.ctrlKey || e.metaKey ? "toggle" : "only";
    setPicked({ names, set: pick(chosen.set, shown, i, mode, chosen.anchor), anchor: e.shiftKey ? chosen.anchor ?? i : i });
  };
  const setAll = (set: Set<number>) => setPicked({ names, set, anchor: null });

  // the timeline's window over the shot (its frames, else the curves' own)
  const frames = shotFrames.length ? shotFrames : data.frames;
  const bounds = frames.length ? fullView(frames[0], frames.at(-1)!) : fullView(0, 1);
  const boundsKey = `${bounds.start}:${bounds.end}`;
  const view = zoom?.key === boundsKey ? zoom.view : bounds;

  return (
    <div className="curves">
      <div className="curves-list">
        <div className="curves-list-head">
          <div className="curves-head-row">
            <span className="curves-title">{title ?? t("ui.timeline.curves")}</span>
            <span className="curves-count tnum">
              {chosen.set.size} / {data.names.length - hidden.size}
            </span>
          </div>
          <div className="curves-head-row curves-list-tools">
            <input className="field curves-search" type="search" value={typed} onChange={(e) => setTyped(e.target.value)} placeholder={t("ui.common.search")} />
            <Button size="sm" tone="ghost" onClick={() => setAll(new Set([...chosen.set, ...shown]))}>
              {t("ui.timeline.all")}
            </Button>
            <Button size="sm" tone="ghost" onClick={() => setAll(new Set())}>
              {t("ui.common.none")}
            </Button>
            <Button size="sm" tone="ghost" onClick={() => setAll(new Set(mostMoving(data.values, FIRST, hidden)))}>
              {t("ui.timeline.most_moving")}
            </Button>
            <Button
              size="sm"
              tone="ghost"
              on={original && pair.size > 0}
              disabled={!pair.size}
              tip={pair.size ? undefined : tipOf("disabled", t("ui.timeline.no_original"))}
              onClick={() => setOriginal(!original)}
            >
              {t("ui.timeline.show_original")}
            </Button>
          </div>
        </div>
        <div className="curves-items" role="listbox" aria-multiselectable="true" aria-label={t("ui.timeline.channels")}>
          {shown.map((i) => (
            <div
              key={i}
              role="option"
              aria-selected={chosen.set.has(i)}
              className={`curves-item${chosen.set.has(i) ? " on" : ""}`}
              onMouseDown={(e) => e.shiftKey && e.preventDefault()} // Shift: no text selection
              onClick={(e) => click(i, e)}
              onMouseEnter={() => setHover(i)}
              onMouseLeave={() => setHover(null)}
            >
              <i style={{ background: color(i) }} />
              <span className="curves-name" data-user-data {...tipAttrs(tipOf("truncated", data.names[i]))}>{data.names[i]}</span>
              <span className="curves-value tnum">{k >= 0 ? formatValue(data.values[i][k]) : ""}</span>
            </div>
          ))}
          {!shown.length && <div className="curves-none">{t("ui.timeline.no_channel_matching", { query: typed })}</div>}
        </div>
      </div>
      <Graph data={data} picked={chosen.set} hover={hover} view={view} frame={frame} frames={frames}
        before={original ? pair : undefined}
        onFrame={setFrame} onZoom={(v) => setZoom(v ? { key: boundsKey, view: v } : null)} bounds={bounds} />
    </div>
  );
}

const PAD = { left: 52, right: 10, top: 8, bottom: 20 };

function Graph({ data, picked, hover, view, frame, frames, bounds, before, onFrame, onZoom }: {
  data: CurvesData; picked: Set<number>; hover: number | null; view: { start: number; end: number }; frame: number; frames: number[];
  bounds: { start: number; end: number }; before?: ReadonlyMap<number, number>;  // absent while 显示原始 is off
  onFrame: (f: number) => void; onZoom: (v: { start: number; end: number } | null) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const size = useRefSize(el);
  // the wheel zooms the timeline's window around the pointer; a listener of its own, to keep the page from scrolling
  const latest = useRef({ view, bounds, w: size.w, onZoom });
  latest.current = { view, bounds, w: size.w, onZoom };
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      const { view: v, bounds: b, w, onZoom: zoomTo } = latest.current;
      const pw = w - PAD.left - PAD.right;
      if (pw <= 0) return;
      e.preventDefault();
      zoomTo(zoomView(v, xFrame(v, e.clientX - node.getBoundingClientRect().left - PAD.left, pw), Math.exp(-e.deltaY * 0.0015), b));
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, []);

  const pw = size.w - PAD.left - PAD.right;
  const ph = size.h - PAD.top - PAD.bottom;
  const drawn = [...picked].sort((a, b) => a - b);
  // each picked channel's 原始, drawn under it in the same colour: it belongs to the channel, so it is never listed
  // or picked on its own
  const originals = drawn.map((i) => [i, before?.get(i)] as const).filter((p): p is readonly [number, number] => p[1] !== undefined);
  const [lo, hi] = fitRange(data.values, data.frames, [...drawn, ...originals.map(([, j]) => j)], view.start, view.end);
  const x = (f: number) => PAD.left + ((f - view.start) / (view.end - view.start)) * pw;
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * ph;
  // 换帧与时间线同一个做法（editor/scrub.ts：拖动期间不取帧，followDrag 收尾）
  const frameAt = (cx: number) => (frames.length ? nearest(frames, xFrame(view, cx - el.current!.getBoundingClientRect().left - PAD.left, pw)) : null);
  const k = data.frames.indexOf(frame);
  // the frames in view, with one on each side so a curve runs to the edge
  const first = Math.max(0, data.frames.findIndex((f) => f >= view.start) - 1);
  const after = data.frames.findIndex((f) => f > view.end);
  const last = after < 0 ? data.frames.length - 1 : after;

  return (
    <div
      ref={el}
      className="curves-graph"
      onPointerDown={(e) => e.button === 0 && startScrub(e, frameAt, onFrame)}
      onDoubleClick={() => onZoom(null)}
    >
      {pw > 0 && ph > 0 && (
        <svg width={size.w} height={size.h} role="img" aria-label={t("ui.timeline.graph")}>
          <rect x={PAD.left} y={PAD.top} width={pw} height={ph} className="curves-plot" />
          {(() => {
            const vt = valueTicks(lo, hi, Math.max(3, Math.floor(ph / 26)));
            return vt.values.map((v) => (
              <g key={`v${v}`}>
                <line x1={PAD.left} x2={PAD.left + pw} y1={y(v)} y2={y(v)} className={v === 0 ? "curves-grid zero" : "curves-grid"} />
                <text x={PAD.left - 6} y={y(v) + 3.5} className="curves-axis" textAnchor="end">
                  {formatValue(v, vt.step)}
                </text>
              </g>
            ));
          })()}
          {(() => {
            const t = ticks(view.start, view.end, pw);
            return (
              <>
                {t.minor.map((f) => (
                  <line key={`m${f}`} x1={x(f)} x2={x(f)} y1={PAD.top + ph} y2={PAD.top + ph + 3} className="curves-tick" />
                ))}
                {t.major.map((f) => (
                  <g key={`f${f}`}>
                    <line x1={x(f)} x2={x(f)} y1={PAD.top} y2={PAD.top + ph} className="curves-grid" />
                    <text x={x(f)} y={PAD.top + ph + 14} className="curves-axis" textAnchor="middle">
                      {f}
                    </text>
                  </g>
                ))}
              </>
            );
          })()}
          <svg x={PAD.left} y={PAD.top} width={pw} height={ph} overflow="hidden">
            <g transform={`translate(${-PAD.left} ${-PAD.top})`}>
              {originals.map(([i, j]) => (
                <polyline
                  key={`b${j}`}
                  fill="none"
                  stroke={color(i)}
                  strokeWidth={1.4}
                  strokeDasharray="4 3"
                  strokeOpacity={hover === null || hover === i ? 0.45 : 0.15}
                  points={data.values[j].slice(first, last + 1).map((v, n) => `${x(data.frames[first + n]).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}
                />
              ))}
              {drawn.map((i) => (
                <polyline
                  key={i}
                  fill="none"
                  stroke={color(i)}
                  strokeWidth={hover === i ? 2.6 : 1.4}
                  strokeOpacity={hover === null || hover === i ? 1 : 0.3}
                  points={data.values[i].slice(first, last + 1).map((v, n) => `${x(data.frames[first + n]).toFixed(1)},${y(v).toFixed(1)}`).join(" ")}
                />
              ))}
              {k >= 0 && drawn.map((i) => <circle key={`d${i}`} cx={x(frame)} cy={y(data.values[i][k])} r={2.6} fill={color(i)} />)}
            </g>
          </svg>
          {frame >= view.start && frame <= view.end && <line x1={x(frame)} x2={x(frame)} y1={PAD.top} y2={PAD.top + ph} className="curves-cursor" />}
        </svg>
      )}
      {!drawn.length && <div className="curves-empty">{t("ui.timeline.pick_channels")}</div>}
    </div>
  );
}

const color = (i: number) => `hsl(${(i * 137.5) % 360} 75% 62%)`;

// ------------------------------------------------------------------ curves in the parameter panel

const SPARK = { w: 280, h: 76, pad: 4 };

/** The curve of a parameter driven by a wire that carries a value per frame (a zoom lens's Focal Length, a per-frame switch):
 * a thin line over a faint grid, a vertical line at the current frame and a dot on it, with that frame's value written
 * beside the dot and the frame number in small type under it, and no other words on the plot. A wire
 * carrying one value draws nothing: the row already states the number.
 *
 * It follows playback, because it reads 当前帧 from the viewer like everything else on the stage. */
export function ParamCurve({ fp, title }: { fp: string; title: string }) {
  const frame = useViewer((s) => s.frame);
  const data = useDescribed<CurvesData>("curves", [fp])[0];
  if (!data) return null;
  const { frames, values, names } = data;
  if (!frames.length || !values.length) return null;
  const first = frames[0];
  const last = frames.at(-1)!;
  const span = Math.max(1, last - first);
  const drawn = values.map((_, i) => i);
  const [lo, hi] = fitRange(values, frames, drawn, first, last);
  const x = (f: number) => SPARK.pad + ((f - first) / span) * (SPARK.w - 2 * SPARK.pad);
  const y = (v: number) => SPARK.pad + (1 - (v - lo) / (hi - lo || 1)) * (SPARK.h - 2 * SPARK.pad);
  const k = frames.indexOf(nearest(frames, frame)); // the curve may not have every frame of the shot
  const at = frames[k];
  const now = drawn.map((i) => formatValue(values[i][k], (hi - lo) / 100 || 0.01)).join(" · ");
  const left = x(at) < SPARK.w * 0.6;
  return (
    <div className="param-curve">
      <span className="param-curve-name">{title}</span>
      <svg viewBox={`0 0 ${SPARK.w} ${SPARK.h}`} role="img" aria-label={t("ui.timeline.curve_of", { name: title })} preserveAspectRatio="none">
        {[0.25, 0.5, 0.75].map((p) => (
          <line key={p} className="curves-grid" x1={0} x2={SPARK.w} y1={SPARK.pad + p * (SPARK.h - 2 * SPARK.pad)} y2={SPARK.pad + p * (SPARK.h - 2 * SPARK.pad)} vectorEffect="non-scaling-stroke" />
        ))}
        {drawn.map((i) => (
          <polyline key={i} className="param-curve-line" fill="none" vectorEffect="non-scaling-stroke"
            points={frames.map((f, n) => `${x(f).toFixed(1)},${y(values[i][n]).toFixed(1)}`).join(" ")} />
        ))}
        <line className="curves-cursor" x1={x(at)} x2={x(at)} y1={0} y2={SPARK.h} vectorEffect="non-scaling-stroke" />
        {drawn.map((i) => (
          <circle key={`d${i}`} className="param-curve-dot" cx={x(at)} cy={y(values[i][k])} r={2.5} />
        ))}
        <text className="param-curve-now" x={left ? x(at) + 6 : x(at) - 6} y={Math.min(SPARK.h - 16, Math.max(12, y(values[0][k]) - 8))} textAnchor={left ? "start" : "end"}>
          {now}
        </text>
        <text className="param-curve-frame" x={left ? x(at) + 6 : x(at) - 6} y={Math.min(SPARK.h - 5, Math.max(24, y(values[0][k]) + 4))} textAnchor={left ? "start" : "end"}>
          {at}
        </text>
      </svg>
      <span className="param-curve-names" data-user-data>
        {names.join(" · ")}
      </span>
    </div>
  );
}

