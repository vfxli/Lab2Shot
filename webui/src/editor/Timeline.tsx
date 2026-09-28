import { useEffect, useRef, useState } from "react";
import { usePreferences, type LoopMode } from "../state/preferences";
import { useLook } from "../state/look";
import { framesShown, useRulerView, useViewLoads, useViewer } from "../state/viewer";
import { IconPause, IconPlay } from "../ui/icons";
import {
  advance, follow, frameX, fullView, gaps, nearest, panView, parseFrame, playRange, runs, setRange,
  tick, ticks, xFrame, zoomView, type MarkLayer, type Range, type View,
} from "../model/timelineMath";
import { onItsWay } from "../transfer/frames";
import { ZoomBar } from "../ui/ZoomBar";
import { useRefSize } from "../platform/size";
import { IconButton } from "../ui/Button";
import { MoreMenu } from "./TimelineMenu";
// 时间线的数据来源（帧、已计算的帧、标尺上的标记）在 timelineSource.ts 中，与绘制标尺无关
import { useSource } from "./timelineSource";
import { CookRange } from "./Chrome";

/** The timeline under both stages, one thin row: 倒放 / 播放, the current frame, the frame ruler, the 2D zoom,
 * 计算范围, the rate and 播放方式 (MoreMenu). The ruler is in the shot's own frame numbers (zoomed with the wheel, moved
 * by dragging with the middle or right button) and carries the data's frames, the playback range with its handles,
 * the frames already in the browser (the cache bar), the marks the displayed node's results carry and the frames
 * missing. Steps and jumps are keys only (← → ↑ ↓, editor/App.tsx). The arithmetic is in model/timelineMath.ts; the
 * loop mode belongs to state/preferences.ts (this browser's way of playing, not the graph's).
 *
 * The whole strip shows no hover tips (data-no-tips, platform/tips.ts); the 播放方式 menu it opens keeps its own. */

/** Plays while the store says so: moves the frame on inside the playback range at the shot's rate, as the loop mode
 * says, waiting for a frame not yet in the browser. Returns the frames actually shown per second while playing. */
function usePlayback(prefs: { mode: LoopMode }): number | null {
  const playing = useViewer((s) => s.playing);
  const [shownRate, setShownRate] = useState<number | null>(null);
  useEffect(() => {
    if (!playing) {
      setShownRate(null);
      return;
    }
    const frames0 = framesShown();
    const v0 = useViewer.getState();
    const r0 = playRange(frames0, useLook.getState().playback);
    if (r0 && prefs.mode === "once") {
      // played once to the end: playing again starts over, as in Houdini
      const list = frames0.filter((f) => f >= r0[0] && f <= r0[1]);
      if (v0.frame === (v0.playDir > 0 ? list.at(-1) : list[0])) useViewer.setState({ frame: v0.playDir > 0 ? list[0] : list.at(-1)! });
    }
    const start = performance.now() / 1000;
    const done = () => useViewLoads.getState().catchingUp && useViewLoads.setState({ catchingUp: false });
    let clock = { t0: start, done: 0 };
    let reported = start;
    const shown: number[] = [];
    let raf = 0;
    const loop = () => {
      const now = performance.now() / 1000;
      const v = useViewer.getState();
      const frames = framesShown();
      const range = playRange(frames, useLook.getState().playback);
      // 按 Nuke / Houdini 的方式播放：不跳帧，按顺序播放每一帧，边播放边缓存。因此第一遍慢于实时，
      // 但连续且完整填充缓存；第二遍起达到实时。
      // 下一帧尚未到达浏览器时原地等待（steps 0），并将时钟对齐到当前时刻，不累积欠帧；帧到达后继续播放。
      // 不能只依赖 `tick` 的非实时分支：该分支仍每拍前进一帧，会超前于取帧进度，每一遍只能取到零散的子集，
      // 多次往返也无法填满缓存。
      // 二维舞台只认可已解码的帧（`decoded`），三维舞台在没有该信息时使用 `loaded`（三维数据块解开即可绘制）。
      // 二维不能使用 `loaded`：它将本机文件的每一帧都视为已到达，本机序列会始终被判定为跟得上，解码跟不上时显示上一帧。
      const loadsNow = useViewLoads.getState();
      const loadedNow = loadsNow.decoded ?? loadsNow.loaded;
      let keepingUp = true;
      if (loadedNow && range) {
        const one = advance(frames, range, v.frame, v.playDir, prefs.mode, 1);
        // 与 Nuke 相同，全程只有一条规则：按顺序逐帧前进，不跳帧；目标帧未缓存时等待其到达
        // （因此第一遍较慢）；到达末尾后回到第一帧继续，同样遵循有缓存则实时、无缓存则等待，
        // 不区分前进与回绕。
        // 但不得等待没有被请求的帧，否则会无限等待（例如到达末尾需回到第一帧，而第一帧无人请求）。
        // 取帧顺序本身会随播放回绕到开头（transfer/frameWindow.ts order），此处是第二道保障。
        // 三维逐帧数据块不经过取帧账本，其加载状态由 useViewLoads.pending 提供（view/scene.ts pending）
        keepingUp = loadedNow.includes(one.frame) || !(onItsWay(one.frame) || loadsNow.pending?.includes(one.frame));
      }
      // 等待期间时钟同步暂停，否则等待期间会累积欠帧，帧到达后连续跳过多帧，重新造成跳帧
      const t = keepingUp ? tick(clock, now, v.fps) : { steps: 0, clock: { t0: now, done: 0 } };
      clock = t.clock;
      if (useViewLoads.getState().catchingUp !== !keepingUp) useViewLoads.setState({ catchingUp: !keepingUp });
      if (t.steps && range) {
        const a = advance(frames, range, v.frame, v.playDir, prefs.mode, t.steps);
        useViewer.setState({ frame: a.frame, playDir: a.dir, ...(a.stopped ? { playing: false } : {}) });
        shown.push(now);
      }
      while (shown.length && shown[0] < now - 1) shown.shift();
      if (now - reported > 0.25 && now - start > 0.5) {
        reported = now;
        setShownRate(shown.length / Math.min(1, now - start));
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      done();
    };
  }, [playing, prefs.mode]);
  return shownRate;
}

export function Timeline({ stage2d }: { stage2d: boolean }) {
  const frames = useViewer((s) => s.frames);
  const frame = useViewer((s) => s.frame);
  const playing = useViewer((s) => s.playing);
  const playDir = useViewer((s) => s.playDir);
  const fps = useViewer((s) => s.fps);
  const playback = useLook((s) => s.playback);
  const setPlayback = useLook((s) => s.setPlayback);
  const setFrame = useViewer((s) => s.setFrame);
  const play = useViewer((s) => s.play);
  const { layers } = useSource(); // 色带只表示能否实时播放，不画结果位于服务器上的帧
  const prefsMode = usePreferences((s) => s.playbackMode);
  const setPreferencesPlayback = usePreferences((s) => s.setPlayback);
  const prefs = { mode: prefsMode };
  const shownRate = usePlayback(prefs);
  const has = frames.length > 0;
  const range = playRange(frames, playback);

  const prefer = (p: Partial<{ mode: LoopMode }>) => setPreferencesPlayback(p);
  const setEnd = (end: 0 | 1, f: number) => range && setPlayback(setRange(frames, end ? [range[0], f] : [f, range[1]], end));
  const rate = Number(fps.toFixed(2));
  const slow = shownRate !== null && shownRate < fps * 0.9;

  return (
    <div className="timeline" data-no-tips>
      {/* 倒放与正放并列置于最左侧，与 DCC 软件一致 */}
      <IconButton aria-label="倒着播放" tone="ghost" size="sm" layout="tl-playpause" on={playing && playDir < 0} onClick={() => play(-1)} disabled={!has}>
        {playing && playDir < 0 ? <IconPause size={13} /> : <IconPlay size={13} back />}
      </IconButton>
      <IconButton aria-label="播放" tone="ghost" size="sm" layout="tl-playpause" on={playing && playDir > 0} onClick={() => play(1)} disabled={!has}>
        {playing && playDir > 0 ? <IconPause size={13} /> : <IconPlay size={13} />}
      </IconButton>
      <FrameField
        className="tl-current"
        value={has ? frame : null}
        label="当前帧"
        onCommit={(f) => setFrame(nearest(frames, f))}
      />
      <Ruler frames={frames} frame={frame} range={range} layers={layers} onFrame={setFrame} onEnd={setEnd} />
      <ZoomBar stage2d={stage2d} />
      <CookRange />
      {/* 帧率：未播放时显示镜头设定的帧率，播放时显示当前每秒实际绘制的帧数，与 Nuke 相同只显示一个数。
          宽度固定，开始播放时该格不变宽，整条时间条不发生位移。 */}
      <span className={`tl-fps tnum${slow ? " slow" : ""}`}>
        {`${shownRate !== null ? shownRate.toFixed(1) : rate} fps`}
      </span>
      <MoreMenu prefs={prefs} prefer={prefer} />
    </div>
  );
}

/** A frame number typed freely, taken on Enter or when the field is left (a frame the data does not have goes to the
 * nearest one); Escape, or what is not a whole number, puts back the value. */
function FrameField({ value, label, className, onCommit }: { value: number | null; label: string; className: string; onCommit: (f: number) => void }) {
  const shown = value === null ? "" : String(value);
  const [text, setText] = useState(shown);
  useEffect(() => setText(shown), [shown]);
  const commit = () => {
    const f = parseFrame(text);
    if (f === null || value === null) setText(shown);
    else if (f !== value) onCommit(f);
    else setText(shown);
  };
  return (
    <input
      className={`field num tl-field ${className}`}
      value={text}
      aria-label={label}
      placeholder="—"
      inputMode="numeric"
      spellCheck={false}
      disabled={value === null}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") e.currentTarget.blur();
        else if (e.key === "Escape") {
          setText(shown);
          const field = e.currentTarget; // React clears currentTarget once the handler returns: taken now, blurred next frame
          requestAnimationFrame(() => field.blur());
        }
      }}
    />
  );
}

interface RulerProps {
  frames: number[];
  frame: number;
  range: Range | null;
  layers: MarkLayer[];
  onFrame: (f: number) => void;
  onEnd: (end: 0 | 1, f: number) => void;
}

type Drag = { kind: "scrub" } | { kind: "pan"; x: number; view: View } | { kind: "end"; end: 0 | 1 };

const DENSE = 300; // more marks than this in view are drawn as runs

function Ruler({ frames, frame, range, layers, onFrame, onEnd }: RulerProps) {
  const loaded = useViewLoads((s) => s.loaded); // in the 3D view now (null: nothing there comes frame by frame)
  const stale = useViewLoads((s) => s.stale); // 主源显示的是上一次的结果：整条色带为土黄色
  const el = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const width = useRefSize(el).w;
  const bounds = frames.length ? fullView(frames[0], frames.at(-1)!) : null;
  const boundsKey = bounds ? `${bounds.start}:${bounds.end}` : "";
  const zoom = useRulerView((s) => s.zoom);
  const setZoom = useRulerView((s) => s.setZoom); // another shot: the whole of it again (the curve editor follows it)
  const view = bounds && zoom?.key === boundsKey ? zoom.view : bounds;
  const zoomed = view !== bounds;
  const setView = (v: View) => setZoom({ key: boundsKey, view: v });


  // playing or stepping past the edge of a zoomed ruler moves it along
  useEffect(() => {
    if (zoomed && view && bounds) {
      const next = follow(view, frame, bounds);
      if (next !== view) setView(next);
    }
  }, [frame]); // eslint-disable-line react-hooks/exhaustive-deps

  // the wheel zooms around the pointer (sideways: moves); a listener of its own, to keep the page from scrolling
  const latest = useRef({ view, bounds, width });
  latest.current = { view, bounds, width };
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      const { view: v, bounds: b, width: w } = latest.current;
      if (!v || !b || !w) return;
      e.preventDefault();
      const x = e.clientX - node.getBoundingClientRect().left;
      const sideways = Math.abs(e.deltaX) > Math.abs(e.deltaY) || e.shiftKey;
      if (sideways) setZoom({ key: `${b.start}:${b.end}`, view: panView(v, ((e.shiftKey ? e.deltaY : e.deltaX) / w) * (v.end - v.start), b) });
      else setZoom({ key: `${b.start}:${b.end}`, view: zoomView(v, xFrame(v, x, w), Math.exp(-e.deltaY * 0.0015), b) });
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, []);

  if (!view || !width) return <div ref={el} className="tl-ruler empty" />;

  const x = (f: number) => frameX(view, f, width);
  const cells = (a: number, b: number) => ({ left: x(a - 0.5), width: Math.max(1, x(b + 0.5) - x(a - 0.5)) });
  const localX = (e: React.PointerEvent) => e.clientX - el.current!.getBoundingClientRect().left;
  const at = (e: React.PointerEvent) => xFrame(view, localX(e), width);
  const t = ticks(view.start, view.end, width);
  const ppf = width / (view.end - view.start);
  const seen = (f: number) => f >= view.start - 1 && f <= view.end + 1;

  return (
    <div
      ref={el}
      className="tl-ruler"
      onContextMenu={(e) => e.preventDefault()}
      onDoubleClick={() => setZoom(null)}
      onPointerDown={(e) => {
        if (!frames.length) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        const grip = (e.target as HTMLElement).dataset.end;
        if (e.button === 1 || e.button === 2) drag.current = { kind: "pan", x: e.clientX, view };
        else if (grip !== undefined) drag.current = { kind: "end", end: grip === "1" ? 1 : 0 };
        else {
          drag.current = { kind: "scrub" };
          useViewer.setState({ scrubbing: true }); // 拖动过程中不请求帧，松开后才请求
          onFrame(nearest(frames, at(e)));
        }
        e.preventDefault(); // middle button: no page autoscroll
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d || !bounds) return;
        if (d.kind === "scrub") onFrame(nearest(frames, at(e)));
        else if (d.kind === "end") onEnd(d.end, Math.round(at(e)));
        else setView(panView(d.view, ((d.x - e.clientX) / width) * (d.view.end - d.view.start), bounds));
      }}
      onPointerUp={() => (drag.current = null, useViewer.setState({ scrubbing: false }))}
      onPointerCancel={() => (drag.current = null, useViewer.setState({ scrubbing: false }))}
    >
      {t.minor.map((f) => (
        <div key={`m${f}`} className="tl-tick" style={{ left: x(f) }} />
      ))}
      {t.major.map((f) => (
        <div key={f} className="tl-tick major" style={{ left: x(f) }}>
          <span className="tl-num tnum">{f}</span>
        </div>
      ))}
      <div className="tl-data" style={cells(frames[0], frames.at(-1)!)} />
      {range && <div className="tl-play" style={cells(range[0], range[1])} />}
      {gaps(frames).map(([a, b]) => (
        <div key={`g${a}`} className="tl-gap" style={cells(a, b)} />
      ))}
      {layers.map((l, i) => {
        const inView = l.frames.filter(seen);
        return inView.length > DENSE || ppf < 3
          ? runs(inView).map(([a, b]) => <div key={`${l.name}${a}`} className={`tl-mark-run k${i % 3}`} style={cells(a, b)} />)
          : inView.map((f) => <div key={`${l.name}${f}`} className={`tl-mark k${i % 3}${f === frame ? " on" : ""}`} style={{ left: x(f) }} />);
      })}
      {range && (
        <>
          <div className="tl-grip" data-end="0" style={{ left: x(range[0] - 0.5) }} />
          <div className="tl-grip out" data-end="1" style={{ left: x(range[1] + 0.5) }} />
        </>
      )}
      {/* 一条色带，两种颜色，只表示能否实时播放以及是否为最新结果：
          绿 = 已在浏览器中，播放无须再下载（账本中的 loaded：已解码、已持有字节、本机代理已在硬盘上。
          播放器等待的是 `decoded`，已持有字节的帧在播放到时仍需数毫秒解码，因此此处表示「无须再下载」而非「已解码」）；
          土黄 = 可实时播放，但显示的是上一次的结果（state/stale.ts）。不绘制结果是否位于服务器上。 */}
      {loaded &&
        runs(loaded).map(([a, b]) => (
          <div key={`l${a}`} className={`tl-cache${stale ? " stale" : ""}`} style={cells(a, b)} />
        ))}
      <div className={`tl-head${x(frame) < 20 ? " at-start" : x(frame) > width - 20 ? " at-end" : ""}`} style={{ left: x(frame) }}>
        <span className="tl-head-num tnum">{frame}</span>
      </div>
    </div>
  );
}
