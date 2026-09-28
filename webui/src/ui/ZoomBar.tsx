import "./zoombar.css";
import { useEffect, useState } from "react";
import { useView2D } from "../state/viewer";
import { Button } from "./Button";

/** 适应 / 1:1 / the current zoom%, centred under the timeline (in the empty space of its control row). Shown whenever
 * the viewer is on the 2D stage (Timeline mounts it), whatever draws it; the stage shown applies the asks
 * (state/view2d.ts useView2DNav) and the zoom comes back from the viewer's one 2D view. Greyed while there is no
 * picture to zoom: on the 3D stage (unless it looks through a camera), or before a picture is drawn. */
export function ZoomBar({ stage2d = true }: { stage2d?: boolean }) {
  const zoomPercent = useView2D((s) => s.zoomPercent);
  const navigable = useView2D((s) => s.navigable);
  const fit = useView2D((s) => s.fit);
  const one = useView2D((s) => s.one);
  const goTo = useView2D((s) => s.goTo);
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(String(zoomPercent));

  useEffect(() => {
    if (!editing) setText(String(zoomPercent));
  }, [zoomPercent, editing]);

  const commit = () => {
    const n = Number(text);
    if (Number.isFinite(n) && n > 0) goTo(n);
    setEditing(false);
  };

  const on = stage2d && navigable;
  return (
    <div className="tl-group tl-zoom">
      <Button tone="ghost" size="sm" onClick={fit} disabled={!on}>
        适应
      </Button>
      <Button tone="ghost" size="sm" onClick={one} disabled={!on}>
        1:1
      </Button>
      {editing && on ? (
        <input
          className="field num tl-field tl-zoom-field"
          autoFocus
          value={text}
          aria-label="缩放百分比"
          inputMode="decimal"
          spellCheck={false}
          onChange={(e) => setText(e.target.value)}
          onFocus={(e) => e.currentTarget.select()}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
            else if (e.key === "Escape") {
              setText(String(zoomPercent));
              requestAnimationFrame(() => e.currentTarget?.blur());
            }
          }}
        />
      ) : (
        <Button tone="ghost" size="sm" layout="tl-zoom-pct tnum" onClick={() => setEditing(true)} disabled={!on}>
          {on ? `${zoomPercent}%` : "—"}
        </Button>
      )}
    </div>
  );
}
