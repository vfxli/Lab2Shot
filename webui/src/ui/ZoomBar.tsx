import "./zoombar.css";
import { useState } from "react";
import { Num } from "./controls";
import { useView2D } from "../state/viewer";
import { Button } from "./Button";
import { t } from "../i18n/t";

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

  const on = stage2d && navigable;
  return (
    <div className="tl-group tl-zoom">
      <Button tone="ghost" size="sm" onClick={fit} disabled={!on}>
        {t("ui.view.fit")}
      </Button>
      <Button tone="ghost" size="sm" onClick={one} disabled={!on}>
        1:1
      </Button>
      {editing && on ? (
        <Num className="tl-field tl-zoom-field" autoFocus label={t("ui.view.zoom_percent")} value={zoomPercent} min={0} openMin
          onChange={goTo} onDone={() => setEditing(false)} />
      ) : (
        <Button tone="ghost" size="sm" layout="tl-zoom-pct tnum" onClick={() => setEditing(true)} disabled={!on}>
          {on ? `${zoomPercent}%` : "—"}
        </Button>
      )}
    </div>
  );
}
