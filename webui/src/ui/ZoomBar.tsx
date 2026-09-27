import "./zoombar.css";
import { useEffect, useState } from "react";
import { useView2D } from "../state/viewer";
import { Button } from "./Button";

const NOTHING = "这里没有画面可以缩放：先让节点算出结果，或者显示一个有画面的节点";
const IN_3D = "现在看的是三维场景，不是二维画面：三维里用鼠标转视角和推拉。切到 2D，或者让视图透过一台相机看，这几样就能用";

/** 适应 / 1:1 / the current zoom%, centred under the timeline (in the empty space of its control row). Shown whenever
 * the viewer is on the 2D stage (Timeline mounts it), whatever draws it; the stage shown applies the asks
 * (view2dState.ts's useView2DNav) and the zoom comes back from the viewer's one 2D view. */
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
  const why = !stage2d ? IN_3D : navigable ? undefined : NOTHING;
  return (
    <div className="tl-group tl-zoom" data-tip={why}>
      <Button tip={why ?? "适应整幅画面 · F"} tone="ghost" size="sm" onClick={fit} disabled={!on}>
        适应
      </Button>
      <Button tip={why ?? "1:1 显示：一个画面像素对一个屏幕像素"} tone="ghost" size="sm" onClick={one} disabled={!on}>
        1:1
      </Button>
      {editing && on ? (
        <input
          className="field num tl-field tl-zoom-field"
          autoFocus
          value={text}
          aria-label="缩放百分比"
          data-tip="缩放百分比：输入数字，回车确定，Esc 放弃"
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
        <Button tip={why ?? "点击输入缩放百分比"} tone="ghost" size="sm" layout="tl-zoom-pct tnum" onClick={() => setEditing(true)} disabled={!on}>
          {on ? `${zoomPercent}%` : "—"}
        </Button>
      )}
    </div>
  );
}
