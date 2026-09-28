import { useRef, useState } from "react";
import { HUD_SEG } from "../ui/Button";
import { useDismiss } from "../platform/dismiss";
import { useViewCamera, VIEW_NAMES, type ViewName } from "../state/viewer";

/** 视角：透视 / 顶 / 前 / 侧，或透过场景中的某台相机观看；以及框显全部 / 框显选中。
 *
 * 位于画面上方的工具栏中，而非画面左下角：视角与框显属于视图工具，与 2D / 3D、显示种类、视图设置位于同一行。
 *
 * 此工具栏须在不加载 three.js 的情况下绘制（three.js 只随三维舞台按需加载，见 editor/Viewer.tsx 的 Stage3D），
 * 因此不引用三维舞台的任何模块：所需的两项信息（场景中的相机列表与当前选中项）由三维舞台计算后写入 state/viewTools.ts。 */
const VIEW_TIPS: Record<ViewName, string> = {
  persp: "透视：自由转动的视角",
  top: "顶视图：从上往下看，正交；左键平移",
  front: "前视图：沿 -Z 看，正交；左键平移",
  side: "侧视图：从右往左看，正交；左键平移",
};

/** The view menu (Houdini's viewport camera menu): 透视 / 顶 / 前 / 侧 and every camera of the scene by its place in
 * the hierarchy; and framing. */
export function ViewButtons() {
  const view = useViewCamera((s) => s.view);
  const setView = useViewCamera((s) => s.setView);
  const setLook = useViewCamera((s) => s.setLook);
  const frame = useViewCamera((s) => s.frame);
  // 场景中的相机列表与当前选中项：由三维舞台计算后写入 state（viewTools.ts），此处只负责绘制
  const cameras = useViewCamera((s) => s.cameras);
  const selected = useViewCamera((s) => s.selected);
  const look = useViewCamera((s) => s.look);
  const [open, setOpen] = useState(false);
  const anchor = useRef<HTMLDivElement>(null);
  useDismiss(open, anchor, () => setOpen(false));
  const current = cameras.find((c) => c.key === look);
  const label = current ? current.label.split("/").pop() || current.label : VIEW_NAMES[view];
  return (
    <>
      <div className="vo-anchor" ref={anchor}>
        <div className={`seg ${HUD_SEG}`}>
          <button className={`view-menu-button${open ? " on" : ""}`} onClick={() => setOpen(!open)}>
            {current && <span className="dim">相机</span>}
            {label} ▾
          </button>
        </div>
        {open && (
          <div className="popover view-menu glass strong" role="menu" aria-label="视角">
            {(Object.keys(VIEW_NAMES) as ViewName[]).map((v) => (
              <button
                key={v}
                role="menuitem"
                className={`view-menu-item${!current && view === v ? " on" : ""}`}
                onClick={() => {
                  setView(v);
                  setOpen(false);
                }}
                data-tip={VIEW_TIPS[v]}
              >
                {VIEW_NAMES[v]}
              </button>
            ))}
            <div className="view-menu-cat">相机</div>
            {cameras.length ? (
              cameras.map((c) => (
                <button
                  key={c.key}
                  role="menuitem"
                  className={`view-menu-item${c.key === look ? " on" : ""}`} data-user-data
                  onClick={() => {
                    setLook(c.key);
                    setOpen(false);
                  }}
                  data-tip={`${c.label}\n透过这台相机看：它每一帧的位置、Focal Length 跟着时间线走，${c.width} × ${c.height} 的画框外变暗。转动视图就离开相机`}
                >
                  {c.label}
                </button>
              ))
            ) : (
              <div className="view-menu-none">场景里没有相机</div>
            )}
          </div>
        )}
      </div>
      <div className={`seg ${HUD_SEG}`}>
        {/* H / F frame the same way while the pointer is over the view */}
        <button onClick={() => frame("all")}>
          框显全部
        </button>
        <button onClick={() => frame("selected")} disabled={!selected}>
          框显选中
        </button>
      </div>
    </>
  );
}
