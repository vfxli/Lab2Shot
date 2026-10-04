import { useRef, useState } from "react";
import { HUD_SEG } from "../ui/Button";
import { useDismiss } from "../platform/dismiss";
import { slotCamera, useViewCamera, VIEW_NAMES, VIEWER_SLOT, type ViewName } from "../state/viewer";
import { useHandleView, type Reference } from "../state/handleView";
import { packetOf, useResults } from "../state/results";
import { useCookInputs } from "../state/cookInputs";
import { useTypes } from "../state/catalog";
import { typeOf } from "../state/items";
import { nodeRef } from "../graph/naming";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 视图工具栏中视角与框显按钮的唯一所在。视角：透视 / 顶 / 前 / 侧，或透过场景中的某台相机观看；以及框显全部 / 框显选中。
 *
 * 位于画面上方的工具栏中，而非画面左下角：视角与框显属于视图工具，与 2D / 3D、显示种类、视图设置位于同一行。
 *
 * 此工具栏须在不加载 three.js 的情况下绘制（three.js 只随三维舞台按需加载，见 editor/Viewer.tsx 的 Stage3D），
 * 因此不引用三维舞台的任何模块：所需的两项信息（场景中的相机列表与当前选中项）由三维舞台计算后写入 state/viewTools.ts。 */
/** 视角菜单（对应 Houdini 视口的相机菜单）：透视 / 顶 / 前 / 侧，以及场景中按层级位置列出的每台相机；另有框显。 */
export function ViewButtons() {
  const view = useViewCamera((s) => slotCamera(s, VIEWER_SLOT).view);
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
  const current = cameras.find((c) => c.key === look?.key);
  const label = current ? current.label.split("/").pop() || current.label : VIEW_NAMES[view];
  return (
    <>
      <div className="vo-anchor" ref={anchor}>
        <div className={`seg ${HUD_SEG}`}>
          <button className={`view-menu-button${open ? " on" : ""}`} onClick={() => setOpen(!open)}>
            {current && <span className="dim">{t("ui.view.camera")}</span>}
            {label} ▾
          </button>
        </div>
        {open && (
          <div className="popover view-menu glass strong" role="menu" aria-label={t("ui.view.view_menu")}>
            {(Object.keys(VIEW_NAMES) as ViewName[]).map((v) => (
              <button
                key={v}
                role="menuitem"
                className={`view-menu-item${!current && view === v ? " on" : ""}`}
                onClick={() => {
                  setView(v);
                  setOpen(false);
                }}
              >
                {VIEW_NAMES[v]}
              </button>
            ))}
            <div className="view-menu-cat">{t("ui.view.cameras")}</div>
            {cameras.length ? (
              cameras.map((c) => (
                <button
                  key={c.key}
                  role="menuitem"
                  className={`view-menu-item${c.key === look?.key ? " on" : ""}`} data-user-data
                  onClick={() => {
                    setLook(c);
                    setOpen(false);
                  }}
                  {...tipAttrs(tipOf("truncated", c.label))}
                >
                  {c.label}
                </button>
              ))
            ) : (
              <div className="view-menu-none">{t("ui.view.no_cameras")}</div>
            )}
          </div>
        )}
      </div>
      <div className={`seg ${HUD_SEG}`}>
        {/* 指针位于视图上时，H / F 键执行相同的框显 */}
        <button onClick={() => frame("all")}>
          {t("ui.view.frame_all")}
        </button>
        <button onClick={() => frame("selected")} disabled={!selected}>
          {t("ui.view.frame_selected")}
        </button>
      </div>
    </>
  );
}

/** 「参考」菜单：把另一个节点已有的三维结果（它的一个口）半透明、另一种颜色叠在当前显示的内容上，只看不改
 * （state/handleView.ts reference → view/stageLayers.tsx ReferenceLayer）。列出的是除显示节点以外、有三维元素结果的
 * 每个口；类型按目录判断（state/items.ts typeOf 的 in_3d）。 */
export function ReferenceMenu({ shown }: { shown: string }) {
  const reference = useHandleView((s) => s.reference);
  const setReference = useHandleView((s) => s.setReference);
  const reply = useResults((s) => s.reply);
  const labels = useCookInputs((s) => s.nodes);
  const types = useTypes();
  const [open, setOpen] = useState(false);
  const anchor = useRef<HTMLDivElement>(null);
  useDismiss(open, anchor, () => setOpen(false));
  const rows = Object.entries(reply?.nodes ?? {}).flatMap(([node, st]) => (node === shown ? [] : st.ports.outputs
    .filter((q) => typeOf(types, q.type)?.in_3d === "element")
    .map((q) => ({ node, port: q.name, label: `${nodeRef(node, labels[node]?.typeId)} · ${q.label}`, cooked: !!packetOf(st, q.name) }))));
  const on = reference ? rows.find((r) => r.node === reference.node && r.port === reference.port) : undefined;
  const pick = (r: Reference | null) => {
    setReference(r);
    setOpen(false);
  };
  return (
    <div className="vo-anchor" ref={anchor}>
      <div className={`seg ${HUD_SEG}`}>
        <button className={`view-menu-button${open ? " on" : ""}`} onClick={() => setOpen(!open)}>
          {on ? <><span className="dim">{t("ui.view.reference")}</span>{on.label}</> : t("ui.view.reference")} ▾
        </button>
      </div>
      {open && (
        <div className="popover view-menu glass strong" role="menu" aria-label={t("ui.view.reference")}>
          <button role="menuitem" className={`view-menu-item${!reference ? " on" : ""}`} onClick={() => pick(null)}>{t("ui.view.no_reference")}</button>
          <div className="view-menu-cat">{t("ui.view.node_3d_results")}</div>
          {/* 只有有包的口能选（state/results.ts packetOf）；没算过的灰着、写「还没算」，选了也画不出东西 */}
          {rows.length ? rows.map((r) => (
            <button key={`${r.node}/${r.port}`} role="menuitem" data-user-data disabled={!r.cooked}
              className={`view-menu-item${on === r ? " on" : ""}`} onClick={() => pick({ node: r.node, port: r.port })}
              {...tipAttrs(tipOf("disabled", r.cooked ? undefined : t("ui.view.reference_not_cooked")))}>
              {r.label}{!r.cooked && <span className="dim">{t("ui.view.not_cooked_paren")}</span>}
            </button>
          )) : <div className="view-menu-none">{t("ui.view.no_other_3d_results")}</div>}
        </div>
      )}
    </div>
  );
}
