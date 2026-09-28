import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useKeyLayer, useShortcut } from "../platform/keys";
import "./menu.css";

/** The site's single menu: a floating list of rows at a point on the screen (a node's context menu, the account menu, a
 * file menu). It closes on any click outside it, on scroll and on Esc, and every row describes its action (`tip`,
 * required). A row is 28 px high; content at its right end (a shortcut, a count, a short explanation) goes in `desc`,
 * and a keyboard shortcut in a <Kbd>.
 *
 * Glass, like every floating layer (ui/glass.css); a docked list is not a menu but a panel with rows.
 *
 * 菜单挂载在 document.body 上（portal），而非打开它的控件旁。菜单使用 `position: fixed`，按屏幕坐标定位；
 * 而按 CSS 规范，只要某个祖先带有 `transform`，`fixed` 即改为相对该祖先定位。节点图画布
 * （React Flow 的 `.react-flow__viewport`）始终带有 transform，因此节点上参数下拉框打开的菜单
 * 会被置于缩放坐标系中并落到屏幕外，用户看到的是点击无反应，而菜单实际已打开。
 * 挂载到 body 上即不存在该祖先，一处修改即可使所有使用菜单的位置（节点参数、节点右键菜单、账号、文件、时间线）均正确。 */

export interface MenuRow {
  key: string;
  label: ReactNode;
  tip: string;
  desc?: ReactNode;
  off?: boolean; // present but unavailable now: greyed, its tip explains why
  run?: () => void;
}

export function Menu({ at, rows, label, onClose, width, layout }: {
  at: { x: number; y: number };
  rows: MenuRow[];
  label: string; // what this menu is about (aria-label)
  onClose: () => void;
  width?: number; // 最小宽度（使下拉列表与触发器对齐）；某一行文字更长时按文字宽度扩展，菜单中的文字不得截断
  layout?: string; // the page's own name for this menu (it can be found from outside the page by this); never a style of its own
}) {
  const box = useRef<HTMLDivElement>(null);
  // 无论在何处打开，菜单都必须完整位于窗口内：`at` 只是打开位置，菜单自身高度须测量后才能得知。
  // 测量后回夹一次：下方空间不足时贴窗口下沿，右侧空间不足时贴右沿。
  // 弹层沿固定方向展开，超出即越界，越界的行用户无法点击，
  // 表现为点击无反应。在此统一处理一次，所有使用菜单的位置均正确
  // （节点右键菜单、账号、文件、时间线，以及每个 ui/Select.tsx 的下拉框）；
  // 画面底部「切换人」的下拉框紧贴窗口下沿，人数达到数十时列表的一半会在窗口外。
  // 使用 useLayoutEffect：在绘制前完成回夹，不会先闪现再跳动。
  const [place, setPlace] = useState<{ x: number; y: number } | null>(null);
  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const fit = () => {
      // 测量布局尺寸（offsetWidth），而非 getBoundingClientRect：菜单打开时有一个由小到大的缩放动画，
      // 此时 rect 为缩小后的尺寸，据此回夹会相差十余像素，导致右侧超出窗口
      const w = el.offsetWidth, h = el.offsetHeight;
      const edge = 8;
      const x = Math.max(edge, Math.min(at.x, window.innerWidth - edge - w));
      const y = Math.max(edge, Math.min(at.y, window.innerHeight - edge - h));
      setPlace((had) => (had && had.x === x && had.y === y ? had : { x, y }));
    };
    fit();
    // 菜单打开期间自身尺寸变化时（网页字体延迟加载、某行文字改变）也需重新回夹：测量的是其实际尺寸，而非打开时的尺寸
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => ro.disconnect();
  }, [at.x, at.y, rows.length, width]);
  useEffect(() => {
    // 点击菜单自身不视为点击外部。此处按元素判断（`contains`），不依赖事件冒泡被何处拦截：
    // 挂载到 body 后，React 的合成事件与该 window 监听不在同一条冒泡路径上
    const close = (e: Event) => {
      if (!(e.target instanceof Node) || !box.current?.contains(e.target)) onClose();
    };
    // after this click: the click that opened the menu must not close it again
    const later = setTimeout(() => {
      window.addEventListener("pointerdown", close);
      window.addEventListener("wheel", close);
    });
    return () => {
      clearTimeout(later);
      window.removeEventListener("pointerdown", close);
      window.removeEventListener("wheel", close);
    };
  }, [onClose]);
  useShortcut({ keys: ["escape"], inText: true, run: () => onClose() }, { layer: useKeyLayer(true) }); // a key layer: keys go to the menu only
  return createPortal(
    <div
      ref={box}
      className={layout ? `menu ${layout} glass` : "menu glass"}
      role="menu"
      aria-label={label}
      style={{ left: place?.x ?? at.x, top: place?.y ?? at.y, ...(width ? { minWidth: width } : {}) }}
      onPointerDown={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {rows.map((r) => (
        <div
          key={r.key}
          role="menuitem"
          data-key={r.key} // 这一行的键（下拉框中即该项的值），写在元素上，从页面外可以按它找到这一行
          aria-disabled={r.off || undefined}
          className={r.off ? "menu-item off" : "menu-item"}
          data-tip={r.tip}
          onPointerUp={
            r.off || !r.run
              ? undefined
              : (e) => {
                  e.stopPropagation();
                  r.run!();
                  onClose();
                }
          }
        >
          <span className="menu-label">{r.label}</span>
          {r.desc !== undefined && r.desc !== "" && <span className="menu-desc">{r.desc}</span>}
        </div>
      ))}
    </div>,
    document.body,
  );
}

/** The key to press, in a menu row or a tooltip: always a key, never a descriptive word. */
export function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}
