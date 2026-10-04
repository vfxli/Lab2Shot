import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useKeyLayer, useShortcut } from "../platform/keys";
import "./menu.css";
import { tipAttrs, type Tip } from "../platform/tips";

/** 站点唯一的菜单：在屏幕上某一点浮出的一列行（节点右键菜单、账号菜单、文件菜单）。点击菜单外任意处、滚动或按 Esc
 * 即关闭。一行的悬停提示（`tip`）只写行上看不到的：不可用的原因、操作的后果；行上已写着的不重复。行高 28 px；
 * 行右端的内容（快捷键、计数、简短说明）放在 `desc` 中，键盘快捷键用 <Kbd>。
 *
 * 浮层的玻璃边光与阴影、不透明的底（ui/glass.css glass solid：下面面板的边框不透上来），层级在所有能打开它的层之上（tokens.css --z-menu）；停靠在页面中的列表不是菜单，而是带行的面板。
 *
 * 菜单挂载在 document.body 上（portal），而非打开它的控件旁。菜单使用 `position: fixed`，按屏幕坐标定位；
 * 而按 CSS 规范，只要某个祖先带有 `transform`，`fixed` 即改为相对该祖先定位。节点图画布
 * （React Flow 的 `.react-flow__viewport`）始终带有 transform，因此节点上参数下拉框打开的菜单
 * 会被置于缩放坐标系中并落到屏幕外，用户看到的是点击无反应，而菜单实际已打开。
 * 挂载到 body 上即不存在该祖先，所有使用菜单的位置（节点参数、节点右键菜单、账号、文件、时间线）均按屏幕坐标正确定位。 */

export interface MenuRow {
  key: string;
  label: ReactNode;
  tip?: Tip | null;
  desc?: ReactNode;
  off?: boolean; // 存在但当前不可用：置灰，提示中说明原因
  run?: () => void;
}

export function Menu({ at, rows, label, onClose, width, layout }: {
  at: { x: number; y: number };
  rows: MenuRow[];
  label: string; // 该菜单的用途（aria-label）
  onClose: () => void;
  width?: number; // 最小宽度（使下拉列表与触发器对齐）；某一行文字更长时按文字宽度扩展，菜单中的文字不得截断
  layout?: string; // 该菜单在页面里的名字（从页面外据此找到它）；不代表特定样式
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
    // 在本次点击之后再监听：打开菜单的那次点击不得又把它关掉
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
  useShortcut({ keys: ["escape"], inText: true, run: () => onClose() }, { layer: useKeyLayer(true) }); // 独立的按键层：按键只交给菜单
  return createPortal(
    <div
      ref={box}
      className={layout ? `menu ${layout} glass solid` : "menu glass solid"}
      role="menu"
      aria-label={label}
      style={{ left: place?.x ?? at.x, top: place?.y ?? at.y, ...(width ? { minWidth: width } : {}) }}
      onPointerDown={(e) => e.stopPropagation()}
      onContextMenu={(e) => e.preventDefault()}
    >
      {/* 行放在自己的滚动层里，面板（.menu：底色、圆角、玻璃边光）本身不滚：边光是面板上一层 position: absolute 的
          ::before，面板若自己滚，它会跟着内容往上走，在打开时的底边位置留下一道线；线以下的行在 Chrome 里还会没有底色 */}
      <div className="menu-rows">
      {rows.map((r) => (
        <div
          key={r.key}
          role="menuitem"
          data-key={r.key} // 这一行的键（下拉框中即该项的值），写在元素上，从页面外可以按它找到这一行
          aria-disabled={r.off || undefined}
          className={r.off ? "menu-item off" : "menu-item"}
          {...tipAttrs(r.tip)}
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
      </div>
    </div>,
    document.body,
  );
}

/** 菜单行或悬停提示中要按的键：始终写键名，不写描述性文字。 */
export function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}
