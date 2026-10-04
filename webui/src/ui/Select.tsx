import { useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { Menu, type MenuRow } from "./Menu";
import { IconChevron } from "./icons";
import { tipAttrs, type Tip } from "../platform/tips";

/** 站点唯一的下拉框。
 *
 * 不使用原生 `<select>`：其弹出列表由系统绘制（直角、白底、蓝色高亮），与整体的暗色圆角风格不一致，
 * 且无法用 CSS 修改。因此这里只沿用原生 select 的触发器外形（一个 field 加一个小箭头），列表交由站点已有的
 * 菜单组件（ui/Menu.tsx：玻璃效果、圆角、暗色、Esc 关闭、点击外部关闭），一处实现，各处一致。
 *
 * 用法与原生相同：`value` 为当前值，`options` 为 {value, label} 列表，`onPick` 返回新值。
 * 选项的 `tip` 只在有看不到的内容时给（不能选的原因）；未提供时选项没有悬停提示，不重复选项自身的文字。
 *
 * 其他属性（`data-field`：从页面外据此找到该下拉框；`data-user-data`：内容为用户或文件提供的文字，允许截断，
 * 完整文字在悬停提示中，见 styles/01-user-data.css）原样附加到触发器上，写法与 ui/Button.tsx 的 Chip 相同。 */

export interface SelectOption {
  value: string;
  label: ReactNode;
  tip?: Tip | null;
  off?: boolean; // 存在但当前不可选：置灰并在提示中注明原因（控件不得时隐时现）
}

export function Select({ value, options, onPick, label, tip, className = "", disabled, width, layout, ...rest }:
  Omit<ButtonHTMLAttributes<HTMLButtonElement>, "className" | "value" | "onClick"> & {
  value: string;
  options: SelectOption[];
  onPick: (value: string) => void;
  label: string; // 下拉框的用途（aria-label）
  tip?: Tip | null; // 悬停提示（platform/tips.ts：说明用途）
  className?: string;
  disabled?: boolean;
  width?: number; // 弹出列表的最小宽度（默认与触发器同宽；选项名更长时按选项名扩展，不截断）
  layout?: string; // 该列表在页面里的名字，从页面外据此找到它；不代表特定样式
}) {
  const { "data-user-data": userData, ...attrs } = rest as Record<string, unknown>;
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  const box = useRef<HTMLButtonElement | null>(null);
  const now = options.find((o) => o.value === value);
  const rows: MenuRow[] = options.map((o) => ({
    key: o.value,
    label: o.label,
    tip: o.tip,
    off: o.off,
    desc: o.value === value && !o.off ? "✓" : undefined, // 不可选的项（「选一个环节」这类占位）不是选中的项，不打勾
    run: () => onPick(o.value),
  }));
  const open = () => {
    const r = box.current?.getBoundingClientRect();
    if (r) setAt({ x: r.left, y: r.bottom + 2 });
  };
  return (
    <>
      <button ref={box} type="button" className={`field select-trigger ${className}`.trim()} aria-label={label}
        {...tipAttrs(tip)} disabled={disabled} onClick={open} {...attrs}>
        {/* 「用户或文件提供的文字」标记附加在文字元素上而非按钮上：截断规则只作用于带标记的元素
            （styles/01-user-data.css，只允许该选择器带省略号），而文字位于此 span 中 */}
        <span className="select-now" {...(userData === undefined ? {} : { "data-user-data": userData })}
          {...tipAttrs(userData === undefined ? undefined : { why: "truncated", text: typeof now?.label === "string" ? now.label : value })}>
          {now?.label ?? value}
        </span>
        <IconChevron size={9} />
      </button>
      {at && (
        <Menu at={at} rows={rows} label={label} layout={layout}
          width={width ?? box.current?.getBoundingClientRect().width}
          onClose={() => setAt(null)} />
      )}
    </>
  );
}
