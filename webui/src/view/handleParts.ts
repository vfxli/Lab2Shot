/** 二维手柄工具的公共定义，供 `view/handles2d.ts` 与 `view/figure2d.ts` 使用：
 * 条目结构、条目解析，以及「先放置后调整」类手柄的颜色。
 */

export interface Pt {
  x: number; // 图像像素坐标
  y: number;
}

export interface Entry {
  frame: number;
  v: number[];
}

/** 解析一条条目，格式为「帧:x1,y1,…」（语法见 `lab2shot/nodes/handles.py` HANDLE_KINDS）。 */
export const parse = (s: string): Entry => {
  const [f, rest] = s.split(":");
  return { frame: Number(f), v: rest.split(",").map(Number) };
};

/** 「先放置后调整」类手柄（平面四角、火柴人）统一使用的黄色，用于提示该手柄可拖动。 */
export const CORNER_COLOR = "#FFD60A";
