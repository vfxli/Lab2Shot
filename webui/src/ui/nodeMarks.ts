import { createElement, Fragment, type ReactElement } from "react";
import type { StandingMark } from "../api";

/** The marks a node's own declaration gives it (nodes/applies.py standing_marks): a short word on its bottom row, with
 * the whole sentence in its 数据信息 card — 「输入图像需要去畸变」 on a node that treats the plate as a pinhole lens. Each is a
 * message, and its level decides its colour: red only for a production risk (P), the ordinary notice colour for the
 * rest (red is kept for what can cause a production accident; a standing notice like the pinhole one is not one).
 * A template card says nothing of them. Nothing here decides which node has a mark. Plain createElement, no JSX. */
const NODE_MARK_CLASS = "node-mark";

export function NodeMarks({ marks }: { marks?: readonly StandingMark[] }): ReactElement | null {
  if (!marks?.length) return null;
  return createElement(
    Fragment,
    null,
    ...marks.map((m) => createElement("span", { key: m.code, className: NODE_MARK_CLASS, "data-code": m.code, "data-level": m.level }, m.mark)),
  );
}
