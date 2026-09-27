/** The two alert colours needed as values by canvases and inline styles (CSS uses the tokens in ui/tokens.css, which
 * webui/tests/nodeMarks.test.ts keeps equal to these). Red is reserved for production risk (message level P); ordinary
 * errors (a wrong wire, a missing port, a failed or blocked node) are purple. */
export const PRODUCTION_RISK_COLOR = "#ff5b4f"; // --production-risk
export const ERROR_COLOR = "#b98cff"; // --error

/** The colour of a node the catalogue does not place anywhere (an unknown type, a node whose extension was removed):
 * the muted grey of --text-3, as a value for the canvas and the glyphs (api/catalog.ts nodeCategory). */
export const NEUTRAL_COLOR = "#69717c"; // --text-3

/** The node graph's dot grid colour, which @xyflow/react's <Background> takes as a value rather than a class: the
 * hairline colour, slightly fainter. */
export const GRAPH_DOT_COLOR = "rgba(190, 205, 225, 0.07)";

/** 二维视图背景「纯色」的可选档位：颜色与名称在此一并定义（界面上只显示名称，不显示色值）。
 * 纯黑（查看 alpha 时以真正的黑色为底）、近黑、中灰、近白，另加绿幕绿和洋红两个对比色。 */
export const BG_CHOICES = [
  { value: "#000000", name: "纯黑" },
  { value: "#1a1a1c", name: "近黑" },
  { value: "#808083", name: "中灰" },
  { value: "#e8e8ec", name: "近白" },
  { value: "#00b140", name: "绿幕绿" },
  { value: "#ff00ff", name: "洋红" },
] as const;

export const bgColourOf = (picked: string): string => picked || BG_CHOICES[0].value;

/** 透明区域棋盘格的两种灰色：二维舞台用画布绘制（view/overlays.ts drawBackground），
 * 背景下拉框中的色样用 CSS 绘制。值只在此处定义，令牌 `--checker-light` / `--checker-dark`
 * 须与之保持一致（由 webui/tests/nodeMarks.test.ts 检查，做法与 --production-risk 相同）。 */
export const CHECKER_LIGHT = "#4a4a4e"; // --checker-light
export const CHECKER_DARK = "#323234"; // --checker-dark

