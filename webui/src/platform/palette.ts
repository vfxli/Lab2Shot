/** The error colour needed as a value by canvases and inline styles (CSS uses the token in ui/tokens.css): an ordinary
 * error — a wrong wire, a missing port, a failed or blocked node — is purple (production risk, message level P, is a
 * separate red token used only in CSS). */
export const ERROR_COLOR = "#b98cff"; // --error

/** The accent colour (--accent in ui/tokens.css) as a value for canvases: the 2D stage marks the people a 「选人」 picks
 * with it (view/overlays.ts drawBoxes), the same colour as every other "chosen / on" state of the page. */
export const ACCENT_COLOR = "#6aa8ff"; // --accent

/** The colour of a node the catalogue does not place anywhere (an unknown type, a node whose extension was removed):
 * the muted grey of --text-3, as a value for the canvas and the glyphs (api/catalog.ts nodeCategory). */
export const NEUTRAL_COLOR = "#69717c"; // --text-3

/** The node graph's dot grid colour, which @xyflow/react's <Background> takes as a value rather than a class: the
 * hairline colour, slightly fainter. */
export const GRAPH_DOT_COLOR = "rgba(190, 205, 225, 0.07)";

/** The choices of the 2D view's plain-colour background, colour and name defined together here (the interface shows only
 * the name, never the value): pure black (a true black under the image when inspecting alpha), near black, mid grey,
 * near white, and two contrast colours, green-screen green and magenta. */
export const BG_CHOICES = [
  { value: "#000000", name: "纯黑" },
  { value: "#1a1a1c", name: "近黑" },
  { value: "#808083", name: "中灰" },
  { value: "#e8e8ec", name: "近白" },
  { value: "#00b140", name: "绿幕绿" },
  { value: "#ff00ff", name: "洋红" },
] as const;

export const bgColourOf = (picked: string): string => picked || BG_CHOICES[0].value;

/** The two greys of the checkerboard behind transparent areas: the 2D stage draws it on a canvas (view/overlays.ts
 * drawBackground), the swatch in the background pull-down draws it in CSS. The values are defined only here; the tokens
 * `--checker-light` / `--checker-dark` must match them (the same arrangement as ERROR_COLOR / --error). */
export const CHECKER_LIGHT = "#4a4a4e"; // --checker-light
export const CHECKER_DARK = "#323234"; // --checker-dark

