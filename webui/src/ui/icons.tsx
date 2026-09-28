import "./brand.css";
import "./categories.css";
import markSmall from "../brand/mark-96.png";
import markBig from "../brand/mark-192.png";
// Small line icons in the SF Symbols spirit: 1.6px strokes, round caps.

type P = { size?: number; color?: string };

const base = (size: number, color: string, children: React.ReactNode) => (
  <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    {children}
  </svg>
);

export const IconPlay = ({ size = 14, color = "currentColor", back = false }: P & { back?: boolean }) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden style={back ? { transform: "scaleX(-1)" } : undefined}>
    <path d="M4.5 2.8v10.4a.6.6 0 0 0 .9.5l8.4-5.2a.6.6 0 0 0 0-1L5.4 2.3a.6.6 0 0 0-.9.5Z" fill={color} />
  </svg>
);

export const IconPause = ({ size = 14, color = "currentColor" }: P) => (
  <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden>
    <rect x="3.5" y="2.5" width="3.2" height="11" rx="1" fill={color} />
    <rect x="9.3" y="2.5" width="3.2" height="11" rx="1" fill={color} />
  </svg>
);

export const IconEye = ({ size = 13, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <path d="M1.8 8S4.2 3.6 8 3.6 14.2 8 14.2 8 11.8 12.4 8 12.4 1.8 8 1.8 8Z" />
      <circle cx="8" cy="8" r="2" />
    </>
  ));

export const IconGrid = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <rect x="2.5" y="2.5" width="4.5" height="4.5" rx="1.2" />
      <rect x="9" y="2.5" width="4.5" height="4.5" rx="1.2" />
      <rect x="2.5" y="9" width="4.5" height="4.5" rx="1.2" />
      <rect x="9" y="9" width="4.5" height="4.5" rx="1.2" />
    </>
  ));

export const IconUndo = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, <path d="M5.8 3.8 3 6.6l2.8 2.8M3 6.6h6.7a3.3 3.3 0 0 1 0 6.6H7.5" />);

export const IconRedo = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, <path d="M10.2 3.8 13 6.6l-2.8 2.8M13 6.6H6.3a3.3 3.3 0 0 0 0 6.6h2.2" />);

export const IconFit = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" />);

export const IconGroup = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <rect x="2" y="2.5" width="12" height="11" rx="2" strokeDasharray="2.2 1.8" />
      <rect x="4.5" y="6" width="3" height="2.6" rx=".6" />
      <rect x="8.8" y="8.6" width="3" height="2.6" rx=".6" />
    </>
  ));

export const IconMap = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <rect x="2" y="3" width="12" height="10" rx="2" />
      <rect x="8" y="7.5" width="4" height="3.5" rx=".8" />
    </>
  ));

export const IconPlus = ({ size = 14, color = "currentColor" }: P) => base(size, color, <path d="M8 3v10M3 8h10" />);

export const IconMinus = ({ size = 14, color = "currentColor" }: P) => base(size, color, <path d="M3 8h10" />);

export const IconClose = ({ size = 12, color = "currentColor" }: P) => base(size, color, <path d="m4 4 8 8M12 4l-8 8" />);

/** Three dots: 「还有别的」, the action menu of a template card, a node menu row or a category. */
export const IconMore = ({ size = 13, color = "currentColor" }: P) =>
  base(size, color, <path d="M4 8h.01M8 8h.01M12 8h.01" />);

/** 「i」 in a circle: 「这里有说明」, the mark at a node's bottom right that opens 数据信息. It is never an exclamation mark:
 * that shape denotes a warning on this page (the corner mark), and the two must remain distinct. */
export const IconInfo = ({ size = 12, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7.2v3.6" />
      <path d="M8 5.1v.1" />
    </>
  ));

/** Down (open: 展开) or up (close: 收起). */
export const IconChevron = ({ size = 12, color = "currentColor", up = false }: P & { up?: boolean }) =>
  base(size, color, <path d={up ? "M4 10 8 6l4 4" : "M4 6l4 4 4-4"} />);

/** A node with a row on it: a parameter that is also shown on the node's body. `filled`: it is shown.
    (The mark on a parameter row: editor/ParamPanel.tsx OnNodePin.) */
export const IconOnNode = ({ size = 11, color = "currentColor", filled = false }: P & { filled?: boolean }) =>
  base(size, color, (
    <>
      <rect x="2" y="3" width="12" height="10" rx="2.2" />
      <path d="M2 6.6h12" />
      <rect x="4.6" y="8.8" width="6.8" height="1.9" rx=".9" fill={filled ? color : "none"} strokeWidth={filled ? 0.8 : 1.2} />
    </>
  ));

export const IconCamera = ({ size = 13, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <rect x="1.8" y="4.5" width="9" height="7" rx="1.5" />
      <path d="m10.8 7 3.4-2v6l-3.4-2" />
    </>
  ));

export const IconBone = ({ size = 13, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <circle cx="4" cy="12" r="1.6" />
      <circle cx="12" cy="4" r="1.6" />
      <path d="M5.2 10.8 10.8 5.2" />
    </>
  ));

// a file parameter's buttons (editor/FileParam.tsx): a folder, a file, an image sequence
export const IconFolder = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, <path d="M2 4.5c0-.6.4-1 1-1h3.2l1.4 1.5H13c.6 0 1 .4 1 1v6.5c0 .6-.4 1-1 1H3c-.6 0-1-.4-1-1V4.5Z" />);

export const IconFile = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <path d="M4 1.8h5.2L12.5 5v9.2H4V1.8Z" />
      <path d="M9 1.8V5h3.5" />
    </>
  ));

/** A stack of frames: an image sequence. */
export const IconFrames = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <rect x="2" y="4.5" width="9" height="8" rx="1" />
      <path d="M4.5 2.5h8c.6 0 1 .4 1 1V10" />
    </>
  ));

export const IconOpen = ({ size = 14, color = "currentColor" }: P) =>
  base(size, color, (
    <>
      <path d="M2 4.5c0-.6.4-1 1-1h3.2l1.4 1.5H12c.6 0 1 .4 1 1V7" />
      <path d="M2.5 13.5 4.3 8h10.2l-1.8 5.5H2.5Z" />
    </>
  ));

/** Lines of text: the log. */
export const BrandMark = ({ big = false, hero = false }: { big?: boolean; hero?: boolean }) => (
  <img className={`brand-mark${big ? " big" : ""}${hero ? " hero" : ""}`} src={big || hero ? markBig : markSmall} alt="" aria-hidden />
);

/** A speech bubble: 提交反馈. */
