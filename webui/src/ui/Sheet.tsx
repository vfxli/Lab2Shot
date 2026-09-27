import "./sheet.css";
import { KeyLayer, useKeyLayer, useShortcut } from "../platform/keys";
import { createPortal } from "react-dom";
import { IconClose } from "./icons";
import { IconButton } from "./Button";

/** A modal panel: title, close button, scrolling body; a click on the dimmed backdrop or Esc closes it. Rendered at
 * the page's top level, so it covers the whole window wherever it is opened from. Glass like every floating layer;
 * `solid` for a window of tables and long lists (队列, 日志), which read best on a solid ground.
 *
 * `bare`: the sheet draws no title row — the page inside puts the name where it belongs (从模板新建: on top of the
 * category rail). The name is still the sheet's, for anyone reading the page aloud, and the way out stays in the
 * same corner. */
export function Sheet({ title, width, height, solid, bare, onClose, children }: {
  title: string; width?: number; height?: number; solid?: boolean; bare?: boolean; onClose: () => void; children: React.ReactNode;
}) {
  const layer = useKeyLayer(true); // while open, keys go to the sheet and what it holds only (platform/keys.ts)
  useShortcut({ keys: ["escape"], inText: true, run: () => onClose() }, { layer });
  const close = (
    <IconButton tip="关闭（Esc）" tone="ghost" onClick={onClose} aria-label="关闭">
      <IconClose />
    </IconButton>
  );
  return createPortal(
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div
        className={`sheet glass${solid ? " solid" : ""}${bare ? " bare" : ""}${height ? " fixed" : ""}`}
        role="dialog"
        aria-label={title}
        style={{ ...(width ? { width } : {}), ...(height ? { height } : {}) }}
      >
        {bare ? (
          <div className="sheet-close">{close}</div>
        ) : (
          <div className="sheet-head">
            <h2>{title}</h2>
            {close}
          </div>
        )}
        <div className="sheet-body">
          <KeyLayer.Provider value={layer}>{children}</KeyLayer.Provider>
        </div>
      </div>
    </div>,
    document.body,
  );
}
