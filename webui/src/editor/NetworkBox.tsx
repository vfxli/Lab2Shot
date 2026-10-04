import { memo, useState } from "react";
import { Handle, NodeResizer, Position, type Node, type NodeProps } from "@xyflow/react";
import { toggleBox } from "../graph/actions";
import { useLook, type Box } from "../state/look";
import { BOX_COLORS, BOX_HEAD } from "../graph/nodes";
import { Swatches } from "../ui/Swatches";
import { composing } from "../platform/keys";
import { useT } from "../i18n/t";
import { boxLabel } from "../graph/naming";
import { tipAttrs, tipOf } from "../platform/tips";

export type BoxNode = Node<{ box: Box }, "box">;

/**
 * Group box (Houdini network box): a titled, tinted frame behind nodes.
 * Drag it by the title bar to move everything inside; collapse it to a single bar.
 * The body lets clicks through, so box-select and the node menu still work inside.
 */
export const NetworkBox = memo(function NetworkBox({ id, data, selected }: NodeProps<BoxNode>) {
  const t = useT(); // memoised (xyflow): re-renders by itself when the language changes
  const { box } = data;
  const setBox = useLook((s) => s.setBox);
  const [editing, setEditing] = useState(false);

  return (
    <div className={`netbox${box.collapsed ? " collapsed" : ""}${selected ? " selected" : ""}`} style={{ ["--box" as string]: box.color }}>
      <NodeResizer
        isVisible={!!selected && !box.collapsed}
        minWidth={200}
        minHeight={BOX_HEAD + 60}
        lineClassName="netbox-resize-line"
        handleClassName="netbox-resize-handle"
        onResize={(_, r) => setBox(id, { x: r.x, y: r.y, w: r.width, h: r.height })}
      />
      {/* wires into a collapsed box attach here */}
      <Handle type="target" id="in" position={Position.Left} isConnectable={false} className="netbox-port" style={{ top: BOX_HEAD / 2 }} />
      <Handle type="source" id="out" position={Position.Right} isConnectable={false} className="netbox-port" style={{ top: BOX_HEAD / 2 }} />
      <div className="netbox-head">
        <button className="netbox-fold nodrag" aria-label={box.collapsed ? t("ui.node.expand") : t("ui.node.box_fold")} onClick={() => toggleBox(id)}>
          <svg width={10} height={10} viewBox="0 0 10 10" style={{ transform: box.collapsed ? "rotate(-90deg)" : undefined }}>
            <path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
        {editing ? (
          <input
            className="netbox-title-input nodrag"
            autoFocus
            defaultValue={boxLabel(box)}
            onBlur={(e) => {
              // left as it was shown (the default): keeps no name of its own, so it still follows the language
              const v = e.target.value.trim();
              setBox(id, { label: !v || v === boxLabel(box) ? box.label : v });
              setEditing(false);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !composing(e)) (e.target as HTMLInputElement).blur();
              if (e.key === "Escape") setEditing(false);
              e.stopPropagation();
            }}
          />
        ) : (
          <span className="netbox-title" data-user-data onDoubleClick={() => setEditing(true)} {...tipAttrs(tipOf("truncated", boxLabel(box)))}>
            {boxLabel(box)}
          </span>
        )}
        {box.collapsed && <span className="netbox-count">{t("ui.node.menu_nodes", { count: box.members.length })}</span>}
        {selected && (
          <Swatches label={t("ui.node.box_color")} size="sm" layout="netbox-colors nodrag" value={box.color} colors={BOX_COLORS} onChange={(color) => setBox(id, { color })} />
        )}
      </div>
    </div>
  );
});
