import type { ReactNode } from "react";
import "./table.css";
import { tipAttrs, type Tip } from "../platform/tips";

/** A list as a table: columns declared once with their header, an optional header tip (only what the header word
 * cannot say) and cell;
 * a row can be picked (the one picked is marked) and a row can be dimmed (no longer in use: a deleted account, an
 * invite that can no longer be used). Both are the component's own states — a page never adds a class of its own to a row. */

export interface Column<T> {
  id: string;
  label: string;
  tip?: Tip | null;
  cell: (row: T) => ReactNode;
  className?: string;
  width?: string; // a fixed width for this column (the rest share what is left): several tables of the same columns
                  // under one another line up, whatever each one holds
}

export function Table<T>({ rows, columns, rowKey, picked, onPick, dim, empty, className }: {
  rows: T[];
  columns: Column<T>[];
  rowKey: (row: T) => string;
  picked?: string | null;
  onPick?: (row: T) => void;
  dim?: (row: T) => boolean; // this row is no longer in use: its text steps back
  empty?: ReactNode;
  className?: string;
}) {
  return (
    <div className={`tbl-wrap${className ? ` ${className}` : ""}`}>
      <table className="tbl">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.id} className={c.className} style={c.width ? { width: c.width } : undefined} {...tipAttrs(c.tip)}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const key = rowKey(r);
            return (
              <tr
                key={key}
                data-row={key}
                className={`${onPick ? "pickable" : ""}${picked === key ? " picked" : ""}${dim?.(r) ? " dim" : ""}`}
                onClick={onPick ? () => onPick(r) : undefined}
              >
                {columns.map((c) => (
                  <td key={c.id} className={c.className}>
                    {c.cell(r)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
      {!rows.length && empty && <div className="tbl-empty">{empty}</div>}
    </div>
  );
}
