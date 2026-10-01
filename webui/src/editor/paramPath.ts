/** Where a control's value sits in its node: a parameter `name`, or a table's cell. A cell is named two ways:
 * - `table[i].column`, by the row's place: the ids the server's answers use (nodes/applies.py: `layers[2].scale` is
 *   greyed, `layers[2].scale=x` is an option's state), right for the rows as they are drawn now;
 * - `table[name=n].column`, by the row's own name (the row's identity, editor/ParamTable.tsx): what an editor window that
 *   stays open keeps (`stableAt`), so a row removed or moved above it does not move the window onto another row.
 * A control is keyed, licensed, asked about and written by one of these (editor/ParamControls.tsx Control `at`), never by
 * its column's bare name, which can repeat a top-level parameter's (a retarget's `motion_pose.scale` and its `scale`). */

import type { NodeTypeDef, ParamDef } from "../api";

const CELL = /^([^[\]]+)\[(?:(\d+)|name=(.*))\]\.([^\]]+)$/;

type Row = Record<string, unknown>;
const rowsOf = (params: Record<string, unknown>, table: string): Row[] => (Array.isArray(params[table]) ? (params[table] as Row[]) : []);

/** The cell's table, row index (-1: no such row now) and column; null for a top-level parameter. */
function cellOf(params: Record<string, unknown>, at: string): { table: string; i: number; column: string } | null {
  const cell = CELL.exec(at);
  if (!cell) return null;
  const [, table, index, name, column] = cell;
  const rows = rowsOf(params, table);
  const i = index !== undefined ? Number(index) : rows.findIndex((r) => r && typeof r === "object" && String(r.name) === name);
  return { table, i: i < rows.length ? i : -1, column };
}

/** The definition and value at `at` (null: the node has no such parameter, row or column). */
export function paramAt(def: NodeTypeDef, params: Record<string, unknown>, at: string): { p: ParamDef; value: unknown } | null {
  const cell = cellOf(params, at);
  if (!cell) {
    const p = def.params.find((q) => q.name === at);
    return p ? { p, value: params[at] } : null;
  }
  const p = def.params.find((q) => q.name === cell.table)?.items?.find((f) => f.name === cell.column);
  const row = cell.i >= 0 ? rowsOf(params, cell.table)[cell.i] : undefined;
  return p && row && typeof row === "object" ? { p, value: row[cell.column] } : null;
}

/** The node's parameters to write for `v` at `at`: the parameter itself, or the whole table with that one cell changed
 * (nothing when that row is gone). */
export function placeAt(params: Record<string, unknown>, at: string, v: unknown): Record<string, unknown> {
  const cell = cellOf(params, at);
  if (!cell) return { [at]: v };
  if (cell.i < 0) return {};
  return { [cell.table]: rowsOf(params, cell.table).map((r, j) => (j === cell.i ? { ...r, [cell.column]: v } : r)) };
}

/** `at` by the row's name, for whatever outlives this drawing of the table (an open editor window). */
export function stableAt(params: Record<string, unknown>, at: string): string {
  const cell = cellOf(params, at);
  if (!cell || cell.i < 0) return at;
  const rows = rowsOf(params, cell.table);
  const name = rows[cell.i]?.name;
  // a name that is not the row's alone (no name, or another row has it too) does not identify it: the place does
  const own = name !== undefined && name !== null && rows.filter((r) => r && typeof r === "object" && String(r.name) === String(name)).length === 1;
  return own ? `${cell.table}[name=${String(name)}].${cell.column}` : `${cell.table}[${cell.i}].${cell.column}`;
}

/** `at` by the row's place now, as the server's answers name it (null: that row is gone). */
export function indexAt(params: Record<string, unknown>, at: string): string | null {
  const cell = cellOf(params, at);
  if (!cell) return at;
  return cell.i < 0 ? null : `${cell.table}[${cell.i}].${cell.column}`;
}
