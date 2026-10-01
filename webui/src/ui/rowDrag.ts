import { useRef, useState } from "react";

// the wording 拖动改顺序, written in one place (the same thing is called by the same word everywhere)
export const DRAG_TIP = "拖动改顺序";

/** 拖动改顺序, written once: the parameter table's rows (editor/ParamTable.tsx) and a list setting's rows
 * (admin/ListField.tsx) drag this way, so a table that needs it takes this implementation instead of writing its own.
 *
 * `onDrop(from, to)` is told the two positions in the list as it is drawn; what that means (reordering an array,
 * a setting's list) is the caller's. `grip(i)` goes on the handle (it alone is `draggable`, so
 * selecting text in a row still works), `row(i)` on the row it may be dropped on, and `over` is the row the pointer
 * is on right now (the caller draws it, class `drag-over`).
 *
 * `can(from, to)`: may this row be dropped there at all.
 * A row it says no to is not made a drop target — the browser then shows its own 禁止 cursor and the row does not
 * light up, so a drop that would do nothing never looks like one that worked (a silent no-op is the worst outcome:
 * the user assumes they missed and keeps trying). */
export function useRowDrag(onDrop: (from: number, to: number) => void, can: (from: number, to: number) => boolean = () => true) {
  const from = useRef<number | null>(null);
  const [over, setOver] = useState<number | null>(null);
  const end = () => {
    from.current = null;
    setOver(null);
  };
  return {
    over,
    grip: (i: number) => ({
      draggable: true,
      onDragStart: (e: React.DragEvent) => {
        from.current = i;
        e.dataTransfer.effectAllowed = "move";
      },
      onDragEnd: end,
    }),
    row: (i: number) => ({
      onDragOver: (e: React.DragEvent) => {
        if (from.current === null) return;
        if (!can(from.current, i)) {
          setOver((d) => (d === i ? null : d)); // no preventDefault: not a drop target, so the pointer says 禁止
          return;
        }
        e.preventDefault();
        setOver(i);
      },
      onDragLeave: () => setOver((d) => (d === i ? null : d)),
      onDrop: (e: React.DragEvent) => {
        e.preventDefault();
        const start = from.current;
        end();
        if (start !== null && start !== i && can(start, i)) onDrop(start, i);
      },
    }),
  };
}
