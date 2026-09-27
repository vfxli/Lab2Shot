import { useRef, useState } from "react";

// 拖动改顺序 的说法，一处写（同一个东西处处叫同一个词）：能拖的手柄、队列里那条（多一句顺序能管多久）、
// 和不能拖时变灰写的原因
export const DRAG_TIP = "拖动改顺序";
export const DRAG_QUEUE_TIP = "拖动改这条队列的顺序；排好的顺序只在这次服务运行期间有效，重启服务后回到公平排队";
export const DRAG_QUEUED_ONLY = "只有排队中的任务能改顺序：正在算的和已经结束的动不了";

/** 拖动改顺序, written once: the parameter table's rows (editor/ParamTable.tsx) and the administrator's queue
 * (ui/QueueTables.tsx) drag the same way, so there is one implementation of it and not one per table.
 *
 * `onDrop(from, to)` is told the two positions in the list as it is drawn; what that means — reordering an array,
 * or asking the server to move a job — is the caller's. `grip(i)` goes on the handle (it alone is `draggable`, so
 * selecting text in a row still works), `row(i)` on the row it may be dropped on, and `over` is the row the pointer
 * is on right now (the caller draws it, class `drag-over`).
 *
 * `can(from, to)`: may this row be dropped there at all (the queue: only onto another waiting job of the same lane).
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
