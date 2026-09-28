/** Which person a click on the picture picks: the viewer's copy of the server's rule (lab2shot/ops/run.py _items_pick,
 * "at_point", with _box_at). The 2D stage uses it to accept a click of the "person" handle, to light the person under
 * the pointer, and to light the people the picks point at before 计算 (view/handles2d.ts); the cook of 「选人」 runs the
 * server's. `lab2shot check people` runs both on the same boxes and clicks and requires the same answer.
 *
 * The rule: each person's box on the click's frame, or, on a frame where the person has none, the box of the nearest
 * frame that has one (equal distance: the earlier frame); of the boxes that hold the point (edges included), the smallest,
 * then the smaller id, then the earlier in the list. No imports: the check runs this file on its own. */

interface PeopleBoxes {
  people: { id: number; boxes?: Record<string, number[]> }[];
}

/** The person's box on `frame`, else on the nearest frame that has one (null: the person has no box at all). */
function boxAt(boxes: Record<string, number[]> | undefined, frame: number): number[] | null {
  if (!boxes) return null;
  const here = boxes[String(frame)];
  if (here) return here;
  let best: [number, number] | null = null; // [distance, frame]
  for (const k of Object.keys(boxes).map(Number).sort((a, b) => a - b)) {
    const d = Math.abs(k - frame);
    if (best === null || d < best[0]) best = [d, k];
  }
  return best === null ? null : boxes[String(best[1])];
}

/** The id of the person a click at `p` on `frame` picks, or null when no box holds it. */
export function personAt(data: PeopleBoxes | null, frame: number, p: { x: number; y: number }): number | null {
  let best: [number, number, number] | null = null; // [area, id, index]
  (data?.people ?? []).forEach((t, i) => {
    const b = boxAt(t.boxes, frame);
    if (!b || p.x < b[0] || p.x > b[2] || p.y < b[1] || p.y > b[3]) return;
    const key: [number, number, number] = [(b[2] - b[0]) * (b[3] - b[1]), t.id, i];
    if (best === null || key[0] < best[0] || (key[0] === best[0] && (key[1] < best[1] || (key[1] === best[1] && key[2] < best[2])))) best = key;
  });
  return best === null ? null : (best as [number, number, number])[1];
}
