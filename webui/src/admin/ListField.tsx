import { useEffect, useRef } from "react";
import { Button, IconButton } from "../ui/Button";
import { useRowDrag } from "../ui/rowDrag";
import { composing } from "../platform/keys";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** A setting that is a list of short names (kind "list": 环节, 编译目标架构), edited one row per item: the name, a grip
 * to drag the row to another place, a button to remove it, and 「添加一条」 at the bottom. The name comes first so
 * its field starts where every other setting's control starts. Every list setting uses this one editor.
 *
 * It only edits the rows as typed, blank ones included; what a row may hold, that there is at least one and that no
 * two are the same is checked by the settings page (the server's words) and again by the server when it is saved.
 * `twice`: the rows that repeat an earlier one, marked so the admin sees which. */
export function ListField({ label, items, twice, onChange, shown = {} }: {
  label: string;
  items: string[];
  twice: ReadonlySet<number>;
  onChange: (items: string[]) => void;
  // items that are ids with words of their own (the server's Setting.item_labels: a factory 环节 such as `animation`):
  // shown as those words, not typed over (taken away with ×, an administrator's own typed instead)
  shown?: Readonly<Record<string, string>>;
}) {
  const inputs = useRef<(HTMLInputElement | null)[]>([]);
  const focus = useRef<number | null>(null); // the row to put the cursor in once it is drawn (a row just added)
  useEffect(() => {
    if (focus.current === null) return;
    inputs.current[focus.current]?.focus();
    focus.current = null;
  });
  const drag = useRowDrag((from, to) => {
    const next = [...items];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    onChange(next);
  });
  const add = (at: number) => {
    onChange([...items.slice(0, at), "", ...items.slice(at)]);
    focus.current = at;
  };

  return (
    <div className="set-list" role="list" aria-label={label}>
      {items.map((item, i) => (
        <div key={i} role="listitem" className={`set-list-row${drag.over === i ? " drag-over" : ""}`} {...drag.row(i)}>
          <input
            ref={(el) => void (inputs.current[i] = el)}
            className={`field${twice.has(i) ? " bad" : ""}`}
            value={shown[item] ?? item}
            readOnly={item in shown}
            spellCheck={false}
            aria-label={t("ui.admin.list.row", { label, n: i + 1 })}
            {...tipAttrs(twice.has(i) ? tipOf("error", t("ui.admin.list.repeated")) : undefined)}
            onChange={(e) => onChange(items.map((x, j) => (j === i ? e.target.value : x)))}
            onKeyDown={(e) => {
              if (e.key !== "Enter" || composing(e)) return;
              e.preventDefault();
              add(i + 1); // Enter starts the next row, as in a list typed by hand
            }}
          />
          <span className="set-list-grip" {...drag.grip(i)}>
            ⠿
          </span>
          <IconButton tone="ghost" aria-label={t("ui.admin.list.remove")} onClick={() => onChange(items.filter((_, j) => j !== i))}>
            ×
          </IconButton>
        </div>
      ))}
      <Button tone="ghost" layout="set-list-add" onClick={() => add(items.length)}>
        {t("ui.admin.list.add")}
      </Button>
    </div>
  );
}
