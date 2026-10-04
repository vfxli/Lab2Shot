import { useState, type ReactNode } from "react";
import { Button, IconButton } from "./Button";
import { IconMore, IconPlus } from "./icons";
import { Menu } from "./Menu";
import "./categories.css";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** Catalogue navigation for the site: the first-level categories down the left with their item counts, and rows of
 * filter chips above the list. The template sheet and the node menu both use these components, so a category looks the
 * same wherever it is shown.
 *
 * The chips of a filter row are ui/Button.tsx's Chip (`size="md"`, with the colour of what it filters and its count);
 * this component only lays them out and labels the row. */

interface Category {
  id: string;
  label: string;
  count?: number;
  color?: string; // the category's own colour (the wire colour of what it delivers)
  glyph?: ReactNode; // drawn before the name instead of the colour bar (the node menu's category glyphs)
  fixed?: boolean; // not a category of the tree (未分类): accepts dropped items, but cannot be dragged, renamed or removed
}

/** One band of the rail: its rows, a heading above them when the caller provides one, and whether a manager may modify
 * it (the templates panel's 「我的模板」 band is not managed by anyone). */
export interface RailBand {
  id: string;
  rows: Category[];
  heading?: string;
  managed?: boolean;
  fixedOnly?: boolean; // the band holds only fixed rows (未分类): no 「新建分类」 below it
}

/** The management actions the rail offers (editor/Templates.tsx, editor/NodeMenu.tsx, for a login the server reports as
 * a manager of that tree): add a category to a band, rename or remove one, drag one before another, and accept an item
 * dropped on one (the item's id is carried in the drag as `itemType`). The rail has no notion of roles; the caller decides. */
export interface RailManage {
  itemType: string;
  itemWord: string; // the kind of item dropped, for the 删除 row's tip: 模板卡, 节点
  onAdd: (band: string) => void;
  onRename: (id: string) => void;
  onRemove: (id: string) => void;
  onReorder: (id: string, beforeId: string | null) => void; // dragged `id` is placed before `beforeId` (null: at the end of its band)
  onDropItem: (id: string, itemId: string) => void;
}

const CATEGORY_TYPE = "application/x-lab2shot-category";

/** The rail of first-level categories, in bands separated by spacing (no divider between them). */
export function CategoryRail({ bands, chosen, onChoose, label, title, manage, onHover }: {
  bands: RailBand[];
  chosen: string;
  onChoose: (id: string) => void;
  label: string; // what is being navigated (aria-label): 模板分类, 节点分类
  title?: string; // the sheet's own title above the rail (「新建节点图」 sits above the categories)
  manage?: RailManage;
  onHover?: (id: string) => void; // a pointer resting on a row selects it (the node menu browses by hover)
}) {
  const [over, setOver] = useState<string | null>(null); // the row under the drag ("end:<band>": the end of a band)
  const [menu, setMenu] = useState<{ id: string; at: { x: number; y: number } } | null>(null);
  const accepts = (e: React.DragEvent) => manage && (e.dataTransfer.types.includes(CATEGORY_TYPE) || e.dataTransfer.types.includes(manage.itemType));
  const dropOn = (e: React.DragEvent, id: string | null) => {
    if (!manage) return;
    e.preventDefault();
    setOver(null);
    const item = e.dataTransfer.getData(manage.itemType);
    const moved = e.dataTransfer.getData(CATEGORY_TYPE);
    if (item && id) manage.onDropItem(id, item);
    else if (moved && moved !== id) manage.onReorder(moved, id);
  };
  const row = (c: Category, managed: boolean) => {
    const button = (
      <button
        key={c.id}
        type="button"
        aria-current={c.id === chosen}
        {...tipAttrs(tipOf("truncated", c.label))}
        style={c.color ? ({ ["--c" as string]: c.color }) : undefined}
        onClick={() => onChoose(c.id)}
        onMouseMove={onHover ? () => onHover(c.id) : undefined}
      >
        {c.glyph ?? (c.color && <i />)}
        {c.label}
        {c.count !== undefined && <b>{c.count}</b>}
      </button>
    );
    if (!managed || !manage) return button;
    const takes = (e: React.DragEvent) => c.fixed ? e.dataTransfer.types.includes(manage.itemType) : accepts(e); // 未分类 accepts items only
    return (
      <div
        key={c.id}
        className={`cat-row${over === c.id ? " drag-over" : ""}`}
        draggable={!c.fixed}
        onDragStart={(e) => {
          e.dataTransfer.setData(CATEGORY_TYPE, c.id);
          e.dataTransfer.effectAllowed = "move";
        }}
        onDragOver={(e) => {
          if (!takes(e)) return;
          e.preventDefault();
          setOver(c.id);
        }}
        onDragLeave={() => setOver((d) => (d === c.id ? null : d))}
        onDrop={(e) => dropOn(e, c.id)}
      >
        {button}
        {!c.fixed && (
          <IconButton tone="ghost" size="sm" aria-label={t("ui.misc.category_actions", { label: c.label })} onClick={(e) => setMenu({ id: c.id, at: { x: e.clientX, y: e.clientY } })}>
            <IconMore />
          </IconButton>
        )}
      </div>
    );
  };
  const menuFor = menu && manage && bands.flatMap((b) => (b.managed ? b.rows : [])).find((c) => c.id === menu.id);
  return (
    <nav className="cat-rail" aria-label={label}>
      {title && <h3>{title}</h3>}
      {bands.map((band, i) => (
        <div key={band.id} style={{ display: "contents" }}>
          {i > 0 && <div className="cat-gap" />}
          {band.heading && <h4 className="cat-band">{band.heading}</h4>}
          {band.rows.map((c) => row(c, !!band.managed))}
          {band.managed && !band.fixedOnly && manage && (
            <div
              className={`cat-end${over === `end:${band.id}` ? " drag-over" : ""}`}
              onDragOver={(e) => {
                if (!e.dataTransfer.types.includes(CATEGORY_TYPE)) return;
                e.preventDefault();
                setOver(`end:${band.id}`);
              }}
              onDragLeave={() => setOver((d) => (d === `end:${band.id}` ? null : d))}
              onDrop={(e) => dropOn(e, null)}
            >
              <Button tone="ghost" size="sm" onClick={() => manage.onAdd(band.id)}>
                <IconPlus /> {t("ui.misc.category_new")}
              </Button>
            </div>
          )}
        </div>
      ))}
      {menuFor && manage && (
        <Menu
          at={menu.at}
          label={t("ui.misc.category_actions", { label: menuFor.label })}
          width={160}
          onClose={() => setMenu(null)}
          rows={[
            { key: "rename", label: t("ui.common.rename"), run: () => manage.onRename(menuFor.id) },
            { key: "remove", label: t("ui.common.delete"), tip: tipOf("consequence", t("ui.misc.category_remove_tip", { items: manage.itemWord })), run: () => manage.onRemove(menuFor.id) },
          ]}
        />
      )}
    </nav>
  );
}

/** The rows of filter chips over a list. */
export function Filters({ children }: { children: ReactNode }) {
  return <div className="filters">{children}</div>;
}

/** One labelled row of chips. */
export function FilterRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="filter-row" role="group" aria-label={label}>
      <span className="filter-label">{label}</span>
      {children}
    </div>
  );
}
