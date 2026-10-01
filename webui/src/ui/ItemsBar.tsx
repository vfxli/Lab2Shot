import { render } from "../messages/format";
import { blocksOf, chainOf, itemsOf, itemWord, useItems, viewPath, type ScopeItem } from "../state/items";
import { useTypes } from "../state/catalog";
import { showItem } from "../graph/actions";
import { useResults } from "../state/results";
import { Select, type SelectOption } from "./Select";
import "./items.css";

/** 条目选择条: the glass pill at the bottom centre of the viewer, with one row per 逐项处理 block containing the shown
 * node, selecting which item the view is on. As a viewer control it sits in the kit beside ui/ZoomBar.tsx and
 * ui/DisplayOptions.tsx (view/ is below ui/, so a component built from kit components cannot live under view/).
 *
 * It is view state (state/items.ts): changing it cooks nothing, marks no result stale and is never undone; it is only
 * sent with the next status request so that the node answers for that item. It appears only while the shown node is
 * inside a block; the blocks come from the status reply (engine/scopes.py) and are never derived from the wires here.
 *
 * One fixed-width drop-down per block. Not a row of segments (with dozens of people it would run off the screen), and
 * not a switch between two forms by count: if the control changed form when the count went from 3 to 4, users could
 * not remember where it is.
 *
 * Layout invariants:
 *   · the control floats over the stage with `position: absolute` and takes no part in layout, so its changes never
 *     squeeze the canvas;
 *   · the drop-down trigger has a fixed width (items.css `.items-select`) whatever the names' length or count; a long
 *     name is cut in the trigger, with the full name on hover (`data-user-data`: only user data may be cut);
 *   · the open list is the menu component's job (ui/Menu.tsx: glass, rounded, scrolls itself when too tall, never
 *     crosses the window's edge), so with dozens of items only the list grows and the control itself never moves. */

/** The open list is a little wider than the trigger: names show in full in the list (cut only in the trigger). */
const MENU_WIDE = 240;

function ItemPick({ where, name, kind, items, chosen }: { where: string; name: string; kind: string; items: ScopeItem[]; chosen: string }) {
  const index = Math.max(0, items.findIndex((i) => i.key === chosen));
  const options: SelectOption[] = items.map((i) => ({ value: i.key, label: i.name, tip: i.name }));
  return (
    <div className="items-pick">
      <span className="items-kind">{kind}</span>
      <Select
        className="items-select"
        data-user-data
        layout="items-menu"
        value={items[index]?.key ?? ""}
        options={options}
        onPick={(key) => showItem(where, key)}
        label={render("I-EACH-ITEMS")}
        tip={render("I-EACH-PICK", { block: name })}
        width={MENU_WIDE}
      />
      <span className="items-at tnum">{render("I-EACH-ITEMAT", { index: index + 1, total: items.length })}</span>
    </div>
  );
}

/** The bar for the shown node: one row per 逐项处理 block containing it, selecting which item of that block the view is
 * on. When no block contains it, nothing is drawn.
 *
 * A list result takes no place here: the view draws every entry of the list itself (view/plan.ts `expand`), so there
 * is nothing to pick; a node that splits a list shows the upstream picture and needs no item picker. */
export function ItemsBar({ nodeId }: { nodeId: string | null }) {
  const reply = useResults((s) => s.reply);
  const view = useItems((s) => s.view);
  const types = useTypes();
  const blocks = blocksOf(reply);
  const chain = nodeId ? chainOf(blocks, nodeId) : [];
  const path = nodeId ? viewPath(blocks, nodeId, view) : [];
  // the name of one item of this block: the block's 「条目」 output type as resolved by the server (the same value
  // editor/BlockFrame.tsx shows in the frame's title; the page derives none of it)
  const eachOf = (begin: string) =>
    render("I-EACH-EACHOF", { kind: itemWord(types, reply?.nodes[begin]?.ports.outputs.find((p) => p.name === "item")?.type ?? "") });
  const rows = chain.map((scope, n) => ({ where: scope.begin, name: scope.name, kind: eachOf(scope.begin), items: itemsOf(scope, path.slice(0, n)), chosen: path[n] ?? "" }));
  const shown = rows.filter((r) => r.items?.length);
  if (!shown.length) return null;
  return (
    <div className="items-bar hud-pill glass static" data-tip={render("I-EACH-ITEMTIP")}>
      {shown.map(({ where, name, kind, items, chosen }) => (
        <ItemPick key={where} where={where} name={name} kind={kind} items={items!} chosen={chosen} />
      ))}
    </div>
  );
}
