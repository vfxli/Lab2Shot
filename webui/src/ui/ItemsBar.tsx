import { render } from "../messages/format";
import { blocksOf, chainOf, itemsOf, itemWord, useItems, viewPath, type ScopeItem } from "../state/items";
import { useTypes } from "../state/catalog";
import { showItem } from "../graph/actions";
import { useResults } from "../state/results";
import { Select, type SelectOption } from "./Select";
import "./items.css";

/** 条目选择条: the glass pill at the bottom centre of the viewer, with one row per 逐项处理 block containing the shown
 * node, selecting which item the view is on. As a viewer control it sits in the kit beside ui/ZoomBar.tsx and
 * ui/DisplayOptions.tsx (webui/tests/layers.test.ts: view/ is below ui/, so a component built from kit components
 * cannot live under view/).
 *
 * It is view state (state/items.ts): changing it cooks nothing, marks no result stale and is never undone; it is only
 * sent with the next status request so that the node answers for that item. It appears only while the shown node is
 * inside a block; the blocks come from the status reply (engine/scopes.py) and are never derived from the wires here.
 *
 * 每个块一个固定宽度的下拉框。不使用横排分段控件（人数达到数十时会超出画面），也不按人数在两种形式间切换：
 * 人数从 3 变为 4 时界面形式随之改变，用户难以记住控件位置。
 *
 * 该布局的不变量：
 *   · 本控件以 `position: absolute` 浮于舞台之上，不参与任何排版，因此其变化不会挤压画布；
 *   · 下拉框触发器宽度固定（items.css 的 `.items-select`），不随人名长短或人数变化；
 *     名称过长时在触发器中截断，悬停显示全名（`data-user-data`，只有用户数据允许截断）；
 *   · 展开的列表由菜单组件负责（ui/Menu.tsx：玻璃效果、圆角、超高时自行滚动、贴近窗口边缘时不越界），
 *     条目达到数十条时只有列表变长，控件本身不移动。 */

/** 展开的列表比触发器略宽：列表中需完整显示人名（仅在触发器上截断）。 */
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
 * 列表类结果在此不占任何位置：视图直接绘制列表中的每一条（view/plan.ts 的 `expand`），
 * 因此无需挑选；拆分列表的节点显示的是上游原图，不需要挑选条目的控件。 */
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
