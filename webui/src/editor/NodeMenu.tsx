import { useEffect, useMemo, useRef, useState } from "react";
import { nodeCategory, type NodeTypeDef, type TreeCategory } from "../api";
import { api } from "../api";
import { adminApi } from "../api/admin";
import { nodeUsable, nodeWhy, shown } from "../api/applies";
import { addChain, addNode } from "../graph/actions";
import { looseFix, looseType, looseTypeLabel, loosePort, makesType } from "../graph/rules";
import { snapshotNow } from "../graph/snapshot";
import { msg, reasonOf } from "../messages/message";
import { setCatalog, useCatalog } from "../state/catalog";
import { usePreferences } from "../state/preferences";
import { say } from "../state/say";
import { useSession } from "../state/session";
import { useViewer } from "../state/viewer";
import { Button, IconButton } from "../ui/Button";
import { CategoryGlyph } from "../ui/CategoryGlyph";
import { CategoryRail, type RailBand, type RailManage } from "../ui/Categories";
import { useConfirm } from "../ui/Confirm";
import { IconMore, IconPlus } from "../ui/icons";
import { Menu } from "../ui/Menu";
import { NameSheet, TextSheet, type Naming } from "../ui/NameSheet";
import { composing } from "../platform/keys";

/** What a menu row adds: one node type, or several added together, each feeding the next (the last takes the wire). */
type Offer = NodeTypeDef[];

const LOOSE = "_none"; // the rail's row for 未分类: node types the administrator has not placed yet
const NODE_TYPE = "application/x-lab2shot-nodetype"; // a node type in a drag: its id
const SUB_TYPE = "application/x-lab2shot-menusub"; // a subcategory heading in a drag: its id
const NEUTRAL = "#8E8E93";

/** How tall the menu is: enough for the whole rail (the two bands' 18 rows) without scrolling it. Browsing takes
 * exactly this height, the same for every category: the box must not grow and shrink under the pointer as one
 * category's list is longer than another's; each column scrolls inside instead. Searching
 * only caps at it, so two hits make a small menu. ui/menu.css .menu.wide keeps the same number. */
const MENU_TALL = 700;

/** Node menu (Tab / right-click / double-click, or a wire let go on empty canvas).
 * Browsing: two bands of categories on the left (the core's above the projects') and the hovered category's nodes
 * on the right, grouped by its subcategories. The bands, their order, their names, their categories, their
 * subcategories and where every node sits are all the server's (Catalog.menu: the administrator's data files
 * menu/categories.json and menu/nodes.json, lab2shot/categories.py): no node list, no project name and no category
 * word is written here. A node type placed nowhere (a newly merged one) is in
 * 「未分类」 until a manager drags it somewhere.
 * Typing: one flat list across both bands, each row tagged with its category and project.
 * For a wire: only the node types it can go to, everywhere in the menu; the one picked gets the wire. A wire out of an
 * input whose usage checks name a node to put in front of it (「选人」 before a solver's people) is offered that node
 * first, alone and after each node that can feed it (sam_3d_body.detect_people → core.select_people), as 推荐.
 *
 * 管理功能同样位于此处，与模板面板一致：具有节点分类管理权限的登录
 * （由服务器计算的 menu.edit）在同一菜单中可使用以下操作：分类可拖动排序、重命名、删除，每个区底部为「新建分类」；二级分类
 * 标题相同；将节点拖到左栏的分类或右侧的二级分类标题上即归入该分类；「未分类」中为尚未归类的节点。 */
export function NodeMenu() {
  const applies = useSession((s) => s.state?.applies);
  const menu = useViewer((s) => s.menu);
  const catalog = useCatalog();
  const openMenu = useViewer((s) => s.openMenu);
  const recentNodes = usePreferences((s) => s.recentNodes);
  const pushRecentNode = usePreferences((s) => s.pushRecentNode);
  const [q, setQ] = useState("");
  const [active, setActive] = useState(0);
  const [cat, setCat] = useState<string | null>(null);
  const [naming, setNaming] = useState<Naming | null>(null);
  const [editing, setEditing] = useState<NodeTypeDef | null>(null); // the node type whose name and description are being edited
  const [ask, confirmSheet] = useConfirm();
  const input = useRef<HTMLInputElement>(null);
  // a hover picks a row only when the pointer really moved: rows that appear under a resting pointer (the menu opened
  // over it, the list changed while typing) must not take the choice Enter makes
  const pointer = useRef<string | null>(null);
  const moved = (e: React.MouseEvent) => {
    const at = `${e.clientX},${e.clientY}`;
    const was = pointer.current;
    pointer.current = at;
    return was !== null && was !== at;
  };
  const manage = shown(applies, "menu.edit"); // this login manages the node menu's categories: the server says so

  // the node types offered: all, or those a wire can go to, with the type the wire stands for, and what is recommended
  // for it (the graph does not change while the menu is open)
  const { offered, wireType, recommended } = useMemo(() => {
    const all = catalog?.nodes ?? [];
    const wire = menu?.wire;
    if (!wire) return { offered: all, wireType: "", recommended: [] as Offer[] };
    const s = snapshotNow();
    const can = all.filter((n) => loosePort(s, wire, n.id));
    const t = looseType(s, wire) ?? "";
    const fix = can.find((n) => n.id === looseFix(s, wire));
    // What is recommended before it: the node types that MAKE this data themselves (rules.ts makesType). A node that
    // only passes data through (「切换」, 「命名」) can still take the wire, but never answers "where does this come from".
    const feeds = fix ? can.filter((n) => n !== fix && makesType(s.catalog, n, t)) : [];
    return { offered: can, wireType: looseTypeLabel(s, wire), recommended: fix ? [[fix], ...feeds.map((n) => [n, fix])] : [] };
  }, [catalog, menu]);
  // A node sits under one category (the administrator's placing), found through the one lookup (api/catalog.ts
  // nodeCategory), never through a second label written here; "" is 未分类.
  const byCat = useMemo(() => {
    const out: Record<string, NodeTypeDef[]> = {};
    for (const n of offered) (out[nodeCategory(catalog, n).id] ??= []).push(n);
    return out;
  }, [catalog, offered]);
  // the rail, in the server's bands and their order. A category with nothing to offer here (a wire that nothing in it
  // can take, an extension not installed) is left out, and the rest keep their order, so a row never moves to a place
  // the user has to hunt for; a manager sees every category (an empty one is a drop target)
  const bands = useMemo(() => {
    if (!catalog) return [];
    const tree = catalog.menu.categories;
    const out = catalog.menu.sections
      .map((section) => ({ section, rows: tree.filter((c) => c.section === section.id && (manage || (byCat[c.id]?.length ?? 0) > 0)) }))
      .filter((b) => b.rows.length > 0 || manage);
    return out;
  }, [catalog, byCat, manage]);
  const looseCount = byCat[""]?.length ?? 0;
  // every category the menu shows, in the order the bands put them (未分类 first: what still needs a place is seen at once)
  const categories = useMemo(() => [...(looseCount || manage ? [LOOSE] : []), ...bands.flatMap((b) => b.rows.map((c) => c.id))], [bands, looseCount, manage]);
  const recent = useMemo(() => (menu ? recentNodes : []).map((id) => offered.find((n) => n.id === id)).filter(Boolean) as NodeTypeDef[], [menu, offered, recentNodes]);

  const results = useMemo((): Offer[] => {
    if (!catalog || !q.trim()) return [];
    const needle = q.trim().toLowerCase();
    // best match first: the name itself, then project / id / category, then the description
    const score = (n: NodeTypeDef) => {
      const label = n.label.toLowerCase();
      if (label === needle) return 0;
      if (label.startsWith(needle)) return 1;
      if (label.includes(needle)) return 2;
      if (n.project.toLowerCase().includes(needle) || n.id.includes(needle)) return 3;
      if ((nodeCategory(catalog, n)?.label ?? "").includes(needle)) return 4;
      if (n.description.toLowerCase().includes(needle)) return 5;
      return -1;
    };
    // what is recommended comes first when any of its nodes matches
    const picked = recommended.filter((o) => o.some((n) => score(n) >= 0));
    const rest = offered
      .filter((n) => !picked.some((o) => o.length === 1 && o[0] === n))
      .map((n) => [n, score(n)] as const)
      .filter(([, sc]) => sc >= 0)
      .sort((a, b) => a[1] - b[1] || categories.indexOf(nodeCategory(catalog, a[0]).id || LOOSE) - categories.indexOf(nodeCategory(catalog, b[0]).id || LOOSE))
      .map(([n]) => [n]);
    return [...picked, ...rest];
  }, [catalog, offered, recommended, q, categories]);

  useEffect(() => {
    setQ("");
    setActive(0);
    setCat(null);
    pointer.current = null;
    if (menu) setTimeout(() => input.current?.focus(), 0);
  }, [menu]);
  useEffect(() => setActive(0), [q]);

  if (!menu || !catalog) return null;
  const tree = catalog.menu.categories;

  const pick = (o: Offer | undefined) => {
    if (!o) return;
    for (const n of o) pushRecentNode(n.id);
    if (o.length > 1 && menu.wire) addChain(o.map((n) => n.id), menu.flowX, menu.flowY, menu.wire);
    else addNode(o[0].id, menu.flowX, menu.flowY, menu.wire);
  };
  const placeholder = menu.wire
    ? `搜索${menu.wire.side === "source" ? "接收" : "给出"}${wireType}的 ${offered.length} 个节点`
    : `搜索 ${offered.length} 个节点：名字、项目、分类`;

  const shownCat = cat ?? categories.find((c) => c === LOOSE ? looseCount > 0 : byCat[c]?.length) ?? categories[0] ?? null;
  const loosely = shownCat === LOOSE;
  const searching = q.trim().length > 0;
  // browsing shows two columns and a sentence per row: wide enough that a description takes two or three lines
  // instead of six, and tall enough that the whole rail is there without scrolling it (18 rows + 2 headings)
  const width = searching ? 560 : 760;
  const height = Math.min(MENU_TALL, window.innerHeight - 100);
  // whole inside the window, a margin from its edges
  const left = Math.max(12, Math.min(menu.x, window.innerWidth - width - 16));
  const top = Math.max(12, Math.min(menu.y, window.innerHeight - height - 12));

  // ---- the manager's operations (every one asks the server, which answers with the whole menu; nothing is decided here)
  const act = async (run: () => Promise<{ sections: unknown; categories: TreeCategory[]; placed: Record<string, string>; problem: string }>) => {
    try {
      const got = await run();
      setCatalog({ ...catalog, menu: got as typeof catalog.menu });
    } catch (e) {
      say(msg("E-REQUEST-REFUSED", { status: 0, detail: reasonOf(e as Error) }));
      api.catalog().then(setCatalog, () => undefined); // whatever happened, the menu shows what the server now has (App.tsx's way)
    }
  };
  const nextId = (prefix: string) => `${prefix}${Date.now().toString(36)}`; // lab2shot/categories.py ID: [a-z][a-z0-9_]{1,30}
  const treeOf = (id: string) => tree.find((c) => c.id === id);
  const subOf = (id: string) => tree.flatMap((c) => c.subs.map((s) => ({ ...s, parent: c.id }))).find((s) => s.id === id);
  const railManage: RailManage | undefined = manage
    ? {
        itemType: NODE_TYPE,
        itemWord: "节点",
        onAdd: (band) => setNaming({ title: "新建分类", label: "分类的名字", initial: "",
          save: (name) => act(() => adminApi.saveMenuCategory({ id: nextId("c"), label: name, section: band, rank: tree.length + 1 })) }),
        onRename: (id) => {
          const c = treeOf(id);
          if (c) setNaming({ title: "重命名分类", label: "分类的名字", initial: c.label, save: (name) => act(() => adminApi.saveMenuCategory({ id, label: name, tip: c.tip, color: c.color, rank: c.rank, section: c.section })) });
        },
        onRemove: async (id) => {
          const c = treeOf(id);
          if (!c || !(await ask({ title: "删掉分类", say: msg("N-CATEGORY-REMOVE", { name: c.label }), yes: "删掉", tip: "从节点菜单里去掉它和它的二级分类；节点进「未分类」", danger: true }))) return;
          void act(() => adminApi.removeMenuCategory(id));
        },
        onReorder: (id, beforeId) => {
          if (id === LOOSE || beforeId === LOOSE) return;
          const moved = treeOf(id);
          const before = beforeId ? treeOf(beforeId) : undefined;
          if (!moved || (before && before.section !== moved.section)) return; // a category stays in its band
          const order = tree.filter((c) => c.section === moved.section && c.id !== id);
          const at = before ? order.findIndex((c) => c.id === before.id) : order.length;
          if (at < 0) return;
          order.splice(at, 0, moved);
          // the whole band's order in one write (its other band keeps its own ranks: the server ranks within `parent` only)
          void act(() => adminApi.orderMenuCategories(order.map((c) => c.id)));
        },
        onDropItem: (id, typeId) => void act(() => adminApi.placeNode(typeId, id === LOOSE ? "" : id)),
      }
    : undefined;
  const addSub = () => shownCat && !loosely && setNaming({ title: "新建二级分类", label: "二级分类的名字", initial: "",
    save: (name) => act(() => adminApi.saveMenuCategory({ id: nextId("s"), parent: shownCat, label: name, rank: (treeOf(shownCat)?.subs.length ?? 0) + 1 })) });
  const renameSub = (id: string) => {
    const s = subOf(id);
    if (s) setNaming({ title: "重命名二级分类", label: "二级分类的名字", initial: s.label, save: (name) => act(() => adminApi.saveMenuCategory({ id, parent: s.parent, label: name, tip: s.tip, rank: s.rank })) });
  };
  const removeSub = async (id: string) => {
    const s = subOf(id);
    if (!s || !(await ask({ title: "删掉二级分类", say: msg("N-CATEGORY-REMOVE", { name: s.label }), yes: "删掉", tip: "从节点菜单里去掉它；节点进「未分类」", danger: true }))) return;
    void act(() => adminApi.removeMenuCategory(id));
  };
  const reorderSub = (id: string, beforeId: string | null) => {
    const parent = shownCat && treeOf(shownCat);
    if (!parent) return;
    const order = parent.subs.filter((s) => s.id !== id);
    const moved = parent.subs.find((s) => s.id === id);
    const at = beforeId ? order.findIndex((s) => s.id === beforeId) : order.length;
    if (!moved || at < 0) return;
    order.splice(at, 0, moved);
    void act(() => adminApi.orderMenuCategories(order.map((s) => s.id), parent.id));
  };
  const placeNode = (typeId: string, where: string) => void act(() => adminApi.placeNode(typeId, where));
  const editText = async (n: NodeTypeDef, label: string, description: string) => {
    try {
      await adminApi.editNodeText(n.id, label, description);
    } catch (e) {
      say(msg("E-REQUEST-REFUSED", { status: 0, detail: reasonOf(e as Error) }));
      throw e; // the sheet stays open with the words (ui/NameSheet.tsx TextSheet)
    }
    api.catalog().then(setCatalog, () => undefined); // the words are on the class now: the catalogue carries them
  };

  const Row = ({ o, i, tag }: { o: Offer; i?: number; tag?: boolean }) => {
    const n = o[0];
    const c = nodeCategory(catalog, n);
    const isActive = i !== undefined && i === active;
    const chain = o.length > 1;
    // Until the session says what this account may use,
    // a row shows itself with its own sentence, never greyed with nothing in place of the reason.
    const available = !applies || o.every((m) => nodeUsable(applies, m.id));
    return (
      <div
        className={`menu-item${isActive ? " active" : ""}${available ? "" : " off"}`}
        draggable={manage && !chain}
        onDragStart={(e) => {
          e.dataTransfer.setData(NODE_TYPE, n.id);
          e.dataTransfer.effectAllowed = "move";
        }}
        onMouseMove={(e) => moved(e) && i !== undefined && setActive(i)}
        onClick={() => pick(o)}
      >
        <CategoryGlyph category={c.id} color={isActive ? "#fff" : c.color} />
        <div style={{ minWidth: 0, flex: 1 }}>
          <div className="menu-title">
            <span className="menu-title-text">{o.map((m) => m.label).join(" → ")}</span>
            {o.some((m) => !m.at_defaults.licence.commercial) && <span className="nc-badge">非商用</span>}
          </div>
          <div className="menu-desc">
            {!available ? nodeWhy(applies, o.find((m) => !nodeUsable(applies, m.id))!.id) : chain ? `一起加上并接好：${o.map((m) => `「${m.label}」`).join("接")}` : n.description}
          </div>
        </div>
        {tag && <span className="menu-project">{chain ? "推荐" : `${c.label || "未分类"} · ${n.project}`}</span>}
        {manage && !chain && (
          <IconButton tip="编辑名字和说明" tone="ghost" size="xs" aria-label={`编辑 ${n.label}`} onClick={(e) => (e.stopPropagation(), setEditing(n))}>
            <IconMore />
          </IconButton>
        )}
      </div>
    );
  };

  // nodes of the shown category, grouped by its subcategories in the tree's own order, the ones placed on the category
  // itself first (no heading). A category with one subcategory and nothing loose is a flat list: a heading that repeats
  // the category's name says nothing. A manager sees every subcategory (an empty one is a drop target).
  const shownTree = loosely ? undefined : bands.flatMap((b) => b.rows).find((c) => c.id === shownCat);
  const here = loosely ? (byCat[""] ?? []) : (byCat[shownCat ?? ""] ?? []);
  const inSub = (n: NodeTypeDef) => nodeCategory(catalog, n).sub;
  const looseHere = here.filter((n) => !inSub(n));
  const subGroups: { id: string; label: string; tip: string; nodes: NodeTypeDef[] }[] = (shownTree?.subs ?? [])
    .map((s) => ({ id: s.id, label: s.label, tip: s.tip, nodes: here.filter((n) => inSub(n) === s.id) }))
    .filter((g) => g.nodes.length > 0 || manage);
  const flat = !manage && subGroups.length === 1 && !looseHere.length;
  const groups = [
    ...(looseHere.length ? [{ id: "", label: "", tip: shownTree?.tip ?? "", nodes: looseHere }] : []),
    ...subGroups.map((g) => (flat ? { ...g, label: "" } : g)),
  ];

  const railBands: RailBand[] = [
    ...(looseCount || manage ? [{ id: "loose", managed: manage, fixedOnly: true, rows: [{ id: LOOSE, label: "未分类", tip: "还没有归到任何分类的节点：管理员拖到一个分类上", count: looseCount, glyph: <CategoryGlyph category="" color={NEUTRAL} />, fixed: true }] }] : []),
    ...bands.map(({ section, rows }) => ({
      id: section.id, heading: section.label, tip: section.tip, managed: manage,
      rows: rows.map((c) => ({ id: c.id, label: c.label, tip: c.tip, count: byCat[c.id]?.length ?? 0, glyph: <CategoryGlyph category={c.id} color={c.color} /> })),
    })),
  ];

  // a sheet asked from the menu (a name, a node's words, a confirmation) sits below the menu's own layer: the menu
  // steps aside while the sheet is open and is back, as it was, when the sheet closes (its state is kept)
  const asking = !!naming || !!editing || !!confirmSheet;
  return (
    <>
      {!asking && <div style={{ position: "fixed", inset: 0, zIndex: 49 }} onMouseDown={() => openMenu(null)} />}
      <div className="menu wide glass" style={{ left, top, width, ...(searching ? { maxHeight: height } : { height }), ...(asking ? { visibility: "hidden" as const } : {}) }}>
        <input
          ref={input}
          value={q}
          data-tip="打字筛选，上下键选，回车加上"
          placeholder={placeholder}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (!searching) {
              if (e.key === "Escape") openMenu(null);
              return;
            }
            if (e.key === "ArrowDown") (setActive((a) => Math.min(a + 1, results.length - 1)), e.preventDefault());
            else if (e.key === "ArrowUp") (setActive((a) => Math.max(a - 1, 0)), e.preventDefault());
            else if (e.key === "Enter" && !composing(e)) pick(results[active]);
            else if (e.key === "Escape") openMenu(null);
          }}
        />
        {searching ? (
          <div className="menu-list">
            {results.map((o, i) => (
              <Row key={o.map((n) => n.id).join(">")} o={o} i={i} tag />
            ))}
            {!results.length && <div className="menu-desc" style={{ padding: 12 }}>没有匹配的节点</div>}
          </div>
        ) : (
          <div className="menu-browse">
            <div className="menu-cats">
              {recent.length > 0 && <div className="menu-cat">最近用过</div>}
              {recent.map((n) => (
                <div key={`r-${n.id}`} className="menu-catrow" onClick={() => pick([n])}>
                  <CategoryGlyph category={nodeCategory(catalog, n).id} color={nodeCategory(catalog, n).color} />
                  <span className="menu-catname">{n.label}</span>
                </div>
              ))}
              {/* the bands and their categories are the administrator's tree (Catalog.menu), drawn by the same rail as the
                  templates panel: the same drag, rename, remove and 「新建分类」 there and here */}
              <CategoryRail
                label="节点分类"
                bands={railBands}
                chosen={shownCat ?? ""}
                onChoose={setCat}
                onHover={setCat}
                manage={railManage}
              />
              {catalog.menu.problem && <div className="menu-desc" style={{ padding: 12 }}>{catalog.menu.problem}</div>}
            </div>
            <div className="menu-list">
              {recommended.length > 0 && (
                <div className="menu-recommended">
                  <div className="menu-cat">推荐</div>
                  {recommended.map((o) => (
                    <Row key={o.map((n) => n.id).join(">")} o={o} />
                  ))}
                </div>
              )}
              {groups.map((g) => (
                <MenuBand key={g.id || "loose"} group={g} manage={manage && !loosely} dropWhere={g.id || shownCat || ""}
                  onDropNode={placeNode} onDropSub={reorderSub} onRename={g.id ? () => renameSub(g.id) : undefined} onRemove={g.id ? () => void removeSub(g.id) : undefined}>
                  {g.nodes.map((n) => (
                    <Row key={n.id} o={[n]} />
                  ))}
                </MenuBand>
              ))}
              {manage && shownCat && !loosely && (
                <div className="menu-addsub">
                  <Button tip="在这个分类下新建一个二级分类" tone="ghost" size="sm" onClick={addSub}>
                    <IconPlus /> 新建二级分类
                  </Button>
                </div>
              )}
              {!groups.length && <div className="menu-desc" style={{ padding: 12 }}>{loosely ? "没有未分类的节点" : "没有能接上的节点"}</div>}
            </div>
          </div>
        )}
      </div>
      {naming && <NameSheet {...naming} onClose={() => setNaming(null)} />}
      {editing && (
        <TextSheet title={`编辑「${editing.label}」`} nameLabel="名字" textLabel="说明" initialName={editing.label} initialText={editing.description} nameMax={30} textMax={520}
          save={(label, description) => editText(editing, label, description)} onClose={() => setEditing(null)} />
      )}
      {confirmSheet}
    </>
  );
}

/** One subcategory band of the node list: its heading (drags to reorder, takes a dropped node, has its own menu when
 * managed) and its rows. */
function MenuBand({ group, manage, dropWhere, onDropNode, onDropSub, onRename, onRemove, children }: {
  group: { id: string; label: string; tip: string; nodes: NodeTypeDef[] };
  manage: boolean;
  dropWhere: string; // where a node dropped on this heading goes (the subcategory, or the category itself for the loose band)
  onDropNode: (typeId: string, where: string) => void;
  onDropSub: (id: string, beforeId: string | null) => void;
  onRename?: () => void;
  onRemove?: () => void;
  children: React.ReactNode;
}) {
  const [over, setOver] = useState(false);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const accepts = (e: React.DragEvent) => manage && (e.dataTransfer.types.includes(NODE_TYPE) || (!!group.id && e.dataTransfer.types.includes(SUB_TYPE)));
  return (
    <div
      className={over ? "drag-over" : undefined}
      onDragOver={(e) => {
        if (!accepts(e)) return;
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node | null)) setOver(false);
      }}
      onDrop={(e) => {
        if (!accepts(e)) return;
        e.preventDefault();
        setOver(false);
        const node = e.dataTransfer.getData(NODE_TYPE);
        const sub = e.dataTransfer.getData(SUB_TYPE);
        if (node) onDropNode(node, dropWhere);
        else if (sub && sub !== group.id) onDropSub(sub, group.id);
      }}
    >
      {(group.label || manage) && (
        <div
          className="menu-cat menu-subhead"
          data-tip={manage ? `${group.tip}\n把节点拖到这里就归到这个分类；拖动标题改顺序` : group.tip}
          draggable={manage && !!group.id}
          onDragStart={(e) => {
            e.dataTransfer.setData(SUB_TYPE, group.id);
            e.dataTransfer.effectAllowed = "move";
          }}
        >
          <span>{group.label || "直接归在这个分类下"}</span>
          {manage && <span className="menu-count">{group.nodes.length}</span>}
          {manage && onRename && (
            <IconButton tip="重命名、删除" tone="ghost" size="xs" aria-label={`${group.label} 的操作`} onClick={(e) => (e.stopPropagation(), setMenu({ x: e.clientX, y: e.clientY }))}>
              <IconMore />
            </IconButton>
          )}
        </div>
      )}
      {children}
      {manage && !group.nodes.length && <div className="menu-desc" style={{ padding: "4px 12px 10px" }}>没有节点：把节点拖到这个标题上</div>}
      {menu && onRename && (
        <Menu
          at={menu}
          label={`${group.label} 的操作`}
          width={160}
          onClose={() => setMenu(null)}
          rows={[
            { key: "rename", label: "重命名", tip: "改这个二级分类的名字", run: onRename },
            { key: "remove", label: "删除", tip: "删掉这个二级分类：归在它下面的节点进「未分类」，不会跟着删", run: onRemove },
          ]}
        />
      )}
    </div>
  );
}
