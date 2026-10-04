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
import { NamesSheet, TextsSheet, type LangSaved, type Namings } from "../ui/NameSheet";
import { composing } from "../platform/keys";
import { t } from "../i18n/t";
import { getLang } from "../i18n/lang";
import { tipOf } from "../platform/tips";

/** What a menu row adds: one node type, or several added together, each feeding the next (the last takes the wire). */
type Offer = NodeTypeDef[];

const LOOSE = "_none"; // the rail's row for 未分类: node types the administrator has not placed yet
const NODE_TYPE = "application/x-lab2shot-nodetype"; // a node type in a drag: its id
const SUB_TYPE = "application/x-lab2shot-menusub"; // a subcategory heading in a drag: its id
// the server's limits per language (lab2shot/categories.py, lab2shot/nodes/text.py)
const CATEGORY_MAX = { zh: 10, en: 32 };
const NODE_NAME_MAX = { zh: 30, en: 48 };
const NODE_TEXT_MAX = { zh: 520, en: 1400 };
/** A node menu category's name in every language, as written (the rename sheet's fields). */
const menuWords = (id: string) => adminApi.menuCategories().then((m) => m.words[id]?.label ?? {});
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
 * first, alone and after each node that can feed it (sam_3d_body.detect_people → select_people), as 推荐.
 *
 * Management lives here too, as in the template panel: a login allowed to manage the node categories (menu.edit, worked
 * out by the server) gets these operations in the same menu: categories can be dragged to reorder, renamed and deleted,
 * with 「新建分类」 at the bottom of each band; subcategory headings likewise; dropping a node on a category in the rail
 * or on a subcategory heading on the right files it there; 「未分类」 holds the nodes not yet filed. */
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
  const [naming, setNaming] = useState<Namings | null>(null);
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
    // best match first: the type name or its description in words (either one: 「fbx.import」 and 「导入 FBX」 find the
    // same row), then project / category, then the long description
    const score = (n: NodeTypeDef) => {
      const names = [n.id, n.subtitle.toLowerCase()];
      if (names.some((s) => s === needle)) return 0;
      if (names.some((s) => s.startsWith(needle))) return 1;
      if (names.some((s) => s.includes(needle))) return 2;
      if (n.project.toLowerCase().includes(needle)) return 3;
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
    ? t(menu.wire.side === "source" ? "ui.editor.menu_search_taking" : "ui.editor.menu_search_giving", { type: wireType, count: offered.length })
    : t("ui.editor.menu_search", { count: offered.length });

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
        itemWord: t("ui.editor.menu_item_word"),
        onAdd: (band) => setNaming({ title: t("ui.editor.cat_new"), label: t("ui.editor.cat_name"), max: CATEGORY_MAX,
          save: (name) => act(() => adminApi.saveMenuCategory({ id: nextId("c"), label: name, section: band, rank: tree.length + 1 })) }),
        onRename: (id) => {
          const c = treeOf(id);
          if (c) setNaming({ title: t("ui.editor.cat_rename"), label: t("ui.editor.cat_name"), initial: { [getLang()]: c.label }, load: () => menuWords(id), max: CATEGORY_MAX,
            save: (name) => act(() => adminApi.saveMenuCategory({ id, label: name, color: c.color, rank: c.rank, section: c.section })) });
        },
        onRemove: async (id) => {
          const c = treeOf(id);
          if (!c || !(await ask({ title: t("ui.editor.cat_remove"), say: msg("N-CATEGORY-REMOVE", { name: c.label }), yes: t("ui.common.delete"), tip: tipOf("consequence", t("ui.editor.cat_remove_tip")), danger: true }))) return;
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
  const addSub = () => shownCat && !loosely && setNaming({ title: t("ui.editor.sub_new"), label: t("ui.editor.sub_name"), max: CATEGORY_MAX,
    save: (name) => act(() => adminApi.saveMenuCategory({ id: nextId("s"), parent: shownCat, label: name, rank: (treeOf(shownCat)?.subs.length ?? 0) + 1 })) });
  const renameSub = (id: string) => {
    const s = subOf(id);
    if (s) setNaming({ title: t("ui.editor.sub_rename"), label: t("ui.editor.sub_name"), initial: { [getLang()]: s.label }, load: () => menuWords(id), max: CATEGORY_MAX,
      save: (name) => act(() => adminApi.saveMenuCategory({ id, parent: s.parent, label: name, rank: s.rank })) });
  };
  const removeSub = async (id: string) => {
    const s = subOf(id);
    if (!s || !(await ask({ title: t("ui.editor.sub_remove"), say: msg("N-CATEGORY-REMOVE", { name: s.label }), yes: t("ui.common.delete"), tip: tipOf("consequence", t("ui.editor.sub_remove_tip")), danger: true }))) return;
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
  const editText = async (n: NodeTypeDef, subtitle: LangSaved, description: LangSaved) => {
    try {
      await adminApi.editNodeText(n.id, subtitle, description);
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
            {/* 大字是类型名，旁边小字是它的副标题（节点上最上面那行大字） */}
            <span className="menu-title-text">{o.map((m) => m.id).join(" → ")}</span>
            <span className="menu-title-desc">{o.map((m) => m.subtitle).join(" → ")}</span>
            {o.some((m) => !m.at_defaults.licence.commercial) && <span className="nc-badge">{t("ui.editor.noncommercial")}</span>}
          </div>
          <div className="menu-desc">
            {!available ? nodeWhy(applies, o.find((m) => !nodeUsable(applies, m.id))!.id) : chain ? t("ui.editor.chain", { chain: o.map((m) => m.subtitle).join(" → ") }) : n.description}
          </div>
        </div>
        {tag && <span className="menu-project">{chain ? t("ui.editor.recommended") : `${c.label || t("ui.editor.uncategorized")} · ${n.project}`}</span>}
        {manage && !chain && (
          <IconButton tone="ghost" size="xs" aria-label={t("ui.editor.node_text_edit_of", { node: n.subtitle })} onClick={(e) => (e.stopPropagation(), setEditing(n))}>
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
  const subGroups: { id: string; label: string; nodes: NodeTypeDef[] }[] = (shownTree?.subs ?? [])
    .map((s) => ({ id: s.id, label: s.label, nodes: here.filter((n) => inSub(n) === s.id) }))
    .filter((g) => g.nodes.length > 0 || manage);
  const flat = !manage && subGroups.length === 1 && !looseHere.length;
  const groups = [
    ...(looseHere.length ? [{ id: "", label: "", nodes: looseHere }] : []),
    ...subGroups.map((g) => (flat ? { ...g, label: "" } : g)),
  ];

  const railBands: RailBand[] = [
    ...(looseCount || manage ? [{ id: "loose", managed: manage, fixedOnly: true, rows: [{ id: LOOSE, label: t("ui.editor.uncategorized"), count: looseCount, glyph: <CategoryGlyph category="" color={NEUTRAL} />, fixed: true }] }] : []),
    ...bands.map(({ section, rows }) => ({
      id: section.id, heading: section.label, managed: manage,
      rows: rows.map((c) => ({ id: c.id, label: c.label, count: byCat[c.id]?.length ?? 0, glyph: <CategoryGlyph category={c.id} color={c.color} /> })),
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
            {!results.length && <div className="menu-desc" style={{ padding: 12 }}>{t("ui.editor.menu_no_match")}</div>}
          </div>
        ) : (
          <div className="menu-browse">
            <div className="menu-cats">
              {recent.length > 0 && <div className="menu-cat">{t("ui.editor.recent")}</div>}
              {recent.map((n) => (
                <div key={`r-${n.id}`} className="menu-catrow" onClick={() => pick([n])}>
                  <CategoryGlyph category={nodeCategory(catalog, n).id} color={nodeCategory(catalog, n).color} />
                  <span className="menu-catname">{n.id}</span>
                  <span className="menu-title-desc">{n.subtitle}</span>
                </div>
              ))}
              {/* the bands and their categories are the administrator's tree (Catalog.menu), drawn by the same rail as the
                  templates panel: the same drag, rename, remove and 「新建分类」 there and here */}
              <CategoryRail
                label={t("ui.editor.node_categories")}
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
                  <div className="menu-cat">{t("ui.editor.recommended")}</div>
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
                  <Button tone="ghost" size="sm" onClick={addSub}>
                    <IconPlus /> {t("ui.editor.sub_new")}
                  </Button>
                </div>
              )}
              {!groups.length && <div className="menu-desc" style={{ padding: 12 }}>{loosely ? t("ui.editor.menu_no_loose") : t("ui.editor.menu_no_wirable")}</div>}
            </div>
          </div>
        )}
      </div>
      {naming && <NamesSheet {...naming} onClose={() => setNaming(null)} />}
      {editing && (
        <TextsSheet title={t("ui.editor.node_text_edit_of", { node: editing.id })} nameLabel={t("ui.editor.node_text_name")} textLabel={t("ui.editor.node_text_description")}
          load={() => adminApi.nodeText(editing.id).then((w) => ({ name: w.subtitle, text: w.description }))} nameMax={NODE_NAME_MAX} textMax={NODE_TEXT_MAX}
          save={(subtitle, description) => editText(editing, subtitle, description)} onClose={() => setEditing(null)} />
      )}
      {confirmSheet}
    </>
  );
}

/** One subcategory band of the node list: its heading (drags to reorder, takes a dropped node, has its own menu when
 * managed) and its rows. */
function MenuBand({ group, manage, dropWhere, onDropNode, onDropSub, onRename, onRemove, children }: {
  group: { id: string; label: string; nodes: NodeTypeDef[] };
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
          draggable={manage && !!group.id}
          onDragStart={(e) => {
            e.dataTransfer.setData(SUB_TYPE, group.id);
            e.dataTransfer.effectAllowed = "move";
          }}
        >
          <span>{group.label || t("ui.editor.sub_direct")}</span>
          {manage && <span className="menu-count">{group.nodes.length}</span>}
          {manage && onRename && (
            <IconButton tone="ghost" size="xs" aria-label={t("ui.editor.sub_actions", { name: group.label })} onClick={(e) => (e.stopPropagation(), setMenu({ x: e.clientX, y: e.clientY }))}>
              <IconMore />
            </IconButton>
          )}
        </div>
      )}
      {children}
      {manage && !group.nodes.length && <div className="menu-desc" style={{ padding: "4px 12px 10px" }}>{t("ui.editor.sub_empty")}</div>}
      {menu && onRename && (
        <Menu
          at={menu}
          label={t("ui.editor.sub_actions", { name: group.label })}
          width={160}
          onClose={() => setMenu(null)}
          rows={[
            { key: "rename", label: t("ui.common.rename"), run: onRename },
            { key: "remove", label: t("ui.common.delete"), tip: tipOf("consequence", t("ui.editor.sub_remove_nodes_tip")), run: onRemove },
          ]}
        />
      )}
    </div>
  );
}
