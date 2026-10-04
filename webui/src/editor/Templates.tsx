/** 从模板新建: the single owner of the template browser. The first-level categories run down the left with how many
 * cards each holds, and the chosen one's cards sit on the right in sections of its subcategories.
 *
 * Every word of the tree comes from the server (/api/templates: the administrator's file templates/_categories.json,
 * lab2shot/categories.py). Nothing about a category is written here or anywhere in code: where a card sits is its own
 * file's word (meta.deliverable), and a card whose file names no place (a newly saved preset, or one whose category was
 * removed) is 未分类 until a manager drags it somewhere.
 *
 * Management lives here as well, not on the admin page: a login that manages templates (templates.create, computed by
 * the server; no role is checked here) gets the following operations, which nobody else sees:
 *   - a card can be dragged onto a first-level category in the left rail or onto a subcategory heading on the right to
 *     place it there; the menu at the card's top right offers on / off, copy, edit, properties and delete;
 *   - first-level categories in the left rail can be dragged to reorder, renamed and deleted; 「新建分类」 is at the bottom;
 *   - subcategory headings on the right can be dragged to reorder, renamed and deleted; 「新建二级分类」 comes last;
 *   - 「保存为预设模板」 in the file menu asks only for a name (Chrome.tsx); the saved card lands in 「未分类」 and is then
 *     dragged into a category. */

import { useState } from "react";
import { placeIn, type GraphJSON, type TemplateInfo, type TreeCategory } from "../api";
import { adminApi } from "../api/admin";
import { shown } from "../api/applies";
import { useCatalog } from "../state/catalog";
import { usePreferences } from "../state/preferences";
import { useSession } from "../state/session";
import { useViewer } from "../state/viewer";
import { CategoryRail, type RailBand, type RailManage } from "../ui/Categories";
import { Button, IconButton } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
import { Empty } from "../ui/Empty";
import { IconMore, IconPlus } from "../ui/icons";
import { render } from "../messages/format";
import { msg, reasonOf } from "../messages/message";
import { Loading } from "../ui/Loading";
import { Menu } from "../ui/Menu";
import { NamesSheet, TextSheet, type Namings } from "../ui/NameSheet";
import { Sheet } from "../ui/Sheet";
import { say } from "../state/say";
import { MyTemplateCards, MyTemplatesState, useMyTemplates } from "./MyTemplates";
import { refreshTemplates, useTemplates } from "./templatesList";
import { t as tr } from "../i18n/t";
import { listSep } from "../i18n/words";
import { getLang } from "../i18n/lang";
import "./templates.css";
import { tipOf } from "../platform/tips";

export { refreshTemplates, useTemplates } from "./templatesList";

// 「我的模板」 band: graphs this account saved itself. They belong to no deliverable category, so they form a band of
// their own in the rail, apart from the server's category tree.
const MINE = "mine";
const LOOSE = "_none"; // the rail's row for 未分类 (a card whose file names no place, or a place missing from the tree)
const CARD_TYPE = "application/x-lab2shot-template"; // a card in a drag: its template id
const SUB_TYPE = "application/x-lab2shot-subcategory"; // a subcategory heading in a drag: its id
const CATEGORY_MAX = { zh: 10, en: 32 }; // the server's limit per language (lab2shot/categories.py)
/** A templates panel category's name in every language, as written (the rename sheet's fields). */
const categoryWords = (id: string) => adminApi.categories().then((r) => r.words?.[id]?.label ?? {});
const ADMIN = "admin"; // TemplateInfo.owner of a project preset (lab2shot/site/library.py): the one kind an administrator may delete
const looseCat = (): TreeCategory => ({ id: "", label: tr("ui.templates.loose"), color: "#8E8E93", rank: 0, section: "", subs: [] });

/** Which categories the templates panel shows, each with how many cards it holds. */
function railOf(tree: TreeCategory[], list: TemplateInfo[], every: boolean): TreeCategory[] {
  // a manager sees every category (an empty one is a drop target, a new one is empty); everyone else only the ones with cards
  return tree.filter((c) => every || list.some((t) => t.category === c.id));
}

export function TemplatesSheet({ onOpen }: { onOpen: (g: GraphJSON) => void }) {
  const open = useViewer((s) => s.templatesOpen);
  const setOpen = useViewer((s) => s.setTemplatesOpen);
  const [again, setAgain] = useState(0);
  const page = useTemplates(open ? `${open}:${again}` : null); // read again every time the panel opens (a card just saved shows at once)
  const catalog = useCatalog();
  const group = usePreferences((s) => s.browseGroup);
  const setGroup = usePreferences((s) => s.setBrowseGroup);
  const [typed, setTyped] = useState("");
  const my = useMyTemplates(open); // 「我的模板」 is read again every time the panel opens
  const applies = useSession((st) => st.state)?.applies;
  const manage = shown(applies, "templates.create"); // this login manages the templates: the server says so
  const [naming, setNaming] = useState<Namings | null>(null);
  const [ask, confirmSheet] = useConfirm();
  const [menu, setMenu] = useState<{ t: TemplateInfo; at: { x: number; y: number } } | null>(null); // a card's menu: every hook before the early returns below
  const [props, setProps] = useState<TemplateInfo | null>(null); // the card whose properties are open
  const [editing, setEditing] = useState<TemplateInfo | null>(null); // the card whose name and intro are being edited

  if (!open) return null;
  if (!catalog || !page) {
    return (
      <Sheet title={tr("ui.templates.title")} onClose={() => setOpen(false)}>
        <Loading what={tr("ui.templates.loading_what")} />
      </Sheet>
    );
  }
  const { templates: list, categories: tree, problem } = page;

  /** Something changed on the server: the cards and the tree are read again (the page carries both). */
  const reload = async () => {
    refreshTemplates();
    setAgain((n) => n + 1);
  };
  const act = async (run: () => Promise<unknown>) => {
    try {
      await run();
    } catch (e) {
      say(msg("E-REQUEST-REFUSED", { status: 0, detail: reasonOf(e as Error) }));
    } finally {
      await reload(); // whatever happened, the page shows what the server now has
    }
  };
  const removeCategory = (id: string) => act(async () => {
    const got = await adminApi.removeCategory(id);
    if (got.problem) say(msg("W-TEMPLATES-STUCK", { detail: got.problem }));
  });
  const nextId = (prefix: string) => `${prefix}${Date.now().toString(36)}`; // lab2shot/categories.py ID: [a-z][a-z0-9_]{1,30}
  const treeOf = (id: string) => tree.find((c) => c.id === id);
  const subOf = (id: string) => tree.flatMap((c) => c.subs.map((s) => ({ ...s, parent: c.id }))).find((s) => s.id === id);

  const rail = railOf(tree, list, manage);
  const looseCount = list.filter((t) => !t.category).length;
  const own = group === MINE; // the 「我的模板」 band: graphs saved by this account, outside the deliverable categories
  const chosen = rail.find((c) => c.id === group) ?? (group === LOOSE && (manage || looseCount) ? looseCat() : own ? undefined : rail[0]);
  const cat = chosen ?? null;
  const loosely = cat?.id === ""; // 未分类 is the chosen "category"
  // There is no recycle bin here (to the user a delete is a delete). The bin is on the admin side: when something must
  // come back, an administrator sees what the user deleted on the 「用户」 page and restores it with 「恢复」
  const mineBand = [{ id: MINE, label: tr("ui.templates.mine"), count: my.view?.mine.length ?? 0 }];
  const all = list.filter((t) => t.category === (cat?.id ?? "\0")); // the whole category
  const needle = typed.trim().toLowerCase();
  const matches = (t: TemplateInfo) => [t.name, t.intro, ...t.projects.map((p) => p.title)].some((s) => s.toLowerCase().includes(needle));
  // a search looks through every category (the person typing a name does not know which category holds it); the
  // results are banded by category › subcategory, and a band drops a card into that subcategory or category as usual
  const found = needle ? list.filter(matches) : all;
  type Band = { key: string; id: string; sub: boolean; label: string; items: TemplateInfo[] };
  const bandsOf = (c: { id: string; label: string; subs: { id: string; label: string }[] } | null, items: TemplateInfo[], prefix: string): Band[] => {
    // a manager sees every subcategory of the chosen category (an empty one is a drop target); everyone else only the
    // ones with cards. Cards placed on the category itself get a band of their own only when there are any: a card is
    // dropped on the category through the rail on the left, so an empty band here would only repeat the category's name
    const subs = c ? c.subs.filter((s) => (manage && !needle) || items.some((t) => t.deliverable === s.id)) : [];
    const loose = items.filter((t) => !subs.some((s) => s.id === t.deliverable));
    return [
      ...subs.map((s) => ({ key: s.id, id: s.id, sub: true, label: prefix + s.label, items: items.filter((t) => t.deliverable === s.id) })),
      ...(loose.length ? [{ key: c?.id || "loose", id: c?.id ?? "", sub: false, label: c?.label ?? tr("ui.templates.loose"), items: loose }] : []),
    ];
  };
  const sections: Band[] = needle
    ? [
        ...tree.flatMap((c) => bandsOf(c, found.filter((t) => t.category === c.id), `${c.label} › `)),
        ...bandsOf(null, found.filter((t) => !t.category), ""),
      ]
    : bandsOf(cat && !loosely ? cat : null, found, "");

  // ---- the manager's operations (every one asks the server and reads back; the page decides nothing about who may)
  const railManage: RailManage | undefined = manage
    ? {
        itemType: CARD_TYPE,
        itemWord: tr("ui.templates.item_word"),
        onAdd: () => setNaming({ title: tr("ui.templates.cat_new"), label: tr("ui.templates.cat_name"), max: CATEGORY_MAX, save: (name) => act(() => adminApi.saveCategory({ id: nextId("c"), label: name, rank: tree.length + 1 })) }),
        onRename: (id) => {
          const c = treeOf(id);
          if (c) setNaming({ title: tr("ui.templates.cat_rename"), label: tr("ui.templates.cat_name"), initial: { [getLang()]: c.label }, load: () => categoryWords(id), max: CATEGORY_MAX, save: (name) => act(() => adminApi.saveCategory({ id, label: name, color: c.color, rank: c.rank })) });
        },
        onRemove: async (id) => {
          const c = treeOf(id);
          if (!c || !(await ask({ title: tr("ui.templates.cat_remove"), say: msg("N-CATEGORY-REMOVE", { name: c.label }), yes: tr("ui.common.delete"), tip: tipOf("consequence", tr("ui.templates.cat_remove_tip")), danger: true }))) return;
          void removeCategory(id);
        },
        onReorder: (id, beforeId) => {
          if (id === LOOSE || beforeId === LOOSE) return; // 未分类 is not a category: it keeps its place
          const order = tree.filter((c) => c.id !== id);
          const at = beforeId ? order.findIndex((c) => c.id === beforeId) : order.length;
          const moved = treeOf(id);
          if (!moved || at < 0) return;
          order.splice(at, 0, moved);
          void act(() => adminApi.orderCategories(order.map((c) => c.id)));
        },
        onDropItem: (id, cardId) => void act(() => adminApi.placeTemplate(cardId, id === LOOSE ? "" : id)),
      }
    : undefined;

  const addSub = () => cat && !loosely && setNaming({ title: tr("ui.templates.sub_new"), label: tr("ui.templates.sub_name"), max: CATEGORY_MAX,
    save: (name) => act(() => adminApi.saveCategory({ id: nextId("s"), parent: cat.id, label: name, rank: (treeOf(cat.id)?.subs.length ?? 0) + 1 })) });
  const renameSub = (id: string) => {
    const s = subOf(id);
    if (s) setNaming({ title: tr("ui.templates.sub_rename"), label: tr("ui.templates.sub_name"), initial: { [getLang()]: s.label }, load: () => categoryWords(id), max: CATEGORY_MAX, save: (name) => act(() => adminApi.saveCategory({ id, parent: s.parent, label: name, rank: s.rank })) });
  };
  const removeSub = async (id: string) => {
    const s = subOf(id);
    if (!s || !(await ask({ title: tr("ui.templates.sub_remove"), say: msg("N-CATEGORY-REMOVE", { name: s.label }), yes: tr("ui.common.delete"), tip: tipOf("consequence", tr("ui.templates.sub_remove_tip")), danger: true }))) return;
    void removeCategory(id);
  };
  const reorderSub = (id: string, beforeId: string | null) => {
    const parent = cat && treeOf(cat.id);
    if (!parent) return;
    const order = parent.subs.filter((s) => s.id !== id);
    const moved = parent.subs.find((s) => s.id === id);
    const at = beforeId ? order.findIndex((s) => s.id === beforeId) : order.length;
    if (!moved || at < 0) return;
    order.splice(at, 0, moved);
    void act(() => adminApi.orderCategories(order.map((s) => s.id), parent.id));
  };
  const place = (cardId: string, where: string) => void act(() => adminApi.placeTemplate(cardId, where));
  const cardMenu = (t: TemplateInfo, at: { x: number; y: number }) => setMenu({ t, at });

  const bands: RailBand[] = [
    { id: "cats", managed: manage, rows: [
      // 未分类 first, so a manager sees at once what still needs a place; for everyone else only when there is something in it
      ...(looseCount || manage ? [{ id: LOOSE, label: tr("ui.templates.loose"), count: looseCount, fixed: true }] : []),
      ...rail.map((c) => ({ id: c.id, label: c.label, color: c.color, count: list.filter((t) => t.category === c.id).length })),
    ] },
    { id: "mine", rows: mineBand },
  ];

  return (
    <Sheet title={tr("ui.templates.title")} width={1180} height={760} bare onClose={() => setOpen(false)}>
      <div className="tpl-browse">
        <CategoryRail
          label={tr("ui.templates.rail")}
          title={tr("ui.templates.rail_title")}
          chosen={own ? group : loosely ? LOOSE : (cat?.id ?? "")}
          onChoose={setGroup}
          bands={bands}
          manage={railManage}
        />
        <div className="tpl-main">
          {/* 我的模板: graphs saved on the server by this account; not part of the search or the project filter */}
          {own ? (
            <>
              <div className="tpl-head">
                <div className="tpl-head-what">
                  <h2>{mineBand.find((b) => b.id === group)?.label}</h2>
                  <p>{tr("ui.templates.mine_intro")}</p>
                </div>
              </div>
              <div className="tpl-cards">
                <MyTemplatesState view={my.view} problem={my.problem} />
                {my.view && (
                  <MyTemplateCards
                    view={my.view}
                    onOpen={(g) => (onOpen(g), setOpen(false))}
                    onChanged={my.set}
                  />
                )}
              </div>
            </>
          ) : (
          <>
          <div className="tpl-head">
            <div className="tpl-head-what">
              <h2>{cat?.label}</h2>
              {manage && <p>{tr(loosely ? "ui.templates.manage_loose" : "ui.templates.manage_hint")}</p>}
            </div>
            <input
              className="field"
              type="search"
              value={typed}
              aria-label={tr("ui.templates.search")}
              placeholder={tr("ui.templates.search_placeholder")}
              onChange={(e) => setTyped(e.target.value)}
            />
          </div>
          {problem && <p className="tpl-problem" role="alert">{problem}</p>}
          <div className="tpl-cards">
            {sections.map((s) => (
              <Section
                key={s.key}
                section={s}
                heading={(sections.length > 1 || manage || !!needle) && !loosely}
                manage={manage && !needle}
                dropWhere={s.id}
                onDropCard={place}
                onDropSub={reorderSub}
                onRename={s.sub ? () => renameSub(s.id) : undefined}
                onRemove={s.sub ? () => void removeSub(s.id) : undefined}
              >
                <div className="tpl-grid">
                  {s.items.map((t) => (
                    <Card key={t.id} t={t} manage={manage} onMenu={cardMenu} onOpen={(g) => (onOpen(g), setOpen(false))} />
                  ))}
                </div>
              </Section>
            ))}
            {manage && cat && !loosely && (
              <div className="tpl-addsub">
                <Button tone="ghost" size="sm" onClick={addSub}>
                  <IconPlus /> {tr("ui.templates.sub_new")}
                </Button>
              </div>
            )}
            {!found.length && (!manage || loosely) && (
              <Empty title={tr(needle ? "ui.templates.empty_search" : loosely ? "ui.templates.empty_loose" : "ui.templates.empty_cat")} hint={tr(needle ? "ui.templates.empty_search_hint" : loosely ? "ui.templates.empty_loose_hint" : "ui.templates.empty_cat_hint")} />
            )}
          </div>
          </>
          )}
        </div>
      </div>
      {menu && (
        <Menu
          at={menu.at}
          label={tr("ui.templates.actions_of", { name: menu.t.name })}
          width={190}
          onClose={() => setMenu(null)}
          rows={[
            { key: "switch", label: tr(menu.t.enabled === false ? "ui.templates.switch_on" : "ui.templates.switch_off"),
              tip: menu.t.enabled === false ? undefined : tipOf("consequence", render("N-TEMPLATES-OFF")),
              run: () => void act(() => adminApi.switchTemplate(menu.t.id, menu.t.enabled === false)) },
            { key: "copy", label: tr("ui.common.copy"),
              run: () => void act(async () => { const got = await adminApi.copyTemplate(menu.t.id); say(msg("I-TEMPLATES-COPIED", { name: got.name })); }) },
            { key: "edit", label: tr("ui.common.edit"), run: () => setEditing(menu.t) },
            { key: "props", label: tr("ui.templates.props"), run: () => setProps(menu.t) },
            { key: "delete", label: tr("ui.common.delete"), tip: menu.t.owner === ADMIN ? tipOf("consequence", tr("ui.templates.delete_tip")) : tipOf("disabled", tr("ui.templates.delete_adapter")), off: menu.t.owner !== ADMIN,
              run: async () => {
                if (!(await ask({ title: tr("ui.templates.delete_title"), say: msg("N-TEMPLATES-DELETE", { name: menu.t.name }), yes: tr("ui.common.delete"), tip: tipOf("consequence", tr("ui.templates.delete_confirm_tip")), danger: true }))) return;
                void act(() => adminApi.deleteTemplate(menu.t.id));
              } },
          ]}
        />
      )}
      {naming && <NamesSheet {...naming} onClose={() => setNaming(null)} />}
      {props && <PropsSheet t={props} tree={tree} catalog={catalog} onClose={() => setProps(null)} />}
      {editing && (
        <TextSheet title={tr("ui.templates.edit_title", { name: editing.name })} nameLabel={tr("ui.templates.name")} textLabel={tr("ui.templates.intro")} initialName={editing.name} initialText={editing.intro} nameMax={60} textMax={240}
          save={async (name, intro) => {
            try {
              await adminApi.editTemplate(editing.id, name, intro);
            } catch (e) {
              say(msg("E-REQUEST-REFUSED", { status: 0, detail: reasonOf(e as Error) }));
              throw e; // the sheet stays open with the words (ui/NameSheet.tsx TextSheet)
            }
            await reload();
          }} onClose={() => setEditing(null)} />
      )}
      {confirmSheet}
    </Sheet>
  );
}

/** One subcategory band: its heading (drags to reorder, takes a dropped card, has its own menu when managed) and its cards. */
function Section({ section, heading, manage, dropWhere, onDropCard, onDropSub, onRename, onRemove, children }: {
  section: { id: string; label: string; items: TemplateInfo[] };
  heading: boolean;
  manage: boolean;
  dropWhere: string; // where a card dropped on this heading goes (the subcategory, or the category itself for the loose band)
  onDropCard: (cardId: string, where: string) => void;
  onDropSub: (id: string, beforeId: string | null) => void;
  onRename?: () => void;
  onRemove?: () => void;
  children: React.ReactNode;
}) {
  const [over, setOver] = useState(false);
  const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
  const accepts = (e: React.DragEvent) => manage && (e.dataTransfer.types.includes(CARD_TYPE) || (!!section.id && e.dataTransfer.types.includes(SUB_TYPE)));
  return (
    <section
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
        const card = e.dataTransfer.getData(CARD_TYPE);
        const sub = e.dataTransfer.getData(SUB_TYPE);
        if (card) onDropCard(card, dropWhere);
        else if (sub && sub !== section.id) onDropSub(sub, section.id);
      }}
    >
      {heading && (
        <h4
          className="sec-title"
          draggable={manage && !!section.id}
          onDragStart={(e) => {
            e.dataTransfer.setData(SUB_TYPE, section.id);
            e.dataTransfer.effectAllowed = "move";
          }}
        >
          {section.label || tr("ui.templates.no_sub")}
          <b className="tnum">{section.items.length}</b>
          {manage && onRename && (
            <IconButton tone="ghost" size="xs" aria-label={tr("ui.templates.actions_of", { name: section.label })} onClick={(e) => setMenu({ x: e.clientX, y: e.clientY })}>
              <IconMore />
            </IconButton>
          )}
        </h4>
      )}
      {children}
      {manage && !section.items.length && <p className="tpl-empty-band">{tr("ui.templates.empty_band")}</p>}
      {menu && onRename && (
        <Menu
          at={menu}
          label={tr("ui.templates.actions_of", { name: section.label })}
          width={160}
          onClose={() => setMenu(null)}
          rows={[
            { key: "rename", label: tr("ui.common.rename"), run: onRename },
            { key: "remove", label: tr("ui.common.delete"), tip: tipOf("consequence", tr("ui.templates.sub_remove_menu_tip")), run: onRemove },
          ]}
        />
      )}
    </section>
  );
}

/** Every property of one card: what the server says of it, laid out as rows, nothing computed here. */
function PropsSheet({ t, tree, catalog, onClose }: { t: TemplateInfo; tree: TreeCategory[]; catalog: NonNullable<ReturnType<typeof useCatalog>>; onClose: () => void }) {
  const at = placeIn(tree, t.deliverable);
  const where = at.id ? [at.label, at.subLabel].filter(Boolean).join(" › ") : t.deliverable ? tr("ui.templates.loose_was", { was: t.deliverable }) : tr("ui.templates.loose");
  const source = t.owner === ADMIN ? tr("ui.templates.source_admin") : t.owner === "adapter" ? tr("ui.templates.source_adapter", { adapter: t.adapter ?? "" }) : tr("ui.templates.source_user", { user: t.owner });
  const kb = t.bytes / 1024;
  const rows: [string, string][] = [
    [tr("ui.templates.name"), t.name],
    [tr("ui.templates.intro"), t.intro || "—"],
    [tr("ui.templates.prop.source"), source],
    [tr("ui.templates.prop.file"), t.path],
    [tr("ui.templates.prop.author"), t.author || tr("ui.templates.prop.unknown")],
    [tr("ui.templates.prop.created"), t.created ? t.created.replace("T", " ") : tr("ui.templates.prop.unknown")],
    [tr("ui.templates.prop.updated"), new Date(t.updated * 1000).toLocaleString(getLang() === "zh" ? "zh-CN" : "en-US", { hour12: false })],
    [tr("ui.templates.prop.category"), where],
    [tr("ui.templates.prop.state"), tr(t.enabled === false ? "ui.templates.prop.state_off" : "ui.templates.prop.state_on")],
    [tr("ui.templates.prop.projects"), t.projects.length ? t.projects.map((p) => p.title).join(listSep()) : tr("ui.templates.prop.core_only")],
    [tr("ui.templates.prop.licence"), t.licence_word ? (t.licence.length ? tr("ui.templates.prop.licence_list", { word: t.licence_word, tags: t.licence.map((id) => catalog.tags[id]?.label ?? id).join(listSep()) }) : t.licence_word) : "—"],
    [tr("ui.templates.prop.size"), kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`],
    [tr("ui.templates.prop.nodes"), String(t.graph.nodes.length)],
    [tr("ui.templates.prop.id"), t.id],
  ];
  return (
    <Sheet title={tr("ui.templates.props_of", { name: t.name })} width={560} onClose={onClose}>
      <dl className="tpl-props">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="dialog-row">
        <Button tone="ghost" onClick={onClose}>{tr("ui.common.close")}</Button>
      </div>
    </Sheet>
  );
}

/** One template card: its name with the project it is built on in the quieter colour, one line of what it does, and
 * one bottom row with the strictest licence word and the year; no hover (its intro says what it does; the projects it
 * uses are in 属性). A manager drags the card to another category (the id rides in the drag) and opens its menu. */
function Card({ t, manage, onMenu, onOpen }: {
  t: TemplateInfo; manage: boolean;
  onMenu: (t: TemplateInfo, at: { x: number; y: number }) => void; onOpen: (g: GraphJSON) => void;
}) {
  // The name is 「deliverable · project」 (the template file's own full string, templates/*.json); the card swaps the two:
  // the main title is the third-party extension's name (ViPE, COLMAP, MonST3R …), the subtitle the deliverable.
  const cut = t.name.indexOf(" · ");
  const [what, built] = cut < 0 ? [t.name, ""] : [t.name.slice(cut + 3), ` · ${t.name.slice(0, cut)}`];
  // the graph opens as it is: it never records which template it came from (saved again, it is an ordinary graph)
  const open = () => onOpen(t.graph);
  return (
    <article
      className={`tpl-card${t.enabled === false ? " dim" : ""}${manage ? " managed" : ""}`}
      role="button"
      tabIndex={0}
      draggable={manage}
      onDragStart={(e) => {
        e.dataTransfer.setData(CARD_TYPE, t.id);
        e.dataTransfer.effectAllowed = "move";
      }}
      onClick={open}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), open())}
    >
      <h5 className="tpl-name">
        <span className="tpl-name-text">
          {what}
          {built && <span className="tpl-built">{built}</span>}
        </span>
        {t.enabled === false && (
          <span className="tpl-off">
            {render("I-TEMPLATE-DISABLED")}
          </span>
        )}
        {manage && (
          <IconButton tone="ghost" size="xs" layout="tpl-menu" aria-label={tr("ui.templates.actions_of", { name: t.name })}
            onClick={(e) => (e.stopPropagation(), onMenu(t, { x: e.clientX, y: e.clientY }))}>
            <IconMore />
          </IconButton>
        )}
      </h5>
      <p className="tpl-intro">{t.intro}</p>
      {/* 默认的选择下许可更严、换一种选择有更宽的路线时（服务器给的 best_licence_word / best_commercial）：一小行说出来 */}
      {t.best_licence_word && t.best_licence_word !== t.licence_word && (
        <p className="tpl-route">
          {tr(t.best_commercial ? "ui.templates.route_commercial" : "ui.templates.route_other", { word: t.licence_word || "—", best: t.best_licence_word })}
        </p>
      )}
      <div className="tpl-foot">
        {t.licence_word && (
          <span className={`chip${t.commercial ? " ok" : " nc"}`}>
            {t.licence_word}
          </span>
        )}
        {/* the year sits at the far right of the row */}
        {t.year != null && <span className="chip year-tag at-end">{t.year}</span>}
      </div>
    </article>
  );
}
