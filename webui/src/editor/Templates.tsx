import { useState } from "react";
import { placeIn, type GraphJSON, type TemplateInfo, type TreeCategory } from "../api";
import { adminApi } from "../api/admin";
import { shown } from "../api/applies";
import { useCatalog } from "../state/catalog";
import { usePreferences } from "../state/preferences";
import { useSession } from "../state/session";
import { useViewer } from "../state/viewer";
import { CategoryRail, FilterRow, Filters, type RailBand, type RailManage } from "../ui/Categories";
import { Button, Chip, IconButton } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
import { Empty } from "../ui/Empty";
import { IconMore, IconPlus } from "../ui/icons";
import { render } from "../messages/format";
import { msg, reasonOf } from "../messages/message";
import { Loading } from "../ui/Loading";
import { Menu } from "../ui/Menu";
import { NameSheet, TextSheet, type Naming } from "../ui/NameSheet";
import { Sheet } from "../ui/Sheet";
import { say } from "../state/say";
import { MyTemplateCards, MyTemplatesState, useMyTemplates } from "./MyTemplates";
import { refreshTemplates, useTemplates } from "./templatesList";
import "./templates.css";

export { refreshTemplates, useTemplates } from "./templatesList";

// 「我的模板」一段中的两项：自己保存的节点图与回收站中的节点图。它们不属于交付物分类，
// 因此在分类栏中单独成段，不混入服务器提供的分类树。
const MINE = "mine";
const BIN = "bin";
const LOOSE = "_none"; // the rail's row for 未分类 (a card whose file names no place, or a place the tree no longer has)
const CARD_TYPE = "application/x-lab2shot-template"; // a card in a drag: its template id
const SUB_TYPE = "application/x-lab2shot-subcategory"; // a subcategory heading in a drag: its id
const ADMIN = "admin"; // TemplateInfo.owner of a project preset (lab2shot/library.py): the one kind an administrator may delete
const LOOSE_CAT: TreeCategory = { id: "", label: "未分类", tip: "还没有归到任何分类的模板：管理员把它拖到左边的分类上", color: "#8E8E93", rank: 0, section: "", subs: [] };

/** 从模板新建: the first-level categories down the left with how many cards each holds, and the chosen one's cards on the
 * right in sections of its subcategories.
 *
 * Every word of the tree comes from the server (/api/templates: the administrator's file templates/_categories.json,
 * lab2shot/categories.py). Nothing about a category is written here or anywhere in code: where a card sits is its own
 * file's word (meta.deliverable), and a card whose file names no place (a newly saved preset, or one whose category was
 * removed) is 未分类 until a manager drags it somewhere.
 *
 * 管理功能同样位于此处，而非后台：具有模板管理权限的登录（由服务器计算的 templates.create，此处不检查角色）可使用以下操作，
 * 其他使用者均不可见：
 *   - 卡片可拖到左栏的一级分类或右侧的二级分类标题上以归入该分类；卡片右上角菜单提供开 / 关、复制、属性、删除；
 *   - 左栏的一级分类可拖动排序、重命名、删除，底部为「新建分类」；
 *   - 右侧的二级分类标题可拖动排序、重命名、删除，末尾为「新建二级分类」；
 *   - 文件菜单中的「保存为预设模板」只需填写名称（Chrome.tsx），保存的卡片位于「未分类」，再拖入某个分类。 */

/** Which categories the templates panel shows, each with how many cards it holds. */
function railOf(tree: TreeCategory[], list: TemplateInfo[], every: boolean): TreeCategory[] {
  // a manager sees every category (an empty one is a drop target, a new one is empty); everyone else only the ones with cards
  return tree.filter((c) => every || list.some((t) => t.category === c.id));
}

export function TemplatesSheet({ onOpen }: { onOpen: (g: GraphJSON) => void }) {
  const open = useViewer((s) => s.templatesOpen);
  const setOpen = useViewer((s) => s.setTemplatesOpen);
  const [again, setAgain] = useState(0);
  const page = useTemplates(open ? `${open}:${again}` : null); // 每次打开模板面板时重新读取（刚保存的卡片打开即可见）
  const catalog = useCatalog();
  const group = usePreferences((s) => s.browseGroup);
  const setGroup = usePreferences((s) => s.setBrowseGroup);
  const [typed, setTyped] = useState("");
  // which project the help page asked to show ("" none): the cards are narrowed to those that use it, with one chip
  // saying so and taking it off again
  const [project, setProject] = useState("");
  const my = useMyTemplates(open); // 每次打开模板面板时重新读取「我的模板」
  const applies = useSession((st) => st.state)?.applies;
  const manage = shown(applies, "templates.create"); // this login manages the templates: the server says so
  const [naming, setNaming] = useState<Naming | null>(null);
  const [ask, confirmSheet] = useConfirm();
  const [menu, setMenu] = useState<{ t: TemplateInfo; at: { x: number; y: number } } | null>(null); // a card's menu: every hook before the early returns below
  const [props, setProps] = useState<TemplateInfo | null>(null); // the card whose properties are open
  const [editing, setEditing] = useState<TemplateInfo | null>(null); // the card whose name and intro are being edited

  if (!open) return null;
  if (!catalog || !page) {
    return (
      <Sheet title="从模板新建" onClose={() => setOpen(false)}>
        <Loading what="模板" />
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
  const own = group === MINE || group === BIN; // 「我的模板」一段：自己保存的节点图，不属于交付物分类
  const chosen = rail.find((c) => c.id === group) ?? (group === LOOSE && (manage || looseCount) ? LOOSE_CAT : own ? undefined : rail[0]);
  const cat = chosen ?? null;
  const loosely = cat?.id === ""; // 未分类 is the chosen "category"
  // 前台不设回收站（对使用者而言删除即删除）。回收站位于后台：使用者需要恢复时，管理员可在「用户」页
  // 查看其删除的内容，并点击「恢复」找回
  const mineBand = [
    { id: MINE, label: "我的模板", tip: "自己存到服务器上的节点图：换台电脑登录也在", count: my.view?.mine.length ?? 0 },
  ];
  const all = list.filter((t) => t.category === (cat?.id ?? "\0")); // the whole category, whatever the search box or a project chip narrows it to
  const needle = typed.trim().toLowerCase();
  const of = project ? list.find((t) => t.projects.some((p) => p.name === project))?.projects.find((p) => p.name === project) : undefined;
  const mine = all.filter((t) => !project || t.projects.some((p) => p.name === project));
  const found = needle
    ? mine.filter((t) => [t.name, t.intro, ...t.projects.map((p) => p.title)].some((s) => s.toLowerCase().includes(needle)))
    : mine;
  // a manager sees every subcategory of the category (an empty one is a drop target); everyone else only the ones with cards
  const subs = cat ? cat.subs.filter((s) => manage || found.some((t) => t.deliverable === s.id)) : [];
  const loose = found.filter((t) => !subs.some((s) => s.id === t.deliverable));
  // cards placed on the category itself (not in a subcategory) get a band of their own only when there are any: a card is
  // dropped on the category through the rail on the left, so an empty band here would only repeat the category's name
  const sections: { id: string; label: string; tip: string; items: TemplateInfo[] }[] = [
    ...subs.map((s) => ({ ...s, items: found.filter((t) => t.deliverable === s.id) })),
    ...(loose.length ? [{ id: "", label: cat?.label ?? "", tip: cat?.tip ?? "", items: loose }] : []),
  ];

  // ---- the manager's operations (every one asks the server and reads back; the page decides nothing about who may)
  const railManage: RailManage | undefined = manage
    ? {
        itemType: CARD_TYPE,
        itemWord: "模板卡",
        onAdd: () => setNaming({ title: "新建分类", label: "分类的名字", initial: "", save: (name) => act(() => adminApi.saveCategory({ id: nextId("c"), label: name, rank: tree.length + 1 })) }),
        onRename: (id) => {
          const c = treeOf(id);
          if (c) setNaming({ title: "重命名分类", label: "分类的名字", initial: c.label, save: (name) => act(() => adminApi.saveCategory({ id, label: name, tip: c.tip, color: c.color, rank: c.rank })) });
        },
        onRemove: async (id) => {
          const c = treeOf(id);
          if (!c || !(await ask({ title: "删掉分类", say: msg("N-CATEGORY-REMOVE", { name: c.label }), yes: "删掉", tip: "从分类树里去掉它和它的二级分类；模板进「未分类」", danger: true }))) return;
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

  const addSub = () => cat && !loosely && setNaming({ title: "新建二级分类", label: "二级分类的名字", initial: "",
    save: (name) => act(() => adminApi.saveCategory({ id: nextId("s"), parent: cat.id, label: name, rank: (treeOf(cat.id)?.subs.length ?? 0) + 1 })) });
  const renameSub = (id: string) => {
    const s = subOf(id);
    if (s) setNaming({ title: "重命名二级分类", label: "二级分类的名字", initial: s.label, save: (name) => act(() => adminApi.saveCategory({ id, parent: s.parent, label: name, tip: s.tip, rank: s.rank })) });
  };
  const removeSub = async (id: string) => {
    const s = subOf(id);
    if (!s || !(await ask({ title: "删掉二级分类", say: msg("N-CATEGORY-REMOVE", { name: s.label }), yes: "删掉", tip: "从分类树里去掉它；模板进「未分类」", danger: true }))) return;
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
      ...(looseCount || manage ? [{ id: LOOSE, label: LOOSE_CAT.label, tip: LOOSE_CAT.tip, count: looseCount, fixed: true }] : []),
      ...rail.map((c) => ({ id: c.id, label: c.label, tip: c.tip, color: c.color, count: list.filter((t) => t.category === c.id).length })),
    ] },
    { id: "mine", rows: mineBand },
  ];

  return (
    <Sheet title="从模板新建" width={1180} height={760} bare onClose={() => setOpen(false)}>
      <div className="tpl-browse">
        <CategoryRail
          label="模板分类"
          title="新建节点图"
          chosen={own ? group : loosely ? LOOSE : (cat?.id ?? "")}
          onChoose={setGroup}
          bands={bands}
          manage={railManage}
        />
        <div className="tpl-main">
          {/* 我的模板 / 回收站：自己保存在服务器上的节点图，不参与镜头筛选 */}
          {own ? (
            <>
              <div className="tpl-head">
                <div className="tpl-head-what">
                  <h2>{mineBand.find((b) => b.id === group)?.label}</h2>
                  <p>{mineBand.find((b) => b.id === group)?.tip}</p>
                </div>
              </div>
              <div className="tpl-cards">
                <MyTemplatesState view={my.view} problem={my.problem} />
                {my.view && (
                  <MyTemplateCards
                    view={my.view}
                    bin={group === BIN}
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
              {manage && <p>{loosely ? "这些模板还没有分类：拖到左边的分类上就归到那里" : "拖动卡片到左边的分类或下面的二级分类标题上就归到那里；卡片右上角的菜单里开关、复制、编辑、看属性、删除"}</p>}
            </div>
            <input
              className="field"
              type="search"
              value={typed}
              aria-label="搜索模板"
              placeholder="搜索模板、项目"
              data-tip="只列出名字、简介或项目名里有这些字的模板"
              onChange={(e) => setTyped(e.target.value)}
            />
          </div>
          {problem && <p className="tpl-problem" role="alert">{problem}</p>}
          {project && (
            <Filters>
              <FilterRow label="项目">
                <Chip size="md" on tip="只看用到这个项目的模板：再点一次取消，看这一类的全部模板" onClick={() => setProject("")}>
                  {of?.title ?? project}
                </Chip>
              </FilterRow>
            </Filters>
          )}
          <div className="tpl-cards">
            {sections.map((s) => (
              <Section
                key={s.id || "loose"}
                section={s}
                heading={(sections.length > 1 || manage) && !loosely}
                manage={manage}
                dropWhere={s.id || cat?.id || ""}
                onDropCard={place}
                onDropSub={reorderSub}
                onRename={s.id ? () => renameSub(s.id) : undefined}
                onRemove={s.id ? () => void removeSub(s.id) : undefined}
              >
                <div className="tpl-grid">
                  {s.items.map((t) => (
                    <Card key={t.id} t={t} catalog={catalog} manage={manage} onMenu={cardMenu} onOpen={(g) => (onOpen(g), setOpen(false))} />
                  ))}
                </div>
              </Section>
            ))}
            {manage && cat && !loosely && (
              <div className="tpl-addsub">
                <Button tip="在这个分类下新建一个二级分类" tone="ghost" size="sm" onClick={addSub}>
                  <IconPlus /> 新建二级分类
                </Button>
              </div>
            )}
            {!found.length && (!manage || loosely) && (
              <Empty title={needle ? "没有符合搜索的模板" : loosely ? "没有未分类的模板" : "这个分类还没有模板"} hint={needle ? "换一个搜索词，或者点左边别的分类" : loosely ? "新存的预设模板和新装的兼容层带来的模板会先出现在这里" : "在节点图里点右键添加节点，自己接一张"} />
            )}
          </div>
          </>
          )}
        </div>
      </div>
      {menu && (
        <Menu
          at={menu.at}
          label={`${menu.t.name} 的操作`}
          width={190}
          onClose={() => setMenu(null)}
          rows={[
            { key: "switch", label: menu.t.enabled === false ? "开启" : "关闭",
              tip: menu.t.enabled === false ? "重新开启：所有账号在「模板」里又看得到它" : render("N-TEMPLATES-OFF", { name: menu.t.name }),
              run: () => void act(() => adminApi.switchTemplate(menu.t.id, menu.t.enabled === false)) },
            { key: "copy", label: "复制", tip: "复制成一张新的预设卡：节点图一样，归同一分类，名字后加「副本」，之后可改可删",
              run: () => void act(async () => { const got = await adminApi.copyTemplate(menu.t.id); say(msg("I-TEMPLATES-COPIED", { name: got.name })); }) },
            { key: "edit", label: "编辑", tip: "改这张卡的名字和简介：写进它自己的文件", run: () => setEditing(menu.t) },
            { key: "props", label: "属性", tip: "这张卡的全部属性：来源、文件、谁创建的、什么时候、分类、用到的项目、许可", run: () => setProps(menu.t) },
            { key: "delete", label: "删除", tip: menu.t.owner === ADMIN ? "删掉这个项目预设：它的文件从 templates/ 里删掉" : "兼容层自带的模板删不了，只能关闭或拖到别的分类", off: menu.t.owner !== ADMIN,
              run: async () => {
                if (!(await ask({ title: "删掉预设模板", say: msg("N-TEMPLATES-DELETE", { name: menu.t.name }), yes: "删掉", tip: "文件从 templates/ 里删掉，找不回来", danger: true }))) return;
                void act(() => adminApi.deleteTemplate(menu.t.id));
              } },
          ]}
        />
      )}
      {naming && <NameSheet {...naming} onClose={() => setNaming(null)} />}
      {props && <PropsSheet t={props} tree={tree} catalog={catalog} onClose={() => setProps(null)} />}
      {editing && (
        <TextSheet title={`编辑「${editing.name}」`} nameLabel="名字" textLabel="简介" initialName={editing.name} initialText={editing.intro} nameMax={40} textMax={240}
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
  section: { id: string; label: string; tip: string; items: TemplateInfo[] };
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
          data-tip={manage ? `${section.tip}\n把模板卡拖到这里就归到这个分类；拖动标题改顺序` : section.tip}
          draggable={manage && !!section.id}
          onDragStart={(e) => {
            e.dataTransfer.setData(SUB_TYPE, section.id);
            e.dataTransfer.effectAllowed = "move";
          }}
        >
          {section.label || "没有二级分类"}
          <b className="tnum">{section.items.length}</b>
          {manage && onRename && (
            <IconButton tip="重命名、删除" tone="ghost" size="xs" aria-label={`${section.label} 的操作`} onClick={(e) => setMenu({ x: e.clientX, y: e.clientY })}>
              <IconMore />
            </IconButton>
          )}
        </h4>
      )}
      {children}
      {manage && !section.items.length && <p className="tpl-empty-band">没有模板：把卡片拖到这个标题上</p>}
      {menu && onRename && (
        <Menu
          at={menu}
          label={`${section.label} 的操作`}
          width={160}
          onClose={() => setMenu(null)}
          rows={[
            { key: "rename", label: "重命名", tip: "改这个二级分类的名字", run: onRename },
            { key: "remove", label: "删除", tip: "删掉这个二级分类：归在它下面的模板进「未分类」，不会跟着删", run: onRemove },
          ]}
        />
      )}
    </section>
  );
}

/** Every property of one card: what the server says of it, laid out as rows, nothing computed here. */
function PropsSheet({ t, tree, catalog, onClose }: { t: TemplateInfo; tree: TreeCategory[]; catalog: NonNullable<ReturnType<typeof useCatalog>>; onClose: () => void }) {
  const at = placeIn(tree, t.deliverable);
  const where = at.id ? [at.label, at.subLabel].filter(Boolean).join(" › ") : t.deliverable ? `未分类（原来归在 ${t.deliverable}，那个分类已经删了）` : "未分类";
  const source = t.owner === ADMIN ? "项目预设（templates/ 文件夹，管理员存的）" : t.owner === "adapter" ? `兼容层自带（${t.adapter} 的兼容层，只读）` : `用户 ${t.owner}`;
  const kb = t.bytes / 1024;
  const rows: [string, string][] = [
    ["名字", t.name],
    ["简介", t.intro || "—"],
    ["来源", source],
    ["文件", t.path],
    ["创建者", t.author || "未记录"],
    ["创建时间", t.created ? t.created.replace("T", " ") : "未记录"],
    ["最后修改", new Date(t.updated * 1000).toLocaleString("zh-CN", { hour12: false })],
    ["分类", where],
    ["状态", t.enabled === false ? "已关闭：普通账号看不到" : "开启"],
    ["用到的项目", t.projects.length ? t.projects.map((p) => p.title).join("、") : "只用核心节点"],
    ["许可", t.licence_word ? `${t.licence_word}${t.licence.length ? `：${t.licence.map((id) => catalog.tags[id]?.label ?? id).join("、")}` : ""}` : "—"],
    ["大小", kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb.toFixed(1)} KB`],
    ["节点数", String(t.graph.nodes.length)],
    ["编号", t.id],
  ];
  return (
    <Sheet title={`${t.name} 的属性`} width={560} onClose={onClose}>
      <dl className="tpl-props">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="dialog-row">
        <Button tip="关上" tone="ghost" onClick={onClose}>关闭</Button>
      </div>
    </Sheet>
  );
}

/** One template card: its name with the project it is built on in the quieter colour, one line of what it does, and
 * one bottom row with the strictest licence word and the year. The projects it uses appear in the card's hover, not on
 * the card (the face carries only the name, one sentence, the licence word and the year). A manager drags the card to
 * another category (the id rides in the drag) and opens its menu. */
function Card({ t, catalog, manage, onMenu, onOpen }: {
  t: TemplateInfo; catalog: NonNullable<ReturnType<typeof useCatalog>>; manage: boolean;
  onMenu: (t: TemplateInfo, at: { x: number; y: number }) => void; onOpen: (g: GraphJSON) => void;
}) {
  // 名称为「交付物 · 项目名」（服务器生成的完整字符串，nodes/templates），卡片上将两部分对调：
  // 主标题为第三方扩展包名称（ViPE、COLMAP、MonST3R 等），副标题为交付物名称。
  const cut = t.name.indexOf(" · ");
  const [what, built] = cut < 0 ? [t.name, ""] : [t.name.slice(cut + 3), ` · ${t.name.slice(0, cut)}`];
  // the graph says which template it came from, so a job submitted from it counts towards this card
  const open = () => onOpen({ ...t.graph, meta: { ...t.graph.meta, template: t.id } });
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
      data-tip={[t.projects.length ? `用到：${t.projects.map((p) => p.title).join("、")}` : "", manage ? "拖到左边的分类或一个二级分类标题上就归到那里" : ""].filter(Boolean).join("\n")}
      onClick={open}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), open())}
    >
      <h5 className="tpl-name">
        <span className="tpl-name-text">
          {what}
          {built && <span className="tpl-built">{built}</span>}
        </span>
        {t.enabled === false && (
          <span className="tpl-off" data-tip={render("N-TEMPLATE-DISABLED")}>
            {render("I-TEMPLATE-DISABLED")}
          </span>
        )}
        {manage && (
          <IconButton tip="开关、复制、编辑、属性、删除" tone="ghost" size="xs" layout="tpl-menu" aria-label={`${t.name} 的操作`}
            onClick={(e) => (e.stopPropagation(), onMenu(t, { x: e.clientX, y: e.clientY }))}>
            <IconMore />
          </IconButton>
        )}
      </h5>
      <p className="tpl-intro">{t.intro}</p>
      <div className="tpl-foot">
        {t.licence_word && (
          <span className={`chip${t.commercial ? " ok" : " nc"}`} data-tip={licenceTip(catalog, t)}>
            {t.licence_word}
          </span>
        )}
        {/* 年份位于该行最右侧 */}
        {t.year != null && <span className="chip year-tag at-end" data-tip="论文 / 发布年份">{t.year}</span>}
      </div>
    </article>
  );
}

/** Why the card carries that one licence word: the tag table's own sentence for each tag it holds (the word itself is
 * the strictest of them, nodes/tags.py strictest, on the server). */
function licenceTip(catalog: NonNullable<ReturnType<typeof useCatalog>>, t: TemplateInfo): string {
  return t.licence.map((id) => catalog.tags[id]).filter(Boolean).map((tag) => `${tag.label}：${tag.tip}`).join("\n");
}
