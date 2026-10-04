import { useEffect, useRef, useState } from "react";
import { api, type GraphJSON, type MyTemplates, type SavedGraph, type TreeCategory } from "../api";
import { ApiError } from "../platform/http";
import { fileJSON } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { Button } from "../ui/Button";
import { Empty } from "../ui/Empty";
import { Loading } from "../ui/Loading";
import { Sheet } from "../ui/Sheet";
import { useConfirm } from "../ui/Confirm";
import { messageOf, msg, reasonOf } from "../messages/message";
import { say } from "../state/say";
import { agoText, sizeText } from "../platform/format";
import { adminApi } from "../api/admin";
import { Select } from "../ui/Select";
import { refreshTemplates, useTemplates } from "./templatesList";
import { pick, t as tr } from "../i18n/t";
import { LANGS } from "../i18n/lang";
import { widthCount } from "../ui/NameSheet";
import { tipAttrs, tipOf } from "../platform/tips";

// a card's intro: at most this many full-width characters, by display width (lab2shot/site/library.py TEXT_WIDTH / 2)
const INTRO_MAX = 240;

/** 我的模板：节点图除保存为本机 JSON 文件外，还可保存在服务器上并归属当前账号，在其他计算机登录后仍可使用。
 * 删除的模板进入服务器端回收站，管理员可在管理后台「用户」详情页中查看并恢复。
 *
 * 名称与简介为用户输入的文本，一律按纯文本显示（data-user-data），不解析任何标记。 */

/** 「我的模板」列表的唯一读取与刷新入口。`active` 表示该区域当前可见：每次打开模板面板时重新读取，
 * 因此刚通过「文件」菜单保存的模板在打开面板时即可见。 */
/** Whether a document name is still the default (empty, or the unnamed word in some language). */
const unnamed = (n: string): boolean => !n || LANGS.some((l) => n === tr("ui.common.unnamed", {}, l));

export function useMyTemplates(active: boolean): { view: MyTemplates | null; problem: string; set: (v: MyTemplates) => void } {
  const [view, setView] = useState<MyTemplates | null>(null);
  const [problem, setProblem] = useState("");
  useEffect(() => {
    if (!active) return;
    let live = true;
    api.myTemplates().then(
      (v) => live && (setView(v), setProblem("")),
      (e: Error) => live && (setProblem(reasonOf(e)), say(messageOf(e))),
    );
    return () => {
      live = false;
    };
  }, [active]);
  return { view, problem, set: setView };
}

/** 分类 id（二级或一级，`meta.deliverable`）在树里的位置：[一级, 二级]；树里没有的（已删掉）是「未分类」。 */
function placeOf(tree: TreeCategory[], where: string): [string, string] {
  if (!where) return ["", ""];
  if (tree.some((c) => c.id === where)) return [where, ""];
  const top = tree.find((c) => c.subs?.some((s) => s.id === where));
  return top ? [top.id, where] : ["", ""];
}

/** 保存到我的模板：填写名称与简介。仅保存节点图与参数，不包含素材。
 * 超出存储配额时，服务器返回的消息会说明已用量、上限及可释放空间的位置。
 *
 * 文档是从模板打开的：名字、简介（预设还有分类）预填为那张模板现在的值，可以改。和已有的（预设：管理员自己的项目预设；
 * 我的模板：这个账号自己的）同名时，服务器回 409（E-TEMPLATES-SAMENAME / E-LIBRARY-SAMENAME），这里问一句是否覆盖，
 * 确认后带 `replace` 再发；取消就不存。和扩展包自带的预设同名时服务器直接拒绝（E-TEMPLATES-SAMENAMEREADONLY）。 */
export function SaveToLibrarySheet({ onClose, onSaved, preset = false }: { onClose: () => void; onSaved?: () => void; preset?: boolean }) {
  // `preset`：文件菜单「保存为预设模板」（有模板管理权限的账号）。存成所有账号可见的卡片；
  // 分类默认为「未分类」，也可在此选择一级分类及其下的二级分类。
  // 预填的是文档自己的名字、简介和分类（meta.name / intro / deliverable：从模板打开的文档带着模板文件里的这几项）；
  // 节点图不记来自哪张模板。空白新建的文档（名字还是占位的「未命名」，state/cookInputs.ts）名字留空
  const docMeta = useCookInputs((s) => s.meta);
  const page = useTemplates(preset ? "save-preset" : null);
  const tree = page?.categories ?? [];
  const [name, setName] = useState(() => unnamed(pick(docMeta.name)) ? "" : pick(docMeta.name));
  const [intro, setIntro] = useState(() => pick(docMeta.intro));
  const [top, setTop] = useState("");
  const [sub, setSub] = useState("");
  const touched = useRef(false); // 用户动过表单：分类树即使读到得晚，也不覆盖用户已选的
  const placed = useRef(false);
  useEffect(() => {
    if (!preset || !page || placed.current || touched.current) return;
    placed.current = true;
    const [t, s2] = placeOf(tree, docMeta.deliverable ?? "");
    setTop(t);
    setSub(s2);
  }, [preset, page, tree, docMeta.deliverable]);
  const edit = <T,>(set: (v: T) => void) => (v: T) => ((touched.current = true), set(v), setProblem(""));
  const subs = tree.find((c) => c.id === top)?.subs ?? [];
  const [problem, setProblem] = useState("");
  const [busy, setBusy] = useState(false);
  const [ask, confirmSheet] = useConfirm();
  const introCount = widthCount(intro.trim(), INTRO_MAX);
  const ready = !!name.trim() && !introCount[2] && !busy;

  const send = async (replace: boolean): Promise<boolean> => {
    const entry = { name: name.trim(), intro: intro.trim(), graph: fileJSON() as GraphJSON, replace };
    if (preset) {
      const done = await adminApi.createTemplate({ ...entry, deliverable: sub || top });
      refreshTemplates();
      say(msg(done.replaced ? "I-TEMPLATES-REPLACED" : "I-LIBRARY-LISTED", { name: entry.name }));
    } else {
      const done = await api.saveTemplate(entry);
      say(msg(done.replaced ? "I-LIBRARY-REPLACED" : "I-LIBRARY-SAVED", { name: entry.name }));
    }
    return true;
  };

  const save = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      let saved: boolean;
      try {
        saved = await send(false);
      } catch (e) {
        // 同名：问一句，确认才覆盖；取消就不存（弹窗留着，可以改名再存）
        if (!(e instanceof ApiError && (e.code === "E-TEMPLATES-SAMENAME" || e.code === "E-LIBRARY-SAMENAME"))) throw e;
        const yes = await ask({
          title: tr("ui.templates.replace_title"),
          say: msg(preset ? "N-TEMPLATES-REPLACE" : "N-LIBRARY-REPLACE", { name: name.trim() }),
          yes: tr("ui.templates.replace"),
        });
        saved = yes && (await send(true));
      }
      if (!saved) return;
      onSaved?.();
      onClose();
    } catch (e) {
      setProblem(reasonOf(e as Error));
      say(messageOf(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet title={tr(preset ? "ui.templates.save_preset" : "ui.templates.save_mine")} width={460} onClose={onClose}>
      <div className="lib-form">
        <label className="login-field">
          <span>{tr("ui.templates.name")}</span>
          <input
            className="field"
            value={name}
            autoFocus
            maxLength={60}
            aria-label={tr("ui.templates.name_aria")}
            onChange={(e) => edit(setName)(e.target.value)}
          />
        </label>
        <label className="login-field">
          <span>{tr("ui.templates.intro")}<b className={`field-count${introCount[2] ? " over" : ""}`}>{introCount[0]} / {introCount[1]}</b></span>
          <textarea
            className="field"
            rows={4}
            value={intro}
            maxLength={2 * INTRO_MAX}
            aria-label={tr("ui.templates.intro_aria")}
            onChange={(e) => edit(setIntro)(e.target.value)}
          />
        </label>
        {preset && (
          <div className="login-field">
            <span>{tr("ui.templates.prop.category")}</span>
            <div className="lib-pick">
              <Select label={tr("ui.templates.top_cat")} value={top}
                options={[{ value: "", label: tr("ui.templates.loose") }, ...tree.map((c) => ({ value: c.id, label: c.label }))]}
                onPick={(v) => (edit(setTop)(v), setSub(""))} />
              <Select label={tr("ui.templates.sub_cat")} tip={top ? undefined : tipOf("disabled", tr("ui.templates.top_first"))} value={sub} disabled={!top || !subs.length}
                options={[{ value: "", label: top ? tr("ui.templates.no_sub_pick") : "—" }, ...subs.map((c) => ({ value: c.id, label: c.label }))]}
                onPick={edit(setSub)} />
            </div>
          </div>
        )}
        <p className="lib-note">{tr(preset ? "ui.templates.save_preset_note" : "ui.templates.save_mine_note")}</p>
        {problem && (
          <p className="login-problem" role="alert">
            {problem}
          </p>
        )}
        <div className="dialog-row">
          <Button tone="ghost" onClick={onClose}>
            {tr("ui.common.cancel")}
          </Button>
          <Button tip={ready ? undefined : tipOf("disabled", tr("ui.templates.name_first"))} tone="primary" disabled={!ready} onClick={() => void save()}>
            {busy ? tr("ui.templates.saving") : tr("ui.common.save")}
          </Button>
        </div>
      </div>
      {confirmSheet}
    </Sheet>
  );
}

/** 「我的模板」区域的卡片：用户保存的模板。 */
export function MyTemplateCards({ view, onOpen, onChanged }: {
  view: MyTemplates;
  onOpen: (g: GraphJSON) => void;
  onChanged: (v: MyTemplates) => void;
}) {
  const [busy, setBusy] = useState("");
  const [ask, confirmSheet] = useConfirm();
  const items = view.mine;

  const open = async (g: SavedGraph) => {
    setBusy(g.id);
    try {
      const got = await api.openTemplate(g.id);
      // 与预设一样原样打开（Templates.tsx）：节点图从不记录它来自哪张模板
      onOpen(got.graph);
    } catch (e) {
      say(msg("E-REQUEST-REFUSED", { status: 0, detail: reasonOf(e as Error) }));
    } finally {
      setBusy("");
    }
  };

  const act = async (g: SavedGraph, run: () => Promise<MyTemplates>) => {
    setBusy(g.id);
    try {
      onChanged(await run());
    } catch (e) {
      say(msg("E-REQUEST-REFUSED", { status: 0, detail: reasonOf(e as Error) }));
    } finally {
      setBusy("");
    }
  };

  const remove = async (g: SavedGraph) => {
    // 编辑器不提供恢复入口；误删的模板由管理员在管理后台恢复
    if (!(await ask({ title: tr("ui.templates.mine_delete"), say: msg("N-LIBRARY-BIN", { name: g.name }), yes: tr("ui.common.delete"), tip: tipOf("consequence", tr("ui.templates.mine_delete_tip")) }))) return;
    await act(g, () => api.binTemplate(g.id));
  };

  if (!items.length) {
    return (
      <Empty
        title={tr("ui.templates.mine_empty")}
        hint={tr("ui.templates.mine_empty_hint")}
      />
    );
  }
  return (
    <div className="tpl-grid">
      {items.map((g) => (
        <article key={g.id} className="tpl-card lib-card">
          <h5 className="tpl-name">
            <span className="tpl-name-text" data-user-data {...tipAttrs(tipOf("truncated", g.name))}>
              {g.name}
            </span>
          </h5>
          <p className="tpl-intro" data-user-data {...tipAttrs(tipOf("truncated", g.intro))}>
            {g.intro || tr("ui.templates.no_intro")}
          </p>
          <div className="lib-meta dim tnum">
            <span>{sizeText(g.bytes)}</span>
            <span>{agoText(g.updated)}</span>
          </div>
          <div className="lib-acts">
            <Button tone="primary" size="sm" disabled={busy === g.id} onClick={() => void open(g)}>
              {tr("ui.common.open")}
            </Button>
            <Button tip={tipOf("consequence", tr("ui.templates.mine_delete_tip"))} tone="ghost" size="sm" disabled={busy === g.id} onClick={() => void remove(g)}>
              {tr("ui.common.delete")}
            </Button>
          </div>
        </article>
      ))}
      {confirmSheet}
    </div>
  );
}

/** 列表读取中与读取失败时的显示。 */
export function MyTemplatesState({ view, problem }: { view: MyTemplates | null; problem: string }) {
  if (problem) return <Empty title={problem} hint={tr("ui.templates.reload_hint")} />;
  if (!view) return <Loading what={tr("ui.templates.mine")} />;
  return null;
}
