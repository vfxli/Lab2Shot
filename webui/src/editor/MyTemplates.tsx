import { useEffect, useState } from "react";
import { api, type GraphJSON, type MyTemplates, type SavedGraph } from "../api";
import { fileJSON } from "../graph/actions";
import { useCookInputs } from "../state/cookInputs";
import { Button } from "../ui/Button";
import { Empty } from "../ui/Empty";
import { Loading } from "../ui/Loading";
import { Sheet } from "../ui/Sheet";
import { useConfirm } from "../ui/Confirm";
import { msg, reasonOf } from "../messages/message";
import { say } from "../state/say";
import { agoText, sizeText } from "../platform/format";
import { adminApi } from "../api/admin";
import { Select } from "../ui/Select";
import { refreshTemplates, useTemplates } from "./templatesList";

/** 我的模板：节点图除保存为本机 JSON 文件外，还可保存在服务器上并归属当前账号，在其他计算机登录后仍可使用。
 * 删除的模板进入服务器端回收站，管理员可在管理后台「用户」详情页中查看并恢复。
 *
 * 名称与简介为用户输入的文本，一律按纯文本显示（data-user-data），不解析任何标记。 */

/** 「我的模板」列表的唯一读取与刷新入口。`active` 表示该区域当前可见：每次打开模板面板时重新读取，
 * 因此刚通过「文件」菜单保存的模板在打开面板时即可见。 */
export function useMyTemplates(active: boolean): { view: MyTemplates | null; problem: string; set: (v: MyTemplates) => void } {
  const [view, setView] = useState<MyTemplates | null>(null);
  const [problem, setProblem] = useState("");
  useEffect(() => {
    if (!active) return;
    let live = true;
    api.myTemplates().then(
      (v) => live && (setView(v), setProblem("")),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [active]);
  return { view, problem, set: setView };
}

/** 保存到我的模板：填写名称与简介。仅保存节点图与参数，不包含素材。
 * 超出存储配额时，服务器返回的消息会说明已用量、上限及可释放空间的位置。 */
export function SaveToLibrarySheet({ onClose, onSaved, preset = false }: { onClose: () => void; onSaved?: () => void; preset?: boolean }) {
  // `preset`：文件菜单「保存为预设模板」（有模板管理权限的账号）。存成所有账号可见的卡片；
  // 分类默认为「未分类」，也可在此选择一级分类及其下的二级分类。
  const docName = useCookInputs((s) => s.meta.name);
  const [name, setName] = useState(docName);
  const [intro, setIntro] = useState("");
  const tree = useTemplates(preset ? "save-preset" : null)?.categories ?? [];
  const [top, setTop] = useState("");
  const [sub, setSub] = useState("");
  const subs = tree.find((c) => c.id === top)?.subs ?? [];
  const [problem, setProblem] = useState("");
  const [busy, setBusy] = useState(false);
  const ready = !!name.trim() && !busy;

  const save = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      if (preset) {
        await adminApi.createTemplate({ name: name.trim(), intro: intro.trim(), deliverable: sub || top, graph: fileJSON() as GraphJSON });
        refreshTemplates();
        say(msg("I-LIBRARY-LISTED", { name: name.trim() }));
      } else {
        await api.saveTemplate({ name: name.trim(), intro: intro.trim(), graph: fileJSON() as GraphJSON });
        say(msg("I-LIBRARY-SAVED", { name: name.trim() }));
      }
      onSaved?.();
      onClose();
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet title={preset ? "保存为预设模板" : "保存到我的模板"} width={460} onClose={onClose}>
      <div className="lib-form">
        <label className="login-field">
          <span data-tip={preset ? "卡片上的名字：所有人在「模板」里看到的" : "以后在「模板」的「我的模板」里按这个名字找它"}>名字</span>
          <input
            className="field"
            value={name}
            autoFocus
            maxLength={40}
            aria-label="模板名字"
            data-tip={preset ? "卡片上的名字：所有人在「模板」里看到的" : "以后在「模板」的「我的模板」里按这个名字找它"}
            onChange={(e) => (setName(e.target.value), setProblem(""))}
          />
        </label>
        <label className="login-field">
          <span data-tip={preset ? "卡片上的简介：所有人在「模板」里看到的" : "一句话：这张图是做什么的，给自己以后看"}>简介</span>
          <textarea
            className="field"
            rows={4}
            value={intro}
            maxLength={240}
            aria-label="模板简介"
            data-tip={preset ? "卡片上的简介：所有人在「模板」里看到的" : "一句话：这张图是做什么的，给自己以后看"}
            onChange={(e) => (setIntro(e.target.value), setProblem(""))}
          />
        </label>
        {preset && (
          <div className="login-field">
            <span data-tip="卡片归入的分类；不选则进入「未分类」，之后可在「模板」中拖动调整">分类</span>
            <div className="lib-pick">
              <Select label="一级分类" tip="一级分类" value={top}
                options={[{ value: "", label: "未分类" }, ...tree.map((c) => ({ value: c.id, label: c.label }))]}
                onPick={(v) => (setTop(v), setSub(""))} />
              <Select label="二级分类" tip={top ? "二级分类；不选则直接归入所选的一级分类" : "先选择一级分类"} value={sub} disabled={!top || !subs.length}
                options={[{ value: "", label: top ? "（不选二级分类）" : "—" }, ...subs.map((c) => ({ value: c.id, label: c.label }))]}
                onPick={setSub} />
            </div>
          </div>
        )}
        <p className="lib-note">{preset ? "保存为所有账号可见的预设模板。仅保存节点图与参数，不包含素材；分类可随时在「模板」中调整。" : "只存节点图和参数，素材不随模板保存：用的时候自己选文件。存在服务器上，换台电脑登录也在。"}</p>
        {problem && (
          <p className="login-problem" role="alert">
            {problem}
          </p>
        )}
        <div className="dialog-row">
          <Button tip="不存了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={ready ? (preset ? "存成预设模板" : "存到我的模板") : "先给它起个名字"} tone="primary" disabled={!ready} onClick={() => void save()}>
            {busy ? "保存中…" : "保存"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

/** 「我的模板」区域的卡片：用户保存的模板，以及回收站中的模板（灰显，仅提供「恢复」）。 */
export function MyTemplateCards({ view, bin, onOpen, onChanged }: {
  view: MyTemplates;
  bin: boolean;
  onOpen: (g: GraphJSON) => void;
  onChanged: (v: MyTemplates) => void;
}) {
  const [busy, setBusy] = useState("");
  const [ask, confirmSheet] = useConfirm();
  const items = bin ? view.bin : view.mine;

  const open = async (g: SavedGraph) => {
    setBusy(g.id);
    try {
      const got = await api.openTemplate(g.id);
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
    if (!(await ask({ title: "删掉模板", say: msg("N-LIBRARY-BIN", { name: g.name }), yes: "删掉", tip: "从「我的模板」里拿走；删错了找管理员，他在后台还能帮你恢复" }))) return;
    await act(g, () => api.binTemplate(g.id));
  };

  if (!items.length) {
    return (
      <Empty
        title={bin ? "回收站是空的" : "还没有存过自己的模板"}
        hint={bin ? "在「我的模板」里删掉的会先放到这里。" : "搭好一张节点图，在「文件」菜单里点「保存到我的模板」。"}
      />
    );
  }
  return (
    <div className="tpl-grid">
      {items.map((g) => (
        <article key={g.id} className={`tpl-card lib-card${bin ? " dim" : ""}`}>
          <h5 className="tpl-name">
            <span className="tpl-name-text" data-user-data data-tip={g.name}>
              {g.name}
            </span>
          </h5>
          <p className="tpl-intro" data-user-data data-tip={g.intro}>
            {g.intro || "没有简介"}
          </p>
          <div className="lib-meta dim tnum">
            <span data-tip={`这张模板占 ${sizeText(g.bytes)}，算在你的磁盘配额里`}>{sizeText(g.bytes)}</span>
            <span data-tip={bin ? "放进回收站的时间" : "最近一次保存的时间"}>{agoText(bin ? (g.deleted ?? g.updated) : g.updated)}</span>
          </div>
          <div className="lib-acts">
            {bin ? (
              <Button tip="拿回「我的模板」" tone="ghost" size="sm" disabled={busy === g.id} onClick={() => void act(g, () => api.restoreTemplate(g.id))}>
                恢复
              </Button>
            ) : (
              <>
                <Button tip="用这张模板新建一张节点图" tone="primary" size="sm" disabled={busy === g.id} onClick={() => void open(g)}>
                  打开
                </Button>
                <Button tip="删掉这张模板；删错了找管理员，他在后台还能帮你恢复" tone="ghost" size="sm" disabled={busy === g.id} onClick={() => void remove(g)}>
                  删除
                </Button>
              </>
            )}
          </div>
        </article>
      ))}
      {confirmSheet}
    </div>
  );
}

/** 列表读取中与读取失败时的显示。 */
export function MyTemplatesState({ view, problem }: { view: MyTemplates | null; problem: string }) {
  if (problem) return <Empty title={problem} hint="刷新页面再试一次。" />;
  if (!view) return <Loading what="我的模板" />;
  return null;
}
