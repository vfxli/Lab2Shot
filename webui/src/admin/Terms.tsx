import { useEffect, useState } from "react";
import { adminApi, type TermsAdminView } from "../api/admin";
import { shown, usable, why } from "../api/applies";
import { msg, reasonOf } from "../messages/message";
import { whenText } from "../platform/format";
import { useSignedIn } from "../state/session";
import { TermsSheet, type TermsDoc } from "../terms";
import { Button } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";

/** 用户协议与隐私政策, a card of 注册设置 (lab2shot/terms, server/terms.py): both texts as written, to read and edit,
 * with a preview of each as a person reads it (the placeholders filled in). Saving keeps the administrator's copy
 * (under the server's work folder), which is then the one in effect; 恢复自带 goes back to the program's own texts.
 * Either, when the texts change, makes the next version, which every account but the owner's is asked to agree to
 * before it goes on: both ask first. Whether this card is there is the server's answer (terms.edit through
 * available.py), not a role check here. */

type Texts = Record<"agreement" | "privacy", string>;

const textsOf = (v: TermsAdminView): Texts => ({
  agreement: v.documents.find((d) => d.id === "agreement")?.text ?? "",
  privacy: v.documents.find((d) => d.id === "privacy")?.text ?? "",
});

export function TermsAdminCard() {
  const state = useSignedIn();
  const may = shown(state?.applies, "terms.edit");
  const [view, setView] = useState<TermsAdminView | null>(null);
  const [draft, setDraft] = useState<Texts>({ agreement: "", privacy: "" });
  const [problem, setProblem] = useState("");
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<TermsDoc | null>(null);
  const [ask, confirmSheet] = useConfirm();

  const take = (v: TermsAdminView) => (setView(v), setDraft(textsOf(v)), setProblem(""));
  useEffect(() => {
    if (!may) return;
    let live = true;
    adminApi.terms().then(
      (v) => live && take(v),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [may]);

  if (!may) return null;
  const saved = view ? textsOf(view) : null;
  const dirty = !!saved && (saved.agreement.trim() !== draft.agreement.trim() || saved.privacy.trim() !== draft.privacy.trim());
  const edits = usable(state?.applies, "terms.edit");
  const most = view?.most ?? 0;
  const docs = view?.documents ?? [];
  const blocked = docs.map((d) => [...draft[d.id].trim()].length > most ? `《${d.title}》最多 ${most} 个字` : !draft[d.id].trim() ? `《${d.title}》不能是空的` : "").find(Boolean) ?? "";
  const filled = (text: string) => Object.entries(view?.fills ?? {}).reduce((t, [name, value]) => t.split(name).join(value), text);

  const act = async (run: () => Promise<TermsAdminView>) => {
    setBusy(true);
    try {
      take(await run());
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setBusy(false);
    }
  };
  const save = async () => {
    if (view && (await ask({ title: "保存协议", say: msg("N-TERMS-SAVE", { version: view.version + 1 }), yes: "保存", tip: "保存，所有账号下次使用前重新同意" })))
      await act(() => adminApi.saveTerms(draft));
  };
  const reset = async () => {
    if (await ask({ title: "恢复自带的文字", say: msg("N-TERMS-RESET"), yes: "恢复", tip: "删掉管理员改的这一份，用程序自带的文字", danger: true }))
      await act(() => adminApi.resetTerms());
  };

  return (
    <div className="set-card" data-group="terms">
      <h3>用户协议与隐私政策</h3>
      <p className="adm-lede">
        注册页要勾选「我已阅读并同意」才能注册；管理员建的账号第一次登录时也要先同意。一个版本号管两份文字：保存改过的文字就是新的一版，所有账号（主人的除外）下次打开页面时要重新同意。纯文字，「# 」开头是标题，「## 」是小标题，「- 」是一条，「**字**」加粗。
      </p>
      <div className="set-row set-status">
        <span className="set-label" data-tip="现在生效的是第几版、从什么时候起、是程序自带的文字还是管理员改过的">
          版本
        </span>
        <span className="set-ctl set-static" data-user-data>
          {view ? `第 ${view.version} 版 · ${whenText(view.at)} 起 · ${view.edited ? `管理员改过（${view.by || "管理员"}）` : view.by ? `恢复成程序自带（${view.by}）` : "程序自带"}` : "读取中…"}
        </span>
      </div>
      <div className="set-row set-status">
        <span className="set-label" data-tip="要同意的账号（能用的，主人的除外）里，有几个已经同意了这一版">
          已同意
        </span>
        <span className="set-ctl set-static" data-user-data>
          {view ? `${view.agreed} / ${view.accounts} 个账号同意了这一版` : ""}
        </span>
      </div>
      {docs.map((d) => (
        <div key={d.id} className="set-row tall" data-key={`terms.${d.id}`}>
          <span className="set-label" data-tip={`《${d.title}》的原文：最多 ${most} 个字`}>
            {d.title}
          </span>
          <span className="set-ctl">
            <textarea
              className="field area terms-edit"
              value={draft[d.id]}
              rows={12}
              spellCheck={false}
              aria-label={d.title}
              data-tip={`《${d.title}》的原文：纯文字，最多 ${most} 个字`}
              disabled={!edits}
              onChange={(e) => (setDraft({ ...draft, [d.id]: e.target.value }), setProblem(""))}
            />
          </span>
          <span className="set-tags">
            <span className="set-what">
              <Button tip={`看《${d.title}》在注册页上打开的样子（占位符按现在的设置填好）`} tone="ghost" onClick={() => setPreview({ id: d.id, title: d.title, text: filled(draft[d.id]) })}>
                预览
              </Button>
            </span>
          </span>
        </div>
      ))}
      <div className="set-row set-status tall">
        <span className="set-label" data-tip="文字里可以写这些占位符：用户看到的是按现在的设置填好的值，设置改了跟着变，不算新的一版">
          占位符
        </span>
        <span className="set-ctl terms-fills" data-user-data>
          {Object.entries(view?.fills ?? {}).map(([name, value]) => (
            <span key={name} className="chip" data-tip={`写 ${name}，用户看到的是 ${value}`}>
              {name} = {value}
            </span>
          ))}
        </span>
      </div>
      {(problem || blocked) && <div className="set-why bad">{problem || blocked}</div>}
      <div className="set-row">
        <span className="set-label" />
        <span className="set-ctl">
          {/* Not the page's primary (filled) button: 设置 already has one, and a page has at most one. */}
          <Button tip={why(state?.applies, "terms.edit") || blocked || "保存两份文字：是新的一版，所有账号下次使用前要重新同意"} disabled={busy || !dirty || !!blocked || !edits} onClick={() => void save()}>
            {busy ? "保存中…" : dirty ? "保存协议" : "已保存"}
          </Button>
          <Button tip={why(state?.applies, "terms.edit") || (view?.edited ? "删掉管理员改的这一份，恢复成程序自带的文字" : "现在用的就是程序自带的文字")} tone="ghost" danger disabled={busy || !view?.edited || !edits} onClick={() => void reset()}>
            恢复自带
          </Button>
          {dirty && (
            <Button tip="放弃还没保存的修改" tone="ghost" disabled={busy} onClick={() => view && take(view)}>
              还原
            </Button>
          )}
        </span>
      </div>
      {preview && <TermsSheet doc={preview} onClose={() => setPreview(null)} />}
      {confirmSheet}
    </div>
  );
}
