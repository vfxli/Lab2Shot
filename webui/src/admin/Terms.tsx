import { useEffect, useState } from "react";
import { LabelRow } from "../ui/LabelRow";
import { adminApi, type TermsAdminView, type TermsTexts } from "../api/admin";
import { shown, usable, why } from "../api/applies";
import { msg, reasonOf } from "../messages/message";
import { whenText } from "../platform/format";
import { useSignedIn } from "../state/session";
import { TermsSheet, type TermsDoc } from "../terms";
import { Button } from "../ui/Button";
import { useConfirm } from "../ui/Confirm";
import { t } from "../i18n/t";
import { LANGS, type Lang } from "../i18n/lang";
import { tipOf } from "../platform/tips";

/** 用户协议与隐私政策, a card of 注册设置 (lab2shot/terms, server/terms.py): both texts as written, to read and edit,
 * with a preview of each as a person reads it (the placeholders filled in). Saving keeps the administrator's copy
 * (under the server's work folder), which is then the one in effect; 恢复自带 goes back to the program's own texts.
 * Either, when the texts change, makes the next version, which every account but the owner's is asked to agree to
 * before it goes on: both ask first. Whether this card is there is the server's answer (terms.edit through
 * available.py), not a role check here. */

// each document in every interface language: the administrator edits both, and a person reads the one in their
// language (the public /api/terms)
type Texts = Record<"agreement" | "privacy", TermsTexts>;

const EMPTY: TermsTexts = { zh: "", en: "" };
const textsOf = (v: TermsAdminView): Texts => ({
  agreement: { ...EMPTY, ...v.documents.find((d) => d.id === "agreement")?.texts },
  privacy: { ...EMPTY, ...v.documents.find((d) => d.id === "privacy")?.texts },
});
const ENTRIES = (["agreement", "privacy"] as const).flatMap((id) => LANGS.map((lang) => [id, lang] as const));

export function TermsAdminCard() {
  const state = useSignedIn();
  const may = shown(state?.applies, "terms.edit");
  const [view, setView] = useState<TermsAdminView | null>(null);
  const [draft, setDraft] = useState<Texts>({ agreement: EMPTY, privacy: EMPTY });
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
  const dirty = !!saved && ENTRIES.some(([id, lang]) => saved[id][lang].trim() !== draft[id][lang].trim());
  const edits = usable(state?.applies, "terms.edit");
  const most = view?.most ?? 0;
  const docs = view?.documents ?? [];
  const titled = (id: string, lang: Lang) => t("ui.admin.terms.in_lang", { title: docs.find((d) => d.id === id)?.title ?? id, lang: t(`lang.${lang}`) });
  const blocked = ENTRIES.map(([id, lang]) => [...draft[id][lang].trim()].length > most ? t("ui.admin.terms.too_long", { title: titled(id, lang), most }) : !draft[id][lang].trim() ? t("ui.admin.terms.empty", { title: titled(id, lang) }) : "").find(Boolean) ?? "";
  const filled = (text: string) => Object.entries(view?.fills ?? {}).reduce((s, [name, value]) => s.split(name).join(value), text);

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
    if (view && (await ask({ title: t("ui.admin.terms.save"), say: msg("N-TERMS-SAVE", { version: view.version + 1 }), yes: t("ui.admin.common.save") })))
      await act(() => adminApi.saveTerms(draft));
  };
  const reset = async () => {
    if (await ask({ title: t("ui.admin.terms.reset_title"), say: msg("N-TERMS-RESET"), yes: t("ui.admin.terms.reset_yes"), danger: true }))
      await act(() => adminApi.resetTerms());
  };

  return (
    <div className="set-card lgrid" data-group="terms">
      <h3>{t("ui.admin.terms.title")}</h3>
      <p className="adm-lede">
        {t("ui.admin.terms.lede")}
      </p>
      <LabelRow className="set-row set-status" labelClass="set-label" label={t("ui.admin.terms.version")} ctlClass="set-ctl set-static">
        <span data-user-data>{view ? t("ui.admin.terms.version_value", { version: view.version, since: whenText(view.at), source: view.edited ? t("ui.admin.terms.edited", { by: view.by || t("ui.admin.rights.admin") }) : view.by ? t("ui.admin.terms.restored", { by: view.by }) : t("ui.admin.terms.builtin") }) : t("ui.admin.common.reading")}</span>
      </LabelRow>
      <LabelRow className="set-row set-status" labelClass="set-label" label={t("ui.admin.terms.agreed")} ctlClass="set-ctl set-static">
        <span data-user-data>{view ? t("ui.admin.terms.agreed_value", { agreed: view.agreed, accounts: view.accounts }) : ""}</span>
      </LabelRow>
      {docs.flatMap((d) => LANGS.map((lang) => (
        <LabelRow key={`${d.id}.${lang}`} className="set-row tall top" data-key={`terms.${d.id}.${lang}`} labelClass="set-label" label={titled(d.id, lang)} ctlClass="set-ctl"
          tail={<span className="set-tags">
            <span className="set-what">
              <Button tone="ghost" onClick={() => setPreview({ id: d.id, title: titled(d.id, lang), text: filled(draft[d.id][lang]) })}>
                {t("ui.admin.terms.preview")}
              </Button>
            </span>
          </span>}>
            <textarea
              className="field area terms-edit"
              value={draft[d.id][lang]}
              rows={12}
              spellCheck={false}
              aria-label={titled(d.id, lang)}
              disabled={!edits}
              onChange={(e) => (setDraft({ ...draft, [d.id]: { ...draft[d.id], [lang]: e.target.value } }), setProblem(""))}
            />
        </LabelRow>
      )))}
      <LabelRow className="set-row set-status tall" labelClass="set-label" label={t("ui.admin.terms.placeholders")} ctlClass="set-ctl terms-fills">
        <span data-user-data>{Object.entries(view?.fills ?? {}).map(([name, value]) => (
            <span key={name} className="chip">
              {name} = {value}
            </span>
          ))}</span>
      </LabelRow>
      {(problem || blocked) && <div className="set-why bad lrow-under">{problem || blocked}</div>}
      <LabelRow className="set-row" labelClass="set-label" label="" ctlClass="set-ctl">
        {/* Not the page's primary (filled) button: 设置 already has one, and a page has at most one. */}
          <Button tip={why(state?.applies, "terms.edit") || blocked ? tipOf("disabled", why(state?.applies, "terms.edit") || blocked) : tipOf("consequence", t("ui.admin.terms.save_tip"))} disabled={busy || !dirty || !!blocked || !edits} onClick={() => void save()}>
            {busy ? t("ui.admin.common.saving") : dirty ? t("ui.admin.terms.save") : t("ui.admin.common.saved")}
          </Button>
          <Button tip={why(state?.applies, "terms.edit") ? tipOf("disabled", why(state?.applies, "terms.edit")) : view?.edited ? tipOf("consequence", t("ui.admin.terms.reset_tip")) : tipOf("disabled", t("ui.admin.terms.reset_none"))} tone="ghost" danger disabled={busy || !view?.edited || !edits} onClick={() => void reset()}>
            {t("ui.admin.terms.reset")}
          </Button>
          {dirty && (
            <Button tip={tipOf("consequence", t("ui.admin.common.revert_tip"))} tone="ghost" disabled={busy} onClick={() => view && take(view)}>
              {t("ui.admin.common.revert")}
            </Button>
          )}
      </LabelRow>
      {preview && <TermsSheet doc={preview} onClose={() => setPreview(null)} />}
      {confirmSheet}
    </div>
  );
}
