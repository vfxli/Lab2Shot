import { useEffect, useState } from "react";
import { Button } from "./Button";
import { Sheet } from "./Sheet";
import { composing } from "../platform/keys";
import { t } from "../i18n/t";
import { getLang, LANGS, type Lang } from "../i18n/lang";
import { tipAttrs, tipOf } from "../platform/tips";

/** A single name entered in a small sheet (a new category, a rename): the only text input a manager uses here; the
 * templates panel's and the node menu's categories both use it, and a task group's rename (ui/QueueTables.tsx).
 * `max`: the most characters (the server's limit for this kind of name). `empty`: saving an empty name is allowed and
 * means this (the field's placeholder says it); without it a name is required. */
export function NameSheet({ title, label, initial, save, onClose, max = 10, empty }: {
  title: string; label: string; initial: string; save: (name: string) => Promise<void>; onClose: () => void;
  max?: number; empty?: string;
}) {
  const [name, setName] = useState(initial);
  const [busy, setBusy] = useState(false);
  const ready = (!!name.trim() || empty !== undefined) && name.trim() !== initial && !busy;
  const go = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      await save(name.trim());
      onClose();
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={title} width={420} onClose={onClose}>
      <div className="lib-form">
        <label className="login-field">
          <span>{label}</span>
          <input className="field" value={name} autoFocus maxLength={max} aria-label={label} placeholder={empty}
            {...tipAttrs(tipOf("value", t("ui.misc.at_most_chars", { count: max })))}
            onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && !composing(e) && void go()} />
        </label>
        <div className="dialog-row">
          <Button tone="ghost" onClick={onClose}>{t("ui.common.cancel")}</Button>
          <Button tip={ready ? undefined : tipOf("disabled", empty !== undefined ? t("ui.misc.name_same") : t("ui.misc.name_first"))} tone="primary" disabled={!ready} onClick={() => void go()}>{busy ? t("ui.misc.saving") : t("ui.common.save")}</Button>
        </div>
      </div>
    </Sheet>
  );
}

/** The sheet's parameters: the title, the field's label, the initial name and the handler for the entered name. */
export interface Naming {
  title: string;
  label: string;
  initial: string;
  save: (name: string) => Promise<void>;
}

/** How wide a text shows: a full-width character (Chinese, Japanese, Korean, full-width forms) 2, any other 1 — the
 * server's rule (lab2shot/site/library.py width: Unicode East Asian Width W and F). */
export function textWidth(text: string): number {
  let n = 0;
  for (const c of text) n += WIDE.test(c) ? 2 : 1;
  return n;
}
const WIDE = /[\u1100-\u115F\u2E80-\u303E\u3041-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uA000-\uA4CF\uAC00-\uD7A3\uF900-\uFAFF\uFE30-\uFE4F\uFF00-\uFF60\uFFE0-\uFFE6]|[\u{1F300}-\u{1F64F}\u{1F900}-\u{1F9FF}\u{20000}-\u{3FFFD}]/u;

/** A text held to a display width (`most` full-width characters = 2 × `most` half-width ones), counted the way the
 * page's language reads it: in Chinese full-width characters (half-width ones count half), in English characters
 * (full-width ones count two). [shown count, shown limit, too wide]. */
export function widthCount(text: string, most: number): [number, number, boolean] {
  const w = textWidth(text);
  return getLang() === "zh" ? [Math.ceil(w / 2), most, w > 2 * most] : [w, 2 * most, w > 2 * most];
}

/** A name and a paragraph entered in a small sheet (a template card's name and intro). */
export function TextSheet({ title, nameLabel, textLabel, initialName, initialText, nameMax, textMax, save, onClose }: {
  title: string; nameLabel: string; textLabel: string; initialName: string; initialText: string;
  // the server's limits (lab2shot/site/library.py): the name in characters, the text by display width, in full-width
  // characters (TEXT_WIDTH / 2: 240 Chinese characters or 480 English ones, widthCount)
  nameMax: number; textMax: number;
  save: (name: string, text: string) => Promise<void>; onClose: () => void;
}) {
  const [name, setName] = useState(initialName);
  const [text, setText] = useState(initialText);
  const [busy, setBusy] = useState(false);
  const [count, limit, tooWide] = widthCount(text.trim(), textMax);
  const ready = !!name.trim() && (name.trim() !== initialName || text.trim() !== initialText) && !tooWide && !busy;
  const go = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      await save(name.trim(), text.trim());
      onClose();
    } catch {
      // the caller has reported the failure (a message); the sheet stays open with the text preserved
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={title} width={560} onClose={onClose}>
      <div className="lib-form">
        <label className="login-field">
          <span>{nameLabel}<b className="field-count">{name.length} / {nameMax}</b></span>
          <input className="field" value={name} autoFocus maxLength={nameMax} aria-label={nameLabel}
            onChange={(e) => setName(e.target.value)} />
        </label>
        <label className="login-field">
          <span>{textLabel}<b className={`field-count${tooWide ? " over" : ""}`}>{count} / {limit}</b></span>
          <textarea className="field" value={text} rows={6} maxLength={2 * textMax} aria-label={textLabel}
            onChange={(e) => setText(e.target.value)} />
        </label>
        <div className="dialog-row">
          <Button tone="ghost" onClick={onClose}>{t("ui.common.cancel")}</Button>
          <Button tip={ready ? undefined : tipOf("disabled", t("ui.misc.name_or_change"))} tone="primary" disabled={!ready} onClick={() => void go()}>{busy ? t("ui.misc.saving") : t("ui.common.save")}</Button>
        </div>
      </div>
    </Sheet>
  );
}

/** A display text in each language, as written (a missing or empty one: not written in that language). */
export type LangText = Partial<Record<Lang, string>>;

/** What a two-language edit hands back: the text in every language (the server's GET answers both: the sheet edits
 * only once it has read them). */
export type LangSaved = LangText;

const trimmed = (v: LangText): LangText => Object.fromEntries(LANGS.map((l) => [l, (v[l] ?? "").trim()])) as LangText;
const same = (a: LangText, b: LangText): boolean => LANGS.every((l) => (a[l] ?? "").trim() === (b[l] ?? "").trim());
const filled = (v: LangText): LangText => Object.fromEntries(LANGS.filter((l) => (v[l] ?? "").trim()).map((l) => [l, (v[l] ?? "").trim()])) as LangText;

/** Reads the texts to edit in every language (`load`), or starts from `initial` (a new one): [texts, read, failed]. Not
 * read (`failed`): nothing is edited (saving one language blind could not show what the other says). */
function useLangTexts(load: (() => Promise<LangText>) | undefined, initial: LangText) {
  const [texts, setTexts] = useState<LangText>(initial);
  const [read, setRead] = useState<LangText | null>(load ? null : initial);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    if (!load) return;
    let alive = true;
    load().then(
      (got) => { if (alive) { setTexts(got); setRead(got); } },
      () => { if (alive) setFailed(true); },
    );
    return () => { alive = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- read once when the sheet opens
  return [texts, setTexts, read, failed] as const;
}

/** A name in both languages entered in a small sheet (a category of the templates panel or the node menu: the
 * administrator names it in Chinese and in English at once). `load` reads what is written now (a rename); without it
 * the fields start from `initial` (a new one). `max`: the server's limit per language. When what is written could not be
 * read the sheet says so and edits nothing. */
export interface Namings {
  title: string;
  label: string;
  initial?: LangText;
  load?: () => Promise<LangText>;
  max: Record<Lang, number>;
  save: (names: LangSaved) => Promise<void>;
}

export function NamesSheet({ title, label, initial = {}, load, max, save, onClose }: Namings & { onClose: () => void }) {
  const [names, setNames, read, failed] = useLangTexts(load, initial);
  const [busy, setBusy] = useState(false);
  const ready = read !== null && LANGS.some((l) => (names[l] ?? "").trim()) && !same(names, read) && !busy;
  const go = async () => {
    if (!ready) return;
    setBusy(true);
    try {
      await save(filled(names));
      onClose();
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={title} width={460} onClose={onClose}>
      <div className="lib-form">
        {read === null && <p className="muted">{failed ? t("ui.misc.words_unread") : t("ui.misc.loading_words")}</p>}
        {read !== null && LANGS.map((l, i) => {
          const what = t("ui.misc.in_lang", { label, lang: t(`lang.${l}`) });
          return (
            <label className="login-field" key={l}>
              <span>{what}</span>
              <input className="field" value={names[l] ?? ""} autoFocus={i === 0} maxLength={max[l]} aria-label={what}
                lang={l} {...tipAttrs(tipOf("value", t("ui.misc.at_most_chars", { count: max[l] })))}
                onChange={(e) => setNames({ ...names, [l]: e.target.value })} onKeyDown={(e) => e.key === "Enter" && !composing(e) && void go()} />
            </label>
          );
        })}
        <div className="dialog-row">
          <Button tone="ghost" onClick={onClose}>{t("ui.common.cancel")}</Button>
          <Button tip={ready ? undefined : tipOf("disabled", t("ui.misc.name_or_change"))} tone="primary" disabled={!ready} onClick={() => void go()}>{busy ? t("ui.misc.saving") : t("ui.common.save")}</Button>
        </div>
      </div>
    </Sheet>
  );
}

/** A name and a paragraph in both languages (a node's name and description in the node menu): read with `load`, saved
 * as {lang: text} for each; not read, nothing is edited. */
export function TextsSheet({ title, nameLabel, textLabel, load, nameMax, textMax, save, onClose }: {
  title: string; nameLabel: string; textLabel: string;
  load: () => Promise<{ name: LangText; text: LangText }>;
  nameMax: Record<Lang, number>; textMax: Record<Lang, number>; // the server's limits per language (lab2shot/nodes/text.py)
  save: (name: LangSaved, text: LangSaved) => Promise<void>; onClose: () => void;
}) {
  const [both, setBoth] = useState<{ name: LangText; text: LangText } | null>(null);
  const [read, setRead] = useState<{ name: LangText; text: LangText } | null>(null);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let alive = true;
    load().then(
      (got) => { if (alive) { setBoth(got); setRead(got); } },
      () => { if (alive) setFailed(true); },
    );
    return () => { alive = false; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps -- read once when the sheet opens
  const ready = !!both && !!read && LANGS.some((l) => (both.name[l] ?? "").trim())
    && (!same(both.name, read.name) || !same(both.text, read.text)) && !busy;
  const go = async () => {
    if (!ready || !both) return;
    setBusy(true);
    try {
      await save(filled(both.name), trimmed(both.text));
      onClose();
    } catch {
      // the caller has reported the failure (a message); the sheet stays open with the text preserved
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={title} width={600} onClose={onClose}>
      <div className="lib-form">
        {!both && <p className="muted">{failed ? t("ui.misc.words_unread") : t("ui.misc.loading_words")}</p>}
        {both && LANGS.map((l, i) => {
          const n = both.name[l] ?? "", x = both.text[l] ?? "";
          const nameWhat = t("ui.misc.in_lang", { label: nameLabel, lang: t(`lang.${l}`) });
          const textWhat = t("ui.misc.in_lang", { label: textLabel, lang: t(`lang.${l}`) });
          return (
            <div key={l}>
              <label className="login-field">
                <span>{nameWhat}<b className="field-count">{n.length} / {nameMax[l]}</b></span>
                <input className="field" value={n} autoFocus={i === 0} maxLength={nameMax[l]} aria-label={nameWhat} lang={l}
                  onChange={(e) => setBoth({ ...both, name: { ...both.name, [l]: e.target.value } })} />
              </label>
              <label className="login-field">
                <span>{textWhat}<b className="field-count">{x.length} / {textMax[l]}</b></span>
                <textarea className="field" value={x} rows={4} maxLength={textMax[l]} aria-label={textWhat} lang={l}
                  onChange={(e) => setBoth({ ...both, text: { ...both.text, [l]: e.target.value } })} />
              </label>
            </div>
          );
        })}
        <div className="dialog-row">
          <Button tone="ghost" onClick={onClose}>{t("ui.common.cancel")}</Button>
          <Button tip={ready ? undefined : tipOf("disabled", t("ui.misc.name_or_change"))} tone="primary" disabled={!ready} onClick={() => void go()}>{busy ? t("ui.misc.saving") : t("ui.common.save")}</Button>
        </div>
      </div>
    </Sheet>
  );
}
