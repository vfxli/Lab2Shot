import { formatValue, type Params } from "../messages/format.ts";
import { getLang, useLang, type Lang } from "./lang.ts";
import { templateFor } from "./words.ts";

/** The page's words: `t("ui.account.language")`, `t("ui.account.language_switch", { other: "English" })`, in the
 * page's language (i18n/lang.ts). The key is written out whole (`lab2shot check i18n` finds every key the page uses
 * that way and every key no code uses); one made from pieces is only for the families made from data (lang.<code>).
 * A key in no table is a bug, said loudly. Placeholders and formats as the messages' ({name}, {name:spec}); a sentence
 * written in plural forms (`<key>.one` / `.other`) reads in the one its number takes (`count`, i18n/words.ts). */

const PLACEHOLDER = /\{(\w+)(?::([^{}]*))?\}/g;

export function t(key: string, params: Params = {}, lang: Lang = getLang()): string {
  const template = templateFor(key, params, lang);
  if (template === undefined) throw new Error(`${key} is in no catalogue (lab2shot/i18n/<lang>/ui/*.toml)`);
  return template.replace(PLACEHOLDER, (_, name: string, spec: string | undefined) => {
    if (!(name in params)) throw new Error(`${key}: parameter ${name} missing`);
    return formatValue(params[name], spec, lang);
  });
}

/** `t` for a component that must re-render by itself when the language changes (one memoised, or outside the
 * page's root): it subscribes to the language. */
export function useT(): typeof t {
  useLang((s) => s.lang);
  return t;
}

/** A display field that may hold both languages (a built-in template's {"zh": …, "en": …}): a string as it is (a
 * user's own words), an object as its text in the page's language, else in the other one (lab2shot/i18n pick). */
export function pick(v: unknown, lang: Lang = getLang()): string {
  if (typeof v === "string") return v;
  if (v && typeof v === "object") {
    const o = v as Record<string, unknown>;
    for (const l of [lang, "zh", "en"]) if (typeof o[l] === "string" && o[l]) return o[l] as string;
    return "";
  }
  return v == null ? "" : String(v);
}
