import { create } from "zustand";

/** The page's language: what every word it shows is in (lab2shot/i18n LANGS). One store for the whole page, the gate
 * included (it is tiny): a component that shows words reads them with `t()` (i18n/t.ts) while it renders, and the page's
 * root re-renders when the language changes (gate.tsx, site.tsx subscribe to it), so the words follow without a reload.
 *
 * Where it comes from: before anything is asked, the `lang` cookie (this browser's choice), else the browser's own
 * languages, else zh (the server's order, server/lang.py, without the account); then the language the server says
 * it speaks to this login (/api/auth/state `lang`: the account's choice first). Changing it (the account menu,
 * state/language.ts) also writes the cookie, so the login page of this browser is in it too. */

export const LANGS = ["zh", "en"] as const;
export type Lang = (typeof LANGS)[number];
export const DEFAULT_LANG: Lang = "zh";
const COOKIE = "lang";
const COOKIE_DAYS = 400;

/** `v` as one of LANGS (zh, zh-CN, en-US …), or null. */
export function normalLang(v: unknown): Lang | null {
  if (typeof v !== "string") return null;
  const head = v.trim().toLowerCase().split(/[-_.@]/)[0];
  return (LANGS as readonly string[]).includes(head) ? (head as Lang) : null;
}

function fromCookie(): Lang | null {
  try {
    const m = /(?:^|;\s*)lang=([^;]*)/.exec(document.cookie);
    return m ? normalLang(decodeURIComponent(m[1])) : null;
  } catch {
    return null;
  }
}

function fromBrowser(): Lang | null {
  try {
    for (const l of navigator.languages ?? [navigator.language]) {
      const got = normalLang(l);
      if (got) return got;
    }
  } catch {
    /* no navigator: the default */
  }
  return null;
}

const htmlLang = (lang: Lang): string => (lang === "zh" ? "zh-CN" : "en");

/** Who the page speaks to now (lab2shot/i18n APP_SUFFIX): "graph" — someone working on the node graph (node mode),
 * every word as written; "app" — someone using a card (app mode, focus mode: no nodes, no wires, no node names): a word
 * or message that has a `<key>.app` form is said in it (i18n/words.ts templateFor, messages/format.ts, messages/message.ts
 * textOf). Set by the editor's mode (editor/AppMode.tsx), the one switch; the page's root re-renders when it changes. */
export type Phrasing = "graph" | "app";

interface LangState {
  lang: Lang;
  phrasing: Phrasing;
}

export const useLang = create<LangState>(() => ({ lang: fromCookie() ?? fromBrowser() ?? DEFAULT_LANG, phrasing: "graph" }));

export const getLang = (): Lang => useLang.getState().lang;

export const getPhrasing = (): Phrasing => useLang.getState().phrasing;

/** Speak to whoever uses a card (`"app"`) or works on the graph (`"graph"`) from now on. */
export function setPhrasing(phrasing: Phrasing): void {
  if (phrasing !== getPhrasing()) useLang.setState({ phrasing });
}

/** Speak `lang` from now on (the server said so, or the user chose it); `remember`: keep it in this browser's cookie. */
export function setLang(lang: unknown, remember = false): void {
  const got = normalLang(lang);
  if (!got) return;
  if (remember) {
    try {
      document.cookie = `${COOKIE}=${got}; path=/; max-age=${COOKIE_DAYS * 86400}; samesite=lax`;
    } catch {
      /* cookies off: this page still follows it */
    }
  }
  try {
    document.documentElement.lang = htmlLang(got);
  } catch {
    /* no document */
  }
  if (got !== getLang()) useLang.setState({ lang: got });
}
