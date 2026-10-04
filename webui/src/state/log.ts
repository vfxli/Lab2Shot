import { useSyncExternalStore } from "react";
import { account } from "./session";
import { clientInfo } from "../platform/client";
import { readLocalJSON, writeLocal } from "../platform/util";
import { stampText } from "../platform/format";
import { t } from "../i18n/t";
import { getLang, type Lang } from "../i18n/lang";
import { render, type Params } from "../messages/format";
import { namesNode, nodeNameOf } from "../messages/shorten";
import type { Said as Word } from "../messages/format";

/** The page's log: every message the page showed (state/say.ts, with its code), each cook's progress and the page's
 * own errors, kept in the browser (it survives a reload) so the user can open it, copy it for support, or send it to the
 * server's log. It only records and makes no decisions.
 *
 * An entry keeps the message as what it is (its code, its parameters, the node it is about) beside its words, so a
 * reader who switched language reads it in theirs (`entryText`): the page's own messages are said again here, the
 * server's by the server (POST /api/said: `wantedAgain` / `keepAgain`, editor/LogSheet.tsx); until then, or for one
 * that cannot be said again, its words as they were. */

export type Level = "info" | "ok" | "warn" | "error";

interface LogEntry {
  t: number;
  level: Level;
  text: string;
  code?: string; // the message's code: users quote it, developers search for it
  params?: Record<string, unknown>; // its parameters (said again in another language)
  node?: Word | string; // the node it is about, as a message points at it (graph/naming.ts nodeWord): put in front of it once
  body?: string; // its own words (without the node), in `lang`
  lang?: Lang; // the language `text` is in
  count?: number; // repeat count of the same entry (same level, code and text) while it is still among the latest entries
  last?: number; // time of the latest repeat (t: the first occurrence)
}

const RECENT = 20; // an entry repeated while it is among this many latest entries is counted on it rather than listed again

const KEY = "lab2shot.log";
const MAX = 500;

let entries: LogEntry[] = readLocalJSON<LogEntry[]>(KEY, []);
let unseen = 0; // errors since the log window was last opened
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());

/** What a message entry keeps besides its words (state/say.ts): its parameters, its own words and the node it is about. */
export interface Said {
  params?: Record<string, unknown>;
  body?: string;
  node?: Word | string;
}

/** The name (id) of the node an entry is about. */
export const nameOf = (node: Word | string | undefined): string => nodeNameOf(node);

export function log(level: Level, text: string, code?: string, said: Said = {}): void {
  const now = Date.now();
  const from = Math.max(0, entries.length - RECENT);
  const again = entries.findIndex((e, i) => i >= from && e.level === level && e.text === text && e.code === code);
  if (again >= 0) {
    const e = entries[again];
    entries = entries.map((x, i) => (i === again ? { ...e, count: (e.count ?? 1) + 1, last: now } : x));
  } else {
    const kept = code ? { code, ...said, lang: getLang() } : {};
    entries = [...entries.slice(-(MAX - 1)), { t: now, level, text, ...kept }];
    if (level === "error") unseen++;
  }
  writeLocal(KEY, JSON.stringify(entries));
  notify();
}

/** The latest `n` entries (feedback sends them). */
export const recentLog = (n: number): LogEntry[] => entries.slice(-n);

export function clearLog(): void {
  entries = [];
  unseen = 0;
  writeLocal(KEY, "[]");
  notify();
}

export function markSeen(): void {
  unseen = 0;
  notify();
}

// ------------------------------------------------------------------ said again in the reader's language

// the server's words for a message it said before, by language, code and parameters (POST /api/said)
const saidAgain = new Map<string, string | null>();
const againKey = (e: LogEntry, lang: Lang) => `${lang}\u0000${e.code}\u0000${JSON.stringify(e.params ?? {})}`;

/** An entry's own words in `lang`: the page's own message said again here, else the server's words (kept by
 * `keepAgain`); undefined when there are none (yet). */
function bodyIn(e: LogEntry, lang: Lang): string | undefined {
  try {
    return render(e.code!, (e.params ?? {}) as Params, lang);
  } catch {
    const got = saidAgain.get(againKey(e, lang));
    return got ?? undefined;
  }
}

/** An entry as the reader reads it: in the page's language now when it was said in another (and can be said again). */
export function entryText(e: LogEntry, lang: Lang = getLang()): string {
  if (!e.code || !e.lang || e.lang === lang) return e.text;
  const body = bodyIn(e, lang);
  if (body === undefined) return e.text;
  return e.node && !namesNode(e.params, nameOf(e.node)) ? t("ui.state.log_node", { node: e.node, text: body }, lang) : body;
}

/** The server's messages among `list` that `lang` has no words for yet: what to ask POST /api/said for. */
export function wantedAgain(list: readonly LogEntry[], lang: Lang = getLang()): { code: string; params: Record<string, unknown> }[] {
  const out = new Map<string, { code: string; params: Record<string, unknown> }>();
  for (const e of list) {
    if (!e.code || !e.lang || e.lang === lang || saidAgain.has(againKey(e, lang)) || bodyIn(e, lang) !== undefined) continue;
    out.set(againKey(e, lang), { code: e.code, params: e.params ?? {} });
  }
  return [...out.values()];
}

/** Keeps what the server said again (`texts`, in the order of `asked`) for `lang`, and redraws the log. */
export function keepAgain(asked: { code: string; params: Record<string, unknown> }[], texts: (string | null)[], lang: Lang): void {
  asked.forEach((m, i) => saidAgain.set(againKey({ t: 0, level: "info", text: "", code: m.code, params: m.params }, lang), texts[i] ?? null));
  entries = [...entries];
  notify();
}

const subscribe = (l: () => void) => (listeners.add(l), () => listeners.delete(l));
export const useLog = () => useSyncExternalStore(subscribe, () => entries);
export const useUnseenErrors = () => useSyncExternalStore(subscribe, () => unseen);

const LEVEL_KEY = { info: "ui.state.level_info", ok: "ui.state.level_ok", warn: "ui.state.level_warn", error: "ui.state.level_error" } as const;
export const levelText = (l: Level): string => t(LEVEL_KEY[l]);

/** When an entry occurred: its time, plus for a repeated entry the count and the latest time (×390，到 23:29:59). */
export const entryTime = (e: LogEntry): string =>
  `${stampText(e.t / 1000)}${e.count ? `  ${t("ui.state.log_repeated", { count: e.count, last: stampText((e.last ?? e.t) / 1000) })}` : ""}`;

/** The log as text for pasting: user and location first, then every entry. */
export function logText(graph: string): string {
  const c = clientInfo();
  const head = [
    t("ui.state.log_title"),
    t("ui.state.log_copied", { time: stampText(Date.now() / 1000) }),
    t("ui.state.log_page", { url: location.href }),
    t("ui.state.log_browser", { agent: navigator.userAgent }),
    t("ui.state.log_account", { account: account()?.username ?? t("ui.state.log_no_login"), platform: c.platform, timezone: c.timezone }),
    t("ui.state.log_graph", { graph }),
    "",
  ];
  return [...head, ...entries.map((e) => `${entryTime(e)}  [${levelText(e.level)}]  ${e.code ? `[${e.code}] ` : ""}${entryText(e)}`)].join("\n");
}
