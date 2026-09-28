import { useSyncExternalStore } from "react";
import { account } from "./session";
import { clientInfo } from "../platform/client";
import { readLocalJSON, writeLocal } from "../platform/util";
import { stampText } from "../platform/format";

/** The page's log: every message the page showed (state/say.ts, with its code), each cook's progress and the page's
 * own errors, kept in the browser (it survives a reload) so the user can open it, copy it for support, or send it to the
 * server's log. It only records and makes no decisions. */

export type Level = "info" | "ok" | "warn" | "error";

interface LogEntry {
  t: number;
  level: Level;
  text: string;
  code?: string; // the message's code: users quote it, developers search for it
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

export function log(level: Level, text: string, code?: string): void {
  const now = Date.now();
  const from = Math.max(0, entries.length - RECENT);
  const again = entries.findIndex((e, i) => i >= from && e.level === level && e.text === text && e.code === code);
  if (again >= 0) {
    const e = entries[again];
    entries = entries.map((x, i) => (i === again ? { ...e, count: (e.count ?? 1) + 1, last: now } : x));
  } else {
    entries = [...entries.slice(-(MAX - 1)), { t: now, level, text, ...(code ? { code } : {}) }];
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

const subscribe = (l: () => void) => (listeners.add(l), () => listeners.delete(l));
export const useLog = () => useSyncExternalStore(subscribe, () => entries);
export const useUnseenErrors = () => useSyncExternalStore(subscribe, () => unseen);

const LEVEL = { info: "信息", ok: "完成", warn: "注意", error: "出错" } as const;
export const levelText = (l: Level) => LEVEL[l];

/** When an entry occurred: its time, plus for a repeated entry the count and the latest time (×390，到 23:29:59). */
export const entryTime = (e: LogEntry): string => `${stampText(e.t / 1000)}${e.count ? `  ×${e.count}，到 ${stampText((e.last ?? e.t) / 1000)}` : ""}`;

/** The log as text for pasting: user and location first, then every entry. */
export function logText(graph: string): string {
  const c = clientInfo();
  const head = [
    "Lab2Shot 网页日志",
    `复制时间：${stampText(Date.now() / 1000)}`,
    `页面：${location.href}`,
    `浏览器：${navigator.userAgent}`,
    `账号：${account()?.username ?? "（没登录）"} · ${c.platform} · ${c.timezone}`,
    `节点图：${graph}`,
    "",
  ];
  return [...head, ...entries.map((e) => `${entryTime(e)}  [${levelText(e.level)}]  ${e.code ? `[${e.code}] ` : ""}${e.text}`)].join("\n");
}
