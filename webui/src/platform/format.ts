/** Formatting of byte counts, durations, timestamps and frame ranges, shared by the editor and the admin pages.
 * Pure but for the page's language (its words: lab2shot/i18n/<lang>/ui/format.toml). */

import { t } from "../i18n/t.ts";

// ------------------------------------------------------------------ sizes

/** 1.2 GB, 340 MB, 12 KB (1024-based, like a file browser). */
export function sizeText(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) (v /= 1024), i++;
  return `${v < 10 ? v.toFixed(1) : Math.round(v)} ${units[i]}`;
}

/** A size already in GB (memory, disk, video memory): 7.5 GB, 24 GB, 3.5 TB. */
export function gbText(gb: number): string {
  if (gb >= 1024) return `${(gb / 1024).toFixed(1)} TB`;
  return `${gb < 10 && !Number.isInteger(gb) ? gb.toFixed(1) : gb.toFixed(0)} GB`;
}

/** An amount given in megabytes (a card's memory, as the server reports it): 512 MB, 4.3 GB, 24 GB. */
export const mbText = (mb: number): string => (mb < 1024 ? `${Math.round(mb)} MB` : gbText(mb / 1024));

/** How a transfer speed is written (the only such formatter: the top bar's live traffic and a parameter row's upload
 * speed both use it): number and unit given separately, the unit B/s, KB/s or MB/s by magnitude, the number always
 * with two decimals (at most 999.99), so the top bar can give it a fixed width that does not jump. */
export function rateParts(bytesPerSecond: number): [string, string] {
  const b = Math.max(0, bytesPerSecond);
  if (b < 1000) return [b.toFixed(2), "B/s"];
  if (b < 1e6) return [(b / 1e3).toFixed(2), "KB/s"];
  return [(b / 1e6).toFixed(2), "MB/s"];
}

/** The same as one text: 1.40 MB/s. */
export const rateText = (bytesPerSecond: number): string => rateParts(bytesPerSecond).join(" ");

// ------------------------------------------------------------------ lengths of time

/** An exact length: 45 s, 12 m 3 s, 2 h 5 m (in the page's language). */
export function durationText(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return t("ui.format.seconds", { s });
  const m = Math.floor(s / 60);
  return m < 60 ? t("ui.format.minutes_seconds", { m, s: s % 60 }) : t("ui.format.hours_minutes", { h: Math.floor(m / 60), m: m % 60 });
}

/** Hours of computing: 0.25 h, 12.5 h. */
export function hoursText(seconds: number): string {
  if (!seconds) return t("ui.format.hours", { h: 0 });
  const h = seconds / 3600;
  return h < 0.01 ? t("ui.format.hours_tiny") : t("ui.format.hours", { h: h < 10 ? h.toFixed(2) : h.toFixed(1) });
}

const now = () => Date.now() / 1000;

/** How long ago `t` was (seconds since the epoch): just now, 5 minutes ago, 3 hours ago, 2 days ago. */
export function agoText(at: number): string {
  const s = Math.max(0, now() - at);
  if (s < 60) return t("ui.format.just_now");
  if (s < 3600) return t("ui.format.minutes_ago", { n: Math.floor(s / 60) });
  if (s < 86400) return t("ui.format.hours_ago", { n: Math.floor(s / 3600) });
  return t("ui.format.days_ago", { n: Math.floor(s / 86400) });
}

/** How long something has lasted since `t`: under a minute, 5 minutes, 3 hours, 2 days (running for …). */
export function lastedText(at: number): string {
  const s = Math.max(0, now() - at);
  if (s < 60) return t("ui.format.under_minute");
  if (s < 3600) return t("ui.format.minutes", { n: Math.floor(s / 60) });
  if (s < 86400) return t("ui.format.hours_long", { n: Math.floor(s / 3600) });
  return t("ui.format.days", { n: Math.floor(s / 86400) });
}

// ------------------------------------------------------------------ moments (seconds since the epoch)

const at = (t: number) => new Date(t * 1000);
/** The locale dates and times are written in (the page's language's: ui.format.locale). */
const locale = (): string => t("ui.format.locale");

/** 14:05 */
export const clockText = (t: number): string => at(t).toLocaleTimeString(locale(), { hour: "2-digit", minute: "2-digit" });

/** 9/14 14:05 */
export const whenText = (t: number): string => at(t).toLocaleString(locale(), { hour12: false, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** 9/14 14:05:09 */
export const whenSecondsText = (t: number): string =>
  at(t).toLocaleString(locale(), { hour12: false, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** 2026/9/14 14:05 */
export const fullTimeText = (t: number): string =>
  at(t).toLocaleString(locale(), { hour12: false, year: "numeric", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** 2026/9/14 14:05:09, everything */
export const stampText = (t: number): string => at(t).toLocaleString(locale(), { hour12: false });

/** 2026年9月14日 */
export const dayText = (t: number): string => at(t).toLocaleDateString(locale(), { year: "numeric", month: "long", day: "numeric" });

// ------------------------------------------------------------------ frames

/** 1001–1124 */
export const rangeText = (first: number, last: number): string => `${first}–${last}`;

/** A point count or other count, large ones in the language's own steps with one decimal (zh: 30 万 / 1.2 亿; en: 300K
 * / 1.2M; ui.format.count_steps lists them as factor=unit;…), which reads more easily than 300000. Every point count
 * on the page is shown through this function, never through a toLocaleString of its own. */
export function countText(n: number): string {
  const x = Math.round(n);
  const steps = t("ui.format.count_steps").split(";").map((p) => p.split("=")).map(([f, unit]) => [Number(f), unit] as const);
  for (const [factor, unit] of steps.sort((a, b) => b[0] - a[0])) {
    if (factor > 0 && x >= factor) return t("ui.format.count", { n: (x / factor).toFixed(1).replace(/\.0$/, ""), unit });
  }
  return String(x);
}
