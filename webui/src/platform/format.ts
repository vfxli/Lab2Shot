/** Formatting of byte counts, durations, timestamps and frame ranges, shared by the editor and the admin pages.
 * Pure; no imports. */

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

/** A transfer rate in bytes per second: 1.4 MB/s, 320 KB/s. */
export const rateText = (bytesPerSecond: number): string =>
  bytesPerSecond >= 1e6 ? `${(bytesPerSecond / 1e6).toFixed(1)} MB/s` : `${Math.max(1, Math.round(bytesPerSecond / 1e3))} KB/s`;

// ------------------------------------------------------------------ lengths of time

/** An exact length: 45 秒, 12 分 3 秒, 2 小时 5 分. */
export function durationText(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s} 秒`;
  const m = Math.floor(s / 60);
  return m < 60 ? `${m} 分 ${s % 60} 秒` : `${Math.floor(m / 60)} 小时 ${m % 60} 分`;
}

/** An estimate, as precisely as it deserves: 40 秒, 12 分, 2 小时 40 分. */
export function roughlyText(seconds: number): string {
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} 秒`;
  const m = Math.round(seconds / 60);
  return m < 60 ? `${m} 分` : `${Math.floor(m / 60)} 小时${m % 60 ? ` ${m % 60} 分` : ""}`;
}

/** Remaining time, approximated: 约 1 分钟内, 约 3 分钟, 约 1.5 小时. */
export const leftText = (seconds: number): string =>
  seconds < 60 ? "约 1 分钟内" : seconds < 3600 ? `约 ${Math.ceil(seconds / 60)} 分钟` : `约 ${(seconds / 3600).toFixed(1)} 小时`;

/** Hours of computing: 0.25 小时, 12.5 小时. */
export function hoursText(seconds: number): string {
  if (!seconds) return "0 小时";
  const h = seconds / 3600;
  return h < 0.01 ? "<0.01 小时" : `${h < 10 ? h.toFixed(2) : h.toFixed(1)} 小时`;
}

const now = () => Date.now() / 1000;

/** How long ago `t` was (seconds since the epoch): 刚刚, 5 分钟前, 3 小时前, 2 天前. */
export function agoText(t: number): string {
  const s = Math.max(0, now() - t);
  if (s < 60) return "刚刚";
  if (s < 3600) return `${Math.floor(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时前`;
  return `${Math.floor(s / 86400)} 天前`;
}

/** How long something has lasted since `t`: 不到 1 分钟, 5 分钟, 3 小时, 2 天 (已运行 …). */
export function lastedText(t: number): string {
  const s = Math.max(0, now() - t);
  if (s < 60) return "不到 1 分钟";
  if (s < 3600) return `${Math.floor(s / 60)} 分钟`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小时`;
  return `${Math.floor(s / 86400)} 天`;
}

// ------------------------------------------------------------------ moments (seconds since the epoch)

const at = (t: number) => new Date(t * 1000);

/** 14:05 */
export const clockText = (t: number): string => at(t).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });

/** 9/14 14:05 */
export const whenText = (t: number): string => at(t).toLocaleString("zh-CN", { hour12: false, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** 9/14 14:05:09 */
export const whenSecondsText = (t: number): string =>
  at(t).toLocaleString("zh-CN", { hour12: false, month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** 2026/9/14 14:05 */
export const fullTimeText = (t: number): string =>
  at(t).toLocaleString("zh-CN", { hour12: false, year: "numeric", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });

/** 2026/9/14 14:05:09, everything */
export const stampText = (t: number): string => at(t).toLocaleString("zh-CN", { hour12: false });

/** 2026年9月14日 */
export const dayText = (t: number): string => at(t).toLocaleDateString("zh-CN", { year: "numeric", month: "long", day: "numeric" });

// ------------------------------------------------------------------ frames

/** 1001–1124 */
export const rangeText = (first: number, last: number): string => `${first}–${last}`;

/** 第 1001–1124 帧 */
export const framesLabel = (first: number, last: number): string => `第 ${rangeText(first, last)} 帧`;

/** 格式化点数或计数：一万以上以「万」、一亿以上以「亿」为单位，保留一位小数
 * （「30 万 / 100 万」比「300000」易读）。页面中所有点数的显示均须使用本函数，不得各处自行调用 toLocaleString。 */
export function countText(n: number): string {
  const x = Math.round(n);
  if (x >= 1e8) return `${(x / 1e8).toFixed(1).replace(/\.0$/, "")} 亿`;
  if (x >= 1e4) return `${(x / 1e4).toFixed(1).replace(/\.0$/, "")} 万`;
  return String(x);
}
