// Small helpers shared by several components.

/** localStorage, which throws in private windows and when storage is blocked: failures read as "nothing stored". */
export function readLocal(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeLocal(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    /* private mode: the preference is not remembered */
  }
}

export function readLocalJSON<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(readLocal(key) ?? "") as T;
  } catch {
    return fallback;
  }
}

/** The frame under a pointer on a horizontal track spanning frames[0]..frames[last] (nearest existing frame). */
export function frameAt(frames: number[], clientX: number, rect: DOMRect): number {
  const first = frames[0] ?? 0;
  const last = frames.at(-1) ?? first;
  const target = first + Math.min(1, Math.max(0, (clientX - rect.left) / rect.width)) * (last - first);
  return frames.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a), first);
}

/** Hands the browser something to save as a file (a URL or a blob URL; uses the browser's download). */
export function download(href: string, name: string): void {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
}

/** Copies text to the clipboard. Pages served over http:// from another machine have no clipboard API; the legacy method is used there. */
export async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
}

/** A site's name for a link's text: smpl-x.is.tue.mpg.de for https://smpl-x.is.tue.mpg.de/download.php. */
export const host = (url: string) => url.replace(/^https?:\/\//, "").replace(/\/.*$/, "");

/** Every link from the editor to the help pages opens the same tab (and navigates it), never a new tab per click.
 * Same-origin links without rel="noopener", so the browser finds the tab it opened before. */
export const HELP_TAB = "lab2shot-help";

/** The reason a node cannot be cooked, for places where the node is already named (its footer, the viewer showing it):
 * the name prefix is removed (「导入 USD」出错：文件里没有模型 /set/gone… → 文件里没有模型 /set/gone…) so only the reason shows. */
export function ownReason(text: string, label: string): string {
  const own = `「${label}」`;
  return (text.startsWith(own) && text.slice(own.length).replace(/^(出错：|的)/, "")) || text;
}

/** Reads text from the clipboard. Pages served over http:// from another machine have no clipboard API, and the
 * permission may be refused, so the result can be empty; the caller then asks the user to paste the text manually
 * rather than failing silently. */
export async function readClipboard(): Promise<string> {
  try {
    return await navigator.clipboard.readText();
  } catch {
    return "";
  }
}
