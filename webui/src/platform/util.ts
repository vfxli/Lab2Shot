// Small helpers shared by several components.

/** localStorage, which throws in private windows and when storage is blocked: failures read as "nothing stored". */
export function readLocal(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

/** Whether it was kept: false in private mode, and when the browser's storage for this site is full
 * (QuotaExceededError). A preference simply isn't remembered then; the working copy (editor/autosave.ts) says so. */
export function writeLocal(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export function removeLocal(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    /* storage blocked: nothing was kept either */
  }
}

/** The keys kept under `prefix` (all of them, in no order). */
export function localKeys(prefix: string): string[] {
  try {
    return Object.keys(localStorage).filter((k) => k.startsWith(prefix));
  } catch {
    return [];
  }
}

export function readLocalJSON<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(readLocal(key) ?? "") as T;
  } catch {
    return fallback;
  }
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

/** Whether an address from the server's data (a project's repository, licence, download page) may be a link: a web
 * address only. Anything else (javascript:, data:, a path) is shown as text, never followed. */
export const webAddress = (url: string | null | undefined): url is string => !!url && /^https?:\/\/[^\s]+$/i.test(url);

/** A site's name for a link's text: smpl-x.is.tue.mpg.de for https://smpl-x.is.tue.mpg.de/download.php. */
export const host = (url: string) => url.replace(/^https?:\/\//, "").replace(/\/.*$/, "");

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
