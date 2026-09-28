import { run } from "../platform/db";
import { randomId } from "../platform/randomId";

/** Native file dialogs and the file and folder handles they return. A web page never sees a path on the user's
 * disk: the user picks a file or folder in the system dialog and the browser grants the page that single handle
 * (Chrome and Edge, on localhost or over HTTPS). Handles are persisted in the browser database so they survive a
 * reload; graphs store only the key. After a reload the browser may require the user's permission again before
 * writing, via its own prompt, which is shown only when requested from a user gesture. Browsers without these
 * dialogs pick files through a file input and save through downloads. */

type Access = { mode: "read" | "readwrite" };

interface Permissions {
  queryPermission(d: Access): Promise<PermissionState>;
  requestPermission(d: Access): Promise<PermissionState>;
}

export interface FileType {
  description: string;
  accept: Record<string, string[]>; // mime type -> suffixes
}

interface Dialogs {
  showOpenFilePicker(o: { id: string; types: FileType[] }): Promise<FileSystemFileHandle[]>;
  showSaveFilePicker(o: { id: string; types: FileType[]; suggestedName: string }): Promise<FileSystemFileHandle>;
  showDirectoryPicker(o: { id: string; mode: "read" | "readwrite" }): Promise<FileSystemDirectoryHandle>;
}

const dialogs = window as unknown as Partial<Dialogs>;

/** Whether the page can open and save the user's files through the system's dialogs (the graph's file); otherwise a
 * graph is opened through a file input and saved as a download. */
export const canWrite = (["showOpenFilePicker", "showSaveFilePicker"] as const).every((f) => typeof dialogs[f] === "function");

/** Whether the page can be given a folder of the user's to read (the view's originals, files/localDirs.ts). */
export const canReadFolder = typeof dialogs.showDirectoryPicker === "function";

/** Resolves to null when the dialog is closed without a selection. */
async function chosen<T>(dialog: Promise<T>): Promise<T | null> {
  try {
    return await dialog;
  } catch (e) {
    if ((e as Error).name === "AbortError") return null;
    throw e;
  }
}

// `id`: the dialog opens in the last location used by a dialog with the same id.
export const openFile = (id: string, types: FileType[]) => chosen(dialogs.showOpenFilePicker!({ id, types }).then(([h]) => h));
export const saveFile = (id: string, types: FileType[], suggestedName: string) => chosen(dialogs.showSaveFilePicker!({ id, types, suggestedName }));
/** Picks a folder read-only. Used when the view renders from local originals (`files/localDirs.ts`), which never
 * writes, so requesting `readwrite` would ask for an unnecessary permission. */
export const readFolder = (id: string) => chosen(dialogs.showDirectoryPicker!({ id, mode: "read" }));

interface Kept {
  id: string;
  handle: FileSystemHandle;
}

/** Local directories (`files/localDirs.ts`) are stored in the same object store, with ids carrying this prefix.
 *
 * A separate store would require an IndexedDB version upgrade, which waits while a tab of an older page holds the
 * database open (`platform/db.ts`); a prefix in the existing store needs no schema change. */
export const DIR_PREFIX = "dir:";

/** Persists a file or folder chosen by the user and returns its key; choosing the same entry again reuses its key. */
export async function keep(handle: FileSystemHandle): Promise<string> {
  const all = await run<Kept[]>("handles", "readonly", (s) => s.getAll());
  let id: string | undefined;
  // Authorized directories (DIR_PREFIX) are never matched: they are records of their own (files/localDirs.ts), and
  // keeping a handle here must never reuse or replace one of them.
  for (const k of all) if (!id && !k.id.startsWith(DIR_PREFIX) && (await k.handle.isSameEntry(handle))) id = k.id;
  id ??= randomId();
  await run("handles", "readwrite", (s) => s.put({ id, handle }));
  return id;
}

/** A persisted file or folder, or undefined when this browser does not hold it (chosen in another browser, or
 * cleared). */
export async function kept<H extends FileSystemHandle>(id: string | undefined): Promise<H | undefined> {
  if (!id) return undefined;
  const k = await run<Kept | undefined>("handles", "readonly", (s) => s.get(id)).catch(() => undefined);
  return k?.handle as H | undefined;
}

/** Whether write access is currently granted. Never prompts, so it is safe for work not triggered by a click. */
async function writable(h: FileSystemHandle): Promise<boolean> {
  return (await (h as FileSystemHandle & Permissions).queryPermission({ mode: "readwrite" })) === "granted";
}

/** Whether read access is currently granted, without prompting. After a reload the browser usually resets the
 * permission to "prompt", and prompting requires a user gesture. The view path must never interrupt the user with a
 * prompt (files/localDirs.ts, constraint 1): when not readable it silently falls back to the server proxy, and the user
 * can grant access from 「文件 · 授权本机素材文件夹…」. */
export async function readable(h: FileSystemHandle): Promise<boolean> {
  try {
    return (await (h as FileSystemHandle & Permissions).queryPermission({ mode: "read" })) === "granted";
  } catch {
    return false;
  }
}

/** All persisted handles with their keys (files/localDirs.ts enumerates the folders the user authorized). */
export async function keptAll(): Promise<{ id: string; handle: FileSystemHandle }[]> {
  return await run<Kept[]>("handles", "readonly", (s) => s.getAll()).catch(() => []);
}

/** Ensures write access, requesting it through the browser prompt when needed. Must be called from a click. */
export async function allowWrite(h: FileSystemHandle): Promise<boolean> {
  try {
    return (await writable(h)) || (await (h as FileSystemHandle & Permissions).requestPermission({ mode: "readwrite" })) === "granted";
  } catch {
    return false; // no user gesture available to request permission
  }
}
