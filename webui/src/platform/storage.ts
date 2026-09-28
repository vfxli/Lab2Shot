import { localKeys, readLocalJSON, removeLocal, writeLocal } from "./util";

/** The one place 用户首选项 (state/preferences.ts) touches localStorage: every key it uses is named "lab2shot.<name>".
 * Reading and writing goes through readLocalJSON/writeLocal (util.ts), which swallow the private-window /
 * blocked-storage case: a preference simply isn't remembered then. */

export function readPref<T>(key: string, fallback: T): T {
  return readLocalJSON<T>(`lab2shot.${key}`, fallback);
}

export function writePref(key: string, value: unknown): boolean {
  return writeLocal(`lab2shot.${key}`, typeof value === "string" ? value : JSON.stringify(value));
}

export function removePref(key: string): void {
  removeLocal(`lab2shot.${key}`);
}

/** The keys kept under `prefix`, without "lab2shot.". */
export function prefKeys(prefix: string): string[] {
  return localKeys(`lab2shot.${prefix}`).map((k) => k.slice("lab2shot.".length));
}
