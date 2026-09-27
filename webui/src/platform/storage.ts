import { readLocal, readLocalJSON, writeLocal } from "./util";

/** The one place 用户首选项 (state/preferences.ts) touches localStorage. Every key it uses is listed here, named
 * "lab2shot.<name>" (the prefix every key in this browser's storage already used, kept so nothing already saved on a
 * user's machine is orphaned by this refactor). Reading and writing goes through readLocal/writeLocal (util.ts),
 * which already swallow the private-window / blocked-storage case: a preference simply isn't remembered then. */

export function readPref<T>(key: string, fallback: T): T {
  return readLocalJSON<T>(`lab2shot.${key}`, fallback);
}

export function writePref(key: string, value: unknown): void {
  writeLocal(`lab2shot.${key}`, typeof value === "string" ? value : JSON.stringify(value));
}

export function readPrefString(key: string): string | null {
  return readLocal(`lab2shot.${key}`);
}
