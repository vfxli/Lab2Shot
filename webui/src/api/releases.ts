import { json } from "../platform/http";
import type { Words } from "./catalog";

/** Lines in both languages ({zh: [...], en: [...]}, as many in each: lab2shot check releases). */
export type Lines = Partial<Record<"zh" | "en", string[]>>;

/** One version in the release notes (lab2shot/releases.py, read from CHANGELOG.toml at the repository's root). Every
 * text is in both languages: shown in the page's (i18n/t.ts pick, lines()). */
export interface Release {
  name: Words; // 名称
  date: string; // 日期, YYYY-MM-DD
  about: Words; // 说明
  changes: Lines; // 更新: a few short lines, possibly none
  admin?: Lines; // lines for the back office: the server sends them to administrators only
}

/** 「更新说明」 (GET /api/releases): the project's address and the versions, newest first. Plain text: the page never
 * reads markup from it. */
export interface ReleaseNotes {
  project: string;
  releases: Release[];
}

export const readReleases = () => json<ReleaseNotes>("GET", "/api/releases");
