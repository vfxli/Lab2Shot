import { json } from "../platform/http";

/** One version in the release notes (lab2shot/releases.py, read from CHANGELOG.toml at the repository's root). */
export interface Release {
  name: string; // 名称
  date: string; // 日期, YYYY-MM-DD
  about: string; // 说明
  changes: string[]; // 更新: a few short lines, possibly none
  admin?: string[]; // lines for the back office: the server sends them to administrators only
}

/** 「更新说明」 (GET /api/releases): the project's address and the versions, newest first. Plain text: the page never
 * reads markup from it. */
export interface ReleaseNotes {
  project: string;
  releases: Release[];
}

export const readReleases = () => json<ReleaseNotes>("GET", "/api/releases");
