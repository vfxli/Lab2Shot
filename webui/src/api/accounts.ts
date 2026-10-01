// The admin calls about accounts (lab2shot/accounts.py, server/users.py, server/quota.py, resources.py): what an
// account is, its logins, its disk quota, and the row actions the registry declares. api/admin.ts merges them into
// adminApi and re-exports these types; admin pages read them only from there.

import { json } from "../platform/http";
import type { Availability } from "./applies";
import type { TagInfo } from ".";
import type { AccountUsage, Traffic } from "./library";

/** A row action the registry declares (lab2shot/resources.py Act), already filtered by the server for this login:
 * every action returned is usable. `{id}` in `path` is replaced by the value of the row's first column. */
export interface ResourceAct {
  id: string;
  label: string;
  tip: string;
  method: string;
  path: string;
  danger: boolean;
}

/** 用户: the accounts (lab2shot/accounts.py, server/users.py). */
export interface UserRow {
  applies: Availability;
  // what this login may do to the account (account.edit, account.expiry, account.enable, account.tags, account.role,
  // account.password, account.delete, account.purge, account.logins, account.quota, account.quota_set):
  // server/available.py ACCOUNT, resolved by account()
  id: number;
  username: string;
  name: string; // Chinese name
  department: string;
  role: string;
  role_label: string;
  owner: boolean; // the built-in administrator account
  all_nodes: boolean; // uses every node, whatever its tags
  tags: string[]; // what it may use (nodes/tags.py), the implied ones not listed
  expires: number | null; // null: never (the administrator)
  enabled: boolean;
  deleted: number | null;
  created: number;
  last_login: number | null;
  password_set: number | null;
  password_by: string;
  no_password: boolean; // the built-in administrator with no password yet: nobody can log in as it
  state: string; // 可以用, or why not (未设密码 for the built-in administrator with no password yet)
  state_tip: string;
  usable_now: boolean; // it can log in now (not disabled, expired or deleted)
  jobs: number;
  last_job: number | null;
  quota_gb: number | null; // the account's own disk quota (null: the default from the settings)
  traffic: Traffic; // the network traffic the account used (lab2shot/server/traffic.py)
  presence: Presence; // 在线 (lab2shot/accounts.py presence()); never a raw session count (最近登录 has the real detail)
}

/** Where a request came from: a browser as its name and system ("Chrome · Windows"), a DCC plugin or the command line
 * by its app and computer. */
export interface Place {
  kind: "web" | "client";
  device: string;
  hostname: string;
}

/** An account's 在线 (lab2shot/accounts.py presence()): where it made a request within the last `window_s` seconds
 * (OnlineSummary), and otherwise when and where it was last active (null: never). */
export interface Presence {
  online: Place[];
  active: number | null;
  where: Place | null;
}

/** The overview's 在线 tile: accounts that made a request within the last `window_s` seconds (lab2shot/accounts.py
 * online()). */
export interface OnlineSummary {
  count: number;
  browser: number;
  client: number;
  who: { browser: string[]; client: string[] };
  window_s: number;
}

/** Where an account is online right now: at most one row per kind (a browser and a DCC plugin/command line may both
 * be online at once; lab2shot/accounts.py start()). */
export interface OnlinePlace {
  kind: "web" | "client";
  ip: string;
  device: string;
  hostname: string;
  seen: number;
}

export interface LoginAttempt {
  at: number;
  ok: boolean;
  reason: string; // "" on success
  kind: "web" | "client";
  ip: string;
  device: string;
  hostname: string;
  ended: { kind: string; device: string }[];
}

export interface LoginWindow {
  logins: number;
  ips: number;
  devices: number;
  failures: number;
}

export interface UserLogins {
  online: OnlinePlace[];
  recent: LoginAttempt[];
  summary: { "7d": LoginWindow; "30d": LoginWindow; suspicious: boolean; worst_day_failures: number };
  threshold_tip: string;
}

export interface UsersView {
  users: UserRow[];
  departments: string[];
  tags: Record<string, TagInfo>;
  allowed_new: string[];
  roles: { id: string; label: string; tip: string }[]; // the roles this login may give
  default_role: string; // a new account's, unless one is picked
  online: OnlineSummary;
}

export interface NewUser {
  username: string;
  password: string;
  name: string;
  department: string;
  expires: number; // seconds since the epoch
  tags?: string[]; // only with users.tags (none: the default for a new account)
  role?: string; // only with users.role
}

export type UserChange = Partial<Pick<UserRow, "name" | "department" | "expires" | "enabled" | "tags" | "role">>;


/** One item of the rights sheet: its short name (what the UI shows), the ability ticking it grants (the sentence shown
 * on hover), and whether it is ticked. Defined and laid out by the server alone (lab2shot/roles.py sheet()); the page
 * draws it as given, with no words or computation of its own. */
export interface RightItem {
  id: string;
  label: string;
  what: string;
  on: boolean;
}

/** One role's sheet: its name, its sections, how many items are ticked, whether it is still the default, who changed it last. */
export interface RightsSheetView {
  role: string;
  label: string;
  tip: string;
  groups: { label: string; items: RightItem[] }[];
  count: number;
  total: number;
  default: boolean;
  default_count: number;
  updated: number;
  updated_by: string;
}

/** The rights sheets of the admin page's 用户 (lab2shot/server/users.py /api/admin/rights): one per role that can be given. */
export interface RightsView {
  sheets: RightsSheetView[];
}

export const accountsApi = {
  // 二级管理员 rights: ticked by a first-level administrator in 用户 (admins.manage). A change applies from the next request, without logging in again
  rights: () => json<RightsView>("GET", "/api/admin/rights"),
  setRights: (role: string, on: string[]) => json<RightsView>("PUT", "/api/admin/rights", { role, on }),
  resetRights: (role: string) => json<RightsView>("DELETE", `/api/admin/rights/${encodeURIComponent(role)}`),
  // disk quota (the quota and its setting both live on the 用户 page): what the account holds, and changing its limit per account
  userQuota: (id: number) => json<AccountUsage>("GET", `/api/admin/users/${id}/quota`),
  setUserQuota: (id: number, gb: number | null) => json<AccountUsage>("PUT", `/api/admin/users/${id}/quota`, { gb }),
  // row actions (restore, move to the recycle bin, delete for good ...): method and path come from the server; no address is written here
  rowAct: (act: ResourceAct, rowId: string) =>
    json<unknown>(act.method as "POST" | "PUT" | "DELETE", act.path.replace("{id}", encodeURIComponent(rowId)), act.method === "DELETE" ? undefined : {}),
};
