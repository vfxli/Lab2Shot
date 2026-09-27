// 账号相关的后台接口 (lab2shot/accounts.py, server/users.py, server/quota.py, resources.py)：账号的定义、
// 登录、磁盘配额，以及登记表声明的行操作。api/admin.ts 将其并入 adminApi 并转发这些类型，后台页面仍只从该处读取。

import { json } from "../platform/http";
import type { Availability } from "./applies";
import type { TagInfo } from ".";
import type { AccountUsage, Traffic } from "./library";

/** 登记表声明的一个行操作（lab2shot/resources.py Act），服务器已按当前登录过滤：返回的操作均可使用。
 * `path` 中的 `{id}` 替换为该行第一列的值。 */
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
  // account.password, account.delete, account.logins): server/available.py account()
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
  quota_gb: number | null; // 该账号自身的磁盘配额（null：使用设置中的默认值）
  traffic: Traffic; // 该账号使用的网络流量（lab2shot/server/traffic.py）
  online: { browser: boolean; client: boolean }; // never a raw session count (最近登录 has the real detail)
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
  online: { count: number; browser: number; client: number; who: { browser: string[]; client: string[] } };
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


/** 权限表中的一项：短名（界面显示的名称）、勾选后授予的能力（悬停显示该句）、当前是否勾选。
 * 由服务器统一定义并排版（lab2shot/roles.py sheet()），网页按其绘制，不另写文字，也不自行计算。 */
export interface RightItem {
  id: string;
  label: string;
  what: string;
  on: boolean;
}

/** 单个角色的勾选表：名称、分段、已勾选条数、是否仍为默认配置、最近修改者。 */
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

/** 后台「用户」中的权限表（lab2shot/server/users.py /api/admin/rights）：每个可分配的角色一张。 */
export interface RightsView {
  sheets: RightsSheetView[];
}

export const accountsApi = {
  // 二级管理员的权限：由一级管理员在「用户」中勾选（admins.manage）。修改后下一个请求即生效，无须重新登录
  rights: () => json<RightsView>("GET", "/api/admin/rights"),
  setRights: (role: string, on: string[]) => json<RightsView>("PUT", "/api/admin/rights", { role, on }),
  resetRights: (role: string) => json<RightsView>("DELETE", `/api/admin/rights/${encodeURIComponent(role)}`),
  // 磁盘配额（配额及其设置均位于用户管理页面）：账号的占用量，以及按账号修改上限
  userQuota: (id: number) => json<AccountUsage>("GET", `/api/admin/users/${id}/quota`),
  setUserQuota: (id: number, gb: number | null) => json<AccountUsage>("PUT", `/api/admin/users/${id}/quota`, { gb }),
  // 行操作（恢复、移入回收站、永久删除等）：方法与路径均由服务器提供，此处不写任何地址
  rowAct: (act: ResourceAct, rowId: string) =>
    json<unknown>(act.method as "POST" | "PUT" | "DELETE", act.path.replace("{id}", encodeURIComponent(rowId)), act.method === "DELETE" ? undefined : {}),
};
