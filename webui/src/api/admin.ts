import type { QueueView, ServerInfo, ServerNoticeText } from ".";
import { ApiError, json } from "../platform/http";
import { accountsApi, type NewUser, type OnlineSummary, type ResourceAct, type UserChange, type UserLogins, type UserRow, type UsersView } from "./accounts";
import { templatesApi } from "./templates";

export type { AdminCategory, CategoryEdit } from "./templates";
export type { LoginAttempt, LoginWindow, NewUser, OnlinePlace, OnlineSummary, Place, Presence, ResourceAct, RightItem, RightsSheetView, RightsView, UserChange, UserLogins, UserRow, UsersView } from "./accounts";

/** The admin page's own calls: settings (lab2shot/config.py SCHEMA, server/settings.py), the overview and restarting
 * the server (server/restart.py). The queue, disk, models kept loaded, usage and log are in api/index.ts (api.admin). */

export type SettingValue = number | boolean | string | string[];

/** One setting as the schema declares it, with its value. */
export interface SettingDef {
  key: string; // <section>.<name>, as in config/local.toml
  group: string;
  label: string;
  help: string;
  tip: string; // the hover text, made by the server: what it does, plus its default in words
  default: SettingValue;
  // The three values in words, as the server states them (lab2shot/config.py Setting.says): 开 / 关, a choice's label,
  // a number with its unit, what leaving it empty means. The page never spells any of that a second time.
  default_text: string;
  value_text: string;
  running_text: string;
  kind: "number" | "int" | "bool" | "choice" | "text" | "list" | "multi"; // multi: any of the options
  min: number | null;
  max: number | null;
  unit: string;
  options: { value: string; label: string }[];
  restart: boolean; // takes effect after the server restarts
  why: string; // and why
  admin: boolean; // false: changed in the file only
  locked: string; // why this login may not change it, as the server says ("" it may): a right it lacks (lab2shot/roles.py setting_needs)
  only_if: string; // a switch it needs on to matter
  empty: string; // text: what leaving it empty means
  value: SettingValue; // what the settings file sets (or the default): what the page edits
  running: SettingValue; // what this server goes by (differs until a restart for the ones that need it)
  overridden: string; // what overrides the file for this run ("" nothing): the command line, the environment
  auto: string; // "" a fixed default; otherwise how this machine works its default out (cores, cards)
  source: string; // the tag next to it: the administrator changed it, or the default was worked out here ("" neither)
  source_tip: string; // and what that means, said by the server (lab2shot/config.py Settings._source)
}

/** What the server can tell that goes with a group of settings (read only). */
export interface StatusRow {
  label: string;
  value: string;
  tip: string;
}

/** A settings page (lab2shot/config.py PAGES): its groups, one card each, in order. */
export interface SettingsPage {
  id: string; // the page's id; its section of the admin page is settings-<id>
  label: string;
  tip: string;
  groups: { id: string; label: string }[];
}

export interface SettingsView {
  file: string;
  pages: SettingsPage[];
  settings: SettingDef[];
  pending: string[]; // keys saved that wait for a restart
  status: Record<string, StatusRow[]>;
}

export interface Overview {
  memory: { total_gb: number; available_gb: number; keep_free_gb: number };
  disk: { path: string; total: number; free: number };
  resident: { processes: number; vram_mb: number };
  server: { version: string; boot: string; started: number; pid: number; address: string; command: string };
  pending: { label: string; page: string }[]; // settings waiting for a restart, each with the settings page it is on
  feedback_new?: number; // users' feedback not yet looked at (only for a login that may open 用户反馈)
  registering: Registering;
}

/** 概览的「今天和最近」 (GET /api/admin/overview/recent; server/settings.py admin_overview_recent). The first day of
 * each period (server's local time), and each group of numbers with the section it opens; a group the login may not
 * open that section for is not in the answer at all. */
export interface RecentView {
  days: Record<RecentPeriod, string>;
  access?: { section: string; online: OnlineSummary } & Record<"today" | "days7" | "month", { people: number; logins: number; failed: number }>;
  accounts?: { section: string } & Record<"today" | "week" | "month", { all: number; self: number }>;
  tasks?: { section: string } & Record<"today" | "days7" | "month", { all: number; done: number; partial: number; failed: number; cancelled: number }>;
  traffic?: { section: string } & Record<"today" | "days7" | "month", number>;
  feedback?: { section: string; open: number; new: number } & Record<"today" | "days7" | "month", { came: number }>;
}

export type RecentPeriod = "today" | "days7" | "week" | "month";

/** 自行注册 now (lab2shot/registration.py counts): open or not, registrations lately against the site-wide limits,
 * and whether that paused registering ("hour" / "day"; "" not paused). */
export interface Registering {
  open: boolean;
  invite: boolean;
  hour: number;
  day: number;
  per_hour: number;
  per_day: number;
  paused: "" | "hour" | "day";
}

/** 邀请码 (server/invites.py): a code as the list shows it, the code itself included (the page copies it). */
export interface Invite {
  id: number;
  code: string; // as it is handed out
  hint: string; // its first characters: what logs and audit lines name it by
  note: string;
  uses_max: number | null; // null: no limit
  used: number;
  expires: number | null; // null: never
  enabled: boolean;
  created: number;
  created_by: string;
  state: string; // 可以用 / 已停用 / 已过期 / 已用完
  usable: boolean;
  accounts: { id: number; username: string; name: string; at: number; ip: string; enabled: boolean; deleted: number | null }[];
}

export interface InvitesView {
  invites: Invite[];
  registering: Registering;
  rules: { min: number; max: number; note_most: number; uses_most: number }; // what a typed code, a note and a use limit may be
}

export interface Registered {
  id: number;
  username: string;
  name: string;
  department: string;
  role: string;
  enabled: boolean;
  registered: number;
  invite: string; // the code's first characters ('' without one)
  ip: string;
  managed: boolean; // this login may disable it
}

export interface RegisteredFilter {
  invite?: number;
  since?: number;
  until?: number;
}

/** The server refused some changes: key -> why. */
export class SettingsRefused extends Error {
  errors: Record<string, string>;
  constructor(message: string, errors: Record<string, string>) {
    super(message);
    this.errors = errors;
  }
}


/** 用户反馈 (lab2shot/feedback.py). */
export type FeedbackStatus = "new" | "seen" | "solved";

interface FeedbackItem {
  id: string;
  at: number;
  user: number; // the account that sent it
  username: string;
  person: string; // its Chinese name (「已删除的用户」 once deleted)
  department: string;
  category: "" | "error" | "usage" | "idea";
  category_label: string;
  text: string;
  title: string; // the first line
  status: FeedbackStatus;
  status_label: string;
  reply: string; // the administrator's answer: the user sees it in 我的反馈
  replied: number | null;
  replied_by: string;
  changed: number | null; // when what the user sees (status, reply) last changed
  unread: boolean; // the user has not looked since
  note: string; // the administrator's own: the user never sees it
  updated: number | null;
  updated_by: string;
  images: string[]; // file names (feedbackFile)
  bytes: number;
}

/** What came with a feedback: what the page collected (diagnostics.ts) and what the server added. */
interface FeedbackBundle {
  page: {
    collected?: number;
    page?: string;
    browser?: Record<string, unknown>;
    log?: { t: number; level: string; text: string; count?: number; last?: number }[];
    errors?: { t: number; count: number; last: number; kind: string; message: string; where: string; stack: string }[];
    requests?: { t: number; method: string; url: string; status: number; message: string; ms: number }[];
    editor?: Record<string, unknown>;
    help?: Record<string, unknown>; // a help page: which one, the project it shows, its install
    graph?: unknown;
  };
  server: {
    revision?: string;
    client?: Record<string, unknown>;
    jobs?: {
      id: string;
      title: string;
      state: string;
      submitted: number;
      error: string | null;
      error_log: { file: string; lines: string[] } | null;
      server_log: string[];
    }[];
    server_log?: string[];
    [more: string]: unknown;
  };
}

export type FeedbackDetail = FeedbackItem & { bundle: FeedbackBundle | null };

/** The records' database (lab2shot/database). */
export interface DatabaseView {
  path: string;
  bytes: number;
  version: number;
  checked: { at?: number; ok?: boolean; detail?: string };
  last_backup: { at: number; file: string; reason: string } | null;
  keep: number;
  backups: { name: string; bytes: number; at: number }[];
}

/** What looked like probing (auth.Watch). */
interface SuspiciousEvent {
  t: number;
  kind: string;
  detail: string;
  who: string;
  client: string;
  user: string; // the account, when the request carried one
  path: string;
  method: string;
  counted: boolean; // 是否计入封禁阈值的 40 条：页面与服务器版本不一致导致的 404 会被记录但不计入
}

export interface SecurityView {
  events: SuspiciousEvent[];
  counts: Record<string, number>;
  blocked: { client: string; until: number; who: string; user: string; why: string }[];
  online: OnlineSummary;
  limits: Record<string, number>;
}


/** 按用户查一切 (lab2shot/resources.py): every kind of thing an account owns that this login may see, with how
 * many it has (GET /api/admin/users/{id}/resources), and one page of one kind with the columns it declares
 * (.../resources/{kind}). The 用户 detail page draws a tab per kind; the types are here so the routes are named once. */
export interface UserResourceTab {
  kind: string;
  label: string;
  section: string; // where a row opens (the admin page's section)
  count: number;
}

export interface UserResourcePage {
  kind: string;
  label: string;
  section: string;
  columns: { key: string; label: string; says: string }[]; // says: "size" 表示字节数（由 platform/format.ts 格式化），"" 表示按值本身显示
  when: string; // the column the 时间 filter reads ("" no such filter)
  state_column: string; // the column the 状态 filter reads ("" no such filter)
  states: string[]; // the values that column takes for this account, in the order they appear
  dim: boolean; // rows may carry `__dim`: no longer in use, drawn greyed by the shared table
  acts: ResourceAct[]; // what this login may do to one row ([]: no action column)
  total: number; // rows matching the search and the filters (the page is a cut of them)
  offset: number;
  rows: Record<string, unknown>[];
}

/** What a resource list asks for: the search, the state chip and the time chip (the server filters, so the
 * 用户 detail page and a resource's own list page ask the same way). */
interface ResourceQuery {
  offset?: number;
  limit?: number;
  q?: string;
  since?: number; // only rows at or after this moment (0: all of them)
  state?: string; // only rows whose state column is this ("" all of them)
}

/** 用户协议 and 隐私政策 as the administrator edits them (lab2shot/server/terms.py): the texts as written, the
 * placeholders they may use with what each says now, and how many of the accounts that must agree did. */
export interface TermsAdminView {
  version: number;
  at: number;
  by: string; // who saved this version; "" for the program's own text
  edited: boolean; // the administrator's copy is in effect
  most: number; // characters one text may have
  documents: { id: "agreement" | "privacy"; title: string; text: string }[];
  fills: Record<string, string>; // placeholder -> what it says now
  accounts: number;
  agreed: number;
}

export const adminApi = {
  ...templatesApi,
  users: () => json<UsersView>("GET", "/api/admin/users"),
  userResources: (id: number) => json<{ user: number; tabs: UserResourceTab[] }>("GET", `/api/admin/users/${id}/resources`),
  userResource: (id: number, kind: string, { offset = 0, limit = 50, q = "", since = 0, state = "" }: ResourceQuery = {}) =>
    json<UserResourcePage>("GET",
      `/api/admin/users/${id}/resources/${encodeURIComponent(kind)}?${new URLSearchParams({ offset: String(offset), limit: String(limit), q, since: String(since), state })}`),
  createUser: (u: NewUser) => json<UsersView & { user: UserRow }>("POST", "/api/admin/users", u),
  changeUser: (id: number, change: UserChange) => json<UsersView & { user: UserRow }>("PUT", `/api/admin/users/${id}`, change),
  resetPassword: (id: number, password: string) => json<UsersView>("POST", `/api/admin/users/${id}/password`, { password }),
  deleteUser: (id: number) => json<UsersView & { jobs_stopped: number; outputs: number }>("DELETE", `/api/admin/users/${id}`),
  // 永久删除已删除的账号：账号行被移除，用户名可供新账号重用；
  // 任务记录、反馈、登录记录保留，统计中显示为「已删除的用户」；保存在服务器上的节点图一并删除
  purgeUser: (id: number) => json<UsersView & { jobs: number; feedback: number; graphs: number }>("DELETE", `/api/admin/users/${id}/purge`),
  userLogins: (id: number) => json<UserLogins>("GET", `/api/admin/users/${id}/logins`),
  ...accountsApi,
  security: () => json<SecurityView>("GET", "/api/admin/security"),
  unblock: (client: string) => json<SecurityView>("POST", "/api/admin/security/unblock", { client }),
  database: () => json<DatabaseView>("GET", "/api/admin/db"),
  backupNow: () => json<DatabaseView>("POST", "/api/admin/db/backup", {}),
  checkNow: () => json<DatabaseView>("POST", "/api/admin/db/check", {}),
  settings: () => json<SettingsView>("GET", "/api/admin/settings"),
  saveSettings: (values: Record<string, SettingValue>): Promise<SettingsView> =>
    json<SettingsView>("PUT", "/api/admin/settings", { values }).catch((e) => {
      throw e instanceof ApiError && e.status === 400 ? new SettingsRefused(e.message, (e.body?.errors as Record<string, string>) ?? {}) : e;
    }),
  overview: () => json<Overview>("GET", "/api/admin/overview"),
  recent: () => json<RecentView>("GET", "/api/admin/overview/recent"),
  invites: () => json<InvitesView>("GET", "/api/admin/invites"),
  createInvite: (i: { code: string; note: string; uses_max: number | null; expires: number | null }) =>
    json<InvitesView & { made: Invite }>("POST", "/api/admin/invites", i),
  changeInvite: (id: number, i: { note: string; uses_max: number | null; expires: number | null; enabled: boolean }) =>
    json<InvitesView>("PUT", `/api/admin/invites/${id}`, i),
  deleteInvite: (id: number) => json<InvitesView>("DELETE", `/api/admin/invites/${id}`),
  registered: (f: RegisteredFilter) =>
    json<{ accounts: Registered[] }>("GET", `/api/admin/registrations?${new URLSearchParams(Object.entries(f).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]))}`),
  disableRegistered: (f: RegisteredFilter) => json<InvitesView & { disabled: string[] }>("POST", "/api/admin/registrations/disable", f),
  terms: () => json<TermsAdminView>("GET", "/api/admin/terms"),
  saveTerms: (t: { agreement: string; privacy: string }) => json<TermsAdminView>("PUT", "/api/admin/terms", t),
  resetTerms: () => json<TermsAdminView>("DELETE", "/api/admin/terms"),
  notice: () => json<ServerNoticeText>("GET", "/api/admin/notice"),
  setNotice: (n: { text: string; tone: string; on: boolean }) => json<ServerNoticeText>("PUT", "/api/admin/notice", n),
  queueSwitches: (switches: { gpu?: boolean; compute?: boolean }) => json<QueueView>("PUT", "/api/admin/queue/switches", switches),
  restart: (mode: "drain" | "now") => json<ServerInfo>("POST", "/api/admin/restart", { mode }),
  callOff: () => json<ServerInfo>("POST", "/api/admin/restart/cancel", {}),
  feedback: ({ status, since, person }: { status: string; since?: number; person: string }) =>
    json<{ items: FeedbackItem[]; counts: Record<FeedbackStatus, number> }>("GET", 
      `/api/admin/feedback?${new URLSearchParams({ status, person, ...(since ? { since: String(since) } : {}) })}`,
    ),
  feedbackDetail: (id: string) => json<FeedbackDetail>("GET", `/api/admin/feedback/${id}`),
  answerFeedback: (id: string, reply: { status: FeedbackStatus; reply: string; note: string }) => json<FeedbackItem>("PUT", `/api/admin/feedback/${id}`, reply),
  deleteFeedback: (id: string) => json<unknown>("DELETE", `/api/admin/feedback/${id}`),
  feedbackFile: (id: string, name: string) => `/api/admin/feedback/${id}/files/${encodeURIComponent(name)}`,
  feedbackDownload: (id: string) => `/api/admin/feedback/${id}/download`,
};
