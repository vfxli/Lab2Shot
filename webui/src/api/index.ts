// Backend contract. Shapes mirror lab2shot/nodes/base.py describe() and server/app.py.

import { clipboardUrl } from "../transfer/frameKey";
import type { Availability, MessageJson } from "./applies";
import { clientInfo, deviceId } from "../platform/client";
import { ApiError, json } from "../platform/http";
import { followEvents } from "../platform/events";
import type { Output, Upload } from "./files";
import { graphKept, graphRef, unknownGraph, type GraphRef } from "../model/graphSync";
import type { BackgroundTask } from "./tasks";
import type { Catalog, GraphJSON, OcioInfo, TemplatesPage } from "./catalog";
import type { Licence, ManualNeed, ManualView } from "./extensions";
import type { DiskUsage, HistoryJob, JobLoad, JobRecord, QueueView, ResidentView, ServerLoad } from "./queue";
import { normalizeStatus, type Choice, type ClipboardText, type ItemsPage, type StatusReply } from "./status";
import { usageFilled, type UsageStats } from "./usage";
import { libraryApi } from "./library";
import { getLang, type Lang } from "../i18n/lang";

export * from "./catalog";
export * from "./progress";
export * from "./queue";
export * from "./usage";
export * from "./status";
export * from "./extensions";
export * from "./library";

/** An account (lab2shot/accounts.py) as its own pages know it. */
export interface Account {
  id: number;
  username: string;
  name: string;
  department: string; // its value (stored)
  department_label: string; // how it shows (lab2shot/accounts.py department_label)
  role: string; // lab2shot/roles.py, shown by its label: what the pages offer comes from AuthState's availability
  role_label: string;
  tags: string[]; // what it may use besides the basics (nodes/tags.py)
  expires: number | null; // null: never (the built-in administrator account)
  lang?: string; // the language the account chose ("": none, the browser's)
}

/** Another login of this account (same kind: a browser, or a DCC plugin/command line) ended this one: one place
 * online per account, per kind (lab2shot/accounts.py start()). `detail` is the ready-made sentence the gate shows. */
interface KickedInfo {
  at: number;
  ip: string;
  device: string;
  kind: "web" | "client";
  detail: string;
}

/** This browser's login: the account (null: not logged in) and what applies to it on the admin side, resolved by the
 * server (lab2shot/server/available.py; read only through applies.ts); until when its rights last (the
 * password typed within the last three days); for the built-in administrator account, whether the password is still the
 * default one and whether a 口令 is set on the server. `kicked`: this browser is not logged in because another login of
 * the same account ended it; the gate shows this instead of a plain login form. */
export interface AuthState {
  user: Account | null;
  applies: Availability; // what of the admin side is there for it (applies.ts reads it)
  expires?: number | null;
  admin_until?: number;
  passphrase?: boolean;
  kicked?: KickedInfo;
  settings_pages?: SettingsPageEntry[]; // signed in: the admin side list's 设置 band (lab2shot/config.py PAGES)
  lang?: string; // what the server speaks to this login (server/lang.py): the page's language (i18n/lang.ts)
}

/** One settings page as the admin side list shows it: its id (its section of the admin page is settings-<id>), name
 * and hover text. */
export interface SettingsPageEntry {
  id: string;
  label: string;
  tip: string;
}


/** An ask about the graph: the graph goes once per version, later asks name it (model/graphSync.ts); a server that
 * does not have that version (restarted) gets the whole graph once more. */
async function askGraph<T>(url: string, graph: GraphJSON, extra: Record<string, unknown>, signal?: AbortSignal): Promise<T> {
  const asking = (ref: GraphRef) => json<T>("POST", url, { ...ref, ...extra }, signal ? { signal } : undefined);
  let ref = graphRef(graph);
  let data: T;
  try {
    data = await asking(ref);
  } catch (e) {
    if (!(e instanceof ApiError && unknownGraph(e.status, e.body as { graph?: string } | null))) throw e;
    data = await asking((ref = graphRef(graph, true)));
  }
  graphKept(ref);
  return data;
}


/** The queue as the editor's 队列 window reads it: the poll (/api/queue) carries the jobs going and only the version of
 * the account's task history; the history itself (/api/queue/history: every unexpired task, their groups and cache
 * marks) is asked for when that version is not the one kept here, so an unchanged history is never sent again. The
 * answer is the same either way: the poll's with `history` filled in. */
// kept per language too: the history's titles, errors and reasons are said in the language it was asked in (a page that
// switched language asks again, server/farm.py queue_history)
type HistoryKept = { version: string; history: HistoryJob[]; lang?: Lang };
let historyKept: HistoryKept | null = null;
// the history being asked for, for the version it was asked for: calls at the same moment (a page opening polls and
// resumes its jobs at once) share one request
let historyAsked: { version: string | undefined; lang: Lang; answer: Promise<HistoryKept> } | null = null;

async function queueWithHistory(load: boolean): Promise<QueueView> {
  const view = await json<QueueView>("GET", `/api/queue${load ? "" : "?load=0"}`);
  const lang = getLang();
  if (!historyKept || historyKept.version !== view.history_version || historyKept.lang !== lang) {
    if (historyAsked?.version !== view.history_version || historyAsked?.lang !== lang) {
      const answer = json<HistoryKept>("GET", "/api/queue/history").then((got) => ({ ...got, lang }));
      historyAsked = { version: view.history_version, lang, answer };
      answer.then((got) => (historyKept = got), () => undefined).finally(() => {
        if (historyAsked?.answer === answer) historyAsked = null;
      });
    }
    historyKept = await historyAsked!.answer;
  }
  return { ...view, history: historyKept.history };
}

/** One submission's own: its submit key and the step's time limit (graph/submitLine.ts). */
interface Submitting {
  submit?: string;
  signal?: AbortSignal;
}

export const api = {
  catalog: () => json<Catalog>("GET", "/api/catalog"),
  templates: () => json<TemplatesPage>("GET", "/api/templates"),
  // my templates and my disk use (api/library.ts)
  ...libraryApi,
  // installing and what is downloaded by hand: the administrator's (logged in)
  installs: {
    /** Installs (background tasks of the farm): every extension's state, the ones running now, the checklist, start
     * one, read its steps and output lines since a count, cancel, roll back, uninstall. */
    extensions: () => json<{ extensions: ExtensionRow[] }>("GET", "/api/admin/extensions"),
    list: () => json<{ jobs: InstallTask[] }>("GET", "/api/admin/installs"),
    preflight: (name: string) => json<Checklist>("GET", `/api/admin/extensions/${encodeURIComponent(name)}/preflight`),
    start: (name: string, rebuild = false) => json<InstallTask>("POST", "/api/admin/installs", { name, rebuild }),
    job: (id: string, since: number) => json<InstallTask>("GET", `/api/admin/installs/${encodeURIComponent(id)}?since=${since}`),
    cancel: (id: string) => json<InstallTask>("POST", `/api/admin/installs/${encodeURIComponent(id)}/cancel`, {}),
    rollback: (name: string) => json<unknown>("POST", `/api/admin/extensions/${encodeURIComponent(name)}/rollback`, {}),
    uninstall: (name: string) => json<unknown>("POST", `/api/admin/extensions/${encodeURIComponent(name)}/uninstall`, {}),
  },
  manual: {
    view: () => json<ManualView>("GET", "/api/admin/manual"), // looking sorts the inbox: what is recognised gets installed
    licence: (file: string) => json<Licence>("GET", `/api/admin/manual/licence?file=${encodeURIComponent(file)}`),
    /** Only the built-in administrator's own click on 同意并安装 (logged in): records who accepted which licence, then installs. */
    accept: (file: string, licence: string) => json<ManualView>("POST", "/api/admin/manual/accept", { file, licence, client: clientInfo() }),
  },
  ocio: () => json<OcioInfo>("GET", "/api/ocio"),
  derive: (typeId: string, params: Record<string, unknown>) => json<Record<string, unknown>>("POST", `/api/nodes/${encodeURIComponent(typeId)}/derive`, { params }),
  // the reverse path: a pasted piece of text -> a set of this node's parameter values. The server knows no format; the node parses it itself
  paste: (typeId: string, text: string) => json<Record<string, unknown>>("POST", `/api/nodes/${encodeURIComponent(typeId)}/paste`, { text }),
  choices: (typeId: string, params: Record<string, unknown>, inputs: Record<string, string>) =>
    json<Record<string, Choice>>("POST", `/api/nodes/${encodeURIComponent(typeId)}/choices`, { params, inputs }),
  // `view`: which item each 逐项处理 block is showing (state/items.ts). It is a view setting, so the nodes
  // inside a block answer for that item; it never changes what a cook is.
  // `handles`: whose handle data to send (`node`, "" none: state/handleView.ts handleNode) and the key of
  // the copy the page holds of it (StatusReply handle_data): while that still matches, the reply leaves the (large)
  // data out
  // `show`: the displayed node's outputs the viewer shows, as `cook` sends them: its plan is judged as that cook would be
  // `signal`: the submission's own time limit (graph/submitLine.ts); the editor's own asks have none
  stopReadings: () => json<{ stopped: number }>("POST", "/api/readings/stop", {}),
  // `holds`: the other nodes whose 「计算」 the page offers (a card's buttons): the reply says why each can't cook now
  status: (graph: GraphJSON, cookInputs: number, display: string | null, view: Record<string, string> = {}, handles: { node: string; key: string } = { node: "", key: "" }, show: string[] = [], signal?: AbortSignal, holds: string[] = []) =>
    askGraph<StatusReply>("/api/status", graph, {
      cook_inputs: cookInputs, display, ...(show.length ? { show } : {}), ...(Object.keys(view).length ? { view } : {}), ...(holds.length ? { holds } : {}),
      ...(handles.node ? { handle_node: handles.node } : {}), ...(handles.key ? { handle_key: handles.key } : {}),
    }, signal).then(normalizeStatus),
  /** Item by item for one node inside a 逐项处理 block: a page at a time, of the graph version the
   * status reply came with. The status reply itself is never one entry per item. */
  nodeItems: (graph: string, node: string, offset = 0, limit = 50) =>
    json<ItemsPage>("GET", `/api/status/${encodeURIComponent(graph)}/node/${encodeURIComponent(node)}/items?offset=${offset}&limit=${limit}`),
  // POST /api/jobs, the one way anything queues a graph (server/farm.py JobRequest): exactly one of cook (a node and
  // its upstream) or deliver ([]: every 「输出」 in the graph, like Nuke's Render All). `version`: the cook inputs'
  // version of the graph (state/cookInputs.ts), which every event of the job carries back; a node_done applies to the
  // page only while it is still that version (graph/streamDone.ts).
  // `show`: the output ports the viewer is showing (the server cooks only what they need); left out when there are none.
  // `follows`: the account's own job this one follows (the focus mode's 「计算」, editor/AppMode.tsx: a DCC plugin finds
  // its job's newest result by it). `submit`: the click's submit key (graph/submitLine.ts): sent again after a lost
  // answer, it never makes a second job; `signal`: the step's time limit
  cook: (graph: GraphJSON, version: number, target: string, show: string[] = [], follows?: string, o: Submitting = {}) =>
    askGraph<{ job: string }>("/api/jobs", graph, { cook: target, version, ...(show.length ? { show } : {}), ...(follows ? { follows } : {}), ...(o.submit ? { submit: o.submit } : {}), client: clientInfo() }, o.signal),
  deliver: (graph: GraphJSON, version: number, follows?: string, o: Submitting = {}) =>
    askGraph<{ job: string }>("/api/jobs", graph, { deliver: [], version, ...(follows ? { follows } : {}), ...(o.submit ? { submit: o.submit } : {}), client: clientInfo() }, o.signal),
  /** A job's state now (the first check after submitting: is it there); `since` at its maximum, so no event comes back. */
  jobState: (job: string, signal?: AbortSignal) =>
    json<{ state: string; done: boolean }>("GET", `/api/jobs/${encodeURIComponent(job)}/state?since=${Number.MAX_SAFE_INTEGER}`, undefined, signal ? { signal } : undefined),
  // the job's events as a stream that mends itself (platform/events.ts: nothing else opens an EventSource)
  // after a refusal, the reason is asked through the job's state (no stream; since at its maximum, so no event comes back)
  cookEvents: (job: string) => followEvents(`/api/jobs/${job}/events`, `/api/jobs/${job}/state?since=${Number.MAX_SAFE_INTEGER}`),
  cancelCook: (job: string) => json<{ ok: boolean }>("POST", `/api/jobs/${job}/cancel`, {}),
  /** Deletes one finished task: its whole task folder (graph, footage, output folders and zips, log) goes; cache only it
   * refers to is cleaned by the server afterwards. A task still computing is cancelled first. */
  forgetJob: (job: string) => json<{ ok: boolean }>("DELETE", `/api/jobs/${encodeURIComponent(job)}`),
  // free space in one click: delete all of one's own finished tasks and the space they hold
  forgetAllJobs: () => json<{ ok: boolean; jobs: number; skipped: number; bytes: number }>("DELETE", "/api/jobs"),
  /** Deletes one of one's own task groups: each finished task in it goes whole, as a single delete would; tasks waiting or computing are skipped (`skipped`). */
  forgetGroup: (key: string) =>
    json<{ ok: boolean; jobs: number; skipped: number; bytes: number }>("DELETE", `/api/task-groups/${encodeURIComponent(key)}`),
  /** Renames one of one's own task groups (display only; the grouping stays); an empty name returns to the automatic one. The answer is the name the group shows now. */
  renameGroup: (key: string, name: string) =>
    json<{ ok: boolean; name: string }>("PUT", `/api/task-groups/${encodeURIComponent(key)}/name`, { name }),
  // load: with the machine's load (the queue panel open); without it an idle queue's answer stays the same (a 304)
  queue: (load = true) => queueWithHistory(load),
  /** How busy the server is, for the top bar's load pill: small enough to ask for on every poll. */
  load: () => json<ServerLoad>("GET", "/api/load"),
  /** One of the account's jobs again: its graph as submitted and whether its results are still cached. */
  job: (id: string) => json<JobLoad>("GET", `/api/jobs/${encodeURIComponent(id)}`),
  /** Every output of the account's tasks still kept on the server, newest first (lab2shot/transfer/outputs.py). */
  outputs: () => json<Output[]>("GET", "/api/outputs"),
  uploads: {
    describe: (ref: string) => json<Upload>("GET", `/api/uploads/describe?ref=${encodeURIComponent(ref)}`),
    /** Declares an upload: its files, the sha256 of each file's content (computed by the page on the user's machine),
     * and the first few tens of KB of the first file. From these the server computes the upload's id (the same formula
     * `POST /api/uploads` uses once the bytes have arrived), reads the layers with its own describe_file, and returns
     * the final reference. No data bytes are transferred. When the head sent is too short to read the layers, the
     * answer carries `need` (how many more bytes) and the page sends them; empty layers are never returned. */
    declare: (body: { name: string; files: Record<string, string>; sizes: Record<string, number>; origin: object;
                      head_of?: string; head?: string; head_whole?: number }) =>
      json<{ ref: string; missing: string[]; layers?: Record<string, unknown>; need?: number }>("POST", "/api/uploads/declare", body),
    /** Which of these names (per folder) make sequences: lab2shot/io/sequence.py, the one place that decides it. */
    sequences: (folders: string[][]) =>
      json<{ folders: { sequences: { name: string; frames: number[]; files: number[] }[]; singles: number[]; junk: number[] }[] }>("POST", "/api/uploads/sequences", { folders }),
  },
  admin: {
    queue: () => json<QueueView>("GET", "/api/admin/queue"),
    authorize: (uuids: string[]) => json<QueueView>("PUT", "/api/admin/gpus", { authorized: uuids }),
    cancel: (job: string) => json<{ ok: boolean }>("POST", `/api/admin/jobs/${job}/cancel`, {}),
    forget: (job: string) => json<{ ok: boolean }>("DELETE", `/api/admin/jobs/${encodeURIComponent(job)}`),
    // one account's group of tasks, every finished task of it (the same path as the account's own 删除组)
    forgetGroup: (user: number, key: string) =>
      json<{ ok: boolean; jobs: number; skipped: number; bytes: number }>("DELETE", `/api/admin/task-groups/${user}/${encodeURIComponent(key)}`),
    // rename one account's group (the server records who renamed whose group to what)
    renameGroup: (user: number, key: string, name: string) =>
      json<{ ok: boolean; name: string }>("PUT", `/api/admin/task-groups/${user}/${encodeURIComponent(key)}/name`, { name }),
    // 插队: put a task still to finish at the front of the queue; the server records who moved what from where
    // (lab2shot/server/access.py audit) and answers with the queue as it now is
    first: (job: string) => json<QueueView>("POST", `/api/admin/jobs/${encodeURIComponent(job)}/first`, {}),
    // `user`: only this account's jobs (the queue's 「按人」), filtered by the server
    history: (user: number | null = null) => json<JobRecord[]>("GET", `/api/admin/history${user === null ? "" : `?user=${user}`}`),
    disk: (fresh = false) => json<DiskUsage>("GET", `/api/admin/disk${fresh ? "?fresh=1" : ""}`),
    log: (lines = 500) => json<{ file: string; lines: string[] }>("GET", `/api/admin/log?lines=${lines}`),
    clean: (area: string, days: number) => json<{ removed: number; bytes: number }>("POST", "/api/admin/disk/clean", { area, days }),
    graphUrl: (job: string) => `/api/admin/jobs/${job}/graph`,
    resident: () => json<ResidentView>("GET", "/api/admin/resident"),
    offload: (id: string) => json<ResidentView>("POST", `/api/admin/resident/${id}/offload`, {}),
    unload: (id: string) => json<ResidentView>("POST", `/api/admin/resident/${id}/unload`, {}),
    usage: (since: number | null, until: number | null) => {
      const q = new URLSearchParams({ tz: String(-new Date().getTimezoneOffset()) });
      if (since !== null) q.set("since", String(since));
      if (until !== null) q.set("until", String(until));
      return json<UsageStats>("GET", `/api/admin/usage?${q}`).then(usageFilled);
    },
    resetUsage: () => json<{ start: number }>("POST", "/api/admin/usage/reset", {}),
    undoReset: () => json<{ start: number | null }>("POST", "/api/admin/usage/reset/undo", {}),
  },
  /** The node text an output-settings node wrote for another application to paste: which application reads
   * it, the file it was written as, and the text itself. Only a result whose meta declares one has it. */
  clipboard: (fp: string) => json<ClipboardText>("GET", clipboardUrl(fp)),
  sendLog: (text: string) => json<{ ok: boolean }>("POST", "/api/logs", { text, client: clientInfo() }),
  /** Logging in with an account (lab2shot/server/auth.py); the administrator's forgotten password with the 口令. */
  /** The account's own choices (server/lang.py). */
  me: {
    lang: (lang: string) => json<{ lang: string }>("PUT", "/api/me/lang", { lang }),
    // messages the server said before, said again in the page's language now (the log after a switch; server/lang.py)
    said: (messages: { code: string; params: Record<string, unknown> }[]) =>
      json<{ texts: (string | null)[] }>("POST", "/api/said", { messages }),
  },
  auth: {
    state: () => json<AuthState>("GET", "/api/auth/state"),
    login: (username: string, password: string) => json<AuthState>("POST", "/api/auth/login", { username, password, device_id: deviceId() }),
    logout: () => json<AuthState>("POST", "/api/auth/logout", {}),
    change: (current: string, next: string) => json<AuthState>("POST", "/api/auth/password", { current, new: next }),
    recover: (passphrase: string, next: string) =>
      json<AuthState>("POST", "/api/auth/recover", { passphrase, new: next, device_id: deviceId() }),
  },
};

/** The server itself (GET /api/server, lab2shot/server/settings.py): which run of it answers, whether a restart is coming. */
export interface RestartState {
  state: "draining" | "restarting"; // draining: waiting for the jobs running
  mode: "drain" | "now";
  since: number;
  // what it waits for: only a logged-in page is told (the login page learns only where to connect next)
  running?: number;
  tasks?: MessageJson[]; // the background tasks still going (their titles): an install, a card's check
  port: number; // where the next server listens
  https: boolean;
}

/** Without a login (the login page) only `boot`, `ui`, `restart` (where to connect next) and `account` come. */
export interface ServerInfo {
  boot: string;
  started?: number;
  version?: string;
  ui: number; // when the built page was made: another one after a restart means the page should reload
  restart: RestartState | null;
  notice?: number; // when the administrator's notice last changed: another number means reading /api/notice again
  account: number | null; // the account this browser is logged in as now (null: none): another than the page's opens it again
  // the two local-proxy settings (the administrator's 「视图 · 本机代理尺寸 / 本机缓存上限」, lab2shot/config.py): the
  // browser makes and evicts its local proxies by them (transfer/localProxy)
  view?: { local_px: number; local_cache_gb: number };
  // the administrator's 「任务保留天数」 (lab2shot/config.py tasks.keep_days): how long a paused upload in the browser may
  // sit untouched before it is purged (transfer/uploads.ts purgeStale); the same number as the server's retention of
  // tasks and cache, never a separate one. A server that does not send it (an older version): nothing is purged
  tasks?: { keep_days: number };
}

/** The administrator's notice (GET /api/notice): one line every page shows at the top while it is on. The type is
 * here so the route is named once. */
export interface ServerNoticeText {
  text: string;
  tone: "info" | "notice" | "warn" | "risk"; // the four meanings of the design tokens (lab2shot/server/notice.py TONES)
  on: boolean;
  updated: number; // when it last changed (0: never)
  by?: string; // which administrator wrote it
}

/** One extension's row in the admin page's 扩展包 section (GET /api/admin/extensions, lab2shot/server/installs.py
 * extension_list): its install state, what is missing, what this login may do (`actions`, computed by the server; the
 * page checks no role), and its latest install task. `manual`: the items still to be downloaded by hand. */
export interface ExtensionRow {
  name: string;
  title: string;
  summary: string;
  nodes: number; // how many nodes the extension adds to the editor
  installed: boolean;
  ready: boolean;
  label: string; // its state in brief: 已就绪 / 未安装 / 模型未齐 / 需要重装…
  reason: string; // why it is not usable yet ("" when 已就绪)
  manual: ManualNeed[];
  actions: Availability;
  job: InstallTask | null;
}

/** An install (lab2shot/server/installs.py): a background task of the farm in the one progress format, with the state
 * of each step beside it for the step bar (the task's `done` of `total` and `label` say which step it is on). */
export interface InstallTask extends BackgroundTask {
  steps: InstallStep[];
}

export interface Checklist {
  name: string;
  title: string;
  ready: boolean;
  checks: { kind: string; label: string; state: "ok" | "notice" | "warning" | "blocked"; message: MessageJson }[];
}

export interface InstallStep {
  id: string;
  label: string;
  state: "waiting" | "running" | "done" | "skipped" | "failed" | "cancelled";
  message?: MessageJson;
}

