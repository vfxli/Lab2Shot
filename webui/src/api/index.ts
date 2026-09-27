// Backend contract. Shapes mirror lab2shot/nodes/base.py describe() and server/app.py.

import type { Availability, MessageJson } from "./applies";
import { clientInfo, deviceId } from "../platform/client";
import { ApiError, json } from "../platform/http";
import { followEvents } from "../platform/events";
import { addressOf, type Delivery, type Upload } from "./deliveries";
import { graphKept, graphRef, unknownGraph, type GraphRef } from "../model/graphSync";
import type { BackgroundTask } from "./tasks";
import type { Catalog, GraphJSON, OcioInfo, TemplatesPage } from "./catalog";
import type { Licence, ManualNeed, ManualView } from "./extensions";
import type { DiskArea, JobLoad, JobRecord, QueueView, ResidentView, ServerLoad } from "./queue";
import type { BoxesData, Choice, ClipboardText, CurvesData, ItemsPage, Manifest, StatusReply, TracksData } from "./status";
import type { UsageStats } from "./usage";
import { libraryApi } from "./library";

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
  department: string;
  role: string; // lab2shot/roles.py, shown by its label: what the pages offer comes from AuthState's availability
  role_label: string;
  tags: string[]; // what it may use besides the basics (nodes/tags.py)
  expires: number | null; // null: never (the built-in administrator account)
}

/** Another login of this account (same kind: a browser, or a DCC plugin/command line) ended this one: one place
 * online per account, per kind (lab2shot/accounts.py start()). `detail` is the ready-made sentence the gate shows. */
export interface KickedInfo {
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
}


/** An ask about the graph: the graph goes once per version, later asks name it (model/graphSync.ts); a server that
 * does not have that version (restarted) gets the whole graph once more. */
async function askGraph<T>(url: string, graph: GraphJSON, extra: Record<string, unknown>): Promise<T> {
  const asking = (ref: GraphRef) => json<T>("POST", url, { ...ref, ...extra });
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

/** 透过指定相机查看一帧底图（`lab2shot/server/packets.py frame` 的 `through` / `at`）：相机包的指纹及该相机在
 * 场景中的路径。相机带畸变时，服务器按其镜头对该帧去畸变后发送（`view/proxy.py through_picture_file`），
 * 不带畸变时发送普通版本。仅在三维舞台透过带畸变的相机观看时使用（view/Stage3D.tsx）。 */
export interface Through {
  fp: string;
  at: string;
}

/** 地址中用于区分字节内容的两项（lab2shot/server/wire.py versioned）：包的代次 `g`（状态回复中的 gens，
 * 同一指纹重新计算后改变）与代理档位 `px`（包说明中的 proxy.px）。地址包含这两项，服务器才会标记 immutable，
 * 浏览器缓存才不会在重算或更换档位后返回旧字节；页面自身的缓存键（transfer/ident.ts）使用同一对值，两端标识一致。 */
export interface Version {
  g: string;
  px?: number;
}

const versionQuery = (v?: Version): string[] => (v ? [...(v.g ? [`g=${v.g}`] : []), ...(v.px ? [`px=${v.px}`] : [])] : []);
const query = (parts: string[]): string => (parts.length ? `?${parts.join("&")}` : "");

export const api = {
  catalog: () => json<Catalog>("GET", "/api/catalog"),
  templates: () => json<TemplatesPage>("GET", "/api/templates"),
  // 我的模板与个人磁盘占用 (api/library.ts)
  ...libraryApi,
  // installing and what is downloaded by hand: the administrator's (logged in)
  installs: {
    /** Installs (background tasks of the farm): every extension's state, the ones running now, the checklist, start
     * one, read its steps and output lines since a count, cancel, roll back, uninstall. */
    extensions: () => json<{ extensions: ExtensionRow[] }>("GET", "/api/admin/extensions"),
    list: () => json<{ jobs: InstallTask[] }>("GET", "/api/admin/installs"),
    preflight: (name: string) => json<Checklist>("GET", `/api/admin/extensions/${encodeURIComponent(name)}/preflight`),
    start: (name: string, force = false) => json<InstallTask>("POST", "/api/admin/installs", { name, force }),
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
  // 反向路径：粘贴的一段文本 → 该节点的一组参数值。服务器不识别格式，由节点自身解析
  paste: (typeId: string, text: string) => json<Record<string, unknown>>("POST", `/api/nodes/${encodeURIComponent(typeId)}/paste`, { text }),
  choices: (typeId: string, params: Record<string, unknown>, inputs: Record<string, string>) =>
    json<Record<string, Choice>>("POST", `/api/nodes/${encodeURIComponent(typeId)}/choices`, { params, inputs }),
  // `view`: which item each 逐项处理 block is showing (state/items.ts). It is a view setting, so the nodes
  // inside a block answer for that item; it never changes what a cook is.
  status: (graph: GraphJSON, cookInputs: number, display: string | null, view: Record<string, string> = {}) =>
    askGraph<StatusReply>("/api/status", graph, { cook_inputs: cookInputs, display, ...(Object.keys(view).length ? { view } : {}) }),
  /** Item by item for one node inside a 逐项处理 block: a page at a time, of the graph version the
   * status reply came with. The status reply itself is never one entry per item. */
  nodeItems: (graph: string, node: string, offset = 0, limit = 50) =>
    json<ItemsPage>("GET", `/api/status/${encodeURIComponent(graph)}/node/${encodeURIComponent(node)}/items?offset=${offset}&limit=${limit}`),
  // POST /api/jobs, the one way anything queues a graph (server/farm.py JobRequest): exactly one of cook (a node and
  // its upstream), shown (the viewer asks because the node is shown: only what it shows, only light, never a delivery)
  // or deliver ([]: every 「输出」 in the graph, like Nuke's Render All). `version`: the cook inputs' version of the graph
  // (state/cookInputs.ts), which every event of the job carries back; a node_done applies to the page only while it
  // is still that version (graph/streamDone.ts).
  // `show`: the output ports the viewer is showing (the server cooks only what they need); left out when there are none
  cook: (graph: GraphJSON, version: number, target: string, shown = false, show: string[] = []) =>
    askGraph<{ job: string }>("/api/jobs", graph, { [shown ? "shown" : "cook"]: target, version, ...(show.length ? { show } : {}), client: clientInfo() }),
  deliver: (graph: GraphJSON, version: number) => askGraph<{ job: string }>("/api/jobs", graph, { deliver: [], version, client: clientInfo() }),
  // the job's events as a stream that mends itself (platform/events.ts: nothing else opens an EventSource)
  cookEvents: (job: string) => followEvents(`/api/jobs/${job}/events`),
  cancelCook: (job: string) => json<{ ok: boolean }>("POST", `/api/jobs/${job}/cancel`, {}),
  /** 删除一条已结束的任务：该记录与其留在服务器上的交付包一并删除，计算结果的缓存保留
   * （缓存按指纹共享，可能仍被其他任务使用；需要释放空间时使用该行的「清理」）。正在计算的任务先取消。 */
  forgetJob: (job: string) => json<{ ok: boolean }>("DELETE", `/api/jobs/${encodeURIComponent(job)}`),
  // 一键释放空间：删除自己全部已结束的任务及其占用的空间
  forgetAllJobs: () => json<{ ok: boolean; jobs: number; skipped: number; bytes: number }>("DELETE", "/api/jobs"),
  // load: with the machine's load (the queue panel open); without it an idle queue's answer stays the same (a 304)
  queue: (load = true) => json<QueueView>("GET", `/api/queue${load ? "" : "?load=0"}`),
  /** How busy the server is, for the top bar's load pill: small enough to ask for on every poll. */
  load: () => json<ServerLoad>("GET", "/api/load"),
  /** One of the account's jobs again: its graph as submitted and whether its results are still cached. */
  job: (id: string) => json<JobLoad>("GET", `/api/jobs/${encodeURIComponent(id)}`),
  deliveries: {
    mine: () => json<Delivery[]>("GET", "/api/deliveries"),
    // Every path is the delivery's address (transfer/deliveries.py address): the node, or node.<hash> for one item of
    // a 逐项处理 block (`addressOf`), never the bare node id.
    get: (d: Pick<Delivery, "run" | "node" | "address">) => json<Delivery>("GET", `/api/deliveries/${d.run}/${addressOf(d)}`),
    state: (d: Pick<Delivery, "run" | "node" | "address">, state: "saved" | "dismissed") =>
      json<Delivery>("POST", `/api/deliveries/${d.run}/${addressOf(d)}/state`, { state }),
    /** Every package one 「输出」 delivered in that run, one per item of a block. */
    batch: (run: string, node: string) => `/api/deliveries/${run}/${node}/batch`,
  },
  uploads: {
    describe: (ref: string) => json<Upload>("GET", `/api/uploads/describe?ref=${encodeURIComponent(ref)}`),
    /** 申报一份上传：包含的文件、各文件内容的 sha256（由网页在本机计算），以及第一个文件开头的数十 KB。
     * 服务器据此计算该上传的 id（与字节传输完成后 `POST /api/uploads` 使用同一公式），用其自身的
     * describe_file 读取图层，并返回最终引用。此过程不传输任何数据字节。
     * 提供的文件头不足以读出图层时返回 `need`（需要追加的字节数），由网页补发，不会返回空图层。 */
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
    // 拖拽插队: put a waiting job at `position` (1-based) of its own lane; the server records who moved what, from
    // where to where (lab2shot/server/access.py audit) and answers with the queue as it now is
    place: (job: string, position: number) => json<QueueView>("POST", `/api/admin/jobs/${job}/place`, { position }),
    history: () => json<JobRecord[]>("GET", "/api/admin/history"),
    disk: () => json<DiskArea[]>("GET", "/api/admin/disk"),
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
      return json<UsageStats>("GET", `/api/admin/usage?${q}`);
    },
    resetUsage: () => json<{ start: number }>("POST", "/api/admin/usage/reset", {}),
    undoReset: () => json<{ start: number | null }>("POST", "/api/admin/usage/reset/undo", {}),
  },
  manifest: (fp: string) => json<Manifest>("GET", `/api/packet/${fp}`),
  /** 一帧供视图显示的代理图（`lab2shot/server/packets.py frame`）。
   *
   * 视图只使用代理，不区分无损与有损：尺寸为管理员设置的档位（包说明中的 `proxy.px`），
   * 由服务器在计算完成后生成。不含黑白点参数：黑白点属于显示方式，在浏览器的显卡上计算（view/look.ts）。 */
  packetFrameUrl: (fp: string, frame: number, through?: Through, v?: Version) =>
    `/api/packet/${fp}/frame/${frame}.png` + query([...(through ? [`through=${encodeURIComponent(through.fp)}`, `at=${encodeURIComponent(through.at)}`] : []), ...versionQuery(v)]),
  /** 一帧中单条通道的原始数据（`lab2shot/server/packets.py frame_channel`，格式见 transfer/plane.ts），
   * 而非生成好的显示图：通道选择、黑白点、着色与合成均在浏览器中计算。
   * `name` 为 R / G / B / A 或 valid，有效通道见 `/api/packet/{fp}` 的 `channels.names`。
   * 发送的是代理：按 `proxy.px` 档位缩放并压缩后的版本（没有其他档位可切换）。 */
  channelUrl: (fp: string, frame: number, name: string, v?: Version) => `/api/packet/${fp}/frame/${frame}/channel/${name}` + query(versionQuery(v)),
  boxes: (fp: string) => json<BoxesData>("GET", `/api/packet/${fp}/boxes`),
  tracks: (fp: string) => json<TracksData>("GET", `/api/packet/${fp}/tracks`),
  curves: (fp: string) => json<CurvesData>("GET", `/api/packet/${fp}/curves`),
  /** The node text an output-settings node wrote for another application to paste: which application reads
   * it, the file it was written as, and the text itself. Only a result whose meta declares one has it. */
  clipboard: (fp: string) => json<ClipboardText>("GET", `/api/packet/${fp}/clipboard`),
  /** 视频一帧的代理图。 */
  videoFrameUrl: (fp: string, frame: number, v?: Version) => `/api/view/${fp}/video/${frame}.png` + query(versionQuery(v)),
  /** 三维视图的描述（各数据块的地址由服务器提供，均带代次）；`g` 为该包的代次（transfer/gens.ts）。 */
  sceneUrl: (fp: string, g: string) => `/api/packet/${fp}/scene` + query(g ? [`g=${g}`] : []),
  pointsUrl: (fp: string, camera: string | null, g: string) => `/api/view/${fp}/points` + query([...(camera ? [`camera=${camera}`] : []), ...(g ? [`g=${g}`] : [])]),
  sendLog: (text: string) => json<{ ok: boolean }>("POST", "/api/logs", { text, client: clientInfo() }),
  /** Logging in with an account (lab2shot/server/auth.py); the administrator's forgotten password with the 口令. */
  auth: {
    state: () => json<AuthState>("GET", "/api/auth/state"),
    login: (username: string, password: string) => json<AuthState>("POST", "/api/auth/login", { username, password, device_id: deviceId() }),
    logout: () => json<AuthState>("POST", "/api/auth/logout", {}),
    change: (current: string, next: string) => json<AuthState>("POST", "/api/auth/password", { current, new: next }),
    recover: (passphrase: string, next: string) => json<AuthState>("POST", "/api/auth/recover", { passphrase, new: next }),
  },
};

/** The server itself (GET /api/server, lab2shot/server/settings.py): which run of it answers, whether a restart is coming. */
export interface RestartState {
  state: "draining" | "restarting"; // draining: waiting for the jobs running
  mode: "drain" | "now";
  since: number;
  running: number;
  tasks: MessageJson[]; // the background tasks still going (their titles): an install, a card's check
  port: number; // where the next server listens
  https: boolean;
}

export interface ServerInfo {
  boot: string;
  started: number;
  version: string;
  ui: number; // when the built page was made: another one after a restart means the page should reload
  restart: RestartState | null;
  notice: number; // when the administrator's notice last changed: another number means reading /api/notice again
  // 本机代理的两项参数（管理员设置「视图 · 本机代理尺寸 / 本机缓存上限」，lab2shot/config.py）：
  // 浏览器据此生成与淘汰本机代理（transfer/localProxy）。旧版服务器不提供该项时，页面使用默认值 1024 / 10
  view?: { local_px: number; local_cache_gb: number };
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

/** 后台「扩展包」区域中一个扩展包的条目（GET /api/admin/extensions，lab2shot/server/installs.py
 * extension_list）：安装状态、缺失项、当前登录可执行的操作（`actions`，由服务器计算，网页不检查角色），
 * 以及最近一次安装任务。`manual` 为仍需手动下载的项目。 */
export interface ExtensionRow {
  name: string;
  title: string;
  summary: string;
  nodes: number; // 该扩展包为编辑器添加的节点数量
  installed: boolean;
  ready: boolean;
  label: string; // 状态简述：已就绪 / 未安装 / 模型未齐 / 需要重装…
  reason: string; // 尚不可用的原因（已就绪时为 ""）
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

