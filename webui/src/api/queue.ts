// The farm as the page sees it (lab2shot/farm): the queue, jobs, their clients and cache marks, the machine's load, disk
// areas, models kept loaded, the job log. Re-exported by api/index.ts.

import type { JobProgress } from "./progress";
import type { MessageJson } from "./applies";
import type { Output } from "./files";
import type { GraphJSON } from "./catalog";
import type { StorageGate } from "./library";
import type { ServerInfo } from "./index";

/** The farm's queue (lab2shot/farm): the same view for the editor's 队列 and the admin page. */
interface QueueGpu {
  index: number;
  name: string;
  short_name: string;
  uuid: string;
  memory_mb: number;
  used_mb: number;
  utilization: number;
  temperature: number;
  authorized: boolean; // takes jobs
  busy: boolean; // a node of a job runs on it (anyone's)
  job: string | null; // the job whose node runs on it, when it is the viewer's (the administrator: any)
  compute_cap: string; // nvidia-smi's compute capability ("8.9", "12.0"); "" when it could not be read
  // the administrator only (null for everyone else): every installed GPU extension on this card, as assumed (its
  // declared architectures and the environment scan say it runs here), refused (they say it cannot) or unknown
  // (the environment could not be probed), each with its reason (lab2shot/farm/scheduler/compat.py card_extensions)
  extensions: Record<GpuFitState, GpuFitItem[]> | null;
}

export type GpuFitState = "assumed" | "refused" | "unknown";

export interface GpuFitItem {
  extension: string;
  title: string;
  message: { code: string; level: string; text: string; params: Record<string, unknown> };
}

export interface JobClient {
  who: string; // the account: 张三（zhangsan）
  app: string; // web / maya / houdini / nuke / cli ...
  // the administrator's view adds the account and what the request showed (lab2shot/farm/clients.py)
  user?: number;
  username?: string;
  name?: string;
  department?: string;
  details?: Record<string, unknown>; // ip, user_agent, and what the client said of itself (hostname, OS user, platform ...)
}

// partial: 部分失败, a node failed (its error said at it) and what did not need it was cooked
export type JobState = "queued" | "running" | "done" | "partial" | "failed" | "cancelled" | "interrupted";

export interface QueueJob {
  id: string; // "" for someone else's (anonymous load)
  anonymous?: boolean; // someone else's job: only where it stands and how far it is
  title: string;
  graph?: string; // the graph's own id (meta.id): only for one's own jobs
  targets: string[]; // the nodes it cooks, by their labels
  nodes?: string[]; // and their ids
  frames: [number, number] | null; // the frame range it cooks (null: every frame of the inputs)
  state: JobState;
  position: number | null; // place among the waiting tasks (the queue's order: 插队 first, then as they came in)
  cards?: string[]; // the cards its nodes are on now: only for whoever may see the cards (farm.cards)
  submitted: number;
  started: number | null;
  finished: number | null;
  // cook progress (api/progress.ts): the same data the event stream sends, so the queue panel and the node draw the
  // same thing (lab2shot/farm/queue.py Job.progress_json). `{}`: not computing now (waiting or finished)
  now: JobProgress | Record<string, never>;
  error: string | null;
  reason: string; // cancelled: why, when not by the one who started it
  stopping: boolean;
  // why a waiting task has not started, only for what will not clear by itself (the administrator switched computing
  // off, N-QUEUE-PAUSED; no machine on this server can ever compute it, N-QUEUE-NOMACHINEEVER). Waiting for a card, for
  // memory or behind other tasks clears by itself, is simply 「排队中」 to the user, and goes to `waiting_detail`
  // below (see lab2shot/farm/queue.py for how waiting reasons are classified). A running task has it too while one of
  // its nodes waits for what needs somebody to act (no card authorized: N-QUEUE-NOCARD)
  waiting?: MessageJson | null;
  waiting_detail?: MessageJson | null; // the reason about the cards (N-QUEUE-GPUOFF): only for whoever may see the cards (farm.cards)
  mine: boolean;
  client?: JobClient; // not for someone else's
  outputs?: Output[]; // what its 「输出」 packed (the download is there once its zip is written)
  group?: TaskGroup; // the account's group it is in (a task; never on someone else's anonymous row)
}

/** Which of its account's groups a task is in (lab2shot/transfer/groups.py): the same footage (content of what its
 * input nodes read), or without footage the same template in one fixed two-hour slot of the server's clock. `name`
 * is the user's own data (a file's or a graph's name): shown as plain text only. `slot`: the slot's start (seconds)
 * for a group without footage, null with footage. `count`: how many tasks the account has in it now. `first`: when its
 * first task was submitted (seconds). `twin`: another of the account's groups with footage has the same name and this
 * one was not renamed, so the page adds `first` to tell them apart (「sh030_plate · 9月28日 14:05」, only as shown).
 * `renamed`: its user or an administrator gave it `name` (shown exactly so). */
export interface TaskGroup {
  key: string;
  name: string;
  slot: number | null;
  count: number;
  first: number;
  twin: boolean;
  renamed: boolean;
}

/** Whether a finished job's results are still cached (server: farm/queue.py cache_mark): all (全在), some (部分),
 * none (已清理). */
export interface CacheMark {
  mark: "all" | "some" | "none";
  cached: number;
  nodes: number;
  why: string; // it can't be planned any more (an upload cleaned)
}

/** One of the account's own finished jobs, from the job log, with its cache mark. */
export interface HistoryJob extends JobRecord {
  cache: CacheMark | null;
}

/** A job loaded again (GET /api/jobs/{id}): its graph exactly as submitted. */
export interface JobLoad {
  id: string;
  title: string;
  submitted: number;
  graph: GraphJSON;
  cache: CacheMark;
  nodes: string[]; // what it computed (node ids): the focus mode's 「计算」 computes the same again
  follows: string; // the job it followed ("" none)
}

/** How busy the server machine is (lab2shot/farm/load.py): numbers only. */
interface MachineLoad {
  cpu_percent: number | null; // all cores, since the last look (null the first time)
  cores: number;
  memory_gb: { used: number; total: number };
  disk_gb: { free: number; total: number }; // the work folder's disk
}

/** The one small answer every page may ask for often (GET /api/load): how long the queue
 * is, how many compute slots are busy, and the machine's CPU and memory. `cards` (busy or not, memory in use) is in
 * everyone's answer. */
export interface ServerLoad {
  queue: { waiting: number; running: number };
  slots: { cpu: { busy: number; total: number }; gpu: { busy: number; total: number } }; // 计算位: nodes at once, by kind
  cpu_pct: number | null; // null until the machine has been read twice
  ram_pct: number | null;
  cards?: { busy: boolean; mem_pct: number | null }[];
  // the only thing an idle editor asks for: these three barely change, and polling the queue separately for them
  // would double the idle traffic (mostly request headers and the session cookie); carrying them in this one answer saves that
  switches: { gpu: boolean; compute: boolean };
  // 数据盘低于「暂停新计算的剩余空间」（lab2shot/farm/policy.py space）：所有账号的新计算暂停，恢复后自动接着
  disk_low?: boolean;
  max_frames: number;
  storage?: StorageGate | null;
  server?: ServerInfo;  // comes with 「服务本身」: a logged-in page keeps this as its only poll
}

export interface QueueView {
  machine: MachineLoad;
  gpus?: QueueGpu[]; // the cards: only for whoever may see them (farm.cards)
  // what the scheduler goes by (lab2shot/farm/scheduler/pools.py): the cards and CPU nodes one task holds at once, the
  // CPU nodes the whole machine runs at once
  limits: { task_gpus: number; task_cpus: number; cpu_nodes: number };
  // the most frames one submission may cook (lab2shot/config.py queue.max_frames, set by the administrator in 设置): the
  // page stops a submission over it before sending, and the server checks again on its own in farm/queue.py submit (a
  // submission that bypasses the page is refused all the same)
  max_frames: number;
  // 显卡任务, 计算任务 (lab2shot/config.py queue.gpu_jobs, queue.compute_jobs): gpu off holds back the GPU nodes of
  // every task that has not started (it still queues, as when no GPU is authorized); compute off refuses every new
  // task outright (never queued; farm/queue.py submit).
  switches: { gpu?: boolean; compute: boolean }; // gpu: only for whoever may see the cards
  disk_low?: boolean; // as ServerLoad's
  jobs: QueueJob[];
  history?: HistoryJob[]; // the account's own finished tasks, newest first
  // the version of that history (lab2shot/farm/queue.py listed_version): the poll carries this, api.queue fetches
  // /api/queue/history when it changes and fills `history` in
  history_version?: string;
  // only the three numbers that tell whether the account is full (lab2shot/server/quota.py gate): when full, every
  // 「计算」 entry is greyed (graph/actions.ts cookHold, state/quota.ts), which works with the queue window closed too.
  // The three-part breakdown and the traffic are not here: they change throughout a cook while this answer is polled
  // every 1.5–30 s, and an answer that differs every time never hits its ETag, turning every poll that should be a
  // 304 into a full resend. The queue window's usage bar asks /api/my/storage once for the breakdown, outside the poll.
  // The admin queue answer leaves this out: an account's usage concerns that account only.
  storage?: StorageGate;
}

/** The server's disk as last measured (lab2shot/farm/queue.py Farm.disk): measuring walks the whole data disk, so it
 * runs in the background and the answer says when it was measured and whether a new measurement is running. */
export interface DiskUsage {
  areas: DiskArea[] | null; // null: never measured yet
  at: number | null; // when measured (seconds)
  measuring: boolean;
  space?: DiskSpace;
}

/** The data disk against 暂停新计算的剩余空间 (lab2shot/farm/policy.py space): below `floor` every account's new
 * computing pauses (submissions and uploads refused, waiting tasks held), and resumes by itself above it. */
export interface DiskSpace {
  path: string;
  total: number;
  free: number;
  pct: number; // free, % of total
  floor_pct: number; // the setting
  floor: number; // bytes
  low: boolean;
}

/** The server's disk, per area (task folders, cache, uploads). */
export interface DiskArea {
  id: string;
  label: string;
  note: string;
  bytes: number;
  items: number;
  idle_7_bytes: number; // unused for 7 days
  idle_30_bytes: number;
}

/** A worker process kept between jobs with its models loaded (lab2shot/engine/resident.py). */
interface ResidentModel {
  name: string;
  gpu_mb: number; // its tensors on the GPU
  ram_mb: number; // its tensors in RAM
  on_gpu: boolean;
}

export interface ResidentProcess {
  id: string;
  extension: string;
  title: string;
  gpu: string | null; // UUID; "" no GPU
  gpu_name: string;
  pid: number;
  state: "busy" | "moving" | "gpu" | "ram";
  models: ResidentModel[];
  vram_mb: number; // what PyTorch holds on the GPU (the CUDA context comes on top)
  ram_mb: number; // the process's resident memory
  idle_s: number;
  jobs: number;
  started: number;
}

export interface ResidentView {
  available_gb: number;
  processes: ResidentProcess[];
}

export interface JobRecord {
  event: "submitted" | "finished";
  id: string;
  title: string;
  targets: string[];
  frames: [number, number] | null;
  state: JobState;
  submitted: number;
  started: number | null;
  finished: number | null;
  cards?: string[]; // the models of the cards its nodes ran on: only for whoever may see the cards (farm.cards)
  error: string | null;
  reason: string;
  outputs: Output[];
  client: JobClient;
  group?: TaskGroup;
  follows?: string; // the account's own job this one followed (POST /api/jobs follows): "" or absent none
}

