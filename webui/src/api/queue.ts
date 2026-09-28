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

/** When a job should finish (running) or start (waiting), from the records of earlier cooks; partial: no earlier. */
interface Eta {
  at: number;
  partial: boolean;
}

export interface QueueJob {
  id: string; // "" for someone else's (anonymous load)
  anonymous?: boolean; // someone else's job: only where it stands and when it should end
  title: string;
  graph?: string; // the graph's own id (meta.id): only for one's own jobs
  targets: string[]; // the nodes it cooks, by their labels
  nodes?: string[]; // and their ids
  frames: [number, number] | null; // the frame range it cooks (null: every frame of the inputs)
  eta: Eta | null;
  state: JobState;
  position: number | null; // place among the waiting tasks (the queue's order: 插队 first, then as they came in)
  cards?: string[]; // the cards its nodes are on now: only for whoever may see the cards (farm.cards)
  submitted: number;
  started: number | null;
  finished: number | null;
  // 计算进度（api/progress.ts）：与事件流中发送的是同一份数据，队列面板与节点绘制的内容一致
  // （lab2shot/farm/queue.py Job.progress_json）。`{}` 表示当前未在计算（排队中或已结束）
  now: JobProgress | Record<string, never>;
  error: string | null;
  reason: string; // cancelled: why, when not by the one who started it
  stopping: boolean;
  // 排队任务尚未开始的原因，只列出不会自行解除的情况（管理员关闭了计算 N-QUEUE-PAUSED、该服务器上
  // 没有能够计算它的机器 N-QUEUE-NOMACHINEEVER）；等待显卡、等待内存、前方排队等情况会自行解除，对使用者而言即「排队中」，
  // 归入下方 `waiting_detail`（见 lab2shot/farm/queue.py 对等待原因的分类）
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
 * none (已清理), and what computing the rest again should take. */
export interface CacheMark {
  mark: "all" | "some" | "none";
  cached: number;
  nodes: number;
  seconds: number;
  unknown: number; // nodes without a timing record
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
  // 编辑器空闲时只查询这一项：这三项几乎不变，若单独为其轮询队列会使空闲流量翻倍
  // （主要是请求头与会话 cookie），合并为一项才能节省
  switches: { gpu: boolean; compute: boolean };
  max_frames: number;
  storage?: StorageGate | null;
  server?: ServerInfo;  // 随「服务本身」一并返回：已登录的页面只保留这一项轮询
}

export interface QueueView {
  machine: MachineLoad;
  gpus?: QueueGpu[]; // the cards: only for whoever may see them (farm.cards)
  // what the scheduler goes by (lab2shot/farm/scheduler/pools.py): the cards and CPU nodes one task holds at once, the
  // CPU nodes the whole machine runs at once
  limits: { task_gpus: number; task_cpus: number; cpu_nodes: number };
  // 单次提交可计算的最大帧数（lab2shot/config.py queue.max_frames，由管理员在「设置」中配置）：网页据此在提交前
  // 拦截，服务器在 farm/queue.py submit 中独立再次校验（绕过网页直接提交同样会被拒绝）
  max_frames: number;
  // 显卡任务, 计算任务 (lab2shot/config.py queue.gpu_jobs, queue.compute_jobs): gpu off holds back the GPU nodes of
  // every task that has not started (it still queues, as when no GPU is authorized); compute off refuses every new
  // task outright (never queued; farm/queue.py submit).
  switches: { gpu?: boolean; compute: boolean }; // gpu: only for whoever may see the cards
  jobs: QueueJob[];
  history?: HistoryJob[]; // the account's own finished tasks, newest first
  // the version of that history (lab2shot/farm/queue.py listed_version): the poll carries this, api.queue fetches
  // /api/queue/history when it changes and fills `history` in
  history_version?: string;
  // 只包含判断是否已满的三个数（lab2shot/server/quota.py gate）：已满时置灰「提交」与右键菜单中的「计算」
  // （state/quota.ts），队列窗口关闭时也能判断。
  // 三项明细与流量不在此处：它们在计算过程中持续变化，而该回答每 1.5–30 秒轮询一次；
  // 若回答每次都不同，ETag 将始终无法命中，本应返回 304 的轮询都会变成完整重发。明细由队列窗口中的占用条
  // 自行查询一次 /api/my/storage，不属于轮询。后台的队列回答不包含此项：账号的占用量只与该账号相关。
  storage?: StorageGate;
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
}

