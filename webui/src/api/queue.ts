// The farm as the page sees it (lab2shot/farm): the queue, jobs, their clients and cache marks, the machine's load, disk
// areas, models kept loaded, the job log. Re-exported by api/index.ts.

import type { JobProgress } from "./progress";
import type { MessageJson } from "./applies";
import type { Delivery } from "./deliveries";
import type { GraphJSON } from "./catalog";
import type { StorageGate } from "./library";
import type { ServerInfo } from "./index";

/** The farm's queue (lab2shot/farm): the same view for the editor's 队列 and the admin page. */
export interface QueueGpu {
  index: number;
  name: string;
  short_name: string;
  uuid: string;
  memory_mb: number;
  used_mb: number;
  utilization: number;
  temperature: number;
  authorized: boolean; // takes jobs
  busy: boolean; // a job runs on it (anyone's)
  job: string | null; // the job running on it, when it is the viewer's (the administrator: any)
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

export type JobState = "queued" | "running" | "done" | "failed" | "cancelled" | "interrupted";

/** When a job should finish (running) or start (waiting), from the records of earlier cooks; partial: no earlier. */
export interface Eta {
  at: number;
  partial: boolean;
}

/** Where a job runs (lab2shot/engine/policy.py). */
export type Lane = "light" | "heavy" | "gpu";

export interface QueueJob {
  id: string; // "" for someone else's (anonymous load)
  anonymous?: boolean; // someone else's job: only where it stands and when it should end
  shown?: boolean; // started because a node was shown (not a click)
  title: string;
  targets: string[]; // the nodes it cooks, by their labels
  nodes?: string[]; // and their ids
  frames: [number, number] | null; // the frame range it cooks (null: every frame of the inputs)
  eta: Eta | null;
  state: JobState;
  position: number | null; // place among the waiting jobs of its lane
  lane: Lane; // light: at once; heavy: the CPU lane; gpu: waits for a GPU (lab2shot/engine/policy.py)
  gpu_name: string; // the GPU it runs or ran on ("" none)
  submitted: number;
  started: number | null;
  finished: number | null;
  // 计算进度（api/progress.ts）：与事件流中发送的是同一份数据，队列面板与节点绘制的内容一致
  // （lab2shot/farm/queue.py Job.progress_json）。`{}` 表示当前未在计算（排队中或已结束）
  now: JobProgress | Record<string, never>;
  // a job of a 逐项处理 block is cooked in 计算单元 (farm/units.py): how many items are through, and how many cards it
  // is on at once ({} / 0 for a job without a block)
  units?: { done: number; items: number; running: number };
  cards?: number;
  error: string | null;
  reason: string; // cancelled: why, when not by the one who started it
  stopping: boolean;
  // 排队任务尚未开始的原因，只列出不会自行解除的情况（管理员关闭了计算 N-QUEUE-PAUSED、该服务器上
  // 没有能够计算它的机器 N-QUEUE-NOMACHINEEVER）；等待显卡、等待内存、前方排队等情况会自行解除，对使用者而言即「排队中」，
  // 归入下方 `waiting_detail`（farm/queue.py SELF_CLEARING）
  waiting?: MessageJson | null;
  waiting_detail?: MessageJson | null; // the reason about the cards (N-QUEUE-GPUOFF): only for whoever may see the cards (farm.cards)
  mine: boolean;
  client?: JobClient; // not for someone else's
  outputs?: Delivery[];
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
export interface MachineLoad {
  cpu_percent: number | null; // all cores, since the last look (null the first time)
  cores: number;
  memory_gb: { used: number; total: number };
  disk_gb: { free: number; total: number }; // the work folder's disk
}

/** The one small answer every page may ask for often (GET /api/load): how long the queue
 * is, how many compute slots are busy, and the machine's CPU and memory. `cards` is only in the answer of an account
 * that may see the cards (the route's `hides` takes the field out of everyone else's), so a page never judges a role:
 * the key is either present or absent. */
export interface ServerLoad {
  queue: { waiting: number; running: number };
  slots: { busy: number; total: number };
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
  limits: { light: number; heavy: number }; // jobs the lanes without a GPU run at once
  // 单次提交可计算的最大帧数（lab2shot/config.py queue.max_frames，由管理员在「设置」中配置）：网页据此在提交前
  // 拦截，服务器在 farm/queue.py submit 中独立再次校验（绕过网页直接提交同样会被拒绝）
  max_frames: number;
  // 显卡任务, 计算任务 (lab2shot/config.py queue.gpu_jobs, queue.compute_jobs): gpu off pauses only the GPU lane (a
  // GPU job still queues, as when no GPU is authorized); compute off refuses anything but pure viewing outright (never
  // queued): a GPU node, heavy CPU work, or any delivery, even of a cached result.
  switches: { gpu?: boolean; compute: boolean }; // gpu: only for whoever may see the cards
  jobs: QueueJob[];
  history?: HistoryJob[]; // the account's own finished jobs (a click started them), newest first
  // 只包含判断是否已满的三个数（lab2shot/server/quota.py gate）：已满时置灰「提交」与右键菜单中的「计算」
  // （state/quota.ts），队列窗口关闭时也能判断。
  // 四项明细与流量不在此处：它们在计算过程中持续变化，而该回答每 1.5–30 秒轮询一次；
  // 若回答每次都不同，ETag 将始终无法命中，本应返回 304 的轮询都会变成完整重发。明细由队列窗口中的占用条
  // 自行查询一次 /api/my/storage，不属于轮询。后台的队列回答不包含此项：账号的占用量只与该账号相关。
  storage?: StorageGate;
}

/** The server's disk, per area (cache, uploads, deliveries). */
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
export interface ResidentModel {
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
  lane: Lane;
  submitted: number;
  started: number | null;
  finished: number | null;
  gpu: string | null;
  gpu_name: string;
  error: string | null;
  reason: string;
  outputs: Delivery[];
  client: JobClient;
}

/** How much a project or node type was used in a range (GET /api/admin/usage, lab2shot/farm/usage.py). */
