/** 节点右上角的状态格：节点当前阶段只在此处判定。
 *
 * 三路输入（用户机器上的文件处于上传的哪一步、本机代理生成到第几帧、服务器报告的计算进度）在此合成为一个词，
 * `editor/GraphNode.tsx` 的 `.gnode-state` 只读取此处的结果，自身不拼接任何文字。
 * 判定规则为下方的表 `RULES`：自上而下，取第一条成立的规则，优先级一目了然，不分散在 if 链中。
 *
 * 用户从选择序列到计算完成依次看到：
 *   读文件中 → 本机缓存（数字走到头）→ 已就位 → 〔点「计算」〕→ 上传中 → 排队中 → 计算中 · 加载模型 → 计算中 →
 *   取回结果 → 已缓存
 *
 * 格宽固定（`editor/styles/08a-node.css --node-state-w` 60 px，字号 10.5 px），最多容纳 5 个汉字，
 * 因此格内只写词，字节数、帧数、速度均放入悬停提示 `tip`。例如「本机缓存 37/200」放不下，格内显示「本机缓存」，
 * 数字放在提示中；「计算中 · 加载模型」格内显示「加载模型」，提示第一行写全。
 *
 * 缓存账本的活动（预读中 / 解码中）尚未接入：账本的 `doing` 接口确定后在表中增加两行，位于
 * 「本机缓存」之下、「服务器状态」之上。 */
import type { NodeStatus } from "./graph";
import type { UploadTask } from "./uploads";
import type { NodeDelivery } from "./results";
import { PHASE_TEXT, progressTip, type JobProgress } from "../api/progress";
import { localTier, type ProxyProgress } from "../transfer/localProxy";
import { eta, howFar } from "../transfer/uploadText";
import { agoText, rateText, sizeText } from "../platform/format";

/** 状态格可用的颜色，即 `.gnode-state.<tone>` 已有的类名（08a-node.css），此处不新增。 */
export type PhaseTone = "error" | "cooked" | "queued" | "cooking" | "pending" | "idle" | "skipped";

export interface NodePhase {
  word: string; // 格内文字（≤ 5 个汉字）
  tip: string; // 悬停提示：数字均放在此处
  tone: PhaseTone;
}

export interface PhaseInput {
  /** 提交被拦下的原因（已去掉开头的节点名，`platform/util.ts ownReason`）；undefined 表示未被拦下 */
  blocked?: string;
  /** 当前账号无法使用该节点的原因（`api/applies.ts nodeWhy`）；undefined 表示可用 */
  unusable?: string;
  /** 该节点某个文件参数的上传任务（`transfer/uploads.ts useNodeUpload`） */
  upload?: UploadTask;
  /** 本机代理生成进度，按该节点所有参数合计（`transfer/localProxy useLocalProxies`，键以 `<节点id>|` 开头） */
  proxy?: ProxyProgress;
  /** 「输出」的上一次交付，仅在未计算、未出错时提供（GraphNode 的 `deliveryShown`；页脚读取同一个值） */
  delivery?: NodeDelivery;
  /** 服务器报告的状态（`state/results.ts byNode`） */
  status: NodeStatus;
  /** 该节点的计算进度（`state/results.ts now`）；未在计算时为 null */
  progress?: JobProgress | null;
  /** 存在上一次的结果可用，但参数已修改（`state/stale.ts staleNode`）：视图显示该结果，时间线为土黄色 */
  stale?: boolean;
}

// ---------------------------------------------------------------- 各档的文字

type Shown = Exclude<NodeDelivery["state"], "dismissed">;

/** 「输出」不缓存（每次均重新交付），因此任务结束后不能按普通的 未计算 / 已缓存 显示：该格显示上一次交付
 * 进行到哪一步。已忽略（dismissed）的交付不在节点上显示。 */
const DELIVERY_WORD: Record<Shown, string> = { pending: "已交付", saved: "已保存", downloaded: "已下载", expired: "已过期" };
const DELIVERY_TONE: Record<Shown, PhaseTone> = { pending: "pending", saved: "cooked", downloaded: "cooked", expired: "idle" };
const DELIVERY_TIP = (d: NodeDelivery): string =>
  ({
    pending: `已交付 · ${agoText(d.at)}\n写好了，等提交它的电脑取回（这个浏览器提交的话，会自动存或弹出下载）`,
    saved: `已经存进这台电脑：${d.name}`,
    downloaded: `已经作为浏览器下载发出去：${d.name}`,
    expired: "服务器只保留几天，已经删了：再提交一次，有缓存的话很快",
  })[d.state as Shown];

/** 服务器状态对应的文字。`graph/nodes.ts STATUS_TEXT` 供页脚和信息面板使用（「排队」「计算中…」），
 * 该格使用「排队中」「计算中」。 */
const STATUS_WORD: Record<NodeStatus, string> = { idle: "未计算", queued: "排队中", cooked: "已缓存", cooking: "计算中", error: "出错", skipped: "已跳过" };

/** 计算期间格内显示当前所处阶段（api/progress.ts 的四个阶段），提示第一行写全「计算中 · 加载模型」。 */
const PHASE_FULL: Record<JobProgress["phase"], string> = { queued: "排队中", loading: "计算中 · 加载模型", computing: "计算中", fetching: "计算中 · 取回结果" };

const proxyBusy = (p?: ProxyProgress): boolean => !!p && p.total > 0 && p.done + p.failed < p.total;
const proxyLine = (p: ProxyProgress): string => `本机缓存 ${p.done}/${p.total} 帧${p.failed ? `，${p.failed} 帧没做出来` : ""}`;

// ---------------------------------------------------------------- 规则表

interface Rule {
  /** 该档成立的条件 */
  when: (i: PhaseInput) => boolean;
  say: (i: PhaseInput) => NodePhase;
}

const up = (...states: UploadTask["state"][]) => (i: PhaseInput) => !!i.upload && states.includes(i.upload.state);

/** 优先级：自上而下，取第一条成立的规则。
 *
 * | 档 | 成立条件 | 格内文字 | 色 | 提示中的补充说明 |
 * |---|---|---|---|---|
 * | 已拦下 | 提交被网页拦下（`blocked`） | 已拦下 | error | 拦下的原因 |
 * | 不可用 | 当前账号无法使用该节点 | 不可用 | error | 无法使用的原因 |
 * | 读文件中 | 上传任务 `reading`：本机计算内容指纹 | 读文件中 | cooking | 读到第几个文件，尚未上传任何字节 |
 * | 上传中 | 上传任务 `sending` / `elsewhere`（另一个标签页在上传） | 上传中 | cooking | 帧数或百分比、字节数、速度、剩余时间 |
 * | 断网重连 | 上传任务 `waiting` | 断网重连 | queued | 中断位置、已上传量、恢复后自动续传 |
 * | 服务器整理中 | 上传任务 `finishing` | 整理中 | cooking | 文件已全部上传，服务器正在组装输入 |
 * | 本机缓存 | 本机代理尚未完成（done + failed < total） | 本机缓存 | cooking | 已完成帧数、失败帧数、存储位置 |
 * | 交付 | 「输出」的上一次交付（仅在未计算、未出错时） | 已交付 / 已保存 / 已下载 / 已过期 | pending / cooked / cooked / idle | 交付进行到哪一步 |
 * | 计算中 | 服务器 `cooking` 且有该节点的进度 | 加载模型 / 计算中 / 取回结果 | cooking | 「计算中 · 加载模型」及解算器的步数与总体进度 |
 * | 已就位 | 服务器 `idle` 且上传任务 `picked`：已申报，字节仍在用户机器上 | 已就位 | idle | 大小、点「计算」时才上传；本机缓存已完成的帧数 |
 * | 服务器状态 | 其余情况 | 未计算 / 排队中 / 计算中 / 已缓存 / 出错 / 已跳过 | 同名 | 有进度时显示进度 |
 *
 * 上传任务的 `paused`（页面刷新后浏览器不再允许读取这些文件）和 `failed`（服务器未接收）不决定该格：
 * 这两种情况显示在左下角的灰色文字中（`ui/UploadState.tsx` / `editor/NodeFoot.tsx uploadNote`），该格仍显示服务器状态。
 *
 * 「服务器整理中」超出格宽（最多 5 个汉字），格内显示「整理中」，提示第一行写全。 */
const RULES: readonly Rule[] = [
  { when: (i) => !!i.blocked, say: (i) => ({ word: "已拦下", tone: "error", tip: i.blocked! }) },
  { when: (i) => !!i.unusable, say: (i) => ({ word: "不可用", tone: "error", tip: i.unusable! }) },
  {
    when: up("reading"),
    say: ({ upload: t }) => ({
      word: "读文件中", tone: "cooking",
      tip: `读文件中 ${t!.done}/${t!.files.length} 个文件\n正在读你机器上的文件、算内容指纹。还没有任何字节上传：算完节点上就有输出口，字节要到点「计算」时才传`,
    }),
  },
  {
    when: up("sending", "elsewhere"),
    say: ({ upload: t }) => ({
      word: "上传中", tone: "cooking",
      tip: t!.state === "elsewhere"
        ? `上传中 ${howFar(t!)}\n这份文件正在这个浏览器的另一个标签页里上传：传完以后这里也会用上`
        : [`上传中 ${howFar(t!)} · 已传 ${sizeText(t!.sent)} / ${sizeText(t!.bytes)}`, t!.rate ? rateText(t!.rate) : "", eta(t!)].filter(Boolean).join(" · ")
          + "\n服务器上已经有的文件不再传；网络断了会自己接着传，不用管",
    }),
  },
  {
    when: up("waiting"),
    say: ({ upload: t }) => ({
      word: "断网重连", tone: "queued",
      tip: `断网重连 · 已传 ${howFar(t!)}（${sizeText(t!.sent)} / ${sizeText(t!.bytes)}）\n连不上服务器（${t!.error}）。已经传上去的部分都在服务器上，网络一恢复就从断开的地方接着传，不用重新选`,
    }),
  },
  {
    when: up("finishing"),
    say: ({ upload: t }) => ({
      word: "整理中", tone: "cooking",
      tip: `服务器整理中\n所有文件都传上去了（${sizeText(t!.bytes)}），服务器正在把它们整理成一份输入`,
    }),
  },
  {
    when: (i) => proxyBusy(i.proxy),
    say: ({ proxy: p }) => ({
      word: "本机缓存", tone: "cooking",
      tip: `${proxyLine(p!)}\n正在后台把每一帧缩到 ${localTier()} 档、存进浏览器的硬盘缓存，视图从它画；一个字节都不上传。期间什么都能操作`,
    }),
  },
  {
    when: (i) => !!i.delivery,
    say: ({ delivery: d }) => ({ word: DELIVERY_WORD[d!.state as Shown], tone: DELIVERY_TONE[d!.state as Shown], tip: DELIVERY_TIP(d!) }),
  },
  {
    when: (i) => i.status === "cooking" && !!i.progress,
    say: ({ progress: p }) => ({ word: PHASE_TEXT[p!.phase], tone: "cooking", tip: `${PHASE_FULL[p!.phase]}\n${progressTip(p!)}` }),
  },
  {
    when: (i) => i.status === "idle" && !!i.stale,
    say: () => ({ word: "已过期", tone: "queued", tip: "已过期：参数改过了，视图里是上一次算的结果（时间线上土黄的那段能实时播）。点「计算」更新" }),
  },
  {
    when: (i) => i.status === "idle" && up("picked")(i),
    say: ({ upload: t, proxy: p }) => ({
      word: "已就位", tone: "idle",
      tip: [
        `已就位 · ${t!.name} 在你的机器上，一个字节都没上传（${sizeText(t!.bytes)}）`,
        p && p.total > 0 ? `${proxyLine(p)}，已经生成好` : "",
        "点「计算」时先把接了线的通道传上去，传完自动接着算——不用在这儿等",
      ].filter(Boolean).join("\n"),
    }),
  },
  {
    when: () => true,
    say: ({ status, progress: p }) => ({ word: STATUS_WORD[status], tone: status, tip: p ? progressTip(p) : STATUS_WORD[status] }),
  },
];

/** 节点当前所处的阶段：规则表中第一条成立的档。 */
export function nodePhase(i: PhaseInput): NodePhase {
  return RULES.find((r) => r.when(i))!.say(i);
}

/** 按节点合计本机代理进度（`useLocalProxies` 的键为 `<节点id>|<参数名>`，一个节点可能有多个文件参数）；
 * 没有任何条目时返回 undefined。 */
export function proxyOfNode(tasks: Record<string, ProxyProgress>, node: string): ProxyProgress | undefined {
  const head = `${node}|`;
  let sum: ProxyProgress | undefined;
  for (const [key, p] of Object.entries(tasks)) {
    if (!key.startsWith(head)) continue;
    sum = sum ? { done: sum.done + p.done, total: sum.total + p.total, failed: sum.failed + p.failed } : { ...p };
  }
  return sum;
}
