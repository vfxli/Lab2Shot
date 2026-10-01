/** 节点右上角的状态格：节点当前阶段只在此处判定。
 *
 * 三路输入（用户机器上的文件处于上传的哪一步、本机代理生成到第几帧、服务器报告的计算进度）在此合成为一个词，
 * `editor/GraphNode.tsx` 的 `.gnode-state` 只读取此处的结果，自身不拼接任何文字。
 * 判定规则为下方的表 `RULES`：自上而下，取第一条成立的规则，优先级一目了然，不分散在 if 链中。
 *
 * 用户从选择序列到计算完成依次看到：
 *   读文件中 → 本机缓存 → 已就位 → 〔点「计算」〕→ 上传中 → 排队中 → 加载模型 → 计算中 → 取回结果 → 已缓存
 *
 * 格宽固定（`editor/styles/08a-node.css --node-state-w` 60 px，字号 10.5 px），最多容纳 5 个汉字，格内只写词。
 * 画布上的节点不显示悬停提示（platform/tips.ts `[data-no-tips]`），所以这里只有词，没有补充说明：
 * 上传的进度在底行左侧的灰字里（transfer/uploadText.ts uploadNote），计算进度是节点底边的进度条，数字在队列面板里。
 *
 * 表中没有缓存账本的活动（预读中 / 解码中）：这两档若加入，位于「本机缓存」之下、「服务器状态」之上。 */
import type { NodeStatus } from "./graph";
import type { UploadTask } from "./uploads";
import type { Output } from "../api/files";
import { PHASE_TEXT, type JobProgress } from "../api/progress";
import type { ProxyProgress } from "../transfer/localProxy";

/** 状态格的颜色，作为类名 `.gnode-state.<tone>` 加上（08a-node.css：cooked / queued / cooking / error 有各自的颜色，
 * idle / skipped 用默认颜色）。 */
type PhaseTone = "error" | "cooked" | "queued" | "cooking" | "idle" | "skipped";

interface NodePhase {
  word: string; // 格内文字（≤ 5 个汉字）
  tone: PhaseTone;
}

interface PhaseInput {
  /** 提交被网页拦下 */
  blocked: boolean;
  /** 当前账号无法使用该节点（`api/applies.ts nodeUsable`） */
  unusable: boolean;
  /** 该节点某个文件参数的上传任务（`transfer/uploads.ts useNodeUpload`） */
  upload?: UploadTask;
  /** 本机代理生成进度，按该节点所有参数合计（`transfer/localProxy useLocalProxies`，键以 `<节点id>|` 开头） */
  proxy?: ProxyProgress;
  /** 「输出」上一次整理打包好的结果，仅在未计算、未出错时提供（GraphNode 的 `outputShown`；页脚的「下载」读取同一个值） */
  output?: Output;
  /** 服务器报告的状态（`state/results.ts byNode`） */
  status: NodeStatus;
  /** 该节点的计算进度（`state/results.ts running`）；未在计算时为 null */
  progress?: JobProgress | null;
  /** 存在上一次的结果可用，但参数已修改（`editor/GraphNode.tsx useStaleNode`，规则 `state/stale.ts staleNode`）：视图显示该结果，时间线为土黄色 */
  stale?: boolean;
}

// ---------------------------------------------------------------- 各档的文字

/** 服务器状态对应的文字。`graph/nodes.ts STATUS_TEXT` 供页脚和信息面板使用（「排队」「计算中…」），
 * 该格使用「排队中」「计算中」。 */
const STATUS_WORD: Record<NodeStatus, string> = { idle: "未计算", queued: "排队中", cooked: "已缓存", cooking: "计算中", error: "出错", skipped: "已跳过" };

const proxyBusy = (p?: ProxyProgress): boolean => !!p && p.total > 0 && p.done + p.failed < p.total;

// ---------------------------------------------------------------- 规则表

interface Rule {
  /** 该档成立的条件 */
  when: (i: PhaseInput) => boolean;
  say: (i: PhaseInput) => NodePhase;
}

const up = (...states: UploadTask["state"][]) => (i: PhaseInput) => !!i.upload && states.includes(i.upload.state);

/** 优先级：自上而下，取第一条成立的规则。
 *
 * | 档 | 成立条件 | 格内文字 | 色 |
 * |---|---|---|---|
 * | 已拦下 | 提交被网页拦下（`blocked`） | 已拦下 | error |
 * | 不可用 | 当前账号无法使用该节点 | 不可用 | error |
 * | 读文件中 | 上传任务 `reading`：本机计算内容指纹，尚未上传任何字节 | 读文件中 | cooking |
 * | 上传中 | 上传任务 `sending` / `elsewhere`（另一个标签页在上传） | 上传中 | cooking |
 * | 断网重连 | 上传任务 `waiting`：恢复后自动续传 | 断网重连 | queued |
 * | 服务器整理中 | 上传任务 `finishing`：文件已全部上传，服务器正在组装输入 | 整理中 | cooking |
 * | 本机缓存 | 本机代理尚未完成（done + failed < total） | 本机缓存 | cooking |
 * | 已打包 | 「输出」上一次整理打包好的结果（仅在未计算、未出错时） | 已打包 / 已删除 | cooked / idle |
 * | 计算中 | 服务器 `cooking` 且有该节点的进度 | 加载模型 / 计算中 / 取回结果 | cooking |
 * | 已过期 | 服务器 `idle` 且有参数修改前的结果（`stale`）：视图显示的是上一次的结果 | 已过期 | queued |
 * | 已就位 | 服务器 `idle` 且上传任务 `picked`：已申报，字节仍在用户机器上，点「计算」时才上传 | 已就位 | idle |
 * | 服务器状态 | 其余情况 | 未计算 / 排队中 / 计算中 / 已缓存 / 出错 / 已跳过 | 同名 |
 *
 * 上传任务的 `paused`（页面刷新后浏览器不再允许读取这些文件）和 `failed`（服务器未接收）不决定该格：
 * 这两种情况显示在左下角的灰色文字中（`ui/UploadState.tsx` / `editor/NodeFoot.tsx uploadNote`），该格仍显示服务器状态。 */
const RULES: readonly Rule[] = [
  { when: (i) => i.blocked, say: () => ({ word: "已拦下", tone: "error" }) },
  { when: (i) => i.unusable, say: () => ({ word: "不可用", tone: "error" }) },
  { when: up("reading"), say: () => ({ word: "读文件中", tone: "cooking" }) },
  { when: up("sending", "elsewhere"), say: () => ({ word: "上传中", tone: "cooking" }) },
  { when: up("waiting"), say: () => ({ word: "断网重连", tone: "queued" }) },
  { when: up("finishing"), say: () => ({ word: "整理中", tone: "cooking" }) },
  { when: (i) => proxyBusy(i.proxy), say: () => ({ word: "本机缓存", tone: "cooking" }) },
  { when: (i) => !!i.output, say: ({ output: o }) => ({ word: o!.gone ? "已删除" : "已打包", tone: o!.gone ? "idle" : "cooked" }) },
  { when: (i) => i.status === "cooking" && !!i.progress, say: ({ progress: p }) => ({ word: PHASE_TEXT[p!.phase], tone: "cooking" }) },
  { when: (i) => i.status === "idle" && !!i.stale, say: () => ({ word: "已过期", tone: "queued" }) },
  { when: (i) => i.status === "idle" && up("picked")(i), say: () => ({ word: "已就位", tone: "idle" }) },
  { when: () => true, say: ({ status }) => ({ word: STATUS_WORD[status], tone: status }) },
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
