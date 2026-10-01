/** 计算进度的唯一数据格式：队列面板（ui/Queue.tsx）与节点（editor/GraphNode.tsx）绘制同一份数据，
 * 服务器也只发送这一份（`lab2shot/progress.py`）。
 *
 * 该数据经两条路径到达，结构完全相同：
 *   - 事件流：`{type: "progress", ...}`（由 graph/follow.ts 接收，存入 state/results.ts 的 `now`）；
 *   - 队列回答：`QueueJob.now`（api/queue.ts）。
 *
 * 进度条只依据 `at`（整个任务的完成量，0–1，服务器保证单调不减，页面不做「只取最大值」
 * 之类的修补）；`done` / `total` 是解算器当前步骤的计数（如内部迭代次数），只作为文字显示（悬停提示），
 * 不得用于计算进度条宽度，否则进度条会来回伸缩。 */

export type Phase = "queued" | "loading" | "computing" | "fetching";

export interface JobProgress {
  phase: Phase;
  node: string; // 正在计算的节点 id（"" 表示尚未开始计算任何节点）
  label: string; // 该节点的名称
  note: string; // 解算器报告的当前步骤（如「检测人物」），仅为一句文字
  done: number; // 当前步骤的计数（不驱动进度条；view/partial.ts 以 done > 0 判断是否已写出第一帧）
  total: number;
  at: number | null; // 整个任务的完成量，0–1；null 表示存在无历史记录的节点、无法估计，此时绘制不确定进度条
}

/** 四个阶段的用词，全站唯一定义（对应 lab2shot/progress.py 定义的四个阶段；终端版本位于 lab2shot/cli/jobs.py PHASE_WORD）。
 * 与 graph/nodes.ts STATUS_TEXT 性质相同：简短的状态词，不属于提示或报错，因此不进入消息目录。 */
export const PHASE_TEXT: Record<Phase, string> = {
  queued: "排队中",
  loading: "加载模型",
  computing: "计算中",
  fetching: "取回结果",
};

/** 悬停提示中的完整说明：阶段名称、解算器当前步骤、该步骤的计数。
 * 该计数只在此处显示，节点右上角不显示。 */
export function progressTip(p: JobProgress): string {
  const step = p.total > 0 ? `${p.note || "这一步"} ${p.done} / ${p.total}` : p.note;
  const line = [p.label, PHASE_TEXT[p.phase], step].filter(Boolean).join(" · ");
  return p.at === null ? `${line}\n第一次算这个节点，没有历史记录，算不出走了百分之几` : `${line}\n整个计算走了 ${Math.round(p.at * 100)}%`;
}
