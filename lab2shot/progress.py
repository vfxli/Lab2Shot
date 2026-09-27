"""计算进度的唯一实现：队列面板和节点上显示的是同一份描述，服务器也只发送这一份。

进度条必须有属于任务的定义，不能是最后一条 worker 消息的副产品。worker 每进入一个阶段就重新开始一组
`done / total`（`worker_sdk.progress`），一个节点常有多组（SAM 3D Body 一个节点有四组：按帧检测人物、
按帧估计人体姿态、按张估计焦距、按人数锁定体型 · 平滑 · 生成骨骼）。若页面按最近一组计算宽度，进度条会走满、
归零、再走满，`12/195` 这类数字也会随分母变化而跳动：分母在计算过程中改变时，进度条必然回缩。

做法：

1. 分母在开始时固定（`farm/queue.py Job.fix_budget`，在 `_start` 中确定显卡之后计算一次）：该任务要计算的
   每个节点实例（`farm/timings.py planned()` 列出的 `Work`）按历史记录预测的秒数之和即为分母。这与
   队列上「剩余约 X」使用同一份预测（`Job.left` / `Job.remaining`），进度条与剩余时间不会互相矛盾。
2. 分子只增不减：已完成的节点实例累加其自身份额（`Job.spent`），正在计算的实例再加上已运行的秒数，
   并以其自身份额为上限。该上限保证实例完成时分子不回退，因此 `at` 单调不减由构造保证，而不依赖前端
   「只取最大值」来弥补。
3. 无法估计时不显示刻度：任一节点没有历史记录（首次计算）时 `at` 为 `None`，页面显示往复移动的不定
   进度条，不使用编造的分数。
4. `done` / `total` 仅用作文字（悬停提示中显示「检测人物 12 / 195」），不驱动进度条：它们是解算器内部的计数，
   使用者无需关注。此外 `webui/src/view/partial.ts` 以 `done > 0` 判断该节点是否已写出第一帧（边算边看）。

四个阶段的判据（机械判据，不依赖字符串推测）：

| 阶段 | 判据 |
|---|---|
| `queued` 排队中 | 任务仍在队列中（`Job.state == "queued"`） |
| `loading` 加载模型 | `engine/external.py run_job` 启动 worker 进程之前发出的 `phase`：此阶段启动进程、加载权重、读取素材 |
| `computing` 计算 | 第一条 `progress`（开始有可计数的内容）。当前步骤由 worker 自身的 `stage` 名称说明，显示在旁边的文字中 |
| `fetching` 取回结果 | worker 写完 `result.json`、`run_job` 返回之后发出的 `phase`：核心将 raw 转换为数据包并写盘 |

核心自身的节点（不运行 worker）没有 `loading` / `fetching` 两个阶段，`node_start` 之后直接进入 `computing`：
它们没有加载模型的步骤，显示该阶段即为虚构。
"""

from __future__ import annotations

# 只有这四个阶段。本文件位于 `lab2shot/` 顶层，是因为两处需要使用同一套名称：`engine/external.py` 发出这些名称
# （它是唯一知道 worker 进程何时启动、何时写完结果的地方），`farm/queue.py` 将其整合为对外的描述。
# 界面上的四个词只在 webui/src/api/progress.ts PHASE_TEXT 一处定义（与 graph/nodes.ts STATUS_TEXT 属于同一类：
# 简短的状态词，不是提示或报错，因此不进入消息目录）。
QUEUED = "queued"
LOADING = "loading"
COMPUTING = "computing"
FETCHING = "fetching"
PHASES = (QUEUED, LOADING, COMPUTING, FETCHING)

# 计算进行中时进度条最多显示到此处：估计时间用完而计算仍在进行时，停在此处等待实际完成。
# 显示满 100% 后继续转动会误导使用者（估计只是估计），回缩则更不可接受。
NEARLY = 0.99

# 节点实例刚开始、尚未报告任何进度时 `now` 的初始值（每个字段都存在，页面无需处处使用 `?? 0`）。
BLANK: dict = {"phase": QUEUED, "node": "", "label": "", "note": "", "done": 0, "total": 0, "at": None}

PUBLIC = ("phase", "node", "label", "note", "done", "total")  # 对外发送的字段（`at` 每次按当前时间另行计算）


def fraction(spent: float, running: float, share: float | None, budget: float | None) -> float | None:
    """整个任务的完成比例（0–1），`None` 表示无法估计。

    `spent` 为已完成实例的预测秒数之和，`running` 为正在计算的实例已运行的秒数，
    `share` 为正在计算实例自身的预测秒数（`None`：当前没有计算任何内容），`budget` 为开始时固定的分母。

    单调不减由此处的构造保证：`spent` 只增不减；`running` 以 `share` 为上限，因此实例完成、
    `spent` 计入 `share` 的时刻，数值恰好衔接，不会回退。
    """
    if not budget or budget <= 0:
        return None
    done = spent + (min(max(running, 0.0), share) if share else 0.0)
    return round(min(done / budget, NEARLY), 4)
