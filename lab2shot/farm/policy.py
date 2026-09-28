"""调度参数：the numbers the queue and the scheduler go by, in one place.

Every one of them is a setting the administrator can change (lab2shot/config.py, the page 计算与显卡),
read here — once, where it is used — so no module keeps its own copy and nothing has to be restarted for a change to
apply:

    单节点超时       queue.node_minutes      how long one node may run before it is stopped (it fails, what needs it is skipped)
    单任务最多占卡数 queue.task_gpus         GPUs one task holds at once (a card runs one node at a time)
    单任务 CPU 节点上限 queue.task_cpus      nodes without a GPU one task runs at once
    全局 CPU 节点上限 queue.cpu_nodes        nodes without a GPU the whole machine runs at once (default: cores ÷ 8;
                                             each of them uses several cores: it is a count of nodes, not of cores)
    帧数上限         queue.max_frames        the frames one submission may cook
    显卡任务 · 计算任务 queue.gpu_jobs · queue.compute_jobs   the two switches (a task that has not started waits)
    显存余量         queue.vram_margin_gb    headroom kept free beyond a node's own declared need
    显卡优先顺序     queue.gpu_order         CUDA numbers preferred first among the free cards a node fits on
"""

from __future__ import annotations

from ..config import settings


def node_timeout_s() -> float:
    return float(settings()["queue.node_minutes"]) * 60.0


def task_gpus() -> int:
    return int(settings()["queue.task_gpus"])


def task_cpus() -> int:
    return int(settings()["queue.task_cpus"])


def cpu_nodes() -> int:
    return int(settings()["queue.cpu_nodes"])


def max_frames() -> int:
    """The most frames one submission may cook; the server refuses a submission beyond it, saying so."""
    return int(settings()["queue.max_frames"])


def margin_gb() -> float:
    return float(settings()["queue.vram_margin_gb"])


def gpu_order() -> list[int]:
    """The administrator's preferred cards (CUDA numbers, first preferred first); [] when not set."""
    from ..config import gpu_order as parse

    return parse(str(settings()["queue.gpu_order"]))


def gpu_enabled() -> bool:
    """显卡任务 (queue.gpu_jobs): off is exactly like no GPU being authorized for a task that has not started: its GPU
    nodes wait until this (or an authorization) is turned back on; a task that started goes on to its end."""
    return bool(settings()["queue.gpu_jobs"])


def compute_enabled() -> bool:
    """计算任务 (queue.compute_jobs): off refuses every new task at submission (Farm.submit), and holds back every
    task that had not started when it was switched off; a task that started goes on to its end."""
    return bool(settings()["queue.compute_jobs"])
