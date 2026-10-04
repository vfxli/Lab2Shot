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
    暂停新计算的剩余空间 storage.pause_free_pct  below this share of the data disk free, no new task is taken or
                                             started (`space`); uploads keep it too (server/transfer.py)
    每账号保留的已完成任务 tasks.keep_most   finished tasks an account keeps; the oldest beyond go (farm/queue.py
                                             trim_finished): a guard of the database, not of the disk
"""

from __future__ import annotations

import shutil
import threading
import time

from ..config import settings
from ..messages import Msg


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


def release_below_gb() -> float:
    """让出显存的空闲线 (resident.release_below_gb): a card with less memory nobody uses than this gets it back from
    our idle kept-loaded models (scheduler/placement.py pressed_cards); 0 never, and nothing when models are not kept."""
    return float(settings()["resident.release_below_gb"]) if settings()["resident.keep"] else 0.0


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


def keep_most() -> int:
    """每账号保留的已完成任务 (tasks.keep_most): the finished tasks an account keeps at most."""
    return int(settings()["tasks.keep_most"])


# ------------------------------------------------------------------ the server's disk (暂停新计算的剩余空间)

GB = 1 << 30
UPLOAD_KEEP_FREE = 10 * GB  # an upload never leaves the disk with less than this, whatever the share says (a small disk)
SPACE_S = 5.0  # how old the disk's figures may be: asked by the scheduler every pass, read at most this often
_space: dict = {"at": 0.0, "usage": None}
_space_lock = threading.Lock()


def _disk_usage() -> tuple[int, int]:
    """(total, free) bytes of the disk the data location is on (config.py paths.data_dir: tasks, caches, uploads).
    A test replaces this function to play a full disk without filling one."""
    folder = settings().data_dir
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    got = shutil.disk_usage(folder)
    return got.total, got.free


def forget_space() -> None:
    """The next `space()` reads the disk again (a test that changed `_disk_usage`; the admin's 刷新)."""
    with _space_lock:
        _space["at"] = 0.0


def space() -> dict:
    """The data disk now: {"total", "free", "pct" (the share free, %), "floor_pct" (the setting), "floor" (bytes),
    "low"}. Below the floor every account's new computing pauses (Farm.submit refuses, the scheduler holds back what has
    not started: `space_low`) and resumes by itself once it is above again; a task that started goes on to its end."""
    with _space_lock:
        if _space["usage"] is None or time.time() - _space["at"] > SPACE_S:
            try:
                _space["usage"] = _disk_usage()
            except OSError:
                _space["usage"] = (0, 0)
            _space["at"] = time.time()
        total, free = _space["usage"]
    pct = float(settings()["storage.pause_free_pct"])
    floor = int(total * pct / 100)
    return {"total": total, "free": free, "pct": 100.0 * free / total if total else 100.0, "floor_pct": pct,
            "floor": floor, "low": bool(total) and free < floor}


def space_low() -> Msg | None:
    """Why no new computing is taken now because the data disk is short (None: it is not): the refusal a submission
    gets (B-QUEUE-DISKLOW) — the waiting tasks are told N-QUEUE-DISKLOW by the scheduler."""
    s = space()
    if not s["low"]:
        return None
    return Msg("B-QUEUE-DISKLOW", free_pct=s["pct"], free_gb=s["free"] / GB, pct=s["floor_pct"])


def disk_now() -> tuple[int, int]:
    """(total, free) of the data disk this moment (not the figures `space` keeps for SPACE_S): an upload's check."""
    try:
        return _disk_usage()
    except OSError:
        return 0, 0


def upload_floor(total: int) -> int:
    """What an upload must leave free on the data disk of `total` bytes: the same share new computing pauses below,
    and never less than UPLOAD_KEEP_FREE (one rule for both: what the disk keeps free for the tasks already going)."""
    return max(int(total * float(settings()["storage.pause_free_pct"]) / 100), UPLOAD_KEEP_FREE)
