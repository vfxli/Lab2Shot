"""How busy the server machine is, for the queue everyone sees (next to its GPUs): the CPU (all cores, over the last
few seconds), the memory and the disk of the work folder. Aggregate numbers only: no paths, processes or users."""

from __future__ import annotations

import os
import shutil
import threading
import time

from ..config import machine_memory_gb, settings
from ..engine.resident import available_gb

FRESH_S = 1.0  # read again at most this often, however many pages ask

_lock = threading.Lock()
_last: dict = {"at": 0.0, "cpu": None, "view": None}  # when it was read, the CPU counters then, what it said


def _cpu_times() -> tuple[int, int] | None:
    """(busy, total) jiffies of every core since boot (/proc/stat), None where there is none."""
    try:
        with open("/proc/stat", encoding="ascii") as f:
            fields = [int(x) for x in f.readline().split()[1:]]
    except (OSError, ValueError):
        return None
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)  # idle + iowait
    return sum(fields) - idle, sum(fields)


def now() -> dict:
    """{"cpu_percent": the CPU busy since the last read (None the first time), "cores", "memory_gb": {"used",
    "total"}, "disk_gb": {"free", "total"}} of the work folder's disk."""
    with _lock:
        t = time.time()
        if _last["view"] is not None and t - _last["at"] < FRESH_S:
            return _last["view"]
        cpu, before = _cpu_times(), _last["cpu"]
        percent = None
        if cpu and before and cpu[1] > before[1]:
            percent = round(100 * (cpu[0] - before[0]) / (cpu[1] - before[1]))
        total = machine_memory_gb()
        work = settings().work_dir
        while not work.exists():  # not made yet: the disk it will be on
            work = work.parent
        disk = shutil.disk_usage(work)
        view = {"cpu_percent": percent, "cores": os.cpu_count() or 0,
                "memory_gb": {"used": round(max(total - available_gb(), 0), 1), "total": round(total, 1)},
                "disk_gb": {"free": round(disk.free / 2**30, 1), "total": round(disk.total / 2**30, 1)}}
        _last.update(at=t, cpu=cpu, view=view)
        return view
