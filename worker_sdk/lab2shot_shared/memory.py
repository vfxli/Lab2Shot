"""The machine's memory as the worker and the core's queue both read it."""

from __future__ import annotations

def available_gb() -> float:
    """System memory free for more work (MemAvailable: page cache that can be dropped counts as free). Used by the
    worker and by the core's queue alike."""
    try:
        for line in open("/proc/meminfo", encoding="ascii"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return float("inf")  # not Linux: no guard


def total_gb() -> float:
    """The machine's memory altogether (MemTotal): what no amount of waiting can exceed."""
    try:
        for line in open("/proc/meminfo", encoding="ascii"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1 << 20)
    except OSError:
        pass
    return float("inf")
