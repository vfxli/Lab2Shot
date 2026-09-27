"""GPU scheduling: which GPU (if any, of possibly several on a multi-card workstation) a job
should run on, and why it must wait when none fit. Separate concerns kept apart so each can change on its own:

    inventory.py     live GPU state (nvidia-smi), refreshed by a background thread — never inside the queue's lock
    compat.py        can a runtime's environment (an extension's, or the core's) run on a given architecture: the
                     extension's declared architectures (Extension.env_archs) and the install-time probe, nothing
                     else
    requirements.py  what a job needs (runtimes, VRAM), derived once from its planned nodes
    placement.py     place(): pure placement policy — eligibility, best-fit, ageing fairness

farm/queue.py is the only caller: its GPU lanes ask place() for the next job they should run and report the
result; extensions/gpu_archs.py stays the install-time recorder that compat.py reads (and probes lazily when an
environment predates it). No other module computes GPU compatibility or reads nvidia-smi directly (the queue
stays thin).
"""

from __future__ import annotations

from .compat import Fit, card_extensions, fit, runtime_record, wait_reason
from .inventory import GpuState, Host, LocalHost, Snapshot, local_host
from .placement import Ahead, Placement, Wait, eligible, place, reclaimable_cards
from .requirements import Requirement, requirement_for

__all__ = [
    "Ahead",
    "Fit",
    "GpuState",
    "Host",
    "LocalHost",
    "Placement",
    "Requirement",
    "Snapshot",
    "Wait",
    "card_extensions",
    "eligible",
    "fit",
    "local_host",
    "place",
    "reclaimable_cards",
    "requirement_for",
    "runtime_record",
    "wait_reason",
]
