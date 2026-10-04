"""Scheduling by node: the machine's places (each authorized GPU, the CPU slots) and which task's node gets one, with
the time limit of a node. Separate concerns kept apart so each can change on its own:

    inventory.py     live GPU state (nvidia-smi), refreshed by a background thread — never inside a lock of the farm's
    compat.py        can a runtime's environment (an extension's, or the core's) run on a given architecture: the
                     extension's declared architectures (Extension.env_archs) and the install-time probe, nothing
                     else
    placement.py     place(): which free card a GPU node fits on (architecture, VRAM, the preference order), and why
                     none does
    pools.py         Pools: the places, the rules of who gets one (queue order, the limits of a task and of the
                     machine, memory), the time limit, and background work; one dispatcher
                     thread decides

The engine asks through the interface it defines itself (engine/resources.py: Need, Ticket, Resources), which
TaskResources implements for one task; farm/queue.py builds the Pools and hands each task's cook its TaskResources.
extensions/gpu_archs.py is the install-time recorder that compat.py reads (and has probe, on a thread of its own, an
environment that was never probed or changed since). No other module computes GPU compatibility or reads nvidia-smi
directly.
"""

from __future__ import annotations

from .compat import Fit, card_extensions, fit, runtime_record, wait_reason
from .inventory import GpuState, Host, LocalHost, Snapshot, local_host
from .placement import eligible, place, pressed_cards, reclaimable_cards
from .pools import Pools, TaskResources, Ticket

__all__ = [
    "Fit",
    "GpuState",
    "Host",
    "LocalHost",
    "Pools",
    "Snapshot",
    "TaskResources",
    "Ticket",
    "card_extensions",
    "eligible",
    "fit",
    "local_host",
    "place",
    "pressed_cards",
    "reclaimable_cards",
    "runtime_record",
    "wait_reason",
]
