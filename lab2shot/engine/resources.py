"""What a cook asks for before it runs a node: a GPU or a CPU slot, by the node's resolved cost. The engine defines
this interface and the farm implements it (farm/scheduler: the machine's pools, who gets which slot, the time limit),
handing one to Engine.cook, so the engine never imports the farm (the same way it gets its services: nodes/services.py).

    Need       what one node needs to run: a GPU (its runtime's environment must run on the card, its declared VRAM
               must be free there) or a CPU slot, and the system memory it takes at its peak (the memory guard)
    Ticket     one node's request, as the scheduler answers it: granted once it has its place (`gpu`: the card's UUID,
               "" for a CPU slot; `gpu_name`: its model, what the timing records keep); `expired`, once the node ran past
               its time limit, the scheduler's words for it (it then sets the node's stop: a worker's process is killed,
               a core node stops at its next frame, CookContext.each_done / check_stop), which the node fails with
    Resources  ask(need, stop, woken): a ticket at once, never waiting; `woken()` is called (on the scheduler's
               thread, holding nothing) when it is granted. done(ticket): its place goes back the moment the node is
               through, or the request is withdrawn when it was never granted (the cook was stopped)

A node holds its place only while it runs: the engine asks when the node's inputs are there and it has to compute,
and gives the place back as soon as it is through, or while it waits for another cook of the same result (then it asks
again if it still has to compute: engine/cook.py _Place)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..messages import Msg

CPU, GPU = "cpu", "gpu"


@dataclass(frozen=True)
class Need:
    kind: str  # CPU or GPU
    runtime: str  # the node type's runtime ("core", or its extension): a card must run it (its architecture)
    vram_gb: float  # a GPU node's declared peak VRAM (its resolved cost); 0 for a CPU slot
    ram_gb: float  # the system memory it takes at its peak (its resolved cost): kept free for it before it starts
    label: str  # the node's label: what the administrator sees waiting


class Ticket(Protocol):
    @property
    def granted(self) -> bool: ...

    gpu: str
    gpu_name: str
    expired: Msg | None


class Resources(Protocol):
    def ask(self, need: Need, stop: threading.Event, woken: Callable[[], None]) -> Ticket: ...

    def done(self, ticket: Ticket) -> None: ...
