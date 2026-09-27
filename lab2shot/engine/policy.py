"""Cook classification: whether a cook runs immediately or is queued, and whether displaying a node may start it.

Every cook belongs to the logged-in account (lab2shot/accounts.py). All computation runs on the server; the page
only displays results. The model follows Houdini (SOPs cook for the viewport; a ROP renders or submits to Deadline)
and Nuke (the Viewer computes; a Write renders).

Each node declares two properties (nodes/base.py): its cost (NodeDef.cost, resolved against its parameters by
nodes/applies.py to LIGHT, HEAVY CPU or GPU; derived from where it runs unless declared explicitly) and whether it
delivers files (NodeDef.delivers: 「输出」). A cook is classified by the nodes it actually computes, i.e. its targets
and their upstream nodes whose results are not cached (Engine.computes), never by the whole graph. The rules below
are defined here and nowhere else:

- 交互计算 (interactive lane, LIGHT): every node to compute is light and none delivers. The cook runs immediately in
  a small dedicated pool; the node shown in the viewer is cooked this way automatically (by_itself).
- 排队任务 (queued: HEAVY, GPU): at least one node to compute is heavy CPU work or requires a GPU. It starts only on
  an explicit click and waits in the CPU lane (limited concurrency, set by the administrator) or for a GPU.
- 交付 (delivers): 「输出」 is among the nodes to compute (always the case when it is cooked, since it caches
  nothing). It starts only on an explicit click, never by displaying a node. It runs immediately when all upstream
  nodes are cached or light; otherwise the whole job is queued as above.

The status reply reports the classification for every node (engine/evaluation.py: what a click and a display would
cook), the web editor only reads it (webui/src/graph/rules.ts), and the queue enforces it (farm/queue.py Farm.submit).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..nodes.applies import GPU, HEAVY, LIGHT

LANES = (LIGHT, HEAVY, GPU)  # cook lanes ordered from cheapest; a cook runs in the lane its costliest node requires


@dataclass(frozen=True)
class CookKind:
    lane: str  # LIGHT: immediately, in the interactive pool; HEAVY: the CPU lane; GPU: waits for a GPU
    delivers: bool  # delivers files to the user (「输出」)

    @property
    def queues(self) -> bool:
        """Whether the cook is queued (CPU lane or GPU)."""
        return self.lane != LIGHT

    @property
    def by_itself(self) -> bool:
        """Whether displaying a node may start the cook: light and free of side effects."""
        return not self.queues and not self.delivers

    @property
    def case(self) -> str:
        """The case name used by the page: deliver, queue or interactive."""
        return "deliver" if self.delivers else "queue" if self.queues else "interactive"

    def describe(self) -> dict:
        return {"lane": self.lane, "delivers": self.delivers, "queues": self.queues, "by_itself": self.by_itself, "case": self.case}


def classify(nodes: Iterable[tuple[str, bool]]) -> CookKind:
    """Classify a cook that computes the given nodes (each given as its resolved lane and whether it delivers). A cook
    with nothing to compute (everything cached) is interactive."""
    nodes = list(nodes)
    lane = max((lane for lane, _ in nodes), key=LANES.index, default=LIGHT)
    return CookKind(lane, any(delivers for _, delivers in nodes))
