"""What a GPU job needs, derived once in this one place from the nodes it will actually compute (timings.planned's
Work items already used for its time estimate): which runtimes it touches (each checked against a card's
architecture by compat.py) and how much VRAM its biggest GPU node needs (nodes/base.py NodeDef.vram_gb_for, the
same declared numbers behind the 计算量档位 shown on the node)."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Requirement:
    """A GPU job's needs, at the point it is being considered for placement."""

    runtimes: frozenset[str]  # extension names (or "core") of its GPU-needing nodes
    vram_gb: float  # its biggest GPU node's peak VRAM (GB); nodes run one at a time within a job, never summed
    estimated_seconds: float | None  # best current estimate of its own run time; None: no record to go by yet

    @property
    def needs_gpu(self) -> bool:
        return bool(self.runtimes)


NONE = Requirement(frozenset(), 0.0, None)


def requirement_for(job) -> Requirement:
    """`job`: farm.queue.Job. Nodes that do not need a GPU (job.works entries with .gpu False) contribute nothing:
    a job may compute some light/heavy nodes alongside its GPU ones, but only the GPU ones matter here."""
    from ...engine import Engine
    from ...nodes import node_types

    types = node_types()
    gpu_works = [w for w in job.works if w.gpu and w.type in types]
    if not gpu_works:
        return NONE
    runtimes = frozenset(types[w.type].runtime for w in gpu_works)
    engine = Engine(job.graph)
    vram = max((engine.graph.resolved(w.node).cost.vram_gb for w in gpu_works), default=0.0)
    seconds = job.remaining(time.time())
    return Requirement(runtimes, vram, seconds[0] if seconds else None)
