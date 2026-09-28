"""Which card (if any) a GPU node runs on: `place()` is pure — it takes the node's Need (engine/resources.py), an
inventory Snapshot and the cards a node of ours is on right now, and returns the card or why none fits — so it can be
tried over fake inventories with no thread, no lock and no real machine. Who is first to a card that frees is not
decided here: the pools hand a free card to the first task in queue order that has a GPU node waiting (pools.py).

Eligibility: a candidate card must be authorized and idle (one GPU node per card at
a time), the node's runtime must fit its architecture (compat.fit), and its free VRAM (GpuState.free_mb: nvidia-smi's
memory.total minus the foreign part of memory.used — an idle card's usage is a foreign program's, or our own idle
kept-loaded models, which make way for the node) must cover the node's declared need plus 显存余量 (policy.margin_gb).

Preference among eligible cards: first the administrator's order (显卡优先顺序, policy.gpu_order: listed cards in that
order, the others after them), then the smallest one that is still sufficient (keeps a machine's bigger cards free for
the nodes that need them), then the newer architecture, then the least utilized. It only ever chooses among cards that
are free at once: it never changes whose turn it is.
"""

from __future__ import annotations

from ...engine.resources import Need
from ...messages import Msg
from .. import policy
from . import compat
from .inventory import GpuState, Snapshot

# 显存余量 (policy.margin_gb, queue.vram_margin_gb): headroom kept free beyond the node's own declared need. Declared
# needs are PyTorch's peaks; a fresh worker's CUDA context comes on top (about 0.4-0.5 GB on an RTX 4090, 0.5-0.7 GB
# on an RTX 5090), so a margin of 0.5 can leave a correctly declared node no room at all, which is why the setting
# will not go below it.


def runs(need: Need, gpu: GpuState) -> bool:
    """The node's environment runs on this card's architecture (compat.fit: its declaration, nothing else)."""
    return compat.fit(need.runtime, gpu).ok is True


def eligible(need: Need, gpu: GpuState) -> bool:
    return gpu.authorized and runs(need, gpu) and gpu.free_gb >= need.vram_gb + policy.margin_gb()


def _cc(gpu: GpuState) -> tuple[int, int]:
    major, _, minor = gpu.compute_cap.partition(".")
    try:
        return (int(major), int(minor or 0))
    except ValueError:
        return (0, 0)


def _rank(gpu: GpuState, order: list[int] = ()) -> tuple:
    """Sort key for "best" among eligible cards: the administrator's order (`order`: CUDA numbers, unlisted cards
    after the listed ones), then smallest sufficient, then newest architecture, then least utilized."""
    major, minor = _cc(gpu)
    listed = order.index(gpu.index) if gpu.index in order else len(order)
    return (listed, gpu.memory_mb, -major, -minor, gpu.utilization)


def place(need: Need, snapshot: Snapshot, busy: set[str]) -> GpuState | Msg:
    """The card `need` runs on now, or why it waits (reason). `busy`: the authorized cards a node of ours is on right
    now (never candidates: a card runs one GPU node at a time)."""
    idle = [g for g in snapshot.authorized() if g.uuid not in busy]
    candidates = sorted((g for g in idle if eligible(need, g)), key=lambda g: _rank(g, policy.gpu_order()))
    return candidates[0] if candidates else reason(need, snapshot, busy)


# A worker process's CUDA context, which the process does not report (its vram_mb counts only its tensors): an upper
# bound of the figures in the 显存余量 note above
CONTEXT_GB = 0.8


def reclaimable_cards(need: Need, snapshot: Snapshot, busy: set[str], idle_processes: dict[str, int]) -> set[str]:
    """The idle cards where ending this server's own idle kept-loaded processes (`idle_processes`: GPU UUID -> how
    many) would let `need` fit: the card can run it, is large enough, and what is missing is no more than those
    processes' unreported CUDA contexts. Empty when ending them could not help (a node larger than any card, another
    program holding the memory), so warm models are never ended for nothing."""
    want = need.vram_gb + policy.margin_gb()
    out: set[str] = set()
    for g in snapshot.authorized():
        count = idle_processes.get(g.uuid, 0)
        if not count or g.uuid in busy or not runs(need, g) or g.memory_mb / 1024 < want:
            continue
        if 0 < want - g.free_gb <= count * CONTEXT_GB:  # short, and by no more than those contexts
            out.add(g.uuid)
    return out


def reason(need: Need, snapshot: Snapshot, busy: set[str] = frozenset()) -> Msg:
    """Why no card takes the node right now, about the cards that could take it (a message: the administrator's
    detail — users are told only what they can act on, farm/queue.py):
    0. no card is authorized to take jobs at all;
    1. architecture (compat.py: never changes without reinstalling): no authorized card can run it at all;
    2. busy: every card that can run it has a node of ours; said with the idle cards that cannot run it named, since
       those are what an artist sees free and asks about;
    3. memory: among the idle cards that can run it, the roomiest, with its current numbers and only foreign usage
       called another program's (our own idle kept-loaded models make way: GpuState.ours_mb)."""
    authorized = list(snapshot.authorized())
    if not authorized:
        return Msg("N-GPU-NOCARD")
    arch = compat.wait_reason({need.runtime}, authorized, list(snapshot.gpus))
    if arch is not None:
        return arch
    fit = [g for g in authorized if runs(need, g)]
    idle = [g for g in fit if g.uuid not in busy]
    if not idle:
        waits_for = "、".join(sorted({g.short_name for g in fit}))
        cannot = sorted({g.short_name for g in authorized if g not in fit and g.uuid not in busy})
        if not cannot:
            return Msg("N-GPU-WAITBUSY", gpus=waits_for)
        return Msg("N-GPU-WAITBUSYOTHERS", gpus=waits_for, projects=compat.runtime_title(need.runtime), idle=cannot)
    best = max(idle, key=lambda g: g.free_gb)
    if best.foreign_mb >= 512:
        return Msg("N-GPU-WAITVRAMFOREIGN", need=need.vram_gb, gpu=best.short_name, free=best.free_gb,
                   foreign=best.foreign_mb / 1024)
    return Msg("N-GPU-WAITVRAM", need=need.vram_gb, gpu=best.short_name, free=best.free_gb)
