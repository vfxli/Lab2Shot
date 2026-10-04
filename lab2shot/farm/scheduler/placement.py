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

import math

from ...engine.resources import GPU, Need
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


def holds(runtime: str, vram_gb: float, snapshot: Snapshot) -> bool:
    """A card authorized for jobs runs `runtime` and is large enough for `vram_gb` with 显存余量, busy or not: whether a
    node that steps down on a smaller card runs its full tier on this machine (nodes/applies.py Cost.vram_full_gb)."""
    want = vram_gb + policy.margin_gb()
    return any(runs(Need(GPU, runtime, vram_gb, 0.0, ""), g) and g.memory_mb / 1024 >= want for g in snapshot.authorized())


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


def reclaimable_cards(need: Need, snapshot: Snapshot, busy: set[str], idle_context_mb: dict[str, int]) -> dict[str, int]:
    """The idle cards where ending this server's own idle kept-loaded processes would let `need` fit, each with the
    MB it is short (GPU UUID -> MB): the card can run it, is large enough, and what is missing is no more than what
    those processes hold beyond their reported memory (`idle_context_mb`: GPU UUID -> their CUDA contexts, as each
    measured its own: engine/resident.py Pool.idle_context_mb). Empty when ending them could not help (a node larger
    than any card, another program holding the memory), so warm models are never ended for nothing."""
    want = need.vram_gb + policy.margin_gb()
    out: dict[str, int] = {}
    for g in snapshot.authorized():
        held = idle_context_mb.get(g.uuid, 0)
        if not held or g.uuid in busy or not runs(need, g) or g.memory_mb / 1024 < want:
            continue
        short = math.ceil((want - g.free_gb) * 1024)
        if 0 < short <= held:  # short, and by no more than those contexts
            out[g.uuid] = short
    return out


def pressed_cards(snapshot: Snapshot, busy: set[str], below_gb: float) -> dict[str, int]:
    """The authorized cards no node of ours is on whose memory nobody uses (GpuState.unused_mb: our own idle models'
    counted as used) is below `below_gb` (resident.release_below_gb; 0: never), each with the MB it is short (GPU
    UUID -> MB): our idle kept-loaded models there give it back (engine/resident.py Pool.relieve), so another
    program, or another server's job, gets it."""
    if below_gb <= 0:
        return {}
    floor = int(below_gb * 1024)
    return {g.uuid: floor - g.unused_mb for g in snapshot.authorized()
            if g.uuid not in busy and g.unused_mb < floor}


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
        waits_for = sorted({g.short_name for g in fit})  # a list: the message joins it in its language
        cannot = sorted({g.short_name for g in authorized if g not in fit and g.uuid not in busy})
        if not cannot:
            return Msg("N-GPU-WAITBUSY", gpus=waits_for)
        return Msg("N-GPU-WAITBUSYOTHERS", gpus=waits_for, projects=compat.runtime_title(need.runtime), idle=cannot)
    best = max(idle, key=lambda g: g.free_gb)
    if best.foreign_mb >= 512:
        return Msg("N-GPU-WAITVRAMFOREIGN", need=need.vram_gb, gpu=best.short_name, free=best.free_gb,
                   foreign=best.foreign_mb / 1024)
    return Msg("N-GPU-WAITVRAM", need=need.vram_gb, gpu=best.short_name, free=best.free_gb)
