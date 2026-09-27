"""Which GPU (if any) a job should run on: `place()` is pure — it takes a Requirement, an inventory Snapshot and
who else is waiting, and returns a Placement or a Wait with a reason — so it can be unit-tested over fake
inventories with no thread, no lock and no real machine.

Eligibility: a candidate card must be authorized, every one of the job's runtimes must fit its architecture
(compat.fit), and its free VRAM (GpuState.free_mb: nvidia-smi's memory.total minus the foreign part of memory.used —
the only cards ever offered here are ones no job of ours is running on, so what is used there is a foreign
program's, or our own idle kept-loaded models, which make way for the job) must cover the job's declared need plus
MARGIN_GB headroom.

Preference among eligible cards: first the administrator's order (显卡优先顺序, policy.gpu_order: listed cards in that
order, the others after them), then the smallest one that is still sufficient (bin-packing — keeps a machine's bigger
cards free for the jobs that actually need them), then the newer architecture (a rough stand-in for "faster": there
is no cross-vendor speed number here, just compute capability), then the least utilized.

Fairness, two rules, both here and nowhere else (farm/queue.py only supplies the facts they read — who is waiting,
for how long, whose account, how many cards that account already holds — and never judges a second time):

1. Ageing: a job waiting for a big card must not starve forever behind a stream of small jobs that keep grabbing
   every card that frees up before the big card does. `ahead`: every OTHER queued GPU job's own Requirement, how long
   it has waited and whose account it is. A job that has waited 久等优先 (policy.age_s) or longer, and whose OWN
   eligible-card set (on this same snapshot) is no larger than the job being placed now, reserves the first of ITS
   eligible cards in the same preference order: the job being placed skips that card (falling back to another eligible one, or to Wait if that was
   its only option — an aged job's reservation never manufactures a card out of nothing, it only re-orders who gets
   the one that exists).
2. 单账号占卡 (policy.cards_per_account): one account may hold at most a share of the cards taking jobs (50% by
   default; the share is a percentage so 2 / 4 / 8 cards all work with no setting changed). `account` is whose job
   this is and `held` how many cards that account has a job of ours on right now; every `Ahead` carries the same two
   for the job waiting. The cap exists to stop ONE account hogging the cards, so it only bites while turning this job
   away would really hand a card to somebody else — that is, while some OTHER account is waiting with a job that is
   itself still under the cap AND that one of the idle cards could run. With nobody else in line, with the others
   waiting for cards none of these would fit, or with every waiting account already at its own cap, the rule stands
   aside: holding a card back then leaves it idle for nobody, which is the rule working against its own purpose.
   On a one-card machine the cap is 0 — off — because the only
   card can never be another account's share. This is NOT the same thing as the borrowing rule in farm/units.py
   (「a running job borrows a second card only while no waiting job would be placed there」): that one is about the
   items of ONE job, this one is about the several jobs of ONE account.

Known limits (not solved here):
  - the reservation is recomputed fresh on every place() call from the current `ahead` list; there is no ledger of
    "this card is promised to job X", so with several aged big jobs and several cards, two lane threads could both
    skip the same card in the same instant and a different one both consider free, wasting a cycle (self-corrects
    on the next poll — the queue's memory-wait loop already tolerates rounds where nothing more happens);
  - the rule only ever protects ONE aged job per place() call (the pickiest with the least eligible cards); several
    aged jobs of equal pickiness are served in map-iteration order, not by whichever is oldest among them;
  - `ahead` must be recomputed by the caller from the live job list on every call (it is not cached here), which
    costs one Requirement derivation (an Engine + NodeDef.vram_gb_for per GPU node) per queued job per poll — fine
    at the queue depths this farm sees, not scale-tested beyond that;
  - 单账号占卡 counts cards, not time or work: an account holding its share of the cards with tiny jobs blocks the
    same as one holding them with long jobs. A per-department share (rather than per-account) is not attempted.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...messages import Msg
from .. import policy
from . import compat
from .inventory import GpuState, Snapshot
from .requirements import Requirement

# 显存余量 (policy.margin_gb, queue.vram_margin_gb): headroom kept free beyond the job's own declared need. Declared
# needs are PyTorch's peaks; a fresh worker's CUDA context comes on top (about 0.4-0.5 GB on an RTX 4090, 0.5-0.7 GB
# on an RTX 5090), so a margin of 0.5 can leave a correctly declared job no room at all, which is why the setting
# will not go below it.


@dataclass(frozen=True)
class Placement:
    gpu: GpuState


@dataclass(frozen=True)
class Wait:
    reason: Msg  # reason(): no card authorized, architecture, the busy cards it waits for, or memory


@dataclass(frozen=True)
class Ahead:
    """One other queued GPU job whose claim to a card comes first: what it needs, how long it has waited (the ageing
    rule), whose account it is and how many cards that account already holds (单账号占卡)."""

    requirement: Requirement
    waiting_seconds: float
    account: int = 0  # the account id (farm/clients.py Client.user); 0: not said
    held: int = 0  # cards that account has a job of ours on right now


def eligible(requirement: Requirement, gpu: GpuState) -> bool:
    if not gpu.authorized:
        return False
    if any(compat.fit(r, gpu).ok is not True for r in requirement.runtimes):
        return False
    return gpu.free_gb >= requirement.vram_gb + policy.margin_gb()


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


def place(requirement: Requirement, snapshot: Snapshot, busy_uuids: set[str], ahead: list[Ahead] = (),
          account: int = 0, held: int = 0) -> Placement | Wait:
    """`busy_uuids`: authorized GPUs a job of ours is already running on right now (never candidates: this farm
    runs one job per GPU at a time, matching the pre-scheduler queue's own rule). `account`: whose job this is
    (farm/clients.py Client.user); `held`: how many cards that account has a job of ours on right now — the two
    facts 单账号占卡 reads."""
    idle = [g for g in snapshot.authorized() if g.uuid not in busy_uuids]
    order = policy.gpu_order()
    candidates = sorted((g for g in idle if eligible(requirement, g)), key=lambda g: _rank(g, order))
    if not candidates:
        return Wait(reason(requirement, snapshot, busy_uuids))
    if (over := _over_share(snapshot, idle, ahead, account, held)) is not None:
        return Wait(over)
    reserved = _aged_reservations(idle, candidates, ahead, order)
    left = [g for g in candidates if g.uuid not in reserved] or candidates  # an aged reservation never starves
    return Placement(left[0])                                              # everyone: it only re-orders, never empties


# A worker process's CUDA context, which the process does not report (its vram_mb counts only its tensors): an upper
# bound of the figures in the 显存余量 note above
CONTEXT_GB = 0.8


def reclaimable_cards(requirement: Requirement, snapshot: Snapshot, busy_uuids: set[str],
                      idle_processes: dict[str, int]) -> set[str]:
    """The idle cards where ending this server's own idle kept-loaded processes (`idle_processes`: GPU UUID -> how
    many) would let `requirement` fit: the card can run it, is large enough, and what is missing is no more than
    those processes' unreported CUDA contexts. Empty when ending them could not help (a job larger than any card,
    another program holding the memory), so warm models are never ended for nothing."""
    need = requirement.vram_gb + policy.margin_gb()
    out: set[str] = set()
    for g in snapshot.authorized():
        count = idle_processes.get(g.uuid, 0)
        if not count or g.uuid in busy_uuids:
            continue
        if any(compat.fit(r, g).ok is not True for r in requirement.runtimes) or g.memory_mb / 1024 < need:
            continue
        if 0 < need - g.free_gb <= count * CONTEXT_GB:  # short, and by no more than those contexts
            out.add(g.uuid)
    return out


def _over_share(snapshot: Snapshot, idle: list[GpuState], ahead: list[Ahead], account: int, held: int) -> Msg | None:
    """单账号占卡, the second fairness rule (module doc 2): why this job may not take another card right now, although
    one fits — None when it may. Off when the share is 100% or fewer than two cards take jobs (policy), and off
    unless turning this job away would really hand a card to someone else: another account, itself still under the
    cap, waiting for a job one of the idle cards could run. Otherwise the card would only sit idle for nobody."""
    cards = len(list(snapshot.authorized()))
    cap = policy.cards_per_account(cards)
    if not cap or held < cap:
        return None
    if not any(_would_take(a, account, cap, idle) for a in ahead):
        return None
    return Msg("N-GPU-WAITSHARE", held=held, cap=cap, cards=cards)


def _would_take(a: Ahead, account: int, cap: int, idle: list[GpuState]) -> bool:
    """Would this other waiting job really start on one of the idle cards if the job being placed stood back? Only
    then is there any point in turning that job away (module doc 2)."""
    return a.account != account and a.held < cap and any(eligible(a.requirement, g) for g in idle)


def _aged_reservations(idle: list[GpuState], candidates: list[GpuState], ahead: list[Ahead],
                       order: list[int] = ()) -> set[str]:
    reserved = set()
    age_s = policy.age_s()
    for a in ahead:
        if a.waiting_seconds < age_s:
            continue
        its_candidates = sorted((g for g in idle if eligible(a.requirement, g)), key=lambda g: _rank(g, order))
        if its_candidates and len(its_candidates) <= len(candidates):
            reserved.add(its_candidates[0].uuid)
    return reserved


def reason(requirement: Requirement, snapshot: Snapshot, busy_uuids: set[str] = frozenset()) -> Msg:
    """Why nothing fits right now, about the cards that could take it (a message: the administrator's detail — users
    are told only that the job waits for a machine, farm/queue.py):
    0. no card is authorized to take jobs at all;
    1. architecture (compat.py: never changes without reinstalling): no authorized card can run it at all;
    2. busy: every card that can run it has a job of ours; said with the idle cards that cannot run it named, since
       those are what an artist sees free and asks about (otherwise a job waiting for a busy card would point at an
       idle card it can never use);
    3. memory: among the idle cards that can run it, the roomiest, with today's numbers and only foreign usage
       called another program's (our own idle kept-loaded models make way: GpuState.ours_mb)."""
    authorized = list(snapshot.authorized())
    if not authorized:
        return Msg("N-GPU-NOCARD")
    arch = compat.wait_reason(set(requirement.runtimes), authorized, list(snapshot.gpus))
    if arch is not None:
        return arch
    runs = [g for g in authorized if all(compat.fit(r, g).ok is True for r in requirement.runtimes)]
    idle = [g for g in runs if g.uuid not in busy_uuids]
    if not idle:
        waits_for = "、".join(sorted({g.short_name for g in runs}))
        cannot = sorted({g.short_name for g in authorized if g not in runs and g.uuid not in busy_uuids})
        if not cannot:
            return Msg("N-GPU-WAITBUSY", gpus=waits_for)
        titles = "、".join(sorted({compat.runtime_title(r) for r in requirement.runtimes
                                   if any(compat.fit(r, g).ok is not True for g in authorized if g not in runs)}))
        return Msg("N-GPU-WAITBUSYOTHERS", gpus=waits_for, projects=titles, idle=cannot)
    best = max(idle, key=lambda g: g.free_gb)
    if best.foreign_mb >= 512:
        return Msg("N-GPU-WAITVRAMFOREIGN", need=requirement.vram_gb, gpu=best.short_name, free=best.free_gb,
                   foreign=best.foreign_mb / 1024)
    return Msg("N-GPU-WAITVRAM", need=requirement.vram_gb, gpu=best.short_name, free=best.free_gb)
