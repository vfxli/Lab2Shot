"""The administrator's cards: artists never see or choose a card, the scheduler places jobs; the
administrator gets everything about the cards in one view, and sees what a change does before making it.

    view(farm)                 every card this machine has: model, VRAM, architecture, whether it takes jobs, what it
                               runs and its load, which GPU extensions 能跑 / 不能跑 / 未知 there and why
                               (scheduler.compat.card_extensions), the parameter tiers it makes possible; every tier
                               and which cards enable it; every GPU node's measured VRAM; every GPU node waiting for a
                               card, with why it waits and whether any authorised card could ever run it
    consequences(farm, uuids)  what authorising exactly `uuids` would change: tiers no card could run any more (or
                               could now), waiting nodes that could never start, GPU extensions no authorised card runs;
                               said as messages, before the switch is made

Everything is detected (nvidia-smi, the environments' declared and probed architectures): nothing maps environments
to cards by hand, and a new card needs no configuration. Whether an environment runs on a card is decided in one
place only: the architectures the extension declares (Extension.env_archs, scheduler/compat.py). What a tier needs is the node's own declaration
(NodeDef.cost, its OptionTraits): measured peak VRAM plus the scheduler's margin, and an environment that runs there.
Who may see any of it is server/available.py's (FIELDS, the farm.cards capability)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from lab2shot_shared.gpu_arch import cap_to_sm

from ..errors import Invalid
from ..messages import Msg
from ..engine.resources import GPU
from .scheduler import compat
from .scheduler.inventory import GpuState
from . import policy


@dataclass(frozen=True)
class Tier:
    """A GPU node at one setting: its default (param ""), or a choice that declares its own measured VRAM."""

    node: str
    label: str
    runtime: str
    vram_gb: float
    param: str = ""
    option: str = ""

    @property
    def id(self) -> str:
        return f"{self.node}:{self.param}={self.option}" if self.param else self.node

    def message(self) -> Msg:
        return (Msg("I-CARDS-TIEROPTION", node=self.label, setting=self.param, option=self.option, vram=self.vram_gb) if self.param
                else Msg("I-CARDS-TIER", node=self.label, vram=self.vram_gb))


def tiers() -> list[Tier]:
    """Every GPU node's tiers, from its declarations: its default (Cost), each choice whose trait measures its own VRAM
    (OptionTrait.vram_gb), and each measured setting of a heavy parameter (the parameter spec's `measured`: setting ->
    GB, None when the setting changes the time only; nodes/applies.py setting_vram places a job by the same numbers)."""
    from ..nodes import node_types
    from ..nodes.applies import ParamIn

    out = []
    for t in sorted(node_types().values(), key=lambda t: t.id):
        if not (t.cost.gpu or any(tr.gpu for tr in t.traits)):
            continue
        found = {t.id: Tier(t.id, t.label, t.runtime, float(t.cost.vram_gb))}
        for tr in t.traits:
            if tr.vram_gb is not None and isinstance(tr.when, ParamIn):
                found |= {x.id: x for x in (Tier(t.id, t.label, t.runtime, float(tr.vram_gb), tr.when.name, str(v)) for v in tr.when.values)}
        for spec in t.param_specs():
            found |= {x.id: x for x in (Tier(t.id, t.label, t.runtime, float(gb), spec["name"], str(setting))
                                         for setting, gb in (spec.get("measured") or {}).items() if gb is not None)}
        out += found.values()
    return out


def fits(vram_gb: float, runtimes, gpu: GpuState) -> bool:
    """Could `gpu` ever run this (its whole memory, not what is free now; every runtime's environment runs there)?"""
    return gpu.memory_mb / 1024 >= vram_gb + policy.margin_gb() and all(compat.fit(r, gpu).ok is True for r in runtimes)


def _authorised(gpus, uuids) -> list[GpuState]:
    return [replace(g, authorized=g.uuid in uuids) for g in gpus]


def _waiting(farm) -> list:
    """The GPU nodes waiting for a card (farm/scheduler/pools.py Ticket), with the job each is of."""
    jobs = {j.id: j for j in farm.jobs_now()}
    return [(t, jobs[t.task]) for t in farm.pools.now() if t.need.kind == GPU and not t.granted and t.task in jobs]


def _enabled(all_tiers: list[Tier], gpus: list[GpuState]) -> set[str]:
    return {t.id for t in all_tiers if any(g.authorized and fits(t.vram_gb, (t.runtime,), g) for g in gpus)}


def _card(g: GpuState, running: dict, all_tiers: list[Tier]) -> dict:
    ticket, job = running.get(g.uuid, (None, None))
    return {"uuid": g.uuid, "index": g.index, "name": g.name, "model": g.short_name, "memory_gb": round(g.memory_mb / 1024, 1),
            "arch": cap_to_sm(g.compute_cap) if g.compute_cap else "", "compute_cap": g.compute_cap, "authorized": g.authorized,
            "load": {"utilization": g.utilization, "used_gb": round(g.used_mb / 1024, 1), "temperature": g.temperature},
            "running": {"job": job.id, "title": job.title, "who": job.client.who, "node": ticket.need.label} if job else None,
            "extensions": compat.card_extensions(g),
            "tiers": [t.id for t in all_tiers if fits(t.vram_gb, (t.runtime,), g)]}


def view(farm) -> dict:
    gpus = list(farm.host.snapshot().gpus)
    jobs = {j.id: j for j in farm.jobs_now()}
    running = {t.gpu: (t, jobs[t.task]) for t in farm.pools.now() if t.granted and t.gpu and t.task in jobs}  # card -> the node on it
    all_tiers = tiers()
    enabled = _enabled(all_tiers, gpus)
    from ..extensions import extensions
    from ..nodes import node_types

    titles = {name: ext.title for name, ext in extensions().items()}  # the page groups tiers and nodes by extension
    return {
        "cards": [_card(g, running, all_tiers) for g in gpus],
        "tiers": [{"id": t.id, "node": t.node, "label": t.label, "param": t.param, "option": t.option, "vram_gb": t.vram_gb,
                   "runtime": t.runtime, "runtime_title": titles.get(t.runtime, t.runtime),
                   "message": t.message().json(), "available": t.id in enabled,
                   "cards": [g.uuid for g in gpus if fits(t.vram_gb, (t.runtime,), g)]} for t in all_tiers],
        "nodes": [{"node": t.id, "label": t.label, "runtime": t.runtime, "runtime_title": titles.get(t.runtime, t.runtime),
                   "vram_gb": t.cost.vram_gb, "vram_measured": t.cost.vram_measured,
                   "measured_on": t.cost.measured_on, "note": t.cost.said}
                  for t in sorted(node_types().values(), key=lambda t: t.id) if t.cost.gpu or any(tr.gpu for tr in t.traits)],
        "waiting": [_waiting_row(t, j, gpus) for t, j in _waiting(farm)],
        # hourly average utilisation: from the nvidia-smi reading already being made, no extra call
        # (scheduler/inventory.py _note_hour)
        "hourly": farm.host.hourly() if hasattr(farm.host, "hourly") else [],
    }


def _waiting_row(ticket, job, gpus: list[GpuState]) -> dict:
    need = ticket.need
    ever = any(g.authorized and fits(need.vram_gb, (need.runtime,), g) for g in gpus)
    return {"job": job.id, "title": job.title, "who": job.client.who, "node": need.label, "vram_gb": need.vram_gb,
            "runtimes": [need.runtime], "reason": ticket.reason.json() if ticket.reason else None, "runnable_ever": ever}


def consequences(farm, uuids: list[str]) -> dict:
    """What authorising exactly `uuids` would change (nothing is changed)."""
    gpus = list(farm.host.snapshot().gpus)
    known = {g.uuid for g in gpus}
    if unknown := [u for u in uuids if u not in known]:
        raise Invalid(Msg("E-GPU-UNKNOWN", uuids=unknown))
    after = _authorised(gpus, set(uuids))
    all_tiers = tiers()
    by_id = {t.id: t for t in all_tiers}
    was, will = _enabled(all_tiers, gpus), _enabled(all_tiers, after)
    lost, gained = [by_id[i] for i in sorted(was - will)], [by_id[i] for i in sorted(will - was)]
    stuck = list({job.id: job for t, job in _waiting(farm)
                  if not any(g.authorized and fits(t.need.vram_gb, (t.need.runtime,), g) for g in after)}.values())
    runs = lambda ext, cards: any(g.authorized and compat.fit(ext.name, g).ok is True for g in cards)  # noqa: E731
    dropped = [ext for ext in compat._gpu_extensions() if runs(ext, gpus) and not runs(ext, after)]
    messages = []
    if lost:
        messages.append(Msg("W-CARDS-TIERSLOST", tiers=[t.message() for t in lost]))
    if stuck:
        messages.append(Msg("W-CARDS-JOBSSTUCK", count=len(stuck), jobs=[j.title for j in stuck]))
    if dropped:
        messages.append(Msg("W-CARDS-EXTENSIONSLOST", extensions=[e.title for e in dropped]))
    if gained:
        messages.append(Msg("I-CARDS-TIERSGAINED", tiers=[t.message() for t in gained]))
    if not messages:
        messages.append(Msg("I-CARDS-NOCHANGE"))
    return {"authorized": sorted(set(uuids)), "lost_tiers": [t.id for t in lost], "gained_tiers": [t.id for t in gained],
            "stuck_jobs": [{"job": j.id, "title": j.title, "who": j.client.who} for j in stuck],
            "unrunnable_extensions": [e.name for e in dropped], "messages": [m.json() for m in messages]}
