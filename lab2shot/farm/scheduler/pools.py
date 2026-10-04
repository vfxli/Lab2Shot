"""The machine's places for nodes, and who gets them (按节点调度). A task's cook
(engine/cook.py) asks for one place per node once that node's inputs are there and it has to compute (engine/
resources.py: a GPU or a CPU slot, by its resolved cost), and gives it back the moment the node is through. The rules
are these, and nothing else:

- each authorized card is one GPU slot: one GPU node at a time on it;
- the machine runs at most 全局 CPU 节点上限 nodes without a GPU at once (policy.cpu_nodes);
- one task holds at most 单任务最多占卡数 cards and runs at most 单任务 CPU 节点上限 nodes without a GPU at once;
- a place that frees goes to the first task in queue order (the queue's: first come, first served, an administrator's
  插队 at the front) that has a node waiting for that kind of place and is under its own limits;
- a GPU node goes only on a card its environment runs on, with its declared VRAM free there (placement.py); the
  administrator's 显卡优先顺序 chooses among the cards that are free at once;
- no node starts while the machine has less memory free than 保留内存 (or what the node declares, whichever is more):
  idle kept-loaded models make way first (engine/resident.py free_ram), and the nodes after it wait too. What a node
  granted takes is counted as taken from then on: within a pass, and for RAM_SETTLE_S after it, while it may not have
  taken it yet (free memory is read once a pass and says nothing of a node that is still starting). A node that
  declares more than the machine has would wait for ever: it says so (W-QUEUE-NEVERRAM, told to the user) and holds
  up nobody;
- a card no node of ours is on that runs short of memory nobody uses (让出显存线, resident.release_below_gb: another
  program's or another server's use included) gets it back from our idle kept-loaded models there (placement.py
  pressed_cards, engine/resident.py relieve);
- a node runs no longer than 单节点超时: then its stop is set and it fails with the reason (Ticket.expired);
- no node of an extension starts while the installer switches that extension's environment (`withholding`, a moment).

A task that has not started yet (none of its nodes ever had a place) gets none while the queue is held for a restart,
or while the administrator has switched 计算任务 (or, for its GPU nodes, 显卡任务) off; a task that started goes on to
its end.

Background work uses the pools too (the viewer's proxies of what was computed, `later`): it takes a free CPU slot
only while no node waits that could have it, and it never delays a node (a node takes its slot even while background
work is there).

One thread (the dispatcher, `run`) makes every decision, so no two decisions race: asking, giving back, a changed
setting or order only wake it. It never holds the pools' lock while models make way (seconds each), and never calls
the queue while holding it."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from ...config import machine_memory_gb
from ...engine.resident import available_gb, keep_free_gb
from ...engine.resident import pool as resident
from ...engine.resources import CPU, GPU, Need
from ...messages import Msg
from .. import policy
from . import placement
from .inventory import Host

POLL_S = 1.0  # how often the dispatcher looks again with nothing else happening: time limits, another program
# freeing a card's memory, the machine's memory (none of these is an event of the pools')
BACKGROUND_NICE = 19  # background work's CPU priority: the lowest
# the waits that are only about video memory (placement.reason, step 3): idle kept-loaded models may make way
VRAM_WAITS = frozenset({"N-GPU-WAITVRAM", "N-GPU-WAITVRAMFOREIGN"})
RAM_SETTLE_S = 60.0  # a node granted this recently may not have taken the memory it declared yet (a worker loading
# its model): what it declared is counted as taken until then; after it, the machine's free memory says it


@dataclass(eq=False)
class Ticket:
    """One node's request (engine/resources.py Ticket): whose task, what it needs, the node's stop (set when it ran
    past its time limit), and who to wake when it is granted. `reason`: why it waits now, the administrator's detail
    (the card it waits for, the memory, a limit), None while it is granted or nothing needs saying."""

    task: str  # the job's id
    need: Need
    stop: threading.Event
    woken: Callable[[], None]
    asked: float = field(default_factory=time.monotonic)
    since: float | None = None  # when it was granted (monotonic)
    gpu: str = ""  # the card's UUID ("" a CPU slot)
    gpu_name: str = ""  # its model (the timing records)
    expired: Msg | None = None
    reason: Msg | None = None

    @property
    def granted(self) -> bool:
        return self.since is not None


class TaskResources:
    """One task's side of the pools, as its cook takes it (engine/resources.py Resources)."""

    def __init__(self, pools: Pools, task: str) -> None:
        self.pools, self.task = pools, task

    def ask(self, need: Need, stop: threading.Event, woken: Callable[[], None]) -> Ticket:
        return self.pools.ask(Ticket(self.task, need, stop, woken))

    def done(self, ticket: Ticket) -> None:
        self.pools.done(ticket)


@dataclass
class _Pass:
    """What one of the dispatcher's passes decided, done once the lock is let go."""

    granted: list[Ticket] = field(default_factory=list)
    started: list[str] = field(default_factory=list)  # tasks whose first node got its place now
    free_ram: tuple[float, set[str]] | None = None  # memory to make way for, and the runtimes to spare
    reclaim: tuple[Need, set[str]] | None = None  # a GPU node short of video memory only, and the busy cards
    busy: set[str] = field(default_factory=set)  # the cards a node of ours is on (granted in this pass too)
    background: list[Callable[[], None]] = field(default_factory=list)


class Pools:
    """The places and the dispatcher. `line()`: the tasks in queue order, each (its id, whether it started);
    `held()`: the queue is held (a restart is coming); `started(task)`: told once, when a task's first node got its
    place; `spawn(work)`: runs background work on a thread of the farm (Farm._spawn: close() waits for it). The farm
    gives the four; the pools never import the queue."""

    def __init__(self, host: Host, line: Callable[[], list[tuple[str, bool]]], held: Callable[[], bool],
                 started: Callable[[str], None], spawn: Callable[[Callable[[], None]], None]) -> None:
        self.host, self.line, self.held, self.started, self.spawn = host, line, held, started, spawn
        self.cond = threading.Condition()
        self.tickets: list[Ticket] = []  # asked and not given back, granted or not, in the order asked
        self._later: deque[Callable[[], None]] = deque()
        self._background = 0  # background work running now
        self._begun: set[str] = set()  # tasks told `started` already
        self._changes = 0  # counts every wake: the dispatcher waits only when nothing happened during its pass
        self._ended = False
        self._withheld: set[str] = set()  # runtimes whose environment is being switched: no node of theirs starts
        self._relieved: dict[str, float] = {}  # card -> when its idle models last gave memory back (_make_way)

    # ------------------------------------------------------------------ what the cooks and the queue call

    def ask(self, ticket: Ticket) -> Ticket:
        with self.cond:
            self.tickets.append(ticket)
            self._wake()
        return ticket

    def done(self, ticket: Ticket) -> None:
        """A node is through (its place goes back), or its request is withdrawn."""
        with self.cond:
            if ticket in self.tickets:
                self.tickets.remove(ticket)
            self._wake()

    def later(self, work: Callable[[], None]) -> None:
        """Background work, in the order handed over: run when a CPU slot is free and no node waits for it."""
        with self.cond:
            self._later.append(work)
            self._wake()

    def wake(self) -> None:
        """Something the rules read changed (a setting, the queue's order, the cards authorized): decide again now."""
        with self.cond:
            self._wake()

    def forget(self, task: str) -> None:
        """The task ended: it is not told `started` again should its id ever come back (a task held over a restart)."""
        with self.cond:
            self._begun.discard(task)

    def end(self) -> None:
        with self.cond:
            self._ended = True
            self._wake()

    def _wake(self) -> None:
        self._changes += 1
        self.cond.notify_all()

    @contextmanager
    def withholding(self, runtime: str) -> Iterator[None]:
        """No node of `runtime` (an extension) gets a place while this lasts; those holding one keep it. The installer
        switches the extension's environment inside it (the installer's Live, built in server/installs.py live), so no
        node starts in the environment being replaced between its check that none runs there and the switch."""
        with self.cond:
            self._withheld.add(runtime)
        try:
            yield
        finally:
            with self.cond:
                self._withheld.discard(runtime)
                self._wake()

    def now(self) -> list[Ticket]:
        """Every request this moment (a copy), granted or not: what the queue and the cards page show."""
        with self.cond:
            return list(self.tickets)

    # ------------------------------------------------------------------ the dispatcher

    def run(self) -> None:
        """Decide, act on what was decided outside the lock, and wait for the next change (or POLL_S); until end()."""
        while True:
            with self.cond:
                if self._ended:
                    return
                seen = self._changes
            line, held = self.line(), self.held()  # the queue's, asked holding nothing of the pools'
            with self.cond:
                done = self._decide(line, held)
            for ticket in done.granted:
                ticket.woken()
            for task in done.started:
                self.started(task)
            for work in done.background:
                self.spawn(lambda work=work: self._run_later(work))
            if self._make_way(done):
                continue  # models made way: decide again at once
            with self.cond:
                if not self._ended and self._changes == seen:
                    self.cond.wait(POLL_S)

    def _run_later(self, work: Callable[[], None]) -> None:
        """Background work on its own thread of the farm, at the lowest CPU priority (the threads it starts inherit
        it: Linux keeps a niceness per thread), so a node computing beside it gets the cores first."""
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), BACKGROUND_NICE)
        except (AttributeError, OSError):  # a system without per-thread priorities: it runs at the usual one
            pass
        try:
            work()
        finally:
            with self.cond:
                self._background -= 1
                self._wake()

    def _make_way(self, done: _Pass) -> bool:
        """What stands in the way is our own idle kept-loaded models: they make way (seconds each, holding nothing).
        True when any did. A card short of free memory for anyone (placement.pressed_cards: another program, another
        server) gets it back from them too; a card is relieved again only once a reading taken after the last time
        says it still needs it."""
        if done.free_ram is not None:
            need, spare = done.free_ram
            if resident().free_ram(need, spare):
                return True
        snapshot = self.host.snapshot()
        if done.reclaim is not None:
            need, busy = done.reclaim
            cards = placement.reclaimable_cards(need, snapshot, busy, resident().idle_context_mb())
            if cards and resident().clear_idle(cards):
                return True
        pressed = {g: mb for g, mb in placement.pressed_cards(snapshot, done.busy, policy.release_below_gb()).items()
                   if self._relieved.get(g, 0.0) < snapshot.at}
        if pressed and resident().relieve(pressed):
            now = time.time()
            self._relieved.update({g: now for g in pressed})
            return True
        return False

    def _decide(self, line: list[tuple[str, bool]], held_now: bool) -> _Pass:
        """(Holding the lock) one pass over the rules of the module doc: time limits first, then the requests in queue
        order, then background work in what is left."""
        out = _Pass()
        self._expire()
        order = {task: i for i, (task, _) in enumerate(line)}
        new = {task for task, started in line if not started}
        held: dict[str, dict[str, int]] = {}  # task -> kind -> places it holds now
        busy: set[str] = set()  # cards a node of ours is on
        slots = 0  # CPU slots nodes hold
        for t in self.tickets:
            if t.granted:
                held.setdefault(t.task, {GPU: 0, CPU: 0})[t.need.kind] += 1
                busy |= {t.gpu} if t.gpu else set()
                slots += t.need.kind == CPU
        cpus, gpus, most = policy.task_cpus(), policy.task_gpus(), policy.cpu_nodes()
        waiting = sorted((t for t in self.tickets if not t.granted and t.task in order),
                         key=lambda t: (order[t.task], t.asked))
        snapshot, short, now = self.host.snapshot(), None, time.monotonic()
        # the memory free for what is granted now: what the machine has free, less what the nodes granted a moment ago
        # declared and may not have taken yet (RAM_SETTLE_S); each grant below takes its share at once
        free_gb = available_gb() - sum(t.need.ram_gb for t in self.tickets if t.granted and now - t.since < RAM_SETTLE_S)
        total_gb = machine_memory_gb()
        for t in waiting:
            mine = held.setdefault(t.task, {GPU: 0, CPU: 0})
            if t.need.runtime in self._withheld:  # its environment is being switched (a moment): nothing to say
                t.reason = None
                continue
            if t.task in new and (held_now or (paused := self._paused(t.need.kind)) is not None):
                t.reason = None if held_now else paused  # nothing of a task that has not started (held: nothing to say)
                continue
            if total_gb and t.need.ram_gb > total_gb:  # it never fits: it says so, and nothing behind it waits for it
                t.reason = Msg("W-QUEUE-NEVERRAM", node=t.need.label, need=t.need.ram_gb, total=total_gb)
                continue
            if t.need.kind == GPU and mine[GPU] >= gpus:
                t.reason = Msg("N-QUEUE-TASKGPUS", most=gpus)
                continue
            if t.need.kind == CPU and mine[CPU] >= cpus:
                t.reason = Msg("N-QUEUE-TASKCPUS", most=cpus)
                continue
            if short is not None:  # the memory is short for one before it: every node after it waits too
                t.reason = short
                continue
            card = None
            if t.need.kind == GPU:
                card = placement.place(t.need, snapshot, busy)
                if isinstance(card, Msg):
                    t.reason = card
                    if card.code in VRAM_WAITS and out.reclaim is None:
                        out.reclaim = (t.need, set(busy))
                    continue
            elif slots >= most:
                t.reason = Msg("N-QUEUE-CPUBUSY", most=most)
                continue
            if free_gb < (need := keep_free_gb(t.need.ram_gb)):
                t.reason = short = Msg("N-QUEUE-WAITRAM", need=need, free=free_gb)
                out.free_ram = (need, {t.need.runtime})
                continue
            mine[t.need.kind] += 1
            free_gb -= t.need.ram_gb
            if card is not None:
                busy.add(card.uuid)
                self._grant(t, card.uuid, card.short_name, out)
            else:
                slots += 1
                self._grant(t, "", "", out)
        out.busy = set(busy)
        room = most - slots - self._background  # what no node computes on: background work may have it
        while short is None and room > 0 and self._later:
            out.background.append(self._later.popleft())
            self._background += 1
            room -= 1
        return out

    def _grant(self, t: Ticket, gpu: str, name: str, out: _Pass) -> None:
        t.since, t.gpu, t.gpu_name, t.reason = time.monotonic(), gpu, name, None
        out.granted.append(t)
        if t.task not in self._begun:
            self._begun.add(t.task)
            out.started.append(t.task)

    @staticmethod
    def _paused(kind: str) -> Msg | None:
        """Why a task that has not started gets no place of this kind now: the administrator switched 计算任务 (or,
        for a GPU node, 显卡任务) off, or the data disk is below 暂停新计算的剩余空间; None when it may have one."""
        if not policy.compute_enabled():
            return Msg("N-QUEUE-PAUSED")
        if (low := policy.space()).get("low"):  # 暂停新计算的剩余空间: back by itself once the disk has room again
            return Msg("N-QUEUE-DISKLOW", free_pct=low["pct"], pct=low["floor_pct"])
        if kind == GPU and not policy.gpu_enabled():
            return Msg("N-QUEUE-GPUOFF")
        return None

    def _expire(self) -> None:
        """Every granted node past 单节点超时 is stopped: a worker's process is killed, a core node stops at its next
        frame; it fails with this reason (engine/cook.py)."""
        limit, now = policy.node_timeout_s(), time.monotonic()
        for t in self.tickets:
            if t.granted and t.expired is None and now - t.since > limit:
                t.expired = Msg("E-QUEUE-NODETIMEOUT", node=t.need.label, minutes=limit / 60)
                t.stop.set()
