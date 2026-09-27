"""The job queue. Every cook, from every user (web pages, DCC plugins, the command line), runs here.

What a job is comes from the nodes it computes (engine/policy.py), and says which lane it runs in:

- a light job (reading a file, a value, delivering what is cached) runs at once, in the interactive pool: a few at a
  time (立即计算名额, farm/policy.py), no longer than the administrator's 立即计算限时, and not too many from one client
  (PER_CLIENT at once, PER_MINUTE a minute, per account), so a shared link can't keep the server busy;
- a heavy CPU job (视频转序列, rasterizing every frame) waits in the CPU lane: the administrator's CPU 任务数 at a
  time, in the order they came in;
- a GPU job waits for a GPU: each GPU the administrator authorized takes one job at a time, in the order the jobs
  came in.

Every job is its account's (farm/clients.py). A job in any lane starts only when the machine has the memory it keeps
free. Jobs without a GPU node see no GPU (their workers get none). The queue is the same for everyone; a user sees
their own jobs and, of everyone else's, only how many are ahead of theirs and when they should end (anonymous load);
the administrator sees every job with everything known about who started it.

Every job is kept: its record in the database (lab2shot/database), written when it is submitted and updated when it
ends, and the graph it was submitted with as a file in the submitting account's folder
(work/users/<username>/jobs/<job id>.json; the database keeps only that file's path, `job_graph` reads it). The end says, per node type, how its nodes served the job (computed or answered without computing), which the usage
statistics count (usage.py). The time of every node that computes is recorded, and every job is estimated from those
records (timings.py): the queue says when a running job should finish and when a waiting one should start.

Before the server restarts (server/restart.py) the queue is held: no job starts, new ones still come in and wait. The
waiting jobs are kept in the database (held) and queued again, under their own ids, by the server that comes next.
"""

from __future__ import annotations

import asyncio
import bisect
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

from .. import logs, progress
from ..config import settings
from ..data.units import PERCENT
from ..database import Database, db, json_of, json_text
from ..engine import EVALUATIONS, CookCancelled, CookError, Engine, Evaluation, Graph, GraphError
from ..engine.evaluation import Inst
from ..engine.evaluations import content_key
from ..engine.policy import GPU, HEAVY, LANES, LIGHT, CookKind
from ..engine.resident import available_gb, keep_free_gb
from ..engine.resident import pool as resident
from ..errors import Invalid, MessageError, NotFound, TooMany, Unavailable, message_of
from ..io.atomic import write_text
from ..messages import Msg
from ..serving import Account, serving
from ..transfer import deliveries
from . import disk, load, policy, scheduler, streaming, timings, units
from .clients import Client
from .tasks import Tasks

log = logs.get("farm")

FARM_SESSION = "farm"  # the EvaluationCache session for the queue's own internal reads (frame_range, params for
# usage stats, submit's cook-kind check, cache_mark): not a browser session, but a graph asked about here and
# through /api/status is still the same content, so a page checking a job's graph reads what the farm already built


def _eval_for(graph: Graph, account: Account) -> Evaluation:
    """The shared Evaluation of `graph` for `account` (engine/evaluations.py), for the farm's own read-only asks about
    a graph it has no job for (a finished job's cache mark). A running job reads its own (Job.eval): one job, one
    evaluation. Never one that cooks: Engine keeps its own, see its docstring."""
    return EVALUATIONS.get(FARM_SESSION, content_key(graph), lambda: graph, account)

KEEP_FINISHED = 100  # finished jobs the queue still shows
MEMORY_POLL_S = 10.0
# How often to re-check while a GPU is occupied by another program. The queue's own events (submit, finish, cancel)
# wake the waiting thread, but another program releasing the GPU is not a queue event; waiting without a timeout would
# leave the job waiting after the card is free, with the page still showing the old reason. Memory waits are polled
# the same way.
CARD_POLL_S = 10.0
# The placement reasons that mean only video memory is short (farm/scheduler/placement.py reason, step 3)
VRAM_WAITS = frozenset({"N-GPU-WAITVRAM", "N-GPU-WAITVRAMFOREIGN"})
ACTIVE = ("queued", "running")
TIDY_S = 3600.0  # the disk is cleaned by the settings at least this often (and after every job)
# one client's light jobs waiting or running, and submitted in a minute. How many of everyone's run at once
# (立即计算名额) and how many wait for the pool before one more is refused (排队上限) are the administrator's:
# farm/policy.py, the one place the queue's and the scheduler's five tunables are read.
PER_CLIENT = 3
PER_MINUTE = 120
BOUND_S = 1.0  # how often running light jobs are held to 立即计算限时
LANE_WORDS = {LIGHT: "立即计算", HEAVY: "CPU 队列", GPU: "要显卡"}
SPREAD_S = 0.2  # how often a running job looks whether another idle card could take some of its items (farm/units.py)


def _gpu_enabled() -> bool:
    """显卡任务 (queue.gpu_jobs): off is exactly like no GPU being authorized; a GPU job may still be submitted and
    queues, it just never starts until this (or an authorization) is turned back on."""
    return bool(settings()["queue.gpu_jobs"])


def _compute_enabled() -> bool:
    """计算任务 (queue.compute_jobs): off refuses any new job that is not pure light viewing (a GPU node, heavy CPU
    work, or a delivery: 「输出」, even of a cached result) at submission (Farm.submit), and pauses the heavy and GPU
    lanes for whatever queued before it was switched off. Viewing (导入 / 读取序列 / 数值 / a cached result shown) is
    never affected: it is judged by CookKind, not this switch."""
    return bool(settings()["queue.compute_jobs"])


# Waiting for resources is shown to the user as 「排队中」 without an additional message: the user sees only four phases,
# 「排队中」, 「加载模型」, the computation steps and 「取回结果」; waiting for a card is an internal detail of the first.
#
# The single criterion is what the user can do after reading the message:
#   - nothing, and the wait resolves by itself -> no message. Busy cards (N-GPU-WAITBUSY / WAITVRAM / WAITSHARE /
#     WAITSEVERAL), waiting for memory (N-QUEUE-WAITRAM) and jobs ahead (N-QUEUE-BEHIND) belong here; the queue
#     position is already drawn on the queue row and the node (webui/src/graph/nodes.ts waitText, QueueJob.position),
#     and a message would repeat it.
#   - someone must act, otherwise the wait never ends -> the message is shown in full (no unexplained waiting):
#     N-QUEUE-NOMACHINEEVER (architecture mismatch; never starts without reinstalling the extension or changing the
#     card), N-QUEUE-NOCARD (no card is authorized for jobs; an administrator must enable one),
#     N-QUEUE-PAUSED (the administrator paused computation; switching off GPU jobs shows the user this message too).
#     None of these messages mention 「显卡」, "GPU" or "RTX": users never see cards; which card and why is in the
#     administrator's field `waiting_detail` (N-GPU-*, N-QUEUE-GPUOFF).
#
# The messages not shown to the user are not dropped: they remain in `waiting_detail`, where administrators (sessions
# with farm.cards) can see which card is awaited (the `hides` of the two routes in server/farm.py remove this field from
# other replies).
#
# Waits that resolve by themselves and waits that do not must not share a message: a job that never starts because of
# an architecture mismatch, described like a busy queue, would lead the user to wait indefinitely. So self-resolving
# waits get no message, and the others state which case applies.
#
# A generic "waiting for an available machine" (`N-QUEUE-NOMACHINE`) would cover both kinds; it is not in the message
# catalogue and must not be added back.
NEVER_CODES = ("W-GPU-",)  # fixed at install: without reinstalling the extension or changing the card, the job never starts on this server
NOCARD_CODES = ("N-GPU-NOCARD",)  # no card is authorized for jobs: computation can start once an administrator enables one, which requires action
NOMACHINE_EVER = Msg("N-QUEUE-NOMACHINEEVER")
NOMACHINE_NOCARD = Msg("N-QUEUE-NOCARD")

# self-resolving waits (memory, jobs ahead): moved into `waiting_detail`; the user sees only 「排队中」
SELF_CLEARING = ("N-QUEUE-WAITRAM", "N-QUEUE-BEHIND")


def _told(detail: Msg | None) -> Msg | None:
    """What the user is told about a GPU job waiting for a card, or None when no action is needed: the cards are just
    busy, so it starts by itself and 「排队中」 already says everything (see the block above). `detail` is the card
    reason (placement.reason), which only a session with farm.cards ever sees."""
    code = detail.code if detail is not None else ""
    if code.startswith(NEVER_CODES):
        return NOMACHINE_EVER
    return NOMACHINE_NOCARD if code.startswith(NOCARD_CODES) else None

EVENT_CAP = 1000  # events a job keeps (Job._prune): a long cook's progress would grow them without end
PASSING = ("progress",)  # the events a later one of the same node replaces (computation progress has this single
# outward event type, lab2shot/progress.py: stage / progress / phase are folded into Job.now and sent as one description)

@dataclass(eq=False)
class Job:
    title: str
    graph: Graph
    targets: list[str]
    client: Client
    force: bool = False
    kind: CookKind = CookKind(GPU, False)  # its lane, and whether it delivers (engine/policy.py)
    ram_gb: float = 0.0  # system memory its biggest node needs (NodeDef.ram_gb)
    shown: bool = False  # the viewer asked for it because a node is shown (not a click: not in the user's history)
    graph_id: str = ""  # the graph file's own meta.id: "" for a submitter
    # that sends none (a DCC or the command line, with no "open graph" of its own); a job or delivery without one
    # never matches any open document's id, so the page only ever draws it as belonging to no open graph
    show: frozenset[str] | None = None  # the outputs of the node the viewer shows (JobRequest.show; None: every one of
    # them), passed to Engine.cook as `show` when this job cooks that one node: the others it only computes what is
    # wired on from them (engine/cook.py CookContext.wanted)
    version: int | None = None  # the page's cook-inputs version it was submitted at (JobRequest.version): every event carries it
    # back, so a page tells whether a result still answers what it shows now (None: a submitter that sends none)
    # the one read-only Evaluation of this job: everything the queue asks about the graph (its frame
    # range, the parameters the usage record keeps, the partial packets a page follows, the folders a stopped job
    # leaves) is asked of this one object, built once when the job was admitted, for the account that submitted it
    # (Evaluation.account: another account's upload is not there). The cook builds its own on top of the same graph
    # (Engine, which mutates one as it cooks and shares it with nobody).
    eval: Evaluation = None  # type: ignore[assignment]  # Farm.submit hands over the one it checked the job with;
    # without one (a test building a Job directly) the job builds its own for its account, below
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    state: str = "queued"  # queued / running / done / failed / cancelled
    gpu: str | None = None  # UUID of the GPU it runs on ("" none)
    gpu_name: str = ""
    looked: float = 0.0  # when its driver last looked for another card for its items (_spread)
    cards: set[str] = field(default_factory=set)  # other authorized GPUs it borrowed for its 计算单元 (farm/units.py):
    # a card is borrowed only while no waiting job would be placed there, and given back the moment its items are done
    units: dict = field(default_factory=dict)  # 「3/12 条」: {"done", "items", "running"} (Ledger.progress; {} no block)
    # view proxies: one per computed packet, made in the CPU thread pool (`_proxies_of`); the job ends only once they are done
    proxies: list[Future] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    next_n: int = 0  # the number the next event gets: events are numbered for good (poll since, the stream's ids)
    outputs: list[dict] = field(default_factory=list)  # what its 「输出」 delivered: {node, label, run, name, mode, files, bytes}
    error: str | None = None
    error_log: str | None = None  # the failed node's log (a worker's output): feedback attaches it
    reason: str = ""  # why it was cancelled, when not by the one who started it
    # computation progress, a single description (lab2shot/progress.py) used by the queue and the node alike, and the
    # only one the server sends: the phase (排队中 / 加载模型 / 计算 / 取回结果), the node, the current step's name and
    # its count (as text only), plus two internal timestamps (since / stage_since, used for remaining).
    # The outward copy is computed by progress_json() on demand (`at` uses the current clock).
    now: dict = field(default_factory=lambda: dict(progress.BLANK))
    # the progress bar's denominator, fixed when the job starts (fix_budget: computed once after the GPU is assigned
    # and estimate has run): the sum of the predicted seconds, from history, of every node instance the job computes.
    # None: some instance has no estimate (computed for the first time), and no scale is drawn.
    budget: float | None = None
    spent: float = 0.0  # predicted seconds of the instances already computed; it only increases, so the bar never shrinks
    share: dict[str, list[float]] = field(default_factory=dict)  # node id -> predicted seconds of each of its instances, popped as each finishes
    # why it waits although it is first in the queue: `waiting` what everyone is told (only the reasons that need
    # somebody to act) (N-QUEUE-NOMACHINEEVER / NOCARD / GPUOFF / PAUSED; `_told` and the block above it say
    # why the self-clearing ones are not told), `waiting_detail` the card reason (placement.reason), which only a
    # session with farm.cards gets (server/available.py FIELDS)
    waiting: Msg | None = None
    waiting_detail: Msg | None = None
    works: list[timings.Work] = field(default_factory=list)  # the nodes it computes, as their estimates need them
    left: dict[str, float | None] = field(default_factory=dict)  # nodes still to compute -> estimated seconds (None: no record)
    usage: dict[str, dict] = field(default_factory=dict)  # node type -> how its nodes served the job (_serve)
    served: set[str] = field(default_factory=set)  # the nodes counted in `usage`
    stop: threading.Event = field(default_factory=threading.Event)
    db: Database | None = field(default=None, repr=False)  # where its records go: its farm's database
    # a job the farm runs for a subsystem of its own: `priority` orders it
    # in its lane after every waiting job of a lower number (users' cooks are 0, so they always start first; a running
    # job is never interrupted); `then(job, cooked)` runs after every target cooked ({target: its packets}) and may fail
    # the job (a MessageError says why); `ended(job, interrupted)` is told how it ended, whatever the way (interrupted:
    # the server is going away; such a job is not kept over a restart, the subsystem queues it again). Neither is kept in
    # the job's record.
    priority: int = 0
    # 拖拽插队 (Farm.reorder): the administrator dragged this lane into an order of their own, and this is this job's
    # place in it. None: nobody has dragged it, and it takes the place 公平排队 works out (_line). A job that was dragged
    # keeps its place until it starts; jobs that come in afterwards queue behind the dragged ones, by the usual rule.
    order: float | None = None
    then: Callable[[Job, dict], None] | None = field(default=None, repr=False)
    ended: Callable[[Job, bool], None] | None = field(default=None, repr=False)
    cond: threading.Condition = field(default_factory=threading.Condition)
    _frames: list[int] | None | Literal[False] = field(default=False, repr=False, compare=False)  # cache: False = not worked out yet
    _wakers: set[Callable[[], None]] = field(default_factory=set, repr=False, compare=False)  # event streams awaiting `changed`

    def __post_init__(self) -> None:
        if self.eval is None and self.graph is not None and self.client is not None:
            # one job, one evaluation, read-only and for the account that submitted it
            self.eval = Evaluation(self.graph, Account(self.client.user))

    @property
    def done(self) -> bool:
        return self.state not in ACTIVE

    def notify(self) -> None:
        """Wake everyone following the job (called holding `cond`): threads waiting on `cond`, and the event streams
        awaiting `changed` on the server's event loop."""
        self.cond.notify_all()
        for wake in list(self._wakers):
            wake()

    async def changed(self, seen: int, timeout: float) -> bool:
        """Wait on the event loop, holding no thread (an open event stream costs no thread of the server),
        until the job has more than `seen` events or is woken (notify); False when `timeout` seconds passed first."""
        loop = asyncio.get_running_loop()
        woken = asyncio.Event()

        def wake() -> None:
            try:
                loop.call_soon_threadsafe(woken.set)
            except RuntimeError:  # that loop is closed: nobody waits there any more
                pass

        with self.cond:
            if self.next_n > seen:
                return True
            self._wakers.add(wake)
        try:
            await asyncio.wait_for(woken.wait(), timeout)
            return True
        except TimeoutError:
            return False
        finally:
            with self.cond:
                self._wakers.discard(wake)

    @property
    def labels(self) -> list[str]:
        return [self.graph.nodes[t].label for t in self.targets]

    @property
    def frames(self) -> list[int] | None:
        """The frame range it cooks: the graph's narrowed range, or (worked out once, from its own inputs, and kept:
        the graph and targets never change after submission) the full range its inputs cover."""
        if self.graph.frames:
            return list(self.graph.frames)
        if self._frames is False:
            full = self.eval.frame_range(self.targets)
            self._frames = list(full) if full else None
        return self._frames

    def emit(self, event: dict) -> None:
        kind = event.get("type")
        t = time.time()
        timed = None  # a node's time to keep: written once the job's lock is let go
        with self.cond:
            if kind == "output":
                self.outputs.append({**{k: event[k] for k in ("node", "label", "run", "name", "mode", "files", "bytes")}, "graph": self.graph_id})
            elif kind == "error":
                self.error, self.error_log = event["text"], event.get("log")
            if kind in ("message", "error") and event.get("level") in ("E", "W"):  # the server's log carries the code
                label = self.graph.nodes[event["node"]].label if event.get("node") in self.graph.nodes else ""
                logs.say(log, event, about=f"{self.id}「{label}」" if label else self.id)
            elif kind == "node_done":
                self.left.pop(event["node"], None)
                if event["node"] not in self.served:  # a node several targets share serves the job once
                    self.served.add(event["node"])
                    timed = self._serve(event)
            # computation progress: stage / progress / phase events only update this one description
            # (lab2shot/progress.py) and are not forwarded as they are; the folded description is sent, and the queue
            # and the node read the same thing
            out = [event]
            if kind in ("stage", "progress", "phase"):
                self._advance(kind, event, t)
                out = [{"type": "progress", **self.progress_json(t)}]  # the three kinds appear on the wire as a single "progress"
            elif kind in ("node_start", "node_done"):
                self._advance(kind, event, t)
                out = [event, {"type": "progress", **self.progress_json(t)}]  # the node changed: the bar and its text follow
            # every streamed event carries the job's graph id and the page's
            # cook-inputs version it was submitted at: the page only ever draws a delivery on a node of the graph that
            # is open now, and takes a result as still true only at the version it asked with (an event's own "graph"
            # or "version", when it sets one, wins; none does today)
            for one in out:
                self.events.append({"graph": self.graph_id, "version": self.version, **one, "t": round(t, 2), "n": self.next_n})
                self.next_n += 1
            if len(self.events) > EVENT_CAP:
                self._prune()
            self.notify()
        if timed is not None:
            try:
                timings.record(*timed, self.db)
            except OSError as exc:  # the job goes on without the record
                logs.say(log, Msg("W-QUEUE-TIMINGNOTKEPT", job=self.id, node=event["node"], why=str(exc)))

    # ------------------------------------------------------------------ computation progress (rationale in lab2shot/progress.py)

    def _advance(self, kind: str, event: dict, t: float) -> None:
        """(Holding the job's lock) fold an event into `now`; this is the only place where progress advances.

        `node_start` enters the 「计算」 phase (core nodes have no 「加载模型」 phase, which exists only for workers and is
        announced by the `phase` event engine/external.py sends before starting the process); the first `progress`
        means there is something to count; `node_done` adds the instance's share to `spent` (which only increases), so
        the bar never moves backwards."""
        if kind == "node_start":
            self.now = {**progress.BLANK, "phase": progress.COMPUTING, "node": event["node"], "label": event["label"],
                        "since": t, "stage_since": t}
        elif kind == "phase":  # engine/external.py: before starting the worker process / after the worker wrote result.json
            self.now = {**self.now, "phase": event["name"]}
        elif kind == "stage":  # a step named by the worker (「检测人物」): only the accompanying text, the bar does not move
            self.now = {**self.now, "note": event["name"], "done": 0, "total": 0, "stage_since": t}
        elif kind == "progress":
            phase = progress.COMPUTING if self.now.get("phase") == progress.LOADING else self.now.get("phase")
            self.now = {**self.now, "phase": phase, "done": event["done"], "total": event["total"]}
        elif kind == "node_done":
            left = self.share.get(event["node"])
            self.spent += left.pop(0) if left else 0.0  # unplanned instances (already cached) count in neither the denominator nor the numerator
            # at this moment no node is computing (the next node_start follows immediately), so `node` is cleared:
            # the bar advances by `spent`, and this progress event no longer touches the finished node
            # (its row now shows 「用时 N 秒」, which would otherwise be overwritten)
            self.now = {**progress.BLANK, "phase": progress.COMPUTING, "since": t, "stage_since": t}

    def fix_budget(self) -> None:
        """Fix the progress bar's denominator when the job starts: a changing denominator would make the bar grow and
        shrink repeatedly and hide the real progress.

        Called once by `_start` after the GPU is assigned and `estimate()` has run: the predicted seconds, from history,
        of every node instance the job computes (the `Work` entries of timings.planned, one per instance inside
        逐项处理 blocks) are summed into the denominator, which then never changes, so the bar cannot shrink.
        If any instance has no estimate (the node is computed for the first time, without history), the whole budget
        is discarded (None) and the page draws an indeterminate bar rather than a fabricated fraction."""
        history = timings.records(self.db)
        models = [self.gpu_name] if self.gpu_name else []
        with self.cond:
            each = [timings.predict(w, models, history)["seconds"] for w in self.works]
            if not each or any(s is None for s in each):
                self.budget, self.share = None, {}
                return
            self.budget, self.share = float(sum(each)), {}
            for work, seconds in zip(self.works, each):
                self.share.setdefault(work.node, []).append(float(seconds))

    def progress_json(self, t: float) -> dict:
        """(Holding the job's lock) the computation progress, the same data for the queue panel and the node:
        `view()` puts it into `now` as is, and `emit()` sends it on the event stream with `"type": "progress"`."""
        if self.state == "queued":
            return {**progress.BLANK, "phase": progress.QUEUED}
        now = self.now
        running = self.share.get(now.get("node", ""))
        at = progress.fraction(self.spent, t - now.get("since", t), running[0] if running else None, self.budget)
        return {**{k: now.get(k, progress.BLANK[k]) for k in progress.PUBLIC}, "at": at}

    def _prune(self) -> None:
        """Keep a job's events bounded: a node's progress events that a later one of the same node
        replaces go, oldest first, until three quarters of EVENT_CAP are left; everything else (what started, finished,
        was said, delivered or failed) stays. Numbers never change: a page asking from its last number gets what came
        after it, the latest progress included. (`stage` is folded into the `progress` description; there is no
        `stage` event on the wire, see PASSING.)"""
        latest = {(e["type"], e.get("node")): i for i, e in enumerate(self.events) if e.get("type") in PASSING}
        drop = len(self.events) - EVENT_CAP * 3 // 4
        kept = []
        for i, e in enumerate(self.events):
            if drop > 0 and e.get("type") in PASSING and latest[(e["type"], e.get("node"))] != i:
                drop -= 1
                continue
            kept.append(e)
        self.events = kept

    def after(self, n: int) -> list[dict]:
        """The events numbered `n` and later (the ones a page that has everything before `n` still needs)."""
        with self.cond:
            return self.events[bisect.bisect_left(self.events, n, key=lambda e: e["n"]):]

    def _serve(self, done: dict) -> tuple | None:
        """(Holding the job's lock) count how a node served the job, per node type: a run (it computed: its seconds, the
        part of them on a GPU, its frames) or a reuse (answered without computing: a cached result, or a worker's raw
        results from an earlier run, whose time says nothing of the work). A run returns what timings.record keeps of
        it (for estimates), for the caller to write once the lock is let go."""
        nid = done["node"]
        node_type = self.graph.nodes[nid].type
        use = self.usage.setdefault(node_type.id, {"runs": 0, "reuses": 0, "seconds": 0.0, "gpu_seconds": 0.0, "frames": 0})
        if done["cached"] or done["reused"] or done.get("nothing"):  # nothing computed: no time to learn from
            use["reuses"] += 1
            return None
        params, gpu = self.eval.params(nid), self.graph.resolved(nid).cost.gpu
        use["runs"] += 1
        use["frames"] += done["frames"]
        use["seconds"] = round(use["seconds"] + done["seconds"], 1)
        if timings.device(gpu, self.gpu_name) != timings.CPU:
            use["gpu_seconds"] = round(use["gpu_seconds"] + done["seconds"], 1)
        return node_type, params, gpu, self.gpu_name, done

    def estimate(self, models: list[str]) -> None:
        """Estimate its nodes on these GPU models: the ones that take jobs when it is submitted, the one it got when
        it starts."""
        history = timings.records(self.db)
        with self.cond:
            self.left = {w.node: timings.predict(w, models, history)["seconds"] for w in self.works}

    def remaining(self, t: float) -> tuple[float, bool] | None:
        """(seconds still to compute, whether that is only a lower bound), None when nothing can be said. The node
        running takes the longer of what its estimate leaves and what the progress of its current stage says that
        stage still needs; a node without records, or running past its estimate with no progress to go by, makes the
        rest a lower bound."""
        with self.cond:
            now, left = dict(self.now), dict(self.left)
        seconds, partial = 0.0, False
        for nid, est in left.items():
            if nid != now.get("node"):
                seconds += est or 0.0
                partial = partial or est is None
                continue
            guesses = [] if est is None else [est - (t - now["since"])]
            done, total = now.get("done", 0), now.get("total", 0)
            if done and total:
                guesses.append((t - now["stage_since"]) * (total - done) / done)
            if guesses and max(guesses) > 0:
                seconds += max(guesses)
            else:
                partial = True
        return None if partial and not seconds else (seconds, partial)

    def wait_for(self, said: Msg | None, detail: Msg | None = None) -> bool:
        """Why it waits now (`said` for everyone, `detail` for whoever may see the cards; None, None: it does not);
        True when that changed (the queue then tells the waiting jobs again).

        Self-resolving waits (SELF_CLEARING: cards, memory, jobs ahead) are moved into `detail`, leaving the user only
        「排队中」; administrators still see them, as the field's visibility is set by the route's `hides` according to
        farm.cards (server/farm.py)."""
        if said is not None and said.code in SELF_CLEARING:
            said, detail = None, detail or said
        new = (said.json() if said else None, detail.json() if detail else None)
        old = (self.waiting.json() if self.waiting else None, self.waiting_detail.json() if self.waiting_detail else None)
        self.waiting, self.waiting_detail = said, detail
        return new != old

    def waiting_json(self) -> dict:
        """The two reasons as the answers carry them (the detail is taken out for whoever may not see cards)."""
        queued = self.state == "queued"
        return {"waiting": self.waiting.json() if queued and self.waiting else None,
                "waiting_detail": self.waiting_detail.json() if queued and self.waiting_detail else None}

    def poll(self, since: int = 0) -> dict:
        with self.cond:
            return {"state": self.state, "done": self.done, "error": self.error, "outputs": list(self.outputs),
                    "events": self.after(since), "next": self.next_n}

    def view(self, viewer: int | None, admin: bool, position: int | None, eta: dict | None = None) -> dict | None:
        """`eta`: when it should finish (running) or start (waiting), {"at": time, "partial": at the earliest}. A job
        is shown whole to its account (`viewer`) and the administrator; to anyone else only while it waits or runs, as
        anonymous load: its lane, its place and when it should end (no id, no account, nothing of what it cooks).
        None: nothing of it for this viewer (someone else's finished job)."""
        mine = viewer is not None and viewer == self.client.user
        running = self.state == "running"
        base = {"state": self.state, "position": position, "eta": eta, "lane": self.kind.lane, "gpu_name": self.gpu_name,
                "submitted": self.submitted, "started": self.started, "finished": self.finished,
                "stopping": self.stop.is_set() and running, **self.waiting_json()}
        if not (mine or admin):
            return None if self.done else {**base, "id": "", "anonymous": True, "mine": False}
        return {
            **base, "id": self.id, "anonymous": False, "mine": mine, "shown": self.shown,
            "client": self.client.full() if admin else {"who": self.client.who, "app": self.client.app},
            "title": self.title, "graph": self.graph_id, "targets": self.labels, "nodes": self.targets, "frames": self.frames, "error": self.error,
            # computation progress: the same data as sent on the event stream (progress_json, lab2shot/progress.py);
            # the queue panel and the node draw the same thing, and the server does not send two versions
            "reason": self.reason, "now": self.progress_json(time.time()) if running else {},
            "units": self.units,  # 「3/12 条」 (farm/units.py Ledger.progress; {} for a job without a block)
            "cards": len(self.cards) + 1 if self.cards else 0,  # cards it is on at once (0: the one lane card)
            "outputs": [_delivery_state(o) for o in self.outputs],
        }

    def record(self) -> dict:
        """What the job log keeps."""
        return {"id": self.id, "title": self.title, "graph": self.graph_id, "targets": self.labels, "state": self.state, "frames": self.frames,
                "lane": self.kind.lane, "submitted": self.submitted, "started": self.started, "finished": self.finished, "gpu": self.gpu,
                "gpu_name": self.gpu_name, "error": self.error, "error_log": self.error_log, "reason": self.reason, "outputs": self.outputs,
                "client": self.client.full(), "usage": self.usage, "shown": self.shown, "nodes": self.targets, "version": self.version}


def _percent(part: float, whole: float) -> int:
    """`part` of `whole` as a whole percentage (0 when there is no whole): what the load strip shows."""
    return round(PERCENT * part / whole) if whole else 0


def _cards_of(running: list[Job]) -> dict[str, list[Job]]:
    """Which jobs are on each card right now, by its UUID: a job's own lane card and the ones it borrowed for its
    items (farm/units.py). The single answer to 「这张卡在忙吗」: the queue view, the load strip and the placement rule
    (_busy_uuids) all mean the same thing by it."""
    on: dict[str, list[Job]] = {}
    for job in running:
        for uuid_ in {job.gpu} | job.cards:
            if uuid_:
                on.setdefault(uuid_, []).append(job)
    return on


def _delivery_state(output: dict) -> dict:
    """A delivery as the queue shows it: what it is, and what became of it (deliveries.state; gone: expired)."""
    try:
        r = deliveries.record(output["run"], output["node"])
    except NotFound:
        return {**output, "state": "expired", "expires": None}
    return {**output, "state": r["state"], "expires": r["expires"]}


class Farm:
    """The queue of one work folder: its records go to that folder's database, whatever happens to be current when
    a job ends (close() waits for every thread it started)."""

    def __init__(self) -> None:
        self.db = db()
        self._threads: set[threading.Thread] = set()
        self._ended = threading.Event()  # close(): the threads of this queue finish
        self.cond = threading.Condition()
        self.jobs: dict[str, Job] = {}  # in submission order
        # a single removal by disk cleaning and submit placing a job into jobs are mutually exclusive (`removing`):
        # measuring the disk takes minutes, and checking busy() only at the start cannot stop jobs submitted meanwhile.
        # This lock is taken before cond on both sides, in the same order, so there is no deadlock
        self._cleaning = threading.Lock()
        self.lanes: dict[str, threading.Thread] = {}  # GPU UUID -> the thread feeding it
        # this machine's GPUs (farm/scheduler/inventory.py), the farm's own idle kept-loaded models told apart: its own background
        self.host = scheduler.LocalHost(reclaimable=lambda: resident().idle_vram_mb())
        self.authorized = self.host.authorized_uuids()  # thread, never the queue's lock, refreshes it
        self.holding = False  # a restart is coming: no job starts
        self.closed = False  # the waiting jobs are parked for the next server: none comes in any more
        self._recent: dict[int, deque[float]] = {}  # account -> when it submitted its light jobs of the last minute
        self._marks: dict[str, tuple[float, dict]] = {}  # job id -> (when, its cache mark): cache_mark()
        # view proxies are made here, not in the engine: when a packet is computed (node_done) it is handed to these
        # two CPU threads, and the GPU thread continues with the next node without waiting inside the fingerprint lock;
        # the job still ends only once the proxies are done
        self._proxy_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="farm-proxy")
        self.tasks = Tasks(self)  # the work that is not a node graph (farm/tasks.py): on this queue's threads
        disk.sweep_incomplete()  # a process killed mid-write left packets without .complete: they are not results
        self._sync_lanes()
        for lane in (LIGHT, HEAVY):
            self._spawn(self._pool, lane, name=f"farm-{lane}", forever=True)
        self._unpark()
        self._spawn(self._tidy_forever, name="farm-tidy", forever=True)
        self._spawn(self._bound_forever, name="farm-bound", forever=True)

    # ------------------------------------------------------------------ GPUs

    def models(self) -> list[str]:
        """The GPU models that take jobs (what a job's estimate assumes until it gets one)."""
        return list(dict.fromkeys(g.short_name for g in self.host.snapshot().authorized()))

    def authorize(self, uuids: list[str]) -> None:
        """The administrator's choice of GPUs. A GPU taken away finishes its job, then takes no more."""
        self.host.authorize(uuids)
        logs.say(log, Msg("I-QUEUE-GPUSAUTHORIZED", gpus="、".join(uuids)) if uuids else Msg("I-QUEUE-NOGPUAUTHORIZED"))
        with self.cond:
            self.authorized = self.host.authorized_uuids()
            self._sync_lanes()
            self.cond.notify_all()
        resident().end_off(self.authorized)

    def _sync_lanes(self) -> None:
        for gpu in self.authorized:
            if gpu not in self.lanes:
                self.lanes[gpu] = self._spawn(self._lane, gpu, name=f"farm-{gpu}", forever=True)

    def _lane(self, gpu: str) -> None:
        """A GPU taking jobs: one at a time, run on this thread. Paused (queue.gpu_jobs, queue.compute_jobs off)
        exactly like the memory wait below: the job stays queued, saying why. Which job (if any) this lane should
        take next is farm/scheduler's alone (architecture, VRAM, best-fit, ageing): a job scheduler.place() sends
        to a different card is skipped here (a later job may be this one's fit) and never started to fail right
        away (a torch build that does not support the card fails with "no kernel image is available", easily
        mistaken for running out of memory)."""
        def gpu_reason() -> tuple[Msg | None, Msg | None]:
            if not _compute_enabled():
                return Msg("N-QUEUE-PAUSED"), None
            # the administrator switched off GPU jobs: the user must still learn that the job was paused and will not
            # start by itself (the user has to contact an administrator), but the message must not mention 「显卡」
            # since users never see cards; the user reads the same N-QUEUE-PAUSED as when compute jobs are switched off
            # (「管理员重新打开后接着算」), and which switch was turned off is in the administrator's field
            # (N-QUEUE-GPUOFF). It must not say the job starts automatically when a machine is available: it never
            # starts by itself.
            return (Msg("N-QUEUE-PAUSED"), Msg("N-QUEUE-GPUOFF")) if not _gpu_enabled() else (None, None)

        while True:
            with self.cond:
                job = self._next(GPU, lambda: gpu in self.authorized, lambda: _compute_enabled() and _gpu_enabled(), gpu_reason,
                                 fits=lambda j: self._placement(j, gpu) is not None)
                if job is None:
                    del self.lanes[gpu]
                    return
                found = self.host.snapshot().get(gpu)
                name = found.short_name if found else gpu
                self._claim(job, gpu, name)
            if not self._start(job, gpu, name):  # its estimate broke: the job fails, the lane goes on
                self._finish(job, "failed")
                continue
            self._run(job)

    def _busy_uuids(self) -> set[str]:
        """(Holding the queue's lock) authorized GPUs a farm job is already running on: the lane asking is
        always idle by construction (a `_lane` thread only calls `_next` between jobs), so it never needs excluding."""
        return set(_cards_of(self.running()))

    def _ahead(self, job: Job, held: dict[int, int]) -> list[scheduler.Ahead]:
        """(Holding the queue's lock) the queued GPU jobs whose claim to a card comes before `job`'s, how long each
        has waited (scheduler.place()'s ageing rule) and whose account it is with how many cards that account holds
        (单账号占卡). "Before" is the queue's own order (_line: priority, then the administrator's own order, then the
        accounts take turns, then when they came in), the one order for both, so what a waiting job is told about its
        place is what actually happens. A job that is not in the line at all (it is running, and asking
        for another card for its items) is behind every waiting one."""
        now = time.time()
        line = self._line(GPU)
        ahead = line[:line.index(job)] if job in line else line
        return [scheduler.Ahead(scheduler.requirement_for(j), now - j.submitted, j.client.user,
                                held.get(j.client.user, 0)) for j in ahead]

    def _held_by(self) -> dict[int, int]:
        """(Holding the queue's lock) how many authorized GPUs each account has a farm job on right now: the one
        fact 单账号占卡 reads, for this job's account and for everyone waiting (the rule itself is the scheduler's,
        farm/scheduler/placement.py). A card a job borrowed for its items counts too: it is held either way."""
        counts: dict[int, int] = {}
        for jobs in _cards_of(self.running()).values():
            for account in {j.client.user for j in jobs}:
                counts[account] = counts.get(account, 0) + 1
        return counts

    def _placement(self, job: Job, gpu: str) -> scheduler.Placement | None:
        """(Holding the queue's lock) where scheduler.place() would put `job` right now, when that is `gpu`; None
        otherwise (a different card fits it better, or nothing does); a lane only ever takes a job placed on it."""
        req = scheduler.requirement_for(job)
        if not req.needs_gpu:
            return None  # a lane's job list is already filtered to GPU-lane jobs, but never place() a non-GPU one
        placed = self._place(job, req)
        return placed if isinstance(placed, scheduler.Placement) and placed.gpu.uuid == gpu else None

    def _place(self, job: Job, req: scheduler.Requirement) -> scheduler.Placement | scheduler.Wait:
        """(Holding the queue's lock) the one call into the scheduler: the queue hands it the facts (the live
        inventory, which cards are busy, who is waiting and for how long, whose account this job is and how many
        cards that account holds) and the scheduler alone decides. No rule about who may take a card is written
        here."""
        held = self._held_by()
        return scheduler.place(req, self.host.snapshot(), self._busy_uuids(), self._ahead(job, held),
                               job.client.user, held.get(job.client.user, 0))

    def _gpu_reason(self, job: Job, req: scheduler.Requirement | None = None) -> Msg | None:
        """Why `job` cannot start on any authorized GPU right now (no card, architecture, busy cards or memory, not a
        paused switch): the administrator's detail; None when some authorized GPU can take it now. `req`: the job's
        requirement when the caller already has it."""
        req = req or scheduler.requirement_for(job)
        if not req.needs_gpu:
            return None
        placed = self._place(job, req)
        return placed.reason if isinstance(placed, scheduler.Wait) else None

    # ------------------------------------------------------------------ the lanes without a GPU

    def limit(self, lane: str) -> int:
        """How many jobs of a lane without a GPU run at once: the interactive pool's, the administrator's CPU lane."""
        return policy.interactive_pool() if lane == LIGHT else int(settings()["queue.cpu_jobs"])

    def _pool(self, lane: str) -> None:
        """The interactive pool (LIGHT: never paused, viewing always works) or the CPU lane (HEAVY: paused by
        queue.compute_jobs): up to its limit of its jobs run at once, each on a thread of its own, in the order they
        came in; their workers see no GPU."""
        paused = (lambda: not _compute_enabled()) if lane == HEAVY else (lambda: False)
        while True:
            with self.cond:
                job = self._next(lane, lambda: True, lambda: not paused() and self._running(lane) < self.limit(lane),
                                 lambda: (Msg("N-QUEUE-PAUSED"), None) if paused() else (None, None))
                if job is None:
                    return
                self._claim(job, "", "")
            if not self._start(job, "", ""):  # its estimate broke: the job fails, the lane goes on
                self._finish(job, "failed")
                continue
            self._spawn(self._run, job, name=f"job-{job.id}")

    def _running(self, lane: str) -> int:
        return sum(j.state == "running" and j.kind.lane == lane for j in self.jobs.values())

    def _next(self, lane: str, stay: Callable[[], bool], free: Callable[[], bool] = lambda: True,
              reason: Callable[[], tuple[Msg | None, Msg | None]] = lambda: (None, None),
              fits: Callable[[Job], bool] = lambda j: True) -> Job | None:
        """(Holding the queue's lock) wait for the next job of a lane that may start: the first that came in and
        `fits` this caller (a GPU lane skips one farm/scheduler.place() sends to a different card, since a later job
        may be this one's fit), once the lane has room (`free`) and the machine the memory it keeps
        free (models kept loaded make way first). None when the lane ends: the queue closes, or `stay` says no (a
        GPU no longer authorized). `reason`: why it waits although it is first in line, while `free` says no (a
        paused lane; "": no need to say, e.g. just no room yet)."""
        while True:
            if not stay() or self._ended.is_set():
                return None
            in_lane = self._line(lane)
            if lane == GPU and in_lane:  # the head of the GPU queue may fit no authorized GPU at all: say so
                head = in_lane[0]
                req = scheduler.requirement_for(head)
                why = self._gpu_reason(head, req)
                if why is not None and why.code in VRAM_WAITS:
                    # the head waits only for video memory: where this server's own idle kept-loaded processes are
                    # what stands in the way (scheduler.reclaimable_cards), they end (engine/resident.py clear_idle),
                    # outside the queue's lock as in free_ram below
                    cards = scheduler.reclaimable_cards(req, self.host.snapshot(),
                                                        self._busy_uuids(), resident().idle_counts())
                    if cards:
                        self.cond.release()
                        try:
                            resident().clear_idle(cards)
                        finally:
                            self.cond.acquire()
                        continue
                if why is not None and head.wait_for(_told(why), why):
                    self._announce_positions()
            job = next((j for j in in_lane if fits(j)), None)
            if job is None or self.holding or not free():
                if job is not None and job.wait_for(*reason()):
                    self._announce_positions()
                # when jobs are queued but none can be placed on this card, the wait depends on machine state (another
                # program holding VRAM), not on queue events, so it must re-check (CARD_POLL_S). An empty queue or a
                # full lane is woken by the queue's own events.
                self.cond.wait(CARD_POLL_S if lane == GPU and in_lane else None)
                continue
            need, free_gb = keep_free_gb(job.ram_gb), available_gb()
            if free_gb < need:
                # freeing memory waits for model processes to exit one by one (seconds each) and must not hold the
                # queue lock, otherwise /api/queue and submit would block meanwhile. While the lock is released another
                # lane may take the job, it may be cancelled, or a switch may be turned off: after reacquiring the
                # lock, any changed precondition restarts the loop (which waits as needed, without spinning)
                self.cond.release()
                try:
                    made_way = resident().free_ram(need, {n.type.runtime for n in job.graph.nodes.values()})
                finally:
                    self.cond.acquire()
                if made_way:
                    free_gb = available_gb()  # models kept loaded made way (the job's own last)
                if (not stay() or self._ended.is_set() or self.holding or not free() or job.state != "queued"
                        or job.stop.is_set() or not fits(job)):
                    continue
            if free_gb >= need:
                job.wait_for(None)
                return job
            if job.wait_for(Msg("N-QUEUE-WAITRAM", need=need, free=free_gb)):
                self._announce_positions()
            self.cond.wait(MEMORY_POLL_S)

    def _bound_forever(self) -> None:
        """A light job runs no longer than the administrator's 立即计算限时: then it is stopped, saying why (what it
        finished stays in the cache: a click goes on from there)."""
        while not self._ended.wait(BOUND_S):
            minutes = float(settings()["queue.interactive_minutes"])
            for job in self.running():
                if job.kind.lane == LIGHT and not job.stop.is_set() and time.time() - (job.started or time.time()) > minutes * 60:
                    self.cancel(job.id, None, Msg("W-QUEUE-TIMELIMIT", minutes=minutes).text)

    # ------------------------------------------------------------------ jobs

    def submit(self, data: dict, graph: Graph, targets: list[str], client: Client, force: bool = False,
               held: dict | None = None, shown: bool = False, account: Account | None = None, priority: int = 0,
               version: int | None = None, show: frozenset[str] | set[str] | None = None,
               then: Callable[[Job, dict], None] | None = None, ended: Callable[[Job, bool], None] | None = None) -> Job:
        """Queue a cook of `targets` of `graph` (read from `data`, which the job log keeps), in the lane what it
        computes says (engine/policy.py). Raises GraphError / CookError when the graph can't be cooked as it is,
        TooMany when a light one comes too often from one account or finds the pool full.
        `held`: a job the server before a restart kept waiting (its id and when it was submitted, park). `shown`:
        the viewer asked for it because a node is shown: refused unless it may start by itself (light, delivering
        nothing). `account`: whose uploads the graph's files are (transfer/uploads.py Account; the submitter's own by
        default); one that is not theirs is not there, so its node fails and the rest cooks as usual.
        `version`: the page's cook-inputs version, carried back on every event (Job.version); `show`: the
        outputs of the shown node the viewer displays (Job.show). `priority`, `then`,
        `ended`: a subsystem's own job (Job); one with a priority above 0 never runs in
        the interactive pool, at least the CPU lane, so it waits behind every user's cook there too."""
        stored = len(json_text(data).encode("utf-8"))
        if stored > GRAPH_MAX_BYTES:  # the job log keeps the graph: a graph of this size is data, not a node graph
            raise GraphError(Msg("E-JOB-GRAPHTOOBIG", mb=stored / 2**20, most=GRAPH_MAX_BYTES >> 20))
        # at submission every content identity of external files is invalidated and recomputed (io/content.py): the
        # status page normally uses the recorded identity (cheap), but a cook must use the files' current content, so
        # material replaced in place with an unchanged modification time is also detected
        from ..io.content import forget_all

        forget_all()
        ev = Evaluation(graph, account or Account(client.user))  # this job's own, read-only: Job.eval
        ev.check_frames(targets)
        # global frame limit: the maximum number of frames per submission, a single value in the admin 「设置」.
        # Enforced here, so the page, DCC plug-ins, scripts and the command line follow the same rule (direct
        # submissions bypassing the page are refused as well). Light cooks triggered by viewing (shown) are exempt,
        # otherwise long shots could not even be viewed
        if not shown and (span := graph.frames or ev.frame_range(targets)) is not None:
            count = span[1] - span[0] + 1
            if count > (most := policy.max_frames()):
                raise Invalid(Msg("B-JOB-TOOMANYFRAMES", frames=count, most=most, first=span[0], last=span[1]))
        if (problem := timings.unplannable(ev, targets)) is not None:
            raise problem  # nothing of this cook can even be planned: refused now, in the words of the first target
        kind = ev.kind(targets, force)
        if priority > 0 and kind.lane == LIGHT:
            kind = CookKind(HEAVY, kind.delivers)
        if shown and not kind.by_itself:
            raise Invalid(Msg("B-QUEUE-SHOWNDELIVERS") if kind.delivers else Msg("B-QUEUE-SHOWNQUEUES", lane=LANE_WORDS[kind.lane]))
        # 计算任务 (queue.compute_jobs) off: refuses anything but pure viewing outright, never queued (held: a job
        # already accepted before a restart is exempt, it is only being registered again). 显卡任务 (queue.gpu_jobs)
        # off is not refused here: a GPU job may still be submitted and waits, like no GPU being authorized (_lane).
        if held is None and not _compute_enabled() and (kind.lane != LIGHT or kind.delivers):
            raise Unavailable(Msg("B-QUEUE-PAUSED"))
        title = data.get("meta", {}).get("name") or "节点图"
        graph_id = str(data.get("meta", {}).get("id") or "")
        job = Job(title, graph, targets, client, force, kind, ev.needs_ram(targets, force), shown, graph_id,
                  version=version, show=None if show is None else frozenset(show), eval=ev,
                  works=timings.planned(ev, targets, force), db=self.db, priority=priority, then=then, ended=ended, **(held or {}))
        job.estimate(self.models())
        if kind.lane == GPU:
            from .scheduler.compat import runtime_record
            from .scheduler.requirements import requirement_for

            for runtime in sorted(requirement_for(job).runtimes):  # probed and kept now, never under the queue's lock
                runtime_record(runtime)
            # said now, not only once a GPU lane thread next loops (they may all be busy running something else for
            # a while): an incompatible-GPU job never looks like silent progress even before it is first in line.
            if (why := self._gpu_reason(job)) is not None:
                job.wait_for(_told(why), why)
        with self.cond:
            if self.closed:
                raise Unavailable(Msg("E-QUEUE-RESTARTING"))
            if not kind.queues and held is None:
                self._admit(client.user, job.submitted)
        if held is None:  # its record before any lane can see it, and never under the queue's lock
            _log_submitted(job, data)
        with self._cleaning, self.cond:  # _cleaning: not between a disk cleaner's check and its removal (`removing`)
            if self.closed:  # the server began restarting while its record was written: it never queued
                if held is None:
                    self.cond.release()
                    try:
                        _log_finished(job, "cancelled", time.time())
                    finally:
                        self.cond.acquire()
                raise Unavailable(Msg("E-QUEUE-RESTARTING"))
            logs.say(log, Msg({(False, False): "I-QUEUE-SUBMITTED", (False, True): "I-QUEUE-SUBMITTEDDELIVERS", (True, False): "I-QUEUE-REQUEUED",
                                   (True, True): "I-QUEUE-REQUEUEDDELIVERS"}[(bool(held), bool(kind.delivers))], job=job.id, title=title,
                                  targets="、".join(job.labels), lane=LANE_WORDS[kind.lane], who=client.who, ip=client.details.get("ip", "")))
            self.jobs[job.id] = job
            self._announce_positions()
            self.cond.notify_all()
        return job

    def _admit(self, client_id: int, t: float) -> None:
        """(Holding the queue's lock) a light job may come in: its account has fewer than PER_CLIENT light jobs waiting
        or running and submitted fewer than PER_MINUTE in the last minute, and the pool has fewer than 排队上限
        (farm/policy.py) waiting; TooMany otherwise."""
        light = [j for j in self.jobs.values() if j.kind.lane == LIGHT and not j.done]
        if sum(j.client.user == client_id for j in light) >= PER_CLIENT:
            raise TooMany(Msg("E-QUEUE-TOOMANYMINE", count=PER_CLIENT))
        if sum(j.state == "queued" for j in light) >= policy.waiting_max():
            raise TooMany(Msg("E-QUEUE-FULL"))
        recent = self._recent.setdefault(client_id, deque())
        while recent and recent[0] < t - 60:
            recent.popleft()
        if len(recent) >= PER_MINUTE:
            raise TooMany(Msg("E-QUEUE-TOOFAST", count=PER_MINUTE))
        recent.append(t)
        for other in [c for c, times in self._recent.items() if not times or times[-1] < t - 60]:
            del self._recent[other]

    def _claim(self, job: Job, gpu: str, name: str) -> None:
        """(Holding the queue's lock) the job is this lane's from this moment: running, so no other lane takes it and
        the lane's count (_running) has it. What follows (`_start`) reads the database and is done with the lock let go."""
        job.state, job.gpu, job.gpu_name, job.started = "running", gpu, name, time.time()

    def _start(self, job: Job, gpu: str, name: str) -> bool:
        """A claimed job begins: its estimate for the card it got and its progress budget (database reads, done
        outside the queue's lock so /api/queue and submit never wait on them), then it is said to have started. False when the
        estimate itself broke (a bug): said on the job, which the lane then fails; a lane thread never dies of it."""
        try:
            if name:
                job.estimate([name])
            job.fix_budget()  # the progress bar's denominator is fixed at this moment and never changes afterwards (lab2shot/progress.py: a
            # changing denominator would shrink the bar)
        except Exception as exc:  # noqa: BLE001 (the job fails, the lane keeps going)
            logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))
            job.emit({"type": "error", "node": None, **Msg("E-FARM-INTERNAL", detail=str(exc) or type(exc).__name__).json()})
            return False
        logs.say(log, Msg("I-QUEUE-JOBSTARTED", job=job.id, where=name or LANE_WORDS[job.kind.lane]))
        job.emit({"type": "started", "gpu": name, "lane": job.kind.lane})
        with self.cond:
            self._announce_positions()
        return True

    def _announce_positions(self) -> None:
        """(Holding the queue's lock) every waiting job told its place and why it waits: one behind others in its lane
        waits for them (N-QUEUE-BEHIND: how many are ahead, nothing about cards); the first in line says what its lane
        found (the card reason for whoever sees cards; the memory; a paused lane: _next).

        The user is shown only the waits that do not resolve by themselves (criteria: `_told` and the block above
        SELF_CLEARING). Busy cards, waiting for memory and jobs ahead resolve by themselves, and the position is already
        drawn in the interface, so nothing is said; for the user the job is 「排队中」. The rule against unexplained
        waiting still holds: the cases requiring action (N-QUEUE-NOMACHINEEVER, N-QUEUE-NOCARD, N-QUEUE-GPUOFF,
        N-QUEUE-PAUSED) are reported in full."""
        for j, position in self._positions().items():
            if position > 1:
                j.wait_for(Msg("N-QUEUE-BEHIND", ahead=position - 1))
            # the job has just moved from behind others to the head of the line: ask again why the lane has not taken
            # it. Both fields must be checked: N-QUEUE-BEHIND is stored in `waiting_detail` (self-resolving waits are not
            # shown to the user, see SELF_CLEARING); checking only `waiting` would never match, and the head job would
            # keep showing "N ahead" instead of its actual reason
            elif "N-QUEUE-BEHIND" in {m.code for m in (j.waiting, j.waiting_detail) if m is not None}:
                why = self._gpu_reason(j) if j.kind.lane == GPU else None
                j.wait_for(_told(why) if why is not None else None, why)
            j.emit({"type": "queued", "position": position, "lane": j.kind.lane, "gpus": len(self.authorized), **j.waiting_json()})

    def _waiting(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.state == "queued" and not j.stop.is_set()]  # stopped: being withdrawn

    def _line(self, lane: str) -> list[Job]:
        """(Holding the queue's lock) the waiting jobs of a lane in the order they start: by priority (users' cooks
        first), then the administrator's own order where there is one (Job.order, 拖拽插队), then accounts take turns
        (公平排队: the account whose last started job is the oldest is next; an account submitting many waits behind
        the others, one of its jobs at a time; an account that never started anything goes first), then as they came
        in. The order shown is this one too (_positions), so what a waiting job is told matches when it really
        starts."""
        last: dict[int, float] = {}
        for j in self.jobs.values():
            if j.started is not None:
                last[j.client.user] = max(last.get(j.client.user, 0.0), j.started)
        return sorted((j for j in self._waiting() if j.kind.lane == lane),
                      key=lambda j: (j.priority, j.order is None, j.order or 0.0,
                                     last.get(j.client.user, 0.0), j.submitted))

    def reorder(self, job_id: str, position: int) -> tuple[str, int, int]:
        """拖拽插队: put a waiting job at `position` (1-based) of its own lane, and keep that lane in the order the
        administrator dragged it into. Returns (the lane, where it was, where it is now) for the record the caller
        writes (server/access.py audit). NotFound when the job is not waiting any more (it started or was cancelled
        while the page was being dragged); Invalid when `position` is outside the lane.

        Every waiting job of that lane gets a place of its own (Job.order), so what the page shows after the drag is
        exactly what the queue then does; jobs submitted afterwards queue behind them, by the usual rule."""
        with self.cond:
            job = self.jobs.get(job_id)
            if job is None or job.state != "queued" or job.stop.is_set():
                raise NotFound(Msg("E-QUEUE-NOTWAITING", job=job_id))
            lane = job.kind.lane
            line = self._line(lane)
            was = line.index(job) + 1
            if not 1 <= position <= len(line):
                raise Invalid(Msg("E-QUEUE-BADPOSITION", position=position, count=len(line)))
            line.remove(job)
            line.insert(position - 1, job)
            for n, j in enumerate(line):
                j.order = float(n)
            self._announce_positions()
            self.cond.notify_all()
        return lane, was, position

    def _positions(self) -> dict[Job, int]:
        """Each waiting job's place in the line of its lane."""
        return {j: n for lane in LANES for n, j in enumerate(self._line(lane), 1)}

    def _run(self, job: Job) -> None:
        state = "done"
        with serving(job.eval.account):  # the whole job reads files as the account that submitted it (uploads.resolve)
            self._run_job(job)

    def _engine(self, job: Job, gpu: str) -> Engine:
        """One card's engine for a job: its own Evaluation (an Engine mutates the one it cooks on), the job's own
        delivery sink and stop event, so every card of a job delivers and stops as one."""
        owner = {"user": job.client.user, "who": job.client.who, "title": job.title, "graph": job.graph_id}
        return Engine(job.graph, gpu=gpu, stop=job.stop, delivery=deliveries.Sink(job.id, owner, job.emit),
                      evaluation=Evaluation(job.graph, job.eval.account),
                      stream_worker=partial(streaming.run, self))  # streaming: a streaming node's worker runs on a farm thread

    def _proxies_of(self, job: Job, event: dict) -> None:
        """A packet has just been computed (`node_done`, not a cache hit): queue its view proxies in the CPU thread pool
        (`lab2shot/view/proxy.py build`, every channel of every frame scaled and compressed at the administrator's tier).
        The proxy threads are not the GPU thread and do not hold the fingerprint lock; the job waits for them at the end
        of `_run_job`. Compression is not reported to the user, so no events are sent. A failure does not fail the job:
        the frame-serving path builds the proxy again on first request."""
        if event.get("type") != "node_done" or event.get("cached") or not event.get("outputs"):
            return
        from ..data.packet import Packet, packet_dir
        from ..view.proxy import build

        def one(fp: str) -> None:
            try:
                build(Packet.load(packet_dir(fp)))
            except Exception as exc:  # noqa: BLE001 (proxies only affect viewing; the result itself is already written)
                logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))

        for fp in event["outputs"].values():
            job.proxies.append(self._proxy_pool.submit(one, fp))

    def _cook_targets(self, job: Job, engine: Engine, emit, mine: units.Units) -> tuple[dict, bool]:
        """One card's pass over the job's targets: what it cooked, and whether anything failed on it. Every card walks
        the same targets; the ledger (farm/units.py) hands each 计算单元 to one of them."""
        cooked, failed = {}, False

        def seen(event: dict) -> None:  # every card's node_done: its packets' proxies go to the CPU pool
            self._proxies_of(job, event)
            emit(event)

        try:
            for target in job.targets:
                try:
                    # `show` is about the one node the viewer shows: it applies when that node is what this job cooks
                    cooked[target] = engine.cook(target, seen, force=job.force, units=mine,
                                                 show=job.show if len(job.targets) == 1 else None)
                except CookError:  # said where it happened (the engine's error event); the other targets still cook
                    failed = True
        finally:
            mine.over()  # the item this card was on is through: 「3/12 条」
        return cooked, failed

    def _helper(self, job: Job, ledger: units.Ledger, gpu: str, emit) -> None:
        """A borrowed card working on the same job: it takes the items the driver has not, and gives the card back the
        moment it is through. Its errors are the failing item's own (said at its node); the job's state is the
        driver's."""
        ledger.joins(gpu)

        def told(event: dict) -> None:  # a target is "done" once, when the driver is through with it, not per card
            if event.get("type") != "done":
                emit(event)

        try:
            with serving(job.eval.account):
                self._cook_targets(job, self._engine(job, gpu), told, units.Units(ledger, gpu))
        except (CookCancelled, MessageError, GraphError, ValueError, OSError) as exc:
            logs.say(log, Msg("I-QUEUE-CARDLEFT", job=job.id, gpu=gpu, why=str(exc)))
        except Exception as exc:  # a bug on a borrowed card never takes the job down: the driver cooks the rest
            logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))
        finally:
            ledger.leaves(gpu)
            with self.cond:
                job.cards.discard(gpu)
                self.cond.notify_all()

    def _spread(self, job: Job, ledger: units.Ledger, engine: Engine, emit) -> None:
        """Put another idle authorized card on this job's items, if there are items nobody has taken and no waiting job
        would be placed on that card (the waiting jobs' order always comes first). Which items the job has
        is the plan's answer (the driver's evaluation), not a guess: they appear as the lists above them cook."""
        if job.kind.lane != GPU or job.stop.is_set() or self.holding:
            return
        now = time.monotonic()  # every event of a long node is a chance to look, but looking costs a plan: not too often
        if now - job.looked < SPREAD_S:
            return
        job.looked = now
        try:
            ledger.note({i.path for i in engine.eval.order(job.targets)[0] if i.path})
        except (GraphError, CookError, OSError, ValueError):
            return
        with job.cond:  # what the queue shows, kept up to date while it runs (an item finishing also says it, below)
            job.units = ledger.progress()
        if not ledger.free_items():
            return
        with self.cond:
            gpu = self._spare_card(job)
            if gpu is None:
                return
            job.cards.add(gpu)
        self._spawn(self._helper, job, ledger, gpu, emit, name=f"job-{job.id}-{gpu[-6:]}")

    def _spare_card(self, job: Job) -> str | None:
        """(Holding the queue's lock) an authorized GPU `job` may borrow for another 计算单元 right now: one the
        scheduler would place it on, that no waiting job would be placed on (its claim comes first, whatever account
        it belongs to). None: nothing to borrow."""
        req = scheduler.requirement_for(job)
        if not req.needs_gpu:
            return None
        placed = self._place(job, req)
        if not isinstance(placed, scheduler.Placement):
            return None
        gpu = placed.gpu.uuid
        if any(self._placement(waiting, gpu) is not None for waiting in self._line(GPU)):
            return None  # a job waiting in line would start there: it goes first
        return gpu

    def _run_job(self, job: Job) -> None:
        """The job on its own card (the driver), which also hands its items to other idle cards (farm/units.py): the
        driver cooks everything outside every block, so it is the one that delivers, and the one whose end is the
        job's end."""
        state = "done"
        ledger = units.Ledger(job.stop)
        ledger.joins(job.gpu or "")
        ledger.owners[()] = job.gpu or ""  # outside every block is the driver's: it delivers, and the nodes after a
        # block need every item
        ledger.on_progress = lambda progress: self._units_progress(job, progress)
        emit = units.watch(ledger, job.emit)
        engine = self._engine(job, job.gpu or "")

        def told(event: dict) -> None:  # while the driver works, look for another card for the items it left
            emit(event)
            self._spread(job, ledger, engine, emit)

        try:
            cooked, failed = self._cook_targets(job, engine, told, units.Units(ledger, job.gpu or ""))
            if failed:
                state = "failed"
            if job.then is not None and state == "done":
                job.then(job, cooked)  # what the job's submitter does with the cooked results (a benchmark item measures)
        except CookCancelled:
            state = "cancelled"
        except (GraphError, ValueError, OSError, MessageError) as exc:
            state = "failed"
            job.emit({"type": "error", "node": None, **message_of(exc).json()})
        except Exception as exc:  # a bug: the job fails, the lane keeps going
            state = "failed"
            logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))
            job.emit({"type": "error", "node": None, **Msg("E-FARM-INTERNAL", detail=str(exc) or type(exc).__name__).json()})
        finally:  # the driver is through: the cards it borrowed stop waiting and are given back before the job ends
            ledger.close()
            self._wait_for_cards(job)
            if not job.stop.is_set():  # the cook ends only once its proxies are done (failures were reported in _proxies_of; this only waits)
                for f in job.proxies:  # a stopped job does not wait: proxies finish in the background, as stopping must end the job at once
                    f.exception()
        self._finish(job, state)

    def _finish(self, job: Job, state: str) -> None:
        """A job ends. Its record goes into the database first (its jobs row and how its nodes served it); only then is
        it said to be finished (its state, its last events), so whoever sees it finished finds its record; then its
        owner is told (job.ended)."""
        finished = time.time()
        try:
            _log_finished(job, state, finished)
            if state != "done":
                self._clean_partial(job)  # the folders it was writing are incomplete: not results, so removed
        except Exception as exc:  # noqa: BLE001 (the record failed; the job still ends, or it would stay `running` for ever and its lane's card never take another)
            logs.say(log, Msg("E-QUEUE-RECORDFAILED", job=job.id), logs.error_text(exc))
        with self.cond:
            job.state, job.finished = state, finished
            if state == "cancelled":
                job.emit({"type": "cancelled", "reason": job.reason})
            job.emit({"type": "finished", "state": state})
            finished = [j for j in self.jobs.values() if j.done]
            for old in finished[:-KEEP_FINISHED]:
                del self.jobs[old.id]
            self._announce_positions()
            self.cond.notify_all()
        if job.ended is not None:
            try:
                job.ended(job, self.holding or self._ended.is_set())
            except Exception as exc:  # its owner's records: never the queue's end
                logs.say(log, Msg("E-QUEUE-ENDEDFAILED", job=job.id), logs.error_text(exc))
        took = round(job.finished - job.started, 1) if job.started else 0.0
        if state == "failed":
            logs.say(log, Msg("W-QUEUE-JOBFAILED", job=job.id, seconds=took, error=job.error))
        elif state == "done":
            logs.say(log, Msg("I-QUEUE-JOBDONE", job=job.id, seconds=took))
        else:
            logs.say(log, Msg("I-QUEUE-JOBCANCELLED", job=job.id, seconds=took, reason=job.reason or "-"))
        self._spawn(self._tidy, name="farm-tidy-now")

    def _units_progress(self, job: Job, progress: dict) -> None:
        """「3/12 条」: said to whoever follows the job, and kept on it for the queue view."""
        with job.cond:
            job.units = progress
        job.emit({"type": "units", **progress})

    def _wait_for_cards(self, job: Job, seconds: float = 300.0) -> None:
        """Every card this job borrowed is through before the job ends (its records, its state and the cards' release
        are one moment, as they have always been for the one card a job had)."""
        end = time.time() + seconds
        with self.cond:
            while job.cards and time.time() < end:
                self.cond.wait(0.2)

    def _clean_partial(self, job: Job) -> None:
        """A job that ended without finishing (cancelled, failed) leaves the packet folders it was writing incomplete:
        they are not results, so they are removed, each under its fingerprint's lock so a
        concurrent identical cook's folder is never touched. Only the instances this job's own targets need
        (Evaluation.order), never every node of the graph: another branch's folder is another cook's business.

        Only the instances this job itself began (its own node_start events): a job that never ran wrote nothing,
        and an instance it never reached is nobody's partial folder of its own. Otherwise cancelling a job that has
        not started would clean every output packet of its plan; when the same graph is submitted twice, or two
        accounts cook the same material, the folder of the other job still cooking would be deleted and its commit
        would raise FileNotFoundError."""
        if job.started is None:
            return
        with job.cond:
            begun = {(e.get("node"), tuple(e.get("path") or ())) for e in job.events if e.get("type") == "node_start"}
        if not begun:
            return
        ev = job.eval
        try:
            order, _behind = ev.order(job.targets)
        except (GraphError, CookError, OSError, ValueError):
            return
        for inst in order:
            if (inst.node, tuple(inst.path)) not in begun:
                continue
            try:
                plan = ev.plan(*inst)
            except (GraphError, CookError, OSError, ValueError):
                continue
            for fp in plan.outputs.values():
                disk.clean_incomplete(fp)

    def cancel(self, job_id: str, user_id: int | None, reason: str = "") -> None:
        """Stop a job: its account (`user_id`), or the administrator (None), who may say why. Someone else's is not
        there for them."""
        job = self.get(job_id)
        if user_id is not None and job.client.user != user_id:
            raise NotFound(Msg("E-JOB-GONE"))
        logs.say(log, Msg("I-QUEUE-CANCELLEDBYADMIN" if user_id is None else "I-QUEUE-CANCELLEDBYOWNER", job=job.id))
        with self.cond:
            job.reason = reason
            withdrawn = job.state == "queued" and not job.stop.is_set()
            if withdrawn or job.state == "running":
                job.stop.set()  # a waiting one no lane takes any more (_waiting); a running one stops at its next check
            if job.state == "running":
                job.emit({"type": "stopping"})
        if withdrawn:  # its record written outside the queue's lock, then it is said cancelled
            self._finish(job, "cancelled")

    def busy(self) -> bool:
        """Whether anything is cooking or waiting to cook."""
        with self.cond:
            return any(not j.done for j in self.jobs.values())

    def wake(self) -> None:
        """The settings changed: jobs waiting for memory, and for room in the CPU lane, and background tasks waiting
        for 计算任务, look again at once."""
        with self.cond:
            self.cond.notify_all()
        self.tasks.wake()

    @contextmanager
    def removing(self, seen: set[str] | None = None) -> Iterator[bool]:
        """One removal by a disk cleaner (farm/disk.py clean, server/quota.py _drop), and whether it may go ahead: the
        queue has nothing to finish, or, with `seen` (the active jobs known when the cleaner worked out what is
        safe), no job has come in since. Held while the removal happens, and `submit` takes the same lock to put a job
        in: a job is either there before the check or comes after the removal, never in between."""
        with self._cleaning:
            active = self.active_ids()
            yield (not active) if seen is None else active <= seen

    def active_ids(self) -> set[str]:
        """The ids of every job still to finish (queued and running)."""
        with self.cond:
            return {j.id for j in self.jobs.values() if not j.done}

    def jobs_now(self) -> list[Job]:
        """A copy of the job table this moment (the lanes change it under the queue's lock: nobody walks it without)."""
        with self.cond:
            return list(self.jobs.values())

    def _tidy(self) -> None:
        """Housekeeping: the disk by the settings, the database's daily backup."""
        try:
            from ..accounts import forget_old_grants

            disk.tidy(idle=not self.busy(), guard=self.removing)
            forget_old_grants()
            self.db.backup_if_due()
        except Exception as exc:  # housekeeping never stops the queue
            logs.say(log, Msg("E-QUEUE-HOUSEKEEPING"), logs.error_text(exc))

    def _tidy_forever(self) -> None:
        while not self._ended.wait(TIDY_S):
            self._tidy()

    def _spawn(self, target, *args, name: str, forever: bool = False) -> threading.Thread:
        """A thread of this queue (close() waits for it). `forever`: a lane or a housekeeping loop, one that must
        outlive any exception (a lane thread that dies silently leaves its lane with nobody taking jobs): the error is logged and the loop is started again after a moment, until the queue ends."""

        def run() -> None:
            try:
                while True:
                    try:
                        target(*args)
                        return
                    except Exception as exc:  # noqa: BLE001 (a bug in a loop of the queue: reported, never fatal to the lane)
                        logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=name), logs.error_text(exc))
                        if not forever or self._ended.wait(1.0):
                            return
            finally:
                with self.cond:
                    self._threads.discard(threading.current_thread())

        t = threading.Thread(target=run, daemon=True, name=name)
        with self.cond:  # started before close() can see it (it joins only started threads)
            self._threads.add(t)
            t.start()
        return t

    def background(self, target, *args, name: str) -> threading.Thread | None:
        """Work of a subsystem that belongs to this queue (a benchmark run queueing its items): on a thread of the
        queue's own, so close() stops and waits for it like the queue's. A queue being closed starts nothing (None):
        the work would outlive its work folder."""
        if self._ended.is_set():
            return None
        return self._spawn(target, *args, name=name)

    @property
    def ending(self) -> bool:
        """close() was called: background work stops at its next step."""
        return self._ended.is_set()

    def close(self, timeout: float = 60.0) -> None:
        """Stop this queue: no job starts, the running ones stop, every thread it started finishes (their records
        written) before this returns. For a work folder going away (tests); a server ends by exiting. A thread still
        running after `timeout` is a bug that would write into whatever database is current next: it raises, never
        returns as if closed."""
        self._ended.set()
        with self.cond:
            self.holding = True
            self.cond.notify_all()
        self.tasks.stop_all()  # background tasks stop at their next step; their threads are joined below with the rest
        self.stop_running(Msg("N-QUEUE-SERVEREND").text)
        end = time.time() + timeout
        while True:
            with self.cond:
                left = [t for t in self._threads if t is not threading.current_thread()]
                self.cond.notify_all()
            if not left or time.time() > end:
                break
            left[0].join(0.1)
        self.host.stop()  # its own background thread (farm/scheduler/inventory.py): no work folder outlives its test
        if left:
            raise RuntimeError(f"queue closed with threads still running after {timeout:.0f} s: {sorted(t.name for t in left)}")

    # ------------------------------------------------------------------ restarts

    def running(self) -> list[Job]:
        with self.cond:
            return [j for j in self.jobs.values() if j.state == "running"]

    def active(self) -> list[Job]:
        """Every job still to finish (queued and running), whoever submitted it: what may not be cleaned away under
        anyone (busy_fingerprints)."""
        with self.cond:
            return [j for j in self.jobs.values() if j.state in ACTIVE]

    def hold(self) -> None:
        """No job starts from now (the server is about to restart): every lane finishes the jobs it runs and takes no
        more; new jobs are still taken in and wait, light ones too."""
        with self.cond:
            self.holding = True
            self.cond.notify_all()
        logs.say(log, Msg("I-QUEUE-HOLDING"))

    def release(self) -> None:
        """The restart was called off: every lane takes jobs again."""
        with self.cond:
            self.holding = False
            self.cond.notify_all()
        logs.say(log, Msg("I-QUEUE-RELEASED"))

    def stop_user(self, user_id: int) -> int:
        """Stop every job of an account that is being deleted; returns how many."""
        with self.cond:
            live = [j for j in self.jobs.values() if not j.done and j.client.user == user_id]
        for job in live:
            self.cancel(job.id, None, Msg("N-QUEUE-ACCOUNTDELETED").text)
        return len(live)

    def stop_running(self, reason: str) -> None:
        """Stop every running job, saying why (they end cancelled, their finished nodes stay in the cache)."""
        for job in self.running():
            self.cancel(job.id, None, reason)

    def park(self) -> int:
        """Keep the waiting jobs for the server that comes after the restart (the database's held table); returns
        how many. Their records are in the database and their graphs in their files already."""
        with self.cond:
            self.closed = True
            waiting = [j for j in self._waiting() if j.ended is None]  # a subsystem's job: that subsystem queues it again
            for j in self.jobs.values():  # their event streams end (the pages reconnect to the next server)
                with j.cond:
                    j.notify()
        with self.db.write() as c:
            c.execute("DELETE FROM held")
            c.executemany("INSERT INTO held (job_id, position, targets, force) VALUES (?, ?, ?, ?)",
                          [(j.id, i, json_text(j.targets), int(j.force)) for i, j in enumerate(waiting)])
        logs.say(log, Msg("I-QUEUE-PARKED", count=len(waiting)))
        return len(waiting)

    def _unpark(self) -> None:
        """Queue again, under their own ids, the jobs the server before a restart kept waiting. One that can't be
        cooked any more (an upload cleaned meanwhile) ends failed, saying why."""
        from .. import accounts

        held = self.db.rows("SELECT h.job_id, h.targets, h.force, j.submitted, j.record, j.user_id FROM held h "
                            "JOIN jobs j ON j.id = h.job_id ORDER BY h.position")
        with self.db.write() as c:
            c.execute("DELETE FROM held")
        for h in held:
            record = json_of(h["record"])
            try:
                account = accounts.get(h["user_id"])
                if problem := account.usable_now():
                    raise NotFound(problem)
                client = Client.from_record(record.get("client", {}), account)
                data = job_graph(h["job_id"])
                self.submit(data, Graph.from_json(data), json_of(h["targets"]), client, bool(h["force"]),
                            held={"id": h["job_id"], "submitted": h["submitted"]}, account=Account(h["user_id"]),
                            version=record.get("version"))
            except (NotFound, GraphError, CookError, ValueError, OSError) as exc:
                logs.say(log, Msg("W-QUEUE-NOTREQUEUED", job=h["job_id"], why=str(exc)))
                record.update(state="failed", finished=time.time(), error=Msg("E-QUEUE-UNPARK", reason=exc).text)
                with self.db.write() as c:
                    c.execute("UPDATE jobs SET state = 'failed', finished = ?, record = ? WHERE id = ?",
                              (record["finished"], json_text(record), h["job_id"]))

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFound(Msg("E-JOB-GONE"))
        return job

    def view(self, viewer: int | None = None, admin: bool = False) -> dict:
        """The queue as `viewer` (an account) sees it: the machine (how busy its CPU, memory and disk are: numbers
        only), the GPUs (usage, which take jobs, whether one runs a job of theirs), how many jobs the lanes without a
        GPU run at once, the two switches (显卡任务, 计算任务: queue.gpu_jobs, queue.compute_jobs; the page reads them
        here to explain itself before submitting), then the jobs: running, waiting in order, finished (newest first);
        their own whole, everyone else's waiting or running as anonymous load (Job.view); the administrator sees every
        job whole."""
        with self.cond:
            jobs = list(self.jobs.values())
            positions = self._positions()
            # waiting jobs in the order they will start, lane by lane (_line, positions), never the order they were
            # submitted in: the queue must read as what it is going to do, which is also what 拖拽插队 changes
            waiting = sorted(self._waiting(), key=lambda j: (LANES.index(j.kind.lane), positions.get(j, 0)))
        running = [j for j in jobs if j.state == "running"]
        finished = sorted((j for j in jobs if j.done), key=lambda j: -(j.finished or 0))
        order = running + waiting + finished
        etas = self._etas(running, waiting, time.time())
        on = _cards_of(running)  # which card each running job is on, the ones it borrowed for its items included
        return {
            "machine": load.now(),
            # the GPUs that take jobs (a GPU the farm never uses: only for the administrator); whether a
            # session gets this field at all is server/available.py's (FIELDS, farm.cards), and everything else about
            # the cards is the administrator's cards view (farm/cards.py)
            "gpus": [{**g.describe(), "busy": g.uuid in on,
                      "job": next((j.id for j in on.get(g.uuid, ()) if admin or j.client.user == viewer), None)}
                     for g in self.host.snapshot().gpus if admin or g.uuid in self.authorized],
            "limits": {lane: self.limit(lane) for lane in (LIGHT, HEAVY)},
            # maximum frames per submission (queue.max_frames): the page refuses before submitting, and the server
            # refuses independently in submit
            "max_frames": policy.max_frames(),
            "switches": {"gpu": _gpu_enabled(), "compute": _compute_enabled()},
            "jobs": [v for j in order if (v := j.view(viewer, admin, positions.get(j), etas.get(j.id))) is not None],
        }

    def load(self) -> dict:
        """How busy this machine is, for the strip every page shows (server/farm.py /api/load): the queue (waiting and
        running jobs), the 计算位 (how many jobs may run at once, how many do), the CPU and memory, and each card
        (whether a job is on it, how much of its memory is in use). The same numbers as the queue view, from the same
        samples (farm/load.py reads the machine at most once a second however many ask, and the cards are the
        inventory thread's latest snapshot), so a page asking often costs nothing extra and nothing here samples
        anything of its own. The cards are the administrator's (server/available.py FIELDS farm.cards): the route
        takes the field out for everyone else."""
        machine = load.now()
        with self.cond:
            waiting = len(self._waiting())
            running = [j for j in self.jobs.values() if j.state == "running"]
            slots = self.limit(LIGHT) + self.limit(HEAVY) + len(self.authorized)
        on = _cards_of(running)
        memory = machine["memory_gb"]
        return {
            "queue": {"waiting": waiting, "running": len(running)},
            "slots": {"busy": len(running), "total": slots},
            # an idle editor polls only this request: the two switches and the frame limit rarely change and do not
            # justify polling the queue separately. Most of each request is headers and the session cookie, so a longer
            # interval saves little; combining the requests does
            "switches": {"gpu": _gpu_enabled(), "compute": _compute_enabled()},
            "max_frames": policy.max_frames(),
            # values are sent at the precision the page displays: the badge shows integer percentages (`Math.round` in
            # Chrome.tsx). Decimals add nothing and are harmful: polling backoff relies on the reply being identical to
            # the previous one, and a CPU value with decimals changes every time, so backoff would never engage. On an
            # idle machine the integer percentages are stable, the reply is unchanged, and backoff takes effect
            "cpu_pct": None if machine["cpu_percent"] is None else round(machine["cpu_percent"]),
            "ram_pct": _percent(memory["used"], memory["total"]),
            "cards": [{"busy": g.uuid in on, "mem_pct": _percent(g.used_mb, g.memory_mb)}
                      for g in self.host.snapshot().gpus],
        }

    def _etas(self, running: list[Job], waiting: list[Job], t: float) -> dict[str, dict]:
        """When each running job should finish, and each waiting one start: in its lane, the slot that frees first (a
        GPU, a place in the pool or in the CPU lane) takes the next job in line, each job lasting what its estimate
        says. "partial": no earlier than that (a job ahead has nodes without records)."""
        etas = {}
        for j in running:
            left = j.remaining(t)
            if left:
                etas[j.id] = {"at": t + left[0], "partial": left[1]}

        def free(busy: Job | None) -> list:  # a slot: [when it is free, whether that is a lower bound]
            eta = etas.get(busy.id) if busy else {"at": t, "partial": False}
            return [eta["at"], eta["partial"]] if eta else [t, True]

        for lane in LANES:
            if lane == GPU:
                slots = [free(next((j for j in running if j.gpu == gpu), None)) for gpu in self.authorized]
            else:
                busy = [j for j in running if j.kind.lane == lane]
                slots = [free(j) for j in busy] + [free(None)] * max(self.limit(lane) - len(busy), 0)
            for j in (w for w in waiting if w.kind.lane == lane):
                if not slots:
                    break
                slot = min(slots, key=lambda s: s[0])
                if slot[0] > t:
                    etas[j.id] = {"at": slot[0], "partial": slot[1]}
                left = j.remaining(t)
                slot[0] += left[0] if left else 0.0
                slot[1] = slot[1] or not left or left[1]
        return etas


# ------------------------------------------------------------------ job log (the database's jobs and job_usage)

GRAPH_MAX_BYTES = 4 << 20  # the largest node graph a job may carry; the same limit as a template file (library.MAX_BYTES)


SYSTEM_OWNER = "_system"  # the folder of jobs without an account (a username always begins with a letter: accounts.USERNAME)


def graph_file(job_id: str, user_id: int | None) -> str:
    """Where the graph of job `job_id` is kept, relative to the work directory: users/<username>/jobs/<job id>.json,
    in the folder of account `user_id` (lab2shot/accounts.py); a job without an account, or whose account no longer
    exists, under users/_system/jobs/. The account's folder is deleted with the account (library.remove_user)."""
    from .. import accounts

    owner = SYSTEM_OWNER
    if user_id is not None:
        try:
            owner = accounts.get(user_id).username
        except NotFound:
            pass
    return f"users/{owner}/jobs/{job_id}.json"


def job_graph(job_id: str) -> dict:
    """The graph job `job_id` was submitted with, read from its file (the one way every reader gets it). An empty dict
    when the job has no graph file, or the file is missing or unreadable; that is logged."""
    r = db().row("SELECT graph_file FROM jobs WHERE id = ?", (job_id,))
    if r is None or not r["graph_file"]:
        logs.say(log, Msg("W-QUEUE-GRAPHMISSING", job=job_id, path="数据库里没有记它的文件"))
        return {}
    path = settings().work_dir / r["graph_file"]
    try:
        found = json_of(path.read_text(encoding="utf-8"), {})
    except (OSError, ValueError):
        found = None
    if not isinstance(found, dict):
        logs.say(log, Msg("W-QUEUE-GRAPHMISSING", job=job_id, path=str(path)))
        return {}
    return found


def _log_submitted(job: Job, data: dict) -> None:
    """The job's record in the database and its graph in its file (`graph_file`); the file is written first, so a
    record never points at a graph that is not there."""
    rel = graph_file(job.id, job.client.user)
    path = settings().work_dir / rel
    write_text(path, json_text(data))
    try:
        with job.db.write() as c:
            c.execute("INSERT INTO jobs (id, submitted, state, record, graph_file, user_id) VALUES (?, ?, ?, ?, ?, ?)",
                      (job.id, job.submitted, job.state, json_text(job.record()), rel, job.client.user))
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _log_finished(job: Job, state: str, finished: float) -> None:
    """The record of a job that ended in `state` at `finished`, written before the job itself says so (Farm._finish)."""
    record = {**job.record(), "state": state, "finished": finished}
    with job.db.write() as c:
        c.execute("UPDATE jobs SET state = ?, finished = ?, record = ? WHERE id = ?",
                  (state, finished, json_text(record), job.id))
        c.executemany("INSERT INTO job_usage (job_id, node_type, runs, reuses, seconds, gpu_seconds, frames) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      [(job.id, t, u["runs"], u["reuses"], u["seconds"], u["gpu_seconds"], u["frames"]) for t, u in job.usage.items()])


def ensure_finished(job_id: str) -> None:
    """A job still queued or running may not be deleted, nor anything of it (E-QUEUE-STILLRUNNING): cancel it first.
    The first step of every deletion path (server/quota.py drop_for_job, `forget_job`), so a running job's cache and
    material are never removed before its record turns out undeletable."""
    live = farm().jobs.get(job_id)
    if live is not None and not live.done:
        raise Invalid(Msg("E-QUEUE-STILLRUNNING"))


def forget_job(job_id: str, user_id: int | None) -> None:
    """Delete a finished job from the queue.

    Deletes the job's record, its graph file and the delivery packages it keeps on the server (which would otherwise occupy disk
    space); the result cache is not touched: it is cleaned by the admin 「结果保留」 days and may still serve other jobs
    (results of the same fingerprint are shared). Queued or running jobs may not be deleted: cancel them first
    (`Farm.cancel`), after which they count as finished. `user_id`: when not None, only the account's own jobs may be
    deleted (the route already checks ownership; this is a second check)."""
    ensure_finished(job_id)
    queue = farm()
    where = "id = ?" + ("" if user_id is None else " AND user_id = ?")
    args = (job_id,) if user_id is None else (job_id, user_id)
    row = queue.db.row(f"SELECT id, graph_file FROM jobs WHERE {where}", args)
    if row is None:
        raise NotFound(Msg("E-QUEUE-NOJOB", job=job_id))
    from ..transfer import deliveries as deliveries_store

    deliveries_store.forget_run(job_id)
    with queue.db.write() as c:
        c.execute("DELETE FROM job_usage WHERE job_id = ?", (job_id,))
        c.execute(f"DELETE FROM jobs WHERE {where}", args)
    if row["graph_file"]:
        (settings().work_dir / row["graph_file"]).unlink(missing_ok=True)
    with queue.cond:  # the lanes walk `jobs` under this lock: popping without it can kill a lane thread
        queue.jobs.pop(job_id, None)


def history(limit: int = 200, user_id: int | None = None, clicked: bool = False) -> list[dict]:
    """The latest jobs, newest first (of one account, `user_id`; only the ones a click started, `clicked`, not the
    viewer's own showing); a job the server stopped in the middle (a restart) says so. Each says whose it is as the
    account is now (「已删除的用户」 once it is deleted) and its department."""
    from ..accounts import DELETED

    queue = farm()
    active = queue.active_ids()
    where = ["1"]
    args: list = []
    if user_id is not None:
        where.append("j.user_id = ?")
        args.append(user_id)
    if clicked:
        where.append("COALESCE(json_extract(j.record, '$.shown'), 0) = 0")
    out = []
    for r in queue.db.rows(f"SELECT j.id, j.state, j.finished, j.record, j.user_id, u.username, u.name, u.department, u.deleted "
                           f"FROM jobs j LEFT JOIN users u ON u.id = j.user_id WHERE {' AND '.join(where)} "
                           f"ORDER BY j.submitted DESC LIMIT ?", (*args, limit)):
        e = json_of(r["record"])
        who = DELETED if r["deleted"] else f"{r['name']}（{r['username']}）"
        e["client"] = {**e.get("client", {}), "user": r["user_id"], "who": who, "department": r["department"] or ""}
        # what became of each delivery is now, like the queue's own rows (the record kept them as they were made)
        e["outputs"] = [_delivery_state(o) if isinstance(o, dict) and "run" in o else o for o in e.get("outputs") or []]
        ended = r["finished"] is not None
        out.append({**e, "event": "finished" if ended else "submitted",
                    "state": r["state"] if ended or r["id"] in active else "interrupted"})
    return out


def job_row(job_id: str) -> dict:
    """A job's graph as it was submitted (`job_graph`), its record and its account. NotFound when there is no such job
    or its graph is not kept."""
    r = farm().db.row("SELECT record, user_id FROM jobs WHERE id = ?", (job_id,))
    graph = job_graph(job_id) if r is not None else {}
    if not graph:
        raise NotFound(Msg("E-JOB-NOGRAPH"))
    return {"graph": graph, "record": json_of(r["record"]), "user": r["user_id"]}


MARK_S = 30.0  # a job's cache mark is worked out again at most this often


def _done_for(ev: Evaluation, node_id: str, targets: list[str]) -> bool:
    """Whether every instance of the node still has what cooking `targets` wants of it (`Evaluation.demand` /
    `satisfied`, the same rule the cook and the policy use). A node inside a 逐项处理 block has one per item
    (engine/scopes.py), so it is never asked about at the empty path."""
    paths, pending = ev.instances(node_id)
    d = ev.demand(targets)
    return not pending and bool(paths) and all(ev.satisfied(Inst(node_id, p), d.get(Inst(node_id, p), frozenset())) for p in paths)


def _targets_of(ev: Evaluation, record: dict) -> list[str]:
    """The nodes a finished job cooked to, as its record kept them (by id, or by the labels it wrote)."""
    return record.get("nodes") or [n for n, g in ev.graph.nodes.items() if g.label in record.get("targets", [])]


def _needed(ev: Evaluation, targets: list[str]) -> list[str]:
    """The nodes a cook of `targets` keeps results for: 「输出」 keeps nothing of its own."""
    return [n for n in ev.graph.needed(targets) if not ev.graph.nodes[n].type.delivers]


def fingerprints_of(ev: Evaluation, targets: list[str]) -> set[str]:
    """Which cache folders a cook of `targets` reads or writes: every output of every instance of every node it needs
    (a node inside a 逐项处理 block has one instance per item, engine/scopes.py). 按任务清缓存 and the guard that keeps
    a running job's folders out of any cleaning (busy_fingerprints) both ask this one question."""
    found: set[str] = set()
    for node in _needed(ev, targets):
        paths, _pending = ev.instances(node)
        for path in paths:
            found.update(ev.plan(node, path).outputs.values())
    return found


def job_fingerprints(graph: dict, record: dict, account: Account) -> set[str]:
    """Which cache folders one finished job's results sit in (server/quota.py clean_job: 按任务清缓存). Empty when its
    graph can't be planned any more (its uploads were cleaned, a node type is gone); then nothing of it is
    recognisable as this job's, and nothing is removed for it."""
    try:
        ev = _eval_for(Graph.from_json(graph), account)
        return fingerprints_of(ev, _targets_of(ev, record))
    except (NotFound, GraphError, CookError, ValueError, OSError):
        return set()


def busy_fingerprints() -> set[str] | None:
    """The cache folders the queued and running jobs still need, never cleaned away under them (cache cleaning must
    not affect queued or running jobs). None: one of them can't be planned right now, so which folders it needs is not
    known and nothing may be cleaned at all (server/quota.py refuses with E-QUOTA-BUSY rather than guess)."""
    found: set[str] = set()
    for job in farm().active():
        try:
            found |= fingerprints_of(job.eval, job.targets)
        except (NotFound, GraphError, CookError, ValueError, OSError):
            return None
    return found


def forget_marks() -> None:
    """Every job's cache mark is worked out again next time it is asked for, not just the one that was cleaned:
    two jobs of the same graph share the very same cache folders, so cleaning one changes the other's mark too."""
    farm()._marks.clear()


def cache_mark(job_id: str, graph: dict | None, record: dict, account: Account) -> dict:
    """Are a finished job's results still in the cache, worked out from its graph's plan (never guessed from dates):
    {"mark": "all" (全在) / "some" (部分) / "none" (已清理), "cached", "nodes", "seconds": what computing the rest
    again should take (timings), "why": when it can't be planned any more (its uploads were cleaned)}. The nodes
    counted are those whose results the job showed or delivered from (「输出」 keeps nothing of its own). `account`:
    whose it is asked for (the job's own account, or an administrator's: transfer/uploads.py Account). `graph`: the job's
    graph when the caller has it already; None reads it (`job_graph`) only when the mark is worked out again."""
    queue = farm()
    hit = queue._marks.get(job_id)
    if hit and time.time() - hit[0] < MARK_S:
        return hit[1]
    try:
        if not (graph := (job_graph(job_id) if graph is None else graph)):
            raise NotFound(Msg("E-JOB-NOGRAPH"))
        ev = _eval_for(Graph.from_json(graph), account)
        targets = _targets_of(ev, record)
        needed = _needed(ev, targets)
        missing = [n for n in needed if not _done_for(ev, n, targets)]
        history_ = timings.records(queue.db)
        works = [w for w in timings.planned(ev, needed) if w.node in missing]
        guesses = [timings.predict(w, queue.models(), history_)["seconds"] for w in works]
        mark = {"mark": "all" if not missing else "none" if len(missing) == len(needed) else "some",
                "cached": len(needed) - len(missing), "nodes": len(needed),
                "seconds": round(sum(g for g in guesses if g is not None), 1), "unknown": sum(g is None for g in guesses), "why": ""}
    except (NotFound, GraphError, CookError, ValueError, OSError) as exc:  # an upload cleaned, a node type gone
        mark = {"mark": "none", "cached": 0, "nodes": 0, "seconds": 0, "unknown": 0, "why": str(exc)}
    queue._marks[job_id] = (time.time(), mark)
    if len(queue._marks) > 2000:
        queue._marks.clear()
    return mark


class _TheFarm:
    """This process's one farm, made the first time it is asked for, once however many ask at the same moment
    (otherwise pages asking for their first frame at once would each start a farm with its lanes' threads, of which
    only one is ever closed)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.made: Farm | None = None


_THE_FARM = _TheFarm()


def farm() -> Farm:
    """This process's farm (made when first asked)."""
    made = _THE_FARM.made
    if made is not None:
        return made
    with _THE_FARM.lock:
        if _THE_FARM.made is None:
            _THE_FARM.made = Farm()
        return _THE_FARM.made


def started() -> Farm | None:
    """The farm when it has been made, else None (closing what exists without making one)."""
    return _THE_FARM.made


def forget() -> None:
    """The next farm() makes a new farm (a test's own work folder; the one before is closed by whoever made it)."""
    with _THE_FARM.lock:
        _THE_FARM.made = None
