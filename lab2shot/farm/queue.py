"""The job queue. Every cook, from every user (web pages, DCC plugins, the command line), runs here, and every cook is
a task: a click on 「计算」 (a DCC plugin's, the command line's) computes one node and what it needs.

This is a task's lifecycle: its order in the queue, the administrator's 插队, cancelling, its records, submitting it
and its end. Where its nodes run is the scheduler's (farm/scheduler: the machine's GPU and CPU places, who gets one, the
time limit of a node), and the task's own nodes run side by side in its cook (engine/cook.py), each on the place its
cost needs (a GPU, or a CPU slot), held only while it runs. Every task is the same kind of thing: it waits in the
queue, first come first served (an administrator's 插队 puts one at the front), and each of its nodes gets its place
when its inputs are there; a node taken from the cache needs none, so a task whose results are cached is through at
once.

Every job is its account's (farm/clients.py). The queue is the same for everyone; a user sees their own jobs and, of
everyone else's, only how many are ahead of theirs and when they should end (anonymous load); the administrator sees
every job with everything known about who started it.

Every job is kept: its record in the database (lab2shot/database), written when it is submitted and updated when it
ends. Every job has a folder of its own (transfer/tasks.py) with the graph it was submitted with (`job_graph` reads
it) and the footage it reads, hard-linked, and it references the cache entries it computed or reused; it is kept
任务保留天数 after it ends, then goes whole (farm/disk.py). The end says, per node type, how its nodes served the job
(computed or answered without computing), which the usage statistics count (usage.py). The time of every node that
computes is recorded, and every job is estimated from those records (timings.py): the queue says when a running job
should finish and when a waiting one should start. A task ends when its nodes are done; the viewer's proxies of what
it computed are made afterwards, as background work the scheduler runs on CPU slots no node is waiting for.

Before the server restarts (server/restart.py) the queue is held: no task starts, new ones still come in and wait. The
waiting jobs are kept in the database (held) and queued again, under their own ids, by the server that comes next.
"""

from __future__ import annotations

import asyncio
import bisect
import fcntl
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

from .. import logs, progress
from ..data.units import PERCENT
from ..database import Database, db, json_of, json_text
from ..engine import EVALUATIONS, CookCancelled, CookError, Engine, Evaluation, Graph, GraphError
from ..engine.cook import FRAME_THREADS
from ..engine.evaluations import content_key
from ..engine.resident import pool as resident
from ..engine.scopes import Inst
from ..errors import Invalid, MessageError, NotFound, Unavailable, message_of
from ..io.digest import sha256
from ..messages import Msg
from ..serving import Account, Uses, carried, noting, serving
from ..transfer import groups, outputs, tasks
from . import disk, load, policy, scheduler, streaming, timings
from .clients import Client
from .tasks import Tasks

log = logs.get("farm")

FARM_SESSION = "farm"  # the EvaluationCache session for the queue's own internal reads (frame_range, params for
# usage stats, submit's checks, cache_mark): not a browser session, but a graph asked about here and
# through /api/status is still the same content, so a page checking a job's graph reads what the farm already built


def _eval_for(graph: Graph, account: Account) -> Evaluation:
    """The shared Evaluation of `graph` for `account` (engine/evaluations.py), for the farm's own read-only asks about
    a graph it has no job for (a finished job's cache mark). A running job reads its own (Job.eval): one job, one
    evaluation. Never one that cooks: Engine keeps its own, see its docstring."""
    return EVALUATIONS.get(FARM_SESSION, content_key(graph), lambda: graph, account)

KEEP_FINISHED = 100  # finished jobs the queue still shows
ACTIVE = ("queued", "running")
TIDY_S = 3600.0  # the disk is cleaned by the settings at least this often (and after every job)
DISK_STALE_S = 600.0  # the 硬盘 page's figures are measured again when older than this (Farm.disk)
STOP_WAIT_S = 60.0  # how long deleting an account waits for its stopped jobs to end (a worker is killed at once)
TELL_S = 1.0  # how often the waiting tasks are told again why they wait (the scheduler's reasons change by themselves)
# the frames of a packet whose proxies are made holding its lock at once (_proxies_of): four rounds of the threads a part
# is made on (view/proxy.py build, engine/cook.py FRAME_THREADS), so a cook waiting for the packet waits about the same
# on any machine; a 1080p part takes well under a second
PROXY_FRAMES = 4 * FRAME_THREADS


# Waiting for resources is shown to the user as 「排队中」 without an additional message: the user sees only four phases,
# 「排队中」, 「加载模型」, the computation steps and 「取回结果」; waiting for a card is an internal detail of the first.
#
# The single criterion is what the user can do after reading the message:
#   - nothing, and the wait resolves by itself -> no message. Busy cards (N-GPU-WAITBUSY / WAITVRAM / WAITSEVERAL),
#     the CPU slots (N-QUEUE-CPUBUSY), the task's own limits (N-QUEUE-TASKGPUS / TASKCPUS), waiting for memory
#     (N-QUEUE-WAITRAM) and jobs ahead (N-QUEUE-BEHIND) belong here; the queue position is already drawn on the queue
#     row and the node (webui/src/graph/nodes.ts waitText, QueueJob.position), and a message would repeat it.
#   - someone must act, otherwise the wait never ends -> the message is shown in full (no unexplained waiting):
#     N-QUEUE-NOMACHINEEVER (architecture mismatch, never starts without reinstalling the extension or changing the
#     card; or a node declaring more memory than the machine has), N-QUEUE-NOCARD (no card is authorized for jobs; an administrator must enable one),
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
# catalogue and must not be added.
NEVER_CODES = ("W-GPU-", "W-QUEUE-NEVERRAM")  # fixed at install (without reinstalling the extension or changing the
# card), or a node declaring more memory than the machine has: the job never starts on this server
NOCARD_CODES = ("N-GPU-NOCARD",)  # no card is authorized for jobs: computation can start once an administrator enables one, which requires action
NOMACHINE_EVER = Msg("N-QUEUE-NOMACHINEEVER")
NOMACHINE_NOCARD = Msg("N-QUEUE-NOCARD")
PAUSED = ("N-QUEUE-PAUSED", "N-QUEUE-GPUOFF")  # the administrator switched computing (or GPU jobs) off: said to the user as N-QUEUE-PAUSED


def _told(detail: Msg | None) -> Msg | None:
    """What the user is told about a task waiting for its first place, or None when no action is needed: the places
    are just busy, so it starts by itself and 「排队中」 already says everything (see the block above). `detail` is the
    scheduler's reason (farm/scheduler), which only a session with farm.cards ever sees."""
    code = detail.code if detail is not None else ""
    if code.startswith(NEVER_CODES):
        return NOMACHINE_EVER
    if code in PAUSED:
        return Msg("N-QUEUE-PAUSED")
    return NOMACHINE_NOCARD if code.startswith(NOCARD_CODES) else None


EVENT_CAP = 1000  # events a job keeps as a rule (Job._prune): a long cook's progress would grow them without end
EVENT_MOST = 5000  # events a job keeps whatever they are: beyond, the oldest go
PASSING = ("progress",)  # the events a later one of the same node replaces (computation progress has this single
# outward event type, lab2shot/progress.py: stage / progress / phase are folded into Job.now and sent as one description)
ITEM_EVENTS = ("node_start", "node_done")  # of an item's instance (逐项处理): the node's later one says as much for it


def _passing(event: dict) -> tuple | None:
    """What a later event replaces this one for (the node's latest of its kind), or None when it is kept for good: a
    node's progress, and the start or end of one item's instance of a node inside a 逐项处理 block (thousands of items
    are thousands of each; the page shows a node's state, which its latest one gives)."""
    kind = event.get("type")
    if kind in PASSING or (kind in ITEM_EVENTS and event.get("path")):
        return kind, event.get("node")
    return None


def _at(event: dict) -> tuple[str, tuple]:
    """The node instance an event is about (its node and item path)."""
    return event.get("node") or "", tuple(event.get("path") or ())


@dataclass(eq=False)
class Job:
    title: str
    graph: Graph
    targets: list[str]
    client: Client
    force: bool = False
    delivers: bool = False  # it collects and packs files for download (a 「输出」 among what it computes)
    graph_id: str = ""  # the graph file's own meta.id: "" for a submitter
    # that sends none (a DCC or the command line, with no "open graph" of its own); a job without one
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
    ran: dict = field(default_factory=dict)  # Engine.ran of its cook: instance -> what it ran with (plan, params, gpu)
    eval: Evaluation = None  # type: ignore[assignment]  # Farm.submit hands over the one it checked the job with;
    # without one (a Job built directly) the job builds its own for its account, below
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    submitted: float = field(default_factory=time.time)
    started: float | None = None  # when its first node got its place
    finished: float | None = None
    state: str = "queued"  # queued / running / done / partial / failed / cancelled
    ran_on: set[str] = field(default_factory=set)  # the models of the cards its nodes ran on (the record's `cards`)
    # the cache entries it computed or reused in its account's cache (data/packet.py note): a task's references, and
    # what cleaning leaves alone while it runs (farm/disk.py)
    uses: Uses = field(default_factory=Uses)
    events: list[dict] = field(default_factory=list)
    next_n: int = 0  # the number the next event gets: events are numbered for good (poll since, the stream's ids)
    prune_at: int = EVENT_CAP  # the number of events that makes _prune run next
    begun: set[tuple] = field(default_factory=set)  # (node, item path) of every instance it began (node_start)
    outputs: list[dict] = field(default_factory=list)  # what its 「输出」 packed: {task, node, label, pkg, name, bytes, count}
    error: str | None = None
    error_log: str | None = None  # the failed node's log (a worker's output): feedback attaches it
    reason: str = ""  # why it was cancelled, when not by the one who started it
    # computation progress, a single description (lab2shot/progress.py) used by the queue and the node alike, and the
    # only one the server sends: the phase (排队中 / 加载模型 / 计算 / 取回结果), the node, the current step's name and
    # its count (as text only), plus two internal timestamps (since / stage_since). Several of a
    # task's nodes may run at once: each running instance has its own (`active`), and `now` is the one that last said
    # something. The outward copy is computed by progress_json() on demand (`at` uses the current clock).
    now: dict = field(default_factory=lambda: dict(progress.BLANK))
    active: dict[tuple, dict] = field(default_factory=dict)  # (node, item path) -> its description, while it runs
    # the progress bar's denominator, fixed when the job starts (fix_budget: once, when its first node gets its
    # place): the sum of the predicted seconds, from history, of every node instance the job computes.
    # None: some instance has no estimate (computed for the first time), and no scale is drawn.
    budget: float | None = None
    predicted: list[float | None] = field(default_factory=list)  # each of `works`' predicted seconds (estimate)
    spent: float = 0.0  # predicted seconds of the instances already computed; it only increases, so the bar never shrinks
    share: dict[str, list[float]] = field(default_factory=dict)  # node id -> predicted seconds of each of its instances, popped as each finishes
    # why it waits although it is first in the queue: `waiting` what everyone is told (only the reasons that need
    # somebody to act) (N-QUEUE-NOMACHINEEVER / NOCARD / PAUSED; `_told` and the block above it say why the
    # self-clearing ones are not told), `waiting_detail` the scheduler's reason, which only a session with farm.cards
    # gets (server/available.py FIELDS)
    waiting: Msg | None = None
    waiting_detail: Msg | None = None
    position: int | None = None  # its place among the waiting tasks, as it was last told (Farm._announce_positions)
    works: list[timings.Work] = field(default_factory=list)  # the nodes it computes, as their estimates need them
    usage: dict[str, dict] = field(default_factory=dict)  # node type -> how its nodes served the job (_serve)
    served: set[tuple[str, tuple]] = field(default_factory=set)  # the node instances counted in `usage` (_at)
    stop: threading.Event = field(default_factory=threading.Event)
    db: Database | None = field(default=None, repr=False)  # where its records go: its farm's database
    # 插队 (Farm.first): its place in the queue's order. Its submission time, unless the administrator put it at the
    # front (then before every other task still to finish). Only later allocations follow it: nothing running stops.
    order: float = 0.0
    cond: threading.Condition = field(default_factory=threading.Condition)
    _frames: list[int] | None | Literal[False] = field(default=False, repr=False, compare=False)  # cache: False = not worked out yet
    _wakers: set[Callable[[], None]] = field(default_factory=set, repr=False, compare=False)  # event streams awaiting `changed`

    def __post_init__(self) -> None:
        if self.eval is None and self.graph is not None and self.client is not None:
            # one job, one evaluation, read-only and for the account that submitted it
            self.eval = Evaluation(self.graph, Account(self.client.user))
        self.order = self.order or self.submitted

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
        if self._frames is False and self.eval is not None:
            with serving(self.eval.account):  # its account's cache, whoever looks at the queue
                full = self.eval.frame_range(self.targets)
            self._frames = list(full) if full else None
        return self._frames or None

    def emit(self, event: dict) -> None:
        kind = event.get("type")
        t = time.time()
        timed = None  # a node's time to keep: written once the job's lock is let go
        with self.cond:
            if kind == "output":
                self.outputs.append({**{k: event.get(k) for k in ("task", "node", "label", "pkg", "name", "bytes", "count")}, "graph": self.graph_id})
            elif kind == "error":
                self.error, self.error_log = event["text"], event.get("log")
            if kind in ("message", "error") and event.get("level") in ("E", "W"):  # the server's log carries the code
                label = self.graph.nodes[event["node"]].label if event.get("node") in self.graph.nodes else ""
                logs.say(log, event, about=f"{self.id}「{label}」" if label else self.id)
            elif kind == "node_start":
                self.begun.add(_at(event))
            elif kind == "node_done":
                for fp in (event.get("outputs") or {}).values():  # computed or reused: its outputs are the job's
                    self.uses.add(fp)
                if event.get("gpu_name"):
                    self.ran_on.add(event["gpu_name"])
                # each instance serves the job once (a node in a 逐项处理 block: once per item, each its own run or
                # reuse; one cooked again for outputs wanted later, Engine._wanting_more: counted the first time)
                if _at(event) not in self.served:
                    self.served.add(_at(event))
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
            # or "version" would win; no event sets one)
            for one in out:
                self.events.append({"graph": self.graph_id, "version": self.version, **one, "t": round(t, 2), "n": self.next_n})
                self.next_n += 1
            if len(self.events) > self.prune_at:
                self._prune()
            self.notify()
        if timed is not None:
            try:
                timings.record(*timed, self.db)
            except (OSError, MessageError, sqlite3.Error) as exc:  # the job goes on without the record
                logs.say(log, Msg("W-QUEUE-TIMINGNOTKEPT", job=self.id, node=event["node"], why=str(exc)))
        if kind == "node_done":  # what the task references so far is written down as it goes, outside the lock
            try:
                self.uses.write_down(partial(tasks.reference, self.id))
            except (OSError, MessageError, sqlite3.Error) as exc:  # left for the next write, at the latest its end (_log_finished)
                logs.say(log, Msg("W-QUEUE-REFSNOTKEPT", job=self.id, node=event["node"], why=str(exc)))

    # ------------------------------------------------------------------ computation progress (rationale in lab2shot/progress.py)

    def _advance(self, kind: str, event: dict, t: float) -> None:
        """(Holding the job's lock) fold an event into the description of the instance it is about (`active`) and make
        that one `now`; this is the only place where progress advances.

        `node_start` enters the 「计算」 phase (core nodes have no 「加载模型」 phase, which exists only for workers and is
        announced by the `phase` event engine/external.py sends before starting the process); the first `progress`
        means there is something to count; `node_done` adds the instance's share to `spent` (which only increases), so
        the bar never moves backwards."""
        at = _at(event)
        if kind == "node_start":
            self.now = self.active[at] = {**progress.BLANK, "phase": progress.COMPUTING, "node": event["node"],
                                          "label": event["label"], "since": t, "stage_since": t}
            return
        if kind == "node_done":
            left = self.share.get(event["node"])
            # every instance of a node the plan computes has its share in the denominator (timings.planned: a cached one
            # too, since planned counts the node's instances); each done one moves its share over, cached or not, so the
            # two stay in step. An instance the plan did not count (a block's item not known at submission) has no share
            self.spent += left.pop(0) if left else 0.0
            # the finished instance is no longer `now` (its row shows 「用时 N 秒」, which would otherwise be
            # overwritten): the one still running that last said something is; with none, the task's next node waits
            # for its place (排队中) or comes the next moment
            self.active.pop(at, None)
            self.now = next(reversed(self.active.values()), None) or {**progress.BLANK, "phase": progress.QUEUED, "since": t, "stage_since": t}
            return
        now = self.active.get(at)
        if now is None:  # an instance that is not running (a message of a node the cook only looked at)
            return
        if kind == "phase":  # engine/external.py: before starting the worker process / after the worker wrote result.json
            now.update(phase=event["name"])
        elif kind == "stage":  # a step named by the worker (「检测人物」): only the accompanying text, the bar does not move
            now.update(note=event["name"], done=0, total=0, stage_since=t)
        elif kind == "progress":
            phase = progress.COMPUTING if now.get("phase") == progress.LOADING else now.get("phase")
            now.update(phase=phase, done=event["done"], total=event["total"])
        self.active[at] = self.active.pop(at)  # it said something last: the newest of them
        self.now = now

    def fix_budget(self) -> None:
        """Fix the progress bar's denominator when the job starts: a changing denominator would make the bar grow and
        shrink repeatedly and hide the real progress.

        Called once when its first node gets its place (Farm._started, on the scheduler's thread: nothing here reads
        the database): the predicted seconds of every node instance the job computes (the `Work` entries of
        timings.planned, one per instance inside 逐项处理 blocks), as `estimate` predicted them, are summed into the
        denominator, which then never changes, so the bar cannot shrink. If any instance has no estimate (the node is
        computed for the first time, without history), the whole budget is discarded (None) and the page draws an
        indeterminate bar rather than a fabricated fraction."""
        with self.cond:
            each = list(self.predicted)
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
        running = [(t - a.get("since", t), (self.share.get(a["node"]) or [None])[0]) for a in self.active.values()]
        at = progress.fraction(self.spent, running, self.budget)
        return {**{k: self.now.get(k, progress.BLANK[k]) for k in progress.PUBLIC}, "at": at}

    def _prune(self) -> None:
        """Keep a job's events bounded: the ones a later event replaces (`_passing`: a node's progress, an item
        instance's start or end) go, oldest first, until three quarters of EVENT_CAP are left; the rest (a node's start
        and end outside a block, what was said, delivered or failed) stays, up to EVENT_MOST. Numbers never change: a
        page asking from its last number gets what came after it, each node's latest included. The next pruning waits
        until a quarter of EVENT_CAP more have come (`prune_at`), so however many events stay, pruning costs each one
        a bounded share."""
        keys = [_passing(e) for e in self.events]
        latest = {k: i for i, k in enumerate(keys) if k is not None}
        drop = len(self.events) - EVENT_CAP * 3 // 4
        kept = []
        for i, e in enumerate(self.events):
            if drop > 0 and keys[i] is not None and latest[keys[i]] != i:
                drop -= 1
                continue
            kept.append(e)
        self.events = kept[-EVENT_MOST:]
        self.prune_at = max(EVENT_CAP, len(self.events) + EVENT_CAP // 4)

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
        path = tuple(done.get("path", ()))
        ran = self.ran.get(Inst(nid, path))  # as the cook ran it (Engine.ran): its real parameters and cost
        gpu = ran.gpu if ran is not None else self.graph.resolved(nid).cost.gpu
        params, card = (ran.params if ran is not None else {}), done.get("gpu_name", "")
        use["runs"] += 1
        use["frames"] += done["frames"]
        use["seconds"] = round(use["seconds"] + done["seconds"], 1)
        if timings.device(gpu, card) != timings.CPU:
            use["gpu_seconds"] = round(use["gpu_seconds"] + done["seconds"], 1)
        return node_type, params, gpu, card, done

    def estimate(self, models: list[str]) -> None:
        """Estimate its nodes on these GPU models (the ones that take jobs), from the timing records, when it is
        submitted: only the progress bar's denominator (`fix_budget`, once it starts) uses this prediction. The page
        shows no time left or time to start: a time predicted from earlier cooks is not reliable."""
        history = timings.records({w.type for w in self.works}, self.db)
        seconds = [timings.predict(w, models, history.get(w.type, []))["seconds"] for w in self.works]
        with self.cond:
            self.predicted = seconds

    def wait_for(self, detail: Msg | None) -> bool:
        """Why it waits now: the scheduler's reason (None: it does not, or nothing needs saying), kept for whoever may
        see the cards; what everyone is told is `_told` of it (only what needs somebody to act). True when that
        changed (the queue then tells the job again)."""
        said = _told(detail)
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

    def view(self, viewer: int | None, admin: bool, cards: list[str] = ()) -> dict | None:
        """`cards`: the cards its nodes are on now. A job is shown whole to its account (`viewer`) and the
        administrator; to anyone else only while it waits or runs, as anonymous load: its state and place (no id, no
        account, nothing of what it cooks). None: nothing of it for this viewer (someone else's finished job)."""
        mine = viewer is not None and viewer == self.client.user
        frames = self.frames if mine or admin else None  # worked out once, outside the lock (it may read the inputs)
        # one moment of it, taken under its lock: its node threads change what it is running and what it made
        with self.cond:
            running = self.state == "running"
            base = {"state": self.state, "position": self.position if self.state == "queued" else None,
                    "cards": list(cards), "submitted": self.submitted, "started": self.started,
                    "finished": self.finished, "stopping": self.stop.is_set() and running, **self.waiting_json()}
            if not (mine or admin):
                return None if self.done else {**base, "id": "", "anonymous": True, "mine": False}
            # computation progress: the same data as sent on the event stream (progress_json, lab2shot/progress.py);
            # the queue panel and the node draw the same thing, and the server does not send two versions
            now = self.progress_json(time.time()) if running else {}
            made, error, reason = list(self.outputs), self.error, self.reason
        return {
            **base, "id": self.id, "anonymous": False, "mine": mine,
            "client": self.client.full() if admin else {"who": self.client.who, "app": self.client.app},
            "title": self.title, "graph": self.graph_id, "targets": self.labels, "nodes": self.targets, "frames": frames, "error": error,
            "reason": reason, "now": now, "outputs": [outputs.present(o) for o in made],
        }

    def record(self) -> dict:
        """What the job log keeps: a copy, taken under its lock (its node threads add to what it ran on, made and
        counted)."""
        frames = self.frames  # outside the lock (it may read the inputs)
        with self.cond:
            return {"id": self.id, "title": self.title, "graph": self.graph_id, "targets": self.labels, "state": self.state, "frames": frames,
                    "submitted": self.submitted, "started": self.started, "finished": self.finished,
                    "cards": sorted(self.ran_on), "error": self.error, "error_log": self.error_log, "reason": self.reason,
                    "outputs": list(self.outputs), "client": self.client.full(), "usage": {t: dict(u) for t, u in self.usage.items()},
                    "nodes": self.targets, "version": self.version, "show": sorted(self.show) if self.show else None}


def _percent(part: float, whole: float) -> int:
    """`part` of `whole` as a whole percentage (0 when there is no whole): what the load strip shows."""
    return round(PERCENT * part / whole) if whole else 0


def _grouped(rows: list[dict]) -> list[dict]:
    """Each row that is a task says which of its account's groups it is in (transfer/groups.py: {key, name, slot}); an
    anonymous row (someone else's job) never does. The same rows, changed in place."""
    named = groups.of_tasks([r["id"] for r in rows if r.get("id") and not r.get("anonymous")])
    for r in rows:
        if (g := named.get(r.get("id") or "")) is not None:
            r["group"] = g
    return rows


OWNER_FILE = "queue.lock"  # <work>/queue.lock: held (flock) by the one process whose queue the work folder's is


def _own_queue():
    """The work folder's queue is one process's, the server's: its jobs, what they use and what cleaning may take are
    known only there. Another process (the command line, a second server) never builds one of its own beside it: it
    would take every job it does not know for finished (a live task deleted, its cache and footage cleaned, parked
    jobs taken). Held as an flock for the Farm's life: close() lets it go, the process's end or a restart's exec (the
    descriptor is not inherited) too. The file says which process holds it. E-QUEUE-OWNED when another does."""
    from ..config import settings

    path = settings().work_dir / OWNER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    held = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        held.seek(0)
        other = held.read().strip() or "?"
        held.close()
        raise Unavailable(Msg("E-QUEUE-OWNED", pid=other, folder=str(path.parent))) from None
    held.truncate(0)
    held.write(str(os.getpid()))
    held.flush()
    return held


class Farm:
    """The queue of one work folder: its records go to that folder's database, whatever happens to be current when
    a job ends (close() waits for every thread it started). Only one process has it (`_own_queue`)."""

    def __init__(self, scene_done: Callable[[str], None] | None = None) -> None:
        """`scene_done`: called with a scene.* packet's fingerprint once a job computed it (_proxies_of), to make its 3D
        view ahead (the server's view worker: farm() passes what the server gave on_scene_done; farm does not know the
        server). None (the command line, a test's own farm): nothing is made ahead, the view is made when looked at."""
        self.scene_done = scene_done
        self._owner = _own_queue()  # before anything of the work folder is touched
        self.db = db()
        self._threads: set[threading.Thread] = set()
        self._ended = threading.Event()  # close(): the threads of this queue finish
        self.cond = threading.Condition()
        self.jobs: dict[str, Job] = {}  # in submission order
        # a single removal of one account's things by disk cleaning and submit placing a job of that account into jobs
        # are mutually exclusive (`removing`): cleaning takes minutes, and checking the queue only at the start cannot
        # stop jobs submitted meanwhile. One lock per account (_cleaning_of): a job reads only its own account's
        # things, so removing one account's never holds up another's submit. Taken before cond on both sides, in the
        # same order, so there is no deadlock
        self._cleaning: dict[int, threading.Lock] = {}
        self._cleaning_lock = threading.Lock()
        self._came: dict[int, float] = {}  # account -> when a job of it last came into the queue (`removing`)
        self._disk: dict = {"areas": None, "at": None, "measuring": False}  # the 硬盘 page's figures (`disk`)
        self._disk_lock = threading.Lock()
        # this machine's GPUs (farm/scheduler/inventory.py), the farm's own idle kept-loaded models told apart: its own background
        self.host = scheduler.LocalHost(reclaimable=lambda: resident().idle_vram_mb())
        self.authorized = self.host.authorized_uuids()  # thread, never the queue's lock, refreshes it
        self.holding = False  # a restart is coming: no task starts
        self.closed = False  # none comes in any more: the waiting jobs are parked for the next server, or the queue closes
        self._marks: dict[str, tuple[float, dict]] = {}  # job id -> (when, its cache mark): cache_mark()
        self._tidy_lock = threading.Lock()  # _tidy: one at a time, and whether another was asked for meanwhile
        self._tidying = self._tidy_again = False
        # the machine's places for nodes and who gets them (farm/scheduler/pools.py): it asks this queue for its order
        # and is told nothing else; its one dispatcher thread and its background work are this queue's threads
        self.pools = scheduler.Pools(self.host, self._line, lambda: self.holding, self._started,
                                     lambda work: self._spawn(work, name="farm-later"))
        self.tasks = Tasks(self)  # the work that is not a node graph (farm/tasks.py): on this queue's threads
        disk.sweep_incomplete()  # a process killed mid-write left packets without .complete: they are not results
        self._spawn(self.pools.run, name="farm-scheduler", forever=True)
        self._unpark()
        self._spawn(self._tidy_forever, name="farm-tidy", forever=True)
        self._spawn(self._tell_forever, name="farm-tell", forever=True)

    # ------------------------------------------------------------------ GPUs

    def models(self) -> list[str]:
        """The GPU models that take jobs (what a job's estimate assumes)."""
        return list(dict.fromkeys(g.short_name for g in self.host.snapshot().authorized()))

    def authorize(self, uuids: list[str]) -> None:
        """The administrator's choice of GPUs. A GPU taken away finishes the node it runs, then takes no more."""
        self.host.authorize(uuids)
        logs.say(log, Msg("I-QUEUE-GPUSAUTHORIZED", gpus="、".join(uuids)) if uuids else Msg("I-QUEUE-NOGPUAUTHORIZED"))
        with self.cond:
            self.authorized = self.host.authorized_uuids()
        self.pools.wake()
        resident().end_off(self.authorized)

    def _cards_now(self) -> dict[str, list[str]]:
        """Which cards each job's nodes are on right now (job id -> the cards' UUIDs): the one answer to 「这张卡在忙吗」
        the queue view, the load strip and the cards page mean the same thing by."""
        on: dict[str, list[str]] = {}
        for t in self.pools.now():
            if t.granted and t.gpu:
                on.setdefault(t.task, []).append(t.gpu)
        return on

    # ------------------------------------------------------------------ jobs

    def submit(self, data: dict, graph: Graph, targets: list[str], client: Client, force: bool = False,
               held: dict | None = None, version: int | None = None,
               show: frozenset[str] | set[str] | None = None) -> Job:
        """Queue a cook of `targets` of `graph` (read from `data`, which the job log keeps) as a task. Raises GraphError / CookError
        when the graph can't be cooked as it is. `held`: a job the server before a restart kept waiting (its id and
        when it was submitted, park). The job is its submitter's (`client.user`) and runs as that account only
        (lab2shot/serving.py Account, never every account's, whatever rights the submitter has): it reads that account's
        uploads and cache, the same before and after a restart; an upload that is not theirs is not there, so its node
        fails and the rest cooks as usual. `version`: the page's cook-inputs version, carried back on every event
        (Job.version); `show`: the outputs of the shown node the viewer displays (Job.show)."""
        who = Account(client.user)
        with serving(who):  # everything it asks of the graph, in the account's own cache (data/store.py)
            return self._submit(data, graph, targets, client, force, held, who, version, show)

    def _submit(self, data: dict, graph: Graph, targets: list[str], client: Client, force: bool, held: dict | None,
                account: Account, version: int | None, show) -> Job:
        stored = len(json_text(data).encode("utf-8"))
        if stored > GRAPH_MAX_BYTES:  # the job log keeps the graph: a graph of this size is data, not a node graph
            raise GraphError(Msg("E-JOB-GRAPHTOOBIG", mb=stored / 2**20, most=GRAPH_MAX_BYTES >> 20))
        # at submission every content identity of external files is invalidated and recomputed (io/content.py): the
        # status page normally uses the recorded identity (cheap), but a cook must use the files' current content, so
        # material replaced in place with an unchanged modification time is also detected
        from ..io.content import forget_all

        forget_all()
        ev = Evaluation(graph, account)  # this job's own, read-only: Job.eval
        # the one answer: refused as a whole, or what it computes (Evaluation.readiness), for the outputs it shows as
        # its cook will be given them (Job.show -> Engine.cook)
        ready = ev.readiness(targets, force, frozenset(show or ()))
        if ready.refused is not None:
            raise GraphError(ready.refused)
        # global frame limit: the maximum number of frames per submission, a single value in the admin 「设置」.
        # Enforced here, so the page, DCC plug-ins, scripts and the command line follow the same rule (direct
        # submissions bypassing the page are refused as well)
        if (span := graph.frames or ev.frame_range(targets)) is not None:
            count = span[1] - span[0] + 1
            if count > (most := policy.max_frames()):
                raise Invalid(Msg("B-JOB-TOOMANYFRAMES", frames=count, most=most, first=span[0], last=span[1]))
        # a task brings its footage into its folder: at most 单任务上传上限 of uploads, refused here whoever submits
        # (transfer/tasks.py). A job held over a restart has its folder already.
        footage = tasks.footage_of(data) if held is None else {}
        tasks.check_footage(footage)
        # 计算任务 (queue.compute_jobs) off: refuses every new task outright, never queued (held: a job already
        # accepted before a restart is exempt, it is only being registered again). 显卡任务 (queue.gpu_jobs) off is not
        # refused here: a task with GPU nodes may still be submitted and waits.
        if held is None and not policy.compute_enabled():
            raise Unavailable(Msg("B-QUEUE-PAUSED"))
        title = data.get("meta", {}).get("name") or "节点图"
        graph_id = str(data.get("meta", {}).get("id") or "")
        delivers = any(graph.nodes[t].type.delivers for t in targets)
        job = Job(title, graph, targets, client, force, delivers, graph_id, version=version,
                  show=None if show is None else frozenset(show), eval=ev, works=timings.planned(ev, targets, force, ready),
                  db=self.db, **(held or {}))
        job.estimate(self.models())
        if held is None:  # its record before the scheduler can see it, and never under the queue's lock
            _log_submitted(job, data, footage)
        with self._cleaning_of(client.user), self.cond:  # not between a disk cleaner's check and its removal (`removing`)
            closed = self.closed  # checked and the job put in under the one lock: none comes in once it closed
            if not closed:
                self._enqueue(job, client, held, delivers, title)
        if closed:  # the server began restarting (or the queue closing) while its record was written: it never queued
            if held is None:
                _log_finished(job, "cancelled", time.time())  # a database write: never under the queue's locks
            raise Unavailable(Msg("E-QUEUE-CLOSED" if self.ending else "E-QUEUE-RESTARTING"))
        self._spawn(self._run, job, name=f"job-{job.id}")  # its cook: its nodes ask for their places as they come
        self.pools.wake()
        return job

    def _enqueue(self, job: Job, client: Client, held: dict | None, delivers: bool, title: str) -> None:
        """(Holding its account's cleaning lock and the queue's lock) put an admitted job into the queue."""
        logs.say(log, Msg({(False, False): "I-QUEUE-SUBMITTED", (False, True): "I-QUEUE-SUBMITTEDDELIVERS", (True, False): "I-QUEUE-REQUEUED",
                           (True, True): "I-QUEUE-REQUEUEDDELIVERS"}[(bool(held), delivers)], job=job.id, title=title,
                          targets="、".join(job.labels), who=client.who, ip=client.details.get("ip", "")))
        self.jobs[job.id] = job
        self._came[client.user] = time.time()
        self._announce_positions()

    def _line(self) -> list[tuple[str, bool]]:
        """The tasks still to finish in the queue's order (插队 first, then as they came in), each with whether it
        started: what the scheduler hands places out by (farm/scheduler/pools.py)."""
        with self.cond:
            line = sorted((j for j in self.jobs.values() if not j.done), key=lambda j: j.order)
        return [(j.id, j.state == "running") for j in line]

    def _started(self, job_id: str) -> None:
        """The scheduler gave the first place to one of this task's nodes: it is running from now (its progress
        budget fixed). On the scheduler's one thread: nothing here reads the database."""
        with self.cond:
            job = self.jobs.get(job_id)
            if job is None or job.state != "queued":
                return
            job.state, job.started = "running", time.time()
            job.wait_for(None)
        job.fix_budget()  # the progress bar's denominator is fixed at this moment and never changes afterwards
        # (lab2shot/progress.py: a changing denominator would shrink the bar)
        logs.say(log, Msg("I-QUEUE-JOBSTARTED", job=job.id))
        job.emit({"type": "started"})
        with self.cond:
            self._announce_positions()

    def _waiting(self) -> list[Job]:
        """The tasks waiting for their first place, in the queue's order (stopped ones: being withdrawn)."""
        return sorted((j for j in self.jobs.values() if j.state == "queued" and not j.stop.is_set()), key=lambda j: j.order)

    def _announce_positions(self) -> None:
        """(Holding the queue's lock) every waiting task told its place and why it waits, when either changed: one
        behind others waits for them (N-QUEUE-BEHIND: how many are ahead, nothing about cards); the first in line says
        why its nodes wait (the scheduler's reason for the first of them that waits: a card, the memory, a switch).

        The user is shown only the waits that do not resolve by themselves (criteria: `_told` and the block above
        it). Busy places, waiting for memory and jobs ahead resolve by themselves, and the position is already
        drawn in the interface, so nothing is said; for the user the job is 「排队中」. The rule against unexplained
        waiting still holds: the cases requiring action (N-QUEUE-NOMACHINEEVER, N-QUEUE-NOCARD, N-QUEUE-PAUSED) are
        reported in full."""
        why: dict[str, Msg] = {}
        for t in self.pools.now():
            if not t.granted and t.reason is not None and t.task not in why:
                why[t.task] = t.reason
        for position, j in enumerate(self._waiting(), 1):
            reason = why.get(j.id)
            # one behind others waits for them, unless the administrator's switch holds it: that is said to every
            # task it holds (it never starts by itself)
            if position > 1 and not (reason is not None and reason.code in PAUSED):
                reason = Msg("N-QUEUE-BEHIND", ahead=position - 1)
            if j.wait_for(reason) or j.position != position:
                j.position = position
                j.emit({"type": "queued", "position": position, **j.waiting_json()})

    def first(self, job_id: str) -> int:
        """插队: put a task still to finish at the front of the queue (before every other one), for the places handed
        out from now on; nothing running stops. Returns where it was (for the record the caller writes: server/access.py
        audit). NotFound when it is not waiting or running any more (it ended while the page was looked at)."""
        with self.cond:
            job = self.jobs.get(job_id)
            if job is None or job.done or job.stop.is_set():
                raise NotFound(Msg("E-QUEUE-NOTWAITING", job=job_id))
            line = sorted((j for j in self.jobs.values() if not j.done), key=lambda j: j.order)
            was = line.index(job) + 1
            job.order = min(j.order for j in line) - 1.0
            self._announce_positions()
        self.pools.wake()
        return was

    def _run(self, job: Job) -> None:
        # the whole job reads files as the account that submitted it (uploads.resolve) and works in that account's own
        # cache (data/store.py), noting every entry it computes or reuses (Job.uses)
        with serving(job.eval.account), noting(job.uses):
            state = self._cook(job)
        self._finish(job, state)

    def _cook(self, job: Job) -> str:
        """The task's cook, its nodes on the places the scheduler gives them; how it ended: done, partial (部分失败: a
        node failed, and another branch came through all the same), failed (nothing but the failure: every branch
        stands behind it), cancelled."""
        with job.cond:
            owner = {"user": job.client.user, "title": job.title, "graph": job.graph_id, "several": len(job.graph.deliveries()) > 1}
        collector = outputs.Collector(job.id, owner, job.emit)  # its task's outputs (transfer/outputs.py)
        engine = Engine(job.graph, stop=job.stop, collector=collector, evaluation=Evaluation(job.graph, job.eval.account),
                        stream_worker=partial(streaming.run, self))  # streaming: a streaming node's worker runs on a farm thread
        # what the cook found of each instance it ran (its real plan and parameters: Engine.ran), read while it runs by
        # a streaming node's partial folder and the timing records, never the plan from before the cook
        job.ran = engine.ran
        try:
            cooked = engine.cook(job.targets, lambda event: self._seen(job, event),
                                 scheduler.TaskResources(self.pools, job.id), force=job.force, show=job.show)
        except CookCancelled:
            return "cancelled"
        except (GraphError, ValueError, OSError, MessageError) as exc:
            job.emit({"type": "error", "node": None, **message_of(exc).json()})
            return "failed"
        except Exception as exc:  # a bug: the job fails, the queue keeps going
            logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))
            job.emit({"type": "error", "node": None, **Msg("E-FARM-INTERNAL", detail=str(exc) or type(exc).__name__).json()})
            return "failed"
        if not cooked.failed:
            return "done"
        return "partial" if cooked.through else "failed"

    def _seen(self, job: Job, event: dict) -> None:
        """Every event of the task's cook: said on the job, and a packet it computed has its view proxies made later."""
        job.emit(event)
        self._proxies_of(job, event)

    def _proxies_of(self, job: Job, event: dict) -> None:
        """A packet has just been computed (`node_done`, not a cache hit): its view proxies (`lab2shot/view/proxy.py
        build`, every channel of every frame scaled and compressed at the administrator's tier) are background work of
        the scheduler, made on a CPU slot no node is waiting for (farm/scheduler/pools.py later). The task does not
        wait for them: it ends when its nodes are done, and the frame-serving path makes a frame asked for before its
        proxy is there (the one asked for first). Compression is not reported to the user, so no events are sent; a
        failure does not fail anything (the frame-serving path builds the proxy again on first request).

        A packet's proxies are written inside its folder (view/proxy.py `_view/proxy`), so they are made holding the
        packet's lock, the one a cook holds on its outputs (data/locks.py), a few frames at a time (PROXY_FRAMES), the
        lock let go between parts for as long as a waiting cook takes to look again (data/store.py POLL_S): a recook of
        that very packet waits for one part at most, never for the whole packet, and never has its fresh folder filled
        with proxies of the old pixels; a packet recooked or cleaned meanwhile is left."""
        if event.get("type") != "node_done" or event.get("cached") or not event.get("outputs"):
            return
        from ..data.locks import exclusive
        from ..data.packet import Packet, packet_dir, valid
        from ..data.store import POLL_S
        from ..view.proxy import build, frames_of, has_proxy

        def one(fp: str) -> None:
            try:
                with exclusive(fp):
                    # only picture packets have proxies (has_proxy): a node's other outputs (an 「输出设置」's package,
                    # whose `files` is a list of what it packed; a camera, a scene) have no frames to list
                    p = Packet.load(packet_dir(fp)) if valid(packet_dir(fp)) else None
                    frames = frames_of(p) if p is not None and has_proxy(p) else []
                for at in range(0, len(frames), PROXY_FRAMES):
                    if at:
                        time.sleep(POLL_S)  # a cook waiting for the packet takes the lock now
                    with exclusive(fp):
                        if not valid(packet_dir(fp)):  # recooked or cleaned since: its proxies are the new one's
                            return
                        build(Packet.load(packet_dir(fp)), frames=frames[at:at + PROXY_FRAMES])
            except Exception as exc:  # noqa: BLE001 (proxies only affect viewing; the result itself is already written)
                logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))

        def scene(fp: str) -> None:
            """A scene packet (点云、点缓存、逐帧曲线…): its 3D view's chunks made ahead and kept on disk, by whatever
            the farm was given (`scene_done`: the server's view worker, in its background process), so that playing it
            back later is sending files; a scene with nothing per frame has no chunks and costs one look."""
            hook = self.scene_done
            if hook is None:
                return
            try:
                if valid(packet_dir(fp)) and Packet.load(packet_dir(fp)).type.split(".")[0] == "scene":
                    hook(fp)
            except Exception as exc:  # noqa: BLE001 (made ahead only to be faster: made on request otherwise)
                logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=job.id), logs.error_text(exc))

        for fp in event["outputs"].values():
            self.pools.later(carried(partial(one, fp)))  # in the job's account's cache
            self.pools.later(carried(partial(scene, fp)))

    def _finish(self, job: Job, state: str) -> None:
        """A job ends. Its record goes into the database first (its jobs row and how its nodes served it); only then is
        it said to be finished (its state, its last events), so whoever sees it finished finds its record."""
        finished = time.time()
        try:
            _log_finished(job, state, finished)
            if state != "done":
                with serving(job.eval.account):  # in its account's cache, whoever ended it (an administrator's cancel)
                    self._clean_partial(job)  # the folders it was writing are incomplete: not results, so removed
                outputs.discard_unfinished(job.id)  # an output it never packed is not an output
        except Exception as exc:  # noqa: BLE001 (the record failed; the job still ends, or it would stay `running` for ever)
            logs.say(log, Msg("E-QUEUE-RECORDFAILED", job=job.id), logs.error_text(exc))
        with self.cond:
            job.state, job.finished = state, finished
            if state == "cancelled":
                job.emit({"type": "cancelled", "reason": job.reason})
            job.emit({"type": "finished", "state": state})
            job.eval = None  # what it knew of its graph (tens of MB for a large block) is not kept with a finished job
            ended = [j for j in self.jobs.values() if j.done]
            for old in ended[:-KEEP_FINISHED]:
                del self.jobs[old.id]
            self._announce_positions()
        self.pools.forget(job.id)
        self.pools.wake()  # its order is gone: the tasks behind it move up
        took = round(job.finished - job.started, 1) if job.started else 0.0
        if state in ("failed", "partial"):
            logs.say(log, Msg("W-QUEUE-JOBFAILED" if state == "failed" else "W-QUEUE-JOBPARTIAL", job=job.id, seconds=took, error=job.error))
        elif state == "done":
            logs.say(log, Msg("I-QUEUE-JOBDONE", job=job.id, seconds=took))
        else:
            logs.say(log, Msg("I-QUEUE-JOBCANCELLED", job=job.id, seconds=took, reason=job.reason or "-"))
        self._spawn(self._tidy, name="farm-tidy-now")

    def _clean_partial(self, job: Job) -> None:
        """A job that ended without finishing (cancelled, failed, partly failed) leaves the packet folders it was
        writing incomplete: they are not results, so they are removed, each under its fingerprint's lock so a
        concurrent identical cook's folder is never touched. The folders are those of the plans the cook ran them with
        (Job.ran: an item's instance, a node after a pending switch, with the fingerprints the cook found).

        Only the instances this job itself began (Job.begun, its own node_start events): a job that never ran wrote nothing,
        and an instance it never reached is nobody's partial folder of its own. Otherwise cancelling a job that has
        not started would clean every output packet of its plan; when the same graph is submitted twice, or two
        accounts cook the same material, the folder of the other job still cooking would be deleted and its commit
        would raise FileNotFoundError."""
        with job.cond:
            begun = set(job.begun)
        if not begun:
            return
        for node, path in sorted(begun):
            if (ran := job.ran.get(Inst(node, tuple(path)))) is None:
                continue
            for fp in ran.plan.outputs.values():
                disk.clean_incomplete(fp)

    def cancel(self, job_id: str, user_id: int | None, reason: str = "") -> None:
        """Stop a job: its account (`user_id`), or the administrator (None), who may say why. Someone else's is not
        there for them. Its cook stops at once: every node running is stopped, every request withdrawn (a task that
        never started ends the same way)."""
        job = self.get(job_id)
        if user_id is not None and job.client.user != user_id:
            raise NotFound(Msg("E-JOB-GONE"))
        logs.say(log, Msg("I-QUEUE-CANCELLEDBYADMIN" if user_id is None else "I-QUEUE-CANCELLEDBYOWNER", job=job.id))
        with self.cond:
            if job.done or job.stop.is_set():
                return
            job.reason = reason
            job.stop.set()
            if job.state == "running":
                job.emit({"type": "stopping"})
            self._announce_positions()

    def _tell_forever(self) -> None:
        """The waiting tasks are told again why they wait: the scheduler's reasons change without a queue event."""
        while not self._ended.wait(TELL_S):
            with self.cond:
                self._announce_positions()

    def wake(self) -> None:
        """The settings changed: the scheduler decides again at once, and background tasks waiting for 计算任务
        look again."""
        self.pools.wake()
        self.tasks.wake()

    def _cleaning_of(self, user_id: int) -> threading.Lock:
        """The lock between a disk cleaner's removal of an account's things and a job of that account coming in."""
        with self._cleaning_lock:
            return self._cleaning.setdefault(user_id, threading.Lock())

    @contextmanager
    def removing(self, user_id: int, since: float) -> Iterator[bool]:
        """One removal of an account's cache entry, upload or task folder by a disk cleaner (farm/disk.py clean,
        tidy), and whether it may go ahead: the account has no job to finish, and none of it came into the queue since
        the cleaner began (`since`: what it worked out as safe then still is). A job reads only its own account's
        (Farm.submit), so another account's jobs never stop it. Held while the removal happens, and `submit` takes
        the same lock to put a job in: a job is either there before the check or comes after the removal, never in
        between."""
        with self._cleaning_of(user_id):
            yield not self.live_of(user_id) and self._came.get(user_id, 0.0) < since

    def cleaner(self) -> Callable[[int], AbstractContextManager[bool]]:
        """The guard of one cleaning pass beginning now (farm/disk.py Guard)."""
        since = time.time()
        return lambda user_id: self.removing(user_id, since)

    def disk(self, fresh: bool = False) -> dict:
        """The 硬盘 page's figures (farm/disk.py usage): {"areas", "at" (when measured; None never), "measuring"}.
        Measuring walks the whole data disk and may take minutes, so it runs on a thread of this queue and the caller
        gets what was measured last: a new measurement starts when there is none yet, when it is older than
        DISK_STALE_S, or when asked (`fresh`: 刷新, and after cleaning); one at a time."""
        with self._disk_lock:
            at = self._disk["at"]
            if not self._disk["measuring"] and (fresh or at is None or time.time() - at > DISK_STALE_S):
                self._disk["measuring"] = True
                self._spawn(self._measure_disk, name="farm-disk")
            return dict(self._disk)

    def _measure_disk(self) -> None:
        try:
            areas = disk.usage()
            with self._disk_lock:
                self._disk.update(areas=areas, at=time.time())
        finally:
            with self._disk_lock:
                self._disk["measuring"] = False

    def active_ids(self) -> set[str]:
        """The ids of every job still to finish (queued and running)."""
        with self.cond:
            return {j.id for j in self.jobs.values() if not j.done}

    def jobs_now(self) -> list[Job]:
        """A copy of the job table this moment (the queue changes it under its lock: nobody walks it without)."""
        with self.cond:
            return list(self.jobs.values())

    def _tidy(self) -> None:
        """Housekeeping: the disk by the settings, the database's daily backup. One at a time (every job's end and
        the hourly loop ask for it): two would measure and remove the same things at once and could name two backups
        alike. Asked while one runs, that one goes round once more when it is through, so what a job that ended
        meanwhile leaves is still cleaned."""
        with self._tidy_lock:
            if self._tidying:
                self._tidy_again = True
                return
            self._tidying = True
        while True:
            try:
                disk.tidy(guard=self.cleaner())
                self.db.backup_if_due()
            except Exception as exc:  # housekeeping never stops the queue
                logs.say(log, Msg("E-QUEUE-HOUSEKEEPING"), logs.error_text(exc))
            with self._tidy_lock:  # whether to go round again and letting go are one step: no ask falls in between
                if not self._tidy_again or self._ended.is_set():
                    self._tidying = self._tidy_again = False
                    return
                self._tidy_again = False

    def _tidy_forever(self) -> None:
        while not self._ended.wait(TIDY_S):
            self._tidy()

    def _spawn(self, target, *args, name: str, forever: bool = False) -> threading.Thread:
        """A thread of this queue (close() waits for it). `forever`: the scheduler or a housekeeping loop, one that must
        outlive any exception (a scheduler thread that dies silently leaves every node waiting for ever): the error is
        logged and the loop is started again after a moment, until the queue ends."""

        def run() -> None:
            try:
                while True:
                    try:
                        target(*args)
                        return
                    except Exception as exc:  # noqa: BLE001 (a bug in a loop of the queue: reported, never fatal to it)
                        logs.say(log, Msg("E-QUEUE-JOBINTERNAL", job=name), logs.error_text(exc))
                        if not forever or self._ended.wait(1.0):
                            return
            finally:
                with self.cond:
                    self._threads.discard(threading.current_thread())

        t = threading.Thread(target=carried(run), daemon=True, name=name)  # a job's thread goes on doing its work
        with self.cond:  # started before close() can see it (it joins only started threads)
            self._threads.add(t)
            t.start()
        return t

    def background(self, target, *args, name: str) -> threading.Thread | None:
        """Work of a subsystem that belongs to this queue (a streaming worker, farm/streaming.py): on a thread of the
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
        """Stop this queue: no task starts, every one still to finish stops, every thread it started finishes (their
        records written) before this returns. For a work folder going away; a server ends by exiting. A thread still
        running after `timeout` is a bug that would write into whatever database is current next: it raises, never
        returns as if closed."""
        self._ended.set()
        with self.cond:  # nothing starts and nothing comes in: a job taken in now would wait for a queue that runs nothing
            self.holding = self.closed = True
        self.tasks.stop_all()  # background tasks stop at their next step; their threads are joined below with the rest
        for job in [j for j in self.jobs_now() if not j.done]:
            self.cancel(job.id, None, Msg("N-QUEUE-SERVEREND").text)
        self.pools.end()
        end = time.time() + timeout
        while True:
            with self.cond:
                left = [t for t in self._threads if t is not threading.current_thread()]
            if not left or time.time() > end:
                break
            left[0].join(0.1)
        self.host.stop()  # its own background thread (farm/scheduler/inventory.py): no work folder outlives it
        if left:  # still writing into the work folder: nobody else may take its queue
            raise RuntimeError(f"queue closed with threads still running after {timeout:.0f} s: {sorted(t.name for t in left)}")
        self._owner.close()

    # ------------------------------------------------------------------ restarts

    def running(self) -> list[Job]:
        with self.cond:
            return [j for j in self.jobs.values() if j.state == "running"]

    def active(self) -> list[Job]:
        """Every job still to finish (queued and running), whoever submitted it: what may not be cleaned away under
        anyone (in_use)."""
        with self.cond:
            return [j for j in self.jobs.values() if j.state in ACTIVE]

    def hold(self) -> None:
        """No task starts from now (the server is about to restart): the tasks running go on to their end, the waiting
        ones get no place; new jobs are still taken in and wait."""
        with self.cond:
            self.holding = True
        self.pools.wake()
        logs.say(log, Msg("I-QUEUE-HOLDING"))

    def release(self) -> None:
        """The restart was called off: waiting tasks get places again."""
        with self.cond:
            self.holding = False
        self.pools.wake()
        logs.say(log, Msg("I-QUEUE-RELEASED"))

    def live_of(self, user_id: int) -> list[Job]:
        """The account's jobs still to finish (queued, running, or stopping)."""
        with self.cond:
            return [j for j in self.jobs.values() if not j.done and j.client.user == user_id]

    def stop_user(self, user_id: int, wait_s: float = STOP_WAIT_S) -> int:
        """Stop every job of an account that is being deleted and wait until each has ended (its nodes stopped,
        nothing of it written any more), so what is removed after this is not written again under the remover; returns
        how many. Unavailable (E-QUEUE-STILLSTOPPING) when one has not ended after `wait_s`."""
        live = self.live_of(user_id)
        for job in live:
            self.cancel(job.id, None, Msg("N-QUEUE-ACCOUNTDELETED").text)
        until = time.monotonic() + wait_s
        for job in live:
            with job.cond:
                if not job.cond.wait_for(lambda job=job: job.done, max(until - time.monotonic(), 0.0)):
                    raise Unavailable(Msg("E-QUEUE-STILLSTOPPING", count=len(self.live_of(user_id)), seconds=wait_s))
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
            waiting = self._waiting()
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
        cooked any more (an upload cleaned meanwhile) ends failed, saying why. Each one's held row goes only once it
        is this queue's (queued again, or its failure recorded): a server that dies on the way leaves the rest held
        for the next one."""
        from .. import accounts

        held = self.db.rows("SELECT h.job_id, h.targets, h.force, j.submitted, j.record, j.user_id FROM held h "
                            "JOIN jobs j ON j.id = h.job_id ORDER BY h.position")
        with self.db.write() as c:  # a held row whose job record is gone names nothing to queue
            c.execute("DELETE FROM held WHERE job_id NOT IN (SELECT id FROM jobs)")
        for h in held:
            record = json_of(h["record"])
            try:
                account = accounts.get(h["user_id"])
                if problem := account.usable_now():
                    raise NotFound(problem)
                client = Client.from_record(record.get("client", {}), account)
                data = job_graph(h["job_id"])
                self.submit(data, Graph.from_json(data), json_of(h["targets"]), client, bool(h["force"]),
                            held={"id": h["job_id"], "submitted": h["submitted"]}, version=record.get("version"), show=record.get("show"))
            except (NotFound, GraphError, CookError, ValueError, OSError) as exc:
                logs.say(log, Msg("W-QUEUE-NOTREQUEUED", job=h["job_id"], why=str(exc)))
                record.update(state="failed", finished=time.time(), error=Msg("E-QUEUE-UNPARK", reason=exc).text)
                with self.db.write() as c:
                    c.execute("UPDATE jobs SET state = 'failed', finished = ?, record = ? WHERE id = ?",
                              (record["finished"], json_text(record), h["job_id"]))
                    c.execute("DELETE FROM held WHERE job_id = ?", (h["job_id"],))
                continue
            with self.db.write() as c:
                c.execute("DELETE FROM held WHERE job_id = ?", (h["job_id"],))

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise NotFound(Msg("E-JOB-GONE"))
        return job

    def view(self, viewer: int | None = None, admin: bool = False) -> dict:
        """The queue as `viewer` (an account) sees it: the machine (how busy its CPU, memory and disk are: numbers
        only), the GPUs (usage, which take jobs, whether a node of theirs runs on one), the limits the scheduler goes
        by, the two switches (显卡任务, 计算任务: queue.gpu_jobs, queue.compute_jobs; the page reads them here to explain
        itself before submitting), then the jobs: running, waiting in order, finished (newest first); their own whole,
        everyone else's waiting or running as anonymous load (Job.view); the administrator sees every job whole."""
        with self.cond:
            jobs = list(self.jobs.values())
            # waiting jobs in the order they will start (the queue's order, which 插队 changes), never the order
            # they were submitted in: the queue must read as what it is going to do
            waiting = self._waiting()
        running = sorted((j for j in jobs if j.state == "running"), key=lambda j: j.order)
        finished = sorted((j for j in jobs if j.done), key=lambda j: -(j.finished or 0))
        order = running + waiting + finished
        on = self._cards_now()
        snapshot = self.host.snapshot()
        names = {g.uuid: g.short_name for g in snapshot.gpus}
        busy = {card: job for job, cards in on.items() for card in cards}
        return {
            "machine": load.now(),
            # the GPUs that take jobs (a GPU the farm never uses: only for the administrator); whether a
            # session gets this field at all is server/available.py's (FIELDS, farm.cards), and everything else about
            # the cards is the administrator's cards view (farm/cards.py)
            "gpus": [{**g.describe(), "busy": g.uuid in busy,
                      "job": busy.get(g.uuid) if admin or self._mine(busy.get(g.uuid), viewer) else None}
                     for g in snapshot.gpus if admin or g.uuid in self.authorized],
            "limits": {"task_gpus": policy.task_gpus(), "task_cpus": policy.task_cpus(), "cpu_nodes": policy.cpu_nodes()},
            # maximum frames per submission (queue.max_frames): the page refuses before submitting, and the server
            # refuses independently in submit
            "max_frames": policy.max_frames(),
            "switches": {"gpu": policy.gpu_enabled(), "compute": policy.compute_enabled()},
            "jobs": _grouped([v for j in order if (v := j.view(viewer, admin, [names.get(c, c) for c in on.get(j.id, [])])) is not None]),
        }

    def _mine(self, job_id: str | None, viewer: int | None) -> bool:
        job = self.jobs.get(job_id or "")
        return job is not None and viewer is not None and job.client.user == viewer

    def load(self) -> dict:
        """How busy this machine is, for the strip every page shows (server/farm.py /api/load): the queue (waiting and
        running jobs), the 计算位 (the places nodes may have at once: the CPU slots and the authorized cards, and how
        many nodes hold one, CPU and GPU apart), the CPU and memory, and each card (whether a node is on it, how much of
        its memory is in use). The same numbers as the queue view, from the same samples (farm/load.py reads the machine at most once a
        second however many ask, and the cards are the inventory thread's latest snapshot), so a page asking often
        costs nothing extra and nothing here samples anything of its own. Every account gets all of it (server/farm.py
        /api/load)."""
        machine = load.now()
        with self.cond:
            waiting = len(self._waiting())
            running = sum(j.state == "running" for j in self.jobs.values())
            cpu_total, gpu_total = policy.cpu_nodes(), len(self.authorized)
        held = [t for t in self.pools.now() if t.granted]
        busy = {t.gpu for t in held if t.gpu}
        memory = machine["memory_gb"]
        return {
            "queue": {"waiting": waiting, "running": running},
            # 计算位 by kind: CPU nodes at once (setting queue.cpu_nodes, background proxy work included) and one per
            # authorized card
            "slots": {"cpu": {"busy": sum(not t.gpu for t in held), "total": cpu_total},
                      "gpu": {"busy": sum(bool(t.gpu) for t in held), "total": gpu_total}},
            # an idle editor polls only this request: the two switches and the frame limit rarely change and do not
            # justify polling the queue separately. Most of each request is headers and the session cookie, so a longer
            # interval saves little; combining the requests does
            "switches": {"gpu": policy.gpu_enabled(), "compute": policy.compute_enabled()},
            "max_frames": policy.max_frames(),
            # values are sent at the precision the page displays: the badge shows integer percentages (`Math.round` in
            # Chrome.tsx). Decimals add nothing and are harmful: polling backoff relies on the reply being identical to
            # the previous one, and a CPU value with decimals changes every time, so backoff would never engage. On an
            # idle machine the integer percentages are stable, the reply is unchanged, and backoff takes effect
            "cpu_pct": None if machine["cpu_percent"] is None else round(machine["cpu_percent"]),
            "ram_pct": _percent(memory["used"], memory["total"]),
            "cards": [{"busy": g.uuid in busy, "mem_pct": _percent(g.used_mb, g.memory_mb)}
                      for g in self.host.snapshot().gpus],
        }

# ------------------------------------------------------------------ job log (the database's jobs and job_usage)

GRAPH_MAX_BYTES = 4 << 20  # the largest node graph a job may carry; the same limit as a template file (library.MAX_BYTES)


def job_graph(job_id: str) -> dict:
    """The graph job `job_id` was submitted with, read from its task's folder (transfer/tasks.py graph.json; the one
    way every reader gets it). An empty dict when its task is gone (past 任务保留天数, deleted)."""
    try:
        path = tasks.graph_file(job_id)
        found = json_of(path.read_text(encoding="utf-8"), {})
    except (OSError, ValueError, NotFound):
        return {}
    return found if isinstance(found, dict) else {}


def _log_submitted(job: Job, data: dict, footage: dict) -> None:
    """The job's record in the database and its task's folder (its graph and its footage) and row, in the same
    transaction as the job's; the folder is made first, so a record never names a task that is not there, and goes
    again when the record could not be written."""
    # the account's group it goes in (transfer/groups.py), worked out before anything is written
    group = groups.of_graph(data, job.client.user, job.submitted)
    tasks.create(job.id, json_text(data), footage)
    try:
        with job.db.write() as c:
            c.execute("INSERT INTO jobs (id, submitted, state, record, user_id) VALUES (?, ?, ?, ?, ?)",
                      (job.id, job.submitted, job.state, json_text(job.record()), job.client.user))
            tasks.record(c, job.id, job.client.user, job.submitted, footage, group)
    except BaseException:
        tasks.discard(job.id)
        raise


def _log_finished(job: Job, state: str, finished: float) -> None:
    """The record of a job that ended in `state` at `finished`, written before the job itself says so (Farm._finish)."""
    record = {**job.record(), "state": state, "finished": finished}
    job.uses.write_down(partial(tasks.reference, job.id))  # the rest of what it computed or reused
    with job.cond:
        said = list(job.events)
    try:
        tasks.write_log(job.id, record, said)
    except OSError as exc:  # the task goes on without its log; its record is in the database all the same
        logs.say(log, Msg("E-QUEUE-RECORDFAILED", job=job.id), logs.error_text(exc))
    with job.db.write() as c:
        c.execute("UPDATE jobs SET state = ?, finished = ?, record = ? WHERE id = ?",
                  (state, finished, json_text(record), job.id))
        tasks.ended(c, job.id, finished)  # kept 任务保留天数 from now
        c.executemany("INSERT INTO job_usage (job_id, node_type, runs, reuses, seconds, gpu_seconds, frames) VALUES (?, ?, ?, ?, ?, ?, ?)",
                      [(job.id, t, u["runs"], u["reuses"], u["seconds"], u["gpu_seconds"], u["frames"]) for t, u in job.usage.items()])


def ensure_finished(job_id: str) -> None:
    """A job still queued or running may not be deleted, nor anything of it (E-QUEUE-STILLRUNNING): cancel it first.
    The first step of every deletion path (server/quota.py drop_for_job, `forget_job`), so a running job's cache and
    material are never removed before its record turns out undeletable."""
    live = farm().jobs.get(job_id)
    if live is not None and not live.done:
        raise Invalid(Msg("E-QUEUE-STILLRUNNING"))


def forget_job(job_id: str, user_id: int | None) -> int:
    """Delete a finished job: its record, and its task whole (its folder: graph, footage, outputs; its row and the
    cache references it held: transfer/tasks.py remove), its outputs (folders and zips) with it. The cache
    entries only it referenced go with the next cleaning (farm/disk.py). Queued or running jobs may not be deleted:
    cancel them first (`Farm.cancel`), after which they count as finished. `user_id`: when not None, only the account's
    own jobs may be deleted (the route already checks ownership; this is a second check). Returns what its task
    folder took."""
    ensure_finished(job_id)
    queue = farm()
    where = "id = ?" + ("" if user_id is None else " AND user_id = ?")
    args = (job_id,) if user_id is None else (job_id, user_id)
    row = queue.db.row(f"SELECT id FROM jobs WHERE {where}", args)
    if row is None:
        raise NotFound(Msg("E-QUEUE-NOJOB", job=job_id))
    freed = tasks.remove(job_id) if tasks.is_task(job_id) else 0
    with queue.db.write() as c:
        c.execute("DELETE FROM job_usage WHERE job_id = ?", (job_id,))
        c.execute(f"DELETE FROM jobs WHERE {where}", args)
    with queue.cond:  # the scheduler (_line) and every reader walk `jobs` under this lock: popping without it breaks a walk
        queue.jobs.pop(job_id, None)
    if not queue.ending:
        queue._spawn(queue._tidy, name="farm-tidy-now")  # the cache only it referenced goes (once the account has no job to finish)
    return freed


def finished_of(user_id: int) -> list[str]:
    """Every job of the account that the queue window lists (its task is still kept) and that is not queued or running,
    newest first, however many: what 「删除全部」 removes and what the quota's message counts (a page of `history` has a
    limit; this has none)."""
    queue = farm()
    live = queue.active_ids()
    rows = queue.db.rows("SELECT j.id FROM jobs j WHERE j.user_id = ? AND EXISTS (SELECT 1 FROM tasks t WHERE t.id = j.id) "
                         "ORDER BY j.submitted DESC", (user_id,))
    return [r["id"] for r in rows if r["id"] not in live]


def history(limit: int = 200, user_id: int | None = None) -> list[dict]:
    """The latest jobs, newest first (of one account, `user_id`); a job the server stopped in the middle (a restart)
    says so. Each says whose it is as the account is now (「已删除的用户」 once it is deleted) and its department."""
    from ..accounts import DELETED

    queue = farm()
    active = queue.active_ids()
    where = ["1"]
    args: list = []
    if user_id is not None:
        where.append("j.user_id = ?")
        args.append(user_id)
    # a task past 任务保留天数 (or deleted) is gone whole: its record stays for the usage statistics, not in the history
    where.append("EXISTS (SELECT 1 FROM tasks t WHERE t.id = j.id)")
    out = []
    for r in queue.db.rows(f"SELECT j.id, j.state, j.finished, j.record, j.user_id, u.username, u.name, u.department, u.deleted "
                           f"FROM jobs j LEFT JOIN users u ON u.id = j.user_id WHERE {' AND '.join(where)} "
                           f"ORDER BY j.submitted DESC LIMIT ?", (*args, limit)):
        e = json_of(r["record"])
        who = DELETED if r["deleted"] else f"{r['name']}（{r['username']}）"
        e["client"] = {**e.get("client", {}), "user": r["user_id"], "who": who, "department": r["department"] or ""}
        # whether each output is still there is looked at now, like the queue's own rows (the record kept them as they
        # were made)
        e["outputs"] = [outputs.present(o) for o in e.get("outputs") or [] if isinstance(o, dict) and "pkg" in o]
        ended = r["finished"] is not None
        out.append({**e, "event": "finished" if ended else "submitted",
                    "state": r["state"] if ended or r["id"] in active else "interrupted"})
    return _grouped(out)


_RUN = uuid.uuid4().hex  # this run of the server: what a task history shows is never taken for a former run's


def listed_version(user_id: int | None) -> str:
    """What a task history is made of, as a short key that changes whenever any of it may: the account's 队列 window
    (`history(HISTORY, user_id)` with its cache marks, server/farm.py), or with `user_id` None the admin 队列
    page's (`history(HISTORY_MOST)`, every account). Read from the database, never counted by hand, so nothing that
    changes a history has to remember to say so:
      - the tasks (one added, ended, removed: submitted, finished, deleted, expired, an account removed) and the jobs
        (added, ended, deleted);
      - the names groups were given (rename, clearing);
      - an ended task's folder changed (its outputs discarded or removed: tasks.changed_folder);
      - the accounts' rows (whose each record is, as the history says it);
      - this run of the server (an unended task not running any more reads 「中断」);
      - the cache marks' own period (MARK_S: they are worked out again at most that often)."""
    d = db()
    who = (user_id, user_id)
    t = d.row("SELECT COUNT(*) AS n, TOTAL(created) AS made, TOTAL(COALESCE(ended, 0)) AS ended, "
              "SUM(ended IS NULL) AS going FROM tasks WHERE (? IS NULL OR user_id = ?)", who)
    j = d.row("SELECT COUNT(*) AS n, TOTAL(submitted) AS made, TOTAL(COALESCE(finished, 0)) AS ended "
              "FROM jobs WHERE (? IS NULL OR user_id = ?)", who)
    g = d.row("SELECT COUNT(*) AS n, TOTAL(renamed) AS at, GROUP_CONCAT(name, char(31)) AS names "
              "FROM task_group_names WHERE (? IS NULL OR user_id = ?)", who)
    u = [tuple(r) for r in d.rows("SELECT id, username, name, department, deleted FROM users WHERE (? IS NULL OR id = ?) "
                                  "ORDER BY id", who)]
    parts = (_RUN, user_id, tuple(t), tuple(j), tuple(g), u, tasks.folders_changed(), int(time.time() // MARK_S))
    return sha256(repr(parts))[:24]


def job_row(job_id: str) -> dict:
    """A job's graph as it was submitted (`job_graph`), its record and its account. NotFound when there is no such job
    or its graph is not kept."""
    r = farm().db.row("SELECT record, user_id FROM jobs WHERE id = ?", (job_id,))
    graph = job_graph(job_id) if r is not None else {}
    if not graph:
        raise NotFound(Msg("E-JOB-NOGRAPH"))
    return {"graph": graph, "record": json_of(r["record"]), "user": r["user_id"]}


MARK_S = 30.0  # a job's cache mark is worked out again at most this often


def _targets_of(ev: Evaluation, record: dict) -> list[str]:
    """The nodes a finished job cooked to, as its record kept them (by id, or by the labels it wrote)."""
    return record.get("nodes") or [n for n, g in ev.graph.nodes.items() if g.label in record.get("targets", [])]


def _needed(ev: Evaluation, targets: list[str]) -> list[str]:
    """The nodes a cook of `targets` keeps results for: 「输出」 keeps nothing of its own, and a node on a route a switch
    does not take was never cooked (Evaluation.needed), so it does not make the mark 「部分」."""
    return [n for n in ev.needed(targets) if not ev.graph.nodes[n].type.delivers]


def in_use() -> dict[int, set[str]]:
    """The cache entries the jobs still to finish (queued and running, tasks or not) have computed or reused so far,
    per account (Job.uses): cleaning never removes them under a job (farm/disk.py)."""
    found: dict[int, set[str]] = {}
    for job in farm().active():
        found.setdefault(job.eval.account.user_id, set()).update(job.uses.names())
    return found


def forget_marks() -> None:
    """Every job's cache mark is worked out again next time it is asked for, not just the one that was cleaned:
    two jobs of the same graph share the very same cache folders, so cleaning one changes the other's mark too."""
    farm()._marks.clear()


def cache_mark(job_id: str, graph: dict | None, record: dict, owner: int) -> dict:
    """Are a finished job's results still in the cache, worked out from its graph's plan (never guessed from dates):
    {"mark": "all" (全在) / "some" (部分) / "none" (已清理), "cached", "nodes", "why": when it can't be planned any
    more (its uploads were cleaned)}. No estimate of how long computing the rest again would take: one predicted from
    earlier cooks is not reliable, and the page shows none. The nodes counted are those the job's cook needed (「输出」
    keeps nothing of its own), judged by the same readiness its cook read: its targets and the outputs it showed. `owner`: the
    account that submitted the job; the mark is always of its cache, whoever asks (an administrator opening someone's
    job sees what its owner has), so one mark per job is right for everyone. `graph`: the job's graph when the caller
    has it already; None reads it (`job_graph`) only when the mark is worked out again."""
    queue = farm()
    hit = queue._marks.get(job_id)
    if hit and time.time() - hit[0] < MARK_S:
        return hit[1]
    account = Account(owner)
    with serving(account):  # in the owner's own cache (data/store.py)
        mark = _mark(job_id, graph, record, account)
    queue._marks[job_id] = (time.time(), mark)
    if len(queue._marks) > 2000:
        queue._marks.clear()
    return mark


def _mark(job_id: str, graph: dict | None, record: dict, account: Account) -> dict:
    try:
        if not (graph := (job_graph(job_id) if graph is None else graph)):
            raise NotFound(Msg("E-JOB-NOGRAPH"))
        ev = _eval_for(Graph.from_json(graph), account)
        targets = _targets_of(ev, record)
        if not targets:  # none of what it cooked is in its graph any more: nothing can be told of its results
            raise NotFound(Msg("E-JOB-NOTARGETS"))
        needed = _needed(ev, targets)
        # the one answer (Evaluation.readiness), for the cook the job was: its targets and the outputs it showed
        ready = ev.readiness(targets, False, frozenset(record.get("show") or ()))
        if ready.refused is not None:  # it can't be told what is there (the graph can't be planned now): never 「全在」
            raise GraphError(ready.refused)
        computing = {i.node for i in (*ready.computing, *ready.failing, *ready.skipped)}  # what it has no result of
        missing = [n for n in needed if n in computing]
        mark = {"mark": "all" if not missing else "none" if len(missing) == len(needed) else "some",
                "cached": len(needed) - len(missing), "nodes": len(needed), "why": ""}
    except (NotFound, GraphError, CookError, ValueError, OSError) as exc:  # an upload cleaned, a node type gone
        mark = {"mark": "none", "cached": 0, "nodes": 0, "why": str(exc)}
    return mark


class _TheFarm:
    """This process's one farm, made the first time it is asked for, once however many ask at the same moment
    (otherwise pages asking for their first frame at once would each start a farm with its scheduler's threads, of which
    only one is ever closed)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.made: Farm | None = None
        self.scene_done: Callable[[str], None] | None = None  # what the farm is made with (on_scene_done)


_THE_FARM = _TheFarm()


def on_scene_done(hook: Callable[[str], None]) -> None:
    """The server's hook for a computed scene packet (server/view_worker.py scene_done), given to this process's farm
    when it is made (Farm scene_done): the upper layer hands it down, as lab2shot/catalog.py install() does."""
    with _THE_FARM.lock:
        _THE_FARM.scene_done = hook
        if _THE_FARM.made is not None:
            _THE_FARM.made.scene_done = hook


def farm() -> Farm:
    """This process's farm (made when first asked)."""
    made = _THE_FARM.made
    if made is not None:
        return made
    with _THE_FARM.lock:
        if _THE_FARM.made is None:
            _THE_FARM.made = Farm(scene_done=_THE_FARM.scene_done)
        return _THE_FARM.made


def started() -> Farm | None:
    """The farm when it has been made, else None (closing what exists without making one)."""
    return _THE_FARM.made


def forget() -> None:
    """The next farm() makes a new farm (a test's own work folder; the one before is closed by whoever made it)."""
    with _THE_FARM.lock:
        _THE_FARM.made = None
