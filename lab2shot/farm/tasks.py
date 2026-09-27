"""Background tasks: the farm's work that is not a node graph (installing an extension, for one).

Each is a Task of the farm, sharing one job table, one progress format and one "one at a time" check, and each stops
when the server does:

    Kind      what a subsystem declares once about its tasks: its title, whether 计算任务 (queue.compute_jobs) holds it,
              which group runs one at a time and what a second one gets (refused, saying which is busy, or queued behind
              it, a queued one taking later requests in), and the messages it ends with (failed, cancelled)
    Task      one of them: its state, progress (done of total, the step it is on), what it said, its result, its output
              lines (an install's), why it waits; json() is the one progress format every page and client reads
    Tasks     the farm's (Farm.tasks): submit, get, cancel, wait, running; each task runs on a thread of the farm
              (Farm._spawn), so Farm.close() stops every one and waits for it — none outlives the server or its test

A task's work is a function of the task: it reports with progress(), say(), line(), calls check() between steps (a
cancel, or the farm closing, raises Cancelled there) and returns its result message (or None)."""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from .. import logs
from ..config import settings
from ..errors import Invalid, MessageError, NotFound, Unavailable
from ..messages import Msg

log = logs.get("farm")

KEEP = 50  # finished tasks remembered for the pages
LINES = 400  # output lines a task keeps (the newest)
REFUSE, QUEUE = "refuse", "queue"


class Cancelled(Exception):
    """Raised by Task.check(): the task was cancelled, or the farm is closing."""


@dataclass(frozen=True)
class Kind:
    """What one kind of task is. `title(subject)` names a task of it (a message); `group`: tasks of one group run one
    at a time (None: no limit); `busy`: when one of its group is active, REFUSE (`refused(the active task)` says why) or
    QUEUE (wait for it; a request while one is already waiting joins that one); `compute`: waits while 计算任务 is off;
    `failed(task, why)` and `cancelled(task)`: the result of a task that failed or was cancelled; `lasting`: it runs for
    the server's whole life (the daily benchmark rerun), so a restart that waits for tasks does not wait for it."""

    id: str
    title: Callable[[str], Msg]
    group: str | None = None
    busy: str = REFUSE
    refused: Callable[[Task], Msg] | None = None
    compute: bool = False
    failed: Callable[[Task, Msg], Msg] | None = None
    cancelled: Callable[[Task], Msg] | None = None
    lasting: bool = False


@dataclass(eq=False)
class Task:
    kind: Kind
    subject: str
    by: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "queued"  # queued / running / done / failed / cancelled
    done: int = 0
    total: int = 0
    label: str = ""
    said: list[Msg] = field(default_factory=list)
    result: Msg | None = None
    waiting: Msg | None = None
    submitted: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    lines: deque = field(default_factory=lambda: deque(maxlen=LINES))
    lines_total: int = 0  # every line ever added (a reader asks for those after its own count)
    ended: threading.Event = field(default_factory=threading.Event)
    work: Callable[[Task], Msg | None] | None = field(default=None, repr=False)

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")

    def title(self) -> Msg:
        return self.kind.title(self.subject)

    def progress(self, done: int | None = None, total: int | None = None, label: str | None = None) -> None:
        self.check()
        if total is not None:
            self.total = total
        if done is not None:
            self.done = done
        if label is not None:
            self.label = label

    def step(self, label: str) -> None:
        """One more step done, and the one it is on now."""
        self.progress(self.done + 1, label=label)

    def say(self, *messages: Msg) -> None:
        self.said.extend(messages)

    def line(self, text: str) -> None:
        self.lines.append(text)
        self.lines_total += 1

    def check(self) -> None:
        """Between steps: a cancelled task (or a closing farm) stops here."""
        if self.stop.is_set():
            raise Cancelled

    def json(self, since: int = 0) -> dict:
        """The one progress format (every kind, every page and client): the lines after `since`, `next` to ask from."""
        kept_from = self.lines_total - len(self.lines)
        start = max(since, kept_from)
        return {"id": self.id, "kind": self.kind.id, "subject": self.subject, "title": self.title().json(), "by": self.by,
                "state": self.state, "done": self.done, "total": self.total, "label": self.label,
                "said": [m.json() for m in self.said], "result": self.result.json() if self.result else None,
                "waiting": self.waiting.json() if self.waiting else None, "submitted": self.submitted,
                "started": self.started, "finished": self.finished,
                "lines": list(self.lines)[start - kept_from:], "next": self.lines_total,
                "last": self.lines[-1] if self.lines else ""}


class Tasks:
    """The background tasks of one farm (module doc)."""

    def __init__(self, farm) -> None:
        self.farm = farm
        self.cond = threading.Condition()
        self._tasks: dict[str, Task] = {}  # in submission order

    # ------------------------------------------------------------------ asking

    def submit(self, kind: Kind, subject: str, by: str, work: Callable[[Task], Msg | None]) -> Task:
        """A task of `kind` about `subject`, asked by `by`, doing `work`. Refused (Invalid, kind.refused) when its group
        is busy and its kind refuses; the waiting one returned when its kind queues and one already waits."""
        if self.farm.ending:
            raise Unavailable(Msg("E-QUEUE-RESTARTING"))
        with self.cond:
            if kind.group is not None:
                active = [t for t in self._tasks.values() if t.active and t.kind.group == kind.group]
                if active and kind.busy == REFUSE:
                    raise Invalid(kind.refused(active[0]) if kind.refused else Msg("E-TASK-BUSY", running=active[0].title()))
                waiting = next((t for t in active if t.state == "queued" and t.kind is kind), None)
                if waiting is not None:
                    return waiting
            task = Task(kind, subject, by, work=work)
            self._tasks[task.id] = task
            for old in [t for t in self._tasks.values() if not t.active][:-KEEP]:
                del self._tasks[old.id]
        if self.farm.background(self._run, task, name=f"farm-task-{kind.id}") is None:  # closing meanwhile
            self._end(task, "cancelled", kind.cancelled(task) if kind.cancelled else None)
        return task

    def get(self, task_id: str, kind: Kind | None = None) -> Task:
        with self.cond:
            found = self._tasks.get(task_id)
        if found is None or (kind is not None and found.kind.group != kind.group):
            raise NotFound(Msg("E-TASK-GONE"))
        return found

    def latest(self, kind: Kind, subject: str) -> Task | None:
        """The newest task of `kind` about `subject` (an install's page asks by the extension's name)."""
        with self.cond:
            return next((t for t in reversed(self._tasks.values()) if t.kind is kind and t.subject == subject), None)

    def running(self, kind: Kind | None = None) -> list[Task]:
        """The active tasks (of `kind`), queued or running."""
        with self.cond:
            return [t for t in self._tasks.values() if t.active and (kind is None or t.kind is kind)]

    def busy(self) -> list[Task]:
        """The active tasks a restart waits for (every kind but the lasting ones)."""
        return [t for t in self.running() if not t.kind.lasting]

    def cancel(self, task_id: str, kind: Kind | None = None) -> Task:
        task = self.get(task_id, kind)
        task.stop.set()
        with self.cond:
            self.cond.notify_all()
        return task

    def stop_all(self) -> None:
        """Every active task stops at its next check (the farm closes, the server restarts at once)."""
        with self.cond:
            for t in self._tasks.values():
                if t.active:
                    t.stop.set()
            self.cond.notify_all()

    def wait(self, task_id: str, seconds: float = 60.0) -> Task:
        """(Tests, the command line, a request that answers when it is done) until the task ends, or `seconds`."""
        task = self.get(task_id)
        task.ended.wait(seconds)
        return task

    def wake(self) -> None:
        """The settings changed (计算任务 back on): waiting tasks look again."""
        with self.cond:
            self.cond.notify_all()

    # ------------------------------------------------------------------ running

    def _may_start(self, task: Task) -> Msg | None:
        """(Holding the lock) why `task` must still wait (None: it may start)."""
        if task.kind.compute and not settings()["queue.compute_jobs"]:
            return Msg("N-QUEUE-PAUSED")
        if task.kind.group is not None:
            before = next((t for t in self._tasks.values() if t is not task and t.kind.group == task.kind.group
                           and (t.state == "running" or (t.state == "queued" and t.submitted < task.submitted))), None)
            if before is not None:
                return Msg("N-TASK-AFTER", before=before.title())
        return None

    def _run(self, task: Task) -> None:
        kind = task.kind
        with self.cond:
            while (why := self._may_start(task)) is not None and not task.stop.is_set() and not self.farm.ending:
                task.waiting = why
                self.cond.wait(1.0)
            task.waiting = None
            if task.stop.is_set() or self.farm.ending:
                state = None
            else:
                task.state, task.started, state = "running", time.time(), "running"
        if state is None:
            self._end(task, "cancelled", kind.cancelled(task) if kind.cancelled else None)
            return
        logs.say(log, Msg("I-TASK-STARTED", id=task.id, title=task.title(), by=task.by or "-"))
        try:
            self._end(task, "done", task.work(task))
        except Cancelled:
            self._end(task, "cancelled", kind.cancelled(task) if kind.cancelled else None)
        except MessageError as exc:
            self._end(task, "failed", kind.failed(task, exc.message) if kind.failed else exc.message)
        except Exception as exc:  # a task never takes the server down: the reason is said and logged
            why = Msg("E-TASK-INTERNAL", detail=f"{type(exc).__name__}: {str(exc)[:200]}")
            logs.say(log, why, logs.error_text(exc))
            self._end(task, "failed", kind.failed(task, why) if kind.failed else why)

    def _end(self, task: Task, state: str, result: Msg | None) -> None:
        with self.cond:
            task.state, task.finished, task.waiting = state, time.time(), None
            if result is not None:
                task.result = result
            self.cond.notify_all()
        task.ended.set()
        logs.say(log, Msg("I-TASK-ENDED" if state == "done" else "W-TASK-ENDED", id=task.id, title=task.title(), state=state,
                          said=task.result or "-"))

