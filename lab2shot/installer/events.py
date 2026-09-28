"""What an install tells whoever watches it: which step it is on, its log, how far a download got.

One sink interface, three watchers:

    TaskSink     the farm background task the server runs an install as (server/installs.py INSTALL): the steps are the
                 task's progress (done of total, the step it is on), log lines its output lines, coded lines its lines
                 and what it said, a download a line every tenth — the one progress format of farm/tasks.py
    ConsoleSink  the command line (lab2shot ext install): the same, printed
    Recorder     keeps every event, for a caller that reads back what an install said

and LogFile, which passes every event on to one of them and also writes it to a file: every install is logged under
work/logs/installs/ (run.install), however it was started, so what it downloaded, unpacked, linked and compiled can be
read afterwards.

An event is a plain dict:

    {"type": "plan", "steps": [{"id": "repo", "label": "代码"}, ...]}
    {"type": "step", "step": "packages", "state": "running"}          (+ "message" when it failed)
    {"type": "log", "text": "$ uv pip install ..."}                   (+ "code", "level" for a coded line)
    {"type": "progress", "what": "model.pt", "done": 1048576, "total": 4194304}

A cancel is the farm's (farm/tasks.py Cancelled): an install stops at its next check the way every background task does.
"""

from __future__ import annotations

import threading
import time

from ..data.units import PERCENT
from ..farm.tasks import Cancelled
from ..messages import Msg, said_line
from .plan import LABELS

STATES = ("waiting", "running", "done", "skipped", "failed", "cancelled")
STATE_WORDS = {"waiting": "等待", "running": "开始", "done": "完成", "skipped": "已完成，跳过", "failed": "失败",
               "cancelled": "已取消"}  # a step's state in the install log file (LogFile)
PERCENT_STEP = 10  # a download is a line every this many percent (the task's output lines, the console)

__all__ = ["STATES", "Cancelled", "ConsoleSink", "LogFile", "Recorder", "Sink", "TaskSink"]


class Sink:
    """Receives an install's events. Subclasses override `emit`; `cancel` is set to stop the install."""

    def __init__(self, cancel: threading.Event | None = None) -> None:
        self.cancel = cancel if cancel is not None else threading.Event()
        self._progress_at = 0.0

    def emit(self, event: dict) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def step(self, step: str, state: str, message: Msg | None = None) -> None:
        self.emit({"type": "step", "step": step, "state": state, **({"message": message.json()} if message else {})})

    def log(self, text: str) -> None:
        self.emit({"type": "log", "text": text})

    def say(self, message: Msg) -> None:
        """A coded line in the log (a retry, the source a file came from, a wait)."""
        self.emit({"type": "log", "text": message.text, "code": message.code, "level": message.level})

    def progress(self, what: str, done: int, total: int | None) -> None:
        """A download's bytes; sent at most a few times a second (and always at the end)."""
        now = time.monotonic()
        if total is not None and done < total and now - self._progress_at < 0.25:
            return
        self._progress_at = now
        self.emit({"type": "progress", "what": what, "done": done, "total": total})

    def check(self) -> None:
        """Raise Cancelled when the install was asked to stop."""
        if self.cancel.is_set():
            raise Cancelled()


class _Tenths:
    """A download's percentage in steps of PERCENT_STEP: the step it reached, once (None when it reached none new)."""

    def __init__(self) -> None:
        self._last: dict[str, int] = {}

    def __call__(self, event: dict) -> int | None:
        if not event.get("total"):
            return None
        whole = int(PERCENT)  # 100: a percentage in the log is a whole number, never 100.0
        reached = min(whole, int(whole * event["done"] / event["total"]) // PERCENT_STEP * PERCENT_STEP)
        if self._last.get(event["what"]) == reached:
            return None
        self._last[event["what"]] = reached
        return reached


class TaskSink(Sink):
    """An install as a farm background task (`task`: farm/tasks.py Task): cancelled with the task (its `stop`), the
    steps its progress, the log its output lines, coded lines also what it said."""

    def __init__(self, task) -> None:
        super().__init__(task.stop)
        self.task = task
        self._tenths = _Tenths()

    def emit(self, event: dict) -> None:
        kind, task = event["type"], self.task
        if kind == "plan":
            task.progress(0, len(event["steps"]), "")
        elif kind == "step":
            if event["state"] == "running":
                task.progress(label=LABELS[event["step"]])
            elif event["state"] in ("done", "skipped"):
                task.progress(task.done + 1)
        elif kind == "log":
            task.line(event["text"])
        elif kind == "progress" and (percent := self._tenths(event)) is not None:
            task.line(f"{event['what']} {percent}%")

    def say(self, message: Msg) -> None:
        """What the task said (its messages), and the coded line in its output."""
        self.task.say(message)
        self.task.line(said_line(message.json()))


class Recorder(Sink):
    """Keeps every event (in order)."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


class ConsoleSink(Sink):
    """The command line's: steps as rules, log lines as they come, downloads every tenth."""

    def __init__(self, console, labels: dict[str, str] = LABELS) -> None:
        super().__init__()
        self.console = console
        self.labels = labels
        self._tenths = _Tenths()

    def emit(self, event: dict) -> None:
        kind = event["type"]
        if kind == "step":
            label = self.labels.get(event["step"], event["step"])
            if event["state"] == "running":
                self.console.rule(label)
            elif event["state"] in ("failed", "cancelled"):
                text = event.get("message", {}).get("text", "")
                self.console.print(f"[red]✗ {label}[/red] {text}")
            elif event["state"] == "skipped":
                self.console.print(f"[green]✓[/green] {label}（已完成，跳过）")
        elif kind == "log":
            colour = {"E": "red", "W": "yellow", "B": "red"}.get(event.get("level", ""), "")
            text = event["text"]
            self.console.print(f"[{colour}]{text}[/{colour}]" if colour else text, markup=bool(colour), highlight=False)
        elif kind == "progress" and (percent := self._tenths(event)) is not None:
            self.console.print(f"[dim]{event['what']} {percent}%[/dim]")


class LogFile(Sink):
    """Another sink's events, passed on unchanged and also written to `file` (an open text file), one timestamped
    line each: steps, log lines (every line of the commands it runs), coded lines with their code, downloads every
    tenth. Cancelling is the other sink's (the same event)."""

    def __init__(self, inner: Sink, file, labels: dict[str, str] = LABELS) -> None:
        super().__init__(inner.cancel)
        self.inner = inner
        self.file = file
        self.labels = labels
        self._tenths = _Tenths()

    def _write(self, text: str) -> None:
        self.file.write(f"{time.strftime('%H:%M:%S')} {text}\n")
        self.file.flush()

    def emit(self, event: dict) -> None:
        self.inner.emit(event)
        kind = event["type"]
        if kind == "plan":
            self._write("步骤：" + "、".join(s["label"] for s in event["steps"]))
        elif kind == "step":
            label = self.labels.get(event["step"], event["step"])
            text = event.get("message", {}).get("text", "")
            self._write(f"== {label}：{STATE_WORDS.get(event['state'], event['state'])}" + (f"  {text}" if text else ""))
        elif kind == "log":
            self._write((f"[{event['code']}] " if event.get("code") else "") + event["text"])
        elif kind == "progress" and (percent := self._tenths(event)) is not None:
            self._write(f"{event['what']} {percent}%")

    def say(self, message: Msg) -> None:
        """The other sink's own handling of a coded line (a task keeps what it said), and the line in the file."""
        self.inner.say(message)
        self._write(f"[{message.code}] {message.text}")
