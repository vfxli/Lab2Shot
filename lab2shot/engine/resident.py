"""Worker processes kept between jobs: a model loaded once serves the next job at once (an artist changes a parameter
and cooks again). The worker's side is lab2shot_worker.serving (serve, @resident).

One process per extension and GPU. After a job it stays with the models that job used; a process that loaded none
ends. What the administrator set (/admin 设置, read each time, so a change applies at once) and the machine's memory
come first:
  - resident.keep off: no process stays after its job;
  - a job about to run on a GPU gets that GPU: the other processes there move their models to RAM when the machine
    still keeps enough free (memory.keep_free_gb, or the node's ram_gb; resident.to_ram off: never), else they end;
    at most resident.per_gpu wait on one GPU (the least recently used beyond that end);
  - when a job needs more RAM than is free, idle processes end, least recently used first (free_ram): the queue's
    memory guard asks before a job starts, and every worker job before it runs (cooks outside the queue too);
  - a job that runs out of GPU memory in a process that ran jobs before, or next to other processes on its GPU, runs
    once more in a new process alone on the GPU: a model kept loaded never makes a job fail;
  - a card running short of free memory (resident.release_below_gb), another program's or another server's use
    included, gets it back from the idle processes there, least recently used first: models to RAM, and processes
    end when that is not enough (their CUDA contexts); a queued job short only of those contexts ends as few as
    it needs (release_plan decides all three, a pure function);
  - a process idle for longer than resident.idle_minutes ends; one whose models are in RAM after
    resident.ram_idle_minutes (its CUDA context and RAM copy are held for nothing meanwhile);
  - the admin page (/admin) lists them and moves one's models to RAM or ends it.
A process is replaced when its code, environment or weights change (the worker identity); a job that fails or is
stopped ends its process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import atexit
import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from lab2shot_worker import PREFIX, REPO_ENV, WEIGHTS_ENV
from lab2shot_worker.serving import available_gb

from .. import logs
from ..config import machine_memory_gb, settings
from ..errors import Invalid, NotFound
from ..messages import Both, Msg
from ..process import worker_cpus

if TYPE_CHECKING:
    from ..extensions import Extension

log = logs.get("workers")

STOP_GRACE_S = 2  # a stopped worker gets this long to exit before it is killed: a stop has to free the card quickly
EXIT_S = 5  # a process asked to end gets this long before it is killed
READER_S = 5.0  # once a process has ended, its output reader gets this long to read the last of it
REPLY_S = 600  # the longest moving a process's models to RAM may take
REAP_S = 10.0
# a process's CUDA context and other memory beyond its reserved tensors, while it has not measured its own
# (lab2shot_worker/serving.py context_mb): the upper end of what was measured (about 0.4-0.5 GB on an RTX 4090,
# 0.5-0.7 GB on an RTX 5090)
CONTEXT_MB_GUESS = 800
TO_RAM, END = "to_ram", "end"


@dataclass(frozen=True)
class Held:
    """What release_plan reads of one idle process on a card. vram_mb: what its models leaving the GPU frees that the
    caller does not count as free already; context_mb: what only its ending frees beyond that."""

    key: object
    last_used: float
    on_gpu: bool
    vram_mb: int
    context_mb: int


def release_plan(held: list[Held], *, need_mb: int | None = None, keep_most: int | None = None, to_ram: bool = True,
                 ram_free_gb: float = float("inf"), keep_free_gb: float = 0.0) -> list[tuple[object, str]]:
    """Which idle processes of one card give up its memory, and how: [(key, TO_RAM | END)] in the order to act, least
    recently used first; a process not named stays as it is.

    keep_most: at most this many stay, the least recently used beyond them end (resident.per_gpu).
    need_mb None (a job takes the card): every other one still on the GPU leaves it. need_mb (free this much on the
    card): they leave it least recently used first until that much is free, and when all of them leaving it is not
    enough, they end (least recently used first) until it is. Leaving the GPU is moving to RAM when `to_ram` and the
    machine keeps `keep_free_gb` free after it (`ram_free_gb` now, less each move), else ending; moving frees vram_mb,
    ending vram_mb and context_mb."""
    order = sorted(held, key=lambda h: h.last_used)
    plan: dict[object, str] = {}
    excess = order[: max(0, len(order) - keep_most)] if keep_most is not None else []
    freed, ram = 0, ram_free_gb
    for h in excess:
        plan[h.key] = END
        freed += h.vram_mb + h.context_mb
    rest = order[len(excess):]

    def leave(h: Held) -> None:
        nonlocal freed, ram
        if to_ram and ram - h.vram_mb / 1024 >= keep_free_gb:
            ram -= h.vram_mb / 1024
            plan[h.key] = TO_RAM
            freed += h.vram_mb
        else:
            plan[h.key] = END
            freed += h.vram_mb + h.context_mb

    for h in rest:
        if need_mb is not None and freed >= need_mb:
            break
        if h.on_gpu:
            leave(h)
    if need_mb is not None:
        for h in rest:
            if freed >= need_mb:
                break
            if plan.get(h.key) != END:
                freed += h.context_mb + (0 if plan.get(h.key) == TO_RAM else h.vram_mb)
                plan[h.key] = END
    return list(plan.items())


def keep_free_gb(ram_gb: float = 0.0) -> float:
    """The memory the machine keeps free around a job whose biggest node needs `ram_gb`: GPU jobs start, and models
    move to RAM, only above it (memory.keep_free_gb, as the administrator set it now)."""
    return max(float(settings()["memory.keep_free_gb"]), ram_gb)


def _keeping() -> bool:
    return bool(settings()["resident.keep"])


def worker_environment(ext: Extension, gpu: str | None) -> dict[str, str]:
    """One worker process's environment: what an extension's Python always runs in (Extension.run_env, the same
    one the installer's self-check uses, so the check proves the environment the worker then runs in), plus what
    only a worker needs. `gpu`: CUDA_VISIBLE_DEVICES ("" hides every GPU); None keeps what this process sees."""
    env = ext.run_env(gpu=gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["LAB2SHOT_CORE_PID"] = str(os.getpid())  # a worker process ends when this process is gone
    # the extension's install location, the same for every job of the process (a job's own paths come with its job file)
    env[REPO_ENV] = str(ext.paths.repo.absolute())
    env[WEIGHTS_ENV] = str(ext.paths.weights.absolute())
    return env


NICE = 5  # worker processes run at lower priority than the server, so the page, viewer and queue on the same machine stay responsive


def hold_back(pid: int) -> None:
    """Restrict a worker process to the cores left by 「保留核心数」 and give it lower priority than the server.

    An unrestricted worker process (a single COLMAP can saturate all cores for minutes) makes the browser and the
    server on the same machine unresponsive. Core affinity is a hard guarantee: however many tasks run, the reserved
    cores stay free. Failing to set it (other systems, insufficient permissions) is not an error; the thread limit
    still applies.

    The priority is only lowered relative to the server, not forced to NICE: on Linux an unprivileged process can
    raise its nice value but not lower it (that requires CAP_SYS_NICE). With the server at nice 0 the worker gets
    NICE (5); with the server started at nice 19 (`nice -n 19`), the kernel rejects this setpriority (EACCES)
    and the worker stays at 19, which is still correct as it is not ahead of the server. The result is
    `max(NICE, this process's nice)`; it must not be forced to 5."""
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(pid, set(worker_cpus()))
    except OSError:
        pass
    try:
        os.setpriority(os.PRIO_PROCESS, pid, NICE)
    except OSError:
        pass


def event_of(line: str) -> dict | None:
    """A worker event line (PREFIX + JSON), or None for log text."""
    if not line.startswith(PREFIX):
        return None
    try:
        return json.loads(line[len(PREFIX):])
    except json.JSONDecodeError:
        return None


@dataclass(eq=False)
class Process:
    """One worker process: `python worker.py --serve` in its extension's environment, on one GPU."""

    ext: Extension
    gpu: str | None
    identity: str
    keep: bool  # stays after its job (False: another job had the kept one busy, or none are kept)
    proc: subprocess.Popen
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    jobs: int = 0
    state: str = "busy"  # busy (a job) / moving (its models to RAM) / idle / ending
    models: list[dict] = field(default_factory=list)  # as the worker last reported them
    vram_mb: int = 0
    context_mb: int | None = None  # measured by the worker (lab2shot_worker/serving.py); None: not yet
    in_ram_since: float | None = None  # when its models last went to RAM (resident.ram_idle_minutes)
    lines: queue.Queue = field(default_factory=queue.Queue)  # its output; None when it ended
    reader: threading.Thread | None = None  # reads its output into `lines`; joined once the process has ended (_close)

    @classmethod
    def start(cls, ext: Extension, gpu: str | None, identity: str, keep: bool) -> Process:
        proc = subprocess.Popen(
            [str(ext.paths.python), str(ext.worker_script), "--serve"],
            cwd=ext.paths.repo,
            env=worker_environment(ext, gpu),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            # a process group of its own: some workers run the actual computation in child processes (Pixel3DMM,
            # UniRig via subprocess.run), so stopping must kill the whole group (killpg in kill). Killing only the
            # worker would leave the grandchildren occupying the GPU until they finish, with the queue showing
            # 「已取消」 while the GPU is still busy
            start_new_session=True,
        )
        hold_back(proc.pid)  # 「保留核心数」: it runs only on the cores left for computation (process.py worker_cpus)
        p = cls(ext, gpu, identity, keep, proc)
        p.reader = threading.Thread(target=p._read, daemon=True, name=f"worker-{ext.name}")
        p.reader.start()
        return p

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    @property
    def on_gpu(self) -> bool:
        return any(m["on_gpu"] for m in self.models)

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def send(self, command: dict) -> bool:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(command) + "\n")
            self.proc.stdin.flush()
            return True
        except (OSError, ValueError):  # it ended
            return False

    def drain(self) -> None:
        """What it printed while idle goes to the server log."""
        while True:
            try:
                line = self.lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                self.lines.put(None)
                return
            logs.say(log, Msg("I-RESIDENT-LINE", project=self.ext.name, line=line.rstrip()), debug=True)

    def reply(self, kind: str, seconds: float) -> dict | None:
        """Wait for the worker's `kind` event (other output goes to the server log); None when it ended or took too long."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                line = self.lines.get(timeout=0.5)
            except queue.Empty:
                continue
            if line is None:
                return None
            event = event_of(line)
            if event is not None and event.get("type") == kind:
                return event
            logs.say(log, Msg("I-RESIDENT-LINE", project=self.ext.name, line=line.rstrip()), debug=True)
        return None

    def report(self, event: dict) -> None:
        self.models, self.vram_mb = event.get("models") or [], int(event.get("vram_mb") or 0)
        if event.get("context_mb") is not None:
            self.context_mb = int(event["context_mb"])
        self.in_ram_since = None if self.on_gpu else (self.in_ram_since or time.time())

    def held(self, counted: bool = False) -> Held:
        """What release_plan reads of it (`counted`: its vram_mb is free to the caller already)."""
        return Held(self, self.last_used, self.on_gpu, 0 if counted else self.vram_mb,
                    CONTEXT_MB_GUESS if self.context_mb is None else self.context_mb)

    def end(self, grace: float = EXIT_S) -> None:
        """Ask it to exit; kill it when it does not."""
        self.state = "ending"
        if self.alive:
            self.send({"cmd": "exit"})
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                self.kill()
        self._close()

    def kill(self) -> None:
        """Stop it now (a stopped job, a worker that died): ask the operating system first, then kill the whole process
        group, so a program the worker started for the job (Pixel3DMM / UniRig run steps in a child process) dies with
        it and the card is free at once; STOP_GRACE_S between the ask and the kill. The group is killed even when the
        worker itself has already ended: what it started may still hold the card."""
        self.state = "ending"
        self._signal(signal.SIGTERM)
        try:
            self.proc.wait(STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            pass
        self._signal(signal.SIGKILL)
        self.proc.wait()
        self._close()

    def _signal(self, sig: int) -> None:
        """`sig` to the worker's process group, or to the process alone where there is no such thing; a group already
        gone is nothing to signal. The group is named by the worker's pid (start(): a session of its own), which
        stays valid while anything in the group lives, the worker ended or not."""
        try:
            if hasattr(os, "killpg"):
                os.killpg(self.proc.pid, sig)
            else:
                self.proc.send_signal(sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def _close(self) -> None:
        """Its pipes closed and, once it has ended, its reader thread joined: nothing of a process outlives it (a server
        restart, a test's teardown). A reader still reading after READER_S means something the worker started still
        holds its output open: said in the log (closing the pipe under a blocked read would not wake it)."""
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        if self.alive or self.reader is None or self.reader is threading.current_thread():
            return
        self.reader.join(READER_S)
        if self.reader.is_alive():
            logs.say(log, Msg("W-RESIDENT-READERLEFT", project=self.ext.title))

    def ram_mb(self) -> int:
        """Its resident memory (VmRSS)."""
        try:
            for line in open(f"/proc/{self.proc.pid}/status", encoding="ascii"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) >> 10
        except OSError:
            pass
        return 0

    def view(self) -> dict:
        state = self.state if self.state in ("busy", "moving") else ("gpu" if self.on_gpu else "ram")
        return {"id": self.id, "extension": self.ext.name, "title": self.ext.title, "gpu": self.gpu, "pid": self.proc.pid,
                "state": state, "models": self.models, "vram_mb": self.vram_mb, "context_mb": self.context_mb,
                "ram_mb": self.ram_mb(),
                "idle_s": round(time.time() - self.last_used) if self.state == "idle" else 0, "jobs": self.jobs,
                "started": self.started}


@dataclass
class Outcome:
    ok: bool
    error: str | None = None  # the worker's own message (None: take the log's last line)
    oom: bool = False  # it failed because the GPU was full
    clean: bool = True  # it ran in a new process with no other on its GPU (running it again alone would not help)


class Pool:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.procs: list[Process] = []
        self._reaper: threading.Thread | None = None
        self._reaper_stop = threading.Event()  # shutdown(): the reaper ends (started again by the next process)

    # ------------------------------------------------------------------ running jobs

    def run(self, ext: Extension, gpu: str | None, identity: str, job_file: Path, on_line: Callable[[str], None],
            stop: threading.Event, keep_free: float, stage: Callable[..., None]) -> Outcome:
        """Run a job in the extension's process on `gpu` (a new one when there is none, it is busy or its code
        changed). `on_line` gets every line the worker prints for this job; the job stops when `stop` is set.
        `keep_free`: memory the machine keeps free when models move to RAM (keep_free_gb of the node's ram_gb)."""
        p = self._take(ext, gpu, identity)
        try:
            self._make_room(p, keep_free, stage)
            self.free_ram(keep_free, {ext.name})  # the queue's guard, per node (and for cooks outside the queue)
            with self.lock:
                clean = p.jobs == 0 and not any(o is not p and o.gpu == gpu for o in self.procs)
            if p.jobs:
                p.drain()
            p.send({"cmd": "job", "job": str(job_file), "keep_free_gb": keep_free})
            reply = None
            while reply is None and not stop.is_set():
                try:
                    line = p.lines.get(timeout=0.5)
                except queue.Empty:
                    # ended while a program it started still holds its output open: the end of the output never comes
                    if not p.alive:
                        break
                    continue
                if line is None:  # it ended without finishing the job (what it printed says why)
                    break
                on_line(line)
                event = event_of(line)
                if event is not None and event.get("type") == "job_end":
                    reply = event
        except BaseException:  # the cook was stopped while the worker reported (or a bug): the process goes
            self._forget(p)
            p.kill()
            raise
        if reply is None or not reply["ok"]:
            self._forget(p)
            p.kill() if stop.is_set() or not p.alive else p.end()  # a worker that died: what it started goes too
            return Outcome(False, (reply or {}).get("error"), bool((reply or {}).get("oom")), clean)
        with self.lock:
            p.report(reply)
            p.jobs += 1
            p.last_used = time.time()
            stays = p.keep and bool(p.models) and _keeping()
            p.state = "idle" if stays else "ending"
        if not stays:
            self._forget(p)
            p.end()
        return Outcome(True, clean=clean)

    def _take(self, ext: Extension, gpu: str | None, identity: str) -> Process:
        """The extension's idle process on `gpu` (once its models are back from moving to RAM), else a new one (kept
        after its job unless another is kept there)."""
        while True:
            with self.lock:
                same = [p for p in self.procs if p.ext.name == ext.name and p.gpu == gpu]
                if not any(p.state == "moving" for p in same):
                    stale = [p for p in same if p.state == "idle" and (p.identity != identity or not p.alive)]
                    for p in stale:
                        p.state = "ending"
                        self.procs.remove(p)
                    chosen = next((p for p in same if p.state == "idle"), None)
                    if chosen is None:
                        keep = _keeping() and not any(p.state == "busy" for p in same)
                        chosen = Process.start(ext, gpu, identity, keep)
                        self.procs.append(chosen)
                    chosen.state = "busy"
                    self._start_reaper()
                    break
            time.sleep(0.2)
        for p in stale:
            logs.say(log, Msg("I-RESIDENT-REPLACED", title=ext.title))
            p.end()
        return chosen

    def _make_room(self, p: Process, keep_free: float, stage: Callable[..., None]) -> None:
        """The GPU `p` runs on is the job's: the other processes there move their models to RAM (or end when the
        machine would keep less than `keep_free` GB free, or the administrator lets no models make way to RAM);
        beyond resident.per_gpu the least recently used end (release_plan)."""
        if p.gpu == "":
            return
        s = settings()
        with self.lock:
            others = [o for o in self.procs if o is not p and o.gpu == p.gpu and o.state == "idle"]
            keep_most = int(s["resident.per_gpu"])
            excess = set(sorted(others, key=lambda o: o.last_used)[: max(0, len(others) - keep_most)])
            plan = release_plan([o.held() for o in others], keep_most=keep_most, to_ram=bool(s["resident.to_ram"]),
                                ram_free_gb=available_gb(), keep_free_gb=keep_free)
            for o, _ in plan:
                o.state = "moving"
        for o, action in plan:
            if action == TO_RAM:
                stage("resident_to_ram", title=Both.of(lambda: o.ext.title))
                self._offload(o, keep_free)
                continue
            plain = o in excess or not s["resident.to_ram"]
            stage("resident_unload" if plain else "resident_unload_no_ram", title=Both.of(lambda: o.ext.title))
            self._forget(o)
            o.end()

    def _carry_out(self, plan: list[tuple[object, str]], keep_free: float, why: str) -> int:
        """Act on a release_plan whose processes are marked moving: models to RAM, or the process ends. How many
        acted."""
        for o, action in plan:
            if action == TO_RAM:
                logs.say(log, Msg("I-RESIDENT-TORAM", title=o.ext.title))
                self._offload(o, keep_free)
            else:
                logs.say(log, Msg("I-RESIDENT-UNLOADED", title=o.ext.title, why=why))
                self._forget(o)
                o.end()
        return len(plan)

    def _offload(self, p: Process, keep_free: float) -> None:
        """Move `p`'s models to RAM (it is marked moving); it ends when that fails or it holds nothing afterwards."""
        p.drain()
        reply = p.reply("offloaded", REPLY_S) if p.send({"cmd": "offload", "keep_free_gb": keep_free}) else None
        with self.lock:
            if reply is not None:
                p.report(reply)
            keep = reply is not None and bool(p.models)
            p.state = "idle" if keep else "ending"
        if not keep:
            self._forget(p)
            p.end()

    def _forget(self, p: Process) -> None:
        with self.lock:
            p.state = "ending"
            if p in self.procs:
                self.procs.remove(p)

    def _end_where(self, which: Callable[[Process], bool], why: str) -> int:
        """End the idle processes `which` picks; returns how many."""
        with self.lock:
            gone = [p for p in self.procs if p.state == "idle" and which(p)]
            for p in gone:
                p.state = "ending"
            self.procs = [p for p in self.procs if p not in gone]
        for p in gone:
            logs.say(log, Msg("I-RESIDENT-UNLOADED", title=p.ext.title, why=why))
            p.end()
        return len(gone)

    def clear_gpu(self, gpu: str | None) -> None:
        """Nothing else on `gpu`: a job ran out of its memory next to kept models and runs again alone."""
        self._end_where(lambda p: p.gpu == gpu, "out of GPU memory: the job runs again alone")

    def idle_context_mb(self) -> dict[str, int]:
        """GPU UUID -> what the idle kept-loaded processes there hold beyond their reported vram_mb (their CUDA
        contexts: as each measured it, else CONTEXT_MB_GUESS): what only ending them frees. For the scheduler's
        reclaimable_cards."""
        found: dict[str, int] = {}
        with self.lock:
            for p in self.procs:
                if p.state == "idle" and p.gpu:
                    found[p.gpu] = found.get(p.gpu, 0) + p.held().context_mb
        return found

    def _release(self, gpu: str, why: str, counted: bool = False, keep_free: float = 0.0, **plan_args) -> int:
        """release_plan over `gpu`'s idle processes, carried out; how many acted (`counted`: Process.held)."""
        with self.lock:
            here = [p for p in self.procs if p.state == "idle" and p.gpu == gpu]
            plan = release_plan([p.held(counted) for p in here], keep_free_gb=keep_free, **plan_args)
            for p, _ in plan:
                p.state = "moving"
        return self._carry_out(plan, keep_free, why)

    def clear_idle(self, short_mb: dict[str, int]) -> int:
        """A queued job waits for video memory on cards (GPU UUID -> MB it is short there) that ending idle processes
        would free: a process whose models moved to RAM still holds its CUDA context, which the scheduler sees as
        another program's memory; only ending the process frees it. As few end as cover the shortfall, least recently
        used first (their vram_mb the scheduler counts as free already)."""
        return sum(self._release(gpu, "making GPU memory for a queued job", need_mb=need, to_ram=False, counted=True)
                   for gpu, need in short_mb.items())

    def relieve(self, short_mb: dict[str, int]) -> int:
        """Cards running short of free memory (GPU UUID -> MB short of resident.release_below_gb, whoever uses it:
        another program, another server's job) get it back from their idle processes: models to RAM, least recently
        used first, and processes end when that is not enough (release_plan). How many acted."""
        if not short_mb:
            return 0
        s, keep = settings(), keep_free_gb()
        return sum(self._release(gpu, "the GPU is short of free memory", keep_free=keep, need_mb=need,
                                 to_ram=bool(s["resident.to_ram"]), ram_free_gb=available_gb())
                   for gpu, need in short_mb.items())

    # ------------------------------------------------------------------ memory guard and housekeeping

    def free_ram(self, need_gb: float, spare: set[str] = frozenset()) -> bool:
        """End idle processes, least recently used first (those of extensions in `spare` last), until the machine has
        `need_gb` free. True when any ended."""
        freed = False
        while available_gb() < need_gb:
            with self.lock:
                idle = [p for p in self.procs if p.state == "idle"]
                if not idle:
                    break
                p = min(idle, key=lambda p: (p.ext.name in spare, p.last_used))
            if self._end_where(lambda o: o is p, f"the job needs {need_gb:.0f} GB of RAM, {available_gb():.0f} GB available"):
                freed = True
        return freed

    def end_extension(self, name: str) -> None:
        """Its environment is being (re)installed."""
        self._end_where(lambda p: p.ext.name == name, "extension reinstalled")

    def end_off(self, authorized: list[str]) -> None:
        """GPUs the administrator no longer lets the farm use keep no models."""
        self._end_where(lambda p: bool(p.gpu) and p.gpu not in authorized, "GPU no longer takes jobs")

    def _start_reaper(self) -> None:
        if self._reaper is None:
            self._reaper_stop = threading.Event()
            self._reaper = threading.Thread(target=self._reap_forever, args=(self._reaper_stop,), daemon=True, name="resident-reaper")
            self._reaper.start()

    def _reap_forever(self, stop: threading.Event) -> None:
        while not stop.wait(REAP_S):
            try:
                self.tidy()
            except Exception as exc:  # housekeeping must not stop
                logs.say(log, Msg("E-RESIDENT-TIDYFAILED"), logs.error_text(exc))

    def tidy(self) -> None:
        """End the idle processes the settings no longer keep: past the idle time (a process whose models are in RAM:
        past resident.ram_idle_minutes since they went there, or since its last job when later), or all when models
        are not kept."""
        s, now = settings(), time.time()
        keeping, limit, in_ram = _keeping(), float(s["resident.idle_minutes"]) * 60, float(s["resident.ram_idle_minutes"]) * 60

        def done(p: Process) -> bool:
            if not p.alive or not keeping or now - p.last_used >= limit:
                return True
            return p.in_ram_since is not None and now - max(p.in_ram_since, p.last_used) >= in_ram

        self._end_where(done, "idle timeout" if keeping else "the administrator turned kept models off")
        self.bound_ram()

    def bound_ram(self) -> None:
        """The machine's memory the idle processes may keep: together at most resident.ram_pct of it (their VmRSS:
        models moved to RAM, and whatever else a process kept), and never so much that the machine has less than
        memory.keep_free_gb available. Beyond either, the least recently used end first. Checked by the reaper every
        REAP_S, so processes left loaded by jobs that came one after another, each on a GPU with room, stop adding up
        in RAM when no new job comes to ask for it (free_ram only acts when a job starts)."""
        most_mb = machine_memory_gb() * 1024 * float(settings()["resident.ram_pct"]) / 100
        keep_mb, free_mb = keep_free_gb() * 1024, available_gb() * 1024
        with self.lock:
            idle = sorted((p for p in self.procs if p.state == "idle"), key=lambda p: p.last_used)
        held = {p: p.ram_mb() for p in idle}
        total = sum(held.values())
        for p in idle:  # least recently used first; what one ending frees is counted at once (it takes a moment to go)
            if total <= most_mb and free_mb >= keep_mb:
                return
            why = (f"idle resident processes hold {total / 1024:.0f} GB of RAM, the limit is {most_mb / 1024:.0f} GB"
                   if total > most_mb else f"{free_mb / 1024:.0f} GB of RAM available, below the {keep_mb / 1024:.0f} GB kept free")
            if self._end_where(lambda o, p=p: o is p, why):
                total -= held[p]
                free_mb += held[p]

    def shutdown(self) -> None:
        """The server ends or restarts: every process ends, busy or not (a restart keeps the server's process id, which
        the workers watch to end with it)."""
        with self.lock:
            gone, self.procs = self.procs, []
            for p in gone:
                p.state = "ending"
            reaper, stop, self._reaper = self._reaper, self._reaper_stop, None
        for p in gone:
            logs.say(log, Msg("I-RESIDENT-UNLOADEDEND", title=p.ext.title))
            p.end()
        if reaper is not None:  # its housekeeping reads the settings of whatever work folder is current: never outlives the pool's
            stop.set()
            if reaper is not threading.current_thread():
                reaper.join()

    # ------------------------------------------------------------------ admin page

    def view(self) -> dict:
        with self.lock:
            procs = [p for p in self.procs if p.state != "ending"]
        return {"available_gb": round(available_gb(), 1), "processes": [p.view() for p in procs]}

    def idle_vram_mb(self) -> dict[str, int]:
        """GPU UUID -> the VRAM (MB, as each worker last reported it) its idle kept-loaded processes hold: memory a
        job starting on that card can have, because _make_room moves them to RAM or ends them first. For the GPU
        scheduler's inventory (farm/scheduler/inventory.py GpuState.ours_mb), so the farm's own models are never taken for
        another program's."""
        found: dict[str, int] = {}
        with self.lock:
            for p in self.procs:
                if p.state == "idle" and p.gpu:
                    found[p.gpu] = found.get(p.gpu, 0) + p.vram_mb
        return found

    def _idle(self, pid: str) -> Process:
        """(Holding the lock) the idle process `pid`: what the administrator acts on. The caller marks it in the same
        hold, so a job never takes it between the check and the mark (_take)."""
        p = next((p for p in self.procs if p.id == pid), None)
        if p is None:
            raise NotFound(Msg("E-RESIDENT-GONE"))
        if p.state != "idle":
            raise Invalid(Msg("E-RESIDENT-BUSY"))
        return p

    def offload(self, pid: str) -> None:
        """The administrator moves a process's models to RAM."""
        free, keep = available_gb(), keep_free_gb()
        with self.lock:
            p = self._idle(pid)
            if not p.on_gpu:
                return
            if free - p.vram_mb / 1024 < keep:
                raise Invalid(Msg("E-RESIDENT-NORAM", need=p.vram_mb / 1024, free=free, keep=keep))
            p.state = "moving"
        logs.say(log, Msg("I-RESIDENT-TORAM", title=p.ext.title))
        self._offload(p, keep)

    def unload(self, pid: str) -> None:
        """The administrator ends a process: its RAM and GPU memory are free (_end_where ends it only while idle)."""
        with self.lock:
            p = self._idle(pid)
        self._end_where(lambda o: o is p, "unloaded by the administrator")


@cache
def pool() -> Pool:
    p = Pool()
    atexit.register(p.shutdown)
    return p
