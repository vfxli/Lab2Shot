"""Restarting the server from the admin page (/admin): settings that need it take effect (the port, HTTPS, the OCIO
config ...), and a server that misbehaves starts clean.

    drain   no job starts any more (new ones are still taken in and wait); once the jobs running and the farm's
            background tasks (an install: farm/tasks.py; not the lasting ones) are finished, the
            server restarts
    now     the running jobs stop (they end cancelled, saying why; the nodes they finished stay in the cache) and the
            background tasks are interrupted (an install started again goes on where it stopped); then it restarts

Either way nothing waiting is lost: the next server queues the waiting jobs again under their own ids (farm/queue.py
park), and the pages following them carry on (their event streams reconnect with Last-Event-ID, server/farm.py).

The same steps can end in a stop instead (`then` "stop": the configuration menu's 停止服务 and the one-click update
before it changes the code stop the service this way, cli/service.py stop): the waiting jobs are parked all the same,
and the next server that starts queues them again; the process ends instead of replacing itself.

The restart itself: the worker processes kept loaded end first (they end with the server's process, but exec keeps
its id), then the server stops listening, then the process replaces itself with exactly what started it: the
interpreter, command line, current folder and environment taken the moment serve() began (Launch), never what the
process changed since. The new process checks it is the same checkout (ROOT) before it opens anything
(E-RESTART-OTHERCHECKOUT). Everything the farm keeps is in work/farm and survives. Only a server started by
`lab2shot ui` (serve below) can restart itself.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from .. import logs
from ..errors import Invalid, Unavailable
from ..messages import Msg, said_line

log = logs.get("admin")

BOOT = uuid.uuid4().hex[:12]  # this server process's run (a restart gives the next one another)
STARTED = time.time()
NOW_REASON = Msg("N-JOB-RESTARTNOW")
POLL_S = 0.5
RESTART_ROOT = "LAB2SHOT_RESTART_ROOT"  # set only for the process a restart becomes: the checkout it must be


@dataclass(frozen=True)
class Launch:
    """How this server started, taken once as serve() begins. A restart becomes exactly this again: the same
    interpreter, command line, folder, environment, checkout and cores, whatever the running process changed since (an
    in-process LAB2SHOT_WORK_DIR, a chdir, the reservation process.apply_reservation narrowed it to)."""

    executable: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, str] = field(hash=False)
    root: str

    @classmethod
    def now(cls) -> Launch:
        from ..config import ROOT

        env = {k: v for k, v in os.environ.items() if k != RESTART_ROOT}
        return cls(sys.executable, tuple(sys.orig_argv), os.getcwd(), env, str(ROOT))

    def replace_process(self) -> None:
        from ..process import GIVEN_CPUS

        os.chdir(self.cwd)
        if hasattr(os, "sched_setaffinity"):  # the cores it started with, not the share 「保留核心数」 narrowed it to
            os.sched_setaffinity(0, set(GIVEN_CPUS))
        os.execve(self.executable, list(self.argv), {**self.env, RESTART_ROOT: self.root})


def restarted_elsewhere() -> Msg | None:
    """A process a restart became, running another checkout's code than the server before it (a different
    interpreter or editable install): it must not open the work folder."""
    from ..config import ROOT

    expected = os.environ.get(RESTART_ROOT)
    if expected and expected != str(ROOT):
        return Msg("E-RESTART-OTHERCHECKOUT", root=str(ROOT), expected=expected)
    return None


@dataclass(eq=False)
class Restarter:
    """The restart's steps, over the queue (hold, release, stop_running, running, park, and its background tasks: busy,
    stop_all) and the kept worker processes (shutdown). `stop_server` is set by the server that can restart (serve)."""

    farm: Callable[[], object]
    pool: Callable[[], object]
    stop_server: Callable[[], None] | None = None
    state: str = ""  # "" / draining (waiting for what runs) / restarting
    mode: str = ""  # drain / now
    then: str = "restart"  # restart / stop: what follows once nothing runs (stop: the process ends, serve below)
    since: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    _waiter: threading.Thread | None = None

    def request(self, mode: str, then: str = "restart") -> None:
        if mode not in ("drain", "now"):
            raise Invalid(Msg("E-RESTART-MODE", mode=mode))
        if self.stop_server is None:
            raise Unavailable(Msg("E-RESTART-NOTUI"))
        with self.lock:
            if self.state == "restarting":
                return
            if not self.state:
                self.since = time.time()
            self.state, self.mode, self.then = "draining", mode, then
        if then == "stop":
            logs.say(log, Msg("I-RESTART-ASKEDSTOPNOW" if mode == "now" else "I-RESTART-ASKEDSTOPDRAIN"))
        else:
            logs.say(log, Msg("I-RESTART-ASKEDNOW" if mode == "now" else "I-RESTART-ASKEDDRAIN"))
        self.farm().hold()
        if mode == "now":
            self.farm().stop_running(NOW_REASON)
            self.farm().tasks.stop_all()
        with self.lock:
            if self._waiter is None or not self._waiter.is_alive():
                self._waiter = threading.Thread(target=self._wait, daemon=True, name="restart")
                self._waiter.start()

    def call_off(self) -> None:
        """The administrator changed their mind while it waited for the jobs running."""
        with self.lock:
            if self.state != "draining":
                return
            self.state = self.mode = ""
            self.then = "restart"
        self.farm().release()

    def _wait(self) -> None:
        while True:
            with self.lock:
                if self.state != "draining":
                    return
            if not self.farm().running() and not self.farm().tasks.busy():
                break
            time.sleep(POLL_S)
        with self.lock:
            if self.state != "draining":
                return
            self.state = "restarting"
        self.farm().park()
        self.pool().shutdown()
        logs.say(log, Msg("I-RESTART-STOPPING" if self.then == "stop" else "I-RESTART-GOING"))
        assert self.stop_server is not None
        self.stop_server()

    def replacing(self) -> bool:
        """Whether the process becomes the next server once it stops listening (a restart), rather than ending."""
        return self.state == "restarting" and self.then == "restart"

    def view(self) -> dict | None:
        """What the pages show (None: no restart coming)."""
        if not self.state:
            return None
        from ..config import settings

        s = settings()
        running = self.farm().running()
        return {"state": self.state, "mode": self.mode, "since": self.since, "running": len(running),
                "tasks": [t.title().json() for t in self.farm().tasks.busy()],
                "port": s.value("server.port"), "https": s.value("server.https")}  # where the next server listens


@cache
def restarter() -> Restarter:
    from ..engine.resident import pool
    from ..farm import farm

    return Restarter(farm, pool)


def _thread_dumps() -> None:
    """kill -USR1 <pid> writes every thread's Python stack into work/logs/threads.txt (faulthandler: it works even while
    the process is stuck in a lock, where no Python handler would run). Said in the log with its code at every start,
    after a restart's exec too."""
    import faulthandler
    import signal

    from ..config import settings

    target = settings().work_dir / "logs" / "threads.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    global _DUMPS
    _DUMPS = target.open("a", encoding="utf-8")  # kept open for the process's life: the handler writes to its descriptor
    faulthandler.register(signal.SIGUSR1, file=_DUMPS, all_threads=True, chain=False)
    logs.say(log, Msg("I-SERVER-DUMPREADY", pid=os.getpid(), file=str(target)))


_DUMPS = None


def import_everything():
    """The whole application imported on the thread that starts the server, before any background task runs (an
    install, say), and returns the ASGI app. An import holds a lock: a task importing something
    while the next server starts after a restart's exec makes that start wait on it. Every module of ours (adapters.import_core walks the package: not a list
    of what tasks happen to reach today), then the node catalogue with every extension's nodes (adapters, the SDK,
    USD). The rule: after this returns, no thread but this one may be the first to import a module of ours."""
    from ..adapters import import_core
    from ..nodes.registry import node_types

    import_core()
    from .app import app as application

    node_types()
    return application


RECORD = "server.json"  # <work>/server.json: {"pid", "address"} of the server running on this work folder (cli/service.py reads it)


def _record(host: str | None, port: int = 0, https: bool = False) -> None:
    """Write (or, with host None, remove) this work folder's note of the server running on it: a second `lab2shot ui`
    or the configuration menu learns from it that one is running — and where, even after the port or HTTPS setting
    changed underneath (the menu asks the *current* setting's address, which would then answer nothing)."""
    import json

    from ..config import settings
    from ..io.atomic import write_text

    at = settings().work_dir / RECORD
    try:
        if host is None:
            at.unlink(missing_ok=True)
            return
        shown = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
        write_text(at, json.dumps({"pid": os.getpid(), "address": f"{'https' if https else 'http'}://{shown}:{port}"}))
    except OSError:
        pass


def recorded_address() -> str | None:
    """Where the server running on this work folder said it listens (RECORD); None when none is running. A record
    whose process no longer exists is stale (a crash or a kill) and is ignored."""
    import json

    from ..config import settings

    try:
        said = json.loads((settings().work_dir / RECORD).read_text(encoding="utf-8"))
        os.kill(int(said["pid"]), 0)
        return str(said["address"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def serve(host: str, port: int, ssl: dict) -> None:
    """Run the server (`lab2shot ui`) until it stops; when it stopped to restart, become the next one."""
    if problem := restarted_elsewhere():
        sys.exit(said_line(Msg("E-SERVER-NOSTART", reason=problem).json()))
    os.environ.pop(RESTART_ROOT, None)
    launch = Launch.now()
    from ..process import apply_memory_reuse
    from ..engine.cook import FRAME_THREADS

    memory = apply_memory_reuse(start=True, arenas=FRAME_THREADS)  # 「内存复用」: set before the queue's and the network's threads start, so the arena limit holds for them too

    import uvicorn

    from ..database import DatabaseError, close_all, db
    from ..farm import farm
    from ..workdir import WorkDirError

    # no "server: uvicorn" header; the address a proxy claims (X-Forwarded-For) is never taken as the client's: a
    # tunnel on this machine makes every request come from 127.0.0.1, and anyone could send that header
    application = import_everything()
    config = uvicorn.Config(application, host=host, port=port, log_level="warning", server_header=False,
                            proxy_headers=False, timeout_graceful_shutdown=3, **ssl)  # open event streams end on their own (farm.py)
    config.load()  # uvicorn's own lazy imports (its protocol modules) now, not inside server.run() after the tasks started
    server = uvicorn.Server(config)
    r = restarter()
    r.stop_server = lambda: setattr(server, "should_exit", True)
    try:
        db(upgrade=True).backup_if_due()  # the database opens: this checkout's, checked, brought up to date
    except (DatabaseError, WorkDirError) as exc:
        sys.exit(said_line(Msg("E-SERVER-NOSTART", reason=exc).json()))
    try:
        farm()  # the queue starts now: jobs held over a restart go on at once
    except Unavailable as exc:  # another process has this work folder's queue (farm/queue.py _own_queue)
        sys.exit(said_line(Msg("E-SERVER-NOSTART", reason=exc).json()))
    from . import tools

    tools.warm()  # the tools' signatures in the background: a plugin's first tool list does not wait for them
    from ..process import apply_reservation, apply_thread_pools

    apply_reservation()  # 「保留核心数」: the server process itself is kept to the cores left for computing (light cooks run in it)
    pools = apply_thread_pools()  # the math libraries' own thread pools: per-frame parallelism is the engine's alone
    logs.say(log, Msg("I-RESTART-STARTED", host=host, listen=port, pid=os.getpid()))
    if memory:
        logs.say(log, memory)
    logs.say(log, pools)
    _thread_dumps()
    _record(host, port, bool(ssl))
    try:
        server.run()
    finally:
        if not r.replacing():
            _record(None)
    if not r.replacing():
        return
    close_all()  # everything written is in the database file before the next server opens it
    sys.stdout.flush()
    sys.stderr.flush()
    launch.replace_process()
