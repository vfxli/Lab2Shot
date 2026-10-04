"""The view worker: the 3D viewer's data (server/view_data.py) is made in processes of their own, never in the web
server's. Reading a production scene takes time and memory; made in the server, one user's big file would hold it for
minutes and grow it by gigabytes while the page took half a minute to open for everyone else. So:

- LANES processes; a view is always made, kept and served by the same one (chosen by its key), which takes one
  request at a time; they run niced: the web server comes first;
- a request waits at most WAIT_S for its lane (another view being made there);
- making a view, or a part of it, takes at most BUILD_S, and the process at most MEMORY_GB of memory: beyond, the
  process is ended (the next request starts a new one) and the viewer is told why, in words. The data is never cut
  down to fit: the view has to show what the data holds;
- a worker keeps the views it made (VIEWS, the least recently used let go first) and their chunks;
- every question names the account it is asked for (lab2shot/serving.py): the worker reads that account's own cache
  (data/store.py) and keeps its views apart, so one account's view is never answered to another.

The web server's side waits in the thread its request runs in: the routes that ask a lane are declared on the WAITS
lane of threads (server/wire.py), so however many of them wait, no other route waits for a thread."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import zlib
from collections import OrderedDict, deque

from .. import logs
from .words import Word
from ..errors import NotFound, Unavailable, Unviewable
from ..messages import Msg

LANES = 2
BUILD_S = 120.0  # the longest making a view (or one of its parts) may take
START_S = 90.0  # the longest a new worker may take to start (it loads USD's libraries)
ANSWER_S = 10.0  # the longest a request's thread waits for its answer; the build goes on, the page asks again (E-VIEW-PREPARING)
PENDING_MAX = 32  # asks a lane holds at once (each a thread waiting its turn): above it a request is refused (E-VIEW-BUSY)
PENDING_PER_ACCOUNT = 4  # ... of them one account's (the rest refused as busy); answered taking turns with the others
RECENT = 16  # answers finished after their request gave up waiting, kept for the request that comes back
MEMORY_GB = 8.0  # the most memory a worker may hold
VIEWS = 6
ENV = ("LAB2SHOT_WORK_DIR", "LAB2SHOT_CONFIG")  # where the results are: a worker follows the server's

log = logs.get("view")


def _settings_now() -> tuple:
    """All of the server's current settings (key -> current value), sent to the worker with every request.

    The worker is a separate `spawn`ed process; its own `settings()` reflects the file as read when it started and
    would not see later changes made in the admin page, yet it uses the 「点云上限」 `view.points_max_mb` to decide the
    thinning step (`server/view_data.py _budget`). Without sending the settings with each request, a changed limit would
    silently have no effect.

    No subset is selected: a selection list would need manual maintenance, and a setting left out of it would again have no effect.
    The whole set is sent and the worker overrides its own with it, so it sees exactly what the server sees."""
    from ..config import SCHEMA, settings

    s = settings()
    return tuple((k, s[k]) for k in SCHEMA)


def describe(how: tuple, url: str) -> str:
    """The description (JSON) of the view `how` names (view_data.build), its parts at `url` ({part} in it). Its chunks
    are then made ahead in the background, if they have not been (prebuild_later): a view looked at is one to be played."""
    text = _lane(how).ask(how, "describe", url)
    prebuild_later(how)
    return text


def part(how: tuple, name: str) -> bytes:
    """A part of the view `how` names, gzipped (view_data.View.part). There is a single variant: thinning always
    applies, and one address is one sequence of bytes."""
    return _lane(how).ask(how, "part", (name,))


def prebuild_later(how: tuple) -> None:
    """Make every chunk of the view `how` ahead, in the background, and keep them on disk (view_data.store_chunk), for
    the account asking now. The web server then sends those files without asking a lane (packets.py scene_part,
    view.py points_part): one lane answers one question at a time, so the chunks a page asks for in parallel were made
    one after another there, and a whole shot could never come faster than that lane.

    It has a process of its own (`_ahead`, a lane like the others, niced the same), so the lanes that answer pages are
    never held up by it; views are made there in the order asked, one chunk per question (a question's time limit,
    BUILD_S, is per chunk). A view already asked for in this server (same account, same packets' generations) is not
    asked again; a partial result (points_partial) never is: its frames keep coming."""
    from ..serving import account

    if how[0] not in ("scene", "points"):
        return
    key = (account().user_id, how, _generation(how))
    with _ahead_guard:
        if key in _ahead_seen:
            return
        _ahead_seen[key] = None
        while len(_ahead_seen) > AHEAD_SEEN:
            _ahead_seen.popitem(last=False)
        global _ahead_thread
        if _ahead_thread is None or not _ahead_thread.is_alive():
            _ahead_thread = threading.Thread(target=_ahead_loop, daemon=True, name="view-ahead")
            _ahead_thread.start()
    _ahead_todo.put(key)


def scene_done(fp: str) -> None:
    """A scene packet has just been computed (the farm's scene_done, handed down by server/app.py through
    farm/queue.py on_scene_done): its view's chunks made ahead."""
    prebuild_later(("scene", fp))


def _ahead_loop() -> None:
    while True:
        key = _ahead_todo.get()  # (account, how, generation): worked out when it was asked for, as that account
        who, how, _ = key
        try:
            count = _ahead.run(who, how, "chunks", ())
            for k in range(count):
                _ahead.run(who, how, "store", (k,))
        except Exception as exc:  # noqa: BLE001 (made ahead only to be faster: the chunks are still made on request)
            logs.say(log, Msg("E-VIEW-FAILED", detail=f"building 3D view data ahead: {exc}"), about=str(how))
            with _ahead_guard:  # asked again the next time it is looked at (nothing here reads the disk: this thread
                _ahead_seen.pop(key, None)  # serves no account, and a failure while failing would end it)


def shutdown() -> None:
    with _ahead.lock:
        _ahead.end()
    for lane in _lanes:
        with lane.lock:
            lane.end()


class _Ask:
    """One question to a lane's worker: answered on a thread of the lane's, waited for by whoever asked (several
    requests for the same thing wait on the same one)."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value = None
        self.error: BaseException | None = None

    def result(self):
        if self.error is not None:
            raise self.error
        return self.value


class _Lane:
    def __init__(self) -> None:
        self.lock = threading.Lock()  # the worker process answers one question at a time
        self.proc = None
        self.conn = None
        self._guard = threading.Condition()
        self._pending: dict[tuple, _Ask] = {}  # being answered now (or waiting their turn)
        # the asks waiting, by account: answered taking turns, one account's at a time (_take_turns), so the asks one
        # account piles up never come before another's next
        self._waiting: dict[int, deque] = {}
        self._served: dict[int, int] = {}  # account -> the turn it last had (never: 0)
        self._turn = 0
        self._turns: threading.Thread | None = None
        self._recent: OrderedDict[tuple, _Ask] = OrderedDict()  # answered after their requests stopped waiting

    def ask(self, how: tuple, op: str, arg: str):
        """The answer to (how, op, arg). A request's thread waits ANSWER_S at most: a view that takes two minutes to
        make would otherwise hold one of the WAITS lane's threads for two minutes, and a few of them would stop every
        view of everyone. Past that the request answers E-VIEW-PREPARING (503) while the question goes on being
        answered here; the page asks again and finds the answer kept (`_recent`) or nearly there."""
        from ..serving import account

        who = account().user_id  # whose cache the view is made from: part of every key, here and in the worker
        key = (who, how, op, arg)
        with self._guard:
            kept = self._recent.pop(key, None)
            if kept is not None:
                return kept.result()
            a = self._pending.get(key)
            if a is None:
                if len(self._pending) >= PENDING_MAX or sum(k[0] == who for k in self._pending) >= PENDING_PER_ACCOUNT:
                    raise Unavailable(Msg("E-VIEW-BUSY"))
                a = self._pending[key] = _Ask()
                self._waiting.setdefault(who, deque()).append((key, a))  # the key names the account
                if self._turns is None or not self._turns.is_alive():
                    self._turns = threading.Thread(target=self._take_turns, daemon=True, name="view-turns")
                    self._turns.start()
                self._guard.notify()
        if not a.done.wait(ANSWER_S):
            raise Unavailable(Msg("E-VIEW-PREPARING"))
        return a.result()

    def run(self, who: int, how: tuple, op: str, arg):
        """The answer to (how, op, arg) for account `who`, waited for here (at most BUILD_S once it is this one's turn):
        what a request's thread does through `ask`, and the background making ahead (prebuild_later) directly."""
        with self.lock:
            self.start()
            self.conn.send((tuple((k, os.environ.get(k)) for k in ENV), _settings_now(), who, how, op, arg))
            reply = self.wait(BUILD_S, Word("server.view.building"))
        return self._value(how, reply)

    def _take_turns(self) -> None:
        """Answer the waiting asks one at a time, the accounts taking turns: the next one is the first waiting of the
        account that had its turn longest ago (or never)."""
        while True:
            with self._guard:
                while not self._waiting:
                    self._guard.wait()
                who = min(self._waiting, key=lambda w: self._served.get(w, 0))
                asks = self._waiting[who]
                key, a = asks.popleft()
                if not asks:
                    del self._waiting[who]
                self._turn += 1
                self._served[who] = self._turn
                if len(self._served) > 10_000:  # the ones served longest ago are as if never: forgotten first
                    for old in sorted(self._served, key=self._served.get)[:5_000]:
                        del self._served[old]
            self._answer(key, a)

    def _answer(self, key: tuple, a: _Ask) -> None:
        who, how, op, arg = key
        try:
            a.value = self.run(who, how, op, arg)
        except BaseException as exc:  # noqa: BLE001 — carried to the request that asked (or the one that comes back)
            a.error = exc
        finally:
            with self._guard:
                self._pending.pop(key, None)
                self._recent[key] = a
                while len(self._recent) > RECENT:
                    self._recent.popitem(last=False)
            a.done.set()

    def _value(self, how: tuple, reply: tuple):
        if reply[0] == "ok":
            return reply[1]
        if reply[0] == "refused":  # (the error's kind and its message: an error itself does not pickle)
            raise reply[1](reply[2])
        if reply[0] == "memory":
            raise Unviewable(Msg("E-VIEW-OUTOFMEMORY"))
        failed = Msg("E-VIEW-FAILED", detail=reply[1])
        logs.say(log, failed, about=str(how))
        raise Unviewable(failed)

    def start(self) -> None:
        if self.proc is not None and self.proc.is_alive():
            return
        self.end()
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child,), daemon=True, name="lab2shot-view")
        self.proc.start()
        child.close()
        self.wait(START_S, Word("server.view.starting"))

    def wait(self, seconds: float, doing: str):
        deadline = time.monotonic() + seconds
        while True:
            left = deadline - time.monotonic()
            if _rss_gb(self.proc.pid) > MEMORY_GB:
                self.end()
                raise Unviewable(Msg("E-VIEW-TOOMUCHMEMORY", doing=doing, gb=MEMORY_GB))
            if self.conn.poll(max(0.0, min(0.25, left))):
                try:
                    return self.conn.recv()
                except (EOFError, OSError):
                    self.end()
                    raise Unavailable(Msg("E-VIEW-CRASHED", doing=doing)) from None
            if left <= 0:
                self.end()
                raise Unviewable(Msg("E-VIEW-TIMEOUT", doing=doing, seconds=seconds))

    def end(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.join(5)
        if self.conn is not None:
            self.conn.close()
        self.proc = self.conn = None


def _generation(how: tuple) -> tuple:
    """The generation of each packet `how` names: `created` in its manifest (recomputing the same fingerprint makes
    data/packet.py fresh_dir swap in a new folder and rewrite the manifest, i.e. a new generation).
    Built views must be keyed by (how, generation): View.source holds the USD stage, and keyed by how alone a recomputed
    packet would keep showing old data in this process. The streaming view (points_partial) names a folder that is
    scanned each time and has no generation."""
    from ..data.packet import MANIFEST, packet_dir

    if how[0] == "scene":
        fps = (how[1],)
    elif how[0] == "points":
        fps = (how[1], how[2])
    else:
        return ()
    import json

    stamps: list[object] = []
    for fp in fps:
        try:
            path = packet_dir(fp) / MANIFEST if fp else None
            # generation = `created` in the manifest (the g in page addresses is the same)
            stamps.append(json.loads(path.read_text(encoding="utf-8"))["created"] if path else 0)
        except (OSError, ValueError, KeyError, TypeError):  # the packet does not exist (build reports it itself), the fingerprint is invalid, or the manifest is not one Packet.commit wrote
            stamps.append(0)
    return tuple(stamps)


def _rss_gb(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
    except (OSError, ValueError, IndexError):
        return 0.0


_lanes = [_Lane() for _ in range(LANES)]
_ahead = _Lane()  # making chunks ahead (prebuild_later): a process of its own, never one a page waits on
_ahead_todo: queue.Queue = queue.Queue()
_ahead_guard = threading.Lock()
_ahead_seen: OrderedDict = OrderedDict()
_ahead_thread: threading.Thread | None = None
AHEAD_SEEN = 1000  # views asked to be made ahead, remembered so as not to ask twice
atexit.register(shutdown)


def _lane(how: tuple) -> _Lane:
    return _lanes[zlib.crc32(repr(how).encode()) % len(_lanes)]


# ------------------------------------------------------------------ the worker's side


def _serve(conn) -> None:
    """A worker: makes and keeps views, answers one request at a time until the server closes the pipe."""
    try:
        os.nice(10)
    except OSError:
        pass
    from ..config import settings
    from ..serving import Account, serving
    from .view_data import Refused, build, store_chunk

    views: OrderedDict = OrderedDict()
    env_now = conf_now = None
    conn.send(("ready",))
    while True:
        try:
            env, conf, who, how, op, arg = conn.recv()
        except (EOFError, OSError):
            return
        try:
            if env != env_now:  # another work folder or config: what was kept belongs to the old one
                for k, v in env:
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                settings.cache_clear()
                views.clear()
                env_now, conf_now = env, None
            if conf != conf_now:
                # The server's settings changed (an administrator changed a setting such as 「点云上限」, or this is
                # the first request): override this process's settings with the server's and discard the views already
                # built, since the thinning step is applied when a view is built (view_data.cloud_per_frame /
                # depth_grid) and keeping old views would leave the changed setting without effect. Views are built on
                # demand, so a discarded view is rebuilt on the next request.
                s = settings()
                for k, v in conf:
                    s.command[k] = (v, "server")
                    if k in s.running:
                        s.running[k] = v
                views.clear()
                conf_now = conf
            with serving(Account(who)):  # the asking account's own cache (data/store.py), nobody else's
                key = (who, how, _generation(how))
                view = views.pop(key, None) or build(how)
                views[key] = view
                while len(views) > VIEWS:
                    views.popitem(last=False)
                if op == "describe":
                    out = view.json(arg)
                elif op == "chunks":  # making ahead (prebuild_later): how many chunks there are
                    out = len(view.chunks)
                elif op == "store":  # making ahead: chunk arg[0] kept on disk
                    out = store_chunk(view, how, _generation(how), *arg)
                else:
                    try:
                        out = view.part(*arg)
                    except KeyError:
                        raise Refused(NotFound(Msg("E-VIEW-NOPART", part=arg))) from None
            conn.send(("ok", out))
        except Refused as exc:
            conn.send(("refused", type(exc.error), exc.error.message))
        except MemoryError:
            views.clear()
            conn.send(("memory",))
        except Exception as exc:  # noqa: BLE001 (said to the viewer; the traceback in the server's log)
            traceback.print_exc()
            conn.send(("error", str(exc) or type(exc).__name__))
