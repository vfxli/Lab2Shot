"""Live GPU inventory: one background thread reads nvidia-smi every INVENTORY_S seconds; every other reader
(farm/queue.py's lanes, the admin page) only reads the latest Snapshot. nvidia-smi never runs while the queue's
lock is held: a queue held for an `nvidia-smi` subprocess call would block every lane, light jobs included, for
however long the driver takes to answer, worse under load.

Host is the seam for the future: this machine is LocalHost; a second workstation or a remote render node would be a Host implementation of its own that farm/scheduler/placement.py never has to know
about; it only asks a Host for a Snapshot and to authorize UUIDs. Not built here: today there is exactly one Host,
this machine.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from ... import logs
from ...messages import Msg
from .. import gpus as gpumod

log = logs.get("farm")

INVENTORY_S = 2.0  # nvidia-smi polling interval when idle
# Slow readings reduce the polling rate: while computations run, one `nvidia-smi` query may take seconds to tens of
# seconds (or time out). Spawning a subprocess that takes several seconds every 2 seconds would keep one or two
# nvidia-smi processes pending at all times. Instead of checking whether tasks are running, the duration of a reading
# is itself the signal: the next wait is `last duration x SLOW_FACTOR`, capped at SLOW_MAX_S. When idle a reading
# takes tens of milliseconds and polling stays at every 2 seconds.
SLOW_FACTOR = 10.0
SLOW_MAX_S = 60.0
HOURS_KEPT = 24 * 7  # retention of the hourly average GPU utilisation


@dataclass(frozen=True)
class GpuState:
    """One GPU as the scheduler sees it. `used_mb` is nvidia-smi's memory.used: every program on the card.

    On a card no farm job is running on (the only ones `place()` ever considers) that usage is two things:
    `ours_mb`, what this server's own idle kept-loaded models hold there (engine/resident.py: they always move to
    RAM or end when a job starts on that card, so it is memory the job can have), and the rest, a foreign
    program's. Counting the farm's own kept-loaded models as foreign would make a job needing nearly the whole card wait
    on an idle card until the idle timeout unloaded them."""

    uuid: str
    index: int
    name: str
    short_name: str
    compute_cap: str  # nvidia-smi's ("8.9", "12.0"); "" when it could not be read
    memory_mb: int
    used_mb: int
    utilization: int
    temperature: int
    authorized: bool
    ours_mb: int = 0  # held by this server's idle kept-loaded models: freed for a job that starts here

    @property
    def foreign_mb(self) -> int:
        """Used by something the farm cannot free (another program, or a farm job running right now)."""
        return max(0, self.used_mb - self.ours_mb)

    @property
    def free_mb(self) -> int:
        """What a job starting on this card can have: everything but foreign usage."""
        return max(0, self.memory_mb - self.foreign_mb)

    @property
    def free_gb(self) -> float:
        return self.free_mb / 1024

    def describe(self) -> dict:
        return {"index": self.index, "name": self.name, "short_name": self.short_name, "uuid": self.uuid,
                "memory_mb": self.memory_mb, "used_mb": self.used_mb, "ours_mb": self.ours_mb,
                "utilization": self.utilization, "temperature": self.temperature, "compute_cap": self.compute_cap,
                "authorized": self.authorized}


@dataclass(frozen=True)
class Snapshot:
    """The inventory at one moment (`at`, time.time()); "" (empty) before the first read ever completes."""

    gpus: tuple[GpuState, ...] = ()
    at: float = 0.0

    def get(self, uuid: str) -> GpuState | None:
        return next((g for g in self.gpus if g.uuid == uuid), None)

    def authorized(self) -> tuple[GpuState, ...]:
        return tuple(g for g in self.gpus if g.authorized)


class Host:
    """A place GPU jobs can run. LocalHost (below) is the only one today; a second workstation or a remote render
    node later implements this same interface (farm/scheduler/placement.py's place() takes a Snapshot and never
    cares which Host it came from)."""

    def snapshot(self) -> Snapshot:
        raise NotImplementedError

    def authorized_uuids(self) -> list[str]:
        raise NotImplementedError

    def authorize(self, uuids: list[str]) -> None:
        raise NotImplementedError

    def hourly(self) -> list[dict]:
        """Hourly average utilisation, one entry per hour: `{"hour": Unix seconds // 3600, "average": {uuid: 0-100}}`.
        A host with no records yet returns an empty list (the admin page then shows that there is no data yet)."""
        return []


@dataclass
class LocalHost(Host):
    """This machine's GPUs (nvidia-smi), authorized by UUID (kept in the database, lab2shot/farm/gpus.py) exactly
    as before this module existed. The first snapshot is read synchronously (a caller never sees an empty one);
    afterwards a daemon thread refreshes it every INVENTORY_S seconds.

    `reclaimable`: GPU UUID -> MB this server's own idle kept-loaded models hold there (farm/queue.py passes
    engine/resident.py's Pool.idle_vram_mb); read with each nvidia-smi reading, so the two describe one moment."""

    reclaimable: Callable[[], dict[str, int]] = field(default=lambda: {}, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _snapshot: Snapshot = field(default_factory=Snapshot)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _hour: int | None = field(default=None)  # the hour currently being accumulated (Unix seconds // 3600)
    _sums: dict[str, tuple[float, int]] = field(default_factory=dict)  # uuid -> (sum of utilisation, sample count)

    def __post_init__(self) -> None:
        self.refresh()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-inventory")
        self._thread.start()

    def _loop(self) -> None:
        wait = INVENTORY_S
        while not self._stop.wait(wait):
            if self._stop.is_set():  # the stop signal arrived during the wait: do not start another reading (one reading takes up to 10 s)
                return
            began = time.monotonic()
            try:
                self.refresh()
            except Exception as exc:  # noqa: BLE001 (one unreadable nvidia-smi answer must not end the inventory for good)
                logs.say(log, Msg("W-GPU-INVENTORYFAILED", error=str(exc) or type(exc).__name__))
            took = time.monotonic() - began
            wait = max(INVENTORY_S, min(SLOW_MAX_S, took * SLOW_FACTOR))

    def refresh(self) -> None:
        """Read nvidia-smi now (never call this while holding the queue's lock: it is a subprocess call)."""
        found = gpumod.inventory()
        authed = set(gpumod.authorized())
        ours = self.reclaimable()
        gpus = tuple(
            GpuState(g.uuid, g.index, g.name, g.short_name, g.compute_cap, g.memory_mb, g.used_mb, g.utilization,
                     g.temperature, g.uuid in authed, min(g.used_mb, ours.get(g.uuid, 0)))
            for g in found
        )
        with self._lock:
            self._snapshot = Snapshot(gpus, time.time())
        self._note_hour(gpus)

    # ---------------------------------------------------------------- hourly average utilisation
    def _note_hour(self, gpus: tuple[GpuState, ...]) -> None:
        """Accumulate this reading's utilisation into the current hour's bucket; when the hour changes, write the
        previous hour's average to the database.

        No extra cost: the values come from the nvidia-smi reading already being made (utilization.gpu in gpus.py
        QUERY); this only adds and counts once, and writes a few tens of bytes to the database once per hour."""
        hour = int(time.time() // 3600)
        with self._lock:
            if self._hour is not None and hour != self._hour:
                done = {u: round(total / count) for u, (total, count) in self._sums.items() if count}
                past, self._sums = (self._hour, done), {}
            else:
                past = None
            self._hour = hour
            for g in gpus:
                total, count = self._sums.get(g.uuid, (0.0, 0))
                self._sums[g.uuid] = (total + g.utilization, count + 1)
        if past is not None:
            self._write_hour(*past)

    def _write_hour(self, hour: int, averages: dict[str, int]) -> None:
        from ...database import db

        try:
            kept = [r for r in db().meta("gpus.hourly", []) if isinstance(r, dict) and r.get("hour", 0) > hour - HOURS_KEPT]
            kept.append({"hour": hour, "average": averages})
            db().set_meta("gpus.hourly", kept[-HOURS_KEPT:])
        except Exception:  # losing an hour of statistics must not stop the inventory thread
            pass

    def hourly(self) -> list[dict]:
        """Hourly average GPU utilisation of recent days (one value per card, 0-100): past hours are read from the
        database and the current hour is computed from the samples collected so far, so values are visible right after
        startup."""
        from ...database import db

        try:
            rows = [r for r in db().meta("gpus.hourly", []) if isinstance(r, dict)]
        except Exception:
            rows = []
        with self._lock:
            now, sums = self._hour, dict(self._sums)
        if now is not None and sums:
            rows = [r for r in rows if r.get("hour") != now]
            rows.append({"hour": now, "average": {u: round(t / c) for u, (t, c) in sums.items() if c}, "running": True})
        return sorted(rows, key=lambda r: r.get("hour", 0))[-HOURS_KEPT:]

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snapshot

    def authorized_uuids(self) -> list[str]:
        return gpumod.authorized()

    def authorize(self, uuids: list[str]) -> None:
        """Raises ValueError (gpus.authorize) for a UUID this machine does not have."""
        gpumod.authorize(uuids, gpumod.inventory())
        self.refresh()  # the next snapshot reflects the change at once, not after up to INVENTORY_S

    def stop(self) -> None:
        """Stop, and return only once the thread has actually stopped.

        The wait is bounded by the real maximum duration of one reading (the timeout in gpus.py), not the polling
        interval `INVENTORY_S`: under full load an nvidia-smi reading may still be in progress, and waiting only one
        interval would leave the thread alive into the next test."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=INVENTORY_S + gpumod.QUERY_TIMEOUT_S + 1.0)


@cache
def local_host() -> LocalHost:
    return LocalHost()
