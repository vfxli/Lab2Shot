"""What happens again and again, told once and counted: the one way the server keeps a flood of the same thing from
filling a record (the admin action log's refusals: access.audit; the suspicious-activity log lines: auth.Watch).

The first of a key within `window` seconds is told at once (`first(key, at, data)`: it writes its record and gives back
what names it, a row id); the ones after it are only counted, and the count is told (`again(handle, count, last)`) at most every
FLUSH_S, by a thread that looks every FLUSH_S, and when the process ends: a count is never left untold because nothing
of the key came again. A key is reserved before its first is told, so two at once never both tell a first."""

from __future__ import annotations

import atexit
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable

FLUSH_S = 10.0  # the most a count waits before it is told
KEPT = 10_000  # keys remembered at once; the oldest go first, their counts told


class Repeats:
    def __init__(self, window: float, first: Callable[[Hashable, float, object], object],
                 again: Callable[[object, int, float], None]) -> None:
        self.window, self.first, self.again = window, first, again
        self.lock = threading.Lock()
        self.seen: OrderedDict[Hashable, list] = OrderedDict()  # key -> [handle, first at, untold, last at]
        _all.append(self)
        _start()

    def happened(self, key: Hashable, data: object = None) -> None:
        """One more of `key`; `data`: what its first tells (the others' is only counted)."""
        t = time.time()
        with self.lock:
            seen = self.seen.get(key)
            if seen is not None and t - seen[1] < self.window:
                seen[2] += 1
                seen[3] = t
                return
            old = self.seen.pop(key, None)
            self.seen[key] = slot = [None, t, 0, t]  # reserved: another of the key meanwhile is counted on it
            gone = [self.seen.popitem(last=False)[1] for _ in range(max(0, len(self.seen) - KEPT))]
        for left in [old, *gone]:
            self._tell(left)
        slot[0] = self.first(key, t, data)

    def flush(self) -> None:
        """Tell every count not told yet (the thread, every FLUSH_S; and when the process ends)."""
        with self.lock:
            due = [s for s in self.seen.values() if s[2]]
        for seen in due:
            self._tell(seen)

    def _tell(self, seen: list | None) -> None:
        if seen is None or seen[0] is None:
            return
        with self.lock:
            count, last, seen[2] = seen[2], seen[3], 0
        if count:
            self.again(seen[0], count, last)


_all: list[Repeats] = []
_thread: threading.Thread | None = None


def _start() -> None:
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_loop, daemon=True, name="repeats")
        _thread.start()
        atexit.register(lambda: [r.flush() for r in _all])


def _loop() -> None:
    while True:
        time.sleep(FLUSH_S)
        for r in list(_all):
            try:
                r.flush()
            except Exception:  # noqa: BLE001 (a count is not worth the thread)
                pass
