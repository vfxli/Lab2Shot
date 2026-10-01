"""Recent: the one bounded table for answers worth keeping a while (a skeleton read for an editor, the handle data of
the displayed node): the last `size` keys asked for, the least recently used one dropped first. Shared by threads (a
status request and a choices request run on the server's thread pool at once), so every read and write holds one lock
and a read never meets a key half removed; Evaluation's own tables are engine/records.py Memo, which grows with the
evaluation instead."""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Generic, TypeVar

V = TypeVar("V")
_MISSING = object()


class Recent(Generic[V]):
    def __init__(self, size: int) -> None:
        self._size = size
        self._d: OrderedDict[Hashable, V] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, default=None):
        """The value kept for `key` (now the most recent), or `default`."""
        with self._lock:
            value = self._d.get(key, _MISSING)
            if value is _MISSING:
                return default
            self._d.move_to_end(key)
            return value

    def put(self, key: Hashable, value: V) -> V:
        """Keep `value` for `key`, dropping the least recently used keys beyond the size; the value."""
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self._size:
                self._d.popitem(last=False)
            return value

    def get_or(self, key: Hashable, make: Callable[[], V]) -> V:
        """The value kept for `key`, else `make()` kept and returned (made outside the lock: two threads may both
        make it, the later one kept — a cost, never a wrong answer)."""
        value = self.get(key, _MISSING)
        return self.put(key, make()) if value is _MISSING else value


def packet_key(packet) -> tuple:
    """What a table keyed by a packet keys it by: the account reading it, its fingerprint and its generation
    (Packet.created) — a recook of the same fingerprint is read afresh, and another account's packet of that
    fingerprint is never what this one reads. Every Recent table over packets keys by this."""
    from .serving import account

    return account(), packet.fingerprint, packet.created
