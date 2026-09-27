"""Locks on cache entries, held while one cook produces them (the one implementation: Store.lock, data/store.py).

They exclude other threads of the server and other processes on this machine alike (the queue's jobs on several GPUs,
the GPU tests); a remote host's store replaces the lock implementation without the callers changing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from .store import current


@contextmanager
def exclusive(key: str, on_wait: Callable[[], None] | None = None, check: Callable[[], None] | None = None) -> Iterator[None]:
    """Hold the lock on `key` (a fingerprint or job key). `on_wait` is called once if another holder makes this one
    wait; `check` is called while waiting and may raise to give up (a stopped cook)."""
    with current().lock(key, on_wait, check):
        yield
