"""The store: the server's work folder and the locks on its cache entries.

    root   the work folder (LAB2SHOT_WORK_DIR): cache, uploads, deliveries, bench media all live under it
    cache  one folder per packet or worker job, named by its fingerprint (data/packet.py): root/cache, or wherever
           the setting 「缓存位置」 points (config.py paths.cache_dir — a data disk in production)
    lock   an flock on a cache key, under cache/.locks: it excludes other threads of the server and other processes on
           this machine alike (the queue's jobs on several GPUs, the GPU tests)

What the store writes down names files relative to the folder it is in (data/packet.py file_ref), so the whole work
folder can move to another path and the cache stays valid. `current()` follows the settings: a test (or a second
server) that points LAB2SHOT_WORK_DIR somewhere else gets the store of that folder.
"""

from __future__ import annotations

import fcntl
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ..config import settings

POLL_S = 0.5


class Store:
    def __init__(self, root: Path, cache: Path | None = None) -> None:
        self.root = Path(root)
        self._cache = Path(cache) if cache else None  # 「缓存位置」指到别处时（config.py paths.cache_dir）

    @property
    def cache(self) -> Path:
        return self._cache or self.root / "cache"

    @property
    def locks_dir(self) -> Path:
        return self.cache / ".locks"

    @contextmanager
    def lock(self, key: str, on_wait: Callable[[], None] | None = None, check: Callable[[], None] | None = None) -> Iterator[None]:
        """Hold the lock on `key` (a fingerprint or job key). `on_wait` is called once if another holder makes this
        one wait; `check` is called while waiting and may raise to give up (a stopped cook)."""
        folder = self.locks_dir
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{key}.lock", "w") as f:
            waited = False
            while True:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not waited and on_wait:
                        on_wait()
                    waited = True
                    if check:
                        check()
                    time.sleep(POLL_S)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def sweep_locks(self, older_than_s: float) -> int:
        """Delete lock files not touched for `older_than_s`. A lock file that is held is flock'd by its holder, so
        the sweep first tries the lock itself (non-blocking) and leaves the file alone when it is taken — flock is
        released when a process ends, so a holder that is merely slow is never deleted from under it."""
        folder = self.locks_dir
        if not folder.is_dir():
            return 0
        now = time.time()
        removed = 0
        for path in folder.glob("*.lock"):
            try:
                if now - path.stat().st_mtime <= older_than_s:
                    continue
                with path.open() as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
        return removed


_store: Store | None = None


def current() -> Store:
    """The store of the work folder as configured now (a changed LAB2SHOT_WORK_DIR gives the store of the new folder)."""
    global _store
    root, cache = settings().work_dir, settings().cache_dir
    if _store is None or _store.root != root or _store.cache != cache:
        _store = Store(root, cache)
    return _store
