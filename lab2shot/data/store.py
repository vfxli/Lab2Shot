"""The store: the server's work folder, its data location and the locks on its cache entries.

    root   the work folder (LAB2SHOT_WORK_DIR): the database, logs, settings' companions
    data   the data location 「数据位置」 (config.py paths.data_dir, work/data when empty; a data disk in production):
           every account's cache and uploads, and every task's folder, so every hard link between them is on one
           file system
    cache  each account's own: data/cache/<account id>/, one folder per packet or worker job, named by its fingerprint
           (data/packet.py), with what hangs off it (<fp>_display, <fp>_work, <fp>_failed, <key>_job). Whose it is, is
           whose work is being done now (lab2shot/serving.py): the same graph cooked by two accounts is cooked, kept and
           cleaned twice, and neither ever reads, or learns anything of, the other's. A thread doing nobody's work in
           particular (ANYONE: the command line, a tool) has no cache: it names the account it works for
           (serving(Account(id)), `cache_of`) or is refused (NoAccount) rather than guess one
    lock   an flock on a cache key, under the account's cache/.locks: it excludes other threads of the server and
           other processes on this machine alike (the queue's jobs on several GPUs); held exclusive by
           who writes or removes the entry, shared by the cooks reading it

What the store writes down names files relative to the folder it is in (data/packet.py file_ref), so the whole data
location can move to another path and the cache stays valid. `current()` follows the settings: a test (or a second
server) that points LAB2SHOT_WORK_DIR somewhere else gets the store of that folder.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from ..config import settings
from ..errors import MessageError
from ..messages import Msg
from ..serving import account

POLL_S = 0.5


class NoAccount(MessageError, RuntimeError):
    """A cache was asked for while no account's work is being done (a programming error: the caller names the account)."""

    status = 500


def _names(path: Path, f) -> bool:
    """Whether `path` still names the open file `f`: a lock file the sweep deleted while it was open locks nothing,
    since whoever opens the path next gets a new file."""
    try:
        here, held = os.stat(path), os.fstat(f.fileno())
    except FileNotFoundError:
        return False
    return (here.st_dev, here.st_ino) == (held.st_dev, held.st_ino)


class Store:
    def __init__(self, root: Path, data: Path | None = None) -> None:
        self.root = Path(root)
        self.data = Path(data) if data else self.root / "data"

    def cache_of(self, user_id: int) -> Path:
        """Account `user_id`'s own cache: data/cache/<id>. The one place an account's cache root is named."""
        if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
            raise NoAccount(Msg("E-CACHE-NOACCOUNT"))
        return self.data / "cache" / str(user_id)

    @property
    def cache(self) -> Path:
        """The cache of the account whose work is being done now (lab2shot/serving.py)."""
        return self.cache_of(account().user_id)

    def cached_accounts(self) -> list[int]:
        """Every account that has a cache folder (housekeeping walks them one by one, each as that account)."""
        base = self.data / "cache"
        return sorted(int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()) if base.is_dir() else []

    @property
    def locks_dir(self) -> Path:
        return self.cache / ".locks"

    @contextmanager
    def lock(self, key: str, on_wait: Callable[[], None] | None = None, check: Callable[[], None] | None = None,
             shared: bool = False) -> Iterator[None]:
        """Hold the lock on `key` (a fingerprint or job key) in the current account's cache: exclusive (who writes or
        removes the entry), or `shared` with others that share it (who reads it). `on_wait` is called once if another
        holder makes this one wait; `check` is called while waiting and may raise to give up (a stopped cook)."""
        folder = self.locks_dir
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{key}.lock"
        waited = False
        while True:
            f = open(path, "w")  # noqa: SIM115 (closed below, or when the hold ends)
            try:
                while True:
                    try:
                        fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if not waited and on_wait:
                            on_wait()
                        waited = True
                        if check:
                            check()
                        time.sleep(POLL_S)
            except BaseException:
                f.close()
                raise
            if _names(path, f):
                break
            f.close()  # swept (sweep_locks) between the open and the lock: the lock is the file now at `path`
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    def sweep_locks(self, older_than_s: float) -> int:
        """Delete lock files not touched for `older_than_s`, in every account's cache. A lock file that is held is
        flock'd by its holder, so the sweep first tries the lock itself (non-blocking) and leaves the file alone when it
        is taken — flock is released when a process ends, so a holder that is merely slow is never deleted from under
        it. One that opened the file before it went and locks it after takes the lock again on the new file (`lock`)."""
        now = time.time()
        removed = 0
        for user_id in self.cached_accounts():
            folder = self.cache_of(user_id) / ".locks"
            for path in folder.glob("*.lock") if folder.is_dir() else []:
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
# The store `using` put in place, held per context (contextvars, as serving.py holds the account): a thread judging a
# template's routes in a scratch cache (engine/templates.py _empty_cache) must never move the cooks running on other
# threads into that cache, which vanishes with it. A plain global did exactly that.
_using: ContextVar[Store | None] = ContextVar("lab2shot_store", default=None)


def current() -> Store:
    """The store of the work folder as configured now (a changed LAB2SHOT_WORK_DIR gives the store of the new folder),
    or the one `using` put in place in this context."""
    global _store
    if (put := _using.get()) is not None:
        return put
    root, data = settings().work_dir, settings().data_dir
    if _store is None or _store.root != root or _store.data != data:
        _store = Store(root, data)
    return _store


def using_now() -> Store | None:
    """The store `using` put in place in this context, if any (serving.py carried hands it to another thread)."""
    return _using.get()


@contextmanager
def using(store: Store | None) -> Iterator[Store | None]:
    """Work on `store` instead of the configured one while this lasts, in this context only (`lab2shot check` cache: a
    scratch folder, so the check never touches the work folder; a card's routes judged in an empty cache)."""
    token = _using.set(store)  # None: the configured one, as a thread handed work without an override
    try:
        yield store
    finally:
        _using.reset(token)
