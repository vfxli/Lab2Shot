"""Whose work this process is doing right now: one account, or nobody in particular.

Every layer needs to know it — the server answers one account's request, the farm runs one account's job, the engine
reads that account's files, the upload store decides whose an upload is — so it lives here, at the bottom, where all
of them may look. It says nothing about permissions (lab2shot/roles.py) and nothing about uploads (the rule that
another account's upload is not there is transfer/uploads.py's, the one place that decides it): this is only who.

It is a context variable, so it follows the request or the job onto the threads they hand work to (server/wire.py
off_loop, the farm's own threads through `serving(...)` again) and never leaks into another one:

    with serving(Account(user_id)):    # server/access.py Guard for every request, farm/queue.py for every job
        ...                            # whatever runs in here reads that account's files
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    """Whose files are read: one account's own (`user_id`), or every account's (`all_accounts`: an administrator
    holding data.others — the back office looks at a user's graph, job or result as that user has it)."""

    user_id: int
    all_accounts: bool = False


ANYONE = Account(0)  # no account is being served: this process's own work (the command line, a tool script, a
# worker). It reads no account's files: a server request and a farm job always name one, and code that forgot to
# carry theirs onto its thread must find nothing rather than every account's (all_accounts is only ever given on
# purpose: server/access.py account_of, for a login holding data.others).
NOBODY = Account(-1)  # a request without a session: it owns nothing (account ids start at 1)

_serving: ContextVar[Account] = ContextVar("lab2shot_serving", default=ANYONE)


def account() -> Account:
    """Whose work this thread / task is doing now."""
    return _serving.get()


@contextmanager
def serving(who: Account):
    """Do what follows as `who`."""
    token = _serving.set(who)
    try:
        yield who
    finally:
        _serving.reset(token)


# ------------------------------------------------------------------ what a job touches in its account's cache


class Uses:
    """The cache entries one job computed or reused (their names in its account's cache: a packet fingerprint, a worker
    job's `<key>_job`), noted as the engine goes (data/packet.py note). The farm keeps one per job: a task writes them
    down as its references (transfer/tasks.py, the tasks' task_cache), and while any job runs what it noted is never
    cleaned away under it (farm/disk.py)."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._names: set[str] = set()
        self._told: set[str] = set()

    def add(self, name: str) -> None:
        with self._lock:
            self._names.add(name)

    def names(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._names)

    def write_down(self, write) -> None:
        """Hand the names noted and not written down yet to `write` (the task's references: transfer/tasks.py
        reference), outside the lock; they count as written only once it returned, so a write that fails leaves them
        for the next one."""
        with self._lock:
            new = sorted(self._names - self._told)
        if not new:
            return
        write(new)
        with self._lock:
            self._told.update(new)


_uses: ContextVar[Uses | None] = ContextVar("lab2shot_uses", default=None)


def uses() -> Uses | None:
    """The job being done now notes its cache entries here (None: no job, e.g. a status request)."""
    return _uses.get()


@contextmanager
def noting(into: Uses | None):
    """Note the cache entries what follows touches into `into`."""
    token = _uses.set(into)
    try:
        yield into
    finally:
        _uses.reset(token)


def carried(fn):
    """`fn` as it runs on another thread, doing the same account's work (and noting into the same job) as the thread
    that hands it over now: a thread pool's or a new thread's worker starts with nobody's context otherwise, and a
    cache or an upload would then be looked for as nobody's. Every hand-over of work to another thread goes through
    this (the farm's threads, the engine's per-frame threads, the viewer's). The store `using` put in place (data/store.py) goes
    with it the same way: a check cooking in a scratch cache keeps its threads in that cache. So does the language its
    words are said in (lab2shot/i18n current): a job cooking for an English reader says its messages in English on
    every thread of it."""
    from . import i18n
    from .data.store import using_now, using

    who, into, store, lang = account(), uses(), using_now(), i18n.current()

    def run(*args, **kwargs):
        with serving(who), noting(into), using(store), i18n.using(lang):
            return fn(*args, **kwargs)

    return run
