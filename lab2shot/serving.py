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


ANYONE = Account(0, all_accounts=True)  # no account is being served: this process's own work (the command line, a
# tool script, a test, a worker). A server request and a farm job always name one.
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
