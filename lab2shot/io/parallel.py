"""Worker processes for CPU-heavy per-frame work (colour conversion, overlay rendering).

The web server is multi-threaded, and fork() from a threaded process can
deadlock the child, so pools start their workers from a clean forkserver instead.
Work functions must be importable module-level functions.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor


def process_pool(workers: int = 8) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("forkserver"))


def in_flight(pool, jobs, most: int, ordered: bool = True):
    """Run `jobs` on `pool` with at most `most` in flight at once, yielding each result.

    `jobs`: a sequence of `(key, fn, *args)`; yields `(key, result)`.
    With `ordered` (the default) results come in the order of `jobs`. The jobs do not depend on each other, so the
    results equal those of running them one by one (STMap per-frame resampling and EXR writing rely on this).
    Without it they come as they finish, for callers that put results back by frame number (a layered render).

    The number in flight is bounded, so the memory a shot takes does not grow with its length (a 4K frame is tens of
    MB; hundreds of them cannot be held at once).

    The kind of pool (threads or processes) and any progress reporting are the caller's.
    """
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, wait
    from itertools import islice

    rest = iter(jobs)

    def submit(job):
        key, fn, *args = job
        return pool.submit(fn, *args), key

    if ordered:
        flight = deque(submit(job) for job in islice(rest, most))
        while flight:
            fut, key = flight.popleft()
            got = fut.result()
            for job in islice(rest, 1):
                flight.append(submit(job))
            yield key, got
        return
    pending: dict = {}
    for job in rest:
        fut, key = submit(job)
        pending[fut] = key
        while len(pending) >= most:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                yield pending.pop(f), f.result()
    for f in list(pending):
        yield pending.pop(f), f.result()
