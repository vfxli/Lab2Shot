"""THE one way a background thread has something done on the DCC's main thread and waits for the answer: the step is
queued through the host's run_on_main (which returns at once), the background thread waits — never the main thread —
and stops waiting when its job is cancelled or the DCC is quitting (host.is_exiting). The framework (jobs.py) and the
hosts (Maya's finish_export / prepare_import) all use this; a host only provides run_on_main.

Reading animation (a camera, a skeleton) over the frame range is `sample`: SAMPLE_STEP frames per main-thread step, so
the DCC stays responsive between steps and a cancel is heard between them; a host only says how to read some frames."""

from __future__ import annotations

import threading
import traceback

from . import log
from .connection import Cancelled

SAMPLE_STEP = 25  # frames read in one main-thread step


def call(host, fn, cancelled=lambda: False):
    """`fn()` on the main thread, its answer (or its exception) here. Called on the main thread itself (a test, a
    batch run), it simply runs it."""
    if threading.current_thread() is threading.main_thread():
        return fn()
    box, ready = {}, threading.Event()

    def step():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - handed back to the background thread
            box["error"] = exc
            box["trace"] = traceback.format_exc()
        finally:
            ready.set()

    host.run_on_main(step)
    while not ready.wait(0.2):
        if cancelled() or host.is_exiting():
            raise Cancelled()
    if "error" in box:
        log.get().error("main-thread step failed: %s", box.get("trace"))
        raise box["error"]
    return box.get("value")


def sample(host, first: int, last: int, read_frames, cancelled=lambda: False):
    """`read_frames(frames)` on the main thread for frames first … last, SAMPLE_STEP at a time, the answers joined in
    frame order: lists one after another, dicts of lists key by key (each key's rows one after another). None when
    the range is empty."""
    out = None
    for start in range(int(first), int(last) + 1, SAMPLE_STEP):
        frames = list(range(start, min(int(last), start + SAMPLE_STEP - 1) + 1))
        got = call(host, lambda fr=frames: read_frames(fr), cancelled)
        if isinstance(got, dict):
            out = {} if out is None else out
            for key, rows in got.items():
                out.setdefault(key, []).extend(rows)
        else:
            out = ([] if out is None else out) + list(got or [])
    return out
