"""Streaming while computing: a streaming node's worker runs on a thread of the farm, so the node's convert() reads
each frame_{f}.npz as it appears (RawOutput.frames) instead of waiting for the worker to finish.

Threads are started only by the farm, and only through Farm.background (queue.py _spawn): Farm.close() stops them and
waits for them, so no worker thread outlives its work folder. The farm hands this runner to the engine when it builds
the Engine (queue.py _cook passes stream_worker=partial(streaming.run, farm)); the engine puts it into the CookContext,
and WorkerNode.cook calls ctx.run_worker_streaming() for a node with `streams = True`. Layers depend downwards only
(farm -> engine -> nodes): neither the engine nor the nodes import the farm.

`done` is a threading.Event set when the worker ends; `error()` gives its exception once it has ended (None while it
runs or when it succeeded). RawOutput.frames waits for each frame's file and calls a frame missing only once the worker
has ended, so a worker that fails midway re-raises its real error, not a missing frame. `halt()` stops the worker when
the node's convert() fails: the node is through, and gives its card back, only once the worker has ended; otherwise the
same card would be handed to another node while the worker still runs on it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..engine import external
from ..errors import CookCancelled

# (raw folder, done, error(), halt()) — a node that streams reads its frames through RawOutput.frames with the first
# three; halt() stops the worker when the node gives up on it (its convert() failed)
StreamRun = tuple[Path, threading.Event, Callable[[], Exception | None], Callable[[], None]]


def run(farm: Any, ctx: Any, image: Any, *, extra: dict | None = None, inputs: dict | None = None,
        record: dict | None = None) -> StreamRun:
    """Run the worker of a streaming node on a thread of `farm` (Farm.background: close() stops and waits for it) and
    return its raw folder at once, with a `done` event set when the worker ends, an `error()` that gives its
    exception once it has one, and a `halt()` that stops it (engine/external.py Halting: the node's own stop stays
    unset, so the node fails with its own error). The folder is engine/external.py's job_folder (the same key run_job
    works out), so RawOutput.frames reads the very folder the worker writes. A farm already closing starts nothing: the
    cook ends cancelled."""
    folder = external.job_folder(ctx, image, extra=extra, inputs=inputs)
    done = threading.Event()
    errors: list[BaseException] = []
    env = external.Halting(ctx)

    def run_worker() -> None:
        try:  # blocks until the worker ended
            external.run_worker(ctx, image, extra=extra, inputs=inputs, record=record, env=env)
        except Exception as exc:  # noqa: BLE001 — kept, and re-raised by RawOutput.frames / the node after `done`
            errors.append(exc)
        finally:
            done.set()

    if farm.background(run_worker, name=f"farm-stream-{ctx.node_id}") is None:
        errors.append(CookCancelled())
        done.set()
    return folder, done, (lambda: errors[0] if errors else None), env.halt
