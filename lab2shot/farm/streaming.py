"""边算边传的并发上下文（stream_transfer_spec D.2、wave-2 D.2）：一个 streaming 节点的 worker 跑在农场自己的线程上，
于是节点的 convert() 能一边算一边用 RawOutput.follow 跟着读已经出现的 frame_{f}.npz，不等 worker 算完。

线程只能开在农场（R047 py.threads_outside_farm），而且只经农场的 Farm.background（queue.py _spawn）：关服时
Farm.close() 停下并等完它，不会有 worker 线程活过它的工作文件夹。农场把这个 runner 通过 Engine 的构造交给引擎
（queue._run 传 stream_worker=partial(streaming.run, farm)），引擎把它放进 CookContext，节点 WorkerNode.cook 对
`streams = True` 的节点调 ctx.run_worker_streaming()。这样分层只向下：farm -> engine -> nodes，引擎和节点都不往上
导入农场。

返回值 `done` 是 worker 结束时置位的 threading.Event；`error()` 在 worker 结束后返回它的异常（跑着或成功时是 None）。
follow 等每一帧的文件出现，worker 结束后才判缺帧，所以 worker 半路出错时 follow 重抛的是真实错误而不是「缺帧」。
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..engine import external
from ..errors import CookCancelled

# (raw folder, done, error()) — a node that streams reads its frames through RawOutput.follow with these
StreamRun = tuple[Path, threading.Event, Callable[[], Exception | None]]


def run(farm: Any, ctx: Any, image: Any, *, extra: dict | None = None, inputs: dict | None = None,
        record: dict | None = None) -> StreamRun:
    """Run the worker of a streaming node on a thread of `farm` (Farm.background: close() stops and waits for it) and
    return its raw folder at once, with a `done` event set when the worker ends and an `error()` that gives its
    exception once it has one. The folder is engine/external.py's job_folder (the same key run_job works out), so follow
    reads the very folder the worker writes. A farm already closing starts nothing: the cook ends cancelled."""
    folder = external.job_folder(ctx, image, extra=extra, inputs=inputs)
    done = threading.Event()
    errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            ctx.run_worker(image, extra=extra, inputs=inputs, record=record)  # blocks until the worker ended
        except Exception as exc:  # noqa: BLE001 — kept, and re-raised by follow / the node after `done`
            errors.append(exc)
        finally:
            done.set()

    if farm.background(run_worker, name=f"farm-stream-{ctx.node_id}") is None:
        errors.append(CookCancelled())
        done.set()
    return folder, done, (lambda: errors[0] if errors else None)
