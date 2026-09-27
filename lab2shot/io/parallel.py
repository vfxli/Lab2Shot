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
    """将 `jobs` 提交给 `pool` 执行，同时执行中的任务不超过 `most` 个，逐项产出结果。

    `jobs`：`(key, fn, *args)` 序列；产出 `(key, 结果)`。
    `ordered` 为真（默认）时按 `jobs` 的顺序产出。各项结果互不依赖，因此与逐项顺序计算的结果完全相同
    （STMap 逐帧重采样、写 EXR 依赖此性质）。
    `ordered` 为假时按完成顺序产出，适用于结果随后按帧号归位的场景（如分层渲染）。

    同时执行的任务数有上限，因此一段镜头占用的内存不随其长度增长（一帧 4K 图像为数十 MB，数百帧无法同时
    驻留内存）。

    池的类型（线程池或进程池）以及是否报告进度由调用方决定。
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
