"""Reading frames ahead of the GPU and writing results behind it: the one implementation every worker uses (a
worker starts no thread pool of its own).

    FrameReader  frames read a few ahead on threads, in the order the worker says it will ask for them; what was read
                 and not taken yet stays under a byte budget, so a long or large shot never fills the RAM
    Writer       result files written on threads, a bounded number in flight (the loop waits for a slow disk), the
                 first write error raised at the next call, never lost
    thread_map   a function over in-memory items on threads (resizing a stack of frames), results in order

Needs numpy only (the reader's `read` decides what a frame is: an array, a tensor, anything with nbytes or numel).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Generic, TypeVar

from .files import read_frame, save_npz

T = TypeVar("T")
R = TypeVar("R")


def _bytes(item: Any) -> int:
    """What one read frame holds in memory: numpy's nbytes, torch's numel x element size, else nothing counted."""
    if hasattr(item, "nbytes"):
        return int(item.nbytes)
    if hasattr(item, "numel") and hasattr(item, "element_size"):
        return int(item.numel() * item.element_size())
    return 0


class FrameReader(Generic[T]):
    """Frame i of `paths` as read(path) returns it, decoded ahead on threads.

    get(i, upcoming)   frame i; `upcoming` (the indices the worker asks for next, in its order) are read ahead, at most
                       `ahead` of them; default i+1, i+2 ... Reads queued for frames not in that window are dropped.
    take(indices)      several frames at once (a batch): all read in parallel, returned in order
    reader[i], len()   a sequence, for libraries that index frames themselves (upcoming = the next ones)

    `keep`: the last frames taken stay in memory (a model that steps back a frame or two, a cache of a whole normal
    shot); `budget_bytes`: frames read and not yet taken, plus those kept, above it: nothing more is read ahead (the
    frame asked for is always read). An error in `read` (fail() included) is raised where the frame is asked for."""

    def __init__(self, paths: Sequence[Path], read: Callable[[Path], T] = read_frame, *, threads: int = 3,
                 ahead: int = 6, keep: int = 0, budget_bytes: int = 2 << 30):
        self.paths = list(paths)
        self.read, self.ahead, self.keep, self.budget = read, int(ahead), int(keep), int(budget_bytes)
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(threads)))
        self._pending: dict[int, Future] = {}
        self._kept: OrderedDict[int, T] = OrderedDict()
        self._closed = False

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> T:
        return self.get(index)

    def __iter__(self) -> Iterator[T]:
        for i in range(len(self.paths)):
            yield self.get(i)

    def _held(self) -> int:
        done = sum(_bytes(f.result()) for f in self._pending.values() if f.done() and not f.exception())
        return done + sum(_bytes(v) for v in self._kept.values())

    def request(self, indices: Iterable[int]) -> None:
        """Queue reads of these frames (not already read, queued or kept), whatever the budget: they will be asked for."""
        for i in indices:
            if i not in self._pending and i not in self._kept:
                self._pending[i] = self._pool.submit(self.read, self.paths[i])

    def _read_ahead(self, index: int, upcoming: Sequence[int] | None) -> None:
        window = [j for j in (range(index + 1, len(self.paths)) if upcoming is None else upcoming) if j != index]
        window = window[: self.ahead]
        wanted = {index, *window}
        for j in [j for j in self._pending if j not in wanted]:  # the worker went elsewhere: not needed soon
            self._pending.pop(j).cancel()
        for j in window:
            if j in self._pending or j in self._kept:
                continue
            if self._held() > self.budget:
                break
            self._pending[j] = self._pool.submit(self.read, self.paths[j])

    def _taken(self, index: int, item: T) -> T:
        if self.keep:
            self._kept[index] = item
            self._kept.move_to_end(index)
            while len(self._kept) > self.keep:
                self._kept.popitem(last=False)
        return item

    def get(self, index: int, upcoming: Sequence[int] | None = None) -> T:
        if index in self._kept:
            self._kept.move_to_end(index)
            item = self._kept[index]
        else:
            self.request([index])
            item = self._taken(index, self._pending.pop(index).result())
        self._read_ahead(index, upcoming)
        return item

    def take(self, indices: Sequence[int]) -> list[T]:
        indices = list(indices)
        self.request(indices)
        return [self._kept[i] if i in self._kept else self._taken(i, self._pending.pop(i).result()) for i in indices]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._pending.clear()
        self._kept.clear()

    def __enter__(self) -> FrameReader[T]:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class Writer:
    """Result files written behind the loop, on `threads` threads.

    submit(write, *args)  run write(*args, **kwargs) on a thread (an atomic writer: save_npz, save_png, recon.save_frame)
    npz(path, compression, **arrays)  save_npz on a thread

    At most `max_pending` writes are in flight: one more waits for the oldest, so a slow disk never piles a shot of
    arrays up in memory. A write that failed raises at the next submit / close. close() waits for every write
    (idempotent); leaving the with block does too, and a write's error then is raised unless the block is already
    ending with one."""

    def __init__(self, *, threads: int = 2, max_pending: int = 8):
        self._pool = ThreadPoolExecutor(max_workers=max(1, int(threads)))
        self._in_flight: list[Future] = []
        self.max_pending = max(1, int(max_pending))
        self._closed = False

    def _check(self) -> None:
        for f in [f for f in self._in_flight if f.done()]:
            self._in_flight.remove(f)
            f.result()

    def submit(self, write: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        if self._closed:
            raise RuntimeError("the writer is closed")
        self._check()
        while len(self._in_flight) >= self.max_pending:
            self._in_flight.pop(0).result()
        self._in_flight.append(self._pool.submit(write, *args, **kwargs))

    def npz(self, path: Path, compression: int = 1, **arrays: Any) -> None:
        self.submit(save_npz, path, compression, **arrays)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            while self._in_flight:
                self._in_flight.pop(0).result()
        finally:
            self._pool.shutdown(wait=True)

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, kind, error, trace) -> None:
        if kind is None:
            self.close()
            return
        try:
            self.close()
        except BaseException:  # the block's own error is the one to report
            pass


def thread_map(function: Callable[[T], R], items: Iterable[T], *, threads: int = 4) -> list[R]:
    """[function(item) for item in items], on `threads` threads, in order (numpy and OpenCV release the GIL)."""
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as pool:
        return list(pool.map(function, items))
