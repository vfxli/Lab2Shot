"""What this process does to itself: the cores it runs on, how its allocator keeps freed memory, and the size of the
thread pools the libraries under the per-frame nodes bring along.

Settings (config.py) only say what is wanted; this module applies it to the running process. Only the process that
owns the running server calls the apply_* functions: serve() when it starts (server/restart.py) and the admin page
after a save that changed one of these settings (server/settings.py). Any other process that reads or saves the
settings (the setup menu, the command line) leaves itself as it is.
"""

from __future__ import annotations

import os

from .config import cpu_cores, settings
from .messages import Msg


# The cores the system gave this process (all of them, or what taskset / a cgroup allows), read once when this module
# is first imported: apply_reservation then narrows the server itself to its share, and a set read after that would
# take the reserved cores off a second time (a smaller budget and fewer cores for every worker, and fewer again at each
# change of 「保留核心数」). A restart gives them back before it becomes the next server (server/restart.py).
GIVEN_CPUS = tuple(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else tuple(range(cpu_cores()))


def worker_cpus() -> list[int]:
    """The core ids a cook may use; the reserved cores at the end are never used for computation.

    Computed in one place: worker processes set their affinity from it (`engine/resident.py`), and extensions with
    their own thread parameter (COLMAP's `num_threads`) receive its count. All worker processes share the same set of
    cores rather than being assigned per task, so the reserved cores stay free however many tasks run, leaving room
    for the browser and other programs. Always taken from GIVEN_CPUS, never from this process's cores now."""
    have = list(GIVEN_CPUS)
    keep = int(settings()["queue.reserved_cores"])  # type: ignore[arg-type]
    return have[: max(1, len(have) - keep)]


def cpu_budget() -> int:
    """Maximum number of threads a cook may use (the length of `worker_cpus()`)."""
    return len(worker_cpus())


def apply_reservation() -> None:
    """Restrict this process (the server itself) to the cores available for computation: light computations (file
    reads, masks, numeric operations) run in the server process, so without this the reserved cores would not be kept
    free. Called at service start and after the setting changes; failing to set the affinity is not an error.

    Set on every thread of the process: an affinity belongs to a thread (0 is only the calling one), and a new thread
    takes its creator's. The queue's threads start before the server applies this, and a changed setting is applied on
    a request's thread; set on that one thread only, every frame the queue computes would still run on all the cores."""
    if not hasattr(os, "sched_setaffinity"):
        return
    cpus, done = set(worker_cpus()), set()
    # listed again until no thread is new: one started during the pass by a thread not yet set took the old cores
    while threads := set(os.listdir("/proc/self/task")) - done:
        for thread in threads:
            try:
                os.sched_setaffinity(int(thread), cpus)
            except OSError:  # a thread that ended meanwhile
                pass
        done |= threads


# 「内存复用」: glibc's allocator (mallopt). By default glibc hands a freed frame buffer straight back to the system
# (a big buffer is mmapped and unmapped when freed, and the heap top is trimmed past 128 KB), so the next frame's
# buffer is fresh memory the kernel has to fault in page by page again. Keeping it makes the per-frame nodes that
# allocate a buffer per frame (image_merge, alpha_merge) markedly faster, with identical pixels (the arithmetic does not
# change). The heap top pad matters most, then big buffers staying out of mmap; capping the arenas bounds what is held
# at no cost in speed. Set with mallopt in this process only, not with MALLOC_*
# environment variables: those would pass on to every extension worker the server starts (their memory is torch's,
# and not ours to tune), and mallopt also lets 「复用内存上限」 change while the server runs.
_M_TRIM_THRESHOLD, _M_TOP_PAD, _M_MMAP_THRESHOLD, _M_ARENA_MAX = -1, -2, -3, -8
# One arena per frame thread: the server passes engine/cook.py FRAME_THREADS at start (this module sits below the
# engine and does not import it). More arenas only hold more memory: with no cap, glibc makes up to 8 per core.
_arenas = 0  # the arena cap this process (the server) set when it started; 0: it did not


def _glibc():
    """This process's C library when it is glibc (mallopt / malloc_trim), else None."""
    import ctypes
    import platform

    if platform.libc_ver()[0] != "glibc":
        return None
    try:
        libc = ctypes.CDLL(None)
        return libc if hasattr(libc, "mallopt") and hasattr(libc, "malloc_trim") else None
    except OSError:
        return None


def apply_memory_reuse(start: bool = False, arenas: int = 0) -> Msg | None:
    """Keep freed memory for the next frame instead of handing it back to the system (「内存复用」): called once when
    the server starts (`start`, with `arenas` its frame threads) and again, in that process only, when 「复用内存上限」
    changes, with the arena count it started with. Each arena keeps up to its share of the limit free at its top,
    takes up to 64 MB more at a time when it grows, and serves buffers up to that size itself (not with mmap).
    「内存复用」 itself is fixed at start (a restart setting): the arena cap only holds for arenas not yet made, and a
    glibc told these values does not go back to adapting its own. The line to log; None when nothing is to say."""
    global _arenas
    s = settings()
    if not start and not _arenas:
        return None  # not the server, or 「内存复用」 was off when it started: nothing of this process to change
    if not s["memory.reuse"]:
        return Msg("I-MEMORY-REUSEOFF")
    libc = _glibc()
    if libc is None:
        return Msg("W-MEMORY-NOGLIBC")
    gb = float(s["memory.reuse_gb"])  # type: ignore[arg-type]
    count = max(1, arenas) if start else _arenas
    each = min(int(gb * 2**30 / count), 2**31 - 1)  # mallopt takes an int
    wanted = ([(_M_ARENA_MAX, count)] if start else []) + [
        (_M_TRIM_THRESHOLD, each), (_M_TOP_PAD, min(each, 64 * 2**20)), (_M_MMAP_THRESHOLD, each)]
    if not all(libc.mallopt(param, value) == 1 for param, value in wanted):
        return Msg("W-MEMORY-NOREUSE")
    if not start:
        libc.malloc_trim(0)  # a lower limit holds at once, not only after the next free
    _arenas = count
    return Msg("I-MEMORY-REUSE", gb=gb, arenas=count, mb=each // 2**20)


# The thread pools of the libraries under the per-frame nodes, in the server process. Per-frame parallelism has one
# place, the engine's frame threads (engine/cook.py FRAME_THREADS, CookContext.each_done), but two libraries bring
# their own pool as well:
# - numpy's OpenBLAS starts one thread per core of the machine when it loads and splits every matrix product big
#   enough over them (a frame's million points times a 3 x 3 rotation is), so each of the frame threads fans out to
#   all the cores again. It runs on one thread here: the nodes that multiply per frame (world_position,
#   depth_normal) get faster and use far less CPU, and no core node gets slower.
# - OpenEXR's pool (OpenImageIO's exr_threads, one thread per core of the machine) compresses and decompresses the
#   blocks of each frame. It stays as large as the cores this process runs on (apply_reservation), not smaller: the
#   frame threads alone do not fill the cores, and nodes that write their frames one after another (Constant) have
#   only this pool; a smaller pool makes image_merge, channel_merge and Constant slower.
# Set by calls in this process, not with OPENBLAS_NUM_THREADS / OPENIMAGEIO_OPTIONS: the command line has loaded numpy
# long before serve(), and the variables would pass on to every process the server starts (extension workers get
# their own limits, extensions/spec.py).


def _openblas():
    """numpy's own OpenBLAS (the wheel carries it as numpy.libs/libscipy_openblas64_-*.so), None when it is not there."""
    import ctypes
    import glob

    import numpy

    found = glob.glob(os.path.join(os.path.dirname(os.path.dirname(numpy.__file__)), "numpy.libs", "libscipy_openblas*.so"))
    try:
        lib = ctypes.CDLL(found[0]) if found else None  # already loaded by numpy: the same library, not a second copy
    except OSError:
        return None
    return lib if hasattr(lib, "scipy_openblas_set_num_threads64_") else None


def apply_thread_pools() -> Msg:
    """Size the libraries' pools in this process (above): OpenBLAS to one thread, OpenEXR's to the cores this process
    runs on. Called by the server after apply_reservation, at start and when 「保留核心数」 changes. The line to log."""
    import OpenImageIO as oiio

    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpu_cores()
    oiio.attribute("exr_threads", cores)
    blas = _openblas()
    if blas is None:
        return Msg("W-THREADS-NOBLAS", exr=cores)
    blas.scipy_openblas_set_num_threads64_(1)
    return Msg("I-THREADS-POOLS", exr=cores)


def apply_changed(changed: list[str]) -> list[Msg]:
    """Apply, in this process (the server), the settings of this module that a save just changed (`changed`, the keys
    Settings.save returns); the lines to log. 「保留核心数」 narrows the server to its new share of cores at once
    (worker processes take it when they start) and OpenEXR's pool follows the cores; 「复用内存上限」 holds at once
    while 「内存复用」 is on in this process."""
    said = []
    if "queue.reserved_cores" in changed:
        apply_reservation()
        said.append(apply_thread_pools())
    if "memory.reuse_gb" in changed and (memory := apply_memory_reuse()):
        said.append(memory)
    return said
