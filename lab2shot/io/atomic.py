"""Atomic and durable file writes.

A file is written completely or not at all. Renaming alone protects against a process that dies while writing; after a
system crash or power loss the rename may reach the disk before the data does, leaving an empty file under the final
name. The data is therefore flushed to the disk before the rename, and a completion marker is written only after the
files it vouches for are on the disk (flush_tree)."""

from __future__ import annotations

import errno
import os
import stat
import uuid
from pathlib import Path

# fsync failures that say the file system cannot flush, not that the data is lost: ignored (an odd mount inside a
# worker folder must not fail a finished cook). Any other error, EIO above all, propagates.
_UNSUPPORTED = {errno.EINVAL, errno.EROFS, errno.ENOTSUP, errno.EBADF}


def _fsync_dir(folder: Path) -> None:
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some file systems do not support fsync on a directory
    finally:
        os.close(fd)


def write_text(path: Path, text: str) -> None:
    """`path` holds `text`, whole or not at all. Written aside under a name of this call's own first: two writers of the
    same file at once each rename a complete file of theirs, the last one standing, never one's half into the other's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def put_in_place(part: Path, target: Path) -> None:
    """Rename a file written aside (`part`, complete and closed) to `target`, durably: its data reaches the disk
    before the rename and the rename after it, so `target` is never an empty or partial file after a crash."""
    _fsync_file(str(part))
    os.replace(part, target)
    _fsync_dir(target.parent)


# How many files flush_tree hands to the disk at once. One at a time, every fsync waits for its own journal commit;
# issued together the file system commits them in a few batches. 100 finished 1080p frames (3 MB each), ext4:
# one at a time 0.35 s, sixteen at once 0.11 s. (syncfs, one call for the whole file system, is as fast, but it also
# waits for whatever else is being written on that disk, another cook's frames or an upload.)
FLUSH_THREADS = 16


def _fsync_file(path: str) -> None:
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return  # removed meanwhile (a temporary file) or not readable
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def flush_tree(folder: Path, skip: tuple[str, ...] = ("_partial",)) -> None:
    """Flush every regular file under `folder` and the folders themselves to the disk. Symbolic links, pipes and other
    special files are left alone; folders named in `skip` (temporary content about to be removed) are not descended
    into. Called before a completion marker is written, so that a marker never survives a crash without its data.

    The files are flushed together (FLUSH_THREADS at once), then the folders (their entries: the renames that put the
    files in place); it returns once every one of them is on the disk, and raises the first error of any, so what
    the marker vouches for is exactly what it was when each was flushed one by one."""
    from concurrent.futures import ThreadPoolExecutor

    files, folders = [], []
    for root, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip]
        files += [os.path.join(root, name) for name in names]
        folders.append(Path(root))
    if len(files) < 2:
        for path in files:
            _fsync_file(path)
        for root in folders:
            _fsync_dir(root)
        return
    with ThreadPoolExecutor(min(FLUSH_THREADS, len(files)), thread_name_prefix="l2s-flush") as pool:
        for _ in pool.map(_fsync_file, files):  # map raises the first error once every flush is done
            pass
        for _ in pool.map(_fsync_dir, folders):
            pass


def mark(marker: Path) -> None:
    """Create a completion marker durably, after flush_tree() of what it vouches for."""
    marker.touch()
    _fsync_dir(marker.parent)
