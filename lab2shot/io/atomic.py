"""Atomic and durable file writes.

A file is written completely or not at all. Renaming alone protects against a process that dies while writing; after a
system crash or power loss the rename may reach the disk before the data does, leaving an empty file under the final
name. The data is therefore flushed to the disk before the rename, and a completion marker is written only after the
files it vouches for are on the disk (flush_tree)."""

from __future__ import annotations

import errno
import os
import stat
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def flush_tree(folder: Path, skip: tuple[str, ...] = ("_partial",)) -> None:
    """Flush every regular file under `folder` and the folders themselves to the disk. Symbolic links, pipes and other
    special files are left alone; folders named in `skip` (temporary content about to be removed) are not descended
    into. Called before a completion marker is written, so that a marker never survives a crash without its data."""
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip]
        for name in files:
            path = os.path.join(root, name)
            try:
                if not stat.S_ISREG(os.lstat(path).st_mode):
                    continue
                fd = os.open(path, os.O_RDONLY)
            except OSError:
                continue  # removed meanwhile (a temporary file) or not readable
            try:
                os.fsync(fd)
            except OSError as exc:
                if exc.errno not in _UNSUPPORTED:
                    raise
            finally:
                os.close(fd)
        _fsync_dir(Path(root))


def mark(marker: Path) -> None:
    """Create a completion marker durably, after flush_tree() of what it vouches for."""
    marker.touch()
    _fsync_dir(marker.parent)
