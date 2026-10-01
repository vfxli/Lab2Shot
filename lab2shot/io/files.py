"""Server-side file and folder access for the data code. An externally supplied name (a set still, a dataset folder,
a file named by a sample) is resolved to a path only inside the folder it belongs to; this is the single check, and
every reader of such names goes through it.

    link_or_copy(src, dst)  give dst the bytes of src: a hard link on the same file system, a copy (metadata preserved)
                         across file systems; the single implementation of this operation
    stats(folder, seen)  every regular file under a folder with its lstat (symbolic links neither followed nor
                         counted); with `seen`, each inode once (hard links to one set of bytes count once)
    folder_bytes(folders)  what these folders take on the disk together, by `stats` across all of them: the single
                         measure of a folder's size (tasks, cache entries, uploads, quotas)
    mtime(path)          a file's modification time, 0.0 when it is not there
    inside(base, name)   base / name resolved (symlinks followed); rejected (E-SOURCE-OUTSIDE) when name is empty or
                         absolute, or when the result falls outside base (via .. or a link pointing out of it)
    members(path)        the member names of a zip, a tar (any compression), a folder (relative file paths) or a file
    open_member(path, m) open one member for reading
    is_archive_name(name) whether a name has an archive extension (zip, 7z, rar, tar and its compressed forms)
    pictures(folder, match)  the image files of a folder, sorted, as the sequence reader selects them (its extensions,
                         junk files excluded); `match` restricts results to names matching this wildcard
    unpack(path, folder, keep)  extract the contents into folder (only names accepted by keep()); a tar never writes
                         outside folder (the "data" filter) and zipfile sanitises zip member names. This is the single
                         place where archives are unpacked (installer weights, manually downloaded files).
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from fnmatch import fnmatch
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..errors import Invalid
from ..messages import Msg


NAME_MAX = 255  # bytes one part of a name may take: what every file system this runs on allows (Linux NAME_MAX, NTFS 255)
PATH_MAX = 1024  # bytes the whole name may take, sub-folders included


def inside(base: Path | str, name: str) -> Path:
    text = str(name)
    if not text or "\x00" in text or PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute() or PureWindowsPath(text).drive:
        raise Invalid(Msg("E-SOURCE-OUTSIDE", name=text[:80]))
    if len(text.encode()) > PATH_MAX or any(len(part.encode()) > NAME_MAX for part in PurePosixPath(text).parts):
        # no file on the disk has such a name: said so, never an OSError of the file system (a 500) when it is looked at
        raise Invalid(Msg("E-FILE-NAMETOOLONG", name=text[:80], most=NAME_MAX, whole=PATH_MAX))
    root = Path(base).resolve()
    path = (root / text).resolve()
    if path == root or not path.is_relative_to(root):
        raise Invalid(Msg("E-SOURCE-OUTSIDE", name=text[:80]))
    return path


def stats(folder: Path, seen: set[tuple[int, int]] | None = None) -> Iterator[tuple[Path, os.stat_result]]:
    """Every regular file under `folder` with its lstat; with `seen` (the inodes already met, shared by several
    walks), each inode once: hard links to one set of bytes count once. Symbolic links, to files or folders, are
    neither followed nor counted."""
    for base, _, names in os.walk(folder):
        for name in names:
            f = os.path.join(base, name)
            try:
                st = os.lstat(f)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if seen is not None:
                if (st.st_dev, st.st_ino) in seen:
                    continue
                seen.add((st.st_dev, st.st_ino))
            yield Path(f), st


def folder_bytes(folders: Iterable[Path]) -> int:
    """What these folders take on the disk together, each inode once across them (a task's footage is linked, not
    copied; an upload's files are links to its blobs)."""
    seen: set[tuple[int, int]] = set()
    return sum(st.st_size for folder in folders for _, st in stats(folder, seen))


def mtime(path: Path) -> float:
    """`path`'s modification time; 0.0 when it is not there."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def link_or_copy(src: Path | str, dst: Path | str) -> None:
    """Give dst the bytes of src: a hard link when both are on the same file system, otherwise a copy with metadata."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def members(path: Path) -> list[str]:
    path = Path(path)
    if path.is_dir():
        return sorted(p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file())
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            return tf.getnames()
    return [path.name]


@contextmanager
def open_member(path: Path, member: str):
    path = Path(path)
    if path.is_dir():
        with (path / member).open("rb") as fh:
            yield fh
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
            yield fh
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf, tf.extractfile(member) as fh:
            yield fh
    else:
        with path.open("rb") as fh:
            yield fh


def unpack(path: Path, folder: Path, keep: Callable[[str], bool] | None = None) -> None:
    path, folder = Path(path), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, folder, dirs_exist_ok=True)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(folder, [n for n in zf.namelist() if keep is None or keep(n)])
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            tf.extractall(folder, [m for m in tf.getmembers() if keep is None or keep(m.name)], filter="data")
    else:
        shutil.copy2(path, folder / path.name)


ARCHIVES = (".zip", ".7z", ".rar", ".tar", ".tgz", ".gz", ".bz2", ".xz")  # .gz covers .tar.gz


def is_archive_name(name: str) -> bool:
    return name.lower().endswith(ARCHIVES)


def pictures(folder: Path, match: str = "") -> list[str]:
    from .sequence import IMAGE_EXTS, is_junk

    return sorted(p.name for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not is_junk(p.name) and (not match or fnmatch(p.name, match)))
