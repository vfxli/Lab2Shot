"""Server-side file and folder access for the data code. An externally supplied name (a set still, a dataset folder,
a file named by a sample) is resolved to a path only inside the folder it belongs to; this is the single check, and
every reader of such names goes through it.

    numbered(names)      file name -> frame number, as the sequence reader (「读取序列」, io/sequence.py) reads a folder:
                         the number where the names differ, including TartanAir's 000000_left.png; names without a
                         number are omitted
    link_or_copy(src, dst)  give dst the bytes of src: a hard link on the same file system, a copy (metadata preserved)
                         across file systems; the single implementation of this operation
    inside(base, name)   base / name resolved (symlinks followed); rejected (E-SOURCE-OUTSIDE) when name is empty or
                         absolute, or when the result falls outside base (via .. or a link pointing out of it)
    members(path)        the member names of a zip, a tar (any compression), a folder (relative file paths) or a file
    open_member(path, m) open one member for reading
    is_archive_name(name) whether a name has an archive extension (zip, 7z, rar, tar and its compressed forms)
    pictures(folder, match)  the image files of a folder, sorted, as the sequence reader selects them (its extensions,
                         junk files excluded); `match` restricts results to names matching this wildcard
    unpack(path, folder, keep)  extract the contents into folder (only names accepted by keep()); a tar never writes
                         outside folder (the "data" filter) and zipfile sanitises zip member names. This is the single
                         place where archives are unpacked (installer weights, manually downloaded files, benchmark
                         data scripts).
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
import shutil
import tarfile
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..errors import Invalid
from ..messages import Msg


def inside(base: Path | str, name: str) -> Path:
    text = str(name)
    if not text or "\x00" in text or PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute() or PureWindowsPath(text).drive:
        raise Invalid(Msg("E-SOURCE-OUTSIDE", name=text[:80]))
    root = Path(base).resolve()
    path = (root / text).resolve()
    if path == root or not path.is_relative_to(root):
        raise Invalid(Msg("E-SOURCE-OUTSIDE", name=text[:80]))
    return path


def numbered(names) -> dict[str, int]:
    """Imports only the standard library and lab2shot's messages, so a tool running in an extension's own environment
    can call it with the project on its path."""
    from .sequence import group_names

    return {name: frame for _, _, _, frames in group_names(names).sequences for frame, name in frames.items()}


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
