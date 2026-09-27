"""Composing the tree a pinned checkout's own scripts expect, **beside** it, out of symlinks.

原始仓库永远不改：several upstream installers clone extra repositories *into* their checkout,
copy replacement files over its files, download weights into it and compile C extensions in
place. Lab2Shot never writes into a pinned checkout, so the tree they expect is built next to
it: every entry is a symlink to the original, and only the folders that must hold something
else — a replaced file, a downloaded weight, a compiler's output — are real.

Used by an extension's own `codebase.py` (imported by its `EnvSpec.build` script and by its
worker). Standard library only, so a build script can import it before anything else is there.

    Pixel3DMM  the tree its `env_paths` expects: four checkouts, replacement files, weights
    Mesh4D     the tree `cd hy3dshape; python infer.py` expects, plus three folders a compiler
               writes into (`copy_tree`)
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def link(at: Path, target: Path) -> None:
    """`at` becomes a symlink to `target`, replacing whatever is there."""
    at.parent.mkdir(parents=True, exist_ok=True)
    if at.is_symlink():
        if os.readlink(at) == str(target):
            return
        at.unlink()
    elif at.exists():
        at.unlink() if at.is_file() else shutil.rmtree(at)
    at.symlink_to(target)


def mirror(src: Path, dst: Path, overrides: dict[str, Path | None]) -> None:
    """`dst` mirrors `src`: every entry becomes a symlink to the one in `src`, except the paths named in
    `overrides` (relative to `src` -> what to link instead, or None to leave that entry to the caller —
    a folder a compiler writes into, put there with `copy_tree`) and the directories leading to them,
    which are real directories mirrored the same way. An override whose folder is not in `src` is
    created. Running it again rebuilds the tree, so an upstream bump is picked up by the next install."""
    dst.mkdir(parents=True, exist_ok=True)
    exact: dict[str, Path | None] = {}
    deeper: dict[str, dict[str, Path | None]] = {}
    for rel, target in overrides.items():
        head, sep, tail = rel.partition("/")
        if sep:
            deeper.setdefault(head, {})[tail] = target
        else:
            exact[head] = target
    for entry in sorted(src.iterdir()) if src.is_dir() else ():
        if entry.name in exact:
            continue
        if entry.name in deeper:
            mirror(entry, dst / entry.name, deeper.pop(entry.name))
        else:
            link(dst / entry.name, entry)
    for name, target in exact.items():
        if target is not None:
            link(dst / name, target)
    for name, rest in deeper.items():  # only what src does not have at all
        mirror(src / name, dst / name, rest)


def copy_tree(src: Path, dst: Path, keep: tuple[str, ...] = ("__pycache__", "*.so", "build", "*.egg-info")) -> None:
    """A real copy of one folder of the checkout, for the ones a compiler writes into (`--inplace`
    builds, `pip install -e .`). Replaced on every compose, so a rebuilt environment never links
    against yesterday's objects; `keep` names what is not copied over."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*keep))
