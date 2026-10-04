"""An account's space on the server: every byte it holds, each once, and what deleting its finished tasks would free.

What an account holds (server/quota.py: its quota is all of it):

    tasks      every task folder of the account (transfer/tasks.py: graph, footage linked from its uploads, outputs, logs)
    uploads    its uploads folder (transfer/uploads.py home_of) beyond what a task folder links already
    cache      its own cache (data/store.py data/cache/<account id>/: packets, what hangs off them, worker results)
               beyond what the two above hold

Bytes are counted by inode: a hard link between a task's footage and an upload, or between two cache folders, is one
set of bytes, counted once, in the first area above that holds it.

What deleting finished tasks frees is worked out from the same walk: every inode gets the set of finished tasks
whose deletion would free it (its *owners*):

    a task folder's file          that task (a task not finished: none can, so it is never freed)
    a cache entry's file          every finished task whose references reach the entry (task_cache, and through the
                                  manifests' deps what those are made of, as farm/disk.py _kept_entries keeps them);
                                  none can free it when a live task or a job still to finish reaches it, or when no
                                  task does (it goes with the cleaning, whoever deletes what)
    an upload's file              none (an upload goes by its own rule, 任务保留天数 after its last use)

An inode held by several holders has the union of their owners, and none can free it when one of them can't. A set of
tasks frees exactly the inodes whose owners are all in the set: `Footprint.frees`. So a task's own figure in the
queue window is what only it frees, and the preview of 「腾出空间」 (`Footprint.plan`, oldest first) adds the bytes two
of its tasks share. Deleting them frees the same bytes: their folders at once (farm/queue.py forget_job), the cache
only they reached right after (farm/disk.py collect_named), the same entries this walk found.

A complete cache entry never changes (data/packet.py: packets are immutable once complete; a worker's raw result
once it is done), so what it holds is read once and remembered, like an ended task's folder (transfer/tasks.py
inodes_of); the rest is walked every time. Measuring is the quota's (server/quota.py): in the background, never
while a poll waits.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..data.packet import COMPLETE, DEPS, MANIFEST, base_of, worker_done
from ..data.store import current
from ..io.files import stats

Key = tuple[int, int]
NONE: frozenset[str] = frozenset()  # an inode no deletion of tasks frees (`_owners` keeps it as None)


@dataclass
class Footprint:
    """One measurement of an account (`measure`)."""

    at: float
    areas: dict[str, int]  # tasks / uploads / cache: bytes, each inode once, in the first area holding it
    sets: dict[frozenset[str], int]  # owners -> bytes that exactly these finished tasks together hold
    finished: list[str]  # the account's finished tasks, oldest first (as 「腾出空间」 takes them)
    entries: dict[str, frozenset[str]] = field(default_factory=dict)  # cache entry (base name) -> its owners

    @property
    def total(self) -> int:
        return sum(self.areas.values())

    def own(self) -> dict[str, int]:
        """Each finished task: what deleting it alone frees (its folder's bytes no one else holds, and the cache only
        it reaches)."""
        out = dict.fromkeys(self.finished, 0)
        for owners, size in self.sets.items():
            if len(owners) == 1:
                (t,) = owners
                out[t] = out.get(t, 0) + size
        return out

    def frees(self, ids) -> int:
        """What deleting these tasks together frees."""
        chosen = set(ids)
        return sum(size for owners, size in self.sets.items() if owners <= chosen)

    def entries_of(self, ids) -> set[str]:
        """The cache entries (base names) deleting these tasks leaves no one reaching: freed with them."""
        chosen = set(ids)
        return {name for name, owners in self.entries.items() if owners and owners <= chosen}

    def plan(self, want: int) -> tuple[list[str], int]:
        """The oldest finished tasks whose deletion frees at least `want` bytes (all of them when even that is not
        enough), and what they free together. Taken strictly oldest first: what a user asked for is 「删除最旧的」."""
        left = {owners: len(owners) for owners in self.sets}
        by_task: dict[str, list[frozenset[str]]] = {}
        for owners in self.sets:
            for t in owners:
                by_task.setdefault(t, []).append(owners)
        chosen, freed = [], 0
        for t in self.finished:
            if freed >= want and chosen:
                break
            chosen.append(t)
            for owners in by_task.get(t, ()):
                left[owners] -= 1
                if left[owners] == 0:
                    freed += self.sets[owners]
        return chosen, freed


# A complete cache entry's folder, what it holds: path -> ((its folder's inode, mtime), inode -> size)
_ENTRIES: dict[str, tuple[tuple[int, int], dict[Key, int]]] = {}
# A complete packet's deps: path of its manifest -> ((inode, mtime), the base names it is made of)
_DEPS: dict[str, tuple[tuple[int, int], tuple[str, ...]]] = {}
_memo_lock = threading.Lock()


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns


def _inodes(folder: Path) -> dict[Key, int]:
    return {(st.st_dev, st.st_ino): st.st_size for _, st in stats(folder)}


def _complete(d: Path) -> bool:
    if d.name.endswith("_job"):
        return worker_done(d / "raw")
    if base_of(d.name) != d.name:  # _display (made bit by bit as frames are looked at), _work, _failed
        return False
    return (d / COMPLETE).exists()


def _entry_inodes(d: Path) -> dict[Key, int]:
    """What one cache folder holds; a complete one read once (`_ENTRIES`)."""
    stamp = _stamp(d)
    key = str(d)
    with _memo_lock:
        kept = _ENTRIES.get(key)
    if stamp is not None and kept is not None and kept[0] == stamp:
        return kept[1]
    found = _inodes(d)
    if stamp is not None and _complete(d):
        with _memo_lock:
            _ENTRIES[key] = (stamp, found)
    return found


def _deps(root: Path, name: str) -> tuple[str, ...]:
    """The base names packet `name` is made of (its manifest's deps: packets and soft), as _kept_entries follows them."""
    path = root / name / MANIFEST
    stamp = _stamp(path)
    if stamp is None:
        return ()
    with _memo_lock:
        kept = _DEPS.get(str(path))
    if kept is not None and kept[0] == stamp:
        return kept[1]
    try:
        deps = json.loads(path.read_text(encoding="utf-8")).get(DEPS) or {}
        found = tuple(str(fp) for fp in (*(deps.get("packets") or ()), *(deps.get("soft") or ())) if fp)
    except (OSError, ValueError, AttributeError, TypeError):
        found = ()
    with _memo_lock:
        _DEPS[str(path)] = (stamp, found)
    return found


def forget(root: Path) -> None:
    """Drop what is remembered of the folders under `root` (an account's cache removed whole)."""
    prefix = str(root) + os.sep
    with _memo_lock:
        for memo in (_ENTRIES, _DEPS):
            for k in [k for k in memo if k.startswith(prefix)]:
                del memo[k]


def measure(user_id: int) -> Footprint:
    """Walk account `user_id`'s task folders, uploads and cache (see the module doc). Takes as long as the files not
    remembered take to stat: on a farm or quota thread, never while a poll waits."""
    from ..database import db
    from ..transfer import tasks, uploads
    from .queue import in_use, started

    queue = started()
    live = queue.active_ids() if queue is not None else set()
    rows = db().rows("SELECT t.id, t.ended, j.submitted FROM tasks t LEFT JOIN jobs j ON j.id = t.id "
                     "WHERE t.user_id = ? ORDER BY COALESCE(j.submitted, t.created)", (user_id,))
    finished = [r["id"] for r in rows if r["ended"] is not None and r["submitted"] is not None and r["id"] not in live]
    deletable = set(finished)

    owners: dict[Key, frozenset[str] | None] = {}  # None: no deletion of tasks frees it
    size: dict[Key, int] = {}
    areas = {"tasks": 0, "uploads": 0, "cache": 0}

    def hold(inodes: dict[Key, int], area: str, by: frozenset[str] | None) -> None:
        for k, n in inodes.items():
            if k not in size:
                size[k] = n
                areas[area] += n
                owners[k] = by
            else:
                was = owners[k]
                owners[k] = None if was is None or by is None else was | by

    for r in rows:
        hold(tasks.inodes_of(r["id"], r["ended"] is not None), "tasks",
             frozenset((r["id"],)) if r["id"] in deletable else None)
    hold(_inodes(uploads.home_of(user_id)), "uploads", None)

    # who reaches each cache entry: a finished task (by its references and what they are made of), or someone no
    # deletion here removes (a live task, a job still to finish)
    root = current().cache_of(user_id)
    refs: dict[str, set[str]] = {}
    for r in db().rows("SELECT c.task_id, c.name FROM task_cache c JOIN tasks t ON t.id = c.task_id WHERE t.user_id = ?",
                       (user_id,)):
        refs.setdefault(r["task_id"], set()).add(r["name"])
    reach: dict[str, set[str]] = {}
    pinned: set[str] = set()

    def walk(start, mark) -> None:
        seen: set[str] = set()
        todo = list(start)
        while todo and len(seen) < 1_000_000:
            name = todo.pop()
            if name in seen or "/" in name or name.startswith("."):
                continue
            seen.add(name)
            mark(name)
            todo += _deps(root, name)

    for t, names in refs.items():
        if t in deletable:
            walk(names, lambda n, t=t: reach.setdefault(n, set()).add(t))
        else:
            walk(names, pinned.add)
    walk(in_use().get(user_id, set()) if queue is not None else (), pinned.add)

    entries: dict[str, frozenset[str]] = {}
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.name.startswith(".") or not d.is_dir():
            continue
        base = base_of(d.name)
        by = None if base in pinned or base not in reach else frozenset(reach[base])
        entries[base] = by or NONE
        hold(_entry_inodes(d), "cache", by)

    sets: dict[frozenset[str], int] = {}
    for k, by in owners.items():
        if by:
            sets[by] = sets.get(by, 0) + size[k]
    return Footprint(at=time.time(), areas=areas, sets=sets, finished=finished, entries=entries)
