"""The server's disk, for the administrator: what the cache, the uploads and the deliveries take, and removing what
has not been used for a while. Everything here can be made again: cooked results are cooked again, uploads are sent
again by whoever picks the file, deliveries are cooked again from the cache.

"Last used" is kept by the engine: a cached packet's .complete file, a model result's .worker_complete file and an
upload's manifest get the current time whenever they are used (data/packet.py `used`).

Cleaning also happens by itself, by the administrator's settings (/admin 设置): deliveries past their days always;
uploads and cached results unused for their days, and the cache beyond its size (least recently used first), only
while the queue is empty (`tidy`: after every job, and every hour).
"""

from __future__ import annotations

import os
import json
import shutil
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path

from .. import logs
from ..config import settings
from ..data import packet as packets
from ..data.packet import COMPLETE, Packet, cache_root, packet_dir, worker_done
from ..data.store import current
from ..engine.evaluations import EVALUATIONS
from ..errors import Invalid
from ..messages import Msg
from ..transfer import deliveries, uploads

DAY = 86400

log = logs.get("disk")


@dataclass
class Item:
    path: Path
    size: int  # bytes only this item frees
    last_used: float


def _size(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(base, f)).st_size
            except OSError:
                pass
    return total


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _cache_items() -> list[Item]:
    """Cooked packets (<fp>, <fp>_display: the packet's fingerprint; <fp>_work, <fp>_failed: the node's) and model
    results (<key>_job)."""
    out = []
    root = cache_root()
    for d in root.iterdir() if root.is_dir() else []:
        if not d.is_dir() or d.name.startswith("."):
            continue
        marker = d / "raw" / ".worker_complete" if d.name.endswith("_job") else d / COMPLETE
        out.append(Item(d, _size(d), max(_mtime(marker), _mtime(d))))
    return out


def _upload_items() -> list[Item]:
    """Upload sets of every account, made or only declared (transfer/uploads.py on_disk, where the layout is defined); the
    blobs only they use are freed with them (`clean`)."""
    return [Item(u["path"], u["bytes"], u["used"]) for u in uploads.on_disk()]


def _delivery_items() -> list[Item]:
    """A run's deliveries; their files are hard links into the cache, so only those the cache no longer has count."""
    root = deliveries.root()
    own = lambda d: sum(f.stat().st_size for f in d.rglob("*") if f.is_file() and f.stat().st_nlink == 1)  # noqa: E731
    return [Item(d, own(d), _mtime(d)) for d in root.iterdir() if d.is_dir()] if root.is_dir() else []


def _kept(days_key: str) -> str:
    days = int(settings()[days_key])
    return f"超过 {days} 天没用过的自动删除" if days else "不自动删除"


AREAS = {  # label, what it is and how long it is kept (as set now), its items
    "cache": ("缓存", lambda: f"算好的节点结果和模型的原始结果；删了的节点下次要重新算；{_kept('storage.cache_days')}", _cache_items),
    "uploads": ("上传的素材", lambda: f"用户上传的素材；删了之后用到它的节点图要重新选文件；{_kept('storage.upload_days')}", _upload_items),
    "deliveries": ("待取回的结果", lambda: f"「输出」交出、等用户电脑取回的结果（硬链接到缓存，缓存删了也还在）；超过 {deliveries.keep_days()} 天自动删除", _delivery_items),
}


def usage() -> list[dict]:
    """Per area: size, how many items, how many of them unused for 7 and 30 days."""
    now = time.time()
    out = []
    for area, (label, note, items) in AREAS.items():
        found = items()
        # uploads: each file once, and what is still going up
        size = _size(uploads.root() / "blobs") + _size(uploads.root() / "parts") if area == "uploads" else sum(i.size for i in found)
        idle = {d: sum(i.size for i in found if now - i.last_used > d * DAY) for d in (7, 30)}
        out.append({"id": area, "label": label, "note": note(), "bytes": size, "items": len(found),
                    "idle_7_bytes": idle[7], "idle_30_bytes": idle[30]})
    return out


Guard = Callable[[], AbstractContextManager[bool]]  # farm/queue.py Farm.removing: one removal, and whether it may go ahead


def clean(area: str, days: float, limit_bytes: int | None = None, guard: Guard | None = None) -> dict:
    """Remove the items of `area` unused for more than `days` days and, with `limit_bytes`, the least recently used
    ones beyond that size. `guard` (farm/queue.py Farm.removing) is entered around every single removal and says
    whether the queue is still empty: measuring the disk (`_size` walks every file) can take minutes, and a job
    submitted meanwhile must not lose its inputs, so checking once before the pass is not enough.
    The pass stops at the first removal the guard refuses, and reports what it did until then."""
    if area not in AREAS:
        raise Invalid(Msg("E-DISK-NOAREA", area=area))
    cutoff = time.time() - days * DAY
    items = sorted(AREAS[area][2](), key=lambda i: i.last_used)
    over = sum(i.size for i in items) - limit_bytes if limit_bytes is not None else 0
    removed, freed = 0, 0
    for item in items:
        if item.last_used < cutoff or over > 0:
            with (guard() if guard is not None else nullcontext(True)) as go:
                if not go:
                    break  # a task arrived in the queue: stop this pass; the rest is cleaned next time
                _clean_one(area, item)
            over -= item.size
            removed += 1
            freed += 0 if area == "uploads" else item.size  # uploads: the contents freed are counted below
    if area == "uploads":  # file contents no upload uses any more (uploads.orphan_blobs says which), each under the guard too
        for blob in uploads.orphan_blobs():
            with (guard() if guard is not None else nullcontext(True)) as go:
                if not go:
                    break
                freed += uploads.drop_blob(blob)
    if removed and area in ("cache", "uploads"):  # a plan or an upload identity this evaluated may no longer hold
        EVALUATIONS.bump()
    return {"removed": removed, "bytes": freed}


def _clean_one(area: str, item: Item) -> None:
    """Remove one item of `area` (what `clean` decided to remove), with what hangs off it."""
    if area == "cache":
        packets.remove(item.path.name, "clean")  # single entry point: also removes _display / _work / _failed and notifies the engine
    elif area == "uploads":  # sets/<account>/<id>: the one way an upload leaves (transfer/uploads.py remove_set)
        _, dropped = uploads.remove_set(int(item.path.parent.name), item.path.name)
        if dropped:
            logs.say(log, Msg("I-DISK-STALEPACKETS", sid=item.path.name, count=dropped))
    else:
        shutil.rmtree(item.path, ignore_errors=True)


def tidy(idle: bool, guard: Guard | None = None) -> None:
    """Clean by the administrator's settings: deliveries past their days; with the queue empty (`idle`), uploads and
    cached results unused for their days, and the cache beyond its size. `guard`: re-asked before every removal
    (`clean`), since `idle` was true when this began, not necessarily minutes later."""
    deliveries.prune()
    uploads.prune_parts()  # files that stopped going up long ago
    current().sweep_locks(DAY)  # lock files no one has touched for a day (a held lock is never deleted: try before unlink)
    if not idle:
        return
    s = settings()
    limit = int(s["storage.cache_gb"]) * 10**9
    for area, days, cap in (("uploads", int(s["storage.upload_days"]), None), ("cache", int(s["storage.cache_days"]), limit or None)):
        if days or cap:
            done = clean(area, days or float("inf"), cap, guard)
            if done["removed"]:
                logs.say(log, Msg("I-DISK-AUTOCLEANED", area=AREAS[area][0], count=done["removed"], mb=done["bytes"] / 1e6))


def sweep_incomplete() -> int:
    """When a farm starts: what a process killed mid-write left behind (a packet folder without its `.complete`, a
    worker's raw folder without its `.worker_complete`) is not a result, so it is removed and never kept as cached
    (an incomplete packet is never a packet). A cook's `_work` scratch and a node's `_failed` record are not
    packets: they stay (the failure record is what a reloaded page reads after a restart).

    "Nothing is cooking yet" is true of this process only: another one may be cooking in the same work folder (a
    command-line farm beside the server, two GPU test runs), and an entry it is writing is incomplete exactly because
    it is being written. So each entry is removed under the lock its cook holds (data/locks.py: the fingerprint, a
    worker job's key) and one that is held is left alone; otherwise a second farm on the same work folder would
    remove a `_job` folder a worker is still writing.

    The key here is the folder's own name: for a packet folder that is the output fingerprint
    (`_hash([node fp, port])`, engine/evaluation.py), not the node's. The node lock and the output packet lock are
    different locks: engine/cook.py `_cook_node`, inside the node lock, also holds the lock of each of its output packet
    folders until the node finishes, and that is the lock tried here. If either side changes the lock key, the rule of
    not removing what is being written no longer holds."""
    removed = 0
    root = cache_root()
    for d in root.iterdir() if root.is_dir() else []:
        name = d.name
        if not d.is_dir() or name.startswith(".") or name.endswith(("_work", "_failed")):
            continue
        job = name.endswith("_job")
        done = (lambda d=d: worker_done(d / "raw")) if job else (lambda d=d: Packet.exists(d))
        if not done() and _remove_unless_held(name.removesuffix("_job") if job else name, d, done):
            removed += 1
    return removed


class _Held(Exception):
    """The entry's lock is held by a cook: leave its folder alone."""


def _remove_unless_held(key: str, folder: Path, done: Callable[[], bool]) -> bool:
    """Remove an incomplete cache entry unless a cook holds its lock (`key`: what that cook locks). Re-checked under
    the lock (`done`): it may have been completed while this waited its turn. True when it was removed."""
    from ..data.locks import exclusive

    def busy() -> None:
        raise _Held

    try:
        with exclusive(key, check=busy):
            if done():
                return False
            packets.remove(folder.name, "incomplete")
            return True
    except _Held:
        return False


def clean_incomplete(fp: str) -> bool:
    """Remove the packet `fp` when it is not complete and no other cook is writing it (a cancelled job's partial
    folder): the exclusive lock on the packet's own fingerprint (the one engine/cook.py `_cook_node` holds for
    every output folder it writes, see `sweep_incomplete`) guards against removing what a concurrent identical cook
    is producing. True when it was removed."""
    d = packet_dir(fp)
    if not d.exists() or Packet.exists(d):
        return False
    return _remove_unless_held(fp, d, lambda: Packet.exists(d))
