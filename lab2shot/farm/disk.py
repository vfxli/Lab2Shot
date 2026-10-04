"""The server's disk: what tasks, caches and uploads take, and cleaning by task.

Everything is kept for a task (transfer/tasks.py) and goes with it:

    tasks       a task's folder (its graph, footage, outputs: folders and zips, logs) goes whole 任务保留天数 after the task ended
                (its row and the cache references it held go with it: the database's tasks, task_cache, task_uploads)
    cache       an account's cache entry (data/store.py: a packet with what hangs off it, a worker job's raw results)
                goes once no live task of the account references it (task_cache: what the task computed or reused) and
                no job still to finish is using it (farm/queue.py in_use): every cook is a task, so nothing else is
                kept
    uploads     an upload no live task reads (task_uploads) goes 任务保留天数 after it was last used; its bytes go when
                nothing links them any more (a task's footage keeps them: the same account needs no second upload)

"Last used" is kept by the engine: a cached packet's .complete file, a model result's .worker_complete file and an
upload's manifest get the current time whenever they are used (data/packet.py `used`; a task's submission for its
uploads). Cleaning runs by itself after every job and every hour (`tidy`). A job reads only its own account's cache and
uploads (farm/queue.py Farm.submit), so what removes cache entries or uploads leaves alone every account that has a job
to finish, and asks the queue before every single removal whether the account still has none (`guard`,
farm/queue.py Farm.removing): a job submitted meanwhile never loses what it needs, and a farm that is never idle is
still cleaned. The administrator's 硬盘 page shows the same areas, measured in the background (`usage`, Farm.disk),
and may clean what is older (`clean`), by the same rules: never what a live task references.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path

from .. import accounts, i18n, logs
from ..data import packet as packets
from ..data.packet import COMPLETE, DEPS, MANIFEST, Packet, base_of, cache_root, packet_dir, worker_done
from ..data.store import current
from ..engine.evaluations import EVALUATIONS
from ..errors import Invalid
from ..io.files import folder_bytes, mtime
from ..messages import Msg
from ..serving import Account, serving
from ..transfer import tasks, uploads

DAY = 86400

log = logs.get("disk")


@dataclass
class Item:
    path: Path
    size: int  # bytes only this item frees
    last_used: float
    user_id: int = 0  # whose it is


def _entry_used(d: Path) -> float:
    marker = d / "raw" / ".worker_complete" if d.name.endswith("_job") else d / COMPLETE
    return max(mtime(marker), mtime(d))


def _cache_items() -> list[Item]:
    """Cooked packets (<fp>, <fp>_display: the packet's fingerprint; <fp>_work, <fp>_failed: the node's) and model
    results (<key>_job), in every account's own cache (data/store.py: data/cache/<account id>/<entry>)."""
    out = []
    store = current()
    for user_id in store.cached_accounts():
        root = store.cache_of(user_id)
        for d in root.iterdir() if root.is_dir() else []:
            if d.is_dir() and not d.name.startswith("."):
                out.append(Item(d, folder_bytes([d]), _entry_used(d), user_id))
    return out


def _upload_items() -> list[Item]:
    """Upload sets of every account, made or only declared (transfer/uploads.py on_disk, where the layout is defined); the
    blobs only they use are freed with them (`clean`)."""
    return [Item(u["path"], u["bytes"], u["used"], u["user_id"]) for u in uploads.on_disk()]


def _task_items() -> list[Item]:
    """Every task's folder (transfer/tasks.py), its last use being when it ended (a running task: now)."""
    from ..database import db

    now = time.time()
    return [Item(tasks.task_dir(r["id"]), folder_bytes([tasks.task_dir(r["id"])]), r["ended"] or now, r["user_id"])
            for r in db().rows("SELECT id, user_id, ended FROM tasks")]


AREAS = {  # its items; its label and note (what it is and how long it is kept, as set now): farm.disk.<area>.*
    "tasks": _task_items,
    "cache": _cache_items,
    "uploads": _upload_items,
}


def worded(areas: list[dict]) -> list[dict]:
    """The measured areas with their label and note in the language now (a measurement is kept, its words are not)."""
    return [{**a, "label": i18n.t(f"farm.disk.{a['id']}.label"),
             "note": i18n.t(f"farm.disk.{a['id']}.note", days=tasks.keep_days())} for a in areas]


def usage() -> list[dict]:
    """Per area: size, how many items, how many of them unused for 7 and 30 days. It walks the whole data disk (minutes
    on a big one): only ever on a farm thread (Farm.disk), never while a request waits."""
    now = time.time()
    out = []
    for area, items in AREAS.items():
        found = items()
        # uploads: each file once, and what is still going up
        size = folder_bytes([uploads.root()]) if area == "uploads" else sum(i.size for i in found)
        idle = {d: sum(i.size for i in found if now - i.last_used > d * DAY) for d in (7, 30)}
        out.append({"id": area, "bytes": size, "items": len(found),
                    "idle_7_bytes": idle[7], "idle_30_bytes": idle[30]})
    return out


# farm/queue.py Farm.removing: one removal of an account's, and whether it may go ahead (the account has had no job to
# finish since the cleaning began)
Guard = Callable[[int], AbstractContextManager[bool]]


def _go(guard: Guard | None, user_id: int):
    return guard(user_id) if guard is not None else nullcontext(True)


# ------------------------------------------------------------------ what is still wanted


def _kept_entries(user_id: int) -> set[str]:
    """The cache entries of the account that stay (base names, data/packet.py base_of): what its live tasks reference
    and what its jobs still to finish use, with every packet those name in turn (a list's items, a camera's plate,
    the packet a file lives in: their manifests' deps), so keeping a result keeps what it is made of."""
    from .queue import in_use

    want = tasks.referenced(user_id) | in_use().get(user_id, set())
    seen: set[str] = set()
    todo = list(want)
    with serving(Account(user_id)):
        while todo and len(seen) < 1_000_000:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            try:
                deps = json.loads((packet_dir(name) / MANIFEST).read_text(encoding="utf-8")).get(DEPS) or {}
            except (OSError, ValueError, Invalid):
                continue
            todo += [str(fp) for fp in (*(deps.get("packets") or ()), *(deps.get("soft") or ()))]
    return seen


COLLECT_MIN_AGE_S = 600.0  # an entry used more recently than this is never collected: something may have made it a
# moment ago and not named it yet (a job references what a node made once the node is through; the viewer's display
# copies are no task's)


def collect(guard: Guard | None = None, older_than_s: float = 0.0) -> dict:
    """Remove every cache entry no live task references and no job still to finish uses, last used more than
    COLLECT_MIN_AGE_S (for the administrator's 清理: `older_than_s`, when longer) ago, in every account's cache (each as
    that account, through data/packet.py remove). An account with a job to finish is left for the next time (`guard`),
    and so is an entry whose lock a writer holds (a worker's job, the viewer's proxies being made, a cook:
    `_remove_unless_held`)."""
    removed, freed = 0, 0
    cutoff = time.time() - max(older_than_s, COLLECT_MIN_AGE_S)
    store = current()
    for user_id in store.cached_accounts():
        keep = _kept_entries(user_id)
        root = store.cache_of(user_id)
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if not d.is_dir() or d.name.startswith(".") or base_of(d.name) in keep or _entry_used(d) >= cutoff:
                continue
            size = folder_bytes([d])
            with _go(guard, user_id) as go, serving(Account(user_id)):
                if not go:
                    break  # a job of the account came in: the rest of its cache waits for the next pass
                if not _remove_unless_held(d.name, "clean", lambda d=d: _entry_used(d) < cutoff):
                    continue
            removed += 1
            freed += size
    return {"removed": removed, "bytes": freed}


def collect_named(user_id: int, names: set[str], guard: Guard | None = None) -> dict:
    """Remove the account's cache entries named (base names) that no live task references and no job still to finish
    uses any more: the ones the tasks just deleted alone referenced (server/quota.py free: what its preview counted,
    freed now rather than with the next cleaning). Unlike `collect` no minimum age: each was named by a task, so
    nothing is making it unnamed this moment. The same guard (nothing of an account with a job to finish: such an
    entry stays for the next cleaning) and the same lock rule (`_remove_unless_held`)."""
    if not names:
        return {"removed": 0, "bytes": 0}
    keep = _kept_entries(user_id)
    root = current().cache_of(user_id)
    removed, freed = 0, 0
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        base = base_of(d.name)
        if not d.is_dir() or d.name.startswith(".") or base not in names or base in keep:
            continue
        size = folder_bytes([d])
        with _go(guard, user_id) as go, serving(Account(user_id)):
            if not go:
                break  # a job of the account came in: the rest waits for the next cleaning
            if not _remove_unless_held(d.name, "clean", lambda base=base: base not in keep):  # the guard: no job came meanwhile
                continue
        removed += 1
        freed += size
    return {"removed": removed, "bytes": freed}


def expire_tasks(now: float | None = None) -> dict:
    """Tasks past 任务保留天数 since they ended go whole (transfer/tasks.py remove), their outputs with them; and task
    folders no row names any more (a purged account's) go too."""
    removed, freed = 0, 0
    for task_id in tasks.expired(now):
        size = tasks.remove(task_id)
        logs.say(log, Msg("I-TASK-EXPIRED", task=task_id, days=tasks.keep_days(), mb=size / 1e6))
        removed += 1
        freed += size
    for folder in tasks.stray_folders():
        size = folder_bytes([folder])
        shutil.rmtree(folder, ignore_errors=True)
        logs.say(log, Msg("I-TASK-STRAY", task=folder.name, mb=size / 1e6))
        freed += size
    return {"removed": removed, "bytes": freed}


def clear_uploads(guard: Guard | None = None, days: float | None = None) -> dict:
    """Uploads no live task reads, not used for 任务保留天数 (or `days`), go (transfer/uploads.py remove_set), and then
    every account's bytes nothing links any more."""
    days = tasks.keep_days() if days is None else days
    cutoff = time.time() - days * DAY
    removed, freed = 0, 0
    reading = {u: tasks.uploads_in_use(u) for u in uploads.accounts()}
    for item in sorted(_upload_items(), key=lambda i: i.last_used):
        if item.last_used >= cutoff or item.path.name in reading.get(item.user_id, set()):
            continue
        with _go(guard, item.user_id) as go:
            if not go:
                continue
            got, dropped = uploads.remove_set(item.user_id, item.path.name)
            if dropped:
                logs.say(log, Msg("I-DISK-STALEPACKETS", sid=item.path.name, count=dropped))
        removed += 1
        freed += got
    for user_id in uploads.accounts():
        for blob in uploads.orphan_blobs(user_id):
            with _go(guard, user_id) as go:
                if not go:
                    break
                freed += uploads.drop_blob(blob)
    if removed:
        EVALUATIONS.bump()
        logs.say(log, Msg("I-DISK-UPLOADSCLEARED", count=removed, days=int(days), mb=freed / 1e6))
    return {"removed": removed, "bytes": freed}


def clean(area: str, days: float, guard: Guard | None = None) -> dict:
    """The administrator's 清理 on the 硬盘 page: what in `area` is older than `days`, by the same rules as the
    cleaning that runs by itself (never a cache entry a live task references or a job uses, never an upload a live task
    reads, a task only once it ended that long ago, nothing of an account with a job to finish). `guard` (farm/queue.py
    Farm.removing) is entered around every single removal."""
    if area not in AREAS:
        raise Invalid(Msg("E-DISK-NOAREA", area=area))
    if area == "cache":
        return collect(guard, days * DAY)
    if area == "uploads":
        return clear_uploads(guard, days)
    if area == "tasks":
        done = {"removed": 0, "bytes": 0}
        cutoff = time.time() - days * DAY
        from ..database import db

        for r in db().rows("SELECT id, user_id FROM tasks WHERE ended IS NOT NULL AND ended < ?", (cutoff,)):
            with _go(guard, r["user_id"]) as go:
                if not go:
                    continue
                done["bytes"] += tasks.remove(r["id"])
                done["removed"] += 1
        return done
    raise Invalid(Msg("E-DISK-NOAREA", area=area))


def tidy(guard: Guard | None = None) -> None:
    """Cleaning by task, by itself (after every job and every hour): tasks past 任务保留天数 go whole, tasks no farm
    runs any more end; the cache entries no live task references and the uploads no task used, of every account that
    has no job to finish (`guard`, asked before every removal: the cleaning takes minutes)."""
    from .queue import farm

    from .queue import trim_finished

    tasks.unended(farm().active_ids())
    done = expire_tasks()
    if trim_finished():  # 每账号保留的已完成任务: the oldest beyond it went, their cache goes with `collect` below
        done["removed"] += 1
    uploads.prune_parts()  # files that stopped going up long ago
    accounts.prune_logins()  # login-log entries and ended sessions kept past their window (accounts.py)
    current().sweep_locks(DAY)  # lock files no one has touched for a day (a held lock is never deleted: try before unlink)
    clear_uploads(guard)
    got = collect(guard)
    if got["removed"]:
        logs.say(log, Msg("I-DISK-CACHECOLLECTED", count=got["removed"], mb=got["bytes"] / 1e6))
    if done["removed"]:
        from .queue import forget_marks

        forget_marks()


def forget_account(user_id: int) -> None:
    """An account purged for good (lab2shot/accounts.py purge): its tasks, its cache and its uploads go now, so an id
    SQLite may give a later account never finds anything of it."""
    from ..database import db

    for r in db().rows("SELECT id FROM tasks WHERE user_id = ?", (user_id,)):
        tasks.remove(r["id"])
    store = current()
    root = store.cache_of(user_id)
    with serving(Account(user_id)):
        for d in sorted(root.iterdir()) if root.is_dir() else []:
            if d.is_dir() and not d.name.startswith("."):
                packets.remove(d.name, "clean")
    shutil.rmtree(root, ignore_errors=True)  # its locks
    uploads.forget_account(user_id)


def sweep_incomplete() -> int:
    """When a farm starts: what a process killed mid-write left behind (a packet folder without its `.complete`, a
    worker's raw folder without its `.worker_complete`) is not a result, so it is removed and never kept as cached
    (an incomplete packet is never a packet). A cook's `_work` scratch and a node's `_failed` record are not
    packets: they stay (the failure record is what a reloaded page reads after a restart).

    "Nothing is cooking yet" is true of this process only: another one may be cooking in the same cache (a server of
    another work folder whose 数据位置 is this one), and an entry it is writing is incomplete exactly because it is
    being written. So each entry is removed under the lock its cook holds (`_remove_unless_held`: the fingerprint, a
    worker job's key) and one that is held is left alone; otherwise that process's `_job` folder a worker is still
    writing would be removed.

    The key here is the folder's own name: for a packet folder that is the output fingerprint
    (`_hash([node fp, port])`, engine/evaluation.py), not the node's. The node lock and the output packet lock are
    different locks: engine/cook.py `_compute`, inside the node lock, also holds the lock of each of its output packet
    folders until the node finishes, and that is the lock tried here. If either side changes the lock key, the rule of
    not removing what is being written no longer holds."""
    removed = 0
    store = current()
    for user_id in store.cached_accounts():
        with serving(Account(user_id)):  # each account's own cache, its own locks
            root = cache_root()
            for d in root.iterdir() if root.is_dir() else []:
                name = d.name
                if not d.is_dir() or name.startswith(".") or name.endswith(("_work", "_failed")):
                    continue
                done = (lambda d=d: worker_done(d / "raw")) if name.endswith("_job") else (lambda d=d: Packet.exists(d))
                if not done() and _remove_unless_held(name, "incomplete", lambda done=done: not done()):
                    removed += 1
    return removed


class _Held(Exception):
    """An entry's lock is held by its writer: leave its folder alone."""


def _locks_of(name: str) -> list[str]:
    """The locks (data/locks.py) whoever writes cache entry `name` holds while writing it: a worker job's folder
    `<key>_job` its key's (engine/external.py), a display copy `<fp>_display` its own name's (data/packet.py produce),
    a node's `_work` and `_failed` its node fingerprint's (engine/cook.py), a packet its own; removing a packet removes
    its display copy with it (data/packet.py remove), so that one's lock too."""
    if name.endswith("_job"):
        return [name.removesuffix("_job")]
    if name.endswith("_display"):
        return [name]
    if base_of(name) != name:  # _work, _failed
        return [base_of(name)]
    return [name, name + "_display"]


def _remove_unless_held(name: str, why: str, still: Callable[[], bool]) -> bool:
    """Remove cache entry `name` (data/packet.py remove, saying `why`) unless a writer holds one of its locks
    (`_locks_of`): each is tried without waiting, and one that is held leaves the entry alone. `still()` is asked
    again holding them (it may have been completed, or come to be wanted, meanwhile). True when it was removed. The
    one rule every removal of a cache entry by cleaning goes through: an entry is never removed under its writer."""
    from ..data.locks import exclusive

    def busy() -> None:
        raise _Held

    try:
        with ExitStack() as held:
            for key in sorted(_locks_of(name)):  # one order everywhere (engine/cook.py takes several the same way)
                held.enter_context(exclusive(key, check=busy))
            if not still():
                return False
            packets.remove(name, why)
            return True
    except _Held:
        return False


def clean_incomplete(fp: str) -> bool:
    """Remove the packet `fp` when it is not complete and no other cook is writing it (a cancelled job's partial
    folder): the exclusive lock on the packet's own fingerprint (the one engine/cook.py `_compute` holds for
    every output folder it writes, see `sweep_incomplete`) guards against removing what a concurrent identical cook
    is producing. True when it was removed."""
    d = packet_dir(fp)
    if not d.exists() or Packet.exists(d):
        return False
    return _remove_unless_held(fp, "incomplete", lambda: not Packet.exists(d))
