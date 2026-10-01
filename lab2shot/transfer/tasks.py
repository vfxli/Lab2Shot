"""Task folders: what one task keeps on disk, the one place that knows their layout.

A task is one compute request a user submitted to the farm (a click on 「计算」, a DCC plugin's or the command line's
job): every cook is one (the viewer only shows what is computed). Each task has a folder of its own under the data location (config.py paths.data_dir, 「数据位置」):

    <数据位置>/tasks/<task id>/
        graph.json          the graph as it was submitted (the one copy the server keeps of it)
        footage/<upload id>/...
                            the uploads its graph reads, as they were when it was submitted: hard links to the
                            account's own uploaded bytes (a real copy where the file system can't link), so the task
                            keeps its footage whatever becomes of the upload afterwards, without a second copy
        output/             what its output nodes collect (one sub-folder per output), unpacked: DCC plugins fetch
                            single files from here
        <zip name>.zip      the same outputs packed, for the browser's download, beside output/ (zip_path)
        logs/               its record and what it said as it went (write_log)

The folder is named by the task id, which the server makes (the farm's job id: 12 hexadecimal characters); nothing a
user typed (a graph's name, a file's name, a group's name) ever becomes part of a path here. Whose a task is, when it
ended and which of its account's groups it is in are its row in the database (tasks: id, user_id, created, ended,
group_key, group_name, group_slot; transfer/groups.py): a task is shown, fetched from and deleted only by its own
account (or an administrator holding data.others), which `owned` answers.

How long a task is kept: 任务保留天数 (tasks.keep_days) days after it ends, then it goes whole: its folder, its row and
the cache references it holds (farm/disk.py). An output is recorded against its task by being written under
output_dir(task) (and its zip at zip_path(task, name)): the folder is the record, nothing else is kept about it, and it
goes with the task.

Files in a task folder are written once and never changed in place: footage is hard-linked, so changing one name's
bytes would change every other name's too (io/files.py link_or_copy; uploads are made read-only as they complete).
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import settings
from ..errors import NotFound
from ..io.files import folder_bytes, stats
from ..messages import Msg

ID = re.compile(r"^[0-9a-f]{12}$")  # a task id as the farm makes it (farm/queue.py Job.id): nothing else names a folder here
GRAPH = "graph.json"
FOOTAGE = "footage"
OUTPUT = "output"
LOGS = "logs"


def root() -> Path:
    """Every task's folder: <数据位置>/tasks."""
    return settings().data_dir / "tasks"


def task_dir(task_id: str) -> Path:
    """The folder of task `task_id`. An id that is not one the farm makes is not a task (NotFound): it never reaches a
    path."""
    if not isinstance(task_id, str) or not ID.match(task_id):
        raise NotFound(Msg("E-TASKDIR-GONE"))
    return root() / task_id


def graph_file(task_id: str) -> Path:
    """The graph the task was submitted with."""
    return task_dir(task_id) / GRAPH


def footage_dir(task_id: str) -> Path:
    """The task's own links to the uploads its graph reads: footage/<upload id>/<the upload's files>."""
    return task_dir(task_id) / FOOTAGE


def output_dir(task_id: str) -> Path:
    """Where the task's outputs are collected, unpacked (one sub-folder per output node's result)."""
    return task_dir(task_id) / OUTPUT


def zip_path(task_id: str, name: str) -> Path:
    """Where the task's packed outputs go: <task folder>/<name>.zip. `name` is made by the server (it holds the user id
    and the task id, never a name a user typed); anything else is refused."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", name or "") or name.startswith("."):
        raise NotFound(Msg("E-TASKDIR-GONE"))
    return task_dir(task_id) / f"{name}.zip"


def logs_dir(task_id: str) -> Path:
    """The task's logs."""
    return task_dir(task_id) / LOGS


def keep_days() -> int:
    """任务保留天数: how many days a task is kept after it ended."""
    return int(settings()["tasks.keep_days"])


def owner(task_id: str) -> int:
    """The account the task belongs to (NotFound: no such task, or it is gone)."""
    from ..database import db

    task_dir(task_id)  # the id's form, before it reaches a query (it is a bound parameter there all the same)
    r = db().row("SELECT user_id FROM tasks WHERE id = ?", (task_id,))
    if r is None:
        raise NotFound(Msg("E-TASKDIR-GONE"))
    return int(r["user_id"])


# ------------------------------------------------------------------ a task's life: made, referencing, ended, gone


GB = 1 << 30
DAY = 86400


def footage_of(graph: dict) -> dict[str, Path]:
    """The uploads a graph reads, as the account being served now has them (transfer/uploads.py: another account's is
    not there): upload id -> its folder. One that is not there (declared and not sent, cleaned, someone else's) is left
    out: its node fails at itself when the task cooks."""
    from . import uploads

    out = {}
    for sid in sorted(uploads.refs_in(graph)):
        folder = uploads.set_folder(sid)
        if folder is not None:
            out[sid] = folder
    return out


def footage_bytes(footage: dict[str, Path]) -> int:
    """What the task's footage takes (each inode once): the one number 单任务上传上限 is held against."""
    return folder_bytes(footage.values())


def upload_limit() -> int:
    """单任务上传上限, in bytes."""
    return int(float(settings()["tasks.upload_gb"]) * GB)


def check_footage(footage: dict[str, Path]) -> None:
    """Refuse a task whose uploads together are above 单任务上传上限 (TooLarge, said in words): held on the server,
    whatever the page or a client checked before."""
    from ..errors import TooLarge

    size, most = footage_bytes(footage), upload_limit()
    if size > most:
        raise TooLarge(Msg("B-TASK-UPLOADTOOBIG", mb=size / 2**20, most=most / GB, most_mb=most / 2**20))


def create(task_id: str, graph_text: str, footage: dict[str, Path]) -> Path:
    """The task's folder: its graph and its footage (every file of each upload it reads, hard-linked; a real copy where
    the file system can't link: io/files.py link_or_copy). Nothing of it is ever changed afterwards. Returns the folder.
    Its row is written by the caller in the same transaction as the job's (record)."""
    from ..io.atomic import write_text
    from ..io.files import link_or_copy
    from ..data.packet import used

    folder = task_dir(task_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_text(graph_file(task_id), graph_text)
    for sid, src in footage.items():
        for f, _st in stats(src):  # every name: two frames of the same bytes are two files of the footage
            target = footage_dir(task_id) / sid / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                link_or_copy(f, target)
        used(src.with_suffix(".json"))  # a task used this upload now: it is kept at least 任务保留天数 from here
    return folder


def discard(task_id: str) -> None:
    """A folder `create` made for a task whose row could not be written: it goes again (it never was a task)."""
    import shutil

    shutil.rmtree(task_dir(task_id), ignore_errors=True)


def record(c, task_id: str, user_id: int, created: float, footage: dict[str, Path], group: dict | None = None) -> None:
    """The task's row and the uploads it reads, in the caller's transaction (`c`: database.Database.write()). `group`:
    the group it belongs to (transfer/groups.py of_graph), kept on its row."""
    g = group or {"key": "", "name": "", "slot": None}
    c.execute("INSERT INTO tasks (id, user_id, created, ended, group_key, group_name, group_slot) VALUES (?, ?, ?, NULL, ?, ?, ?)",
              (task_id, user_id, created, g["key"], g["name"], g["slot"]))
    c.executemany("INSERT OR IGNORE INTO task_uploads (task_id, upload) VALUES (?, ?)", [(task_id, sid) for sid in footage])


def reference(task_id: str, names) -> None:
    """The task computed or reused these cache entries (in its account's cache; data/packet.py base_of names): an
    entry some live task references is never cleaned (farm/disk.py)."""
    from ..database import db

    rows = [(task_id, str(n)) for n in names if n]
    if rows:
        with db().write() as c:
            c.executemany("INSERT OR IGNORE INTO task_cache (task_id, name) SELECT ?, ? WHERE EXISTS "
                          "(SELECT 1 FROM tasks WHERE id = ?)", [(t, n, t) for t, n in rows])


def write_log(task_id: str, record: dict, events: list[dict]) -> None:
    """The task's log as it ended, in logs/: its record (logs/record.json: state, error, the failed node's log, what it
    delivered, how its nodes served it) and what it said as it went (logs/events.jsonl, one event a line)."""
    import json

    from ..io.atomic import write_text

    folder = logs_dir(task_id)
    folder.mkdir(parents=True, exist_ok=True)
    write_text(folder / "record.json", json.dumps(record, ensure_ascii=False, indent=1))
    write_text(folder / "events.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))


def ended(c, task_id: str, at: float) -> None:
    """The task ended at `at` (done, failed or cancelled): kept 任务保留天数 from now. In the caller's transaction."""
    c.execute("UPDATE tasks SET ended = ? WHERE id = ? AND ended IS NULL", (at, task_id))


def is_task(task_id: str) -> bool:
    from ..database import db

    return db().row("SELECT 1 FROM tasks WHERE id = ?", (task_id,)) is not None


def remove(task_id: str) -> int:
    """A task goes whole: its folder, its row, its references (task_cache, task_uploads go with the row), and, when it
    was the last task of its group, the name the group was renamed to (task_group_names: a group is its tasks, so the
    same footage coming back later starts with the automatic name). The only place a task row is deleted (task
    deletion, group deletion, expiry, account removal all come here). Returns what its folder took. The cache entries
    only it referenced are cleaned by the next cleaning (farm/disk.py)."""
    import shutil

    from ..database import db

    folder = task_dir(task_id)
    size = folder_bytes([folder])
    with db().write() as c:
        row = c.execute("SELECT user_id, group_key FROM tasks WHERE id = ?", (task_id,)).fetchone()
        c.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if row is not None and row["group_key"]:
            c.execute("DELETE FROM task_group_names WHERE user_id = ? AND group_key = ? AND NOT EXISTS "
                      "(SELECT 1 FROM tasks WHERE user_id = ? AND group_key = ?)",
                      (row["user_id"], row["group_key"], row["user_id"], row["group_key"]))
    shutil.rmtree(folder, ignore_errors=True)
    changed_folder(task_id)
    return size


def expired(now: float | None = None) -> list[str]:
    """The tasks past 任务保留天数 since they ended."""
    import time

    from ..database import db

    cutoff = (now if now is not None else time.time()) - keep_days() * DAY
    return [r["id"] for r in db().rows("SELECT id FROM tasks WHERE ended IS NOT NULL AND ended < ?", (cutoff,))]


def unended(active: set[str]) -> int:
    """Tasks still marked running that no farm runs any more (the server stopped while they ran, or a held one could
    not be queued again): they end now, so they are kept 任务保留天数 from here and then go. Returns how many."""
    import time

    from ..database import db

    stale = [r["id"] for r in db().rows("SELECT id FROM tasks WHERE ended IS NULL") if r["id"] not in active]
    held = {r["job_id"] for r in db().rows("SELECT job_id FROM held")}
    stale = [t for t in stale if t not in held]
    if stale:
        with db().write() as c:
            c.executemany("UPDATE tasks SET ended = ? WHERE id = ? AND ended IS NULL", [(time.time(), t) for t in stale])
    return len(stale)


STRAY_AFTER_S = 3600  # a folder without a row is stray only once it is this old


def stray_folders(now: float | None = None) -> list[Path]:
    """Folders under tasks/ that no task row names any more (a purged account's, or a submission that stopped between
    its folder and its row). A submission writes the folder first and its row right after (farm/queue.py): a folder
    younger than STRAY_AFTER_S may be one whose row is being written this very moment, and is left alone."""
    import time

    from ..database import db

    base = root()
    if not base.is_dir():
        return []
    known = {r["id"] for r in db().rows("SELECT id FROM tasks")}
    t = time.time() if now is None else now
    out = []
    for d in base.iterdir():
        if not d.is_dir() or d.name in known or d.name.startswith("."):
            continue
        try:
            if t - d.stat().st_mtime < STRAY_AFTER_S:
                continue
        except OSError:  # gone meanwhile
            continue
        out.append(d)
    return out


def referenced(user_id: int) -> set[str]:
    """Every cache entry a live task of the account references (base names, data/packet.py base_of)."""
    from ..database import db

    return {r["name"] for r in db().rows("SELECT DISTINCT c.name FROM task_cache c JOIN tasks t ON t.id = c.task_id "
                                         "WHERE t.user_id = ?", (user_id,))}


def uploads_in_use(user_id: int) -> set[str]:
    """The uploads (ids) some live task of the account reads: they are kept while the task lives."""
    from ..database import db

    return {r["upload"] for r in db().rows("SELECT DISTINCT u.upload FROM task_uploads u JOIN tasks t ON t.id = u.task_id "
                                           "WHERE t.user_id = ?", (user_id,))}


def of_account(user_id: int) -> list[dict]:
    """The account's tasks, newest first: id, created, ended, bytes (its folder, each inode once)."""
    from ..database import db

    return [{"id": r["id"], "created": r["created"], "ended": r["ended"], "state": "进行中" if r["ended"] is None else "已结束",
             "bytes": folder_bytes([task_dir(r["id"])])}
            for r in db().rows("SELECT id, created, ended FROM tasks WHERE user_id = ? ORDER BY created DESC", (user_id,))]


# An ended task's folder does not change any more (its log and outputs are written before it ends), so what it holds,
# inode -> size, is looked at once and remembered here, by task id. Whatever changes an ended task's folder says so
# (`changed_folder`: its unpacked outputs discarded, an account's outputs removed) and `remove` forgets it. A running
# task's folder is looked at every time.
_INODES: dict[str, dict[tuple[int, int], int]] = {}


def _inodes(folder: Path) -> dict[tuple[int, int], int]:
    return {(st.st_dev, st.st_ino): st.st_size for _, st in stats(folder)}


_folders_changed = 0  # how many times an ended task's folder changed (or went) since the server started


def changed_folder(task_id: str) -> None:
    """Something changed an ended task's folder (or removed it): what it holds is looked at again when next asked
    (account_bytes), and the outputs the task history shows may have changed (farm/queue.py listed_version)."""
    global _folders_changed
    _INODES.pop(task_id, None)
    _folders_changed += 1


def folders_changed() -> int:
    """How many times an ended task's folder changed since the server started (changed_folder)."""
    return _folders_changed


def account_bytes(user_id: int, beside: Path | None = None) -> tuple[int, int]:
    """What every task folder of the account takes (each inode once across them: footage is linked between tasks; its
    outputs, folders and zips, are in them): the tasks' part of its quota, exactly io/files.py `folder_bytes` of those
    folders; an ended task's folder is not walked again (_INODES). And what `beside` (the account's uploads) takes
    beyond that: bytes linked into a task are the task's, counted once."""
    from ..database import db

    held: dict[tuple[int, int], int] = {}
    for r in db().rows("SELECT id, ended FROM tasks WHERE user_id = ?", (user_id,)):
        inodes = _INODES.get(r["id"]) if r["ended"] is not None else None
        if inodes is None:
            inodes = _inodes(task_dir(r["id"]))
            if r["ended"] is not None:
                _INODES[r["id"]] = inodes
        held.update(inodes)
    more = {k: size for k, size in _inodes(beside).items() if k not in held} if beside is not None else {}
    return sum(held.values()), sum(more.values())
