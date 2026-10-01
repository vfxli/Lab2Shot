"""The server's records in one SQLite database: jobs and how their nodes served them (usage statistics), node timings,
the accounts and their sessions, tasks, licences accepted, the queue's small state,
users' feedback. Big payloads stay files (cache packets, uploads, task folders, feedback's diagnostics and
screenshots); settings stay in config/local.toml.

    work/db/lab2shot.db              the database (with -wal and -shm next to it while it is open)
    work/db/backups/                 online copies: daily, before every migration and every update, and on request; the
                                     newest database.backups (数据库备份份数) are kept. Each has a <name>.files/ folder
                                     beside it with the files rows point to (FILED: work/feedback/), hard links where
                                     the disk allows
    work/db/lab2shot.lock            held by every process that has the database open (restoring needs it free)

Data safety comes first:
  - WAL journal, synchronous=FULL: a transaction that committed survives a crash or power loss; one killed half way
    leaves nothing behind;
  - one write connection per process and one lock around it: every write goes through Database.write(), short
    transactions (BEGIN IMMEDIATE), foreign keys on; every read (rows, row, meta) on its thread's own read-only
    connection, which under WAL never waits for a write (a request checking its session is never held up by a job
    writing its log), except inside a thread's own transaction, where it reads what that transaction wrote so far;
  - at open an integrity check: a damaged database stops the server with what to do; it is never replaced by an empty
    one. A database that went missing while backups exist stops it too;
  - schema versions with explicit migrations (schema.py), each after a backup;
  - backups through SQLite's online backup API, each checked before it counts; restoring is `lab2shot db restore`,
    with the server stopped (the file it replaces is kept).
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .. import logs, workdir
from ..config import settings
from ..errors import MessageError
from ..io.files import link_or_copy
from ..messages import Msg
from . import schema

log = logs.get("db")


def json_text(value) -> str:
    """A JSON column's stored form: compact JSON, non-ASCII kept as characters (the one place a value becomes a JSON
    column — the meta table and every other JSON column go through here)."""
    return json.dumps(value, ensure_ascii=False)


def json_of(text: str | None, default=None):
    """A JSON column read back: `default` when the column is empty (NULL or '')."""
    return json.loads(text) if text else default


LIKE_ESCAPE = "\\"  # what the ESCAPE clause of every LIKE of this project uses
SEARCH_MAX = 200  # characters a person may type into a search box: longer is refused, never scanned


def contains(text: str) -> str:
    """What someone typed into a search box, as the pattern of a `LIKE ? ESCAPE '\\'` (the one place a search becomes
    SQL): its own %, _ and \\ are the characters they typed, never wildcards — otherwise typing "%" would quietly mean
    "everything" and "a_c" would match "abc"."""
    from ..errors import Invalid
    from ..messages import Msg

    typed = text.strip()
    if len(typed) > SEARCH_MAX:
        raise Invalid(Msg("E-SEARCH-TOOLONG", most=SEARCH_MAX))
    for ch in (LIKE_ESCAPE, "%", "_"):
        typed = typed.replace(ch, LIKE_ESCAPE + ch)
    return f"%{typed}%"


FILE = "lab2shot.db"
DAILY_S = 86400


class DatabaseError(MessageError, RuntimeError):
    """The database can't be used as it is; the message says what to do. The server does not start."""

    status = 500


def folder(work_dir: Path | None = None) -> Path:
    return (work_dir or settings().work_dir) / "db"


def backups_folder(work_dir: Path | None = None) -> Path:
    return folder(work_dir) / "backups"


def _restore_hint(work_dir: Path) -> Msg:
    found = sorted(backups_folder(work_dir).glob("*.db"))
    if not found:
        return Msg("N-DB-NOBACKUP")
    return Msg("N-DB-RESTOREHINT", backup=str(found[-1]), name=found[-1].name)


def _fsync(path: Path) -> None:
    """On the disk, not only in the machine's cache (a file, or a folder's entries)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


FILED = ("feedback",)  # work/<name>/: files the database's rows point to (feedback screenshots), kept with every backup


def _files_of(backup: Path) -> Path:
    """Where a backup keeps the files its rows point to (work/feedback/...): next to it, its name with .files for .db."""
    return backup.with_suffix(".files")


def _keep_files(work_dir: Path, backup: Path) -> None:
    """The files the rows point to (user feedback's screenshots and diagnostics), kept with the backup: hard links
    (they are written once and never changed, so a link costs nothing and stays what it was)."""
    for name in FILED:
        src = work_dir / name
        for f in src.rglob("*") if src.is_dir() else []:
            if f.is_file():
                dst = _files_of(backup) / name / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(f, dst)


def check_file(path: Path) -> Msg | str:
    """Why the database at `path` can't be trusted (a message; "" it can): SQLite's integrity and foreign-key checks."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            found = [r[0] for r in conn.execute("PRAGMA integrity_check")]
            if found != ["ok"]:
                return Msg("E-DB-INTEGRITY", detail="; ".join(found[:5]))
            if conn.execute("PRAGMA foreign_key_check").fetchone():
                return Msg("E-DB-BADREFS")
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        return Msg("E-DB-UNREADABLE", detail=str(exc))
    return ""


def _version_of(path: Path) -> int:
    """The schema version of the database at `path`, read without opening it for writing."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _checked(problem: Msg | str) -> dict:
    """The last check, as the admin page shows it: ok, its text ("ok") and its code."""
    return {"at": time.time(), "ok": not problem, "detail": problem.text if problem else "ok", "code": problem.code if problem else ""}


BEFORE_KEPT = 5  # backups taken before a change (the database upgraded: before-vN; the program updated: before-update)


def _before_change(backup: Path) -> bool:
    """Was this backup taken before a change (its reason, the last part of its name, starts with before-)?"""
    return backup.stem.split("-", 3)[-1].split("~", 1)[0].startswith("before-")


def _backup_file(path: Path, work_dir: Path, reason: str) -> Path:
    """A checked copy of the database at `path` in the backups folder (SQLite's backup API: one consistent snapshot of
    what has committed, while a writer may go on), with the files its rows point to; then the oldest go: of the ones
    taken before a change, beyond BEFORE_KEPT; of the others (daily, manual), beyond 数据库备份份数. Counted apart, so
    the daily ones never push out the copy from before an upgrade."""
    target = backups_folder(work_dir) / f"lab2shot-{time.strftime('%Y%m%d-%H%M%S')}-{reason}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    n = 1
    while target.exists():
        n += 1
        target = target.with_name(f"{target.stem.rsplit('~', 1)[0]}~{n}.db")
    partial = target.with_suffix(".partial")
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    dst = sqlite3.connect(partial)
    try:
        # a new file of its own: no journal of its own, one fsync once it is whole (below), not one per page run
        dst.execute("PRAGMA journal_mode=OFF")
        dst.execute("PRAGMA synchronous=OFF")
        src.backup(dst)
        # the backup copies the source's page 1, WAL mode with it: back to one file, or checking it below would leave
        # a -wal and -shm beside it that no longer match its name once it is renamed
        dst.execute("PRAGMA journal_mode=DELETE")
    finally:
        dst.close()
        src.close()
    if problem := check_file(partial):
        partial.unlink(missing_ok=True)
        raise DatabaseError(Msg("E-DB-BACKUPFAILED", reason=problem))
    _fsync(partial)
    partial.replace(target)
    _fsync(target.parent)
    _keep_files(work_dir, target)
    kept = sorted(backups_folder(work_dir).glob("*.db"))
    for before, keep in ((False, int(settings()["database.backups"])), (True, BEFORE_KEPT)):
        for old in [b for b in kept if _before_change(b) == before][:-keep]:
            old.unlink()
            shutil.rmtree(_files_of(old), ignore_errors=True)
    logs.say(log, Msg("I-DB-BACKEDUP", name=target.name))
    return target


class Database:
    """One database file, opened once per process."""

    def __init__(self, work_dir: Path, upgrade: bool = False) -> None:
        """`upgrade`: an older database may be migrated (only the server starting and `lab2shot db upgrade` say so,
        and only while no other process has it open); any other opener refuses it (E-DB-OLDER)."""
        workdir.own(work_dir)  # never another checkout's work folder (lab2shot/workdir.py)
        self.work_dir = work_dir
        self.path = folder(work_dir) / FILE
        self._lock = threading.RLock()
        self._depth = 0  # transactions nested in this thread (the inner ones join the outer)
        self.checked: dict = {}
        self.conn: sqlite3.Connection | None = None  # the one writer
        self._writer = 0  # the thread inside a transaction now (its reads see what it wrote so far)
        # thread -> its read-only connection and the lock a read holds on it (uncontended: only close() takes it too, so a
        # connection is never closed under a read in progress, which crashes SQLite)
        self._readers: dict[int, tuple[sqlite3.Connection, threading.Lock]] = {}
        self._readers_lock = threading.Lock()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._holder = open(self.path.with_suffix(".lock"), "a")  # noqa: SIM115 (held while open)
        fcntl.flock(self._holder, fcntl.LOCK_SH)
        try:
            self._open(upgrade)
        except BaseException:
            self.close()  # refused: nothing of this process keeps the file or its lock
            raise

    def _open(self, upgrade: bool) -> None:
        fresh = not self.path.exists()
        if fresh and any(backups_folder(self.work_dir).glob("*.db")):
            raise DatabaseError(Msg("E-DB-MISSING", path=str(self.path), hint=_restore_hint(self.work_dir)))
        version = 0
        if not fresh:
            problem = check_file(self.path)
            self.checked = _checked(problem)
            if problem:
                raise DatabaseError(Msg("E-DB-DAMAGED", path=str(self.path), reason=problem, hint=_restore_hint(self.work_dir)))
            version = _version_of(self.path)
        # everything that refuses happens before the file is opened for writing: a refused database is left exactly
        # as it was, byte for byte
        if version > schema.VERSION:
            raise DatabaseError(Msg("E-DB-NEWER", path=str(self.path), version=version, known=schema.VERSION))
        upgrading = schema.FIRST <= version < schema.VERSION  # records to keep
        if upgrading:
            # only on purpose (the server starting, `lab2shot db upgrade`), only with the database to itself (a server
            # still running on it would go on with code that does not know the new version), a copy first
            if not upgrade:
                raise DatabaseError(Msg("E-DB-OLDER", path=str(self.path), version=version, known=schema.VERSION))
            try:
                fcntl.flock(self._holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DatabaseError(Msg("E-DB-UPGRADEBUSY", path=str(self.path), version=version, known=schema.VERSION)) from None
        try:
            self.conn = self._connect(write=True)
            if upgrading:
                self.backup(f"before-v{version + 1}")
            self._migrate(self.version)
        finally:
            if upgrading:
                fcntl.flock(self._holder, fcntl.LOCK_SH)
        if version == 0:
            self.checked = _checked("")
            self.backup("created")

    # ------------------------------------------------------------------ the one path in and out

    def _connect(self, write: bool) -> sqlite3.Connection:
        """A connection with this database's settings, every one alike: the writer (it makes the journal WAL), or a
        thread's reader (read-only: it can't write, and under WAL it reads the last commit without waiting)."""
        if write:
            conn = sqlite3.connect(self.path, autocommit=True, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
        else:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, autocommit=True, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        for pragma in ("synchronous=FULL", "foreign_keys=ON", "busy_timeout=30000"):
            conn.execute(f"PRAGMA {pragma}")
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        """A short transaction: everything in it is kept, or nothing (an error rolls it back). Nested in the same
        thread, the inner one joins the outer."""
        with self._lock:
            if self._depth:
                self._depth += 1
                try:
                    yield self.conn
                finally:
                    self._depth -= 1
                return
            self.conn.execute("BEGIN IMMEDIATE")
            self._depth, self._writer = 1, threading.get_ident()
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except BaseException:
                self._roll_back()
                raise
            finally:
                self._depth, self._writer = 0, 0

    def _roll_back(self) -> None:
        """End a failed transaction, the COMMIT itself among the failures (a deferred constraint, a full disk): the
        writer never stays inside it, or every later write would be refused. SQLite may have ended it already; a
        rollback that fails too says nothing the first error has not, so it never takes that error's place."""
        if self.conn.in_transaction:
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass

    def _reader(self) -> tuple[sqlite3.Connection, threading.Lock]:
        """This thread's read-only connection and its lock (made the first time it reads; those of threads that ended
        go). Once the database is closed there is none (E-DB-CLOSED)."""
        me = threading.get_ident()
        with self._readers_lock:
            if self._closed:
                raise DatabaseError(Msg("E-DB-CLOSED", path=str(self.path)))
            found = self._readers.get(me)
            if found is None:
                alive = {t.ident for t in threading.enumerate()}
                for gone in [t for t in self._readers if t not in alive]:
                    conn, lock = self._readers.pop(gone)
                    with lock:
                        conn.close()
                found = self._readers[me] = (self._connect(write=False), threading.Lock())
            return found

    def rows(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        """A read: on this thread's own connection, never waiting for a write; inside this thread's own transaction,
        on the writer (what it wrote so far). A read the database's closing overtook is refused (E-DB-CLOSED)."""
        if self._writer == threading.get_ident():
            return self.conn.execute(sql, params).fetchall()
        conn, lock = self._reader()
        with lock:
            if self._closed:
                raise DatabaseError(Msg("E-DB-CLOSED", path=str(self.path)))
            return conn.execute(sql, params).fetchall()

    def row(self, sql: str, params: tuple | dict = ()) -> sqlite3.Row | None:
        found = self.rows(sql, params)
        return found[0] if found else None

    def meta(self, key: str, default=None):
        r = self.row("SELECT value FROM meta WHERE key = ?", (key,))
        return json_of(r["value"]) if r else default

    def set_meta(self, key: str, value) -> None:
        with self.write() as c:
            c.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                      (key, json_text(value)))

    # ------------------------------------------------------------------ versions

    @property
    def version(self) -> int:
        return self.rows("PRAGMA user_version")[0][0]

    def _migrate(self, version: int) -> None:
        for to, what, sql in schema.pending(version):
            with self.write() as c:
                c.executescript(sql)
                c.execute(f"PRAGMA user_version = {to}")
            logs.say(log, Msg("I-DB-UPGRADED", version=to, what=what))

    # ------------------------------------------------------------------ backups

    def backup(self, reason: str) -> Path:
        """An online copy (SQLite's backup API: consistent while the server runs), checked before it counts; then
        the oldest beyond 数据库备份份数 (database.backups) go."""
        target = _backup_file(self.path, self.work_dir, reason)
        self.set_meta("backup.last", {"at": time.time(), "file": target.name, "reason": reason})
        return target

    def backup_if_due(self) -> Path | None:
        last = self.meta("backup.last")
        if last is None or time.time() - last["at"] >= DAILY_S:
            return self.backup("daily")
        return None

    def check(self) -> dict:
        """Check the database now (the page's 检查完整性): the same check as at open (check_file), on a connection of
        its own: it never holds the writer, and writes go on while it reads."""
        self.checked = _checked(check_file(self.path))
        return self.checked

    def status(self) -> dict:
        """For the admin page: where, how big, which version, the last check and backup, the backups kept."""
        size = sum(p.stat().st_size for p in self.path.parent.glob(FILE + "*") if p.is_file())
        kept = sorted(backups_folder(self.work_dir).glob("*.db"), reverse=True)
        return {"path": str(self.path), "bytes": size, "version": self.version, "checked": self.checked,
                "last_backup": self.meta("backup.last"), "keep": int(settings()["database.backups"]),
                "backups": [{"name": p.name, "bytes": p.stat().st_size, "at": p.stat().st_mtime} for p in kept]}

    def close(self) -> None:
        """Close every connection: each reader once the read in progress on it (another thread's) has ended; reads after
        this are refused (E-DB-CLOSED)."""
        with self._readers_lock:
            self._closed = True
            for conn, lock in self._readers.values():
                with lock:
                    conn.close()
            self._readers.clear()
        with self._lock:
            if self.conn is not None:
                self.conn.close()
            self._holder.close()


# ------------------------------------------------------------------ the database of this process


_open: dict[Path, Database] = {}
_open_lock = threading.Lock()


def db(upgrade: bool = False) -> Database:
    """The database of the current work folder, opened (checked) the first time. An older database is brought up to
    date only with `upgrade` (the server starting, `lab2shot db upgrade`); otherwise it is refused (E-DB-OLDER)."""
    work = settings().work_dir
    with _open_lock:
        found = _open.get(work)
        if found is None:
            found = _open[work] = Database(work, upgrade)
        return found


def close_all() -> None:
    with _open_lock:
        for d in _open.values():
            d.close()
        _open.clear()


def backup_stopped(reason: str) -> Path:
    """Back up a stopped installation before upgrading, including a schema older than this program knows.

    The ownership guard and exclusive database lock apply; no writer is opened and no migration runs.
    """
    work = settings().work_dir
    workdir.own(work)
    path = folder(work) / FILE
    with open(path.with_suffix(".lock"), "a") as holder:
        try:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DatabaseError(Msg("E-DB-INUSE")) from None
        return _backup_file(path, work, reason)


def restore(name: str, work_dir: Path | None = None) -> Path:
    """Put a backup in place of the database (`lab2shot db restore`): the server must be stopped. The backup is
    checked first; the database it replaces (and its journal) is kept next to it as lab2shot.db.replaced-<time>.
    Returns where the replaced one went."""
    work_dir = work_dir or settings().work_dir
    source = backups_folder(work_dir) / name
    if not source.is_file():
        raise DatabaseError(Msg("E-DB-NOSUCHBACKUP", path=str(source)))
    if problem := check_file(source):
        raise DatabaseError(Msg("E-DB-BADBACKUP", reason=problem))
    path = folder(work_dir) / FILE
    with open(path.with_suffix(".lock"), "a") as holder:
        try:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DatabaseError(Msg("E-DB-INUSE")) from None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        replaced = path.with_name(f"{FILE}.replaced-{stamp}")
        for suffix in ("", "-wal", "-shm"):
            p = path.with_name(FILE + suffix)
            if p.exists():
                os.replace(p, replaced.with_name(replaced.name + suffix))
        shutil.copyfile(source, path)
        _fsync(path)
        _fsync(path.parent)
        kept = _files_of(source)
        for part in FILED:  # the files the restored rows point to, where they are missing now
            for f in (kept / part).rglob("*") if (kept / part).is_dir() else []:
                dst = work_dir / part / f.relative_to(kept / part)
                if f.is_file() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dst)
        fcntl.flock(holder, fcntl.LOCK_UN)
    logs.say(log, Msg("I-DB-RESTORED", name=name, kept=str(replaced)))
    return replaced
