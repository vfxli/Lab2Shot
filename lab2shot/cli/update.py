"""Upgrade the code the administrator has already pulled.

Stop the service with the installed version, run git pull --ff-only manually, then ./setup.sh update.
The updater never contacts a Git remote. It records the previous installed commit (or the last manual pull's
starting commit on the first upgrade), backs up the stopped database without migrating it, syncs environments,
builds, checks, migrates and starts the service. A failure restores that commit, data and local managed files.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer
from rich import box
from rich.markup import escape
from rich.rule import Rule
from rich.table import Column, Table

from .. import i18n
from ..installer.migrate import undo as undo_migration
from .base import abort_types, console, note, ok, say
from .service import (LOG, NotStopped, address, build_webui, initialized, lab2shot_command, launch, listening_pid,
                      page_status, recorded_address, running, service_port, sync_env, terminate)


# Local edits to these files are preserved during environment updates and saved for a possible code rollback.
MANAGED_DIRS = ("menu", "templates")
INSTALLED = "installed.json"  # work/: written only after a successful upgrade
UPDATE_STEPS = ("cli.update.step.preflight", "cli.update.step.record", "cli.update.step.managed", "cli.update.step.stop",
                "cli.update.step.environment", "cli.update.step.check", "cli.update.step.start", "cli.update.step.rollback",
                "cli.update.step.report")
# the extensions whose environment was built for another spec than the new code's (extensions/status.py
# E-EXT-OUTDATED): listed by the new code in a process of its own, as JSON [[name, title], ...]
STALE_EXTENSIONS = (
    "import json\n"
    "from lab2shot.extensions import extensions\n"
    "from lab2shot.extensions.status import extension_status\n"
    "print(json.dumps([[e.name, e.title] for e in extensions().values()\n"
    "                  if (extension_status(e).get('message') or {}).get('code') == 'E-EXT-OUTDATED'], ensure_ascii=False))\n")


class UpdateFailed(Exception):
    """A step of the update did not succeed; `msg` says which and why."""

    def __init__(self, msg) -> None:
        super().__init__(msg.text)
        self.msg = msg


@dataclass
class Update:
    """What the update found and did, step by step: what the rollback undoes and the report says."""

    head: str  # the commit before (full)
    branch: str
    target: str  # the code already pulled by the administrator
    local: list[tuple[str, str]]  # locally changed managed files: (path, "M" changed / "D" deleted / "?" untracked)
    database: bool  # the work folder has a database
    record: Path | None = None  # work/updates/<time>/
    backup: str = ""  # the database backup taken once the service stopped, in step 4 (a file name in work/db/backups/)
    settings_file: Path | None = None
    settings_copy: Path | None = None  # None: there was no settings file
    reached: set[str] = field(default_factory=set)  # code / sync / web / db / start: begun, so to be undone


def _git(*args: str) -> subprocess.CompletedProcess:
    """git in this checkout, its output captured: an argument list, never a shell string."""
    from ..config import ROOT

    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _managed(rel: str) -> bool:
    from pathlib import PurePosixPath

    parts = PurePosixPath(rel).parts
    return len(parts) == 2 and parts[0] in MANAGED_DIRS and parts[1].endswith(".json")


def _root_path(rel: str) -> Path:
    """A path git named, inside the checkout (io.files.inside: no .., no link out of it)."""
    from ..config import ROOT
    from ..io.files import inside

    return inside(ROOT, rel)


def _within(rel: str) -> bool:
    from ..errors import MessageError

    try:
        _root_path(rel)
        return True
    except MessageError:
        return False


def _changes() -> list[tuple[str, str]]:
    """Every change of the working tree (`git status --porcelain -z`): (XY status, path)."""
    items = _git("status", "--porcelain=v1", "-z", "--untracked-files=all").stdout.split("\0")
    found, i = [], 0
    while i < len(items):
        entry = items[i]
        i += 1
        if len(entry) < 4:
            continue
        if entry[0] in "RC":
            i += 1  # a rename or copy: the original path follows
        found.append((entry[:2], entry[3:]))
    return found


def _version(commit: str = "HEAD") -> tuple[str, str]:
    """(short hash, subject) of a commit of this checkout."""
    got = _git("log", "-1", "--format=%h%x00%s", commit).stdout.strip().split("\0", 1)
    return (got[0], got[1]) if len(got) == 2 else (commit[:9], "")


def _step(i: int) -> None:
    title = i18n.t("cli.update.step.title", step=i, steps=len(UPDATE_STEPS), name=i18n.t(UPDATE_STEPS[i - 1]))
    console.print(Rule(f"[bold]{title}[/bold]", align="left", style="cyan"))


def _previous_commit(target: str) -> str:
    """Prefer the last successful deployment; on first use accept only the latest HEAD reflog's manual pull.
    A stale ORIG_HEAD is not a deployment record. A no-change run uses the current commit as its rollback baseline.
    """
    import json

    from ..config import ROOT, settings
    from ..messages import Msg

    installed = settings().work_dir / INSTALLED
    try:
        saved = json.loads(installed.read_text(encoding="utf-8"))
        if saved["root"] != str(ROOT):
            raise ValueError(i18n.t("cli.update.baseline.other_root"))
        previous = str(saved["commit"])
    except FileNotFoundError:
        rows = _git("reflog", "show", "--format=%H%x00%gs", "-n", "2", "HEAD").stdout.splitlines()
        latest = rows[0].split("\0", 1) if rows else []
        if len(latest) == 2 and latest[0] == target and latest[1].startswith(("pull:", "pull ")) and len(rows) > 1:
            previous = rows[1].split("\0", 1)[0]
        else:
            note(i18n.t("cli.update.baseline.no_record"))
            previous = target
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UpdateFailed(Msg("E-UPDATE-BASELINE", reason=exc)) from exc
    if _git("cat-file", "-e", previous + "^{commit}").returncode or _git("merge-base", "--is-ancestor", previous, target).returncode:
        raise UpdateFailed(Msg("E-UPDATE-BASELINE", reason=i18n.Word("cli.update.baseline.not_ancestor")))
    return previous


def _update_preflight() -> Update:
    """Check only local state. Stop the old service before pulling so its workers cannot load new code mid-job."""
    from ..config import ROOT, settings
    from ..messages import Msg
    from ..workdir import own

    top = _git("rev-parse", "--show-toplevel") if shutil.which("git") else None
    if top is None or top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT.resolve():
        raise UpdateFailed(Msg("E-UPDATE-NOTGIT", root=str(ROOT)))
    target = _git("rev-parse", "HEAD").stdout.strip()
    branch = _git("symbolic-ref", "--short", "-q", "HEAD").stdout.strip()
    if not branch:
        raise UpdateFailed(Msg("E-UPDATE-DETACHED", head=target[:9]))
    for tool, how in (("uv", i18n.t("cli.update.baseline.how_uv")), ("npm", i18n.t("cli.update.baseline.how_npm"))):
        if not shutil.which(tool):
            raise UpdateFailed(Msg("E-UPDATE-NOTOOL", tool=tool, how=how))
    changes = _changes()
    local = [(p, "?" if xy == "??" else "D" if xy == " D" else "M") for xy, p in changes
             if _managed(p) and xy in (" M", " D", "??") and _within(p)]
    mine = {p for p, _ in local}
    dirty = [p for xy, p in changes if xy != "??" and p not in mine]
    if dirty:
        raise UpdateFailed(Msg("E-UPDATE-DIRTY", count=len(dirty), files=dirty[:10] + (["……"] if len(dirty) > 10 else [])))
    port = service_port()
    if running():
        raise UpdateFailed(Msg("E-UPDATE-RUNNING"))
    if (pid := listening_pid(port)) is not None:
        raise UpdateFailed(Msg("E-UPDATE-PORTBUSY", listen=port, pid=pid))
    own(settings().work_dir)
    previous = _previous_commit(target)
    clash = [p for xy, p in changes if xy == "??" and p not in mine
             and _git("ls-tree", "--name-only", previous, "--", p).stdout.strip()]
    if clash:
        raise UpdateFailed(Msg("E-UPDATE-DIRTY", count=len(clash), files=clash[:10]))
    ok(i18n.t("cli.update.baseline.local", branch=escape(branch), version=_version(target)[0], previous=_version(previous)[0]))
    return Update(previous, branch, target, local, initialized())


def _update_record(u: Update) -> None:
    """Step 2: save the rollback commit and settings before any environment or database change."""
    from ..config import settings
    from ..database import DatabaseError
    from ..messages import Msg
    from ..workdir import WorkDirError

    s = settings()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    u.record = s.work_dir / "updates" / stamp
    n = 1
    while u.record.exists():  # two runs within one second (the first one called off)
        n += 1
        u.record = u.record.with_name(f"{stamp}~{n}")
    try:
        u.record.mkdir(parents=True, exist_ok=False)
        u.settings_file = Path(s.file)
        if u.settings_file.is_file():
            u.settings_copy = u.record / ("settings" + u.settings_file.suffix)
            shutil.copy2(u.settings_file, u.settings_copy)
        _write_record(u)
    except (OSError, DatabaseError, WorkDirError) as exc:
        raise UpdateFailed(Msg("E-UPDATE-RECORD", reason=exc)) from exc
    ok(i18n.t("cli.update.record.git", version=_version(u.head)[0], subject=escape(_version(u.head)[1]), branch=escape(u.branch)))
    note(i18n.t("cli.update.record.plan"))
    ok(i18n.t("cli.update.record.settings_copied", path=u.settings_copy) if u.settings_copy
       else i18n.t("cli.update.record.settings_none", path=u.settings_file))
    note(i18n.t("cli.update.record.where", path=u.record))


def _write_record(u: Update) -> None:
    """work/updates/<time>/record.json: what this update found and kept (rewritten once the database is backed up)."""
    import json

    (u.record / "record.json").write_text(json.dumps(
        {"head": u.head, "branch": u.branch, "target": u.target, "database_backup": u.backup,
         "settings_file": str(u.settings_file), "settings_copy": str(u.settings_copy or ""), "managed": u.local,
         "at": time.time()}, ensure_ascii=False, indent=1), encoding="utf-8")


def _update_backup_db(u: Update) -> None:
    """Step 4: the database backed up with the service stopped, so nothing is written after it; a rollback
    restoring it loses nothing (the jobs kept for the next server are in it)."""
    from ..database import DatabaseError, backup_stopped, close_all
    from ..messages import Msg
    from ..workdir import WorkDirError

    if not u.database:
        note(i18n.t("cli.update.record.no_database"))
        return
    try:
        close_all()
        u.backup = backup_stopped("before-update").name
        _write_record(u)
    except (OSError, DatabaseError, WorkDirError) as exc:
        raise UpdateFailed(Msg("E-UPDATE-RECORD", reason=exc)) from exc
    ok(i18n.t("cli.update.record.database", backup=u.backup))


def _update_backup_managed(u: Update) -> None:
    """Save local managed files in this run's record, leaving existing .old backups untouched."""
    if not u.local:
        ok(i18n.t("cli.update.managed.none"))
        return
    table = Table(Column(i18n.t("cli.update.managed.file"), overflow="fold"), i18n.t("cli.update.managed.change"),
                  Column(i18n.t("cli.update.managed.backup"), overflow="fold"),
                  box=box.SIMPLE_HEAD, header_style="bold")
    for rel, kind in u.local:
        backup = u.record / "managed" / rel
        if kind != "D":
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_root_path(rel), backup)
        kinds = {"M": "cli.update.managed.changed", "D": "cli.update.managed.deleted", "?": "cli.update.managed.untracked"}
        table.add_row(escape(rel), i18n.t(kinds[kind]),
                      i18n.t("cli.update.managed.deletion") if kind == "D" else escape(str(backup)))
    console.print(table)
    ok(i18n.t("cli.update.managed.saved"))


def _update_stop(u: Update) -> None:
    """Recheck before backing up; do not open an old database to authenticate a stop after the manual pull."""
    from ..messages import Msg

    if running():
        raise UpdateFailed(Msg("E-UPDATE-RUNNING"))
    if (pid := listening_pid(service_port())) is not None:
        raise UpdateFailed(Msg("E-UPDATE-PORTBUSY", listen=service_port(), pid=pid))
    ok(i18n.t("cli.update.run.stopped"))


def _lab2shot(*args: str) -> int:
    """`lab2shot <args>` of this checkout's environment in a process of its own: the code on disk now (the new one)."""
    from ..config import ROOT

    note("$ lab2shot " + " ".join(args))
    return subprocess.run(lab2shot_command(*args), cwd=ROOT).returncode


def _update_environment(u: Update) -> None:
    """Step 5: use the code already on disk; never fetch or pull inside the updater."""
    from ..messages import Msg

    u.reached.add("code")
    u.reached.add("sync")
    if not sync_env():
        raise UpdateFailed(Msg("E-UPDATE-SYNC"))
    u.reached.add("web")
    if not build_webui():
        raise UpdateFailed(Msg("E-UPDATE-WEB"))


def _update_check(u: Update) -> None:
    """Step 6, with the new code: lab2shot check, the database upgrade (it backs up first), the integrity check."""
    from ..messages import Msg

    code = _lab2shot("check")
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-CHECK", code=code))
    ok(i18n.t("cli.update.run.checked"))
    # what is installed, as the new code identifies and lays it out (installer/migrate.py): journalled for the rollback
    u.reached.add("migrate")
    code = _lab2shot("ext", "migrate", "--since", u.head, "--journal", str(u.record / "migrate"))
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-MIGRATE", code=code))
    ok(i18n.t("cli.update.run.migrated"))
    if not u.database:
        note(i18n.t("cli.update.run.no_database"))
        return
    u.reached.add("db")
    code = _lab2shot("db", "upgrade")
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-DBUPGRADE", code=code))
    code = _lab2shot("db", "check")
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-DBCHECK", code=code))
    ok(i18n.t("cli.update.run.database"))



def _update_start(u: Update) -> None:
    """Step 7: start the new version and confirm that it answers: /api/server, and the main page."""
    from ..config import settings
    from ..messages import Msg

    log = settings().work_dir / "logs" / LOG
    seen = log.stat().st_size if log.is_file() else 0
    u.reached.add("start")
    u.reached.add("db")  # the server opens (and may upgrade) the database, even where step 6 had none to upgrade
    if not launch():
        _show_output(log, seen)
        raise UpdateFailed(Msg("E-UPDATE-NOSTART", log=str(log)))
    status = page_status()
    if status != 200:
        raise UpdateFailed(Msg("E-UPDATE-NOPAGE", address=recorded_address() or address(), status=status, log=str(log)))
    ok(i18n.t("cli.update.run.started", address=recorded_address() or address()))


def _show_output(log: Path, seen: int, lines: int = 20) -> None:
    """The last lines the new version wrote to its log after byte `seen`: why it did not start, on screen now, before
    the old version started by the rollback writes its own lines after them."""
    try:
        with open(log, "rb") as f:
            f.seek(seen)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return
    said = [line for line in text.splitlines() if line.strip()][-lines:]
    if said:
        note(i18n.t("cli.update.run.output", log=log, lines=len(said)))
        for line in said:
            console.print(f"    {escape(line)}", highlight=False)


def _update_rollback(u: Update) -> list[str]:
    """Step 8: undo what was begun, in reverse: the new service stopped, git back to the recorded commit, the local
    managed files and the settings file as they were, the database backup taken in step 4 back in place (only when
    the new version may have opened the database: until then it is exactly as the stopped service left it), the
    environment synced and the page built again, and the old version started. Returns what could not be done (empty:
    all)."""
    from ..database import DatabaseError, close_all, restore

    problems: list[str] = []
    if "start" in u.reached:  # the new version this update started (it may not answer): ended by its pid, unasked
        port = service_port()
        pid = listening_pid(port)
        if pid is not None:
            try:
                terminate(pid, port)
                ok(i18n.t("cli.update.rollback.new_stopped", port=port, pid=pid))
            except NotStopped as exc:
                problems.append(i18n.t("cli.update.rollback.new_not_stopped", error=exc))
        elif running():
            problems.append(i18n.t("cli.update.rollback.new_no_pid", port=port))
    if "code" in u.reached:
        short = _version(u.head)[0]
        note(f"$ git reset --hard {short}")
        if _git("reset", "--hard", u.head).returncode != 0:
            problems.append(i18n.t("cli.update.rollback.git_failed", version=short))
        else:
            ok(i18n.t("cli.update.rollback.code", version=short))
        restored = 0
        for rel, kind in u.local:  # reset --hard set every managed file back to the commit: the local edits again
            path = _root_path(rel)
            bak = u.record / "managed" / rel
            try:
                if kind == "D":
                    path.unlink(missing_ok=True)
                elif bak.is_file():
                    shutil.copy2(bak, path)
                else:
                    problems.append(i18n.t("cli.update.rollback.backup_gone", backup=bak, path=rel))
                    continue
                restored += 1
            except OSError as exc:
                problems.append(i18n.t("cli.update.rollback.file_failed", path=rel, error=exc))
        if u.local:
            ok(i18n.t("cli.update.rollback.managed", count=restored))
    if "migrate" in u.reached:  # the installed extensions as the old code knows them
        undone = undo_migration(u.record / "migrate")
        if undone:
            problems += [i18n.t("cli.update.rollback.migrate_failed", error=e) for e in undone]
        else:
            ok(i18n.t("cli.update.rollback.migrate"))
    if u.settings_file is not None:
        try:
            if u.settings_copy is not None:
                if not u.settings_file.is_file() or u.settings_file.read_bytes() != u.settings_copy.read_bytes():
                    shutil.copy2(u.settings_copy, u.settings_file)
                    ok(i18n.t("cli.update.rollback.settings", path=u.settings_file))
            elif u.settings_file.is_file():
                u.settings_file.unlink()  # there was none before: every setting at its default again
                ok(i18n.t("cli.update.rollback.settings_removed", path=u.settings_file))
        except OSError as exc:
            problems.append(i18n.t("cli.update.rollback.settings_failed", error=exc))
    if "db" in u.reached and u.backup:
        try:
            close_all()
            replaced = restore(u.backup)
            ok(i18n.t("cli.update.rollback.database", backup=u.backup, moved=replaced))
        except DatabaseError as exc:
            problems.append(i18n.t("cli.update.rollback.database_failed", backup=u.backup, error=exc))
    if "sync" in u.reached and not sync_env():
        problems.append(i18n.t("cli.update.rollback.sync_failed"))
    if "web" in u.reached and not build_webui():
        problems.append(i18n.t("cli.update.rollback.web_failed"))
    if not running():
        if not launch():
            problems.append(i18n.t("cli.update.rollback.start_failed"))
        elif page_status() != 200:
            problems.append(i18n.t("cli.update.rollback.page_failed"))
        else:
            ok(i18n.t("cli.update.rollback.started", address=recorded_address() or address()))
    return problems


def _stale_extensions() -> list[tuple[str, str]] | str:
    """The installed extensions the new code wants installed again (E-EXT-OUTDATED), asked of the new code in a
    process of its own; a string says why they could not be listed."""
    import json

    from ..config import ROOT

    got = subprocess.run([sys.executable, "-c", STALE_EXTENSIONS], cwd=ROOT, capture_output=True, text=True)
    try:
        rows = json.loads(got.stdout.strip().splitlines()[-1]) if got.returncode == 0 else None
    except (ValueError, IndexError):
        rows = None
    if rows is None:
        return (got.stderr.strip().splitlines() or [i18n.t("cli.update.run.exit_code", code=got.returncode)])[-1]
    return [(str(r[0]), str(r[1])) for r in rows]


def _online() -> None:
    """Which version is online now: the checkout's commit, and what the service answering says of itself."""
    from ..config import settings
    from ..messages import Msg

    short, subject = _version()
    state = running()
    if state:
        say(Msg("N-UPDATE-ONLINE", version=short, subject=subject, release=state.get("version", "?"),
                 boot=state.get("boot", "?"), address=recorded_address() or address()))
    else:
        say(Msg("W-UPDATE-OFFLINE", version=short, log=str(settings().work_dir / "logs" / LOG)))


def _update_installed(u: Update) -> None:
    """Record a successful deployment atomically; later manual pulls use it as their rollback baseline."""
    import json

    from ..config import ROOT, settings
    from ..io.atomic import write_text
    from ..messages import Msg

    try:
        write_text(settings().work_dir / INSTALLED, json.dumps(
            {"root": str(ROOT), "commit": u.target, "at": time.time()}, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise UpdateFailed(Msg("E-UPDATE-RECORD", reason=exc)) from exc


def one_click_update() -> None:
    """Complete an already pulled version, restoring the previous deployment on failure after backups finish."""
    # Rollback changes files on disk: import the modules it uses while they still belong to this version.
    from .. import accounts, database  # noqa: F401
    from ..client import Lab2ShotError  # noqa: F401
    from ..io import files  # noqa: F401
    from ..server import restart, tls  # noqa: F401
    from . import accounts as _cli_accounts  # noqa: F401
    from ..messages import Msg

    _step(1)
    try:
        u = _update_preflight()
    except UpdateFailed as exc:
        say(exc.msg)
        raise typer.Exit(1)
    if not typer.confirm(i18n.t("cli.update.run.confirm"), default=True):
        note(i18n.t("cli.update.run.cancelled"))
        return
    try:
        _step(2)
        _update_record(u)
        _step(3)
        _update_backup_managed(u)
        _step(4)
        _update_stop(u)
        _update_backup_db(u)
    except abort_types():
        console.print()
        say(Msg("N-UPDATE-CANCELLED"), quiet=True)
        raise typer.Exit(1)
    except (UpdateFailed, OSError) as exc:
        say(exc.msg if isinstance(exc, UpdateFailed) else Msg("E-UPDATE-RECORD", reason=exc))
        raise typer.Exit(1)
    try:
        _step(5)
        _update_environment(u)
        _step(6)
        _update_check(u)
        _step(7)
        _update_start(u)
        _update_installed(u)
    except (UpdateFailed, *abort_types()) as exc:
        reason = exc.msg if isinstance(exc, UpdateFailed) else Msg("E-UPDATE-INTERRUPTED")
        console.print()
        say(reason)
        _step(8)
        try:
            problems = _update_rollback(u)
        except abort_types():
            problems = [i18n.t("cli.update.run.interrupted")]
        _report_failure(u, reason, problems)
        raise typer.Exit(1)
    _step(9)
    before, after = _version(u.head)[0], _version()[0]
    say(Msg("N-UPDATE-DONE", before=before, after=after, address=recorded_address() or address()))
    _online()
    stale = _stale_extensions()
    if isinstance(stale, str):
        say(Msg("W-UPDATE-REINSTALLUNKNOWN", reason=stale))
    elif stale:
        say(Msg("W-UPDATE-REINSTALL", names=[i18n.Word("cli.update.run.extension", title=title, name=name) for name, title in stale]))
    else:
        say(Msg("N-UPDATE-NOREINSTALL"))
    note(i18n.t("cli.update.run.done", path=u.record))
    say(Msg("N-UPDATE-MENUOLD"), quiet=True)
    raise SystemExit(0)  # the synchronized environment should be loaded by a fresh menu process


def _report_failure(u: Update, reason, problems: list[str]) -> None:
    """Step 9 after a failure: rolled back to which version, and which one is online now."""
    from ..messages import Msg

    _step(9)
    short = _version()[0]
    if problems:
        say(Msg("E-UPDATE-ROLLBACKINCOMPLETE", reason=reason, problems=problems, record=str(u.record)))
    else:
        say(Msg("E-UPDATE-ROLLEDBACK", version=short, reason=reason, address=recorded_address() or address(),
                 database=Msg("N-UPDATE-DBRESTORED", backup=u.backup) if "db" in u.reached and u.backup else Msg("N-UPDATE-DBUNTOUCHED")))
    _online()
