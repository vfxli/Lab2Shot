"""一键更新 (`lab2shot setup update`, the menu's 「一键更新」): this checkout brought to its upstream's latest commit, or
left as it was.

Nine steps: the preflight (a clean working tree on a branch that follows a remote one, the remote reachable, the update
a fast-forward); the record (the commit, a copy of the settings file); the locally changed managed files copied to
<file>.old; the service stopped (cli/service.py stop, the menu's own 停止服务), then the database backed up; git pull
--ff-only, uv sync, npm ci and the build; the new code's checks (`lab2shot check`, `lab2shot db upgrade`, `lab2shot db
check`); the new version started and its page answering. Any failure from the stop on rolls the checkout, the managed
files, the settings file, the database and the environment back and starts the old version again; the last step says
which version is online, the extensions to install again, and asks what becomes of the overwritten managed files.

git, uv and npm run with argument lists, never a shell string; the new version runs only in processes of its own.
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

from .base import abort_types, console, err, menu_table, note, ok, pick, say
from .service import (LOG, NotStopped, address, build_webui, initialized, lab2shot_command, launch, listening_pid,
                      page_status, recorded_address, running, service_port, stop, sync_env, terminate)


# The files the administrator edits in place in the checkout (the templates page, the category trees, node placement):
# menu/*.json and templates/*.json, templates/_categories.json among them. A pull must not silently take the local
# version away, nor may a local edit block the pull: they are copied to <file>.old first, and the choice is asked last.
MANAGED_DIRS = ("menu", "templates")
UPDATE_STEPS = ("预检", "记录现状", "备份受管文件", "停止服务", "更新", "质检", "启动服务", "回退", "汇报结果")
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
    upstream: str  # the remote branch it follows, e.g. origin/main
    remote: str
    target: str  # the upstream commit fetched (full)
    incoming: set[str]  # the paths the pull changes
    local: list[tuple[str, str]]  # locally changed managed files: (path, "M" changed / "D" deleted / "?" untracked)
    database: bool  # the work folder has a database
    record: Path | None = None  # work/updates/<time>/
    backup: str = ""  # the database backup taken once the service stopped, after step 4 (a file name in work/db/backups/)
    settings_file: Path | None = None
    settings_copy: Path | None = None  # None: there was no settings file
    reached: set[str] = field(default_factory=set)  # managed / pull / sync / web / db / start: begun, so to be undone

    @property
    def overwritten(self) -> list[tuple[str, str]]:
        """The locally changed managed files the pull changes: set back to the committed version before it."""
        return [(p, k) for p, k in self.local if p in self.incoming]


def _git(*args: str) -> subprocess.CompletedProcess:
    """git in this checkout, its output captured: an argument list, never a shell string."""
    from ..config import ROOT

    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)


def _git_live(*args: str) -> int:
    """git in this checkout, its output (and any credential prompt) on the terminal; returns the exit code."""
    from ..config import ROOT

    note("$ git " + " ".join(args))
    return subprocess.run(["git", "-C", str(ROOT), *args]).returncode


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
    console.print(Rule(f"[bold]第 {i} 步，共 {len(UPDATE_STEPS)} 步：{UPDATE_STEPS[i - 1]}[/bold]", align="left", style="cyan"))




def _update_preflight() -> Update | None:
    """Step 1: a git checkout on a branch that follows a remote one, no change of its own outside the managed files,
    uv and npm present (the rollback needs them too), the service port free unless this work folder's service holds
    it, the remote reachable (git fetch), and the update a fast-forward. None: nothing to update (said).

    A process on the port that does not answer /api/server is not this work folder's service: the update leaves it
    alone and does not begin, since neither the new version nor the old one could start on that port."""
    from ..config import ROOT
    from ..messages import Msg

    top = _git("rev-parse", "--show-toplevel") if shutil.which("git") else None
    if top is None or top.returncode != 0 or Path(top.stdout.strip()).resolve() != ROOT.resolve():
        raise UpdateFailed(Msg("E-UPDATE-NOTGIT", root=str(ROOT)))
    head = _git("rev-parse", "HEAD").stdout.strip()
    branch = _git("symbolic-ref", "--short", "-q", "HEAD").stdout.strip()
    if not branch:
        raise UpdateFailed(Msg("E-UPDATE-DETACHED", head=head[:9]))
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote = _git("config", "--get", f"branch.{branch}.remote").stdout.strip()
    if upstream.returncode != 0 or not remote:
        raise UpdateFailed(Msg("E-UPDATE-NOUPSTREAM", branch=branch))
    upstream_name = upstream.stdout.strip()
    for tool, how in (("uv", "重新运行 ./setup.sh 由脚本引导安装，或参阅 https://docs.astral.sh/uv/"), ("npm", "先安装 Node.js（https://nodejs.org/）")):
        if not shutil.which(tool):
            raise UpdateFailed(Msg("E-UPDATE-NOTOOL", tool=tool, how=how))
    changes = _changes()
    local = [(p, "?" if xy == "??" else "D" if xy == " D" else "M") for xy, p in changes
             if _managed(p) and xy in (" M", " D", "??") and _within(p)]
    mine = {p for p, _ in local}
    dirty = [p for xy, p in changes if xy != "??" and p not in mine]
    if dirty:
        raise UpdateFailed(Msg("E-UPDATE-DIRTY", count=len(dirty), files=dirty[:10] + (["……"] if len(dirty) > 10 else [])))
    ok(f"工作区干净（受管文件之外没有未提交的改动）：分支 {escape(branch)}，版本 {_version()[0]}，跟随 {escape(upstream_name)}。")
    port = service_port()
    if not running() and (pid := listening_pid(port)) is not None:
        raise UpdateFailed(Msg("E-UPDATE-PORTBUSY", listen=port, pid=pid))
    code = _git_live("fetch", remote)
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-FETCH", remote=remote, code=code))
    target = _git("rev-parse", "@{upstream}").stdout.strip()
    behind = int(_git("rev-list", "--count", "HEAD..@{upstream}").stdout.strip() or 0)
    ahead = int(_git("rev-list", "--count", "@{upstream}..HEAD").stdout.strip() or 0)
    if behind == 0:
        say(Msg("N-UPDATE-AHEAD", branch=branch, upstream=upstream_name, ahead=ahead) if ahead
             else Msg("N-UPDATE-UPTODATE", branch=branch, upstream=upstream_name, head=_version()[0]))
        return None
    if ahead:
        raise UpdateFailed(Msg("E-UPDATE-DIVERGED", branch=branch, upstream=upstream_name, ahead=ahead, behind=behind))
    incoming = {p for p in _git("diff", "--name-only", "-z", "HEAD", target).stdout.split("\0") if p}
    clash = [p for xy, p in changes if xy == "??" and p in incoming and not _managed(p)]
    if clash:
        raise UpdateFailed(Msg("E-UPDATE-UNTRACKED", files=clash[:10]))
    ok(f"已连上远程仓库 {escape(remote)}：{escape(upstream_name)} 比本地多 {behind} 个提交，可以快进。")
    log = _git("log", "--format=%h  %s", "-n", "15", f"HEAD..{target}").stdout.splitlines()
    for line in log:
        console.print(f"    {escape(line)}", highlight=False)
    if behind > len(log):
        note(f"    ……另有 {behind - len(log)} 个提交")
    return Update(head, branch, upstream_name, remote, target, incoming, local, initialized())


def _update_record(u: Update) -> None:
    """Step 2: the commit and a copy of the settings file, in work/updates/<time>/record.json. The database is backed up
    once the service has stopped (_update_backup_db, after step 4): taken while it runs, the backup would miss what the
    service writes until it stops (the jobs finishing during a drain, the jobs kept for the next server), and a rollback
    restoring it would lose them."""
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
    ok(f"git 版本：{_version(u.head)[0]}「{escape(_version(u.head)[1])}」（分支 {escape(u.branch)}）。")
    note("数据库在服务停止之后再备份（第 4 步之后）：停止之前服务还会写入。")
    ok(f"设置文件：{'已复制到 ' + str(u.settings_copy) if u.settings_copy else str(u.settings_file) + ' 不存在，全部为默认值'}。")
    note(f"本次更新的记录：{u.record}")


def _write_record(u: Update) -> None:
    """work/updates/<time>/record.json: what this update found and kept (rewritten once the database is backed up)."""
    import json

    (u.record / "record.json").write_text(json.dumps(
        {"head": u.head, "branch": u.branch, "upstream": u.upstream, "target": u.target, "database_backup": u.backup,
         "settings_file": str(u.settings_file), "settings_copy": str(u.settings_copy or ""), "managed": u.local,
         "at": time.time()}, ensure_ascii=False, indent=1), encoding="utf-8")


def _update_backup_db(u: Update) -> None:
    """Right after step 4: the database backed up with the service stopped, so nothing is written after it; a rollback
    restoring it loses nothing (the jobs kept for the next server are in it)."""
    from ..database import DatabaseError, close_all, db
    from ..messages import Msg
    from ..workdir import WorkDirError

    if not u.database:
        note("工作目录还没有数据库（服务从未启动过），无需备份。")
        return
    try:
        u.backup = db().backup("before-update").name
        close_all()  # the new version's upgrade (step 6) needs the database to itself
        _write_record(u)
    except (OSError, DatabaseError, WorkDirError) as exc:
        raise UpdateFailed(Msg("E-UPDATE-RECORD", reason=exc)) from exc
    ok(f"数据库已备份：{u.backup}（work/db/backups/；服务已停止，备份之后不再有写入）。")


def _update_backup_managed(u: Update) -> None:
    """Step 3: every locally changed managed file copied beside itself as <file>.old (a deleted one has nothing to copy)."""
    if not u.local:
        ok("受管文件（menu/*.json、templates/*.json）没有本地改动。")
        return
    table = Table(Column("受管文件", overflow="fold"), "本地改动", Column(".old", overflow="fold"), "本次更新是否改动它",
                  box=box.SIMPLE_HEAD, header_style="bold")  # fold, never cut: a path cut short names no file
    for rel, kind in u.local:
        path = _root_path(rel)
        if kind != "D":
            shutil.copy2(path, path.with_name(path.name + ".old"))
        table.add_row(escape(rel), {"M": "已修改", "D": "已删除", "?": "新增（未纳入 git）"}[kind],
                      "—" if kind == "D" else escape(rel + ".old"), "[yellow]是，将被覆盖[/yellow]" if rel in u.incoming else "否，保持不动")
    console.print(table)
    ok(f"已备份 {sum(1 for _, k in u.local if k != 'D')} 个受管文件（原地复制为 .old）。")


def _update_stop(u: Update) -> None:
    """Step 4: the service stopped the one way the menu's 停止服务 takes too (cli/service.py stop): after the jobs
    running (drain) or at once, the waiting jobs kept for the next server."""
    if not running():
        note("服务没有在运行，无需停止；更新后启动新版本。")
        return
    try:
        stop(cancel=("取消更新", "什么都不改（第 2、3 步的备份保留）"))
    except NotStopped as exc:
        raise UpdateFailed(exc.message) from exc


def _lab2shot(*args: str) -> int:
    """`lab2shot <args>` of this checkout's environment in a process of its own: the code on disk now (the new one)."""
    from ..config import ROOT

    note("$ lab2shot " + " ".join(args))
    return subprocess.run(lab2shot_command(*args), cwd=ROOT).returncode


def _update_pull(u: Update) -> None:
    """Step 5: the managed files the pull changes set back to the committed version (their .old is kept), then
    git pull --ff-only, uv sync, npm ci and the build."""
    from ..messages import Msg

    u.reached.add("managed")
    tracked = [rel for rel, kind in u.overwritten if kind in "MD"]
    if tracked and (code := _git_live("checkout", "HEAD", "--", *tracked)) != 0:
        raise UpdateFailed(Msg("E-UPDATE-PULL", code=code))
    for rel, kind in u.overwritten:
        if kind == "?":
            _root_path(rel).unlink(missing_ok=True)  # its .old holds it; the pull brings the file of the same name
    u.reached.add("pull")
    # from the upstream the preflight checked; git's own advice on a refused pull (merge, rebase) is wrong here: the
    # update puts everything back itself
    code = _git_live("-c", "advice.diverging=false", "pull", "--ff-only")
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-PULL", code=code))
    ok(f"代码已更新：{_version(u.head)[0]} → {_version()[0]}。")
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
    ok("lab2shot check：没有发现问题。")
    if not u.database:
        note("工作目录还没有数据库，无需升级与检查：新版本的服务首次启动时创建。")
        return
    u.reached.add("db")
    code = _lab2shot("db", "upgrade")
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-DBUPGRADE", code=code))
    code = _lab2shot("db", "check")
    if code != 0:
        raise UpdateFailed(Msg("E-UPDATE-DBCHECK", code=code))
    ok("数据库已升级并通过完整性检查。")



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
    ok(f"服务已启动并正常响应：{recorded_address() or address()}（/api/server 与网页均已回答）。")


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
        note(f"新版本的服务输出（{log} 的最后 {len(said)} 行）：")
        for line in said:
            console.print(f"    {escape(line)}", highlight=False)


def _update_rollback(u: Update) -> list[str]:
    """Step 8: undo what was begun, in reverse: the new service stopped, git back to the recorded commit, the local
    managed files and the settings file as they were, the database backup taken after step 4 back in place (only when
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
                ok(f"新版本的服务已停止（端口 {port}，进程 {pid} 已退出）。")
            except NotStopped as exc:
                problems.append(f"新版本的服务没有停下：{exc}")
        elif running():
            problems.append(f"新版本的服务（端口 {port}）在回答，但看不到它的进程号，没能停下")
    if "managed" in u.reached or "pull" in u.reached:
        short = _version(u.head)[0]
        note(f"$ git reset --hard {short}")
        if _git("reset", "--hard", u.head).returncode != 0:
            problems.append(f"git 没能回到 {short}")
        else:
            ok(f"代码已回到 {short}。")
        restored = 0
        for rel, kind in u.local:  # reset --hard set every managed file back to the commit: the local edits again
            path = _root_path(rel)
            bak = path.with_name(path.name + ".old")
            try:
                if kind == "D":
                    path.unlink(missing_ok=True)
                elif bak.is_file():
                    shutil.copy2(bak, path)
                else:
                    problems.append(f"{rel}.old 不见了，{rel} 没能还原")
                    continue
                restored += 1
            except OSError as exc:
                problems.append(f"{rel} 没能还原：{exc}")
        if u.local:
            ok(f"{restored} 个受管文件已还原为更新前本地的内容。")
    if u.settings_file is not None:
        try:
            if u.settings_copy is not None:
                if not u.settings_file.is_file() or u.settings_file.read_bytes() != u.settings_copy.read_bytes():
                    shutil.copy2(u.settings_copy, u.settings_file)
                    ok(f"设置文件已还原：{u.settings_file}。")
            elif u.settings_file.is_file():
                u.settings_file.unlink()  # there was none before: every setting at its default again
                ok(f"更新中生成的设置文件已移除：{u.settings_file}。")
        except OSError as exc:
            problems.append(f"设置文件没能还原：{exc}")
    if "db" in u.reached and u.backup:
        try:
            close_all()
            replaced = restore(u.backup)
            ok(f"数据库已恢复为 {u.backup}（新版本用过的数据库移至 {replaced}）。")
        except DatabaseError as exc:
            problems.append(f"数据库没能恢复为 {u.backup}：{exc}")
    if "sync" in u.reached and not sync_env():
        problems.append("uv sync 没能恢复旧版本的 Python 环境")
    if "web" in u.reached and not build_webui():
        problems.append("旧版本的网页没能重新构建")
    if not running():
        if not launch():
            problems.append("旧版本的服务没能启动")
        elif page_status() != 200:
            problems.append("旧版本的服务启动了，但网页打不开")
        else:
            ok(f"旧版本的服务已启动并正常响应：{recorded_address() or address()}。")
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
        return (got.stderr.strip().splitlines() or [f"退出码 {got.returncode}"])[-1]
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


def _update_managed_choice(u: Update) -> None:
    """Step 9, last: the managed files the update overwrote, and the choice: all back from .old, or the new version."""
    from ..messages import Msg

    if not u.overwritten:
        ok("本次更新没有覆盖任何本地改动过的受管文件。")
        return
    table = Table(Column("被覆盖的受管文件", overflow="fold"), "本地改动", Column("本地的内容在", overflow="fold"),
                  box=box.SIMPLE_HEAD, header_style="bold")
    for rel, kind in u.overwritten:
        table.add_row(escape(rel), {"M": "已修改", "D": "已删除", "?": "新增（未纳入 git）"}[kind],
                      "（本地已删除，还原即再次删除）" if kind == "D" else escape(rel + ".old"))
    console.print(table)
    console.print(menu_table([("1", "全部从 .old 还原", "每个文件恢复为更新前本地的内容（本地删除的再次删除）"),
                               ("2", "使用新版本", "保留更新带来的内容；本地的旧内容仍在 .old 中")]))
    try:
        while (choice := pick("2")) not in ("1", "2"):
            err("没有该编号，请重新输入。")
    except abort_types():
        console.print()
        choice = "2"
    if choice == "2":
        say(Msg("N-UPDATE-MANAGEDKEPT"))
        return
    for rel, kind in u.overwritten:
        path = _root_path(rel)
        if kind == "D":
            path.unlink(missing_ok=True)
        else:
            shutil.copy2(path.with_name(path.name + ".old"), path)
    say(Msg("N-UPDATE-MANAGEDRESTORED", count=len(u.overwritten)))


def one_click_update() -> None:
    """一键更新: the nine steps (above). Any failure from step 4 on rolls back (step 8);
    before it nothing has changed but the backups. Ctrl-C: before step 5 the update is called off, from step 5 on it
    rolls back."""
    # Everything this process uses is imported before the pull: after it the files on disk are the new version's,
    # and a module imported late would be new code among old. The new code runs only in processes of its own
    # (lab2shot check, db upgrade, db check, ui, the extension list)
    from .. import accounts, database  # noqa: F401
    from ..client import Lab2ShotError  # noqa: F401
    from ..io import files  # noqa: F401
    from ..messages import Msg, catalogue
    from ..server import restart, tls  # noqa: F401

    from . import accounts as _cli_accounts  # noqa: F401

    catalogue()
    _step(1)
    try:
        u = _update_preflight()
    except UpdateFailed as exc:  # nothing has changed: the version online stays as it is
        say(exc.msg)
        raise typer.Exit(1)
    if u is None:
        return
    doing = ("接下来会停止服务几分钟：拉取代码、同步环境、构建网页、检查，再启动" if running()
             else "服务现在没有运行：拉取代码、同步环境、构建网页、检查，再启动新版本")
    if not typer.confirm(f"是否开始更新？（{doing}）", default=True):
        note("已取消，未作任何改动。")
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
    except UpdateFailed as exc:
        say(exc.msg)
        if not running():  # step 4 began and the service is down: bring the version it ran back (step 8)
            _step(8)
            _report_failure(u, exc.msg, _update_rollback(u))
        raise typer.Exit(1)
    try:
        _step(5)
        _update_pull(u)
        _step(6)
        _update_check(u)
        _step(7)
        _update_start(u)
    except (UpdateFailed, *abort_types()) as exc:
        reason = exc.msg if isinstance(exc, UpdateFailed) else Msg("E-UPDATE-INTERRUPTED")
        console.print()
        say(reason)
        _step(8)
        try:
            problems = _update_rollback(u)
        except abort_types():
            problems = ["回退被中断（Ctrl-C）"]
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
        say(Msg("W-UPDATE-REINSTALL", names=[f"{title}（{name}）" for name, title in stale]))
    else:
        say(Msg("N-UPDATE-NOREINSTALL"))
    _update_managed_choice(u)
    note(f"本次更新的记录：{u.record}")
    say(Msg("N-UPDATE-MENUOLD"), quiet=True)
    raise SystemExit(0)  # this process is the old program: leave, rather than go on with old code over new files


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
