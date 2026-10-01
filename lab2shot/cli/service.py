"""The service of this checkout on its work folder, for the configuration menu (cli/setup.py) and the one-click update
(cli/update.py) alike: whether it runs and where it listens, the environment it runs from (uv sync, the page build),
and starting and stopping it.

The service is asked over HTTP (/api/server needs no login), so looking creates nothing; it is controlled through the
local client (machine token, cli/accounts.py local_client), and started as `lab2shot ui` in a process of its own, so it
runs the code on disk now whatever this process imported.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import typer
from rich.markup import escape

from ..errors import MessageError
from .base import abort_types, console, err, menu_table, note, ok, pick, say, warn

LOG = "ui_stdout.log"  # the background service's terminal output, under <work>/logs/


# ------------------------------------------------------------------ whether it runs, and where


def initialized() -> bool:
    """Whether the work folder's database exists. Before the service's first start there is nothing to inspect, and
    inspecting it (local_client → machine token → db()) would create work/, owner.json, the database and the token;
    read-only items must not do so."""
    from ..config import settings
    from ..database import FILE

    return (settings().work_dir / "db" / FILE).is_file()


def running():
    """The service on this work folder, if it answers: {"boot", "started", ...} or None. Queried over plain HTTP
    (`/api/server` requires no login), so the query creates nothing (no work/, no database, no machine token) and
    cannot collide with a server that is creating them at that moment (the first start polls this while the server
    is still building the database). The address queried is the recorded one when a server left its record
    (server/restart.py RECORD: the port or HTTPS setting may have changed since it started), otherwise the address
    of the current settings."""
    import json
    import ssl
    import urllib.request

    where = recorded_address() or address()
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # this machine's own self-signed certificate
        with urllib.request.urlopen(f"{where}/api/server", timeout=3, context=ctx) as r:
            got = json.loads(r.read().decode("utf-8"))
        return got if isinstance(got, dict) and "boot" in got else None
    except (OSError, ValueError):
        return None


def recorded_address() -> str | None:
    """Where a server running on this work folder reported that it listens (server/restart.py RECORD); None when none
    is running. A record whose process no longer exists is stale (a crash or a kill) and is ignored."""
    from ..server import restart as server

    return server.recorded_address()


def _now(key: str) -> object:
    """What is set now (the file as it is on disk, or this run's environment), not what this process started with.
    settings()[key] gives a setting that needs a restart as this process started with it: right for the server, wrong
    for this menu, which is not the server: the port, HTTPS and listening address it uses are those of the server it
    starts next, so a change made in 「设置」 earlier in the same menu session applies at once. A running server's own
    address comes from its record (recorded_address)."""
    from ..config import settings

    s = settings()
    s._reload_if_changed()  # a hand edit of config/local.toml while the menu is open
    return s.value(key)


def address() -> str:
    return f"{'https' if _now('server.https') else 'http'}://127.0.0.1:{_now('server.port')}"


def listening_pid(port: int) -> int | None:
    """The process listening on `port` (Linux `ss`); None when there is none, or when it cannot be determined."""
    if not shutil.which("ss"):
        return None
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{port} " in line and "pid=" in line:
            try:
                return int(line.split("pid=")[1].split(",")[0])
            except ValueError:
                return None
    return None


def service_port() -> int:
    """The port of the service on this work folder: the recorded one when a server left its record (it may have been
    started with --port), otherwise the one in the settings."""
    from urllib.parse import urlparse

    recorded = recorded_address()
    try:
        if recorded and (port := urlparse(recorded).port):
            return port
    except ValueError:
        pass
    return int(_now("server.port"))


def _command_of(pid: int) -> str:
    """The command line of process `pid` (Linux /proc), for the confirmation; "" when it cannot be read."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:80]
    except OSError:
        return ""


def _pid_alive(pid: int) -> bool:
    """Whether process `pid` still exists. A server this menu started is its child: once ended it is reaped here,
    or it would linger as a zombie that os.kill still finds."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass  # not this process's child
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def page_status() -> int | str:
    """What the running service answers for its main page: the HTTP status, or why there is none."""
    import ssl
    import urllib.error
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # this machine's own self-signed certificate
    try:
        with urllib.request.urlopen(f"{recorded_address() or address()}/", timeout=10, context=ctx) as r:
            return r.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except OSError as exc:
        return str(exc)


# ------------------------------------------------------------------ the environment it runs from


def sync_env() -> bool:
    """uv sync; returns whether the environment is ready (the one-click update goes on, or rolls back, on it)."""
    from ..config import ROOT

    if not shutil.which("uv"):
        err("未找到 uv。请重新运行 ./setup.sh 由脚本引导安装，或参阅 https://docs.astral.sh/uv/ 。")
        return False
    note(f"$ uv sync（工作目录：{ROOT}）")
    if subprocess.run(["uv", "sync"], cwd=ROOT).returncode == 0:
        ok("Python 环境已就绪。")
        return True
    mirror = _pypi_mirror()
    if mirror and typer.confirm(f"执行失败。是否使用 PyPI 镜像（{mirror}）重试？", default=True):
        env = {**os.environ, "UV_INDEX_URL": mirror}
        if subprocess.run(["uv", "sync"], cwd=ROOT, env=env).returncode == 0:
            ok("Python 环境已就绪（经由镜像）。")
            return True
    err("执行失败，请查看上方输出。如网络不可用，请在「安装与环境 → 扩展包编译与下载设置 → 设置下载镜像」中配置镜像，或配置代理后重试。")
    return False


def _pypi_mirror() -> str:
    from ..config import settings

    text = str(settings()["install.mirror_pypi"] or "").strip()
    return text.split()[0] if text else ""


def build_webui() -> bool:
    """npm ci and npm run build; returns whether the page is built (the one-click update goes on, or rolls back, on it)."""
    from ..config import ROOT

    if not shutil.which("npm"):
        err("未找到 npm。请先安装 Node.js（https://nodejs.org/）。")
        return False
    web = ROOT / "webui"
    # npm ci installs exactly what package-lock.json pins and never rewrites it (npm install would: a build must not
    # leave a changed file in the checkout); a lockfile that no longer matches package.json is an error, not a re-solve
    note(f"$ npm ci（工作目录：{web}）")
    if subprocess.run(["npm", "ci"], cwd=web).returncode != 0:
        if not typer.confirm("执行失败。是否使用 npm 镜像（registry.npmmirror.com）重试？", default=True):
            return False
        if subprocess.run(["npm", "ci", "--registry=https://registry.npmmirror.com"], cwd=web).returncode != 0:
            err("执行失败，请查看上方输出。")
            return False
    note(f"$ npm run build（工作目录：{web}）")
    if subprocess.run(["npm", "run", "build"], cwd=web).returncode != 0:
        err("执行失败，请查看上方输出。")
        return False
    ok("网页已构建完成。")
    return True


# ------------------------------------------------------------------ starting and stopping it


def start(background: bool) -> None:
    """Start the service (in the background, or in this terminal). Ends with typer.Exit(1) when it could not: the port
    is taken, the page is not built, or the service did not answer in time."""
    from ..config import WEBUI_DIST, settings
    from ..database import close_all

    s = settings()
    port = int(_now("server.port"))
    if running():
        recorded = recorded_address()
        note(f"服务已在运行：{recorded or address()}。")
        if recorded and recorded != address():
            warn(f"当前设置的地址为 {address()}；端口或 HTTPS 的修改需重启服务后生效。")
        return
    if (pid := listening_pid(port)) is not None:
        err(f"端口 {port} 已被进程 {pid} 占用，不再启动新的服务。如该进程不是 Lab2Shot，请在「设置 → 网络与安装」中更换端口。")
        raise typer.Exit(1)
    if not WEBUI_DIST.is_dir():
        err("网页尚未构建。请先执行「安装与环境 → 构建网页界面」。")
        raise typer.Exit(1)
    close_all()  # this menu's own handle on the database must not block the server's upgrade at start
    if not background:
        note(f"服务在前台运行，按 Ctrl-C 停止：{address()}")
        subprocess.call(lab2shot_command("ui"))
        return
    log = s.work_dir / "logs" / LOG
    if launch():
        ok(f"服务已启动：{recorded_address() or address()}（日志：{log}）。")
        _first_time_hints()
        return
    err(f"服务未能在 120 秒内启动，请查看日志 {log}。")
    raise typer.Exit(1)  # `./setup.sh start` (and a script calling it) must see that it failed


def lab2shot_command(*args: str) -> list[str]:
    """`lab2shot <args>` of this checkout's environment (`ui`, or the update's `check` and `db ...`): run as a separate
    process, it runs the code on disk now (after an update, the new code), whatever this menu's own process imported."""
    exe = Path(sys.executable).parent / "lab2shot"
    return [str(exe), *args] if exe.is_file() else [sys.executable, "-c", "from lab2shot.cli import app; app()", *args]


def launch() -> bool:
    """Start the service in the background, its output appended to work/logs/ui_stdout.log, and wait (at most 120
    seconds) until it answers; returns whether it does.

    A service that has not answered by then is ended (its whole process group): left running it would keep the
    database and the queue, so the one-click update could neither restore the database nor start the old version, and
    it might still come up later on the port."""
    from ..config import settings

    logs = settings().work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / LOG, "ab") as out:  # the child keeps its own copy of the descriptor
        proc = subprocess.Popen(lab2shot_command("ui"), stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    note("正在等待服务启动（最长 120 秒）……")
    for _ in range(120):
        time.sleep(1)
        if running():
            return True
        if proc.poll() is not None:  # it ended without answering (the log says why): no use waiting on
            return False
    warn(f"服务 120 秒内没有回答，结束它（进程 {proc.pid}）。")
    _end_group(proc)
    return False


def _end_group(proc: subprocess.Popen) -> None:
    """End a process launch() started, with every process it started (start_new_session: its group): SIGTERM, then
    SIGKILL after KILL_WAIT_S seconds."""
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
            proc.wait(timeout=KILL_WAIT_S)
            break
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue
    try:
        os.killpg(proc.pid, signal.SIGKILL)  # what it started and outlived it
    except ProcessLookupError:
        pass


def lan_address() -> str:
    """The address other machines on the network reach this one by: an IPv4 address of its own interfaces
    (server/tls.py own_addresses), not 127.0.0.1; unknown: a placeholder."""
    from ..server import tls

    lan = sorted(ip for ip in tls.own_addresses() if ip not in ("127.0.0.1", "::1") and ":" not in ip)
    return lan[0] if lan else "<本机地址>"


def ca_url(port: int) -> str:
    """Where a user's computer downloads the root certificate of this server's self-signed HTTPS (server/tls.py)."""
    return f"https://{lan_address()}:{port}/api/tls/ca.pem"


def _first_time_hints() -> None:
    from .. import accounts

    if _now("server.https"):
        note(f"已启用 HTTPS：每台用户电脑需安装一次证书。请在浏览器中打开 {ca_url(int(_now('server.port')))} 下载，"
              "并导入系统的「受信任的根证书颁发机构」。")
    if accounts.admin().no_password:
        warn(f"管理员 {accounts.admin().username} 尚未设置密码，目前任何人都无法登录。请执行「账号与安全 → 设置管理员密码」。")


# the choice stop offers: the running server stops through its own route either way (server/restart.py `then` stop)
STOP_CHOICES = (("1", "等当前任务算完再停", "不再开始新任务；计算中的任务算完后停止。排队的任务留给下一次启动的服务接着排"),
                ("2", "立即停止", "计算中的任务立即停下（已算完的节点留在缓存里）；排队的任务留给下一次启动的服务"))
NOW_WAIT_S = 120  # how long a stop "now" may take before it counts as failed (a drain waits as long as the jobs run)
KILL_WAIT_S = 30  # how long a process ended by its pid may take to leave the port


class NotStopped(MessageError):
    """The service did not stop: it refused, it cannot be seen, or it did not end in time; the message says which."""

    status = 409


def stop(cancel: tuple[str, str] = ("返回", "不停止服务，返回上一级菜单")) -> bool:
    """Stop the service on this work folder: the one way, for the menu's 停止服务 and the one-click update's step 4.

    The running server is asked through its own route (/api/admin/stop, the local client's machine token), after the
    choice STOP_CHOICES offers or `cancel` (its label and note): once the jobs running are done (drain), or at once
    (now); either way the waiting jobs are kept for the next server. Then it is waited for until nothing listens on the
    port and the process has ended; Ctrl-C while it waits calls the stop off, and the queue goes on.

    Only a service that cannot be asked at all is ended by its pid instead, after confirmation: nothing answers on the
    port while a process listens there, or this checkout cannot open its records (the machine token): a broken or
    foreign server. A server that answers but refuses the route is an error (NotStopped).

    Returns whether a service was stopped (False: none was running). Raises NotStopped when it is still there, and the
    abort types (abort_types) when the choice was `cancel`, the confirmation was declined or the wait was called off."""
    from ..client import Lab2ShotError
    from ..database import DatabaseError, close_all
    from ..messages import Msg
    from ..workdir import WorkDirError

    from .accounts import local_client

    port = service_port()
    answering = running() is not None  # asked first: the port's pid alone does not say who listens there
    pid = listening_pid(port)
    lab = None
    if answering:
        try:
            lab = local_client()
        except (WorkDirError, DatabaseError) as exc:
            warn(f"服务在回答，但本项目目录打不开它的记录，无法用本机令牌请它停止：{escape(str(exc))}")
    if lab is None:
        if not _end(pid, port, answering):
            return False
    else:
        if pid is None:
            raise NotStopped(Msg("E-SERVICE-NOPID", address=recorded_address() or address()))
        console.print(menu_table([*STOP_CHOICES, ("0", *cancel)]))
        choice = pick("1")
        if choice not in ("1", "2"):
            raise KeyboardInterrupt
        mode = "drain" if choice == "1" else "now"
        try:
            lab.admin_stop(mode)
        except Lab2ShotError as exc:
            raise NotStopped(Msg("E-SERVICE-STOPREFUSED", reason=str(exc))) from exc
        try:
            waited = 0
            while listening_pid(port) is not None or _pid_alive(pid):
                if mode == "now" and waited >= NOW_WAIT_S:
                    raise NotStopped(Msg("E-SERVICE-STILLUP", seconds=waited, listen=port))
                if waited % 15 == 0:
                    n = ((running() or {}).get("restart") or {}).get("running")
                    note(f"正在等待服务停止（已等 {waited} 秒" + (f"，还有 {n} 个任务在计算" if n else "") + "；按 Ctrl-C 取消停止）……")
                time.sleep(1)
                waited += 1
        except abort_types():
            try:
                lab.admin_restart_cancel()
            except Lab2ShotError:
                pass
            say(Msg("N-SERVICE-STOPCANCELLED"), quiet=True)
            raise
    close_all()  # the machine token opened the database here: whatever comes next (an upgrade, a restore) needs it alone
    ok(f"服务已停止（端口 {port}，进程 {pid} 已退出）。")
    return True


def _end(pid: int | None, port: int, answering: bool) -> bool:
    """stop's fallback for a service that cannot be asked: the process listening on `port` ended by its pid, once
    confirmed. Returns False when there is none; raises NotStopped when it does not end, KeyboardInterrupt when the
    confirmation is declined."""
    from ..messages import Msg

    if pid is None:
        if answering:
            raise NotStopped(Msg("E-SERVICE-NOPID", address=recorded_address() or address()))
        note(f"端口 {port} 上没有服务。以其他端口（--port）启动的服务无法在此识别，请先在「设置 → 网络与安装」中将端口改为该端口，或手动停止该进程。")
        return False
    name = _command_of(pid)
    hint = "" if answering else "（它不回答 /api/server：可能不是 Lab2Shot、属于另一个项目目录的服务，或已失去响应）"
    if not typer.confirm(f"是否结束端口 {port} 上的进程 {pid}{'（' + name + '）' if name else ''}{hint}？正在计算的任务将被中断", default=False):
        raise KeyboardInterrupt
    terminate(pid, port)
    return True


def terminate(pid: int, port: int) -> None:
    """End process `pid` (SIGTERM) and wait up to KILL_WAIT_S seconds until nothing listens on `port` and the process
    has ended. Raises NotStopped when it did not."""
    from ..messages import Msg

    try:
        os.kill(pid, 15)
    except PermissionError as exc:
        raise NotStopped(Msg("E-SERVICE-NOPERMISSION", pid=pid)) from exc
    except ProcessLookupError:
        pass  # ended already; the port is waited for all the same
    for _ in range(KILL_WAIT_S):
        if listening_pid(port) is None and not _pid_alive(pid):
            return
        time.sleep(1)
    raise NotStopped(Msg("E-SERVICE-STILLUP", seconds=KILL_WAIT_S, listen=port))


def restart() -> None:
    from .accounts import admin_restart_cmd

    if not running():
        warn("服务未运行。请执行「服务 → 启动服务（后台）」。")
        return
    mode = "drain" if typer.confirm("是否等待正在计算的任务完成后再重启？（选择否将立即中断这些任务）", default=True) else "now"
    admin_restart_cmd(mode=mode, wait=True)
