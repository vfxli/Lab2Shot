"""`lab2shot setup`: the single interactive place to install, configure and run a server, for whoever has just cloned
the repository and for the administrator afterwards.

The menu is grouped by purpose. The top level lists the groups (first-time installation wizard, installation and
environment, accounts and security, service, settings, database, status overview); each group opens a second-level
menu, and the settings group opens a third level per settings group of the admin page. Every item performs exactly
one task and explains, before it runs, what it does, why, what it writes and where.

Nothing here is a second implementation: passwords go through lab2shot.accounts, settings through
lab2shot.config.Settings.save (the admin page's own path, with the same validation), the running service through the
local client (machine token), the database through lab2shot.database (status, backup, check, upgrade, restore) and
lab2shot.workdir (claim), and the server itself through `lab2shot ui`.

`setup.sh` at the repository root brings a fresh clone this far (uv and the Python environment, each only when
chosen) and then runs this command. `lab2shot setup <step>` runs one item directly without the menu.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import typer
from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from .base import app, console, when

LOG = "ui_stdout.log"  # the background service's terminal output, under <work>/logs/


# ------------------------------------------------------------------ output conventions

# One colour and one symbol per kind of message, used by every item: success, warning, error, and neutral notes.


def _ok(text: str) -> None:
    console.print(f"[green]✓[/green] {text}")


def _warn(text: str) -> None:
    console.print(f"[yellow]![/yellow] {text}")


def _err(text: str) -> None:
    console.print(f"[red]✗[/red] {text}")


def _note(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


def _mark(state: bool | None) -> str:
    """A table cell for a check: passed, failed, or not applicable."""
    return "" if state is None else ("[green]✓[/green]" if state else "[red]✗[/red]")


def _menu_table(rows: list[tuple[str, str, str]]) -> Table:
    """A menu as a table: number, name, description."""
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("编号", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("名称", no_wrap=True)
    table.add_column("说明")
    for key, name, note in rows:
        table.add_row(key, name, note)
    return table


def _pick(default: str = "0") -> str:
    """Read a menu number. Ctrl-C and end of input propagate (typer.Abort / click.Abort) to the caller."""
    return typer.prompt("请输入编号", default=default).strip().lower()


def _abort_types() -> tuple[type[BaseException], ...]:
    import click

    return (KeyboardInterrupt, typer.Abort, click.Abort)  # typer 0.27 has its own Abort, distinct from click's


# ------------------------------------------------------------------ what the menu inspects


def _initialized() -> bool:
    """Whether the work folder's database exists. Before the service's first start there is nothing to inspect, and
    inspecting it (local_client → machine token → db()) would create work/, owner.json, the database and the token;
    read-only items must not do so."""
    from ..config import settings
    from ..database import FILE

    return (settings().work_dir / "db" / FILE).is_file()


def _service():
    """The service on this work folder, if it answers: {"boot", "started", ...} or None. Queried over plain HTTP
    (`/api/server` requires no login), so the query creates nothing (no work/, no database, no machine token) and
    cannot collide with a server that is creating them at that moment (the first start polls this while the server
    is still building the database). The address queried is the recorded one when a server left its record
    (server/restart.py RECORD: the port or HTTPS setting may have changed since it started), otherwise the address
    of the current settings."""
    import json
    import ssl
    import urllib.request

    address = _recorded_address() or _address()
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # this machine's own self-signed certificate
        with urllib.request.urlopen(f"{address}/api/server", timeout=3, context=ctx) as r:
            got = json.loads(r.read().decode("utf-8"))
        return got if isinstance(got, dict) and "boot" in got else None
    except (OSError, ValueError):
        return None


def _recorded_address() -> str | None:
    """Where a server running on this work folder reported that it listens (server/restart.py RECORD); None when none
    is running. A record whose process no longer exists is stale (a crash or a kill) and is ignored."""
    import json

    from ..config import settings
    from ..server.restart import RECORD

    try:
        said = json.loads((settings().work_dir / RECORD).read_text(encoding="utf-8"))
        os.kill(int(said["pid"]), 0)
        return str(said["address"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _trouble(exc: Exception) -> None:
    """A WorkDirError or DatabaseError raised by any item: the message, followed by the available remedies."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    _err(escape(str(exc)))
    if isinstance(exc, WorkDirError):
        _note("可选的处理方式：改用另一个工作目录（设置环境变量 LAB2SHOT_WORK_DIR 后重新进入菜单）；"
              "或停止使用该目录的服务后，在本菜单「数据库 → 接管工作目录」中将其归属本项目目录。")
    elif isinstance(exc, DatabaseError):
        _note("启动服务时会先自动备份，再升级数据库；也可在「数据库 → 升级」中手动执行。"
              "数据库损坏或缺失时，可在「数据库 → 从备份恢复」中恢复。本菜单的其他项不会自动升级数据库。")


def _address() -> str:
    from ..config import settings

    s = settings()
    return f"{'https' if s['server.https'] else 'http'}://127.0.0.1:{s['server.port']}"


def _listening_pid(port: int) -> int | None:
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


def _admin_state() -> str:
    """The administrator's password state for the header, read only when the database already exists."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    if not _initialized():
        return "[dim]尚未初始化（首次启动服务时创建管理员账号）[/dim]"
    from .. import accounts

    try:
        owner = accounts.admin()
    except (WorkDirError, DatabaseError, OSError):
        return "[red]无法读取[/red]（详见「数据库 → 状态」）"
    if owner.no_password:
        return f"{owner.username}，[yellow]尚未设置密码，目前无法登录[/yellow]"
    return f"{owner.username}，[green]已设置密码[/green]"


def _header() -> None:
    """The title panel above the top-level menu: version, service, administrator, work folder."""
    from .. import __version__
    from ..config import settings

    state = _service()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    if state:
        grid.add_row("服务", f"[green]运行中[/green]  {_recorded_address() or _address()}（启动编号 {state.get('boot', '?')}）")
    else:
        grid.add_row("服务", f"[dim]未运行[/dim]  按当前设置的地址为 {_address()}")
    grid.add_row("管理员", _admin_state())
    grid.add_row("工作目录", str(settings().work_dir))
    grid.add_row("设置文件", str(settings().file))
    console.print(Panel(grid, title=f"[bold]Lab2Shot {__version__} 配置菜单[/bold]", title_align="left",
                        box=box.ROUNDED, border_style="cyan", padding=(0, 1)))


# ------------------------------------------------------------------ installation and environment


NODE_NEED = "Node.js 20.19 及以上（20.x）或 22.12 及以上"  # webui/package.json "engines" (^20.19.0 || >=22.12.0), as vite 8 requires


def _node_ok(version: str) -> bool:
    """`node --version` ("v22.12.0") against what vite 8 requires: ^20.19 or >=22.12. Unreadable: treated as adequate."""
    try:
        parts = tuple(int(p) for p in version.lstrip("v").split(".")[:3])
    except ValueError:
        return True
    return (20, 19) <= parts < (21,) or parts >= (22, 12)


def _check_environment() -> None:
    """One table: what the service requires on this machine, and whether it is present."""
    from ..config import ROOT, WEBUI_DIST, settings
    from ..workdir import owner

    rows: list[tuple[str, bool | None, str]] = []
    rows.append(("Python", True, sys.version.split()[0] + f"（{sys.executable}）"))
    rows.append(("uv", bool(shutil.which("uv")), shutil.which("uv") or "未安装。请参阅 https://docs.astral.sh/uv/ ，或重新运行 ./setup.sh 由脚本引导安装。"))
    node = shutil.which("node")
    node_v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip() if node else ""
    if not node:
        rows.append(("Node.js", False, f"未安装。构建网页需要 {NODE_NEED}（https://nodejs.org/）。"))
    elif _node_ok(node_v):
        rows.append(("Node.js", True, node_v))
    else:
        rows.append(("Node.js", False, f"{node_v} 版本过低。构建网页（vite 8）需要 {NODE_NEED}（https://nodejs.org/）。"))
    rows.append(("网页构建", WEBUI_DIST.is_dir(), str(WEBUI_DIST) if WEBUI_DIST.is_dir() else "尚未构建。请执行「安装与环境 → 构建网页」。"))
    smi = shutil.which("nvidia-smi")
    gpus = ""
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
        gpus = "；".join(line.strip() for line in out.splitlines() if line.strip())
    rows.append(("显卡驱动", bool(smi), gpus or "未找到 nvidia-smi。没有显卡时服务仍可启动，但无法执行显卡计算任务。"))
    s = settings()
    work = s.work_dir
    who = owner(work) if work.is_dir() else None
    if who is None:
        rows.append(("工作目录", True, f"{work}（{'已存在' if work.is_dir() else '将在首次启动服务时创建'}）"))
    elif who == str(ROOT):
        rows.append(("工作目录", True, f"{work}（归属本项目目录）"))
    else:
        rows.append(("工作目录", False, f"{work} 归属另一个项目目录 {who}。请改用其他 LAB2SHOT_WORK_DIR，或停止该处的服务后执行「数据库 → 接管工作目录」。"))
    rows.append(("设置文件", None, f"{s.file}（{'已存在' if Path(s.file).is_file() else '不存在，全部设置取默认值；修改设置后自动生成'}）"))
    table = Table("项目", "", "说明", box=box.SIMPLE_HEAD, header_style="bold")
    for name, ok, note in rows:
        table.add_row(name, _mark(ok), note)
    console.print(table)
    missing = [name for name, ok, _ in rows if ok is False]
    if missing:
        _warn(f"以下项目需要处理：{'、'.join(missing)}。")
    else:
        _ok("运行服务所需的条件均已具备。")


def _sync_env() -> None:
    from ..config import ROOT

    if not shutil.which("uv"):
        _err("未找到 uv。请重新运行 ./setup.sh 由脚本引导安装，或参阅 https://docs.astral.sh/uv/ 。")
        return
    _note(f"$ uv sync（工作目录：{ROOT}）")
    if subprocess.run(["uv", "sync"], cwd=ROOT).returncode == 0:
        _ok("Python 环境已就绪。")
        return
    mirror = _pypi_mirror()
    if mirror and typer.confirm(f"执行失败。是否使用 PyPI 镜像（{mirror}）重试？", default=True):
        env = {**os.environ, "UV_INDEX_URL": mirror}
        if subprocess.run(["uv", "sync"], cwd=ROOT, env=env).returncode == 0:
            _ok("Python 环境已就绪（经由镜像）。")
            return
    _err("执行失败，请查看上方输出。如网络不可用，请在「安装与环境 → 内核配置 → 镜像设置」中配置镜像，或配置代理后重试。")


def _pypi_mirror() -> str:
    from ..config import settings

    text = str(settings()["install.mirror_pypi"] or "").strip()
    return text.split()[0] if text else ""


def _toolchain() -> None:
    """The core build kit every compiled extension requires: what the settings point to, and whether it is present."""
    from ..config import settings
    from lab2shot_worker.build import compute_caps

    s = settings()
    cuda_home = s["build.cuda_home"] or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    rows = [("git", shutil.which("git"), "用于获取扩展包的源代码"),
            ("C 编译器", shutil.which(s["build.cc"] or "cc"), f"设置 build.cc = {s['build.cc'] or '（系统默认）'}"),
            ("C++ 编译器", shutil.which(s["build.cxx"] or "c++"), f"设置 build.cxx = {s['build.cxx'] or '（系统默认）'}"),
            ("CUDA 工具包", (Path(cuda_home) / "bin" / "nvcc").is_file() and str(Path(cuda_home) / "bin" / "nvcc"), f"设置 build.cuda_home = {s['build.cuda_home'] or '（$CUDA_HOME 或 /usr/local/cuda）'}"),
            ("ninja", shutil.which("ninja"), "可加快 CUDA 算子的编译（可选）")]
    table = Table("项目", "", "说明", box=box.SIMPLE_HEAD, header_style="bold")
    for name, found, note in rows:
        table.add_row(name, _mark(bool(found)), f"{found}  {note}" if found else note)
    console.print(table)
    caps = compute_caps()
    console.print(f"显卡架构：{'、'.join(caps) if caps else '（未检测到显卡）'}。编译 CUDA 算子时按此架构生成代码。")
    from ..installer import toolchain

    state, why = toolchain.problem()
    text = f"nvcc 与编译器的兼容性：{why.text if why else '兼容'}"
    {"ok": _ok, "warning": _warn, "blocked": _err}[state](text)


def _install_formats() -> None:
    """The format modules (Alembic, FBX): every card that reads or writes DCC scene files depends on them."""
    from ..extensions import extensions
    from ..extensions.status import extension_status

    from .extensions import _checklist, _run_install

    mods = [e for e in extensions().values() if e.format_module]
    for ext in mods:
        state = extension_status(ext)
        console.print(f"[bold]{ext.title}[/bold]  {'已安装' if state['installed'] else '未安装'}  许可证：{ext.license.name}")
        if state["installed"] and not typer.confirm("是否重新安装？", default=False):
            continue
        if not typer.confirm("是否安装？", default=True):
            continue
        if not _checklist(ext):
            _warn("上方存在未满足的条件，请先解决后再安装。")
            continue
        _run_install(ext, stop=False)


def _hf_login() -> None:
    """The weights whose repository requires an approved access request, and the Hugging Face login they need."""
    from huggingface_hub import get_token

    from ..extensions import extensions

    rows = [(ext.title, w.key, w.page) for ext in sorted(extensions().values(), key=lambda e: e.title.lower())
            for w in ext.weights if w.gated]
    table = Table("扩展包", "权重", "申请访问的页面", box=box.SIMPLE_HEAD, header_style="bold")
    for title, key, page in rows:
        table.add_row(title, key, page)
    console.print(table)
    console.print("上述权重的仓库须先申请访问：使用 Hugging Face 账号打开对应页面，按页面要求填写并同意许可条款，"
                  "获批后本机须以该账号登录，安装器方可下载。未使用这些扩展包时无需登录。")
    if os.environ.get("HF_TOKEN"):
        _ok("已登录：令牌来自环境变量 HF_TOKEN。")
        return
    if get_token():
        _ok("已登录：本机已保存 Hugging Face 令牌。")
        if not typer.confirm("是否重新登录？", default=False):
            return
    elif not typer.confirm("本机尚未登录 Hugging Face。是否现在登录？", default=True):
        return
    _note("令牌在 https://huggingface.co/settings/tokens 创建，权限选择 Read 即可。")
    hf = Path(sys.executable).parent / "hf"
    if not hf.is_file():
        _err(f"未找到 {hf}。请先执行「安装与环境 → Python 环境」。")
        return
    if subprocess.run([str(hf), "auth", "login"]).returncode != 0:
        _err("登录未完成，请查看上方输出。")
        return
    _ok("登录完成。")


MANUAL_STATES = {"ready": "[green]已就绪[/green]", "consent": "[yellow]待同意许可[/yellow]",
                 "unrecognised": "[red]文件无法识别[/red]", "missing": "未下载"}


def _manual_downloads() -> None:
    """The files people download by hand: where, which one, what for, and whether each is in place. The files already
    in downloads/ are recognised and installed first."""
    from ..extensions import manual

    view = manual.check()
    for row in view["items"]:
        needs = "、".join(e["title"] for e in row["needed_by"]) or "—"
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", no_wrap=True)
        grid.add_column(overflow="fold")
        grid.add_row("状态", MANUAL_STATES.get(row["state"], escape(row["state"]))
                     + (f"  {escape(row['installed'])}" if row["installed"] else ""))
        grid.add_row("用途", escape(row["what"]))
        grid.add_row("所需扩展包", escape(needs))
        grid.add_row("下载页面", escape(row["page"]))
        grid.add_row("下载哪一项", escape(row["download"]))
        grid.add_row("文件名", escape(row["filename"]))
        grid.add_row("说明", escape(row["note"]))
        for f in row["files"]:
            grid.add_row("downloads/ 中", escape(f"{f['name']}：{f.get('why') or '待同意许可'}"))
        console.print(Panel(grid, title=f"[bold]{escape(row['title'])}[/bold]", title_align="left", border_style="blue",
                            box=box.ROUNDED))
    for f in view["unknown"]:
        _warn(escape(f"downloads/ 中的 {f['name']} 无法识别：{f['why']}"))
    console.print(f"下载的文件原样放入 {escape(view['inbox']['path'])}（无需解压或改名），再执行本项即可识别并安装；"
                  "原始文件随后移入 downloads/installed/。标为「待同意许可」的项目须在管理后台「扩展包 → 手动下载」中阅读许可协议并同意后安装。"
                  "完整说明见 docs/manual-downloads.md。")


KERNEL_ITEMS: tuple[tuple[str, str, str, Callable[[], None]], ...] = (
    ("1", "编译环境设置", "CUDA 路径、C 与 C++ 编译器",
     lambda: _edit_settings(("build.cuda_home", "build.cc", "build.cxx"), "编译环境设置")),
    ("2", "工具链检查", "git、编译器、nvcc、显卡架构，以及 nvcc 与 GCC 的兼容性（只读）", lambda: _toolchain()),
    ("3", "镜像设置", "Hugging Face、PyPI、GitHub 镜像；官方地址无法访问时按顺序尝试",
     lambda: _edit_settings(("install.mirror_hf", "install.mirror_pypi", "install.mirror_github"), "镜像设置")),
    ("4", "重试设置", "下载失败时的重试次数与最长等待时间",
     lambda: _edit_settings(("install.retries", "install.backoff_max"), "重试设置")),
    ("5", "安装格式模块", "Alembic、FBX：读写 DCC 场景文件的模板均依赖这两个模块", lambda: _install_formats()),
)


def _kernel() -> None:
    """The third-level menu of 内核配置: what compiling extensions and building their environments depend on."""
    while True:
        console.print(Rule("[bold]内核配置[/bold]", align="left", style="cyan"))
        console.print(_menu_table([(k, name, note) for k, name, note, _ in KERNEL_ITEMS] + [("0", "返回", "返回上一级菜单")]))
        try:
            choice = _pick()
        except _abort_types():
            console.print()
            return
        if choice == "0":
            return
        found = next((item for item in KERNEL_ITEMS if item[0] == choice), None)
        if found is None:
            _err("没有该编号，请重新输入。")
            continue
        _attempt(found[3])


def _build_webui() -> None:
    from ..config import ROOT

    if not shutil.which("npm"):
        _err("未找到 npm。请先安装 Node.js（https://nodejs.org/）。")
        return
    web = ROOT / "webui"
    _note(f"$ npm install（工作目录：{web}）")
    if subprocess.run(["npm", "install"], cwd=web).returncode != 0:
        if not typer.confirm("执行失败。是否使用 npm 镜像（registry.npmmirror.com）重试？", default=True):
            return
        if subprocess.run(["npm", "install", "--registry=https://registry.npmmirror.com"], cwd=web).returncode != 0:
            _err("执行失败，请查看上方输出。")
            return
    _note(f"$ npm run build（工作目录：{web}）")
    if subprocess.run(["npm", "run", "build"], cwd=web).returncode != 0:
        _err("执行失败，请查看上方输出。")
        return
    _ok("网页已构建完成。")


# ------------------------------------------------------------------ accounts and security


def _not_initialized() -> None:
    _warn("工作目录尚未初始化。账号与数据库由首次启动的服务创建，请先执行「服务 → 启动服务（后台）」，再返回此处设置。")


def _admin_password() -> None:
    if not _initialized():  # this item never creates work/ or the database: the first start of the service does
        _not_initialized()
        return
    from .. import accounts

    from .accounts import admin_password

    owner = accounts.admin()
    if owner.no_password:
        _warn(f"管理员 {owner.username} 尚未设置密码，在设置之前任何人都无法登录。请现在设置。")
    else:
        console.print(f"管理员 {owner.username} 的密码于 {when(owner.password_set or 0)} 经由{owner.password_by}设置。")
        if not typer.confirm("是否重新设置？", default=False):
            return
    admin_password()


def _passphrase() -> None:
    if not _initialized():  # this item never creates work/ or the database: the first start of the service does
        _not_initialized()
        return
    from .. import accounts

    from .accounts import admin_passphrase

    phrase = accounts.passphrase()
    if phrase is not None:
        console.print(f"找回口令已于 {when(phrase['set'])} 设置。")
        if not typer.confirm("是否更换？", default=False):
            return
    else:
        console.print("找回口令用于忘记管理员密码的情形：在登录页选择「忘记密码」，凭此口令设置新密码。")
    admin_passphrase()


# ------------------------------------------------------------------ settings


def _ask_setting(key: str, current: object):
    """Ask for one setting in the form its kind requires: a numbered choice, yes or no, a number, a list, or text."""
    from ..config import SCHEMA

    s = SCHEMA[key]
    if s.kind == "choice":
        options = list(s.options)
        for i, (value, label) in enumerate(options, 1):
            console.print(f"  [bold cyan]{i}[/bold cyan]  {label}（{value}）" + ("  [dim]← 当前值[/dim]" if value == current else ""))
        n = typer.prompt(f"{s.label}（请输入编号）", default=str(next((i for i, (v, _) in enumerate(options, 1) if v == current), 1)))
        try:
            return options[int(n) - 1][0]
        except (ValueError, IndexError):
            _warn("编号无效，保留原值。")
            return current
    if s.kind == "bool":
        return typer.confirm(f"{s.label}：是否开启？", default=bool(current))
    bounds = ""
    if s.kind in ("int", "number") and (s.min is not None or s.max is not None):
        bounds = f"（范围 {s.min if s.min is not None else ''}–{s.max if s.max is not None else ''}{' ' + s.unit if s.unit else ''}）"
    if s.kind == "int":
        return typer.prompt(s.label + bounds, default=int(current), type=int)
    if s.kind == "number":
        return typer.prompt(s.label + bounds, default=float(current), type=float)
    if s.kind == "list":
        return typer.prompt(f"{s.label}（以空格分隔）", default=s.says(current))
    text = typer.prompt(s.label + ("（留空表示" + s.empty + "）" if getattr(s, "empty", "") else ""), default=str(current or ""), show_default=bool(current))
    return text.strip()


def _save_settings(changes: dict[str, object]) -> list[str] | None:
    """Save through Settings.save (the admin page's path and validation). Returns the keys that changed, or None when
    the values were refused (the reasons are printed)."""
    from ..config import InvalidSettings, SCHEMA, settings

    s = settings()
    try:
        changed = s.save(changes)
    except InvalidSettings as exc:
        for key, problem in exc.problems.items():
            _err(escape(problem.text))
        return None
    if not changed:
        _note("没有改动。")
        return []
    _ok(f"已保存：{'、'.join(SCHEMA[k].label for k in changed)}（写入 {s.file}）。")
    return changed


def _offer_restart(keys: list[str]) -> None:
    """After settings that take effect only on restart have changed: list them, and offer to restart the service."""
    from ..config import SCHEMA, settings

    keys = [k for k in dict.fromkeys(keys) if SCHEMA[k].restart]
    if not keys:
        return
    s = settings()
    table = Table("名称", "新值", "需要重启的原因", box=box.SIMPLE_HEAD, header_style="bold")
    for k in keys:
        table.add_row(SCHEMA[k].label, escape(SCHEMA[k].says(s.file_value(k))), SCHEMA[k].restart)
    console.print(Panel(table, title="[bold]以下设置需重启服务后生效[/bold]", title_align="left", border_style="yellow", box=box.ROUNDED))
    if not _service():
        _note("服务当前未运行；下次启动服务时即采用新的设置。")
        return
    if typer.confirm("服务正在运行。是否现在重启服务？", default=False):
        _attempt(_restart)
    else:
        _note("稍后可在「服务 → 重启服务」中重启。")


def _edit_settings(keys: tuple[str, ...], title: str) -> None:
    """Ask a fixed list of settings in turn, then save them together."""
    from ..config import SCHEMA, settings

    s = settings()
    console.print(f"[bold]{title}[/bold]  （直接按回车保留当前值；写入 {s.file.name}，仅记录与默认值不同的项）")
    changes = {}
    for key in keys:
        spec = SCHEMA[key]
        if spec.only_if and not changes.get(spec.only_if, s.file_value(spec.only_if)):
            continue
        if spec.help:
            _note(spec.help)
        changes[key] = _ask_setting(key, s.file_value(key))
    changed = _save_settings(changes)
    if changed:
        _offer_restart(changed)


def _source_of(key: str) -> str:
    """Where a setting's value comes from, as a short table cell."""
    from ..config import SCHEMA, settings

    s = settings()
    spec = SCHEMA[key]
    if key in s.command:
        return f"[magenta]已被覆盖（{s.command[key][1]}）[/magenta]"
    if not spec.admin:
        return "[dim]仅配置文件[/dim]"
    if key in s.saved:
        return "[cyan]已修改[/cyan]"
    if spec.auto:
        return "[dim]按本机计算[/dim]"
    return "[dim]默认[/dim]"


def _setting_detail(key: str, pending: list[str]) -> None:
    """The fourth level: one setting in full, then modify it or restore its default."""
    from ..config import SCHEMA, settings

    while True:
        s = settings()
        spec = SCHEMA[key]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", no_wrap=True)
        grid.add_column(overflow="fold")
        grid.add_row("设置键", key)
        grid.add_row("说明", escape(spec.help))
        grid.add_row("当前值", escape(spec.says(s.file_value(key))))
        if key in s.command:
            grid.add_row("本次运行", f"{escape(spec.says(s.value(key)))}（由{s.command[key][1]}指定，优先于设置文件）")
        grid.add_row("默认值", escape(spec.says(spec.default_now)) + (f"（{escape(spec.auto.says())}）" if spec.auto else ""))
        grid.add_row("生效方式", f"[yellow]需重启服务后生效[/yellow]：{spec.restart}" if spec.restart else "保存后立即生效")
        if spec.only_if:
            grid.add_row("前提", f"仅在「{SCHEMA[spec.only_if].label}」开启时起作用")
        console.print(Panel(grid, title=f"[bold]{spec.label}[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
        if not spec.admin:
            _warn(f"该项只能在配置文件 {s.file} 中修改，本菜单与管理后台均只显示不修改。")
            return
        rows = [("1", "修改", "输入新的值并保存")]
        if key in s.saved:
            rows.append(("2", "恢复默认值", f"恢复为 {escape(spec.says(spec.default_now))}"))
        rows.append(("0", "返回", "返回上一级菜单"))
        console.print(_menu_table(rows))
        choice = _pick()
        if choice == "0":
            return
        if choice == "1":
            changed = _save_settings({key: _ask_setting(key, s.file_value(key))})
        elif choice == "2" and key in s.saved:
            changed = _save_settings({key: spec.default_now})
        else:
            _err("没有该编号，请重新输入。")
            continue
        if changed:
            pending.extend(k for k in changed if SCHEMA[k].restart)
        return


def _settings_group(group: str, pending: list[str]) -> None:
    """The third level: every setting of one admin-page group, with its value, default and how it takes effect."""
    from ..config import GROUPS, SCHEMA, settings

    keys = [k for k, spec in SCHEMA.items() if spec.group == group]
    while True:
        s = settings()
        console.print(Rule(f"[bold]设置 → {GROUPS[group]}[/bold]", align="left", style="cyan"))
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("编号", justify="right", style="bold cyan", no_wrap=True)
        table.add_column("名称", no_wrap=True)
        table.add_column("当前值")
        table.add_column("默认值")
        table.add_column("生效", no_wrap=True)
        table.add_column("来源", no_wrap=True)
        for i, k in enumerate(keys, 1):
            spec = SCHEMA[k]
            name = spec.label
            if spec.only_if and not s.file_value(spec.only_if):
                name = f"[dim]{name}（需开启「{SCHEMA[spec.only_if].label}」）[/dim]"
            table.add_row(str(i), name, escape(spec.says(s.value(k))), escape(spec.says(spec.default_now)),
                          "[yellow]需重启[/yellow]" if spec.restart else "立即", _source_of(k))
        table.add_row("0", "返回", "", "", "", "")
        console.print(table)
        try:
            choice = _pick()
        except _abort_types():
            console.print()
            return
        if choice == "0":
            return
        try:
            key = keys[int(choice) - 1]
        except (ValueError, IndexError):
            _err("没有该编号，请重新输入。")
            continue
        try:
            _setting_detail(key, pending)
        except _abort_types():
            console.print()
            _note("已取消，未作修改。")


def _settings_menu() -> None:
    """The second level of 设置: the admin page's groups. Leaving it lists the changes that await a restart."""
    from ..config import GROUPS, SCHEMA, settings

    pending: list[str] = []
    ids = list(GROUPS)
    while True:
        s = settings()
        console.print(Rule("[bold]设置[/bold]", align="left", style="cyan"))
        _note(f"与管理后台「设置」页为同一份设置，写入 {s.file}，仅记录与默认值不同的项。")
        rows = []
        for i, g in enumerate(ids, 1):
            specs = [spec for spec in SCHEMA.values() if spec.group == g]
            edited = sum(1 for spec in specs if spec.key in s.saved)
            restart = sum(1 for spec in specs if spec.restart)
            rows.append((str(i), GROUPS[g], f"共 {len(specs)} 项；已修改 {edited} 项；需重启生效 {restart} 项"))
        rows.append(("0", "返回", "返回上一级菜单" + ("，并列出需重启生效的改动" if pending else "")))
        console.print(_menu_table(rows))
        file_only = [spec for spec in SCHEMA.values() if not spec.admin]
        console.print(Panel(
            "以下设置请在管理后台或配置文件中修改：\n"
            + "".join(f"  · {spec.label}（{spec.key}）：只能在配置文件 {s.file.name} 中修改，或以环境变量 LAB2SHOT_WORK_DIR 指定。\n" for spec in file_only)
            + "  · 账号、部门归属、存储配额与管理员通知：请在管理后台（/admin）的相应页面中管理。\n"
            + "  · 接受任务的显卡：可在本菜单「服务 → 显卡授权」或管理后台中设置。",
            title="[bold]说明[/bold]", title_align="left", border_style="dim", box=box.ROUNDED))
        try:
            choice = _pick()
        except _abort_types():
            console.print()
            break
        if choice == "0":
            break
        try:
            group = ids[int(choice) - 1]
        except (ValueError, IndexError):
            _err("没有该编号，请重新输入。")
            continue
        _settings_group(group, pending)
    _offer_restart(pending)


# ------------------------------------------------------------------ service


def _gpus() -> None:
    from ..client import Lab2ShotError

    from .accounts import local_client

    if not _service():
        _warn("服务未运行。显卡由运行中的服务识别，请先启动服务再进行授权。")
        return
    try:
        cards = local_client().admin_cards()["cards"]
    except Lab2ShotError as exc:
        _err(escape(str(exc)))
        return
    if not cards:
        _warn("服务未识别到任何显卡。")
        return
    table = Table("编号", "显卡", "显存", "架构", "状态", box=box.SIMPLE_HEAD, header_style="bold")
    for i, c in enumerate(cards, 1):
        table.add_row(str(i), c["model"], f"{c['memory_gb']:g} GB", c["arch"], "[green]接受任务[/green]" if c["authorized"] else "[dim]不接受任务[/dim]")
    console.print(table)
    picked = typer.prompt("请输入接受任务的显卡编号（以空格分隔；all 表示全部，none 表示全部不接受）", default="all")
    if picked.strip() == "all":
        uuids = [c["uuid"] for c in cards]
    elif picked.strip() == "none":
        uuids = []
    else:
        try:
            uuids = [cards[int(n) - 1]["uuid"] for n in picked.split()]
        except (ValueError, IndexError):
            _err("编号无效，未作修改。")
            return
    try:
        local_client().admin_authorize(uuids)
    except Lab2ShotError as exc:
        _err(escape(str(exc)))
        return
    _ok(f"已授权 {len(uuids)} 张显卡接受任务。")


def _start(background: bool) -> None:
    from ..config import WEBUI_DIST, settings
    from ..database import close_all

    s = settings()
    port = int(s["server.port"])
    if _service():
        recorded = _recorded_address()
        _note(f"服务已在运行：{recorded or _address()}。")
        if recorded and recorded != _address():
            _warn(f"当前设置的地址为 {_address()}；端口或 HTTPS 的修改需重启服务后生效。")
        return
    if (pid := _listening_pid(port)) is not None:
        _err(f"端口 {port} 已被进程 {pid} 占用，不再启动新的服务。如该进程不是 Lab2Shot，请在「设置 → 网络」中更换端口。")
        return
    if not WEBUI_DIST.is_dir():
        _warn("网页尚未构建。请先执行「安装与环境 → 构建网页」。")
        return
    close_all()  # this menu's own handle on the database must not block the server's upgrade at start
    exe = Path(sys.executable).parent / "lab2shot"
    cmd = [str(exe), "ui"] if exe.is_file() else [sys.executable, "-c", "from lab2shot.cli import app; app()", "ui"]
    if not background:
        _note(f"服务在前台运行，按 Ctrl-C 停止：{_address()}")
        subprocess.call(cmd)
        return
    logs = s.work_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = open(logs / LOG, "ab")
    subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    _note("正在等待服务启动（最长 120 秒）……")
    for _ in range(120):
        time.sleep(1)
        if _service():
            _ok(f"服务已启动：{_recorded_address() or _address()}（日志：{logs / LOG}）。")
            _first_time_hints()
            return
    _err(f"服务未能在 120 秒内启动，请查看日志 {logs / LOG}。")


def _first_time_hints() -> None:
    from .. import accounts
    from ..config import settings

    if settings()["server.https"]:
        from ..server import tls

        # the address other machines reach this one by (hostname -I), not 127.0.0.1; unknown: the placeholder serve.py uses
        lan = [ip for ip in tls.names()[1] if ip not in ("127.0.0.1", "::1") and ":" not in ip]
        host = lan[0] if lan else "<本机地址>"
        _note(f"已启用 HTTPS：每台用户电脑需安装一次证书。请在浏览器中打开 https://{host}:{settings()['server.port']}/api/tls/ca.pem 下载，"
              "并导入系统的「受信任的根证书颁发机构」。")
    if accounts.admin().no_password:
        _warn(f"管理员 {accounts.admin().username} 尚未设置密码，目前任何人都无法登录。请执行「账号与安全 → 管理员密码」。")


def _command_of(pid: int) -> str:
    """The command line of process `pid` (Linux /proc), for the confirmation; "" when it cannot be read."""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()[:80]
    except OSError:
        return ""


def _service_port() -> int:
    """The port of the service on this work folder: the recorded one when a server left its record (it may have been
    started with --port), otherwise the one in the settings."""
    from urllib.parse import urlparse

    from ..config import settings

    recorded = _recorded_address()
    try:
        if recorded and (port := urlparse(recorded).port):
            return port
    except ValueError:
        pass
    return int(settings()["server.port"])


def _stop() -> None:
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    port = _service_port()
    try:
        state = _service()  # query the service first: the port's pid alone does not say who listens there
    except (WorkDirError, DatabaseError):
        state = None  # stopping must still work when this checkout may not open the records: by pid below
    pid = _listening_pid(port)
    if pid is None:
        if state:
            _warn(f"服务正在运行（{_address()}），但无法获得其进程号：通常是由其他用户启动（ss 只显示当前用户的进程），或系统中没有 ss 命令。"
                  "请由启动服务的用户停止该进程。")
        else:
            _note(f"端口 {port} 上没有服务。以其他端口（--port）启动的服务无法在此识别，请先在「设置 → 网络」中将端口改为该端口，或手动停止该进程。")
        return
    name = _command_of(pid)
    note = "" if state else "（无法通过本机令牌确认：该进程可能不是 Lab2Shot，或属于另一个项目目录的服务）"
    if not typer.confirm(f"是否停止端口 {port} 上的进程 {pid}{'（' + name + '）' if name else ''}{note}？正在计算的任务将被中断", default=False):
        return
    try:
        os.kill(pid, 15)
    except PermissionError:
        _err(f"没有权限停止进程 {pid}：该进程由其他用户启动，请以该用户（或 sudo）执行 kill {pid}。")
        return
    except ProcessLookupError:
        _note("该进程已不存在。")
        return
    for _ in range(30):
        time.sleep(1)
        if _listening_pid(port) is None:
            _ok("服务已停止。")
            return
    _warn("服务尚未退出，请稍候或自行检查该进程。")


def _restart() -> None:
    from .accounts import admin_restart_cmd

    if not _service():
        _warn("服务未运行。请执行「服务 → 启动服务（后台）」。")
        return
    mode = "drain" if typer.confirm("是否等待正在计算的任务完成后再重启？（选择否将立即中断这些任务）", default=True) else "now"
    admin_restart_cmd(mode=mode, wait=True)


# ------------------------------------------------------------------ database

REASONS = {"manual": "手动备份", "daily": "每日自动备份", "created": "创建数据库时"}


def _backup_reason(name: str) -> str:
    """What a backup was made for, from its file name (lab2shot-<date>-<time>-<reason>[~n].db, database/__init__.py)."""
    parts = Path(name).stem.split("-", 3)
    reason = parts[3].split("~", 1)[0] if len(parts) == 4 else ""
    if reason.startswith("before-v") and reason[len("before-v"):].isdigit():
        return f"升级至第 {reason[len('before-v'):]} 版前自动备份"
    return REASONS.get(reason, reason or "未知")


def _backup_time(path: Path) -> str:
    """When a backup was made: from its file name, otherwise from its modification time."""
    parts = path.stem.split("-", 3)
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S"))
    except (IndexError, ValueError):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))


def _backups() -> list[Path]:
    """The backups kept for this work folder, newest first. Read from the folder only: the database is not opened."""
    from ..database import backups_folder

    found = backups_folder()
    return sorted(found.glob("*.db"), reverse=True) if found.is_dir() else []


def _size(n: float) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.1f} KB"


def _backups_table(found: list[Path]) -> Table:
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("编号", justify="right", style="bold cyan", no_wrap=True)
    table.add_column("时间", no_wrap=True)
    table.add_column("大小", justify="right", no_wrap=True)
    table.add_column("来由")
    table.add_column("文件名", style="dim")
    for i, p in enumerate(found, 1):
        table.add_row(str(i), _backup_time(p), _size(p.stat().st_size), _backup_reason(p.name), p.name)
    return table


def _db_status() -> None:
    from ..config import settings
    from ..database import FILE, DatabaseError, db, folder
    from ..database import schema
    from ..workdir import WorkDirError

    found = _backups()
    if not _initialized():  # read-only: never create work/ or the database for a look
        if found:
            _warn(f"数据库文件 {folder() / FILE} 不存在，但备份目录中有 {len(found)} 份备份。可在「数据库 → 从备份恢复」中恢复。")
            console.print(_backups_table(found))
        else:
            _note(f"数据库尚未创建：首次启动服务时在 {settings().work_dir / 'db'} 中创建。")
        return
    try:
        s = db().status()
    except (WorkDirError, DatabaseError) as exc:
        _trouble(exc)
        if found:
            console.print(_backups_table(found))
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("文件", s["path"])
    grid.add_row("大小", _size(s["bytes"]) + "（含预写日志文件）")
    grid.add_row("版本", f"第 {s['version']} 版（本程序对应第 {schema.VERSION} 版）")
    checked = s["checked"]
    if checked:
        grid.add_row("完整性", ("[green]完好[/green]" if checked.get("ok") else f"[red]存在问题：{escape(checked.get('detail', ''))}[/red]")
                     + f"（检查于 {when(checked['at'])}）")
    else:
        grid.add_row("完整性", "[dim]尚未检查[/dim]")
    last = s["last_backup"]
    grid.add_row("上次备份", f"{when(last['at'])}，{_backup_reason(last['file'])}（{last['file']}）" if last else "[yellow]尚无备份[/yellow]")
    grid.add_row("保留份数", f"{s['keep']} 份（设置项「备份份数」；超出时删除最旧的备份）")
    console.print(Panel(grid, title="[bold]数据库[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    if found:
        console.print(_backups_table(found))
    else:
        _note("备份目录中没有备份。")


def _db_backup() -> None:
    from ..database import db

    if not _initialized():
        _warn("数据库尚未创建，无需备份。首次启动服务时会创建数据库并自动备份一次。")
        return
    target = db().backup("manual")
    _ok(f"已备份：{target}（{_size(target.stat().st_size)}）。")


def _db_check() -> None:
    from ..database import db

    if not _initialized():
        _warn("数据库尚未创建，无可检查的内容。")
        return
    found = db().check()
    if found["ok"]:
        _ok("数据库完好：完整性检查与外键检查均已通过。")
    else:
        _err(f"数据库存在问题：{escape(found['detail'])}。可在「数据库 → 从备份恢复」中恢复。")


def _ensure_stopped(action: str) -> bool:
    """For an action that requires the database to itself: when the service is running, say so and offer to stop it.
    Returns whether the service is (now) stopped."""
    if not _service():
        return True
    _warn(f"服务正在运行。{action}前必须停止服务。")
    console.print(_menu_table([("1", "现在停止服务", "停止本工作目录的服务；正在计算的任务将被中断"), ("0", "返回", "不执行任何操作")]))
    if _pick() != "1":
        return False
    _stop()
    if _service():
        _err("服务仍在运行，已取消操作。")
        return False
    return True


def _offer_start() -> None:
    _warn("请重新启动服务，以使用新的数据库。")
    if typer.confirm("是否现在启动服务（后台）？", default=False):
        _start(background=True)
    else:
        _note("稍后可在「服务 → 启动服务（后台）」中启动。")


def _db_restore() -> None:
    from ..database import FILE, close_all, folder, restore

    found = _backups()
    if not found:
        _warn("备份目录中没有备份，无法恢复。")
        return
    console.print(_backups_table(found))
    choice = typer.prompt("请输入要恢复的备份编号（0 表示返回）", default="0").strip()
    if choice == "0":
        return
    try:
        source = found[int(choice) - 1]
    except (ValueError, IndexError):
        _err("编号无效，未作任何操作。")
        return
    if not _ensure_stopped("恢复数据库"):
        return
    current = folder() / FILE
    stamp = time.strftime("%Y%m%d-%H%M%S")
    consequences = Table.grid(padding=(0, 2))
    consequences.add_column(style="bold", no_wrap=True)
    consequences.add_column(overflow="fold")
    consequences.add_row("恢复来源", f"{source.name}（{_backup_time(source)}，{_backup_reason(source.name)}）")
    consequences.add_row("当前数据库", f"移至 {current.with_name(f'{FILE}.replaced-<时间>')}，例如 {FILE}.replaced-{stamp}；不会删除，可随时找回。"
                         if current.exists() else "不存在，将直接放入备份。")
    consequences.add_row("数据变化", "备份之后产生的记录（账号、任务、使用统计、结果记录等）将不在恢复后的数据库中。")
    consequences.add_row("附带文件", "备份附带的反馈截图与基准文件中，工作目录现已缺失的部分会复制回原位置；已有文件不受影响。")
    console.print(Panel(consequences, title="[bold]恢复的后果[/bold]", title_align="left", border_style="yellow", box=box.ROUNDED))
    if not typer.confirm("是否确认恢复？", default=False):
        _note("已取消，未作任何操作。")
        return
    again = typer.prompt("为防止误操作，请再次输入所选备份的编号", default="").strip()
    if again != choice:
        _note("两次输入的编号不一致，已取消，未作任何操作。")
        return
    close_all()  # this menu's own handle on the database would otherwise hold it (E-DB-INUSE)
    replaced = restore(source.name)
    _ok(f"已从 {source.name} 恢复数据库。")
    _ok(f"原数据库已移至 {replaced}。")
    _offer_start()


def _db_upgrade() -> None:
    from ..database import close_all, db, schema

    if not _initialized():
        _warn("数据库尚未创建，无需升级。首次启动服务时会创建最新版本的数据库。")
        return
    if not _ensure_stopped("升级数据库"):
        return
    close_all()  # the upgrade needs the database to itself, including this menu's own handle
    d = db(upgrade=True)
    _ok(f"数据库为第 {d.version} 版（本程序对应第 {schema.VERSION} 版）：{d.path}。升级前的数据库已自动备份。")


def _db_claim() -> None:
    from ..config import ROOT, settings
    from ..workdir import claim, owner

    work = settings().work_dir
    if not work.is_dir():
        _note(f"工作目录 {work} 尚不存在，无需接管：首次启动服务时自动归属本项目目录。")
        return
    was = owner(work)
    if was == str(ROOT):
        _ok(f"工作目录 {work} 已归属本项目目录 {ROOT}，无需接管。")
        return
    _warn(f"工作目录 {work} 目前归属 {was or '（未记录）'}。接管后，其他项目目录（其他分支、测试检出）的进程均无法再打开它。"
          "请先停止原来使用该目录的服务。")
    if not typer.confirm(f"是否将其改为归属 {ROOT}？", default=False):
        _note("已取消，未作任何操作。")
        return
    claim(work)
    _ok(f"工作目录 {work} 现归属 {ROOT}。")


# ------------------------------------------------------------------ status overview


def _status() -> None:
    from .. import accounts
    from ..config import settings
    from ..extensions import extensions

    s = settings()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    if not _initialized():  # read-only: never create work/, the database or the machine token for a look
        grid.add_row("服务", f"[dim]未运行[/dim]（按当前设置的地址为 {_address()}）")
        grid.add_row("工作目录", f"{s.work_dir}，尚未初始化：首次启动服务时创建数据库、管理员账号与本机令牌")
        grid.add_row("管理员", f"{accounts.ADMIN_NAME}，没有默认密码：服务首次启动后请在「账号与安全 → 管理员密码」中设置，设置之前无法登录")
        grid.add_row("设置文件", str(s.file))
        console.print(Panel(grid, title="[bold]状态总览[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
        return
    state = _service()
    if state:
        grid.add_row("服务", f"[green]运行中[/green]  {_recorded_address() or _address()}，启动编号 {state.get('boot', '?')}")
    else:
        grid.add_row("服务", f"[dim]未运行[/dim]（按当前设置的地址为 {_address()}）")
    owner = accounts.admin()
    grid.add_row("管理员", f"{owner.username}，" + ("[yellow]尚未设置密码，目前无法登录[/yellow]" if owner.no_password else "[green]已设置密码[/green]"))
    grid.add_row("找回口令", "[green]已设置[/green]" if accounts.passphrase() else "[yellow]未设置[/yellow]")
    users = [u for u in accounts.listing() if not u["deleted"]]
    grid.add_row("账号", f"{len(users)} 个（在管理后台「用户」页中新建）")
    grid.add_row("扩展包", f"{len(extensions())} 个可安装（已安装的情况见管理后台「扩展包」页，或执行 lab2shot ext list）")
    from ..database import db

    d = db().status()
    last = d["last_backup"]
    grid.add_row("数据库", f"第 {d['version']} 版，{_size(d['bytes'])}；上次备份 {when(last['at']) if last else '无'}；共 {len(d['backups'])} 份备份")
    grid.add_row("工作目录", str(s.work_dir))
    grid.add_row("设置文件", f"{s.file}（已修改 {len(s.saved)} 项）")
    console.print(Panel(grid, title="[bold]状态总览[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))


# ------------------------------------------------------------------ the menu


# Before each item runs, four facts are stated: the operation, its purpose, what it writes, and its scope. Every item
# touches the project folder only (.venv, webui, config, work, third_party, downloads); system components (gcc, CUDA,
# Node.js, the display driver) are only checked here, never installed. Two steps write to the user's home folder: the
# installation of uv in setup.sh (to ~/.local/bin) and hf-login (the Hugging Face token).
PROJECT = "项目目录内；不安装系统级组件"
READ_ONLY = "只读，不作任何修改"


@dataclass(frozen=True)
class Step:
    name: str  # the step name of `lab2shot setup <step>`
    label: str
    summary: str  # one line for the menu table
    what: str  # 操作
    why: str  # 用途
    changes: str  # 写入位置
    scope: str  # 影响范围
    run: Callable[[], None]


def _wizard() -> None:
    """The first-time installation wizard: the essential steps in order, each explained and confirmed, each skippable."""
    names = WIZARD
    console.print(Panel(
        "本向导按顺序引导完成首次安装所需的六个步骤：" + " → ".join(STEPS[n].label for n in names) + "。\n"
        "每一步开始前均说明其内容并询问是否执行；选择否即跳过该步骤。按 Ctrl-C 可随时退出向导。",
        title="[bold]首次安装向导[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    results: list[tuple[str, str]] = []
    for i, name in enumerate(names, 1):
        step = STEPS[name]
        console.print(Rule(f"[bold]第 {i} 步，共 {len(names)} 步：{step.label}[/bold]", align="left", style="cyan"))
        _explain(step)
        if not typer.confirm("是否执行此步骤？", default=True):
            _note("已跳过此步骤。")
            results.append((step.label, "[dim]已跳过[/dim]"))
            continue
        results.append((step.label, "[green]已执行[/green]" if _attempt(step.run) else "[yellow]未完成[/yellow]"))
    table = Table("步骤", "结果", box=box.SIMPLE_HEAD, header_style="bold")
    for label, result in results:
        table.add_row(label, result)
    console.print(Panel(table, title="[bold]向导结果[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    _note("未完成或已跳过的步骤可随时在对应分组中单独执行。")


STEP_LIST: tuple[Step, ...] = (
    Step("wizard", "首次安装向导", "依次完成 Python 环境、构建网页、检查环境、启动服务、设置管理员密码、显卡授权",
         "依次引导执行首次安装所需的六个步骤，每一步均可跳过", "从全新检出到可以登录使用", "取决于所执行的步骤", PROJECT, _wizard),
    # 安装与环境
    Step("env", "Python 环境", "按 uv.lock 安装依赖（uv sync）",
         "执行 uv sync，按 uv.lock 安装依赖", "提供服务与命令行的运行环境", ".venv/；包缓存位于 ~/.cache/uv", PROJECT, _sync_env),
    Step("web", "构建网页", "安装前端依赖并构建网页界面",
         "执行 npm install 与 npm run build", "生成服务向浏览器提供的页面", "webui/node_modules/、webui/dist/", PROJECT, _build_webui),
    Step("check", "检查环境", "列出运行服务所需条件的状态（只读）",
         "列出 Python、uv、Node.js、网页构建、显卡驱动、工作目录与设置文件的状态", "确认缺少的条件", "无", READ_ONLY, _check_environment),
    Step("kernel", "内核配置", "编译环境、工具链、镜像、重试与格式模块",
         "编译环境设置、工具链检查、下载镜像、重试策略、安装格式模块（Alembic、FBX）", "扩展包在独立环境中编译 CUDA 算子、获取代码与模型权重时所依赖的设置",
         "config/local.toml；格式模块安装至 third_party/，原始文件移入 downloads/installed/", PROJECT, _kernel),
    Step("hf-login", "Hugging Face 登录", "列出须申请访问的模型，并在本机登录 Hugging Face",
         "列出仓库须申请访问的权重及其申请页面，并执行 hf auth login", "获批后，安装器以本机登录的账号下载这些权重",
         "Hugging Face 的令牌文件（默认 ~/.cache/huggingface/token，设置了 HF_HOME 时位于其中）", "当前用户主目录中的 Hugging Face 登录状态；不改动项目目录", _hf_login),
    Step("downloads", "手动下载", "需自行下载的文件：下载地址、下载哪一项、是否就绪",
         "识别并安装 downloads/ 中的文件，列出每一项的下载页面、所需文件与状态",
         "部分模型须在其网站注册或同意许可后才能下载，安装器无法代为获取",
         "third_party/_body_models/ 或对应扩展包的 weights/；原始文件移入 downloads/installed/", PROJECT, _manual_downloads),
    # 账号与安全
    Step("password", "管理员密码", "设置或重设管理员账号的密码",
         "设置或重设管理员账号的密码", "新安装时管理员没有密码，设置之前任何人都无法登录",
         "work/db 中的账号记录（需先启动过一次服务）；立即生效，已登录的浏览器须重新登录", PROJECT, _admin_password),
    Step("passphrase", "找回口令", "设置忘记管理员密码时使用的口令",
         "设置忘记管理员密码时使用的口令", "登录页「忘记密码」凭此口令重设密码；仅可在本机设置",
         "work/db 中保存口令的哈希值（需先启动过一次服务）", PROJECT, _passphrase),
    # 服务
    Step("start", "启动服务（后台）", "在后台启动服务，日志写入 work/logs/",
         "在后台启动 lab2shot ui", "提供网页与 DCC 插件所连接的服务",
         "work/logs/ui_stdout.log；首次启动时由服务创建 work/（数据库、管理员账号、本机令牌）", PROJECT, lambda: _start(background=True)),
    Step("start-fg", "启动服务（前台）", "在当前终端运行服务，便于查看日志",
         "在当前终端启动 lab2shot ui，按 Ctrl-C 停止", "排查问题时直接查看日志", "与后台启动相同", PROJECT, lambda: _start(background=False)),
    Step("restart", "重启服务", "重启运行中的服务，可等待任务完成",
         "通过本机令牌重启运行中的服务，可选择等待正在计算的任务完成", "使需重启生效的设置生效", "无", PROJECT, _restart),
    Step("stop", "停止服务", "停止监听端口的服务进程",
         "停止监听端口的服务进程", "关机、更换端口或停用服务", "无；正在计算的任务将被中断", PROJECT, _stop),
    Step("gpus", "显卡授权", "选择接受计算任务的显卡",
         "选择接受计算任务的显卡", "未授权的显卡不参与计算", "运行中服务的授权名单（work/db）", PROJECT, _gpus),
    # 设置
    Step("settings", "设置", "按管理后台的分组查看与修改全部设置",
         "列出管理后台「设置」页的全部设置项，显示当前值、默认值与生效方式，并可逐项修改",
         "在不打开浏览器的情况下管理服务器设置", "config/local.toml（仅记录与默认值不同的项）；部分设置需重启服务后生效", PROJECT, _settings_menu),
    Step("server", "服务器设置", "监听范围、端口、HTTPS 与对外域名",
         "依次询问监听范围、端口、HTTPS 与对外域名（与「设置 → 网络」相同）", "局域网访问需监听 0.0.0.0；浏览器直接写入用户文件夹需要 HTTPS",
         "config/local.toml；启用 HTTPS 后首次启动时在 work/tls/ 生成自签名证书；需重启服务后生效", PROJECT,
         lambda: _edit_settings(("server.host", "server.port", "server.https", "server.names"), "服务器设置")),
    # 数据库
    Step("db-status", "状态", "数据库文件、大小、版本、完整性与全部备份（只读）",
         "显示数据库的文件、大小、版本、最近一次完整性检查与备份，并列出全部备份", "了解数据库的当前状况", "无", READ_ONLY, _db_status),
    Step("db-backup", "立即备份", "立即备份一份数据库（服务运行时亦可）",
         "以 SQLite 在线备份方式复制数据库，并校验副本", "在重要操作之前保留一份可恢复的数据",
         "work/db/backups/ 中新增一份备份及其附带文件；超出保留份数时删除最旧的备份", PROJECT, _db_backup),
    Step("db-check", "完整性检查", "检查数据库的完整性与外键",
         "执行 SQLite 完整性检查与外键检查", "确认数据库可以信任", "无（仅记录检查结果）", PROJECT, _db_check),
    Step("db-restore", "从备份恢复", "用一份备份替换当前数据库（需先停止服务）",
         "列出全部备份，以所选备份替换当前数据库", "数据库损坏、缺失或需要回退时恢复数据",
         "work/db/lab2shot.db；原数据库移至 work/db/lab2shot.db.replaced-<时间>，不会删除", PROJECT, _db_restore),
    Step("db-upgrade", "升级", "将数据库升级至本程序对应的版本（需先停止服务）",
         "先自动备份，再将数据库升级至本程序对应的版本", "更新 Lab2Shot 后使数据库与程序一致；启动服务时也会自动执行",
         "work/db/lab2shot.db；work/db/backups/ 中新增一份升级前的备份", PROJECT, _db_upgrade),
    Step("db-claim", "接管工作目录", "使工作目录归属本项目目录",
         "将工作目录的归属改为本项目目录", "此后其他项目目录（其他分支、测试检出）的进程无法打开该目录",
         "工作目录中的归属记录（owner.json）", PROJECT, _db_claim),
    # 状态总览
    Step("status", "状态总览", "服务、管理员、口令、账号、扩展包、数据库与工作目录（只读）",
         "显示服务、管理员密码、找回口令、账号数、扩展包数、数据库与工作目录的概况", "总览当前状况", "无", READ_ONLY, _status),
)
STEPS: dict[str, Step] = {s.name: s for s in STEP_LIST}
WIZARD = ("env", "web", "check", "start", "password", "gpus")


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    summary: str
    steps: tuple[str, ...] = ()  # a second-level menu of these steps; empty: `single` runs directly
    single: str = ""


MENU: tuple[Group, ...] = (
    Group("1", "首次安装向导", "按顺序完成 Python 环境、构建网页、检查环境、启动服务、设置管理员密码、显卡授权", single="wizard"),
    Group("2", "安装与环境", "Python 环境、构建网页、检查环境、内核配置、Hugging Face 登录、手动下载",
          ("env", "web", "check", "kernel", "hf-login", "downloads")),
    Group("3", "账号与安全", "管理员密码、找回口令", ("password", "passphrase")),
    Group("4", "服务", "启动、停止、重启服务，显卡授权", ("start", "start-fg", "restart", "stop", "gpus")),
    Group("5", "设置", "管理后台「设置」页的全部设置项，按分组查看与修改", single="settings"),
    Group("6", "数据库", "状态、备份、完整性检查、从备份恢复、升级、接管工作目录",
          ("db-status", "db-backup", "db-check", "db-restore", "db-upgrade", "db-claim")),
    Group("7", "状态总览", "服务、管理员、数据库与工作目录的概况（只读）", single="status"),
)


def _explain(step: Step) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("操作", step.what)
    grid.add_row("用途", step.why)
    grid.add_row("写入位置", step.changes)
    grid.add_row("影响范围", f"[green]{step.scope}[/green]" if step.scope == READ_ONLY else step.scope)
    group = next((g for g in MENU if step.name in g.steps), None)
    title = f"{group.label} → {step.label}" if group else step.label
    console.print(Panel(grid, title=f"[bold]{title}[/bold]", title_align="left", border_style="blue", box=box.ROUNDED))


def _group_menu(group: Group) -> None:
    """A second-level menu: the steps of one group; 0 returns to the top level."""
    while True:
        console.print()
        console.print(Rule(f"[bold]{group.key}  {group.label}[/bold]", align="left", style="cyan"))
        rows = [(str(i), STEPS[n].label, STEPS[n].summary) for i, n in enumerate(group.steps, 1)]
        console.print(_menu_table(rows + [("0", "返回", "返回上一级菜单")]))
        try:
            choice = _pick()
        except _abort_types():
            console.print()
            return
        if choice == "0":
            return
        try:
            step = STEPS[group.steps[int(choice) - 1]]
        except (ValueError, IndexError):
            _err("没有该编号，请重新输入。")
            continue
        console.print()
        _explain(step)
        _run(step)


HELP_STEPS = "、".join(s.name for s in STEP_LIST)


@app.command()
def setup(step: str = typer.Argument("", help=f"直接执行某一项而不进入菜单。可用的步骤名：{HELP_STEPS}")) -> None:
    """交互式配置菜单：按用途分组（首次安装向导、安装与环境、账号与安全、服务、设置、数据库、状态总览），每一项仅在选择后执行；执行之前先说明操作、用途、写入位置与影响范围。首次使用请运行 ./setup.sh。

    步骤名：wizard、env、web、check、kernel、hf-login、downloads、password、passphrase、start、start-fg、restart、stop、gpus、settings、server、db-status、db-backup、db-check、db-restore、db-upgrade、db-claim、status。"""
    if step:
        found = STEPS.get(step)
        if found is None:
            _err(f"没有名为 {escape(step)} 的步骤。可用的步骤名：{HELP_STEPS}。")
            raise typer.Exit(2)
        _explain(found)
        if not _run(found, menu=False):
            raise typer.Exit(1)
        return
    while True:
        console.print()
        _header()
        console.print(_menu_table([(g.key, g.label, g.summary) for g in MENU] + [("0", "退出", "退出配置菜单")]))
        try:
            choice = _pick()
        except _abort_types():  # Ctrl-C or end of input at the top level: leave quietly
            console.print()
            return
        if choice == "0":
            return
        group = next((g for g in MENU if g.key == choice), None)
        if group is None:
            _err("没有该编号，请重新输入。")
            continue
        if group.single:
            console.print()
            _explain(STEPS[group.single])
            _run(STEPS[group.single])
        else:
            _group_menu(group)


def _attempt(fn: Callable[[], None], menu: bool = True) -> bool:
    """Run one item, ending back at the menu whatever happens inside it: Ctrl-C in a sub-prompt is typer.Abort /
    click.Abort rather than KeyboardInterrupt; a command the menu calls (the restart item) ends with typer.Exit on
    failure; a work folder this checkout may not use, or a database that is refused, is a message, not a traceback.
    `menu` False (`lab2shot setup <step>`): Ctrl-C ends the command quietly and an Exit keeps its code.
    Returns whether the item finished."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    try:
        fn()
        return True
    except _abort_types():
        console.print()
        _note("已取消" + ("，返回上一级菜单。" if menu else "。"))
        if not menu:
            raise typer.Exit(130)
    except typer.Exit as exc:
        if not menu:
            raise
        if not exc.exit_code:
            return True
        _note("此项未能完成，返回上一级菜单。")
    except (WorkDirError, DatabaseError) as exc:
        _trouble(exc)
    return False


def _run(step: Step, menu: bool = True) -> bool:
    """One step from the menu (or from `lab2shot setup <step>` when `menu` is False); see _attempt."""
    return _attempt(step.run, menu)
