"""`lab2shot setup`: the single interactive place to install, configure and run a server, for whoever has just cloned
the repository and for the administrator afterwards.

The menu is grouped by purpose. The top level lists the wizards (first-time installation, the extension build and
download settings, the extension packages), the one-click update, and the groups (installation and environment,
service, accounts and security, settings, database, status overview); each group opens a second-level menu. Two
open a third level: the settings group, one per settings page of the admin page (config.PAGES), and 扩展包编译与下载设置 under
installation and environment, with its six items. Every item performs exactly one task and explains, before it
runs, what it does, why, what it writes and where.

Nothing here is a second implementation: passwords go through lab2shot.accounts, settings through
lab2shot.config.Settings.save (the admin page's own path, with the same validation), the running service through the
local client (machine token), the database through lab2shot.database (status, backup, check, upgrade, restore) and
lab2shot.workdir (claim), and the server itself through `lab2shot ui`. Two parts live in modules of their own: the
service (whether it runs, the environment it runs from, starting and stopping it: cli/service.py) and the one-click
update (`update`: cli/update.py), which uses the same service functions; the output conventions are cli/base.py's.

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
from rich.console import Group as Stack
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from .base import abort_types, app, console, err, mark, menu_table, note, ok, pick, say, warn, when
from .service import (NotStopped, address, build_webui, initialized, recorded_address, restart, running, start, stop,
                      sync_env)
from .update import one_click_update


# ------------------------------------------------------------------ what the menu inspects


def _trouble(exc: Exception) -> None:
    """A WorkDirError, DatabaseError or NotStopped raised by any item: the message, followed by the available remedies."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    err(escape(str(exc)))
    if isinstance(exc, WorkDirError):
        note("可选的处理方式：改用另一个工作目录（设置环境变量 LAB2SHOT_WORK_DIR 后重新进入菜单）；"
              "或停止使用该目录的服务后，在本菜单「数据库 → 将工作目录归属本项目」中将其归属本项目目录。")
    elif isinstance(exc, DatabaseError):
        note("启动服务时会先自动备份，再升级数据库；也可在「数据库 → 升级数据库版本」中手动执行。"
              "数据库损坏或缺失时，可在「数据库 → 从备份恢复数据库」中恢复。本菜单的其他项不会自动升级数据库。")


def _admin_state() -> str:
    """The administrator's password state for the header, read only when the database already exists."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    if not initialized():
        return "[dim]尚未初始化（首次启动服务时创建管理员账号）[/dim]"
    from .. import accounts

    try:
        owner = accounts.admin()
    except (WorkDirError, DatabaseError, OSError):
        return "[red]无法读取[/red]（详见「数据库 → 查看数据库状态」）"
    if owner.no_password:
        return f"{owner.username}，[yellow]尚未设置密码，目前无法登录[/yellow]"
    return f"{owner.username}，[green]已设置密码[/green]"


def _header() -> None:
    """The title panel above the top-level menu: version, service, administrator, work folder."""
    from .. import __version__
    from ..config import settings

    state = running()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    if state:
        grid.add_row("服务", f"[green]运行中[/green]  {recorded_address() or address()}（启动编号 {state.get('boot', '?')}）")
    else:
        grid.add_row("服务", f"[dim]未运行[/dim]  按当前设置的地址为 {address()}")
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
    rows.append(("网页构建", WEBUI_DIST.is_dir(), str(WEBUI_DIST) if WEBUI_DIST.is_dir() else "尚未构建。请执行「安装与环境 → 构建网页界面」。"))
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
        rows.append(("工作目录", False, f"{work} 归属另一个项目目录 {who}。请改用其他 LAB2SHOT_WORK_DIR，或停止该处的服务后执行「数据库 → 将工作目录归属本项目」。"))
    rows.append(("设置文件", None, f"{s.file}（{'已存在' if Path(s.file).is_file() else '不存在，全部设置取默认值；修改设置后自动生成'}）"))
    table = Table("项目", "", "说明", box=box.SIMPLE_HEAD, header_style="bold")
    for name, passed, text in rows:
        table.add_row(name, mark(passed), text)
    console.print(table)
    missing = [name for name, passed, _ in rows if passed is False]
    if missing:
        warn(f"以下项目需要处理：{'、'.join(missing)}。")
    else:
        ok("运行服务所需的条件均已具备。")


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
    for name, found, text in rows:
        table.add_row(name, mark(bool(found)), f"{found}  {text}" if found else text)
    console.print(table)
    from lab2shot_shared.gpu_arch import target_label

    here = [target_label("sm_" + c.replace(".", "")) for c in compute_caps()]
    console.print(f"本机显卡的架构：{'、'.join(here) if here else '（未检测到显卡）'}（仅供参考，编译不按它）")
    _check_targets(Path(cuda_home) / "bin" / "nvcc", list(s.value("build.archs")))
    from ..installer import toolchain

    state, why = toolchain.problem()
    text = f"nvcc 与编译器的兼容性：{why.text if why else '兼容'}"
    {"ok": ok, "warning": warn, "blocked": err}[state](text)


def _check_targets(nvcc: Path, archs: list[str]) -> None:
    """The compile targets (build.archs), each checked against what this nvcc can compile for (--list-gpu-arch)."""
    from lab2shot_shared.gpu_arch import target_label

    console.print(f"编译目标架构：{'、'.join(target_label(a) for a in archs)}（设置「编译目标架构」）")
    if not nvcc.is_file():
        return
    try:
        listed = subprocess.run([str(nvcc), "--list-gpu-arch"], capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return
    missing = [a for a in archs if a.replace("sm_", "compute_") not in listed]
    if missing:
        warn(f"这个 CUDA 工具包（{nvcc}）编不了：{'、'.join(target_label(a) for a in missing)}。"
              "请换一个更新的 CUDA 工具包（「设置编译器与 CUDA」），或在「选择编译目标架构」中去掉它们。")
    else:
        ok("CUDA 工具包支持所选的每一个编译目标架构。")


def _choose_archs() -> None:
    """The compile targets (setting build.archs), chosen by number and named by platform, never by graphics card."""
    from lab2shot_shared.gpu_arch import TARGET_LABELS, target_label

    from ..config import SCHEMA, settings

    current = list(settings().value("build.archs"))
    tokens = list(TARGET_LABELS)
    note("扩展包里的 CUDA 代码只为这里选中的架构编译，编出来的只能在这些架构上运行，与编译这台机器插的是什么卡无关。"
          f"默认：{'、'.join(target_label(a) for a in SCHEMA['build.archs'].default)}。")
    table = Table("编号", "架构", "", box=box.SIMPLE_HEAD, header_style="bold")
    for i, token in enumerate(tokens, 1):
        table.add_row(str(i), target_label(token), "[green]已选[/green]" if token in current else "")
    console.print(table)
    answer = typer.prompt("要编译的架构编号（以空格分隔；直接回车保留当前）", default="", show_default=False).strip()
    if not answer:
        note(f"保留当前：{'、'.join(target_label(a) for a in current)}。")
        return
    numbers = answer.replace(",", " ").replace("，", " ").split()
    if not all(n.isdigit() and 1 <= int(n) <= len(tokens) for n in numbers):
        err("编号无效，未修改。")
        return
    picked = {tokens[int(n) - 1] for n in numbers}
    _save_settings({"build.archs": [t for t in tokens if t in picked]})  # kept in the table's order, oldest first


def _install_formats() -> None:
    """The format modules (Alembic, FBX): every card that reads or writes DCC scene files depends on them. A hand
    download one needs that waits for its licence (the FBX SDK) is shown and accepted here first (_consent_pending)."""
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
        _consent_pending(ext.name)
        if not _checklist(ext):
            warn("上方存在未满足的条件，请先解决后再安装。")
            continue
        _run_install(ext, stop=False)


def _install_extensions() -> None:
    """Extension packages from the command line: every one with its state; the chosen ones installed in turn, each
    after the licences of its hand downloads (_consent_pending) and the installer's checklist, the same install as
    `lab2shot ext install` (the admin page's 安装 runs it as a farm task). The format modules are 扩展包编译与下载设置's."""
    from ..extensions import extensions
    from ..extensions.status import extension_status

    from .extensions import _checklist, _run_install

    exts = [e for e in extensions().values() if not e.format_module]
    ready = {}
    table = Table("编号", "扩展包", "状态", "许可证", box=box.SIMPLE_HEAD, header_style="bold")
    for i, ext in enumerate(exts, 1):
        st = extension_status(ext)
        ready[ext.name] = st["ready"]
        color = "green" if st["ready"] else "dim" if st["label"] == "未安装" else "yellow"
        table.add_row(str(i), f"{escape(ext.title)}（{ext.name}）", f"[{color}]{escape(st['label'])}[/{color}]", escape(ext.license.name))
    console.print(table)
    answer = typer.prompt("要安装的编号（以空格分隔；a 表示全部未就绪的扩展包；直接回车跳过）", default="", show_default=False).strip()
    if not answer:
        note("未选择扩展包。")
        return
    if answer.lower() == "a":
        chosen = [e for e in exts if not ready[e.name]]
    else:
        numbers = answer.replace(",", " ").replace("，", " ").split()
        if not all(n.isdigit() and 1 <= int(n) <= len(exts) for n in numbers):
            err("编号无效，未安装任何扩展包。")
            return
        chosen = list(dict.fromkeys(exts[int(n) - 1] for n in numbers))
    if not chosen:
        note("全部扩展包均已就绪。")
        return
    results: list[tuple[str, str]] = []
    for i, ext in enumerate(chosen, 1):
        console.print(Rule(f"[bold]{i}/{len(chosen)}  {escape(ext.title)}（{ext.name}）[/bold]", align="left", style="cyan"))
        console.print(f"许可证：{escape(ext.license.name)} — {escape(ext.license.summary)}", highlight=False)
        _consent_pending(ext.name)
        if not _checklist(ext):
            warn("上方存在未满足的条件，请先解决后再安装。")
            results.append((ext.title, "[yellow]条件未满足[/yellow]"))
            continue
        why = _run_install(ext, stop=False)
        results.append((ext.title, "[green]已安装[/green]" if not why else f"[red]失败[/red]  {escape(why)}"))
    summary = Table("扩展包", "结果", box=box.SIMPLE_HEAD, header_style="bold")
    for title, result in results:
        summary.add_row(escape(title), result)
    console.print(Panel(summary, title="[bold]安装结果[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))


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
        ok("已登录：令牌来自环境变量 HF_TOKEN。")
        return
    if get_token():
        ok("已登录：本机已保存 Hugging Face 令牌。")
        if not typer.confirm("是否重新登录？", default=False):
            return
    elif not typer.confirm("本机尚未登录 Hugging Face。是否现在登录？", default=True):
        return
    note("令牌在 https://huggingface.co/settings/tokens 创建，权限选择 Read 即可。")
    hf = Path(sys.executable).parent / "hf"
    if not hf.is_file():
        err(f"未找到 {hf}。请先执行「安装与环境 → 安装 Python 依赖」。")
        return
    if subprocess.run([str(hf), "auth", "login"]).returncode != 0:
        err("登录未完成，请查看上方输出。")
        return
    ok("登录完成。")


MANUAL_STATES = {"ready": "[green]已就绪[/green]", "consent": "[yellow]待同意许可[/yellow]",
                 "unrecognised": "[red]文件无法识别[/red]", "missing": "未下载"}


def _manual_downloads() -> None:
    """The files people download by hand: where, which one, what for, and whether each is in place. The files already
    in downloads/ are recognised and installed first; the ones whose licence must be accepted are shown and, on the
    user's yes, installed (_consent_pending)."""
    from ..extensions import manual

    manual.check()
    _consent_pending()
    view = manual.view()
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
        warn(escape(f"downloads/ 中的 {f['name']} 无法识别：{f['why']}"))
    console.print(f"下载的文件原样放入 {escape(view['inbox']['path'])}（无需解压或改名），再执行本项即可识别并安装；"
                  "原始文件随后移入 downloads/installed/。标为「待同意许可」的项目，再次执行本项时显示其许可协议，同意后即安装；"
                  "也可在管理后台「扩展包 → 手动下载」中阅读并同意。完整说明见 docs/manual-downloads.md。")


def _cli_who() -> dict:
    """Who accepts a licence at this machine's command line, recorded like the page's click (farm Client.full): the
    administrator account, with what a DCC client says of itself (lab2shot/client.py: app, hostname, user, platform)."""
    import getpass
    import platform
    import socket

    from .. import accounts
    from ..farm.clients import Client

    return Client.of(accounts.admin(), {"app": "cli", "hostname": socket.gethostname(), "user": getpass.getuser(),
                                        "platform": platform.platform()}).full()


def _consent_pending(ext_name: str = "") -> None:
    """The files in downloads/ that wait for their licence to be accepted (all of them, or those `ext_name` needs): the
    licence taken from the file is shown in full and, on the user's yes, accepted through manual.accept, the page's
    同意并安装 path (the acceptance is recorded in the database: who, when, which version and licence)."""
    from ..errors import MessageError
    from ..extensions import manual

    rows = [r for r in manual.view()["items"] if any(f["state"] == "consent" for f in r["files"])
            and (not ext_name or any(e["name"] == ext_name for e in r["needed_by"]))]
    if not rows:
        return
    if not initialized():
        warn("同意许可协议的记录写入数据库，须先启动过一次服务（「服务 → 启动服务（后台）」）后再执行本项。")
        return
    for row in rows:
        for f in (f for f in row["files"] if f["state"] == "consent"):
            title = row["title"]
            try:
                note(f"正在从 {f['name']} 中取出 {title} 的许可协议……")
                shown = manual.licence(f["name"])
                console.print(Panel(escape(shown["text"]), title=f"[bold]{escape(title)} 许可协议[/bold]（{escape(f['name'])}）",
                                    title_align="left", border_style="yellow", box=box.ROUNDED))
                if not typer.confirm(f"是否已阅读并同意 {title} 的许可协议，并安装？", default=False):
                    note(f"未同意，{title} 未安装；文件留在 downloads/ 中，再次执行本项时重新询问。")
                    continue
                manual.accept(f["name"], shown["sha256"], _cli_who())
            except MessageError as exc:
                err(escape(str(exc)))
                continue
            ok(f"已同意 {title} 的许可协议并完成安装（同意记录已写入数据库）。")


# 扩展包编译与下载设置的六项（steps in STEP_LIST: `lab2shot setup build` etc. runs one directly); also its wizard's order:
# the compile targets first, then the compilers, then the check of both
BUILD_SETUP = ("archs", "build", "toolchain", "mirrors", "retries", "formats")


def _build_setup() -> None:
    """The third-level menu of 扩展包编译与下载设置: what compiling extensions and building their environments depend on."""
    while True:
        console.print(Rule("[bold]扩展包编译与下载设置[/bold]", align="left", style="cyan"))
        rows = [(str(i), STEPS[n].label, STEPS[n].summary) for i, n in enumerate(BUILD_SETUP, 1)]
        console.print(menu_table(rows + [("0", "返回", "返回上一级菜单")]))
        try:
            choice = pick()
        except abort_types():
            console.print()
            return
        if choice == "0":
            return
        try:
            step = STEPS[BUILD_SETUP[int(choice) - 1]]
        except (ValueError, IndexError):
            err("没有该编号，请重新输入。")
            continue
        _attempt(step.run)


# ------------------------------------------------------------------ accounts and security


def _not_initialized() -> None:
    warn("工作目录尚未初始化。账号与数据库由首次启动的服务创建，请先执行「服务 → 启动服务（后台）」，再返回此处设置。")


def _admin_password() -> None:
    if not initialized():  # this item never creates work/ or the database: the first start of the service does
        _not_initialized()
        return
    from .. import accounts

    from .accounts import admin_password

    owner = accounts.admin()
    if owner.no_password:
        warn(f"管理员 {owner.username} 尚未设置密码，在设置之前任何人都无法登录。请现在设置。")
    else:
        console.print(f"管理员 {owner.username} 的密码于 {when(owner.password_set or 0)} 经由{owner.password_by}设置。")
        if not typer.confirm("是否重新设置？", default=False):
            return
    admin_password()


def _passphrase() -> None:
    if not initialized():  # this item never creates work/ or the database: the first start of the service does
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
    """Ask for one setting in the form its kind requires: a numbered choice (or several), yes or no, a number, a list, or text."""
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
            warn("编号无效，保留原值。")
            return current
    if s.kind == "multi":
        options = list(s.offered())
        for i, (value, label) in enumerate(options, 1):
            console.print(f"  [bold cyan]{i}[/bold cyan]  {label}（{value}）" + ("  [dim]← 已选[/dim]" if value in current else ""))
        text = typer.prompt(f"{s.label}（输入要选的编号，用空格隔开；直接回车保留，输入 - 全不选）", default="", show_default=False).strip()
        if not text:
            return current
        if text == "-":
            return []
        numbers = text.replace("，", " ").replace(",", " ").split()
        if not all(n.isdigit() and 1 <= int(n) <= len(options) for n in numbers):
            warn("编号无效，保留原值。")
            return current
        return [options[int(n) - 1][0] for n in numbers]
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
        return _ask_list(key, list(current))
    # The current value is said in words, apart from what empty means (a bracketed default right after 「留空表示只用官方」
    # would read as if the mirror were the official address). Enter keeps the value, so "-" is what makes it empty
    shown = str(current or "")
    hint = f"当前：{shown}；直接回车保留" if shown else "当前：空"
    if s.empty:
        hint += f"；输入 - 清空，空表示{s.empty}" if shown else f"，表示{s.empty}"
    text = typer.prompt(f"{s.label}（{hint}）", default="", show_default=False).strip()
    if not text:
        return shown
    return "" if text == "-" and s.empty else text


def _ask_list(key: str, items: list[str]) -> list[str]:
    """A list setting (kind "list": 环节, 编译目标架构) edited item by item, as on the admin page: the items numbered in
    order; add one (at the end, or before a number), remove one, or move one to another place, until Enter. What an
    item may hold is checked when the list is saved (Settings.save, the admin page's path and validation)."""
    from ..config import SCHEMA

    label = SCHEMA[key].label
    while True:
        table = Table("编号", label, box=box.SIMPLE_HEAD, header_style="bold")
        for i, item in enumerate(items, 1):
            table.add_row(str(i), escape(item))
        console.print(table)
        note("+ 名字：添加到最后；+ 名字 编号：插到该编号之前；- 编号：删除；编号 新位置：移到新位置；直接回车：完成。")
        text = typer.prompt(label, default="", show_default=False).strip()
        if not text:
            return items
        parts = text.split()
        n = len(items)

        def number(x: str, most: int) -> bool:
            return x.isdigit() and 1 <= int(x) <= most

        if parts[0] == "+" and len(parts) in (2, 3) and (len(parts) == 2 or number(parts[2], n + 1)):
            items.insert(int(parts[2]) - 1 if len(parts) == 3 else n, parts[1])
        elif parts[0] == "-" and len(parts) == 2 and number(parts[1], n):
            items.pop(int(parts[1]) - 1)
        elif len(parts) == 2 and number(parts[0], n) and number(parts[1], n):
            items.insert(int(parts[1]) - 1, items.pop(int(parts[0]) - 1))
        else:
            err("看不懂这一句，请按上面的写法输入（名字里不能有空格）。")


def _provide_options() -> None:
    """Import the layers that provide settings' options (config.provide_choices): the menu shows and saves them."""
    from ..nodes import tags  # noqa: F401  (注册可用模型类别)
    from ..view import proxy  # noqa: F401  (视图代理尺寸)


def _save_settings(changes: dict[str, object]) -> list[str] | None:
    """Save through Settings.save (the admin page's path and validation). Returns the keys that changed, or None when
    the values were refused (the reasons are printed)."""
    _provide_options()
    from ..config import InvalidSettings, SCHEMA, settings

    s = settings()
    try:
        changed = s.save(changes)
    except InvalidSettings as exc:
        for key, problem in exc.problems.items():
            err(escape(problem.text))
        return None
    if not changed:
        note("没有改动。")
        return []
    ok(f"已保存：{'、'.join(SCHEMA[k].label for k in changed)}（写入 {s.file}）。")
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
    if not running():
        note("服务当前未运行；下次启动服务时即采用新的设置。")
        return
    if typer.confirm("服务正在运行。是否现在重启服务？", default=False):
        _attempt(restart)
    else:
        note("稍后可在「服务 → 重启服务」中重启。")


def _edit_settings(keys: tuple[str, ...], title: str) -> None:
    """Ask a fixed list of settings in turn, then save them together."""
    _provide_options()
    from ..config import SCHEMA, settings

    s = settings()
    console.print(f"[bold]{title}[/bold]  （直接按回车保留当前值；写入 {s.file.name}，仅记录与默认值不同的项）")
    changes = {}
    for key in keys:
        spec = SCHEMA[key]
        if spec.only_if and not changes.get(spec.only_if, s.file_value(spec.only_if)):
            continue
        if spec.help:
            note(spec.help)
        changes[key] = _ask_setting(key, s.file_value(key))
    changed = _save_settings(changes)
    if changed:
        _offer_restart(changed)


# ------------------------------------------------------------------ network: how the server is reached


@dataclass(frozen=True)
class Access:
    """One way the server is reached, and the value it writes for every network setting but the port (asked last in
    each). A value of None is asked after the choice. Every network setting is listed, so a choice leaves nothing from
    an earlier one behind."""
    key: str
    label: str
    purpose: str
    values: dict[str, object]
    shown: dict[str, str]  # how the choice table states a value of None: what is asked, and its default


# ③ has two cases, answered by 「代理在哪里」: on this machine (or its connections arrive through a tunnel on this
# machine, as frp does), Lab2Shot listens on loopback only and trusts 127.0.0.1; on another machine, it has to listen on
# every interface for the proxy to reach it, and trusts that machine's address only
PROXY_LOCAL = {"server.host": "127.0.0.1", "server.trusted_proxies": "127.0.0.1"}
PROXY_REMOTE = {"server.host": "0.0.0.0"}  # server.trusted_proxies: the proxy machine's address, asked, required
ACCESS = (
    Access("1", "只在这台电脑上自己用", "只有这台电脑上的浏览器、DCC 插件和命令行能连；不加密（本机之内不经过网络）",
           {"server.host": "127.0.0.1", "server.https": False, "server.trusted_proxies": "", "server.names": ""}, {}),
    Access("2", "局域网里的人直接访问这台电脑", "Lab2Shot 自己用自签证书提供 HTTPS，监听所有网卡；每台用户电脑装一次根证书",
           {"server.host": "0.0.0.0", "server.https": True, "server.trusted_proxies": "", "server.names": None},
           {"server.names": "（询问，可留空：只在别人用另外的名字访问时填）"}),
    Access("3", "前面有 Nginx 等反向代理，用正规证书对外", "反向代理用正规证书提供 HTTPS 并转发到这里；Lab2Shot 只认代理转告的用户地址。"
           "再问代理在哪里：(a) 在这台电脑上，或经本机的 frp 等隧道转来（默认）；(b) 在另一台机器上",
           {"server.host": None, "server.https": False, "server.trusted_proxies": None, "server.names": ""},
           {"server.host": '= (a) "127.0.0.1"  (b) "0.0.0.0"',
            "server.trusted_proxies": '= (a) "127.0.0.1"  (b) 代理那台机器的 IP（询问，必填）'}),
)
NETWORK_KEYS = ("server.host", "server.https", "server.trusted_proxies", "server.names", "server.port")


def _toml(value: object) -> str:
    """A value as config/local.toml writes it: what a technical reader compares with the file."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value) if isinstance(value, int) else f'"{value}"'


def _access_now(s) -> str:
    """Which way the current settings amount to, as the default choice: a trusted proxy is 3, listening on every
    interface without one is 2, otherwise 1."""
    if s.file_value("server.trusted_proxies"):
        return "3"
    return "2" if s.file_value("server.host") == "0.0.0.0" else "1"


def _access_rows(s) -> list[tuple[str, str, str]]:
    """The three ways as menu rows: the purpose, then every setting each writes (key = value)."""
    rows = []
    for a in ACCESS:
        lines = [f"{k} {a.shown[k]}" if a.values[k] is None else f"{k} = {_toml(a.values[k])}" for k in NETWORK_KEYS[:-1]]
        lines.append(f"server.port （询问，默认 {s.file_value('server.port')}）")
        rows.append((a.key, a.label, escape("\n".join([a.purpose, *lines]))))
    return rows


def _proxy_remote(current: str) -> str:
    """The address of the proxy machine (③ b): required, and of the form the admin page accepts (Setting.check);
    asked until it is both."""
    from ..config import SCHEMA

    spec = SCHEMA["server.trusted_proxies"]
    while True:
        text = typer.prompt("代理那台机器的 IP（地址或网段，用空格隔开；必填）", default=current or None,
                            show_default=bool(current)).strip()
        if not text:
            err("必须填写代理那台机器的 IP。")
            continue
        if problem := spec.check(text):
            err(escape(problem.text))
            continue
        return text


def _network() -> None:
    """The network step (首次安装向导, and 服务 → 设置网络与端口): ask how the server is reached, write the settings that
    way needs through Settings.save (the admin page's path and validation), then show what was written and what the
    other side (a user's computer, the proxy) has to do."""
    _provide_options()
    from ..config import SCHEMA, settings
    from .service import ca_url, lan_address

    s = settings()
    console.print(f"[bold]服务器怎样被访问？[/bold]  （按所选写入 {s.file.name}；以后可在「设置 → 网络与安装」中逐项修改）")
    console.print(menu_table(_access_rows(s) + [("0", "返回", "不作修改")]))
    console.print(Panel(
        "两者都能让别的电脑用 HTTPS 打开 Lab2Shot，区别在于由谁面对用户：\n"
        "  ② Lab2Shot 自己面对用户：它用自签证书提供 HTTPS，监听所有网卡；每台用户电脑装一次它的根证书；"
        "不信任任何转发头（X-Forwarded-For 等），按连接本身的地址认用户。\n"
        "  ③ Lab2Shot 在代理后面：代理用正规证书提供 HTTPS，用户什么都不用装；Lab2Shot 自己不加密，"
        "只从可信代理的地址认它转告的用户真实 IP（按 IP 的注册限制、密码输错计数、登录记录都靠它）。\n"
        "     (a) 代理在这台电脑上，或经本机的 frp 等隧道转来：只监听本机，别人绕不过代理直接连它；可信代理 127.0.0.1。\n"
        "     (b) 代理在另一台机器上：须监听所有网卡代理才连得到；可信代理只填那台机器的 IP，"
        "并应在防火墙上只让那台机器连这个端口。",
        title="[bold]② 与 ③ 的区别[/bold]", title_align="left", border_style="dim", box=box.ROUNDED))
    while True:
        choice = pick(_access_now(s))
        if choice == "0":
            note("未作修改。")
            return
        access = next((a for a in ACCESS if a.key == choice), None)
        if access is None:
            err("没有该编号，请重新输入。")
            continue
        changes = {k: v for k, v in access.values.items() if v is not None}
        remote = False
        if access.key == "2":
            note("对外域名只写进 Lab2Shot 自签证书里的名字：别人用这台电脑的名字或局域网 IP 访问时直接回车；"
                 "用另外的名字（例如内网穿透给的域名）访问时填上，用空格隔开。")
            changes["server.names"] = _ask_setting("server.names", s.file_value("server.names"))
        if access.key == "3":
            was_remote = s.file_value("server.host") == "0.0.0.0" and bool(s.file_value("server.trusted_proxies"))
            console.print("  [bold cyan]1[/bold cyan]  (a) 在这台电脑上，或经本机的 frp 等隧道转来\n"
                          "  [bold cyan]2[/bold cyan]  (b) 在另一台机器上")
            where = typer.prompt("代理在哪里（请输入编号）", default="2" if was_remote else "1").strip()
            if where not in ("1", "2"):
                err("没有该编号，请重新选择。")
                continue
            remote = where == "2"
            if remote:
                changes |= PROXY_REMOTE
                changes["server.trusted_proxies"] = _proxy_remote(s.file_value("server.trusted_proxies") if was_remote else "")
            else:
                changes |= PROXY_LOCAL
        changes["server.port"] = _ask_setting("server.port", s.file_value("server.port"))
        changed = _save_settings(changes)
        if changed is not None:
            break
        note("以上设置未保存，请重新选择。")
    table = Table("设置键", "名称", "值", "", box=box.SIMPLE_HEAD, header_style="bold")
    for k in NETWORK_KEYS:
        table.add_row(k, SCHEMA[k].label, escape(_toml(s.file_value(k))), "[cyan]已改[/cyan]" if k in changed else "")
    port = s.file_value("server.port")
    way = f"访问方式：{access.key} {access.label}"
    if access.key == "3":
        way += "；代理在另一台机器上" if remote else "；代理在这台电脑上，或经本机的隧道转来"
    hints = [way]
    if access.key == "2":
        hints.append(f"每台用户电脑需安装一次根证书：在浏览器中打开 {ca_url(port)} 下载，导入系统的「受信任的根证书颁发机构」。")
    if access.key == "3":
        target = f"{lan_address()}:{port}" if remote else f"127.0.0.1:{port}"
        hints.append("反向代理须把用户的地址写进 X-Forwarded-For，是改写而不是追加"
                     "（Nginx：proxy_set_header X-Forwarded-For $remote_addr;）；"
                     "还须原样转发用户访问的地址，包括端口，否则登录等操作会被当作别的网站发来的而拒绝"
                     "（Nginx：proxy_set_header Host $http_host;，不能用去掉了端口的 $host），"
                     f"并转发到 http://{target}。")
    if remote:
        hints.append(f"Lab2Shot 现在监听所有网卡：请在防火墙上只允许代理那台机器（{s.file_value('server.trusted_proxies')}）"
                     f"连接本机的 {port} 端口，别的机器不应能绕过代理直接连它。")
    hints.append("以上各项以后都可在「设置 → 网络与安装」中修改。")
    console.print(Panel(Stack(table, *(escape(h) for h in hints)), title=f"[bold]已写入 {s.file}[/bold]",
                        title_align="left", border_style="green", box=box.ROUNDED))
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
            warn(f"该项只能在配置文件 {s.file} 中修改，本菜单与管理后台均只显示不修改。")
            return
        rows = [("1", "修改", "输入新的值并保存")]
        if key in s.saved:
            rows.append(("2", "恢复默认值", f"恢复为 {escape(spec.says(spec.default_now))}"))
        rows.append(("0", "返回", "返回上一级菜单"))
        console.print(menu_table(rows))
        choice = pick()
        if choice == "0":
            return
        if choice == "1":
            changed = _save_settings({key: _ask_setting(key, s.file_value(key))})
        elif choice == "2" and key in s.saved:
            changed = _save_settings({key: spec.default_now})
        else:
            err("没有该编号，请重新输入。")
            continue
        if changed:
            pending.extend(k for k in changed if SCHEMA[k].restart)
        return


def _settings_page(page: str, pending: list[str]) -> None:
    """The third level: every setting of one settings page (config.PAGES, as the admin page lays it out), under its
    groups' headings, with its value, default and how it takes effect. 网络与安装 also offers the network step
    (_network): the access way, which sets its network settings together."""
    from ..config import PAGES, SCHEMA, settings

    groups = PAGES[page].groups
    keys = [k for g in groups for k, spec in SCHEMA.items() if spec.group == g]
    wizard = str(len(keys) + 1) if page == "network" else ""
    while True:
        s = settings()
        console.print(Rule(f"[bold]设置 → {PAGES[page].label}[/bold]", align="left", style="cyan"))
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column("编号", justify="right", style="bold cyan", no_wrap=True)
        table.add_column("名称", no_wrap=True)
        table.add_column("当前值")
        table.add_column("默认值")
        table.add_column("生效", no_wrap=True)
        table.add_column("来源", no_wrap=True)
        for i, k in enumerate(keys, 1):
            spec = SCHEMA[k]
            if len(groups) > 1 and (i == 1 or SCHEMA[keys[i - 2]].group != spec.group):
                table.add_row("", f"[bold]{groups[spec.group]}[/bold]", "", "", "", "")
            name = spec.label
            if spec.only_if and not s.file_value(spec.only_if):
                name = f"[dim]{name}（需开启「{SCHEMA[spec.only_if].label}」）[/dim]"
            table.add_row(str(i), name, escape(spec.says(s.value(k))), escape(spec.says(spec.default_now)),
                          "[yellow]需重启[/yellow]" if spec.restart else "立即", _source_of(k))
        if wizard:
            table.add_row(wizard, "按访问方式设置网络", "本机自用、局域网直连或反向代理之后：一次设好访问范围、HTTPS、可信代理与对外域名，再设端口",
                          "", "", "")
        table.add_row("0", "返回", "", "", "", "")
        console.print(table)
        try:
            choice = pick()
        except abort_types():
            console.print()
            return
        if choice == "0":
            return
        if choice != wizard:
            try:
                key = keys[int(choice) - 1]
            except (ValueError, IndexError):
                err("没有该编号，请重新输入。")
                continue
        try:
            if choice == wizard:
                _network()
            else:
                _setting_detail(key, pending)
        except abort_types():
            console.print()
            note("已取消，未作修改。")


def _settings_menu() -> None:
    """The second level of 设置: the settings pages, the same as the admin page's 设置 band (config.PAGES). Leaving it
    lists the changes that await a restart."""
    _provide_options()
    from ..config import PAGES, SCHEMA, settings

    pending: list[str] = []
    ids = list(PAGES)
    while True:
        s = settings()
        console.print(Rule("[bold]设置[/bold]", align="left", style="cyan"))
        note(f"与管理后台「设置」一组的各页为同一份设置，写入 {s.file}，仅记录与默认值不同的项。")
        rows = []
        for i, page in enumerate(ids, 1):
            specs = [spec for spec in SCHEMA.values() if spec.group in PAGES[page].groups]
            edited = sum(1 for spec in specs if spec.key in s.saved)
            needs_restart = sum(1 for spec in specs if spec.restart)
            rows.append((str(i), PAGES[page].label, f"共 {len(specs)} 项；已修改 {edited} 项；需重启生效 {needs_restart} 项"))
        rows.append(("0", "返回", "返回上一级菜单" + ("，并列出需重启生效的改动" if pending else "")))
        console.print(menu_table(rows))
        file_only = [spec for spec in SCHEMA.values() if not spec.admin]
        console.print(Panel(
            "以下设置请在管理后台或配置文件中修改：\n"
            + "".join(f"  · {spec.label}（{spec.key}）：只能在配置文件 {s.file.name} 中修改，或以环境变量 LAB2SHOT_WORK_DIR 指定。\n" for spec in file_only)
            + "  · 「注册设置」页的邀请码、「账号设置」页的管理员通知，以及每个账号的环节与配额：请在管理后台（/admin）中管理。\n"
            + "  · 接受任务的显卡：可在本菜单「服务 → 选择计算用显卡」或管理后台中设置。",
            title="[bold]说明[/bold]", title_align="left", border_style="dim", box=box.ROUNDED))
        try:
            choice = pick()
        except abort_types():
            console.print()
            break
        if choice == "0":
            break
        try:
            page = ids[int(choice) - 1]
        except (ValueError, IndexError):
            err("没有该编号，请重新输入。")
            continue
        _settings_page(page, pending)
    _offer_restart(pending)


# ------------------------------------------------------------------ service


def _gpus() -> None:
    from ..client import Lab2ShotError

    from .accounts import local_client

    if not running():
        warn("服务未运行。显卡由运行中的服务识别，请先启动服务再进行授权。")
        return
    try:
        cards = local_client().admin_cards()["cards"]
    except Lab2ShotError as exc:
        err(escape(str(exc)))
        return
    if not cards:
        warn("服务未识别到任何显卡。")
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
            err("编号无效，未作修改。")
            return
    try:
        local_client().admin_authorize(uuids)
    except Lab2ShotError as exc:
        err(escape(str(exc)))
        return
    ok(f"已授权 {len(uuids)} 张显卡接受任务。")


# ------------------------------------------------------------------ database

REASONS = {"manual": "手动备份", "daily": "每日自动备份", "created": "创建数据库时", "before-update": "一键更新前备份"}


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
    if not initialized():  # read-only: never create work/ or the database for a look
        if found:
            warn(f"数据库文件 {folder() / FILE} 不存在，但备份目录中有 {len(found)} 份备份。可在「数据库 → 从备份恢复数据库」中恢复。")
            console.print(_backups_table(found))
        else:
            note(f"数据库尚未创建：首次启动服务时在 {settings().work_dir / 'db'} 中创建。")
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
    grid.add_row("保留份数", f"{s['keep']} 份（设置项「数据库备份份数」；超出时删除最旧的备份）")
    console.print(Panel(grid, title="[bold]数据库[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    if found:
        console.print(_backups_table(found))
    else:
        note("备份目录中没有备份。")


def _db_backup() -> None:
    from ..database import db

    if not initialized():
        warn("数据库尚未创建，无需备份。首次启动服务时会创建数据库并自动备份一次。")
        return
    target = db().backup("manual")
    ok(f"已备份：{target}（{_size(target.stat().st_size)}）。")


def _db_check() -> None:
    from ..database import db

    if not initialized():
        warn("数据库尚未创建，无可检查的内容。")
        return
    found = db().check()
    if found["ok"]:
        ok("数据库完好：完整性检查与外键检查均已通过。")
    else:
        err(f"数据库存在问题：{escape(found['detail'])}。可在「数据库 → 从备份恢复数据库」中恢复。")


def _ensure_stopped(action: str) -> bool:
    """For an action that requires the database to itself: when the service is running, say so and offer to stop it
    (the menu's own 停止服务, cli/service.py stop). Returns whether the service is (now) stopped."""
    if not running():
        return True
    warn(f"服务正在运行。{action}前必须停止服务。")
    try:
        stop(cancel=("返回", f"不停止服务，不{action}"))
    except NotStopped as exc:
        say(exc.message)
        return False
    except abort_types():
        console.print()
        note("已取消，未作任何操作。")
        return False
    if running():
        err("服务仍在运行，已取消操作。")
        return False
    return True


def _offer_start() -> None:
    warn("请重新启动服务，以使用新的数据库。")
    if typer.confirm("是否现在启动服务（后台）？", default=False):
        try:
            start(background=True)
        except typer.Exit:
            pass  # the database is restored all the same; start said why the service did not start
    else:
        note("稍后可在「服务 → 启动服务（后台）」中启动。")


def _db_restore() -> None:
    from ..database import FILE, close_all, folder, restore

    found = _backups()
    if not found:
        warn("备份目录中没有备份，无法恢复。")
        return
    console.print(_backups_table(found))
    choice = typer.prompt("请输入要恢复的备份编号（0 表示返回）", default="0").strip()
    if choice == "0":
        return
    try:
        source = found[int(choice) - 1]
    except (ValueError, IndexError):
        err("编号无效，未作任何操作。")
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
        note("已取消，未作任何操作。")
        return
    again = typer.prompt("为防止误操作，请再次输入所选备份的编号", default="").strip()
    if again != choice:
        note("两次输入的编号不一致，已取消，未作任何操作。")
        return
    close_all()  # this menu's own handle on the database would otherwise hold it (E-DB-INUSE)
    replaced = restore(source.name)
    ok(f"已从 {source.name} 恢复数据库。")
    ok(f"原数据库已移至 {replaced}。")
    _offer_start()


def _db_upgrade() -> None:
    from ..database import close_all, db, schema

    if not initialized():
        warn("数据库尚未创建，无需升级。首次启动服务时会创建最新版本的数据库。")
        return
    if not _ensure_stopped("升级数据库"):
        return
    close_all()  # the upgrade needs the database to itself, including this menu's own handle
    d = db(upgrade=True)
    ok(f"数据库为第 {d.version} 版（本程序对应第 {schema.VERSION} 版）：{d.path}。升级前的数据库已自动备份。")


def _db_claim() -> None:
    from ..config import ROOT, settings
    from ..workdir import claim, owner

    work = settings().work_dir
    if not work.is_dir():
        note(f"工作目录 {work} 尚不存在，无需接管：首次启动服务时自动归属本项目目录。")
        return
    was = owner(work)
    if was == str(ROOT):
        ok(f"工作目录 {work} 已归属本项目目录 {ROOT}，无需接管。")
        return
    warn(f"工作目录 {work} 目前归属 {was or '（未记录）'}。接管后，其他项目目录（其他分支、测试检出）的进程均无法再打开它。"
          "请先停止原来使用该目录的服务。")
    if not typer.confirm(f"是否将其改为归属 {ROOT}？", default=False):
        note("已取消，未作任何操作。")
        return
    claim(work)
    ok(f"工作目录 {work} 现归属 {ROOT}。")


# ------------------------------------------------------------------ status overview


def _status() -> None:
    from .. import accounts
    from ..config import settings
    from ..extensions import extensions

    s = settings()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    if not initialized():  # read-only: never create work/, the database or the machine token for a look
        grid.add_row("服务", f"[dim]未运行[/dim]（按当前设置的地址为 {address()}）")
        grid.add_row("工作目录", f"{s.work_dir}，尚未初始化：首次启动服务时创建数据库、管理员账号与本机令牌")
        grid.add_row("管理员", f"{accounts.ADMIN_NAME}，没有默认密码：服务首次启动后请在「账号与安全 → 设置管理员密码」中设置，设置之前无法登录")
        grid.add_row("设置文件", str(s.file))
        console.print(Panel(grid, title="[bold]状态总览[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
        return
    state = running()
    if state:
        grid.add_row("服务", f"[green]运行中[/green]  {recorded_address() or address()}，启动编号 {state.get('boot', '?')}")
    else:
        grid.add_row("服务", f"[dim]未运行[/dim]（按当前设置的地址为 {address()}）")
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
# Node.js, the display driver) are only checked here, never installed. Apart from uv's package cache (~/.cache/uv), two
# steps write to the user's home folder: the installation of uv in setup.sh (to ~/.local/bin) and hf-login (the Hugging
# Face token).
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
    run: Callable[[], object]


def _wizard(title: str, purpose: str, names: tuple[str, ...]) -> None:
    """A wizard (首次安装, 扩展包编译与下载设置, 扩展包安装): its steps in order, each explained and confirmed, each skippable."""
    console.print(Panel(
        f"本向导按顺序引导完成{purpose}的 {len(names)} 个步骤：" + " → ".join(STEPS[n].label for n in names) + "。\n"
        "每一步开始前均说明其内容并询问是否执行；选择否即跳过该步骤。按 Ctrl-C 可随时退出向导。",
        title=f"[bold]{title}[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    results: list[tuple[str, str]] = []
    for i, name in enumerate(names, 1):
        step = STEPS[name]
        console.print(Rule(f"[bold]第 {i} 步，共 {len(names)} 步：{step.label}[/bold]", align="left", style="cyan"))
        _explain(step)
        if not typer.confirm("是否执行此步骤？", default=True):
            note("已跳过此步骤。")
            results.append((step.label, "[dim]已跳过[/dim]"))
            continue
        results.append((step.label, "[green]已执行[/green]" if _attempt(step.run) else "[yellow]未完成[/yellow]"))
    table = Table("步骤", "结果", box=box.SIMPLE_HEAD, header_style="bold")
    for label, result in results:
        table.add_row(label, result)
    console.print(Panel(table, title="[bold]向导结果[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    note("未完成或已跳过的步骤可随时在对应分组中单独执行。")


STEP_LIST: tuple[Step, ...] = (
    Step("wizard", "首次安装向导", "依次完成安装 Python 依赖、构建网页界面、检查运行条件、设置网络与端口、启动服务、设置管理员密码、选择计算用显卡",
         "依次引导执行首次安装所需的七个步骤，每一步均可跳过", "从全新检出到可以登录使用", "取决于所执行的步骤", PROJECT,
         lambda: _wizard("首次安装向导", "首次安装所需", WIZARD)),
    Step("build-wizard", "扩展包编译与下载设置向导", "依次完成选择编译目标架构、设置编译器与 CUDA、检查编译工具、设置下载镜像、设置下载重试、安装 Alembic 与 FBX 模块",
         "依次引导执行扩展包编译与下载设置的六个步骤，每一步均可跳过", "扩展包编译与下载所依赖的设置，以及读写 DCC 场景文件的格式模块",
         "取决于所执行的步骤", PROJECT, lambda: _wizard("扩展包编译与下载设置向导", "扩展包编译与下载设置", BUILD_SETUP)),
    Step("ext-wizard", "扩展包安装向导", "依次完成登录 Hugging Face、安装手动下载的文件、安装扩展包",
         "依次引导执行安装扩展包的三个步骤，每一步均可跳过", "安装各算法所在的扩展包，以及它们所需的模型与手动下载的文件",
         "取决于所执行的步骤", PROJECT, lambda: _wizard("扩展包安装向导", "扩展包安装", EXT_WIZARD)),
    # 一键更新
    Step("update", "一键更新", "先停止服务并手动 git pull --ff-only，再升级已拉取的本地代码：备份、同步环境、构建网页、检查、迁移、启动；失败回退",
         "使用刚拉取的新版升级程序，检查工作区与停服状态，记录上次部署版本（首次升级从最近一次手动拉取记录取得）、"
         "备份设置、本地受管文件与旧版数据库，执行 uv sync、npm ci 与构建网页、lab2shot check、数据库升级与完整性检查、"
         "启动服务并确认其正常响应；环境同步之后失败，即恢复记录的代码、数据库和环境，用旧版本上线",
         "完成已拉取代码的部署；git pull 由管理员在运行本步骤之前手动执行",
         ".venv/、webui/node_modules/ 与 webui/dist/；work/db（更新前备份一份）；work/installed.json 中的部署版本；"
         "work/updates/<时间>/ 中的更新记录、设置文件与本地受管文件副本；失败回退时恢复项目代码",
         "项目目录与工作目录内；请先用原版本停止服务再拉取；不自动重装扩展包", one_click_update),
    # 安装与环境
    Step("env", "安装 Python 依赖", "按 uv.lock 创建服务与命令行所用的 Python 环境（uv sync）",
         "执行 uv sync，按 uv.lock 安装依赖", "提供服务与命令行的运行环境", ".venv/；包缓存位于 ~/.cache/uv", PROJECT, sync_env),
    Step("web", "构建网页界面", "安装前端依赖并构建网页界面（npm）",
         "执行 npm ci（严格按 package-lock.json 安装，不改动它）与 npm run build", "生成服务向浏览器提供的页面", "webui/node_modules/、webui/dist/", PROJECT, build_webui),
    Step("check", "检查运行条件", "列出 Python、Node.js、网页、显卡驱动等运行服务所需条件的状态（只读）",
         "列出 Python、uv、Node.js、网页构建、显卡驱动、工作目录与设置文件的状态", "确认缺少的条件", "无", READ_ONLY, _check_environment),
    Step("build-setup", "扩展包编译与下载设置", "扩展包编译与下载所需的设置，以及 Alembic、FBX 模块",
         "编译环境设置、工具链检查、下载镜像、重试策略、安装格式模块（Alembic、FBX）", "扩展包在独立环境中编译 CUDA 算子、获取代码与模型权重时所依赖的设置",
         "config/local.toml；格式模块安装至 third_party/，原始文件移入 downloads/installed/", PROJECT, _build_setup),
    # 扩展包编译与下载设置（安装与环境 → 扩展包编译与下载设置的三级菜单，及其向导）
    Step("archs", "选择编译目标架构", "扩展包的 CUDA 代码为哪几代架构编译（默认 Ada Lovelace 与 Blackwell）",
         "列出可选的架构（按平台名称），按编号选择", "编出来的 CUDA 代码只能在所选架构上运行，与编译这台机器插的卡无关",
         "config/local.toml（仅记录与默认值不同的项）；已装好的扩展包要重装才按新架构编译", PROJECT, _choose_archs),
    Step("build", "设置编译器与 CUDA", "CUDA 路径、C 与 C++ 编译器",
         "依次询问 CUDA 路径、C 编译器与 C++ 编译器", "扩展包安装时在独立环境中编译 CUDA 算子所用的工具",
         "config/local.toml（仅记录与默认值不同的项）", PROJECT,
         lambda: _edit_settings(("build.cuda_home", "build.cc", "build.cxx"), "编译环境设置")),
    Step("toolchain", "检查编译工具", "git、编译器、nvcc、编译目标架构，以及 nvcc 与 GCC 的兼容性（只读）",
         "检查 git、C 与 C++ 编译器、nvcc，nvcc 是否支持所选的每个编译目标架构，以及 nvcc 与 GCC 的版本是否兼容",
         "确认编译扩展包所需的工具齐全且版本相容",
         "无", READ_ONLY, lambda: _toolchain()),
    Step("mirrors", "设置下载镜像", "Hugging Face、PyPI、GitHub 镜像；官方地址无法访问时按顺序尝试",
         "依次询问 Hugging Face、PyPI 与 GitHub 镜像", "官方地址无法访问时按顺序尝试镜像；从镜像下载的文件须与官方校验值一致才使用",
         "config/local.toml（仅记录与默认值不同的项）", PROJECT,
         lambda: _edit_settings(("install.mirror_hf", "install.mirror_pypi", "install.mirror_github"), "镜像设置")),
    Step("retries", "设置下载重试", "下载失败时的重试次数与最长等待时间",
         "依次询问重试次数与两次重试之间的最长等待时间", "网络不稳定时减少安装失败",
         "config/local.toml（仅记录与默认值不同的项）", PROJECT,
         lambda: _edit_settings(("install.retries", "install.backoff_max"), "重试设置")),
    Step("formats", "安装 Alembic 与 FBX 模块", "读写 DCC 场景文件（Alembic、FBX）的模板均依赖这两个模块",
         "安装 Alembic 与 FBX 格式模块；FBX SDK 须先放入 downloads/，并在此阅读、同意其许可协议",
         "读写 DCC 场景文件的模板均依赖这两个模块",
         "third_party/ 中的格式模块；原始文件移入 downloads/installed/；许可协议的同意记录写入 work/db", PROJECT, _install_formats),
    Step("hf-login", "登录 Hugging Face", "列出须申请访问的模型，并在本机登录 Hugging Face",
         "列出仓库须申请访问的权重及其申请页面，并执行 hf auth login", "获批后，安装器以本机登录的账号下载这些权重",
         "Hugging Face 的令牌文件（默认 ~/.cache/huggingface/token，设置了 HF_HOME 时位于其中）", "当前用户主目录中的 Hugging Face 登录状态；不改动项目目录", _hf_login),
    Step("downloads", "安装手动下载的文件", "识别并安装放入 downloads/ 的文件（SMPL、FLAME、FBX SDK 等），列出仍缺的文件及其下载页面",
         "识别并安装 downloads/ 中的文件，列出每一项的下载页面、所需文件与状态",
         "部分模型须在其网站注册或同意许可后才能下载，安装器无法代为获取",
         "third_party/_body_models/ 或对应扩展包的 weights/；原始文件移入 downloads/installed/；许可协议的同意记录写入 work/db",
         PROJECT, _manual_downloads),
    Step("extensions", "安装扩展包", "列出全部扩展包及其状态，选择后依次安装",
         "列出全部扩展包，按所选编号依次执行安装前检查与安装（与管理后台「扩展包」页的「安装」相同）",
         "各算法节点在其扩展包的独立环境中运行", "third_party/<项目名>/：代码、独立环境与模型权重", PROJECT, _install_extensions),
    # 账号与安全
    Step("password", "设置管理员密码", "设置或重设管理员账号 admin 的密码",
         "设置或重设管理员账号的密码", "新安装时管理员没有密码，设置之前任何人都无法登录",
         "work/db 中的账号记录（需先启动过一次服务）；立即生效，已登录的浏览器须重新登录", PROJECT, _admin_password),
    Step("passphrase", "设置找回口令", "设置忘记管理员密码时用于重设密码的口令",
         "设置忘记管理员密码时使用的口令", "登录页「忘记密码」凭此口令重设密码；仅可在本机设置",
         "work/db 中保存口令的哈希值（需先启动过一次服务）", PROJECT, _passphrase),
    # 服务
    Step("start", "启动服务（后台）", "在后台启动服务，日志写入 work/logs/",
         "在后台启动 lab2shot ui", "提供网页与 DCC 插件所连接的服务",
         "work/logs/ui_stdout.log；首次启动时由服务创建 work/（数据库、管理员账号、本机令牌）", PROJECT, lambda: start(background=True)),
    Step("start-fg", "启动服务（前台）", "在当前终端运行服务，便于查看日志",
         "在当前终端启动 lab2shot ui，按 Ctrl-C 停止", "排查问题时直接查看日志", "与后台启动相同", PROJECT, lambda: start(background=False)),
    Step("restart", "重启服务", "重启运行中的服务，可等待任务完成",
         "通过本机令牌重启运行中的服务，可选择等待正在计算的任务完成", "使需重启生效的设置生效", "无", PROJECT, restart),
    Step("stop", "停止服务", "停止运行中的服务，可等待任务完成",
         "通过本机令牌请运行中的服务停止，可选择等当前任务算完再停或立即停止，并等到它退出；服务不回答时，确认后结束监听端口的进程",
         "关机、更换端口或停用服务", "无；排队的任务留给下一次启动的服务，选择立即停止时计算中的任务停下", PROJECT, stop),
    Step("gpus", "选择计算用显卡", "选择接受计算任务的显卡（显卡授权）",
         "选择接受计算任务的显卡", "未授权的显卡不参与计算", "运行中服务的授权名单（work/db）", PROJECT, _gpus),
    # 设置
    Step("settings", "全部设置", "按管理后台「设置」一组的各页查看与修改全部设置",
         "按管理后台「设置」一组的各页列出全部设置项，显示当前值、默认值与生效方式，并可逐项修改",
         "在不打开浏览器的情况下管理服务器设置", "config/local.toml（仅记录与默认值不同的项）；部分设置需重启服务后生效", PROJECT, _settings_menu),
    Step("server", "设置网络与端口", "按访问方式（本机自用、局域网直连、反向代理之后）设好访问范围、HTTPS、可信代理与对外域名，再设端口",
         "先问服务器怎样被访问，列出每种方式写入的每项设置；按所选写入访问范围、HTTPS、可信代理与对外域名，再询问端口，最后汇总实际写入的值",
         "三种访问方式各需一组确定的网络设置；选好方式即全部写对，不必逐项推敲",
         "config/local.toml（与「设置 → 网络与安装」为同一份）；开启 HTTPS 时，服务启动时在 work/tls/ 生成自签证书；需重启服务后生效", PROJECT,
         _network),
    # 数据库
    Step("db-status", "查看数据库状态", "数据库文件、大小、版本、完整性与全部备份（只读）",
         "显示数据库的文件、大小、版本、最近一次完整性检查与备份，并列出全部备份", "了解数据库的当前状况", "无", READ_ONLY, _db_status),
    Step("db-backup", "立即备份数据库", "立即备份一份数据库（服务运行时亦可）",
         "以 SQLite 在线备份方式复制数据库，并校验副本", "在重要操作之前保留一份可恢复的数据",
         "work/db/backups/ 中新增一份备份及其附带文件；超出保留份数时删除最旧的备份", PROJECT, _db_backup),
    Step("db-check", "检查数据库完整性", "检查数据库的完整性与外键",
         "执行 SQLite 完整性检查与外键检查", "确认数据库可以信任", "无（仅记录检查结果）", PROJECT, _db_check),
    Step("db-restore", "从备份恢复数据库", "用一份备份替换当前数据库（需先停止服务）",
         "列出全部备份，以所选备份替换当前数据库", "数据库损坏、缺失或需要回退时恢复数据",
         "work/db/lab2shot.db；原数据库移至 work/db/lab2shot.db.replaced-<时间>，不会删除", PROJECT, _db_restore),
    Step("db-upgrade", "升级数据库版本", "将数据库升级至本程序对应的版本（需先停止服务）",
         "先自动备份，再将数据库升级至本程序对应的版本", "更新 Lab2Shot 后使数据库与程序一致；启动服务时也会自动执行",
         "work/db/lab2shot.db；work/db/backups/ 中新增一份升级前的备份", PROJECT, _db_upgrade),
    Step("db-claim", "将工作目录归属本项目", "使工作目录归属本项目目录，其他项目目录的服务此后不能使用它",
         "将工作目录的归属改为本项目目录", "此后其他项目目录（其他分支、测试检出）的进程无法打开该目录",
         "工作目录中的归属记录（owner.json）", PROJECT, _db_claim),
    # 状态总览
    Step("status", "状态总览", "服务、管理员、口令、账号、扩展包、数据库与工作目录（只读）",
         "显示服务、管理员密码、找回口令、账号数、扩展包数、数据库与工作目录的概况", "总览当前状况", "无", READ_ONLY, _status),
)
STEPS: dict[str, Step] = {s.name: s for s in STEP_LIST}
# 服务器设置在启动服务之前：端口、访问范围与 HTTPS 只在服务启动时生效，默认端口 8765 可能已被本机的另一个服务占用
WIZARD = ("env", "web", "check", "server", "start", "password", "gpus")
EXT_WIZARD = ("hf-login", "downloads", "extensions")


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    summary: str
    steps: tuple[str, ...] = ()  # a second-level menu of these steps; empty: `single` runs directly
    single: str = ""


MENU: tuple[Group, ...] = (
    # the wizards first, in the order a new installation goes through them (one short line each: the menu table is one
    # row per item, the wizard's own panel lists every step); then the one-click update of an installation that runs;
    # then the groups in the same order: install, run the service (the network is set before it starts), accounts (they
    # exist once it has started), settings, database, overview
    Group("1", "首次安装向导", f"从全新检出到可以登录使用：依赖、网页、网络、启动、管理员密码、显卡（{len(WIZARD)} 步）", single="wizard"),
    Group("2", "扩展包编译与下载设置向导", f"编译目标架构、编译器与 CUDA、下载镜像与重试、Alembic 与 FBX 模块（{len(BUILD_SETUP)} 步）", single="build-wizard"),
    Group("3", "扩展包安装向导", f"Hugging Face 登录、手动下载的文件、安装扩展包（{len(EXT_WIZARD)} 步）", single="ext-wizard"),
    Group("4", "一键更新", "停服并手动拉取后，升级本地代码；失败回退并重新上线", single="update"),
    Group("5", "安装与环境", "Python 依赖、网页界面、运行条件、扩展包编译与下载设置、Hugging Face、手动下载的文件、扩展包",
          ("env", "web", "check", "build-setup", "hf-login", "downloads", "extensions")),
    Group("6", "服务", "网络与端口，启动、重启、停止服务，选择计算用显卡", ("server", "start", "start-fg", "restart", "stop", "gpus")),
    Group("7", "账号与安全", "管理员密码、找回口令（须先启动过一次服务）", ("password", "passphrase")),
    Group("8", "设置", "管理后台「设置」一组的全部设置项，按相同的各页查看与修改", single="settings"),
    Group("9", "数据库", "查看状态、检查完整性、备份、恢复、升级版本、将工作目录归属本项目",
          ("db-status", "db-check", "db-backup", "db-restore", "db-upgrade", "db-claim")),
    Group("10", "状态总览", "服务、管理员、数据库与工作目录的概况（只读）", single="status"),
)


def _explain(step: Step) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row("操作", step.what)
    grid.add_row("用途", step.why)
    grid.add_row("写入位置", step.changes)
    grid.add_row("影响范围", f"[green]{step.scope}[/green]" if step.scope == READ_ONLY else step.scope)
    group = next((g for g in MENU if (step.name if step.name not in BUILD_SETUP else "build-setup") in g.steps), None)
    title = (f"{group.label} → " if group else "") + ("扩展包编译与下载设置 → " if step.name in BUILD_SETUP else "") + step.label
    console.print(Panel(grid, title=f"[bold]{title}[/bold]", title_align="left", border_style="blue", box=box.ROUNDED))


def _group_menu(group: Group) -> None:
    """A second-level menu: the steps of one group; 0 returns to the top level."""
    while True:
        console.print()
        console.print(Rule(f"[bold]{group.key}  {group.label}[/bold]", align="left", style="cyan"))
        rows = [(str(i), STEPS[n].label, STEPS[n].summary) for i, n in enumerate(group.steps, 1)]
        console.print(menu_table(rows + [("0", "返回", "返回上一级菜单")]))
        try:
            choice = pick()
        except abort_types():
            console.print()
            return
        if choice == "0":
            return
        try:
            step = STEPS[group.steps[int(choice) - 1]]
        except (ValueError, IndexError):
            err("没有该编号，请重新输入。")
            continue
        console.print()
        _explain(step)
        _run(step)


HELP_STEPS = "、".join(s.name for s in STEP_LIST)


@app.command()
def setup(step: str = typer.Argument("", help=f"直接执行某一项而不进入菜单。可用的步骤名：{HELP_STEPS}")) -> None:
    """交互式配置菜单：按用途分组（首次安装向导、扩展包编译与下载设置向导、扩展包安装向导、一键更新、安装与环境、服务、账号与安全、设置、数据库、状态总览），每一项仅在选择后执行；执行之前先说明操作、用途、写入位置与影响范围。首次使用请运行 ./setup.sh。

    步骤名：wizard、build-wizard、ext-wizard、update、env、web、check、build-setup、archs、build、toolchain、mirrors、retries、formats、hf-login、downloads、extensions、password、passphrase、start、start-fg、restart、stop、gpus、settings、server、db-status、db-backup、db-check、db-restore、db-upgrade、db-claim、status。"""
    if step:
        found = STEPS.get(step)
        if found is None:
            err(f"没有名为 {escape(step)} 的步骤。可用的步骤名：{HELP_STEPS}。")
            raise typer.Exit(2)
        _explain(found)
        if not _run(found, menu=False):
            raise typer.Exit(1)
        return
    while True:
        console.print()
        _header()
        console.print(menu_table([(g.key, g.label, g.summary) for g in MENU] + [("0", "退出", "退出配置菜单")]))
        try:
            choice = pick()
        except abort_types():  # Ctrl-C or end of input at the top level: leave quietly
            console.print()
            return
        if choice == "0":
            return
        group = next((g for g in MENU if g.key == choice), None)
        if group is None:
            err("没有该编号，请重新输入。")
            continue
        if group.single:
            console.print()
            _explain(STEPS[group.single])
            _run(STEPS[group.single])
        else:
            _group_menu(group)


def _attempt(fn: Callable[[], object], menu: bool = True) -> bool:
    """Run one item, ending back at the menu whatever happens inside it: Ctrl-C in a sub-prompt is typer.Abort /
    click.Abort rather than KeyboardInterrupt; a command the menu calls (the restart item) ends with typer.Exit on
    failure; a work folder this checkout may not use, a database that is refused, or a service that did not stop, is a
    message, not a traceback.
    `menu` False (`lab2shot setup <step>`): Ctrl-C ends the command quietly and an Exit keeps its code.
    Returns whether the item finished."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    try:
        fn()
        return True
    except abort_types():
        console.print()
        note("已取消" + ("，返回上一级菜单。" if menu else "。"))
        if not menu:
            raise typer.Exit(130)
    except typer.Exit as exc:
        if not menu:
            raise
        if not exc.exit_code:
            return True
        note("此项未能完成，返回上一级菜单。")
    except (WorkDirError, DatabaseError, NotStopped) as exc:
        _trouble(exc)
    return False


def _run(step: Step, menu: bool = True) -> bool:
    """One step from the menu (or from `lab2shot setup <step>` when `menu` is False); see _attempt."""
    return _attempt(step.run, menu)
