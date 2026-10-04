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

from .. import i18n
from .base import abort_types, app, console, err, mark, menu_table, note, ok, pick, say, warn, when
from .service import (NotStopped, address, build_webui, initialized, recorded_address, restart, running, start, stop,
                      sync_env)
from .update import one_click_update


WIDE_COMMA = chr(0xFF0C)  # the comma a Chinese keyboard types between numbers: read as a space, like ","


# ------------------------------------------------------------------ what the menu inspects


def _trouble(exc: Exception) -> None:
    """A WorkDirError, DatabaseError or NotStopped raised by any item: the message, followed by the available remedies."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    err(escape(str(exc)))
    if isinstance(exc, WorkDirError):
        note(i18n.t("cli.setup.trouble.workdir"))
    elif isinstance(exc, DatabaseError):
        note(i18n.t("cli.setup.trouble.database"))


def _admin_state() -> str:
    """The administrator's password state for the header, read only when the database already exists."""
    from ..database import DatabaseError
    from ..workdir import WorkDirError

    if not initialized():
        return f"[dim]{i18n.t('cli.setup.header.admin_uninitialized')}[/dim]"
    from .. import accounts

    try:
        owner = accounts.admin()
    except (WorkDirError, DatabaseError, OSError):
        return i18n.t("cli.setup.header.admin_unreadable")
    if owner.no_password:
        return i18n.t("cli.setup.header.admin_no_password", name=owner.username)
    return i18n.t("cli.setup.header.admin_password_set", name=owner.username)


def _header() -> None:
    """The title panel above the top-level menu: version, service, administrator, work folder."""
    from .. import __version__
    from ..config import settings

    state = running()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    if state:
        grid.add_row(i18n.t("cli.setup.header.service"), i18n.t("cli.setup.header.running",
                     address=recorded_address() or address(), boot=state.get('boot', '?')))
    else:
        grid.add_row(i18n.t("cli.setup.header.service"), i18n.t("cli.setup.header.stopped", address=address()))
    grid.add_row(i18n.t("cli.setup.header.admin"), _admin_state())
    grid.add_row(i18n.t("cli.setup.header.work_dir"), str(settings().work_dir))
    grid.add_row(i18n.t("cli.setup.header.settings_file"), str(settings().file))
    console.print(Panel(grid, title=f"[bold]{i18n.t('cli.setup.header.title', version=__version__)}[/bold]", title_align="left",
                        box=box.ROUNDED, border_style="cyan", padding=(0, 1)))


# ------------------------------------------------------------------ installation and environment


NODE_NEED = "Node.js ^20.19 || >=22.12"  # webui/package.json "engines" (^20.19.0 || >=22.12.0), as vite 8 requires


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
    rows.append(("Python", True, i18n.t("cli.setup.env.python", version=sys.version.split()[0], path=sys.executable)))
    rows.append(("uv", bool(shutil.which("uv")), shutil.which("uv") or i18n.t("cli.setup.env.uv_missing")))
    node = shutil.which("node")
    node_v = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip() if node else ""
    if not node:
        rows.append(("Node.js", False, i18n.t("cli.setup.env.node_missing", need=NODE_NEED)))
    elif _node_ok(node_v):
        rows.append(("Node.js", True, node_v))
    else:
        rows.append(("Node.js", False, i18n.t("cli.setup.env.node_old", version=node_v, need=NODE_NEED)))
    rows.append((i18n.t("cli.setup.env.webui"), WEBUI_DIST.is_dir(),
                 str(WEBUI_DIST) if WEBUI_DIST.is_dir() else i18n.t("cli.setup.env.webui_missing")))
    smi = shutil.which("nvidia-smi")
    gpus = ""
    if smi:
        out = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
        gpus = i18n.t("cli.setup.env.gpu_sep").join(line.strip() for line in out.splitlines() if line.strip())
    rows.append((i18n.t("cli.setup.env.driver"), bool(smi), gpus or i18n.t("cli.setup.env.driver_missing")))
    s = settings()
    work = s.work_dir
    who = owner(work) if work.is_dir() else None
    if who is None:
        rows.append((i18n.t("cli.setup.env.work_dir"), True, i18n.t("cli.setup.env.work_dir_exists", path=work)
                     if work.is_dir() else i18n.t("cli.setup.env.work_dir_later", path=work)))
    elif who == str(ROOT):
        rows.append((i18n.t("cli.setup.env.work_dir"), True, i18n.t("cli.setup.env.work_dir_ours", path=work)))
    else:
        rows.append((i18n.t("cli.setup.env.work_dir"), False, i18n.t("cli.setup.env.work_dir_other", path=work, owner=who)))
    rows.append((i18n.t("cli.setup.env.settings_file"), None, i18n.t("cli.setup.env.settings_file_exists", path=s.file)
                 if Path(s.file).is_file() else i18n.t("cli.setup.env.settings_file_missing", path=s.file)))
    table = Table(i18n.t("cli.setup.table.item"), "", i18n.t("cli.setup.table.description"), box=box.SIMPLE_HEAD,
                  header_style="bold")
    for name, passed, text in rows:
        table.add_row(name, mark(passed), text)
    console.print(table)
    missing = [name for name, passed, _ in rows if passed is False]
    if missing:
        warn(i18n.t("cli.setup.env.missing", items=i18n.separator().join(missing)))
    else:
        ok(i18n.t("cli.setup.env.all_ok"))


def _toolchain() -> None:
    """The core build kit every compiled extension requires: what the settings point to, and whether it is present."""
    from ..config import settings
    from lab2shot_worker.build import compute_caps

    s = settings()
    cuda_home = s["build.cuda_home"] or os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    system = i18n.t("cli.setup.toolchain.system_default")
    rows = [("git", shutil.which("git"), i18n.t("cli.setup.toolchain.git")),
            (i18n.t("cli.setup.toolchain.cc"), shutil.which(s["build.cc"] or "cc"),
             i18n.t("cli.setup.toolchain.setting", key="build.cc", value=s['build.cc'] or system)),
            (i18n.t("cli.setup.toolchain.cxx"), shutil.which(s["build.cxx"] or "c++"),
             i18n.t("cli.setup.toolchain.setting", key="build.cxx", value=s['build.cxx'] or system)),
            (i18n.t("cli.setup.toolchain.cuda"), (Path(cuda_home) / "bin" / "nvcc").is_file() and str(Path(cuda_home) / "bin" / "nvcc"),
             i18n.t("cli.setup.toolchain.setting", key="build.cuda_home",
                    value=s['build.cuda_home'] or i18n.t("cli.setup.toolchain.cuda_default"))),
            ("ninja", shutil.which("ninja"), i18n.t("cli.setup.toolchain.ninja"))]
    table = Table(i18n.t("cli.setup.table.item"), "", i18n.t("cli.setup.table.description"), box=box.SIMPLE_HEAD,
                  header_style="bold")
    for name, found, text in rows:
        table.add_row(name, mark(bool(found)), f"{found}  {text}" if found else text)
    console.print(table)
    from lab2shot_shared.gpu_arch import target_label

    here = [target_label("sm_" + c.replace(".", "")) for c in compute_caps()]
    console.print(i18n.t("cli.setup.toolchain.here", archs=i18n.separator().join(here) if here
                         else i18n.t("cli.setup.toolchain.no_gpu")))
    _check_targets(Path(cuda_home) / "bin" / "nvcc", list(s.value("build.archs")))
    from ..installer import toolchain

    state, why = toolchain.problem()
    text = i18n.t("cli.setup.toolchain.compatible", state=why.text if why else i18n.t("cli.setup.toolchain.compatible_ok"))
    {"ok": ok, "warning": warn, "blocked": err}[state](text)


def _check_targets(nvcc: Path, archs: list[str]) -> None:
    """The compile targets (build.archs), each checked against what this nvcc can compile for (--list-gpu-arch)."""
    from lab2shot_shared.gpu_arch import target_label

    console.print(i18n.t("cli.setup.toolchain.targets", archs=i18n.separator().join(target_label(a) for a in archs)))
    if not nvcc.is_file():
        return
    try:
        listed = subprocess.run([str(nvcc), "--list-gpu-arch"], capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return
    missing = [a for a in archs if a.replace("sm_", "compute_") not in listed]
    if missing:
        warn(i18n.t("cli.setup.toolchain.unsupported", nvcc=nvcc,
                    archs=i18n.separator().join(target_label(a) for a in missing)))
    else:
        ok(i18n.t("cli.setup.toolchain.supported"))


def _choose_archs() -> None:
    """The compile targets (setting build.archs), chosen by number and named by platform, never by graphics card."""
    from lab2shot_shared.gpu_arch import TARGET_LABELS, target_label

    from ..config import SCHEMA, settings

    current = list(settings().value("build.archs"))
    tokens = list(TARGET_LABELS)
    note(i18n.t("cli.setup.archs.intro",
                default=i18n.separator().join(target_label(a) for a in SCHEMA['build.archs'].default)))
    table = Table(i18n.t("cli.menu.number"), i18n.t("cli.setup.archs.arch"), "", box=box.SIMPLE_HEAD, header_style="bold")
    for i, token in enumerate(tokens, 1):
        table.add_row(str(i), target_label(token), f"[green]{i18n.t('cli.setup.common.selected')}[/green]" if token in current else "")
    console.print(table)
    answer = typer.prompt(i18n.t("cli.setup.archs.prompt"), default="", show_default=False).strip()
    if not answer:
        note(i18n.t("cli.setup.archs.kept", archs=i18n.separator().join(target_label(a) for a in current)))
        return
    numbers = answer.replace(",", " ").replace(WIDE_COMMA, " ").split()
    if not all(n.isdigit() and 1 <= int(n) <= len(tokens) for n in numbers):
        err(i18n.t("cli.setup.common.bad_number_unchanged"))
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
        console.print(f"[bold]{ext.title}[/bold]  " + i18n.t("cli.setup.ext.format_line", state=i18n.t(
            "cli.setup.ext.installed") if state['installed'] else i18n.t("cli.setup.ext.not_installed"), license=ext.license.name))
        if state["installed"] and not typer.confirm(i18n.t("cli.setup.ext.reinstall"), default=False):
            continue
        if not typer.confirm(i18n.t("cli.setup.ext.install"), default=True):
            continue
        _consent_pending(ext.name)
        if not _checklist(ext):
            warn(i18n.t("cli.setup.ext.unmet"))
            continue
        _run_install(ext, stop=False)


def _install_extensions() -> None:
    """Extension packages from the command line: every one with its state; the chosen ones installed in turn, each
    after the licences of its hand downloads (_consent_pending) and the installer's checklist, the same install as
    `lab2shot ext install` (the admin page's 安装 runs it as a farm task). The format modules are 扩展包编译与下载设置's."""
    from ..extensions import extensions
    from ..extensions.status import extension_status

    from .extensions import _checklist, _run_install

    # a base (no nodes of its own) is never chosen on its own: installing an extension that runs in it installs it
    exts = [e for e in extensions().values() if not e.format_module and not e.is_base]
    ready = {}
    table = Table(i18n.t("cli.menu.number"), i18n.t("cli.setup.ext.extension"), i18n.t("cli.setup.ext.state"),
                  i18n.t("cli.setup.ext.license"), box=box.SIMPLE_HEAD, header_style="bold")
    for i, ext in enumerate(exts, 1):
        st = extension_status(ext)
        ready[ext.name] = st["ready"]
        not_installed = (st.get("message") or {}).get("code") == "E-EXT-NOTINSTALLED"  # the 「未安装」 branch
        color = "green" if st["ready"] else "dim" if not_installed else "yellow"
        table.add_row(str(i), i18n.t("cli.setup.ext.title", title=escape(ext.title), name=ext.name),
                      f"[{color}]{escape(st['label'])}[/{color}]", escape(ext.license.name))
    console.print(table)
    answer = typer.prompt(i18n.t("cli.setup.ext.prompt"), default="", show_default=False).strip()
    if not answer:
        note(i18n.t("cli.setup.ext.none_chosen"))
        return
    if answer.lower() == "a":
        chosen = [e for e in exts if not ready[e.name]]
    else:
        numbers = answer.replace(",", " ").replace(WIDE_COMMA, " ").split()
        if not all(n.isdigit() and 1 <= int(n) <= len(exts) for n in numbers):
            err(i18n.t("cli.setup.ext.bad_number"))
            return
        chosen = list(dict.fromkeys(exts[int(n) - 1] for n in numbers))
    if not chosen:
        note(i18n.t("cli.setup.ext.all_ready"))
        return
    results: list[tuple[str, str]] = []
    for i, ext in enumerate(chosen, 1):
        console.print(Rule(f"[bold]{i}/{len(chosen)}  {i18n.t('cli.setup.ext.title', title=escape(ext.title), name=ext.name)}[/bold]",
                           align="left", style="cyan"))
        console.print(i18n.t("cli.setup.ext.license_line", name=escape(ext.license.name), summary=escape(ext.license.summary)),
                      highlight=False)
        _consent_pending(ext.name)
        if not _checklist(ext):
            warn(i18n.t("cli.setup.ext.unmet"))
            results.append((ext.title, f"[yellow]{i18n.t('cli.setup.ext.result_unmet')}[/yellow]"))
            continue
        why = _run_install(ext, stop=False)
        results.append((ext.title, f"[green]{i18n.t('cli.setup.ext.installed')}[/green]" if not why
                        else f"[red]{i18n.t('cli.setup.ext.result_failed')}[/red]  {escape(why)}"))
    summary = Table(i18n.t("cli.setup.ext.extension"), i18n.t("cli.setup.common.result"), box=box.SIMPLE_HEAD, header_style="bold")
    for title, result in results:
        summary.add_row(escape(title), result)
    console.print(Panel(summary, title=f"[bold]{i18n.t('cli.setup.ext.results')}[/bold]", title_align="left",
                        border_style="cyan", box=box.ROUNDED))


def _hf_login() -> None:
    """The weights whose repository requires an approved access request, and the Hugging Face login they need."""
    from huggingface_hub import get_token

    from ..extensions import extensions

    rows = [(ext.title, w.key, w.page) for ext in sorted(extensions().values(), key=lambda e: e.title.lower())
            for w in ext.weights if w.gated]
    table = Table(i18n.t("cli.setup.ext.extension"), i18n.t("cli.setup.hf.weights"), i18n.t("cli.setup.hf.page"),
                  box=box.SIMPLE_HEAD, header_style="bold")
    for title, key, page in rows:
        table.add_row(title, key, page)
    console.print(table)
    console.print(i18n.t("cli.setup.hf.intro"))
    if os.environ.get("HF_TOKEN"):
        ok(i18n.t("cli.setup.hf.env_token"))
        return
    if get_token():
        ok(i18n.t("cli.setup.hf.saved_token"))
        if not typer.confirm(i18n.t("cli.setup.hf.again"), default=False):
            return
    elif not typer.confirm(i18n.t("cli.setup.hf.now"), default=True):
        return
    note(i18n.t("cli.setup.hf.token_hint"))
    hf = Path(sys.executable).parent / "hf"
    if not hf.is_file():
        err(i18n.t("cli.setup.hf.no_cli", path=hf))
        return
    if subprocess.run([str(hf), "auth", "login"]).returncode != 0:
        err(i18n.t("cli.setup.hf.failed"))
        return
    ok(i18n.t("cli.setup.hf.done"))


MANUAL_STATES = {"ready": "cli.setup.manual.state.ready", "consent": "cli.setup.manual.state.consent",
                 "unrecognised": "cli.setup.manual.state.unrecognised", "missing": "cli.setup.manual.state.missing"}


def _manual_downloads() -> None:
    """The files people download by hand: where, which one, what for, and whether each is in place. The files already
    in downloads/ are recognised and installed first; the ones whose licence must be accepted are shown and, on the
    user's yes, installed (_consent_pending)."""
    from ..extensions import manual

    manual.check()
    _consent_pending()
    view = manual.view()
    for row in view["items"]:
        needs = i18n.separator().join(e["title"] for e in row["needed_by"]) or "—"
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", no_wrap=True)
        grid.add_column(overflow="fold")
        state = MANUAL_STATES.get(row["state"])
        grid.add_row(i18n.t("cli.setup.manual.status"), (i18n.t(state) if state else escape(row["state"]))
                     + (f"  {escape(row['installed'])}" if row["installed"] else ""))
        grid.add_row(i18n.t("cli.setup.manual.what"), escape(row["what"]))
        grid.add_row(i18n.t("cli.setup.manual.needed_by"), escape(needs))
        grid.add_row(i18n.t("cli.setup.manual.page"), escape(row["page"]))
        grid.add_row(i18n.t("cli.setup.manual.download"), escape(row["download"]))
        grid.add_row(i18n.t("cli.setup.manual.filename"), escape(row["filename"]))
        grid.add_row(i18n.t("cli.setup.table.description"), escape(row["note"]))
        for f in row["files"]:
            grid.add_row(i18n.t("cli.setup.manual.in_downloads"), escape(i18n.t(
                "cli.setup.manual.file", name=f['name'], why=f.get('why') or i18n.t("cli.setup.manual.awaiting_consent"))))
        console.print(Panel(grid, title=f"[bold]{escape(row['title'])}[/bold]", title_align="left", border_style="blue",
                            box=box.ROUNDED))
    for f in view["unknown"]:
        warn(escape(i18n.t("cli.setup.manual.unknown", name=f['name'], why=f['why'])))
    console.print(i18n.t("cli.setup.manual.how", inbox=escape(view['inbox']['path'])))


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
        warn(i18n.t("cli.setup.consent.uninitialized"))
        return
    for row in rows:
        for f in (f for f in row["files"] if f["state"] == "consent"):
            title = row["title"]
            try:
                note(i18n.t("cli.setup.consent.reading", name=f['name'], title=title))
                shown = manual.licence(f["name"])
                console.print(Panel(escape(shown["text"]), title=i18n.t("cli.setup.consent.panel", title=escape(title),
                                                                        name=escape(f['name'])),
                                    title_align="left", border_style="yellow", box=box.ROUNDED))
                if not typer.confirm(i18n.t("cli.setup.consent.ask", title=title), default=False):
                    note(i18n.t("cli.setup.consent.declined", title=title))
                    continue
                manual.accept(f["name"], shown["sha256"], _cli_who())
            except MessageError as exc:
                err(escape(str(exc)))
                continue
            ok(i18n.t("cli.setup.consent.done", title=title))


# 扩展包编译与下载设置的六项（steps in STEP_LIST: `lab2shot setup build` etc. runs one directly); also its wizard's order:
# the compile targets first, then the compilers, then the check of both
BUILD_SETUP = ("archs", "build", "toolchain", "mirrors", "retries", "formats")


def _build_setup() -> None:
    """The third-level menu of 扩展包编译与下载设置: what compiling extensions and building their environments depend on."""
    while True:
        console.print(Rule(f"[bold]{STEPS['build-setup'].label}[/bold]", align="left", style="cyan"))
        rows = [(str(i), STEPS[n].label, STEPS[n].summary) for i, n in enumerate(BUILD_SETUP, 1)]
        console.print(menu_table(rows + [("0", i18n.t("cli.setup.common.back"), i18n.t("cli.setup.common.back_up"))]))
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
            err(i18n.t("cli.setup.common.no_such_number"))
            continue
        _attempt(step.run)


# ------------------------------------------------------------------ accounts and security


def _not_initialized() -> None:
    warn(i18n.t("cli.setup.account.uninitialized"))


def _admin_password() -> None:
    if not initialized():  # this item never creates work/ or the database: the first start of the service does
        _not_initialized()
        return
    from .. import accounts

    from .accounts import admin_password

    owner = accounts.admin()
    if owner.no_password:
        warn(i18n.t("cli.setup.account.no_password", name=owner.username))
    else:
        console.print(i18n.t("cli.setup.account.password_set", name=owner.username, when=when(owner.password_set or 0),
                             by=accounts.password_by_text(owner.password_by)))
        if not typer.confirm(i18n.t("cli.setup.account.set_again"), default=False):
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
        console.print(i18n.t("cli.setup.account.passphrase_set", when=when(phrase['set'])))
        if not typer.confirm(i18n.t("cli.setup.account.passphrase_change"), default=False):
            return
    else:
        console.print(i18n.t("cli.setup.account.passphrase_intro"))
    admin_passphrase()


# ------------------------------------------------------------------ settings


def _ask_setting(key: str, current: object):
    """Ask for one setting in the form its kind requires: a numbered choice (or several), yes or no, a number, a list, or text."""
    from ..config import SCHEMA

    s = SCHEMA[key]
    if s.kind == "choice":
        options = list(s.options)
        for i, (value, label) in enumerate(options, 1):
            console.print(f"  [bold cyan]{i}[/bold cyan]  " + i18n.t("cli.setup.ask.option", label=label, value=value)
                          + (f"  [dim]{i18n.t('cli.setup.ask.current_mark')}[/dim]" if value == current else ""))
        n = typer.prompt(i18n.t("cli.setup.ask.choice", label=s.label),
                         default=str(next((i for i, (v, _) in enumerate(options, 1) if v == current), 1)))
        try:
            return options[int(n) - 1][0]
        except (ValueError, IndexError):
            warn(i18n.t("cli.setup.ask.bad_number_kept"))
            return current
    if s.kind == "multi":
        options = list(s.offered())
        for i, (value, label) in enumerate(options, 1):
            console.print(f"  [bold cyan]{i}[/bold cyan]  " + i18n.t("cli.setup.ask.option", label=label, value=value)
                          + (f"  [dim]{i18n.t('cli.setup.ask.selected_mark')}[/dim]" if value in current else ""))
        text = typer.prompt(i18n.t("cli.setup.ask.multi", label=s.label), default="", show_default=False).strip()
        if not text:
            return current
        if text == "-":
            return []
        numbers = text.replace(WIDE_COMMA, " ").replace(",", " ").split()
        if not all(n.isdigit() and 1 <= int(n) <= len(options) for n in numbers):
            warn(i18n.t("cli.setup.ask.bad_number_kept"))
            return current
        return [options[int(n) - 1][0] for n in numbers]
    if s.kind == "bool":
        return typer.confirm(i18n.t("cli.setup.ask.bool", label=s.label), default=bool(current))
    bounds = ""
    if s.kind in ("int", "number") and (s.min is not None or s.max is not None):
        bounds = i18n.t("cli.setup.ask.bounds", min=s.min if s.min is not None else '',
                        max=s.max if s.max is not None else '', unit=' ' + s.unit if s.unit else '')
    if s.kind == "int":
        return typer.prompt(s.label + bounds, default=int(current), type=int)
    if s.kind == "number":
        return typer.prompt(s.label + bounds, default=float(current), type=float)
    if s.kind == "list":
        return _ask_list(key, list(current))
    # The current value is said in words, apart from what empty means (a bracketed default right after 「留空表示只用官方」
    # would read as if the mirror were the official address). Enter keeps the value, so "-" is what makes it empty
    shown = str(current or "")
    hint = i18n.t("cli.setup.ask.current", value=shown) if shown else i18n.t("cli.setup.ask.current_empty")
    if s.empty:
        hint += (i18n.t("cli.setup.ask.clear", empty=s.empty) if shown
                 else i18n.t("cli.setup.ask.empty_means", empty=s.empty))
    text = typer.prompt(i18n.t("cli.setup.ask.text", label=s.label, hint=hint), default="", show_default=False).strip()
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
        table = Table(i18n.t("cli.menu.number"), label, box=box.SIMPLE_HEAD, header_style="bold")
        for i, item in enumerate(items, 1):
            table.add_row(str(i), escape(item))
        console.print(table)
        note(i18n.t("cli.setup.ask.list_help"))
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
            err(i18n.t("cli.setup.ask.list_bad"))


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
        note(i18n.t("cli.setup.settings.unchanged"))
        return []
    ok(i18n.t("cli.setup.settings.saved", names=i18n.separator().join(SCHEMA[k].label for k in changed), file=s.file))
    return changed


def _offer_restart(keys: list[str]) -> None:
    """After settings that take effect only on restart have changed: list them, and offer to restart the service."""
    from ..config import SCHEMA, settings

    keys = [k for k in dict.fromkeys(keys) if SCHEMA[k].restart]
    if not keys:
        return
    s = settings()
    table = Table(i18n.t("cli.menu.name"), i18n.t("cli.setup.restart.new_value"), i18n.t("cli.setup.restart.why"),
                  box=box.SIMPLE_HEAD, header_style="bold")
    for k in keys:
        table.add_row(SCHEMA[k].label, escape(SCHEMA[k].says(s.file_value(k))), SCHEMA[k].restart)
    console.print(Panel(table, title=f"[bold]{i18n.t('cli.setup.restart.title')}[/bold]", title_align="left",
                        border_style="yellow", box=box.ROUNDED))
    if not running():
        note(i18n.t("cli.setup.restart.not_running"))
        return
    if typer.confirm(i18n.t("cli.setup.restart.ask"), default=False):
        _attempt(restart)
    else:
        note(i18n.t("cli.setup.restart.later"))


def _edit_settings(keys: tuple[str, ...], title: str) -> None:
    """Ask a fixed list of settings in turn, then save them together."""
    _provide_options()
    from ..config import SCHEMA, settings

    s = settings()
    console.print(f"[bold]{title}[/bold]  " + i18n.t("cli.setup.settings.edit_hint", file=s.file.name))
    changes = {}
    for key in keys:
        spec = SCHEMA[key]
        if spec.only_if and not changes.get(spec.only_if, s.file_value(spec.only_if)):
            continue
        if spec.note:
            note(spec.note)
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
    Access("1", i18n.t("cli.setup.network.local.label"), i18n.t("cli.setup.network.local.purpose"),
           {"server.host": "127.0.0.1", "server.https": False, "server.trusted_proxies": "", "server.names": ""}, {}),
    Access("2", i18n.t("cli.setup.network.lan.label"), i18n.t("cli.setup.network.lan.purpose"),
           {"server.host": "0.0.0.0", "server.https": True, "server.trusted_proxies": "", "server.names": None},
           {"server.names": i18n.t("cli.setup.network.lan.names")}),
    Access("3", i18n.t("cli.setup.network.proxy.label"), i18n.t("cli.setup.network.proxy.purpose"),
           {"server.host": None, "server.https": False, "server.trusted_proxies": None, "server.names": ""},
           {"server.host": '= (a) "127.0.0.1"  (b) "0.0.0.0"',
            "server.trusted_proxies": i18n.t("cli.setup.network.proxy.trusted")}),
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
        lines.append(i18n.t("cli.setup.network.port_line", port=s.file_value('server.port')))
        rows.append((a.key, a.label, escape("\n".join([a.purpose, *lines]))))
    return rows


def _proxy_remote(current: str) -> str:
    """The address of the proxy machine (③ b): required, and of the form the admin page accepts (Setting.check);
    asked until it is both."""
    from ..config import SCHEMA

    spec = SCHEMA["server.trusted_proxies"]
    while True:
        text = typer.prompt(i18n.t("cli.setup.network.proxy_ip"), default=current or None,
                            show_default=bool(current)).strip()
        if not text:
            err(i18n.t("cli.setup.network.proxy_ip_required"))
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
    console.print(i18n.t("cli.setup.network.question", file=s.file.name))
    console.print(menu_table(_access_rows(s) + [("0", i18n.t("cli.setup.common.back"), i18n.t("cli.setup.network.no_change"))]))
    console.print(Panel(i18n.t("cli.setup.network.difference"),
                        title=f"[bold]{i18n.t('cli.setup.network.difference_title')}[/bold]", title_align="left",
                        border_style="dim", box=box.ROUNDED))
    while True:
        choice = pick(_access_now(s))
        if choice == "0":
            note(i18n.t("cli.setup.network.unchanged"))
            return
        access = next((a for a in ACCESS if a.key == choice), None)
        if access is None:
            err(i18n.t("cli.setup.common.no_such_number"))
            continue
        changes = {k: v for k, v in access.values.items() if v is not None}
        remote = False
        if access.key == "2":
            note(i18n.t("cli.setup.network.names_hint"))
            changes["server.names"] = _ask_setting("server.names", s.file_value("server.names"))
        if access.key == "3":
            was_remote = s.file_value("server.host") == "0.0.0.0" and bool(s.file_value("server.trusted_proxies"))
            console.print(f"  [bold cyan]1[/bold cyan]  (a) {i18n.t('cli.setup.network.proxy_here')}\n"
                          f"  [bold cyan]2[/bold cyan]  (b) {i18n.t('cli.setup.network.proxy_elsewhere')}")
            where = typer.prompt(i18n.t("cli.setup.network.proxy_where"), default="2" if was_remote else "1").strip()
            if where not in ("1", "2"):
                err(i18n.t("cli.setup.network.bad_choice"))
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
        note(i18n.t("cli.setup.network.not_saved"))
    table = Table(i18n.t("cli.setup.settings.key"), i18n.t("cli.menu.name"), i18n.t("cli.setup.network.value"), "",
                  box=box.SIMPLE_HEAD, header_style="bold")
    for k in NETWORK_KEYS:
        table.add_row(k, SCHEMA[k].label, escape(_toml(s.file_value(k))),
                      f"[cyan]{i18n.t('cli.setup.network.changed')}[/cyan]" if k in changed else "")
    port = s.file_value("server.port")
    way = i18n.t("cli.setup.network.way", key=access.key, label=access.label)
    if access.key == "3":
        way += i18n.t("cli.setup.network.way_remote") if remote else i18n.t("cli.setup.network.way_local")
    hints = [way]
    if access.key == "2":
        hints.append(i18n.t("cli.setup.network.ca_hint", url=ca_url(port)))
    if access.key == "3":
        target = f"{lan_address()}:{port}" if remote else f"127.0.0.1:{port}"
        hints.append(i18n.t("cli.setup.network.proxy_hint", target=target))
    if remote:
        hints.append(i18n.t("cli.setup.network.firewall_hint", proxy=s.file_value('server.trusted_proxies'), port=port))
    hints.append(i18n.t("cli.setup.network.later"))
    console.print(Panel(Stack(table, *(escape(h) for h in hints)),
                        title=f"[bold]{i18n.t('cli.setup.network.written', file=s.file)}[/bold]",
                        title_align="left", border_style="green", box=box.ROUNDED))
    _offer_restart(changed)


def _source_of(key: str) -> str:
    """Where a setting's value comes from, as a short table cell."""
    from ..config import SCHEMA, settings

    s = settings()
    spec = SCHEMA[key]
    if key in s.command:
        return f"[magenta]{i18n.t('cli.setup.source.overridden', by=s.command[key][1])}[/magenta]"
    if not spec.admin:
        return f"[dim]{i18n.t('cli.setup.source.file_only')}[/dim]"
    if key in s.saved:
        return f"[cyan]{i18n.t('cli.setup.source.modified')}[/cyan]"
    if spec.auto:
        return f"[dim]{i18n.t('cli.setup.source.auto')}[/dim]"
    return f"[dim]{i18n.t('cli.setup.source.default')}[/dim]"


def _setting_detail(key: str, pending: list[str]) -> None:
    """The fourth level: one setting in full, then modify it or restore its default."""
    from ..config import SCHEMA, settings

    while True:
        s = settings()
        spec = SCHEMA[key]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold", no_wrap=True)
        grid.add_column(overflow="fold")
        grid.add_row(i18n.t("cli.setup.settings.key"), key)
        grid.add_row(i18n.t("cli.setup.table.description"), escape(spec.note))
        grid.add_row(i18n.t("cli.setup.settings.current"), escape(spec.says(s.file_value(key))))
        if key in s.command:
            grid.add_row(i18n.t("cli.setup.settings.this_run"), i18n.t("cli.setup.settings.this_run_value",
                         value=escape(spec.says(s.value(key))), by=s.command[key][1]))
        grid.add_row(i18n.t("cli.setup.settings.default"), escape(spec.says(spec.default_now))
                     + (i18n.t("cli.setup.settings.auto", how=escape(spec.auto.says())) if spec.auto else ""))
        grid.add_row(i18n.t("cli.setup.settings.takes_effect"), i18n.t("cli.setup.settings.needs_restart", why=spec.restart)
                     if spec.restart else i18n.t("cli.setup.settings.at_once"))
        if spec.only_if:
            grid.add_row(i18n.t("cli.setup.settings.requires"), i18n.t("cli.setup.settings.only_if", label=SCHEMA[spec.only_if].label))
        console.print(Panel(grid, title=f"[bold]{spec.label}[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
        if not spec.admin:
            warn(i18n.t("cli.setup.settings.file_only", file=s.file))
            return
        rows = [("1", i18n.t("cli.setup.settings.modify"), i18n.t("cli.setup.settings.modify_hint"))]
        if key in s.saved:
            rows.append(("2", i18n.t("cli.setup.settings.reset"),
                         i18n.t("cli.setup.settings.reset_hint", value=escape(spec.says(spec.default_now)))))
        rows.append(("0", i18n.t("cli.setup.common.back"), i18n.t("cli.setup.common.back_up")))
        console.print(menu_table(rows))
        choice = pick()
        if choice == "0":
            return
        if choice == "1":
            changed = _save_settings({key: _ask_setting(key, s.file_value(key))})
        elif choice == "2" and key in s.saved:
            changed = _save_settings({key: spec.default_now})
        else:
            err(i18n.t("cli.setup.common.no_such_number"))
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
        console.print(Rule(f"[bold]{i18n.t('cli.setup.settings.page_title', page=PAGES[page].label)}[/bold]",
                           align="left", style="cyan"))
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
        table.add_column(i18n.t("cli.menu.number"), justify="right", style="bold cyan", no_wrap=True)
        table.add_column(i18n.t("cli.menu.name"), no_wrap=True)
        table.add_column(i18n.t("cli.setup.settings.current"))
        table.add_column(i18n.t("cli.setup.settings.default"))
        table.add_column(i18n.t("cli.setup.settings.effect"), no_wrap=True)
        table.add_column(i18n.t("cli.setup.settings.source"), no_wrap=True)
        for i, k in enumerate(keys, 1):
            spec = SCHEMA[k]
            if len(groups) > 1 and (i == 1 or SCHEMA[keys[i - 2]].group != spec.group):
                table.add_row("", f"[bold]{groups[spec.group]}[/bold]", "", "", "", "")
            name = spec.label
            if spec.only_if and not s.file_value(spec.only_if):
                name = f"[dim]{i18n.t('cli.setup.settings.needs_on', name=name, label=SCHEMA[spec.only_if].label)}[/dim]"
            table.add_row(str(i), name, escape(spec.says(s.value(k))), escape(spec.says(spec.default_now)),
                          f"[yellow]{i18n.t('cli.setup.settings.restart_short')}[/yellow]" if spec.restart
                          else i18n.t("cli.setup.settings.at_once_short"), _source_of(k))
        if wizard:
            table.add_row(wizard, i18n.t("cli.setup.settings.network_wizard"), i18n.t("cli.setup.settings.network_wizard_hint"),
                          "", "", "")
        table.add_row("0", i18n.t("cli.setup.common.back"), "", "", "", "")
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
                err(i18n.t("cli.setup.common.no_such_number"))
                continue
        try:
            if choice == wizard:
                _network()
            else:
                _setting_detail(key, pending)
        except abort_types():
            console.print()
            note(i18n.t("cli.setup.common.cancelled_unchanged"))


def _settings_menu() -> None:
    """The second level of 设置: the settings pages, the same as the admin page's 设置 band (config.PAGES). Leaving it
    lists the changes that await a restart."""
    _provide_options()
    from ..config import PAGES, SCHEMA, settings

    pending: list[str] = []
    ids = list(PAGES)
    while True:
        s = settings()
        console.print(Rule(f"[bold]{i18n.t('cli.setup.settings.title')}[/bold]", align="left", style="cyan"))
        note(i18n.t("cli.setup.settings.intro", file=s.file))
        rows = []
        for i, page in enumerate(ids, 1):
            specs = [spec for spec in SCHEMA.values() if spec.group in PAGES[page].groups]
            edited = sum(1 for spec in specs if spec.key in s.saved)
            needs_restart = sum(1 for spec in specs if spec.restart)
            rows.append((str(i), PAGES[page].label, i18n.t("cli.setup.settings.page_counts", total=len(specs), edited=edited,
                                                            restart=needs_restart)))
        rows.append(("0", i18n.t("cli.setup.common.back"), i18n.t("cli.setup.settings.back_listing") if pending
                     else i18n.t("cli.setup.common.back_up")))
        console.print(menu_table(rows))
        file_only = [spec for spec in SCHEMA.values() if not spec.admin]
        console.print(Panel(
            i18n.t("cli.setup.settings.elsewhere") + "\n"
            + "".join(i18n.t("cli.setup.settings.elsewhere_file", label=spec.label, key=spec.key, file=s.file.name) + "\n"
                      for spec in file_only)
            + i18n.t("cli.setup.settings.elsewhere_admin") + "\n"
            + i18n.t("cli.setup.settings.elsewhere_gpus"),
            title=f"[bold]{i18n.t('cli.setup.table.description')}[/bold]", title_align="left", border_style="dim", box=box.ROUNDED))
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
            err(i18n.t("cli.setup.common.no_such_number"))
            continue
        _settings_page(page, pending)
    _offer_restart(pending)


# ------------------------------------------------------------------ service


def _gpus() -> None:
    from ..client import Lab2ShotError

    from .accounts import local_client

    if not running():
        warn(i18n.t("cli.setup.gpus.not_running"))
        return
    try:
        cards = local_client().admin_cards()["cards"]
    except Lab2ShotError as exc:
        err(escape(str(exc)))
        return
    if not cards:
        warn(i18n.t("cli.setup.gpus.none"))
        return
    table = Table(i18n.t("cli.menu.number"), i18n.t("cli.setup.gpus.gpu"), i18n.t("cli.setup.gpus.memory"),
                  i18n.t("cli.setup.archs.arch"), i18n.t("cli.setup.ext.state"), box=box.SIMPLE_HEAD, header_style="bold")
    for i, c in enumerate(cards, 1):
        table.add_row(str(i), c["model"], f"{c['memory_gb']:g} GB", c["arch"],
                      f"[green]{i18n.t('cli.setup.gpus.on')}[/green]" if c["authorized"] else f"[dim]{i18n.t('cli.setup.gpus.off')}[/dim]")
    console.print(table)
    picked = typer.prompt(i18n.t("cli.setup.gpus.prompt"), default="all")
    if picked.strip() == "all":
        uuids = [c["uuid"] for c in cards]
    elif picked.strip() == "none":
        uuids = []
    else:
        try:
            uuids = [cards[int(n) - 1]["uuid"] for n in picked.split()]
        except (ValueError, IndexError):
            err(i18n.t("cli.setup.common.bad_number_unchanged"))
            return
    try:
        local_client().admin_authorize(uuids)
    except Lab2ShotError as exc:
        err(escape(str(exc)))
        return
    ok(i18n.t("cli.setup.gpus.done", count=len(uuids)))


# ------------------------------------------------------------------ database

REASONS = {"manual": "cli.setup.db.reason.manual", "daily": "cli.setup.db.reason.daily", "created": "cli.setup.db.reason.created",
           "before-update": "cli.setup.db.reason.before_update"}


def _backup_reason(name: str) -> str:
    """What a backup was made for, from its file name (lab2shot-<date>-<time>-<reason>[~n].db, database/__init__.py)."""
    parts = Path(name).stem.split("-", 3)
    reason = parts[3].split("~", 1)[0] if len(parts) == 4 else ""
    if reason.startswith("before-v") and reason[len("before-v"):].isdigit():
        return i18n.t("cli.setup.db.reason.before_version", version=reason[len('before-v'):])
    if reason in REASONS:
        return i18n.t(REASONS[reason])
    return reason or i18n.t("cli.setup.db.reason.unknown")


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
    table.add_column(i18n.t("cli.menu.number"), justify="right", style="bold cyan", no_wrap=True)
    table.add_column(i18n.t("cli.setup.db.time"), no_wrap=True)
    table.add_column(i18n.t("cli.setup.db.size"), justify="right", no_wrap=True)
    table.add_column(i18n.t("cli.setup.db.origin"))
    table.add_column(i18n.t("cli.setup.manual.filename"), style="dim")
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
            warn(i18n.t("cli.setup.db.missing_with_backups", path=folder() / FILE, count=len(found)))
            console.print(_backups_table(found))
        else:
            note(i18n.t("cli.setup.db.not_created", path=settings().work_dir / 'db'))
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
    grid.add_row(i18n.t("cli.setup.db.file"), s["path"])
    grid.add_row(i18n.t("cli.setup.db.size"), i18n.t("cli.setup.db.size_value", size=_size(s["bytes"])))
    grid.add_row(i18n.t("cli.setup.db.version"), i18n.t("cli.setup.db.version_value", version=s['version'], ours=schema.VERSION))
    checked = s["checked"]
    if checked:
        grid.add_row(i18n.t("cli.setup.db.integrity"),
                     (f"[green]{i18n.t('cli.setup.db.intact')}[/green]" if checked.get("ok")
                      else f"[red]{i18n.t('cli.setup.db.damaged', detail=escape(checked.get('detail', '')))}[/red]")
                     + i18n.t("cli.setup.db.checked_at", when=when(checked['at'])))
    else:
        grid.add_row(i18n.t("cli.setup.db.integrity"), f"[dim]{i18n.t('cli.setup.db.unchecked')}[/dim]")
    last = s["last_backup"]
    grid.add_row(i18n.t("cli.setup.db.last_backup"), i18n.t("cli.setup.db.last_backup_value", when=when(last['at']),
                 reason=_backup_reason(last['file']), file=last['file']) if last
                 else f"[yellow]{i18n.t('cli.setup.db.no_backup')}[/yellow]")
    grid.add_row(i18n.t("cli.setup.db.keep"), i18n.t("cli.setup.db.keep_value", count=s['keep']))
    console.print(Panel(grid, title=f"[bold]{i18n.t('cli.setup.db.title')}[/bold]", title_align="left", border_style="cyan",
                        box=box.ROUNDED))
    if found:
        console.print(_backups_table(found))
    else:
        note(i18n.t("cli.setup.db.no_backups"))


def _db_backup() -> None:
    from ..database import db

    if not initialized():
        warn(i18n.t("cli.setup.db.backup_not_created"))
        return
    target = db().backup("manual")
    ok(i18n.t("cli.setup.db.backed_up", path=target, size=_size(target.stat().st_size)))


def _db_check() -> None:
    from ..database import db

    if not initialized():
        warn(i18n.t("cli.setup.db.check_not_created"))
        return
    found = db().check()
    if found["ok"]:
        ok(i18n.t("cli.setup.db.check_ok"))
    else:
        err(i18n.t("cli.setup.db.check_failed", detail=escape(found['detail'])))


def _ensure_stopped(action: str, keep: str) -> bool:
    """For an action that requires the database to itself: when the service is running, say so and offer to stop it
    (the menu's own 停止服务, cli/service.py stop). `action` (what must wait for the stop) and `keep` (the choice not
    to stop) as the menu says them. Returns whether the service is (now) stopped."""
    if not running():
        return True
    warn(i18n.t("cli.setup.db.must_stop", action=action))
    try:
        stop(cancel=(i18n.t("cli.setup.common.back"), keep))
    except NotStopped as exc:
        say(exc.message)
        return False
    except abort_types():
        console.print()
        note(i18n.t("cli.setup.common.cancelled_nothing"))
        return False
    if running():
        err(i18n.t("cli.setup.db.still_running"))
        return False
    return True


def _offer_start() -> None:
    warn(i18n.t("cli.setup.db.start_again"))
    if typer.confirm(i18n.t("cli.setup.db.start_now"), default=False):
        try:
            start(background=True)
        except typer.Exit:
            pass  # the database is restored all the same; start said why the service did not start
    else:
        note(i18n.t("cli.setup.db.start_later"))


def _db_restore() -> None:
    from ..database import FILE, close_all, folder, restore

    found = _backups()
    if not found:
        warn(i18n.t("cli.setup.db.restore_none"))
        return
    console.print(_backups_table(found))
    choice = typer.prompt(i18n.t("cli.setup.db.restore_prompt"), default="0").strip()
    if choice == "0":
        return
    try:
        source = found[int(choice) - 1]
    except (ValueError, IndexError):
        err(i18n.t("cli.setup.db.bad_number_nothing"))
        return
    if not _ensure_stopped(i18n.t("cli.setup.db.restore_action"), i18n.t("cli.setup.db.restore_keep")):
        return
    current = folder() / FILE
    stamp = time.strftime("%Y%m%d-%H%M%S")
    consequences = Table.grid(padding=(0, 2))
    consequences.add_column(style="bold", no_wrap=True)
    consequences.add_column(overflow="fold")
    consequences.add_row(i18n.t("cli.setup.db.restore_source"), i18n.t("cli.setup.db.restore_source_value", name=source.name,
                         time=_backup_time(source), reason=_backup_reason(source.name)))
    consequences.add_row(i18n.t("cli.setup.db.restore_current"), i18n.t("cli.setup.db.restore_current_moved",
                         pattern=current.with_name(f'{FILE}.replaced-' + i18n.t('cli.setup.db.time_placeholder')), example=f"{FILE}.replaced-{stamp}")
                         if current.exists() else i18n.t("cli.setup.db.restore_current_none"))
    consequences.add_row(i18n.t("cli.setup.db.restore_data"), i18n.t("cli.setup.db.restore_data_value"))
    consequences.add_row(i18n.t("cli.setup.db.restore_files"), i18n.t("cli.setup.db.restore_files_value"))
    console.print(Panel(consequences, title=f"[bold]{i18n.t('cli.setup.db.restore_title')}[/bold]", title_align="left",
                        border_style="yellow", box=box.ROUNDED))
    if not typer.confirm(i18n.t("cli.setup.db.restore_confirm"), default=False):
        note(i18n.t("cli.setup.common.cancelled_nothing"))
        return
    again = typer.prompt(i18n.t("cli.setup.db.restore_again"), default="").strip()
    if again != choice:
        note(i18n.t("cli.setup.db.restore_mismatch"))
        return
    close_all()  # this menu's own handle on the database would otherwise hold it (E-DB-INUSE)
    replaced = restore(source.name)
    ok(i18n.t("cli.setup.db.restored", name=source.name))
    ok(i18n.t("cli.setup.db.restored_moved", path=replaced))
    _offer_start()


def _db_upgrade() -> None:
    from ..database import close_all, db, schema

    if not initialized():
        warn(i18n.t("cli.setup.db.upgrade_not_created"))
        return
    if not _ensure_stopped(i18n.t("cli.setup.db.upgrade_action"), i18n.t("cli.setup.db.upgrade_keep")):
        return
    close_all()  # the upgrade needs the database to itself, including this menu's own handle
    d = db(upgrade=True)
    ok(i18n.t("cli.setup.db.upgraded", version=d.version, ours=schema.VERSION, path=d.path))


def _db_claim() -> None:
    from ..config import ROOT, settings
    from ..workdir import claim, owner

    work = settings().work_dir
    if not work.is_dir():
        note(i18n.t("cli.setup.claim.missing", path=work))
        return
    was = owner(work)
    if was == str(ROOT):
        ok(i18n.t("cli.setup.claim.ours", path=work, root=ROOT))
        return
    warn(i18n.t("cli.setup.claim.other", path=work, owner=was or i18n.t("cli.setup.claim.unrecorded")))
    if not typer.confirm(i18n.t("cli.setup.claim.ask", root=ROOT), default=False):
        note(i18n.t("cli.setup.common.cancelled_nothing"))
        return
    claim(work)
    ok(i18n.t("cli.setup.claim.done", path=work, root=ROOT))


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
        grid.add_row(i18n.t("cli.setup.header.service"), i18n.t("cli.setup.status.stopped", address=address()))
        grid.add_row(i18n.t("cli.setup.header.work_dir"), i18n.t("cli.setup.status.work_dir_new", path=s.work_dir))
        grid.add_row(i18n.t("cli.setup.header.admin"), i18n.t("cli.setup.status.admin_new", name=accounts.ADMIN_NAME))
        grid.add_row(i18n.t("cli.setup.header.settings_file"), str(s.file))
        console.print(Panel(grid, title=f"[bold]{i18n.t('cli.setup.status.title')}[/bold]", title_align="left",
                            border_style="cyan", box=box.ROUNDED))
        return
    state = running()
    if state:
        grid.add_row(i18n.t("cli.setup.header.service"), i18n.t("cli.setup.status.running",
                     address=recorded_address() or address(), boot=state.get('boot', '?')))
    else:
        grid.add_row(i18n.t("cli.setup.header.service"), i18n.t("cli.setup.status.stopped", address=address()))
    owner = accounts.admin()
    grid.add_row(i18n.t("cli.setup.header.admin"), i18n.t("cli.setup.header.admin_no_password", name=owner.username)
                 if owner.no_password else i18n.t("cli.setup.header.admin_password_set", name=owner.username))
    grid.add_row(i18n.t("cli.setup.status.passphrase"), f"[green]{i18n.t('cli.setup.status.set')}[/green]" if accounts.passphrase()
                 else f"[yellow]{i18n.t('cli.setup.status.unset')}[/yellow]")
    users = [u for u in accounts.listing() if not u["deleted"]]
    grid.add_row(i18n.t("cli.setup.status.accounts"), i18n.t("cli.setup.status.accounts_value", count=len(users)))
    grid.add_row(i18n.t("cli.setup.ext.extension"), i18n.t("cli.setup.status.extensions_value", count=len(extensions())))
    from ..database import db

    d = db().status()
    last = d["last_backup"]
    grid.add_row(i18n.t("cli.setup.db.title"), i18n.t("cli.setup.status.database_value", version=d['version'], size=_size(d['bytes']),
                 last=when(last['at']) if last else i18n.t("cli.setup.status.none"), count=len(d['backups'])))
    grid.add_row(i18n.t("cli.setup.header.work_dir"), str(s.work_dir))
    grid.add_row(i18n.t("cli.setup.header.settings_file"), i18n.t("cli.setup.status.settings_value", path=s.file, count=len(s.saved)))
    console.print(Panel(grid, title=f"[bold]{i18n.t('cli.setup.status.title')}[/bold]", title_align="left", border_style="cyan",
                        box=box.ROUNDED))


# ------------------------------------------------------------------ the menu


# Before each item runs, four facts are stated: the operation, its purpose, what it writes, and its scope. Every item
# touches the project folder only (.venv, webui, config, work, third_party, downloads); system components (gcc, CUDA,
# Node.js, the display driver) are only checked here, never installed. Apart from uv's package cache (~/.cache/uv), two
# steps write to the user's home folder: the installation of uv in setup.sh (to ~/.local/bin) and hf-login (the Hugging
# Face token).
PROJECT = i18n.t("cli.setup.scope.project")
READ_ONLY = i18n.t("cli.setup.scope.read_only")


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
        i18n.t("cli.setup.wizard.intro", purpose=purpose, count=len(names), steps=" → ".join(STEPS[n].label for n in names)),
        title=f"[bold]{title}[/bold]", title_align="left", border_style="cyan", box=box.ROUNDED))
    results: list[tuple[str, str]] = []
    for i, name in enumerate(names, 1):
        step = STEPS[name]
        console.print(Rule(f"[bold]{i18n.t('cli.setup.wizard.step', i=i, count=len(names), label=step.label)}[/bold]",
                           align="left", style="cyan"))
        _explain(step)
        if not typer.confirm(i18n.t("cli.setup.wizard.ask"), default=True):
            note(i18n.t("cli.setup.wizard.skipped_note"))
            results.append((step.label, f"[dim]{i18n.t('cli.setup.wizard.skipped')}[/dim]"))
            continue
        results.append((step.label, f"[green]{i18n.t('cli.setup.wizard.done')}[/green]" if _attempt(step.run)
                        else f"[yellow]{i18n.t('cli.setup.wizard.unfinished')}[/yellow]"))
    table = Table(i18n.t("cli.setup.wizard.steps"), i18n.t("cli.setup.common.result"), box=box.SIMPLE_HEAD, header_style="bold")
    for label, result in results:
        table.add_row(label, result)
    console.print(Panel(table, title=f"[bold]{i18n.t('cli.setup.wizard.results')}[/bold]", title_align="left",
                        border_style="cyan", box=box.ROUNDED))
    note(i18n.t("cli.setup.wizard.later"))


STEP_LIST: tuple[Step, ...] = (
    Step("wizard", i18n.t("cli.setup.step.wizard.label"), i18n.t("cli.setup.step.wizard.summary"),
         i18n.t("cli.setup.step.wizard.what"), i18n.t("cli.setup.step.wizard.why"), i18n.t("cli.setup.step.wizard.changes"), PROJECT,
         lambda: _wizard(i18n.t("cli.setup.step.wizard.label"), i18n.t("cli.setup.step.wizard.purpose"), WIZARD)),
    Step("build-wizard", i18n.t("cli.setup.step.build_wizard.label"), i18n.t("cli.setup.step.build_wizard.summary"),
         i18n.t("cli.setup.step.build_wizard.what"), i18n.t("cli.setup.step.build_wizard.why"),
         i18n.t("cli.setup.step.build_wizard.changes"), PROJECT, lambda: _wizard(i18n.t("cli.setup.step.build_wizard.label"), i18n.t("cli.setup.step.build_wizard.purpose"), BUILD_SETUP)),
    Step("ext-wizard", i18n.t("cli.setup.step.ext_wizard.label"), i18n.t("cli.setup.step.ext_wizard.summary"),
         i18n.t("cli.setup.step.ext_wizard.what"), i18n.t("cli.setup.step.ext_wizard.why"),
         i18n.t("cli.setup.step.ext_wizard.changes"), PROJECT, lambda: _wizard(i18n.t("cli.setup.step.ext_wizard.label"), i18n.t("cli.setup.step.ext_wizard.purpose"), EXT_WIZARD)),
    # 一键更新
    Step("update", i18n.t("cli.setup.step.update.label"), i18n.t("cli.setup.step.update.summary"),
         i18n.t("cli.setup.step.update.what"),
         i18n.t("cli.setup.step.update.why"),
         i18n.t("cli.setup.step.update.changes"),
         i18n.t("cli.setup.step.update.scope"), one_click_update),
    # 安装与环境
    Step("env", i18n.t("cli.setup.step.env.label"), i18n.t("cli.setup.step.env.summary"),
         i18n.t("cli.setup.step.env.what"), i18n.t("cli.setup.step.env.why"), i18n.t("cli.setup.step.env.changes"), PROJECT, sync_env),
    Step("web", i18n.t("cli.setup.step.web.label"), i18n.t("cli.setup.step.web.summary"),
         i18n.t("cli.setup.step.web.what"), i18n.t("cli.setup.step.web.why"), i18n.t("cli.setup.step.web.changes"), PROJECT, build_webui),
    Step("check", i18n.t("cli.setup.step.check.label"), i18n.t("cli.setup.step.check.summary"),
         i18n.t("cli.setup.step.check.what"), i18n.t("cli.setup.step.check.why"), i18n.t("cli.setup.step.check.changes"), READ_ONLY, _check_environment),
    Step("build-setup", i18n.t("cli.setup.step.build_setup.label"), i18n.t("cli.setup.step.build_setup.summary"),
         i18n.t("cli.setup.step.build_setup.what"), i18n.t("cli.setup.step.build_setup.why"),
         i18n.t("cli.setup.step.build_setup.changes"), PROJECT, _build_setup),
    # 扩展包编译与下载设置（安装与环境 → 扩展包编译与下载设置的三级菜单，及其向导）
    Step("archs", i18n.t("cli.setup.step.archs.label"), i18n.t("cli.setup.step.archs.summary"),
         i18n.t("cli.setup.step.archs.what"), i18n.t("cli.setup.step.archs.why"),
         i18n.t("cli.setup.step.archs.changes"), PROJECT, _choose_archs),
    Step("build", i18n.t("cli.setup.step.build.label"), i18n.t("cli.setup.step.build.summary"),
         i18n.t("cli.setup.step.build.what"), i18n.t("cli.setup.step.build.why"),
         i18n.t("cli.setup.step.build.changes"), PROJECT,
         lambda: _edit_settings(("build.cuda_home", "build.cc", "build.cxx"), i18n.t("cli.setup.step.build.title"))),
    Step("toolchain", i18n.t("cli.setup.step.toolchain.label"), i18n.t("cli.setup.step.toolchain.summary"),
         i18n.t("cli.setup.step.toolchain.what"),
         i18n.t("cli.setup.step.toolchain.why"),
         i18n.t("cli.setup.step.toolchain.changes"), READ_ONLY, lambda: _toolchain()),
    Step("mirrors", i18n.t("cli.setup.step.mirrors.label"), i18n.t("cli.setup.step.mirrors.summary"),
         i18n.t("cli.setup.step.mirrors.what"), i18n.t("cli.setup.step.mirrors.why"),
         i18n.t("cli.setup.step.mirrors.changes"), PROJECT,
         lambda: _edit_settings(("install.mirror_hf", "install.mirror_pypi", "install.mirror_github"), i18n.t("cli.setup.step.mirrors.title"))),
    Step("retries", i18n.t("cli.setup.step.retries.label"), i18n.t("cli.setup.step.retries.summary"),
         i18n.t("cli.setup.step.retries.what"), i18n.t("cli.setup.step.retries.why"),
         i18n.t("cli.setup.step.retries.changes"), PROJECT,
         lambda: _edit_settings(("install.retries", "install.backoff_max"), i18n.t("cli.setup.step.retries.title"))),
    Step("formats", i18n.t("cli.setup.step.formats.label"), i18n.t("cli.setup.step.formats.summary"),
         i18n.t("cli.setup.step.formats.what"),
         i18n.t("cli.setup.step.formats.why"),
         i18n.t("cli.setup.step.formats.changes"), PROJECT, _install_formats),
    Step("hf-login", i18n.t("cli.setup.step.hf_login.label"), i18n.t("cli.setup.step.hf_login.summary"),
         i18n.t("cli.setup.step.hf_login.what"), i18n.t("cli.setup.step.hf_login.why"),
         i18n.t("cli.setup.step.hf_login.changes"), i18n.t("cli.setup.step.hf_login.scope"), _hf_login),
    Step("downloads", i18n.t("cli.setup.step.downloads.label"), i18n.t("cli.setup.step.downloads.summary"),
         i18n.t("cli.setup.step.downloads.what"),
         i18n.t("cli.setup.step.downloads.why"),
         i18n.t("cli.setup.step.downloads.changes"),
         PROJECT, _manual_downloads),
    Step("extensions", i18n.t("cli.setup.step.extensions.label"), i18n.t("cli.setup.step.extensions.summary"),
         i18n.t("cli.setup.step.extensions.what"),
         i18n.t("cli.setup.step.extensions.why"), i18n.t("cli.setup.step.extensions.changes"), PROJECT, _install_extensions),
    # 账号与安全
    Step("password", i18n.t("cli.setup.step.password.label"), i18n.t("cli.setup.step.password.summary"),
         i18n.t("cli.setup.step.password.what"), i18n.t("cli.setup.step.password.why"),
         i18n.t("cli.setup.step.password.changes"), PROJECT, _admin_password),
    Step("passphrase", i18n.t("cli.setup.step.passphrase.label"), i18n.t("cli.setup.step.passphrase.summary"),
         i18n.t("cli.setup.step.passphrase.what"), i18n.t("cli.setup.step.passphrase.why"),
         i18n.t("cli.setup.step.passphrase.changes"), PROJECT, _passphrase),
    # 服务
    Step("start", i18n.t("cli.setup.step.start.label"), i18n.t("cli.setup.step.start.summary"),
         i18n.t("cli.setup.step.start.what"), i18n.t("cli.setup.step.start.why"),
         i18n.t("cli.setup.step.start.changes"), PROJECT, lambda: start(background=True)),
    Step("start-fg", i18n.t("cli.setup.step.start_fg.label"), i18n.t("cli.setup.step.start_fg.summary"),
         i18n.t("cli.setup.step.start_fg.what"), i18n.t("cli.setup.step.start_fg.why"), i18n.t("cli.setup.step.start_fg.changes"), PROJECT, lambda: start(background=False)),
    Step("restart", i18n.t("cli.setup.step.restart.label"), i18n.t("cli.setup.step.restart.summary"),
         i18n.t("cli.setup.step.restart.what"), i18n.t("cli.setup.step.restart.why"), i18n.t("cli.setup.step.restart.changes"), PROJECT, restart),
    Step("stop", i18n.t("cli.setup.step.stop.label"), i18n.t("cli.setup.step.stop.summary"),
         i18n.t("cli.setup.step.stop.what"),
         i18n.t("cli.setup.step.stop.why"), i18n.t("cli.setup.step.stop.changes"), PROJECT, stop),
    Step("gpus", i18n.t("cli.setup.step.gpus.label"), i18n.t("cli.setup.step.gpus.summary"),
         i18n.t("cli.setup.step.gpus.what"), i18n.t("cli.setup.step.gpus.why"), i18n.t("cli.setup.step.gpus.changes"), PROJECT, _gpus),
    # 设置
    Step("settings", i18n.t("cli.setup.step.settings.label"), i18n.t("cli.setup.step.settings.summary"),
         i18n.t("cli.setup.step.settings.what"),
         i18n.t("cli.setup.step.settings.why"), i18n.t("cli.setup.step.settings.changes"), PROJECT, _settings_menu),
    Step("server", i18n.t("cli.setup.step.server.label"), i18n.t("cli.setup.step.server.summary"),
         i18n.t("cli.setup.step.server.what"),
         i18n.t("cli.setup.step.server.why"),
         i18n.t("cli.setup.step.server.changes"), PROJECT,
         _network),
    # 数据库
    Step("db-status", i18n.t("cli.setup.step.db_status.label"), i18n.t("cli.setup.step.db_status.summary"),
         i18n.t("cli.setup.step.db_status.what"), i18n.t("cli.setup.step.db_status.why"), i18n.t("cli.setup.step.db_status.changes"), READ_ONLY, _db_status),
    Step("db-backup", i18n.t("cli.setup.step.db_backup.label"), i18n.t("cli.setup.step.db_backup.summary"),
         i18n.t("cli.setup.step.db_backup.what"), i18n.t("cli.setup.step.db_backup.why"),
         i18n.t("cli.setup.step.db_backup.changes"), PROJECT, _db_backup),
    Step("db-check", i18n.t("cli.setup.step.db_check.label"), i18n.t("cli.setup.step.db_check.summary"),
         i18n.t("cli.setup.step.db_check.what"), i18n.t("cli.setup.step.db_check.why"), i18n.t("cli.setup.step.db_check.changes"), PROJECT, _db_check),
    Step("db-restore", i18n.t("cli.setup.step.db_restore.label"), i18n.t("cli.setup.step.db_restore.summary"),
         i18n.t("cli.setup.step.db_restore.what"), i18n.t("cli.setup.step.db_restore.why"),
         i18n.t("cli.setup.step.db_restore.changes"), PROJECT, _db_restore),
    Step("db-upgrade", i18n.t("cli.setup.step.db_upgrade.label"), i18n.t("cli.setup.step.db_upgrade.summary"),
         i18n.t("cli.setup.step.db_upgrade.what"), i18n.t("cli.setup.step.db_upgrade.why"),
         i18n.t("cli.setup.step.db_upgrade.changes"), PROJECT, _db_upgrade),
    Step("db-claim", i18n.t("cli.setup.step.db_claim.label"), i18n.t("cli.setup.step.db_claim.summary"),
         i18n.t("cli.setup.step.db_claim.what"), i18n.t("cli.setup.step.db_claim.why"),
         i18n.t("cli.setup.step.db_claim.changes"), PROJECT, _db_claim),
    # 状态总览
    Step("status", i18n.t("cli.setup.step.status.label"), i18n.t("cli.setup.step.status.summary"),
         i18n.t("cli.setup.step.status.what"), i18n.t("cli.setup.step.status.why"), i18n.t("cli.setup.step.status.changes"), READ_ONLY, _status),
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
    Group("1", i18n.t("cli.setup.group.g1.label"), i18n.t("cli.setup.group.g1.summary", count=len(WIZARD)), single="wizard"),
    Group("2", i18n.t("cli.setup.group.g2.label"), i18n.t("cli.setup.group.g2.summary", count=len(BUILD_SETUP)), single="build-wizard"),
    Group("3", i18n.t("cli.setup.group.g3.label"), i18n.t("cli.setup.group.g3.summary", count=len(EXT_WIZARD)), single="ext-wizard"),
    Group("4", i18n.t("cli.setup.group.g4.label"), i18n.t("cli.setup.group.g4.summary"), single="update"),
    Group("5", i18n.t("cli.setup.group.g5.label"), i18n.t("cli.setup.group.g5.summary"),
          ("env", "web", "check", "build-setup", "hf-login", "downloads", "extensions")),
    Group("6", i18n.t("cli.setup.group.g6.label"), i18n.t("cli.setup.group.g6.summary"), ("server", "start", "start-fg", "restart", "stop", "gpus")),
    Group("7", i18n.t("cli.setup.group.g7.label"), i18n.t("cli.setup.group.g7.summary"), ("password", "passphrase")),
    Group("8", i18n.t("cli.setup.group.g8.label"), i18n.t("cli.setup.group.g8.summary"), single="settings"),
    Group("9", i18n.t("cli.setup.group.g9.label"), i18n.t("cli.setup.group.g9.summary"),
          ("db-status", "db-check", "db-backup", "db-restore", "db-upgrade", "db-claim")),
    Group("10", i18n.t("cli.setup.group.g10.label"), i18n.t("cli.setup.group.g10.summary"), single="status"),
)


def _explain(step: Step) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")
    grid.add_row(i18n.t("cli.setup.explain.what"), step.what)
    grid.add_row(i18n.t("cli.setup.explain.why"), step.why)
    grid.add_row(i18n.t("cli.setup.explain.changes"), step.changes)
    grid.add_row(i18n.t("cli.setup.explain.scope"), f"[green]{step.scope}[/green]" if step.scope == READ_ONLY else step.scope)
    group = next((g for g in MENU if (step.name if step.name not in BUILD_SETUP else "build-setup") in g.steps), None)
    title = (f"{group.label} → " if group else "") + (f"{STEPS['build-setup'].label} → " if step.name in BUILD_SETUP else "") + step.label
    console.print(Panel(grid, title=f"[bold]{title}[/bold]", title_align="left", border_style="blue", box=box.ROUNDED))


def _group_menu(group: Group) -> None:
    """A second-level menu: the steps of one group; 0 returns to the top level."""
    while True:
        console.print()
        console.print(Rule(f"[bold]{group.key}  {group.label}[/bold]", align="left", style="cyan"))
        rows = [(str(i), STEPS[n].label, STEPS[n].summary) for i, n in enumerate(group.steps, 1)]
        console.print(menu_table(rows + [("0", i18n.t("cli.setup.common.back"), i18n.t("cli.setup.common.back_up"))]))
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
            err(i18n.t("cli.setup.common.no_such_number"))
            continue
        console.print()
        _explain(step)
        _run(step)


HELP_STEPS = i18n.separator().join(s.name for s in STEP_LIST)


@app.command(help=i18n.t("cli.setup.help", steps=HELP_STEPS))
def setup(step: str = typer.Argument("", help=i18n.t("cli.setup.step_help", steps=HELP_STEPS))) -> None:
    """The interactive configuration menu (its --help: cli.setup.help)."""
    if step:
        found = STEPS.get(step)
        if found is None:
            err(i18n.t("cli.setup.no_such_step", step=escape(step), steps=HELP_STEPS))
            raise typer.Exit(2)
        _explain(found)
        if not _run(found, menu=False):
            raise typer.Exit(1)
        return
    while True:
        console.print()
        _header()
        console.print(menu_table([(g.key, g.label, g.summary) for g in MENU]
                                 + [("0", i18n.t("cli.setup.common.quit"), i18n.t("cli.setup.common.quit_hint"))]))
        try:
            choice = pick()
        except abort_types():  # Ctrl-C or end of input at the top level: leave quietly
            console.print()
            return
        if choice == "0":
            return
        group = next((g for g in MENU if g.key == choice), None)
        if group is None:
            err(i18n.t("cli.setup.common.no_such_number"))
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
        note(i18n.t("cli.setup.common.cancelled_back") if menu else i18n.t("cli.setup.common.cancelled"))
        if not menu:
            raise typer.Exit(130)
    except typer.Exit as exc:
        if not menu:
            raise
        if not exc.exit_code:
            return True
        note(i18n.t("cli.setup.common.unfinished_back"))
    except (WorkDirError, DatabaseError, NotStopped) as exc:
        _trouble(exc)
    return False


def _run(step: Step, menu: bool = True) -> bool:
    """One step from the menu (or from `lab2shot setup <step>` when `menu` is False); see _attempt."""
    return _attempt(step.run, menu)
