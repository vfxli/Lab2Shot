"""`lab2shot ext`: extension packages — list, look at one, install."""

from __future__ import annotations

from typing import Optional

import typer
from rich.table import Table

from ..extensions import broken_extensions, extensions, get_extension
from .base import console, failed
from .base import group as _group

group = _group("扩展包管理：列出、查看、安装。")


@group.command("list")
def ext_list() -> None:
    """列出所有扩展包和安装状态。"""
    from ..extensions.status import extension_status

    table = Table("名称", "说明", "状态", "许可证")
    for ext in extensions().values():
        st = extension_status(ext)
        color = "green" if st["ready"] else "dim" if st["label"] == "未安装" else "yellow"
        table.add_row(ext.name, ext.summary, f"[{color}]{st['label']}[/{color}]", ext.license.name)
    for folder, why in broken_extensions().items():
        table.add_row(folder, why, "[red]加载失败[/red]", "")
    console.print(table)


@group.command("info")
def ext_info(name: str = typer.Argument(..., help="扩展包名称")) -> None:
    """查看扩展包详情：来源、许可证、权重、参数。"""
    from ..extensions.status import weight_rows

    ext = get_extension(name)
    console.print(f"[bold]{ext.title}[/bold]  ({ext.name})\n{ext.summary}\n")
    console.print(f"主页      {ext.homepage}")
    console.print(f"源码      {ext.source.url} @ {ext.source.commit[:10]}")
    console.print(f"许可证    {ext.license.name} — {ext.license.summary}\n          {ext.license.url}")
    stack = " · ".join(x for x in (" ".join(ext.env.torch), ext.env.torch_backend) if x)
    console.print(f"环境      Python {ext.env.python}" + (f" · {stack}" if stack else ""))
    console.print(f"位置      {ext.paths.root}\n")
    table = Table("权重", "说明", "状态")
    labels = {"ok": "[green]已就位[/green]", "missing": "未下载", "pending": "[yellow]等待审批[/yellow]", "manual": "[yellow]需手动下载[/yellow]",
              "consent": "[yellow]待同意许可协议[/yellow]"}
    for row in weight_rows(ext):
        table.add_row(row["key"], row["note"], labels[row["status"]])
    console.print(table)


def _checklist(ext) -> bool:
    """Print the installer's checklist before an install; True when nothing blocks."""
    from ..installer.preflight import preflight

    checklist = preflight(ext)
    marks = {"ok": "[green]✓[/green]", "notice": "[cyan]·[/cyan]", "warning": "[yellow]![/yellow]", "blocked": "[magenta]✗[/magenta]"}
    for c in checklist.checks:
        console.print(f"{marks[c.state]} {c.json()['label']}  {c.message.text}", highlight=False)
    return checklist.ready


def _run_install(ext, force: bool = False, only: tuple[str, ...] = (), stop: bool = True) -> str:
    """One extension's install (or just `only`). Returns "" when it worked, else what stopped it. `stop`: end the
    command at once (one extension was named); False: hand the reason back so the rest still run."""
    from ..errors import MessageError
    from ..installer import LABELS, ConsoleSink, install

    try:
        install(ext, ConsoleSink(console, LABELS), force=force, only=only)
    except MessageError as exc:
        console.print(f"[magenta]{exc.code}[/magenta] {exc}", highlight=False)
        if stop:
            raise typer.Exit(1)
        return str(exc)
    return ""


@group.command("install")
def ext_install(
    name: str = typer.Argument(..., help="扩展包名称"),
    force: bool = typer.Option(False, "--force", help="重新执行所有步骤并创建新环境（保留旧环境以便回退）"),
) -> None:
    """安装扩展包：先执行安装前检查（手动下载、许可、Hugging Face 申请、磁盘空间、显卡），全部满足后开始安装。
    依次拉取锁定版本的代码、在旧环境旁创建新环境、下载并校验权重、执行自检，自检通过后启用。
    可重复执行：已完成的步骤将跳过，从失败的步骤继续安装。"""
    ext = get_extension(name)
    console.print(f"[bold]{ext.title}[/bold] 许可证：{ext.license.name} — {ext.license.summary}")
    if not _checklist(ext):
        raise typer.Exit(1)
    _run_install(ext, force=force)


@group.command("preflight")
def ext_preflight(name: str = typer.Argument(..., help="扩展包名称")) -> None:
    """安装前检查：列出缺少的条件及处理方法，不执行任何安装。"""
    if not _checklist(get_extension(name)):
        raise typer.Exit(1)


@group.command("selfcheck")
def ext_selfcheck(
    name: Optional[str] = typer.Argument(None, help="扩展包名称（省略时必须指定 --all）"),
    all_: bool = typer.Option(False, "--all", help="所有已安装、但当前代码版本的自检尚未通过的扩展包"),
) -> None:
    """仅执行自检（在 CPU 上导入包并执行一次小规模计算，无需素材）：自检通过后卡片方可使用。"""
    from ..extensions.status import extension_status
    from ..messages import Msg

    targets = [get_extension(name)] if name else [e for e in extensions().values() if (st := extension_status(e))["installed"]
                                                  and st["selfcheck"] != "passed"] if all_ else []
    if not targets:
        raise typer.BadParameter("必须指定扩展包名称，或使用 --all")
    # one extension that cannot be checked never stops the others: every one is tried, the summary says which did not
    # pass, and the command ends non-zero when any did not
    stuck = []
    for ext in targets:
        console.rule(ext.title)
        if _run_install(ext, only=("selfcheck",), stop=len(targets) == 1):
            stuck.append(ext.title)
    if len(targets) > 1:
        console.print(Msg("I-INSTALL-SELFCHECKALL", passed=len(targets) - len(stuck), failed=len(stuck),
                          names=("：" + "、".join(stuck)) if stuck else "").text)
    if stuck:
        raise typer.Exit(1)


@group.command("adopt")
def ext_adopt(
    name: str = typer.Argument(..., help="扩展包名称"),
    check_only: bool = typer.Option(False, "--check", help="仅核对，不登记"),
) -> None:
    """登记一份手动安装的环境：核对其与声明是否一致（代码 commit、权重、放置到检出目录中的文件、
    Python 版本与锁定的依赖、导入自检），全部一致后写入安装记录。
    不一致时拒绝登记并说明差异，不写入任何记录，不修改任何文件（如需放置文件，请使用 `ext place`）。"""
    from ..errors import MessageError
    from ..installer import ConsoleSink
    from ..installer.adopt import adopt, check

    ext = get_extension(name)
    if check_only:
        problems = check(ext)
        for m in problems:
            console.print(f"[yellow]✗[/yellow] {m.text}", highlight=False)
        if problems:
            raise typer.Exit(1)
        console.print("[green]✓[/green] 与声明一致，可执行 adopt", highlight=False)
        return
    try:
        adopt(ext, ConsoleSink(console))
    except MessageError as exc:
        console.print(str(exc), highlight=False)
        raise typer.Exit(1)


@group.command("place")
def ext_place(name: str = typer.Argument(..., help="扩展包名称")) -> None:
    """将已下载的文件放置到上游代码固定的位置（EnvSpec.places），不执行其他操作：不下载、不编译、不修改环境。
    适用于权重被误删或 `adopt` 报告缺少文件的情况，无需重新安装整个扩展包。"""
    from ..errors import MessageError
    from ..installer import ConsoleSink
    from ..installer.place import place

    try:
        place(get_extension(name), ConsoleSink(console))
    except MessageError as exc:
        console.print(str(exc), highlight=False)
        raise typer.Exit(1)


@group.command("rollback")
def ext_rollback(name: str = typer.Argument(..., help="扩展包名称")) -> None:
    """回退到上一次安装切换之前的环境（仅当该环境仍保留在磁盘上时可用）。"""
    from ..errors import MessageError
    from ..installer import rollback

    try:
        rollback(get_extension(name))
    except MessageError as exc:
        console.print(str(exc), highlight=False)
        raise typer.Exit(1)


@group.command("uninstall")
def ext_uninstall(
    name: str = typer.Argument(..., help="扩展包名称"),
    yes: bool = typer.Option(False, "--yes", help="确认卸载（删除环境、代码、缓存和安装记录；保留模型文件）"),
) -> None:
    """卸载扩展包：删除环境、代码、缓存和安装记录；保留模型文件，重新安装时无需再次下载。"""
    from ..installer import uninstall

    if not yes:
        raise typer.BadParameter("卸载必须使用 --yes 确认")
    uninstall(get_extension(name))
