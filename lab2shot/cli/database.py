"""`lab2shot db`: the database of this work folder — status, backup, check, upgrade, claim, restore."""

from __future__ import annotations

import typer

from .base import console, failed, group as _group, when

group = _group("数据库管理：账号、任务记录、使用统计与结果记录均保存在 work/db/lab2shot.db。")


@group.command("status")
def db_status() -> None:
    """显示数据库的文件、大小、版本、完整性和备份。"""
    from ..database import db

    s = db().status()
    console.print(f"文件      {s['path']}（{s['bytes'] / 1e6:.1f} MB，第 {s['version']} 版）")
    console.print(f"完整性    {s['checked'].get('detail', '')}")
    last = s["last_backup"]
    console.print(f"上次备份  {when(last['at']) + ' ' + last['file'] if last else '无'}")
    for b in s["backups"]:
        console.print(f"  {b['name']}  {b['bytes'] / 1e6:.1f} MB")


@group.command("backup")
def db_backup() -> None:
    """立即创建一份备份（服务运行期间亦可执行）。"""
    from ..database import db

    console.print(f"备份已保存至 {db().backup('manual')}")


@group.command("check")
def db_check() -> None:
    """检查数据库的完整性。发现问题时以状态码 1 结束（一键更新据此判断，cli/update.py）。"""
    from ..database import db

    found = db().check()
    console.print("[green]完好[/green]" if found["ok"] else f"[red]发现问题：{found['detail']}[/red]")
    if not found["ok"]:
        raise typer.Exit(1)


@group.command("upgrade")
def db_upgrade() -> None:
    """将数据库升级至当前 Lab2Shot 版本（升级前自动备份）：必须先停止服务。启动服务时也会自动执行此步骤。"""
    import sqlite3

    from ..database import DatabaseError, db
    from ..workdir import WorkDirError

    try:
        d = db(upgrade=True)
    except (DatabaseError, WorkDirError, sqlite3.Error) as exc:  # a migration that fails in SQLite: said, not traced
        failed(exc)
    console.print(f"[green]数据库版本为第 {d.version} 版[/green]：{d.path}")


@group.command("claim")
def db_claim(yes: bool = typer.Option(False, "--yes", help="不再确认")) -> None:
    """由当前 Lab2Shot（本项目文件夹）接管工作目录：此后其他项目文件夹（分支、测试）中的进程均无法打开该目录。
    执行前必须先停止原先使用该目录的服务。"""
    from ..config import ROOT, settings
    from ..workdir import claim, owner

    work = settings().work_dir
    was = owner(work)
    if not yes and not typer.confirm(f"工作目录 {work} 当前属于 {was or '（未记录）'}，是否改为属于 {ROOT}？"):
        raise typer.Exit(1)
    claim(work)
    console.print(f"[green]工作目录 {work} 现已属于 {ROOT}[/green]")


@group.command("restore")
def db_restore(name: str = typer.Argument(..., help="work/db/backups/ 中的备份文件名（可通过 lab2shot db status 列出）")) -> None:
    """以指定备份替换当前数据库：必须先停止服务。被替换的数据库保留在 work/db/ 中（lab2shot.db.replaced-<时间>），不会删除。"""
    from ..database import DatabaseError, restore

    try:
        replaced = restore(name)
    except DatabaseError as exc:
        failed(exc)
    console.print(f"[green]已恢复 {name}[/green]；原数据库保留在 {replaced}。现在可以启动服务。")
