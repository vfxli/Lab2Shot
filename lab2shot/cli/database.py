"""`lab2shot db`: the database of this work folder — status, backup, check, upgrade, claim, restore."""

from __future__ import annotations

import typer

from .. import i18n
from .base import console, failed, group as _group, when

group = _group(i18n.t("cli.db.help"))


@group.command("status", help=i18n.t("cli.db.status.help"))
def db_status() -> None:
    from ..database import db

    s = db().status()
    console.print(i18n.t("cli.db.status.file", path=s["path"], mb=s["bytes"] / 1e6, version=s["version"]))
    console.print(i18n.t("cli.db.status.integrity", detail=s["checked"].get("detail", "")))
    last = s["last_backup"]
    console.print(i18n.t("cli.db.status.last_backup", backup=when(last["at"]) + " " + last["file"] if last
                         else i18n.t("cli.db.status.no_backup")))
    for b in s["backups"]:
        console.print(f"  {b['name']}  {b['bytes'] / 1e6:.1f} MB")


@group.command("backup", help=i18n.t("cli.db.backup.help"))
def db_backup() -> None:
    from ..database import db

    console.print(i18n.t("cli.db.backup.done", file=db().backup("manual")))


@group.command("check", help=i18n.t("cli.db.check.help"))
def db_check() -> None:
    from ..database import db

    found = db().check()
    console.print(i18n.t("cli.db.check.ok") if found["ok"] else i18n.t("cli.db.check.bad", detail=found["detail"]))
    if not found["ok"]:
        raise typer.Exit(1)


@group.command("upgrade", help=i18n.t("cli.db.upgrade.help"))
def db_upgrade() -> None:
    import sqlite3

    from ..database import DatabaseError, db
    from ..workdir import WorkDirError

    try:
        d = db(upgrade=True)
    except (DatabaseError, WorkDirError, sqlite3.Error) as exc:  # a migration that fails in SQLite: said, not traced
        failed(exc)
    console.print(i18n.t("cli.db.upgrade.done", version=d.version, path=d.path))


@group.command("claim", help=i18n.t("cli.db.claim.help"))
def db_claim(yes: bool = typer.Option(False, "--yes", help=i18n.t("cli.db.claim.yes"))) -> None:
    from ..config import ROOT, settings
    from ..workdir import claim, owner

    work = settings().work_dir
    was = owner(work)
    if not yes and not typer.confirm(i18n.t("cli.db.claim.confirm", work=work, owner=was or i18n.t("cli.db.claim.unrecorded"),
                                                root=ROOT)):
        raise typer.Exit(1)
    claim(work)
    console.print(i18n.t("cli.db.claim.done", work=work, root=ROOT))


@group.command("restore", help=i18n.t("cli.db.restore.help"))
def db_restore(name: str = typer.Argument(..., help=i18n.t("cli.db.restore.name"))) -> None:
    from ..database import DatabaseError, restore

    try:
        replaced = restore(name)
    except DatabaseError as exc:
        failed(exc)
    console.print(i18n.t("cli.db.restore.done", name=name, replaced=replaced))
