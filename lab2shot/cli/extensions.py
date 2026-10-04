"""`lab2shot ext`: extension packages — list, look at one, install."""

from __future__ import annotations

from typing import Optional

import typer
from rich.table import Table

from .. import i18n
from ..extensions import broken_extensions, extensions, get_extension
from .base import console
from .base import group as _group

group = _group(i18n.t("cli.ext.help"))


@group.command("list", help=i18n.t("cli.ext.list.help"))
def ext_list() -> None:
    from ..extensions.status import extension_status

    table = Table(i18n.t("cli.ext.list.name"), i18n.t("cli.ext.list.summary"), i18n.t("cli.ext.list.state"),
                  i18n.t("cli.ext.list.license"))
    for ext in extensions().values():
        st = extension_status(ext)
        color = "green" if st["ready"] else "dim" if not st["installed"] else "yellow"
        table.add_row(ext.name, ext.summary, f"[{color}]{st['label']}[/{color}]", ext.license.name)
    for folder, why in broken_extensions().items():
        table.add_row(folder, why, "[red]" + i18n.t("cli.ext.list.broken") + "[/red]", "")
    console.print(table)


@group.command("info", help=i18n.t("cli.ext.info.help"))
def ext_info(name: str = typer.Argument(..., help=i18n.t("cli.ext.name"))) -> None:
    from ..extensions.status import weight_rows

    ext = get_extension(name)
    console.print(f"[bold]{ext.title}[/bold]  ({ext.name})\n{ext.summary}\n")
    console.print(i18n.t("cli.ext.info.homepage", url=ext.homepage))
    console.print(i18n.t("cli.ext.info.source", url=ext.source.url, commit=ext.source.commit[:10]))
    console.print(i18n.t("cli.ext.info.license", name=ext.license.name, summary=ext.license.summary) + f"\n          {ext.license.url}")
    stack = " · ".join(x for x in (" ".join(ext.env.torch), ext.env.torch_backend) if x)
    console.print(i18n.t("cli.ext.info.environment", python=ext.env.python) + (f" · {stack}" if stack else ""))
    console.print(i18n.t("cli.ext.info.location", path=ext.paths.root) + "\n")
    table = Table(i18n.t("cli.ext.info.weight"), i18n.t("cli.ext.info.note"), i18n.t("cli.ext.info.state"))
    labels = {"ok": i18n.t("cli.ext.weight.ok"), "missing": i18n.t("cli.ext.weight.missing"),
              "pending": i18n.t("cli.ext.weight.pending"), "manual": i18n.t("cli.ext.weight.manual"),
              "consent": i18n.t("cli.ext.weight.consent")}
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


def _run_install(ext, revert: bool = False, rebuild: bool = False, only: tuple[str, ...] = (), stop: bool = True) -> str:
    """One extension's install (or just `only`). Returns "" when it worked, else what stopped it. `stop`: end the
    command at once (one extension was named); False: hand the reason back so the rest still run."""
    from ..errors import MessageError
    from ..installer import LABELS, ConsoleSink, install

    try:
        install(ext, ConsoleSink(console, LABELS), revert=revert, rebuild=rebuild, only=only)
    except MessageError as exc:
        console.print(f"[magenta]{exc.code}[/magenta] {exc}", highlight=False)
        if stop:
            raise typer.Exit(1)
        return str(exc)
    return ""


@group.command("install", help=i18n.t("cli.ext.install.help"))
def ext_install(
    name: str = typer.Argument(..., help=i18n.t("cli.ext.name")),
    revert: bool = typer.Option(False, "--revert", help=i18n.t("cli.ext.install.revert")),
    rebuild: bool = typer.Option(False, "--rebuild", help=i18n.t("cli.ext.install.rebuild")),
) -> None:
    ext = get_extension(name)
    console.print(i18n.t("cli.ext.install.license", title=ext.title, name=ext.license.name, summary=ext.license.summary))
    if not _checklist(ext):
        raise typer.Exit(1)
    _run_install(ext, revert=revert, rebuild=rebuild)


@group.command("preflight", help=i18n.t("cli.ext.preflight.help"))
def ext_preflight(name: str = typer.Argument(..., help=i18n.t("cli.ext.name"))) -> None:
    if not _checklist(get_extension(name)):
        raise typer.Exit(1)


@group.command("selfcheck", help=i18n.t("cli.ext.selfcheck.help"))
def ext_selfcheck(
    name: Optional[str] = typer.Argument(None, help=i18n.t("cli.ext.selfcheck.name")),
    all_: bool = typer.Option(False, "--all", help=i18n.t("cli.ext.selfcheck.all")),
) -> None:
    from ..extensions.status import extension_status
    from ..messages import Msg

    targets = [get_extension(name)] if name else [e for e in extensions().values() if (st := extension_status(e))["installed"]
                                                  and st["selfcheck"] != "passed"] if all_ else []
    if not targets:
        raise typer.BadParameter(i18n.t("cli.ext.selfcheck.no_target"))
    # one extension that cannot be checked never stops the others: every one is tried, the summary says which did not
    # pass, and the command ends non-zero when any did not
    stuck = []
    for ext in targets:
        console.rule(ext.title)
        if _run_install(ext, only=("selfcheck",), stop=len(targets) == 1):
            stuck.append(ext.title)
    if len(targets) > 1:
        console.print(Msg("I-INSTALL-SELFCHECKALL", passed=len(targets) - len(stuck), failed=len(stuck),
                          names=i18n.t("cli.ext.selfcheck.names", names=i18n.separator().join(stuck)) if stuck else "").text)
    if stuck:
        raise typer.Exit(1)


@group.command("adopt", help=i18n.t("cli.ext.adopt.help"))
def ext_adopt(
    name: str = typer.Argument(..., help=i18n.t("cli.ext.name")),
    check_only: bool = typer.Option(False, "--check", help=i18n.t("cli.ext.adopt.check")),
) -> None:
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
        console.print("[green]✓[/green] " + i18n.t("cli.ext.adopt.matches"), highlight=False)
        return
    try:
        adopt(ext, ConsoleSink(console))
    except MessageError as exc:
        console.print(str(exc), highlight=False)
        raise typer.Exit(1)


@group.command("place", help=i18n.t("cli.ext.place.help"))
def ext_place(name: str = typer.Argument(..., help=i18n.t("cli.ext.name"))) -> None:
    from ..errors import MessageError
    from ..installer import ConsoleSink
    from ..installer.place import place

    try:
        place(get_extension(name), ConsoleSink(console))
    except MessageError as exc:
        console.print(str(exc), highlight=False)
        raise typer.Exit(1)


@group.command("migrate", help=i18n.t("cli.ext.migrate.help"))
def ext_migrate(
    since: Optional[str] = typer.Option(None, "--since", help=i18n.t("cli.ext.migrate.since")),
    journal: Optional[str] = typer.Option(None, "--journal", help=i18n.t("cli.ext.migrate.journal")),
) -> None:
    from pathlib import Path

    from ..installer.migrate import migrate

    r = migrate(Path(journal) if journal else None, since)
    for name, base in r.moved:
        console.print("[green]✓[/green] " + i18n.t("cli.ext.migrate.moved", name=name, base=base), highlight=False)
    for name in r.split:
        console.print("[green]✓[/green] " + i18n.t("cli.ext.migrate.split", name=name), highlight=False)
    for name, old, new in r.rewritten:
        console.print("[green]✓[/green] " + i18n.t("cli.ext.migrate.rewritten", name=name, old=old, new=new), highlight=False)
    for name in r.outdated:
        console.print("[yellow]![/yellow] " + i18n.t("cli.ext.migrate.outdated", name=name), highlight=False)
    for path in r.left:
        console.print("[cyan]·[/cyan] " + i18n.t("cli.ext.migrate.left", path=path), highlight=False)
    if not (r.moved or r.split or r.rewritten or r.outdated or r.left):
        console.print("[green]✓[/green] " + i18n.t("cli.ext.migrate.none"), highlight=False)


@group.command("rollback", help=i18n.t("cli.ext.rollback.help"))
def ext_rollback(name: str = typer.Argument(..., help=i18n.t("cli.ext.name"))) -> None:
    from ..errors import MessageError
    from ..installer import rollback

    try:
        rollback(get_extension(name))
    except MessageError as exc:
        console.print(str(exc), highlight=False)
        raise typer.Exit(1)


@group.command("uninstall", help=i18n.t("cli.ext.uninstall.help"))
def ext_uninstall(
    name: str = typer.Argument(..., help=i18n.t("cli.ext.name")),
    yes: bool = typer.Option(False, "--yes", help=i18n.t("cli.ext.uninstall.yes")),
) -> None:
    from ..installer import uninstall

    if not yes:
        raise typer.BadParameter(i18n.t("cli.ext.uninstall.confirm"))
    uninstall(get_extension(name))
