"""What every command of `lab2shot` shares: the one Typer app, the terminal, and how a command stops on an error;
and the output conventions of the interactive menu (`lab2shot setup`, its one-click update)."""

from __future__ import annotations

import os
import sys
import time
from typing import NoReturn

import typer
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .. import __version__, client, i18n
from ..site import catalog


def language(argv: list[str], environ) -> str:
    """The command line's language: --lang zh|en, else its environment (client.language_of_environment: LAB2SHOT_LANG,
    then LC_ALL / LC_MESSAGES / LANG; Chinese unless one names another language: C, POSIX or nothing said reads
    Chinese, the product's default). Read before any command is declared: their help is in it."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv) and i18n.normal(argv[i + 1]):
            return i18n.normal(argv[i + 1])
        if arg.startswith("--lang=") and i18n.normal(arg.split("=", 1)[1]):
            return i18n.normal(arg.split("=", 1)[1])
    return client.language_of_environment(environ)


LANG = language(sys.argv[1:], os.environ)
i18n.set_process(LANG)  # every command, its help and output, and the server's words it asks for (client.py) in it
client.set_lang(LANG)

catalog.install()  # the node types and what planning needs, for every command (lab2shot/site/catalog.py)

console = Console()
app = typer.Typer(help=i18n.t("cli.app.help"), no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def _main(version: bool = typer.Option(False, "--version", help=i18n.t("cli.option.version")),
          lang: str = typer.Option("", "--lang", metavar="zh|en", help=i18n.t("cli.option.lang"))) -> None:
    # --lang was read before the commands were declared (LANG); here it is only accepted
    if version:
        console.print(f"lab2shot {__version__}")
        raise typer.Exit()


def failed(exc: BaseException | str, code: int = 1) -> NoReturn:
    """Say what went wrong (in red) and end the command with `code`."""
    console.print(f"[red]{exc}[/red]")
    raise typer.Exit(code)


def when(t: float) -> str:
    """A moment as the terminal shows it."""
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t))


def group(help: str) -> typer.Typer:
    """A command group (`lab2shot db ...`); lab2shot/cli/__init__.py puts it in the tree."""
    return typer.Typer(help=help, no_args_is_help=True)


# ------------------------------------------------------------------ output conventions

# One colour and one symbol per kind of message, used by every item of the menu (cli/setup.py, cli/service.py,
# cli/update.py): success, warning, error, and neutral notes.


def ok(text: str) -> None:
    console.print(f"[green]✓[/green] {text}")


def warn(text: str) -> None:
    console.print(f"[yellow]![/yellow] {text}")


def err(text: str) -> None:
    console.print(f"[red]✗[/red] {text}")


def note(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


def mark(state: bool | None) -> str:
    """A table cell for a check: passed, failed, or not applicable."""
    return "" if state is None else ("[green]✓[/green]" if state else "[red]✗[/red]")


def menu_table(rows: list[tuple[str, str, str]]) -> Table:
    """A menu as a table: number, name, description."""
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column(i18n.t("cli.menu.number"), justify="right", style="bold cyan", no_wrap=True)
    table.add_column(i18n.t("cli.menu.name"), no_wrap=True)
    table.add_column(i18n.t("cli.menu.description"))
    for key, name, text in rows:
        table.add_row(key, name, text)
    return table


def pick(default: str = "0") -> str:
    """Read a menu number. Ctrl-C and end of input propagate (typer.Abort / click.Abort) to the caller."""
    return typer.prompt(i18n.t("cli.menu.pick"), default=default).strip().lower()


def abort_types() -> tuple[type[BaseException], ...]:
    import click

    return (KeyboardInterrupt, typer.Abort, click.Abort)  # typer 0.27 has its own Abort, distinct from click's


def say(msg, quiet: bool = False) -> None:
    """A message of the catalogue, in the colour of its kind (a notice as a success, or `quiet` as a neutral note)."""
    {"E": err, "W": warn, "B": err, "P": err}.get(msg.level, note if quiet else ok)(escape(msg.text))
