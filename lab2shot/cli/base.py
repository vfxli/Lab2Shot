"""What every command of `lab2shot` shares: the one Typer app, the terminal, and how a command stops on an error."""

from __future__ import annotations

import time
from typing import NoReturn

import typer
from rich.console import Console

from .. import __version__, catalog

catalog.install()  # the node types and what planning needs, for every command (lab2shot/catalog.py)

console = Console()
app = typer.Typer(help="Lab2Shot：将论文算法转化为 CG 制作可用的工具。", no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def _main(version: bool = typer.Option(False, "--version", help="显示版本号")) -> None:
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
