"""`lab2shot gpus | color | inspect`: looking at this machine and at inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .. import i18n
from .base import app, console, failed


@app.command(help=i18n.t("cli.tools.gpus.help"))
def gpus() -> None:
    from ..farm import gpus as farm_gpus

    found = farm_gpus.inventory()
    if not found:
        failed(i18n.t("cli.tools.gpus.none"))
    allowed = farm_gpus.authorized()
    table = Table("#", i18n.t("cli.tools.gpus.card"), "UUID", i18n.t("cli.tools.gpus.memory"),
                  i18n.t("cli.tools.gpus.utilization"), "")
    for g in found:
        mark = i18n.t("cli.tools.gpus.taking") if g.uuid in allowed else i18n.t("cli.tools.gpus.not_taking")
        table.add_row(str(g.index), g.name, g.uuid, f"{g.used_mb} / {g.memory_mb} MiB", f"{g.utilization} %", mark)
    console.print(table)
    if not allowed:
        console.print(i18n.t("cli.tools.gpus.none_authorized"))


@app.command(help=i18n.t("cli.tools.color.help"))
def color(input: Optional[str] = typer.Argument(None, help=i18n.t("cli.tools.color.input"))) -> None:
    from ..io.color import load_config

    from ..io.color import working_space

    cfg = load_config()
    console.print(i18n.t("cli.tools.color.config", name=cfg.name, origin=cfg.origin, uri=cfg.uri, space=working_space(cfg)))
    if input:
        from ..io.sources import open_source

        hint = open_source(input).colorspace_hint if Path(input).exists() or "#" in input else input
        console.print(f"{input} → [bold]{cfg.colorspace_for_file(hint)}[/bold]")


@app.command("inspect", help=i18n.t("cli.tools.inspect.help"))
def inspect_input(input: str = typer.Argument(..., help=i18n.t("cli.tools.inspect.input"))) -> None:
    from ..io.color import load_config
    from ..io.sequence import format_frame_range
    from ..io.sources import open_source

    src = open_source(input)
    cfg = load_config()
    kinds = {"sequence": "cli.tools.inspect.sequence", "video": "cli.tools.inspect.video", "still": "cli.tools.inspect.still"}
    console.print(i18n.t("cli.tools.inspect.kind", kind=i18n.t(kinds[src.kind])))
    console.print(i18n.t("cli.tools.inspect.path", path=src.display))
    console.print(i18n.t("cli.tools.inspect.frames", range=format_frame_range(src.frames), count=len(src.frames)))
    console.print(i18n.t("cli.tools.inspect.resolution", width=src.width, height=src.height))
    console.print(i18n.t("cli.tools.inspect.fps", fps=src.fps if src.fps else i18n.t("cli.tools.inspect.fps_unknown")))
    console.print(i18n.t("cli.tools.inspect.colorspace", space=cfg.colorspace_for_file(src.colorspace_hint)))
