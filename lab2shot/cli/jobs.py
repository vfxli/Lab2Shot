"""`lab2shot templates | cook`: node graphs as tools — list the templates, cook a graph through the service's queue."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

from .. import i18n
from .base import app, console, failed


@app.command(help=i18n.t("cli.jobs.templates.help"))
def templates() -> None:
    import json

    from rich.markup import escape

    from ..engine.templates import exposed_params
    from ..site.library import presets as all_templates

    for t in all_templates():
        console.print(f"[bold]{t['id']}[/bold]  {t['name']}")
        if t["intro"]:
            console.print(f"  {t['intro']}")
        shown: list[str] = []  # 已经列出的组（参数界面是一棵树：组名一级一级缩进，组里的参数再缩进一级）
        for x in exposed_params(t["graph"]):
            path = x["group"]
            same = next((i for i, (a, b) in enumerate(zip(shown, path)) if a != b), min(len(shown), len(path)))
            for depth in range(same, len(path)):
                console.print("  " * (depth + 1) + f"[bold]▸ {escape(path[depth])}[/bold]")
            shown = list(path)
            spec = x["param"] or {}
            if spec.get("widget") == "button":  # 按钮（计算、下载）：网页上点，不用 --set 传值
                line = "  " * (len(path) + 1) + i18n.t("cli.jobs.templates.button", label=escape(x['label']))
                if x.get("hide_when"):
                    line += "  [dim]" + escape(f"Hide When {x['hide_when']}") + "[/dim]"
                console.print(line)
                continue
            line = "  " * (len(path) + 1) + f"--set [cyan]{escape(x['name'])}[/cyan]=…  {escape(x['label'])}"
            if x.get("widget") == "menu" and x.get("options"):  # 参数界面给了下拉：列出自己填的值和显示名
                line += "  [dim]" + escape(i18n.t("cli.jobs.templates.options", options=" | ".join(f"{json.dumps(o['value'], ensure_ascii=False)}={o['label']}" for o in x["options"]))) + "[/dim]"
            elif spec.get("options"):
                line += "  [dim]" + escape(i18n.t("cli.jobs.templates.options", options=" | ".join(map(str, spec["options"])))) + "[/dim]"
            if x.get("hide_when"):
                line += "  [dim]" + escape(f"Hide When {x['hide_when']}") + "[/dim]"
            if x.get("show_on_change"):
                line += "  [dim]" + i18n.t("cli.jobs.templates.show_on_change") + "[/dim]"
            if x.get("disable_when"):
                line += "  [dim]" + escape(f"Disable When {x['disable_when']}") + "[/dim]"
            if x["value"] not in (None, ""):
                line += "  [dim]" + escape(i18n.t("cli.jobs.templates.current", value=repr(x['value']))) + "[/dim]"
            console.print(line)
        console.print()


# 终端中显示的四个阶段名称（阶段定义见 lab2shot/progress.py；网页中的对应文本见 webui/src/api/progress.ts PHASE_TEXT）
PHASE_WORD = {"queued": "cli.jobs.phase.queued", "loading": "cli.jobs.phase.loading", "computing": "cli.jobs.phase.computing",
              "fetching": "cli.jobs.phase.fetching"}


def _progress(bar: Progress, task) -> callable:
    """The job's events (farm/queue.py) as the bar and lines of a terminal."""
    labels: dict[str, str] = {}

    def on_event(e: dict) -> None:
        kind = e.get("type")
        name = labels.get(e.get("node") or "", e.get("node") or "")
        if kind == "upload":
            bar.reset(task, total=e["total"], completed=e["done"], description=i18n.t("cli.jobs.progress.upload", name=Path(e['path']).name))
        elif kind == "queued":
            said = (e.get("waiting") or {}).get("text") or ""  # only what needs someone to act (farm/queue.py _told)
            bar.reset(task, total=None, description=i18n.t("cli.jobs.progress.queued_why", position=e['position'], why=said) if said
                      else i18n.t("cli.jobs.progress.queued", position=e['position']))
        elif kind == "started":
            bar.reset(task, total=None, description=i18n.t("cli.jobs.progress.started"))
        elif kind == "node_start":
            labels[e["node"]] = e["label"]
            bar.reset(task, total=None, description=e["label"])
        elif kind == "node_done":
            took = i18n.t("cli.jobs.progress.cached") if e["cached"] else i18n.t("cli.jobs.progress.seconds", seconds=f"{e['seconds']:>5}")
            bar.console.print(f"[dim]{took}[/dim] {name}")
        elif kind == "progress":
            # 计算进度只有这一种事件（lab2shot/progress.py）：包含阶段、节点、当前步骤名称，以及
            # `at`（整个任务的完成比例，0–1，**单调不减**；None 表示存在无法估计进度的节点，此时显示无刻度的进度条）
            where = " · ".join(x for x in (name, (i18n.t(PHASE_WORD[e["phase"]]) if e.get("phase") in PHASE_WORD else ""), e.get("note") or "") if x)
            at = e.get("at")
            bar.reset(task, total=None if at is None else 1000, completed=0 if at is None else int(at * 1000),
                      description=where)
        elif kind == "message" and e.get("level") != "I":
            bar.console.print(f"[yellow]! [{e['code']}] {name + ' · ' if name else ''}{e['text']}[/yellow]")
        elif kind == "error":
            bar.console.print(f"[red]✗ [{e['code']}] {name + ' · ' if name else ''}{e['text']}[/red]")
            if e.get("log") and Path(e["log"]).exists():
                bar.console.print("[dim]" + "\n".join(Path(e["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-20:]) + "[/dim]")

    return on_event


@app.command(help=i18n.t("cli.jobs.cook.help"))
def cook(
    graph: str = typer.Argument(..., help=i18n.t("cli.jobs.cook.graph")),
    set_: Optional[list[str]] = typer.Option(None, "--set", "-s", help=i18n.t("cli.jobs.cook.set")),
    node: Optional[list[str]] = typer.Option(None, "--node", "-n", help=i18n.t("cli.jobs.cook.node")),
    frames: Optional[str] = typer.Option(None, "--frames", "-f", help=i18n.t("cli.jobs.cook.frames")),
    force: bool = typer.Option(False, "--force", help=i18n.t("cli.jobs.cook.force")),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help=i18n.t("cli.jobs.cook.out")),
    extract: Optional[Path] = typer.Option(None, "--extract", "-x", help=i18n.t("cli.jobs.cook.extract")),
    server: Optional[str] = typer.Option(None, "--server", help=i18n.t("cli.jobs.cook.server")),
) -> None:
    from ..client import Lab2Shot, Lab2ShotError, parse_value
    from ..engine.graph import GraphError
    from ..engine.templates import load_graph
    from .accounts import local_client

    started = time.time()
    values = {}
    for item in set_ or []:
        key, eq, value = item.partition("=")
        if not eq:
            failed(i18n.t("cli.jobs.cook.bad_set", item=item), 2)
        values[key.strip()] = parse_value(value)
    path = Path(graph).expanduser()
    try:
        template = load_graph(path) if path.suffix.lower() == ".json" else graph
    except GraphError as exc:
        failed(exc, 2)
    # on this machine: the machine token (only someone who can read this machine's database gets it), never a password login
    lab = Lab2Shot(server, app="cli") if server else local_client()

    with Progress("{task.description}", BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console) as bar:
        task = bar.add_task(i18n.t("cli.jobs.progress.submit"), total=None)
        try:
            written = lab.run(template, values, node or None, force, _progress(bar, task), str(out) if out else None, frames,
                              str(extract) if extract else None)
        except KeyboardInterrupt:  # the client has stopped the job
            bar.stop()
            console.print("[yellow]" + i18n.t("cli.jobs.progress.cancelled") + "[/yellow]")
            raise typer.Exit(130)
        except Lab2ShotError as exc:
            bar.stop()
            failed(exc)
    for f in written:
        console.print(f"[green]✓[/green] {f}")
    minutes, seconds = divmod(int(time.time() - started), 60)
    console.print(i18n.t("cli.jobs.progress.done_minutes", minutes=minutes, seconds=seconds) if minutes
                  else i18n.t("cli.jobs.progress.done", seconds=seconds))
