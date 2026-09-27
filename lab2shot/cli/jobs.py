"""`lab2shot templates | cook`: node graphs as tools — list the templates, cook a graph through the service's queue."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

from .base import app, console, failed


@app.command()
def templates() -> None:
    """列出模板及其对外参数（lab2shot cook 与 DCC 插件使用这些参数名）。"""
    from rich.markup import escape

    from ..engine.templates import exposed_params, templates as all_templates

    for t in all_templates():
        console.print(f"[bold]{t['id']}[/bold]  {t['name']}")
        if t["intro"]:
            console.print(f"  {t['intro']}")
        for x in exposed_params(t["graph"]):
            spec = x["param"] or {}
            line = f"  --set [cyan]{x['name']}[/cyan]=…  {x['label']}"
            if spec.get("options"):
                line += "  [dim]" + escape("可选 " + " | ".join(map(str, spec["options"]))) + "[/dim]"
            if x["value"] not in (None, ""):
                line += "  [dim]" + escape(f"当前 {x['value']!r}") + "[/dim]"
            console.print(line)
        console.print()


# 终端中显示的四个阶段名称（阶段定义见 lab2shot/progress.py；网页中的对应文本见 webui/src/api/progress.ts PHASE_TEXT）
PHASE_WORD = {"queued": "排队中", "loading": "加载模型", "computing": "计算", "fetching": "取回结果"}


def _progress(bar: Progress, task) -> callable:
    """The job's events (farm/queue.py) as the bar and lines of a terminal."""
    labels: dict[str, str] = {}

    def on_event(e: dict) -> None:
        kind = e.get("type")
        name = labels.get(e.get("node") or "", e.get("node") or "")
        if kind == "upload":
            bar.reset(task, total=e["total"], completed=e["done"], description=f"上传 {Path(e['path']).name}")
        elif kind == "queued":
            where = {"heavy": "CPU 队列", "light": "立即计算"}.get(e["lane"], "")
            bar.reset(task, total=None, description=f"{where}排队第 {e['position']} 位" +
                      ("（尚无已授权的显卡）" if e["lane"] == "gpu" and e.get("gpus") == 0 else ""))
        elif kind == "started":
            bar.reset(task, total=None, description=f"开始计算{'（' + e['gpu'] + '）' if e.get('gpu') else ''}")
        elif kind == "node_start":
            labels[e["node"]] = e["label"]
            bar.reset(task, total=None, description=e["label"])
        elif kind == "node_done":
            took = "缓存   " if e["cached"] else f"{e['seconds']:>5} 秒"
            bar.console.print(f"[dim]{took}[/dim] {name}")
        elif kind == "progress":
            # 计算进度只有这一种事件（lab2shot/progress.py）：包含阶段、节点、当前步骤名称，以及
            # `at`（整个任务的完成比例，0–1，**单调不减**；None 表示存在无法估计进度的节点，此时显示无刻度的进度条）
            where = " · ".join(x for x in (name, PHASE_WORD.get(e.get("phase", ""), ""), e.get("note") or "") if x)
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


@app.command()
def cook(
    graph: str = typer.Argument(..., help="节点图 .json 文件或模板名（可通过 lab2shot templates 列出）"),
    set_: Optional[list[str]] = typer.Option(None, "--set", "-s", help="设置参数，格式为 对外参数名=值 或 节点id.参数名=值；可指定多次"),
    node: Optional[list[str]] = typer.Option(None, "--node", "-n", help="仅计算指定节点（输出设置节点的结果由其连接的「输出」交付）；默认计算所有已设置保存位置的「输出」"),
    frames: Optional[str] = typer.Option(None, "--frames", "-f", help="仅计算指定帧段，例如 1001-1020，必须位于输入的帧范围内；默认使用节点图中保存的帧范围，未保存时使用输入的全部帧"),
    force: bool = typer.Option(False, "--force", help="目标节点不使用缓存，重新计算"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="未指定「保存到」的「输出」保存到此文件夹，默认为当前文件夹，按模板名命名"),
    extract: Optional[Path] = typer.Option(None, "--extract", "-x", help="取回 tar 包后解压到此文件夹（每个输出设置对应一个子文件夹）"),
    server: Optional[str] = typer.Option(None, "--server", help="Lab2Shot 服务地址，默认为本机 <server.port>（启用 HTTPS 时使用 https），以管理员身份计算；连接其他服务前须先执行 lab2shot login --server 地址"),
) -> None:
    """计算节点图：以模板和对外参数批量处理镜头。输入文件从本机上传（相同文件仅上传一次），「输出」交付的结果取回到本机：
    保存为一个 tar 包（每个输出设置对应一个子文件夹），或直接写入一个文件夹。任务进入服务的队列，与网页和 DCC 插件的任务
    共同排队并共用缓存。在服务器本机上以管理员身份计算；连接其他服务（--server）时使用经 lab2shot login 登录的账号。

    示例：lab2shot cook sam_3d_body_moving_camera -s input=shots/sh030.mov -s output=shots/sh030/sh030.tar -x shots/sh030/lab2shot
    """
    from ..client import Lab2Shot, Lab2ShotError, parse_value
    from ..engine.graph import GraphError
    from ..engine.templates import load_graph
    from .accounts import local_client

    started = time.time()
    values = {}
    for item in set_ or []:
        key, eq, value = item.partition("=")
        if not eq:
            failed(f"--set 的格式必须为 名称=值：{item}", 2)
        values[key.strip()] = parse_value(value)
    path = Path(graph).expanduser()
    try:
        template = load_graph(path) if path.suffix.lower() == ".json" else graph
    except GraphError as exc:
        failed(exc, 2)
    # on this machine: the machine token (only someone who can read this machine's database gets it), never a password login
    lab = Lab2Shot(server, app="cli") if server else local_client()

    with Progress("{task.description}", BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), console=console) as bar:
        task = bar.add_task("提交", total=None)
        try:
            written = lab.run(template, values, node or None, force, _progress(bar, task), str(out) if out else None, frames,
                              str(extract) if extract else None)
        except KeyboardInterrupt:  # the client has stopped the job
            bar.stop()
            console.print("[yellow]已取消[/yellow]")
            raise typer.Exit(130)
        except Lab2ShotError as exc:
            bar.stop()
            failed(exc)
    for f in written:
        console.print(f"[green]✓[/green] {f}")
    minutes, seconds = divmod(int(time.time() - started), 60)
    console.print(f"已完成，用时 {f'{minutes}分' if minutes else ''}{seconds}秒")
