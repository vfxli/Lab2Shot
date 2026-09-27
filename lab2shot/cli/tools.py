"""`lab2shot gpus | color | inspect`: looking at this machine and at inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from .base import app, console, failed


@app.command()
def gpus() -> None:
    """列出显卡及其 UUID，并标明接受任务的显卡（在管理页面 /admin 中授权）。"""
    from ..farm import gpus as farm_gpus

    found = farm_gpus.inventory()
    if not found:
        failed("未找到显卡（nvidia-smi 不可用）")
    allowed = farm_gpus.authorized()
    table = Table("#", "显卡", "UUID", "显存", "占用", "")
    for g in found:
        mark = "[green]接受任务[/green]" if g.uuid in allowed else "[dim]不接受任务[/dim]"
        table.add_row(str(g.index), g.name, g.uuid, f"{g.used_mb} / {g.memory_mb} MiB", f"{g.utilization} %", mark)
    console.print(table)
    if not allowed:
        console.print("[yellow]尚未授权任何显卡：需要显卡的任务将持续排队。请在管理页面 http://<服务器>:端口/admin 中授权。[/yellow]")


@app.command()
def color(input: Optional[str] = typer.Argument(None, help="可选：查看指定文件默认识别的色彩空间")) -> None:
    """显示当前 OCIO 配置，以及文件的默认输入色彩空间。"""
    from ..io.color import load_config

    from ..io.color import working_space

    cfg = load_config()
    console.print(f"OCIO 配置  {cfg.name}\n来源       {cfg.origin}：{cfg.uri}\n工作空间   {working_space(cfg)}（所有输入在读取时统一转换至该空间，输出时再从该空间转换）")
    if input:
        from ..io.sources import open_source

        hint = open_source(input).colorspace_hint if Path(input).exists() or "#" in input else input
        console.print(f"{input} → [bold]{cfg.colorspace_for_file(hint)}[/bold]")


@app.command("inspect")
def inspect_input(input: str = typer.Argument(..., help="序列图、文件夹、任意一帧或视频")) -> None:
    """查看输入的识别结果：帧范围、分辨率、帧率、色彩空间。"""
    from ..io.color import load_config
    from ..io.sequence import format_frame_range
    from ..io.sources import open_source

    src = open_source(input)
    cfg = load_config()
    console.print(f"类型       {({'sequence': '序列图', 'video': '视频', 'still': '单张图片'})[src.kind]}")
    console.print(f"路径       {src.display}")
    console.print(f"帧         {format_frame_range(src.frames)}（共 {len(src.frames)} 帧）")
    console.print(f"分辨率     {src.width} × {src.height}")
    console.print(f"帧率       {src.fps if src.fps else '未知（序列图默认为 24）'}")
    console.print(f"色彩空间   {cfg.colorspace_for_file(src.colorspace_hint)}（按文件格式判断；可在读取节点上修改）")
