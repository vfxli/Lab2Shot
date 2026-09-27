"""`lab2shot ui`: the web page, which is also the service DCC plugins connect to."""

from __future__ import annotations

from typing import Optional

import typer

from .base import app, console


@app.command()
def ui(
    port: Optional[int] = typer.Option(None, "--port", help="端口，仅对本次启动有效；默认使用管理页面「设置」中的端口"),
    host: Optional[str] = typer.Option(None, "--host", help="监听地址，仅对本次启动有效；默认按管理页面「设置」中的访问范围。0.0.0.0 允许局域网内的计算机和 DCC 插件连接"),
    https: Optional[bool] = typer.Option(None, "--https/--http", help="使用 HTTPS，仅对本次启动有效；默认按管理页面「设置」。局域网内的浏览器若要将结果直接保存到用户的文件夹、将节点图保存回原文件，必须使用 HTTPS；使用 HTTP 时保存操作改为下载"),
) -> None:
    """启动网页界面（同时作为 DCC 插件连接的服务）。可在管理页面 /admin 中修改设置、重启此服务。"""
    from ..config import ROOT, WEBUI_DIST, settings
    from ..extensions import manual
    from ..server import restart, tls

    s = settings()
    flags = {"server.host": (host, f"--host {host}"), "server.port": (port, f"--port {port}"),
             "server.https": (https, "--https" if https else "--http")}
    for key, (value, flag) in flags.items():  # the admin page says these override its settings for this run
        if value is not None:
            s.run_with(key, value, f"启动命令 {flag}")
    host, port, https = s["server.host"], s["server.port"], s["server.https"]
    inbox = manual.ensure_inbox()  # the one folder for everything downloaded by hand
    console.print(f"手动下载的文件请放入 {inbox.relative_to(ROOT)}（位于 Lab2Shot 项目文件夹下）")
    if not WEBUI_DIST.is_dir():
        console.print("[yellow]界面尚未构建，请执行：cd webui && npm install && npm run build[/yellow]")
    ssl = {}
    if https:
        cert, key = tls.ensure()
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        hosts, ips = tls.names()
        console.print(f"HTTPS 证书适用于 {', '.join(hosts + ips)}")
        console.print(f"每台使用的计算机须安装一次证书：在浏览器中打开 https://<本机>:{port}/api/tls/ca.pem 下载，并安装到系统的「受信任的根证书」")
    scheme = "https" if https else "http"
    console.print(f"Lab2Shot 界面 → [bold]{scheme}://localhost:{port}[/bold]" + (f"（监听 {host}）" if host != "127.0.0.1" else ""))
    restart.serve(host, port, ssl)
