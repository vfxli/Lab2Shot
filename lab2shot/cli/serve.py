"""`lab2shot ui`: the web page, which is also the service DCC plugins connect to."""

from __future__ import annotations

from typing import Optional

import typer

from .. import i18n
from .base import app, console


@app.command(help=i18n.t("cli.serve.help"))
def ui(
    port: Optional[int] = typer.Option(None, "--port", help=i18n.t("cli.serve.port")),
    host: Optional[str] = typer.Option(None, "--host", help=i18n.t("cli.serve.host")),
    https: Optional[bool] = typer.Option(None, "--https/--http", help=i18n.t("cli.serve.https")),
) -> None:
    from ..config import ROOT, WEBUI_DIST, settings
    from ..extensions import manual
    from ..server import restart, tls

    s = settings()
    flags = {"server.host": (host, f"--host {host}"), "server.port": (port, f"--port {port}"),
             "server.https": (https, "--https" if https else "--http")}
    for key, (value, flag) in flags.items():  # the admin page says these override its settings for this run
        if value is not None:
            s.run_with(key, value, flag)  # the option itself: the admin page says it in its reader's language
    host, port, https = s["server.host"], s["server.port"], s["server.https"]
    inbox = manual.ensure_inbox()  # the one folder for everything downloaded by hand
    console.print(i18n.t("cli.serve.inbox", folder=inbox.relative_to(ROOT)))
    if not WEBUI_DIST.is_dir():
        console.print(i18n.t("cli.serve.not_built"))
    ssl = {}
    if https:
        cert, key = tls.ensure()
        ssl = {"ssl_certfile": str(cert), "ssl_keyfile": str(key)}
        hosts, ips = tls.names()
        console.print(i18n.t("cli.serve.certificate", names=", ".join(hosts + ips)))
        console.print(i18n.t("cli.serve.install_ca", port=port))
    scheme = "https" if https else "http"
    console.print(i18n.t("cli.serve.address", url=f"{scheme}://localhost:{port}")
                  + (i18n.t("cli.serve.listening", host=host) if host != "127.0.0.1" else ""))
    restart.serve(host, port, ssl)
