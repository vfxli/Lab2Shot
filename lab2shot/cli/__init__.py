"""Command line: `lab2shot ...` (pyproject.toml's entry point is `lab2shot.cli:app`).

One module per family of commands; this file is the one place that puts them together. The commands of a module
register themselves on `base.app` when it is imported (in this order), its group (`lab2shot db ...`) is added below.

    extensions  lab2shot ext list | info | install
    tools       lab2shot gpus | color | inspect
    serve       lab2shot ui
    setup       lab2shot setup（交互式配置：环境、管理员密码、设置、显卡、启动/停止）
    check       lab2shot check（项目不变量：端口与引文、分类、消息、模板、通道、缓存、删除入口、路由）
    accounts    lab2shot login | logout, lab2shot admin ...
    database    lab2shot db ...
    jobs        lab2shot templates | cook
"""

from __future__ import annotations

# isort: off  (the order of these imports is the order `lab2shot --help` lists the commands in)
from .base import app
from . import tools, serve, setup, check, accounts, jobs  # noqa: F401 (commands register on import)
from . import extensions, database
# isort: on

app.add_typer(extensions.group, name="ext")
app.add_typer(database.group, name="db")
app.add_typer(accounts.admin, name="admin")

__all__ = ["app"]
