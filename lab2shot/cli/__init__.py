"""Command line: `lab2shot ...` (pyproject.toml's entry point is `lab2shot.cli:app`).

One module per family of commands; this file is the one place that puts them together. The commands of a module
register themselves on `base.app` when it is imported (in this order), its group (`lab2shot db ...`) is added below.

    tools       lab2shot gpus | color | inspect
    serve       lab2shot ui
    setup       lab2shot setup（交互式配置：环境、管理员密码、设置、显卡、启动/停止）；它用到的服务启停在 service.py，
                一键更新在 update.py（二者都不注册命令）
    check       lab2shot check（项目不变量：端口与引文、节点体参数、分类、消息、模板、通道、缓存、地址代次、删除入口、路由、
                网页与计算的两处同一规则、更新说明）
    accounts    lab2shot login | logout, lab2shot admin ...
    jobs        lab2shot templates | cook
    extensions  lab2shot ext list | info | install | preflight | selfcheck | adopt | place | rollback | uninstall
    database    lab2shot db ...
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
