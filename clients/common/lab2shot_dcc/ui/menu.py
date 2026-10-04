"""The Lab2Shot menu of a DCC's menu bar, the same in every DCC: its items in order and what each does (新建节点 /
打开面板 / 登录 / 队列 / 打开网页 / 日志文件夹). A host only hangs them on its own menu bar, in the plugin's language
now (each item's word: paths.text(key)), and builds it again when the language changes. Every item is guarded: an
error is logged and shown in the panel, never raised into the DCC."""

from __future__ import annotations

from .. import paths
from ..guard import guarded

SEPARATOR = None  # a line between two groups of items


def menu_items(plugin, open_panel) -> list:
    """[(word key, callback) | SEPARATOR] in menu order. `open_panel()`: the host's way of showing the panel (its own
    dock); every callback takes and ignores whatever the DCC's menu passes it."""

    def new_node(*_):
        node = plugin.new_node()
        plugin.host.select([node])  # the new node selected (a host where selecting writes the user's nodes: no-op)
        open_panel()

    def login(*_):
        from ..qt import exec_
        from .login import LoginDialog

        exec_(LoginDialog(plugin, plugin.host.main_window(), done=lambda user: plugin.refresh_tools()))

    def show_queue(*_):
        from ..qt import exec_
        from .queue import QueueDialog

        exec_(QueueDialog(plugin, plugin.host.main_window()))

    def open_web(*_):
        plugin.open_in_web()

    def open_logs(*_):
        from .widgets import open_folder

        open_folder(paths.user_dir())

    return [
        ("dcc.menu.new_node", guarded(new_node, what="new node")),
        ("dcc.menu.open_panel", guarded(lambda *_: open_panel(), what="open panel")),
        SEPARATOR,
        ("dcc.menu.login", guarded(login, what="login")),
        ("dcc.menu.queue", guarded(show_queue, what="queue")),
        ("dcc.menu.open_web", guarded(open_web, what="open web")),
        SEPARATOR,
        ("dcc.menu.log_folder", guarded(open_logs, what="log folder")),
    ]
