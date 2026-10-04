"""The Lab2Shot menu in Maya's main window: the framework's items (lab2shot_dcc.ui.menu: 新建节点 / 打开面板 / 登录 /
队列 / 打开网页 / 日志文件夹) hung on Maya's menu bar, and how Maya shows the panel (a dockable workspaceControl)."""

from __future__ import annotations

import maya.cmds as cmds

from lab2shot_dcc import log, paths
from lab2shot_dcc.guard import guarded
from lab2shot_dcc.ui.menu import SEPARATOR, menu_items

MENU = "Lab2ShotMenu"
_plugin = None
_panel = None


def plugin():
    global _plugin
    if _plugin is None:
        from lab2shot_dcc.plugin import Plugin

        from .host import MayaHost

        _plugin = Plugin(MayaHost())
    return _plugin


@guarded(what="open panel")
def open_panel(*_):
    global _panel
    from lab2shot_dcc.ui.shell import PanelShell

    if _panel is not None:
        try:
            _panel.close()
        except RuntimeError:  # already gone with its dock
            pass
    _panel = PanelShell(plugin())
    plugin().host.show_panel(_panel, "Lab2Shot")


@guarded(what="menu language")
def _reinstall():
    if cmds.menu(MENU, exists=True):
        install()


def install() -> None:
    """Add the menu (once; again it replaces itself). Called by userSetup.py when Maya is idle."""
    log.setup("maya")
    plugin()  # made first: it sets the plugin's language (the user's choice, else Maya's), the menu's words are in it
    plugin().on_language(_reinstall)  # the language changed in the panel: the menu again, in it
    if cmds.menu(MENU, exists=True):
        cmds.deleteUI(MENU)
    cmds.menu(MENU, label="Lab2Shot", parent="MayaWindow", tearOff=True)
    for item in menu_items(plugin(), open_panel):
        if item is SEPARATOR:
            cmds.menuItem(divider=True)
        else:
            key, callback = item
            cmds.menuItem(label=paths.text(key), command=callback)
    log.get().info("Lab2Shot menu loaded")
