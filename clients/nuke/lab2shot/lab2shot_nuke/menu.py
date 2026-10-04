"""The Lab2Shot menu in Nuke's menu bar (the framework's items, lab2shot_dcc.ui.menu: new node / open panel / log in /
queue / open in the web page / log folder) and the panel in the Pane menu (panel.py: docked, kept with saved
workspaces). Every item goes through lab2shot_dcc.guard: an error is logged and shown, never raised into Nuke. The
menu is built again in the new language when the user changes it in the panel."""

from __future__ import annotations

import nuke

from lab2shot_dcc import log, paths
from lab2shot_dcc.guard import guarded
from lab2shot_dcc.ui.menu import SEPARATOR, menu_items

MENU = "Lab2Shot"
_plugin = None


def plugin():
    global _plugin
    if _plugin is None:
        from lab2shot_dcc.plugin import Plugin

        from .host import NukeHost

        _plugin = Plugin(NukeHost())
    return _plugin


@guarded(what="open panel")
def open_panel(*_):
    from . import panel

    panel.dock()


def _items() -> None:
    """The menu's items in the plugin's language now (again: the old ones go first)."""
    bar = nuke.menu("Nuke")
    menu = bar.findItem(MENU) or bar.addMenu(MENU)
    menu.clearMenu()
    for item in menu_items(plugin(), open_panel):
        if item is SEPARATOR:
            menu.addSeparator()
        else:
            key, callback = item
            menu.addCommand(paths.text(key), callback)


@guarded(what="menu")
def install() -> None:
    """The menu and the panel's registration. Called by menu.py when Nuke's interface starts."""
    from . import panel

    log.setup("nuke")
    plugin()  # made first: it sets the plugin's language (the user's choice, else Chinese), the menu's words are in it
    plugin().on_language(guarded(_items, what="menu language"))
    _items()
    panel.register()
    log.get().info("Lab2Shot menu loaded")
