"""The Lab2Shot panel docked in a Nuke pane: nukescripts.panels.registerWidgetAsPanel registers it in the Pane menu
and for workspaces (a saved workspace brings it back); Nuke makes the pane's widget by calling `Holder()` with no
arguments (the class is named to Nuke as a string: WIDGET). A Holder holds the framework's panel
(lab2shot_dcc.ui.shell.PanelShell): one handed over by the host (show_window) or a new one of the plugin's.

Only one Lab2Shot pane at a time: opening it again raises the one there.
"""

from __future__ import annotations

import weakref

from lab2shot_dcc import log
from lab2shot_dcc.qt import QtWidgets

PANEL_ID = "com.lab2shot.Panel"
NAME = "Lab2Shot"  # the product's name in the Pane menu and on the tab (a name, not a word to translate)
WIDGET = "__import__('lab2shot_nuke.panel', fromlist=['Holder']).Holder"
_handed: list = []  # a panel widget the host was asked to show, waiting for the Holder Nuke is about to make
_current = None  # weak reference to the Holder in a pane now


class Holder(QtWidgets.QWidget):
    def __init__(self, parent=None):
        global _current
        super().__init__(parent)
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        if _handed:
            shell = _handed.pop()
        else:
            from lab2shot_dcc.ui.shell import PanelShell

            from . import menu

            shell = PanelShell(menu.plugin())
        box.addWidget(shell)
        _current = weakref.ref(self)
        # the pane closed (the widget goes with it, closeEvent or not): the panel in it stops listening
        holding = {"shell": shell}
        self.holding = holding
        self.destroyed.connect(lambda *_: _let_go(holding["shell"]))

    @property
    def shell(self):
        return self.holding["shell"]

    @shell.setter
    def shell(self, widget):
        self.holding["shell"] = widget


def _let_go(shell) -> None:
    """A panel whose pane went without closing it stops listening (PanelShell.release, as its closeEvent does)."""
    try:
        shell.release()
    except Exception:  # noqa: BLE001 - going away
        log.get().warning("closing the Lab2Shot pane: listeners not removed")


def current():
    holder = _current() if _current is not None else None
    try:
        if holder is not None and holder.isVisible():
            return holder
    except RuntimeError:  # deleted with its pane
        pass
    return None


def register() -> None:
    """The Pane menu's Lab2Shot, and the panel for saved workspaces."""
    from nukescripts import panels

    panels.registerWidgetAsPanel(WIDGET, NAME, PANEL_ID)


def dock(widget=None) -> None:
    """Show the panel in a pane (beside the Properties). A pane there already: raised, and when `widget` (a
    PanelShell the host was handed) is given it takes the old panel's place; else a new pane tab holding `widget`
    or a new panel."""
    import nuke
    from nukescripts import panels

    holder = current()
    if holder is not None:
        if widget is not None and widget is not holder.shell:
            old = holder.shell
            holder.layout().replaceWidget(old, widget)
            holder.shell = widget
            try:
                old.close()
                old.deleteLater()
            except RuntimeError:
                pass
        holder.window().raise_()
        return
    if widget is not None:
        _handed.append(widget)
    made = panels.registerWidgetAsPanel(WIDGET, NAME, PANEL_ID, True)
    made.addToPane(nuke.getPaneFor("Properties.1") or nuke.getPaneFor("DAG.1"))
