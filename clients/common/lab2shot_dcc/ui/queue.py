"""The queue: this account's jobs on the server and how busy it is (asked on a background thread)."""

from __future__ import annotations

from ..guard import guarded
from ..paths import text
from ..qt import QtWidgets
from . import theme

STATES = {"queued": "dcc.queue.state.queued", "running": "dcc.queue.state.running", "done": "dcc.queue.state.done",
          "failed": "dcc.queue.state.failed", "cancelled": "dcc.queue.state.cancelled"}  # a job's state -> its word's key


def _state(state) -> str:
    return text(STATES[state]) if state in STATES else str(state or "")


class QueueDialog(QtWidgets.QDialog):
    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.plugin = plugin
        self.setObjectName("L2SDialog")
        theme.apply(self)
        self.setWindowTitle(text("dcc.queue.title"))
        self.resize(520, 360)
        layout = QtWidgets.QVBoxLayout(self)
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list)
        again = QtWidgets.QPushButton(text("dcc.queue.refresh"))
        again.setObjectName("L2SButton")
        again.clicked.connect(guarded(self.refresh))
        layout.addWidget(again)
        self.refresh()

    def refresh(self, *_):
        self.list.clear()
        self.list.addItem(text("dcc.queue.asking"))
        self.plugin._background("queue", lambda: self.plugin.conn.session().queue(), self._show)

    def _show(self, got, error):
        self.list.clear()
        if error:
            self.list.addItem(error)
            return
        for j in got.get("jobs") or []:
            where = text("dcc.queue.position", position=j["position"]) if j.get("position") else _state(j.get("state"))
            self.list.addItem(f"{where}  {j.get('title') or text('dcc.queue.others')}  {j.get('id', '')}")
        for j in (got.get("history") or [])[:30]:
            self.list.addItem(f"{_state(j.get('state'))}  {j.get('title', '')}  {j.get('id', '')}")
        if not self.list.count():
            self.list.addItem(text("dcc.queue.empty"))
