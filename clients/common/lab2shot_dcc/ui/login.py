"""Connecting: the server's address as the user types it (never filled in for a first connection; the last one the
user connected to is offered again), the account and password. No sign-up here: accounts are made on the web page."""

from __future__ import annotations

from .. import connection, paths
from ..guard import guarded
from ..qt import QtCore, QtWidgets
from . import theme


class LoginDialog(QtWidgets.QDialog):
    def __init__(self, plugin, parent=None, done=None):
        super().__init__(parent)
        self.plugin, self.done_callback = plugin, done
        self.setObjectName("L2SDialog")
        theme.apply(self)
        self.setWindowTitle(paths.text("dcc.login.title"))
        self.setMinimumWidth(420)
        form = QtWidgets.QFormLayout()
        self.address = QtWidgets.QLineEdit(plugin.conn.server)
        self.address.setPlaceholderText(paths.text("dcc.login.address_hint"))
        self.username = QtWidgets.QLineEdit()
        self.password = QtWidgets.QLineEdit()
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)
        form.addRow(paths.text("dcc.login.server"), self.address)
        form.addRow(paths.text("dcc.login.username"), self.username)
        form.addRow(paths.text("dcc.login.password"), self.password)
        self.language = QtWidgets.QComboBox()  # the plugin's language: the DCC's own, or one the user picks
        client = paths.client_module()
        for value, label in ((connection.LANG_AUTO, client.text("dcc.language.host", host=plugin.host.label)),
                             *((lang, client.text(f"lang.{lang}")) for lang in client.LANGS)):
            self.language.addItem(label, value)
        self.language.setCurrentIndex(max(0, self.language.findData(connection.language_setting())))
        self.language.currentIndexChanged.connect(guarded(self._language, what="language"))
        form.addRow(client.text("dcc.settings.language"), self.language)
        note = QtWidgets.QLabel(paths.text("dcc.login.no_account"))
        note.setProperty("l2s", "muted")
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.button = QtWidgets.QPushButton(paths.text("dcc.login.connect"))
        self.button.setObjectName("L2SPrimary")
        self.button.setDefault(True)
        self.button.clicked.connect(self._go)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(self.status)
        layout.addWidget(self.button, alignment=QtCore.Qt.AlignRight)

    def _language(self, *_):
        self.plugin.set_language(self.language.currentData())

    @guarded(what="login")
    def _go(self, *_):
        if not self.address.text().strip() or not self.username.text().strip() or not self.password.text():
            self.status.setText(paths.text("dcc.login.fill_all"))
            return
        self.button.setEnabled(False)
        self.status.setText(paths.text("dcc.login.connecting"))
        self.plugin.login(self.address.text(), self.username.text().strip(), self.password.text(), self._answer)

    def _answer(self, user, error):
        self.button.setEnabled(True)
        if error:
            self.status.setText(error)
            return
        self.password.clear()
        self.accept()
        if self.done_callback is not None:
            self.done_callback(user)
