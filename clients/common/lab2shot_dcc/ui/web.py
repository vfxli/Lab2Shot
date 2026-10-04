"""「在网页里打开」: the job's own node graph in a web window inside the DCC (Qt WebEngine, which Maya, Houdini and
Nuke all bring), logged in by a one-time ticket (lab2shot_client.Lab2Shot.embed), else the system browser.

- The address holds a ticket good once for 60 seconds: opened at once, never kept, never logged (only its first six
  characters, as the server logs it).
- The window has a profile of its own (off the record): its cookie is the embedded window's, under the plugin's login;
  it never shares or touches the user's browser login.
- HTTPS with the server's own authority (a development server: the authority shipped with the plugin): Chromium in
  the window does not know it, so it asks (certificateError). The certificate is accepted only when it is exactly the
  one this machine's Python verified for that same host and port against the shipped authority, host name included
  (`verified_leaf`) — never "accept anything". A bought certificate never asks.
- Done when any of: the page asks to close the window (windowCloseRequested), its title turns `lab2shot:done`, the
  page's root element says data-lab2shot="done" (asked with runJavaScript every DONE_POLL_MS: never waited for). Then
  `on_done()` runs once (the plugin fetches what the page computed) and the window closes.
- Falls back to the system browser (webbrowser.open, the same address) when Qt WebEngine is missing or the page
  cannot load; then there is no "done" signal: the user fetches with 「取回结果」.
"""

from __future__ import annotations

import socket
import ssl
import urllib.parse
import webbrowser

from .. import log, paths
from ..guard import guarded
from ..qt import QtCore, QtWidgets

DONE_POLL_MS = 1500
# The page's signals to a DCC plugin, defined once on the page: webui/src/editor/dccSignals.ts (DONE_TITLE and the
# <html data-lab2shot="focus|done" data-lab2shot-job="…"> marks). These are the same names; change them there and here
# together.
DONE_TITLE = "lab2shot:done"
MARK_ATTRIBUTE = "lab2shot"  # document.documentElement.dataset.lab2shot
MARK_DONE = "done"


def _ticket_free(url: str) -> str:
    """The address as the log may say it: the ticket cut to its first six characters."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qs(parts.query)
    ticket = (query.get("ticket") or [""])[0]
    return url.replace(ticket, ticket[:6] + "…") if ticket else url


def verified_leaf(url: str, cafile: str | None) -> bytes | None:
    """The server's own certificate (DER) as this machine's Python verifies it — system certificates plus the shipped
    authority, host name checked; None when it does not verify."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https":
        return None
    context = ssl.create_default_context()
    if cafile:
        context.load_verify_locations(cafile=cafile)
    try:
        with socket.create_connection((parts.hostname, parts.port or 443), timeout=10) as raw:
            with context.wrap_socket(raw, server_hostname=parts.hostname) as tls:
                return tls.getpeercert(binary_form=True)
    except (OSError, ssl.SSLError):
        return None


def _engine():
    try:
        from ..qt import QT6

        if QT6:
            from PySide6 import QtWebEngineCore, QtWebEngineWidgets  # type: ignore
        else:  # pragma: no cover - PySide2
            from PySide2 import QtWebEngineWidgets  # type: ignore
            QtWebEngineCore = QtWebEngineWidgets  # noqa: N806 - Qt5 keeps the page classes there
        return QtWebEngineCore, QtWebEngineWidgets
    except ImportError:
        return None


def run_js(page, code: str, callback) -> None:
    """page.runJavaScript with a result callback: Qt 6 takes (code, world id, callback), Qt 5 (code, callback)."""
    try:
        page.runJavaScript(code, 0, callback)
    except TypeError:
        page.runJavaScript(code, callback)


class WebWindow(QtWidgets.QWidget):
    """The embedded page; `done` emitted once, by whichever of the three signals comes first."""

    done = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        core, widgets = _engine()
        self._url, self._finished = url, False
        self._leaf = None  # the certificate verified by Python (asked lazily, once)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = widgets.QWebEngineView(self)
        self.profile = core.QWebEngineProfile(self)  # no name: off the record, its own cookies
        page_class = core.QWebEnginePage if hasattr(core, "QWebEnginePage") else widgets.QWebEnginePage
        self.page = page_class(self.profile, self.view)
        self.view.setPage(self.page)
        layout.addWidget(self.view)
        self.page.windowCloseRequested.connect(guarded(self._finish, what="web window closed"))
        self.page.titleChanged.connect(guarded(self._title, what="web title"))
        self.page.loadFinished.connect(guarded(self._loaded, what="web load"))
        if hasattr(self.page, "certificateError"):
            self.page.certificateError.connect(guarded(self._certificate, what="web certificate"))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(guarded(self._ask_done, what="web state"))
        self.timer.start(DONE_POLL_MS)
        log.get().info("embedded window opens %s", _ticket_free(url))
        self.view.setUrl(QtCore.QUrl(url))

    def _certificate(self, error):
        """Accept only the very certificate Python verified for this host against the shipped authority."""
        url = error.url().toString() if hasattr(error, "url") else self._url
        if self._leaf is None:
            self._leaf = verified_leaf(url, paths.bundled_ca()) or b""
        chain = error.certificateChain() if hasattr(error, "certificateChain") else []
        presented = bytes(chain[0].toDer()) if chain else b""
        if self._leaf and presented == self._leaf:
            error.acceptCertificate()
            return
        log.get().warning("embedded window: certificate refused: %s", error.description() if hasattr(error, "description") else "")
        error.rejectCertificate()
        self.failed.emit(paths.text("dcc.web.certificate"))

    def _loaded(self, ok: bool) -> None:
        if not ok and not self._finished:
            self.failed.emit(paths.text("dcc.web.not_loaded"))

    def _title(self, title: str) -> None:
        if title == DONE_TITLE:
            self._finish()

    def _ask_done(self) -> None:
        if not self._finished:
            run_js(self.page, "document.documentElement.dataset.%s || ''" % MARK_ATTRIBUTE,
                   guarded(lambda state: self._finish() if state == MARK_DONE else None, what="web state"))

    def _finish(self, *_):
        if self._finished:
            return
        self._finished = True
        self.timer.stop()
        self.done.emit()

    def closeEvent(self, event):  # noqa: N802 - Qt's name
        try:
            self.timer.stop()
        finally:
            super().closeEvent(event)


def open_job(host, url: str, on_done, browser, say=None) -> object | None:
    """Open `url` in an embedded window docked by the host (host.show_window), or, when that cannot be, in the system
    browser: `browser()` opens it there with a fresh ticket (the window may have used this one). `on_done()` once the
    page says it is done. Returns the window (None: the system browser)."""
    if _engine() is None:
        log.get().info("no Qt WebEngine: opening the system browser")
        webbrowser.open(url)
        if say:
            say(paths.text("dcc.web.no_engine"))
        return None
    window = WebWindow(url)

    def close():  # after the signal that asked for it has returned: a widget is never deleted inside its own signal
        QtCore.QTimer.singleShot(0, guarded(lambda: host.close_window(window), what="close web window"))

    def finished():
        close()
        on_done()

    def fell_back(why: str):
        close()
        log.get().info("embedded window failed (%s): opening the system browser", why)
        browser()
        if say:
            say(paths.text("dcc.web.fell_back", why=why))

    window.done.connect(guarded(finished, what="web done"))
    window.failed.connect(guarded(fell_back, what="web fallback"))
    host.show_window(window, paths.text("dcc.web.title"), "Lab2ShotWebControl")
    return window
