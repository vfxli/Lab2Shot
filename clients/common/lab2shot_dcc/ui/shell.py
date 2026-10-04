"""The Lab2Shot panel, the same in every DCC with Qt (设计_DCC新面板.md; 设计_DCC插件_Maya.md §14): a top bar (the
breadcrumb 「← 全部工具 › 节点 · 工具」 instead of a node list, the connection, a menu), a banner when something stops
the work (not logged in, the server away, the tool list failed), and two pages — the home page (home.py) and the tool
page of one Lab2Shot node (tool_page.py).

Which node: the Lab2Shot node selected in the DCC, switched to by itself (a tool page when it has a tool, else the home
page with its name in the breadcrumb); or one of 「本场景的任务」; or the one a card made. A card on the home page gives
its tool to the node shown when that node has none, else makes a new node (one task, one node: a node's versions never
end up under another tool); a card of 「适合当前选中」 also binds what is selected to every input that takes it.

Every slot is guarded (an error is a sentence in the panel and a line in the log); nothing waits: the server is asked
on background threads, the panel reads progress and the selection on its own timer (REFRESH_MS), throttled. The
language can change while it is open: the whole panel is built again in it (every word is taken when it is built),
the page, the node and the home page's search and tab kept.
"""

from __future__ import annotations

from .. import connection, guard, log, paths, view_model
from ..guard import guarded
from ..paths import text
from ..qt import QtCore, QtWidgets, exec_
from . import theme, widgets
from .home import HomePage
from .login import LoginDialog
from .tool_page import ToolPage

REFRESH_MS = 300
TASKS_EVERY = 5  # ticks between two looks at the scene's Lab2Shot nodes (added, removed, renamed)


class PanelShell(QtWidgets.QWidget):
    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SRoot")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.plugin, self.host = plugin, plugin.host
        theme.apply(self)
        self.node: str | None = None
        self.page = "home"
        self._dirty_selection = False
        self._ticks = 0
        self._search, self._tab = "", ""
        self._root = QtWidgets.QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)
        self._build()
        guard.on_error(self._error)
        plugin.on_language(self._language)
        self._selection_job = self.host.on_selection_changed(guarded(self._selection_changed))
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(guarded(self._tick, what="panel refresh"))
        self.timer.start(REFRESH_MS)
        picked = self.host.selected_node()
        if picked:
            self.open_node(picked)
        else:
            self.go_home()
        if self.plugin.conn.has_token() and not self.plugin.tools and not self.plugin.loading:
            self.refresh_tools()

    # ---- building

    def _build(self) -> None:
        bar = QtWidgets.QFrame()
        bar.setObjectName("L2STopBar")
        bar.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(8, 6, 6, 6)
        row.setSpacing(4)
        self.back = QtWidgets.QToolButton()
        self.back.setObjectName("L2SBack")
        self.back.setText(text("dcc.ui.top.all_tools"))
        self.back.setIcon(theme.icon("chevron", "text-2", rotate=90))
        self.back.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.back.setCursor(QtCore.Qt.PointingHandCursor)
        self.back.setAutoRaise(True)
        self.back.setToolTip(text("dcc.ui.top.all_tools_tip"))
        self.back.clicked.connect(guarded(lambda *_: self.go_home()))
        row.addWidget(self.back)
        self.crumb = widgets.ElidedLabel("", "crumb")
        self.crumb.setObjectName("L2SCrumb")
        row.addWidget(self.crumb, 1)
        self.server_dot = QtWidgets.QLabel()
        row.addWidget(self.server_dot)
        self.server_label = widgets.ElidedLabel("", "faint")
        self.server_label.setObjectName("L2SServer")
        self.server_label.setMinimumWidth(100)
        self.server_label.setMaximumWidth(170)
        row.addWidget(self.server_label)
        more = widgets.tool_button("more", text("dcc.ui.top.menu"))
        more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        more.setMenu(self._menu(more))
        row.addWidget(more)
        self._root.addWidget(bar)

        self.banner = widgets.Banner()
        holder = QtWidgets.QWidget()
        h = QtWidgets.QVBoxLayout(holder)
        h.setContentsMargins(10, 8, 10, 0)
        h.addWidget(self.banner)
        self.banner_holder = holder
        self._root.addWidget(holder)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.setObjectName("L2SStack")
        self.home = HomePage(self.plugin)
        self.home.search_text, self.home.tab = self._search, self._tab or self.home.tab
        if self._search:
            self.home.search.setText(self._search)
        self.home.tool_chosen.connect(guarded(self.tool_chosen))
        self.home.node_chosen.connect(guarded(self.open_node))
        self.tool_page = ToolPage(self.plugin, self)
        self.stack.addWidget(self.home)
        self.stack.addWidget(self.tool_page)
        self._root.addWidget(self.stack, 1)

    def _menu(self, owner) -> QtWidgets.QMenu:
        menu = QtWidgets.QMenu(owner)
        menu.setObjectName("L2SMenu")
        menu.addAction(text("dcc.ui.menu.login"), guarded(lambda *_: self._login()))
        menu.addAction(text("dcc.ui.menu.refresh"), guarded(lambda *_: self.refresh_tools()))
        menu.addSeparator()
        menu.addAction(text("dcc.ui.menu.new_node"), guarded(lambda *_: self._new_node()))
        menu.addAction(text("dcc.ui.menu.local_graph"), guarded(lambda *_: self._add_local()))
        menu.addAction(text("dcc.ui.menu.queue"), guarded(lambda *_: self._queue()))
        langs = menu.addMenu(text("dcc.ui.menu.language"))
        langs.setObjectName("L2SLanguages")  # each language named in itself (lang.*): the one place another script shows
        client = paths.client_module()
        chosen = connection.language_setting()
        for value, label in ((connection.LANG_AUTO, text("dcc.language.host", host=self.host.label)),
                             *((lang, text(f"lang.{lang}")) for lang in client.LANGS)):
            act = langs.addAction(label)
            act.setCheckable(True)
            act.setChecked(value == chosen)
            act.triggered.connect(guarded(lambda *_, v=value: self.plugin.set_language(v)))
        menu.addSeparator()
        menu.addAction(text("dcc.ui.menu.log_folder"), guarded(lambda *_: widgets.open_folder(paths.user_dir())))
        return menu

    # ---- pages

    def go_home(self, keep_node: bool = True) -> None:
        self.page = "home"
        if not keep_node:
            self.node = None
        self.stack.setCurrentWidget(self.home)
        self.home.refresh(self.host.selection_types(), self.node)
        self._refresh_top()

    def open_node(self, node: str | None) -> None:
        self.node = node
        tool = self.plugin.tool(self.host.load(node)) if node else None
        named = bool(node and (self.host.load(node).get("tool") or {}).get("id"))
        if node and (tool is not None or named):
            self.page = "tool"
            self.tool_page.show_node(node)
            self.stack.setCurrentWidget(self.tool_page)
            self._refresh_top()
        else:
            self.go_home()

    def tool_chosen(self, tool_id: str, suggested: bool) -> None:
        tool = next((t for t in self.plugin.tools if t.get("id") == tool_id), None)
        if tool is None:
            return
        scene_nodes = self.host.nodes()
        shown = self.host.load(self.node) if self.node and self.node in scene_nodes else {}
        if view_model.choose_node(self.node, scene_nodes, shown) == view_model.REUSE:
            node = self.node
        else:
            node = self.plugin.new_node()  # one task, one node: a node that has a tool keeps it
        self.plugin.set_tool(node, tool)
        bound = []
        if suggested:  # 「适合当前选中」: what is selected goes into the inputs that take it (view_model.auto_bind)
            for item in view_model.auto_bind(tool, self.host.selection_types(), self.host.exports):
                if not self.plugin.bind_selected(node, item):
                    bound.append(item.get("label") or item["param"])
        self.open_node(node)
        if bound:
            self.tool_page.say(text("dcc.ui.input.bound", label=text("list.sep").join(bound)))

    # ---- the top bar and the banner

    def _refresh_top(self) -> None:
        on_tool = self.page == "tool"
        self.back.setVisible(on_tool or bool(self.node))
        parts = []
        if self.node:
            parts.append(self.host.node_name(self.node))
            state = self.host.load(self.node)
            name = ((self.plugin.tool(state) or state.get("tool") or {}).get("name")) if on_tool else ""
            if name:
                parts.append(str(name))
        crumb = " · ".join(parts)
        self.crumb.set_text(("› " + crumb) if crumb and self.back.isVisible() else (crumb or "Lab2Shot"))
        conn = self.plugin.conn
        state, said = view_model.connection_state(conn, self.plugin)
        self.server_dot.setPixmap(theme.dot(theme.token(state), 7, theme.ratio(self)))
        self.server_dot.setToolTip(conn.server)
        self.server_label.set_text(said)
        self._refresh_banner()

    def _refresh_banner(self) -> None:
        conn = self.plugin.conn
        if not conn.server:
            self.banner.say(text("dcc.ui.banner.not_connected"), text("dcc.ui.banner.login"), guarded(self._login))
        elif not conn.has_token():
            self.banner.say(text("dcc.ui.banner.not_logged_in", server=conn.server), text("dcc.ui.banner.login"),
                            guarded(self._login))
        elif self.plugin.tools_error and not self.plugin.loading:
            self.banner.say(text("dcc.ui.banner.tools_failed", error=self.plugin.tools_error),
                            text("dcc.ui.banner.retry"), guarded(self.refresh_tools), kind="error")
        else:
            self.banner.say("")
        self.banner_holder.setVisible(self.banner.isVisibleTo(self.banner_holder))

    # ---- the timer: progress, the selection (throttled), the scene's nodes now and then

    def _tick(self) -> None:
        self._ticks += 1
        if self._dirty_selection:
            self._dirty_selection = False
            picked = self.host.selected_node()
            # a Lab2Shot node selected now: shown (also the one shown last, when the home page is up)
            if picked and (picked != self.node or self.page != "tool"):
                self.open_node(picked)
            else:
                types = self.host.selection_types()
                if self.page == "home":
                    self.home.refresh_selection(types)
                else:
                    self.tool_page.selection_changed(types)
        if self.page == "tool":
            self.tool_page.tick()
        if self._ticks % TASKS_EVERY == 0:
            if self.node and self.node not in self.host.nodes():  # deleted, or another scene opened
                self.node = None
                self.go_home()
            elif self.page == "home":
                self.home.refresh_tasks(self.node)
            elif self.node:
                self._refresh_top()

    def _selection_changed(self) -> None:
        self._dirty_selection = True

    def _error(self, said: str) -> None:
        try:
            if self.page == "tool":
                self.tool_page.say(text("dcc.ui.error", error=said, log=log.file()), error=True)
            else:
                self.banner.say(text("dcc.ui.error", error=said, log=log.file()), kind="error")
                self.banner_holder.show()
        except RuntimeError:  # the panel is gone
            guard.off_error(self._error)

    # ---- the tool list, the language

    def refresh_tools(self) -> None:
        self.plugin.refresh_tools(guarded(self._tools_loaded))
        self._refresh_top()
        if self.page == "home":
            self.home.refresh()

    def _tools_loaded(self, error: str) -> None:
        try:
            self._refresh_top()
            if self.page == "home":
                self.home.refresh()
            else:
                self.tool_page.rebuild()
        except RuntimeError:  # the panel is gone
            pass

    def _language(self) -> None:
        """The language changed: the panel is built again in it (every word is taken when it is built), what was
        shown kept."""
        try:
            self._search, self._tab = self.home.search_text, self.home.tab
        except RuntimeError:  # the panel is gone
            self.plugin.off_language(self._language)
            return
        widgets.clear(self._root)
        self._build()
        if self.page == "tool" and self.node:
            self.open_node(self.node)
        else:
            self.go_home()

    # ---- the menu's

    def _login(self, *_):
        dialog = LoginDialog(self.plugin, self, done=lambda user: self.refresh_tools())
        exec_(dialog)
        self._refresh_top()

    def _new_node(self) -> None:
        self.node = self.plugin.new_node()
        self.go_home()

    def _add_local(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(self, text("dcc.ui.menu.local_graph_title"), "",
                                                              text("dcc.ui.menu.local_graph_filter"))
        if path:
            self.plugin.add_local(path, guarded(self._tools_loaded))

    def _queue(self) -> None:
        from .queue import QueueDialog

        exec_(QueueDialog(self.plugin, self))

    def release(self) -> None:
        """Stop listening — the refresh timer, the DCC's selection, the errors, the language — so nothing calls a panel
        that is gone: on closing, and from a DCC whose dock can go without closing the panel (Nuke's pane). Safe to
        call again, and on a panel whose Qt side is already deleted."""
        try:
            self.timer.stop()
        except RuntimeError:  # deleted with the panel
            pass
        job, self._selection_job = self._selection_job, None
        self.host.off_selection_changed(job)
        guard.off_error(self._error)
        self.plugin.off_language(self._language)

    def closeEvent(self, event):  # noqa: N802 - Qt's name
        try:
            self.release()
        finally:
            super().closeEvent(event)
