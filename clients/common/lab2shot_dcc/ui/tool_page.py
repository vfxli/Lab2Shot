"""The tool page of one Lab2Shot node: the tool's name and what it does, then three steps from the top down —
① 输入 (a card per input: what is bound, a sequence's first and last frame and how many; 「用选中的」「选文件」「清除」),
② 设置 (common parameters, 「高级」 folded), ③ 计算 (one big button that becomes 「取消」, the progress and the run's
own words right under it) — and the result versions in a row (设计_DCC新面板.md §1.2).

Everything it asks of the scene goes through the plugin and its host; it knows no tool and no DCC."""

from __future__ import annotations

import os
import threading

from .. import connection, guard, log, view_model
from ..guard import guarded
from ..paths import text
from ..qt import QtCore, QtWidgets
from . import theme, widgets
from .params import ParamForm


INTRO_MOST = 180  # characters of a tool's intro the page shows under its name


class InputCard(QtWidgets.QFrame):
    def __init__(self, row: dict, page, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SInput")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        theme.mark(self, "bound", row["bound"])
        self.row, self.page = row, page
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(12, 9, 10, 9)
        box.setSpacing(5)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        title = widgets.label(row["label"], "cardtitle", wrap=True)
        title.setMinimumWidth(60)
        top.addWidget(title, 1)
        top.addWidget(widgets.Chip(row["type_label"], theme.type_colour(row["type"])), 0, QtCore.Qt.AlignTop)
        box.addLayout(top)
        said = widgets.ElidedLabel(row["text"], "" if row["bound"] else ("warn" if row["required"] else "faint"))
        said.setObjectName("L2SUserText" if row["bound"] else "L2SInputState")
        box.addWidget(said)
        if row["detail"]:
            box.addWidget(widgets.label(row["detail"], "faint", name="L2SUserText"))
        holder = QtWidgets.QWidget()
        buttons = widgets.FlowLayout(holder, 6)  # wraps in a narrow panel
        self.use = widgets.button(text("dcc.ui.input.use_selected"), icon="plus")
        self.use.clicked.connect(guarded(lambda *_: page.bind_selected(row["item"])))
        pick = widgets.button(text("dcc.ui.input.pick_sequence" if row["sequence"] else "dcc.ui.input.pick_file"),
                              icon="frames" if row["sequence"] else "file")
        pick.clicked.connect(guarded(lambda *_: page.bind_file(row["item"])))
        clear = widgets.button(text("dcc.ui.input.clear"), "ghost")
        clear.setEnabled(row["bound"])
        clear.clicked.connect(guarded(lambda *_: page.unbind(row["item"])))
        for b in (self.use, pick, clear):
            buttons.addWidget(b)
        box.addWidget(holder)

    def selection(self, types: list[str]) -> None:
        """「用选中的」 greyed when nothing selected fits this input; its tooltip then says what to select
        (view_model.input_use)."""
        host = self.page.host
        use = view_model.input_use(self.row, types, host.exports, host.label, host.why_not)
        self.use.setEnabled(use["enabled"])
        self.use.setToolTip(use["why"])


class RunBox(QtWidgets.QFrame):
    """③ 计算: the button (计算 / 取消), the progress, the run's own words (the server's when it failed)."""

    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SRunBox")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.page = page
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(8)
        self.button = widgets.button(text("dcc.ui.run.compute"), "primary", icon="play")
        self.button.setMinimumHeight(36)
        self.button.clicked.connect(guarded(lambda *_: page.compute_or_cancel()))
        box.addWidget(self.button)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setObjectName("L2SProgress")
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.hide()
        box.addWidget(self.progress)
        row = QtWidgets.QHBoxLayout()
        self.status = widgets.label("", "muted", wrap=True, name="L2SStatus")
        self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        row.addWidget(self.status, 1)
        self.log_button = widgets.button(text("dcc.ui.run.open_log"), "ghost")
        self.log_button.clicked.connect(guarded(lambda *_: widgets.open_folder(log.file() or "")))
        self.log_button.hide()
        row.addWidget(self.log_button, 0, QtCore.Qt.AlignTop)
        box.addLayout(row)
        self.shown = object()

    def show_snapshot(self, snap: dict | None, missing: list[str], blocked: str = "") -> None:
        """`blocked`: why 「计算」 cannot be pressed now (the scene never saved: Plugin.why_not_saved), "" when it can."""
        key = (repr(snap), tuple(missing), blocked)
        if key == self.shown:
            return
        self.shown = key
        busy = snap is not None and not snap.get("done")
        self.button.setText(text("dcc.ui.run.cancel") if busy else text("dcc.ui.run.compute"))
        self.button.setIcon(theme.icon("close" if busy else "play", "text" if busy else "on-accent"))
        theme.mark(self.button, "busy", busy)
        self.button.setEnabled(busy or not blocked)  # a running job can always be cancelled
        self.button.setToolTip("" if busy else blocked)
        failed = bool(snap and snap.get("error"))
        if blocked and not busy:
            said = blocked
        elif snap is None:
            said = text("dcc.ui.run.missing", inputs=text("list.sep").join(missing)) if missing else ""
        else:
            said = view_model.run_line(snap)
        self.status.setText(said)
        theme.mark(self.status, "l2s", "error" if failed else "muted")
        self.log_button.setVisible(failed)
        if busy:
            self.progress.show()
            if snap.get("fraction") is None:
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 1000)
                self.progress.setValue(int(1000 * float(snap["fraction"])))
        else:
            self.progress.hide()


class VersionCard(widgets.Card):
    def __init__(self, v: dict, page, parent=None):
        super().__init__(parent, "L2SVersion")
        self.v = v
        theme.mark(self, "imported", v["imported"])
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(10, 7, 10, 7)
        box.setSpacing(2)
        box.addWidget(widgets.label(v["label"], "cardtitle"))
        box.addWidget(widgets.label(v["made"][5:16] if len(v["made"]) >= 16 else v["made"], "faint"))
        box.addWidget(widgets.label(v["status"], "ok" if v["imported"] else "warn"))
        self.setToolTip(v["folder"])
        self.setProperty("user_tip", True)  # the tooltip is the user's own folder (their data, in their words)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(guarded(lambda pos: page.version_menu(self, pos)))
        self.clicked.connect(guarded(lambda: page.version_clicked(v)))
        self.setFixedWidth(118)


class ToolPage(QtWidgets.QWidget):
    """One node's tool page. `back()` (to the home page) and `say(sentence)` are the shell's."""

    def __init__(self, plugin, shell, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SToolPage")
        self.plugin, self.host, self.shell = plugin, plugin.host, shell
        self.node: str | None = None
        self.input_cards: list[InputCard] = []
        self._frames: dict = {}  # (file, folder's mtime) -> 「首帧 … 尾帧 · N 帧」, scanned on a background thread
        self._scanning: set = set()
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.body = QtWidgets.QWidget()
        self.body.setObjectName("L2SToolBody")
        self.scroll = widgets.scroll(self.body)
        outer.addWidget(self.scroll)
        self.layout_ = QtWidgets.QVBoxLayout(self.body)
        self.layout_.setContentsMargins(14, 12, 14, 16)
        self.layout_.setSpacing(12)
        self.run_box: RunBox | None = None
        self.versions_row: QtWidgets.QHBoxLayout | None = None
        self.advanced: widgets.Collapsible | None = None
        self._shown_done = None  # the finished run the page was last rebuilt for

    # ---- reading

    def state(self) -> dict:
        return self.host.load(self.node) if self.node else {}

    def tool(self) -> dict | None:
        return self.plugin.tool(self.state()) if self.node else None

    # ---- building

    def show_node(self, node: str | None) -> None:
        self.node = node
        self.rebuild()

    def rebuild(self) -> None:
        keep = self.scroll.verticalScrollBar().value()
        widgets.clear(self.layout_)
        self.input_cards, self.run_box, self.versions_row, self.advanced = [], None, None, None
        state, tool = self.state(), self.tool()
        if not self.node:
            return
        if tool is None:
            name = (state.get("tool") or {}).get("name")
            said = text("dcc.ui.tool.gone", name=name) if name else text("dcc.ui.tool.none")
            self.layout_.addWidget(widgets.label(said, "muted", wrap=True))
            self.layout_.addStretch(1)
            return
        self._header(tool)
        self.note = widgets.label("", "muted", wrap=True, name="L2SNote")
        self.note.hide()
        self.layout_.addWidget(self.note)

        self.layout_.addWidget(widgets.StepHeader(1, text("dcc.ui.step.inputs")))
        rows = view_model.inputs(tool, state, self.host.describe_binding, self._frames_of)
        if not rows:
            self.layout_.addWidget(widgets.label(text("dcc.ui.input.none"), "faint"))
        for row in rows:
            card = InputCard(row, self)
            self.input_cards.append(card)
            self.layout_.addWidget(card)
        self.selection_changed(self.host.selection_types())

        params = view_model.parameters(tool, state)
        self.layout_.addWidget(widgets.StepHeader(2, text("dcc.ui.step.settings")))
        if params["common"]:
            self.layout_.addWidget(ParamForm(params["common"], self._set_value, self.say))
        elif not params["advanced"]:
            self.layout_.addWidget(widgets.label(text("dcc.ui.param.none"), "faint"))
        if params["advanced"]:
            opened = bool((connection.read_settings().get("advanced_open") or {}).get(tool.get("id"), False))
            self.advanced = widgets.Collapsible(text("dcc.ui.param.advanced", count=len(params["advanced"])), opened)
            self.advanced.body_layout.addWidget(ParamForm(params["advanced"], self._set_value, self.say))
            self.advanced.toggled.connect(guarded(lambda on, tid=tool.get("id"): _remember_open(tid, on)))
            self.layout_.addWidget(self.advanced)

        self.layout_.addWidget(widgets.StepHeader(3, text("dcc.ui.step.compute")))
        self.run_box = RunBox(self)
        self.layout_.addWidget(self.run_box)
        self.tick()

        versions = view_model.versions(state)
        if versions:
            self.layout_.addSpacing(4)
            self.layout_.addWidget(widgets.label(text("dcc.ui.version.title"), "section"))
            strip = QtWidgets.QWidget()
            strip.setObjectName("L2SVersionStrip")
            self.versions_row = QtWidgets.QHBoxLayout(strip)
            self.versions_row.setContentsMargins(0, 0, 0, 4)
            self.versions_row.setSpacing(6)
            for v in versions:
                self.versions_row.addWidget(VersionCard(v, self))
            self.versions_row.addStretch(1)
            area = widgets.scroll(strip, horizontal=True)
            area.setObjectName("L2SVersionScroll")
            area.setFixedHeight(strip.sizeHint().height() + 14)
            self.layout_.addWidget(area)
        self.layout_.addStretch(1)
        QtCore.QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(keep))

    def _header(self, tool: dict) -> None:
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(6)
        title = widgets.label(str(tool.get("name") or ""), "title", wrap=True, name="L2SToolName")
        title.setMinimumWidth(60)
        top.addWidget(title, 1)
        web = widgets.button(text("dcc.ui.tool.open_web"), "ghost", icon="open")
        web.clicked.connect(guarded(lambda *_: self.plugin.open_in_web(self.node, self.say)))
        top.addWidget(web, 0, QtCore.Qt.AlignTop)
        more = widgets.tool_button("more", text("dcc.ui.tool.more"))
        more.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(more)
        fetch = menu.addAction(text("dcc.ui.tool.fetch"))
        fetch.triggered.connect(guarded(lambda *_: self._fetch()))
        has_job = bool((self.state().get("job") or {}).get("id")) or bool(self.state().get("versions"))
        fetch.setEnabled(has_job and not self.plugin.why_not_saved())  # results go next to the saved scene only
        more.setMenu(menu)
        top.addWidget(more, 0, QtCore.Qt.AlignTop)
        self.layout_.addLayout(top)
        intro = str(tool.get("intro") or "").strip()
        if intro:  # its first paragraph, kept short; the whole in the tooltip
            first = intro.split("\n\n")[0].strip()
            said = widgets.label(first if len(first) <= INTRO_MOST else first[:INTRO_MOST].rstrip() + "…", "muted", wrap=True)
            said.setToolTip(intro if len(intro) <= 2000 else intro[:2000] + "…")
            self.layout_.addWidget(said)

    # ---- the timer's (the shell's): progress, and the selection

    def tick(self) -> None:
        if self.run_box is None or not self.node:
            return
        run = self.plugin.runs.get(self.node)
        snap = run.snapshot() if run is not None else None
        tool = self.tool()
        missing = view_model.missing_inputs(tool, self.state()) if tool and snap is None else []
        self.run_box.show_snapshot(snap, missing, self.plugin.why_not_saved())
        done = (self.node, id(run), snap.get("version")) if snap is not None and snap.get("done") else None
        if done is not None and done != self._shown_done:
            self._shown_done = done
            if snap.get("version"):  # a new version (or one imported now): the strip shows it
                QtCore.QTimer.singleShot(0, guarded(self.rebuild))

    def selection_changed(self, types: list[str]) -> None:
        for card in self.input_cards:
            card.selection(types)

    def say(self, sentence: str, error: bool = False) -> None:
        try:
            self.note.setText(sentence)
            theme.mark(self.note, "l2s", "error" if error else "muted")
            self.note.setVisible(bool(sentence))
        except (RuntimeError, AttributeError):  # rebuilt meanwhile
            pass

    # ---- ① inputs

    def bind_selected(self, item: dict) -> None:
        why = self.plugin.bind_selected(self.node, item)
        self.rebuild()
        self.say(why or text("dcc.ui.input.bound", label=item.get("label") or item["param"]), error=bool(why))

    def bind_file(self, item: dict) -> None:
        if item.get("widget") == "sequence":
            # the whole sequence selected (Ctrl+A in its folder), or any one frame of it: the rest is found beside it
            paths, _f = QtWidgets.QFileDialog.getOpenFileNames(self, text("dcc.ui.input.pick_sequence_title"))
            path = [os.path.normpath(x) for x in paths] if len(paths) > 1 else (os.path.normpath(paths[0]) if paths else "")
        else:
            got, _f = QtWidgets.QFileDialog.getOpenFileName(self, text("dcc.ui.input.pick_file_title"))
            path = os.path.normpath(got) if got else ""
        if path:
            self.plugin.bind_file(self.node, item, path)
            self.rebuild()

    def unbind(self, item: dict) -> None:
        self.plugin.unbind(self.node, item["param"])
        self.rebuild()

    def _frames_of(self, binding: dict) -> str:
        """A bound picture's frames, from its folder — scanned on a background thread (a long sequence, a network
        drive never hold the DCC up), kept by (file, the folder's modification time); "" until it is known."""
        f = str(binding.get("file") or "")
        if not f or binding.get("files") or not (binding.get("sequence") or str(binding.get("type", "")).startswith("image")):
            return ""
        try:
            key = (f, os.path.getmtime(os.path.dirname(f) or "."))
        except OSError:
            return ""
        if key in self._frames:
            return self._frames[key]
        if key not in self._scanning:
            self._scanning.add(key)

            def scan():
                said = ""
                try:
                    said = view_model.frames_text(view_model.sequence_frames(f))
                except Exception:  # noqa: BLE001 - a folder that cannot be read: nothing said
                    pass
                self._frames[key] = said
                if said:
                    self.host.run_on_main(lambda: guard.call(self._scanned))

            threading.Thread(target=scan, name="lab2shot-frames", daemon=True).start()
        return ""

    def _scanned(self) -> None:
        try:
            if self.isVisible():
                self.rebuild()
        except RuntimeError:  # the page is gone
            pass

    # ---- ② settings

    def _set_value(self, x: dict, value) -> None:
        self.plugin.set_value(self.node, x["name"], value)
        QtCore.QTimer.singleShot(0, guarded(self.rebuild))  # conditions may show, hide or grey others

    # ---- ③ compute

    def compute_or_cancel(self) -> None:
        if self.plugin.busy(self.node or ""):
            self.plugin.cancel(self.node or "")
            return
        self.run_box.shown = object()
        self.plugin.compute(self.node)
        self.tick()

    def _fetch(self) -> None:
        self.run_box.shown = object()
        self.plugin.fetch(self.node)
        self.tick()

    # ---- versions

    def version_clicked(self, v: dict) -> None:
        if v["imported"]:
            self.host.locate(v["objects"])
            return
        if self.host.confirm(text("dcc.ui.version.import_ask", version=v["label"], folder=v["folder"])):
            self.run_box.shown = object()
            self.plugin.import_version(self.node, v["version"])
            self.tick()

    def version_menu(self, card: VersionCard, pos) -> None:
        menu = QtWidgets.QMenu(card)
        folder = menu.addAction(text("dcc.ui.version.open_folder"))
        folder.triggered.connect(guarded(lambda *_: widgets.open_folder(card.v["folder"])))
        folder.setEnabled(bool(card.v["folder"]) and os.path.isdir(card.v["folder"]))
        menu.exec(card.mapToGlobal(pos)) if hasattr(menu, "exec") else menu.exec_(card.mapToGlobal(pos))


def _remember_open(tool_id: str, opened: bool) -> None:
    got = connection.read_settings().get("advanced_open")
    got = dict(got) if isinstance(got, dict) else {}
    got[str(tool_id)] = bool(opened)
    connection.write_settings(advanced_open=got)
