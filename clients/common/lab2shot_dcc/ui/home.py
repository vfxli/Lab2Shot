"""The home page: the search, 「适合当前选中」, 「本场景的任务」, the category tabs and the tool cards (设计_DCC新面板.md §1).
What is shown is view_model.home's; this only draws it. A card says the tool's name, one line of what it does, and
「需要：画面、摄影机」 (the data types it must be given, each with its colour)."""

from __future__ import annotations

from .. import view_model
from ..paths import text
from ..qt import QtCore, QtWidgets
from . import theme, widgets

CARD_MIN = 210  # px a card wants at least: the grid has as many columns as fit (1–3)
COLUMNS_MOST = 3


class ToolCard(widgets.Card):
    def __init__(self, c: dict, parent=None):
        super().__init__(parent)
        self.tool_id = c["id"]
        self.setProperty("tool", c["id"])
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(12, 10, 12, 10)
        box.setSpacing(5)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        mark = QtWidgets.QLabel()
        mark.setObjectName("L2SCategoryMark")
        mark.setPixmap(theme.category_mark(c["category"], c["colour"], 22, theme.ratio()))
        mark.setFixedSize(22, 22)
        top.addWidget(mark, 0, QtCore.Qt.AlignVCenter)
        name = widgets.ElidedLabel(c["name"], "cardtitle")
        name.setObjectName("L2SCardName")
        top.addWidget(name, 1)
        box.addLayout(top)
        if c["intro"]:
            intro = widgets.ElidedLabel(c["intro"], "muted")
            intro.setObjectName("L2SCardIntro")
            box.addWidget(intro)
        if c["needs"] or c["optional"]:
            chips = QtWidgets.QWidget()
            flow = widgets.FlowLayout(chips, 4)
            if c["needs"]:
                flow.addWidget(widgets.label(text("dcc.ui.card.needs_word"), "faint"))
                for t, n in c["needs"]:
                    flow.addWidget(widgets.Chip(n, theme.type_colour(t)))
            if c["optional"]:
                flow.addWidget(widgets.label(text("dcc.ui.card.optional_word"), "faint"))
                for t, n in c["optional"]:
                    flow.addWidget(widgets.Chip(n, theme.type_colour(t), faint=True))
            box.addWidget(chips)
        self.setToolTip(view_model.needs_line(c))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)


class CardGrid(QtWidgets.QWidget):
    """Cards in as many columns as the width holds (1–COLUMNS_MOST), laid again when the panel is resized."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(8)
        self.cards: list[QtWidgets.QWidget] = []
        self._columns = 0

    def set_cards(self, cards: list[QtWidgets.QWidget]) -> None:
        widgets.clear(self.grid)
        self.cards = list(cards)
        self._columns = 0
        self._lay()

    def columns(self) -> int:
        return max(1, min(COLUMNS_MOST, (self.width() + 8) // (CARD_MIN + 8)))

    def _lay(self) -> None:
        n = self.columns()
        if n == self._columns and self.grid.count() == len(self.cards):
            return
        self._columns = n
        while self.grid.count():
            self.grid.takeAt(0)
        for i, card in enumerate(self.cards):
            self.grid.addWidget(card, i // n, i % n)
        for c in range(COLUMNS_MOST):
            self.grid.setColumnStretch(c, 1 if c < n else 0)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._lay()


class TaskChip(widgets.Card):
    """One Lab2Shot node of the scene: its name, its tool, its state; a click goes to it."""

    def __init__(self, t: dict, current: bool, parent=None):
        super().__init__(parent, "L2STask")
        self.node = t["node"]
        theme.mark(self, "current", current)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(10, 5, 10, 5)
        row.setSpacing(6)
        dot = QtWidgets.QLabel()
        dot.setPixmap(theme.dot(theme.token({"running": "accent", "queued": "accent", "failed": "error",
                                             "done": "orange", "imported": "green"}.get(t["state"], "text-3")),
                                7, theme.ratio()))
        row.addWidget(dot)
        name = widgets.label(t["name"], name="L2SUserText")
        row.addWidget(name)
        if t["tool"]:
            row.addWidget(widgets.label("·", "faint"))
            row.addWidget(widgets.label(t["tool"], "muted"))
        self.setToolTip(t["state_label"])


class HomePage(QtWidgets.QWidget):
    """The home page; `tool_chosen(tool_id, suggested)` when a card is clicked, `node_chosen(node)` for a task."""

    tool_chosen = QtCore.Signal(str, bool)
    node_chosen = QtCore.Signal(str)

    def __init__(self, plugin, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SHome")
        self.plugin, self.host = plugin, plugin.host
        self.search_text, self.tab = "", view_model.ALL
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        body = QtWidgets.QWidget()
        body.setObjectName("L2SHomeBody")
        self.scroll = widgets.scroll(body)
        outer.addWidget(self.scroll)
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(14, 12, 14, 16)
        v.setSpacing(10)
        self.search = widgets.SearchField(text("dcc.ui.home.search"))
        self.search.changed.connect(self._searched)
        v.addWidget(self.search)

        self.suggest_box = QtWidgets.QWidget()
        s = QtWidgets.QVBoxLayout(self.suggest_box)
        s.setContentsMargins(0, 4, 0, 0)
        s.setSpacing(6)
        s.addWidget(widgets.label(text("dcc.ui.home.suggested"), "section"))
        self.suggested = CardGrid()
        s.addWidget(self.suggested)
        v.addWidget(self.suggest_box)

        self.tasks_box = QtWidgets.QWidget()
        t = QtWidgets.QVBoxLayout(self.tasks_box)
        t.setContentsMargins(0, 4, 0, 0)
        t.setSpacing(6)
        t.addWidget(widgets.label(text("dcc.ui.home.tasks"), "section"))
        holder = QtWidgets.QWidget()
        self.tasks_flow = widgets.FlowLayout(holder, 6)
        t.addWidget(holder)
        v.addWidget(self.tasks_box)

        v.addWidget(widgets.hline())
        self.tabs = widgets.TabStrip()
        self.tabs.chosen.connect(self._tab)
        v.addWidget(self.tabs)
        self.empty = widgets.label("", "muted", wrap=True)
        v.addWidget(self.empty)
        self.cards = CardGrid()
        v.addWidget(self.cards)
        v.addStretch(1)
        self._selected: list[str] = []
        self._tasks_key = None

    # ---- drawing

    def refresh(self, selected: list[str] | None = None, current_node: str | None = None) -> None:
        if selected is not None:
            self._selected = list(selected)
        got = view_model.home(self.plugin.tools, self.plugin.categories, self._selected, self.search_text, self.tab,
                              self.plugin.recent(), self.host.prefers, self.host.exports)
        self.tab = got["tab"]
        self.tabs.set_tabs(got["tabs"], self.tab)
        self.suggested.set_cards([self._card(c, True) for c in got["suggested"]])
        self.suggest_box.setVisible(bool(got["suggested"]))
        self.cards.set_cards([self._card(c, False) for c in got["cards"]])
        if not self.plugin.tools:
            said = text("dcc.ui.home.loading") if self.plugin.loading else text("dcc.ui.home.no_tools")
        elif not got["cards"]:
            said = text("dcc.ui.home.nothing_found", search=self.search_text) if self.search_text else ""
        else:
            said = ""
        self.empty.setText(said)
        self.empty.setVisible(bool(said))
        self.refresh_tasks(current_node, force=True)

    def refresh_selection(self, selected: list[str]) -> None:
        """The scene's selection changed: only 「适合当前选中」 is worked out again (the grid's order follows on the
        next full refresh, not under the user's pointer)."""
        if selected == self._selected:
            return
        self._selected = list(selected)
        got = view_model.home(self.plugin.tools, self.plugin.categories, self._selected, self.search_text, self.tab,
                              self.plugin.recent(), self.host.prefers, self.host.exports)
        self.suggested.set_cards([self._card(c, True) for c in got["suggested"]])
        self.suggest_box.setVisible(bool(got["suggested"]))

    def refresh_tasks(self, current_node: str | None, force: bool = False) -> None:
        tasks = []
        for n in self.host.nodes():
            run = self.plugin.runs.get(n)
            state = self.host.load(n)
            tasks.append(view_model.task(n, self.host.node_name(n), state, run.snapshot() if run is not None else None,
                                         self.plugin.tool(state)))
        key = (tuple((t["node"], t["name"], t["tool"], t["state"]) for t in tasks), current_node)
        if key == self._tasks_key and not force:
            return
        self._tasks_key = key
        widgets.clear(self.tasks_flow)
        for t in tasks:
            chip = TaskChip(t, t["node"] == current_node)
            chip.clicked.connect(lambda node=t["node"]: self.node_chosen.emit(node))
            self.tasks_flow.addWidget(chip)
        self.tasks_box.setVisible(bool(tasks))
        self.tasks_flow.parentWidget().updateGeometry()

    def _card(self, c: dict, suggested: bool) -> ToolCard:
        card = ToolCard(c)
        card.clicked.connect(lambda tid=c["id"], s=suggested: self.tool_chosen.emit(tid, s))
        return card

    def _searched(self, said: str) -> None:
        self.search_text = said.strip()
        self.refresh()

    def _tab(self, tab: str) -> None:
        self.tab = tab
        self.refresh()
