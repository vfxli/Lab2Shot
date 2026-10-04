"""The panel's building blocks, knowing nothing of tools or DCCs (设计_DCC新面板.md §1.1): a card that can be clicked,
a data type chip, the search field, a tab strip, a section that folds, a flow layout, a label that elides, a step
header (① ② ③), a banner, the buttons. Their look is the theme's (objectName and dynamic properties: theme.qss), never
a style sheet of their own."""

from __future__ import annotations

import os

from .. import log
from ..qt import QtCore, QtGui, QtWidgets
from . import theme


def open_folder(path: str) -> None:
    """A folder (a file: the folder it is in) in the system's file browser, on every platform (QDesktopServices)."""
    if not path:
        return
    folder = path if os.path.isdir(path) else os.path.dirname(path)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(folder))
    log.get().info("opened %s", os.path.normpath(folder))


def label(text: str = "", role: str = "", wrap: bool = False, name: str = "") -> QtWidgets.QLabel:
    """A label in one of the theme's roles: "" body, muted, faint, title, heading, section, error, mono."""
    w = QtWidgets.QLabel(text)
    if role:
        w.setProperty("l2s", role)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    if wrap:
        w.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
    w.setTextFormat(QtCore.Qt.PlainText)
    return w


def row_note(text: str) -> QtWidgets.QLabel:
    """The note under a row (a template's `note` on a parameter: a consequence or what it is for), faint and wrapped
    across the whole row, as the web page's LabelRow note: read where the row is, never in a hover."""
    return label(text, "faint", wrap=True, name="L2SRowNote")


def button(text: str, kind: str = "", icon: str = "", tip: str = "") -> QtWidgets.QPushButton:
    """A push button: kind "" (a control), "primary" (the page's one filled button), "ghost" (no face until hovered)."""
    b = QtWidgets.QPushButton(text)
    b.setObjectName({"primary": "L2SPrimary", "ghost": "L2SGhost"}.get(kind, "L2SButton"))
    b.setCursor(QtCore.Qt.PointingHandCursor)
    if icon:
        b.setIcon(theme.icon(icon, "on-accent" if kind == "primary" else "text-2"))
        b.setIconSize(QtCore.QSize(14, 14))
    if tip:
        b.setToolTip(tip)
    return b


def tool_button(icon: str, tip: str, rotate: int = 0) -> QtWidgets.QToolButton:
    """A square icon button (more, back, close); always with a tooltip saying what it does."""
    b = QtWidgets.QToolButton()
    b.setObjectName("L2SIconButton")
    b.setIcon(theme.icon(icon, "text-2", rotate=rotate))
    b.setIconSize(QtCore.QSize(16, 16))
    b.setToolTip(tip)
    b.setCursor(QtCore.Qt.PointingHandCursor)
    b.setAutoRaise(True)
    return b


class ElidedLabel(QtWidgets.QLabel):
    """One line, cut with … to the width it is given (the whole text in its tooltip)."""

    def __init__(self, text: str = "", role: str = "", parent=None):
        super().__init__(parent)
        if role:
            self.setProperty("l2s", role)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(20)
        self.set_text(text)

    def set_text(self, text: str) -> None:
        self._full = str(text or "")
        self.setToolTip(self._full if len(self._full) > 20 else "")
        self._fit()

    def full_text(self) -> str:
        return self._full

    def _fit(self) -> None:
        metrics = QtGui.QFontMetrics(self.font())
        super().setText(metrics.elidedText(self._full, QtCore.Qt.ElideRight, max(10, self.width())))

    def resizeEvent(self, event):  # noqa: N802 - Qt's name
        super().resizeEvent(event)
        self._fit()

    def sizeHint(self):  # noqa: N802
        h = super().sizeHint()
        return QtCore.QSize(min(h.width(), 400), h.height())


class Card(QtWidgets.QFrame):
    """A surface that can be clicked: hover and pressed shown by the theme (#L2SCard[hover] …); `clicked` on release."""

    clicked = QtCore.Signal()

    def __init__(self, parent=None, name: str = "L2SCard"):
        super().__init__(parent)
        self.setObjectName(name)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setAttribute(QtCore.Qt.WA_Hover, True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.TabFocus)

    def enterEvent(self, event):  # noqa: N802
        theme.mark(self, "hover", True)
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        theme.mark(self, "hover", False)
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class Chip(QtWidgets.QFrame):
    """A data type: its colour dot and its name."""

    def __init__(self, text: str, colour: str, faint: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SChip")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        if faint:
            self.setProperty("faint", True)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(6, 1, 7, 1)
        row.setSpacing(5)
        d = QtWidgets.QLabel()
        d.setPixmap(theme.dot(colour, 7, theme.ratio()))
        d.setObjectName("L2SChipDot")
        row.addWidget(d)
        t = label(text, "chip")
        row.addWidget(t)


class SearchField(QtWidgets.QLineEdit):
    """The search: what is typed is told (`changed`) once typing pauses (DEBOUNCE_MS)."""

    DEBOUNCE_MS = 150
    changed = QtCore.Signal(str)

    def __init__(self, placeholder: str, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SSearch")
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.DEBOUNCE_MS)
        self._timer.timeout.connect(lambda: self.changed.emit(self.text()))
        self.textChanged.connect(lambda *_: self._timer.start())


class FlowLayout(QtWidgets.QLayout):
    """Items left to right, wrapping to the next row when the width runs out (Qt's own example, kept short)."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items: list = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):  # noqa: N802
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):  # noqa: N802
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):  # noqa: N802
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return QtCore.Qt.Orientations(0)

    def hasHeightForWidth(self):  # noqa: N802
        return True

    def heightForWidth(self, width):  # noqa: N802
        return self._lay(QtCore.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):  # noqa: N802
        super().setGeometry(rect)
        self._lay(rect, False)

    def sizeHint(self):  # noqa: N802
        return self.minimumSize()

    def minimumSize(self):  # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _lay(self, rect, test: bool) -> int:
        m = self.contentsMargins()
        x, y, row = rect.x() + m.left(), rect.y() + m.top(), 0
        right = rect.right() - m.right()
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > right + 1 and row > 0:
                x, y, row = rect.x() + m.left(), y + row + self._gap, 0
            if not test:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), hint))
            x += hint.width() + self._gap
            row = max(row, hint.height())
        return y + row - rect.y() + m.bottom()


def clear(layout: QtWidgets.QLayout) -> None:
    """Remove (and delete) everything a layout holds, nested layouts too."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear(item.layout())


class TabStrip(QtWidgets.QWidget):
    """A row of tabs that wraps; one chosen (`chosen(id)`)."""

    chosen = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("L2STabs")
        self._flow = FlowLayout(self, 4)
        self._group = QtWidgets.QButtonGroup(self)
        self._group.setExclusive(True)
        self._ids: dict = {}

    def set_tabs(self, tabs: list[dict], current: str) -> None:
        for b in list(self._group.buttons()):
            self._group.removeButton(b)
        clear(self._flow)
        self._ids = {}
        for t in tabs:
            b = QtWidgets.QPushButton(t["label"])
            b.setObjectName("L2STab")
            b.setCheckable(True)
            b.setCursor(QtCore.Qt.PointingHandCursor)
            b.setProperty("tab", t["id"])
            if t.get("colour"):
                b.setIcon(QtGui.QIcon(theme.dot(t["colour"], 7, theme.ratio())))
                b.setIconSize(QtCore.QSize(7, 7))
            b.setChecked(t["id"] == current)
            b.clicked.connect(lambda *_, tid=t["id"]: self.chosen.emit(tid))
            self._group.addButton(b)
            self._flow.addWidget(b)
            self._ids[t["id"]] = b
        self.updateGeometry()


class Collapsible(QtWidgets.QWidget):
    """A section that folds: its header (a chevron and a title) opens and closes it; `toggled(open)`."""

    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, opened: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SCollapsible")
        box = QtWidgets.QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        self.header = QtWidgets.QToolButton()
        self.header.setObjectName("L2SFold")
        self.header.setText(title)
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setCursor(QtCore.Qt.PointingHandCursor)
        self.header.setCheckable(True)
        self.header.setAutoRaise(True)
        self.header.clicked.connect(lambda *_: self.set_open(self.header.isChecked(), tell=True))
        box.addWidget(self.header)
        self.body = QtWidgets.QWidget()
        self.body.setObjectName("L2SFoldBody")
        self.body_layout = QtWidgets.QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.body)
        self.set_open(opened)

    def set_open(self, opened: bool, tell: bool = False) -> None:
        self.header.setChecked(opened)
        self.header.setIcon(theme.icon("chevron", "text-2", rotate=0 if opened else -90))
        self.body.setVisible(opened)
        if tell:
            self.toggled.emit(opened)

    def is_open(self) -> bool:
        return self.header.isChecked()


class StepHeader(QtWidgets.QWidget):
    """① 输入 / ② 设置 / ③ 计算: a numbered heading."""

    def __init__(self, number: int, title: str, parent=None):
        super().__init__(parent)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        n = label(str(number), name="L2SStepNumber")
        n.setAlignment(QtCore.Qt.AlignCenter)
        n.setFixedSize(20, 20)
        row.addWidget(n)
        row.addWidget(label(title, "heading"))
        row.addStretch(1)
        self.trailing = row


class Banner(QtWidgets.QFrame):
    """One sentence across the panel (not logged in, the server away, the tool list failed) and what to press."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("L2SBanner")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(12, 8, 8, 8)
        self.text = label("", wrap=True)
        row.addWidget(self.text, 1)
        self.action = button("", "")
        row.addWidget(self.action)
        self._slot = None
        self.action.clicked.connect(lambda *_: self._slot() if self._slot else None)
        self.hide()

    def say(self, text: str, action: str = "", slot=None, kind: str = "warn") -> None:
        if not text:
            self.hide()
            return
        theme.mark(self, "kind", kind)
        self.text.setText(text)
        self.action.setText(action)
        self.action.setVisible(bool(action))
        self._slot = slot
        self.show()


def hline() -> QtWidgets.QFrame:
    line = QtWidgets.QFrame()
    line.setObjectName("L2SHair")
    line.setFixedHeight(1)
    line.setAttribute(QtCore.Qt.WA_StyledBackground, True)
    return line


def scroll(body: QtWidgets.QWidget, horizontal: bool = False) -> QtWidgets.QScrollArea:
    area = QtWidgets.QScrollArea()
    area.setObjectName("L2SScroll")
    area.setWidgetResizable(True)
    area.setFrameShape(QtWidgets.QFrame.NoFrame)
    if horizontal:
        area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    else:
        area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    body.setObjectName(body.objectName() or "L2SScrollBody")
    area.setWidget(body)
    area.viewport().setObjectName("L2SViewport")
    return area
