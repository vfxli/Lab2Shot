"""② 设置: the tool's parameters as its contract says (lab2shot_dcc.contract), the same for every tool of every source
— common ones shown, the template's folded groups under 「高级」 (view_model.parameters). Knows no tool and no DCC."""

from __future__ import annotations

import json

from .. import contract
from ..guard import guarded
from ..paths import text
from ..qt import QtCore, QtWidgets
from . import widgets


LABEL_WIDE, LABEL_MOST = 120, 170  # px a parameter's name keeps on one line at least / takes at most (then wraps)


class ParamForm(QtWidgets.QWidget):
    """Rows of `view_model.parameters(...)[part]`, grouped by the interface's own groups. `set_value(x, value)` and
    `say(sentence)` are the page's (they write the node and redraw)."""

    def __init__(self, rows: list[dict], set_value, say, parent=None):
        super().__init__(parent)
        self.set_value, self.say = set_value, say
        self._current = rows[0]["values"] if rows else {}
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)
        forms: dict[tuple, QtWidgets.QFormLayout] = {}
        for row in rows:
            path = tuple(row["group"])
            form = forms.get(path)
            if form is None:
                box = QtWidgets.QWidget()
                box.setObjectName("L2SGroup")
                v = QtWidgets.QVBoxLayout(box)
                v.setContentsMargins(0, 0, 0, 0)
                v.setSpacing(6)
                if path:
                    v.addWidget(widgets.label(" / ".join(path), "section"))
                form = QtWidgets.QFormLayout()
                form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
                form.setHorizontalSpacing(12)
                form.setVerticalSpacing(6)
                form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
                v.addLayout(form)
                forms[path] = form
                outer.addWidget(box)
            form.addRow(self._label(row), self._field(row))
            if row["note"]:  # the template's note on it, under the row (widgets.row_note)
                form.addRow(widgets.row_note(row["note"]))

    def _label(self, row: dict) -> QtWidgets.QWidget:
        x = row["x"]
        holder = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(holder)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        said = str(x.get("label") or x["name"])
        name = widgets.label(said, "muted", wrap=True)
        name.setMinimumWidth(min(LABEL_WIDE, name.fontMetrics().horizontalAdvance(said) + 4))
        name.setMaximumWidth(LABEL_MOST)
        name.setToolTip(x["name"])
        h.addWidget(name)
        if row["from_scene"]:
            tag = widgets.label(text("dcc.ui.param.from_scene"), name="L2SFromScene")
            tag.setToolTip(text("dcc.ui.param.from_scene_tip"))
            h.addWidget(tag)
        h.addStretch(1)
        return holder

    def _field(self, row: dict) -> QtWidgets.QWidget:
        x, value = row["x"], row["value"]
        spec = x.get("param") or {}
        options = contract.menu_options(x, row["values"])
        if options:
            w = QtWidgets.QComboBox()
            w.setObjectName("L2SCombo")
            # as narrow as the panel: a long option never widens the page (it shows whole in the open list)
            w.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(6)
            for v, label in options:
                w.addItem(label, json.dumps(v))
            i = w.findData(json.dumps(value))
            if i >= 0:
                w.setCurrentIndex(i)
            w.activated.connect(guarded(lambda *_, x=x, w=w: self._set(x, json.loads(w.currentData()))))
        elif spec.get("type") == "boolean" or x.get("widget") == "checkbox":
            w = QtWidgets.QCheckBox()
            w.setToolTip(str(x.get("label") or x["name"]))
            w.setChecked(bool(value))
            w.toggled.connect(guarded(lambda on, x=x: self._set(x, bool(on) if spec.get("type") == "boolean" else int(on))))
        elif spec.get("type") in ("integer", "number", "string"):
            w = QtWidgets.QLineEdit("" if value is None else str(value))
            w.setPlaceholderText(str(spec.get("placeholder") or (text("dcc.ui.param.blank") if spec.get("nullable") else "")))
            w.editingFinished.connect(guarded(lambda x=x, w=w: self._typed(x, w)))
        else:
            w = widgets.label(text("dcc.ui.param.in_web"), "faint", wrap=True)
            w.setToolTip(json.dumps(value, ensure_ascii=False)[:400])
        if x.get("wired") and not x.get("fallback"):
            w.setToolTip(text("dcc.ui.param.wired", wired=x["wired"]))
        w.setEnabled(not row["disabled"])
        return w

    def _set(self, x: dict, value) -> None:
        why = contract.refuses(x, value)
        if why:
            self.say(text("dcc.ui.param.refused", label=x.get("label") or x["name"], why=why))
            return
        self.set_value(x, value)

    def _typed(self, x: dict, w) -> None:
        try:
            value = contract.parse(x, w.text())
        except ValueError:
            self.say(text("dcc.ui.param.bad_value", label=x.get("label") or x["name"], value=w.text()))
            return
        if value == self._values_of(x):
            return
        self._set(x, value)

    def _values_of(self, x: dict):
        return self._current.get(x["name"])
