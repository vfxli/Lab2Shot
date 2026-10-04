"""Qt for the plugin's windows: PySide6 when the DCC brings it (Maya 2025+, Houdini 20+, Nuke 15+), else PySide2 —
found by trying, never by the DCC's version number."""

from __future__ import annotations

try:  # noqa: SIM105
    from PySide6 import QtCore, QtGui, QtWidgets  # type: ignore
    import shiboken6 as _shiboken  # type: ignore

    QT6 = True
except ImportError:  # pragma: no cover - depends on the DCC
    from PySide2 import QtCore, QtGui, QtWidgets  # type: ignore
    import shiboken2 as _shiboken  # type: ignore

    QT6 = False

try:  # the panel's icons are drawn from SVG (both PySide6 and PySide2 bring QtSvg); without it they are left out
    if QT6:
        from PySide6 import QtSvg  # type: ignore
    else:  # pragma: no cover - depends on the DCC
        from PySide2 import QtSvg  # type: ignore
except ImportError:  # pragma: no cover
    QtSvg = None

__all__ = ["QtCore", "QtGui", "QtWidgets", "QtSvg", "QT6", "wrap", "unwrap"]


def wrap(pointer, cls):
    """A Qt object from a pointer a DCC hands out (Maya's MQtUtil)."""
    return _shiboken.wrapInstance(int(pointer), cls)


def unwrap(widget) -> int:
    return int(_shiboken.getCppPointer(widget)[0])


def exec_(dialog) -> int:
    return dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()
