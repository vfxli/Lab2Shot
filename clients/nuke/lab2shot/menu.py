# Lab2Shot for Nuke: Nuke runs this when its interface starts (never in nuke -t): the Lab2Shot menu. Anything that
# goes wrong is written to %USERPROFILE%/lab2shot/nuke.log and Nuke starts as usual.
try:
    import lab2shot_nuke.menu as _lab2shot_menu

    _lab2shot_menu.install()
except Exception:  # noqa: BLE001 - never stop Nuke from starting
    try:
        import os as _os
        import traceback as _tb

        _folder = _os.path.join(_os.environ.get("USERPROFILE") or _os.path.expanduser("~"), "lab2shot")
        _os.makedirs(_folder, exist_ok=True)
        with open(_os.path.join(_folder, "nuke.log"), "a", encoding="utf-8") as _f:
            _f.write("Lab2Shot menu.py failed:\n" + _tb.format_exc())
    except Exception:  # noqa: BLE001
        pass
