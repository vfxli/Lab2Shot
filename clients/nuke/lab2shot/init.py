# Lab2Shot for Nuke: this folder is on Nuke's plugin path (the user's ~/.nuke/init.py adds it). Here only: make
# this folder importable. No interface is built here (init.py also runs in nuke -t and on render nodes); anything
# that goes wrong is written to %USERPROFILE%/lab2shot/nuke.log and Nuke starts as usual.
try:
    import os as _os
    import sys as _sys

    import nuke as _nuke

    _here = ""
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
    except NameError:  # Nuke runs init.py without __file__: the plugin path entry that holds our package
        for _p in _nuke.pluginPath():
            if _os.path.isdir(_os.path.join(_p, "lab2shot_nuke")):
                _here = _p
                break
    if _here and _here not in _sys.path:
        _sys.path.insert(0, _here)
except Exception:  # noqa: BLE001 - never stop Nuke from starting
    try:
        import traceback as _tb

        _folder = _os.path.join(_os.environ.get("USERPROFILE") or _os.path.expanduser("~"), "lab2shot")
        _os.makedirs(_folder, exist_ok=True)
        with open(_os.path.join(_folder, "nuke.log"), "a", encoding="utf-8") as _f:
            _f.write("Lab2Shot init.py failed:\n" + _tb.format_exc())
    except Exception:  # noqa: BLE001
        pass
