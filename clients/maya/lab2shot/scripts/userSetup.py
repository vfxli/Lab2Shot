# Lab2Shot for Maya: one thing only — once Maya is idle, add the Lab2Shot menu. Everything is inside try: whatever goes
# wrong is written to %USERPROFILE%/lab2shot/maya.log and Maya starts as usual (no menu, nothing else touched). Maya
# runs every userSetup.py it finds in the __main__ namespace: the names here start with _lab2shot so none clashes.


def _lab2shot_log(text):
    try:
        import os
        import time

        folder = os.path.join(os.environ.get("USERPROFILE") or os.path.expanduser("~"), "lab2shot")
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "maya.log"), "a", encoding="utf-8") as f:
            f.write("%s userSetup %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), text))
    except Exception:
        pass


def _lab2shot_start():
    try:
        import maya.cmds

        if maya.cmds.about(batch=True):
            return
        import lab2shot_maya.menu

        lab2shot_maya.menu.install()
    except Exception:
        import traceback

        _lab2shot_log("loading the Lab2Shot menu failed:\n" + traceback.format_exc())


try:
    import maya.utils

    maya.utils.executeDeferred(_lab2shot_start)
except Exception:
    import traceback

    _lab2shot_log("userSetup failed:\n" + traceback.format_exc())
