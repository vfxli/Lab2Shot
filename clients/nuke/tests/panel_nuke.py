"""The panel (lab2shot_dcc.ui) off screen in Nuke's own Qt (Nuke's python.exe: PySide6, no `nuke` imported, no
licence taken), with a fake host shaped like Nuke's (its prefers and exports, from clients/nuke's NukeHost) and a tool
list recorded from a server: what a Nuke user gets that a Maya user does not.

    QT_QPA_PLATFORM=offscreen QT_PLUGIN_PATH=<Nuke>/qtplugins PYTHONPATH=<Nuke>/pythonextensions/site-packages \
        "<Nuke>/python.exe" panel_nuke.py <config.json>

config: {"tools_zh", "tools_en": GET /api/tools recorded in each language, "out": pictures, "report": results (json)}.
Checks: pictures first in 「适合当前选中」; a camera and a plate selected → the card binds the camera to cam_usd_path
(Nuke exports a .usda); 「用选中的」 of cam_fbx_path greyed with the reason; 「计算」 greyed with 'save the project first'
while the script was never saved, live once it is; a version card locates (never selects); English without Chinese;
pictures in both languages.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

with open(sys.argv[1], encoding="utf-8") as _f:
    CFG = json.load(_f)
HERE = os.path.dirname(os.path.abspath(__file__))
COMMON_TESTS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "common", "tests")
sys.path[:0] = [os.path.dirname(COMMON_TESTS), COMMON_TESTS]
os.environ["USERPROFILE"] = tempfile.mkdtemp(prefix="l2s_nuke_panel_")  # the plugin's settings: this run's own
os.environ["HOME"] = os.environ["USERPROFILE"]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lab2shot_dcc.qt import QtCore, QtGui, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
for _font in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\segoeui.ttf"):
    if os.path.isfile(_font):
        QtGui.QFontDatabase.addApplicationFont(_font)
if "Segoe UI" in QtGui.QFontDatabase.families():
    app.setFont(QtGui.QFont("Segoe UI", 9))

from fake_host import FakeHost  # noqa: E402
from lab2shot_dcc import connection, paths, view_model  # noqa: E402
from lab2shot_dcc.plugin import Plugin  # noqa: E402

REPORT: dict = {"steps": [], "ok": True, "shots": []}
OUT = CFG["out"]
os.makedirs(OUT, exist_ok=True)


def step(name, ok, detail=""):
    REPORT["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:2000]})
    REPORT["ok"] = REPORT["ok"] and bool(ok)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {str(detail)[:300]}"))


class NukeLikeHost(FakeHost):
    """The fake host with Nuke's data (NukeHost.prefers / exports, read from its source so they cannot drift)."""

    label = "Nuke"

    def __init__(self, project):
        super().__init__(project)
        source = open(os.path.join(os.path.dirname(HERE), "lab2shot", "lab2shot_nuke", "host.py"), encoding="utf-8").read()
        scope: dict = {}
        for line in source.splitlines():
            if line.strip().startswith(("prefers = ", "exports = ")):
                exec(line.strip(), {}, scope)  # noqa: S102 - our own source, two data lines
        self.prefers, self.exports = scope["prefers"], scope["exports"]
        self.selected_lab2shot = None
        self.located = []
        self.selected_by_us = []

    def selected_node(self):
        return self.selected_lab2shot

    def why_not(self, data_type):
        return "Select a Read first" if paths.client_module().get_lang() == "en" else "先在节点图里选中一个 Read"

    def locate(self, refs):
        self.located.append(list(refs))

    def select(self, refs):  # Nuke never selects for the user
        self.selected_by_us.append(list(refs))


def pump(ms=80):
    timer = QtCore.QElapsedTimer()
    timer.start()
    while timer.elapsed() < ms:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)


def shoot(panel, name, width=460, height=900):
    panel.resize(width, height)
    panel.show()
    pump(150)
    path = os.path.join(OUT, f"{name}.png")
    step(f"shot {name}", panel.grab().save(path), path)
    REPORT["shots"].append(path)


def has_chinese(panel) -> list:
    bad = []
    for w in [panel, *panel.findChildren(QtWidgets.QWidget)]:
        if w is not panel and not w.isVisibleTo(panel) or w.objectName() == "L2SUserText":
            continue
        said = w.text() if isinstance(w, (QtWidgets.QLabel, QtWidgets.QAbstractButton)) else ""
        for s in (said, "" if w.property("user_tip") else w.toolTip()):
            if view_model.has_cjk(s):
                bad.append(f"{type(w).__name__}#{w.objectName()}: {s[:60]}")
    return bad


def main():
    project = tempfile.mkdtemp(prefix="l2s_nuke_panel_project_")
    host = NukeLikeHost(project)
    client = paths.client_module()
    server = "http://127.0.0.1:8781"
    with open(client.tokens_file(), "w", encoding="utf-8") as f:
        json.dump({server: {"token": "x", "username": "tester"}}, f)
    connection.write_settings(server=server, lang="zh")
    plugin = Plugin(host)
    plugin.conn.user = {"username": "tester"}
    tools = {lang: json.load(open(CFG[f"tools_{lang}"], encoding="utf-8")) for lang in ("zh", "en")}

    def load(lang):
        plugin.tools, plugin.categories = tools[lang]["tools"], tools[lang]["categories"]
        plugin.formats = tools[lang]["formats"]

    plugin.refresh_tools = lambda done=None: (load(client.get_lang()), done and done(""))
    load("zh")
    read = host.add_object("plate", "image")
    cam = host.add_object("shotcam", "scene.camera")

    from lab2shot_dcc.ui.shell import PanelShell

    for lang in ("zh", "en"):
        plugin.set_language(lang)
        load(lang)
        host.selected = [read]
        panel = PanelShell(plugin)
        panel._selection_changed()
        panel._tick()
        sug = [c.tool_id for c in panel.home.suggested.cards]
        first = next(t for t in plugin.tools if t["id"] == sug[0]) if sug else {}
        delivered = {x for d in first.get("delivers") or [] for x in d.get("types") or []}
        step(f"a Read selected: suggestions, pictures first ({lang})",
             bool(sug) and any(x.startswith(("image", "video")) for x in delivered), [sug[:4], sorted(delivered)])
        shoot(panel, f"nuke_{lang}_01_home_read_selected")

        # a camera and the plate selected; a card whose tool reads both camera files
        host.selected = [cam, read]
        panel._selection_changed()
        panel._tick()
        both = next(t for t in plugin.tools if {"cam_fbx_path", "cam_usd_path"} <= {i["param"] for i in t["inputs"]})
        before = set(host.nodes())
        panel.tool_chosen(both["id"], True)
        pump()
        node = panel.node
        bound = sorted((host.load(node).get("bindings") or {}).keys())
        step(f"suggested card: the camera into cam_usd_path, not cam_fbx_path ({lang})",
             node not in before and "cam_usd_path" in bound and "cam_fbx_path" not in bound and "input" in bound, bound)
        cards = {c.row["item"]["param"]: c for c in panel.tool_page.input_cards}
        fbx_card, usd_card = cards.get("cam_fbx_path"), cards.get("cam_usd_path")
        step(f"「用选中的」 of cam_fbx_path greyed with the reason; cam_usd_path's live ({lang})",
             fbx_card is not None and not fbx_card.use.isEnabled() and ".usda" in fbx_card.use.toolTip()
             and usd_card is not None and usd_card.use.isEnabled(), fbx_card.use.toolTip() if fbx_card else None)
        shoot(panel, f"nuke_{lang}_02_tool_camera_bound")
        if lang == "en":
            bad = has_chinese(panel)
            step("English: no Chinese on the tool page", not bad, bad[:6])

        # a version: its card locates (Nuke frames the Backdrop; it never selects)
        state = host.load(node)
        state["versions"] = [{"version": 1, "job": "j1", "folder": os.path.join(project, "data", "lab2shot", "lab2shot1", "v001"),
                              "group": "g", "namespace": "ns", "objects": ["l2s:a", "l2s:b"], "imported": True,
                              "made": "2026-10-04 09:00:00", "tool": "T"}]
        host.store(node, state)
        panel.tool_page.rebuild()
        panel.tool_page.version_clicked(view_model.versions(host.load(node))[0])
        step(f"version card: located, nothing selected ({lang})",
             host.located and host.located[-1] == ["l2s:a", "l2s:b"] and not host.selected_by_us, host.located[-1:])

        # the script never saved: 「计算」 greyed, the reason its tooltip and status; saved: live again
        host.unsaved_scene = True
        panel.tool_page.tick()
        box = panel.tool_page.run_box
        step(f"unsaved: 「计算」 greyed, 'save the project first' said ({lang})",
             not box.button.isEnabled() and box.button.toolTip() == paths.text("dcc.plugin.save_first")
             and box.status.text() == paths.text("dcc.plugin.save_first"), [box.button.isEnabled(), box.button.toolTip()])
        panel.tool_page.scroll.ensureWidgetVisible(box, 0, 200)
        shoot(panel, f"nuke_{lang}_03_tool_unsaved")
        if lang == "en":
            bad = has_chinese(panel)
            step("English: no Chinese (unsaved)", not bad, bad[:6])
        host.unsaved_scene = False
        panel.tool_page.tick()
        step(f"saved: 「计算」 live ({lang})", box.button.isEnabled() and not box.button.toolTip())
        panel.close()


try:
    main()
except Exception:  # noqa: BLE001
    REPORT["ok"] = False
    REPORT["error"] = traceback.format_exc()
    print(traceback.format_exc())
with open(CFG["report"], "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=1)
print("RESULT", "ok" if REPORT["ok"] else "failed")
