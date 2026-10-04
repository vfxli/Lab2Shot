"""The panel (lab2shot_dcc.ui) drawn off screen with the fake host and a recorded tool list: pictures of the home page
and the tool page in their states, at three widths, in both languages; and the checks a picture cannot make:

- English: no Chinese in any label, button or tool button (the user's own data — node names, file names — excepted:
  objectName L2SUserText);
- every button says something (text or tooltip);
- the look is only on our own root (no style sheet on any other widget; QApplication's style sheet untouched);
- the language changes while the panel is open: rebuilt in it, the page and the node kept.

Any Python with PySide6 (or PySide2) runs it — a DCC's own (mayapy, Nuke's python) without its DCC:

    QT_QPA_PLATFORM=offscreen mayapy panel_shots.py <config.json>

config (UTF-8): {"tools_zh", "tools_en": GET /api/tools recorded in each language, "out": the folder for the pictures,
"report": where the results go (json), "server": the address the recorded list came from}.
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
sys.path[:0] = [os.path.dirname(HERE), HERE]
os.environ["USERPROFILE"] = tempfile.mkdtemp(prefix="l2s_shots_")  # the plugin's settings: this run's own
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lab2shot_dcc.qt import QtCore, QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
# Qt's offscreen platform reads single font files only and falls back to whatever face it found first: the system's
# collection (.ttc) the panel's fonts live in is added here, and the system's UI face made the fallback, as the DCC's
# own Qt (the windows platform) has them
from lab2shot_dcc.qt import QtGui  # noqa: E402

for _ttc in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\segoeui.ttf",
             r"C:\Windows\Fonts\segoeuib.ttf"):
    if os.path.isfile(_ttc):
        QtGui.QFontDatabase.addApplicationFont(_ttc)
if "Segoe UI" in QtGui.QFontDatabase.families():
    app.setFont(QtGui.QFont("Segoe UI", 9))
APP_SHEET = app.styleSheet()

from fake_host import FakeHost  # noqa: E402
from lab2shot_dcc import paths, view_model  # noqa: E402
from lab2shot_dcc.plugin import Plugin  # noqa: E402

REPORT: dict = {"steps": [], "ok": True, "shots": []}
OUT = CFG["out"]
os.makedirs(OUT, exist_ok=True)


def step(name, ok, detail=""):
    REPORT["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:2000]})
    REPORT["ok"] = REPORT["ok"] and bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {str(detail)[:300]}" if detail and not ok else ""), flush=True)


class ShotHost(FakeHost):
    """The fake host with a selection the panel can read, and the hooks the panel uses."""

    prefers = ("scene", "image", "video")

    def __init__(self, project):
        super().__init__(project)
        self.selected_lab2shot = None
        self.located = []
        self.label = "Maya"

    def selected_node(self):
        return self.selected_lab2shot

    def why_not(self, data_type):
        return "Select a camera first" if paths.client_module().get_lang() == "en" else "先选中一台摄影机"

    def locate(self, refs):
        self.located.append(list(refs))


class FakeRun:
    def __init__(self, snap):
        self.snap = dict(snap)

    def snapshot(self):
        return dict(self.snap, job=self.snap.get("job", "a1b2c3"))

    @property
    def finished(self):
        return bool(self.snap.get("done"))

    def cancel(self):
        pass


def pump(ms=60):
    end = QtCore.QElapsedTimer()
    end.start()
    while end.elapsed() < ms:
        app.processEvents(QtCore.QEventLoop.AllEvents, 20)


def shoot(panel, name, width=460, height=900):
    panel.resize(width, height)
    panel.show()  # offscreen: nothing appears anywhere
    pump(120)
    path = os.path.join(OUT, f"{name}.png")
    ok = panel.grab().save(path)
    REPORT["shots"].append(path)
    step(f"shot {name}", ok, path)


def texts(root):
    out = []
    for w in [root, *root.findChildren(QtWidgets.QWidget)]:
        if not w.isVisibleTo(root) and w is not root:
            continue
        if w.objectName() == "L2SUserText":
            continue
        t = ""
        if isinstance(w, (QtWidgets.QLabel, QtWidgets.QAbstractButton)):
            t = w.text()
        elif isinstance(w, QtWidgets.QLineEdit):
            t = w.placeholderText()
        out.append((w, t, w.toolTip()))
    return out


def no_chinese(root, where):
    bad = []
    for w, t, tip in texts(root):
        for s in (t, "" if w.property("user_tip") else tip):
            if view_model.has_cjk(s):
                bad.append(f"{type(w).__name__}#{w.objectName()}: {s[:60]}")
    for menu in root.findChildren(QtWidgets.QMenu):
        if menu.objectName() == "L2SLanguages":  # 中文 / English: each language named in itself
            continue
        for a in menu.actions():
            if view_model.has_cjk(a.text()):
                bad.append(f"menu: {a.text()}")
    step(f"English: no Chinese ({where})", not bad, bad[:8])


def buttons_say(root, where):
    mute = [f"{type(b).__name__}#{b.objectName()}" for b in root.findChildren(QtWidgets.QAbstractButton)
            if b.isVisibleTo(root) and not b.text().strip() and not b.toolTip().strip()]
    step(f"every button says what it does ({where})", not mute, mute[:8])


def sheets_only_on_root(root, where):
    others = [f"{type(w).__name__}#{w.objectName()}" for w in root.findChildren(QtWidgets.QWidget) if w.styleSheet()]
    step(f"the look only on our root ({where})", not others and bool(root.styleSheet()) and app.styleSheet() == APP_SHEET,
         others[:8])


def main():
    project = tempfile.mkdtemp(prefix="l2s_shots_project_")
    host = ShotHost(project)
    client = paths.client_module()
    server = CFG.get("server", "http://127.0.0.1:8780")
    with open(client.tokens_file(), "w", encoding="utf-8") as f:
        json.dump({server: {"token": "x", "username": "admin"}}, f)
    from lab2shot_dcc import connection

    connection.write_settings(server=server, lang="zh")
    plugin = Plugin(host)
    plugin.conn.user = {"username": "admin"}
    tools = {lang: json.load(open(CFG[f"tools_{lang}"], encoding="utf-8")) for lang in ("zh", "en")}

    def load(lang):
        plugin.tools, plugin.categories = tools[lang]["tools"], tools[lang]["categories"]
        plugin.formats = tools[lang]["formats"]

    plugin.refresh_tools = lambda done=None: (load(client.get_lang()), done and done(""))  # the recorded list
    load("zh")

    # the scene: a camera with an image plane, a character, two Lab2Shot nodes of earlier work
    cam = host.add_object("shotCam1", "scene.camera")
    plate = host.add_object("imagePlaneShape1", "image")
    host.add_object("RL_BoneRoot", "scene.character")
    old = plugin.new_node()
    depth = next(t for t in plugin.tools if t["source"] == "preset" and any(i["type"].startswith("image") for i in t["inputs"])
                 and len(t["inputs"]) == 1)
    plugin.set_tool(old, depth)

    from lab2shot_dcc.ui.shell import PanelShell

    for lang in ("zh", "en"):
        plugin.set_language(lang)
        load(lang)
        host.selected = []
        panel = PanelShell(plugin)
        tag = f"{lang}"
        shoot(panel, f"{tag}_01_home_460")
        if lang == "en":
            no_chinese(panel, "home")
        buttons_say(panel, f"home {lang}")
        sheets_only_on_root(panel, f"home {lang}")
        shoot(panel, f"{tag}_02_home_360", 360)
        shoot(panel, f"{tag}_03_home_720", 720)
        host.selected = [cam, plate]
        panel._selection_changed()
        panel._tick()
        sug = panel.home.suggested.cards
        step(f"camera with a plane selected: suggestions ({lang})", bool(sug) and panel.home.suggest_box.isVisibleTo(panel),
             [c.tool_id for c in sug])
        shoot(panel, f"{tag}_04_home_selected")
        panel.home.search.setText("depth" if lang == "en" else "深度")
        pump(300)
        step(f"search filters the grid ({lang})", 0 < len(panel.home.cards.cards) < len(plugin.tools),
             len(panel.home.cards.cards))
        shoot(panel, f"{tag}_05_home_search")
        panel.home.search.setText("")
        pump(300)

        # a card of 「适合当前选中」: a new node (the old one has a tool), what is selected bound to the inputs it fits
        sug = panel.home.suggested.cards
        before = set(host.nodes())
        target = next(c for c in sug if any(i["type"] == "scene.camera" for i in
                                            next(t for t in plugin.tools if t["id"] == c.tool_id)["inputs"])) if any(
            any(i["type"] == "scene.camera" for i in next(t for t in plugin.tools if t["id"] == c.tool_id)["inputs"])
            for c in sug) else sug[0]
        target.clicked.emit()
        pump(100)
        made = set(host.nodes()) - before
        node = panel.node
        state = host.load(node)
        step(f"suggested card: a new node, its tool, the selection bound ({lang})",
             len(made) == 1 and node in made and state.get("tool", {}).get("id") == target.tool_id
             and bool(state.get("bindings")), f"{made} {state.get('bindings')}")
        step(f"tool page shown ({lang})", panel.page == "tool")
        shoot(panel, f"{tag}_06_tool_bound")
        if lang == "en":
            no_chinese(panel, "tool page")
        buttons_say(panel, f"tool page {lang}")
        sheets_only_on_root(panel, f"tool page {lang}")
        if panel.tool_page.advanced is not None:
            panel.tool_page.advanced.set_open(True)
            shoot(panel, f"{tag}_07_tool_advanced_open")
            panel.tool_page.advanced.set_open(False)

        # computing, failed, versions
        plugin.runs[node] = FakeRun({"phase": "running", "text": "计算：深度" if lang == "zh" else "Cooking: Depth",
                                     "fraction": 0.42, "error": "", "done": False})
        panel._tick()
        step(f"computing: the button cancels ({lang})",
             panel.tool_page.run_box.button.text() == paths.text("dcc.ui.run.cancel"))
        panel.tool_page.scroll.ensureWidgetVisible(panel.tool_page.run_box, 0, 200)
        shoot(panel, f"{tag}_08_tool_computing")
        plugin.runs[node] = FakeRun({"phase": "failed", "text": paths.text("dcc.job.failed"),
                                     "error": "CUDA out of memory (8 GB asked, 2 GB free)", "fraction": None, "done": True})
        panel._tick()
        step(f"failed: the server's words, the log button ({lang})",
             "CUDA" in panel.tool_page.run_box.status.text() and panel.tool_page.run_box.log_button.isVisibleTo(panel))
        panel.tool_page.scroll.ensureWidgetVisible(panel.tool_page.run_box, 0, 200)
        shoot(panel, f"{tag}_09_tool_failed")
        plugin.runs.pop(node)
        s = host.load(node)
        s["versions"] = [{"version": n, "made": f"2026-10-0{n} 1{n}:20:00", "tool": target.tool_id, "imported": n != 2,
                          "objects": [cam], "folder": project} for n in (1, 2, 3)]
        host.store(node, s)
        panel.tool_page.rebuild()
        shoot(panel, f"{tag}_10_tool_whole", 460, 2400)
        panel.tool_page.scroll.verticalScrollBar().setValue(panel.tool_page.scroll.verticalScrollBar().maximum())
        shoot(panel, f"{tag}_10_tool_versions")
        shoot(panel, f"{tag}_11_tool_720", 720)
        cards = panel.tool_page.findChildren(QtWidgets.QFrame, "L2SVersion")
        cards[0].clicked.emit()
        step(f"an imported version: located in the scene ({lang})", host.located and host.located[-1] == [cam])

        # a Lab2Shot node selected in the scene: the panel goes to it
        host.selected_lab2shot = old
        panel._selection_changed()
        panel._tick()
        step(f"selecting a Lab2Shot node switches to it ({lang})", panel.node == old and panel.page == "tool")
        host.selected_lab2shot = None
        panel.go_home()
        step(f"home: the scene's tasks ({lang})", panel.home.tasks_box.isVisibleTo(panel)
             and panel.home.tasks_flow.count() == len(host.nodes()))
        shoot(panel, f"{tag}_12_home_tasks")
        grid = panel.home.cards
        step(f"home again: every card shown ({lang})", len(grid.cards) == len(plugin.tools) and grid.cards[0].isVisibleTo(panel)
             and grid.height() > 100, f"{len(grid.cards)} {grid.geometry()} {panel.home.tab!r} {panel.home.search_text!r}")
        panel.close()
        panel.deleteLater()
        pump(50)

    # the language changed with the panel open: rebuilt in it, where it was kept
    plugin.set_language("zh")
    load("zh")
    panel = PanelShell(plugin)
    panel.resize(460, 900)
    panel.show()
    panel.open_node(old)
    pump(100)
    plugin.set_language("en")
    pump(200)
    step("language switch: same node, same page", panel.node == old and panel.page == "tool")
    no_chinese(panel, "after switching the language")
    shoot(panel, "switch_en_tool")
    panel.close()


try:
    main()
except Exception:  # noqa: BLE001
    step("ran through", False, traceback.format_exc())
with open(CFG["report"], "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=1)
print("ALL OK" if REPORT["ok"] else "FAILED", flush=True)
