"""The new panel (lab2shot_dcc.ui.shell) on the Maya host in mayapy — no Maya window, Qt off screen — against a running
Lab2Shot server: the panel's own path from the selection to an imported result, and not one write to the user's nodes.

    QT_QPA_PLATFORM=offscreen mayapy test_panel.py <config.json>

config (UTF-8: a Windows command line mangles what is not ASCII): {"common": clients/common, "scripts":
clients/maya/lab2shot/scripts, "server", "account": {"username", "password"}, "plates": a folder holding one picture
sequence (its frames are bound twice: through a camera's image plane, then 「选序列」 picking them all), "tool": the
tool id the suggested card is (a tool that takes a picture and delivers a camera), "project": the Maya project folder,
"out": the folder for the pictures, "report": where the results go (json), "timeout": seconds the job may take}.

What it proves: an image plane selected → the tool among 「适合当前选中」; its card makes a Lab2Shot node
and binds the plane; 「选序列」 binds the whole sequence instead (every frame, the count shown); 「计算」 from the panel →
submitted, followed, fetched, imported as v001 under |Lab2Shot| (its manifest names each output's colour space); the
version strip shows it and a click selects it; selecting the Lab2Shot node switches the panel to it; the language
switched to English → no Chinese in the panel; through all of it no attribute, connection, name or parent of a node
that was in the scene changes (the only exception Maya's own: defaultColorMgtGlobals connected to a new colour-managed
node of ours, its values as they are), and every storable attribute of the user's nodes is the same after.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

with open(sys.argv[1], encoding="utf-8") as _f:
    CFG = json.load(_f)
sys.path[:0] = [CFG["common"], CFG["scripts"]]
os.environ["USERPROFILE"] = tempfile.mkdtemp(prefix="l2s_panel_")  # the plugin's settings and token: this run's own
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from lab2shot_dcc.qt import QtCore, QtGui, QtWidgets  # noqa: E402

# made before Maya: maya.standalone otherwise makes a QGuiApplication of its own, on which no widget can live
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

import maya.standalone  # noqa: E402

maya.standalone.initialize(name="python")
import maya.api.OpenMaya as om  # noqa: E402
import maya.cmds as cmds  # noqa: E402

for _font in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\segoeui.ttf"):
    if os.path.isfile(_font):  # Qt's offscreen platform reads no font collection by itself (screenshots only)
        QtGui.QFontDatabase.addApplicationFont(_font)
if "Segoe UI" in QtGui.QFontDatabase.families():
    app.setFont(QtGui.QFont("Segoe UI", 9))

REPORT: dict = {"steps": [], "ok": True}
if CFG.get("stacks"):  # where it hangs, if it does: every thread's stack into this file, every few minutes
    import faulthandler

    _stacks = open(CFG["stacks"], "w", encoding="utf-8")
    faulthandler.dump_traceback_later(int(CFG.get("stacks_every", 120)), repeat=True, file=_stacks)
FIRST, LAST = 1001, 1010
MAYA_RESYNC = {("defaultColorMgtGlobals", a) for a in ("cmEnabled", "configFileEnabled", "configFilePath", "workingSpaceName")}


def step(name, ok, detail=""):
    REPORT["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)[:3000]})
    REPORT["ok"] = REPORT["ok"] and bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {str(detail)[:400]}" if detail and not ok else ""), flush=True)


class Watch:
    """Attribute sets, connections, renames, re-parents, removals on every node there now."""

    def __init__(self):
        M = om.MNodeMessage
        self.flags = (M.kAttributeSet | M.kConnectionMade | M.kConnectionBroken | M.kAttributeAdded | M.kAttributeRemoved
                      | M.kAttributeArrayAdded | M.kAttributeArrayRemoved | M.kAttributeLocked | M.kAttributeUnlocked
                      | M.kAttributeRenamed)
        self.events, self.ids = [], []
        sel = om.MSelectionList()
        for n in cmds.ls() or []:
            try:
                sel.add(n)
            except RuntimeError:
                pass
        for i in range(sel.length()):
            obj = sel.getDependNode(i)
            name = om.MFnDependencyNode(obj).name()

            def changed(msg, plug, other, data, name=name):
                if msg & self.flags:
                    self.events.append((name, plug.partialName(useLongNames=True)))

            self.ids.append(om.MNodeMessage.addAttributeChangedCallback(obj, changed))
            self.ids.append(om.MNodeMessage.addNameChangedCallback(obj, lambda n, p, d, name=name: self.events.append((name, "rename"))))
            self.ids.append(om.MNodeMessage.addNodePreRemovalCallback(obj, lambda n, d, name=name: self.events.append((name, "removed"))))
            dag = om.MDagPath.getAPathTo(obj) if obj.hasFn(om.MFn.kDagNode) else None
            if dag is not None:
                self.ids.append(om.MDagMessage.addParentAddedDagPathCallback(
                    dag, lambda c, p, d, name=name: self.events.append((name, "parent"))))

    def stop(self):
        for i in self.ids:
            om.MMessage.removeCallback(i)
        return self


def snapshot(uuids):
    """Every storable attribute a node holds itself (not driven by a connection), its name, parent and connections."""
    out = {}
    for u in uuids:
        node = (cmds.ls(u, long=True) or [None])[0]
        if node is None:
            out[u] = None
            continue
        attrs = {}
        for a in cmds.listAttr(node, write=True) or []:
            try:
                if not cmds.attributeQuery(a.split(".")[-1], node=node, storable=True):
                    continue
                if cmds.connectionInfo(f"{node}.{a}", isDestination=True):
                    continue
                attrs[a] = repr(cmds.getAttr(f"{node}.{a}"))[:2000]
            except Exception:  # noqa: BLE001 - compound / message / multi parents have no single value
                continue
        conns = cmds.listConnections(node, plugs=True, connections=True) or []
        pairs = sorted(f"{conns[i]}<->{conns[i + 1]}" for i in range(0, len(conns), 2)
                       if not conns[i + 1].split("|")[-1].split(".")[0].startswith(("l2s_", "backplate")))
        out[u] = {"name": node, "parent": (cmds.listRelatives(node, parent=True, fullPath=True) or [""])[0],
                  "attrs": attrs, "conns": pairs}
    return out


def main():
    os.makedirs(CFG["out"], exist_ok=True)
    os.makedirs(CFG["project"], exist_ok=True)
    cmds.workspace(CFG["project"], openWorkspace=True)
    cmds.file(new=True, force=True)
    cmds.currentUnit(time="film")
    cmds.playbackOptions(minTime=FIRST, maxTime=LAST)
    cmds.currentTime(FIRST)
    frames = sorted(f for f in os.listdir(CFG["plates"]) if f.lower().endswith((".jpg", ".png", ".exr")))
    first_number = int("".join(ch for ch in os.path.splitext(frames[0])[0] if ch.isdigit())[-5:])
    # the user's scene: a camera carrying the plate as an image plane, and a model of theirs
    cam, cam_shape = cmds.camera(name="shotCam")
    plane = cmds.imagePlane(camera=cam_shape, fileName=os.path.join(CFG["plates"], frames[0]))
    plane_shape = plane[1] if len(plane) > 1 else plane[0]
    cmds.setAttr(plane_shape + ".useFrameExtension", True)
    cmds.setAttr(plane_shape + ".frameOffset", first_number - FIRST)
    cmds.polyCube(name="userBox")
    cmds.select(clear=True)
    user = cmds.ls(cmds.ls(long=True), uuid=True)
    before = snapshot(user)
    watch = Watch()

    from lab2shot_dcc import paths, view_model
    from lab2shot_dcc.plugin import Plugin
    from lab2shot_dcc.ui.shell import PanelShell
    from lab2shot_maya.host import MayaHost

    host = MayaHost()
    plugin = Plugin(host)
    said: dict = {}

    def pump_until(cond, seconds):
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents(QtCore.QEventLoop.AllEvents, 20)
            host.pump(0.02)
            if cond():
                return True
        return False

    plugin.login(CFG["server"], CFG["account"]["username"], CFG["account"]["password"],
                 lambda got, error: said.update(login=error))
    pump_until(lambda: "login" in said, 60)
    step("login", said.get("login") == "", said.get("login"))
    from lab2shot_dcc import connection

    connection.write_settings(lang="zh")
    plugin.lang = connection.apply_language(host)
    plugin.refresh_tools(lambda error: said.update(tools=error))
    pump_until(lambda: "tools" in said, 600)
    step("tool list", not said.get("tools") and len(plugin.tools) > 10, said.get("tools") or len(plugin.tools))

    panel = PanelShell(plugin)
    panel.resize(460, 900)
    panel.show()  # offscreen: no window anywhere
    pump_until(lambda: False, 0.3)

    def shoot(name, height=900):
        panel.resize(460, height)
        pump_until(lambda: False, 0.3)
        path = os.path.join(CFG["out"], f"maya_{name}.png")
        step(f"shot {name}", panel.grab().save(path), path)

    shoot("01_home")
    cmds.select(cam)  # the user selects their camera (it carries the plate): what takes both comes first
    panel._selection_changed()
    pump_until(lambda: False, 0.5)
    sug = [c.tool_id for c in panel.home.suggested.cards]
    step("camera with an image plane selected: suggestions", len(sug) > 0, sug)
    cmds.select(plane_shape)  # the user selects the plate itself
    panel._selection_changed()
    pump_until(lambda: False, 0.5)
    sug = [c.tool_id for c in panel.home.suggested.cards]
    step("the image plane selected: the tool among 「适合当前选中」", CFG["tool"] in sug, sug)
    shoot("02_home_selected")
    card = next(c for c in panel.home.suggested.cards if c.tool_id == CFG["tool"])
    card.clicked.emit()
    pump_until(lambda: False, 0.5)
    node = panel.node
    state = host.load(node) if node else {}
    bound = (state.get("bindings") or {}).get("input") or {}
    step("its card: a Lab2Shot node with the tool, the image plane bound", node in host.nodes()
         and state.get("tool", {}).get("id") == CFG["tool"] and bound.get("ref") and bound.get("sequence"), bound)
    pump_until(lambda: any(str(len(frames)) in w.text() for w in panel.tool_page.findChildren(QtWidgets.QLabel, "L2SUserText")),
               10)
    shown = [w.text() for w in panel.tool_page.findChildren(QtWidgets.QLabel, "L2SUserText")]
    step("the plane's sequence: first, last and how many (scanned off the main thread)",
         any(str(len(frames)) in s and frames[0] in s for s in shown), shown)
    shoot("03_tool_plane_bound")

    # 「选序列」: every frame of the folder picked together
    picked = [os.path.join(CFG["plates"], f) for f in frames]
    QtWidgets.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (picked, ""))
    item = next(c.row["item"] for c in panel.tool_page.input_cards if c.row["param"] == "input")
    panel.tool_page.bind_file(item)
    bound = host.load(node)["bindings"]["input"]
    step("「选序列」: the whole sequence bound", len(bound.get("files") or []) == len(frames), bound.get("label"))
    shown = [w.text() for w in panel.tool_page.findChildren(QtWidgets.QLabel, "L2SUserText")]
    step("the picked sequence said: first … last · count", any(str(len(frames)) in s for s in shown), shown)

    # the scene never saved: 「计算」 greyed with the reason, the plugin refuses (results go next to the saved
    # scene only); saved: it can be pressed
    panel.tool_page.tick()
    button = panel.tool_page.run_box.button
    refused = ""
    try:
        plugin.compute(node)
    except RuntimeError as exc:
        refused = str(exc)
    step("unsaved scene: 「计算」 greyed, the reason said, compute refused",
         not button.isEnabled() and button.toolTip() == paths.text("dcc.plugin.save_first") and refused == button.toolTip(),
         [button.isEnabled(), button.toolTip(), refused])
    # the user saves: what Maya itself writes on a save (defaultRenderGlobals.defaultSurfaceShader …) is the user's
    # doing, not ours — the watch and the snapshot start again after it
    os.makedirs(os.path.join(CFG["project"], "scenes"), exist_ok=True)
    cmds.file(rename=os.path.join(CFG["project"], "scenes", "shot_v001.ma"))
    seen = len(watch.events)
    cmds.file(save=True, type="mayaAscii")
    REPORT["written_by_the_save"] = sorted({str(e[:2]) for e in watch.events[seen:]})
    del watch.events[seen:]
    before = snapshot(user)
    panel.tool_page.tick()
    step("saved: 「计算」 can be pressed", button.isEnabled() and not button.toolTip(), cmds.file(q=True, sceneName=True))

    # 「计算」 from the panel
    panel.tool_page.run_box.button.click()
    pump_until(lambda: plugin.busy(node) and plugin.runs[node].snapshot()["phase"] in ("queued", "running"), 300)
    panel.tool_page.scroll.ensureWidgetVisible(panel.tool_page.run_box, 0, 200)
    shoot("04_tool_computing")
    run = plugin.runs[node]
    pump_until(lambda: run.finished, CFG.get("timeout", 1800))
    snap = run.snapshot()
    step("cooked, fetched and imported from the panel", snap["done"] and not snap["error"] and snap.get("version"), snap)
    pump_until(lambda: False, 1.0)
    state = host.load(node)
    versions = state.get("versions") or []
    label = f"v{int(snap.get('version') or 0):03d}"
    group = cmds.ls(f"|Lab2Shot|{host.node_name(node)}_{label}", long=True)
    step(f"{label} under |Lab2Shot|, imported", versions and versions[-1]["imported"] and bool(group), [versions, group])
    folder = versions[-1]["folder"] if versions else ""
    try:
        manifest = json.load(open(os.path.join(folder, "lab2shot.json"), encoding="utf-8"))
        spaces = [o.get("colorspace") for o in manifest.get("outputs") or []]
        step("the delivery's manifest: each output's colour space", all("colorspace" in o for o in manifest["outputs"]), spaces)
    except Exception as exc:  # noqa: BLE001
        step("the delivery's manifest: each output's colour space", False, exc)
    cards = panel.tool_page.findChildren(QtWidgets.QFrame, "L2SVersion")
    step("the version strip shows it", len(cards) == len(versions) and cards[0].v["label"] == label, len(cards))
    panel.tool_page.scroll.verticalScrollBar().setValue(panel.tool_page.scroll.verticalScrollBar().maximum())
    shoot("05_tool_done_versions")
    cards = panel.tool_page.findChildren(QtWidgets.QFrame, "L2SVersion")
    if cards:
        cards[0].clicked.emit()
        sel = cmds.ls(selection=True, uuid=True) or []
        step("a version clicked: its objects selected", set(versions[-1]["objects"]) & set(sel), sel)

    # the Lab2Shot node selected in the scene: the panel goes to it (from the home page)
    panel.go_home()
    cmds.select(host.node_name(node), replace=True, noExpand=True)
    panel._selection_changed()
    pump_until(lambda: panel.page == "tool", 2)
    step("selecting the Lab2Shot node switches the panel to it", panel.page == "tool" and panel.node == node)

    plugin.set_language("en")  # the panel is built again at once, and again once the tools come back in English
    pump_until(lambda: plugin.loading, 2)
    pump_until(lambda: not plugin.loading, 600)
    pump_until(lambda: False, 0.5)
    bad = []
    for w in panel.findChildren(QtWidgets.QWidget):
        if not w.isVisibleTo(panel) or w.objectName() == "L2SUserText":
            continue
        for s in ((w.text() if isinstance(w, (QtWidgets.QLabel, QtWidgets.QAbstractButton)) else ""),
                  "" if w.property("user_tip") else w.toolTip()):
            if view_model.has_cjk(s):
                bad.append(f"{type(w).__name__}#{w.objectName()}: {s[:50]}")
    step("English: no Chinese in the panel", not bad, bad[:8])
    shoot("06_tool_english", 1400)
    plugin.set_language("zh")

    watch.stop()
    ours = {"defaultColorMgtGlobals"}
    writes = sorted({e for e in watch.events if not (e[0] in ours and (e[0], e[1].split("[")[0]) in MAYA_RESYNC)})
    step("not one write to a node that was in the scene (Maya's own colour-management links aside)", not writes, writes[:20])
    after = snapshot(user)
    changed = [u for u in user if before.get(u) != after.get(u) and not (after.get(u) or {}).get("name", "").endswith(
        "defaultColorMgtGlobals")]
    detail = []
    for u in changed[:5]:
        b, a = before.get(u) or {}, after.get(u) or {}
        diff = {k: (b.get("attrs", {}).get(k), a.get("attrs", {}).get(k)) for k in set(b.get("attrs", {})) | set(a.get("attrs", {}))
                if b.get("attrs", {}).get(k) != a.get("attrs", {}).get(k)}
        detail.append({"node": a.get("name") or b.get("name"), "attrs": dict(list(diff.items())[:5]),
                       "conns": sorted(set(a.get("conns", [])) ^ set(b.get("conns", [])))[:5],
                       "name": (b.get("name"), a.get("name")), "parent": (b.get("parent"), a.get("parent"))})
    step(f"every attribute of the user's {len(user)} nodes the same after", not changed, detail)
    panel.close()
    pump_until(lambda: False, 0.2)


try:
    main()
except Exception:  # noqa: BLE001
    step("ran through", False, traceback.format_exc())
with open(CFG["report"], "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=1)
print("ALL OK" if REPORT["ok"] else "FAILED", flush=True)
maya.standalone.uninitialize()
