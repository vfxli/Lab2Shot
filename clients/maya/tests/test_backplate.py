"""The backplate behind a delivered camera, in mayapy (no interface, no server): the framework's import step
(lab2shot_dcc.jobs.Run._import) with the Maya host, in the three cases of the rule:

1. a picture is bound: the plate shows it (whatever the bound camera carries);
2. no picture, the bound camera has an image plane: the plate shows the camera's picture;
3. neither: no plate, a sentence saying so (输入不全), the camera still imported, no error.

Every case: the plate is a new image plane on OUR camera, in the version's namespace, under |Lab2Shot|; it steps
through the sequence at the picture's frame = Maya's frame + frame_offset; one Ctrl+Z removes it with the version;
not one write to the user's nodes; on Maya's own nodes only what Maya itself does on creating any colour-managed node
(defaultColorMgtGlobals' four plugs connected to the new plate; its values as they are).

    mayapy test_backplate.py <config.json>

config (UTF-8): {"common": clients/common, "scripts": clients/maya/lab2shot/scripts, "picture_a", "picture_b",
"work": a folder for the delivered data, "report": where the results go (json)}. picture A: one frame of a sequence numbered 259… (the camera's own plate); picture B: one frame of another sequence
(the bound picture).
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# paths come in a UTF-8 file: a Windows command line mangles what is not ASCII (the test material's Chinese names)
with open(sys.argv[1], encoding="utf-8") as _f:
    _cfg = json.load(_f)
COMMON, SCRIPTS, PLATE_A, PLATE_B, WORK, REPORT_PATH = (_cfg[k] for k in ("common", "scripts", "picture_a", "picture_b",
                                                                          "work", "report"))
sys.path[:0] = [COMMON, SCRIPTS]

import maya.standalone  # noqa: E402

maya.standalone.initialize(name="python")
import maya.api.OpenMaya as om  # noqa: E402
import maya.cmds as cmds  # noqa: E402

REPORT: dict = {"steps": [], "ok": True}
FIRST, LAST = 1001, 1010
# Maya connects these plugs of defaultColorMgtGlobals to every colour-managed node it creates (probe: imagePlane
# command, createNode imagePlane, createNode file; colour management on or off; a plane already in the scene or not):
# a connection from the node, its values as they are
MAYA_RESYNC = {("defaultColorMgtGlobals", a) for a in ("cmEnabled", "configFileEnabled", "configFilePath", "workingSpaceName")}


def step(name, ok, detail=""):
    REPORT["steps"].append({"step": name, "ok": bool(ok), "detail": str(detail)})
    REPORT["ok"] = REPORT["ok"] and bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


class Watch:
    """Attribute sets, connections, renames, re-parents, removals on every node there now, but `ours`."""

    def __init__(self, ours=()):
        M = om.MNodeMessage
        self.flags = (M.kAttributeSet | M.kConnectionMade | M.kConnectionBroken | M.kAttributeAdded | M.kAttributeRemoved
                      | M.kAttributeArrayAdded | M.kAttributeArrayRemoved | M.kAttributeLocked | M.kAttributeUnlocked
                      | M.kAttributeRenamed)
        self.events, self.ids = [], []
        sel = om.MSelectionList()
        for n in cmds.ls() or []:
            if n not in ours:
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

    def stop(self):
        for i in self.ids:
            om.MMessage.removeCallback(i)
        return self


def links(uuid: str):
    """A node's long name and connections, leaving out defaultColorMgtGlobals' links to our own nodes (l2s_…
    namespaces): Maya makes those for every colour-managed node it creates."""
    node = cmds.ls(uuid, long=True)[0]
    got = cmds.listConnections(node, plugs=True, connections=True) or []
    pairs = [(got[i], got[i + 1]) for i in range(0, len(got), 2)]
    if node.endswith("defaultColorMgtGlobals"):
        pairs = [p for p in pairs if not p[1].split("|")[-1].startswith("l2s_")]
    return node, sorted(pairs)


def colour_globals():
    return {a: cmds.getAttr("defaultColorMgtGlobals." + a) for _n, a in sorted(MAYA_RESYNC)}


def delivered_camera() -> str:
    """A delivered camera as import_worker.py leaves it: scene_data.dump of a keyed camera, in a file."""
    from lab2shot_maya import scene_data

    cmds.file(new=True, force=True)
    cam, _shape = cmds.camera(name="solvedCam", focalLength=40.0)
    cmds.setKeyframe(cam, attribute="translateX", time=FIRST - 742, value=0.0)  # at the picture's frames (259…)
    cmds.setKeyframe(cam, attribute="translateX", time=LAST - 742, value=30.0)
    data = scene_data.dump([cmds.ls(cam, long=True)[0]])
    path = os.path.join(WORK, "solved_scene.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def case(label, bind_picture, camera_plane, scene_json):
    from lab2shot_dcc import jobs, paths
    from lab2shot_maya.host import MayaHost

    cmds.file(new=True, force=True)
    cmds.currentUnit(time="film")
    cmds.playbackOptions(minTime=FIRST, maxTime=LAST)
    cmds.currentTime(FIRST)
    # the user's scene: a camera, with or without its own plate (frames 259… at Maya's 1001…)
    cam, cam_shape = cmds.camera(name="shotCam")
    if camera_plane:
        plane = cmds.imagePlane(camera=cam_shape, fileName=PLATE_A)
        plane_shape = plane[1] if len(plane) > 1 else plane[0]
        cmds.setAttr(plane_shape + ".useFrameExtension", True)
        cmds.setAttr(plane_shape + ".frameOffset", 259 - FIRST)
    host = MayaHost()
    node = host.create_node("lab2shot1")
    bindings = {"cam_fbx_path": {"ref": cmds.ls(cam, uuid=True)[0], "path": cmds.ls(cam, long=True)[0],
                                 "type": "scene.camera", "label": "shotCam"}}
    if bind_picture:  # a picture picked from disk (「选文件」): another sequence, its own frame numbers
        bindings["input"] = {"file": PLATE_B, "label": PLATE_B, "type": "image"}
    host.store(node, {"bindings": bindings})
    user = cmds.ls(cmds.ls(long=True), uuid=True)
    before = {u: links(u) for u in user}
    globals_before = colour_globals()
    watch = Watch({host.node_name(node)})
    run = jobs.Run(host, None, node, {"name": "test"}, job="test-job")
    items = [{"type": "scene.camera", "kinds": ["camera"], "main": scene_json, "files": [scene_json], "name": "scene"}]
    version_dir = os.path.join(WORK, label)
    os.makedirs(version_dir, exist_ok=True)
    try:
        entry = run._import(1, version_dir, "lab2shot1", items, host.load(node))
        error = ""
    except Exception as exc:  # noqa: BLE001
        entry, error = {}, f"{exc}\n{traceback.format_exc()}"
    watch.stop()
    step(f"[{label}] import ran without error", not error, error)
    if error:
        return
    group = cmds.ls(entry["group"], long=True)[0]
    cams = cmds.listRelatives(group, allDescendents=True, type="camera", fullPath=True) or []
    # a plane hangs in its camera's underworld (|cam|camShape->|plane), which listRelatives does not walk
    planes = sorted({cmds.ls(p, long=True)[0] for c in cams
                     for p in cmds.listConnections(c + ".imagePlane", source=True, destination=False, shapes=True) or []})
    step(f"[{label}] delivered camera under |Lab2Shot|", len(cams) == 1, str(cams))
    others = [e for e in watch.events if e not in MAYA_RESYNC]
    step(f"[{label}] zero writes to nodes that were there (Maya's own colour links to a new plate aside)", not others,
         f"{others[:6]}; resync {sorted(set(watch.events) & MAYA_RESYNC)}")
    step(f"[{label}] defaultColorMgtGlobals values unchanged", colour_globals() == globals_before,
         f"{globals_before} -> {colour_globals()}")
    after = {u: links(u) for u in user if cmds.ls(u)}
    step(f"[{label}] nodes that were there: names and connections unchanged (but Maya's colour links to our plate)",
         after == before,
         str([u for u in before if before[u] != after.get(u)][:5]))
    if label == "none":
        step("[none] no plate, the camera still came in", not planes and cams, str(planes))
        step("[none] said why (输入不全), no error", entry.get("note") == paths.text(jobs.NO_BACKPLATE) and not entry.get("backplate"),
             entry.get("note"))
    else:
        want_file, want_offset = (PLATE_B, 0) if label == "input" else (PLATE_A, 259 - FIRST)
        p = planes[0] if planes else ""
        ok = len(planes) == 1
        step(f"[{label}] one new plate, in the version's namespace", ok and p.split("|")[-1].startswith(entry["namespace"] + ":"),
             str(planes))
        if not ok:
            return
        step(f"[{label}] hung on OUR camera, under |Lab2Shot|", p.startswith(cams[0] + "->"), p)
        step(f"[{label}] shows the {'bound picture' if label == 'input' else 'camera’s own picture'}",
             cmds.getAttr(p + ".imageName") == want_file, cmds.getAttr(p + ".imageName"))
        seq = cmds.getAttr(p + ".useFrameExtension")
        offset = cmds.getAttr(p + ".frameOffset")
        ext = [cmds.getAttr(p + ".frameExtension", time=t) for t in (FIRST, LAST, 1500)]
        step(f"[{label}] steps through the sequence: picture frame = Maya frame + {want_offset}",
             seq and offset == want_offset and ext == [FIRST, LAST, 1500], f"use {seq}, offset {offset}, frameExtension {ext}")
        step(f"[{label}] the version records where its plate came from",
             entry.get("backplate", {}).get("from") == label and not entry.get("note"),
             f"{entry.get('backplate')} {entry.get('note')}")
        curve = cmds.listConnections(p + ".frameExtension", source=True, destination=False, plugs=True) or []
        step(f"[{label}] frameExtension driven by our own curve (no link to time1)",
             len(curve) == 1 and curve[0].startswith(entry["namespace"] + ":"), str(curve))
    # one Ctrl+Z: the version and its plate gone
    cmds.undo()
    left = cmds.ls(type="imagePlane", long=True) or []
    mine = [n for n in left if "Lab2Shot" in n or ":" in n.split("|")[-1]]
    step(f"[{label}] one undo removes the version with its plate", not cmds.ls(entry["group"]) and not mine, str(mine))
    cmds.redo()  # redo makes the version again from the files on disk (lab2shotUndoStep: new nodes, new UUIDs)
    again = [n for n in cmds.ls(type="imagePlane", long=True) or [] if n.startswith("|Lab2Shot|")]
    step(f"[{label}] redo brings the version back with {'its plate' if label != 'none' else 'no plate'}",
         cmds.ls("|Lab2Shot|" + entry["group_name"]) and len(again) == (0 if label == "none" else 1), str(again))
    cmds.undo()


def main():
    from lab2shot_maya import undo  # noqa: F401 - the undo plug-in loads on first use

    cmds.undoInfo(state=True, infinity=True)
    os.makedirs(WORK, exist_ok=True)
    scene_json = delivered_camera()
    for label, picture, own in (("input", True, True), ("camera", False, True), ("none", False, False)):
        case(label, picture, own, scene_json)


try:
    main()
except Exception:  # noqa: BLE001
    step("test ran", False, traceback.format_exc())
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=1)
print("ALL OK" if REPORT["ok"] else "FAILED", flush=True)
cmds.file(new=True, force=True)
maya.standalone.uninitialize()
