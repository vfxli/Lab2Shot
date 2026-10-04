"""The Maya host end to end in mayapy (no interface), against a running Lab2Shot server.

    mayapy test_maya.py <config.json>

config: {"scripts": the deployed lab2shot/scripts folder, "server", "account": {"username", "password"},
"character": a character FBX (the user's), "plate": one frame of a picture sequence, "plate_first": its first frame
number, "project": the Maya project folder to use, "tool": the tool id, "frames": [first, last] of Maya's playback,
"report": where to write the results (json)}.

What it proves (安全守则 and §9): bind a picture (an image plane), a camera and a character from the scene; submit;
the result comes in as new objects under |Lab2Shot|<node>_vNNN in namespace l2s_<node>_vNNN; every attribute,
connection, name and parent of the user's objects is the same before and after (and after every other step);
the scene's unit / up axis / frame rate, current frame and selection are unchanged; one Ctrl+Z removes the whole
import; names are numbered, never reused; cancelling and a dropped line leave nothing half done; a scene in metres
with Z up; a read-only project folder and a full disk are refused before anything is written; the scene saved and
opened again fetches its result from the job id the node kept.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

CONFIG = json.load(open(sys.argv[1], encoding="utf-8"))
sys.path.insert(0, CONFIG["scripts"])

import maya.standalone  # noqa: E402

maya.standalone.initialize(name="python")
import maya.cmds as cmds  # noqa: E402

REPORT: dict = {"steps": [], "ok": True}


def step(name, ok, detail=""):
    REPORT["steps"].append({"step": name, "ok": bool(ok), "detail": detail})
    REPORT["ok"] = REPORT["ok"] and bool(ok)
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)


def snapshot(uuids):
    """Every attribute value, connection, name and parent of these nodes (the user's objects)."""
    out = {}
    for u in uuids:
        node = (cmds.ls(u, long=True) or [None])[0]
        if node is None:
            out[u] = None
            continue
        attrs = {}
        # every attribute saved with the scene (what the user's work is); attributes Maya does not save are caches
        # it recomputes (a deformer's cacheSetup), not the user's data
        for a in cmds.listAttr(node, write=True) or []:
            try:
                if not cmds.attributeQuery(a.split(".")[-1], node=node, storable=True):
                    continue
                # a value a connection drives (an animation curve, a parent's scale) is the source's, compared there
                # (the curve's own keys are among the user's nodes); here only what the node itself holds
                if cmds.connectionInfo(f"{node}.{a}", isDestination=True):
                    continue
            except Exception:  # noqa: BLE001
                pass
            try:
                v = cmds.getAttr(f"{node}.{a}")
            except Exception:  # noqa: BLE001 - compound / message / multi parents have no single value
                continue
            attrs[a] = repr(_zero(v))[:2000]
        conns = cmds.listConnections(node, plugs=True, connections=True) or []
        pairs = sorted(f"{conns[i]}<->{conns[i + 1]}" for i in range(0, len(conns), 2)
                       if "l2s_inputs" not in conns[i + 1] and "l2s_results" not in conns[i + 1])
        out[u] = {"name": node, "parent": (cmds.listRelatives(node, parent=True, fullPath=True) or [""])[0],
                  "attrs": attrs, "conns": pairs}
    return out


def _zero(v):
    """-0.0 and 0.0 are the same value (the FBX exporter writes -0.0 back over a rotate axis of 0.0): compared as
    numbers, never as their spelling."""
    if isinstance(v, float):
        return 0.0 if v == 0 else v
    if isinstance(v, (list, tuple)):
        return type(v)(_zero(x) for x in v)
    return v


def diff(a, b):
    said = []
    for u in a:
        if a[u] != b.get(u):
            if a[u] is None or b.get(u) is None:
                said.append(f"{u}: gone")
                continue
            for k in ("name", "parent", "conns"):
                if a[u][k] != b[u][k]:
                    said.append(f"{a[u]['name']} {k}: {a[u][k]} -> {b[u][k]}")
            for k, v in a[u]["attrs"].items():
                if b[u]["attrs"].get(k) != v:
                    said.append(f"{a[u]['name']}.{k}: {v[:80]} -> {str(b[u]['attrs'].get(k))[:80]}")
    return said


class Watch:
    """Every write that lands on a node that was in the scene before (the user's, and Maya's own default nodes): an
    attribute set, a connection made or broken, an attribute or array element added or removed, a lock, a rename, a
    re-parent, a removal — OpenMaya callbacks on each node, from start() to stop(). `ours` (the plugin's Lab2Shot node
    and |Lab2Shot) are left out: writing to them is our job."""

    FLAGS = None

    def __init__(self, nodes, ours=()):
        import maya.api.OpenMaya as om

        self.om = om
        M = om.MNodeMessage
        Watch.FLAGS = (M.kAttributeSet | M.kConnectionMade | M.kConnectionBroken | M.kAttributeAdded
                       | M.kAttributeRemoved | M.kAttributeArrayAdded | M.kAttributeArrayRemoved | M.kAttributeLocked
                       | M.kAttributeUnlocked | M.kAttributeRenamed)
        self.ours = set(ours)
        self.nodes = [n for n in nodes if cmds.objExists(n) and n not in self.ours]
        self.defaults = set(cmds.ls(defaultNodes=True) or []) | {"time1", "sequenceManager1", "hardwareRenderingGlobals",
                                                                 "renderPartition", "lightLinker1", "defaultLightSet",
                                                                 "defaultObjectSet", "initialShadingGroup",
                                                                 "initialParticleSE", "shapeEditorManager", "poseInterpolatorManager",
                                                                 "layerManager", "renderLayerManager", "defaultRenderLayer",
                                                                 "defaultLayer", "lambert1", "standardSurface1",
                                                                 "particleCloud1", "strokeGlobals", "defaultTextureList1",
                                                                 "defaultShaderList1", "defaultRenderUtilityList1"}
        self.events, self.ids = [], []

    def start(self):
        om = self.om
        self.events = []
        sel = om.MSelectionList()
        for n in self.nodes:
            try:
                sel.add(n)
            except RuntimeError:
                continue
        for i in range(sel.length()):
            obj = sel.getDependNode(i)
            handle = om.MObjectHandle(obj)
            name = om.MFnDependencyNode(obj).name()

            def changed(msg, plug, other, data, name=name):
                if msg & Watch.FLAGS:
                    self.events.append(("attr", name, plug.partialName(includeNodeName=True), int(msg),
                                        other.partialName(includeNodeName=True) if not other.isNull else ""))

            self.ids.append(om.MNodeMessage.addAttributeChangedCallback(obj, changed))
            self.ids.append(om.MNodeMessage.addNameChangedCallback(obj, lambda node, prev, d, name=name: self.events.append(("rename", name, prev))))
            self.ids.append(om.MNodeMessage.addNodePreRemovalCallback(obj, lambda node, d, name=name: self.events.append(("removed", name))))
            if obj.hasFn(om.MFn.kDagNode):
                dag = om.MDagPath.getAPathTo(obj)
                self.ids.append(om.MDagMessage.addParentAddedDagPathCallback(dag, lambda c, p, d, name=name: self.events.append(("parent+", name))))
                self.ids.append(om.MDagMessage.addParentRemovedDagPathCallback(dag, lambda c, p, d, name=name: self.events.append(("parent-", name))))
        return self

    def stop(self):
        for i in self.ids:
            try:
                self.om.MMessage.removeCallback(i)
            except RuntimeError:
                pass
        self.ids = []
        return self

    def on_user(self):
        return [e for e in self.events if e[1].split(":")[-1] not in self.defaults and e[1] not in self.defaults]

    def on_defaults(self):
        return [e for e in self.events if e not in self.on_user()]


def scene_settings():
    return (cmds.currentUnit(q=True, linear=True), cmds.upAxis(q=True, axis=True), cmds.currentUnit(q=True, time=True),
            cmds.currentTime(q=True), tuple(cmds.ls(selection=True, long=True) or []),
            cmds.playbackOptions(q=True, minTime=True), cmds.playbackOptions(q=True, maxTime=True))


def main():
    from lab2shot_dcc import connection, jobs, paths, safety
    from lab2shot_dcc.plugin import Plugin
    from lab2shot_maya.host import MayaHost

    cmds.loadPlugin("fbxmaya", quiet=True)
    cmds.undoInfo(state=True, infinity=True)
    os.makedirs(CONFIG["project"], exist_ok=True)
    cmds.workspace(CONFIG["project"], openWorkspace=True)
    cmds.currentUnit(time="ntsc")  # the plate is 30 fps
    first, last = CONFIG["frames"]
    cmds.playbackOptions(minTime=first, maxTime=last, animationStartTime=first, animationEndTime=last)
    cmds.currentTime(first)

    # ---- the user's scene: a character from an FBX, a camera with an image plane on a picture sequence
    before_nodes = set(cmds.ls(long=True))
    cmds.file(CONFIG["character"], i=True, type="FBX", ignoreVersion=True)
    user_char = [n for n in cmds.ls(assemblies=True, long=True) if n not in before_nodes and cmds.listRelatives(n, ad=True, type="joint")]
    cam, cam_shape = cmds.camera(name="shotCam", focalLength=35.0, horizontalFilmAperture=1.417, verticalFilmAperture=0.945)
    grp = cmds.group(cam, name="camRig")
    cmds.setKeyframe(grp, attribute="translateZ", time=first, value=0.0)
    cmds.setKeyframe(grp, attribute="translateZ", time=last, value=-50.0)
    cmds.setAttr(cam + ".translateY", 150)
    cmds.setAttr(cam + ".translateZ", 500)
    plane = cmds.imagePlane(camera=cam_shape, fileName=CONFIG["plate"])
    plane_shape = plane[1] if len(plane) > 1 else plane[0]
    cmds.setAttr(plane_shape + ".useFrameExtension", True)
    cmds.setAttr(plane_shape + ".frameOffset", CONFIG["plate_first"] - first)
    # a camera driven every way a camera is: an animated parent (above), an aim constraint, an expression on its lens
    target = cmds.spaceLocator(name="aimTarget")[0]
    cmds.setAttr(target + ".translateY", 120)
    cmds.aimConstraint(target, cam, aimVector=(0, 0, -1))
    cmds.expression(string=f"{cam_shape}.focalLength = 35 + (frame - {first}) * 0.1;")
    # a plain skeleton with keys, and a character referenced from a file (its nodes are read-only)
    cmds.select(clear=True)
    plain = [cmds.joint(name=f"plainJoint{i}", position=(0, i * 10, 0)) for i in range(4)]
    cmds.setKeyframe(plain[1], attribute="rotateZ", time=first, value=0)
    cmds.setKeyframe(plain[1], attribute="rotateZ", time=last, value=45)
    cmds.file(CONFIG["character"], reference=True, type="FBX", namespace="refChar", ignoreVersion=True)
    ref_root = cmds.ls("refChar:RL_BoneRoot", long=True)[0]
    # the FBX reference reset the playback range (the importer's own doing, before anything of ours): set it again
    cmds.playbackOptions(minTime=first, maxTime=last, animationStartTime=first, animationEndTime=last)
    user_nodes = [n for n in cmds.ls(long=True) if n not in before_nodes]
    user_uuids = cmds.ls(user_nodes, uuid=True)
    cmds.select(clear=True)
    # the scene evaluated once at its current frame before it is compared: values an FBX import leaves are not yet
    # what Maya evaluates them to (moving the time slider changes them by 1e-16 too), which is no change of ours
    cmds.currentTime(first + 1)
    cmds.currentTime(first)
    snap0 = snapshot(user_uuids)
    settings0 = scene_settings()
    step("scene built", user_char and cam and ref_root and cmds.referenceQuery(ref_root, isNodeReferenced=True),
         f"{len(user_uuids)} user nodes, character {user_char}, referenced {ref_root}")
    every_node = cmds.ls() or []  # watched: the user's nodes and Maya's own

    host = MayaHost()
    plugin = Plugin(host)

    def pump_until(cond, seconds):
        end = time.time() + seconds
        while not cond() and time.time() < end:
            host.pump()
        return cond()

    said = {}
    plugin.login(CONFIG["server"], CONFIG["account"]["username"], CONFIG["account"]["password"],
                 lambda got, error: said.update(login=error))
    pump_until(lambda: "login" in said, 60)
    step("login", not said.get("login"), said.get("login") or "")
    plugin.refresh_tools(lambda error: said.update(tools=error))
    pump_until(lambda: "tools" in said, 300)
    step("tools listed", not said["tools"] and plugin.tools, f"{len(plugin.tools)} tools")

    # ranking: a camera with an image plane selected puts tools taking a picture and a camera first
    cmds.select(cam, replace=True)
    types = host.selection_types()
    order = plugin.ranked()
    top = order[0][2][0]
    step("selection types", "scene.camera" in types and "image" in types, str(types))
    step("ranked, not filtered", sum(len(t) for _s, _l, t in order) == len(plugin.tools), f"first: {top['name']}")

    tool = next(t for t in plugin.tools if t["id"] == CONFIG["tool"])
    node = plugin.new_node()
    plugin.set_tool(node, tool)
    inputs = {i["param"]: i for i in tool["inputs"]}
    cmds.select(cam, replace=True)
    watch = Watch(every_node, {host.node_name(node)}).start()
    why_pic = plugin.bind_selected(node, inputs["input"])
    why_cam = plugin.bind_selected(node, inputs["cam_fbx_path"])
    cmds.select(user_char[0], replace=True)
    why_char = plugin.bind_selected(node, inputs["char_fbx_path"])
    cmds.select(clear=True)
    watch.stop()
    step("binding: zero writes to nodes that were there (no connection to the user's objects)", not watch.events,
         str(watch.events[:5]))
    state = host.load(node)
    step("bind picture / camera / character", not (why_pic or why_cam or why_char),
         json.dumps({k: v.get("label") for k, v in state["bindings"].items()}, ensure_ascii=False))
    step("source menus follow the bindings", state["values"].get("cam_src") == 1 and state["values"].get("target_src") == 2,
         str(state["values"]))
    state = plugin.fill_from_scene(node)
    step("context from the scene", state["scene_values"].get("first_frame") == CONFIG["plate_first"]
         and abs(state["scene_values"].get("focal", 0) - 35.0) < 1e-6, str(state["scene_values"]))
    step("binding did not change user objects", not diff(snap0, snapshot(user_uuids)), "; ".join(diff(snap0, snapshot(user_uuids))[:5]))

    # ---- every kind of export, watched: not one write to a node that was there before
    import tempfile

    cases = [("character with blend shapes", user_char[0], "scene.character"),
             ("its skeleton", user_char[0], "scene.skeleton"),
             ("plain skeleton with keys", plain[0], "scene.skeleton"),
             ("camera: animated parent + aim constraint + expression", cam, "scene.camera"),
             ("referenced character", ref_root, "scene.character")]
    for label, obj, kind in cases:
        cmds.select(obj, replace=True)
        selection = cmds.ls(selection=True, long=True)
        undo_before = cmds.undoInfo(q=True, undoName=True)
        folder = tempfile.mkdtemp(prefix="l2s_export_", dir=CONFIG["project"])
        watch = Watch(every_node).start()
        try:
            prepared = host.export({"ref": cmds.ls(obj, uuid=True)[0], "type": kind, "plate_offset": CONFIG["plate_first"] - first}, folder)
            out = host.finish_export({}, prepared, lambda: False)
            error = ""
        except Exception as exc:  # noqa: BLE001
            out, error = "", f"{exc}"
        watch.stop()
        size = os.path.getsize(out) if out and os.path.isfile(out) else 0
        step(f"export {label}: file written, zero writes to existing nodes",
             size > 0 and not watch.events and cmds.ls(selection=True, long=True) == selection
             and cmds.undoInfo(q=True, undoName=True) == undo_before,
             f"{error} {size} bytes; user writes {watch.on_user()[:5]}; Maya-node writes {watch.on_defaults()[:5]}")
    cmds.select(clear=True)
    changed = diff(snap0, snapshot(user_uuids))
    step("after the five exports: user objects equal attribute by attribute", not changed, "; ".join(changed[:5]))

    # ---- compute: export, upload, submit, follow, fetch, import
    settings_before = scene_settings()
    t0 = time.time()
    ours = {host.node_name(node), "Lab2Shot"}
    defaults = [u for u in cmds.ls(["time1", "defaultRenderGlobals", "defaultColorMgtGlobals", "standardSurface1",
                                    "lambert1", "initialShadingGroup", "renderPartition", "lightLinker1"], uuid=True)]
    defaults_before = snapshot(defaults)
    watch = Watch(every_node, ours).start()
    run = plugin.compute(node)
    finished = pump_until(lambda: run.finished, CONFIG.get("timeout", 1800))
    watch.stop()
    snap = run.snapshot()
    default_changes = [d for d in diff(defaults_before, snapshot(defaults)) if " conns:" not in d]
    step("Maya's own nodes the FBX importer sets: values unchanged", not default_changes, "; ".join(default_changes[:5]))
    # the backplate on the delivered camera: Maya links every colour-managed node it makes to these four plugs of
    # defaultColorMgtGlobals (a connection from it; its values stay: checked just above). Nothing else, anywhere.
    resync = {f"defaultColorMgtGlobals.{a}" for a in ("cme", "cfe", "cfp", "wsn")}  # the plug names as Watch spells them
    others = [e for e in watch.events if not (e[0] == "attr" and e[1] == "defaultColorMgtGlobals" and e[2] in resync)]
    step("whole job (export + import): zero writes to any node that was there (Maya's defaults included; Maya's own "
         "colour links to the new backplate aside)", finished and not others,
         f"user {[e for e in watch.on_user() if e in others][:5]}; Maya's own nodes: "
         f"{sorted(set(e[2] if len(e) > 2 else e[1] for e in watch.on_defaults() if e in others))[:8]}")
    step("job done", finished and snap["phase"] == "done", f"{snap['phase']} {snap['text']} {snap['error']} ({time.time() - t0:.0f} s, job {snap['job']})")
    if snap["phase"] != "done":
        return
    state = host.load(node)
    v1 = state["versions"][-1]
    group = cmds.ls(v1["group"], long=True)
    step("new objects under |Lab2Shot|<node>_v001", group and group[0] == "|Lab2Shot|lab2shot1_v001", str(group))
    imported = cmds.listRelatives(group[0], allDescendents=True, fullPath=True) or []
    cams = cmds.listRelatives(group[0], allDescendents=True, type="camera", fullPath=True) or []
    joints = cmds.listRelatives(group[0], allDescendents=True, type="joint", fullPath=True) or []
    meshes = cmds.listRelatives(group[0], allDescendents=True, type="mesh", fullPath=True) or []
    step("camera / skinned character delivered", cams and joints and meshes,
         f"{len(cams)} cameras, {len(joints)} joints, {len(meshes)} meshes")
    ns_ok = all(n.split("|")[-1].startswith("l2s_lab2shot1_v001:") and n.split("|")[-1].count(":") == 1 for n in imported)
    step("namespace l2s_lab2shot1_v001 (one level)", cmds.namespace(exists=":l2s_lab2shot1_v001") and ns_ok,
         [n.split("|")[-1] for n in imported if n.split("|")[-1].count(":") != 1][:5])
    ascii_ok = all(n.isascii() and "FBXASC" not in n for n in imported)
    originals = [cmds.getAttr(n + ".l2s_name") for n in imported if cmds.attributeQuery("l2s_name", node=n, exists=True)]
    step("names cleaned, originals kept on l2s_name", ascii_ok, f"originals: {originals[:3]}")
    # the bound picture behind the delivered camera (jobs.backplate); a plane hangs in its camera's underworld
    planes = [cmds.ls(p, long=True)[0] for c in cams
              for p in cmds.listConnections(c + ".imagePlane", source=True, destination=False, shapes=True) or []]
    step("delivered camera gets a new backplate: the bound picture, its frame offset, in the version's namespace",
         len(planes) == 1 and os.path.basename(cmds.getAttr(planes[0] + ".imageName")) == os.path.basename(CONFIG["plate"])
         and cmds.getAttr(planes[0] + ".frameOffset") == CONFIG["plate_first"] - first
         and planes[0].split("|")[-1].startswith("l2s_lab2shot1_v001:"), str(planes))
    after = scene_settings()
    step("unit / up axis / fps / frame / selection / range unchanged", after == settings_before, f"{settings_before} -> {after}")
    changed = diff(snap0, snapshot(user_uuids))
    step("user objects unchanged attribute by attribute", not changed, "; ".join(changed[:8]))
    files = os.listdir(v1["folder"])
    step("files in <project>/data/lab2shot/<node>/v001", v1["folder"].replace("\\", "/").endswith("data/lab2shot/lab2shot1/v001"), str(files))
    long_paths = [os.path.join(b, f) for b, _d, fs in os.walk(v1["folder"]) for f in fs]
    step("every result path cleaned and within 240 chars", all(len(p) <= 240 and all(ord(c) < 128 for c in os.path.relpath(p, v1["folder"])) for p in long_paths),
         str([os.path.relpath(p, v1["folder"]) for p in long_paths]))
    curves = cmds.ls("l2s_lab2shot1_v001:*", type="animCurve", recursive=True) or []
    keys = (cmds.keyframe(curves, q=True, timeChange=True) or []) if curves else []
    step("keys at Maya frames (plate frames shifted back)", keys and min(keys) >= first - 1 and max(keys) <= last + 1,
         f"{min(keys) if keys else None}..{max(keys) if keys else None}")

    # ---- one undo removes the whole import
    cmds.undo()
    left = cmds.ls("l2s_lab2shot1_v001:*", recursive=True) or []
    gone = not cmds.ls(v1["group"]) and not left
    step("one Ctrl+Z removes the import", gone, str(left[:5]))
    versions_after_undo = len(host.load(node).get("versions") or [])
    cmds.redo()
    redone = (host.load(node).get("versions") or [{}])[-1]
    back = cmds.ls(redone.get("group", ""), long=True)
    joints_back = cmds.listRelatives(back[0], allDescendents=True, type="joint") if back else []
    step("undo also takes the version off the node; redo brings all of it back",
         versions_after_undo == 0 and back and joints_back and len(host.load(node)["versions"]) == 1,
         f"{versions_after_undo} versions after undo, {len(joints_back or [])} joints after redo")
    v1 = host.load(node)["versions"][-1]

    # ---- fetch again: a new version, numbered, nothing replaced (the import alone, watched)
    watch = Watch(every_node, ours).start()
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 600)
    watch.stop()
    step("import alone: zero writes to any node that was there (Maya's defaults included)", not watch.events,
         str(watch.events[:5]))
    state = host.load(node)
    v2 = state["versions"][-1]
    step("fetch again: v002, new group and namespace", run.snapshot()["phase"] == "done" and v2["version"] == 2
         and cmds.ls(v2["group"], long=True)[0] == "|Lab2Shot|lab2shot1_v002" and cmds.ls(v1["group"]),
         run.snapshot()["text"])

    # a name taken by someone else: numbered
    cmds.createNode("transform", name="lab2shot1_v003")
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 600)
    v3 = host.load(node)["versions"][-1]
    step("a taken name gets a number", cmds.ls(v3["group"])[0] == "lab2shot1_v003_002", cmds.ls(v3["group"])[0])

    # ---- cancel in the middle: nothing half done
    base = os.path.dirname(v1["folder"])
    listed = sorted(os.listdir(base))
    groups = cmds.listRelatives("|Lab2Shot", children=True)
    run = plugin.compute(node)
    pump_until(lambda: run.snapshot()["phase"] in ("uploading", "queued", "running") or run.finished, 300)
    run.cancel()
    pump_until(lambda: run.finished, 300)
    step("cancel leaves nothing half done", run.snapshot()["phase"] == "cancelled" and sorted(os.listdir(base)) == listed
         and cmds.listRelatives("|Lab2Shot", children=True) == groups, run.snapshot()["text"])

    # ---- a dropped line while following the job: tried a bounded number of times, then stopped, job id kept
    jobs.RECONNECT_TRIES, jobs.RECONNECT_WAIT = 2, (1, 1)
    real = connection.session

    def broken(server, app, **kw):
        lab = real(server, app, **kw)
        client = paths.client_module()

        def poll(job, since=0):
            raise client.Lab2ShotError("connection refused", 0)
        lab.poll = poll
        return lab

    connection.session = broken
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 300)
    connection.session = real
    snap = run.snapshot()
    step("dropped line: stops with the job id kept", snap["phase"] == "failed" and snap["job"] in snap["error"], snap["error"])
    groups_now = cmds.listRelatives("|Lab2Shot", children=True)
    step("dropped line: nothing imported", groups_now == groups, str(groups_now))

    # ---- server unreachable: reconnects a bounded number of times, saying so, then stops with a clear sentence
    keep = plugin.conn.server
    plugin.conn.server = "https://127.0.0.1:9"
    t0 = time.time()
    run = plugin.fetch(node)
    said_tries = set()
    end = time.time() + 180
    while not run.finished and time.time() < end:
        host.pump()
        if "重连" in run.snapshot()["text"]:
            said_tries.add(run.snapshot()["text"])
    plugin.conn.server = keep
    step("server unreachable: says it reconnects, stops within 2 minutes",
         run.snapshot()["phase"] == "failed" and "连不上" in run.snapshot()["error"] and said_tries and time.time() - t0 < 120,
         f"{time.time() - t0:.0f}s {len(said_tries)} tries said; {run.snapshot()['error']}")

    # ---- a full disk / a read-only project: refused before anything is written
    from lab2shot_dcc import results

    real_free = safety.free_bytes
    safety.free_bytes = lambda folder: 1000
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 120)
    safety.free_bytes = real_free
    step("disk full: refused before downloading", run.snapshot()["phase"] == "failed" and "磁盘空间" in run.snapshot()["error"]
         and sorted(os.listdir(base)) == listed, run.snapshot()["error"])
    real_writable = safety.writable
    safety.writable = lambda folder: f"文件夹不能写：{folder}（只读）"
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 120)
    safety.writable = real_writable
    step("read-only project: refused", run.snapshot()["phase"] == "failed" and "不能写" in run.snapshot()["error"], run.snapshot()["error"])

    # ---- metres and Z up: the result comes in, the settings stay
    cmds.currentUnit(linear="m")
    cmds.upAxis(axis="z")
    settings_mz = scene_settings()
    run = plugin.fetch(node)
    pump_until(lambda: run.finished, 600)
    step("metres + Z up: imported, settings kept", run.snapshot()["phase"] == "done" and scene_settings() == settings_mz,
         f"{run.snapshot()['text']} {scene_settings()}")
    cmds.upAxis(axis="y")
    cmds.currentUnit(linear="cm")

    changed = diff(snap0, snapshot(user_uuids))
    step("user objects still unchanged after every step", not changed, "; ".join(changed[:8]))

    # ---- save, open again, fetch from the job id the node kept
    scene = os.path.join(CONFIG["project"], "scenes", "lab2shot_test.ma")
    os.makedirs(os.path.dirname(scene), exist_ok=True)
    cmds.file(rename=scene)
    cmds.file(save=True, type="mayaAscii", force=True)
    cmds.file(new=True, force=True)
    cmds.file(scene, open=True, force=True)
    nodes = host.nodes()
    state = host.load(nodes[0]) if nodes else {}
    step("reopened: node, job id and versions kept", nodes and state.get("job", {}).get("id") and len(state.get("versions", [])) >= 4,
         f"{len(nodes)} nodes, job {state.get('job', {}).get('id')}")
    bound = state.get("bindings") or {}
    conns = [c for b in bound.values() if host.binding_alive(b)
             for c in (cmds.listConnections(cmds.ls(b["ref"])[0], source=False, destination=True) or [])
             if cmds.nodeType(c) == "network"]
    step("bindings kept as UUID + path, nothing connected to the user's objects",
         len(bound) == 3 and all(b.get("ref") and b.get("path") for b in bound.values()) and not conns, str(conns))
    # the user renames and re-parents the bound camera: still found; a bound object deleted: the panel says rebind
    cam_now = cmds.ls(bound["cam_fbx_path"]["ref"], long=True)[0]
    renamed = cmds.rename(cam_now, "shotCamRenamed")
    cmds.parent(renamed, world=True)
    step("a renamed, re-parented binding is found by its UUID", host.describe_binding(bound["cam_fbx_path"]) == "shotCamRenamed",
         host.describe_binding(bound["cam_fbx_path"]))
    gone = {"ref": "00000000-0000-0000-0000-000000000000", "path": "|nothingHere", "label": "oldCam", "type": "scene.camera"}
    step("a binding that is gone says to bind again", not host.binding_alive(gone) and "重新绑定" in host.describe_binding(gone),
         host.describe_binding(gone))
    run = plugin.fetch(nodes[0])
    pump_until(lambda: run.finished, 600)
    step("reopened: fetch from the kept job id", run.snapshot()["phase"] == "done", run.snapshot()["text"])

    # ---- names: Chinese, spaces, emoji, 300 characters
    weird = ["全身动作 · SAM 3D Body", "my scene 😀 v2", "a" * 300, "123start", "CON", "", "名字/../../etc"]
    cleaned = [safety.clean(w) for w in weird]
    ok = all(c and c[0].isalpha() and all(ch.isalnum() or ch == "_" for ch in c) and c.isascii() and len(c) <= 60 for c in cleaned)
    step("names cleaned (Chinese, spaces, emoji, 300 chars)", ok and len(set(cleaned)) == len(cleaned), str(cleaned))
    step("server paths cannot lead out", safety.under(CONFIG["project"], "../x") is None
         and safety.under(CONFIG["project"], "a/../../x") is None)


try:
    main()
except Exception:  # noqa: BLE001
    step("test crashed", False, traceback.format_exc())
finally:
    with open(CONFIG["report"], "w", encoding="utf-8") as f:
        json.dump(REPORT, f, ensure_ascii=False, indent=1)
    print("ALL OK" if REPORT["ok"] else "SOME FAILED", flush=True)
    # mayapy must leave through maya.standalone.uninitialize (or a normal interpreter exit): os._exit() skips Maya's
    # orderly shutdown and Maya then crashes tearing down its node classes (an access violation in
    # TdependNodeVS::cleanupClass) — so this test must not end with os._exit
    cmds.file(new=True, force=True)
    maya.standalone.uninitialize()
    sys.exit(0 if REPORT["ok"] else 1)
