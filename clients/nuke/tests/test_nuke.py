"""End-to-end in Nuke without its interface: `Nuke17.0.exe -t test_nuke.py <config.json>`, with HOME and USERPROFILE
pointing at a test folder (Nuke keeps ~/.nuke and its recent-files list under HOME, the plugin its login and settings
under USERPROFILE: a test run touches neither of the user's).

The plugin as the server's download packs it (config "plugin_dir": tools/deploy_dcc.py nuke --home <a test home>) runs
one whole job against a Lab2Shot server: a user's script (a Read of a plate with a frame offset, a Camera4 under an
animated Axis4, a GeoImport), the Read selected and bound, the job computed, its delivery fetched and imported. Then:

- zero writes: every node that was in the script — every knob (writeKnobs), its inputs, name, position, `selected` —
  the Root's settings, the current frame and the selection are exactly as before;
- what was made: one Backdrop per version, a Camera4 from the .nk, a GeoImport per USD, a Read per picture (raw for
  data maps), an STMap per ST map (its src the user's Read, its stmap our Read), named <namespace>_<name>;
- one space: the delivered points seen through the delivered camera (in front of it, inside its picture) and the
  ratio of their distances to the depth map's (the same unit gives 1);
- the bound camera exported as a USD camera (samples written to the result for the WSL side to read back with the
  server's own reader);
- the script saved and opened again: the Lab2Shot node's state reads back.

Config: {"plugin_dir", "server", "username", "password", "plate", "plate_first", "plate_last", "plate_offset",
"tool", "stmap_tool" (optional), "work", "result"}. Writes the result JSON; prints "RESULT ok" or "RESULT failed".
Undo is not testable here: nuke -t keeps no undo queue (nuke.Undo.undoSize() stays 0).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

import nuke

getattr(sys.stdout, "reconfigure", lambda **_: None)(encoding="utf-8", errors="replace")
CONFIG = json.load(open(sys.argv[-1], encoding="utf-8"))
sys.path.insert(0, CONFIG["plugin_dir"])

from lab2shot_dcc.plugin import Plugin  # noqa: E402

from lab2shot_nuke.host import NukeHost, _find_ours  # noqa: E402

RESULT: dict = {"checks": {}, "errors": []}
FLAGS = nuke.WRITE_ALL | nuke.WRITE_USER_KNOB_DEFS


def check(name: str, ok: bool, detail=None) -> None:
    RESULT["checks"][name] = {"ok": bool(ok), "detail": detail}
    print(("PASS " if ok else "FAIL ") + name + ("" if detail is None else f": {json.dumps(detail, default=str)[:400]}"))


def snapshot() -> dict:
    nodes = {}
    for n in nuke.allNodes(recurseGroups=True):
        if n.knob("l2s_ref") is not None:
            continue
        nodes[n.fullName()] = {"class": n.Class(), "knobs": n.writeKnobs(FLAGS),
                               "inputs": [(n.input(i).fullName() if n.input(i) else None) for i in range(n.inputs())],
                               "selected": n["selected"].value() if n.knob("selected") else None,
                               "pos": [n.xpos(), n.ypos()]}
    root = nuke.root()
    return {"nodes": nodes, "root": root.writeKnobs(FLAGS), "frame": nuke.frame(),
            "selection": sorted(n.fullName() for n in nuke.selectedNodes())}


def compare(a: dict, b: dict) -> list:
    diffs = []
    for name, was in a["nodes"].items():
        now = b["nodes"].get(name)
        if now is None:
            diffs.append(f"{name}: gone")
            continue
        for k in was:
            if was[k] != now[k]:
                diffs.append(f"{name}.{k}: changed")
    for k in ("root", "frame", "selection"):
        if a[k] != b[k]:
            diffs.append(f"{k}: changed")
    return diffs


def user_script() -> dict:
    read = nuke.nodes.Read(name="plate")
    read["file"].fromUserText(f"{CONFIG['plate']} {CONFIG['plate_first']}-{CONFIG['plate_last']}")
    root = nuke.root()  # after the Read: typing a file into a Read stretches the Root's range to the file's frames
    root["first_frame"].setValue(1)
    root["last_frame"].setValue(CONFIG["plate_last"] - CONFIG["plate_first"] + 1)
    root["fps"].setValue(24)
    read["frame_mode"].setValue("offset")
    read["frame"].setValue(str(CONFIG["plate_offset"]))
    read.setXYpos(0, 0)
    axis = nuke.nodes.Axis4(name="rig")
    axis["translate"].setAnimated()
    axis["translate"].setValueAt(10, 1, 0)
    axis["translate"].setValueAt(30, 40, 0)
    axis["rotate"].setValue([0, 15, 0])
    axis.setXYpos(200, 0)
    cam = nuke.nodes.Camera4(name="shotcam")
    cam.setInput(0, axis)
    cam["translate"].setValue([1, 2, 300])
    cam["focal"].setAnimated()
    cam["focal"].setValueAt(30, 1)
    cam["focal"].setValueAt(40, 40)
    cam["win_translate"].setValue([0.02, -0.01])
    cam.setXYpos(200, 100)
    geo = nuke.nodes.GeoImport(name="set_geo", file=CONFIG["work"] + "/cm.usda")
    geo.setXYpos(400, 0)
    for n in nuke.allNodes():
        n.setSelected(False)
    read.setSelected(True)  # the user selects the plate
    return {"read": read, "camera": cam}


def pump_until(host, done, limit_s: float, say=None) -> bool:
    end = time.time() + limit_s
    last = ""
    while time.time() < end:
        host.pump(0.1)
        if say:
            text = say()
            if text != last:
                print("  ", text)
                last = text
        if done():
            return True
    return False


def run_tool(plugin, host, tool_id: str, label: str) -> str:
    tool = next((t for t in plugin.tools if t["id"] == tool_id), None)
    if tool is None:
        raise RuntimeError(f"no tool {tool_id}")
    node = plugin.new_node()
    plugin.set_tool(node, tool)
    item = next(i for i in tool["inputs"] if i["param"] == "input")
    said = plugin.bind_selected(node, item)
    check(f"{label}: bind the selected Read", said == "", said)
    binding = host.load(node)["bindings"]["input"]
    check(f"{label}: binding keeps the plate's frame offset", binding.get("frame_offset") == CONFIG["plate_offset"]
          and binding.get("sequence") is True, {k: binding.get(k) for k in ("file", "frame_offset", "sequence", "colorspace")})
    scene = host.load(node).get("scene_values") or {}
    check(f"{label}: scene values in the plate's frames", scene.get("first_frame") == CONFIG["plate_first"]
          and scene.get("last_frame") == CONFIG["plate_last"], scene)
    run = plugin.compute(node)
    ok = pump_until(host, lambda: run.finished, 3600, lambda: run.snapshot()["text"])
    snap = run.snapshot()
    RESULT[f"{label}_done_text"] = snap["text"]
    check(f"{label}: job done", ok and snap["phase"] == "done", {"phase": snap["phase"], "text": snap["text"],
                                                                 "error": snap["error"], "job": snap["job"]})
    return node


def describe(node) -> dict:
    d = {"class": node.Class(), "name": node.name(), "pos": [node.xpos(), node.ypos()]}
    for k in ("file", "file_path", "first", "last", "frame_mode", "frame", "raw", "colorspace", "time_offset", "uv",
              "label", "rot_order", "xform_order"):
        if node.knob(k) is not None:
            d[k] = node[k].value()
    d["inputs"] = [(node.input(i).name() if node.input(i) else None) for i in range(node.inputs())]
    if node.Class() == "Camera4":
        d["translate_f1"] = node["translate"].getValueAt(1)
        d["focal_f1"] = node["focal"].getValueAt(1)
        d["animated_keys"] = [k.x for k in node["translate"].animation(0).keys()][:3] if node["translate"].isAnimated() else []
    return d


def stage_points(geo, frame: int) -> list:
    from pxr import Usd, UsdGeom

    oc = nuke.OutputContext()
    oc.setFrame(frame)
    stage = geo.getStage(oc)
    pts = []
    cache = UsdGeom.XformCache(Usd.TimeCode(frame))
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Points) or prim.IsA(UsdGeom.Mesh):
            world = cache.GetLocalToWorldTransform(prim)
            got = UsdGeom.PointBased(prim).GetPointsAttr().Get(Usd.TimeCode(frame)) or []
            step = max(1, len(got) // 4000)
            pts += [world.Transform(p) for p in list(got)[::step]]
    return pts, (stage.GetMetadata("metersPerUnit") if stage else None)


def usd_camera_vs_nk(cam, geo, frames) -> list:
    """The camera prim inside the delivered USD scene, as GeoImport gives it, against the Camera4 made from the .nk:
    world translation and focal length at the same Nuke frames."""
    from pxr import Usd, UsdGeom

    out = []
    for f in frames:
        oc = nuke.OutputContext()
        oc.setFrame(f)
        stage = geo.getStage(oc)
        prim = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)), None) if stage else None
        if prim is None:
            return []
        code = Usd.TimeCode(f)  # GeoImport's stage is in Nuke's frames (its time_offset applied; measured)
        t_usd = list(UsdGeom.XformCache(code).GetLocalToWorldTransform(prim).ExtractTranslation())
        m = [cam["world_matrix"].getValueAt(f, i) for i in range(16)]
        t_nk = [m[3], m[7], m[11]]
        out.append({"frame": f, "usd": t_usd, "nk": t_nk, "translate_diff": max(abs(a - b) for a, b in zip(t_usd, t_nk)),
                    "focal_usd": UsdGeom.Camera(prim).GetFocalLengthAttr().Get(code), "focal_nk": cam["focal"].getValueAt(f)})
    return out


def one_space(cam, geo, depth_read, frame: int) -> dict:
    """Points through the camera at a Nuke frame: how many are in front and inside the picture; the median ratio of
    their camera distance to the depth map's value where they land (cm against the depth map's unit)."""
    pts, mpu = stage_points(geo, frame)
    m = [cam["world_matrix"].getValueAt(frame, i) for i in range(16)]
    r = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]]
    t = [m[3], m[7], m[11]]
    focal, hap = cam["focal"].getValueAt(frame), cam["haperture"].getValueAt(frame)
    vap = cam["vaperture"].getValueAt(frame)
    inside, front, depths = 0, 0, []
    for p in pts:
        d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]]
        c = [sum(r[k][i] * d[k] for k in range(3)) for i in range(3)]  # R^T (p - t): camera space, looking down -Z
        if c[2] >= 0:
            continue
        front += 1
        x, y = focal * c[0] / -c[2] / (hap / 2), focal * c[1] / -c[2] / (vap / 2)
        if abs(x) <= 1 and abs(y) <= 1:
            inside += 1
            depths.append((x, y, -c[2], math.sqrt(sum(v * v for v in c))))
    out = {"frame": frame, "points": len(pts), "front": front, "inside": inside, "stage_meters_per_unit": mpu,
           "camera_translate": t, "focal": focal}
    if depth_read is not None and depths:
        w, h = depth_read.width(), depth_read.height()
        ratios = []
        chan = next((c for c in depth_read.channels() if c.split(".")[-1] in ("Z", "z", "depth", "red", "r")), None)
        nuke.frame(frame)
        for x, y, z, _dist in depths[:: max(1, len(depths) // 60)]:
            px, py = (x + 1) / 2 * w, (y + 1) / 2 * h
            v = nuke.sample(depth_read, chan, px, py) if chan else 0.0
            if v > 0:
                ratios.append(z / v)
        ratios.sort()
        out["depth_channel"] = chan
        out["z_over_depth_median"] = ratios[len(ratios) // 2] if ratios else None
        out["depth_samples"] = len(ratios)
    return out


def make_movie() -> str:
    """A 37-frame movie for the video result (made and its nodes removed before the user's script is built)."""
    path = CONFIG["work"] + "/clip.mov"
    c = nuke.nodes.Constant()
    c["color"].setExpression("frame/100", 0)
    w = nuke.nodes.Write(file=path, file_type="mov")
    w.setInput(0, c)
    nuke.execute(w, 1, 37)
    nuke.delete(w)
    nuke.delete(c)
    return path


def main() -> None:
    nuke.scriptClear()
    movie = make_movie()
    user = user_script()
    host = NukeHost()
    plugin = Plugin(host)
    # the script never saved: nothing is computed; the user saves it, then it is
    refused = ""
    try:
        plugin.compute(plugin.new_node())
    except RuntimeError as exc:
        refused = str(exc)
    from lab2shot_dcc import paths as dcc_paths

    check("unsaved script: compute refused, 'save the project first'", host.unsaved()
          and refused == dcc_paths.text("dcc.plugin.save_first"), refused)
    script = os.path.join(CONFIG["work"], "shot_comp_v001.nk").replace("\\", "/")
    nuke.scriptSaveAs(script, overwrite=1)
    check("saved: the gate open, results next to the script", not host.unsaved() and plugin.why_not_saved() == ""
          and host.project_dir().replace("\\", "/") == os.path.dirname(script), host.project_dir())
    before = snapshot()
    plugin.conn.login(CONFIG["server"], CONFIG["username"], CONFIG["password"])
    errors = []
    plugin.refresh_tools(lambda e: errors.append(e))
    pump_until(host, lambda: not plugin.loading, 120)
    check("tool list", bool(plugin.tools) and not any(errors), {"count": len(plugin.tools), "error": errors})
    import lab2shot_nuke.menu
    import lab2shot_nuke.panel

    check("menu and panel modules load (the interface itself only in Nuke's GUI)",
          callable(lab2shot_nuke.menu.install) and lab2shot_nuke.panel.WIDGET.endswith(".Holder"))
    check("ui_language empty: Chinese by default", host.ui_language() == "" and plugin.lang == "zh", plugin.lang)
    check("selection types", host.selection_types() == ["image"], host.selection_types())

    # the bound camera as a USD camera (export runs on the main thread; finish_export samples it)
    cam_binding = {"ref": "node:shotcam", "path": "shotcam", "fingerprint": "Camera4|", "type": "scene.camera",
                   "label": "shotcam", "plate_offset": CONFIG["plate_offset"]}
    out_dir = os.path.join(CONFIG["work"], "export")
    os.makedirs(out_dir, exist_ok=True)
    prepared = host.export(cam_binding, out_dir)
    usd = host.finish_export(cam_binding, prepared, lambda: False)
    RESULT["camera_export"] = {"file": usd, "nuke": [
        {"frame": f + CONFIG["plate_offset"], "matrix": [user["camera"]["world_matrix"].getValueAt(f, i) for i in range(16)],
         "focal": user["camera"]["focal"].getValueAt(f), "win": [user["camera"]["win_translate"].getValueAt(f, 0),
                                                                user["camera"]["win_translate"].getValueAt(f, 1)]}
        for f in (1, 20, 40)]}
    check("camera exported", os.path.isfile(usd), usd)

    # a camera goes into the input that reads what Nuke exports (.usda): cam_usd_path, not cam_fbx_path
    both = next((t for t in plugin.tools if {"cam_fbx_path", "cam_usd_path"} <= {i["param"] for i in t.get("inputs") or []}), None)
    if both is not None:
        from lab2shot_dcc import view_model

        node0 = plugin.new_node()
        plugin.set_tool(node0, both)
        items = {i["param"]: i for i in both["inputs"]}
        user["read"].setSelected(False)  # the user selects the camera and the plate
        user["camera"].setSelected(True)
        user["read"].setSelected(True)
        chosen = [i["param"] for i in view_model.auto_bind(both, host.selection_types(), host.exports)]
        refused = plugin.bind_selected(node0, items["cam_fbx_path"])
        taken = plugin.bind_selected(node0, items["cam_usd_path"])
        check("camera by format: auto bind picks cam_usd_path", "cam_usd_path" in chosen and "cam_fbx_path" not in chosen,
              chosen)
        check("camera by format: cam_fbx_path refused, cam_usd_path taken", bool(refused) and taken == "",
              {"refused": refused, "taken": taken, "accept": items["cam_usd_path"].get("accept")})
        user["camera"].setSelected(False)  # the user's selection as it was
        RESULT["camera_by_format_tool"] = both["id"]
    RESULT["project_dir"] = host.project_dir()

    node = run_tool(plugin, host, CONFIG["tool"], "main")
    state = host.load(node)
    version = (state.get("versions") or [{}])[-1]
    made = [_find_ours(r) for r in version.get("objects") or []]
    RESULT["made"] = [describe(n) for n in made if n is not None]
    bd = _find_ours(version.get("group", ""))
    check("one Backdrop per version", bd is not None and bd.Class() == "BackdropNode",
          bd.name() if bd is not None else None)
    classes = sorted(n.Class() for n in made if n is not None)
    check("result nodes: one camera (the same camera delivered once), the points, the depth",
          classes == ["Camera4", "GeoImport", "Read"], classes)
    folder = str(version.get("folder") or "").replace("\\", "/")
    check("results next to the script: <script folder>/data/lab2shot/<node>/v001",
          folder.startswith(CONFIG["work"].replace("\\", "/") + "/data/lab2shot/") and folder.endswith("/v001"), folder)
    ns = version.get("namespace", "")
    check("names carry the version's namespace", all(n.name().startswith(ns) for n in made if n is not None), ns)
    inside = all(bd.xpos() <= n.xpos() <= bd.xpos() + bd["bdwidth"].value() for n in made if n is not None) if bd else False
    check("nodes inside the Backdrop", inside)
    cam = next((n for n in made if n is not None and n.Class() == "Camera4"), None)
    geo = next((n for n in made if n is not None and n.Class() == "GeoImport"), None)
    depth = next((n for n in made if n is not None and n.Class() == "Read" and n["raw"].value()), None)
    if cam is not None:
        keys = [k.x for k in cam["translate"].animation(0).keys()] if cam["translate"].isAnimated() else []
        check("camera keys at Nuke's frames (picture frame - offset)", keys[:1] == [1.0] if keys else False, keys[:3])
    if geo is not None:
        check("GeoImport time offset follows the plate", geo["time_offset"].value() == -CONFIG["plate_offset"],
              geo["time_offset"].value())
    if depth is not None:
        check("depth map read raw at the plate's frames", depth["frame_mode"].value() == "offset"
              and depth["frame"].value() == str(CONFIG["plate_offset"]), [depth["first"].value(), depth["last"].value()])
        check("depth Read has no error", not depth.hasError())
    if cam is not None and geo is not None:
        RESULT["camera_in_usd_vs_nk"] = usd_camera_vs_nk(cam, geo, (1, 20, 40))
        if RESULT["camera_in_usd_vs_nk"]:
            worst = max(r["translate_diff"] for r in RESULT["camera_in_usd_vs_nk"])
            check("the USD scene's camera (GeoImport) = the .nk camera (Camera4), same numbers", worst < 0.01,
                  RESULT["camera_in_usd_vs_nk"])

    if CONFIG.get("stmap_tool"):
        user["read"].setSelected(True)  # (was selected, stays)
        node2 = run_tool(plugin, host, CONFIG["stmap_tool"], "stmap")
        v2 = (host.load(node2).get("versions") or [{}])[-1]
        made2 = [_find_ours(r) for r in v2.get("objects") or []]
        RESULT["made_stmap"] = [describe(n) for n in made2 if n is not None]
        sts = [n for n in made2 if n is not None and n.Class() == "STMap"]
        check("STMap per ST map layer", len(sts) >= 1, [describe(n) for n in sts])
        pic = next((n for n in made2 if n is not None and n.Class() == "Read"), None)
        if pic is not None:
            check("ACEScg pictures in nuke-default: colour space left as it is (no such space there)",
                  str(pic["colorspace"].value()).startswith("default") and not pic["raw"].value(), pic["colorspace"].value())
        und = next((n for n in sts if "undistort" in n["uv"].value()), None)
        if und is not None:
            check("STMap undistort: src = the user's Read, stmap = our Read, uv = its layer",
                  und.input(0) is not None and und.input(0).name() == "plate" and und.input(1) is not None
                  and und.input(1).knob("l2s_ref") is not None and not und.hasError(), describe(und))
            w, h = und.width(), und.height()
            check("STMap output has the plate's size", (w, h) == (user["read"].width(), user["read"].height()), [w, h])

    after = snapshot()
    diffs = compare(before, after)
    check("zero writes to the user's script", not diffs, diffs[:20])
    RESULT["user_nodes_checked"] = sorted(before["nodes"])

    # (after the zero-write check: sampling the depth map moves the current frame)
    if cam is not None and geo is not None:
        RESULT["one_space"] = [one_space(cam, geo, depth, f) for f in (1, 20, 40)]
        f1 = RESULT["one_space"][0]
        check("points in front of the camera and in its picture", f1["inside"] > 0.6 * max(1, f1["points"]),
              {k: f1.get(k) for k in ("points", "front", "inside", "z_over_depth_median", "stage_meters_per_unit")})

    # a movie result: its frames from the file, its first frame at the job's first (the Root's first frame)
    group = host.make_group("l2s_movie_test", {"node": "", "version": 9, "tool": "movie"})
    refs = host.import_result({"name": "clip", "type": "video", "files": [movie], "sequence": False,
                               "plate": {"frame_offset": CONFIG["plate_offset"]}}, movie, group, "l2s_movie")
    mov = _find_ours(refs[0])
    check("movie Read: the file's frames, starting at the Root's first frame",
          [mov["first"].value(), mov["last"].value(), mov["frame_mode"].value(), mov["frame"].value()]
          == [1, 37, "start at", "1"] and mov.lastFrame() == 37,
          [mov["first"].value(), mov["last"].value(), mov["frame_mode"].value(), mov["frame"].value(), mov.lastFrame()])

    saved = os.path.join(CONFIG["work"], "saved_test.nk").replace("\\", "/")
    nuke.scriptSaveAs(saved, overwrite=1)
    nuke.scriptClear()
    nuke.scriptOpen(saved)
    host2 = NukeHost()
    nodes = host2.nodes()
    back = next((host2.load(n) for n in nodes if (host2.load(n).get("job") or {}).get("id") == state["job"]["id"]), {})
    check("state reads back after save and open", bool(back.get("versions")) and back.get("job", {}).get("id") == state["job"]["id"],
          {"nodes": len(nodes), "job": back.get("job")})
    check("project dir follows the saved script", host2.project_dir().replace("\\", "/") == os.path.dirname(saved),
          host2.project_dir())


try:
    main()
except Exception:  # noqa: BLE001
    RESULT["errors"].append(traceback.format_exc())
    print(traceback.format_exc())
ok = not RESULT["errors"] and all(c["ok"] for c in RESULT["checks"].values())
RESULT["ok"] = ok
with open(CONFIG["result"], "w", encoding="utf-8") as f:
    json.dump(RESULT, f, ensure_ascii=False, indent=1, default=str)
print("RESULT", "ok" if ok else "failed")
