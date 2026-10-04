"""The framework's pure parts, no server and no DCC: names and paths, the contract (conditions, values, scene context),
the tool order, the delivery's files. Run: python clients/common/tests/test_units.py"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from lab2shot_dcc import catalog, contract, paths, results, safety, usd_camera  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


def test_names():
    weird = ["全身动作 · SAM 3D Body", "my scene 😀 v2", "a" * 300, "123start", "CON", "", "名字/../../etc", "Émile"]
    cleaned = [safety.clean(w) for w in weird]
    check("clean: ASCII word, letter first, <= 60", all(c and c[0].isalpha() and c.isascii() and len(c) <= 60
                                                        and all(ch.isalnum() or ch == "_" for ch in c) for c in cleaned), str(cleaned))
    check("clean: different texts stay different", len(set(cleaned)) == len(cleaned), str(cleaned))
    check("clean: accents kept as letters", safety.clean("Émile") == "Emile", safety.clean("Émile"))
    check("clean_file: extension and frame kept", safety.clean_file("背板 plate.1001.exr") .endswith(".1001.exr"),
          safety.clean_file("背板 plate.1001.exr"))
    check("clean_file: <= 80", len(safety.clean_file("x" * 300 + ".fbx")) <= 80)
    taken = {"lab2shot1_v001", "lab2shot1_v001_002"}
    check("numbered: first free", safety.numbered("lab2shot1_v001", taken.__contains__) == "lab2shot1_v001_003")
    check("numbered: within the limit", len(safety.numbered("a" * 70, lambda n: True if n == "a" * 60 else False, 60)) <= 60)
    with tempfile.TemporaryDirectory() as d:
        check("under: inside", (safety.under(d, "scene/全身 角色.fbx") or "").startswith(d))
        check("under: never out", safety.under(d, "../x") is None and safety.under(d, "a/../../x") is None
              and safety.under(d, "") is None)
        check("writable: a folder", safety.writable(os.path.join(d, "new")) == "")
        open(os.path.join(d, "a.fbx"), "w").close()
        check("free_path: numbered", os.path.basename(safety.free_path(d, "a.fbx")) == "a_002.fbx")


TOOL = {
    "id": "t", "name": "人体", "source": "preset", "category": {"path": ["人体动捕", "全身"]},
    "inputs": [
        {"param": "input", "key": "read.path", "type": "image", "kinds": [], "optional": False, "when": None, "widget": "sequence"},
        {"param": "cam_fbx_path", "key": "cam_fbx.path", "type": "scene.camera", "kinds": ["camera"], "optional": True,
         "when": {"cam_src": 1}, "widget": "file"},
    ],
    "delivers": [{"kinds": ["camera", "character"], "settings": [{"name": "scene", "kinds": ["camera", "character"]}]}],
    "exposed": [
        {"name": "input", "target": ["read.path"], "value": "", "param": {"type": "string", "widget": "sequence"}},
        {"name": "first_frame", "target": ["read.first"], "value": None, "param": {"type": "integer", "nullable": True}},
        {"name": "cam_src", "target": ["cam_src.value"], "value": 3, "widget": "menu",
         "options": [{"value": 3, "label": "ViPE"}, {"value": 1, "label": "FBX"}], "param": {"type": "integer"}},
        {"name": "cam_fbx_path", "target": ["cam_fbx.path"], "value": "", "hide_when": "cam_src != 1", "param": {"type": "string", "widget": "file"}},
        {"name": "cam_fbx_camera", "target": ["cam_fbx.camera"], "value": "", "hide_when": "cam_src != 1",
         "param": {"type": "string", "widget": "choice", "derived_from": ["path"]}},
        {"name": "focal", "target": ["cam.focal_mm"], "value": None, "hide_when": "cam_src in [1, 2]",
         "param": {"type": "number", "nullable": True, "minimum": 1}},
        {"name": "filmback", "target": ["cam.filmback_mm"], "value": None, "disable_when": "not focal", "param": {"type": "number", "nullable": True}},
        {"name": "cook", "target": ["deliver.cook"], "value": None, "param": {"widget": "button"}},
        {"name": "fps", "target": ["out.fps"], "value": 24.0, "wired": "「帧率」的「值」", "fallback": False, "param": {"type": "number"}},
    ],
}


def test_contract():
    shown = [x["name"] for x in contract.parameters(TOOL)]
    check("parameters: inputs, derived and buttons left out", shown == ["first_frame", "cam_src", "focal", "filmback", "fps"], str(shown))
    values = contract.current(TOOL, {})
    focal = next(x for x in TOOL["exposed"] if x["name"] == "focal")
    filmback = next(x for x in TOOL["exposed"] if x["name"] == "filmback")
    check("hide_when (server rules): focal shown with ViPE", not contract.hidden(focal, values))
    check("hide_when: focal hidden with FBX camera", contract.hidden(focal, {**values, "cam_src": 1}))
    check("disable_when: filmback greyed without focal", contract.disabled(filmback, values))
    fps = next(x for x in TOOL["exposed"] if x["name"] == "fps")
    check("wired: greyed", contract.disabled(fps, values))
    check("refuses: below minimum", contract.refuses(focal, 0.5) is not None and contract.refuses(focal, 35.0) is None)
    check("refuses: menu value", contract.refuses(next(x for x in TOOL["exposed"] if x["name"] == "cam_src"), 7) is not None)
    fill = contract.scene_fill(TOOL, {"first_frame": 1001.0, "focal": 35, "fps": 30.0, "nothing": 1})
    check("scene fill: what the tool takes, typed", fill == {"first_frame": 1001, "focal": 35.0}, str(fill))
    sent = contract.submission(TOOL, {"scene_values": fill, "values": {"focal": 50.0},
                                      "files": {"input": "C:/plate.1001.exr", "cam_fbx_path": "C:/cam.fbx"}})
    check("submission: user above scene, files, menus of bound inputs", sent == {"first_frame": 1001, "focal": 50.0, "cam_src": 1,
                                                                          "input": "C:/plate.1001.exr", "cam_fbx_path": "C:/cam.fbx"}, str(sent))
    try:
        contract.submission(TOOL, {"values": {"focal": -1.0}})
        check("submission: a refused value is never sent", False)
    except ValueError:
        check("submission: a refused value is never sent", True)


def test_condition_words():
    """The server's condition rules, loaded on their own, say their sentences in the plugin's language through the
    transport's words (paths.conditions_module sets the `say` / `joined` hooks): the page's ui.conditions.* keys."""
    client, cond = paths.client_module(), paths.conditions_module()
    focal = next(x for x in TOOL["exposed"] if x["name"] == "focal")
    was = client.get_lang()
    try:
        for lang in client.LANGS:
            client.set_lang(lang)
            said = contract.refuses(focal, 0.5)
            want = client.text("ui.conditions.below_min", min=format(1, "g"))
            check(f"conditions say ({lang}): the plugin's words", said == want, f"{said!r} != {want!r}")
            check(f"conditions say ({lang}): its language", any("\u4e00" <= c <= "\u9fff" for c in said) == (lang == "zh"), said)
            wrong = cond.problem("a ==", ["a"])
            check(f"conditions say ({lang}): a syntax error in words", bool(wrong) and "ui.conditions" not in wrong, str(wrong))
    finally:
        client.set_lang(was)


def test_ranking():
    tools = [
        {"id": "a", "name": "抠像", "source": "preset", "inputs": [{"type": "image"}], "delivers": [{"kinds": []}]},
        {"id": "b", "name": "人体", "source": "preset", "inputs": [{"type": "image"}, {"type": "scene.camera"}],
         "delivers": [{"kinds": ["character"]}]},
        {"id": "c", "name": "重定向", "source": "preset", "inputs": [{"type": "scene.character|scene.skeleton"}],
         "delivers": [{"kinds": ["skeleton"]}]},
        {"id": "n", "name": "节点", "source": "node", "inputs": [], "delivers": []},
        {"id": "m", "name": "我的", "source": "mine", "inputs": [], "delivers": [{"kinds": ["camera"]}]},
    ]
    order = catalog.ranked(tools, ["scene.camera", "image"])
    check("ranking: grouped by source", [s for s, _l, _t in order] == ["preset", "node", "mine"])
    check("ranking: what takes the selection first, 3D before 2D", [t["id"] for t in order[0][2]] == ["b", "a", "c"],
          str([t["id"] for t in order[0][2]]))
    check("ranking: nothing filtered", sum(len(t) for _s, _l, t in order) == len(tools))
    order = catalog.ranked(tools, ["scene.skeleton"])
    check("ranking: a skeleton selected puts retargeting first", order[0][2][0]["id"] == "c")


def test_delivery():
    manifest = {"outputs": [{"name": "场景 一", "type": "scene", "main": "场景 一/场景 一_ML_Lab2Shot_X.fbx",
                             "files": ["场景 一/场景 一_ML_Lab2Shot_X.fbx", "场景 一/场景 一.lab2shot.json"]},
                            {"name": "plate", "type": "image.3", "main": "plate/plate.####.exr",
                             "files": ["plate/plate.1001.exr", "plate/plate.1002.exr", "plate/plate.lab2shot.json"]}]}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        top = "全身动作_ML_Lab2Shot_u3_abc"
        z.writestr(f"{top}/lab2shot.json", json.dumps(manifest, ensure_ascii=False))
        for o in manifest["outputs"]:
            for f in o["files"]:
                z.writestr(f"{top}/{f}", b"x")
        z.writestr(f"{top}/../../evil.txt", b"no")
    with tempfile.TemporaryDirectory() as d:
        archive = os.path.join(d, "a.zip")
        open(archive, "wb").write(buffer.getvalue())
        stage = results.staging(d)
        mapping = results.unpack(archive, stage)
        check("unpack: nothing outside", not os.path.exists(os.path.join(d, "evil.txt")))
        local = [os.path.relpath(p, stage) for p in mapping.values()]
        check("unpack: every local name ASCII", all(p.isascii() for p in local), str(local))
        version = os.path.join(d, "v001")
        mapping = results.settle(stage, version, mapping)
        items = results.items(results.read_manifest(version), mapping,
                              [{"settings": [{"name": "场景 一", "kinds": ["camera"], "format": "fbx"}]}])
        check("items: original name, kinds, main file", items[0]["name"] == "场景 一" and items[0]["kinds"] == ["camera"]
              and items[0]["main"].endswith(".fbx") and os.path.isfile(items[0]["main"]), str(items[0]))
        check("items: a sequence's main is its first frame", items[1]["main"].endswith("plate.1001.exr"), items[1]["main"])
        os.remove(items[0]["main"])
        try:
            results.items(results.read_manifest(version), mapping)
            check("items: a missing file is refused", False)
        except RuntimeError:
            check("items: a missing file is refused", True)
        check("next version", results.next_version(d, [3]) == 4)
        bad = os.path.join(d, "bad.zip")
        data = bytearray(buffer.getvalue())
        i = data.find(b"x", 200)
        data[i] = ord("y")
        open(bad, "wb").write(bytes(data))
        try:
            results.unpack(bad, results.staging(d))
            check("unpack: a damaged zip is refused", False)
        except Exception:  # noqa: BLE001
            check("unpack: a damaged zip is refused", True)


def test_focus():
    import os
    import tempfile

    sys.path.insert(0, HERE)
    from fake_host import FakeHost
    from lab2shot_dcc.plugin import Plugin

    plugin = Plugin(FakeHost(tempfile.mkdtemp()))
    tool = {"id": "t", "exposed": [{"name": "cook_detect", "target": ["detect.cook"], "param": {"widget": "button"}},
                                   {"name": "people_picks_pick", "target": ["pick.picks_pick"], "param": {"widget": "button"}}],
            "delivers": [{"node": "deliver"}]}
    plugin.tools = [tool]
    state = {"tool": {"id": "t"}, "job": {"id": "abc"}}
    check("focus: none given, the job's graph without a focus", plugin.web_address(state) == "/#job=abc",
          plugin.web_address(state))
    tool["focus"] = "solve"
    check("focus: the signature's own when it gives one", plugin.web_address(state) == "/#job=abc&focus=solve")
    tool["focus"] = None
    check("focus: null from the signature: the job alone", plugin.web_address(state) == "/#job=abc")
    check("focus: no job yet: the editor", plugin.web_address({"tool": {"id": "t"}}) == "/")


def test_backplate():
    """The picture behind a delivered camera: the bound picture; else the bound camera's own; neither: none,
    said (输入不全) and the import goes on."""
    import tempfile

    sys.path.insert(0, HERE)
    from fake_host import FakeHost
    from lab2shot_dcc import jobs

    host = FakeHost(tempfile.mkdtemp())
    cam_with = host.add_object("shotCam", "scene.camera")
    host.scene[cam_with]["picture"] = {"file": "C:/plates/sh010.1001.exr", "sequence": True, "frame_offset": 1000,
                                       "colorspace": "ACEScg"}
    cam_without = host.add_object("bareCam", "scene.camera")
    pic = {"file": "D:/in/plate_0259.jpg", "type": "image", "label": "plate"}
    camera = lambda ref: {"ref": ref, "type": "scene.camera", "label": "cam"}  # noqa: E731

    got, why = jobs.backplate({"bindings": {"input": pic, "cam_fbx_path": camera(cam_with)}}, host.camera_picture)
    check("backplate: a bound picture wins over the camera's own", got.get("from") == "input"
          and got.get("file") == pic["file"] and got.get("sequence") is True and not why, str(got))
    got, why = jobs.backplate({"bindings": {"cam_fbx_path": camera(cam_with)}}, host.camera_picture)
    check("backplate: no picture: the bound camera's own", got.get("from") == "camera" and got.get("frame_offset") == 1000
          and got.get("colorspace") == "ACEScg" and not why, str(got))
    got, why = jobs.backplate({"bindings": {"cam_fbx_path": camera(cam_without)}}, host.camera_picture)
    check("backplate: neither: none, and said why", got == {} and why == paths.text(jobs.NO_BACKPLATE), str(got))
    got, why = jobs.backplate({"bindings": {}}, host.camera_picture)
    check("backplate: nothing bound: none, and said why", got == {} and why == paths.text(jobs.NO_BACKPLATE))
    check("backplate: a single picked still is not a sequence",
          jobs.backplate({"bindings": {"i": {"file": "/x/still.png", "type": "image"}}}, host.camera_picture)[0]["sequence"] is False)

    # through the import step: plates only behind the cameras the version brought in, inside its namespace
    run = jobs.Run(host, None, host.create_node("lab2shot1"), None)
    result_cam = host.add_object("l2s_v001:cam", "scene.camera")
    model = host.add_object("l2s_v001:mesh", "scene.model")
    made, plate, note = run._backplates([result_cam, model], {"bindings": {"cam_fbx_path": camera(cam_with)}}, "l2s_v001")
    check("import: one plate behind the delivered camera, from the camera's own picture",
          len(made) == 1 and host.scene[made[0]]["camera"] == result_cam and plate == {"from": "camera", "file": "C:/plates/sh010.1001.exr"}
          and not note, f"{made} {plate} {note}")
    made, plate, note = run._backplates([result_cam], {"bindings": {"cam_fbx_path": camera(cam_without)}}, "l2s_v001")
    check("import: input incomplete: no plate, a sentence, no error", made == [] and plate == {} and note == paths.text(jobs.NO_BACKPLATE))
    made, plate, note = run._backplates([model], {"bindings": {"input": pic}}, "l2s_v001")
    check("import: no camera delivered: nothing hung, nothing said", made == [] and plate == {} and note == "")
    host.hang_picture = lambda *a: (_ for _ in ()).throw(RuntimeError("file node refused"))
    made, plate, note = run._backplates([result_cam], {"bindings": {"input": pic}}, "l2s_v001")
    check("import: a plate that cannot be made is said, the import is not failed", made == [] and "file node refused" in note, note)


def test_scene_context():
    """The scene's context, one rule for every DCC (lab2shot_dcc.context): the frame range in the bound picture's own
    frame numbers, its colour space, the unit the host reads, the bound camera's lens or else the picture's camera's."""
    sys.path.insert(0, HERE)
    from fake_host import FakeHost
    from lab2shot_dcc.context import scene_context

    host = FakeHost(tempfile.mkdtemp())
    host.context = {"fps": 25.0, "unit": "m", "range": (1, 48)}
    cam = host.add_object("shotCam", "scene.camera")
    host.scene[cam]["lens"] = (35.0, 36.0)
    other = host.add_object("planeCam", "scene.camera")
    host.scene[other]["lens"] = (50.0, 24.892)
    seq = {"file": "C:/plates/sh010.1001.exr", "type": "image", "sequence": True, "frame_offset": 1000,
           "colorspace": "ACEScg", "camera": other}
    got = scene_context(host, {"input": seq})
    check("context: a sequence's range at the picture's frames", (got.get("first_frame"), got.get("last_frame")) == (1001, 1048), str(got))
    check("context: the picture's colour space in, fps and the host's unit",
          got.get("colorspace_in") == "ACEScg" and got.get("fps") == 25.0 and got.get("unit") == "m", str(got))
    check("context: no camera bound: the picture's own camera", (got.get("focal"), got.get("filmback")) == (50.0, 24.892), str(got))
    got = scene_context(host, {"input": seq, "cam": {"ref": cam, "type": "scene.camera"}})
    check("context: a bound camera wins", (got.get("focal"), got.get("filmback")) == (35.0, 36.0), str(got))
    still = {"file": "D:/in/still.png", "type": "image", "sequence": False, "frame_offset": 0}
    got = scene_context(host, {"input": still})
    check("context: a still: no range, no lens, no colour space",
          not {"first_frame", "last_frame", "focal", "colorspace_in"} & set(got), str(got))
    host.context = {"fps": 24.0, "unit": "", "range": (1, 10)}
    check("context: a unit the tools do not take is left out", "unit" not in scene_context(host, {}))
    check("context: a camera bound by file only has no lens",
          "focal" not in scene_context(host, {"cam": {"file": "/x/cam.fbx", "type": "scene.camera"}}))


def test_sample():
    """Animation read in short main-thread steps (lab2shot_dcc.mainthread.sample), joined in frame order."""
    sys.path.insert(0, HERE)
    from fake_host import FakeHost
    from lab2shot_dcc import mainthread

    host = FakeHost(tempfile.mkdtemp())
    asked = []
    got = mainthread.sample(host, 1, 60, lambda frames: asked.append(frames) or [f * 10 for f in frames])
    check("sample: every frame once, in order", got == [f * 10 for f in range(1, 61)], str(got))
    check("sample: SAMPLE_STEP frames per step", [len(a) for a in asked] == [25, 25, 10], str([len(a) for a in asked]))
    got = mainthread.sample(host, 5, 30, lambda frames: {"a": [("a", f) for f in frames], "b": [f for f in frames]})
    check("sample: dicts joined key by key", got["a"] == [("a", f) for f in range(5, 31)] and got["b"] == list(range(5, 31)), str(got))
    check("sample: an empty range reads nothing", mainthread.sample(host, 10, 9, lambda frames: [1]) is None)


def test_no_dcc():
    import importlib
    import pkgutil

    import lab2shot_dcc

    for m in pkgutil.iter_modules(lab2shot_dcc.__path__):
        if m.name in ("qt",):
            continue
        importlib.import_module(f"lab2shot_dcc.{m.name}")
    loaded = [n for n in sys.modules if n.split(".")[0] in ("maya", "hou", "nuke", "pymel")]
    check("framework imports no DCC module", not loaded, str(loaded))
    folder = os.path.dirname(lab2shot_dcc.__file__)
    said = []
    for base, _d, files in os.walk(folder):
        for f in files:
            if f.endswith(".py"):
                text = open(os.path.join(base, f), encoding="utf-8").read()
                for word in ("import maya", "from maya", "import hou", "import nuke", "cmds."):
                    if word in text:
                        said.append(f"{f}: {word}")
    check("framework names no DCC", not said, str(said))


def test_usd_camera():
    samples = []
    for f in range(1001, 1006):
        a = math.radians(10 * (f - 1001))
        c, s = math.cos(a), math.sin(a)
        samples.append({"frame": f, "matrix": [c, 0, s, f - 1000.0, 0, 1, 0, 2.0, -s, 0, c, 500.0, 0, 0, 0, 1],
                        "focal": 35 + (f - 1001), "haperture": 36, "vaperture": 24, "win": (0.1, -0.05)})
    with tempfile.TemporaryDirectory() as tmp:
        path = usd_camera.write(os.path.join(tmp, "cam.usda"), "shot cam 1", samples, 24.0)
        text = open(path, encoding="utf-8").read()
        check("centimetre stage, Y up", "metersPerUnit = 0.01" in text and 'upAxis = "Y"' in text)
        check("camera name cleaned", 'def Camera "shot_cam_1"' in text)
        try:
            import numpy as np
            from pathlib import Path

            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))  # the repository: the server's reader
            from lab2shot.formats.usd import reader
        except ImportError:
            print("SKIP read back with the server's reader (lab2shot / pxr not importable here)")
            return
        listing = reader.listing(path)
        entry = listing.entries[0]
        cam = reader.camera(Path(path), entry, (1920, 1280))
        want = np.array([np.array(s["matrix"]).reshape(4, 4) for s in samples])
        check("read back: frames", tuple(cam.frames) == tuple(range(1001, 1006)), cam.frames)
        check("read back: matrices", float(np.abs(np.asarray(cam.cam_to_world) - want).max()) < 1e-9)
        check("read back: focal per frame", list(np.asarray(cam.focal_mm).ravel()) == [35, 36, 37, 38, 39])
        check("read back: lens centre (mm) from Nuke's window translate",
              bool(np.allclose(np.asarray(cam.center_mm)[0], [0.1 * 18, -0.05 * 12], atol=1e-6)), str(cam.center_mm[:1]))


if __name__ == "__main__":
    test_names()
    test_contract()
    test_condition_words()
    test_ranking()
    test_delivery()
    test_focus()
    test_backplate()
    test_scene_context()
    test_sample()
    test_no_dcc()
    test_usd_camera()
    print("ALL OK" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
