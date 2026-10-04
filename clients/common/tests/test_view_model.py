"""The panel's view model (lab2shot_dcc.view_model), no Qt and no DCC: the cards and their 「需要：」, the tabs, the
order and 「适合当前选中」 (with each host's preferences), the search, 「最近用过」, the tool page's inputs and its common /
advanced parameters, the versions, the scene's tasks, both languages (English: not one Chinese character); and the
plugin's own additions — 最近用过 kept per server, a kept version imported later without fetching it again.

Run: python clients/common/tests/test_view_model.py"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
os.environ["USERPROFILE"] = tempfile.mkdtemp(prefix="l2s_vm_")  # the plugin's settings: a folder of this test's own

from lab2shot_dcc import paths, view_model as vm  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


CATEGORIES = [
    {"id": "workflow", "label": "CG 预设工作流", "color": "#FF9F0A", "rank": -1, "parent": ""},
    {"id": "camera_track", "label": "摄影机", "color": "#FFD60A", "rank": 0, "parent": ""},
    {"id": "camera_traj", "label": "相机轨迹", "color": "#FFD60A", "rank": 0, "parent": "camera_track"},
    {"id": "depth", "label": "深度与法线", "color": "#AC8E68", "rank": 1, "parent": ""},
    {"id": "body", "label": "人物动作", "color": "#64D2FF", "rank": 2, "parent": ""},
    {"id": "read", "label": "读取", "color": "#8E8E93", "rank": 0, "parent": "", "section": "tools"},
]


def tool(tid, name, source="preset", cat=(), inputs=(), delivers=(), intro="", exposed=()):
    return {"id": tid, "name": name, "intro": intro, "source": source,
            "category": {"id": cat[-1] if cat else "", "path": [c.upper() for c in cat], "ids": list(cat)},
            "inputs": [dict(i) for i in inputs], "delivers": [dict(d) for d in delivers], "exposed": list(exposed)}


IMG = {"param": "input", "key": "r.path", "type": "image", "optional": False, "widget": "sequence", "label": "画面"}
VID = {"param": "input", "key": "r.path", "type": "image|video", "optional": False, "widget": "sequence", "label": "画面"}
CAM = {"param": "cam_fbx_path", "key": "c.path", "type": "scene.camera", "optional": True, "widget": "file", "label": "摄影机"}
CHAR = {"param": "char_fbx_path", "key": "ch.path", "type": "scene.character", "optional": False, "widget": "file",
        "label": "角色"}
D3 = {"types": ["scene"], "kinds": ["camera"]}
D2 = {"types": ["image.1"], "kinds": []}

TOOLS = [
    tool("depth", "深度 · DA3", cat=("depth",), inputs=[IMG], delivers=[D2], intro="单目深度\n第二行"),
    tool("solve", "VidMap 相机解算", cat=("camera_track", "camera_traj"), inputs=[VID, CAM], delivers=[D3],
         intro="从画面解算摄影机"),
    tool("body", "全身动作", cat=("body",), inputs=[IMG, CAM, CHAR], delivers=[D3]),
    tool("wf", "工作流 A", cat=("workflow",), inputs=[IMG], delivers=[D3, D2]),
    tool("mine1", "我的图", source="mine"),
    tool("node~x", "节点 X", source="node", cat=("depth",), inputs=[IMG], delivers=[D2]),
    tool("node~read", "读取", source="node", cat=("read",), delivers=[D2]),
]

EXPOSED = [
    {"name": "first_frame", "target": ["r.first"], "value": None, "label": "首帧", "group": [], "folded": False,
     "param": {"type": "integer", "nullable": True}},
    {"name": "cam_src", "target": ["s.value"], "value": 3, "label": "相机来源", "widget": "menu", "group": [], "folded": False,
     "options": [{"value": 3, "label": "ViPE"}, {"value": 1, "label": "FBX"}], "param": {"type": "integer"}},
    {"name": "focal", "target": ["cam.focal"], "value": None, "label": "焦距", "group": ["高级"], "folded": True,
     "hide_when": "cam_src in [1, 2]", "param": {"type": "number", "nullable": True}},
    {"name": "seed", "target": ["m.seed"], "value": 0, "label": "随机种子", "group": ["高级"], "folded": True,
     "param": {"type": "integer"}},
]


def lang(code):
    paths.client_module().set_lang(code)


def all_text(obj) -> list[str]:
    """Every string a view model result holds (to look for Chinese)."""
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() if k not in ("x", "item", "values", "objects") for s in all_text(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in all_text(v)]
    return []


def test_words():
    lang("zh")
    check("type name (zh)", vm.type_label("scene.camera") == paths.text("type.scene.camera.label") and
          vm.type_label("scene.camera") != "scene.camera")
    check("type name: unknown type said as its id", vm.type_label("no.such") == "no.such")
    check("alternatives joined by type.or", vm.type_names("image|video") == paths.text("type.image.label")
          + paths.text("type.or") + paths.text("type.video.label"), vm.type_names("image|video"))
    lang("en")
    check("type name (en)", vm.type_names("image|video") == "Image or Video", vm.type_names("image|video"))
    lang("zh")


def test_cards():
    lang("zh")
    c = vm.card(TOOLS[2], {"body": "#64D2FF"})
    check("card: needs only the required, each type once", [t for t, _ in c["needs"]] == ["image", "scene.character"],
          str(c["needs"]))
    check("card: optional apart", [t for t, _ in c["optional"]] == ["scene.camera"], str(c["optional"]))
    check("card: category colour", c["colour"] == "#64D2FF" and c["category"] == "body")
    c = vm.card(TOOLS[0])
    check("card: intro's first line", c["intro"] == "单目深度")
    check("card: no category: the default grey", vm.card(TOOLS[4])["colour"] == vm.UNPLACED_COLOUR)
    dup = tool("d", "d", inputs=[IMG, {**CAM, "optional": False}, {**IMG, "param": "x"}, {**CAM, "param": "y"}])
    c = vm.card(dup)
    check("card: a type needed is never listed optional too", [t for t, _ in c["needs"]] == ["image", "scene.camera"]
          and c["optional"] == [], str(c))
    check("needs line", paths.text("type.image.label") in vm.needs_line(vm.card(TOOLS[0])))


def test_home():
    lang("zh")
    h = vm.home(TOOLS, CATEGORIES, [], recent=[])
    ids = [t["id"] for t in h["tabs"]]
    check("tabs: all first, the templates tree's categories with tools in the server's order (workflow first), the "
          "node menu's own band as one 「节点」, mine", ids == ["all", "workflow", "camera_track", "depth", "body",
                                                              "source:node", "source:mine"], str(ids))
    hn = vm.home(TOOLS, CATEGORIES, [], tab="source:node")
    check("「节点」: the single nodes outside the templates tree", [c["id"] for c in hn["cards"]] == ["node~read"])
    check("no selection: nothing suggested", h["suggested"] == [])
    check("all tools listed, none filtered", len(h["cards"]) == len(TOOLS))
    check("order without a selection: what the host prefers (3D first), then the server's",
          [c["id"] for c in h["cards"]][:3] == ["solve", "body", "wf"], str([c["id"] for c in h["cards"]]))
    h = vm.home(TOOLS, CATEGORIES, ["scene.camera", "image"], recent=[])
    check("suggested: what takes the selection, the more it takes the higher",
          [c["id"] for c in h["suggested"]][:2] == ["solve", "body"], str([c["id"] for c in h["suggested"]]))
    check("suggested: at most SUGGEST_MOST", len(h["suggested"]) <= vm.SUGGEST_MOST)
    many = [tool(f"t{i}", f"t{i}", inputs=[IMG]) for i in range(10)]
    check("suggested: capped", len(vm.home(many, [], ["image"])["suggested"]) == vm.SUGGEST_MOST)
    h = vm.home(TOOLS, CATEGORIES, ["scene.character"])
    check("suggested: only tools that take it", [c["id"] for c in h["suggested"]] == ["body"])
    nuke = vm.home(TOOLS, CATEGORIES, [], prefers=("image", "video", "scene"))
    check("another host's preferences (Nuke: pictures first)", [c["id"] for c in nuke["cards"]][0] in ("depth", "wf"),
          str([c["id"] for c in nuke["cards"]]))
    h = vm.home(TOOLS, CATEGORIES, [], tab="depth")
    check("a category's tab: its tools (node tools too)", sorted(c["id"] for c in h["cards"]) == ["depth", "node~x"])
    h = vm.home(TOOLS, CATEGORIES, [], tab="source:mine")
    check("my templates' tab", [c["id"] for c in h["cards"]] == ["mine1"])
    h = vm.home(TOOLS, CATEGORIES, [], tab="gone")
    check("a tab that no longer exists: all", h["tab"] == "all" and len(h["cards"]) == len(TOOLS))
    h = vm.home(TOOLS, CATEGORIES, [], search="相机")
    check("search: name", [c["id"] for c in h["cards"]] == ["solve"], str([c["id"] for c in h["cards"]]))
    h = vm.home(TOOLS, CATEGORIES, [], search="DEPTH da3")
    check("search: every word, any case, category ids too", [c["id"] for c in h["cards"]] == ["depth"],
          str([c["id"] for c in h["cards"]]))
    h = vm.home(TOOLS, CATEGORIES, ["image"], search=paths.text("type.scene.character.label"))
    check("search: by input type name; nothing suggested while searching",
          [c["id"] for c in h["cards"]] == ["body"] and h["suggested"] == [], str([c["id"] for c in h["cards"]]))
    h = vm.home(TOOLS, CATEGORIES, [], tab="recent", recent=["body", "gone", "depth"])
    check("recent: its own tab, last; in the order used, gone ones left out",
          h["tabs"][-1]["id"] == "recent" and [c["id"] for c in h["cards"]] == ["body", "depth"])
    check("recent list: newest first, once, at most RECENT_MOST",
          vm.recent_after(["a", "b", "c"], "b") == ["b", "a", "c"] and len(vm.recent_after([str(i) for i in range(20)], "x"))
          == vm.RECENT_MOST)
    h = vm.home([], CATEGORIES, ["image"])
    check("no tools: only 「全部」, nothing shown", [t["id"] for t in h["tabs"]] == ["all"] and h["cards"] == [])


def test_tool_page():
    lang("zh")
    t = tool("body", "全身动作", inputs=[IMG, CAM, CHAR], exposed=EXPOSED)
    state = {"bindings": {"input": {"file": "/p/a.1001.exr", "files": ["/p/a.1001.exr", "/p/a.1002.exr", "/p/a.1003.exr"],
                                    "type": "image"}},
             "scene_values": {"first_frame": 1001}, "values": {}}
    rows = vm.inputs(t, state, describe=lambda b: "described")
    check("inputs: one per slot, in order", [r["param"] for r in rows] == ["input", "cam_fbx_path", "char_fbx_path"])
    check("inputs: bound said by the host; a sequence's first, last and count",
          rows[0]["bound"] and rows[0]["text"] == "described" and "a.1001.exr" in rows[0]["detail"]
          and "a.1003.exr" in rows[0]["detail"] and "3" in rows[0]["detail"], str(rows[0]))
    check("inputs: unbound says required / optional",
          rows[2]["text"] == paths.text("dcc.ui.input.required") and rows[1]["text"] == paths.text("dcc.ui.input.optional"))
    check("inputs: a host picture's frames from the page's scan",
          vm.inputs(t, {"bindings": {"input": {"ref": "x", "file": "/p/a.1001.exr", "sequence": True}}},
                    frames=lambda b: "1001 … 1010")[0]["detail"] == "1001 … 1010")
    p = vm.parameters(t, state)
    check("parameters: the template's folded groups under 「高级」",
          [r["x"]["name"] for r in p["common"]] == ["first_frame", "cam_src"]
          and [r["x"]["name"] for r in p["advanced"]] == ["focal", "seed"], str({k: [r["x"]["name"] for r in v] for k, v in p.items()}))
    check("parameters: 「来自场景」 marked", p["common"][0]["from_scene"] and not p["common"][1]["from_scene"])
    p = vm.parameters(t, {**state, "values": {"cam_src": 1}})
    check("parameters: hidden by the interface's own conditions", [r["x"]["name"] for r in p["advanced"]] == ["seed"])
    check("missing inputs", vm.missing_inputs(t, state) == ["角色"])
    v = vm.versions({"versions": [{"version": 1, "made": "2026-10-03 10:00:00", "tool": "x", "imported": True,
                                   "objects": ["a"]},
                                  {"version": 2, "made": "2026-10-03 11:00:00", "tool": "x", "imported": False}]})
    check("versions: newest first, v-numbers, imported or not",
          [x["label"] for x in v] == ["v002", "v001"] and not v[0]["imported"] and v[1]["objects"] == ["a"])
    check("frames text: fewer than two says nothing", vm.frames_text(["a.1.exr"]) == "")
    with tempfile.TemporaryDirectory() as d:
        for n in ("sh.1001.exr", "sh.1002.exr", "sh.1003.exr", "other.1001.exr", "notes.txt"):
            open(os.path.join(d, n), "w").close()
        got = vm.sequence_frames(os.path.join(d, "sh.1002.exr"))
        check("a sequence found beside one frame", got == ["sh.1001.exr", "sh.1002.exr", "sh.1003.exr"], str(got))
        check("a video or a still is no sequence", vm.sequence_frames(os.path.join(d, "clip.mov")) == []
              and vm.sequence_frames(os.path.join(d, "notes.txt")) == [])


def test_auto_bind():
    t = tool("x", "x", inputs=[IMG, CAM, dict(IMG, param="ref", optional=True), dict(IMG, param="hdri", optional=True),
                                CHAR])
    got = [i["param"] for i in vm.auto_bind(t, ["scene.camera", "image"])]
    check("auto bind: required ones, then an optional one only for what no input took",
          got == ["input", "cam_fbx_path"], str(got))
    got = [i["param"] for i in vm.auto_bind(t, ["scene.character"])]
    check("auto bind: only what takes the selection", got == ["char_fbx_path"], str(got))
    check("auto bind: nothing selected, nothing bound", vm.auto_bind(t, []) == [])


def test_choose_node():
    """A card chosen on the home page (one task, one node): the node shown when it is there and has no tool, else new."""
    check("choose: the shown node without a tool is used", vm.choose_node("n1", ["n1", "n2"], {}) == vm.REUSE)
    check("choose: a node with a tool keeps it: new node",
          vm.choose_node("n1", ["n1"], {"tool": {"id": "t"}}) == vm.NEW)
    check("choose: a node only named a tool without id is free", vm.choose_node("n1", ["n1"], {"tool": {}}) == vm.REUSE)
    check("choose: no node shown: new node", vm.choose_node(None, ["n1"], {}) == vm.NEW)
    check("choose: the node shown was deleted: new node", vm.choose_node("gone", ["n1"], {}) == vm.NEW)


def test_connection_state():
    """The top bar's dot and words, from the connection and the tool list."""
    from types import SimpleNamespace as NS

    def conn(server="", token=False, user=None):
        return NS(server=server, has_token=lambda: token, user=user)

    def plugin(tools=(), error="", loading=False):
        return NS(tools=list(tools), tools_error=error, loading=loading)

    ready = plugin(tools=[{"id": "t"}])
    check("connection: no server: not connected", vm.connection_state(conn(), ready) == ("text-3", paths.text("dcc.ui.top.offline")))
    check("connection: not logged in: orange, the address without its scheme",
          vm.connection_state(conn("https://lab:8766"), ready) == ("orange", "lab:8766"))
    check("connection: the tool list failed: error",
          vm.connection_state(conn("https://lab:8766", True), plugin(error="boom")) == ("error", "lab:8766"))
    check("connection: loading or no tools yet: orange",
          vm.connection_state(conn("https://lab:8766", True), plugin(loading=True)) == ("orange", "lab:8766")
          and vm.connection_state(conn("https://lab:8766", True), plugin())[0] == "orange")
    check("connection: ready: green, who is logged in",
          vm.connection_state(conn("https://lab:8766", True, {"username": "ann"}), ready) == ("green", "lab:8766 · ann"))


def test_input_use():
    """「用选中的」 of an input card: pressable when something selected goes in; else why not."""
    fbx = dict(CAM, accept=[".fbx"])
    row = next(r for r in vm.inputs(tool("cam", "cam", inputs=[fbx]), {}) if r["param"] == "cam_fbx_path")
    got = vm.input_use(row, ["scene.camera"], {"scene.camera": ".fbx"}, "Maya")
    check("use: the selection goes in", got == {"enabled": True, "why": ""}, str(got))
    got = vm.input_use(row, ["scene.camera"], {"scene.camera": ".usda"}, "Nuke")
    check("use: the right kind in a file the input does not read: greyed, says the formats",
          not got["enabled"] and got["why"] == paths.text("dcc.plugin.wrong_format", host="Nuke", suffix=".usda", accept=".fbx"),
          str(got))
    got = vm.input_use(row, ["image"], {}, "Maya", lambda t: "select a camera: " + t)
    check("use: nothing fits: the host's own words", got == {"enabled": False, "why": "select a camera: scene.camera"}, str(got))
    got = vm.input_use(row, [], {}, "Maya", lambda t: "")
    check("use: the host says nothing: the general words",
          got == {"enabled": False, "why": paths.text("dcc.ui.input.nothing_fits")}, str(got))


def test_menu_items():
    """The DCC menu's items, the same in every DCC (lab2shot_dcc.ui.menu): keys in order, separators between groups."""
    from lab2shot_dcc.ui.menu import SEPARATOR, menu_items

    items = menu_items(object(), lambda: None)
    keys = [None if i is SEPARATOR else i[0] for i in items]
    check("menu: items and separators in order", keys == ["dcc.menu.new_node", "dcc.menu.open_panel", None, "dcc.menu.login",
                                                           "dcc.menu.queue", "dcc.menu.open_web", None, "dcc.menu.log_folder"], str(keys))
    check("menu: every item has a word and a callback",
          all(callable(i[1]) and paths.text(i[0]) for i in items if i is not SEPARATOR))


def test_exports():
    """Host.exports: a selected camera goes into the input that reads the file the host writes (Nuke .usda →
    cam_usd_path; Maya .fbx → cam_fbx_path), in 「适合当前选中」, the order and 「用选中的」 alike."""
    from lab2shot_dcc import catalog

    fbx = dict(CAM, accept=[".fbx"])
    usd = dict(CAM, param="cam_usd_path", key="u.path", accept=[".usd", ".usda", ".usdc", ".usdz"])
    t = tool("cam", "cam", inputs=[dict(IMG, accept=[".exr", ".png"]), fbx, usd])
    nuke_like, maya_like = {"scene.camera": ".usda"}, {"scene.camera": ".fbx"}
    got = [i["param"] for i in vm.auto_bind(t, ["scene.camera", "image"], nuke_like)]
    check("exports: a .usda camera into cam_usd_path", got == ["input", "cam_usd_path"], str(got))
    got = [i["param"] for i in vm.auto_bind(t, ["scene.camera"], maya_like)]
    check("exports: an .fbx camera into cam_fbx_path", got == ["cam_fbx_path"], str(got))
    got = [i["param"] for i in vm.auto_bind(t, ["scene.camera"])]
    check("exports: a host that does not say: the first that fits", got == ["cam_fbx_path"], str(got))
    check("exports: takes", catalog.takes(usd, "scene.camera", nuke_like) and not catalog.takes(fbx, "scene.camera", nuke_like))
    only_fbx = tool("f", "f", inputs=[fbx])
    check("exports: a tool reading only .fbx does not take Nuke's camera",
          catalog.score(only_fbx, ["scene.camera"], 0, None, nuke_like)[0] == 0
          and catalog.score(only_fbx, ["scene.camera"], 0, None, maya_like)[0] == -1)


def test_colorspace_for():
    from lab2shot_dcc import results

    table = {"ACEScg": ["ACES - ACEScg"], "sRGB Encoded Rec.709 (sRGB)": ["Utility - sRGB - Texture", "sRGB"]}
    check("colour space: the server's own name", results.colorspace_for("ACEScg", ["Raw", "ACEScg"], table) == "ACEScg")
    check("colour space: by the host's table, a family path's last part",
          results.colorspace_for("ACEScg", ["Input/ACES/ACES - ACEScg"], table) == "Input/ACES/ACES - ACEScg")
    check("colour space: the table in order, no case",
          results.colorspace_for("srgb encoded rec.709 (srgb)", ["linear", "sRGB"], table) == "sRGB")
    check("colour space: none there, none set", results.colorspace_for("ACEScg", ["linear", "sRGB"], table) == "")
    check("colour space: none asked", results.colorspace_for("", ["linear"], table) == "")


def test_tasks():
    lang("zh")
    check("task: new", vm.task_state({}, None) == "new")
    check("task: running", vm.task_state({}, {"done": False, "phase": "running"}) == "running")
    check("task: queued", vm.task_state({}, {"done": False, "phase": "queued"}) == "queued")
    check("task: failed (this session)", vm.task_state({}, {"done": True, "phase": "failed"}) == "failed")
    check("task: failed (kept on the node)", vm.task_state({"job": {"state": "failed"}}, None) == "failed")
    check("task: a version not imported", vm.task_state({"versions": [{"imported": False}]}, None) == "done")
    check("task: imported", vm.task_state({"versions": [{"imported": True}]}, None) == "imported")
    t = vm.task("n1", "lab2shot1", {"tool": {"name": "深度"}}, None)
    check("task: name, tool, state word", t["name"] == "lab2shot1" and t["tool"] == "深度"
          and t["state_label"] == paths.text("dcc.ui.state.new"))


def test_english():
    """In English not one Chinese character comes from the plugin (the tools' own words are the server's: English
    tools here)."""
    lang("en")
    en_tools = [tool("a", "Depth", cat=("depth",), inputs=[dict(IMG, label="Picture"), dict(CAM, label="Camera")], delivers=[D2], intro="Depth from one picture",
                     exposed=[dict(x, label="L", group=["Advanced"] if x["folded"] else []) for x in EXPOSED])]
    cats = [dict(c, label="Depth") for c in CATEGORIES]
    out = [vm.home(en_tools, cats, ["image"], recent=["a"]), vm.card(en_tools[0]), vm.needs_line(vm.card(en_tools[0])),
           vm.inputs(en_tools[0], {}), vm.parameters(en_tools[0], {}), vm.versions({"versions": [{"version": 1}]}),
           [vm.task("n", "node1", {}, s) for s in (None, {"done": False, "phase": "running"})],
           vm.run_line({"text": "Queued", "error": "x"}), vm.frames_text(["a.1.exr", "a.2.exr"])]
    words = [s for s in all_text(out) if vm.has_cjk(s)]
    check("English: no Chinese anywhere", not words, str(words[:5]))
    lang("zh")


def test_plugin_additions():
    """最近用过 per server in the plugin's settings; a version kept (not imported) brought in later from its own folder,
    as the same version, nothing fetched (Plugin.import_version / jobs.Run version)."""
    from fake_host import FakeHost

    from lab2shot_dcc import connection, jobs, results
    from lab2shot_dcc.plugin import Plugin

    lang("zh")
    project = tempfile.mkdtemp(prefix="l2s_vm_project_")
    host = FakeHost(project)
    plugin = Plugin(host)
    plugin.conn.server = "https://a.example"
    plugin.remember_recent("x")
    plugin.remember_recent("y")
    plugin.remember_recent("x")
    plugin.conn.server = "https://b.example"
    plugin.remember_recent("z")
    check("recent: per server", plugin.recent() == ["z"])
    plugin.conn.server = "https://a.example"
    check("recent: newest first, once", plugin.recent() == ["x", "y"], str(plugin.recent()))

    heard = []
    plugin.on_language(lambda: heard.append(paths.client_module().get_lang()))
    plugin.set_language("en")
    plugin.set_language("zh")
    check("language: listeners told at once, in the new language", heard == ["en", "zh"], str(heard))
    check("language: kept in the settings", connection.read_settings().get("lang") == "zh")

    node = plugin.new_node()
    version_dir = os.path.join(results.node_folder(project, "lab2shot1"), "v001")
    os.makedirs(os.path.join(version_dir, "scene"))
    with open(os.path.join(version_dir, "scene", "scene.fbx"), "w") as f:
        f.write("fbx")
    with open(os.path.join(version_dir, results.MANIFEST), "w", encoding="utf-8") as f:
        json.dump({"node": "deliver", "outputs": [{"name": "scene", "type": "scene", "main": "scene/scene.fbx",
                                                   "files": ["scene/scene.fbx"], "colorspace": ""}]}, f)
    with open(os.path.join(version_dir, results.MAP_FILE), "w", encoding="utf-8") as f:
        json.dump({"scene/scene.fbx": os.path.join(version_dir, "scene", "scene.fbx")}, f)
    state = host.load(node)
    state["versions"] = [{"version": 1, "job": "j1", "folder": version_dir, "group": "", "namespace": "", "objects": [],
                          "imported": False, "made": "2026-10-03 09:00:00", "tool": "T"}]
    state["job"] = {"id": "j9", "state": "done"}
    host.store(node, state)
    run = plugin.import_version(node, 1)
    end = time.time() + 20
    while not run.finished and time.time() < end:
        host.pump(0.05)
    snap = run.snapshot()
    after = host.load(node)
    v = after["versions"]
    check("import later: done, no error", snap["done"] and not snap["error"], str(snap))
    check("import later: the same version, imported, its job and time kept, nothing added",
          len(v) == 1 and v[0]["imported"] and v[0]["job"] == "j1" and v[0]["made"] == "2026-10-03 09:00:00"
          and v[0]["objects"], str(v))
    check("import later: the node's last job untouched", after["job"] == {"id": "j9", "state": "done"}, str(after["job"]))
    check("import later: one undo step", len(host.undo_steps) == 1)
    try:
        plugin.import_version(node, 1)
        check("import later: an imported version is refused", False)
    except RuntimeError:
        check("import later: an imported version is refused", True)
    items = results.items({"outputs": [{"name": "p", "main": "p/p.####.exr", "files": ["p/p.0001.exr"],
                                        "colorspace": "ACEScg"}]}, {"p/p.0001.exr": os.path.join(version_dir, "scene",
                                                                                                 "scene.fbx")})
    check("delivery: each output's colour space", items[0]["colorspace"] == "ACEScg")

    # binding checks the file the host exports against what the input reads
    host.exports = {"scene.camera": ".usda"}
    cam = host.add_object("cam1", "scene.camera")
    host.selected = [cam]
    why = plugin.bind_selected(node, dict(CAM, accept=[".fbx"]))
    check("bind: Nuke's camera refused by an input reading .fbx", bool(why) and ".usda" in why, why)
    why = plugin.bind_selected(node, dict(CAM, param="cam_usd_path", accept=[".usd", ".usda"]))
    check("bind: and taken by the one reading USD", why == "", why)
    host.exports = {}

    # the scene never saved: nothing is computed or fetched; saved: the gate is open
    host.unsaved_scene = True
    said = []
    for name, call in (("compute", lambda: plugin.compute(node)), ("fetch", lambda: plugin.fetch(node))):
        try:
            call()
            said.append((name, ""))
        except RuntimeError as exc:
            said.append((name, str(exc)))
    check("unsaved: compute and fetch refused, saying to save first",
          all(why == paths.text("dcc.plugin.save_first") for _n, why in said) and plugin.why_not_saved(), str(said))
    host.unsaved_scene = False
    check("saved: nothing in the way", plugin.why_not_saved() == "")
    del jobs


def test_ranking_prefers():
    from lab2shot_dcc import catalog

    three = tool("3", "3", delivers=[D3])
    two = tool("2", "2", delivers=[D2])
    check("prefers: default 3D first", catalog.preferred(three) < catalog.preferred(two))
    check("prefers: pictures first", catalog.preferred(two, ("image", "scene")) < catalog.preferred(three, ("image", "scene")))
    check("prefers: nothing delivered goes last", catalog.preferred(tool("0", "0"), ("image",)) == 1)


if __name__ == "__main__":
    test_words()
    test_cards()
    test_home()
    test_tool_page()
    test_auto_bind()
    test_choose_node()
    test_connection_state()
    test_input_use()
    test_menu_items()
    test_exports()
    test_colorspace_for()
    test_tasks()
    test_english()
    test_plugin_additions()
    test_ranking_prefers()
    print("ALL OK" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)
