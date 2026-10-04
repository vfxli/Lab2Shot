"""N0: what Nuke 17 does, measured (设计_DCC插件_Nuke.md §8). Read-only towards anything but a fresh script in a
temporary folder. Run: `Nuke17.0.exe -t probe_n0.py <folder>` (with HOME pointing at a test folder, so Nuke's own
recent-files list is not written); prints one JSON line after "N0>>".

1. Units: GeoImport of the same geometry in a metersPerUnit 0.01 and a 1.0 file; a Camera4 reading a USD camera.
2. One space: a quad 300 cm in front of a camera given as the server's .nk (Camera3 block, cm) and read into a
   Camera4 with readKnobs, rendered by ScanlineRender2: where it lands against where the pinhole puts it.
3. STMap: which input is src and which the map.
4. Read frame modes: the file frame a Read shows at a Nuke frame (offset / start at).
5. GeoImport time_offset; Camera4 world_matrix under an animated Axis4.
6. nuke.nodes.* and the selection; Read.file fromUserText and the Root's frame range; the undo queue in nuke -t.
"""

from __future__ import annotations

import json
import math
import os
import sys

import nuke

DIR = sys.argv[-1].replace("\\", "/").rstrip("/") + "/"
os.makedirs(DIR + "seq", exist_ok=True)
OUT: dict = {}

SCENE = """#usda 1.0
(
    defaultPrim = "shot"
    metersPerUnit = %s
    upAxis = "Y"
)
def Xform "shot"
{
    def Mesh "tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(100, 0, 0), (100, 50, 0), (100, 0, 50)]
    }
    def Camera "cam"
    {
        float focalLength = 35
        float horizontalAperture = 36
        double3 xformOp:translate.timeSamples = { 1: (0, 0, 500), 2: (10, 0, 500) }
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
"""
CAMERA_NK = """Camera3 {
 inputs 0
 xform_order SRT
 rot_order ZXY
 translate {0 0 500}
 rotate {0 30 0}
 focal 35
 haperture 36
 vaperture 24
 name probe_cam
}
"""


def stage_of(node, frame=1):
    from pxr import Usd, UsdGeom

    oc = nuke.OutputContext()
    oc.setFrame(frame)
    st = node.getStage(oc)
    out = {"metersPerUnit": UsdGeom.GetStageMetersPerUnit(st)}
    cache = UsdGeom.XformCache(Usd.TimeCode(frame))
    for p in st.Traverse():
        if p.IsA(UsdGeom.PointBased):
            out[str(p.GetPath())] = [list(x) for x in UsdGeom.PointBased(p).GetPointsAttr().Get(Usd.TimeCode(frame))][:1]
        elif p.IsA(UsdGeom.Xformable):
            out[str(p.GetPath())] = list(cache.GetLocalToWorldTransform(p).ExtractTranslation())
    return out


def units():
    for mpu, name in (("0.01", "cm"), ("1", "m")):
        with open(DIR + name + ".usda", "w") as f:
            f.write(SCENE % mpu)
        OUT["geoimport_" + name] = stage_of(nuke.nodes.GeoImport(file=DIR + name + ".usda"))
    cam = nuke.nodes.Camera4()
    cam["file_source_enabled"].setValue(True)
    cam["file"].setValue(DIR + "cm.usda")
    cam["reload"].execute()
    OUT["camera4_reads_usd_cm"] = {"translate_f2": cam["translate"].getValueAt(2), "focal": cam["focal"].getValueAt(1),
                                   "world_to_meters": cam["world_to_meters"].value()}


def one_space():
    a = math.radians(30)
    rot = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    centre = [0, 0, 500]

    def world(x, y, z):
        return [centre[i] + rot[i][0] * x + rot[i][1] * y + rot[i][2] * z for i in range(3)]

    corners = [world(60 + dx, 30 + dy, -300) for dx, dy in ((-5, -5), (5, -5), (5, 5), (-5, 5))]
    with open(DIR + "quad.usda", "w") as f:
        f.write('#usda 1.0\n(\n    metersPerUnit = 0.01\n    upAxis = "Y"\n)\ndef Mesh "quad"\n{\n'
                "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                "    point3f[] points = [%s]\n}\n" % ", ".join("(%f, %f, %f)" % tuple(c) for c in corners))
    nuke.addFormat("360 240 1 l2s_probe")
    bg = nuke.nodes.Constant(format="l2s_probe")
    bg["color"].setValue([0, 0, 0, 0])
    body = CAMERA_NK[CAMERA_NK.index("{") + 1: CAMERA_NK.rindex("}")]
    cam = nuke.nodes.Camera4()
    cam.readKnobs("\n".join(line for line in body.splitlines() if line.split()[:1] not in (["inputs"], ["name"])))
    geo = nuke.nodes.GeoImport(file=DIR + "quad.usda")
    scene = nuke.nodes.GeoScene()
    scene.setInput(0, cam)
    scene.setInput(1, geo)
    render = nuke.nodes.ScanlineRender2()
    render.setInput(0, bg)
    render.setInput(1, scene)
    render.setInput(2, cam)
    nuke.frame(1)
    hits = [(x, y) for y in range(0, 240, 2) for x in range(0, 360, 2) if nuke.sample(render, "rgba.alpha", x + 0.5, y + 0.5) > 0.5]
    # pinhole: x = 180 + 180 * f * X / (-Z) / (haperture / 2), y = 120 + 120 * f * Y / (-Z) / (vaperture / 2)
    OUT["one_space_render"] = {"predicted_px": [180 + 180 * 35 * 60 / 300 / 18, 120 + 120 * 35 * 30 / 300 / 12],
                               "measured_centroid_px": [sum(h[0] for h in hits) / max(1, len(hits)) + 1,
                                                        sum(h[1] for h in hits) / max(1, len(hits)) + 1],
                               "hits": len(hits)}


def stmap():
    red = nuke.nodes.Constant(format="l2s_probe")
    red["color"].setValue([1, 0, 0, 1])
    uv = nuke.nodes.Constant(format="l2s_probe")
    uv["color"].setValue([0.25, 0.75, 0, 1])
    st = nuke.nodes.STMap()
    st["uv"].setValue("rgb")
    st.setInput(0, red)
    st.setInput(1, uv)
    got = [nuke.sample(st, c, 100.5, 100.5) for c in ("rgba.red", "rgba.green")]
    OUT["stmap"] = {"input0_red_input1_uv_gives": got, "input0_is": "src" if got == [1.0, 0.0] else "stmap"}


def read_modes():
    c = nuke.nodes.Constant()
    c["color"].setExpression("frame/10000", 0)
    w = nuke.nodes.Write(file=DIR + "seq/f.####.exr", file_type="exr")
    w.setInput(0, c)
    nuke.execute(w, 1001, 1005)
    for mode, value in (("offset", "1000"), ("offset", "-1000"), ("start at", "1")):
        rd = nuke.nodes.Read(file=DIR + "seq/f.####.exr", first=1001, last=1005)
        rd["frame_mode"].setValue(mode)
        rd["frame"].setValue(value)
        OUT[f"read_{mode}_{value}"] = {"range": [rd.firstFrame(), rd.lastFrame()],
                                       "file_frame_at": {F: rd.metadata("input/frame", F) for F in (1, 3, 2001, 2003)}}


def time_and_parents():
    g = nuke.nodes.GeoImport(file=DIR + "cm.usda")
    g["time_offset"].setValue(10)
    OUT["geoimport_time_offset_10"] = {F: stage_of(g, F).get("/shot/cam") for F in (2, 12)}
    ax = nuke.nodes.Axis4()
    ax["translate"].setAnimated()
    ax["translate"].setValueAt(10, 1, 0)
    ax["translate"].setValueAt(20, 2, 0)
    ax["rotate"].setValue([0, 90, 0])
    cam = nuke.nodes.Camera4()
    cam.setInput(0, ax)
    cam["translate"].setValue([1, 2, 3])
    OUT["camera4_world_matrix_under_axis"] = {F: [round(cam["world_matrix"].getValueAt(F, i), 4) for i in (3, 7, 11)] for F in (1, 2)}


def side_effects():
    for n in nuke.allNodes():
        n.setSelected(False)
    first = nuke.allNodes()[0]
    first.setSelected(True)
    made = nuke.nodes.NoOp()
    OUT["nuke_nodes_keeps_selection"] = [n.name() for n in nuke.selectedNodes()] == [first.name()] and not made["selected"].value()
    root = nuke.root()
    root["first_frame"].setValue(1)
    root["last_frame"].setValue(5)
    a = nuke.nodes.Read()
    a["file"].setValue(DIR + "seq/f.####.exr")
    a["first"].setValue(1001)
    a["last"].setValue(1005)
    after_set = [root.firstFrame(), root.lastFrame()]
    b = nuke.nodes.Read()
    b["file"].fromUserText(DIR + "seq/f.####.exr 1001-1005")
    OUT["root_range"] = {"before": [1, 5], "after_file_setValue": after_set, "after_fromUserText": [root.firstFrame(), root.lastFrame()]}
    nuke.Undo.begin("probe")
    nuke.nodes.NoOp()
    nuke.Undo.end()
    OUT["undo_queue_in_nuke_t"] = nuke.Undo.undoSize()


for step in (units, one_space, stmap, read_modes, time_and_parents, side_effects):
    try:
        step()
    except Exception as exc:  # noqa: BLE001 - one probe failing says so, the others still run
        OUT[step.__name__ + "_error"] = repr(exc)
OUT["nuke"] = nuke.NUKE_VERSION_STRING
print("N0>>" + json.dumps(OUT, default=str))
