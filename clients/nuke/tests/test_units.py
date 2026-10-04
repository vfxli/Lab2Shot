"""The Nuke plugin's parts that need no Nuke: the delivered .nk blocks (nk.py), the EXR header (exr.py) and the colour
space table (colorspaces.json through lab2shot_dcc.results.colorspace_for). Run: python clients/nuke/tests/test_units.py"""

from __future__ import annotations

import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "lab2shot"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "common"))  # lab2shot_dcc

from lab2shot_nuke import exr, nk  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f": {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


CAMERA_NK = """set cut_paste_input [stack 0]
push $cut_paste_input
Camera3 {
 inputs 0
 xform_order SRT
 rot_order ZXY
 translate {{curve x1001 0 10 x1005 20} 0 500}
 rotate {0 30 0}
 focal 35
 name probe_cam
 label "Lab2Shot: camera, {braces} \\"quoted\\" 360x240"
}
Tracker4 {
 inputs 0
 tracks { { 1 31 2 }
{ { 5 1 20 enable e 1 } }
}
 name t1
 selected true
 xpos 10
}
"""


def test_blocks():
    got = nk.blocks(CAMERA_NK)
    check("two blocks", [b["delivered"] for b in got] == ["Camera3", "Tracker4"], [b["delivered"] for b in got])
    check("classic camera made as Camera4", got[0]["class"] == "Camera4")
    check("place knobs left out", not any(line.split()[0] in nk.PLACE_KNOBS for b in got for line in b["knobs"].splitlines()
                                          if line.strip() and not line.startswith(("{", " "))), got[1]["knobs"])
    check("label kept whole (braces, quotes)", 'label "Lab2Shot: camera, {braces} \\"quoted\\" 360x240"' in got[0]["knobs"],
          got[0]["knobs"])
    check("multi-line knob kept", "tracks { { 1 31 2 }\n{ { 5 1 20 enable e 1 } }\n}" in got[1]["knobs"], got[1]["knobs"])
    check("names read", [b["name"] for b in got] == ["probe_cam", "t1"])


def test_shift():
    knobs = nk.blocks(CAMERA_NK)[0]["knobs"]
    moved = nk.shift_frames(knobs, -1000)
    check("curve keys moved", "{curve x1 0 10 x5 20}" in moved, moved)
    check("nothing else moved", moved.replace("{curve x1 0 10 x5 20}", "") == knobs.replace("{curve x1001 0 10 x1005 20}", ""))
    check("no shift: same text", nk.shift_frames(knobs, 0) == knobs)


NUKE_DEFAULT = ["linear", "sRGB", "sRGBf", "rec709", "Cineon", "Gamma1.8", "Gamma2.2", "Gamma2.4", "raw"]
FN_CG = ["ACES2065-1", "ACEScc", "ACEScct", "ACEScg", "sRGB Encoded Rec.709 (sRGB)", "Linear Rec.709 (sRGB)", "Raw"]
ACES_1 = ["ACES - ACEScg", "ACES - ACES2065-1", "Utility - sRGB - Texture", "Utility - Linear - sRGB", "Utility - Raw"]


def test_colorspaces():
    import json

    from lab2shot_dcc.results import colorspace_for

    table = {k: v for k, v in json.load(open(os.path.join(os.path.dirname(HERE), "lab2shot", "lab2shot_nuke",
                                                          "colorspaces.json"), encoding="utf-8")).items()
             if not k.startswith("_")}
    check("ACEScg in Nuke's cg config: itself", colorspace_for("ACEScg", FN_CG, table) == "ACEScg")
    check("ACEScg in an ACES 1.x config", colorspace_for("ACEScg", ACES_1, table) == "ACES - ACEScg")
    check("ACEScg in nuke-default: not there, not set", colorspace_for("ACEScg", NUKE_DEFAULT, table) == "")
    check("sRGB texture in nuke-default: sRGB", colorspace_for("sRGB Encoded Rec.709 (sRGB)", NUKE_DEFAULT, table) == "sRGB")
    check("linear Rec.709 in nuke-default: linear", colorspace_for("Linear Rec.709 (sRGB)", NUKE_DEFAULT, table) == "linear")
    check("sRGB texture in an ACES 1.x config", colorspace_for("sRGB Encoded Rec.709 (sRGB)", ACES_1, table)
          == "Utility - sRGB - Texture")


def test_exr_header():
    import struct

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "h.exr")
        chans = b"".join(n + b"\0" + struct.pack("<iBBBBii", 1, 0, 0, 0, 0, 1, 1) for n in (b"B", b"G", b"R",
                                                                                         b"stmap_undistort.R"))
        chans += b"\0"
        meta = b'{"rgba": {"colorspace": "ACEScg"}, "stmap_undistort": {"direction": "undistort"}}'
        head = struct.pack("<ii", 20000630, 2)
        head += b"channels\0chlist\0" + struct.pack("<i", len(chans)) + chans
        head += b"lab2shot:layers\0string\0" + struct.pack("<i", len(meta)) + meta + b"\0"
        with open(path, "wb") as f:
            f.write(head)
        got = exr.header(path)
        check("exr: channels", got["channels"] == ["B", "G", "R", "stmap_undistort.R"], str(got))
        check("exr: layers by name", exr.layer_names(got["channels"]) == ["rgba", "stmap_undistort"])
        check("exr: lab2shot:layers", got["layers"]["stmap_undistort"]["direction"] == "undistort")


if __name__ == "__main__":
    test_blocks()
    test_shift()
    test_colorspaces()
    test_exr_header()
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all passed")
    sys.exit(1 if FAILED else 0)
