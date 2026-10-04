"""Reading the user's nodes, never changing them: only value() / getValueAt() / evaluate() / Class() / input(), never
setValue, setInput, setSelected, nuke.frame(f) or anything that writes a knob, the selection or the current frame.

Measured in Nuke 17.0v2 (`Nuke17.0.exe -t`, clients/nuke/tests/probe_n0.py):
- Read `frame_mode` offset with `frame` N: Nuke frame F shows the file's frame F + N (offset 1000 on a 1001-1005
  sequence: the Read's range is 1-5); "start at" S: F shows F - S + first; expression "" (the default): F shows F.
- Camera3 / Camera4 `world_matrix` getValueAt(frame, i) is the camera's world matrix at that frame with every parent
  (Axis) applied: 16 numbers, row-major, column-vector (translation in [3], [7], [11]).
"""

from __future__ import annotations

import os
import re

import nuke

VIDEO = (".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".mxf")
CAMERAS = ("Camera", "Camera2", "Camera3", "Camera4")
GEO_FILES = {"GeoImport": "file", "ReadGeo": "file", "ReadGeo2": "file", "GeoReference": "file_path"}
_HASHES = re.compile(r"#+")
_PRINTF = re.compile(r"%0?(\d*)d")


def file_of(node) -> str:
    """The node's file as Nuke resolves it (TCL expressions evaluated), forward slashes; "" when it has none."""
    knob = node.knob(GEO_FILES.get(node.Class(), "file"))
    if knob is None:
        return ""
    try:
        path = nuke.filename(node) if node.Class() in ("Read", "Write") else knob.evaluate()
    except Exception:  # noqa: BLE001 - an expression that does not evaluate: what is typed
        path = knob.value()
    return str(path or knob.value() or "").replace("\\", "/")


def frame_path(pattern: str, frame: int) -> str:
    """A sequence pattern (####, %04d) at one frame; a plain path as it is."""
    if _HASHES.search(pattern):
        return _HASHES.sub(lambda m: str(int(frame)).zfill(len(m.group(0))), pattern)
    if _PRINTF.search(pattern):
        return _PRINTF.sub(lambda m: str(int(frame)).zfill(int(m.group(1) or 0)), pattern)
    return pattern


def is_sequence(pattern: str) -> bool:
    return bool(_HASHES.search(pattern) or _PRINTF.search(pattern))


def read_offset(node) -> int | None:
    """The bound picture's frame offset (the picture's frame = Nuke's frame + it) of a Read; None when its frame
    expression is not one Lab2Shot can follow."""
    mode = node["frame_mode"].value() if node.knob("frame_mode") else "expression"
    typed = str(node["frame"].value() if node.knob("frame") else "").strip()
    if not typed or typed == "frame":
        return 0
    try:
        number = int(float(typed))
    except ValueError:
        number = None
    if mode == "offset" and number is not None:
        return number
    if mode == "start at" and number is not None:
        return int(node["first"].value()) - number
    m = re.fullmatch(r"frame\s*([+-])\s*(\d+)", typed)
    if mode == "expression" and m:
        return int(m.group(2)) * (1 if m.group(1) == "+" else -1)
    return None


def colorspace_of(node) -> str:
    """A Read's input colour space by name ("" when raw: data, no conversion); "default (sRGB)" is sRGB."""
    if node.knob("raw") is not None and node["raw"].value():
        return ""
    knob = node.knob("colorspace")
    if knob is None:
        return ""
    value = str(knob.value() or "")
    m = re.fullmatch(r"default \((.*)\)", value)
    return m.group(1) if m else value


def picture(node) -> dict:
    """What a Read (or a Write that rendered) shows: {file: its first frame as a file, pattern, sequence, first, last,
    frame_offset, colorspace}. frame_offset None: an expression Lab2Shot cannot follow."""
    pattern = file_of(node)
    first = int(node["first"].value()) if node.knob("first") else 1
    last = int(node["last"].value()) if node.knob("last") else first
    sequence = is_sequence(pattern)
    video = pattern.lower().endswith(VIDEO)
    offset = read_offset(node) if node.Class() == "Read" else 0
    return {"file": frame_path(pattern, first) if sequence else pattern, "pattern": pattern,
            "sequence": sequence or video, "first": first, "last": last, "frame_offset": offset,
            "colorspace": colorspace_of(node) if node.Class() == "Read" else ""}


def camera_samples(node, frames: list[int], offset: int = 0) -> list[dict]:
    """The camera's world matrix and lens at these Nuke frames, keyed at the picture's frames (frame + offset)."""
    out = []
    win = node.knob("win_translate")
    for f in frames:
        out.append({
            "frame": int(f) + int(offset),
            "matrix": [float(node["world_matrix"].getValueAt(f, i)) for i in range(16)],
            "focal": float(node["focal"].getValueAt(f)),
            "haperture": float(node["haperture"].getValueAt(f)),
            "vaperture": float(node["vaperture"].getValueAt(f)),
            "win": (float(win.getValueAt(f, 0)), float(win.getValueAt(f, 1))) if win is not None else (0.0, 0.0),
        })
    return out


def near_far(node) -> tuple[float, float]:
    try:
        return float(node["near"].value()), float(node["far"].value())
    except Exception:  # noqa: BLE001
        return 0.1, 100000.0


def geo_types(node) -> list[str]:
    """The data types of a 3D file node by its file (what the server can read of it)."""
    ext = os.path.splitext(file_of(node))[1].lower()
    if ext in (".usd", ".usda", ".usdc", ".usdz", ".abc"):
        return ["scene"]
    if ext in (".fbx",):
        return ["scene.model", "scene.character"]
    if ext in (".obj",):
        return ["scene.model"]
    if ext in (".ply",):
        return ["scene.points", "scene.gaussian"]
    if ext in (".splat", ".spz"):
        return ["scene.gaussian"]
    return []


def types_of(node) -> list[str]:
    """The data types a node of the user's gives as an input (what selection_types reports)."""
    kind = node.Class()
    if kind in ("Read", "Write"):
        path = file_of(node)
        if not path:
            return []
        if kind == "Write" and not os.path.exists(frame_path(path, int(node["first"].value()) if node.knob("first")
                                                             and node["use_limit"].value() else nuke.root().firstFrame())):
            return []
        return ["video" if path.lower().endswith(VIDEO) else "image"]
    if kind in CAMERAS:
        return ["scene.camera"]
    if kind in GEO_FILES:
        return geo_types(node)
    return []


def fingerprint(node) -> str:
    """How a user's node is found again after a rename: its class and its file (no knob of ours is ever put on it)."""
    return f"{node.Class()}|{file_of(node)}"
