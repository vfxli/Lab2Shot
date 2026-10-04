"""A camera's per-frame samples as a USD camera (.usda text), Python standard library only, no DCC.

Written in Lab2Shot's USD conventions (lab2shot/io/usd.py), so the server reads it as it reads its own: Y up,
metersPerUnit 0.01 (the DCC's numbers are taken as centimetres, as every camera Lab2Shot delivers to a DCC is), time
codes = the picture's own frame numbers, focal length and film back in millimetres (tenths of a centimetre-stage
unit), the transform one matrix per frame (column-vector camera-to-world turned into USD's row-vector layout here),
the lens centre as aperture offsets of the opposite sign (io/usd.py write_camera).

Nothing here is any DCC's: a host that can sample a camera's world matrix and lens (Nuke: Camera4 world_matrix) hands
the samples to `write` in its finish_export.
"""

from __future__ import annotations

import math
import re


def _num(v: float) -> str:
    v = float(v)
    if not math.isfinite(v):
        raise ValueError(f"not a finite number: {v}")
    text = repr(v)
    return "0" if text in ("-0.0", "0.0") else text


def _ident(name: str) -> str:
    word = re.sub(r"[^A-Za-z0-9_]", "_", str(name or "")).strip("_") or "camera"
    return word if word[0].isalpha() else "camera_" + word


def _samples(frames, values, fmt) -> str:
    """A value per frame as a plain value when it never changes, else time samples."""
    shown = [fmt(v) for v in values]
    if len(set(shown)) == 1:
        return " = " + shown[0]
    return ".timeSamples = {\n" + "".join(f"            {int(f)}: {s},\n" for f, s in zip(frames, shown)) + "        }"


def _matrix(m) -> str:
    """A 4x4 column-vector matrix (row-major list of 16) as USD's row-vector matrix4d."""
    rows = [[m[c * 4 + r] for c in range(4)] for r in range(4)]  # transposed
    return "(" + ", ".join("(" + ", ".join(_num(x) for x in row) + ")" for row in rows) + ")"


def write(path: str, name: str, samples: list[dict], fps: float, near_far=(0.1, 100000.0)) -> str:
    """`samples`: [{"frame": the picture's frame number, "matrix": 16 numbers, row-major, column-vector camera to
    world (translation in [3], [7], [11]), "focal", "haperture", "vaperture" (mm), "win": (x, y) Nuke-style lens centre
    off the picture's centre in half film backs, +x right +y up}]. Writes `path`; returns it."""
    if not samples:
        raise ValueError("no camera samples")
    frames = [int(s["frame"]) for s in samples]
    cam = _ident(name)
    h_off = [-float(s.get("win", (0, 0))[0]) * float(s["haperture"]) / 2.0 for s in samples]
    v_off = [-float(s.get("win", (0, 0))[1]) * float(s["vaperture"]) / 2.0 for s in samples]
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "shot"',
        "    metersPerUnit = 0.01",
        '    upAxis = "Y"',
        f"    startTimeCode = {min(frames)}",
        f"    endTimeCode = {max(frames)}",
        f"    timeCodesPerSecond = {_num(fps)}",
        f"    framesPerSecond = {_num(fps)}",
        ")",
        "",
        'def Xform "shot"',
        "{",
        f'    def Camera "{cam}"',
        "    {",
        '        token projection = "perspective"',
        f"        float2 clippingRange = ({_num(near_far[0])}, {_num(near_far[1])})",
        "        float focalLength" + _samples(frames, [s["focal"] for s in samples], _num),
        "        float horizontalAperture" + _samples(frames, [s["haperture"] for s in samples], _num),
        "        float verticalAperture" + _samples(frames, [s["vaperture"] for s in samples], _num),
    ]
    if any(abs(x) > 1e-12 for x in h_off + v_off):
        lines.append("        float horizontalApertureOffset" + _samples(frames, h_off, _num))
        lines.append("        float verticalApertureOffset" + _samples(frames, v_off, _num))
    lines += [
        "        matrix4d xformOp:transform" + _samples(frames, [s["matrix"] for s in samples], _matrix),
        '        uniform token[] xformOpOrder = ["xformOp:transform"]',
        "    }",
        "}",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    return path
