"""The textual form of a Nuke script (.nk): node block layout, per-frame knob values, and valid node names.

A .nk file is Tcl: `NodeClass { knob value ... }`, one block per node in creation order. A pasted snippet begins by
taking the current selection in the comp as its input (`set cut_paste_input [stack 0]` / `push
$cut_paste_input`), which is what Nuke writes when nodes are copied. A knob that varies over time holds a curve:
`{curve x1001 1 2 3}`, i.e. the frame of the first key followed by one value per subsequent frame (a key on every
frame, with `x<frame>` repeated wherever the sequence skips frames). Frame numbers are written as the plate's own and
are never renumbered.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from ...errors import Invalid
from ...messages import Msg

HEAD = ("set cut_paste_input [stack 0]", "push $cut_paste_input")  # a pasted snippet takes the selected node as input
MARK = "Lab2Shot"  # written into every block's label so the result is distinguishable from hand-tracked work


def node_name(name: str, fallback: str) -> str:
    """`name` as a Nuke node name: letters, digits and underscores, starting with a letter (Nuke silently renames any
    other name on paste). `fallback` is used when nothing remains after cleaning."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
    if not cleaned:
        return fallback
    return cleaned if cleaned[0].isalpha() else f"{fallback}_{cleaned}"


def number(value: float, digits: int = 4) -> str:
    """A number in Nuke's notation: no exponent, no trailing zeros (0.5, 1, -12.25). Lens coefficients require more
    digits than positions (`digits`). Non-finite values are rejected (E-NUKE-VALUE): Nuke cannot read nan or inf and
    would silently leave the knob at its default."""
    v = float(value)
    if not math.isfinite(v):
        raise Invalid(Msg("E-NUKE-VALUE", knob="", value=str(value)[:20]))
    text = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-", "-0") else "0"


def curve(frames: Sequence[int], values: Sequence[float], digits: int = 4) -> str:
    """One knob's per-frame values as an animation curve. Constant values are written as a plain number, so a static
    camera has no keys, consistent with a hand-built one."""
    kept = [number(v, digits) for v in values]
    if not kept:
        return "0"
    if len(set(kept)) == 1:
        return kept[0]
    keys, last = [], None
    for f, v in zip(frames, kept):
        keys.append(v if last is not None and f == last + 1 else f"x{f} {v}")
        last = f
    return "{curve " + " ".join(keys) + "}"


def channel(frames: Sequence[int], values: Sequence[float], digits: int = 4) -> str:
    """One channel's value in knob form: a plain number when constant, otherwise its curve wrapped in the braces a knob
    value requires, e.g. `focal {{curve x1001 35 36}}`. A multi-channel knob instead places each channel's curve()
    inside a single pair of braces (see channels())."""
    got = curve(frames, values, digits)
    return "{" + got + "}" if got.startswith("{") else got


def channels(frames: Sequence[int], columns: Sequence[Sequence[float]], digits: int = 4) -> str:
    """A multi-channel knob (translate, rotate, win_translate): each channel's number or curve within one pair of
    braces."""
    return "{%s}" % " ".join(curve(frames, column, digits) for column in columns)


def braced(value: str) -> str:
    """A text value that may contain spaces (a file path, an enumeration label), wrapped in braces so Tcl reads it as
    one value and parse.py can split it back."""
    return "{" + str(value).replace("{", "").replace("}", "") + "}"


def label(text: str) -> str:
    """A node's label stating that the result comes from Lab2Shot and by which method, so it is not mistaken for
    hand-tracked work (a machine-learning result must be marked where it is visible, not only in metadata)."""
    return quoted(f"{MARK}: {text}" if not text.startswith(MARK) else text)


def quoted(text: str) -> str:
    """A string knob's value: Tcl quoting (no line breaks, backslashes and quotes escaped)."""
    cleaned = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{cleaned}"'


def block(kind: str, knobs: Sequence[tuple[str, str]]) -> str:
    """One node block, `Camera3 { ... }`, with one knob per line."""
    return "\n".join([kind + " {"] + [f" {name} {value}" for name, value in knobs] + ["}"])


def script(blocks: Sequence[str]) -> str:
    """The complete snippet: the paste header followed by the node blocks."""
    return "\n".join([*HEAD, *blocks]) + "\n"


# ---- The complete script delivered with a camera solve: the plate, the two branches a compositor needs, and the
# solved camera, wired and ready to open.
#
#     Read (原始画面)
#       ├─ 去畸变 ──────────────────────────────→ the undistorted canvas, what the solve was done on
#       └─────────────────────────┐
#     Read (CG)  ─→ 加畸变 ─→ Merge over ────────→ the CG back on the original plate
#
# Both branches apply the camera's distortion in the form Nuke supports (distortion.py): a 3DE model as its LD_3DE4
# node, any other model as a Read of an ST-map plus an STMap. The Root's `format` is the canvas rather than the plate
# frame, because the undistorted branch and the render are canvas-sized; the plate frame's format is written alongside
# so a compositor can reformat back to it.

PLATE = "plate"  # fixed node names, shared by the delivered script and this module's reader
UNDISTORT, REDISTORT, CG, OVER, CAMERA = "undistort", "redistort", "cg", "over", "camera"


def _format(size: tuple[int, int], pixel_aspect: float, name: str) -> str:
    """A Nuke format: width height x y r t pixel-aspect name."""
    width, height = int(size[0]), int(size[1])
    return braced(f"{width} {height} 0 0 {width} {height} {pixel_aspect:g} {name}")


def write_script(*, plate: str, window, first: int, last: int, fps: float | None = None,
                 camera=None, lens=None, undistort_stmap: str = "", distort_stmap: str = "",
                 focus_cm: float | None = None, pixel_aspect: float = 1.0, cg: str = "") -> str:
    """The text of a whole .nk script.

    plate            the original footage, as Nuke reads a sequence ("/shots/sh010/plate.####.exr")
    window           the plate frame and the canvas formed by its overscan (data/windows.py); sets the Root's format
    first, last      the frame range
    fps              frames per second
    camera           the solved camera, written as Camera3 (None: no camera in the script)
    lens             the distortion both branches apply (None: defined by the ST-maps alone)
    undistort_stmap  the undistort ST-map file; required unless Nuke has a node for the `lens` model
    distort_stmap    the distort ST-map file; same requirement
    focus_cm         3DE's custom focus distance, when the distortion was solved per focus
    pixel_aspect     the plate's pixel aspect (an anamorphic squeeze), written into both formats
    cg               the render to composite over the plate ("": an empty Read for the compositor to set)
    """
    from ...data import units
    from . import distortion
    from .camera import write_camera

    fps = units.DEFAULT_FPS if fps is None else fps
    frames = list(range(int(first), int(last) + 1))
    out = [block("Root", [
        ("inputs", "0"),
        ("format", _format(window.canvas, pixel_aspect, "lab2shot_canvas")),
        ("first_frame", str(int(first))),
        ("last_frame", str(int(last))),
        ("fps", number(fps)),
    ])]
    out.append(block("Read", [
        ("inputs", "0"), ("file", braced(plate)), ("first", str(int(first))), ("last", str(int(last))),
        ("format", _format(window.plate, pixel_aspect, "lab2shot_plate")),
        ("name", PLATE), ("label", braced("原始画面")),
    ]))
    out.append(f"set N{PLATE} [stack 0]")
    out.append(distortion.write_distortion(lens, "undistort", UNDISTORT, stmap=undistort_stmap, window=window,
                                           focus_cm=focus_cm, frames=frames, first=first, last=last))
    out.append(f"set N{UNDISTORT} [stack 0]")
    out.append(block("Read", [
        ("inputs", "0"), ("file", braced(cg)), ("first", str(int(first))), ("last", str(int(last))),
        ("format", _format(window.canvas, pixel_aspect, "lab2shot_canvas")),
        ("name", CG), ("label", braced("接 CG 渲染，按画布尺寸")),
    ]))
    out.append(distortion.write_distortion(lens, "distort", REDISTORT, stmap=distort_stmap, window=window,
                                           focus_cm=focus_cm, frames=frames, first=first, last=last))
    out.append(f"set N{REDISTORT} [stack 0]")
    out.append(f"push $N{PLATE}\npush $N{REDISTORT}")
    out.append(block("Merge2", [
        ("inputs", "2"), ("operation", "over"), ("name", OVER), ("label", braced("贴回原始画面")),
    ]))
    if camera is not None:
        out.append("push 0")
        out.append(write_camera(camera, CAMERA))
    return "\n".join(out) + "\n"
