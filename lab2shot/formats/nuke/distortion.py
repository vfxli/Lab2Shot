"""Writing a lens distortion into a Nuke script: an LD_3DE4 node when the model is a 3DE model, otherwise a Read of
an ST-map followed by an STMap node.

A 3DE model is written as the node 3DE's own exporter produces (the Nuke node of its Lens Distortion Plugin Kit). The
class is LD_ plus the model name with each run of punctuation replaced by one underscore; the seven built-in knobs are
in centimetres (the lens description uses millimetres, so values are divided by ten); `direction` selects undistort or
distort; a parameter that varies over the shot is written as a curve; and the canvas is written as the four
`field_of_view_*_unit` knobs (the plate frame's 0..1 range in canvas coordinates), so the node maps the lens across the
whole canvas rather than only the plate.

Other models (OpenCV, or a lens defined only by ST-maps) have no Nuke node. The same two maps are delivered as ST-map
files, loaded by a Read and applied by an STMap, which requires no plug-in.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from ...data.lens_models import MODELS, Lens
from ...data.windows import Window
from ...errors import Invalid
from ...messages import Msg
from . import script

# Model ids that have a 3DE Nuke node, mapped to the node class (as named by 3DE's export script).
NODE_CLASSES = {
    "3de4_classic": "LD_3DE_Classic_LD_Model",
    "3de4_radial_std_deg4": "LD_3DE4_Radial_Standard_Degree_4",
    "3de4_anamorphic_std_deg4": "LD_3DE4_Anamorphic_Standard_Degree_4",
}
DIRECTIONS = ("undistort", "distort")
# LD_3DE nodes record the lens's own values in centimetres (LDPK tde4_ld_plugin.h).
CM_PER_MM = 0.1
DIGITS = 12  # lens coefficients are small (1e-5 and below); enough digits for a lossless round trip
BBOX_MODE = "merge bbox plus margin with undistorted format"  # keeps the pixels the lens pushes outside the frame


def knob_of(param: str) -> str:
    """A model parameter's knob name: its letters and digits joined by single underscores
    ("Cx02 - Degree 2" -> Cx02_Degree_2), matching 3DE's naming."""
    return "_".join(re.findall(r"[A-Za-z0-9]+", str(param)))


def has_node(lens: Lens | None) -> bool:
    """Whether Nuke has a node for this lens's model (3DE's LD_3DE plug-in); otherwise ST-maps are required."""
    return lens is not None and lens.model in NODE_CLASSES


def series(value, frames: Sequence[int]) -> tuple[list[int], list[float]]:
    """A lens value (a number, or {"frames", "values"} for a shot) as (frames, values); a constant is repeated over
    `frames` (a single value when `frames` is empty)."""
    if isinstance(value, Mapping):
        return [int(f) for f in value["frames"]], [float(v) for v in value["values"]]
    return list(frames), [float(value)] * max(1, len(frames))


def _knob(name: str, value, frames: Sequence[int], scale: float = 1.0) -> tuple[str, str]:
    at, values = series(value, frames)
    return name, script.channel(at, [v * scale for v in values], DIGITS)


def field_of_view(window: Window | None) -> list[tuple[str, str]]:
    """The four field_of_view knobs: the position of the plate frame's 0..1 range within the canvas. Without a canvas,
    the plate frame itself."""
    if window is None or not window.has_overscan:
        box = (0.0, 1.0, 0.0, 1.0)
    else:
        left, top, right, bottom = window.overscan
        width, height = window.plate
        box = (-left / width, (width + right) / width, -bottom / height, (height + top) / height)
    return [(f"field_of_view_{axis}_unit", script.number(v, DIGITS))
            for axis, v in zip(("xa", "xb", "ya", "yb"), box, strict=True)]


def lens_node(lens: Lens, direction: str = "undistort", name: str = "lens", *, window: Window | None = None,
              focus_cm: float | None = None, frames: Sequence[int] = (), note: str = "") -> str:
    """The text of one LD_3DE4 node for a 3DE model. `frames` are the frames over which a constant parameter is
    written; a parameter that already carries its own frames keeps them."""
    if direction not in DIRECTIONS:
        raise ValueError(f"a distortion goes one of {DIRECTIONS}, not {direction!r}")
    if not has_node(lens):
        raise Invalid(Msg("E-NUKE-NOSTMAP", model=lens.label, direction=direction))
    width_mm, height_mm = lens.back
    rows: list[tuple[str, str]] = [("inputs", "1"), ("direction", direction), ("bbox_mode", script.braced(BBOX_MODE))]
    if lens.focal_mm is not None:
        rows.append(_knob("tde4_focal_length_cm", lens.focal_mm, frames, CM_PER_MM))
    rows += [
        ("tde4_filmback_width_cm", script.number(width_mm * CM_PER_MM, DIGITS)),
        ("tde4_filmback_height_cm", script.number(height_mm * CM_PER_MM, DIGITS)),
        ("tde4_lens_center_offset_x_cm", script.number(lens.center_mm[0] * CM_PER_MM, DIGITS)),
        ("tde4_lens_center_offset_y_cm", script.number(lens.center_mm[1] * CM_PER_MM, DIGITS)),
        ("tde4_pixel_aspect", script.number(lens.pixel_aspect, DIGITS)),
    ]
    if focus_cm is not None:
        rows.append(("tde4_custom_focus_distance_cm", script.number(focus_cm, DIGITS)))
    rows += field_of_view(window)
    rows += [_knob(knob_of(spec.name), lens.params[spec.name], frames) for spec in MODELS[lens.model].params]
    rows.append(("name", script.node_name(name, "Lab2Shot_lens")))
    if note:  # 镜头来源写在节点标签上：估算结果须在可见处标明，不能只记录在元数据中
        rows.append(("label", script.label(note)))
    return script.block(NODE_CLASSES[lens.model], rows)


def stmap_nodes(file: str, direction: str, name: str = "stmap", *, first: int = 0, last: int = 0) -> str:
    """A Read of an ST-map and the STMap that applies it. The image to be mapped must already be on the script's stack
    (the Read is placed above it and STMap takes both inputs). The direction is encoded in the ST-map's file name
    (data/layers.py stmap_direction_of_name), which this module's reader relies on."""
    if direction not in DIRECTIONS:
        raise ValueError(f"a distortion goes one of {DIRECTIONS}, not {direction!r}")
    read = script.block("Read", [
        ("inputs", "0"),
        ("file", script.braced(file)),
        *([("first", str(int(first))), ("last", str(int(last)))] if last else []),
        ("colorspace", "linear"),
        ("raw", "true"),
        ("name", script.node_name(f"{name}_map", "Lab2Shot_stmap")),
    ])
    apply = script.block("STMap", [
        ("inputs", "2"),
        ("channels", "rgba"),
        ("uv", "rgb"),
        ("filter", "Cubic"),
        ("name", script.node_name(name, "Lab2Shot_lens")),
    ])
    return read + "\n" + apply


def write_distortion(lens: Lens | None, direction: str, name: str = "lens", *, stmap: str = "",
                     window: Window | None = None, focus_cm: float | None = None,
                     frames: Sequence[int] = (), first: int = 0, last: int = 0) -> str:
    """The distortion in the requested direction, in the form Nuke supports: a 3DE model becomes its LD_3DE4 node,
    any other model a Read of `stmap` and an STMap. If neither applies, E-NUKE-NOSTMAP is raised."""
    if lens is not None and has_node(lens):
        return lens_node(lens, direction, name, window=window, focus_cm=focus_cm, frames=frames)
    if not stmap:
        raise Invalid(Msg("E-NUKE-NOSTMAP", model=lens.label if lens is not None else "ST-map", direction=direction))
    return stmap_nodes(stmap, direction, name, first=first, last=last)


def write_lens(lens: Lens, name: str = "lens", *, note: str = "", window: Window | None = None,
               focus_cm: float | None = None, frames: Sequence[int] = ()) -> str:
    """One lens as a snippet to paste (「复制到 Nuke」 on a node that solves or holds a lens).

    What Nuke can represent depends on the model; 3DE and OpenCV models are not converted into one another:

    - one of the three 3DE models: its LD_3DE4 node including coefficients, the same text 3DE's exporter writes, so
      it can be pasted into a comp and back into 「LensDistortion」;
    - any other model (OpenCV, 「无畸变」): Nuke has no node for it, so the snippet is a Camera3 carrying the
      transferable intrinsics (focal length, film back and principal point). The coefficients are omitted and the
      writing node reports this (N-NUKE-LENSASCAMERA); the distortion itself is delivered as the ST-maps baked by
      「LensDistortion」.
    """
    from .camera import lens_camera

    said = note or lens_note(lens)
    if has_node(lens):
        return script.script([lens_node(lens, "undistort", name, window=window, focus_cm=focus_cm, frames=frames, note=said)])
    if lens.focal_mm is None:
        raise Invalid(Msg("E-NUKE-NOLENSFOCAL", model=lens.label))
    return script.script([lens_camera(lens.focal_mm, lens.back, lens.center_mm, name, said, frames)])


def lens_note(lens: Lens) -> str:
    """The label text for the pasted node: the lens model and its source (data/lens_models.py Lens.source), so an
    estimate is not mistaken for a lens delivered by the tracking department."""
    by = str((lens.source or {}).get("by") or "")
    return f"{lens.label}, {by}" if by else lens.label


def as_camera(lens: Lens) -> bool:
    """Whether this lens is exported as a Camera3 (intrinsics only) because Nuke has no node for its model."""
    return not has_node(lens)
