"""One home for the camera and unit constants every camera-reading path and unit table shares (a small number of
shared concepts, not one per node).

FILMBACK_MM, DEFAULT_WIDTH/HEIGHT are fallbacks, only for a *source* that truly has none of its own (a
USD without a film back, 「创建相机」 without a picture): everything downstream reads a packet's own meta.
DEFAULT_FPS is not a fallback at all; see its comment below. CV_TO_GL and M_TO_CM are
the one OpenCV <-> USD axis-and-scale conversion (a worker's world, metres, +Y down -> the project's, Y up, centimetres),
their numbers from lab2shot_shared.units, which the workers convert with too;
UNITS is the one units-of-a-value table (data/values.py, a file's length unit through to_cm): units of the same kind
convert to one another, of different kinds never.

The rule: 36.0 / 24.0 / 1920 / 1080 are written nowhere else in lab2shot/ or the adapters' node modules, nor is any
multiplication or division by 100 (metres to centimetres is M_TO_CM, a percentage is PERCENT):
those numbers have exactly one home, this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple

from ..i18n import Words

import numpy as np
from lab2shot_shared.units import CV_TO_GL, DEFAULT_FPS, M_TO_CM  # noqa: F401  re-exported (DEFAULT_FPS); shared with workers

FILMBACK_MM = 36.0  # full frame: the film back of a lens nothing says more about
# In Lab2Shot the frame rate is not a property of data: packets (Packet.meta) carry none, only frame numbers flow
# between nodes (an image sequence has no frame rate at all). Where a rate matters it is a parameter, 「帧率」, that a
# wire can drive: on output settings whose format stores time (nodes/output.py fps_param) and on the motion models,
# which work in seconds (nodes/families/rig_motion.py motion_fps_param). Readers whose file records one give it on a
# 「帧率」 output for templates to wire there (读取视频, 导入 BVH, 导入 USD).
# This value (24 fps, film) is the default of those parameters and the time base of scene files (one time code per
# frame, io/usd.py apply_conventions). It is defined in lab2shot_shared.units, like M_TO_CM
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1920, 1080  # only when a source records no picture size (a camera whose file has none, 创建相机 without one)
MASK_THRESHOLD = 0.5
CM_TO_M = 1.0 / M_TO_CM  # a stage written in metres (USD 输出设置 · 米)
PERCENT = 100.0  # a share as a percentage in a message (not a length)


class Unit(NamedTuple):
    """A value's unit: units of the same `kind` convert to one another (multiply by their `per_base`, USD's kind of
    unit, and divide by the other's); of different kinds they never do."""

    kind: str
    per_base: float  # how many of the kind's base unit (cm for length, px for pixels, ...) one of this unit is


# the one units-of-a-value table: parameter units (data/values.py) and the conversions formats read with it (formats/nuke/parse.py)
UNITS: dict[str, Unit] = {
    "mm": Unit("length", 0.1),
    "cm": Unit("length", 1.0),
    "m": Unit("length", M_TO_CM),
    "px": Unit("pixels", 1.0),
    "°": Unit("angle", 1.0),
    "frame": Unit("frames", 1.0),
    "s": Unit("time", 1.0),
    "EV": Unit("exposure", 1.0),
    "fps": Unit("rate", 1.0),  # a frame rate port or parameter (file_video / bvh.import / usd.import -> output settings, motion models)
}


# what each kind of unit is called (unit.kind.<kind>), and each unit as it shows (unit.<id>: 帧 / frames)
UNIT_KINDS: Mapping[str, str] = Words("unit.kind.", tuple(dict.fromkeys(u.kind for u in UNITS.values())))
UNIT_LABELS: Mapping[str, str] = Words("unit.", tuple(UNITS))


def unit_label(unit: str) -> str:
    """A unit as the artist reads it in the language now (unit.<id>, also for a unit no value converts, like the
    factor "x"; "" none; one with no words as it is)."""
    from .. import i18n

    return (i18n.lookup(f"unit.{unit}") or unit) if unit else ""


def to_cm(unit: str) -> float:
    """A length unit's factor to centimetres (a file's recorded unit: a 3D format's own scale)."""
    return UNITS[unit].per_base


def focal_px(focal_mm, filmback_mm, width) -> np.ndarray:
    """A lens in mm at a picture `width`: pixels = focal mm / film back mm x width."""
    return np.asarray(focal_mm, np.float64) / np.asarray(filmback_mm, np.float64) * width


def focal_mm(focal_px, filmback_mm, width) -> np.ndarray:
    """The inverse of focal_px: pixels -> mm at the same film back and width."""
    return np.asarray(focal_px, np.float64) * np.asarray(filmback_mm, np.float64) / width


def fov_x_deg(focal_px, width) -> float:
    """The horizontal field of view, degrees, of a pinhole with this focal length (pixels) at this picture width."""
    return float(np.degrees(2 * np.arctan(width / (2 * np.asarray(focal_px, np.float64)))))


def opencv_poses_to_usd(cam_to_world: np.ndarray, unit_cm: float = M_TO_CM) -> np.ndarray:
    """OpenCV camera-to-world poses in an OpenCV world (e.g. first camera at the origin) -> the project's Y-up world, GL
    cameras, centimetres."""
    flip = np.diag([*CV_TO_GL, 1.0])
    out = flip @ np.asarray(cam_to_world, np.float64) @ flip
    out[..., :3, 3] *= unit_cm
    return out


def opencv_points_to_usd(xyz: np.ndarray, unit_cm: float = M_TO_CM) -> np.ndarray:
    """OpenCV-world points (a method's own reconstruction) -> Y up, centimetres: the same axes as
    opencv_poses_to_usd, without a translation to carry."""
    return np.asarray(xyz, np.float64) * CV_TO_GL * unit_cm


def usd_points_to_m(xyz: np.ndarray) -> np.ndarray:
    """Points in the project's Y-up centimetre world -> the same world in metres: the world a worker reads cameras in
    (`usd_poses_to_opencv_m` turns only the camera's own axes to OpenCV, never the world's), so points sent with a
    camera land in front of it. Not the inverse of `opencv_points_to_usd` (that one comes from a method's own OpenCV
    world). Used by 「ViPE 深度图」, which sends 「ViPE 相机解算」's SLAM points back upstream with the camera."""
    return np.asarray(xyz, np.float64) / M_TO_CM


def usd_poses_to_opencv_m(cam_to_world: np.ndarray) -> np.ndarray:
    """The inverse direction: the project's Y-up, centimetre poses -> OpenCV camera axes, metres, as a worker reads them
    (lab2shot_worker.recon.load_camera), send_camera's conversion."""
    mats = np.asarray(cam_to_world, np.float64) @ np.diag([*CV_TO_GL, 1.0])
    mats[..., :3, 3] /= M_TO_CM
    return mats
