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

from typing import NamedTuple

import numpy as np
from lab2shot_shared.units import CV_TO_GL, DEFAULT_FPS, M_TO_CM  # noqa: F401  re-exported (DEFAULT_FPS); shared with workers

FILMBACK_MM = 36.0  # full frame: the film back of a lens nothing says more about
# In Lab2Shot the frame rate is not a property of data: DCCs have no frame rate when reading a sequence either; it is
# a project-level setting, and without a project concept here it is specified at output. No solver depends on it, and
# videos are read frame by frame. Reader nodes therefore have no frame rate and packets (Packet.meta) carry none; only
# frame numbers flow between nodes.
# This value (24 fps, film) has two uses: the default of the output-settings node's 「帧率」 parameter, and the time
# base of scene files (one time code per frame, io/usd.py apply_conventions). It is defined in lab2shot_shared.units,
# like M_TO_CM
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
    "帧": Unit("frames", 1.0),
    "秒": Unit("time", 1.0),
    "EV": Unit("exposure", 1.0),
}
UNIT_KINDS = {"length": "长度", "pixels": "像素", "angle": "角度", "frames": "帧数", "time": "时间", "exposure": "曝光"}


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


def rotations(mats: np.ndarray) -> np.ndarray:
    """The rotation part of a matrix or a batch of them (...,3,3, or the top-left 3x3 of ...,4,4), its columns made
    unit length (the one rotation-normalization: a camera-to-world matrix read back is not perfectly orthonormal)."""
    r = np.asarray(mats, np.float64)[..., :3, :3]
    return r / np.linalg.norm(r, axis=-2, keepdims=True)


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


def usd_points_to_opencv_m(xyz: np.ndarray) -> np.ndarray:
    """Points in the project's Y-up centimetre space -> OpenCV axes, metres: the inverse of `opencv_points_to_usd`.

    The counterpart of `usd_poses_to_opencv_m` (that one for cameras, this one for points). The axis-flip matrix is
    defined only in this module; adapters must not apply it themselves (back-projection and normal flipping each have a
    single implementation in the project).
    Used by ViPE: 「ViPE 相机解算」 outputs SLAM points as 「点云」, and 「ViPE 深度图」 sends them back upstream."""
    return np.asarray(xyz, np.float64) * CV_TO_GL / M_TO_CM


def usd_poses_to_opencv_m(cam_to_world: np.ndarray) -> np.ndarray:
    """The inverse direction: the project's Y-up, centimetre poses -> OpenCV camera axes, metres, as a worker reads them
    (lab2shot_worker.recon.load_camera), send_camera's conversion."""
    mats = np.asarray(cam_to_world, np.float64) @ np.diag([*CV_TO_GL, 1.0])
    mats[..., :3, 3] /= M_TO_CM
    return mats
