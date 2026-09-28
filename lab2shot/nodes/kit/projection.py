"""Projecting positions onto UV: the maths of 「规范坐标转 UV」, pure functions in one place, so any node that has
positions can project them the same way.

A position map holds each pixel's 3D position in its own space (data/types.py map.position). Where that space is a
canonical one — the same model space in every frame, as face and body models give it — the position of a pixel names
the point of the model it shows, and a fixed projection of it is a UV coordinate: the same point of the face always
lands on the same UV, in every frame and in every shot. That is what makes it usable as a texture coordinate.

Three projections, all around the Y axis (Lab2Shot is Y up), all giving u and v in 0-1:

    cylinder  u from the angle around Y (u = 0.5 straight ahead, +Z), v from the height between the box's floor and
              ceiling: the usual projection for a face or a body, no pinching at the top
    sphere    u the same angle, v from the elevation seen from the box's centre: for a closed, round model
    plane     straight onto the XY plane (a front view): u from X, v from Y, both across the box

The box is the bounding box of the positions themselves (`Box`), taken over every frame of the sequence at once, so
the UV of a point does not move when the model does. u wraps at the back of the model (the seam), as every cylindrical
and spherical projection does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WAYS = ("cylinder", "sphere", "plane")  # 投射方式, in the order the parameter lists them
TINY = 1e-9  # a box side (or a radius) no larger than this holds no shape: its coordinate comes out 0.5


@dataclass(frozen=True)
class Box:
    """The bounding box of the positions a projection is made over: the lowest and highest X, Y, Z (centimetres, or
    whatever unit the canonical space is in — a UV only ever uses ratios of it)."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]

    @property
    def centre(self) -> np.ndarray:
        return (np.asarray(self.low, np.float64) + np.asarray(self.high, np.float64)) / 2.0

    @property
    def size(self) -> np.ndarray:
        return np.asarray(self.high, np.float64) - np.asarray(self.low, np.float64)

    def json(self) -> dict:
        return {"low": [float(v) for v in self.low], "high": [float(v) for v in self.high]}


def grown(box: Box | None, values: np.ndarray, valid: np.ndarray | None = None) -> Box | None:
    """`box` grown to hold the valid positions of one frame (values [...,3], `valid` [...] > 0 where there is a
    position). None and no valid position: still None — nothing has been seen yet."""
    points = np.asarray(values, np.float64).reshape(-1, 3)
    if valid is not None:
        points = points[np.asarray(valid).reshape(-1) > 0]
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return box
    low, high = points.min(axis=0), points.max(axis=0)
    if box is not None:
        low, high = np.minimum(low, box.low), np.maximum(high, box.high)
    return Box(tuple(float(v) for v in low), tuple(float(v) for v in high))


def _across(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Where a coordinate lies across a side of the box, 0-1 (a side of no length: 0.5, the middle)."""
    span = high - low
    if span <= TINY:
        return np.full(values.shape, 0.5)
    return np.clip((values - low) / span, 0.0, 1.0)


def _around_y(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """The angle around the Y axis as u, 0-1: 0.5 straight ahead (+Z), 0 and 1 at the back (-Z, the seam)."""
    return np.mod(0.5 + np.arctan2(x, z) / (2.0 * np.pi), 1.0)


def project(values: np.ndarray, way: str, box: Box) -> np.ndarray:
    """Positions [...,3] -> UV [...,2] in 0-1, projected `way` (WAYS) over `box`. Pure: the same position always gives
    the same UV. Positions that are not finite give (0, 0); the caller says through the validity channel which pixels
    hold a UV at all."""
    if way not in WAYS:
        raise ValueError(f"unknown projection {way!r}")
    p = np.asarray(values, np.float64)
    finite = np.isfinite(p).all(axis=-1)
    p = np.where(finite[..., None], p, 0.0)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    centre = box.centre
    if way == "plane":
        u, v = _across(x, box.low[0], box.high[0]), _across(y, box.low[1], box.high[1])
    elif way == "cylinder":
        u, v = _around_y(x - centre[0], z - centre[2]), _across(y, box.low[1], box.high[1])
    else:
        dx, dy, dz = x - centre[0], y - centre[1], z - centre[2]
        radius = np.sqrt(dx * dx + dy * dy + dz * dz)
        u = _around_y(dx, dz)
        v = 0.5 + np.arcsin(np.clip(np.where(radius > TINY, dy / np.maximum(radius, TINY), 0.0), -1.0, 1.0)) / np.pi
    uv = np.stack([u, v], axis=-1)
    return np.where(finite[..., None], uv, 0.0).astype(np.float32)
