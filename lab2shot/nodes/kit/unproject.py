"""将深度图反投影为点。核心中唯一的 back-projection 实现（camera_points / unproject_depth）。"""

from __future__ import annotations

import numpy as np


def camera_points(z, x, y, focal_px: float, width: int, height: int, cam_to_world: np.ndarray, principal=None) -> np.ndarray:
    """Points [M,3] (cm) at view-axis distances `z` (cm) behind image positions (x, y) (pixels, pixel centres at
    +0.5), through a pinhole with principal point `principal` ((cx, cy) in pixels; the image centre when None) and a
    GL camera-to-world matrix (identity yields camera space). This is the core's only back-projection:
    lab2shot_shared.poses.unproject_at with OpenCV camera axes converted to GL. The principal point is taken from the
    camera (data/camera.py CameraSamples.principal_px), so a principal point written by a solver is respected, matching
    that solver's own projection."""
    from lab2shot_shared.poses import unproject_at

    from ...data.units import CV_TO_GL

    cx, cy = (width / 2, height / 2) if principal is None else (float(principal[0]), float(principal[1]))
    k = np.array([[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]])
    p = unproject_at(z, x, y, k, np.eye(4)) * CV_TO_GL
    return p @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def unproject_depth(z: np.ndarray, focal_px: float, cam_to_world: np.ndarray, rows: np.ndarray, cols: np.ndarray, principal=None) -> np.ndarray:
    """Pixels (rows, cols) of a depth map (cm along the view axis) -> points [M,3] in cm, via camera_points at the
    pixel centres."""

    h, w = z.shape
    return camera_points(z[rows, cols], cols + 0.5, rows + 0.5, focal_px, w, h, cam_to_world, principal)
