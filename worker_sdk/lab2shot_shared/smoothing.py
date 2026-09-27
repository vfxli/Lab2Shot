"""Zero-phase temporal smoothing over a shot, numpy only: values, angles, quaternion tracks. One implementation for the
core's camera smoothing and the workers' people tracks."""

from __future__ import annotations

import numpy as np

from .motion import continuous

def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian filter along axis 0, edges handled by renormalization."""
    if sigma <= 0 or len(values) < 3:
        return values.copy()
    radius = int(np.ceil(3 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    flat = values.reshape(len(values), -1)
    out = np.empty_like(flat, dtype=np.float64)
    for i in range(len(flat)):
        lo, hi = max(0, i - radius), min(len(flat), i + radius + 1)
        k = kernel[lo - i + radius : hi - i + radius]
        out[i] = (k[:, None] * flat[lo:hi]).sum(0) / k.sum()
    return out.reshape(values.shape)


def smooth_angles(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth angle channels (radians) without wrapping artifacts."""
    return gaussian_smooth(np.unwrap(values, axis=0), sigma)


def smooth_quats(q: np.ndarray, sigma: float) -> np.ndarray:
    """[F,4] unit quaternions: sign-align, Gaussian-average, renormalize."""
    s = gaussian_smooth(continuous(q), sigma)
    return s / np.linalg.norm(s, axis=-1, keepdims=True)
