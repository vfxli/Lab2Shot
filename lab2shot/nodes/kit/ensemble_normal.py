"""Normal ensemble kernels: several normal maps of one shot, in one camera space, combined on the unit sphere.

The node (nodes/core/ensemble_normal.py) reads the candidates and writes what these functions return. numpy only.
A normal is a unit 3-vector; NaN marks a pixel where a candidate has none. Angles are in degrees.

Objectives (the node's 「目标」):
- accuracy: per pixel a spherical median (a few Weiszfeld steps from the weighted mean direction), the spread
  s = 1.4826 · weighted median angle to it, candidates further than k·max(s, floor) left out there, the weighted mean
  direction of the rest;
- robust: the spherical median itself (nothing averaged in: one far-off candidate cannot pull it).
Candidates of one model (MoGe's own normals and the normals of MoGe's depth) share one vote (`vote_weights`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ensemble import MAD_TO_SIGMA, wmedian

NORMAL_OBJECTIVES = ("accuracy", "robust")
NORMAL_OUTLIER_K = 3.0  # a candidate further than this many spreads from the median is left out of the mean there
NORMAL_FLOOR_DEG = 5.0  # the spread is never taken below this (candidates that agree closely are all kept)
WEISZFELD_STEPS = 3  # steps towards the spherical median (accuracy); robust takes more
MEDIAN_STEPS = 8


def unit_normals(n: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """[..., 3] -> unit vectors, NaN where there is none (zero length, not finite, or not `valid`)."""
    n = np.asarray(n, np.float64)
    length = np.linalg.norm(np.nan_to_num(n), axis=-1)
    ok = np.isfinite(n).all(-1) & (length > 1e-6)
    if valid is not None:
        ok &= np.asarray(valid, bool)
    return np.where(ok[..., None], n / np.maximum(length, 1e-12)[..., None], np.nan)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between unit vectors (NaN where either has none)."""
    d = np.clip(np.einsum("...k,...k->...", a, b), -1.0, 1.0)
    return np.degrees(np.arccos(d))


def vote_weights(w: Sequence[float], groups: Sequence[int]) -> np.ndarray:
    """Per-candidate weights with every group's members sharing their weight: one model behind several candidates is
    one vote (each member gets its weight divided by the group's size)."""
    w = np.asarray(w, np.float64)
    g = np.asarray(groups)
    out = w.copy()
    for k in set(groups):
        m = g == k
        out[m] = w[m] / m.sum()
    return out


def mean_direction(N: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mean direction of N [C,H,W,3] (NaN = none), per-pixel weights w [C,H,W]; NaN where nothing."""
    ok = np.isfinite(N).all(-1)
    ww = np.where(ok, w, 0.0)
    s = np.einsum("chw,chwk->hwk", ww, np.nan_to_num(N))
    return unit_normals(s, ww.sum(0) > 0)


class NormalConsensus:
    """value [H,W,3] (NaN = none), spread [H,W] degrees (NaN where none), kept [C,H,W], nearest [H,W] (the candidate
    closest to the value, -1 where none)."""

    def __init__(self, value, spread, kept, nearest):
        self.value, self.spread, self.kept, self.nearest = value, spread, kept, nearest


def normal_consensus(N: np.ndarray, w: np.ndarray, objective: str = "accuracy", k: float = NORMAL_OUTLIER_K,
                     floor: float = NORMAL_FLOOR_DEG) -> NormalConsensus:
    """Combine N [C,H,W,3] with per-pixel weights w [C,H,W] (0 = not there) by `objective` (see the module)."""
    N = np.asarray(N, np.float64)
    ok = np.isfinite(N).all(-1) & (w > 0)
    W = np.where(ok, w, 0.0)
    m = mean_direction(N, W)
    for _ in range(MEDIAN_STEPS if objective == "robust" else WEISZFELD_STEPS):
        a = np.radians(angle_deg(N, m[None]))
        m = mean_direction(N, W / np.maximum(np.nan_to_num(a, nan=1.0), np.radians(1.0)))
    a = angle_deg(N, m[None])
    s = MAD_TO_SIGMA * wmedian(np.where(ok, a, np.nan), W)
    if objective == "robust":
        value, keep = m, ok
    else:
        keep = ok & (a <= k * np.maximum(np.nan_to_num(s, nan=0.0), floor)[None])
        value = mean_direction(N, np.where(keep, W, 0.0))
    has = np.isfinite(value).all(-1)
    da = np.where(ok, angle_deg(N, np.nan_to_num(value)[None]), np.inf)
    near = np.where(has & ok.any(0), np.argmin(da, 0), -1)
    return NormalConsensus(value, np.where(has, s, np.nan), keep, near)


def normal_confidence(spread: np.ndarray, floor: float = NORMAL_FLOOR_DEG) -> np.ndarray:
    """0..1 from the candidates' spread (degrees): floor/(floor + spread), one half at `floor`; 0 where none."""
    s = np.asarray(spread, np.float64)
    return np.where(np.isfinite(s), floor / (floor + np.nan_to_num(s, nan=0.0)), 0.0)


def median_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Median angle (degrees) between two normal maps where both have a value; NaN where never."""
    d = angle_deg(a, b)
    return float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")
