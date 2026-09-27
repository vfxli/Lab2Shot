"""Putting one set of 3D points (or one camera path) onto another: the one place the core fits a similarity, measures
what is left over and says when the data cannot decide the fit.

Everything that lines two things up in the core comes through here: 「相机对比」 (align_paths), and the scene tools
that follow (落地, 定比例, 场景对齐). The weighted Umeyama itself is not written again: it lives in
lab2shot_shared.poses, which the workers use too, and `similarity` builds the residuals, the RMS and the degenerate
cases on top of it.

Lengths are centimetres, as everywhere in the core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A set of points whose second spread is under this share of its first runs along a line: the turn about that line
# cannot be read from the positions (a straight dolly, a row of tracked points on a wall's edge).
LINE = 0.05
POINT_CM = 1.0  # points spread less than this are one point: neither turn nor scale can be read from them

SIMILARITY, RIGID, SCALE = "similarity", "rigid", "scale"


@dataclass(frozen=True)
class Fit:
    """dst ≈ s R src + t, with what it cost: `residuals` is the distance left at each pair (cm), `rms` and `max` over
    them, and `degenerate` says when the points themselves could not decide the fit — None (they could), "line" (they
    lie along a line: the turn about it is whatever the fit happened to pick) or "point" (they sit on one spot: no
    turn and no scale at all)."""

    s: float
    R: np.ndarray  # [3,3]
    t: np.ndarray  # [3]
    residuals: np.ndarray  # [N] cm
    degenerate: str | None = None

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.residuals**2))) if len(self.residuals) else 0.0

    @property
    def max(self) -> float:
        return float(self.residuals.max()) if len(self.residuals) else 0.0

    @property
    def matrix(self) -> np.ndarray:
        """The fit as one 4x4 (column vectors), the way data/scene.py transform takes it."""
        m = np.eye(4)
        m[:3, :3] = self.s * self.R
        m[:3, 3] = self.t
        return m

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.s * np.asarray(points, np.float64) @ self.R.T + self.t


def spread(points: np.ndarray) -> np.ndarray:
    """The three singular values of the centred points (cm): how far they reach along their own axes."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    if len(p) < 2:
        return np.zeros(3)
    return np.linalg.svd(p - p.mean(0), compute_uv=False)


def degeneracy(src: np.ndarray, dst: np.ndarray) -> str | None:
    """Whether these pairs can decide a similarity at all: "point" (either side sits on one spot), "line" (either side
    runs along a line, so the turn about it is undecided), None."""
    a, b = spread(src), spread(dst)
    if min(a[0], b[0]) < POINT_CM:
        return "point"
    if min(a[1] / a[0], b[1] / b[0]) < LINE:
        return "line"
    return None


def similarity(src, dst, weights=None, mode: str = SIMILARITY) -> Fit:
    """The fit putting `src` [N,3] onto `dst` [N,3] (cm), each pair as important as `weights` says (None: all alike).

    `mode`: "similarity" scale, turn and move; "rigid" turn and move only (both sides are already real size);
    "scale" the scale alone, with no turn (the two sets are already in the same orientation).
    """
    src = np.asarray(src, np.float64).reshape(-1, 3)
    dst = np.asarray(dst, np.float64).reshape(-1, 3)
    if len(src) != len(dst):
        raise ValueError(f"{len(src)} src points against {len(dst)} dst points")
    w = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64).reshape(-1)
    if len(w) != len(src):
        raise ValueError(f"{len(w)} weights for {len(src)} pairs")
    if mode not in (SIMILARITY, RIGID, SCALE):
        raise ValueError(f"mode must be one of {SIMILARITY}, {RIGID}, {SCALE}, not {mode!r}")
    if not len(src):
        return Fit(1.0, np.eye(3), np.zeros(3), np.zeros(0), "point")

    from lab2shot_shared.poses import umeyama

    degenerate = degeneracy(src, dst)
    if degenerate == "point":  # one spot: neither turn nor scale can be read, and a fit would divide by nothing
        wn = w / w.sum()
        t = wn @ dst - wn @ src
        return Fit(1.0, np.eye(3), t, np.linalg.norm(src + t - dst, axis=1), degenerate)
    if mode == SCALE:
        wn = w / w.sum()
        mu_s, mu_d = wn @ src, wn @ dst
        xs, xd = src - mu_s, dst - mu_d
        denom = float(wn @ (xs**2).sum(1))
        s = float(wn @ (xs * xd).sum(1) / denom) if denom > 1e-18 else 1.0
        R, t = np.eye(3), mu_d - s * mu_s
    else:
        s, R, t = umeyama(src, dst, w)
        if mode == RIGID:
            wn = w / w.sum()
            s, t = 1.0, wn @ dst - R @ (wn @ src)
    residuals = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
    return Fit(float(s), R, t, residuals, degenerate)


def shortest_arc(a, b) -> np.ndarray:
    """The rotation (3x3) taking direction `a` onto direction `b` the short way round, turning about nothing else
    (identity when they already agree; a half turn about any perpendicular axis when they are opposite)."""
    a = np.asarray(a, np.float64).reshape(3)
    b = np.asarray(b, np.float64).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a, b = a / na, b / nb
    axis = np.cross(a, b)
    sin, cos = float(np.linalg.norm(axis)), float(a @ b)
    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        # opposite: turn half way about any axis across `a`
        other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        k = np.cross(a, other)
        k /= np.linalg.norm(k)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = axis / sin
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + sin * K + (1 - cos) * K @ K


def align_paths(pa: np.ndarray, pb: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, str]:
    """(s, R, t, note): the similarity putting camera A's path on reference B's, pb ≈ s R pa + t, from positions
    (`similarity`) where the paths span a plane or more. A path that stays in place or runs along a line leaves the
    rotation about it unknown: the turn then comes from the cameras' orientations (rb ≈ R ra), the scale from the
    positions along the line, and `note` (a message code, "" none) says so."""
    from lab2shot_shared.poses import mean_rotation

    fit = similarity(pa, pb)
    if fit.degenerate == "point":  # a camera that hardly moves: its centre says nothing of turn or scale
        R = mean_rotation(rb @ np.swapaxes(ra, 1, 2))
        still_ref = spread(pb)[0] < POINT_CM
        return 1.0, R, pb.mean(0) - R @ pa.mean(0), "N-COMPARE-REFSTILL" if still_ref else "N-COMPARE-CAMSTILL"
    if fit.degenerate == "line":  # along a line (a straight dolly or truck)
        R = mean_rotation(rb @ np.swapaxes(ra, 1, 2))
        xa, xb = (pa - pa.mean(0)) @ R.T, pb - pb.mean(0)
        s = float((xa * xb).sum() / (xa * xa).sum())
        return s, R, pb.mean(0) - s * R @ pa.mean(0), "N-COMPARE-LINE"
    return fit.s, fit.R, fit.t, ""
