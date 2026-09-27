"""Taking the shake out of a solved camera: a weighted Whittaker smoother (least squares against the second
difference) and the banded solver it needs.

Why not a Gaussian: a Gaussian treats every frequency the same, so turning it up enough to lose the frame-to-frame
jitter also rounds off a hard stop or a whip. A Whittaker smoother lets the strength be said in the words a camera
operator uses — 「shake shorter than N frames」 — and, reweighted, leaves a real change of acceleration alone.

The strength is a cutoff period T in frames: λ = 1 / (2 − 2cos(2π/T))², which puts the response of a wobble of period
T at exactly 0.5. The system is pentadiagonal, so the whole shot costs O(F); the main environment has no scipy and
does not get one for this, so the banded Cholesky is here (`band_cholesky` / `band_solve`, it must agree with a dense
solve).

Every channel of one group (the three position axes, the four quaternion components) is smoothed with the same λ and
the same weights. That matters: it makes the smoother a scalar on the channels, so deshaking and then moving the whole
scene gives exactly what moving it and then deshaking gives.
"""

from __future__ import annotations

import numpy as np

CUTOFF_MIN, CUTOFF_MAX, CUTOFF_DEFAULT = 2.0, 30.0, 6.0
SUDDEN_TIMES = 4.0  # curvature this many times the shot's own is a real change of acceleration, not shake
ROUNDS = 5  # reweighting rounds (the first is the plain smooth)
REFERENCE_CM = 500.0  # how far away the plate is taken to be when no point cloud says


def lam_for(cutoff: float) -> float:
    """The penalty weight whose response at a wobble of period `cutoff` frames is 0.5."""
    return 1.0 / (2.0 - 2.0 * np.cos(2.0 * np.pi / float(cutoff))) ** 2


def band_cholesky(ab: np.ndarray) -> np.ndarray:
    """In-place Cholesky of a symmetric positive definite band matrix in lower band storage (ab[i, j] = A[j+i, j],
    i = 0..k), the LAPACK dpbtf2 recurrence. Returns L in the same storage."""
    ab = np.array(ab, np.float64)
    k, n = ab.shape[0] - 1, ab.shape[1]
    for j in range(n):
        d = ab[0, j]
        if d <= 0.0:
            raise ValueError("the smoothing system is not positive definite")
        d = np.sqrt(d)
        ab[0, j] = d
        kn = min(k, n - 1 - j)
        if kn <= 0:
            continue
        ab[1:kn + 1, j] /= d
        for i in range(1, kn + 1):
            ab[0:kn - i + 1, j + i] -= ab[i, j] * ab[i:kn + 1, j]
    return ab


def band_solve(chol: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b for the factor `chol` band_cholesky gave; b is [n] or [n, C] (a column per channel)."""
    x = np.array(b, np.float64)
    one = x.ndim == 1
    if one:
        x = x[:, None]
    k, n = chol.shape[0] - 1, chol.shape[1]
    for j in range(n):  # L y = b
        x[j] /= chol[0, j]
        for i in range(1, min(k, n - 1 - j) + 1):
            x[j + i] -= chol[i, j] * x[j]
    for j in range(n - 1, -1, -1):  # Lᵀ x = y
        for i in range(1, min(k, n - 1 - j) + 1):
            x[j] -= chol[i, j] * x[j + i]
        x[j] /= chol[0, j]
    return x[:, 0] if one else x


def _system(n: int, lam: float, penalty: np.ndarray) -> np.ndarray:
    """I + λ Dᵀ V D in lower band storage, D the (n-2)×n second difference and V = diag(`penalty`)."""
    ab = np.zeros((3, n))
    ab[0] = 1.0
    w = lam * np.asarray(penalty, np.float64).reshape(-1)
    np.add.at(ab[0], np.arange(n - 2), w)
    np.add.at(ab[0], np.arange(1, n - 1), 4.0 * w)
    np.add.at(ab[0], np.arange(2, n), w)
    np.add.at(ab[1], np.arange(n - 2), -2.0 * w)
    np.add.at(ab[1], np.arange(1, n - 1), -2.0 * w)
    np.add.at(ab[2], np.arange(n - 2), w)
    return ab


def _second_difference(x: np.ndarray) -> np.ndarray:
    return x[:-2] - 2.0 * x[1:-1] + x[2:]


def smooth(x: np.ndarray, cutoff: float, keep_sudden: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Smooth the channels of `x` [F, C] together: shake with a period under `cutoff` frames goes, everything slower
    stays. With `keep_sudden`, frames whose second difference stands far above the jitter level keep their sharpness
    (a hard stop, a whip): their share of the penalty is turned down, over `ROUNDS` reweighting rounds.

    Returns the smoothed channels and, per inner frame (index 1 .. F-2), how much of the penalty was left there
    (1.0 = fully smoothed, towards 0 = left as it came in)."""
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    n = len(x)
    if n < 3:
        return x.copy(), np.ones(max(n - 2, 0))
    lam = lam_for(cutoff)
    penalty = np.ones(n - 2)
    y = band_solve(band_cholesky(_system(n, lam, penalty)), x)
    if not keep_sudden:
        return y, penalty
    for _ in range(ROUNDS - 1):
        kept = np.linalg.norm(_second_difference(y), axis=1)  # the curvature the result still has
        # what a frame's curvature is measured against is the curvature of the result itself, robustly (its MAD about
        # zero), not the shake's own second difference: the shake's is larger by about the square of the cutoff, so
        # nothing would ever stand above it and the reweighting would never do anything
        level = float(np.median(kept))
        if not (level > 1e-12):
            break
        over = np.maximum(kept / (SUDDEN_TIMES * level), 1.0)
        penalty = 1.0 / over**2
        y = band_solve(band_cholesky(_system(n, lam, penalty)), x)
    return y, penalty


SUDDEN_LEFT = 0.25  # a frame keeping less than this share of the penalty is reported as one that was left sharp


def sudden_frames(frames, penalty: np.ndarray) -> list[int]:
    """The frame numbers `smooth` left sharp (its `penalty`, one per inner frame)."""
    return [int(frames[i + 1]) for i in np.flatnonzero(np.asarray(penalty) < SUDDEN_LEFT)]


def plate_shift_px(before: np.ndarray, after: np.ndarray, focal_px: np.ndarray, distance_cm: float = REFERENCE_CM) -> np.ndarray:
    """How far, in pixels, a point `distance_cm` straight ahead of the original camera slides on the plate once the
    camera is the deshaken one: per frame, positions and turn together. Camera-to-world [F,4,4], GL axes (the camera
    looks down -Z)."""
    before = np.asarray(before, np.float64)
    after = np.asarray(after, np.float64)
    point = before[:, :3, 3] - distance_cm * before[:, :3, 2]  # straight ahead of the camera it was solved as
    local = np.einsum("fji,fj->fi", after[:, :3, :3], point - after[:, :3, 3])  # Rᵀ (X − C)
    depth = np.maximum(-local[:, 2], 1e-6)
    focal = np.asarray(focal_px, np.float64).reshape(-1)
    return np.hypot(local[:, 0] / depth, local[:, 1] / depth) * focal


# ------------------------------------------------------------------ a whole camera


def deshake_camera(src, out, strength: float, keep_sudden: bool, rotation: bool, focal: str):
    """The camera with its shake taken out, and what that cost. `focal`: "keep" leaves the focal length alone,
    "smooth" takes the shake out of it too (a zoom). Returns (packet, report) with the report a dict of plain numbers
    the node turns into its messages and its before/after curves."""
    from lab2shot_shared.motion import continuous, matrix_to_quat, quat_to_matrix

    from ...data.camera import CameraSamples
    from ...data.scene import xyz_euler_deg
    from ...data.units import DEFAULT_HEIGHT, DEFAULT_WIDTH

    frames = [int(f) for f in src.meta["frames"]]
    w, h = int(src.meta.get("width") or DEFAULT_WIDTH), int(src.meta.get("height") or DEFAULT_HEIGHT)
    s = CameraSamples.from_packet(src, frames)
    before = s.poses().copy()
    mats = before.copy()
    positions, penalty = smooth(before[:, :3, 3], strength, keep_sudden)
    mats[:, :3, 3] = positions
    turn_penalty = np.ones(max(len(frames) - 2, 0))
    if rotation:
        quats, turn_penalty = smooth(continuous(matrix_to_quat(before[:, :3, :3])), strength, keep_sudden)
        mats[:, :3, :3] = quat_to_matrix(quats / np.linalg.norm(quats, axis=1, keepdims=True))
    lens_before = s.focal_px(w)
    lens = smooth(lens_before[:, None], strength, keep_sudden)[0][:, 0] if focal == "smooth" else lens_before
    shift = plate_shift_px(before, mats, lens_before) if len(frames) else np.zeros(0)
    left = np.minimum(penalty, turn_penalty) if rotation else penalty
    report = {
        "moved_cm": float(np.linalg.norm(mats[:, :3, 3] - before[:, :3, 3], axis=1).max(initial=0.0)),
        "turned_deg": float(np.abs(xyz_euler_deg(mats[:, :3, :3]) - xyz_euler_deg(before[:, :3, :3])).max(initial=0.0)),
        "shift_px": float(shift.max(initial=0.0)),
        "shift_frame": int(frames[int(np.argmax(shift))]) if len(shift) else 0,
        "distance_cm": REFERENCE_CM,
        "sudden": sudden_frames(frames, left),
        "before": {"translate": before[:, :3, 3], "rotate": xyz_euler_deg(before[:, :3, :3]), "focal_px": lens_before},
        "after": {"translate": mats[:, :3, 3], "rotate": xyz_euler_deg(mats[:, :3, :3]), "focal_px": lens},
    }
    keep = {k: v for k, v in src.meta.items() if k not in ("frames", "width", "height")}
    solved = CameraSamples.solved(frames, w, h, lens, mats, filmback_mm=s.filmback_mm,
                                  info={"deshaken": {"strength": strength, "keep_sudden": keep_sudden,
                                                     "rotation": rotation, "focal": focal}})
    return solved.write(out, **keep), report
