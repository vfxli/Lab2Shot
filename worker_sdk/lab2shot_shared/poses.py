"""Camera poses and rotations, numpy only: angles and means of rotations, the weighted similarity fit (Umeyama), axis
vectors, blending and interpolating camera-to-world poses, and pixels to world points (unproject; pixel (i, j) covers
[j, j+1] x [i, i+1], its centre at +0.5). One implementation for the core and the workers.

A rotation as an axis times an angle is motion.rotvec_to_matrix / matrix_to_rotvec (the same pair the SMPL family's
parameters are written in): one implementation, not one per line of work."""

from __future__ import annotations

import bisect
import math

import numpy as np


def rays_at(K: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The camera-space ray at depth 1, (x, y), of image positions (x, y) in pixels, a pixel's centre at +0.5 (K's corner
    convention): the one place a position becomes a ray."""
    return (x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1]


def pixel_rays(K: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The camera-space ray of pixels (rows, cols) at depth 1: rays_at their centres. Broadcasts: rows [H,1] and cols
    [1,W] give a whole frame's rays."""
    return rays_at(K, cols + 0.5, rows + 0.5)


def unproject_at(z: np.ndarray, x: np.ndarray, y: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """World points [M,3] at camera depths `z` under image positions (x, y) (pixels, a centre at +0.5): the one
    back-projection, for the core and the workers (OpenCV camera axes; a GL caller turns them: lab2shot data/scene.py
    camera_points)."""
    rx, ry = rays_at(K, np.asarray(x, np.float64), np.asarray(y, np.float64))
    z = np.asarray(z, np.float64)
    return np.stack([rx * z, ry * z, z], axis=1) @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]


def unproject(depth: np.ndarray, K: np.ndarray, cam_to_world: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """World points [M,3] of whole pixels (rows, cols) of a depth map: unproject_at their centres."""
    return unproject_at(depth[rows, cols], cols + 0.5, rows + 0.5, K, cam_to_world)


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of [N,3,3] rotations (SVD projection of their sum)."""
    U, _, Vt = np.linalg.svd(rotations.sum(0))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    return U @ D @ Vt


def rotation_deg(R: np.ndarray) -> np.ndarray:
    """Rotation angle(s) in degrees of [...,3,3] rotations (pass [..., :3, :3] of a pose)."""
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def umeyama(src: np.ndarray, dst: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted similarity with dst ~ s R src + t."""
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd((xd * w[:, None]).T @ xs)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (w @ (xs**2).sum(1)))
    return s, R, mu_d - s * R @ mu_s


def rigid_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, float]:
    """One rigid transform T with dst[i] ≈ T @ src[i] for two tracks of cam_to_world poses [N,4,4].

    This is the algorithm of the correction step: a solve result comes with its own world, in which it has a reference
    camera. The method's own camera is not used directly, only its difference from the reference camera: the two
    cameras' tracks are aligned rigidly, and the resulting constant transform is the correction applied to the result,
    so that it matches the plate under the user's camera. No scaling: both sides are in real units (metres /
    centimetres), and scaling would change the person's height.

    One implementation shared by the core and the workers: the workers' world merge
    `lab2shot_worker/world_humans.py one_world` (metres) calls this function; the core's 「相机空间转换」
    (`lab2shot/nodes/core/scene.py CameraSpaceConvert`) in its 「整段平滑」 mode calls `scaled_align` below, which
    includes scale (the two cameras have their own scales and cannot be aligned without it).

    Returns (T, rms rotation error in degrees, rms camera-centre error in the poses' own length unit: metres for the
    workers' raw arrays, centimetres in the core).
    """
    from .motion import matrix_to_rotvec

    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    r = mean_rotation(dst[:, :3, :3] @ np.swapaxes(src[:, :3, :3], -1, -2))
    t = (dst[:, :3, 3] - src[:, :3, 3] @ r.T).mean(0)
    big_t = np.eye(4)
    big_t[:3, :3], big_t[:3, 3] = r, t
    moved = big_t @ src
    rel = np.swapaxes(moved[:, :3, :3], -1, -2) @ dst[:, :3, :3]
    ang = np.degrees(np.linalg.norm(matrix_to_rotvec(rel), axis=-1))
    pos = np.linalg.norm(moved[:, :3, 3] - dst[:, :3, 3], axis=-1)
    return big_t, float(np.sqrt(np.mean(ang**2))), float(np.sqrt(np.mean(pos**2)))


def scaled_align(src: np.ndarray, dst: np.ndarray, fit_scale: bool = True) -> tuple[np.ndarray, float, float, float, float]:
    """One similarity T (s·R, t) with dst[i] ≈ T @ src[i] for two tracks of cam_to_world poses [N,4,4] of the same
    shot: what 「相机空间转换」's 「整段平滑」 hangs on a scene (lab2shot/nodes/core/scene.py CameraSpaceConvert).

    The rotation comes from the cameras' orientations (chordal mean of dst·srcᵀ, like rigid_align), the scale and the
    translation from their centres by least squares given that rotation, so a short or nearly straight track still gets a
    well-determined rotation, and only the scale needs the cameras to have moved. Two solvers of one shot each pick their own scale
    (a body's height, a depth model), so a rigid fit cannot put their tracks together; the scale found here is the
    target world's over the source's, and it applies to the whole scene, a person's height included: in the target
    world's units that is their height, or they would not project onto the picture.

    Returns (T, s, spread, rms rotation error in degrees, rms centre error); `spread` is the source centres' RMS
    distance from their mean (their own length unit). `fit_scale=False` keeps s = 1 (rotation and translation only): what the
    caller asks for when the cameras hardly moved, since the scale read from a track a few centimetres long is noise
    (a hand-held shot whose camera track is 5 cm long fitted a scale of 0.034: the person shrinks to 3%)."""
    from .motion import matrix_to_rotvec

    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    r = mean_rotation(dst[:, :3, :3] @ np.swapaxes(src[:, :3, :3], -1, -2))
    cs, cd = src[:, :3, 3], dst[:, :3, 3]
    xs, xd = cs - cs.mean(0), cd - cd.mean(0)
    denom = float((xs**2).sum())
    spread = float(np.sqrt(denom / max(len(cs), 1)))
    s = float((xd * (xs @ r.T)).sum() / denom) if fit_scale and denom > 1e-12 else 1.0
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    t = cd.mean(0) - s * (r @ cs.mean(0))
    big_t = np.eye(4)
    big_t[:3, :3], big_t[:3, 3] = s * r, t
    rel = np.swapaxes(r @ src[:, :3, :3], -1, -2) @ dst[:, :3, :3]
    ang = np.degrees(np.linalg.norm(matrix_to_rotvec(rel), axis=-1))
    pos = np.linalg.norm(s * (cs @ r.T) + t - cd, axis=-1)
    return big_t, s, spread, float(np.sqrt(np.mean(ang**2))), float(np.sqrt(np.mean(pos**2)))


def blend_poses(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Camera-to-world a -> b by t: rotation along the shortest arc, translation linear."""
    Ra, Rb = a[:3, :3], b[:3, :3]
    rel = Ra.T @ Rb
    angle = math.acos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0))
    out = b.copy()
    if angle > 1e-9:
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]) / (2 * math.sin(angle))
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        th = t * angle
        out[:3, :3] = Ra @ (np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k)
    else:
        out[:3, :3] = Ra
    out[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
    return out


def interpolate_poses(known: dict[int, np.ndarray], frames) -> np.ndarray:
    """cam_to_world [F,4,4] for every frame from poses at some of them: blend_poses between the known neighbours,
    held before the first / after the last known frame."""
    keys = sorted(known)
    out = []
    for f in (int(f) for f in frames):
        j = bisect.bisect_left(keys, f)
        if j < len(keys) and keys[j] == f:
            out.append(np.asarray(known[f], np.float64))
            continue
        a, b = keys[max(j - 1, 0)], keys[min(j, len(keys) - 1)]
        out.append(blend_poses(np.asarray(known[a], np.float64), np.asarray(known[b], np.float64),
                               0.0 if a == b else (f - a) / (b - a)))
    return np.stack(out)
