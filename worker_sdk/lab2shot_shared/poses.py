"""Camera poses and rotations, numpy only: angles and means of rotations, the weighted similarity fit (Umeyama), axis
vectors, blending and interpolating camera-to-world poses, and pixels to world points (unproject; pixel (i, j) covers
[j, j+1] x [i, i+1], its centre at +0.5). One implementation for the core and the workers.

A rotation as an axis times an angle is motion.rotvec_to_matrix / matrix_to_rotvec (the same pair the SMPL family's
parameters are written in): one implementation, not one per line of work. So are the "XYZ" Euler angles and the
translate / rotate / scale matrix built from them (euler_xyz_matrix, xyz_euler_deg, trs_matrix): 「3D 变换」, the
transform handles, 「初始姿势」 and camera angles all read and write rotations this way."""

from __future__ import annotations

import bisect

import numpy as np

from .motion import axis_angle, matrix_to_quat, quat_to_matrix, slerp


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


def track_scale(xs: np.ndarray, xd: np.ndarray) -> tuple[float, bool]:
    """The scale putting centred points `xs` [N,3] (already turned) on `xd` by least squares, and whether the two ran
    opposite ways (the fit came out negative): (|s|, reversed). The one fit of a scale between two tracks
    (scaled_align; lab2shot/nodes/kit/align.py align_paths): a negative scale is a mirror no camera path means, so the
    size is kept and the caller says the direction. 1 where the points do not spread or the fit is no number."""
    xs, xd = np.asarray(xs, np.float64), np.asarray(xd, np.float64)
    denom = float((xs ** 2).sum())
    s = float((xd * xs).sum() / denom) if denom > 1e-12 else 1.0
    if not np.isfinite(s) or s == 0:
        return 1.0, False
    return abs(s), s < 0


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
    s = track_scale(xs @ r.T, xd)[0] if fit_scale else 1.0
    t = cd.mean(0) - s * (r @ cs.mean(0))
    big_t = np.eye(4)
    big_t[:3, :3], big_t[:3, 3] = s * r, t
    rel = np.swapaxes(r @ src[:, :3, :3], -1, -2) @ dst[:, :3, :3]
    ang = np.degrees(np.linalg.norm(matrix_to_rotvec(rel), axis=-1))
    pos = np.linalg.norm(s * (cs @ r.T) + t - cd, axis=-1)
    return big_t, s, spread, float(np.sqrt(np.mean(ang**2))), float(np.sqrt(np.mean(pos**2)))


def blend_poses(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Camera-to-world a -> b by t: rotation along the shortest arc (motion.slerp, the one rotation interpolation),
    translation linear."""
    q = slerp(matrix_to_quat(np.asarray(a[:3, :3], np.float64)), matrix_to_quat(np.asarray(b[:3, :3], np.float64)), t)
    out = np.array(b, np.float64)
    out[:3, :3] = quat_to_matrix(q)
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


# ------------------------------------------------------------------ Euler angles and TRS

_X, _Y, _Z = np.eye(3)


def euler_xyz_matrix(rotate_deg) -> np.ndarray:
    """Houdini / Maya "XYZ" order: rotate about X first, then Y, then Z (column vectors). [...,3] degrees ->
    [...,3,3]: three turns about the axes (motion.axis_angle), not a separate formula."""
    r = np.radians(np.asarray(rotate_deg, np.float64))
    return axis_angle(_Z, r[..., 2]) @ axis_angle(_Y, r[..., 1]) @ axis_angle(_X, r[..., 0])


def xyz_euler_deg(rotations: np.ndarray) -> np.ndarray:
    """Rotations [F,3,3] -> "XYZ" angles in degrees [F,3], the inverse of euler_xyz_matrix (at a gimbal lock, Z is 0),
    unwrapped over the frames so a turn past 180° goes on instead of jumping."""
    r = np.asarray(rotations, np.float64).reshape(-1, 3, 3)
    ry = np.arcsin(np.clip(-r[:, 2, 0], -1.0, 1.0))
    locked = np.abs(np.cos(ry)) < 1e-8
    rx = np.where(locked, np.arctan2(-r[:, 1, 2], r[:, 1, 1]), np.arctan2(r[:, 2, 1], r[:, 2, 2]))
    rz = np.where(locked, 0.0, np.arctan2(r[:, 1, 0], r[:, 0, 0]))
    return np.degrees(np.unwrap(np.stack([rx, ry, rz], -1), axis=0))


def trs_matrix(translate=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale=1.0) -> np.ndarray:
    """4x4 (column vectors): scale (one number, or one per axis), then rotate ("XYZ", degrees), then translate. The
    viewer builds the same matrix while a handle is dragged (webui model/places.ts placeMatrix; lab2shot check places
    runs both)."""
    m = np.eye(4)
    m[:3, :3] = euler_xyz_matrix(rotate) @ np.diag(np.broadcast_to(np.asarray(scale, np.float64), 3))
    m[:3, 3] = translate
    return m
