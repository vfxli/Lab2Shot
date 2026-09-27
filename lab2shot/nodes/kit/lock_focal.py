"""Holding a solved camera's focal length still, and moving the camera so the plate still lines up.

Machine-learning solvers (ViPE, MegaSaM, MapAnything ...) estimate the focal length frame by frame. When it wobbles,
the solver pushes the camera back and forth along its own axis to make up for it, so the two curves wobble together.
Setting the focal length to a constant and leaving the camera where it is therefore breaks the match with the plate:
the camera has to be solved again at the locked focal length.

Not a full bundle adjustment: that needs 2D tracks across frames, which these solvers do not hand out, and it would
move the solved 3D as well, so the point cloud would no longer sit on the picture. Instead every frame is solved on
its own (PnP) against anchors the shot already has:

  * anchors: the scene points this frame sees — a per-frame cloud's own points, a static cloud's points inside the
    frustum and in front of the camera — picked evenly over the picture on a 32 x 18 grid, one per cell, nearest its
    centre. No random numbers anywhere: the same input gives the same output, byte for byte.
  * what was seen: the anchors put through the ORIGINAL camera, which is exactly where the solver had them on the
    plate. Wire real 2D tracks in and their measured pixels are used instead, which is better evidence.
  * a closed-form start: keep the orientation and slide along the lens axis by z_med (1 - F / f_i), which is the exact
    answer for a plane square to the lens (undoing the dolly-zoom the solver put in), then Levenberg-Marquardt on the
    six pose numbers with a Huber loss, the focal length held at F.

What is left over (the per-frame reprojection residual) is the part of the change that moving the camera cannot make
up for, because the perspective itself differs. The node reports it.
"""

from __future__ import annotations

import numpy as np

GRID_X, GRID_Y = 32, 18  # the picture's cells, one anchor each: an even spread with no random numbers
MOST_ANCHORS = 2000
MOST_CONSIDERED = 200_000  # points of a static cloud looked at per frame (a stride over a bigger one, deterministic)
FEW_ANCHORS = 30  # under this a frame is only worth the closed-form start
NEAR_CM = 1.0  # a point nearer than this is behind the lens as far as a projection is concerned
ROUNDS = 20  # Levenberg-Marquardt steps
HUBER_PX = 2.0
ZOOM_SHARE = 0.20  # first-to-last focal change this big, and smooth, looks like a real zoom
ZOOM_WANDER = 1.5  # ... "smooth": the path length is at most this many times the straight change


def pixels(points: np.ndarray, pose: np.ndarray, focal_px: float, principal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points [N,3] (cm) through one GL camera (`pose` camera-to-world [4,4], `principal` (cx, cy) in pixels):
    pixel positions [N,2] and how far in front of the lens each is [N] (negative: behind it)."""
    p = (np.asarray(points, np.float64) - pose[:3, 3]) @ pose[:3, :3]  # Rᵀ (X − C), the camera's own axes
    depth = -p[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    return np.stack([focal_px * p[:, 0] / safe + principal[0], principal[1] - focal_px * p[:, 1] / safe], -1), depth


def _even_over_the_picture(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """The indices of an even spread of `uv` over the picture: the 32 x 18 grid, the point nearest each cell's centre.
    Deterministic — ties go to the lower index."""
    col = np.clip((uv[:, 0] / width * GRID_X).astype(np.int64), 0, GRID_X - 1)
    row = np.clip((uv[:, 1] / height * GRID_Y).astype(np.int64), 0, GRID_Y - 1)
    cell = row * GRID_X + col
    centre = np.stack([(col + 0.5) * width / GRID_X, (row + 0.5) * height / GRID_Y], -1)
    away = np.linalg.norm(uv - centre, axis=1)
    order = np.lexsort((away, cell))  # by cell, then by how close to its centre
    first = np.concatenate([[True], cell[order][1:] != cell[order][:-1]])
    return np.sort(order[first])[:MOST_ANCHORS]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def solve_pose(points: np.ndarray, seen: np.ndarray, pose: np.ndarray, focal_px: float,
               principal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One frame's camera pose with the focal length held at `focal_px`: start from `pose` (camera-to-world [4,4]) and
    fit the six pose numbers so the `points` [N,3] land on `seen` [N,2]. Returns the pose and the residual of each
    point (pixels)."""
    from lab2shot_shared.motion import rotvec_to_matrix  # 轴 × 角的转换只有这一份

    m = np.array(pose, np.float64)
    uv, _ = pixels(points, m, focal_px, principal)
    best = np.linalg.norm(uv - seen, axis=1)
    damping = 1e-3
    for _ in range(ROUNDS):
        p = (points - m[:3, 3]) @ m[:3, :3]
        depth = -p[:, 2]
        ahead = depth > NEAR_CM
        if ahead.sum() < 3:
            break
        q, d = p[ahead], depth[ahead]
        here = np.stack([focal_px * q[:, 0] / d + principal[0], principal[1] - focal_px * q[:, 1] / d], -1)
        err = here - seen[ahead]
        # d(pixel)/d(point in camera space), then d(point)/d(pose): a turn about the camera's own axes and a move
        jp = np.zeros((len(q), 2, 3))
        jp[:, 0, 0] = focal_px / d
        jp[:, 0, 2] = focal_px * q[:, 0] / d**2
        jp[:, 1, 1] = -focal_px / d
        jp[:, 1, 2] = -focal_px * q[:, 1] / d**2
        dp = np.zeros((len(q), 3, 6))
        dp[:, :, :3] = np.stack([_skew(x) for x in q])
        dp[:, 0, 3] = dp[:, 1, 4] = dp[:, 2, 5] = -1.0
        j = jp @ dp  # [N,2,6]
        size = np.linalg.norm(err, axis=1)
        w = np.where(size > HUBER_PX, HUBER_PX / np.maximum(size, 1e-9), 1.0)
        jw = j * w[:, None, None]
        h = np.einsum("nij,nik->jk", jw, j)
        g = np.einsum("nij,ni->j", jw, err)
        step = np.linalg.solve(h + damping * np.diag(np.maximum(np.diag(h), 1e-9)) + 1e-12 * np.eye(6), -g)
        trial = np.array(m)
        trial[:3, :3] = m[:3, :3] @ rotvec_to_matrix(step[:3])
        trial[:3, 3] = m[:3, 3] + m[:3, :3] @ step[3:]
        uv, _ = pixels(points, trial, focal_px, principal)
        residual = np.linalg.norm(uv - seen, axis=1)
        if np.median(residual) < np.median(best):
            m, best, damping = trial, residual, max(damping * 0.3, 1e-9)
            if np.linalg.norm(step[3:]) < 1e-7 and np.linalg.norm(step[:3]) < 1e-9:
                break
        else:
            damping *= 10.0
            if damping > 1e9:
                break
    return m, best


def looks_like_a_zoom(focal: np.ndarray) -> bool:
    """Whether a focal curve is a real zoom rather than a wobble: a big change end to end, travelled smoothly."""
    focal = np.asarray(focal, np.float64).reshape(-1)
    if len(focal) < 3:
        return False
    middle = float(np.median(focal))
    straight = abs(float(focal[-1] - focal[0]))
    travelled = float(np.abs(np.diff(focal)).sum())
    return middle > 0 and straight / middle > ZOOM_SHARE and travelled <= ZOOM_WANDER * straight


# ------------------------------------------------------------------ the anchors a shot can offer


def cloud_frames(points, frames: list[int]) -> dict[int, np.ndarray]:
    """A point cloud packet's world points at each of `frames` (cm). A cloud that does not change is read once and
    handed back under every frame — the same array, not a copy."""
    from pxr import Usd, UsdGeom

    from ...data.evaluate import cloud_at
    from ...data.scene import open_scene

    stage = open_scene([points])  # kept alive while its prims are read
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Points)]
    own = [int(f) for f in points.meta.get("frames", [])]
    out: dict[int, np.ndarray] = {}
    still = len(own) < 2
    for f in (own[:1] if still else frames):
        got = [cloud_at(prim, Usd.TimeCode(f))[0] for prim in prims]
        got = [g for g in got if len(g)]
        out[f] = np.concatenate(got) if got else np.zeros((0, 3))
    if still:
        only = out[own[0]] if own else np.zeros((0, 3))
        return dict.fromkeys(frames, only)
    return out


def anchors_from_cloud(cloud: np.ndarray, pose: np.ndarray, focal_px: float, principal: np.ndarray,
                       width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """The points of `cloud` this camera sees, spread evenly over the picture, and where the camera puts them (the
    observations a solver would have had): world points [K,3] and pixels [K,2]."""
    if len(cloud) > MOST_CONSIDERED:
        cloud = cloud[:: int(np.ceil(len(cloud) / MOST_CONSIDERED))]
    uv, depth = pixels(cloud, pose, focal_px, principal)
    inside = (depth > NEAR_CM) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not inside.any():
        return np.zeros((0, 3)), np.zeros((0, 2))
    keep = np.flatnonzero(inside)[_even_over_the_picture(uv[inside], width, height)]
    return cloud[keep], uv[keep]


def tracked_anchors(points, tracks, frames: list[int], width: int, height: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Real evidence, where the shot has it: the 3D tracked points of `points` against the pixels 「2D 跟踪点」 measured
    for them (the two outputs of one 3D tracker, point for point). {} when they do not line up point for point."""
    from ...data.payloads import read_tracks

    t = read_tracks(tracks)
    xy, visible = np.asarray(t["tracks"], np.float64), np.asarray(t["visible"], bool)
    columns = {int(f): i for i, f in enumerate(tracks.meta["frames"])}
    scale = np.array([width / float(tracks.meta.get("width") or width), height / float(tracks.meta.get("height") or height)])
    clouds = cloud_frames(points, frames)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for f in frames:
        column, cloud = columns.get(f), clouds.get(f)
        if column is None or cloud is None or len(cloud) != len(xy):
            return {}
        seen = visible[:, column]
        out[f] = (cloud[seen], xy[seen, column] * scale)
    return out


def lock_focal(camera, points, tracks, out, focal_px: float, info: dict | None = None):
    """The camera with its focal length held at `focal_px` (pixels at the plate's width) and every frame's pose solved
    again so the plate still lines up. `points`: a point cloud to take anchors from (None: none); `tracks`: the 2D
    tracks measured for those points (None: the anchors are put through the original camera instead).

    Returns (packet, report); the report is plain numbers the node turns into its messages and its curves."""
    from ...data.camera import CameraSamples
    from ...data.units import DEFAULT_HEIGHT, DEFAULT_WIDTH, focal_mm

    frames = [int(f) for f in camera.meta["frames"]]
    w, h = int(camera.meta.get("width") or DEFAULT_WIDTH), int(camera.meta.get("height") or DEFAULT_HEIGHT)
    s = CameraSamples.from_packet(camera, frames)
    before = s.poses().copy()
    was = s.focal_px(w)
    mm_x = s.filmback_mm / w
    principal = np.array([w / 2 + float(np.median(s.center_mm[:, 0])) / mm_x,
                          h / 2 - float(np.median(s.center_mm[:, 1])) / (mm_x / (s.pixel_aspect or 1.0))])

    measured = tracked_anchors(points, tracks, frames, w, h) if (points is not None and tracks is not None) else {}
    clouds = {} if measured else (cloud_frames(points, frames) if points is not None else {})
    poses, residual, counts = before.copy(), np.zeros(len(frames)), np.zeros(len(frames), int)
    for i, f in enumerate(frames):
        if measured:
            anchor, seen = measured[f]
        else:
            anchor, seen = anchors_from_cloud(clouds.get(f, np.zeros((0, 3))), before[i], was[i], principal, w, h)
        counts[i] = len(anchor)
        if not len(anchor):
            continue
        depth = -((anchor - before[i, :3, 3]) @ before[i, :3, :3])[:, 2]
        # the closed form: keep the orientation, slide along the lens axis (exact for a plane square to the lens)
        start = np.array(before[i])
        start[:3, 3] = before[i, :3, 3] + float(np.median(depth)) * (1.0 - focal_px / was[i]) * -before[i, :3, 2]
        if len(anchor) < FEW_ANCHORS:
            poses[i] = start
            uv, _ = pixels(anchor, start, focal_px, principal)
            residual[i] = float(np.median(np.linalg.norm(uv - seen, axis=1)))
            continue
        poses[i], left = solve_pose(anchor, seen, start, focal_px, principal)
        residual[i] = float(np.median(left))

    along = np.einsum("fi,fi->f", poses[:, :3, 3] - before[:, :3, 3], -before[:, :3, 2])
    was_mm = focal_mm(was, s.filmback_mm, w)
    report = {
        "focal_mm_before": was_mm,
        "focal_mm_after": float(focal_mm(focal_px, s.filmback_mm, w)),
        "along_cm": along,
        "residual_px": residual,
        "moved_cm": float(np.linalg.norm(poses[:, :3, 3] - before[:, :3, 3], axis=1).max(initial=0.0)),
        "residual_median": float(np.median(residual)) if len(residual) else 0.0,
        "residual_max": float(residual.max(initial=0.0)),
        "worst_frame": int(frames[int(np.argmax(residual))]) if len(residual) else 0,
        "few": [int(frames[i]) for i in np.flatnonzero((counts > 0) & (counts < FEW_ANCHORS))],
        "empty": [int(frames[i]) for i in np.flatnonzero(counts == 0)],
        "anchors": int(np.median(counts)) if len(counts) else 0,
        "measured": int(np.median([len(a) for a, _ in measured.values()])) if measured else 0,
        "zoomlike": looks_like_a_zoom(was_mm),
        "range_mm": (float(np.min(was_mm)), float(np.max(was_mm))),
    }
    keep = {k: v for k, v in camera.meta.items() if k not in ("frames", "width", "height")}
    solved = CameraSamples.solved(frames, w, h, focal_px, poses, filmback_mm=s.filmback_mm,
                                  principal_px=principal, lens=s.lens,
                                  overscan=s.overscan, info={"focal_locked": info or {}})
    return solved.write(out, **keep), report
