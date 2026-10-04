"""Camera ensemble kernel: cameras of several solvers of one shot, already in one world, combined into one.

The node (nodes/core/ensemble_camera.py: ensemble_camera) reads the candidates' camera packets and writes what these
functions return; the smoke evaluation (集成评测/camera) calls the very same functions. numpy only, pure functions.

Cameras arrive in one world (the card moved each into the base camera's with camera_space), OpenCV axes here (x right,
y down, z forward; the node turns Lab2Shot's GL cameras before calling): cam_to_world [C,F,4,4] for C candidates over F
common frames, intrinsics K [C,F,3,3] in pixels of the picture.

The judge is independent of every candidate: 2D tracks of the shot (CoTracker's, wired in), held out the way a
matchmover reads a solve's deviation. Each track's views are split in two alternating halves; the point is
triangulated from one half through the candidate's cameras and reprojected into the other half's frames, and the
other way round. A camera that fits the picture puts every view of a point on one ray bundle; a camera that drifts,
jumps or turned a pan into a dolly does not, and no candidate can lower the error by fitting the noise of the very
observations it is measured on. The error is in pixels and does not change under a similarity of the world, so it
says nothing of scale.
"""

from __future__ import annotations

import numpy as np

from .ensemble import geometric_median, rotation_median

CAMERA_OBJECTIVES = ("accuracy", "robust", "smooth")

TRACK_CAP = 50.0  # px: one held-out observation counts at most this much (a point behind a camera, a mistracked one)
TRACK_MIN_VIEWS = 3  # each half of a track needs this many views to triangulate from / to be measured on
TRACK_MIN_PER_FRAME = 8  # fewer measured tracks than this in a frame: the frame has no judge value


def _normalised_world(c2w: np.ndarray) -> np.ndarray:
    """The same cameras with the world moved and scaled so their centres have mean 0 and RMS spread 1 (a spread of 0
    keeps the scale): the triangulation is then equally conditioned in every unit."""
    c = c2w[:, :3, 3]
    mid = c.mean(0)
    spread = float(np.sqrt(((c - mid) ** 2).sum(1).mean())) or 1.0
    out = np.array(c2w, np.float64)
    out[:, :3, 3] = (c - mid) / spread
    return out


def _projections(c2w: np.ndarray) -> np.ndarray:
    """World-to-camera [F,3,4] of cam_to_world [F,4,4]."""
    w2c = np.linalg.inv(c2w)
    return w2c[:, :3, :4]


def triangulate(P: np.ndarray, xn: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Homogeneous points [N,4] from normalised image positions xn [N,F,2] (K⁻¹ applied) seen through world-to-camera
    P [F,3,4] in the frames `use` [N,F] marks: the linear (DLT) solution, each view adding x·P₃ − P₁ and y·P₃ − P₂;
    the eigenvector of the smallest eigenvalue of their normal matrix. Homogeneous, so a point at infinity (a pure
    pan) is found too."""
    a1 = xn[..., 0:1] * P[None, :, 2, :] - P[None, :, 0, :]
    a2 = xn[..., 1:2] * P[None, :, 2, :] - P[None, :, 1, :]
    w = use.astype(np.float64)
    M = np.einsum("nf,nfa,nfb->nab", w, a1, a1) + np.einsum("nf,nfa,nfb->nab", w, a2, a2)
    _, vec = np.linalg.eigh(M)
    return vec[:, :, 0]


def held_out_errors(c2w: np.ndarray, K: np.ndarray, xy: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """The judge for one candidate: held-out reprojection error [N,F] in pixels (NaN where not measured) of tracks xy
    [N,F,2] (pixels, a pixel's centre at +0.5) visible [N,F], through its cameras c2w [F,4,4] (OpenCV axes) and
    intrinsics K [F,3,3]. Each track's views in frame order go alternately to half A and half B; a point triangulated
    from A is measured in B's frames and the other way round. Tracks with fewer than TRACK_MIN_VIEWS views in either
    half are not measured. Each error is capped at TRACK_CAP."""
    c2w = _normalised_world(np.asarray(c2w, np.float64))
    P = _projections(c2w)
    vis = np.asarray(visible, bool) & np.isfinite(xy).all(-1)
    Kinv = np.linalg.inv(K)
    h = np.concatenate([np.nan_to_num(xy), np.ones(xy.shape[:2] + (1,))], -1)
    xn = np.einsum("fij,nfj->nfi", Kinv, h)[..., :2]
    rank = np.cumsum(vis, 1) - 1
    half_a = vis & (rank % 2 == 0)
    half_b = vis & (rank % 2 == 1)
    ok = (half_a.sum(1) >= TRACK_MIN_VIEWS) & (half_b.sum(1) >= TRACK_MIN_VIEWS)
    err = np.full(vis.shape, np.nan)
    for fit, test in ((half_a, half_b), (half_b, half_a)):
        X = triangulate(P, xn[ok], fit[ok])  # [n,4]
        cam = np.einsum("fij,nj->nfi", P, X)  # [n,F,3] homogeneous
        sign = np.sign(X[:, 3:4])  # the point's own side: X and −X are the same homogeneous point
        z = cam[..., 2] * sign
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.einsum("fij,nfj->nfi", K, cam / cam[..., 2:3])[..., :2]
        e = np.minimum(np.linalg.norm(uv - xy[ok], axis=-1), TRACK_CAP)
        e = np.where(z > 0, np.nan_to_num(e, nan=TRACK_CAP), TRACK_CAP)
        sub = err[ok]
        sub[test[ok]] = e[test[ok]]
        err[ok] = sub
    return err


def frame_errors(err: np.ndarray, least: int = TRACK_MIN_PER_FRAME) -> np.ndarray:
    """Per frame [F]: the median held-out error over the tracks measured there; NaN with fewer than `least`."""
    n = np.isfinite(err).sum(0)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(np.isfinite(err), err, np.nan), 0) if err.size else np.zeros(err.shape[1])
    return np.where(n >= least, med, np.nan)


def median_poses(c2w: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per frame, the rotation median and the geometric median of the centres of the candidates [C,F,4,4] with weight
    w [C,F] > 0 there."""
    C, F = c2w.shape[:2]
    out = np.repeat(np.eye(4)[None], F, 0)
    for f in range(F):
        k = w[:, f] > 0
        if not k.any():
            k = np.ones(C, bool)
        out[f, :3, :3] = rotation_median(c2w[k, f, :3, :3], w[k, f] if (w[k, f] > 0).all() else None)
        out[f, :3, 3] = geometric_median(c2w[k, f, :3, 3], w[k, f] if (w[k, f] > 0).all() else None)
    return out


def window_scores(err: np.ndarray, window: int) -> np.ndarray:
    """Per candidate and frame [C,F]: the median of its per-frame judge errors over the `window` frames centred
    there (NaN frames skipped)."""
    C, F = err.shape
    half = window // 2
    out = np.full((C, F), np.nan)
    for f in range(F):
        a, b = max(0, f - half), min(F, f + half + 1)
        blk = err[:, a:b]
        with np.errstate(invalid="ignore"):
            out[:, f] = np.where(np.isfinite(blk).any(1), np.nanmedian(np.where(np.isfinite(blk), blk, np.nan), 1), np.nan)
    return out


def segment_choice(scores: np.ndarray, allowed: np.ndarray, keep: float = 1.0) -> np.ndarray:
    """Per frame, the allowed candidate [F] with the lowest windowed score, changing only when another one is better
    by more than `keep` (a factor ≥ 1: hysteresis, so the result does not flicker between near equals). −1 where no
    candidate has a score."""
    C, F = scores.shape
    s = np.where(allowed[:, None] & np.isfinite(scores), scores, np.inf)
    out = np.full(F, -1)
    cur = -1
    for f in range(F):
        best = int(np.argmin(s[:, f]))
        if not np.isfinite(s[best, f]):
            out[f] = cur
            continue
        if cur < 0 or not np.isfinite(s[cur, f]) or s[best, f] * keep < s[cur, f]:
            cur = best
        out[f] = cur
    if (out < 0).any() and (out >= 0).any():  # frames before the first scored one take that one
        out[out < 0] = out[out >= 0][0]
    return out


def stitch(c2w: np.ndarray, choice: np.ndarray, blend: int) -> np.ndarray:
    """Per frame the chosen candidate's pose [F,4,4], each switch cross-faded over `blend` frames on either side
    (rotation along the shortest arc, centre linearly) so the camera does not jump where the choice changes."""
    from lab2shot_shared.poses import blend_poses

    F = c2w.shape[1]
    out = np.stack([c2w[max(c, 0), f] for f, c in enumerate(choice)])
    if blend <= 0:
        return out
    switches = [f for f in range(1, F) if choice[f] != choice[f - 1]]
    for s in switches:
        a, b = choice[s - 1], choice[s]
        for f in range(max(0, s - blend), min(F, s + blend)):
            t = (f - (s - blend) + 0.5) / (2 * blend)
            out[f] = blend_poses(c2w[a, f], c2w[b, f], float(np.clip(t, 0.0, 1.0)))
    return out


GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # cam_to_world @ this: Lab2Shot's GL camera axes -> OpenCV's (same world)


def camera_K(cam, width: int) -> np.ndarray:
    """Per frame [F,3,3] intrinsics in pixels of a picture `width` wide (data/camera.py CameraSamples)."""
    n = max(len(cam.frames), 1)
    f = np.broadcast_to(np.asarray(cam.focal_xy_px(width), np.float64).reshape(-1, 2), (n, 2))
    pp = np.broadcast_to(np.asarray(cam.principal_px(width), np.float64).reshape(-1, 2), (n, 2))
    K = np.repeat(np.eye(3)[None], n, 0)
    K[:, 0, 0], K[:, 1, 1], K[:, :2, 2] = f[:, 0], f[:, 1], pp
    return K


def whole_error(judge: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """Per candidate [C]: the median of its per-frame judge errors over the shot, divided by its prior weight (a
    weight of 2 halves it); inf where it has none, or a weight of 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.array([np.nanmedian(j) if np.isfinite(j).any() else np.inf for j in judge])
        if prior is not None:
            e = np.where(np.asarray(prior) > 0, e / np.maximum(np.asarray(prior, np.float64), 1e-12), np.inf)
    return e


CAMERA_CONF_PX = 1.0  # held-out error (px) at which a frame's confidence is 0.5


def camera_confidence(err: np.ndarray) -> np.ndarray:
    """0..1 per frame from the result's own held-out error: 1/(1 + e/CAMERA_CONF_PX); 0 where unmeasured."""
    return np.where(np.isfinite(err), 1.0 / (1.0 + np.nan_to_num(err) / CAMERA_CONF_PX), 0.0)


CAMERA_WINDOW = 30  # frames a stretch of the accuracy objective is judged on
CAMERA_KEEP = 1.2  # another candidate takes over only when its error is this many times lower
CAMERA_BLEND = 5  # frames cross-faded either side of a switch
CAMERA_GATE = 2.0  # a candidate whose whole-shot error is over this many times the best one's is not used


def allowed(whole: np.ndarray, gate: float = CAMERA_GATE) -> np.ndarray:
    """Per candidate [C]: whether the tracks let it be used at all (its whole-shot error within `gate` times the best
    one's). Every candidate when none has a judge value."""
    finite = np.isfinite(whole)
    if not finite.any():
        return np.ones(len(whole), bool)
    return finite & (whole <= gate * np.min(whole[finite]))


def spread_cm(c2w: np.ndarray) -> float:
    """RMS distance of the camera centres [F,4,4] from their mean, in the cameras' unit (cm on the card)."""
    c = np.asarray(c2w, np.float64)[:, :3, 3]
    return float(np.sqrt(((c - c.mean(0)) ** 2).sum(1).mean())) if len(c) else 0.0


def still_base(c2w: np.ndarray, fits: list, least: float) -> int | None:
    """The base camera's index when it barely moves, else None. c2w [C,F,4,4] of the candidates in the base camera's
    world, fits [C] the camera_space fit each came through (`aligned.fit`; the base is the one handed through as its
    own target, "same"; else the first). Under a centre spread of `least` (camera_space's MOVED_CM) camera_space read
    no scale for the others (W-CAMSPACE-STILL): their size cannot be checked against the base."""
    if len(c2w) < 2:
        return None
    base = fits.index("same") if "same" in fits else 0
    return base if spread_cm(c2w[base]) < least else None


def combine(c2w: np.ndarray, judge: np.ndarray, prior: np.ndarray, objective: str,
            only: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The combined cameras [F,4,4] of candidates c2w [C,F,4,4] (one world, any axes: only poses are mixed), with their
    per-frame judge errors judge [C,F] (NaN: unmeasured) and prior weights prior [C]. Returns (result, choice [F]: the
    candidate each frame's pose (and lens) comes from, ok [C]: the candidates used).

    - accuracy: per stretch of CAMERA_WINDOW frames the allowed candidate the tracks fit best, a switch only when
      another is CAMERA_KEEP times better, switches cross-faded over CAMERA_BLEND frames each side;
    - robust: the candidate the tracks fit best over the whole shot, unchanged;
    - smooth: per frame the rotation median and the geometric median of the centres of the allowed candidates; each
      frame's lens from the allowed candidate nearest to that centre.
    Without any judge value every objective is the median of all candidates (weighted by prior).
    `only`: that candidate unchanged, whatever the objective (the base camera when the others' scale to it could not
    be checked: the judge does not see scale, so switching to another candidate could change the world's size)."""
    C, F = c2w.shape[:2]
    if only is not None:
        return c2w[only].copy(), np.full(F, int(only)), np.arange(C) == int(only)
    prior = np.asarray(prior, np.float64)
    whole = whole_error(judge, prior)
    has_judge = bool(np.isfinite(whole).any())
    ok = allowed(whole) & (prior > 0) if has_judge else prior > 0
    if not ok.any():
        ok = np.ones(C, bool)
    if has_judge and objective == "accuracy":
        choice = segment_choice(window_scores(judge, CAMERA_WINDOW), ok, CAMERA_KEEP)
        if (choice < 0).all():
            choice[:] = int(np.argmin(np.where(ok, whole, np.inf)))
        result = stitch(c2w, choice, CAMERA_BLEND)
        return result, choice, np.isin(np.arange(C), choice)
    if has_judge and objective == "robust":
        best = int(np.argmin(np.where(ok, whole, np.inf)))
        choice = np.full(F, best)
        return c2w[best].copy(), choice, np.arange(C) == best
    w = np.repeat((ok * np.maximum(prior, 1e-6))[:, None], F, 1)
    result = median_poses(c2w, w)
    near = np.linalg.norm(c2w[:, :, :3, 3] - result[None, :, :3, 3], axis=-1)
    choice = np.argmin(np.where(ok[:, None], near, np.inf), 0)
    return result, choice, ok
