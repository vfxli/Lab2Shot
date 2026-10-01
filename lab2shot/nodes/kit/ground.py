"""地面和放平：场景里的地在哪个高度（网格最低点 / 点云最密的那一层），以及把世界转到 +Y 朝上的那个旋转。"""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom, UsdSkel

from ...errors import Invalid
from ...messages import Msg
from ...data.packet import Packet
from ...data.scene import camera_of, cameras_in, open_scene


def ground_height(src: Packet, percentile: float, turn: np.ndarray | None = None) -> float:
    """Floor height (cm): the lowest mesh point per frame (a skeleton without a mesh: its lowest joint, the feet), taken
    at a low percentile across frames; heights in the world turned by `turn` (3x3 about the origin) when given."""
    from ...data.evaluate import world_points

    up = (np.eye(3) if turn is None else np.asarray(turn))[1]  # a point's height after the turn: up . p
    stage = open_scene([src])
    skeletons = [UsdSkel.Skeleton(p) for p in stage.Traverse() if p.IsA(UsdSkel.Skeleton)]
    cache = UsdSkel.Cache()
    lows = []
    for f in [int(f) for f in src.meta["frames"]] or [0]:  # a still scene (no frames: a PLY) is read once
        t = Usd.TimeCode(f)
        pts = world_points(stage, t) or [
            np.array([np.array(m).T[:3, 3] for m in cache.GetSkelQuery(s).ComputeJointSkelTransforms(t)]) @
            np.array(UsdGeom.Xformable(s.GetPrim()).ComputeLocalToWorldTransform(t))[:3, :3] +
            np.array(UsdGeom.Xformable(s.GetPrim()).ComputeLocalToWorldTransform(t))[3, :3] for s in skeletons]
        if pts:
            lows.append(min(float((p @ up).min()) for p in pts))
    if not lows:
        raise Invalid(Msg("E-GROUND-NOMESHNOSKELETON"))
    return float(np.percentile(lows, percentile))


GROUND_FRAMES = 12  # frames of a per-frame point cloud looked at for the ground
GROUND_POINTS = 400_000
GROUND_BINS = 400  # height bins over the span of the points below the cameras


def points_ground(src: Packet, turn: np.ndarray | None = None) -> tuple[float, float]:
    """(floor height cm, share of the points below the cameras that lie on it) from the scene's point clouds, Y up (in
    the world turned by `turn` when given): the densest level among the points below the lowest camera (a floor, a
    road: a level surface gathers its points at one height once the scene is upright, walls and plants spread over
    many)."""
    from ...data.camera import CameraSamples
    from ...data.evaluate import scene_points

    up = (np.eye(3) if turn is None else np.asarray(turn))[1]
    stage = open_scene([src])
    frames = [int(f) for f in src.meta["frames"]] or [0]
    looked = frames[:: max(1, len(frames) // GROUND_FRAMES)]
    rng = np.random.default_rng(0)
    ys: list[np.ndarray] = []
    for f in looked:
        for _, pts, _, _ in scene_points(stage, Usd.TimeCode(f)):
            ys.append(pts[rng.choice(len(pts), min(len(pts), GROUND_POINTS // len(looked)), replace=False)] @ up)
    if not ys:
        raise Invalid(Msg("E-GROUND-NOCLOUD"))
    y = np.concatenate(ys)
    cams = [CameraSamples.from_prim(p, looked).cam_to_world[:, :3, 3] @ up for p in cameras_in(stage)]
    below = y[y < min(float(c.min()) for c in cams)] if cams else y
    if len(below) < 100:
        raise Invalid(Msg("E-GROUND-NOFLOORPOINTS", count=len(below)))
    lo, hi = np.percentile(below, [0.5, 99.5])
    if hi - lo < 1e-6:
        return float(lo), 1.0
    counts, edges = np.histogram(below, bins=GROUND_BINS, range=(lo, hi))
    counts = np.convolve(counts, [1, 2, 1], mode="same")
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
    near = np.abs(below - peak) <= 2 * (hi - lo) / GROUND_BINS
    return float(np.median(below[near])), float(near.mean())


def level_rotation(frames: list[int], up_camera: np.ndarray, uncertainty_deg: np.ndarray,
                   scene: Packet) -> tuple[np.ndarray, dict]:
    """The rotation (3x3, about the world origin) that turns the world so up is +Y, from each frame's up in its camera
    (a direction [F,3] in the camera space of each frame: X right, Y up, Z towards the viewer; how far off each may be,
    degrees) and the camera it was seen by (the connected one, else the scene's one camera): each frame's up taken into
    the world by that frame's camera, their robust weighted mean (1/sigma^2, frames more than 3x the median spread off
    dropped) turned onto +Y along the shortest arc (the heading is kept). Returns it and what it found."""
    from lab2shot_shared.poses import rotation_deg

    from .align import shortest_arc
    from ...data.camera import CameraSamples

    up_camera = np.asarray(up_camera, np.float64).reshape(-1, 3)
    length = np.linalg.norm(up_camera, axis=1, keepdims=True)
    if (length < 1e-6).any():
        raise Invalid(Msg("E-GRAVITY-ZERO"))
    up_camera, uncertainty_deg = up_camera / length, np.asarray(uncertainty_deg, np.float64)
    cam_frames = [int(f) for f in scene.meta["frames"]]
    if cam_frames:  # a camera holds between its first and last key
        keep = [i for i, f in enumerate(frames) if cam_frames[0] <= f <= cam_frames[-1]]
        if not keep:
            raise Invalid(Msg("E-GRAVITY-FRAMES", first=frames[0], last=frames[-1], camera_first=cam_frames[0], camera_last=cam_frames[-1]))
        frames, up_camera, uncertainty_deg = [frames[i] for i in keep], up_camera[keep], uncertainty_deg[keep]
    stage, prim = camera_of([scene])  # keep the stage alive while its prim is read
    rot = CameraSamples.from_prim(prim, frames).rotations()
    world = np.einsum("fij,fj->fi", rot, up_camera)
    weight = 1.0 / np.maximum(uncertainty_deg, 0.1) ** 2
    used = np.ones(len(world), bool)
    for _ in range(3):
        up = (weight[used, None] * world[used]).sum(0)
        up /= np.linalg.norm(up)
        off = np.degrees(np.arccos(np.clip(world @ up, -1.0, 1.0)))
        used = off <= max(3.0 * float(np.median(off[used])), 0.5)
    R = shortest_arc(up, (0.0, 1.0, 0.0))  # onto +Y the short way: the heading is kept
    return R, {"tilt_deg": float(rotation_deg(R)), "frames": len(frames), "dropped": int((~used).sum()),
               "spread_deg": float(np.median(off[used])), "world_up_before": up.tolist()}
