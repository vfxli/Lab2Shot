"""Track4World worker: dense 4D tracking. Runs inside third_party/track4world/.venv with the pinned repo on
PYTHONPATH (its `track4world` package; the authors' utils3d fork from vendor/); never imports Lab2Shot core.

    python worker.py <job.json>

Every pixel (every `track_step`-th) of the reference frame is followed through the shot in 3D with the Depth Anything 3
backbone (its metric scale on): upstream's "3d_ff" mode (demo.py forward_video3d_ff), which tracks from the first
frame of the clip it is given. Upstream runs one clip; here a shot is cut into clips of at most `max_frames` frames:

    - forwards from the reference frame, then backwards over the reversed frames before it;
    - consecutive clips share `OVERLAP` frames; each point is handed over at the next clip's first frame (where the
      earlier clip put it: the next clip's maps there are sampled at that position). A point hidden at a hand-over
      is not followed further (it keeps its last position, marked not seen);
    - each clip solves its own world and scale (Depth Anything 3 per clip): it is brought into the first clip's world
      by a similarity on the frames they share (rotation from the shared cameras, scale and shift from the 3D points
      of the same pixels, lab2shot_worker.recon.fit_similarity).

The world is always the method's own (OpenCV axes, metres; result.json world "own"), with its own cameras: upstream
solves them itself (demo.py output[0]['camera_poses']) and takes no camera in.

Output: the 3D point-track contract (lab2shot_worker.point_tracks); the 2D tracks are the model's own 2D tracks; and,
per frame, raw/frame_<n>.npz with the rest of upstream's same return (demo.py:519-520): world_points [h,w,3] (metres,
that world: world_points = c2w @ points, nets/model.py:193) and mask [h,w] (where it has usable geometry), plus the
rgb [h,w,3] uint8 the points are coloured with, and valid: that mask back in the plate's proportions (float32).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import lab2shot_worker.point_tracks as pt
from lab2shot_worker import MemoryBound, fail, progress, read_frame, resident, save_npz, serve, stub_module
from lab2shot_worker.recon import fit_similarity, mean_rotation, save_cameras
from lab2shot_worker.run import Run

CHECKPOINT = "Track4World/track4world_da3.pth"
BACKBONE = "da3nested-giant-large-1.1"
CONFIG = "track4world/config/eval/v1.json"
ITERS = 4  # upstream demo --inference_iters
VISIBLE = 0.5  # upstream evaluation: visibility x confidence > 0.5
OVERLAP = 8  # frames consecutive clips share
MIN_CLIP = 16  # the model's window
# Minimum short side after resizing: the correlation pyramid has 5 levels, each halving the 1/8 feature map, and the
# last level must keep at least 2 -> 1/8 map >= 32 -> short side >= 256
# (repo/track4world/nets/model.py:602 corr_levels=5, nets/blocks.py:150 scale_factor=0.5)
MIN_SIDE = 256
# the parameter memory grows with: 每段最多帧数, stepping down 120, 64, 32, 16 (120 frames: 18.7 GB at 512 px)
MAX_FRAMES = MemoryBound.parameter("max_frames", (120, 64, 32, MIN_CLIP))


def stubs() -> None:
    """Modules upstream imports but never uses on this path: Pi3 (not in the repo: a sparse checkout upstream), Depth
    Anything 3's exporters (moviepy, pycolmap, gsplat ...) and pose alignment to given cameras (evo)."""
    stub_module("track4world.nets.external.pi3")
    stub_module("track4world.nets.external.pi3.models")
    stub_module("track4world.nets.external.pi3.models.pi3", Pi3=None)
    stub_module("depth_anything_3.utils.export", export=None)
    stub_module("depth_anything_3.utils.pose_align", align_poses_umeyama=None, batch_align_poses_umeyama=None)


@resident
def load_model(repo: Path, checkpoint: Path, backbone: Path, device):
    import json

    vendor = os.environ["LAB2SHOT_VENDOR_DIR"]
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    stubs()
    import torchvision.models as tvm
    import track4world.nets.model as tm

    # while the model is built: the backbone from the installed files (upstream: from the Hugging Face hub), and the
    # flow encoder without torchvision's ImageNet ConvNeXt (a download; the checkpoint below holds all its weights)
    hub = tm.DepthAnything3.from_pretrained.__func__
    convnext = tvm.convnext_tiny
    tm.DepthAnything3.from_pretrained = classmethod(lambda cls, _repo, **kw: hub(cls, str(backbone), **kw))
    tvm.convnext_tiny = lambda weights=None, **kw: convnext(weights=None, **kw)
    try:
        config = json.loads((repo / CONFIG).read_text())
        model = tm.Track4World(**config["model"], seqlen=16, use_3d=True, use_model="depthanythingv3")
    finally:
        tm.DepthAnything3.from_pretrained = classmethod(hub)
        tvm.convnext_tiny = convnext
    missing, _ = model.load_pretrained_with_remap(torch.load(checkpoint, map_location="cpu"))
    if any(not k.startswith("backbone.model.da3_metric.") for k in missing):  # the metric branch keeps the backbone's
        fail("E-TRACK4WORLD-WEIGHTS", count=len(missing), example=str(missing[0]))
    model.use_metric_scale = True
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


def run_clip(model, frames: np.ndarray, device) -> dict:
    """One clip (uint8 [T, h, w, 3], tracked from its first frame) -> numpy: flow2d [T,h,w,2] pixels, xyz [T,h,w,3]
    (where each first-frame pixel is, in camera t, metres), vis [T,h,w], points [T,h,w,3] (camera t's own points),
    mask [T,h,w], c2w [T,4,4] (the clip's world), focal (pixels at h x w)."""
    rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        with torch.autocast(device_type="cuda", dtype=torch.float32):  # as upstream's demo (forward_video3d_ff)
            out, _ = model.infer(rgbs, iters=ITERS, sw=None, is_training=False, tracking3d=True)
    geo, motion = out
    h, w = frames.shape[1:3]
    vis = motion["visconf_maps_e"][0].float()
    ks = geo["intrinsics"][0].float().cpu().numpy()  # [T, 3, 3], normalised (principal point 0.5, 0.5)
    k = ks[0]
    xyz = motion["flow_3d"][0].float().cpu().numpy()
    # Upstream unprojects the tracked 2D positions as x / w (nets/model.py:2399-2405, flow2d_c: pixel centres at
    # integers), while its own points and intrinsics put pixel x at (x + 0.5) / w (utils3d image_uv, depth_to_points):
    # the 3D tracks sit half a model pixel up-left of the 2D tracks and of the dense points (measured: 3D tracks
    # projected through the delivered camera land 0.5 x (plate / model) pixels short on both axes, dense points on the
    # pixel centres). Moved to where upstream's own convention puts them: (x + 0.5) / w at the same depth, i.e. in
    # camera space + z * 0.5 / (fx_n * w) sideways and + z * 0.5 / (fy_n * h) down (fx_n, fy_n normalised focals)
    kt = ks[np.minimum(np.arange(len(xyz)), len(ks) - 1)]
    xyz[..., 0] += xyz[..., 2] * (0.5 / (kt[:, 0, 0] * w))[:, None, None]
    xyz[..., 1] += xyz[..., 2] * (0.5 / (kt[:, 1, 1] * h))[:, None, None]
    return {
        "flow2d": motion["flow_2d"][0].float().permute(0, 2, 3, 1).cpu().numpy(),
        "xyz": xyz,
        "vis": (vis[:, 0] * vis[:, 1]).cpu().numpy(),
        "points": geo["points"][0].float().cpu().numpy(),
        "mask": geo["mask"][0].cpu().numpy(),
        "c2w": geo["camera_poses"].float().cpu().numpy().astype(np.float64),
        "focal": float(0.5 * (k[0, 0] * w + k[1, 1] * h)),
    }


def sample(maps: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear samples of maps [h, w, C] at pixels xy [N, 2] (the model's pixels, centres at integers)."""
    h, w = maps.shape[:2]
    x = np.clip(xy[:, 0], 0, w - 1.001)
    y = np.clip(xy[:, 1], 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    m = maps.reshape(h, w, -1)
    v = (m[y0, x0] * (1 - fx) * (1 - fy) + m[y0, x0 + 1] * fx * (1 - fy) + m[y0 + 1, x0] * (1 - fx) * fy
         + m[y0 + 1, x0 + 1] * fx * fy)
    return v.reshape((len(xy),) + maps.shape[2:])


def to_world(c2w: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Camera-space points [..., 3] of one camera -> its world."""
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def to_camera(c2w: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """World points [..., 3] -> that camera's space (to_world undone; c2w's rotation has no scale)."""
    return (pts - c2w[:3, 3]) @ c2w[:3, :3]


def crossfade(c2w: np.ndarray, before: np.ndarray, cam: np.ndarray, t: float) -> np.ndarray:
    """World points [N, 3]: `before` (world) -> `cam` (camera c2w's space) at t (0..1), mixed in that camera so their
    projection moves linearly with t as the 2D tracks do (depth and x/z, y/z each linear; a straight line in 3D would
    project unevenly when the depths differ). A point not finite on either side (upstream marks lost points inf) is
    mixed in the world as it is, so it stays inf."""
    after = to_world(c2w, cam)
    out = (1 - t) * before + t * after
    ok = np.isfinite(before).all(-1) & np.isfinite(after).all(-1)
    a, b = to_camera(c2w, before[ok]), cam[ok]
    z = (1 - t) * a[:, 2] + t * b[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = ((1 - t) * a[:, :2] / a[:, 2:] + t * b[:, :2] / b[:, 2:]) * z[:, None]
    good = np.isfinite(xy).all(-1)
    mixed = out[ok]
    mixed[good] = to_world(c2w, np.c_[xy[good], z[good]])
    out[ok] = mixed
    return out


def similarity_to(world: dict, clip: dict, idx: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) taking this clip's world into the result's, from the frames both have (their cameras' rotation, the
    3D points of the same pixels for scale and shift)."""
    shared = [j for j, g in enumerate(idx) if g in world]
    if not shared:
        return 1.0, np.eye(3), np.zeros(3)
    rel = [world[idx[j]]["c2w"][:3, :3] @ clip["c2w"][j][:3, :3].T for j in shared]
    rot = mean_rotation(np.stack(rel))
    src, dst, depth = [], [], []
    rng = np.random.default_rng(0)
    for j in shared:
        ok = clip["mask"][j] & world[idx[j]]["mask"]
        rows, cols = np.nonzero(ok)
        pick = rng.choice(len(rows), min(len(rows), 40_000 // len(shared)), replace=False) if len(rows) else []
        rows, cols = rows[pick], cols[pick]
        src.append(to_world(clip["c2w"][j], clip["points"][j][rows, cols]))
        dst.append(to_world(world[idx[j]]["c2w"], world[idx[j]]["points"][rows, cols]))
        depth.append(world[idx[j]]["points"][rows, cols, 2])
    s, R, t, _ = fit_similarity(np.concatenate(src), np.concatenate(dst), np.concatenate(depth), R=rot)
    return s, R, t


def clips(n: int, clip_len: int) -> list[tuple[int, int]]:
    """[start, end) of the clips along n frames: at most clip_len each, consecutive ones sharing OVERLAP frames (more
    when the last would be shorter than the model's window)."""
    out, start = [], 0
    while True:
        end = min(n, start + clip_len)
        out.append((start, end))
        if end == n:
            return out
        start = min(end - OVERLAP, max(0, n - max(MIN_CLIP, OVERLAP + 1)))


def chain(model, video: np.ndarray, order: list[int], seeds: np.ndarray, world: dict, clip_len: int, device,
          tick) -> tuple[dict, dict, dict, dict]:
    """Track `seeds` (model pixels on frame order[0]) along `order` (frame indices) clip by clip. Returns per frame
    index: xy [N,2] (model pixels), xyz [N,3] (the result's world, metres), vis [N], score [N] (visibility x
    confidence, 0..1: what vis thresholds; 0 once a point is lost); `world` gains every frame's
    camera and points (the first clip of the whole result sets its world). Over the frames two clips share, the
    earlier clip's positions cross-fade into the later one's."""
    xy, xyz, vis, score = {}, {}, {}, {}
    pos, alive = seeds.astype(np.float64), np.ones(len(seeds), bool)
    prev_end = 0
    plan = clips(len(order), clip_len)
    for k, (start, end) in enumerate(plan):
        idx = order[start:end]
        clip = run_clip(model, video[idx], device)
        s, R, t = similarity_to(world, clip, idx)
        for j, g in enumerate(idx):
            if g not in world:
                c2w = clip["c2w"][j].copy()
                c2w[:3, :3] = R @ c2w[:3, :3]
                c2w[:3, 3] = s * R @ c2w[:3, 3] + t
                world[g] = {"c2w": c2w, "points": clip["points"][j] * s, "mask": clip["mask"][j], "focal": clip["focal"]}
        shared = prev_end - start  # frames the previous clip tracked too
        for j, g in enumerate(idx):
            p2 = sample(clip["flow2d"][j], pos)
            # the 3D tracks go into the world through the camera delivered for frame g (world[g]), so that they
            # project through it onto the 2D tracks. On this clip's own frames that camera is this clip's, taken into
            # the result's world: the same as s * R @ (this clip's world) + t. On the frames the previous clip had
            # (the overlap) it is that clip's camera: this clip's points are placed in it where this clip saw them
            # (its camera-space points, with its focal instead of that camera's). Taken through this clip's own camera
            # there instead, where the fitted (s, R, t) is off by a little, the cross-faded 3D tracks stopped landing on
            # the 2D tracks (measured 640x480 at 640: |error| 95% 11 px on the overlap frames, 0.16 px elsewhere)
            cam = sample(clip["xyz"][j], pos) * s
            if clip["focal"] != world[g]["focal"]:
                cam[:, :2] *= clip["focal"] / world[g]["focal"]
            sure = sample(clip["vis"][j][..., None], pos)[:, 0] * alive
            v = sure > VISIBLE
            if j < shared:
                a = (j + 1) / (shared + 1)
                p2, v = (1 - a) * xy[g] + a * p2, vis[g] if a < 0.5 else v
                p3 = crossfade(world[g]["c2w"], xyz[g], cam, a)
                sure = (1 - a) * score[g] + a * sure
            else:
                p3 = to_world(world[g]["c2w"], cam)
            xy[g], xyz[g], vis[g], score[g] = p2, p3, v, sure
        tick(len(idx) - max(0, shared))
        if k + 1 < len(plan):  # handed over on the next clip's first frame: where this clip put them there
            nxt = order[plan[k + 1][0]]
            pos, alive = xy[nxt], alive & vis[nxt]
        prev_end = end
    return xy, xyz, vis, score


def main(job_path: str) -> None:
    run = Run.start(job_path, "track4world.track", "Track4World")
    job = run.job
    params = job.params
    frames = run.frames()
    numbers = frames.numbers
    ref = numbers.index(params["query_frame"]) if params["query_frame"] is not None else 0
    checkpoint, backbone = job.weights_dir / CHECKPOINT, job.weights_dir / BACKBONE
    run.weights(checkpoint, backbone / "model.safetensors")
    width, height = job.width, job.height
    # As upstream (`repo/demo.py:952-958`): scale uniformly to the long side, then floor both sides to multiples of 64.
    #   scale = min(size/H, size/W);  H, W = int(H*scale), int(W*scale);  H, W = (H//64)*64, (W//64)*64
    # Multiples of 64 are required: otherwise the model's InputPadder pads (1920x1080 at 512 as 512x288 would be padded
    # to 512x320), while `infer` rebuilds the intrinsics from the unpadded aspect ratio. The mismatch silently shrinks
    # the Focal Length and point cloud (-2.7% at 512, -4.5% at 384, -8.2% at 256); with `force_projection` (default True)
    # the points are unprojected with the wrong intrinsics, stretching x/y relative to z. Upstream always feeds
    # multiples of 64, so its padding is always 0 and this case does not occur there.
    _scale = params["resolution"] / max(width, height)
    w, h = max(64, (int(width * _scale) // 64) * 64), max(64, (int(height * _scale) // 64) * 64)
    # Flooring each side on its own squeezes the picture (640x480 at 640 is fed as 640x448, 1920x1080 as 640x320):
    # the plate's proportions at the model's scale, before that flooring, are what the per-frame maps go back to
    # (the core then brings them to the plate's size, and refuses any other proportions: E-FAMILY-ASPECT)
    plate_w, plate_h = round(width * _scale), round(height * _scale)
    # A short side that is too small after resizing makes the model fail, so it is rejected here first.
    # The correlation pyramid has 5 levels (`repo/track4world/nets/model.py:602` corr_levels=5), each halving the
    # 1/8-resolution feature map (`nets/blocks.py:150`, `F.interpolate(fmap2, scale_factor=0.5)`).
    # After 4 halvings at least 2 must remain, so the 1/8 map must be at least 32 and the processed short side at least 256.
    # Example: 1280x534 at the 512 setting becomes 512x192, 1/8 = 24 -> 24, 12, 6, 3, 1; level 5 gets (H:1, W:4),
    # halving again gives 0 and upstream raises "Input and output sizes should be greater than 0, but got input
    # (H: 1, W: 4) output (H: 0, W: 2)". 1920x1080 at the same setting is 512x256 (1/8 = 32), exactly sufficient;
    # a wider aspect ratio fails.
    if min(w, h) < MIN_SIDE:
        fail("E-TRACK4WORLD-SMALLSIDE", width=width, height=height, side=params["resolution"], w=w, h=h, need=MIN_SIDE)

    run.stage("读取画面")
    video = np.stack([cv2.resize(read_frame(p), (w, h), interpolation=cv2.INTER_AREA if w < width else cv2.INTER_LINEAR)
                      for _, p in frames])
    track_step = params["track_step"]
    gy, gx = np.mgrid[track_step // 2:h:track_step, track_step // 2:w:track_step]
    grid = np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float64)  # model pixels, centres at integers
    # the viewer's points on the reference frame first (plate pixels, centres at +0.5 -> the model's)
    user = pt.build_queries(job, pt.TrackParams(0, ref, None, None), width, height) if "points" in job.inputs else None
    clicked = (np.stack([user.xy[:, 0] * w / width - 0.5, user.xy[:, 1] * h / height - 0.5], -1)
               if user is not None else np.zeros((0, 2)))
    seeds = np.concatenate([clicked, grid])  # the grid is every track_step-th pixel of the frame: never empty

    device = torch.device("cuda")
    model = run.model("Track4World（Depth Anything 3 骨干）", load_model, job.repo_dir, checkpoint, backbone, device)

    n, clip_len = len(frames), min(params["max_frames"], len(frames))
    total, done = n, 0

    def tick(k):
        nonlocal done
        done = min(total, done + k)
        progress(done, total, "跟踪")

    run.stage("跟踪整帧的点" + (f"（每段 {clip_len} 帧）" if clip_len < n else ""))
    t2 = time.time()
    # forwards from the reference frame, then the frames before it over the reversed shot
    orders = [o for o in (list(range(ref, n)), list(range(ref, -1, -1))) if len(o) > 1]

    def track(clip_len: int):
        """The whole shot in clips of `clip_len` frames."""
        nonlocal done
        done = 0
        world: dict = {}
        xy, xyz, vis, score = {}, {}, {}, {}
        for order in orders:
            a, b, c, d = chain(model, video, order, seeds, world, clip_len, device, tick)
            for g in order:
                if g not in xy:
                    xy[g], xyz[g], vis[g], score[g] = a[g], b[g], c[g], d[g]
        return clip_len, world, xy, xyz, vis, score

    clip_len, world, xy, xyz, vis, score = run.fit(MAX_FRAMES, track, clip_len)
    track_s = time.time() - t2
    run.frame_seconds.extend([track_s / n] * n)  # the tracking is one pass over the shot: shared out per frame
    # grid points only where the model has reliable geometry on the reference frame (no sky); clicked ones all
    rows = np.clip(np.round(seeds[:, 1]).astype(int), 0, h - 1)
    cols = np.clip(np.round(seeds[:, 0]).astype(int), 0, w - 1)
    keep = world[ref]["mask"][rows, cols] | (np.arange(len(seeds)) < len(clicked))
    seeds = seeds[keep]
    sx, sy = width / w, height / h
    tracks = np.stack([xy[g][keep] for g in range(n)], 1)
    tracks = np.stack([(tracks[..., 0] + 0.5) * sx, (tracks[..., 1] + 0.5) * sy], -1).astype(np.float32)
    points = np.stack([xyz[g][keep] for g in range(n)], 1)  # [N, F, 3] in the first clip's world, metres
    seen = np.stack([vis[g][keep] for g in range(n)], 1)
    seen[:, ref] = True
    confidence = np.clip(np.stack([score[g][keep] for g in range(n)], 1), 0.0, 1.0).astype(np.float32)
    confidence[:, ref] = 1.0  # where each point was placed
    c2w = np.stack([world[g]["c2w"] for g in range(n)])
    # the model's pixels are square (its intrinsics: fx_n * w == fy_n * h); flooring both sides to 64 separately
    # stretched the plate by sx across and sy down, so the plate's focal is focal * sx across and focal * sy down
    # (and the principal point, the model's centre, is the plate's centre on each axis). Equal when nothing was
    # squeezed; the core camera keeps both (kit/cameras.py solved_camera fy_px)
    focal = np.array([world[g]["focal"] for g in range(n)])

    queries = pt.Queries(tracks[:, ref].copy(), np.full(len(seeds), ref, np.int64), np.arange(len(seeds)) < len(clicked))
    pt.save_tracks3d(job.raw_dir, points, tracks, seen, confidence, queries, numbers)
    K = np.zeros((n, 3, 3))
    K[:, 0, 0], K[:, 1, 1] = focal * sx, focal * sy
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = width / 2, height / 2, 1.0
    save_cameras(job.raw_dir, numbers, K, c2w, width, height)
    # the rest of the same upstream return (demo.py:519-520): dense per-frame world points and the validity mask,
    # delivered unchanged (world_points = c2w @ points, nets/model.py:193). rgb is the point colour: the frame fed to the model.
    # These three stay on the model's grid (the point cloud is sampled there; it gives world positions only). valid is
    # the same mask as a picture of the plate: unsqueezed to the plate's proportions (plate_w x plate_h above),
    # bilinear as the core resizes masks (kit/maps.py LINEAR); already in proportion, it is the mask itself
    run.stage("写出稠密点和有效遮罩")
    for g, _ in run.each(range(n), "写出稠密点"):
        world_pts = to_world(c2w[g], np.asarray(world[g]["points"], np.float64))
        mask = np.asarray(world[g]["mask"], bool)
        valid = mask.astype(np.float32)
        if (plate_w, plate_h) != (w, h):
            valid = cv2.resize(valid, (plate_w, plate_h), interpolation=cv2.INTER_LINEAR)
        save_npz(job.raw_dir / f"frame_{numbers[g]}.npz", compression=1,
                 world_points=world_pts.astype(np.float32), mask=mask, rgb=video[g], valid=valid)
    run.finish(
        numbers,
        kind="point_tracks_3d",
        model="Track4World (Depth Anything 3)",
        world="own",
        metric=True,
        processing_size=[w, h],
        track_step=track_step,
        clip_frames=clip_len,
        points=int(len(seeds)),
        visible_fraction=round(float(seen.mean()), 4),
        track_seconds=round(track_s, 1),
    )


if __name__ == "__main__":
    serve(main)
