"""TAPIP3D worker: 3D point tracking in a given depth and camera. Runs inside third_party/tapip3d/.venv with the pinned
TAPIP3D repo on sys.path (its `models`, `utils`, `datasets`, `third_party` packages) and its compiled pointops2;
never imports Lab2Shot core.

    python worker.py <job.json>

Inputs: the plate, job input "depth" (the node's depth maps, metres after files.read_depth) and "camera"
(recon.load_camera: per frame focal and OpenCV cam_to_world in the shot's world, metres); the query points as every
point tracker's (point_tracks.build_queries: grid on the query frame, the viewer's points).

TAPIP3D takes the depth and camera as they are and tracks in that camera's world ("raw" mode). Each query pixel is
lifted to 3D here, the way TAPIP3D's evaluation does (the depth at the pixel, holes filled from the nearest valid
pixel), at the plate's resolution. Upstream's preprocessing (inference.py prepare_inputs) runs as it is: the video
and depth resized to the model's size (depth resized validity-weighted), its flying-pixel filter, the intrinsics
rescaled. Upstream's inference() is followed here with the visibility probability kept (it keeps only a yes / no at
0.9), so the node's 可见门槛 can use it.

Output: the 3D point-track contract (point_tracks.save_tracks3d, world "camera"); the 2D tracks are the 3D points
projected through the given camera.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import lab2shot_worker.point_tracks as pt
from lab2shot_shared.poses import unproject_at
from lab2shot_worker import fail, read_depth, read_frame, resident, serve, stub_module
from lab2shot_worker.frame_io import thread_map
from lab2shot_worker.recon import load_camera
from lab2shot_worker.run import Run

CHECKPOINT = "tapip3d/tapip3d_final.pth"  # as the installer stores it (lab2shot_worker.hf_dest)
SUPPORT_GRID = 16  # upstream inference.py: a 16 x 16 support grid on the first frame, tracked along and dropped
VISIBLE = 0.9  # upstream: sigmoid(visibility logit) >= 0.9
ITERS = 6
# 处理分辨率 -> upstream's resolution_factor: the pixel count of its training size (384 x 512) times this
RESOLUTION_FACTOR = {"standard": 1, "fine": 2}


@resident
def load_model(repo: Path, checkpoint: Path, device):
    """upstream models.from_pretrained, without its encoder's ImageNet / CoTracker initialisation (it would download
    CoTracker3 from torch.hub; the checkpoint's weights replace it anyway)."""
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    stub_module("sophuspy")  # datasets.data_ops imports it for training-time augmentations only
    import models
    from omegaconf import open_dict

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    with open_dict(cfg):
        cfg["model"]["encoder"]["pretrained"] = False
    model = models.from_config(cfg["model"], image_size=cfg["train_dataset"]["resolution"])
    model.load_state_dict(ckpt["weight"], strict=True)
    model.set_eval_mode("raw")
    return model.eval().to(device)


def depth_at(depth: np.ndarray, valid: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Depth under pixels xy [N,2] (centres at +0.5), holes filled from the nearest valid pixel (TAPIP3D's lifting)."""
    from scipy import ndimage

    h, w = depth.shape
    c = np.clip(np.floor(xy[:, 0]).astype(int), 0, w - 1)
    r = np.clip(np.floor(xy[:, 1]).astype(int), 0, h - 1)
    if not valid[r, c].all():
        _, (ri, ci) = ndimage.distance_transform_edt(~valid, return_indices=True)
        r, c = ri[r, c], ci[r, c]
    return depth[r, c]


@torch.inference_mode()
def track(model, video, depths, intrinsics, extrinsics, queries):
    """upstream utils.inference_utils.inference, keeping the visibility probability: coords [T,N,3], vis [T,N]."""
    from einops import repeat
    from utils.inference_utils import _inference_with_grid

    valid = depths[depths > 0].reshape(-1)
    q25 = torch.kthvalue(valid, int(0.25 * len(valid))).values
    q75 = torch.kthvalue(valid, int(0.75 * len(valid))).values
    roi = torch.tensor([1e-7, (q75 + 1.5 * (q75 - q25)).item()], dtype=torch.float32, device=video.device)
    t, _, h, w = video.shape
    n = queries.shape[0]
    model.set_image_size((h, w))
    kw = dict(model=model, num_iters=ITERS, depth_roi=roi, grid_size=SUPPORT_GRID)
    preds, _ = _inference_with_grid(video=video[None], depths=depths[None], intrinsics=intrinsics[None],
                                    extrinsics=extrinsics[None], query_point=queries[None], **kw)
    coords, logits = preds.coords, preds.visibs
    if (queries[:, 0] > 0).any() and not model.bidirectional:
        back, _ = _inference_with_grid(video=video[None].flip(1), depths=depths[None].flip(1),
                                       intrinsics=intrinsics[None].flip(1), extrinsics=extrinsics[None].flip(1),
                                       query_point=torch.cat([t - 1 - queries[:, :1], queries[:, 1:]], -1)[None], **kw)
        before = repeat(torch.arange(t, device=video.device), "t -> 1 t n", n=n) < queries[None, None, :, 0]
        coords = torch.where(before[..., None], back.coords.flip(1), coords)
        logits = torch.where(before, back.visibs.flip(1), logits)
    return coords[0].float().cpu().numpy(), torch.sigmoid(logits[0].float()).cpu().numpy()


def main(job_path: str) -> None:
    run = Run.start(job_path, "tapip3d.track", "TAPIP3D")
    job = run.job
    frames = run.frames()
    numbers = frames.numbers
    params = job.params
    checkpoint = job.weights_dir / CHECKPOINT
    run.weights(checkpoint)
    cam = load_camera(job)
    depth_files = job.listing("depth")
    if cam is None or not depth_files:
        fail("E-TAPIP3D-NODEPTHCAMERA")
    missing = [f for f in numbers if f not in depth_files]
    if missing:
        fail("E-TAPIP3D-DEPTHFRAMES", count=len(missing), first=missing[0])
    width, height = job.width, job.height
    p = pt.parse_params({**params, "resolution": None}, numbers)
    queries = pt.build_queries(job, p, width, height)
    device = torch.device("cuda")

    run.stage("读取画面、深度和相机")
    focal, c2w = cam.at(numbers)
    video, depth, valid = [], [], []
    for i, (f, path) in run.each(frames.pairs, "读取"):
        video.append(read_frame(path))
        d, ok = read_depth(depth_files[f])
        if d.shape != (height, width):
            fail("E-TAPIP3D-DEPTHSIZE", frame=f, width=d.shape[1], height=d.shape[0], plate_width=width, plate_height=height)
        depth.append(d)
        valid.append(ok)
    # the query points lifted into the camera's world (metres)
    xyz_q = np.zeros((len(queries), 3))
    for t in np.unique(queries.t):
        sel = queries.t == t
        z = depth_at(depth[t], valid[t], queries.xy[sel])
        k = np.array([[focal[t], 0.0, width / 2], [0.0, focal[t], height / 2], [0.0, 0.0, 1.0]])
        xyz_q[sel] = unproject_at(z, queries.xy[sel, 0], queries.xy[sel, 1], k, c2w[t])  # the one back-projection

    model = run.model("TAPIP3D", load_model, job.repo_dir, checkpoint, device)

    run.stage("预处理（缩放、去深度边缘飞点）")
    from datasets.data_ops import _filter_one_depth
    from utils.inference_utils import resize_depth_bilinear

    factor = RESOLUTION_FACTOR[params["resolution"]]
    res = (int(model.image_size[0] * np.sqrt(factor)), int(model.image_size[1] * np.sqrt(factor)))
    # intrinsics in TAPIP3D's pixels (centres at integers), rescaled as upstream does (align-corners)
    K = np.zeros((len(frames), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 2, 2] = focal, focal, 1.0
    K[:, 0, 2], K[:, 1, 2] = width / 2 - 0.5, height / 2 - 0.5
    K[:, 0, :] *= (res[1] - 1) / (width - 1)
    K[:, 1, :] *= (res[0] - 1) / (height - 1)
    video = np.stack(thread_map(lambda im: cv2.resize(im, (res[1], res[0]), interpolation=cv2.INTER_LINEAR), video, threads=8))
    depth_in = np.stack(thread_map(lambda d: resize_depth_bilinear(d, (res[1], res[0])), depth, threads=8))
    depth_in = np.stack(thread_map(lambda a: _filter_one_depth(a[0], 0.08, 15, a[1]), zip(depth_in, K), threads=8))

    run.stage(f"跟踪 {len(queries)} 个点")
    t2 = time.time()
    to = dict(device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        coords, vis_p = track(
            model,
            torch.from_numpy(video).permute(0, 3, 1, 2).float().div(255.0).to(**to),
            torch.from_numpy(depth_in).float().to(**to),
            torch.from_numpy(K).float().to(**to),
            torch.from_numpy(np.linalg.inv(c2w)).float().to(**to),
            torch.from_numpy(np.c_[queries.t, xyz_q]).float().to(**to),
        )
    track_s = time.time() - t2
    run.frame_seconds.extend([track_s / len(frames)] * len(frames))  # one pass over the shot: shared out per frame
    xyz = np.transpose(coords.reshape(len(frames), len(queries), 3), (1, 0, 2))  # [N, F, 3]
    conf = np.transpose(vis_p.reshape(len(frames), len(queries)), (1, 0))
    uv = pt.project(xyz, focal, c2w, width, height)
    idx = np.arange(len(queries))
    uv[idx, queries.t] = queries.xy  # the query points themselves (lifted exactly from them)
    inside = (uv[..., 0] >= 0) & (uv[..., 0] <= width) & (uv[..., 1] >= 0) & (uv[..., 1] <= height)
    visible = (conf >= VISIBLE) & inside
    visible[idx, queries.t] = True
    pt.save_tracks3d(job.raw_dir, xyz, np.nan_to_num(uv, nan=-1.0), visible, conf, queries, numbers)
    run.finish(
        numbers,
        kind="point_tracks_3d",
        model="TAPIP3D",
        world="camera",
        metric=None,  # the depth's
        files="tracks3d.npz: xyz [N,F,3] metres in the input camera's world; tracks.npz: the 2D contract",
        processing_size=[res[1], res[0]],
        support_grid=SUPPORT_GRID,
        visible_rule=f"sigmoid(visibility) >= {VISIBLE} (upstream) and inside the picture",
        **pt.track_stats(uv, visible, width, height),
        track_seconds=round(track_s, 1),
    )


if __name__ == "__main__":
    serve(main)
