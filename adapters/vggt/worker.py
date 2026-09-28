"""Meta VGGT worker: feed-forward multi-view reconstruction. Runs inside
third_party/vggt/.venv with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

job["node"] == "vggt.reconstruct"; job["params"]:

    model       "original" (VGGT-1B, CC BY-NC 4.0, default) | "commercial" (VGGT-1B-Commercial)
    max_frames  frames per forward pass (the node offers 32 / 64 / 130). Null: what fits a 24 GB
                GPU at the input size (180k 14x14 patches: 231 frames at 518x294, 131 at 518x518 =
                padded portrait); longer shots are split into overlapping chunks joined by a
                similarity transform
    step        use every Nth frame (default 1; the last frame is always used)
    resolution  long side of the model input, rounded to a multiple of 14 (default 518, VGGT's
                training size). Landscape: width = resolution. Portrait: height =
                resolution, padded white to a square (VGGT's own "pad" mode; it was trained on
                landscape / square images only)

Output: see lab2shot_worker.feedforward (cameras.npz, frame_<n>.npz with depth / confidence /
mask / points, result.json). VGGT's scale is arbitrary (its training normalises every scene),
not metres. `points` is VGGT's own world_points head (vggt.py:45), stored per frame in that
frame's camera.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fit_size, hf_dest, recon, require_weights, resident, serve
from lab2shot_worker.feedforward import Backend, load_state, run
from vggt_models import MODELS

NATIVE = 518
# depth_conf = 1 + exp(x) >= 1, and its scale changes a lot from shot to shot (median 1.2
# on the iPhone follow shot, 12 on the dancer). As in VGGT's own demos the least confident
# share of each pass is dropped (from the mask and the chunk alignment), plus the floor.
CONF_DROP_PERCENT = 30
CONF_FLOOR = 1.05
# 14x14 patches per forward pass, keeping the GPU under ~20 GB of the 24 GB card (above that
# WSL spills into system RAM). Measured on an RTX 4090: 150 frames at 518x518 (205k patches)
# peaked at 19.6 GB allocated / 21.6 GB reserved; memory grows ~0.064 GiB per 1000 patches.
PATCH_BUDGET = 180_000


def input_size(width: int, height: int, resolution: int | None) -> tuple[int, int, int, int]:
    long_side = round((resolution or NATIVE) / 14) * 14
    w, h = fit_size(width, height, long_side, 14)
    if h <= w:
        return w, h, 0, 0
    # Portrait: pad to a square, symmetrically (both sides are multiples of 14: even difference).
    return w, h, (long_side - w) // 2, 0


@resident
def load_model(path: Path):
    from vggt.models.vggt import VGGT

    # No track head: it only runs with query_points, which this node has no input for
    # (VGGT.forward(images, query_points=None), vggt.py:29). The point head is built:
    # world_points (vggt.py:45) is one of the official outputs, so the node has a port
    # for it.
    model = VGGT(enable_track=False, enable_point=True)
    load_state(model, path, ("aggregator.", "camera_head.", "depth_head.", "point_head."))
    return model.cuda().eval()


def make_backend(job) -> Backend:
    variant = job.params["model"]
    repo, _rev, filename, _size, _sha = MODELS[variant]
    path = job.weights_dir / hf_dest(repo, filename)
    require_weights("vggt", path)  # the gated commercial weights: the node says how to get them (run_worker)

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = load_model(path)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @torch.no_grad()
    def infer(images: torch.Tensor, w: int, h: int, pad_w: int, pad_h: int) -> recon.Chunk:
        H, W = images.shape[-2:]
        with torch.autocast("cuda", dtype=dtype):
            tokens, patch_start = model.aggregator(images[None])
        with torch.autocast("cuda", enabled=False):
            pose_enc = model.camera_head(tokens)[-1]
            depth, depth_conf = model.depth_head(tokens, images=images[None], patch_start_idx=patch_start)
            # upstream's second geometry head: per-pixel world points (vggt.py:45 world_points). Official outputs
            # must not be discarded
            world_points, _pts_conf = model.point_head(tokens, images=images[None], patch_start_idx=patch_start)
        del tokens
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))  # world-to-camera, OpenCV
        w2c = torch.eye(4, device=images.device, dtype=torch.float64).repeat(extrinsic.shape[1], 1, 1)
        w2c[:, :3, :] = extrinsic[0].double()
        K = intrinsic[0].double().cpu().numpy()
        K[:, 0, 2] -= pad_w  # principal point in the unpadded region
        K[:, 1, 2] -= pad_h
        sl = (slice(None), slice(pad_h, pad_h + h), slice(pad_w, pad_w + w))
        d = depth[0, ..., 0][sl].float().cpu().numpy()
        c = depth_conf[0][sl].float().cpu().numpy()
        # VGGT's world_points are in the chunk's own world (the first frame's camera). Each chunk has its own world
        # and is scaled as a whole when stitched, so the points are first moved into each frame's camera space with
        # upstream's own extrinsics (a lossless change of coordinates); recon.Stitcher then places them in the shot's
        # world together with the depth (Chunk.scaled), and the node maps them back to world by camera
        # (kit/maps.py family_points).
        pw = world_points[0][sl].float()
        R, t = extrinsic[0, :, :3, :3].float(), extrinsic[0, :, :3, 3].float()  # world -> camera
        pts_cam = torch.einsum("nij,nhwj->nhwi", R, pw) + t[:, None, None, :]
        del world_points, pw
        return recon.Chunk(
            cam_to_world=torch.linalg.inv(w2c).cpu().numpy(),
            K=K,
            depth=d,
            confidence=c,
            usable=c > max(CONF_FLOOR, float(np.percentile(c[:, ::4, ::4], CONF_DROP_PERCENT))),
            extra={"points": pts_cam.cpu().numpy().astype(np.float32)},
            scaled=("points",),
        )

    repo = MODELS[variant][0]
    return Backend(
        name=repo.split("/")[1],
        weights=variant,
        metric=False,
        patch_budget=PATCH_BUDGET,
        input_size=input_size,
        infer=infer,
        confidence=f"VGGT depth_conf = 1 + exp(x) (>= 1, higher = more reliable, scale differs per shot); "
                   f"mask drops the {CONF_DROP_PERCENT}% least confident pixels of each forward pass and conf <= {CONF_FLOOR}",
        points="world_points",  # VGGT's own point head (vggt.py:45), kept per frame in that frame's camera
        models=[repo],
        licence="CC BY-NC 4.0 (non-commercial)" if variant == "original" else "VGGT License (commercial use allowed, no military use)",
    )


def main(job_path: str) -> None:
    run(job_path, "vggt.reconstruct", make_backend)


if __name__ == "__main__":
    serve(main)
