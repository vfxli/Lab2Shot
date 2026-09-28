"""Pi3 (π³) worker: feed-forward multi-view reconstruction. Runs inside
third_party/pi3/.venv with the pinned repo on PYTHONPATH; never imports Lab2Shot core.
The driver (frames, chunks, alignment, outputs) is lab2shot_worker.feedforward.

    python worker.py <job.json>

job["node"] == "pi3.reconstruct"; job["params"]:

    model       "pi3x" (Pi3X, default: smoother, approximate metric scale) | "pi3" (original π³,
                arbitrary scale). Both CC BY-NC 4.0 (non-commercial)
    max_frames  frames per forward pass (the node offers 50 / 100 / 150). Null: what fits a 24 GB
                GPU at the input size (260k 14x14 patches: 200 frames at 672x378); longer
                shots are split into overlapping chunks joined by a similarity transform
    step        use every Nth frame (default 1; the last frame is always used)
    resolution  long side of the model input, rounded to a multiple of 14. Null: Pi3's own
                rule, the input aspect within 255,000 pixels (e.g. 672x378 for 16:9)

Output: see lab2shot_worker.feedforward (frame_<n>.npz: depth, confidence, mask, points). Pi3X
depth / translations are approximately metres (its metric head); original Pi3 is scale-free.
`points` is Pi3's own local_points (pi3.py:212), the camera-space form of its world points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fit_size, hf_dest, recon, require_weights, resident, serve
from lab2shot_worker.feedforward import Backend, load_state, pixel_budget_size, run
from pi3_models import MODELS

PIXEL_LIMIT = 255_000  # upstream load_images_as_tensor / load_multimodal_data
CONF_MIN = 0.1  # sigmoid(conf) threshold of upstream's examples
# 14x14 patches per forward pass, keeping the GPU under ~20 GB of the 24 GB card (above that
# WSL spills into system RAM). RTX 4090 at 672x378: Pi3X with 16-frame conv-head steps takes
# 12.8 GB allocated / 14.4 GB reserved at 168k patches (130 frames), ~0.036 GB per 1000 patches;
# Pi3 15.4 / 18.8 GB at 259k (200 frames).
PATCH_BUDGET = 260_000
HEAD_CHUNK = 16  # Pi3X conv-head frames per step (upstream 64)


def input_size(width: int, height: int, resolution: int | None) -> tuple[int, int, int, int]:
    if resolution:
        w, h = fit_size(width, height, round(resolution / 14) * 14, 14)
    else:
        w, h = pixel_budget_size(width, height, PIXEL_LIMIT)
    return w, h, 0, 0


def focal_from_points(local_points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame fx, fy (pixels) from camera-space points, principal point at the image centre:
    least squares of (u - cx) = fx * X/Z over the confident pixels (pixel centres at j + 0.5)."""
    n, h, w, _ = local_points.shape
    z = local_points[..., 2].clamp_min(1e-6)
    xp, yp = local_points[..., 0] / z, local_points[..., 1] / z
    u = (torch.arange(w, device=z.device, dtype=z.dtype) + 0.5 - w / 2).view(1, 1, w)
    v = (torch.arange(h, device=z.device, dtype=z.dtype) + 0.5 - h / 2).view(1, h, 1)
    wgt = valid.to(z.dtype)
    wgt = torch.where(wgt.flatten(1).sum(1).view(n, 1, 1) > 100, wgt, torch.ones_like(wgt))  # nothing confident: use all
    fx = (wgt * xp * u).flatten(1).sum(1) / (wgt * xp * xp).flatten(1).sum(1).clamp_min(1e-12)
    fy = (wgt * yp * v).flatten(1).sum(1) / (wgt * yp * yp).flatten(1).sum(1).clamp_min(1e-12)
    return fx, fy


@resident
def load_model(variant: str, path: Path):
    if variant == "pi3x":
        from pi3.models.pi3x import Pi3X

        model = Pi3X(use_multimodal=False)  # no pose / intrinsics / depth conditioning branch
        # Its convolutional heads run 64 frames at a time (several GB at 672x378): 16 keeps the
        # peak down with the same result.
        conv_head = model._chunked_conv_head
        model._chunked_conv_head = lambda head, feat, ph, pw, chunk_size=HEAD_CHUNK: conv_head(head, feat, ph, pw, chunk_size)
    else:
        from pi3.models.pi3 import Pi3

        model = Pi3()
    load_state(model, path, ("",))
    return model.cuda().eval()


def make_backend(job) -> Backend:
    variant = job.params["model"]
    repo, _rev, filename, _size, _sha = MODELS[variant]
    path = job.weights_dir / hf_dest(repo, filename)
    require_weights("pi3", path)
    model = load_model(variant, path)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    @torch.no_grad()
    def infer(images: torch.Tensor, w: int, h: int, pad_w: int, pad_h: int) -> recon.Chunk:
        with torch.autocast("cuda", dtype=dtype):
            out = model(images[None])
        local = out["local_points"][0].float()
        conf = torch.sigmoid(out["conf"][0, ..., 0].float())
        poses = out["camera_poses"][0].double()  # camera-to-world, OpenCV
        del out
        valid = conf > CONF_MIN
        fx, fy = focal_from_points(local, valid)
        K = torch.zeros(len(local), 3, 3, dtype=torch.float64)
        K[:, 0, 0], K[:, 1, 1] = fx.double().cpu(), fy.double().cpu()
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = w / 2, h / 2, 1.0
        # Upstream's world points are camera_poses @ local_points (pi3.py:211), so local_points itself (the
        # camera-space point map, pi3.py:212) is handed over: recon.Stitcher scales it with the depth
        # (Chunk.scaled) and the node places it in the world by camera. The same data as the official points,
        # without each chunk's own poses applied first.
        return recon.Chunk(
            cam_to_world=poses.cpu().numpy(),
            K=K.numpy(),
            depth=local[..., 2].cpu().numpy().astype(np.float32),
            confidence=conf.cpu().numpy(),
            usable=valid.cpu().numpy(),
            extra={"points": local.cpu().numpy().astype(np.float32)},
            scaled=("points",),
        )

    repo = MODELS[variant][0]
    return Backend(
        name=repo.split("/")[1],
        weights=variant,
        metric=variant == "pi3x",
        patch_budget=PATCH_BUDGET,
        input_size=input_size,
        infer=infer,
        confidence=f"sigmoid of Pi3's confidence logit (0-1, higher = more reliable); mask needs > {CONF_MIN}",
        points="local_points",  # Pi3's own point map; its world points are camera_poses @ local_points (pi3.py:211)
        models=[repo],
        licence="CC BY-NC 4.0 (non-commercial)",
    )


def main(job_path: str) -> None:
    run(job_path, "pi3.reconstruct", make_backend)


if __name__ == "__main__":
    serve(main)
