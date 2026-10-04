"""The single-image geometry family (MoGe, UniDepth, UniK3D, Depth Anything 3, FaceAnything): the per-frame loop, the geometry
arrays and result.json. Runs inside each extension's own environment (numpy, torch); never imports Lab2Shot core.

Per-frame geometry contract (node side: lab2shot/nodes/families/depth_camera.py
PerFrameDepthCamera), at the input resolution, raw/frame_<n>.npz:

    points      float32 [H,W,3]  camera space, OpenCV (+X right, +Y down, +Z forward), metres
    depth       float32 [H,W]    camera Z, metres
    mask        bool    [H,W]    valid pixels
    intrinsics  float64 [3,3]    pixels at H,W, pixel-corner convention: pixel (i, j)
                                 covers [j, j+1] x [i, i+1], so (W/2, H/2) is the image centre
    confidence  float32 [H,W]    the model's own confidence (larger = more reliable; not
                                 comparable between models); absent when the model has none
    normal      float32 [H,W,3]  camera space, OpenCV; only models that predict normals (MoGe)
    sky         float32 [H,W]    the model's own sky probability; only models that predict it (DA3)
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from lab2shot_shared.poses import pixel_rays

from . import Job, fail, read_frame, say, stub_module
from .frame_io import FrameReader, Writer
from .run import Run

CONVENTION = "OpenCV camera: +X right, +Y down, +Z forward; metres; pixel-corner convention (image centre = W/2, H/2)"


def begin(job: str | Job, node: str | tuple[str, ...], project: str, *, src: str = "") -> Run:
    """Every per-frame geometry worker's first step: Run.start (job file, node check, CUDA, bookkeeping), the pinned repo
    (or its `src` folder) on sys.path, wandb stubbed (UniDepth / UniK3D import it at module level in
    utils/visualization.py for training logs; it is never called at inference, so it is not installed)."""
    run = Run.start(job, node, project)
    repo = run.job.repo_dir / src if src else run.job.repo_dir
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    stub_module("wandb")
    return run


def pinhole_from_fov(fov_x_deg: float, width: int, height: int) -> np.ndarray:
    """Square pixels, principal point at the image centre."""
    f = width / (2 * math.tan(math.radians(fov_x_deg) / 2))
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=np.float64)


def ray_samples(points: torch.Tensor, step: int = 4):
    """Every `step`-th pixel of a point map [H,W,3] -> u, v (pixel centres) and unit ray x, y, z (float64)."""
    h, w = points.shape[:2]
    r = torch.nn.functional.normalize(points[step // 2::step, step // 2::step].double(), dim=-1)
    v, u = torch.meshgrid(torch.arange(h, device=r.device)[step // 2::step].double() + 0.5,
                          torch.arange(w, device=r.device)[step // 2::step].double() + 0.5, indexing="ij")
    return (u.flatten(), v.flatten(), *r.reshape(-1, 3).unbind(-1))


def pinhole_fit(u, v, x, y, z) -> dict:
    """Least-squares pinhole (fx, fy, cx, cy) of rays in front of the camera, with residuals in pixels."""
    ok = z > 0.05
    a, b = (x / z)[ok], (y / z)[ok]
    ones = torch.ones_like(a)
    (fx, cx), (fy, cy) = (torch.linalg.lstsq(torch.stack([s, ones], 1), t[ok][:, None]).solution[:, 0]
                          for s, t in ((a, u), (b, v)))
    res = torch.sqrt((fx * a + cx - u[ok]) ** 2 + (fy * b + cy - v[ok]) ** 2)
    return {"fx": fx.item(), "fy": fy.item(), "cx": cx.item(), "cy": cy.item(),
            "rms_px": res.pow(2).mean().sqrt().item(), "max_px": res.max().item()}


def k_matrix(fit: dict) -> np.ndarray:
    return np.array([[fit["fx"], 0, fit["cx"]], [0, fit["fy"], fit["cy"]], [0, 0, 1]], dtype=np.float64)


def camera_for_network(k: np.ndarray, size: tuple[int, int], model, get_paddings, get_resize_factor) -> np.ndarray:
    """UniDepth / UniK3D (same code) pad the image to their aspect range and resize it to multiples of 14
    (slightly different x and y scales), but scale a given camera by one factor: pre-correct K so the
    camera the network sees matches the resized image, and the output rays match `k` at the input size."""
    h, w = size
    bounds = model.shape_constraints
    lo, hi = bounds["pixels_min"], bounds["pixels_max"]
    step = (hi - lo) / 10
    level = model.resolution_level
    (pad_l, _, pad_t, _), (ph, pw) = get_paddings((h, w), bounds["ratio_bounds"])
    scale, (nh, nw) = get_resize_factor((ph, pw), (lo + level * step, lo + (level + 1) * step))
    gx, gy = nw / pw / scale, nh / ph / scale  # true scale / the one upstream applies
    out = k.copy()
    out[0, 0], out[1, 1] = k[0, 0] * gx, k[1, 1] * gy
    out[0, 2], out[1, 2] = (k[0, 2] + pad_l) * gx - pad_l, (k[1, 2] + pad_t) * gy - pad_t
    return out


def load_network(model_dir: Path, device, module: str, class_name: str, project: str, extension: str):
    """UniDepth / UniK3D (two repositories by the same authors) loading their weights: the network built from
    `<model folder>/config.json`, filled from `model.safetensors`, stopping when keys are missing
    (`E-WORKER-WEIGHTSMISMATCH`, one code for the family, the project a parameter), then moved to the GPU in
    inference mode.

    The two workers differ only in class name, module path and project name, so the caller gives those.
    `@resident` stays on each worker's own one-line `load_model`: the admin page shows a resident loader's
    arguments, and the module name among them would make that line unreadable.

    `interpolation_mode = "bilinear"` and `.eval()` are as upstream does them: UniK3D uses the camera it is given
    only after eval().
    """
    from safetensors.torch import load_file

    config = json.loads((model_dir / "config.json").read_text())
    model = getattr(importlib.import_module(module), class_name)(config)
    missing, unexpected = model.load_state_dict(load_file(model_dir / "model.safetensors"), strict=False)
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project=project, extension=extension, model=model_dir.name,
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    model.interpolation_mode = "bilinear"
    return model.to(device).eval()


def network_camera(fov_x_deg: float | None, width: int, height: int, model, get_paddings, get_resize_factor,
                   pinhole_class, device):
    """The camera condition (`Pinhole`) the network takes when the field of view is known; None otherwise (the
    network estimates it). It is the second argument of UniDepth / UniK3D's `infer(image, camera)`.
    """
    if not fov_x_deg:
        return None
    k_net = camera_for_network(pinhole_from_fov(fov_x_deg, width, height), (height, width),
                               model, get_paddings, get_resize_factor)
    return pinhole_class(K=torch.from_numpy(k_net).float()[None].to(device))


def infer_frame(rgb: np.ndarray, device, model, fov_x_deg: float | None, get_paddings, get_resize_factor,
                pinhole_class):
    """One frame through UniDepth / UniK3D's network: the picture onto the GPU, the camera condition when the field
    of view is known, `infer`; returns the network's output as it is and the point map ([H,W,3], camera space,
    metres)."""
    image = torch.from_numpy(rgb).to(device).permute(2, 0, 1)
    h, w = rgb.shape[:2]
    camera = network_camera(fov_x_deg, w, h, model, get_paddings, get_resize_factor, pinhole_class, device)
    out = model.infer(image, camera)  # fp16 autocast inside
    return out, out["points"][0].permute(1, 2, 0).float()


def unproject_frame(depth: torch.Tensor, k: np.ndarray) -> torch.Tensor:
    """depth [H,W] (camera Z) + pinhole K (pixel-corner convention) -> points [H,W,3] on depth's device: every pixel's
    ray (lab2shot_shared.poses.pixel_rays) scaled by its depth."""
    h, w = depth.shape
    x, y = pixel_rays(k, np.arange(h, dtype=np.float64)[:, None], np.arange(w, dtype=np.float64)[None, :])
    x = torch.as_tensor(x, dtype=torch.float32, device=depth.device).expand(h, w)
    y = torch.as_tensor(y, dtype=torch.float32, device=depth.device).expand(h, w)
    return torch.stack([x * depth, y * depth, depth], dim=-1)


def frame_arrays(points: torch.Tensor, mask: torch.Tensor, intrinsics: np.ndarray,
                 confidence: torch.Tensor | None = None, normal: torch.Tensor | None = None,
                 sky: torch.Tensor | None = None) -> dict:
    """Torch outputs at the input resolution -> the npz arrays of the contract."""
    points = torch.nan_to_num(points.float(), nan=0.0, posinf=0.0, neginf=0.0)
    depth = points[..., 2]
    arrays = {
        "points": points.cpu().numpy().astype(np.float32),
        "depth": depth.cpu().numpy().astype(np.float32),
        "mask": (mask & (depth > 0)).cpu().numpy().astype(bool),
        "intrinsics": np.asarray(intrinsics, dtype=np.float64),
    }
    if confidence is not None:
        arrays["confidence"] = torch.nan_to_num(confidence.float()).cpu().numpy().astype(np.float32)
    if normal is not None:
        arrays["normal"] = torch.nan_to_num(normal.float()).cpu().numpy().astype(np.float32)
    if sky is not None:  # the model's own sky probability, one of its official outputs, passed on as it is
        arrays["sky"] = torch.nan_to_num(sky.float()).cpu().numpy().astype(np.float32)
    return arrays


def run_frames(run: Run, infer: Callable[[np.ndarray, int], dict], word: str = "estimate_geometry", dtype: str = "uint8") -> dict:
    """Read every frame (prefetched), run `infer(rgb [H,W,3] of `dtype`, frame) -> frame_arrays(...)` (an extra
    "_stats" dict collects per-frame numbers), write raw/frame_<n>.npz. Timing and progress are the Run's; returns the
    per-shot statistics finish_geometry writes (with the frame numbers under "frames")."""
    frames = run.frames()
    raw = run.job.raw_dir
    stats: dict[str, list] = {"focal_px": [], "depth_median": []}
    run.stage("estimate_geometry")
    # bounded writes: never hold a whole shot of arrays in RAM
    with FrameReader(frames.paths, lambda path: read_frame(path, dtype), threads=1, ahead=1) as reader, \
            Writer(threads=2, max_pending=4) as writer:
        for i, (frame, _path) in run.each(frames.pairs, word):
            rgb = reader.get(i)
            with run.frame():
                arrays = infer(rgb, frame)
            for key, value in arrays.pop("_stats", {}).items():
                stats.setdefault(key, []).append(value)
            valid = arrays["depth"][arrays["mask"]]
            stats["focal_px"].append(float(arrays["intrinsics"][0, 0]))
            stats["depth_median"].append(float(np.median(valid)) if valid.size else float("nan"))
            writer.npz(raw / f"frame_{frame}.npz", **arrays)
    stats["frames"] = frames.numbers
    return stats


def finish_geometry(run: Run, stats: dict, *, model: str, fov_x: float | None, metric: bool = True,
                    files: str = "frame_<n>.npz: points, depth, mask, intrinsics, confidence", **info) -> None:
    """result.json for a geometry job (kind "geometry"): the family's words, the per-shot statistics, the worker's own
    fields, and the Run's standard timing and memory fields."""
    focal = np.asarray(stats["focal_px"])
    frames = list(stats.pop("frames"))
    if fov_x is None and len(focal) > 1 and (spread := focal.std() / np.median(focal)) > 0.05:
        say("W-MONOGEO-FOCALSPREAD", model=model, low=float(focal.min()), high=float(focal.max()), spread=float(spread))
    run.finish(
        frames,
        kind="geometry",
        model=model,
        metric=metric,
        units="metres" if metric else "relative (scale-invariant)",
        convention=CONVENTION,
        files=files,
        principal_point="pixel-corner convention (image centre = W/2, H/2)",
        fov_x_deg=fov_x,
        fov_source="user" if fov_x else "model",
        width=run.job.width or None,
        height=run.job.height or None,
        frames=frames,  # the whole list, not the standard [first, last]: the converter reads every frame's file by it
        focal_px_median=float(np.median(focal)) if len(focal) else None,
        **stats,
        **info,
    )
