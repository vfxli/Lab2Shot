"""Depth Anything 3 worker. Runs inside third_party/depthanything3/.venv with the
pinned repo's src/ on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

job["node"] == "depthanything3.geometry": each frame on its own (monocular). Params:
    model       "da3nested-giant-large-1.1" (default) | "da3-large-1.1" | "da3-base" | "da3metric-large".
                The any-view network (Giant 1.1 / Large 1.1 / Base) gives depth + camera and
                DA3METRIC-LARGE the metric scale (upstream's "nested" scheme; the official Giant
                checkpoint bundles both). "da3metric-large" alone predicts no camera: needs fov_x_deg.
                Non-commercial (CC BY-NC 4.0): da3nested-giant-large-1.1, da3-large-1.1.
                Apache-2.0: da3-base, da3metric-large.
    fov_x_deg   known horizontal field of view (degrees) or null. DA3 cannot take a focal length
                alone as a condition (its camera encoder needs poses too): a given FOV replaces the
                predicted intrinsics and sets the metric scale (metric depth = focal x output / 300).
    resolution  long image side the network works at, 252-1512 px (default 504, what DA3 is
                trained at; rounded to a multiple of 14). Output is always the input size.
    ray_pose    default false: intrinsics from the camera-token head (upstream default, faster);
                true: from the ray head.
  -> raw/frame_<n>.npz per lab2shot_worker.mono_geometry (points, depth, mask, intrinsics,
     confidence; no normals); mask = not sky (DA3METRIC-LARGE's sky head) and depth > 0.

job["node"] == "depthanything3.reconstruct": the whole shot in multi-view passes (depth that
agrees across frames, a camera per frame, metric scale), through the shared feed-forward
reconstruction driver lab2shot_worker.feedforward (chunks with overlap for long shots,
raw/cameras.npz + frame_<n>.npz). Params: weights = "da3nested-giant-large-1.1"
(default) | "da3-large-1.1" | "da3-base", and the driver's max_frames / step / resolution
(long side, default 504). Cameras from DA3's ray head (upstream: more accurate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lab2shot_worker import check_node, fail, fit_size, load_job, recon, require_weights, resident, serve

from lab2shot_worker.feedforward import Backend, run
from lab2shot_worker.mono_geometry import begin, finish_geometry, frame_arrays, pinhole_from_fov, run_frames, unproject_frame

ANYVIEW = ("da3nested-giant-large-1.1", "da3-large-1.1", "da3-base")
NESTED = "da3nested-giant-large-1.1"
METRIC = "da3metric-large"
NON_COMMERCIAL = {"da3nested-giant-large-1.1", "da3-large-1.1"}
SKY_THRESHOLD = 0.3  # upstream compute_sky_mask
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DEFAULT_LONG_SIDE = 504  # upstream process_res
METRIC_CHUNK = 16  # views per DA3METRIC-LARGE step inside a multi-view pass
CONF_PERCENTILE = 10  # reconstruct: the least confident 10% of a pass are not "valid"
# 14x14 patches per reconstruction pass keeping the GPU under ~18 GB (RTX 4090, bf16, at
# 124 frames x 720 patches = 89k): Giant 1.1 + Metric-Large 19.4 GB allocated (6.2 GB weights +
# 0.136 GB per 1000 patches, nearly all in the Giant backbone), Large 1.1 + Metric 11.6 GB, Base 8.4 GB.
PATCH_BUDGET = {"da3nested-giant-large-1.1": 80_000, "da3-large-1.1": 150_000, "da3-base": 200_000}


def load_net(weights: Path, key: str, device: torch.device) -> nn.Module:
    """The network exactly as in the checkpoint's config.json, built and loaded straight on the GPU
    (the Giant checkpoint is 6.8 GB of fp32: building it in RAM first would double the worker's RAM)."""
    from omegaconf import OmegaConf
    from safetensors.torch import load_model

    from depth_anything_3.cfg import create_object

    folder = weights / key
    require_weights("depthanything3", folder / "model.safetensors", what=f"模型 {key} 的权重")
    config = json.loads((folder / "config.json").read_text())["config"]
    # The 3D-Gaussian head (Giant only) is not used and would import evo / gsplat: not built.
    for section in (config, config.get("anyview", {})):
        section.pop("gs_head", None)
        section.pop("gs_adapter", None)
    with device:
        net = create_object(OmegaConf.create(config))
    holder = nn.Module()  # the checkpoint's keys are "model.*" (upstream's DepthAnything3 wrapper)
    holder.model = net
    missing, unexpected = load_model(holder, str(folder / "model.safetensors"), strict=False, device=str(device))
    unexpected = [k for k in unexpected if ".gs_head." not in k and ".gs_adapter." not in k]
    if missing or unexpected:
        fail("E-WORKER-WEIGHTSMISMATCH", project="Depth Anything 3", extension="depthanything3", model=key,
             missing=len(missing), unexpected=len(unexpected), examples=[*missing[:3], *unexpected[:3]])
    return net.eval()


@resident
def build(weights: Path, model_id: str, device: torch.device) -> nn.Module:
    """A NestedDepthAnything3Net (any-view + metric), or the metric network alone."""
    from depth_anything_3.model.da3 import NestedDepthAnything3Net

    if model_id in (METRIC, NESTED):
        return load_net(weights, model_id, device)
    nested = NestedDepthAnything3Net.__new__(NestedDepthAnything3Net)  # same scheme, separate checkpoints
    nn.Module.__init__(nested)
    nested.da3 = load_net(weights, model_id, device)
    nested.da3_metric = load_net(weights, METRIC, device)
    return nested


def preprocess(images: list[np.ndarray], process_res: int) -> torch.Tensor:
    """Upstream's resize + normalisation -> [1, N, 3, h, w]."""
    from depth_anything_3.utils.io.input_processor import InputProcessor

    tensor, _, _ = InputProcessor()(images, None, None, process_res, "upper_bound_resize",
                                    num_workers=min(8, len(images)), sequential=len(images) == 1)
    return tensor[None].float()


def metric_chunked(metric_net, x: torch.Tensor, chunk: int = METRIC_CHUNK):
    """DA3METRIC-LARGE sees every view on its own (no cross-view attention): run it a few views at a
    time so a long multi-view pass does not hold its activations for all frames at once."""
    from addict import Dict

    if x.shape[1] <= chunk:
        return metric_net(x)
    parts = [metric_net(x[:, i:i + chunk]) for i in range(0, x.shape[1], chunk)]
    return Dict({k: torch.cat([p[k] for p in parts], dim=1) for k in ("depth", "sky")})


def run_nested(net, x: torch.Tensor, k_user: torch.Tensor | None, ray_pose: bool):
    """NestedDepthAnything3Net.forward, except a known focal replaces the predicted one
    before the metric scaling. Returns (output, sky probability, predicted intrinsics)."""
    out = net.da3(x, None, None, export_feat_layers=[], infer_gs=False, use_ray_pose=ray_pose,
                  # 上游的默认（`api.py:141 ref_view_strategy: str = "saddle_balanced"`）；单视图（S ≤ 2）
                  # 不重排（api.py:171-172）
                  ref_view_strategy="saddle_balanced")
    metric = metric_chunked(net.da3_metric, x)
    k_model = out.intrinsics.clone()
    if k_user is not None:
        out.intrinsics = k_user
    out = net._apply_metric_scaling(out, metric)
    out = net._apply_depth_alignment(out, metric)
    out = net._handle_sky_regions(out, metric)
    return out, metric.sky, k_model


def forward(net, x: torch.Tensor, k_user: torch.Tensor | None, ray_pose: bool):
    """-> depth [N,h,w] metres, confidence [N,h,w] or None, sky [N,h,w], K [N,3,3] (processed
    pixels, the one used), predicted K or None, world-to-camera [N,4,4] or None."""
    n, (h, w) = x.shape[1], x.shape[-2:]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        if hasattr(net, "da3_metric"):
            out, sky, k_model = run_nested(net, x, k_user, ray_pose)
            k, conf = out.intrinsics, out.depth_conf.reshape(n, h, w)
            w2c = out.extrinsics.reshape(n, -1, 4)
            if w2c.shape[1] == 3:
                w2c = torch.cat([w2c, torch.tensor([0, 0, 0, 1.0], device=w2c.device).expand(n, 1, 4)], 1)
            k_model = k_model.reshape(n, 3, 3)
        else:  # DA3METRIC-LARGE alone: canonical depth, metric = focal * output / 300
            out = net(x)
            k, conf, w2c, k_model, sky = k_user, None, None, None, out.sky
            focal = (k[..., 0, 0] + k[..., 1, 1]).reshape(n, 1, 1) / 2
            out.depth = out.depth.reshape(n, h, w) * focal / 300.0
    return (out.depth.reshape(n, h, w).float(), None if conf is None else conf.float(), sky.reshape(n, h, w).float(),
            k.reshape(n, 3, 3).double(), k_model, None if w2c is None else w2c.double())


def to_input(depth, conf, sky, k_proc: np.ndarray, size: tuple[int, int]):
    """Processed resolution -> input resolution (bilinear), intrinsics rescaled."""
    (hh, ww), (h, w) = size, depth.shape[-2:]
    up = lambda t: F.interpolate(t[None, None], size=(hh, ww), mode="bilinear", align_corners=False)[0, 0]  # noqa: E731
    k = k_proc.copy()
    k[0] *= ww / w
    k[1] *= hh / h
    depth = up(depth)
    sky_p = up(sky)  # 模型自己的天空概率：mask 用它判，原图也交出去
    return depth, None if conf is None else up(conf), sky_p < SKY_THRESHOLD, k, sky_p


def user_k(fov_x: float | None, n: int, size: tuple[int, int], proc: tuple[int, int], device) -> torch.Tensor | None:
    """The known lens as intrinsics at the processed resolution, [1, N, 3, 3]."""
    if not fov_x:
        return None
    (hh, ww), (h, w) = size, proc
    k = pinhole_from_fov(fov_x, ww, hh)
    k[0] *= w / ww
    k[1] *= h / hh
    return torch.from_numpy(k).float().to(device).expand(1, n, 3, 3).clone()


def geometry(run, net, model_id: str, fov_x, process_res: int, ray_pose: bool) -> None:
    device = torch.device("cuda")

    def infer(rgb: np.ndarray, frame: int) -> dict:
        x = preprocess([rgb], process_res).to(device)
        k_user = user_k(fov_x, 1, rgb.shape[:2], x.shape[-2:], device)
        depth, conf, sky, k, k_model, _ = forward(net, x, k_user, ray_pose)
        k_in = k[0].cpu().numpy()
        depth, conf, mask, k_in, sky_p = to_input(depth[0], None if conf is None else conf[0], sky[0], k_in, rgb.shape[:2])
        arrays = frame_arrays(unproject_frame(depth, k_in), mask, k_in, conf, sky=sky_p)
        if k_model is not None:
            fx = float(k_model[0, 0, 0]) * rgb.shape[1] / x.shape[-1]
            arrays["_stats"] = {"focal_px_model": fx}
        return arrays

    stats = run_frames(run, infer)
    finish_geometry(run, stats, model=model_id, fov_x=fov_x, **info(model_id, process_res, ray_pose))


def info(model_id: str, process_res: int, ray_pose: bool) -> dict:
    return dict(
        model_version="Depth Anything 3",
        networks=[model_id] if model_id in (METRIC, NESTED) else [model_id, METRIC],
        non_commercial=model_id in NON_COMMERCIAL,
        resolution=process_res,
        bf16=True,
        ray_pose=ray_pose,
        mask_source="not sky (DA3METRIC-LARGE sky head < 0.3) and depth > 0; sky depth = 99th percentile of the rest",
        confidence="DA3 any-view depth confidence (larger = more reliable); absent for da3metric-large",
        metric_scale="least-squares fit of the any-view depth to DA3METRIC-LARGE (focal x output / 300) "
                     "on confident non-sky pixels, one scale per pass",
    )


def make_backend(job) -> Backend:
    """The reconstruction backend for lab2shot_worker.feedforward."""
    variant = job.params["model"]
    sys.path.insert(0, str(job.repo_dir / "src"))
    net = build(job.weights_dir, variant, torch.device("cuda"))
    mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)

    def infer(images: torch.Tensor, w: int, h: int, pad_w: int, pad_h: int) -> recon.Chunk:
        x = ((images - mean) / std)[None]
        depth, conf, sky, k, _k_model, w2c = forward(net, x, None, ray_pose=True)
        valid = (sky < SKY_THRESHOLD) & (conf >= torch.quantile(conf.flatten()[::97], CONF_PERCENTILE / 100))
        return recon.Chunk(cam_to_world=torch.linalg.inv(w2c).cpu().numpy(), K=k.cpu().numpy(),
                           depth=depth.cpu().numpy(), confidence=conf.cpu().numpy(), usable=valid.cpu().numpy(),
                           # 天空概率原样带出去
                           extra={"sky": sky.float().cpu().numpy()})

    return Backend(
        name=variant.upper(),
        weights=variant,
        metric=True,
        patch_budget=PATCH_BUDGET[variant],
        input_size=lambda width, height, res: (*fit_size(width, height, res or DEFAULT_LONG_SIDE, 14), 0, 0),
        infer=infer,
        confidence=f"DA3 any-view depth confidence (>= 1, larger = more reliable); valid = not sky and above the "
                   f"pass's {CONF_PERCENTILE}th percentile. Metric scale: DA3METRIC-LARGE, one least-squares scale per pass",
        models=[variant] if variant == NESTED else [variant, METRIC],
        licence="CC BY-NC 4.0 (non-commercial)" if variant in NON_COMMERCIAL else "Apache-2.0",
    )


def main(job_path: str) -> None:
    job = load_job(job_path)
    if check_node(job, "depthanything3.geometry", "depthanything3.reconstruct") == "depthanything3.reconstruct":
        run(job_path, "depthanything3.reconstruct", make_backend)
        return
    run = begin(job, "depthanything3.geometry", "Depth Anything 3", src="src")
    params = run.params
    model_id, process_res, fov_x, ray_pose = params["model"], params["resolution"], params["fov_x_deg"], params["ray_pose"]
    if model_id == METRIC and not fov_x:
        fail("E-DEPTHANYTHING3-NOFOCAL")
    net = run.model("Depth Anything 3 模型", build, run.job.weights_dir, model_id, torch.device("cuda"))
    geometry(run, net, model_id, fov_x, process_res, ray_pose)


if __name__ == "__main__":
    serve(main)
