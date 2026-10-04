"""UniDepth V2 worker: per-frame monocular metric geometry. Runs inside
third_party/unidepth/.venv with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

job["node"] == "unidepth.depth". Params:
    model       "unidepth-v2-vitl14" (default) | "unidepth-v2-vitb14" | "unidepth-v2-vits14"
    fov_x_deg   known horizontal field of view (degrees, 0-180) or null. Given: a pinhole
                camera (principal point at the centre) is fed to the network as its camera
                condition and returned as the intrinsics. Null: UniDepth predicts the camera.
    resolution_level  0-9 (default 9): UniDepth's resolution_level, the network's working size
                between 0.2 and 0.6 megapixels (9 = most detail). Output is always the input size.

Each frame on its own (no temporal model) -> raw/frame_<n>.npz (see lab2shot_worker.mono_geometry):
points, depth, mask, intrinsics, confidence. UniDepth has no sky / invalid-region
output: mask is every pixel with a finite positive depth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import resident, serve

from lab2shot_worker.mono_geometry import (begin, finish_geometry, frame_arrays, infer_frame, k_matrix, load_network,
                           pinhole_fit, pinhole_from_fov, ray_samples, run_frames)


@resident
def load_model(model_dir: Path, device: torch.device):
    return load_network(model_dir, device, "unidepth.models", "UniDepthV2", "UniDepth", "unidepth")


def main(job_path: str) -> None:
    run = begin(job_path, "unidepth.depth", "UniDepth")
    params = run.params
    model_id, level, fov_x = params["model"], params["resolution_level"], params["fov_x_deg"]
    model_dir = run.job.weights_dir / model_id
    run.weights(model_dir / "model.safetensors")

    from unidepth.models.unidepthv2.unidepthv2 import get_paddings, get_resize_factor
    from unidepth.utils.camera import Pinhole

    device = torch.device("cuda")
    model = run.model("load_model", load_model, model_dir, device, stage_params={"model": "UniDepth"})
    model.resolution_level = level

    def infer(rgb: np.ndarray, frame: int) -> dict:
        h, w = rgb.shape[:2]
        out, points = infer_frame(rgb, device, model, fov_x, get_paddings, get_resize_factor, Pinhole)
        # The intrinsics that match the points (upstream converts its predicted K back with a single
        # scale factor, ~1% off after the resize to multiples of 14): a pinhole fit of the output rays.
        fit = pinhole_fit(*ray_samples(points))
        arrays = frame_arrays(points, torch.ones(h, w, dtype=torch.bool, device=device),
                              pinhole_from_fov(fov_x, w, h) if fov_x else k_matrix(fit), out["confidence"][0, 0])
        arrays["_stats"] = {"focal_px_model": float(out["intrinsics"][0, 0, 0]) if fov_x else fit["fx"],
                            "pinhole_rms_px": fit["rms_px"]}
        return arrays

    stats = run_frames(run, infer)
    finish_geometry(run, stats, model=model_id, fov_x=fov_x,
                    model_version="UniDepthV2", resolution_level=level, fp16=True,
                    mask_source="finite positive depth (UniDepth has no sky / invalid-region output)",
                    confidence="UniDepth V2 confidence head (larger = more reliable)",
                    focal_note="focal_px_model: UniDepth's own focal (differs from focal_px when fov_x_deg is given); "
                               "intrinsics = pinhole fit of the output rays (pinhole_rms_px: how well they agree), "
                               "or the given lens")


if __name__ == "__main__":
    serve(main)
