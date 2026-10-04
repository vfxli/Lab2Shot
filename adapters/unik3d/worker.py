"""UniK3D worker: per-frame monocular metric 3D for any lens. Runs inside
third_party/unik3d/.venv with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

job["node"] == "unik3d.depth". Params:
    model       "unik3d-vitl" (default) | "unik3d-vitb" | "unik3d-vits"
    fov_x_deg   known horizontal field of view of a PINHOLE (undistorted) plate, degrees
                0-180, or null. Given: its rays are fed to the network as the camera
                condition. Null: UniK3D predicts the camera (any lens, fisheye included).
    resolution_level  0-9 (default 9): UniK3D's resolution_level, the network's working size
                between 0.2 and 0.6 megapixels. Output is always the input size.

Per frame -> raw/frame_<n>.npz (see lab2shot_worker.mono_geometry): points (= ray x
distance, the true geometry for any lens), depth (camera Z; pixels looking backwards,
Z <= 0, are outside the mask), mask, confidence, and intrinsics = the least-squares
PINHOLE APPROXIMATION of the predicted rays (exact when fov_x_deg is given). Plus the
network's own two arrays behind the node's 「距离图」 and 「射线场」 ports (unik3d.py:394, 397):
distance (along each ray, metres) and rays (the unit ray field itself, OpenCV camera space).

raw/camera.json keeps UniK3D's own camera model per frame, in input pixels (u, v are
continuous pixel coordinates, pixel centres at +0.5):

    lon = (u - cx) * rad_per_px_x          lat = (v - cy) * rad_per_px_y
    s   = (cos(lat) sin(lon), -sin(lat), cos(lat) cos(lon))
    ray = normalize( rsh_cart_3(s)[1:16] @ sh_coeffs )      # sh_coeffs [15, 3]

(rsh_cart_3: real spherical harmonics up to degree 3 from unik3d/utils/sht.py at the
pinned commit; ray in OpenCV camera space) plus the least-squares "pinhole" fit
(fx, fy, cx, cy) of the same rays with its RMS / max residual in pixels, which is what
the npz's intrinsics are.

No fisheye lens parameters are fitted: upstream infer returns neither intrinsics nor distortion coefficients. The
pinhole fit is kept because the npz's intrinsics require it (used inside the family for back-projection and
placement; not delivered as an output port).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import resident, say, serve

from lab2shot_worker.mono_geometry import (begin, finish_geometry, frame_arrays, infer_frame,  # noqa: E402
                           k_matrix, load_network, pinhole_fit, pinhole_from_fov, ray_samples, run_frames)

FIT_STEP = 4  # lens fits use every 4th pixel in x and y


@resident
def load_model(model_dir: Path, device: torch.device):
    # eval() (in load_network): UniK3D uses the given lens rays only in inference mode
    return load_network(model_dir, device, "unik3d.models", "UniK3D", "UniK3D", "unik3d")


def fit_pinhole(rays: torch.Tensor) -> dict:
    """Least-squares pinhole fit (fx, fy, cx, cy + residuals) of a ray field [H,W,3] (input pixels).

    This is what the npz's `intrinsics` are (used inside the family for back-projection; not an output port).
    Not an official UniK3D output: the network's own lens is the ray field itself (the 「射线场」 port)."""
    return pinhole_fit(*ray_samples(rays, FIT_STEP))


def main(job_path: str) -> None:
    run = begin(job_path, "unik3d.depth", "UniK3D")
    params = run.params
    model_id, level, fov_x = params["model"], params["resolution_level"], params["fov_x_deg"]
    model_dir = run.job.weights_dir / model_id
    run.weights(model_dir / "model.safetensors")

    from unik3d.models.unik3d import get_paddings, get_resize_factor
    from unik3d.utils.camera import Pinhole

    device = torch.device("cuda")
    model = run.model("load_model", load_model, model_dir, device, stage_params={"model": "UniK3D"})
    model.resolution_level = level

    # UniK3D's camera module returns (hfov, vfov, cx, cy) at the network's input size and
    # the [15, 3] spherical-harmonics coefficients; infer() does not return them.
    captured = {}
    angular = model.pixel_decoder.angular_module
    hook = angular.register_forward_hook(lambda mod, args, out: captured.update(
        intrinsics=out[0][0].float().cpu(), sh=out[1][0].float().cpu(), shape=tuple(mod.shapes)))
    ratio_bounds = model.shape_constraints["ratio_bounds"]

    cameras = {}

    def infer(rgb: np.ndarray, frame: int) -> dict:
        h, w = rgb.shape[:2]
        out, points = infer_frame(rgb, device, model, fov_x, get_paddings, get_resize_factor, Pinhole)
        rays = out["rays"][0].permute(1, 2, 0).float()
        pin = fit_pinhole(rays)
        k = pinhole_from_fov(fov_x, w, h) if fov_x else k_matrix(pin)

        # The network camera in input pixels (see the module docstring).
        (pad_l, _pad_r, pad_t, _pad_b), (padded_h, padded_w) = get_paddings((h, w), ratio_bounds)
        net_h, net_w = captured["shape"]
        hfov, vfov, cx_net, cy_net = (float(v) for v in captured["intrinsics"])
        cameras[frame] = {
            "cx": cx_net * padded_w / net_w - pad_l,
            "cy": cy_net * padded_h / net_h - pad_t,
            "rad_per_px_x": hfov / padded_w,
            "rad_per_px_y": vfov / padded_h,
            "hfov_deg_network": math.degrees(hfov),
            "sh_coeffs": captured["sh"].tolist(),
            "used": "user pinhole (fov_x_deg)" if fov_x else "predicted",
            "pinhole": pin,
        }
        arrays = frame_arrays(points, torch.ones(h, w, dtype=torch.bool, device=device), k, out["confidence"][0, 0])
        # upstream's two other outputs (unik3d.py:394, 397): distance along each ray (metres) and the ray field itself
        # (unit vectors, OpenCV camera space), read by the node's 「距离图」 and 「射线场」 outputs
        arrays["distance"] = out["distance"][0, 0].float().cpu().numpy().astype(np.float32)
        arrays["rays"] = rays.cpu().numpy().astype(np.float32)
        arrays["_stats"] = {"focal_px_model": None if fov_x else pin["fx"], "pinhole_rms_px": pin["rms_px"]}
        return arrays

    try:
        stats = run_frames(run, infer)
    finally:
        hook.remove()  # the model stays loaded for the next job

    pin_rms = float(np.median(stats["pinhole_rms_px"]))
    if not fov_x and pin_rms > 2.0:
        say("W-UNIK3D-DISTORTED", pinhole=pin_rms)
    (run.job.raw_dir / "camera.json").write_text(json.dumps({
        "model": "UniK3D spherical-harmonics ray field",
        "formula": "lon=(u-cx)*rad_per_px_x; lat=(v-cy)*rad_per_px_y; s=(cos(lat)sin(lon), -sin(lat), cos(lat)cos(lon)); "
                   "ray=normalize(rsh_cart_3(s)[1:16] @ sh_coeffs); u,v continuous input pixels (pixel centres +0.5); "
                   "ray in OpenCV camera space",
        "rsh_cart_3": "unik3d/utils/sht.py at the pinned UniK3D commit",
        "lens_fits": "pinhole: u=fx*x/z+cx, v=fy*y/z+cy; residuals in pixels over every 4th pixel",
        "intrinsics_in_npz": "user pinhole" if fov_x else "pinhole fit",
        "width": run.job.width or None,
        "height": run.job.height or None,
        "frames": {str(f): c for f, c in cameras.items()},
    }, indent=1), encoding="utf-8")

    finish_geometry(run, stats, model=model_id, fov_x=fov_x,
                    model_version="UniK3D", resolution_level=level, fp16=True,
                    camera="camera.json (UniK3D camera model per frame + the pinhole fit)",
                    intrinsics_note="pinhole approximation (least-squares fit of the predicted rays); "
                                    "points are exact for any lens",
                    pinhole_rms_px_median=pin_rms,
                    mask_source="finite depth with Z > 0 (UniK3D has no sky / invalid-region output)",
                    confidence="UniK3D confidence head (larger = more reliable)")


if __name__ == "__main__":
    serve(main)
