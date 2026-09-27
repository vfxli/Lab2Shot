"""AnyCalib worker: lens calibration. Runs inside third_party/anycalib/.venv with
the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

A few evenly spaced frames of the shot -> AnyCalib (DINOv2 ray field -> fit of
the chosen camera model) per frame -> per-parameter median (the lens is assumed
constant over the shot) -> raw/lens.json + two ST-maps at plate resolution:

    raw/stmap_undistort.exr  for each pixel of the undistorted (ideal pinhole) image:
                             where to sample in the distorted plate
    raw/stmap_distort.exr    for each pixel of the distorted plate:
                             where to sample in the undistorted image
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fail, progress, read_frame, require_weights, resident, say, serve
from lab2shot_worker.files import write_exr
from lab2shot_worker.run import Run

NODE = "anycalib.calibrate"
BATCH = 8  # frames per network pass (~320x320 each: small)
ROUNDTRIP_TOL_PX = 0.05  # a mapping is "valid" where project(unproject(p)) returns to p

PIXELS = (
    "continuous pixel coordinates (AnyCalib's convention): origin at the top-left corner of the "
    "top-left pixel, x right, y down; pixel (i, j) has its center at (i + 0.5, j + 0.5), so "
    "cx = W / 2, cy = H / 2 is the exact image center"
)
STMAP = (
    "float32 EXR at plate resolution, channels R,G,B,A. For the output pixel whose center is (x, y) "
    "in continuous pixel coordinates, R,G is the position to sample in the source image, normalised "
    "the Nuke STMap way: R = xs / W, G = 1 - ys / H (origin bottom-left; an identity map is "
    "R = (i + 0.5) / W, G = 1 - (j + 0.5) / H for row j counted from the top). B = 0. A = 1 where the "
    "mapping is valid and the sample lies inside the source image, else 0; R = G = -1 where there is "
    "no valid mapping (outside the lens model's domain). OpenCV: map_x = R * W - 0.5, "
    "map_y = (1 - G) * H - 0.5 for cv2.remap"
)
UNDISTORTED = (
    "ideal pinhole image with the same size and principal point (cx, cy) as the plate and the lens's "
    "paraxial focal lengths focal_px / fy (the pinhole that matches the plate at the image center: "
    "the model's own f for every model except ucm / simple_ucm, where it is f / (1 + xi)); "
    "a CG camera with focal_mm = focal_px / W * filmback_width_mm matches it"
)


def write_stmap(path: Path, rg: np.ndarray, alpha: np.ndarray, attrs: dict[str, str]) -> None:
    """R, G = the ST coordinates, B = 0, A = where the map is valid."""
    rgba = np.zeros((*alpha.shape, 4), np.float32)
    rgba[..., :2] = rg
    rgba[..., 3] = alpha
    write_exr(path, rgba, "RGBA", attrs)


# ----------------------------------------------------------------------------- estimation


@resident
def load_model(weights: Path, checkpoint: str, device: torch.device):
    from anycalib import AnyCalib

    path = weights / f"{checkpoint}.pt"
    require_weights("anycalib", path)
    model = AnyCalib(model_id=None)  # architecture only; weights from the pinned file
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval().to(device)


@torch.inference_mode()
def estimate(model, images: torch.Tensor, cam_id: str, center: tuple[float, float] | None):
    """Upstream AnyCalib.predict, plus an optional fixed principal point.

    images: (B, 3, H, W) RGB in [0, 1]. Returns (B, D) intrinsics in the plate's
    pixel coordinates and (B,) success flags.
    """
    from anycalib.cameras import CameraFactory

    ho, wo = images.shape[-2:]
    target_ar = max(model.AR_RANGE[0], min(ho / wo, model.AR_RANGE[1]))
    target_size = model.compute_target_size(model.RESOLUTION, target_ar)
    im, scale_xy, shift_xy = model.set_im_size(images, target_size)
    b = im.shape[0]
    data = {"image": im, "cam_id": [cam_id] * b}
    if center is not None:
        # AnyCalib fits with a known principal point when data["cxcy"] is given
        # (in the network's resized image coordinates).
        cxcy = torch.tensor(center, device=im.device, dtype=im.dtype) * scale_xy + shift_xy
        data["cxcy"] = cxcy.expand(b, 2)
    pred = model.forward(data)
    cam = CameraFactory.create_from_id(cam_id)
    intrinsics = torch.stack([cam.reverse_scale_and_shift(p, scale_xy, shift_xy) for p in pred["intrinsics"]])
    return intrinsics.double().cpu(), pred["success"].cpu()


def param_names(cam) -> list[str]:
    names = [n for n, _ in sorted(cam.PARAMS_IDX.items(), key=lambda kv: kv[1])][: cam.nparams]
    family = cam.NAME.removeprefix("simple_")
    rename = {"ucm": {"k1": "xi"}, "eucm": {"k1": "alpha", "k2": "beta"}, "fov": {"k1": "omega"}}.get(family, {})
    return [rename.get(n, n) for n in names]


# ----------------------------------------------------------------------------- ST-maps


def focal_center(cam, params: torch.Tensor) -> tuple[float, float, float, float]:
    """Paraxial (image-center) focal lengths and principal point, in pixels.

    The focal of the pinhole that matches the lens at the image center: the
    model's own f for every model except UCM, whose f is scaled by (1 + xi).
    Measured on the model itself (central difference of project() at the axis).
    """
    d = cam.params_to_dict(params)
    cx, cy = d["c"].tolist()
    e = 1e-4
    rays = torch.tensor([[e, 0, 1], [-e, 0, 1], [0, e, 1], [0, -e, 1]], dtype=params.dtype)
    uv, _ = cam.project(params, torch.nn.functional.normalize(rays, dim=-1))
    fx = float(uv[0, 0] - uv[1, 0]) / (2 * e)
    fy = float(uv[2, 1] - uv[3, 1]) / (2 * e)
    return fx, fy, cx, cy


def pixel_grid(w: int, h: int) -> torch.Tensor:
    """(H*W, 2) pixel centers in continuous pixel coordinates, row-major from the top-left."""
    y, x = torch.meshgrid(torch.arange(h, dtype=torch.float64), torch.arange(w, dtype=torch.float64), indexing="ij")
    return torch.stack((x, y), dim=-1).reshape(-1, 2) + 0.5


def undistort_lookup(cam, params, fx, fy, cx, cy, pts):
    """Undistorted (pinhole) pixel -> position in the distorted plate, and validity."""
    rays = torch.stack(((pts[:, 0] - cx) / fx, (pts[:, 1] - cy) / fy, torch.ones_like(pts[:, 0])), dim=-1)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    src, valid = cam.project(params, rays)
    ok = torch.isfinite(src).all(-1) if valid is None else valid & torch.isfinite(src).all(-1)
    # Beyond a fold of the distortion polynomial (e.g. strong barrel k1 < 0) project()
    # still returns numbers, but they do not come back: keep only true inverses.
    back, bvalid = cam.unproject(params, src.nan_to_num(0.0))
    z = back[:, 2].clamp_min(1e-12)
    again = torch.stack((fx * back[:, 0] / z + cx, fy * back[:, 1] / z + cy), dim=-1)
    ok &= (back[:, 2] > 1e-9) & ((again - pts).norm(dim=-1) < ROUNDTRIP_TOL_PX)
    if bvalid is not None:
        ok &= bvalid
    return src, ok


def distort_lookup(cam, params, fx, fy, cx, cy, pts):
    """Distorted plate pixel -> position in the undistorted (pinhole) image, and validity."""
    bearings, valid = cam.unproject(params, pts)
    z = bearings[:, 2]
    zc = z.clamp_min(1e-12)
    dst = torch.stack((fx * bearings[:, 0] / zc + cx, fy * bearings[:, 1] / zc + cy), dim=-1)
    ok = (z > 1e-9) & torch.isfinite(dst).all(-1)
    if valid is not None:
        ok &= valid
    again, pvalid = cam.project(params, bearings.nan_to_num(0.0))
    ok &= (again - pts).norm(dim=-1) < ROUNDTRIP_TOL_PX
    if pvalid is not None:
        ok &= pvalid
    return dst, ok


def to_stmap(pos: torch.Tensor, ok: torch.Tensor, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    pos = pos.numpy().reshape(h, w, 2)
    ok = ok.numpy().reshape(h, w)
    inside = ok & (pos[..., 0] >= 0) & (pos[..., 0] <= w) & (pos[..., 1] >= 0) & (pos[..., 1] <= h)
    rg = np.empty((h, w, 2), np.float64)
    rg[..., 0] = pos[..., 0] / w
    rg[..., 1] = 1.0 - pos[..., 1] / h
    rg[~ok] = -1.0
    return rg.astype(np.float32), inside.astype(np.float32)


def sample_bilinear(img: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Sample (H, W, C) at continuous pixel coordinates (N, 2) (pixel centers at +0.5), clamped at the border."""
    h, w = img.shape[:2]
    x = np.clip(pos[:, 0] - 0.5, 0, w - 1)
    y = np.clip(pos[:, 1] - 0.5, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 2)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 2)
    tx = (x - x0)[:, None]
    ty = (y - y0)[:, None]
    return (
        img[y0, x0] * (1 - tx) * (1 - ty) + img[y0, x0 + 1] * tx * (1 - ty)
        + img[y0 + 1, x0] * (1 - tx) * ty + img[y0 + 1, x0 + 1] * tx * ty
    )


def roundtrip(st_undistort: np.ndarray, a_undistort: np.ndarray, dst: np.ndarray, ok: np.ndarray, w: int, h: int) -> dict:
    """distort -> undistort through the written maps: each plate pixel must land on itself."""
    pts = pixel_grid(w, h).numpy()
    inside = ok & (dst[:, 0] >= 0.5) & (dst[:, 0] <= w - 0.5) & (dst[:, 1] >= 0.5) & (dst[:, 1] <= h - 0.5)
    if not inside.any():
        return {"pixels": 0}
    p, q = pts[inside], dst[inside]
    rg = sample_bilinear(st_undistort.astype(np.float64), q)
    alpha = sample_bilinear(a_undistort[..., None].astype(np.float64), q)[:, 0]
    good = alpha > 0.999  # all four neighbours valid
    back = np.stack((rg[:, 0] * w, (1.0 - rg[:, 1]) * h), axis=-1)
    err = np.linalg.norm(back - p, axis=-1)[good]
    return {
        "pixels": int(good.sum()),
        "mean_px": float(err.mean()),
        "p99_px": float(np.percentile(err, 99)),
        "max_px": float(err.max()),
    }


# ----------------------------------------------------------------------------- main


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "AnyCalib")
    job, params = run.job, run.params
    cam_id, samples, checkpoint, principal_point = params["model"], params["samples"], params["checkpoint"], params["principal_point"]

    sys.path.insert(0, str(job.repo_dir))  # the pinned AnyCalib repo, pure Python
    from anycalib.cameras import CameraFactory

    try:
        cam = CameraFactory.create_from_id(cam_id)
    except (KeyError, ValueError, AssertionError):
        fail("E-ANYCALIB-MODEL", model=str(cam_id), models=list(CameraFactory.FACTORY))
    names = param_names(cam)

    frames = run.frames().pairs
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, min(samples, len(frames)))).astype(int))
    picked = [frames[i] for i in idx]

    device = torch.device("cuda")
    raw = job.raw_dir

    weights = job.weights_dir
    model = run.model("AnyCalib 模型", load_model, weights, checkpoint, device)

    run.stage("估计镜头")
    width = height = None
    per_frame: list[dict] = []
    for start in range(0, len(picked), BATCH):
        chunk = picked[start : start + BATCH]
        arrays = [read_frame(p, "float32") for _, p in chunk]
        shapes = {a.shape[:2] for a in arrays}
        if len(shapes) != 1 or (width is not None and (height, width) not in shapes):
            fail("E-ANYCALIB-SIZES", sizes=[f"{sw}×{sh}" for sh, sw in sorted(shapes)])
        height, width = arrays[0].shape[:2]
        images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(device)
        center = (width / 2.0, height / 2.0) if principal_point == "center" else None
        intrinsics, success = estimate(model, images, cam_id, center)
        for (frame, _), p, ok in zip(chunk, intrinsics, success):
            values = p.tolist()
            if center is not None:
                values[cam.PARAMS_IDX["cx"]], values[cam.PARAMS_IDX["cy"]] = center
            per_frame.append({"frame": frame, "params": values, "success": bool(ok)})
        progress(len(per_frame), len(picked), "估计镜头")
    del model
    estimate_seconds = time.time() - run.t0  # model loading included

    # AnyCalib's own flag is also False when the refinement did not lower the cost
    # (it then keeps the linear fit, usually fine): only drop non-finite / non-positive focals.
    nf = cam.NUM_F
    usable = [f for f in per_frame if all(math.isfinite(v) for v in f["params"]) and min(f["params"][:nf]) > 0]
    if not usable:
        fail("E-ANYCALIB-NOFIT")
    if len(usable) < len(per_frame):
        say("N-ANYCALIB-FAILEDFRAMES", failed=len(per_frame) - len(usable), sampled=len(per_frame))
    flagged = sum(not f["success"] for f in usable)
    if flagged:
        say("N-ANYCALIB-UNCONVERGED", count=flagged)
    table = np.array([f["params"] for f in usable], np.float64)
    combined = np.median(table, axis=0)
    mad = np.median(np.abs(table - combined), axis=0)
    params_t = torch.from_numpy(combined)
    fx, fy, cx, cy = focal_center(cam, params_t)
    spread = float(mad[0] / combined[0]) if combined[0] else 0.0
    if spread > 0.05:
        say("W-ANYCALIB-FOCALSPREAD", spread=spread)
    hfov, hvalid = cam.get_hfov(params_t, width)
    vfov, vvalid = cam.get_vfov(params_t, height)
    diag_fov = 2 * math.degrees(math.atan(0.5 * math.hypot(width / fx, height / fy)))
    if cam.num_k and diag_fov < 30:
        say("W-ANYCALIB-LONGLENS", fov=float(diag_fov))

    run.stage("生成 ST-map")
    pts = pixel_grid(width, height)
    src, src_ok = undistort_lookup(cam, params_t, fx, fy, cx, cy, pts)
    dst, dst_ok = distort_lookup(cam, params_t, fx, fy, cx, cy, pts)
    st_u, a_u = to_stmap(src, src_ok, width, height)
    st_d, a_d = to_stmap(dst, dst_ok, width, height)
    check = roundtrip(st_u, a_u, dst.numpy(), dst_ok.numpy(), width, height)
    attrs = {"model": cam_id, "convention": STMAP}
    write_stmap(raw / "stmap_undistort.exr", st_u, a_u, {**attrs, "map": "undistorted pixel -> distorted plate position"})
    write_stmap(raw / "stmap_distort.exr", st_d, a_d, {**attrs, "map": "distorted plate pixel -> undistorted position"})
    # How much of each image the other one covers (1.0 = no black borders).
    coverage = {"undistorted_from_plate": float(a_u.mean()), "plate_from_undistorted": float(a_d.mean())}
    if coverage["undistorted_from_plate"] < 0.5:
        say("W-ANYCALIB-COVERAGE", share=coverage["undistorted_from_plate"])

    lens = {
        "model": cam_id,
        "params": combined.tolist(),
        "param_names": names,
        "width": width,
        "height": height,
        "focal_px": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "per_frame": per_frame,
        "combine": "median",
        "mad": mad.tolist(),
        "principal_point": principal_point,
        "checkpoint": checkpoint,
        "hfov_deg": math.degrees(float(hfov)) if bool(hvalid) else None,
        "vfov_deg": math.degrees(float(vfov)) if bool(vvalid) else None,
        "focal_note": "focal_px / fy: paraxial pinhole focal in pixels (= f of the model except ucm: f / (1 + xi))",
    }
    (raw / "lens.json").write_text(json.dumps(lens, indent=2), encoding="utf-8")

    radial = cam.NAME.removeprefix("simple_") == "radial"
    run.finish(
        [f for f, _ in picked],
        kind="lens",
        lens="lens.json",
        stmap_undistort="stmap_undistort.exr",
        stmap_distort="stmap_distort.exr",
        model=cam_id,
        checkpoint=checkpoint,
        principal_point=principal_point,
        focal_px=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        params=dict(zip(names, combined.tolist())),
        width=width,
        height=height,
        frames_used=[f["frame"] for f in usable],
        focal_spread_mad=spread,
        coverage=coverage,
        roundtrip=check,
        conventions={
            "pixels": PIXELS,
            "params": (
                f"AnyCalib's {cam.NAME} model (anycalib/cameras/{cam.NAME}.py at the pinned commit), order "
                f"{names}: focal(s) and principal point in pixels (coordinates above), distortion "
                "coefficients unitless on normalised image coordinates"
            ),
            "radial": (
                "x_d = fx * x_u * (1 + k1 r^2 + k2 r^4 + k3 r^6 + k4 r^8) + cx with x_u = X/Z, "
                "r^2 = x_u^2 + y_u^2 (distortion applied to undistorted normalised coordinates, "
                "barrel = k1 < 0). OpenCV equivalent: K = [[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]], "
                "distCoeffs = [k1, k2, 0, 0, k3] (k4 has no OpenCV slot)"
            ) if radial else None,
            "combine": "per-parameter median over the sampled frames (mad = median absolute deviation)",
            "undistorted_image": UNDISTORTED,
            "stmap": STMAP,
            "stmap_undistort": (
                "for each pixel of the UNDISTORTED image: where to sample in the distorted plate. "
                "Nuke: STMap(src = plate, stmap = this, uv = rgb) gives the undistorted plate"
            ),
            "stmap_distort": (
                "for each pixel of the DISTORTED plate: where to sample in the undistorted image. "
                "Nuke: STMap(src = undistorted CG render, stmap = this) puts the plate's lens distortion on it"
            ),
            "roundtrip": "plate pixel -> stmap_distort -> bilinear lookup in stmap_undistort -> back on the plate: error in pixels",
        },
        estimate_seconds=round(estimate_seconds, 1),
    )


if __name__ == "__main__":
    serve(main)
