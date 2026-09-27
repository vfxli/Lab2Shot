"""GeoCalib worker: which way is up in each frame (gravity) and the lens. Runs inside third_party/geocalib/.venv
with the pinned repo on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>        (job["node"] == "geocalib.calibrate")

The lens first, once for the shot (it is not a zoom): the connected one (focal_px), else GeoCalib on up to SAMPLES
evenly spaced frames together with one shared focal (and distortion). Then every `step`-th frame (and the last) on
its own with that focal fixed: up field + latitude field from the network, then GeoCalib's Levenberg-Marquardt fit
of roll and pitch. Raw results:

    raw/gravity.npz   frames [F], up [F,3]: the unit vector against gravity in that frame's OpenCV camera
                      (+X right, +Y down, +Z forward; GeoCalib's "gravity" vector is this up direction),
                      uncertainty_deg [F] (GeoCalib's gravity uncertainty, one sigma), roll_deg [F], pitch_deg [F]
    raw/lens.json     focal_px (pixels at the plate width), model, k1 (distorted models), source, width, height
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fail, progress, read_frame, resident, save_npz, say, serve
from lab2shot_worker.run import Run

NODE = "geocalib.calibrate"
SAMPLES = 16  # frames the shared lens is estimated on
BATCH = 16  # frames per network pass (each is resized to 320 px on its short side)
READ_SHORT = 640  # frames are first shrunk to this short side on the CPU (GeoCalib then resizes to 320 on the GPU)
# the camera model (node parameter) -> the checkpoint trained for it and GeoCalib's model name
MODELS = {"pinhole": ("pinhole", "pinhole"), "simple_radial": ("distorted", "simple_radial"),
          "simple_divisional": ("distorted", "simple_divisional")}


@resident
def load_model(checkpoint: Path, device: torch.device):
    from geocalib import GeoCalib

    return GeoCalib(weights=str(checkpoint)).to(device).eval()  # a local file: nothing is fetched


def images(paths: list[Path], scale: float, device: torch.device) -> torch.Tensor:
    """[B,3,H,W] RGB in 0..1, as GeoCalib's load_image gives, `scale` times the plate's size."""
    import cv2

    def one(p: Path) -> np.ndarray:
        rgb = read_frame(p, "float32")
        return rgb if scale == 1.0 else cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    return torch.from_numpy(np.stack([one(p) for p in paths])).permute(0, 3, 1, 2).to(device)


@torch.no_grad()
def calibrate(model, batch: torch.Tensor, camera_model: str, focal_px: float | None, shared: bool) -> dict:
    """GeoCalib on a batch; `focal_px` fixes the focal (pixels at the batch's size)."""
    priors = {"focal": torch.full((len(batch),), focal_px, device=batch.device)} if focal_px else None
    return model.calibrate(batch, camera_model=camera_model, priors=priors, shared_intrinsics=shared)


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "GeoCalib")
    job, params = run.job, run.params
    weights, camera_model = MODELS[params["fit_model"]]
    checkpoint = job.weights_dir / f"geocalib-{weights}.tar"
    run.weights(checkpoint)
    device = torch.device("cuda")

    frames = job.frames
    used = run.frames(step=params["step"]).pairs  # every step-th frame and the last
    width, height = job.width, job.height
    scale = min(1.0, READ_SHORT / min(width, height))  # focal lengths below are in these pixels, until written
    raw = job.raw_dir

    model = run.model("GeoCalib 模型", load_model, checkpoint, device)

    focal, k1, source = params["focal_px"], None, "user"
    if focal is None:
        run.stage("估计镜头（整段同一个 Focal Length）")
        pick = np.unique(np.linspace(0, len(frames) - 1, min(SAMPLES, len(frames))).round().astype(int))
        camera = calibrate(model, images([frames[i][1] for i in pick], scale, device), camera_model, None, shared=True)["camera"]
        focal = float(camera.f[..., 0].median()) / scale
        k1 = float(camera.dist[..., 0].median()) if hasattr(camera, "dist") else None
        source = "geocalib"

    run.stage("估计每帧的重力方向")
    up, sigma, roll, pitch = [], [], [], []
    for a in range(0, len(used), BATCH):
        chunk = used[a:a + BATCH]
        out = calibrate(model, images([p for _, p in chunk], scale, device), camera_model, focal * scale, shared=False)
        g = out["gravity"]
        up.append(g.vec3d.double().cpu().numpy())
        sigma.append(np.degrees(out["gravity_uncertainty"].double().cpu().numpy()))
        roll.append(np.degrees(g.roll.double().cpu().numpy()))
        pitch.append(np.degrees(g.pitch.double().cpu().numpy()))
        progress(min(a + BATCH, len(used)), len(used), "重力方向")
    up, sigma = np.concatenate(up), np.nan_to_num(np.concatenate(sigma), nan=90.0, posinf=90.0)
    roll, pitch = np.concatenate(roll), np.concatenate(pitch)
    if not np.isfinite(up).all():
        fail("E-GEOCALIB-NOGRAVITY")
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    save_npz(raw / "gravity.npz", frames=np.asarray([f for f, _ in used], np.int64), up=up, uncertainty_deg=sigma,
             roll_deg=roll, pitch_deg=pitch)
    lens = {"focal_px": focal, "model": camera_model, "k1": k1, "source": source, "width": width, "height": height}
    (raw / "lens.json").write_text(json.dumps(lens, indent=1), encoding="utf-8")

    spread = float(np.degrees(np.arccos(np.clip(up @ (up.mean(0) / np.linalg.norm(up.mean(0))), -1, 1))).max())
    if float(np.median(sigma)) > 5.0:
        say("W-GEOCALIB-UNRELIABLE", error=float(np.median(sigma)))
    run.finish(
        [f for f, _ in used],
        kind="gravity",
        model=f"GeoCalib {weights}",
        camera_model=camera_model,
        convention="up: unit vector against gravity in the OpenCV camera of each frame (+X right, +Y down, +Z forward); "
                   "roll / pitch: GeoCalib's angles of it (degrees)",
        frames=[f for f, _ in used],  # the whole list, not the standard [first, last]: gravity.npz has a row per frame
        step=params["step"],
        lens=lens,
        fov_x_deg=math.degrees(2 * math.atan(width / (2 * focal))),
        roll_deg={"median": float(np.median(roll)), "min": float(roll.min()), "max": float(roll.max())},
        pitch_deg={"median": float(np.median(pitch)), "min": float(pitch.min()), "max": float(pitch.max())},
        uncertainty_deg={"median": float(np.median(sigma)), "max": float(sigma.max())},
        spread_deg=spread,
    )


if __name__ == "__main__":
    serve(main)
