"""Microsoft MoGe worker: per-frame monocular geometry. Runs inside
third_party/moge/.venv with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

Each frame on its own (MoGe has no temporal model) -> MoGeModel.infer() ->
raw/frame_<n>.npz (the lab2shot_worker.mono_geometry contract, with normals):

    points      float32 [H,W,3]  camera space, OpenCV (+X right, +Y down, +Z forward), metres
    depth       float32 [H,W]    camera Z, metres
    normal      float32 [H,W,3]  unit normals, same camera space (facing the camera)
    mask        bool    [H,W]    valid pixels (sky / undefined regions are False)
    intrinsics  float64 [3,3]    pixels at H,W; principal point at the image centre
                                 (W/2, H/2): pixel (i, j) covers [j, j+1] x [i, i+1]

Outside the mask the arrays hold the network's raw (unreliable) prediction, not
zeros, so a mask edge can be refined later; non-finite values are set to 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import resident, say, serve
from lab2shot_worker.mono_geometry import begin, finish_geometry, frame_arrays, run_frames

REFINE_STEPS = 3  # MoGe-3's sparse 3D refinement updates (upstream default)
# fp16 autocast for the ViT + heads (the refiner's depth binning stays fp32 upstream).
USE_FP16 = True


@resident
def load_model(checkpoint: Path, device: torch.device):
    """The checkpoint's own config decides the model class (v1 / v2 / v3); the repo is on sys.path (start())."""
    from moge.model import import_model_class_by_version

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config = dict(ckpt["model_config"])
    if config.get("refiner") is not None:
        version = "v3"
    elif config.get("scale_head") is not None:
        version = "v2"
    else:
        version = "v1"
    model = import_model_class_by_version(version)(**config)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="MoGe", extension="moge", model=f"{checkpoint.name}（{version}）",
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    if unexpected:
        say("I-WORKER-UNUSEDWEIGHTS", project="MoGe", model=f"{checkpoint.name}（{version}）",
            count=len(unexpected), examples=list(unexpected[:3]))
    del ckpt
    model = model.to(device).eval()
    return model, version


def normals_from_points(points: torch.Tensor) -> torch.Tensor:
    """[H,W,3] camera-space points -> unit normals facing the camera (for models without a normal head)."""
    p = points.permute(2, 0, 1)[None]
    p = F.pad(p, (1, 1, 1, 1), mode="replicate")[0]
    dx = p[:, 1:-1, 2:] - p[:, 1:-1, :-2]
    dy = p[:, 2:, 1:-1] - p[:, :-2, 1:-1]
    n = torch.linalg.cross(dx, dy, dim=0).permute(1, 2, 0)
    n = F.normalize(n, dim=-1)
    facing = (n * points).sum(-1, keepdim=True) > 0
    return torch.where(facing, -n, n)


def main(job_path: str) -> None:
    run = begin(job_path, "moge.geometry", "MoGe")
    params = run.params
    model_id, resolution_level, fov_x = params["model"], params["resolution_level"], params["fov_x_deg"]
    checkpoint = run.job.weights_dir / model_id / "model.pt"
    run.weights(checkpoint, what=f"模型 {model_id} 的权重")

    device = torch.device("cuda")
    model, version = run.model("MoGe 模型", load_model, checkpoint, device)
    has_normal_head = hasattr(model, "normal_head")
    if not has_normal_head:
        say("N-MOGE-NONORMALS", model=model_id)
    infer_kwargs = {"resolution_level": resolution_level, "fov_x": fov_x, "apply_mask": False, "use_fp16": USE_FP16}
    if version == "v3":
        infer_kwargs["refine_steps"] = REFINE_STEPS

    def infer(rgb: np.ndarray, frame: int) -> dict:
        h, w = rgb.shape[:2]
        out = model.infer(torch.from_numpy(rgb).to(device).permute(2, 0, 1), **infer_kwargs)
        points = torch.nan_to_num(out["points"].float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = out["mask"] if "mask" in out else torch.ones(h, w, dtype=torch.bool, device=device)
        normal = out.get("normal")
        # MoGe's intrinsics are normalized (image = unit square): to pixels at this frame's size.
        k = out["intrinsics"].double().cpu().numpy() * np.array([[w], [h], [1.0]])
        k[2] = (0.0, 0.0, 1.0)
        return frame_arrays(points, mask, k, normal=normals_from_points(points) if normal is None else normal)

    stats = run_frames(run, infer, dtype="float32")
    finish_geometry(run, stats, model=model_id, fov_x=fov_x, metric=version != "v1",
                    files="frame_<n>.npz: points, depth, normal, mask, intrinsics",
                    model_version={"v1": "MoGe-1", "v2": "MoGe-2", "v3": "MoGe-3"}[version],
                    normal_source="model" if has_normal_head else "point map finite differences",
                    normal_convention="unit normals in the same OpenCV camera space, facing the camera",
                    resolution_level=resolution_level, refine_steps=REFINE_STEPS if version == "v3" else 0, fp16=USE_FP16)


if __name__ == "__main__":
    serve(main)
