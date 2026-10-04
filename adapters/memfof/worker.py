"""MEMFOF worker: optical flow of a shot. Runs inside third_party/memfof/.venv with the pinned repo on the path; never
imports Lab2Shot core.

    python worker.py <job.json>

Job and raw layout: lab2shot_worker/optical_flow.py (shared by every optical-flow worker).

MEMFOF sees three frames at once and gives the middle one's backward and forward flow in one pass. Frame i is solved
from (i-1, i, i+1); the first and the last frame repeat themselves for the neighbour they lack, and the direction
towards that repeat is left out. The feature maps of the two frames the next triplet shares are kept (upstream's
fmap_cache), so each frame's features are computed once.

Scale: MEMFOF is trained at 1080p (Tartan-T-TSKH: 872 x 1920 crops). Upstream evaluates lower-resolution footage
(Sintel 1024 wide, KITTI 1242) with the frames upsampled 2x and the flow brought back down, and 1080p footage (Spring)
as it is (model_lit.py scale_and_forward_flow, config/eval/*.json "scale"). The worker does the same: a shot processed
at its own size whose long side is below UPSCALE_BELOW runs 2x upsampled (bilinear), the flow scaled back (bilinear,
x 0.5) and the uncertainty head area-averaged, exactly as upstream; a size the user chose (resolution) is kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import resident, serve
from lab2shot_worker import optical_flow as of
from lab2shot_worker.run import Run

MODEL_DIR = "optical-flow-MEMFOF-Tartan-T-TSKH"
ITERS = 8  # upstream default (MEMFOF.forward)
UPSCALE_BELOW = 1440  # long side: below about 1080p the frames are upsampled 2x (upstream's evaluation of Sintel / KITTI)


def upscale_of(width: int, height: int, resolution) -> int:
    """upstream's `scale` (frames upsampled 2^scale): 1 for a shot at its own size below about 1080p, else 0."""
    return 1 if resolution is None and max(width, height) < UPSCALE_BELOW else 0


def triplet_flow(model, clip: torch.Tensor, cache: list, scale: int) -> tuple[torch.Tensor, torch.Tensor, list]:
    """One triplet [1, 3, 3, h, w] (0..255) -> flow [2, 2, h, w] (backward, forward), info [2, 4, h, w] and the
    feature cache, at the triplet's own size; scale 1: computed 2x upsampled (model_lit.scale_and_forward_flow)."""
    import torch.nn.functional as F

    if scale:
        b, t, c, h, w = clip.shape
        clip = F.interpolate(clip.reshape(b * t, c, h, w), scale_factor=2**scale, mode="bilinear",
                             align_corners=False).reshape(b, t, c, h * 2**scale, w * 2**scale)
    out = model(clip, iters=ITERS, fmap_cache=cache)
    flow, info = out["flow"][-1][0], out["info"][-1][0]  # [2, 2, H, W], [2, 4, H, W]
    if scale:
        flow = F.interpolate(flow, scale_factor=0.5**scale, mode="bilinear", align_corners=False) * 0.5**scale
        info = F.interpolate(info, scale_factor=0.5**scale, mode="area")
    return flow, info, out["fmap_cache"]


@resident
def load_model(model_dir: Path, device):
    from safetensors.torch import load_file

    from memfof.model import MEMFOF

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    # backbone_weights=None: no torchvision ImageNet download; every weight comes from the checkpoint below
    model = MEMFOF(backbone_weights=None, **config)
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    return model.eval().to(device)


def main(job_path: str) -> None:
    run = Run.start(job_path, "memfof.flow", "MEMFOF")
    job = run.job
    shot = of.Shot(job)
    model_dir = job.weights_dir / MODEL_DIR
    run.weights(model_dir / "model.safetensors", model_dir / "config.json")
    device = torch.device("cuda")

    model = run.model("load_model", load_model, model_dir, device, stage_params={"model": "MEMFOF"})

    scale = upscale_of(shot.w, shot.h, job.params["resolution"])
    run.stage("compute_flow", width=shot.w, height=shot.h)
    n = len(shot)
    cache: list = [None, None, None]
    with torch.inference_mode():
        for i, _ in run.each(range(n), "flow"):
            with run.frame():
                ids = (max(i - 1, 0), i, min(i + 1, n - 1))
                clip = torch.from_numpy(np.stack([shot[k] for k in ids])).to(device).permute(0, 3, 1, 2)[None].float()  # 0..255
                flow, info, fmaps = triplet_flow(model, clip, cache, scale)
                flow = flow.permute(0, 2, 3, 1).cpu().numpy()  # [2 (backward, forward), h, w, 2]
                info = info.cpu().numpy()  # [2, 4, h, w]
                conf = [of.confidence(info[d], model.var_min, model.var_max) for d in range(2)]
                of.save_frame(job.raw_dir, shot.frames[i][0],
                              backward=flow[0] if i > 0 else None, backward_confidence=conf[0] if i > 0 else None,
                              forward=flow[1] if i < n - 1 else None, forward_confidence=conf[1] if i < n - 1 else None)
                cache = [fmaps[1], fmaps[2], None]  # the next triplet's first two frames
                shot.release(i)
    shot.done(run, "MEMFOF-Tartan-T-TSKH", upsampled=2**scale)


if __name__ == "__main__":
    serve(main)
