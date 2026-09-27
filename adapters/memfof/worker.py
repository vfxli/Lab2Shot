"""MEMFOF worker: optical flow of a shot. Runs inside third_party/memfof/.venv with the pinned repo on the path; never
imports Lab2Shot core.

    python worker.py <job.json>

Job and raw layout: lab2shot_worker/optical_flow.py (shared by every optical-flow worker).

MEMFOF sees three frames at once and gives the middle one's backward and forward flow in one pass. Frame i is solved
from (i-1, i, i+1); the first and the last frame repeat themselves for the neighbour they lack, and the direction
towards that repeat is left out. The feature maps of the two frames the next triplet shares are kept (upstream's
fmap_cache), so each frame's features are computed once.
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

    model = run.model("MEMFOF 模型", load_model, model_dir, device)

    run.stage(f"计算光流（{shot.w}×{shot.h}）")
    n = len(shot)
    cache: list = [None, None, None]
    with torch.inference_mode():
        for i, _ in run.each(range(n), "光流"):
            with run.frame():
                ids = (max(i - 1, 0), i, min(i + 1, n - 1))
                clip = torch.from_numpy(np.stack([shot[k] for k in ids])).to(device).permute(0, 3, 1, 2)[None].float()  # 0..255
                out = model(clip, iters=ITERS, fmap_cache=cache)
                flow = out["flow"][-1][0].permute(0, 2, 3, 1).cpu().numpy()  # [2 (backward, forward), h, w, 2]
                info = out["info"][-1][0].cpu().numpy()  # [2, 4, h, w]
                conf = [of.confidence(info[d], model.var_min, model.var_max) for d in range(2)]
                of.save_frame(job.raw_dir, shot.frames[i][0],
                              backward=flow[0] if i > 0 else None, backward_confidence=conf[0] if i > 0 else None,
                              forward=flow[1] if i < n - 1 else None, forward_confidence=conf[1] if i < n - 1 else None)
                cache = [out["fmap_cache"][1], out["fmap_cache"][2], None]  # the next triplet's first two frames
                shot.release(i)
    shot.done(run, "MEMFOF-Tartan-T-TSKH")


if __name__ == "__main__":
    serve(main)
