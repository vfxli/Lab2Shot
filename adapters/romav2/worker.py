"""RoMa v2 worker: dense matching of plate frames to another picture. Runs inside third_party/romav2/.venv with the
pinned repo's src/ on the path; never imports Lab2Shot core.

    python worker.py <job.json>

Job: the plate's frames (job frames), the other picture's frames (inputs["other"], a {"frames": {n: png}} listing) and
params["pairs"] [[plate frame, other frame], ...] (the node pairs them). Raw layout: lab2shot_worker/correspondence.py.

Per pair, upstream's RoMaV2.match(A, B) (both pictures squeezed to the setting's square, e.g. 800 x 800 then refined
at 1280 x 1280 for "precise") gives warp_AB: for every pixel of A, where it is in B (normalised [-1, 1], pixel centres
at -1 + 1/n), and overlap_AB, the probability it is seen in B. Those are written on RoMa's own grid; the node brings
them to the plate's size. Matches for other programs come from upstream's balanced sampling (RoMaV2.sample, both
directions), seeded so the same job gives the same matches.

Upstream loads its weights with torch.hub.load_state_dict_from_url and builds DINOv3 ViT-L through torch.hub.load of
facebookresearch/dinov3 at a pinned commit: the worker hands it the installed file and the installed checkout.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import fail, local_hub, read_frame, resident, serve, set_seed
from lab2shot_worker import correspondence as corr
from lab2shot_worker.run import Run

NODE = "romav2.match"
CHECKPOINT = "romav2.0.1.pt"
SEED = 0


@contextmanager
def local_weights(checkpoint: Path):
    """RoMaV2() fetches its weights by URL: the installed file instead."""
    fetch = torch.hub.load_state_dict_from_url

    def load(url, map_location=None, **kw):
        return torch.load(checkpoint, map_location=map_location)

    torch.hub.load_state_dict_from_url = load
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = fetch


@resident
def load_model(checkpoint: Path, dinov3: Path):
    local_hub({"dinov3": dinov3}, "RoMa v2")
    from romav2 import RoMaV2

    with local_weights(checkpoint):
        model = RoMaV2()
    return model.eval()


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "RoMa v2")
    job = run.job
    checkpoint = job.weights_dir / CHECKPOINT
    dinov3 = Path(os.environ["LAB2SHOT_DINOV3_DIR"])
    run.weights(checkpoint, dinov3)
    plate = dict(job.frames)
    other = job.listing("other")
    pairs = [(int(a), int(b)) for a, b in job.params["pairs"]]
    if not pairs:
        fail("E-ROMAV2-NOPAIRS")

    model = run.model("RoMa v2 模型", load_model, checkpoint, dinov3)
    model.apply_setting(job.params["setting"])  # a setting is plain attributes (sizes, both directions): set per job

    run.stage(f"稠密匹配（{len(pairs)} 对）")
    matches = {k: [] for k in ("xy_a", "xy_b", "frame_a", "frame_b", "confidence")}
    size_b = None
    for _i, (fa, fb) in run.each(pairs, "稠密匹配"):
        with run.frame():  # a "frame" here is one pair
            a, b = read_frame(plate[fa]), read_frame(other[fb])
            (ha, wa), (hb, wb) = a.shape[:2], b.shape[:2]
            size_b = size_b or (wb, hb)
            if (wb, hb) != size_b:
                fail("E-ROMAV2-REFERENCESIZE", frame=fb, width=wb, height=hb, other_width=size_b[0], other_height=size_b[1])
            preds = model.match(a, b)
            warp = preds["warp_AB"][0]  # [h, w, 2] in B, normalised
            xy = torch.stack([(warp[..., 0] + 1) / 2 * wb, (warp[..., 1] + 1) / 2 * hb], -1)
            corr.save_frame(job.raw_dir, fa, xy.float().cpu().numpy(), preds["overlap_AB"][0, ..., 0].float().cpu().numpy())
            if job.params["matches"]:
                set_seed(SEED)
                sampled, certainty, _, _ = model.sample(preds, job.params["matches"])
                kpts_a, kpts_b = model.to_pixel_coordinates(sampled, ha, wa, hb, wb)
                matches["xy_a"].append(kpts_a.float().cpu().numpy())
                matches["xy_b"].append(kpts_b.float().cpu().numpy())
                matches["frame_a"].append(np.full(len(kpts_a), fa))
                matches["frame_b"].append(np.full(len(kpts_a), fb))
                matches["confidence"].append(certainty.float().cpu().numpy())
    if job.params["matches"]:
        corr.save_matches(job.raw_dir, **{k: np.concatenate(v) for k, v in matches.items()})
    run.finish([fa for fa, _ in pairs], target={"width": size_b[0], "height": size_b[1]}, setting=job.params["setting"],
               grid=[int(model.W_hr or model.W_lr), int(model.H_hr or model.H_lr)])


if __name__ == "__main__":
    serve(main)
