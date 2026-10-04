"""WAFT worker: optical flow of a shot. Runs inside third_party/waft/.venv with the pinned repo on the path; never
imports Lab2Shot core.

    python worker.py <job.json>

Job and raw layout: lab2shot_worker/optical_flow.py (shared by every optical-flow worker).

WAFT is a two-frame method: frame i's forward flow is flow(i, i+1) and its backward flow flow(i, i-1), both in one
batch. The model is WAFT-a1 with Depth Anything V2 ViT-S features (config/a1/tar-c-t.json, checkpoint tar-c-t.pth).
Upstream builds the network with pretrained parts it loads at construction (Depth Anything V2 from
depth-anything-ckpts/, ResNet-18 layers from timm's hub); the checkpoint holds every one of those weights, so the
worker builds it without them and loads the checkpoint strictly.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from lab2shot_worker import resident, serve
from lab2shot_worker import optical_flow as of
from lab2shot_worker.run import Run

CHECKPOINT = "tar-c-t.pth"
CONFIG = "config/a1/tar-c-t.json"


@contextmanager
def no_pretrained_parts():
    """While WAFT-a1 is built: Depth Anything V2 without its checkpoint file, timm models without a download."""
    import timm

    import model.waft_a1 as waft_a1

    feature, create = waft_a1.DepthAnythingFeature, timm.create_model

    def depth_anything(encoder="vits", pretrained=True):
        return feature(encoder=encoder, pretrained=False)

    def create_model(name, pretrained=False, **kw):
        return create(name, pretrained=False, **kw)

    waft_a1.DepthAnythingFeature, timm.create_model = depth_anything, create_model
    try:
        yield
    finally:
        waft_a1.DepthAnythingFeature, timm.create_model = feature, create


def sdpa_attention() -> None:
    """Depth Anything V2's attention through torch's fused scaled_dot_product_attention. Upstream asks for xFormers
    (README: "Please also install xformers"); without it DINOv2's MemEffAttention falls back to the explicit
    q @ k^T softmax, which holds the whole tokens x tokens matrix (most of WAFT's memory at 1920). Instead of one more
    dependency, the fallback itself is replaced by the same attention computed by torch's memory-efficient kernel
    (the same mathematics: softmax(q k^T / sqrt(d)) v; dropout is 0 in eval): 1920 x 816 pair 9.2 -> 6.2 GB, same speed."""
    import torch.nn.functional as F
    from thirdparty.DepthAnythingV2.depth_anything_v2.dinov2_layers import attention

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(B, N, C)))

    attention.Attention.forward = forward


@resident
def load_model(repo: Path, checkpoint: Path, device):
    from model.waft_a1 import ViTWarpV8

    sdpa_attention()

    args = argparse.Namespace(**json.loads((repo / CONFIG).read_text(encoding="utf-8")))
    with no_pretrained_parts():
        model = ViTWarpV8(args)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.eval().to(device)


def main(job_path: str) -> None:
    run = Run.start(job_path, "waft.flow", "WAFT")
    job = run.job
    shot = of.Shot(job)
    checkpoint = job.weights_dir / CHECKPOINT
    run.weights(checkpoint)
    device = torch.device("cuda")

    model = run.model("load_model", load_model, job.repo_dir, checkpoint, device, stage_params={"model": "WAFT"})

    run.stage("compute_flow", width=shot.w, height=shot.h)
    n = len(shot)
    with torch.inference_mode():
        for i, _ in run.each(range(n), "flow"):
            with run.frame():
                others = [k for k in (i + 1, i - 1) if 0 <= k < n]  # forward first, then backward
                pictures = torch.from_numpy(np.stack([shot[k] for k in [i, *others]])).to(device).permute(0, 3, 1, 2).float()
                out = model(pictures[:1].expand(len(others), -1, -1, -1), pictures[1:])  # 0..255, as upstream wants
                flow = out["flow"][-1].permute(0, 2, 3, 1).cpu().numpy()  # [pairs, h, w, 2]
                info = out["info"][-1].cpu().numpy()  # [pairs, 4, h, w]
                got = {}
                for j, k in enumerate(others):
                    key = "forward" if k == i + 1 else "backward"
                    got[key] = flow[j]
                    got[f"{key}_confidence"] = of.confidence(info[j], model.args.var_min, model.args.var_max)
                of.save_frame(job.raw_dir, shot.frames[i][0], **got)
                shot.release(i)
    shot.done(run, "WAFT-a1 tar-c-t")


if __name__ == "__main__":
    serve(main)
