"""BiRefNet worker: matte of the main subject(s), frame by frame. Runs inside
third_party/birefnet/.venv with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

Frames (display-referred sRGB PNGs) -> resize to the model's input size ->
official BiRefNet (models.birefnet at the pinned commit, weights from weights/)
-> sigmoid of the final prediction -> bilinear back to the frame size ->
raw/frame_<n>.npz:

    alpha  float32 [H,W]  0..1, at the input frame's resolution

Each frame on its own (BiRefNet has no temporal model); frames go through the
network in batches sized to the free GPU memory.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import MemoryBound, fail, fit_size, progress, read_frame, reason, resident, serve, say
from lab2shot_worker.frame_io import FrameReader, Writer
from lab2shot_worker.run import Run

# model param -> (Hugging Face repo, native input size: square side or None = any shape)
MODELS = {
    "general": ("ZhengPeng7/BiRefNet", 1024),
    "matting": ("ZhengPeng7/BiRefNet-matting", 1024),
    "hr": ("ZhengPeng7/BiRefNet_HR", 2048),
    "hr_matting": ("ZhengPeng7/BiRefNet_HR-matting", 2048),
    "dynamic": ("ZhengPeng7/BiRefNet_dynamic", None),
}
DYNAMIC_MAX = 2304  # BiRefNet_dynamic was trained on 256x256 .. 2304x2304
MULTIPLE = 32  # Swin-L: four stride-2 stages after a stride-4 patch embed
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Batches only save per-call overhead: on an RTX 4090 one frame already fills the GPU
# (1024: 86 ms/frame at batch 1, 2 or 4; 2048: 324 ms at batch 1, 350 ms at batch 4).
# So batches are kept small: at most MAX_BATCH frames and BATCH_BUDGET of activations,
# which also leaves room on a shared card (on Windows/WSL an overfull card does not
# raise out-of-memory, it silently spills into system RAM and gets several times slower).
MAX_BATCH = 4
BATCH = MemoryBound.batch((MAX_BATCH, 2, 1))  # each frame is matted on its own: a smaller batch only takes longer
BATCH_BUDGET = 8 * 2**30
GPU_MARGIN = 2 * 2**30  # keep this much of the free GPU memory unused


@resident
def load_model(repo: Path, checkpoint: Path, device: torch.device, fp16: bool):
    """The official model class at the pinned commit + a local safetensors file."""
    from safetensors.torch import load_file

    # The repo's config.py / models import each other as top-level modules
    # (`from config import Config`), and Config() looks for train.sh next to it.
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    cwd = os.getcwd()
    os.chdir(repo)
    try:
        from models.birefnet import BiRefNet
        from utils import check_state_dict

        model = BiRefNet(bb_pretrained=False)  # backbone weights come with the checkpoint
    finally:
        os.chdir(cwd)
    state = check_state_dict(load_file(str(checkpoint)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="BiRefNet", extension="birefnet", model=checkpoint.name,
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    if unexpected:
        say("I-WORKER-UNUSEDWEIGHTS", project="BiRefNet", model=checkpoint.name, count=len(unexpected),
            examples=list(unexpected[:3]))
    del state
    model = model.to(device).eval()
    # Weights always stay fp32; half precision comes from autocast, as upstream does: the bf16 in upstream
    # `config.py:10` is for `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)` (`inference.py:25`, inference
    # at :35 under `with autocast_ctx`). The model itself is fp32, and autocast falls back to fp32 for operators without
    # a bf16 kernel. Casting the whole model with `.to(torch.bfloat16)` removes that fallback, and the backbone's
    # deformable convolution raises `NotImplementedError: "deformable_im2col" not implemented for 'BFloat16'`
    # (torchvision has no bf16 deform_conv2d kernel).
    return model.float()


def input_size(model_id: str, resolution: int, height: int, width: int) -> tuple[int, int]:
    """(height, width) the network sees.

    `resolution` is always a number (the node's 「处理分辨率」 has no automatic option): the network runs at the
    value shown to the user."""
    native = MODELS[model_id][1]
    if native is not None:  # trained on square inputs (the frame is squeezed, as upstream does)
        side = fit_size(native, native, resolution, MULTIPLE)[0]
        return side, side
    # Any shape: keep the frame's aspect ratio, `resolution` is the long side.
    w, h = fit_size(width, height, min(resolution, DYNAMIC_MAX), MULTIPLE)
    return h, w


class Matte:
    def __init__(self, model, size: tuple[int, int], fp16: bool, device: torch.device):
        self.model, self.size, self.device = model, size, device
        # Inputs are always fed as fp32 and half precision is left to autocast (see load_model): casting tensors to
        # bf16 directly would also affect operators without bf16 support, while autocast falls back per operator
        self.autocast = fp16
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images: list[np.ndarray]) -> torch.Tensor:
        """[B] float32 RGB [H,W,3] (same size) -> alpha float32 [B,H,W] on the GPU."""
        h, w = images[0].shape[:2]
        x = torch.from_numpy(np.stack(images)).to(self.device, non_blocking=True).permute(0, 3, 1, 2)
        if (h, w) != self.size:
            # Antialiased bilinear, like upstream's PIL resize.
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        # upstream inference.py:25/35: autocast wraps the forward pass; model and inputs are fp32
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            pred = self.model(x)[-1]
        pred = pred.float().sigmoid()
        if pred.shape[-2:] != (h, w):
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred[:, 0].clamp_(0.0, 1.0)


def main(job_path: str) -> None:
    run = Run.start(job_path, "birefnet.matte", "BiRefNet")  # timing, GPU memory, progress and the standard result.json fields are recorded by Run
    job, params = run.job, run.params

    model_id, resolution, fp16 = params["model"], params["resolution"], params["fp16"]

    repo_id = MODELS[model_id][0]
    weights = job.weights_dir
    checkpoint = weights / repo_id / "model.safetensors"
    run.weights(checkpoint, what=f"模型 {repo_id} 的权重")

    frames = run.frames()
    numbers, height, width = frames.numbers, frames.height, frames.width
    raw = job.raw_dir
    device = torch.device("cuda")
    # No cudnn autotuning: measured no faster here, and its trial workspaces would
    # inflate the per-frame memory measured on the first batch.
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    model = run.model("BiRefNet 模型", load_model, job.repo_dir, checkpoint, device, fp16)

    size = input_size(model_id, resolution, height, width)
    matte = Matte(model, size, fp16, device)

    run.stage("抠像")
    frame_seconds = run.frame_seconds  # each batch's time is divided among its frames; Run derives seconds_per_frame from it
    coverage: list[float] = []
    empty: list[int] = []
    batch_size = 1  # the first batch measures the memory one frame needs
    per_frame_bytes = 0
    # Decode ahead of the GPU, but only a couple of batches (a long shot must not fill the RAM).
    with FrameReader(frames.paths, lambda path: read_frame(path, "float32"), threads=4,
                     ahead=2 * MAX_BATCH) as reader, Writer(threads=2, max_pending=16) as writer:

        def matte_batch(start: int, size: int):
            """The frames from `start`, `size` of them at once: (size, their indices, alpha, seconds, memory before)."""
            batch = list(range(start, min(start + size, len(frames))))
            rgbs = reader.take(batch)
            reader.request(range(batch[-1] + 1, min(batch[-1] + 1 + 2 * MAX_BATCH, len(frames))))
            if any(r.shape[:2] != (height, width) for r in rgbs):
                bad, shape = next((numbers[j], r.shape) for j, r in zip(batch, rgbs) if r.shape[:2] != (height, width))
                fail("E-WORKER-FRAMESIZE", width=shape[1], height=shape[0], where=reason("I-WORKER-ATFRAME", frame=bad),
                     first_width=width, first_height=height)
            torch.cuda.synchronize()
            t = time.time()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            alpha = matte(rgbs).cpu().numpy()
            return size, batch, alpha, time.time() - t, base

        i = 0
        while i < len(frames):
            batch_size, batch, alpha, elapsed, base = run.fit(
                BATCH, lambda size, start=i: matte_batch(start, size), batch_size)
            if i == 0:
                # Size the batches to the free memory (shared GPU: what is free right now).
                per_frame_bytes = max(1, int(1.25 * (torch.cuda.max_memory_allocated() - base)))
                torch.cuda.empty_cache()  # the batch-of-one blocks would not fit the larger batches
                free, _total = torch.cuda.mem_get_info()
                budget = min(BATCH_BUDGET, free - GPU_MARGIN)
                batch_size = int(max(1, min(MAX_BATCH, budget // per_frame_bytes)))
            for k, j in enumerate(batch):
                a = alpha[k]
                frame_seconds.append(elapsed / len(batch))
                coverage.append(float((a > 0.5).mean()))
                if a.max() < 0.5:
                    empty.append(numbers[j])
                writer.npz(raw / f"frame_{numbers[j]}.npz", alpha=a)
            i += len(batch)
            progress(i, len(frames), "抠像")

    if empty:
        say("N-BIREFNET-EMPTY", count=len(empty), frames=empty[:10], more="……" if len(empty) > 10 else "")

    run.finish(
        numbers,  # seconds / load_seconds / seconds_per_frame / gpu_* are Run's standard fields
        kind="matte",
        model=repo_id,
        model_key=model_id,
        files="frame_<n>.npz: alpha float32 [H,W] in 0..1 at the input resolution",
        input_size={"width": size[1], "height": size[0]},
        fp16=fp16,
        width=width,
        height=height,
        batch_size=batch_size,
        network_seconds_per_frame=round(float(np.mean(frame_seconds[1:] or frame_seconds)), 4),
        coverage={"min": min(coverage), "max": max(coverage), "mean": float(np.mean(coverage))},
        empty_frames=empty,
        gpu_per_frame_mb=round(per_frame_bytes / 2**20),
    )


if __name__ == "__main__":
    serve(main)
