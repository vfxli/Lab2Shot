"""Video Depth Anything worker: temporally consistent depth for one shot. Runs
inside third_party/videodepthanything/.venv; never imports Lab2Shot core.

    python worker.py <job.json>

Frames -> Video-Depth-Anything Small / Base / Large (relative or metric), in upstream's overlapping
32-frame windows (the offline mode of video_depth.py: 10 overlapping frames,
2 of them keyframes carried over for scale/shift alignment, 8 cross-faded) ->
raw/frame_<n>.npz (depth float32 [H, W] at the input resolution + "kind").

The windowing is re-implemented here as a stream so a shot of any length keeps
only ~40 frames in memory (upstream's infer_video_depth holds every frame and
every full-resolution depth map in RAM); the maths is upstream's, step for step,
using its own constants and helper functions.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import fail, progress, read_frame, reason, resident, say, serve, shown
from lab2shot_worker.frame_io import FrameReader, Writer
from lab2shot_worker.run import Run

NODE = "videodepthanything.depth"
# model param -> (checkpoint, metric model?, encoder). Small: Apache-2.0; Base / Large: CC-BY-NC-4.0 (non-commercial).
MODELS = {
    "small": ("video_depth_anything_vits.pth", False, "vits"),
    "metric_small": ("metric_video_depth_anything_vits.pth", True, "vits"),
    "base": ("video_depth_anything_vitb.pth", False, "vitb"),
    "metric_base": ("metric_video_depth_anything_vitb.pth", True, "vitb"),
    "large": ("video_depth_anything_vitl.pth", False, "vitl"),
    "metric_large": ("metric_video_depth_anything_vitl.pth", True, "vitl"),
}
# run.py's model_configs at the pinned commit (the same for relative and metric checkpoints).
MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
SIZE_NAME = {"vits": "Small", "vitb": "Base", "vitl": "Large"}
LICENSE = {"vits": "Apache-2.0", "vitb": "CC-BY-NC-4.0 (non-commercial)", "vitl": "CC-BY-NC-4.0 (non-commercial)"}
KIND = {
    True: "metric_depth_m",  # metres along the camera's optical axis (Z depth, not ray length)
    False: "relative_disparity",  # affine-invariant inverse depth: disparity * a + b, a/b unknown but fixed for the shot
}
UNITS = {
    True: "meters (camera Z depth)",
    False: "none: relative disparity (inverse depth up to one unknown scale and shift for the whole shot; larger = nearer)",
}


def use_sdpa() -> None:
    """DINOv2 attention through torch's fused kernel.

    Without xformers upstream falls back to q @ k^T with the full attention matrix
    (32 frames x 6 heads x ~2400^2 tokens); scaled_dot_product_attention is the
    same maths (scale head_dim^-0.5) without materializing it.
    """
    from video_depth_anything.dinov2_layers import attention as dino_attention

    def forward(self, x, attn_bias=None):
        if attn_bias is not None:
            raise NotImplementedError("attn_bias (nested tensors) is not used by Video Depth Anything")
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    dino_attention.Attention.forward = forward
    dino_attention.MemEffAttention.forward = forward


def make_transform(height: int, width: int, resolution: int):
    """Exactly infer_video_depth's preprocessing (incl. its narrower size for > 16:9 frames)."""
    from torchvision.transforms import Compose
    from video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

    ratio = max(height, width) / min(height, width)
    if ratio > 1.78:  # upstream: "we recommend to process video with ratio smaller than 16:9 due to memory limitation"
        resolution = int(resolution * 1.777 / ratio)
        resolution = round(resolution / 14) * 14
    return Compose([
        Resize(
            width=resolution,
            height=resolution,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])


def frame_loader(paths: list[Path], size: tuple[int, int], transform) -> FrameReader:
    """Frames decoded + preprocessed ahead of the GPU on a few threads (a window of 48, the network's pace)."""

    def load(path: Path) -> torch.Tensor:
        rgb = read_frame(path, "float32")
        if rgb.shape[:2] != size:
            fail("E-WORKER-FRAMESIZE", width=rgb.shape[1], height=rgb.shape[0],
                 where=reason("I-WORKER-ATFILE", file=shown(path)), first_width=size[1], first_height=size[0])
        return torch.from_numpy(transform({"image": rgb})["image"])

    return FrameReader(paths, load, threads=4, ahead=48)


def infer_stream(model, load, n: int, metric: bool, fp16: bool, device: str = "cuda"):
    """Upstream VideoDepthAnything.infer_video_depth as a generator.

    `load(i)` -> preprocessed frame i ([3, h, w]); yields (i, depth [h, w] float32 at
    network resolution) for i = 0..n-1 in order, each as soon as no later window can
    change it. Same windows, keyframes, alignment and cross-fade as upstream; the
    only difference is that the scale/shift fit sees network-resolution depth instead
    of depth already resized to the frame size.
    """
    from utils.util import compute_scale_and_shift, get_interpolate_frames
    from video_depth_anything.video_depth import INFER_LEN, INTERP_LEN, KEYFRAMES, OVERLAP

    frame_step = INFER_LEN - OVERLAP
    align_len = OVERLAP - INTERP_LEN
    kf_align_list = KEYFRAMES[:align_len]

    pre_input = None
    ref_align: list[np.ndarray] = []
    aligned: deque[np.ndarray] = deque()  # aligned depth of frames base .. base+len-1
    base = 0
    starts = list(range(0, n, frame_step))
    for w, frame_id in enumerate(starts):
        if pre_input is None:
            cur_input = torch.stack([load(frame_id + i) for i in range(INFER_LEN)])[None].to(device)
        else:
            # Slots 0..OVERLAP-1 are the previous window's KEYFRAMES (first frame of the
            # shot, a mid frame, and the 8 frames that overlap): only new frames are read.
            new = torch.stack([load(frame_id + i) for i in range(OVERLAP, INFER_LEN)])[None].to(device)
            cur_input = torch.cat([pre_input[:, KEYFRAMES], new], dim=1)

        with torch.no_grad(), torch.autocast(device_type=device, enabled=fp16):
            depth = model(cur_input)  # [1, T, h, w]
        depth = [d for d in depth[0].to(torch.float32).cpu().numpy()]

        if w == 0:
            aligned.extend(depth[:INFER_LEN])
            ref_align = [depth[kf] for kf in kf_align_list]
        else:
            curr_align = [depth[i] for i in range(len(kf_align_list))]
            if metric:
                scale, shift = 1.0, 0.0
            else:
                scale, shift = compute_scale_and_shift(
                    np.concatenate(curr_align), np.concatenate(ref_align), np.concatenate(np.ones_like(ref_align) == 1)
                )
            pre_depth = [aligned[-INTERP_LEN + i] for i in range(INTERP_LEN)]
            post_depth = [np.maximum(depth[i] * scale + shift, 0) for i in range(align_len, OVERLAP)]
            for i, d in enumerate(get_interpolate_frames(pre_depth, post_depth)):
                aligned[-INTERP_LEN + i] = d
            for i in range(OVERLAP, INFER_LEN):
                aligned.append(np.maximum(depth[i] * scale + shift, 0))
            ref_align = ref_align[:1] + [np.maximum(depth[kf] * scale + shift, 0) for kf in kf_align_list[1:]]
        pre_input = cur_input

        # The last INTERP_LEN frames get cross-faded by the next window; the rest is final.
        final = base + len(aligned) - (INTERP_LEN if w + 1 < len(starts) else 0)
        while base < min(final, n):
            yield base, aligned.popleft()
            base += 1


@resident
def load_model(checkpoint: Path, encoder: str, metric: bool):
    from video_depth_anything.video_depth import VideoDepthAnything

    model = VideoDepthAnything(**MODEL_CONFIGS[encoder], metric=metric)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to("cuda").eval()


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "Video Depth Anything")
    job, params = run.job, run.params
    model_name, resolution, fp16 = params["model"], params["resolution"], params["fp16"]

    if str(job.repo_dir) not in sys.path:
        sys.path.insert(0, str(job.repo_dir))
    use_sdpa()

    frames = run.frames()
    raw = job.raw_dir
    height, width = frames.height, frames.width

    checkpoint, metric, encoder = MODELS[model_name]
    ckpt_dir = job.weights_dir / "checkpoints"
    run.weights(ckpt_dir / checkpoint)
    model = run.model("load_model", load_model, ckpt_dir / checkpoint, encoder, metric, stage_params={"model": "Video Depth Anything"})

    transform = make_transform(height, width, resolution)
    loader = frame_loader(frames.paths, (height, width), transform)
    net_h, net_w = loader.get(0).shape[1:]
    if len(frames) < 32:
        say("N-VIDEODEPTHANYTHING-SHORTSHOT", count=len(frames))

    run.stage("estimate_depth")
    kind = KIND[metric]
    n = len(frames)
    writer = Writer(threads=2, max_pending=8)
    stats = {"min": np.inf, "max": -np.inf, "median": []}
    # anything past the last frame is the last frame (upstream pads the shot by repeating it)
    for i, depth in infer_stream(model, lambda k: loader.get(min(k, n - 1)), n, metric, fp16):
        # Back to the plate's resolution. align_corners=False puts pixel centres where
        # cv2's downscale took them from (upstream's align_corners=True shifts by <= 0.5 px).
        full = F.interpolate(
            torch.from_numpy(depth)[None, None].to("cuda"), size=(height, width), mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        stats["min"] = min(stats["min"], float(full.min()))
        stats["max"] = max(stats["max"], float(full.max()))
        stats["median"].append(float(np.median(depth)))
        writer.npz(raw / f"frame_{frames.numbers[i]}.npz", depth=full, kind=np.str_(kind))
        progress(i + 1, n, "depth")
    writer.close()
    loader.close()

    run.finish(
        frames.numbers,
        kind="depth",
        depth_kind=kind,
        model=f"Video-Depth-Anything-{'Metric-' if metric else ''}{SIZE_NAME[encoder]}",
        license=LICENSE[encoder],
        units=UNITS[metric],
        files="frame_<frame>.npz: depth float32 [height, width], kind",
        width=width,
        height=height,
        network_size=[int(net_w), int(net_h)],
        resolution=resolution,
        fp16=fp16,
        depth_range=[stats["min"], stats["max"]],
        median_per_frame=stats["median"],
    )


if __name__ == "__main__":
    serve(main)
