"""VideoMaMa worker: mask-guided video matting with a video diffusion prior. Runs
inside third_party/videomama/.venv; never imports Lab2Shot core.

    python worker.py <job.json>

Contract: lab2shot_worker/matte.py (shared with MatAnyone 2).
VideoMaMa needs a guide mask on EVERY frame (a frame without one gets an empty
guide). The computation is upstream's VideoInferencePipeline.run
(pipeline_svd_mask.py at the pinned commit), one denoising step of Stable Video
Diffusion's spatio-temporal UNet fine-tuned by VideoMaMa:

    frames -> [-1,1], guide -> binary 0/1 -> [-1,1] (3 channels)
    latents = VAE.encode(frames), VAE.encode(guide)     (SVD-XT temporal VAE)
    pred = UNet(cat[noise(seed 42), frame latents, guide latents], t=1,
                encoder_hidden_states=0, time ids (fps 7, motion 127, noise aug 0))
    alpha = mean over RGB of VAE.decode(pred) (8 frames per decode call)

Differences from upstream, none of which changes the maths:
* the CLIP image encoder is not loaded: upstream replaces its embedding with zeros;
* frames go through the VAE encoder a few at a time (upstream: all at once);
* alpha stays float (upstream: 8-bit PIL images) and is resized back to the
  input size bilinearly.

Long shots: upstream runs one window of 16 frames (its default --num_frames).
Here the shot is cut into windows of WINDOW frames that overlap by OVERLAP;
the overlapping frames are cross-faded linearly, so GPU memory depends on the
window, not on the shot length. Processing size: the long side at resolution
(default 1024, upstream's 1024x576), snapped to multiples of 64.

Output: raw/frame_<n>.npz alpha float32 [H,W] 0..1 at the input resolution
(VideoMaMa predicts alpha only, no foreground colour).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import MemoryBound, fail, fit_size, progress, resident, serve, set_seed
from lab2shot_worker.matte import GUIDE_THRESHOLD, GuideMasks, Output, frame_reader, morph
from lab2shot_worker.run import Run

MULTIPLE = 64  # VAE /8, then three UNet downsamplings
# what the memory grows with: 处理尺寸 within the node's 256..1024 in 64s (1024 measured 12-15 GB, docs.md)
RESOLUTION = MemoryBound.parameter("resolution", (1024, 768, 512))
WINDOW = 16  # frames per UNet call (upstream --num_frames default)
OVERLAP = 4  # frames shared by consecutive windows, cross-faded
SEED = 42  # upstream --seed default
FPS_ID, MOTION_BUCKET, NOISE_AUG = 7, 127, 0.0  # upstream run() defaults
VAE_ENCODE_BATCH = 4
VAE_DECODE_CHUNK = 8  # upstream decodes 8 frames per call (the temporal decoder mixes them)
UNET_DIR = "SammyLim/VideoMaMa"
SVD_DIR = "stabilityai/stable-video-diffusion-img2vid-xt"


@resident
def load_models(weights: Path, device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        weights / SVD_DIR, subfolder="vae", variant="fp16", torch_dtype=dtype, local_files_only=True,
    )
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        weights / UNET_DIR, subfolder="unet", torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True,
    )
    return vae.to(device).eval(), unet.to(device).eval()


def windows(n: int) -> list[tuple[int, int]]:
    """[start, end) frame ranges of WINDOW frames, consecutive ones sharing >= OVERLAP frames."""
    if n <= WINDOW:
        return [(0, n)]
    out, start = [], 0
    while start + WINDOW < n:
        out.append((start, start + WINDOW))
        start += WINDOW - OVERLAP
    out.append((n - WINDOW, n))  # the last window ends on the last frame
    return out


class Matting:
    def __init__(self, vae, unet, size: tuple[int, int], dtype: torch.dtype, device: torch.device):
        self.vae, self.unet, self.size, self.dtype, self.device = vae, unet, size, dtype, device
        self.sf = vae.config.scaling_factor
        # upstream: zeros_like(CLIP image embedding).unsqueeze(1)
        self.context = torch.zeros((1, 1, unet.config.cross_attention_dim), device=device, dtype=dtype)
        self.time_ids = torch.tensor([[FPS_ID, MOTION_BUCKET, NOISE_AUG]], device=device, dtype=dtype)
        want = unet.config.addition_time_embed_dim * 3
        got = unet.add_embedding.linear_1.in_features
        if want != got:
            fail("E-VIDEOMAMA-UNETMISMATCH", want=want, got=got, dim=unet.config.addition_time_embed_dim)

    def resize(self, x: torch.Tensor) -> torch.Tensor:
        """[T,C,H,W] -> processing size; bilinear with antialiasing when shrinking, like PIL's BILINEAR."""
        if tuple(x.shape[-2:]) == self.size:
            return x
        shrink = x.shape[-2] > self.size[0] or x.shape[-1] > self.size[1]
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False, antialias=shrink)

    def encode(self, video: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """[T,3,h,w] in [-1,1] -> latents [T,4,h/8,w/8] (upstream: sample * sf / sf)."""
        parts = []
        for i in range(0, len(video), VAE_ENCODE_BATCH):
            dist = self.vae.encode(video[i: i + VAE_ENCODE_BATCH].to(self.dtype)).latent_dist
            parts.append(dist.sample(generator=generator))
        return torch.cat(parts)

    @torch.inference_mode()
    def __call__(self, rgbs: list[np.ndarray], guides: list[np.ndarray]) -> torch.Tensor:
        """uint8 [H,W,3] frames + uint8 0/1 [H,W] guides -> alpha float32 [T,h,w] (processing size) on the GPU."""
        dev = self.device
        frames = torch.from_numpy(np.stack(rgbs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
        frames = self.resize(frames) * 2.0 - 1.0
        masks = torch.from_numpy(np.stack(guides)).to(dev)[:, None].float()
        masks = (self.resize(masks) > 0.5).float() * 2.0 - 1.0  # upstream: resize, then threshold at 127
        masks = masks.expand(-1, 3, -1, -1)

        vae_gen = set_seed(SEED, dev)
        cond = self.encode(frames, vae_gen)
        guide = self.encode(masks, vae_gen)
        noise = torch.randn(cond.shape, generator=set_seed(SEED, dev), device=dev,
                            dtype=self.dtype)
        timesteps = torch.full((1,), 1.0, device=dev, dtype=torch.long)
        x = torch.cat([noise, cond, guide], dim=1)[None]  # [1,T,12,h/8,w/8]
        pred = self.unet(x, timesteps, self.context, added_time_ids=self.time_ids).sample[0] / self.sf
        alphas = []
        for i in range(0, len(pred), VAE_DECODE_CHUNK):
            chunk = pred[i: i + VAE_DECODE_CHUNK]
            decoded = self.vae.decode(chunk, num_frames=len(chunk)).sample.float()
            alphas.append((decoded / 2.0 + 0.5).clamp(0, 1).mean(dim=1))
        return torch.cat(alphas)


def main(job_path: str) -> None:
    run = Run.start(job_path, "videomama.matte", "VideoMaMa")
    job, params = run.job, run.params
    resolution, erode_dilate, fp16 = params["resolution"], params["erode_dilate"], params["fp16"]
    # Like upstream's --mixed_precision (fp16 | bf16): float32 gives the same alpha (max difference
    # 0.03, mean 1e-5 on a test shot) but needs ~22 GB, too close to a 24 GB card.
    dtype = torch.float16 if fp16 else torch.bfloat16
    weights = job.weights_dir
    run.weights(weights / UNET_DIR / "unet", weights / SVD_DIR / "vae")

    frames = run.frames()
    numbers, height, width = frames.numbers, frames.height, frames.width
    guides = GuideMasks(job, height, width)
    no_guide = guides.missing(numbers)  # result.json 记下缺了几帧

    device = torch.device("cuda")
    vae, unet = run.model("VideoMaMa", load_models, weights, device, dtype)

    def guide_of(frame: int) -> np.ndarray:
        g = guides.get(frame)
        if g is None:
            return np.zeros((height, width), np.uint8)
        return morph((g > GUIDE_THRESHOLD).astype(np.uint8), erode_dilate)

    run.stage("抠像")
    reader = frame_reader(frames.paths, (height, width))
    spans = windows(len(frames))

    def matte_shot(side: int):
        """The whole shot, window by window, its long side scaled to `side`."""
        size = fit_size(width, height, side, MULTIPLE, upscale=False)[::-1]  # (h, w)
        matting = Matting(vae, unet, size, dtype, device)
        out = Output(job)
        tail: dict[int, torch.Tensor] = {}  # frames of the previous window that the current one overlaps
        try:
            # inference_mode for the whole loop: the cross-fade writes into the network output in place
            with torch.inference_mode():
                for w, (start, end) in enumerate(spans):
                    idx = list(range(start, end))
                    order = idx + list(range(end, min(len(frames), end + reader.ahead)))
                    rgbs = [reader.get(i, order[k:]) for k, i in enumerate(idx)]
                    gds = [guide_of(numbers[i]) for i in idx]
                    pad = WINDOW - len(idx) if len(frames) < WINDOW else 0  # upstream repeats the last frame
                    alpha = matting(rgbs + rgbs[-1:] * pad, gds + gds[-1:] * pad)[: len(idx)]
                    shared = [i for i in idx if i in tail]
                    for k, i in enumerate(shared):  # linear cross-fade from the previous window to this one
                        t = (k + 1) / (len(shared) + 1)
                        alpha[i - start] = (1 - t) * tail[i] + t * alpha[i - start]
                    next_start = spans[w + 1][0] if w + 1 < len(spans) else end
                    full = F.interpolate(alpha[:, None], size=(height, width), mode="bilinear", align_corners=False)[:, 0]
                    full = full.clamp_(0, 1).cpu().numpy()
                    for i in idx:
                        if i < next_start:
                            out.put(numbers[i], full[i - start])
                    tail = {i: alpha[i - start] for i in idx if i >= next_start}
                    progress(min(next_start, len(frames)), len(frames), "抠像")
        except BaseException:
            out.close()  # its writing thread, not kept for a run that failed
            raise
        return side, size, out

    try:
        resolution, size, out = run.fit(RESOLUTION, matte_shot, resolution)
    finally:
        reader.close()

    out.finish(
        run,
        numbers,
        model="VideoMaMa (SammyLim/VideoMaMa UNet + stable-video-diffusion-img2vid-xt VAE, one step)",
        guide="every frame (frames without a mask get an empty guide)",
        frames_without_guide=len(no_guide),
        width=width,
        height=height,
        processing_size={"width": size[1], "height": size[0]},
        window={"frames": WINDOW, "overlap": OVERLAP, "count": len(spans), "blend": "linear cross-fade"},
        params={"resolution": resolution, "erode_dilate": erode_dilate, "fp16": fp16},
        seed=SEED,
        foreground=False,
    )


if __name__ == "__main__":
    serve(main)
