"""NVIDIA Cosmos DiffusionRenderer worker. Runs inside third_party/diffusionrenderer/.venv
with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

Two nodes (job["node"]):

diffusionrenderer.inverse  frames -> G-buffers
    The shot is cut into windows of `max_frames` frames (8k+1, the tokenizer's causal
    chunk) overlapping by `overlap` frames; each window is placed on the model's 16:9
    working canvas (Canvas: aspect kept, borders mirrored), encoded by the Cosmos tokenizer and denoised by the
    inverse 7B DiT once per requested pass (`steps` Euler steps, 15 by default, no guidance, the same
    noise for every window and pass), then decoded. Windows are stitched with a
    linear cross-fade over the overlapping frames (depth is first fitted to the
    previous window by least-squares scale + offset on those frames). ->

    raw/frame_<n>.npz, at the INPUT resolution, every pass (the node has an output for each):
        basecolor  float32 [H,W,3]  base colour (albedo) 0..1, sRGB-encoded (BASECOLOR_SPACE)
        normal     float32 [H,W,3]  unit normals, OpenCV camera (+X right, +Y down, +Z forward),
                                    facing the camera (z < 0 on surfaces seen head-on)
        depth      float32 [H,W]    relative depth (RELATIVE_DEPTH), not metric
        roughness  float32 [H,W]    0..1
        metallic   float32 [H,W]    0..1
    raw/result.json

diffusionrenderer.relight  frames + HDRI -> relit frames
    Inverse rendering as above (all five passes, kept as latents), then per window the
    G-buffers are decoded, re-encoded as conditions together with the HDRI (Reinhard
    LDR, log and direction encodings, upstream's three environment inputs) and the
    forward 7B DiT renders the shot under the new light. Windows cross-fade the same
    way. The HDRI is sampled into the model's camera frame here with PyTorch (upstream
    does this with nvdiffrast, which is not installed: NVIDIA non-commercial licence). ->

    raw/frame_<n>.png  relit frames, sRGB 8-bit, input resolution
    raw/envmap_model.png  the environment exactly as the model received it (Reinhard LDR)
    raw/result.json

Parameters (job["params"], as the node defines them in adapters/diffusionrenderer/nodes.py):
    passes       inverse only: the material channels to compute (default: all five); each one is a model run
    resolution   working canvas width (16:9): 1280 / 1024 / 960 / 768 / 640 / 512
    max_frames   frames per window, 8k+1 (57 = training length, needs ~15 GB extra RAM on a 24 GB card)
    overlap      shared frames between windows, at most max_frames/2 (a short shot's window shrinks: kept within half)
    seed, steps  the sampler
    env_rotate relight only: turns the HDRI about the camera up axis; +90 moves what was straight ahead to the
                 camera's left
    exposure     relight only: HDRI brightness x 2^EV
Inputs: relight needs job["inputs"]["hdri"], an equirectangular EXR/HDR in the Lab2Shot light-probe orientation.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lab2shot_worker import (
    set_seed,
    check_node,
    fail,
    light_probe,
    limit_gpu_memory,
    load_job,
    offload,
    progress,
    read_frame,
    resident,
    serve,
    shown,
    say,
)
from lab2shot_worker.frame_io import Writer
from lab2shot_worker.run import Run

ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER))
import shims  # noqa: E402

PASSES = ("basecolor", "normal", "depth", "roughness", "metallic")
# upstream rendering_utils.GBUFFER_INDEX_MAPPING: the inverse model's pass selector
CONTEXT_INDEX = {"basecolor": 0, "metallic": 1, "roughness": 2, "normal": 3, "depth": 4}
INVERSE_MODEL = "Diffusion_Renderer_Inverse_Cosmos_7B"
FORWARD_MODEL = "Diffusion_Renderer_Forward_Cosmos_7B"
TOKENIZER_DIR = "Cosmos-Tokenize1-CV8x8x8-720p"

NATIVE_H, NATIVE_W = 704, 1280  # the resolution both models were trained at
WINDOWS = (1, 9, 17, 25, 33, 41, 49, 57)  # 8k+1 frames; 57 = training length
# canvas width -> height (16:9, multiples of 16 = 8x tokenizer x 2x DiT patches); 1280 x 704 = training size
CANVASES = {1280: 704, 1024: 576, 960: 528, 768: 432, 640: 352, 512: 288}
# GPU memory plan (RTX 4090, bf16): the 7B DiT + tokenizer weigh 14.2 GB; sampling adds
# ~3.6 GB at 704x1280 x 57 frames (scales with tokens); one tokenizer encode / decode adds about
# (700 + 105 * frames) MB at 704x1280 (scales with area). When DiT + tokenizer work would pass the budget,
# the DiT waits in system RAM (+14.5 GB RAM) while the tokenizer runs, as upstream's --offload_* flags do.
GPU_BUDGET_MB = 20 * 1024
DIT_WEIGHTS_MB = 14300
GPU_MARGIN_MB = 500
MODEL_FPS = 24  # temporal RoPE unit the models were trained with (not the plate's fps)
ENV_LOG_SCALE = 10000.0  # upstream hdr_mapping log_scale
ENV_REINHARD_MAX = 16.0  # upstream hdr_mapping reinhard max_point

BASECOLOR_SPACE = "sRGB-encoded (display-referred) 0..1, like a base-colour texture: convert sRGB->linear before rendering"
RELATIVE_DEPTH = (
    "relative depth 0..1 (0 = near, 1 = far; affine-invariant, the model normalises each window of max_frames "
    "frames to its own range; windows after the first are fitted to the previous one by scale+offset), not metric"
)
NORMAL_CONVENTION = "OpenCV camera space: +X right, +Y down, +Z forward (into the scene); unit length; facing the camera"


# ------------------------------------------------------------------ small helpers


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    out = cv2.resize(img, (w, h), interpolation=interp)
    return out[..., None] if img.ndim == 3 and out.ndim == 2 else out


@dataclass(frozen=True)
class Canvas:
    """The model's working frame: always 16:9 like the training videos (1280 x 704 at full size). The plate is
    scaled to fit inside (aspect kept) and the rest is filled by mirroring the plate's edges; results are cut
    back out of the box (y0, x0, fh, fw). Both models misbehave on other frame shapes (portrait or square
    canvases turn relit faces into colour noise) and treat black borders as open background."""

    h: int
    w: int
    y0: int
    x0: int
    fh: int
    fw: int

    @classmethod
    def fit(cls, in_h: int, in_w: int, width: int) -> "Canvas":
        h = CANVASES[width]
        scale = min(width / in_w, h / in_h)
        fw, fh = min(width, max(16, round(in_w * scale))), min(h, max(16, round(in_h * scale)))
        return cls(h, width, (h - fh) // 2, (width - fw) // 2, fh, fw)

    def place(self, img: np.ndarray) -> np.ndarray:
        img = resize(img, self.fw, self.fh)
        bottom, right = self.h - self.fh - self.y0, self.w - self.fw - self.x0
        if not (self.y0 or self.x0 or bottom or right):
            return img
        return cv2.copyMakeBorder(img, self.y0, bottom, self.x0, right, cv2.BORDER_REFLECT_101)

    def crop(self, img: np.ndarray) -> np.ndarray:
        return img[self.y0:self.y0 + self.fh, self.x0:self.x0 + self.fw]


def plan_windows(n: int, max_frames: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) frame ranges of length `max_frames`, as few as keep at least `overlap` shared frames
    between neighbours, spread evenly over the shot (so the spare frames widen every overlap a little)."""
    if n <= max_frames:
        return [(0, n)]
    count = math.ceil((n - overlap) / (max_frames - overlap))
    starts = [round(i * (n - max_frames) / (count - 1)) for i in range(count)]
    return [(s, s + max_frames) for s in starts]


def save_png(path: Path, rgb01: np.ndarray) -> None:
    part = path.with_name(path.stem + ".part.png")
    bgr = cv2.cvtColor((np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(part), bgr):
        raise OSError(f"Cannot write {path}")
    part.replace(path)


# ------------------------------------------------------------------ environment map


def latlong_vec(h: int, w: int, device) -> torch.Tensor:
    """upstream rendering_utils.latlong_vec, verbatim maths: [H,W,3] directions."""
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / h, 1.0 - 1.0 / h, h, device=device),
        torch.linspace(-1.0 + 1.0 / w, 1.0 - 1.0 / w, w, device=device),
        indexing="ij",
    )
    st, ct = torch.sin(gy * math.pi), torch.cos(gy * math.pi)
    sp, cp = torch.sin(gx * math.pi), torch.cos(gx * math.pi)
    return torch.stack((st * sp, ct, -st * cp), dim=-1)


def model_env_directions(h: int, w: int, device) -> torch.Tensor:
    """Direction (in the model's camera frame, OpenGL-style: +X right, +Y up, +Z towards the
    viewer) that each pixel of the model's environment input stands for. This is upstream's
    `envmap_vec` (fed to the model as env_nrm) and also the direction upstream samples the
    HDRI cube map with (utils_env_proj.process_projected_envmap, identity camera pose)."""
    return -latlong_vec(h, w, device).flip(0).flip(1)


def load_hdri(path: Path) -> np.ndarray:
    """Equirectangular EXR/HDR -> float32 [H,W,3] RGB, scene-linear, NaN/Inf cleaned."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        fail("E-DIFFUSIONRENDERER-HDRIREAD", path=shown(path))
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = cv2.cvtColor(img[..., :3].astype(np.float32), cv2.COLOR_BGR2RGB)
    img = np.nan_to_num(img, nan=0.0, posinf=65504.0, neginf=0.0)
    return np.clip(img, 0.0, 65504.0)


def project_hdri(hdri: np.ndarray, h: int, w: int, env_rotate: float, device) -> torch.Tensor:
    """A Lab2Shot HDRI (lab2shot_worker.light_probe.ORIENTATION: centre column = where the camera looks,
    u = 0.75 camera right, top = camera up) -> the model's [h,w,3] linear environment image.

    For every model env pixel: its direction d in the camera frame (model_env_directions),
    turned by -rotation about the up axis (so the environment turns by +rotation), then
    looked up in our map: the inverse of light_probe.latlong_directions,
    u = atan2(-d.x, d.z) / 2pi (mod 1), v = acos(d.y) / pi.
    Bilinear lookup with wrap-around in u on a copy pre-shrunk (area filter) to about 2x the
    target, so bright small sources are not lost."""
    th, tw = hdri.shape[:2]
    pre_w = min(tw, 2 * w)
    pre_h = max(2, round(th * pre_w / tw))
    src = cv2.resize(hdri, (pre_w, pre_h), interpolation=cv2.INTER_AREA) if pre_w < tw else hdri
    d = model_env_directions(h, w, device)
    a = math.radians(env_rotate)
    c, s = math.cos(a), math.sin(a)
    # d_query = R_y(-a) d
    x = c * d[..., 0] - s * d[..., 2]
    z = s * d[..., 0] + c * d[..., 2]
    y = d[..., 1]
    u = torch.remainder(torch.atan2(-x, z) / (2 * math.pi), 1.0)
    v = torch.acos(y.clamp(-1, 1)) / math.pi
    img = torch.from_numpy(np.ascontiguousarray(src)).to(device).permute(2, 0, 1)[None]  # [1,3,H,W]
    img = torch.cat([img[..., -1:], img, img[..., :1]], dim=-1)  # wrap one column each side
    sw = img.shape[-1]
    gx = ((u * pre_w + 1.0) + 0.5) / sw * 2 - 1  # +1: the padded column; pixel centres
    gy = v * 2 - 1
    grid = torch.stack((gx, gy), dim=-1)[None].float()
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0].permute(1, 2, 0).clamp(min=0)  # [h,w,3]


def env_conditions(env_hdr: torch.Tensor, frames: int) -> dict[str, torch.Tensor]:
    """upstream utils_env_proj.hdr_mapping + inference_forward_renderer: three [1,3,T,H,W] in [-1,1]."""

    def srgb(x):
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp(min=0.0031308) ** (1 / 2.4) - 0.055)

    reinhard = env_hdr * (1 + env_hdr / ENV_REINHARD_MAX**2) / (1 + env_hdr)
    ldr = srgb(reinhard.clamp(0, 1))
    log = srgb(torch.log1p(env_hdr) / math.log1p(ENV_LOG_SCALE)).clamp(0, 1)
    h, w = env_hdr.shape[:2]
    nrm = model_env_directions(h, w, env_hdr.device)

    def video(img, scale):
        v = img * 2 - 1 if scale else img
        return v.permute(2, 0, 1)[None, :, None].expand(1, 3, frames, h, w)

    return {"env_ldr": video(ldr, True), "env_log": video(log, True), "env_nrm": video(nrm, False), "_ldr": ldr}


# ------------------------------------------------------------------ models


@resident
def load_renderer(weights: Path, name: str, device: torch.device):
    """One Cosmos DiffusionRenderer DiT + the shared tokenizer: the network built directly on the GPU in bf16, the
    fp32 checkpoint copied in (memory-mapped)."""
    from cosmos_predict1.diffusion.inference.inference_utils import load_model_by_config, skip_init_linear
    from cosmos_predict1.diffusion.model.model_diffusion_renderer import DiffusionRendererModel

    model = load_model_by_config(
        config_job_name=name,
        config_file="cosmos_predict1/diffusion/config/diffusion_renderer_config.py",
        model_class=DiffusionRendererModel,
    )
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with skip_init_linear(), device:
            model.set_up_model()
    finally:
        torch.set_default_dtype(default_dtype)
    ckpt = weights / name / "model.pt"
    state = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    if "model" in state:
        state = state["model"]
    state = {k: v for k, v in state.items() if "_extra_state" not in k}
    missing, unexpected = model.model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "_extra_state" not in k]
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="Cosmos DiffusionRenderer", extension="diffusionrenderer",
             model=shown(ckpt), missing=len(missing), unexpected=len(unexpected), examples=list(missing[:3]))
    if unexpected:
        say("I-WORKER-UNUSEDWEIGHTS", project="Cosmos DiffusionRenderer", model=shown(ckpt),
            count=len(unexpected), examples=list(unexpected[:3]))
    del state
    model.model.eval().requires_grad_(False)
    with skip_init_linear():
        model.set_up_tokenizer(str(weights / TOKENIZER_DIR))
    model.tokenizer.to(device)
    gc.collect()
    return model


class Renderer:
    """The DiT this job works with (inverse, then forward for relighting) + the shared tokenizer, with optional
    offloading. The models stay loaded between jobs (load_renderer); what a job sets on them it sets every time."""

    def __init__(self, weights: Path, device: torch.device):
        self.weights, self.device = weights, device
        self.model = None
        self.name = None
        self.max_frames = None
        self.offload = False
        self.net_on_gpu = False

    def load(self, name: str) -> float:
        """Switch to the DiT `name`; the previous one leaves the GPU (to RAM for the next job, or freed)."""
        t0 = time.time()
        previous, self.model = self.model, None
        if previous is not None:
            offload(previous)
            del previous
        self.model, self.name = load_renderer(self.weights, name, self.device), name
        self.net_on_gpu = next(self.model.model.parameters()).device.type == "cuda"  # an earlier job may have left it in RAM
        self.max_frames = None
        return time.time() - t0

    def set_window(self, frames: int) -> None:
        """The tokenizer encodes/decodes whole chunks of `frames` (8k+1) pixel frames."""
        if frames == self.max_frames:
            return
        vae = self.model.tokenizer.video_vae
        vae._pixel_chunk_duration = frames
        if frames > 1:
            vae.register_mean_std(str(self.weights / TOKENIZER_DIR))
            vae.latent_mean = vae.latent_mean.to(self.device)
            vae.latent_std = vae.latent_std.to(self.device)
        self.max_frames = frames

    # offloading: the 7B DiT (14.5 GB bf16) leaves the GPU while the tokenizer decodes/encodes

    def net_to_gpu(self) -> None:
        if not self.net_on_gpu:
            self.model.model.to(self.device)
            self.net_on_gpu = True

    def net_to_cpu(self) -> None:
        if self.offload and self.net_on_gpu:
            self.model.model.to("cpu")
            self.net_on_gpu = False
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """[1,3,T,H,W] in [-1,1] -> latent [1,16,(T-1)/8+1,H/8,W/8] (scaled by sigma_data)."""
        self.set_window(video.shape[2])
        return self.model.encode(video.to(self.device, torch.bfloat16)).contiguous()

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """latent -> [T,H,W,3] in [-1,1] on the GPU, bf16 (the tokenizer's own precision)."""
        self.set_window((latent.shape[2] - 1) * 8 + 1)
        out = self.model.decode(latent.to(self.device, torch.bfloat16))
        return out[0].permute(1, 2, 3, 0).clamp(-1, 1)

    @torch.no_grad()
    def sample(self, latent_condition: torch.Tensor, shape: tuple, context_index: int | None, steps: int, seed: int):
        """upstream DiffusionRendererModel.generate_samples_from_batch (guidance 0) with the
        latent condition computed once per window; the noise depends on the seed only."""
        self.net_to_gpu()
        m = self.model
        _, _, t, h, w = shape
        batch = {
            "t5_text_embeddings": torch.zeros(1, 512, 1024, device=self.device, dtype=torch.bfloat16),
            "t5_text_mask": torch.cat([torch.ones(1, 1), torch.zeros(1, 511)], 1).to(self.device, torch.bfloat16),
            "fps": torch.full((1,), float(MODEL_FPS), device=self.device, dtype=torch.bfloat16),
            "num_frames": torch.full((1,), float(t), device=self.device, dtype=torch.bfloat16),
            "image_size": torch.tensor([[h * 8, w * 8]], device=self.device, dtype=torch.bfloat16),
            "padding_mask": torch.zeros(1, 1, h * 8, w * 8, device=self.device, dtype=torch.bfloat16),
            "latent_condition": latent_condition,
            "context_index": torch.full((1,), context_index or 0, device=self.device, dtype=torch.long),
        }
        condition, _ = m.conditioner.get_condition_uncondition(batch)
        m.scheduler.set_timesteps(steps)
        g = set_seed(seed)
        xt = torch.randn(shape, generator=g) * m.scheduler.init_noise_sigma
        for step_t in m.scheduler.timesteps:
            xt = xt.to(**m.tensor_kwargs)
            x_in = m.scheduler.scale_model_input(xt, timestep=step_t)
            tt = step_t.to(**m.tensor_kwargs)
            out = m.net(x=x_in, timesteps=tt, **condition.to_dict())
            xt = m.scheduler.step(out, tt, xt).prev_sample
        return xt

    def condition(self, videos: dict[str, torch.Tensor]) -> torch.Tensor:
        """upstream prepare_diffusion_renderer_latent_conditions (inference mode)."""
        keys = self.model.condition_keys
        self.set_window(next(iter(videos.values())).shape[2])
        return self.model.prepare_diffusion_renderer_latent_conditions(
            videos, condition_keys=keys, condition_drop_rate=0,
            append_condition_mask=self.model.append_condition_mask,
            dtype=torch.bfloat16, device=self.device, latent_shape=None, mode="inference",
        )


# ------------------------------------------------------------------ G-buffer post-processing


def gbuffer_arrays(name: str, video01: np.ndarray) -> np.ndarray:
    """Model output (0..1, [T,H,W,3]) -> the npz value on the working canvas."""
    if name == "basecolor":
        return video01.astype(np.float32)
    if name == "normal":
        n = video01 * 2 - 1  # model camera frame: +X right, +Y up, +Z towards the viewer
        n = n * np.array([1.0, -1.0, -1.0], dtype=np.float32)  # -> OpenCV: +Y down, +Z forward
        return n.astype(np.float32)
    return video01.mean(-1).astype(np.float32)  # grey passes come back as three equal channels


def normalize_normals(n: np.ndarray) -> np.ndarray:
    """Upstream's normalisation (cosmos_predict1/diffusion/inference/diffusion_renderer_pipeline.py:163-172): not a
    hard normalise but a blend by length — vectors shorter than 0.2 stay as they are, 0.2–0.4 blend linearly, longer
    than 0.4 become unit length.

    The model answers pixels it is unsure of with short vectors; stretching every one to unit length would turn that
    noise into full-strength normals, and flat or shadowed areas would get coloured speckle."""
    length = np.linalg.norm(n, axis=-1, keepdims=True)
    unit = n / np.maximum(length, 1e-12)
    blend = np.clip((length - 0.2) / (0.4 - 0.2), 0.0, 1.0)
    return unit * blend + n * (1.0 - blend)


def fit_scale_offset(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """a, b minimising |a*src + b - dst|^2."""
    x, y = src.reshape(-1).astype(np.float64), dst.reshape(-1).astype(np.float64)
    vx = x.var()
    if vx < 1e-12:
        return 1.0, float(y.mean() - x.mean())
    a = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
    if not 0.2 < a < 5.0:  # a degenerate fit would wreck the window: fall back to an offset
        a = 1.0
    return a, float(y.mean() - a * x.mean())


class Stitcher:
    """Cross-fades consecutive windows over their shared frames and hands out finished frames."""

    def __init__(self, windows: list[tuple[int, int]], align: tuple[str, ...] = ()):
        self.windows = windows
        self.align = align
        self.prev = None  # (start, end, {name: [T,...]})

    def add(self, k: int, out: dict[str, np.ndarray]):
        s, e = self.windows[k]
        if self.prev is not None:
            ps, pe, pout = self.prev
            ov = pe - s
            if ov > 0:
                for name in self.align:
                    if name in out:
                        a, b = fit_scale_offset(out[name][:ov], pout[name][s - ps:])
                        out[name] = out[name] * a + b
                for i in range(ov):
                    wgt = (i + 1) / (ov + 1)
                    for name in out:
                        out[name][i] = pout[name][s - ps + i] * (1 - wgt) + out[name][i] * wgt
        next_s = self.windows[k + 1][0] if k + 1 < len(self.windows) else e
        self.prev = (s, e, out)
        return [(f, {n: v[f - s] for n, v in out.items()}) for f in range(s, min(next_s, e))]


# ------------------------------------------------------------------ jobs


INVERSE_NODE, RELIGHT_NODE = "diffusionrenderer.inverse", "diffusionrenderer.relight"


def needs_offload(hw: tuple[int, int], max_frames: int, budget_mb: int) -> bool:
    """True when the DiT and one tokenizer pass (or sampling) do not fit on the GPU together."""
    area = hw[0] * hw[1] / (NATIVE_H * NATIVE_W)
    tokenizer_mb = (700 + 105 * max_frames) * area
    tokens = ((max_frames - 1) // 8 + 1) * (hw[0] // 16) * (hw[1] // 16)
    sampling_mb = 3600 * tokens / (8 * 44 * 80)
    return DIT_WEIGHTS_MB + max(tokenizer_mb, sampling_mb) + GPU_MARGIN_MB > budget_mb


def effective_window(n: int, max_frames: int) -> int:
    """Short shots use the smallest 8k+1 window that holds them (padding repeats the last frame)."""
    if n >= max_frames:
        return max_frames
    return min(w for w in WINDOWS if w >= n)


def load_window(frames, s, e, max_frames, canvas: Canvas) -> torch.Tensor:
    imgs = [canvas.place(read_frame(frames[i][1], dtype="float32")) for i in range(s, e)]
    imgs += [imgs[-1]] * (max_frames - len(imgs))
    v = torch.from_numpy(np.stack(imgs)).permute(3, 0, 1, 2)[None]  # [1,3,T,H,W]
    return v * 2 - 1


def run_inverse_windows(r: Renderer, frames, windows, max_frames, canvas: Canvas, passes, steps, seed, keep_latents: bool,
                        on_window):
    """Inverse model over all windows. on_window(k, {pass: [T,H,W,3] in [-1,1] np}) or latents kept."""
    kept = []
    h, w = canvas.h, canvas.w
    shape = (1, 16, (max_frames - 1) // 8 + 1, h // 8, w // 8)
    total = len(windows) * len(passes)
    done = 0
    for k, (s, e) in enumerate(windows):
        video = load_window(frames, s, e, max_frames, canvas)
        r.net_to_cpu()  # only when offloading: the tokenizer encodes without the DiT on the GPU
        cond = r.condition({"rgb": video})
        latents = {}
        for name in passes:
            latents[name] = r.sample(cond, shape, CONTEXT_INDEX[name], steps, seed)
            done += 1
            progress(done, total, f"逆渲染 第 {k + 1}/{len(windows)} 段 {name}")
        del cond
        if keep_latents:
            kept.append({n: l.cpu() for n, l in latents.items()})
            continue
        r.net_to_cpu()
        decoded = {n: r.decode(l)[: e - s].float().cpu().numpy() for n, l in latents.items()}
        on_window(k, decoded)
    return kept


def run(job) -> None:
    node = check_node(job, INVERSE_NODE, RELIGHT_NODE)
    run = Run.start(job, node, "Cosmos DiffusionRenderer")
    p = run.params
    shot = run.frames()
    frames, n = shot.pairs, len(shot)
    in_h, in_w = shot.height, shot.width
    canvas = Canvas.fit(in_h, in_w, p["resolution"])
    hw = (canvas.h, canvas.w)
    max_frames = effective_window(n, p["max_frames"])
    overlap = min(p["overlap"], max_frames // 2) if max_frames > 1 else 0
    windows = plan_windows(n, max_frames, overlap)

    weights = job.weights_dir
    need = [INVERSE_MODEL] + ([FORWARD_MODEL] if node == RELIGHT_NODE else [])
    run.weights(*(weights / name / "model.pt" for name in need), weights / TOKENIZER_DIR / "decoder.jit")
    hdri = None
    if node == RELIGHT_NODE:
        hdri_path = job.inputs.get("hdri")
        if hdri_path is None:
            fail("E-DIFFUSIONRENDERER-NOHDRI")
        if not hdri_path.is_file():
            fail("E-DIFFUSIONRENDERER-HDRIMISSING", path=shown(hdri_path))
        hdri = load_hdri(hdri_path) * (2.0 ** p["exposure"])

    raw = job.raw_dir
    device = torch.device("cuda")
    shims.install()
    os.chdir(job.repo_dir)
    if str(job.repo_dir) not in sys.path:
        sys.path.insert(0, str(job.repo_dir))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    run.gpu_cap_mb = limit_gpu_memory()  # before the models: the offload decision reads the cap (run.model would set it later)
    r = Renderer(weights, device)
    r.offload = needs_offload(hw, max_frames, min(run.gpu_cap_mb, GPU_BUDGET_MB))
    if r.offload:
        say("N-DIFFUSIONRENDERER-OFFLOAD", width=hw[1], height=hw[0], max_frames=max_frames, budget_gb=GPU_BUDGET_MB // 1024)
    timings = {}

    with run.loading("逆渲染模型（7B）"):
        timings["load_inverse_s"] = round(r.load(INVERSE_MODEL), 1)
    run.stage("逆渲染（拆 G-buffer）")
    t0 = time.time()

    if node == INVERSE_NODE:
        # the channels the node asked for (its wired outputs): each one is a run of the 7B model, so the ones nobody
        # is wired to are not computed at all
        passes = [name for name in PASSES if name in (p.get("passes") or PASSES)] or list(PASSES)
        stitch = Stitcher(windows, align=("depth",))
        writer = Writer(threads=2, max_pending=8)

        def on_window(k, decoded):
            out = {name: gbuffer_arrays(name, (v + 1) / 2) for name, v in decoded.items()}
            for f, vals in stitch.add(k, out):
                arrays = {}
                for name, v in vals.items():
                    v = resize(canvas.crop(v), in_w, in_h)
                    v = normalize_normals(v) if name == "normal" else np.clip(v, 0.0, 1.0)
                    arrays[name] = v.astype(np.float32)
                writer.npz(raw / f"frame_{frames[f][0]}.npz", **arrays)

        with writer:
            run_inverse_windows(r, frames, windows, max_frames, canvas, passes, p["steps"], p["seed"], False, on_window)
        timings["inverse_s"] = round(time.time() - t0, 1)
        run.finish(
            [f for f, _ in frames],
            kind="gbuffers",
            node=node,
            model=INVERSE_MODEL,
            files="frame_<n>.npz: " + ", ".join(passes),
            passes=passes,
            basecolor=BASECOLOR_SPACE,
            normal=NORMAL_CONVENTION,
            depth=RELATIVE_DEPTH,
            roughness="0..1 (Disney/principled BRDF roughness)",
            metallic="0..1",
            **common_result(p, in_h, in_w, canvas, max_frames, overlap, windows, timings, r),
            frames=[f for f, _ in frames],  # the whole list, not the standard [first, last]: a file per frame
        )
        return

    # ---------------------------------------------------------------- relight
    latents = run_inverse_windows(r, frames, windows, max_frames, canvas, list(PASSES), p["steps"], p["seed"], True, None)
    timings["inverse_s"] = round(time.time() - t0, 1)
    with run.loading("正向渲染模型（7B）"):
        timings["load_forward_s"] = round(r.load(FORWARD_MODEL), 1)
    run.stage("按 HDRI 重打光")
    t0 = time.time()
    h, w = hw
    env_hdr = project_hdri(hdri, h, w, p["env_rotate"], device)
    env = env_conditions(env_hdr, max_frames)
    save_png(raw / "envmap_model.png", env.pop("_ldr").cpu().numpy())
    shape = (1, 16, (max_frames - 1) // 8 + 1, h // 8, w // 8)
    stitch = Stitcher(windows)
    writer = Writer(threads=2, max_pending=8)
    env_latents = None
    for k, (s, e) in enumerate(windows):
        r.net_to_cpu()
        if env_latents is None:
            env_latents = {key: r.encode(val) for key, val in env.items()}
        cond = []
        for key in r.model.condition_keys:  # upstream's order, each followed by an all-ones mask channel
            if key in env_latents:
                lat = env_latents[key]
            else:  # the G-buffer video, as upstream feeds the inverse renderer's saved frames
                video = r.decode(latents[k][key])
                if key not in ("basecolor", "normal"):
                    video = video.mean(-1, keepdim=True).expand_as(video)
                lat = r.encode(video.permute(3, 0, 1, 2)[None])
                del video
            cond += [lat, torch.ones_like(lat[:, :1])]
        cond = torch.cat(cond, dim=1)
        latent = r.sample(cond, shape, None, p["steps"], p["seed"])
        del cond
        r.net_to_cpu()
        rgb = ((r.decode(latent)[: e - s].float() + 1) / 2).cpu().numpy()
        progress(k + 1, len(windows), f"重打光 第 {k + 1}/{len(windows)} 段")
        for f, vals in stitch.add(k, {"rgb": rgb}):
            img = resize(canvas.crop(vals["rgb"]), in_w, in_h)
            writer.submit(save_png, raw / f"frame_{frames[f][0]}.png", img)
    writer.close()
    timings["forward_s"] = round(time.time() - t0, 1)
    run.finish(
        [f for f, _ in frames],
        kind="relit",
        node=node,
        model=[INVERSE_MODEL, FORWARD_MODEL],
        files="frame_<n>.png: relit frame, sRGB 8-bit, input resolution; envmap_model.png: the environment the model saw",
        hdri=shown(job.inputs["hdri"]),
        env_rotate=p["env_rotate"],
        exposure_ev=p["exposure"],
        hdri_orientation=light_probe.ORIENTATION,
        rotation=(
            "env_rotate turns the environment about the camera up axis, right-handed: "
            "+90 moves what was straight ahead to the camera's left"
        ),
        **common_result(p, in_h, in_w, canvas, max_frames, overlap, windows, timings, r),
        frames=[f for f, _ in frames],  # the whole list, not the standard [first, last]: a file per frame
    )


def common_result(p, in_h, in_w, canvas: Canvas, max_frames, overlap, windows, timings, r) -> dict:
    return {
        "width": in_w,
        "height": in_h,
        "working_size": {"width": canvas.w, "height": canvas.h},
        "frame_box": {"x": canvas.x0, "y": canvas.y0, "width": canvas.fw, "height": canvas.fh,
                      "note": "where the plate sits in the 16:9 working canvas; the rest is mirrored plate edges"},
        "resolution": p["resolution"],
        "max_frames": max_frames,
        "overlap": overlap,
        "windows": [list(w) for w in windows],
        "seed": p["seed"],
        "steps": p["steps"],
        "offload": r.offload,
        "gpu_budget_mb": GPU_BUDGET_MB,
        "consistency": (
            "each window of max_frames frames is one video diffusion pass (temporally coherent inside); consecutive "
            "windows share `overlap` frames, use the same noise (seed), and are cross-faded linearly over the shared frames"
        ),
        **timings,
        "notice": "Built on NVIDIA Cosmos. Models licensed by NVIDIA Corporation under the NVIDIA Open Model License",
    }


def main(job_path: str) -> None:
    run(load_job(job_path))


if __name__ == "__main__":
    serve(main)
