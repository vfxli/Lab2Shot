"""NVIDIA LuxDiT worker: HDR light probe from one frame. Runs inside
third_party/luxdit/.venv with the pinned repo on sys.path; never imports Lab2Shot core.

    python worker.py <job.json>

The reference frame -> centre crop at the model's
input size -> upstream RGBXEnvCogVideoXPipeline (CogVideoX-5B DiT fine-tuned, real-scene
LoRA) conditioned on the plate and on the direction map of a 128 x 256 lat-long ->
dual tone-mapped envmap (Reinhard LDR + log) -> upstream HDR merge MLP
-> the map, re-projected to Lab2Shot's (DiffusionLight's) lat-long
convention and resized to envmap_width ->

    raw/envmap.exr    float32 RGB lat-long HDR, scene-linear, Rec.709 primaries,
                      envmap_width x envmap_width/2, camera-relative (ORIENTATION)
    raw/preview.png   the same map tone-mapped for display (sRGB, Reinhard white 16, as upstream's *_ldr.png)
    raw/input.png     what the model saw of the reference frame (centre crop at the model's input size)
    raw/result.json

Parameters (job["params"]):
    seed             int >= 0     0
    envmap_width     even int >= 64, default 1024 (the model's own map is 256 x 128; larger is interpolated)
    lora_scale       0..1, default 0.8 (upstream's real-scene setting; 0 = synthetic-domain model)
    steps            20 | 30 | 50 | null, null = 50 (upstream's setting)
    guidance_scale   1..10, default 2.5
    resolution       "auto" | "480x720" | "512x512" | "720x480" (height x width), default auto

The light-probe node sends only the probe frame (probe_plate): LuxDiT's single-frame model. (Upstream's video
model is not used: several times slower and not better on real plates.)

GPU (RTX 4090, bf16): ~13.3 GB peak, ~25-30 s.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lab2shot_worker import light_probe  # noqa: E402
from lab2shot_worker import limit_gpu_memory, progress, read_frame, resident, say, serve, set_seed
from lab2shot_worker.run import Run
from PIL import Image

ENV_RES = (128, 256)  # the model's lat-long (upstream --env_resolution default, the training size)
INPUT_SIZES = {"480x720": (480, 720), "512x512": (512, 512), "720x480": (720, 480)}
REINHARD_WHITE = 16.0  # upstream process_hdr(max_ldr=16): the *_ldr.png tone curve
NODE = "luxdit.light_probe"
MIN_GPU_MB = 16 * 1024  # warn below this much free GPU memory (the 5B bf16 transformer alone is 11 GB)

MODEL = "LuxDiT/luxdit_image"
HDR_MERGER = "LuxDiT/hdr_merge_mlp"
BASE = "CogVideoX-5b-I2V"  # vae/ + scheduler/
DEFAULT_STEPS = 50  # upstream README command

COLORSPACE = {
    "name": "lin_rec709",
    "ocio_hint": "Linear Rec.709 (sRGB)",
    "primaries": "Rec.709 / sRGB (x,y R 0.64,0.33 G 0.30,0.60 B 0.15,0.06), white D65 (0.3127,0.3290)",
    "transfer": (
        "scene-linear. The model paints a Reinhard (white 16) sRGB map and a log1p/log1p(10000) sRGB map; "
        "upstream's HDR merge MLP turns the pair back into linear radiance"
    ),
    "scale": (
        "relative, not absolute: trained on HDRIs whose perspective crops were tone-mapped (sRGB / Filmic / AgX, "
        "sometimes auto-exposed to median 0.5) into the input image, so linear 1.0 is roughly the plate's own "
        "display white. The log channel covers up to 10000"
    ),
    "note": "rendered/HDRI training data in linear sRGB; plates are taken as display sRGB, so the primaries are nominal",
}

ORIENTATION = {**light_probe.ORIENTATION,
    "same_as": "identical to the diffusionlight light probe (same pixel_to_direction), so the same USD DomeLight placement applies",
    "upstream": (
        "LuxDiT's own 128x256 map (envmap_vec) has the direction (sin(Th)*sin(Ph), cos(Th), sin(Th)*cos(Ph)) in the same "
        "camera space, Th = pi*gy, Ph = pi*gx, gx = -1+(2c+1)/W per column c, gy = linspace(1/H, 1-1/H, H) per row: its centre "
        "looks behind the camera and it is mirrored. It is re-sampled here to the grid above (upstream hdr_merger.py does the "
        "same with np.roll(W/2) + a horizontal flip, leaving the half-pixel row offset of its linspace)"
    ),
    "reliability": (
        "directions inside the plate's field of view are mostly copied from the plate (more so with a higher lora_scale); "
        "everything else is the model's guess"
    ),
}

# ----------------------------------------------------------------------------- parameters


def read_params(job) -> dict:
    """The node's parameters, the "default" (None) step count filled in."""
    p = job.params
    return {k: p[k] for k in ("seed", "envmap_width", "lora_scale", "guidance_scale", "resolution")} | {"steps": p["steps"] or DEFAULT_STEPS}


def model_input_size(width: int, height: int, choice: str) -> tuple[int, int]:
    """(height, width) the plate is resized + centre-cropped to (upstream resize_crop)."""
    if choice != "auto":
        return INPUT_SIZES[choice]
    aspect = width / height
    if aspect >= 1.2:
        return INPUT_SIZES["480x720"]
    if aspect <= 1 / 1.2:
        return INPUT_SIZES["720x480"]
    return INPUT_SIZES["512x512"]


# ----------------------------------------------------------------------------- model


@resident
def load_pipeline(repo: Path, weights: Path, with_lora: bool, device):
    """Upstream inference_luxdit.py main(), model part: VAE, 5B transformer, DPM scheduler, LoRA (an adapter: its
    scale is given per call)."""
    from omegaconf import ListConfig, OmegaConf

    sys.path.insert(0, str(repo))
    from diffusers.schedulers import CogVideoXDPMScheduler
    from src.models.custom_autoencoder_kl_cogvideox import AutoencoderKLCogVideoX
    from src.models.custom_cogvideox_transformer_3d import (
        CustomCogVideoXTransformer3DModel,
    )
    from src.pipelines.pipeline_cogvideox_rgbxenv import RGBXEnvCogVideoXPipeline

    cfg = OmegaConf.load(repo / "configs" / "luxdit_base.yaml").model_pipeline
    kwargs = {k: tuple(v) if isinstance(v, (list, ListConfig)) else v for k, v in dict(cfg.transformer_kwargs).items()}
    dtype = torch.bfloat16  # upstream --precision default

    vae = AutoencoderKLCogVideoX.from_pretrained(str(weights / BASE), subfolder="vae", torch_dtype=dtype)
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(str(weights / MODEL), torch_dtype=dtype, **kwargs)
    scheduler = CogVideoXDPMScheduler.from_pretrained(str(weights / BASE), subfolder="scheduler")
    pipe = RGBXEnvCogVideoXPipeline(vae=vae, tokenizer=None, text_encoder=None, transformer=transformer, scheduler=scheduler)
    if with_lora:
        pipe.load_lora_weights(str(weights / MODEL / "lora"), weight_name="model.safetensors", adapter_name="env-lora")
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device), cfg


def run_luxdit(pipe, cfg, rgb: np.ndarray, env_dirs: np.ndarray, size, params, device):
    """rgb [F,H,W,3] in 0..1, env_dirs [F,128,256,3] unit directions -> (ldr, log) [F,128,256,3] in 0..1."""
    cond_labels = dict(cfg.cond_images)
    cond_labels.pop("env_ldr", None)  # the targets: generated, not given
    cond_labels.pop("env_log", None)
    example = {"rgb": rgb.astype(np.float32), "env_nrm": (env_dirs * 0.5 + 0.5).astype(np.float32)}
    targets = list(cfg.additional_target_image)  # denoise_type additional_hidden_states: the envmaps
    _, cond_images = pipe.example2input(example, targets, cond_labels)
    steps = params["steps"]

    def on_step(_pipe, i, _t, kwargs):
        progress(i + 1, steps, f"去噪 {i + 1}/{steps}")
        return {}

    generator = set_seed(params["seed"], device)  # upstream: generator on the VAE's device
    with torch.no_grad():
        pred = pipe(
            prompt=cfg.get("text_prompt", None),
            cond_images=cond_images,
            cond_mapping=cond_labels,
            guidance_scale=params["guidance_scale"],
            use_dynamic_cfg=False,
            num_inference_steps=steps,
            generator=generator,
            height=size[0],
            width=size[1],
            num_frames=rgb.shape[0],
            additional_cond_labels=cfg.get("additional_cond_labels", None),
            attention_kwargs={"scale": params["lora_scale"]},
            denoise_type=cfg.get("denoise_type", "additional_hidden_states"),
            target_labels=cfg.target_image,
            additional_target_labels=targets,
            num_latents=len(targets),
            separate_timesteps=cfg.get("separate_timesteps", False),
            additional_rope_time_only=cfg.get("additional_rope_time_only", True),
            output_type="np",  # floats instead of upstream's 8-bit PNGs: no banding in the HDR merge
            callback_on_step_end=on_step,
        ).frames
    ldr, log = (np.asarray(p[0], dtype=np.float32) for p in pred)  # [F,h,w,3]
    return ldr.clip(0, 1), log.clip(0, 1)


@resident
def load_merger(weights: Path, device):
    from src.models.hdr_model import HDR_MLP

    return HDR_MLP.from_pretrained(str(weights / HDR_MERGER)).to(device).eval()


def merge_hdr(repo: Path, weights: Path, ldr: np.ndarray, log: np.ndarray, device) -> np.ndarray:
    """Upstream hdr_merger.py: HDR_MLP on the two maps, then its seam blending (not its roll / flip)."""
    sys.path.insert(0, str(repo))
    from hdr_merger import seam_blending

    model = load_merger(weights, device)
    with torch.no_grad():
        hdr = model(
            torch.from_numpy(ldr * 2.0 - 1.0).float().to(device),
            torch.from_numpy(log * 2.0 - 1.0).float().to(device),
        ).cpu().numpy()
    return seam_blending(hdr.astype(np.float64), blend_width=int(hdr.shape[1] * 0.02))


# ----------------------------------------------------------------------------- lat-long


def luxdit_directions(repo: Path) -> np.ndarray:
    """Upstream envmap_vec(128, 256): the direction of every pixel of the model's map, OpenGL camera space."""
    sys.path.insert(0, str(repo))
    from src.data.rendering_utils import envmap_vec

    return envmap_vec(list(ENV_RES)).numpy().astype(np.float64)


def to_lab2shot_latlong(src: np.ndarray, width: int) -> np.ndarray:
    """Re-sample LuxDiT's map (envmap_vec layout) onto ORIENTATION's grid, bilinear, wrapping around.

    Output pixel (u, v) looks along (-sin(phi) sin(theta), cos(phi), sin(phi) cos(theta)), theta = 2 pi u,
    phi = pi v. In envmap_vec that direction is Th = phi, Ph = -theta: gx = -2u (wrapped to [-1, 1)),
    gy = v, and its pixel indices are c = (gx + 1) W / 2 - 1/2, r = (gy - 1/H) / ((1 - 2/H) / (H - 1)).
    Maps narrower than the model's are super-sampled and box-filtered.
    """
    sh, sw = src.shape[:2]
    ss = max(1, math.ceil(sw / width))
    height = width // 2
    w, h = width * ss, height * ss
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    gx = np.mod(-2.0 * u + 1.0, 2.0) - 1.0
    col = (gx + 1.0) * sw / 2.0 - 0.5
    row = np.clip((v - 1.0 / sh) / ((1.0 - 2.0 / sh) / (sh - 1)), 0.0, sh - 1.0)
    c0 = np.floor(col).astype(int)
    fc = (col - c0)[None, :, None]
    r0 = np.minimum(np.floor(row).astype(int), sh - 2)
    fr = (row - r0)[:, None, None]
    c0w, c1w = np.mod(c0, sw), np.mod(c0 + 1, sw)
    top = src[r0][:, c0w] * (1 - fc) + src[r0][:, c1w] * fc
    bottom = src[r0 + 1][:, c0w] * (1 - fc) + src[r0 + 1][:, c1w] * fc
    out = top * (1 - fr) + bottom * fr
    if ss > 1:
        out = out.reshape(height, ss, width, ss, 3).mean(axis=(1, 3))
    return out


# ----------------------------------------------------------------------------- output


def srgb(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def preview(hdr: np.ndarray) -> Image.Image:
    """Upstream's LDR tone curve: sRGB(Reinhard with white 16)."""
    x = np.maximum(hdr, 0)
    y = x * (1 + x / REINHARD_WHITE**2) / (1 + x)
    return Image.fromarray((srgb(y).clip(0, 1) * 255 + 0.5).astype(np.uint8))


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "LuxDiT")
    job = run.job
    params = read_params(job)
    ref, plate = light_probe.probe_frame(job)

    weights = job.weights_dir
    run.weights(*(weights / rel for rel in (MODEL, HDR_MERGER, f"{BASE}/vae")))
    repo = job.repo_dir
    raw = job.raw_dir
    device = torch.device("cuda:0")
    run.gpu_cap_mb = limit_gpu_memory()  # before anything else: the warning below needs the cap (run.model would set it later)
    if run.gpu_cap_mb < MIN_GPU_MB:
        say("W-LUXDIT-LOWMEMORY", free=run.gpu_cap_mb / 1024, need=MIN_GPU_MB // 1024)

    run.stage("准备画面")
    sys.path.insert(0, str(repo))
    from src.data.rendering_utils import resize_crop

    frame = read_frame(plate, "float32")
    plate_h, plate_w = frame.shape[:2]
    size = model_input_size(plate_w, plate_h, params["resolution"])
    rgb = resize_crop(frame, list(size))[None]  # the model takes clips: one frame
    Image.fromarray((rgb[0] * 255 + 0.5).astype(np.uint8)).save(raw / "input.png")
    env_dirs = luxdit_directions(repo)[None]

    pipe, cfg = run.model("LuxDiT", load_pipeline, repo, weights, params["lora_scale"] > 0, device)
    run.stage("生成环境图")
    t_gen = time.time()
    ldr, log = run_luxdit(pipe, cfg, rgb, env_dirs, size, params, device)
    t_dit = time.time() - t_gen

    run.stage("合成 HDR")
    hdr_native = merge_hdr(repo, weights, ldr[0], log[0], device)
    width = params["envmap_width"]
    hdr = to_lab2shot_latlong(hdr_native, width).astype(np.float32)
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0).clip(0, None)

    probe = light_probe.save_probe(raw, ref, hdr, preview(hdr))
    stats = light_probe.hdr_stats(hdr)
    if stats["dynamic_range_max_over_median"] < 4:
        say("W-LUXDIT-LOWRANGE", ratio=float(stats["dynamic_range_max_over_median"]))

    run.finish(
        [ref],
        kind="light_probe",
        frame=ref,
        plate=str(plate),
        plate_size=[plate_w, plate_h],
        **probe,
        native_size=[ENV_RES[1], ENV_RES[0]],
        input="input.png",
        colorspace=COLORSPACE,
        orientation=ORIENTATION,
        usd_domelight=light_probe.USD_DOMELIGHT,
        camera_model=(
            f"none given: the model infers the field of view; the plate is scaled and centre-cropped to "
            f"{size[1]}x{size[0]} (w x h), so the map is centred on the plate's centre"
        ),
        model=MODEL,
        seed=params["seed"],
        settings={
            "input_size_hw": list(size),
            "steps": params["steps"],
            "guidance_scale": params["guidance_scale"],
            "lora_scale": params["lora_scale"],
            "scheduler": "CogVideoXDPMScheduler (upstream)",
            "precision": "bf16",
        },
        stats=stats,
        models=[
            f"LuxDiT image transformer ({MODEL}, 5B, bf16)",
            *([f"LuxDiT real-scene LoRA ({MODEL}/lora, scale {params['lora_scale']:g})"] if params["lora_scale"] > 0 else []),
            "LuxDiT HDR merge MLP (hdr_merge_mlp)",
            "CogVideoX-5B-I2V VAE",
        ],
        seconds_generate=round(t_dit, 1),
    )


if __name__ == "__main__":
    serve(main)
