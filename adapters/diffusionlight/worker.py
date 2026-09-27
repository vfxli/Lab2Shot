"""DiffusionLight-Turbo worker: HDR light probe from one frame. Runs inside
third_party/diffusionlight/.venv with the pinned repo on sys.path; never imports
Lab2Shot core.

    python worker.py <job.json>

One frame -> upstream inpaint.py's default "turbo_swapping" algorithm (SDXL +
depth ControlNet on DPT-Hybrid depth, Turbo LoRA swapped for the Exposure LoRA at
t < 800), once per EV with the same seed -> chrome balls -> each ball unwrapped
to lat-long (ball2envmap.py's mirror-ball mapping) -> exposure brackets merged
(exposure2hdr.py) ->

    raw/envmap.exr      float32 RGB lat-long HDR, scene-linear, Rec.709 primaries,
                        envmap_width x envmap_width/2, camera-relative (ORIENTATION)
    raw/preview.png     the inpainted chrome ball at EV0 (256x256, sRGB)
    raw/ball_ev<ev>.png the other exposures
    raw/result.json
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from lab2shot_worker import light_probe, progress, say, serve, set_seed
from lab2shot_worker.run import Run
from PIL import Image

NODE = "diffusionlight.light_probe"

EVS = [0.0, -2.5, -5.0]  # the chrome ball's exposure brackets (upstream's): the HDR is merged from them
GAMMA = 2.4  # exposure2hdr.py --gamma: LDR ball value -> linear
SUPERSAMPLE = 2  # lat-long pixels are sampled 2x2 and averaged (upstream: 4x then resize)

MODELS = {
    "sdxl": "stable-diffusion-xl-base-1.0",
    "controlnet": "controlnet-depth-sdxl-1.0",  # the name must contain "depth" (get_control_signal_type)
    "vae": "sdxl-vae-fp16-fix",
    "depth": "dpt-hybrid-midas",
}
TURBO_LORA = "models/rev3/Flickr2K/Flickr2kPlus_extended/checkpoint-230000"
EXPOSURE_LORA = "models/ThisIsTheFinal-lora-hdr-continuous-largeT@900/0_-5/checkpoint-2500"

COLORSPACE = {
    "name": "lin_rec709",
    "ocio_hint": "Linear Rec.709 (sRGB)",
    "primaries": "Rec.709 / sRGB (x,y R 0.64,0.33 G 0.30,0.60 B 0.15,0.06), white D65 (0.3127,0.3290)",
    "transfer": (
        "scene-linear. Balls are linearised with a pure 2.4 power (upstream exposure2hdr.py, matching how the "
        "Exposure LoRA's training balls were tone-mapped), not the piecewise sRGB curve"
    ),
    "scale": (
        "relative, not absolute: linear 1.0 = display white of the EV0 ball, i.e. roughly the plate's own "
        "exposure. Values above 1 come from the darker exposures (up to 2^-min(ev) = 32 at EV -5)"
    ),
    "note": "SDXL paints in the plate's display RGB (sRGB/Rec.709); there is no calibrated gamut, so the primaries are nominal",
}

ORIENTATION = {**light_probe.ORIENTATION,
    "reliability": (
        "the chrome ball's centre sees behind the camera (u = 0 / 1) best; directions near the viewing direction "
        "(u = 0.5) come from the ball's grazing rim and are the least reliable"
    ),
}



# ----------------------------------------------------------------------------- chrome ball


def mark_hf_cache_current() -> None:
    """transformers 4.34 / diffusers 0.23 try to "migrate" an empty HF cache at import and warn
    loudly when offline. Everything is loaded from local folders, so mark the extension's own
    cache (HF_HOME from worker_env, never ~/.cache) as already migrated."""
    home = os.environ.get("HF_HOME")
    if not home:
        return
    hub = Path(home) / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    for name in ("version.txt", "version_diffusers_cache.txt"):
        if not (hub / name).exists():
            (hub / name).write_text("1")


def load_upstream(repo: Path, weights: Path):
    """Import the pinned research code with its model ids pointed at weights/."""
    mark_hf_cache_current()
    sys.path.insert(0, str(repo))
    from relighting import argument

    argument.VAE_MODELS["sdxl"] = str(weights / MODELS["vae"])  # dict shared with inpainter_2lora
    import relighting.inpainter_2lora as inpainter

    inpainter.DEPTH_ESTIMATOR = str(weights / MODELS["depth"])  # copied name: patch the module global
    import inpaint  # upstream inpaint.py: argument defaults, embedding interpolation, ball placement

    return inpaint, inpainter


def upstream_args(inpaint, repo: Path, evs: list[float], seed: int):
    """inpaint.py's own argparse defaults (turbo_swapping, 30 steps, CFG 5, control 0.5, ...)."""
    args = inpaint.create_argparser().parse_args(
        ["--dataset", "-", "--output_dir", "-", "--ev", ",".join(f"{e:g}" for e in evs), "--seed", str(seed)]
    )
    args = inpaint.backward_compatible_parameters(args)
    args.turbo_lora_path = str(repo / TURBO_LORA)
    args.exposure_lora_path = str(repo / EXPOSURE_LORA)
    assert args.algorithm == "turbo_swapping" and args.model_option == "sdxl" and args.use_controlnet
    return args


def paint_balls(run: Run, inpaint, inpainter, args, plate: Image.Image, seed: int, weights: Path, device):
    """inpaint.py main() for one image and the turbo_swapping algorithm.

    Returns ({ev: 256x256 PIL ball}, SDXL canvas size)."""
    from relighting.ball_processor import get_ideal_normal_ball
    from relighting.image_processor import pil_square_image
    from relighting.mask_utils import MaskGenerator

    with run.loading("SDXL / ControlNet / LoRA"):
        pipe = inpainter.BallInpainter.from_sdxl(
            model=str(weights / MODELS["sdxl"]),
            controlnet=str(weights / MODELS["controlnet"]),
            device=device,
            torch_dtype=torch.float16,
            offload=False,
        )
        pipe.pipeline.load_lora_weights(args.exposure_lora_path)
        pipe.pipeline.fuse_lora(lora_scale=args.exposure_lora_scale)

    image = pil_square_image(plate, (args.img_width, args.img_height))  # fit + black borders
    embedding_dict = inpaint.interpolate_embedding(pipe, args)
    normal_ball, mask_ball = get_ideal_normal_ball(size=args.ball_size + args.ball_dilate)
    x, y, r = inpaint.get_ball_location({}, args)  # centre of the canvas
    mask = MaskGenerator().generate_single(
        image, mask_ball, x - (args.ball_dilate // 2), y - (args.ball_dilate // 2), r + args.ball_dilate
    )

    balls = {}
    run.stage("画铬球")
    for i, (ev, (prompt_embeds, pooled_prompt_embeds)) in enumerate(embedding_dict.items()):
        progress(i, len(embedding_dict), f"EV {ev:g}")
        pipe.pipeline.unfuse_lora()
        pipe.pipeline.unload_lora_weights()
        pipe.pipeline.load_lora_weights(args.turbo_lora_path)
        pipe.pipeline.fuse_lora(lora_scale=args.turbo_lora_scale)
        kwargs = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt": args.negative_prompt,
            "num_inference_steps": args.denoising_step,
            "generator": set_seed(seed),  # same seed for every EV (upstream)
            "image": image,
            "mask_image": mask,
            "strength": 1.0,
            "current_seed": seed,
            "controlnet_conditioning_scale": args.control_scale,
            "height": args.img_height,
            "width": args.img_width,
            "normal_ball": normal_ball,
            "mask_ball": mask_ball,
            "x": x,
            "y": y,
            "r": r,
            "guidance_scale": args.guidance_scale,
            "cross_attention_kwargs": {"scale": args.lora_scale},
            "switch_lora_timestep": args.switch_lora_timestep,
            "exposure_lora_path": args.exposure_lora_path,
            "exposure_lora_scale": args.exposure_lora_scale,
        }
        output = pipe.inpaint_turbo_swapping(**kwargs).images[0]
        balls[ev] = output.crop((x, y, x + r, y + r))
    progress(len(embedding_dict), len(embedding_dict), "完成")
    return balls, image.size


# ----------------------------------------------------------------------------- ball -> lat-long


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def ball_lookup(dirs: np.ndarray, focal_px: float | None, radius_px: float) -> np.ndarray:
    """Where on the ball image each reflected direction is seen: grid_sample coordinates in [-1, 1]
    (x right, y down) over the ball crop, align_corners=True (pixel centres at +-1, as upstream).

    focal_px None: upstream's orthographic mirror ball (view vector +Z everywhere):
    normal = normalize(view + reflected), ball position = normal's x, y.
    focal_px given: a perspective camera at the origin looking at a sphere on the
    optical axis whose silhouette is the painted disk; solved per direction by fixed-point
    iteration (normal = bisector of the direction to the camera and the reflected ray).
    """
    view = np.array([0.0, 0.0, 1.0])
    n = _normalize(dirs + view)
    if focal_px is None:
        gx, gy = n[..., 0], -n[..., 1]
    else:
        sin_a = math.sin(math.atan(radius_px / focal_px))  # sphere radius at centre distance 1
        centre = np.array([0.0, 0.0, -1.0])
        shape = n.shape
        n, flat = n.reshape(-1, 3).copy(), dirs.reshape(-1, 3)
        todo = np.arange(len(n))
        for _ in range(60):  # converges ~sin_a per step; only unconverged points are updated
            new = _normalize(_normalize(-(centre + sin_a * n[todo])) + flat[todo])
            moved = np.abs(new - n[todo]).max(axis=-1) > 1e-9
            n[todo] = new
            todo = todo[moved]
            if not todo.size:
                break
        n = n.reshape(shape)
        p = centre + sin_a * n
        depth = np.maximum(-p[..., 2], 1e-6)
        gx = focal_px * p[..., 0] / depth / radius_px
        gy = -focal_px * p[..., 1] / depth / radius_px
    # Directions the ball cannot show (behind it) sample its rim, like upstream's border padding.
    rho = np.maximum(np.hypot(gx, gy), 1.0)
    return np.stack([gx / rho, gy / rho], axis=-1)


def unwrap(ball: np.ndarray, grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """ball (h, w, 3) float -> lat-long (height, width, 3), bilinear + 2x2 box filter."""
    t_ball = torch.from_numpy(ball).permute(2, 0, 1)[None].float()
    t_grid = torch.from_numpy(grid)[None].float()
    env = F.grid_sample(t_ball, t_grid, mode="bilinear", padding_mode="border", align_corners=True)
    env = F.avg_pool2d(env, SUPERSAMPLE)
    assert env.shape[-2:] == (height, width), env.shape
    return env[0].permute(1, 2, 0).numpy().astype(np.float64)


def merge_exposures(envmaps: dict[float, np.ndarray]) -> np.ndarray:
    """exposure2hdr.py process_image(), on float lat-long maps instead of 8-bit PNGs."""
    evs = sorted(envmaps, reverse=True)
    image0_linear = np.power(envmaps[evs[0]], GAMMA)
    luminances = [(np.power(envmaps[ev], GAMMA) / (2**ev)) @ light_probe.LUMA for ev in evs]
    out = luminances[-1]
    for i in range(len(evs) - 1, 0, -1):
        maxval = 1 / (2 ** evs[i - 1])
        p1 = np.clip((luminances[i - 1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
        p2 = out > luminances[i - 1]
        mask = (p1 * p2).astype(np.float32)
        out = luminances[i - 1] * (1 - mask) + out * mask
    return image0_linear * (out / (luminances[0] + 1e-10))[:, :, None]


# ----------------------------------------------------------------------------- output


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "DiffusionLight")
    job, params = run.job, run.params
    number, path = light_probe.probe_frame(job)
    seed, width, fov_deg, evs = params["seed"], params["envmap_width"], params["fov_deg"], EVS
    height = width // 2

    weights = job.weights_dir
    run.weights(*(weights / name for name in MODELS.values()))
    raw = job.raw_dir
    device = torch.device("cuda:0")  # an index, so the DPT depth pipeline runs on the GPU too

    run.stage("准备画面")
    plate = Image.open(path).convert("RGB")
    inpaint, inpainter = load_upstream(job.repo_dir, weights)
    args = upstream_args(inpaint, job.repo_dir, evs, seed)
    balls, canvas_size = paint_balls(run, inpaint, inpainter, args, plate, seed, weights, device)
    t_paint = time.time() - run.t0

    run.stage("展开铬球 / 合成 HDR")
    # Plate focal length in canvas pixels (pil_square_image scales the long side to 1024).
    scale = min(canvas_size[0] / plate.width, canvas_size[1] / plate.height)
    plate_w_canvas = int(plate.width * scale)
    focal_px = None if fov_deg is None else (plate_w_canvas / 2) / math.tan(math.radians(float(fov_deg)) / 2)
    radius_px = (args.ball_size - 1) / 2  # disk edge at the outer pixel centres (grid +-1)
    grid = ball_lookup(light_probe.latlong_directions(height * SUPERSAMPLE, width * SUPERSAMPLE), focal_px, radius_px)
    envmaps = {ev: unwrap(np.asarray(ball, np.float64) / 255.0, grid, height, width) for ev, ball in balls.items()}
    hdr = merge_exposures(envmaps).astype(np.float32)
    hdr = np.nan_to_num(hdr, nan=0.0, posinf=0.0, neginf=0.0).clip(0, None)

    probe = light_probe.save_probe(raw, number, hdr, balls[0.0])
    ball_files = {"0": probe["preview"]}
    for ev, ball in balls.items():
        if ev != 0.0:
            name = f"ball_ev{ev:g}.png"
            ball.save(raw / name)
            ball_files[f"{ev:g}"] = name
    stats = light_probe.hdr_stats(hdr)
    if stats["dynamic_range_max_over_median"] < 4:
        say("W-DIFFUSIONLIGHT-LOWRANGE", ratio=float(stats["dynamic_range_max_over_median"]))

    run.finish(
        [number],
        kind="light_probe",
        frame=number,
        plate=str(path),
        plate_size=[plate.width, plate.height],
        **probe,
        balls=ball_files,
        colorspace=COLORSPACE,
        orientation=ORIENTATION,
        usd_domelight=light_probe.USD_DOMELIGHT,
        camera_model=(
            "orthographic (upstream ball2envmap.py default: camera infinitely far from the ball)"
            if focal_px is None
            else f"perspective, horizontal FOV {float(fov_deg):g} deg (focal {focal_px:.1f} px on the {canvas_size[0]} px canvas)"
        ),
        fov_deg=None if fov_deg is None else float(fov_deg),
        seed=seed,
        ev=evs,
        algorithm="turbo_swapping (upstream default): Turbo LoRA 1.0, Exposure LoRA 0.75 from t < 800",
        settings={
            "steps": args.denoising_step,
            "guidance_scale": args.guidance_scale,
            "controlnet_conditioning_scale": args.control_scale,
            "ball_size_px": args.ball_size,
            "ball_dilate_px": args.ball_dilate,
            "canvas": list(canvas_size),
            "gamma": GAMMA,
        },
        stats=stats,
        models=[
            "SDXL 1.0 base (fp16)",
            "SDXL depth ControlNet (diffusers, fp16)",
            "SDXL-VAE-FP16-Fix",
            "DPT-Hybrid MiDaS (Intel/dpt-hybrid-midas)",
            "DiffusionLight Turbo LoRA (Flickr2kPlus_extended ckpt 230000)",
            "DiffusionLight Exposure LoRA (0_-5 ckpt 2500)",
        ],
        seconds_inpaint=round(t_paint, 1),
    )


if __name__ == "__main__":
    serve(main)
