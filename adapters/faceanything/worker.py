"""FaceAnything worker. Runs inside third_party/faceanything/.venv-ada-blackwell with the
original repo's src/ on PYTHONPATH; never imports Lab2Shot core.

    python worker.py <job.json>

Node "faceanything.face_maps": frames -> Robust Video Matting alpha (in order, like
FaceAnything's own pipeline) -> FaceAnything (DA3-GIANT + canonical head) in
chunks of frames or one frame at a time -> per frame raw/frame_<n>.npz in the
geometry contract (lab2shot_worker.mono_geometry: points, depth, mask = the RVM
matte on valid pixels, intrinsics, confidence) at the input resolution, plus
canonical [H,W,3] (the face's canonical-space position of each pixel) and alpha
(the soft RVM matte, uint8). Depth is in the model's own units (not metric).

Chunk mode: the model sees `chunk` frames together (their joint prediction is
more consistent than frame by frame). Consecutive chunks share OVERLAP frames;
each chunk's depth is rescaled to agree with the previous chunk on those shared
frames (the model's depth has no fixed scale between separate calls), and the
shared frames keep the earlier chunk's result. Out of GPU memory, the worker
SDK's one policy (lab2shot_worker.fit_memory) decides, on Max Frames per Chunk in chunk
mode and on Resolution one frame at a time.

RTX 4090 (24 GB), resolution 504, peak memory reserved by this process:
one frame 8.6 GB, chunk 8 13.2 GB, 16 14.4-15.6 GB, 24 18.8 GB, 40 20.4 GB.
Chunk 16 is the default (about 0.19 s/frame). 504 is the only resolution the
node offers (the model's training size, the only one measured).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lab2shot_worker import MemoryBound, fail, limit_gpu_memory, progress, read_frame, resident, save_npz, say, serve
from lab2shot_worker.mono_geometry import finish_geometry, frame_arrays, unproject_frame
from lab2shot_worker.run import Run

NODE = "faceanything.face_maps"
# what the memory grows with: the node's options (chunk 8 / 16 / 24 = 13.2 / 15.0 / 18.8 GB at 504 px)
CHUNK = MemoryBound.parameter("max_frames", (24, 16, 8))
RESOLUTION = MemoryBound.parameter("resolution", (504,))  # one frame at a time: the one resolution offered
OVERLAP = 2  # frames shared by consecutive chunks (depth scale alignment)
BASE_MODEL = "da3-giant"  # the DA3 preset FaceAnything is finetuned from (config in repo/src)
GIANT_FEATURE_DIM = 3072
RVM_TARGET_SIDE = 480  # RVM's internal long side: 1920 px plates get FaceAnything's 0.25
RVM_WARMUP = 25  # recurrent-state warm-up passes on the first frame (as in faceanything/background.py)


def output_size(path: Path, resolution: int) -> tuple[int, int]:
    """(h, w) the model works and predicts at: FaceAnything's own preprocessing of this frame."""
    from depth_anything_3.utils.io.input_processor import InputProcessor

    _, (h, w), _, _ = InputProcessor()._process_one(
        str(path), process_res=resolution, process_res_method="upper_bound_resize"
    )
    return h, w


@resident
def load_rvm(rvm_dir: str, ckpt: Path, device: torch.device):
    model = torch.hub.load(rvm_dir, "resnet50", source="local", pretrained=False, trust_repo=True)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def matte(paths: list[Path], device) -> list[np.ndarray]:
    """Robust Video Matting ResNet-50 over the frames in order -> alpha uint8 [H, W] per frame (input size)."""
    ckpt = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / "rvm_resnet50.pth"
    model = load_rvm(os.environ["FACEANYTHING_RVM_DIR"], ckpt, device)

    first = read_frame(paths[0])
    ratio = min(1.0, RVM_TARGET_SIDE / max(first.shape[:2]))
    to_tensor = lambda rgb: torch.from_numpy(rgb).to(device).permute(2, 0, 1)[None].float().div_(255.0)  # noqa: E731
    rec = [None] * 4
    alphas = []
    with torch.inference_mode():
        src = to_tensor(first)
        for _ in range(RVM_WARMUP):
            _, _, *rec = model(src, *rec, ratio)
        for n, path in enumerate(paths):
            _, pha, *rec = model(to_tensor(read_frame(path)), *rec, ratio)
            alphas.append(np.clip(np.rint(pha[0, 0].float().cpu().numpy() * 255.0), 0, 255).astype(np.uint8))
            progress(n + 1, len(paths), "background_matte")
    return alphas


@resident
def load_model(path: Path, device: torch.device):
    """FaceAnything = DA3-GIANT + a DPT canonical ("deformation") head, weights from checkpoint.pt.

    Same network as faceanything.model.load_model, but the DA3 architecture comes
    from the repo's own da3-giant preset instead of downloading DA3-GIANT-1.1
    from Hugging Face (whose weights the checkpoint overwrites anyway).
    """
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model import dpt

    with torch.device(device):  # initialize straight on the GPU: much faster than on the CPU
        model = DepthAnything3(model_name=BASE_MODEL)
        model.model.deformation_head = dpt.DPT(
            GIANT_FEATURE_DIM, output_dim=4, head_name="deformation", use_sky_head=False, activation="linear"
        )
    try:  # memory-mapped: only the model weights are read from the 15 GB file
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError:  # legacy (non-zip) checkpoint format
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    del ckpt, state
    # Only the Gaussian-splatting branch (never run here) may be absent from the checkpoint.
    missing = [k for k in missing if not k.startswith(("model.gs_head.", "model.gs_adapter."))]
    if missing:
        fail("E-WORKER-WEIGHTSMISMATCH", project="FaceAnything", extension="faceanything", model=path.name,
             missing=len(missing), unexpected=len(unexpected), examples=list(missing[:5]))
    if unexpected:
        say("I-WORKER-UNUSEDWEIGHTS", project="FaceAnything", model=path.name, count=len(unexpected),
            examples=list(unexpected[:3]))
    return model.eval()


def infer(model, paths: list[Path], resolution: int) -> dict[str, np.ndarray]:
    """One model call on a list of frames (what faceanything.predict.run_inference does per call)."""
    with torch.inference_mode():
        p = model.inference([str(x) for x in paths], export_dir=None, use_ray_pose=True, process_res=resolution)
    if getattr(p, "deformation", None) is None or getattr(p, "conf", None) is None:
        raise RuntimeError("FaceAnything returned no canonical / confidence maps (checkpoint without canonical head?)")
    return {
        "depth": np.asarray(p.depth, np.float32),
        "canonical": np.asarray(p.deformation, np.float32),
        "conf": np.asarray(p.conf, np.float32),
        "K": np.asarray(p.intrinsics, np.float64),
    }


def to_input_size(depth: np.ndarray, canonical: np.ndarray, conf: np.ndarray, k: np.ndarray, size: tuple[int, int],
                  device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """The model's maps (at its own h x w, a plain resize of the frame) -> the input size (H, W): bilinear maps,
    and K from FaceAnything's pixel-centre convention to the contract's pixel-corner one, scaled per axis."""
    h, w = depth.shape
    H, W = size
    grow = lambda a: torch.nn.functional.interpolate(  # noqa: E731
        torch.from_numpy(np.ascontiguousarray(a)).to(device).permute(2, 0, 1)[None], size=size, mode="bilinear",
        align_corners=False)[0].permute(1, 2, 0)
    maps = grow(np.concatenate([depth[..., None], canonical, conf[..., None]], -1))
    sx, sy = W / w, H / h
    k_in = np.array([[k[0, 0] * sx, 0, (k[0, 2] + 0.5) * sx], [0, k[1, 1] * sy, (k[1, 2] + 0.5) * sy], [0, 0, 1]])
    return maps[..., 0], maps[..., 1:4], maps[..., 4], k_in


def depth_scale(ref: np.ndarray, new: np.ndarray, mask: np.ndarray) -> float:
    """Factor that brings `new` depth to `ref` depth on the shared frames (median ratio on the face)."""
    m = mask & (ref > 0) & (new > 0) & np.isfinite(ref) & np.isfinite(new)
    if m.sum() < 100:
        m = (ref > 0) & (new > 0) & np.isfinite(ref) & np.isfinite(new)
    return float(np.median(ref[m] / new[m])) if m.any() else 1.0


def solve(model, paths: list[Path], alphas: list[np.ndarray], numbers: list[int], chunk: int, resolution: int) -> dict:
    """The whole shot, `chunk` frames per model call at `resolution`: frame index -> maps, the chunks and their depth
    scales, the model's output size (h, w)."""
    n = len(paths)
    h, w = output_size(paths[0], resolution)
    results: dict[int, dict[str, np.ndarray]] = {}  # frame index -> maps of that frame
    chunks, scales = [], []
    pos = 0  # first frame index not solved yet
    size = min(chunk, n)
    while pos < n:
        start = max(0, pos - min(OVERLAP, size - 1))  # every chunk solves at least one new frame
        end = min(n, start + size)
        if end == n and size > 1:
            start = max(0, n - size)  # a full-size last chunk: more context for the tail
        out = infer(model, paths[start:end], resolution)
        if out["depth"].shape[1:] != (h, w):
            raise RuntimeError(f"model output {out['depth'].shape[1:]} != expected {(h, w)}")
        shared = [i for i in range(start, end) if i < pos]
        scale = 1.0
        if shared:
            ref = np.stack([results[i]["depth"] for i in shared])
            new = out["depth"][[i - start for i in shared]]
            mask = np.stack([cv2.resize(alphas[i], (w, h), interpolation=cv2.INTER_AREA) >= 128 for i in shared])  # model size
            scale = depth_scale(ref, new, mask)
        for i in range(max(start, pos), end):
            j = i - start
            results[i] = {
                "depth": out["depth"][j] * np.float32(scale),
                "canonical": out["canonical"][j],
                "conf": out["conf"][j],
                "K": out["K"][j],
            }
        chunks.append([numbers[start], numbers[end - 1]])
        scales.append(round(scale, 5))
        pos = end
        progress(pos, n, "solve")
    return {"results": results, "chunks": chunks, "scales": scales, "size": (h, w), "chunk": size, "resolution": resolution}


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "FaceAnything")
    job, params = run.job, run.params
    mode, resolution = params["mode"], params["resolution"]
    chunk = 1 if mode == "one_by_one" else params["max_frames"]
    device = torch.device("cuda")

    frames = run.frames()
    numbers, paths, n = frames.numbers, frames.paths, len(frames)
    raw = job.raw_dir
    t0 = time.time()
    run.gpu_cap_mb = limit_gpu_memory()  # before the matting model too, not only the face model (run.model would set it later)

    run.stage("background_matte_rvm")
    alphas = matte(paths, device)
    t_matte = time.time() - t0

    model = run.model("load_model", load_model, job.weights_dir / "faceanything" / "checkpoint.pt", device,
                      stage_params={"model": "FaceAnything"})

    run.stage("solve")
    t2 = time.time()
    if mode == "chunk":
        solved = run.fit(CHUNK, lambda c: solve(model, paths, alphas, numbers, c, resolution), chunk)
    else:
        solved = run.fit(RESOLUTION, lambda s: solve(model, paths, alphas, numbers, 1, s), resolution)
    results, (h, w) = solved["results"], solved["size"]
    t_solve = time.time() - t2
    run.frame_seconds.extend([t_solve / n] * n)  # the solve is one pass over all frames: shared out per frame

    run.stage("write_results")
    size_in = alphas[0].shape
    stats: dict[str, list] = {"focal_px": [], "depth_median": [], "valid_fraction": [], "frames": numbers}
    for i, f in enumerate(numbers):
        r = results[i]
        depth, canonical, conf, k_in = to_input_size(r["depth"], r["canonical"], r["conf"], r["K"], size_in, device)
        alpha = alphas[i]
        mask = torch.isfinite(depth) & torch.isfinite(canonical).all(-1) & torch.from_numpy(alpha >= 128).to(device)
        arrays = frame_arrays(unproject_frame(torch.nan_to_num(depth), k_in), mask, k_in, confidence=conf)
        valid = arrays["depth"][arrays["mask"]]
        stats["focal_px"].append(float(k_in[0, 0]))
        stats["depth_median"].append(float(np.median(valid)) if valid.size else float("nan"))
        stats["valid_fraction"].append(round(float(arrays["mask"].mean()), 4))
        save_npz(raw / f"frame_{f}.npz", **arrays, canonical=torch.nan_to_num(canonical).cpu().numpy().astype(np.float32),
                 alpha=alpha)
        progress(i + 1, n, "write_results")
    finish_geometry(
        run,
        stats,
        model="FaceAnything (DA3-GIANT + canonical head), Robust Video Matting ResNet-50",
        fov_x=None,
        metric=False,
        files="frame_<n>.npz: points, depth, mask, intrinsics, confidence, canonical, alpha",
        process_width=w,
        process_height=h,
        mode=mode,
        chunk=solved["chunk"],
        chunk_requested=chunk,
        overlap=OVERLAP if mode == "chunk" else 0,
        chunks=solved["chunks"],
        chunk_depth_scales=solved["scales"],
        resolution=solved["resolution"],
        max_side_requested=resolution,
        canonical_units="FaceAnything canonical face space (meters per the paper)",
        license="CC BY-NC 4.0 (non-commercial)",
        seconds_matte=round(t_matte, 1),
        seconds_solve=round(t_solve, 1),
    )


if __name__ == "__main__":
    serve(main)
