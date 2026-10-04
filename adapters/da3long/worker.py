"""Official DA3-Streaming driver; upstream stays unchanged."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import yaml

from lab2shot_worker import fail, progress, save_npz, say, serve
from lab2shot_worker.recon import save_cameras
from lab2shot_worker.run import Run


def export_chunk(run, shot, prediction, indices, scale, poses, calibration):
    """Restore the upper-bound-resized raster; depth and translations share scale."""
    import cv2

    for local, index in indices:
        z = np.asarray(prediction.depth[local], np.float32) * scale
        score = np.asarray(prediction.conf[local], np.float32) + 1.0  # upstream streaming subtracts 1
        h, w = z.shape
        K = np.asarray(prediction.intrinsics[local], np.float64).copy()
        K[0] *= shot.width / w
        K[1] *= shot.height / h
        calibration[index] = K
        z = cv2.resize(z, (shot.width, shot.height), interpolation=cv2.INTER_LINEAR)
        score = cv2.resize(score, (shot.width, shot.height), interpolation=cv2.INTER_LINEAR)
        save_npz(run.job.raw_dir / f"frame_{shot.numbers[index]}.npz", depth=z, confidence=score,
                 mask=np.isfinite(z) & (z > 0))
        progress(index + 1, len(shot), "write_depth")


# Peak VRAM of one chunk: the official table (da3_streaming/README.md, TUM 504x378: 120 frames 28.3 GB, 60 frames
# 21.2 GB) gives the growth per frame and model-input pixel; BASE_GB is the rest, refitted on our own runs
# (RTX 5090, PyTorch caching allocator's reserved peak).
PER_FRAME_PX_GB = 6.2e-7
BASE_GB = 15.5
PROCESS_RES = 504  # upstream process_res: the long side the model sees (upper_bound_resize)
STEPS = (120, 90, 60, 40, 30)  # chunk sizes stepped down to when a card cannot hold the asked one


def chunk_vram_gb(frames: int, width: int, height: int) -> float:
    scale = PROCESS_RES / max(width, height)
    px = round(width * scale / 14) * 14 * round(height * scale / 14) * 14
    return BASE_GB + PER_FRAME_PX_GB * frames * px


def fitting_chunk(asked: int, overlap: int, width: int, height: int, budget: float | None = None) -> tuple[int, int, float]:
    """The asked chunk size, or the largest smaller step that fits (overlap kept in proportion); the memory it was
    picked by in GB. Upstream's 120 frames need ~30 GB at 4:3: a 24 GB card runs smaller chunks instead of failing.
    `budget`: the VRAM the scheduler gave this run's tier (the job's vram_budget_gb): the step is picked by it alone,
    so the result is the tier the cache key says; a card with less free than that step needs is an error, never a
    further silent step down. None: by the card's free memory."""
    free, _total = torch.cuda.mem_get_info()
    free = (free + torch.cuda.memory_reserved()) / 2**30
    card = float(budget) if budget else free
    size = next((s for s in (asked, *[s for s in STEPS if s < asked])
                 if chunk_vram_gb(s, width, height) <= card or s == min(asked, STEPS[-1])), asked)
    if budget and chunk_vram_gb(size, width, height) > free:
        fail("E-WORKER-VRAMSHORT", free=free, need=chunk_vram_gb(size, width, height))
    return size, (overlap if size == asked else max(2, min(size - 1, round(overlap * size / asked)))), card


def main(job_path: str):
    run = Run.start(job_path, "da3long.reconstruct", "DA3-Long")
    job, params = run.job, run.params
    shot = run.frames(step=params["step"], least=2)
    run.weights(job.weights_dir / "model.safetensors", job.weights_dir / "config.json")
    streaming = job.repo_dir / "da3_streaming"
    sys.path.insert(0, str(streaming))
    import da3_streaming as upstream

    config = yaml.safe_load((streaming / "configs/base_config.yaml").read_text())
    config["Weights"] = {"DA3": str(job.weights_dir / "model.safetensors"),
                         "DA3_CONFIG": str(job.weights_dir / "config.json"),
                         "SALAD": str(job.weights_dir / "dino_salad.ckpt")}
    chunk_size = int(params["max_frames"])
    overlap = int(params["overlap"])
    if overlap >= chunk_size:
        fail("E-DA3LONG-OVERLAP", overlap=overlap, chunk=chunk_size)
    if len(shot) > STEPS[-1]:  # a chunk is at most the shot: a short shot never needs a smaller one
        asked = min(chunk_size, len(shot))
        fitted, fitted_overlap, card = fitting_chunk(asked, overlap, shot.width, shot.height, params.get("vram_budget_gb"))
        if fitted < asked:
            say("W-DA3LONG-VRAMSTEP", asked=chunk_size, chunk=fitted, chunk_overlap=fitted_overlap,
                card=card, need=chunk_vram_gb(asked, shot.width, shot.height))
            chunk_size, overlap = fitted, fitted_overlap
    loops = bool(params["loops"]) and len(shot) > chunk_size
    if loops:
        run.weights(job.weights_dir / "dino_salad.ckpt")
    config["Model"].update(chunk_size=chunk_size, overlap=overlap, loop_enable=loops,
                           align_lib="torch", save_depth_conf_result=False, delete_temp_files=False)
    config["Loop"]["SIM3_Optimizer"]["lang_version"] = "python"
    inputs = job.raw_dir / "_ordered"
    inputs.mkdir(exist_ok=True)
    for i, path in enumerate(shot.paths):
        target = inputs / f"{i:09d}.png"
        if not target.exists():
            target.symlink_to(path.resolve())
    output = job.raw_dir / "_streaming"
    output.mkdir(exist_ok=True)

    original_hub_load = torch.hub.load

    def local_dino(repository, model, *args, **kwargs):
        if repository == "facebookresearch/dinov2":
            # SALAD's complete checkpoint supplies the backbone weights too.
            return original_hub_load(str(job.repo_dir.parent / "dinov2"), model, source="local", pretrained=False)
        return original_hub_load(repository, model, *args, **kwargs)

    with patch.object(torch.hub, "load", local_dino):
        pipeline = run.model("load_model", upstream.DA3_Streaming, str(inputs), str(output), config, stage_params={"model": "DA3-Long"})
        if len(shot) <= chunk_size:
            # Upstream only exports inside its multi-chunk loop. Keep the same
            # official inference for short shots, then serialize every frame.
            pipeline.img_list = [str(p) for p in sorted(inputs.glob("*.png"))]
            pipeline.chunk_indices = [(0, len(shot))]
            run.stage("solve_short_shot")
            prediction = pipeline.process_single_chunk((0, len(shot)), chunk_idx=0)
            chunks = [(0, len(shot), prediction, 1.0, np.eye(3), np.zeros(3))]
        else:
            run.stage("streaming_solve")
            pipeline.run()
            chunks = []
            for index, (start, stop) in enumerate(pipeline.chunk_indices):
                prediction = np.load(output / "_tmp_results_unaligned" / f"chunk_{index}.npy", allow_pickle=True).item()
                s, R, t = (1.0, np.eye(3), np.zeros(3)) if index == 0 else pipeline.sim3_list[index - 1]
                # Upstream keeps each frame from the later overlap chunk.
                end = stop if index == len(pipeline.chunk_indices) - 1 else stop - overlap
                chunks.append((start, end, prediction, float(s), R, t))

    poses = np.repeat(np.eye(4)[None], len(shot), axis=0)
    K = np.zeros((len(shot), 3, 3), np.float64)
    for start, stop, prediction, scale, R, t in chunks:
        extrinsics = np.repeat(np.eye(4)[None], len(prediction.extrinsics), axis=0)
        extrinsics[:, :3] = prediction.extrinsics
        c2w = np.linalg.inv(extrinsics)
        c2w[:, :3, :3] = R @ c2w[:, :3, :3]
        c2w[:, :3, 3] = scale * (c2w[:, :3, 3] @ R.T) + t
        poses[start:stop] = c2w[:stop - start]
        export_chunk(run, shot, prediction, [(i - start, i) for i in range(start, stop)], scale, poses, K)
    if not np.all(K[:, (0, 1), (0, 1)] > 0):
        fail("E-DA3LONG-UNCOVERED")
    # the model's own fx and fy, passed on as they are; a real lens has them (nearly) equal, so a big difference is said
    fx, fy = float(np.median(K[:, 0, 0])), float(np.median(K[:, 1, 1]))
    if abs(fy / fx - 1) > 0.1:
        say("W-DA3LONG-ASPECT", fx=fx, fy=fy, diff=abs(fy / fx - 1))
    poses = np.linalg.inv(poses[0]) @ poses
    save_cameras(job.raw_dir, shot.numbers, K, poses, shot.width, shot.height)
    run.finish(shot.numbers, kind="reconstruction", metric=True,
               chunks=[{"start": start, "end": stop} for start, stop, *_ in chunks],
               loop_closure=loops, models=["DA3NESTED-GIANT-LARGE-1.1"])


if __name__ == "__main__":
    serve(main)
