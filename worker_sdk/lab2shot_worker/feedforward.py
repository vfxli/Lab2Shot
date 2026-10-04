"""Shared driver of the feed-forward multi-view reconstruction workers (VGGT, Pi3, Depth Anything 3).

Runs inside the extension's own environment (numpy, torch, cv2); never imports
Lab2Shot core. A worker supplies a Backend (model loading + one forward
pass over a chunk of frames); this module does everything else:

    frames (every `step`-th, plus the last) -> model input size -> chunks of at most
    `max_frames` frames with overlap -> one forward pass per chunk -> chunks joined
    by a similarity transform fitted on the overlap frames (recon.Stitcher: rotation
    from their cameras, scale and translation from their 3D points), loops closed
    when asked -> world = first frame's camera -> raw/ at the INPUT resolution.

Outputs (raw/):

    cameras.npz     frames        int64   [F]       frame numbers actually used
                    K             float64 [F,3,3]   pixels at the input resolution, pixel (i, j)
                                                    covers [j, j+1] x [i, i+1]; principal
                                                    point at the image centre (W/2, H/2)
                    cam_to_world  float64 [F,4,4]   OpenCV camera (+X right, +Y down, +Z forward);
                                                    world = first used frame's camera
                    width, height int64             input resolution
    frame_<n>.npz   depth         float32 [H,W]     camera Z, same units as cam_to_world
                    confidence    float32 [H,W]     the model's own confidence (see result.json)
                    mask          bool    [H,W]     True = reliable geometry: confident, not on a
                                                    depth edge
                    points        float32 [H,W,3]   only when the backend predicts a point map of
                                                    its own (Chunk.extra["points"]): the model's
                                                    3D point per pixel in that frame's camera
                                                    (OpenCV axes, same units as depth) — the node's
                                                    「点云」 (families/base.py native_points)
    result.json     scale convention, chunks, alignment residuals, focal, drift, time, memory

These models take no mask input: they see the whole frame, moving people included,
and nothing is taken out of the result afterwards. To reconstruct only part of the
picture, black the rest out before the 「图像」 input (人物检测 → 人物框转遮罩 →
图像合成, multiplied), where it is visible on the node graph.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from . import MemoryBound, fail, progress, read_frame, recon, say
from .frame_io import FrameReader
from .run import Run

PATCH = 14  # every model driven here is a ViT with 14 px patches: input sides are multiples of 14
OVERLAP_FRACTION = 0.25  # chunk overlap for long shots (at least 2 frames)
LOOP_WINDOW = 16  # frames on each side of a loop chunk (fewer when a chunk is shorter)
LOOP_CHUNKS = "loop_chunks"  # the scratch folder in the job folder where chunks wait until loops are closed (then deleted)
# what the memory grows with, when it runs out: 每段最多帧数, within every model's node ceiling (DA3 110, VGGT 130, Pi3 150)
MAX_FRAMES = MemoryBound.parameter("max_frames", (100, 64, 32, 16, 8))


# ---------------------------------------------------------------------- backend


@dataclass
class Backend:
    name: str  # model shown in result.json, e.g. "VGGT-1B"
    weights: str  # the variant that was loaded
    metric: bool  # True: depth / translations in (approximate) metres
    # Default frames per forward pass: this many image patches (14x14 px, padding
    # included) fit a 24 GB GPU; memory grows with the patch count.
    patch_budget: int
    # (width, height) -> model input (w, h, pad_w, pad_h): the image is resized to w x h
    # and padded by pad_w / pad_h pixels on each side (0 = no padding).
    input_size: Callable[[int, int, int | None], tuple[int, int, int, int]]
    # images float32 [N,3,H,W] in 0..1 (padded) on the GPU -> recon.Chunk of the unpadded region (h x w, K in
    # its pixels, corner convention; usable = confident enough: the driver also removes depth edges)
    infer: Callable[[torch.Tensor, int, int, int, int], recon.Chunk]
    confidence: str  # what the confidence numbers mean (result.json)
    models: list[str] = field(default_factory=list)
    licence: str = ""
    # The upstream name of the model's own 3D point map (written into result.json). Non-empty: infer() returns a
    # camera-space point map in Chunk.extra["points"] and the driver writes it into every frame's npz; "": depth only.
    points: str = ""


def pixel_budget_size(width: int, height: int, budget: int) -> tuple[int, int]:
    """Pi3's own rule: largest multiple-of-14 size with the input aspect and at most `budget` pixels."""
    s = math.sqrt(budget / (width * height))
    k, m = round(width * s / PATCH), round(height * s / PATCH)
    while (k * PATCH) * (m * PATCH) > budget:
        if k / m > width / height:
            k -= 1
        else:
            m -= 1
    return max(1, k) * PATCH, max(1, m) * PATCH


# ---------------------------------------------------------------------- inputs


def model_image(path: Path, w: int, h: int, pad_w: int, pad_h: int) -> np.ndarray:
    """float32 [3, h + 2 pad_h, w + 2 pad_w] in 0..1, padding white (as VGGT's own loader)."""
    rgb = read_frame(path, "float32")
    interp = cv2.INTER_AREA if rgb.shape[1] > w else cv2.INTER_CUBIC
    rgb = np.clip(cv2.resize(rgb, (w, h), interpolation=interp), 0.0, 1.0)
    if pad_w or pad_h:
        rgb = cv2.copyMakeBorder(rgb, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(1.0, 1.0, 1.0))
    return rgb.transpose(2, 0, 1)


# ---------------------------------------------------------------------- main


def run(job_path: str, node: str, make_backend: Callable[[object], Backend], model: str = "") -> None:
    """`make_backend(job)` validates job.params["model"] and loads the model; `model`: its name as the loading stage
    says it (a proper name, the same in every language; default: the project)."""
    run = Run.start(job_path, node, node.split(".")[0])
    job, params = run.job, run.params
    step, resolution = params["step"], params["resolution"]

    frames = run.frames(step=step)  # every `step`-th frame and the last: the camera always covers the whole shot
    used, frame_numbers = frames.pairs, frames.numbers
    width, height = frames.width, frames.height
    raw = job.raw_dir

    # Above the card's memory WSL spills into system RAM (10x slower): out-of-memory instead (run.model caps it).
    backend = run.model("load_model", make_backend, job, stage_params={"model": model or run.project})
    w, h, pad_w, pad_h = backend.input_size(width, height, resolution)
    patches = ((w + 2 * pad_w) // PATCH) * ((h + 2 * pad_h) // PATCH)
    max_frames = params["max_frames"] or max(2, backend.patch_budget // patches)
    sx, sy = width / w, height / h

    def solve(max_frames: int) -> dict:
        """The whole shot in chunks of at most `max_frames`: stitched, loops closed, every frame saved; the cameras
        and what result.json records."""
        overlap = min(max(2, round(OVERLAP_FRACTION * max_frames)), max_frames - 1)
        chunks = recon.plan_chunks(len(used), max_frames, overlap)
        overlap = overlap if len(chunks) > 1 else 0
        print(f"{backend.name}: {len(used)} frames at {w}x{h} (+pad {pad_w},{pad_h}), "
              f"{len(chunks)} chunk(s) of <= {max_frames}, overlap {overlap}", flush=True)
        loops = params["loops"] and len(chunks) > 2  # loop closure needs chunks that share nothing
        stitch = recon.Stitcher(keep=job.scratch(LOOP_CHUNKS) if loops else None)
        chunk_info = []
        infer_seconds = 0.0
        K_in = np.zeros((len(used), 3, 3))
        c2w = np.zeros((len(used), 4, 4))
        depth_median = [float("nan")] * len(used)
        written = 0

        def save(frames) -> None:
            """Stitched frames at the input resolution; K and cameras collected for cameras.npz."""
            nonlocal written
            for fr in frames:
                i = fr.index
                K_in[i] = fr.K
                K_in[i, 0, :] *= sx
                K_in[i, 1, :] *= sy
                c2w[i] = fr.cam_to_world
                if fr.usable.any():
                    depth_median[i] = float(np.median(fr.depth[fr.usable]))
                d, c, m = recon.at_input_size(fr, width, height)
                extra = {}
                if "points" in fr.extra:  # the model's own point map (Chunk.extra / Chunk.scaled), at the plate size
                    extra["points"] = cv2.resize(fr.extra["points"], (width, height), interpolation=cv2.INTER_LINEAR)
                recon.save_frame(raw, frame_numbers[i], d, c, m, **extra)
                written += 1
                if written % 10 == 0 or written == len(used):
                    progress(written, len(used), "write_depth")

        def reconstruct(indices: list[int]) -> recon.Chunk:
            """The model on these frames together; usable = confident and off depth edges."""
            nonlocal infer_seconds
            images = np.stack(reader.take(indices))
            t = time.time()
            chunk = backend.infer(torch.from_numpy(images).cuda(), w, h, pad_w, pad_h)
            infer_seconds += time.time() - t
            chunk.usable &= ~recon.depth_edges(chunk.depth) & (chunk.depth > 0)
            if loops:
                chunk.looks = recon.thumbnails(images)
            return chunk

        run.stage("feedforward")
        with FrameReader(frames.paths, lambda path: model_image(path, w, h, pad_w, pad_h), threads=4,
                         ahead=0) as reader:
            for ci, (a, b) in enumerate(chunks):
                chunk = reconstruct(list(range(a, b)))
                try:
                    info = stitch.add(a, chunk)
                except ValueError:
                    fail("E-FEEDFWD-STITCH", chunk=ci + 1)
                info["frames"] = [frame_numbers[a], frame_numbers[b - 1]]
                chunk_info.append(info)
                del chunk
                save(stitch.pop(chunks[ci + 1][0] if ci + 1 < len(chunks) else None))  # nothing while loops wait
                progress(ci + 1, len(chunks), "feedforward")
            if loops:
                run.stage("loop_closure")
                window = max(4, min(LOOP_WINDOW, max_frames // 2))
                candidates = stitch.loop_candidates(window, max(2, len(chunks)))
                for k, (first, second) in enumerate(candidates):
                    loop = stitch.add_loop(first, second, reconstruct(first + second))
                    loop["frames"] = [frame_numbers[i] for i in loop["frames"]]
                    progress(k + 1, len(candidates), "loop_closure")
        save(stitch.close())
        return {"max_frames": max_frames, "overlap": overlap, "chunks": chunk_info, "loops": loops, "stitch": stitch,
                "infer_seconds": infer_seconds, "K": K_in, "cam_to_world": c2w, "depth_median": depth_median}

    t_solve = time.time()
    solved = run.fit(MAX_FRAMES, solve, min(max_frames, len(used)))
    run.frame_seconds.extend([(time.time() - t_solve) / len(used)] * len(used))  # the shot is solved in one go: shared out per frame
    stitch, loops = solved["stitch"], solved["loops"]
    shutil.rmtree(job.dir / LOOP_CHUNKS, ignore_errors=True)
    if loops:
        used_loops = sum(bool(lp["used"]) for lp in stitch.loops)
        fix = stitch.loop_correction
        if used_loops:
            say("I-FEEDFWD-LOOPS", loops=len(stitch.loops), used=used_loops, turn_deg=fix["max_turn_deg"],
                move=fix["max_move_over_depth"])
        else:
            say("N-FEEDFWD-NOLOOPS")

    run.stage("write_results")
    K_in, c2w = solved["K"], solved["cam_to_world"]
    recon.save_cameras(raw, frame_numbers, K_in, c2w, width, height)

    numbers = recon.summary(c2w, K_in, width, solved["depth_median"])
    focal = numbers["focal_px"]
    if len(used) > 1 and (spread := (focal["max"] - focal["min"]) / (2 * focal["median"])) > 0.05:
        say("W-FEEDFWD-FOCALSPREAD", model=backend.name, low=focal["min"], high=focal["max"], spread=spread)
    units = ("metres (approximate, from the model's metric scale head)" if backend.metric
             else "arbitrary: the model's own normalised scale (not metres); consistent within the shot")
    run.finish(
        frame_numbers,
        kind="reconstruction",
        cameras="cameras.npz",
        files="cameras.npz: frames, K, cam_to_world, width, height; frame_<n>.npz: depth, confidence, mask"
              + (", points" if backend.points else ""),
        model=backend.name,
        weights=backend.weights,
        models=backend.models,
        weights_license=backend.licence,
        convention=recon.CONVENTION,
        intrinsics_convention="K in pixels at the input resolution; pixel (i, j) covers [j, j+1] x [i, i+1] "
                              "(principal point at the image centre = (W/2, H/2))",
        depth=f"camera Z at the input resolution ({'metres' if backend.metric else 'same arbitrary unit as cam_to_world'}), "
              "raw prediction also outside the mask",
        metric=backend.metric,
        units=units,
        confidence=backend.confidence,
        points=f"the model's own 3D point map ({backend.points}), per pixel in that frame's camera, same units "
               "as depth" if backend.points else "not predicted by this model (depth only)",
        mask="True = reliable geometry: confident and not on a depth edge (the model sees the whole frame, "
             "moving objects included, and nothing is taken out afterwards)",
        width=width,
        height=height,
        model_input={"width": w, "height": h, "pad_width": pad_w, "pad_height": pad_h},
        step=step,
        frames=frame_numbers,  # the whole list, not the standard [first, last]: the converter reads every frame's file by it
        max_frames=solved["max_frames"],
        max_frames_source="user" if params.get("max_frames") is not None else f"default: {backend.patch_budget} patches per pass",
        overlap=solved["overlap"],
        chunks=solved["chunks"],
        alignment="consecutive chunks share `overlap` frames, joined by lab2shot_worker.recon.Stitcher: rotation = mean "
                  "of the shared frames' relative camera orientations, scale + translation = least squares (weights "
                  "1/depth^2, 3x-median outlier trimming) on the 3D points of the same pixels of the shared frames "
                  "(confident, static only); cameras, K and depth cross-fade across the shared frames",
        loops=stitch.loops if loops else "not needed: the shot fits in two chunks" if params["loops"] else "off",
        loop_correction=stitch.loop_correction,
        **numbers,
        infer_seconds=round(solved["infer_seconds"], 1),
    )


def load_state(model: torch.nn.Module, path: Path, required: tuple[str, ...]) -> None:
    """safetensors -> model (strict about the parts this worker uses; unused heads may be absent from the model)."""
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    missing, _unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k.startswith(required)]
    if missing:
        fail("E-FEEDFWD-WEIGHTS", file=path.name, count=len(missing), names=missing[:3])
    del state


