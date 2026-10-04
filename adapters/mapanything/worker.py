"""Meta MapAnything worker: feed-forward multi-view metric reconstruction. Runs
inside third_party/mapanything/.venv with the pinned repo on sys.path; never
imports Lab2Shot core.

    python worker.py <job.json>

node "mapanything.reconstruct". Frames (every `step`-th, plus always the last)
-> chunks of at most `max_frames` views -> MapAnything.infer() per chunk (one
forward pass sees every view of the chunk) -> chunks aligned into one world ->
the reconstruction contract (lab2shot_worker.recon; node side
lab2shot/nodes/families/depth_camera.py WholeShotDepthCamera):

    raw/cameras.npz
        frames        int64   [F]      the reconstructed frames (every step-th + the last)
        K             float64 [F,3,3]  pixels at the input resolution; pixel (i, j) covers
                                       [j, j+1] x [i, i+1] (centre j+0.5): a centred
                                       principal point is (W/2, H/2)
        cam_to_world  float64 [F,4,4]  OpenCV camera (+X right, +Y down, +Z forward);
                                       world = the first frame's camera; metres
        focal_px_model float64 [F]     the network's own focal (differs from K only when
                                       focal_px was given)
    raw/frame_<n>.npz (reconstructed frames only)
        depth       float32 [H,W]  camera Z, metres, input resolution (raw prediction
                                   everywhere, also outside the mask; non-finite -> 0)
        confidence  float32 [H,W]  MapAnything confidence (>= 1, unitless, higher = better)
        mask        bool    [H,W]  valid: unambiguous, not a depth edge, inside the crop
        points      float32 [H,W,3] MapAnything's own 3D point per pixel (pts3d_cam,
                                   model.py:2103), in that frame's camera, metres: the
                                   camera-space form of its world points pts3d
                                   (model.py:2102) — the node's 「点云」

Frames between the reconstructed ones (step > 1) are left to the node to interpolate.

Long shots: consecutive chunks share >= `overlap` frames. Each chunk is its own
metric reconstruction in its own world; lab2shot_worker.recon.Stitcher brings the
next one into the running world (rotation from the shared frames' cameras, scale
and translation from the same pixels' 3D points; unreliable pixels left out)
and cross-fades cameras and depth across the shared frames.
The world scale is the first chunk's metric scale.

MapAnything has no input for masks: it sees the whole frame, moving objects
included, and nothing is taken out afterwards. To reconstruct only part of the
picture, black the rest out before the 「RGB」 input (「ViTDet 人物框」 → 「人物框转遮罩」 →
「图像合成」 set to 留下), where it is visible on the node graph.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from lab2shot_worker import (
    MemoryBound, fail, local_hub, progress, read_frame, recon, reason, require_weights, resident, say, serve,
)
from lab2shot_worker.frame_io import Writer
from lab2shot_worker.run import Run

NODE = "mapanything.reconstruct"
# Views per forward pass, as image tokens (14x14 px patches of the model input: 777 per
# frame at 518x294 / 294x518, 1036 at 518x392, 1369 at 518x518). Every view of a chunk
# attends to every other view. Measured on the RTX 4090 (24 GB), memory-efficient mode,
# 16:9 plates: 150 frames peak 18.5 GB, 200 frames 23.5 GB (the limit), 300 do not fit.
DEFAULT_TOKENS = 150 * 777  # default max_frames: 150 frames at 16:9, 112 at 4:3, 85 square
MAX_TOKENS = 200 * 777
# what the memory grows with, when it runs out: 每段最多帧数, stepping down from 200 to 8
MAX_FRAMES = MemoryBound.parameter("max_frames", (200, 150, 100, 64, 32, 16, 8))
RESOLUTION_SET = 518  # upstream's default: longest side 518 px, aspect from a fixed list
PATCH = 14
OVERLAP_MIN, OVERLAP_MAX = 4, 16  # frames shared by consecutive chunks
# Upstream's edge filter (infer(mask_edges=True)): depth jumps and normal creases.
EDGE_NORMAL_DEG = 5.0
EDGE_DEPTH_RTOL = 0.03
# Upstream's dense-head memory estimate per view (memory-efficient mode, 518 px input).
HEAD_BYTES_PER_VIEW = 680 * 2**20


# ---------------------------------------------------------------------- offline model loading


@resident
def load_model(repo: Path, checkpoint_dir: Path, device: torch.device):
    """What MapAnything.from_pretrained does for a local folder, but built directly on the
    GPU (no 5 GB random init + 5 GB state dict in RAM) and with a strict key check."""
    sys.path.insert(0, str(repo))
    from mapanything.models.mapanything.model import MapAnything
    from safetensors.torch import load_model as load_safetensors

    weights = checkpoint_dir / "model.safetensors"
    require_weights("mapanything", weights)
    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    config["encoder_config"]["torch_hub_pretrained"] = False  # as MapAnything._from_pretrained
    with device:
        model = MapAnything(**config)
    # The file stores each shared tensor once (dense_head.0 == dpt_feature_head, ...):
    # safetensors resolves those aliases; anything else missing is an error.
    load_safetensors(model, str(weights), strict=True, device=str(device))
    # a few buffers are made with torch.FloatTensor(...), which ignores the device context
    return model.to(device).eval()


def cap_dense_head_batches(model, cap_bytes: int) -> None:
    """Upstream sizes the memory-efficient dense-head batches from the card's free memory;
    under limit_gpu_memory() the usable memory is the cap minus what is allocated. Set by every job (its cap)."""

    def minibatch(memory_safety_factor: float = 0.95) -> int:
        free = min(torch.cuda.mem_get_info()[0], cap_bytes - torch.cuda.memory_allocated())
        return max(1, int(free * memory_safety_factor / HEAD_BYTES_PER_VIEW))

    model._compute_adaptive_minibatch_size = minibatch


# ---------------------------------------------------------------------- image geometry


@dataclass(frozen=True)
class Geometry:
    """Input frame (W x H) -> model input (tw x th): uniform resize to rw x rh, then a
    centred crop at (left, top): upstream's preprocessing (fixed_mapping, 518)."""

    W: int
    H: int
    tw: int
    th: int
    rw: int
    rh: int
    left: int
    top: int

    @property
    def sx(self) -> float:
        return self.rw / self.W

    @property
    def sy(self) -> float:
        return self.rh / self.H

    @property
    def tokens(self) -> int:
        return (self.tw // PATCH) * (self.th // PATCH)

    # Two pixel conventions: MapAnything's rays put pixel centres on integers (OpenCV);
    # our K (the reconstruction contract) puts them at +0.5 (pixel (i, j) covers [j, j+1]).

    def to_model_K(self, K: np.ndarray) -> np.ndarray:
        """Our K at the input resolution -> MapAnything's K at the model's input."""
        out = K.astype(np.float64).copy()
        out[0, 0] *= self.sx
        out[1, 1] *= self.sy
        out[0, 2] = K[0, 2] * self.sx - 0.5 - self.left
        out[1, 2] = K[1, 2] * self.sy - 0.5 - self.top
        return out

    def to_input_K(self, K: np.ndarray) -> np.ndarray:
        """MapAnything's K at the model's input -> our K at the input resolution."""
        out = K.astype(np.float64).copy()
        out[0, 0] /= self.sx
        out[1, 1] /= self.sy
        out[0, 2] = (K[0, 2] + self.left + 0.5) / self.sx
        out[1, 2] = (K[1, 2] + self.top + 0.5) / self.sy
        return out

    def input_to_model_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """cv2.remap maps: for every input pixel, where it lies in the model's grid."""
        x = (np.arange(self.W, dtype=np.float32) + 0.5) * self.sx - 0.5 - self.left
        y = (np.arange(self.H, dtype=np.float32) + 0.5) * self.sy - 0.5 - self.top
        return np.broadcast_to(x[None, :], (self.H, self.W)).copy(), np.broadcast_to(y[:, None], (self.H, self.W)).copy()

def plan_geometry(width: int, height: int) -> Geometry:
    from mapanything.utils.image import find_closest_aspect_ratio

    tw, th = find_closest_aspect_ratio(width / height, RESOLUTION_SET)
    scale = max(tw / width, th / height) + 1e-8
    rw, rh = int(np.floor(width * scale)), int(np.floor(height * scale))
    return Geometry(width, height, tw, th, rw, rh, (rw - tw) // 2, (rh - th) // 2)


def plan_max_frames(requested: int | None, geo: Geometry) -> int:
    """Frames per segment: what was asked, at most what 24 GB holds for this picture's aspect (`limit`). The node's
    tiers are measured on 16:9 (nodes.py max_frames_param): on 4:3 or square the same number does not fit, so it steps
    down to the limit and says so (N-MAPANYTHING-FEWERFRAMES) rather than failing: the user filled in nothing wrong."""
    limit = MAX_TOKENS // geo.tokens
    if requested is None:
        return DEFAULT_TOKENS // geo.tokens
    if requested > limit:
        say("N-MAPANYTHING-FEWERFRAMES", requested=requested, limit=limit)
        return limit
    return requested


def model_image(rgb: np.ndarray, geo: Geometry, frame: int) -> torch.Tensor:
    """sRGB uint8 [H,W,3] -> normalized [1,3,th,tw] exactly as upstream's preprocessing."""
    import PIL.Image
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    if rgb.shape[:2] != (geo.H, geo.W):
        fail("E-WORKER-FRAMESIZE", width=rgb.shape[1], height=rgb.shape[0], where=reason("I-WORKER-ATFRAME", frame=frame),
             first_width=geo.W, first_height=geo.H)
    img = PIL.Image.fromarray(rgb)
    resample = PIL.Image.Resampling.LANCZOS if geo.rw < geo.W else PIL.Image.Resampling.BICUBIC
    img = img.resize((geo.rw, geo.rh), resample=resample).crop((geo.left, geo.top, geo.left + geo.tw, geo.top + geo.th))
    norm = IMAGE_NORMALIZATION_DICT["dinov2"]
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    t = (t - norm.mean.view(3, 1, 1)) / norm.std.view(3, 1, 1)
    return t[None]


# ---------------------------------------------------------------------- reconstruction


@dataclass
class Shot:
    job: object
    keyframes: list[tuple[int, Path]]
    geo: Geometry
    K_given: np.ndarray | None

    @property
    def numbers(self) -> list[int]:
        return [f for f, _ in self.keyframes]


def infer_chunk(model, shot: Shot, s0: int, s1: int) -> recon.Chunk:
    """One forward pass over keyframes [s0, s1) -> per view (model resolution): depth, confidence, valid
    (unambiguous, no depth / normal edge), MapAnything's K, cam_to_world (the chunk's own world), and its own
    camera-space point map."""
    from mapanything.utils.geometry import depth_edge, normals_edge, points_to_normals

    geo = shot.geo
    views = []  # frames streamed: only the small model-size images stay in memory
    for j, (frame, path) in enumerate(shot.keyframes[s0:s1]):
        view = {"img": model_image(read_frame(path), geo, frame), "data_norm_type": ["dinov2"],
                "true_shape": np.int32([[geo.th, geo.tw]]), "idx": j, "instance": str(j)}
        if shot.K_given is not None:
            view["intrinsics"] = torch.from_numpy(geo.to_model_K(shot.K_given)).float()[None]
        views.append(view)
    preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                        apply_mask=False, mask_edges=False, apply_confidence_mask=False)
    del views
    out = {k: [] for k in ("depth", "conf", "valid", "K", "c2w", "points")}
    for j, pr in enumerate(preds):
        pw = torch.nan_to_num(pr["pts3d"][0].float()).cpu().numpy()
        d = torch.nan_to_num(pr["depth_z"][0, ..., 0].float()).cpu().numpy()
        ok = pr["non_ambiguous_mask"][0].cpu().numpy().astype(bool) & (d > 0)
        if ok.any():
            normals, normals_ok = points_to_normals(pw, mask=ok)
            ok &= ~(depth_edge(d, rtol=EDGE_DEPTH_RTOL, mask=ok) & normals_edge(normals, tol=EDGE_NORMAL_DEG, mask=normals_ok))
        out["depth"].append(d)
        out["conf"].append(pr["conf"][0].float().cpu().numpy())
        out["valid"].append(ok)
        out["K"].append(pr["intrinsics"][0].double().cpu().numpy())
        out["c2w"].append(pr["camera_poses"][0].double().cpu().numpy())
        # 官方的世界点图 pts3d（model.py:2102）在每一段自己的世界里；它的相机空间形式 pts3d_cam
        # （model.py:2103）是同一份数据换了坐标系，跟着深度一起被 recon.Stitcher 缩放进整片的世界
        # （Chunk.scaled），节点那边再按相机放回世界
        out["points"].append(torch.nan_to_num(pr["pts3d_cam"][0].float()).cpu().numpy())
    del preds
    torch.cuda.empty_cache()
    return recon.Chunk(cam_to_world=np.stack(out["c2w"]), K=np.stack(out["K"]), depth=np.stack(out["depth"]),
                       confidence=np.stack(out["conf"]), usable=np.stack(out["valid"]),
                       extra={"points": np.stack(out["points"])}, scaled=("points",))


def reconstruct(run: Run, model, shot: Shot, max_frames: int) -> dict:
    """All chunks, stitched (lab2shot_worker.recon) -> raw/frame_<n>.npz written; cameras and statistics returned."""
    geo, numbers, n = shot.geo, shot.numbers, len(shot.keyframes)
    raw = shot.job.raw_dir
    overlap = int(np.clip(max_frames // 6, OVERLAP_MIN, OVERLAP_MAX))
    chunks = recon.plan_chunks(n, max_frames, overlap)
    in_x, in_y = geo.input_to_model_maps()
    r = {"K": np.zeros((n, 3, 3)), "c2w": np.zeros((n, 4, 4)), "focal_model": np.zeros(n), "principal": np.zeros((n, 2)),
         "depth_median": np.full(n, np.nan), "chunks": [], "overlap": overlap if len(chunks) > 1 else 0, "infer_seconds": 0.0}
    stitch = recon.Stitcher()
    written = 0
    with Writer(threads=2, max_pending=8) as writer:
        for ci, (s0, s1) in enumerate(chunks):
            run.stage("reconstruct_segment", segment=ci + 1, segments=len(chunks), frames=s1 - s0) if len(chunks) > 1 else run.stage("reconstruct", frames=n)
            t = time.time()
            chunk = infer_chunk(model, shot, s0, s1)
            r["infer_seconds"] += time.time() - t
            try:
                info = stitch.add(s0, chunk)
            except ValueError:
                fail("E-MAPANYTHING-NOOVERLAP", segment=ci + 1)
            info["frames"] = [numbers[s0], numbers[s1 - 1]]
            if ci:
                if info["residual_rms_relative"] > 0.05:
                    say("W-MAPANYTHING-SEAM", segment=ci + 1, residual=float(info["residual_rms_relative"]))
                if not 0.8 < info["scale"] < 1.25:
                    say("W-MAPANYTHING-SEAMSCALE", segment=ci + 1, scale=float(info["scale"]))
            r["chunks"].append(info)
            del chunk

            # Write the frames no later chunk shares, at the input resolution.
            for fr in list(stitch.pop(chunks[ci + 1][0] if ci + 1 < len(chunks) else None)):
                i, frame = fr.index, numbers[fr.index]
                K_in = geo.to_input_K(fr.K)
                r["focal_model"][i] = 0.5 * (K_in[0, 0] + K_in[1, 1])
                r["principal"][i] = K_in[0, 2], K_in[1, 2]
                r["K"][i] = shot.K_given if shot.K_given is not None else K_in
                r["c2w"][i] = fr.cam_to_world
                depth = cv2.remap(fr.depth, in_x, in_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                conf = cv2.remap(fr.confidence, in_x, in_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                # outside the model's crop: invalid (constant border 0)
                ok = cv2.remap(fr.usable.astype(np.uint8), in_x, in_y, cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
                ok &= depth > 0
                if ok.any():
                    r["depth_median"][i] = float(np.median(depth[ok]))
                points = cv2.remap(fr.extra["points"], in_x, in_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                writer.submit(recon.save_frame, raw, frame, depth, conf, ok, points=points)
                written += 1
                progress(written, n, "write_depth")
    return r


def main(job_path: str) -> None:
    run = Run.start(job_path, NODE, "MapAnything")
    job = run.job
    p = job.params  # max_frames None: what fits 24 GB for this frame shape (plan_max_frames)
    repo_id, licence = p["weights_repo"], p["weights_license"]  # from the node (extension.py MODELS, the one table)

    frames = run.frames(step=p["step"])  # every step-th frame and always the last: nothing to extrapolate
    keyframes, width, height = frames.pairs, frames.width, frames.height
    device = torch.device("cuda")

    # UniCeption builds its DINOv2 backbone through torch.hub: from the pinned code (structure only, the trained
    # weights are in MapAnything's model.safetensors)
    dinov2 = Path(os.environ["MAPANYTHING_DINOV2_CODE"])
    run.weights(dinov2 / "hubconf.py", what=reason("I-MAPANYTHING-DINOV2CODE"))
    local_hub({"dinov2": dinov2}, "MapAnything")
    # run.model caps the GPU first: an allocation beyond the card fails (fit_memory decides) instead of spilling into RAM
    model = run.model("load_model", load_model, job.repo_dir, job.weights_dir / repo_id, device,
                      stage_params={"model": "MapAnything"})
    cap_dense_head_batches(model, run.gpu_cap_mb * 2**20)

    geo = plan_geometry(width, height)
    focal_given = p["focal_px"]
    K_given = None if focal_given is None else np.array([[focal_given, 0, width / 2], [0, focal_given, height / 2], [0, 0, 1.0]])
    shot = Shot(job, keyframes, geo, K_given)
    t_solve = time.time()
    max_frames, r = run.fit(MAX_FRAMES, lambda m: (m, reconstruct(run, model, shot, m)),
                            min(plan_max_frames(p["max_frames"], geo), len(keyframes)))
    run.frame_seconds.extend([(time.time() - t_solve) / len(keyframes)] * len(keyframes))  # one pass over the shot: shared out per frame

    numbers = shot.numbers
    recon.save_cameras(job.raw_dir, numbers, r["K"], r["c2w"], width, height, focal_px_model=r["focal_model"])

    focal, fm = r["K"][:, 0, 0], r["focal_model"]
    rel = np.linalg.inv(r["c2w"][0]) @ r["c2w"]
    translation = np.linalg.norm(rel[:, :3, 3], axis=1)
    angle = np.degrees(np.arccos(np.clip((np.trace(rel[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    if focal_given is not None and abs(np.median(fm) / focal_given - 1) > 0.1:
        say("W-MAPANYTHING-FOCALMISMATCH", model=float(np.median(fm)), given=float(focal_given), diff=float(abs(np.median(fm) / focal_given - 1)))
    run.finish(
        numbers,
        kind="reconstruction",
        cameras="cameras.npz",
        files="cameras.npz: frames, K, cam_to_world, focal_px_model; frame_<n>.npz: depth, confidence, mask, points",
        convention=recon.CONVENTION,
        intrinsics_convention="K in pixels at the input resolution; pixel (i, j) covers [j, j+1] x [i, i+1] (centred principal point = W/2, H/2)",
        depth="camera Z in metres at the input resolution, raw prediction also outside the mask",
        confidence="MapAnything confidence, >= 1, unitless, higher = more reliable",
        mask="True = valid: unambiguous (model), not a depth / normal edge, inside the model's crop",
        points="MapAnything's own pts3d_cam (model.py:2103): the 3D point of every pixel in that frame's camera, "
               "metres; the camera-space form of its world points pts3d (model.py:2102)",
        units="metres",
        weights=p["model"],
        weights_repo=repo_id,
        weights_license=licence,
        width=width,
        height=height,
        fps=job.fps,
        frames=numbers,  # the whole list, not the standard [first, last]: the converter reads every frame's file by it
        step=p["step"],
        focal_source="user" if focal_given is not None else "mapanything",
        focal_px={"median": float(np.median(focal)), "min": float(focal.min()), "max": float(focal.max())},
        focal_px_model={"median": float(np.median(fm)), "min": float(fm.min()), "max": float(fm.max()),
                        "per_frame": [round(float(v), 2) for v in fm]},
        principal_point_px={"median": [float(v) for v in np.median(r["principal"], axis=0)]},
        depth_median_m=[None if np.isnan(v) else round(float(v), 4) for v in r["depth_median"]],
        model_input={"width": geo.tw, "height": geo.th, "resized": [geo.rw, geo.rh], "crop_left_top": [geo.left, geo.top]},
        max_frames=max_frames,
        overlap=r["overlap"],
        chunks=r["chunks"],
        chunk_alignment="lab2shot_worker.recon.Stitcher: rotation = mean of the shared frames' relative camera orientations, "
                        "scale + translation = least squares (weights 1/depth^2, 3x-median trimming) on the 3D points of the same "
                        "pixels of the shared frames; cameras, K and depth cross-fade across them",
        memory_efficient=True,
        drift={"max_translation_m": float(translation.max()), "max_rotation_deg": float(angle.max())},
        infer_seconds=round(r["infer_seconds"], 1),
    )


if __name__ == "__main__":
    serve(main)
