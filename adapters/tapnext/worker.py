"""TAPNext++ worker: long-range 2D point tracking. Runs inside
third_party/tapnext/.venv with the pinned tapnet repo on sys.path; never imports
Lab2Shot core.

    python worker.py <job.json>

Job contract and raw/tracks.npz layout: lab2shot_worker/point_tracks.py (shared by
every point-tracking worker).

TAPNext++ is a recurrent (online) tracker: one frame at a time, a fixed-size
state carried from frame to frame, so GPU memory does not grow with the shot
length. Every point is tracked forwards from its query frame to the end, and
backwards (the frames reversed) from its query frame to the start. Frames are
squeezed to a square (upstream's preprocessing) of `resolution` pixels.

    model "512"  tapnextpp_512.ckpt, fine-tuned at 512x512 input (default)
         "256"  tapnextpp_ckpt.pt, 256x256 input (the paper's checkpoint; faster)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

import lab2shot_worker.point_tracks as pt
from lab2shot_worker import progress, require_weights, resident, say, serve
from lab2shot_worker.run import Run

# model -> (checkpoint in weights/, native square input side)
CHECKPOINTS = {"512": ("tapnextpp_512.ckpt", 512), "256": ("tapnextpp_ckpt.pt", 256)}
MODEL_SPACE = 256  # TAPNext++ predicts coordinates on a 256 x 256 grid whatever the input size
MAX_POINTS_PER_PASS = 4096  # point tokens per rollout (more points: several passes over the shot)
# Upstream's VOTS 2026 setup: 64 extra "support" points on an 8 x 8 grid within
# 32 px (at 512 input = 1/16 of the frame) around each user point, tracked along
# for context and then dropped.
SUPPORT_PER_POINT = 64
SUPPORT_RADIUS = 1.0 / 16.0  # of the frame width / height
SUPPORT_MAX_TOTAL = 2048
CERTAINTY_RADIUS = 8  # model-space pixels, upstream tracker_certainty default


@resident
def load_model(repo: Path, checkpoint: Path, side: int, device: torch.device):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    wrapper = TAPNextPP.from_checkpoint(checkpoint, device=device, input_resolution=side)
    return wrapper._model  # the plain TAPNext module: batched tensors are fed to it directly


def support_points(xy: np.ndarray, t: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Upstream's local support grid around each user point (input pixels, same query frame)."""
    per = min(SUPPORT_PER_POINT, SUPPORT_MAX_TOTAL // max(1, len(xy)))
    side = int(np.sqrt(per))
    if side < 2:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.int64)
    rx, ry = SUPPORT_RADIUS * width, SUPPORT_RADIUS * height
    local = pt.grid_points(side, (-rx, -ry, rx, ry))
    pts = (xy[:, None, :] + local[None]).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0.5, width - 0.5)
    pts[:, 1] = np.clip(pts[:, 1], 0.5, height - 0.5)
    return pts.astype(np.float32), np.repeat(t, side * side)


@torch.no_grad()
def rollout(model, frames: pt.Frames, q_t: np.ndarray, q_xy: np.ndarray, device, tick) -> tuple:
    """Track forwards through `frames` (square uint8 frames in processing order) from each query's frame index q_t; q_xy in model space (x, y).
    Returns (xy [N, F, 2] model space, visible logit [N, F], certainty [N, F]);
    frames before a query's own frame are left as NaN / -inf / 0.
    """
    from tapnet.tapnext.tapnext_torch_utils import tracker_certainty

    n, f = len(q_t), len(frames)
    xy = np.full((n, f, 2), np.nan, np.float32)
    logit = np.full((n, f), -np.inf, np.float32)
    cert = np.zeros((n, f), np.float32)
    start = int(q_t.min())
    # TAPNext query layout: (t, y, x), t relative to the first frame given to the model.
    queries = torch.from_numpy(np.stack([q_t - start, q_xy[:, 1], q_xy[:, 0]], -1).astype(np.float32))[None].to(device)
    state = None
    for i in range(start, f):
        frame = torch.from_numpy(frames[i]).to(device, non_blocking=True).float().div_(127.5).sub_(1.0)[None, None]
        with torch.autocast("cuda", dtype=torch.float16):
            if state is None:
                tracks, track_logits, vis_logits, state = model(video=frame, query_points=queries)
            else:
                tracks, track_logits, vis_logits, state = model(video=frame, state=state)
        tracks = tracks.float()  # [1, 1, N, 2] (y, x)
        c = tracker_certainty(tracks, track_logits.float(), CERTAINTY_RADIUS)[0, 0, :, 0]
        xy[:, i] = tracks[0, 0].flip(-1).cpu().numpy()
        logit[:, i] = vis_logits[0, 0, :, 0].float().cpu().numpy()
        cert[:, i] = c.cpu().numpy()
        tick()
    started = np.arange(f)[None, :] >= q_t[:, None]
    xy[~started] = np.nan
    logit[~started] = -np.inf
    cert[~started] = 0.0
    return xy, logit, cert


def track_both_ways(model, frames: pt.Frames, q_t: np.ndarray, q_xy: np.ndarray, device, tick) -> tuple:
    """Forwards from each query frame to the end, backwards from it to the start.
    Returns xy, logit, cert and the forward pass's query-frame offsets."""
    xy, logit, cert = rollout(model, frames, q_t, q_xy, device, tick)
    offsets = pt.anchor(xy, q_t, q_xy)
    late = q_t > 0
    if late.any():
        last = int(q_t[late].max())
        rq = last - q_t[late]
        bxy, blogit, bcert = rollout(model, frames.reversed_upto(last), rq, q_xy[late], device, tick)
        pt.anchor(bxy, rq, q_xy[late])
        before = np.arange(last + 1)[None, :] < q_t[late][:, None]  # frames before each query (natural order)
        sub_xy, sub_logit, sub_cert = xy[late], logit[late], cert[late]
        sub_xy[:, : last + 1][before] = bxy[:, ::-1][before]
        sub_logit[:, : last + 1][before] = blogit[:, ::-1][before]
        sub_cert[:, : last + 1][before] = bcert[:, ::-1][before]
        xy[late], logit[late], cert[late] = sub_xy, sub_logit, sub_cert
    return xy, logit, cert, offsets


def plan_passes(n_points: int) -> list[slice]:
    return [slice(i, min(n_points, i + MAX_POINTS_PER_PASS)) for i in range(0, n_points, MAX_POINTS_PER_PASS)]


def main(job_path: str) -> None:
    run = Run.start(job_path, "tapnext.track", "TAPNext++")
    job = run.job
    frames = run.frames()
    frame_numbers = frames.numbers
    p = pt.parse_params(job.params, frame_numbers)
    ckpt_name, native = CHECKPOINTS[p.model]
    side = p.resolution or native
    if side % 8:
        given, side = side, int(round(side / 8)) * 8
        say("N-TAPNEXT-SIDEROUNDED", given=given, side=side)
    checkpoint = job.weights_dir / ckpt_name
    require_weights("tapnext", checkpoint)  # not run.weights: the extension is "tapnext", the project "TAPNext++"

    width, height = frames.width, frames.height
    queries = pt.build_queries(job, p, width, height)
    n_user = int(queries.user.sum())
    device = torch.device("cuda")

    run.stage("读取画面")
    t0 = time.time()
    video = pt.Frames(frames.paths, (side, side))
    video.preload(lambda d, t: progress(d, t, "读取画面"))
    load_frames_s = time.time() - t0

    model = run.model("TAPNext++ 模型", load_model, job.repo_dir, checkpoint, side, device)

    # User points get upstream's local support points; they are tracked in the
    # same pass (joint attention) and dropped afterwards.
    s_xy, s_t = support_points(queries.xy[queries.user], queries.t[queries.user], width, height) if n_user else (
        np.zeros((0, 2), np.float32), np.zeros(0, np.int64))
    all_xy = np.concatenate([queries.xy, s_xy])
    all_t = np.concatenate([queries.t, s_t])
    to_model = np.array([MODEL_SPACE / width, MODEL_SPACE / height], np.float32)

    passes = plan_passes(len(all_t))
    n_frames = len(frames)
    # frames stepped: forwards from the earliest query, backwards from the latest one
    steps = sum(n_frames - int(all_t[sl].min()) + int(all_t[sl].max()) + int(all_t[sl].max() > 0) for sl in passes)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        progress(done, steps, "跟踪")

    run.stage(f"跟踪 {len(queries)} 个点" + (f"（另加 {len(s_t)} 个辅助点）" if len(s_t) else ""))
    t2 = time.time()
    xy_m = np.empty((len(all_t), n_frames, 2), np.float32)
    logit = np.empty((len(all_t), n_frames), np.float32)
    cert = np.empty((len(all_t), n_frames), np.float32)
    offsets = np.empty((len(all_t), 2), np.float32)
    for sl in passes:
        xy_m[sl], logit[sl], cert[sl], offsets[sl] = track_both_ways(model, video, all_t[sl], all_xy[sl] * to_model,
                                                                     device, tick)
    track_s = time.time() - t2
    run.frame_seconds.extend([track_s / n_frames] * n_frames)  # one pass over the shot: shared out per frame
    offset_px = np.linalg.norm(offsets[: len(queries)] / to_model, axis=-1)  # input pixels

    n = len(queries)
    tracks = xy_m[:n] / to_model  # model space -> input pixels (both use the pixel-centre +0.5 convention)
    vis_prob = 1.0 / (1.0 + np.exp(-logit[:n].astype(np.float64)))
    visible = logit[:n] > 0  # upstream's default occlusion decision
    confidence = (vis_prob * cert[:n]).astype(np.float32)
    # tracks pass through the query points (anchor()); a query point is visible
    idx = np.arange(n)
    tracks[idx, queries.t] = queries.xy  # removes float round-off only
    visible[idx, queries.t] = True

    pt.write_result(
        run,
        tracks=tracks, visible=visible, confidence=confidence, queries=queries, frame_numbers=frame_numbers,
        width=width, height=height, p=p, user_points=n_user, query_offset_px=offset_px,
        read_seconds=load_frames_s, track_seconds=track_s,
        model="TAPNext++",
        checkpoint_choice=p.model,  # the node's 「模型」 parameter: which of the two checkpoints (512 / 256)
        checkpoint=ckpt_name,
        visible_rule="visibility logit > 0 (upstream default)",
        confidence_rule="sigmoid(visibility logit) x TAPNext certainty (probability mass within 8/256 of the frame)",
        processing_resolution=[side, side],
        processing_note="frames squeezed to a square (upstream preprocessing), area-averaged resize",
        direction="forwards from each query frame, backwards (reversed frames) before it",
        support_points=int(len(s_t)),
        passes=len(passes),
    )


if __name__ == "__main__":
    serve(main)
