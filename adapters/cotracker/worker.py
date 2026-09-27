"""CoTracker3 worker: long-range 2D point tracking. Runs inside
third_party/cotracker/.venv with the pinned co-tracker repo on sys.path; never
imports Lab2Shot core.

    python worker.py <job.json>

Job contract and raw/tracks.npz layout: lab2shot_worker/point_tracks.py (shared by
every point-tracking worker).

    mode "offline"  scaled_offline.pth: every point sees a whole window of frames at
                    once, forwards and backwards (best quality; upstream default).
                    A window holds as many frames as fit the GPU (up to
                    OFFLINE_MAX_WINDOW); longer shots are cut into windows that
                    overlap by one frame, and each point continues into the next
                    window from the position the previous window found on that
                    frame (re-queried there).
         "online"   scaled_online.pth: upstream's sliding windows of 16 frames
                    (step 8), memory independent of the shot length. Tracks
                    forwards from each query frame, then backwards over the
                    reversed frames for the part of the shot before it.

Processing resolution: resolution = null -> upstream's 512 x 384 (W x H) whatever
the frame's aspect ratio (the frames are squeezed, as upstream does); a number
-> that long side with the aspect ratio kept (short side a multiple of 32).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

import lab2shot_worker.point_tracks as pt
from lab2shot_worker import fit_size, progress, require_weights, resident, say, serve
from lab2shot_worker.run import Run

CHECKPOINTS = {"offline": "scaled_offline.pth", "online": "scaled_online.pth"}
WINDOW_LEN = {"offline": 60, "online": 16}  # the models' time embeddings (upstream hubconf)
NATIVE_HW = (384, 512)  # upstream model_resolution (H, W)
ITERS = 6  # refinement iterations (upstream predictor)
# Feature encoder batch (upstream: 200 frames at once, ~70 MB of activations per frame
# at 512x384; the result is the same, only the peak memory differs).
FNET_CHUNK = 16
SUPPORT_GRID = 6  # upstream: a 6 x 6 support grid is tracked along with sparse / masked points
VISIBLE_THRESHOLD = {"offline": 0.9, "online": 0.6}  # upstream predictors (offline: vis; online: vis x conf)

# Offline windows: peak GPU memory ~ frames x (FRAME_BYTES_PER_PIXEL x pixels +
# POINT_BYTES x points) + FNET_CHUNK x FNET_BYTES_PER_PIXEL x pixels (the feature
# encoder's activations), fp32, measured on the RTX 4090 (e.g. 120 frames x 800
# points at 512x384: 9.3 GB). Windows are planned to stay under OFFLINE_BUDGET.
OFFLINE_BUDGET = 14 * 2**30
OFFLINE_FRAME_BYTES_PER_PIXEL = 66.0
OFFLINE_POINT_BYTES = 81e3
FNET_BYTES_PER_PIXEL = 443.0
OFFLINE_MAX_WINDOW = 240
OFFLINE_MIN_WINDOW = 48
# 段与段只重叠一帧：上一段在这一帧上算出来的位置，就是下一段里这些点的查询点 (t, x, y)。
# 上游 offline predictor 本身是整段一次、没有分段；这里只改喂数据的方式，不改模型里的任何一步
#（不做多帧重叠的交叉淡化，也不替换 get_track_feat 之类的内部方法）。
OFFLINE_OVERLAP = 1
OFFLINE_MAX_POINTS = 1536  # points per pass (joint attention); more points: several passes


@resident
def load_model(repo: Path, checkpoint: Path, mode: str, hw: tuple[int, int], device):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from cotracker.models.build_cotracker import build_cotracker

    model = build_cotracker(str(checkpoint), offline=mode == "offline", window_len=WINDOW_LEN[mode])
    # The model normalises motion by its resolution; at another size it must know it.
    model.model_resolution = hw
    return model.to(device).eval()


def processing_size(width: int, height: int, resolution: int | None) -> tuple[int, int]:
    """(h, w) the model runs at."""
    if resolution is None:
        return NATIVE_HW
    w, h = fit_size(width, height, resolution, 32, minimum=64)
    return h, w


def support_queries(hw: tuple[int, int]) -> np.ndarray:
    """Upstream's support grid (get_points_on_a_grid(6, (H, W))), model pixels (x, y)."""
    h, w = hw
    margin = w / 64
    xs = np.linspace(margin, w - margin, SUPPORT_GRID)
    ys = np.linspace(margin, h - margin, SUPPORT_GRID)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], -1).astype(np.float32)


def to_tensor(clip: np.ndarray, device) -> torch.Tensor:
    """uint8 [T, h, w, 3] -> float [1, T, 3, h, w] in 0..255 (the model normalises)."""
    return torch.from_numpy(clip).to(device).permute(0, 3, 1, 2)[None].float()


# ---------------------------------------------------------------------- offline


def offline_windows(n_frames: int, window: int) -> list[tuple[int, int]]:
    """Equal windows of at most `window` frames, consecutive ones overlapping by
    OFFLINE_OVERLAP frames."""
    if n_frames <= window:
        return [(0, n_frames)]
    k = math.ceil((n_frames - OFFLINE_OVERLAP) / (window - OFFLINE_OVERLAP))
    length = math.ceil((n_frames + (k - 1) * OFFLINE_OVERLAP) / k)
    step = length - OFFLINE_OVERLAP
    return [(i * step, min(n_frames, i * step + length)) for i in range(k)]


@torch.no_grad()
def offline_sweep(model, video: pt.Frames, q_t: np.ndarray, q_xy: np.ndarray, support: np.ndarray | None,
                  window: int, device, tick) -> tuple:
    """Track points through `video` (processing order) window by window, forwards.

    段与段只重叠一帧：上一段在这一帧上算出来的位置作为下一段的查询点 (t, x, y) 喂给上游 predictor；
    模型内部不做任何改动，段间也不做混合。

    Returns xy [N, F, 2] (model pixels, centre-at-integer), vis, conf [N, F]
    (probabilities) and start [N], the first frame each point has values for
    (the start of the first window holding its query frame; earlier frames are NaN).
    """
    n, f = len(q_t), len(video)
    xy = np.full((n, f, 2), np.nan, np.float32)
    vis = np.zeros((n, f), np.float32)
    conf = np.zeros((n, f), np.float32)
    start = np.full(n, -1, np.int64)
    prev_end = 0
    for s, e in offline_windows(f, window):
        fresh_ids = np.nonzero((q_t >= s) & (q_t < e))[0]
        carried_ids = np.nonzero((start >= 0) & (q_t < s))[0]
        if len(fresh_ids) + len(carried_ids) == 0:
            tick(e - max(s, prev_end))
            prev_end = e
            continue
        ov = max(0, prev_end - s)  # frames shared with the previous window (OFFLINE_OVERLAP)
        # 续上的点：上一段在重叠帧上算出来的位置就是这一段的查询点，查询帧即重叠的那一帧
        seed = xy[carried_ids, s] if len(carried_ids) else np.zeros((0, 2), np.float32)
        rows = [np.stack([q_t[fresh_ids] - s, q_xy[fresh_ids, 0], q_xy[fresh_ids, 1]], -1),
                np.concatenate([np.zeros((len(carried_ids), 1)), seed], -1)]
        if support is not None:
            rows.append(np.concatenate([np.zeros((len(support), 1)), support], -1))
        queries = torch.from_numpy(np.concatenate(rows).astype(np.float32))[None].to(device)
        coords, v, c, _ = model(video=to_tensor(video[s:e], device), queries=queries, iters=ITERS,
                                fmaps_chunk_size=FNET_CHUNK)
        coords = coords[0].permute(1, 0, 2).cpu().numpy()  # [M, T, 2]
        v = v[0].permute(1, 0).cpu().numpy()
        c = c[0].permute(1, 0).cpu().numpy()
        ids = np.concatenate([fresh_ids, carried_ids])
        m = len(ids)
        coords, v, c = coords[:m], v[:m], c[:m]
        # 重叠帧上保留上一段的答案（续点的查询点就来自它），这一段从它之后接着写，不做混合
        for arr, new in ((xy, coords), (vis, v), (conf, c)):
            arr[ids, s + ov:e] = new[:, ov:]
            fresh = start[ids] < 0
            if fresh.any():  # 这一段里第一次出现的点：重叠那一帧也归它
                arr[ids[fresh], s:s + ov] = new[fresh, :ov]
        start[fresh_ids] = np.where(start[fresh_ids] >= 0, start[fresh_ids], s)
        tick(e - max(s, prev_end))
        prev_end = e
    return xy, vis, conf, start


# ---------------------------------------------------------------------- online


@torch.no_grad()
def online_rollout(model, video: pt.Frames, q_t: np.ndarray, q_xy: np.ndarray, support: np.ndarray | None,
                   device, tick) -> tuple:
    """Upstream's online processing: windows of 16 frames, step 8, forwards from the
    earliest query. Returns xy, vis, conf [N, F] (values from each query's frame on)."""
    n, f = len(q_t), len(video)
    first = int(q_t.min())
    rows = [np.stack([q_t - first, q_xy[:, 0], q_xy[:, 1]], -1)]
    if support is not None:
        rows.append(np.concatenate([np.zeros((len(support), 1)), support], -1))
    queries = torch.from_numpy(np.concatenate(rows).astype(np.float32))[None].to(device)
    s_len = model.window_len
    step = s_len // 2
    model.init_video_online_processing()
    length = f - first
    ind = 0
    while True:
        clip = video[first + ind: first + ind + s_len]
        coords, v, c, _ = model(video=to_tensor(clip, device), queries=queries, iters=ITERS, is_online=True)
        tick(min(length, ind + s_len) - (ind + step if ind else 0))
        if ind + s_len >= length:
            break
        ind += step
    xy = np.full((n, f, 2), np.nan, np.float32)
    vis = np.zeros((n, f), np.float32)
    conf = np.zeros((n, f), np.float32)
    xy[:, first:] = coords[0, :, :n].permute(1, 0, 2).cpu().numpy()
    vis[:, first:] = v[0, :, :n].permute(1, 0).cpu().numpy()
    conf[:, first:] = c[0, :, :n].permute(1, 0).cpu().numpy()
    model.init_video_online_processing()  # the model stays loaded: without this shot's state
    before = np.arange(f)[None, :] < q_t[:, None]
    xy[before] = np.nan
    vis[before] = 0
    conf[before] = 0
    return xy, vis, conf


# ---------------------------------------------------------------------- both ways


def first_window_end(mode: str, n_frames: int, window: int) -> int:
    """Offline: queries at or after this frame index need the backward pass."""
    return offline_windows(n_frames, window)[0][1] if mode == "offline" else 1


def track(mode: str, model, video, q_t, q_xy, support, window, device, tick):
    """Forwards, then backwards over the reversed frames for what is still missing.
    Tracks are anchored to their query points (point_tracks.anchor)."""
    if mode == "offline":
        # the first window tracks both ways: only points whose query lies beyond it need the backward pass
        xy, vis, conf, start = offline_sweep(model, video, q_t, q_xy, support, window, device, tick)
        need = start > 0
    else:
        xy, vis, conf = online_rollout(model, video, q_t, q_xy, support, device, tick)
        need = q_t > 0
    offsets = pt.anchor(xy, q_t, q_xy)
    if need.any():
        last = int(q_t[need].max())
        rev = video.reversed_upto(last)
        rq = last - q_t[need]
        if mode == "offline":
            bxy, bvis, bconf, _ = offline_sweep(model, rev, rq, q_xy[need], support, window, device, tick)
        else:
            bxy, bvis, bconf = online_rollout(model, rev, rq, q_xy[need], support, device, tick)
        pt.anchor(bxy, rq, q_xy[need])
        before = np.arange(last + 1)[None, :] < q_t[need][:, None]
        for arr, back in ((xy, bxy), (vis, bvis), (conf, bconf)):
            sub = arr[need]
            sub[:, : last + 1][before] = back[:, ::-1][before]
            arr[need] = sub
    return xy, vis, conf, offsets


def offline_plan(n_frames: int, n_points: int, n_support: int, hw: tuple[int, int], budget: float) -> tuple[int, int]:
    """(window frames, points per pass) whose forward pass fits `budget` bytes."""
    pixels = hw[0] * hw[1]
    per_frame = OFFLINE_FRAME_BYTES_PER_PIXEL * pixels
    budget -= FNET_CHUNK * FNET_BYTES_PER_PIXEL * pixels
    per_pass = max(1, min(n_points, OFFLINE_MAX_POINTS))

    def frames_that_fit(points: int) -> int:
        return int(max(0.0, budget) / (per_frame + OFFLINE_POINT_BYTES * (points + n_support)))

    # fewer points per pass until at least OFFLINE_MIN_WINDOW frames fit
    while per_pass > 64 and frames_that_fit(per_pass) < OFFLINE_MIN_WINDOW:
        per_pass = (per_pass + 1) // 2
    window = max(OFFLINE_OVERLAP * 2, min(frames_that_fit(per_pass), OFFLINE_MAX_WINDOW, n_frames))
    forced = os.environ.get("LAB2SHOT_COTRACKER_WINDOW")  # testing / tuning: fixed window length
    if forced:
        window = max(OFFLINE_OVERLAP * 2, min(int(forced), n_frames))
    return window, per_pass


def main(job_path: str) -> None:
    run = Run.start(job_path, "cotracker.track", "CoTracker3")
    job = run.job
    frames = run.frames()
    frame_numbers = frames.numbers
    p = pt.parse_params(job.params, frame_numbers)
    checkpoint = job.weights_dir / CHECKPOINTS[p.mode]
    require_weights("cotracker", checkpoint)  # not run.weights: the extension is "cotracker", the project "CoTracker3"

    width, height = frames.width, frames.height
    hw = processing_size(width, height, p.resolution)
    queries = pt.build_queries(job, p, width, height)
    n_user = int(queries.user.sum())
    has_mask = "mask" in job.inputs
    # upstream adds the support grid whenever the points are not a full-frame grid
    support = support_queries(hw) if (n_user or has_mask) else None
    device = torch.device("cuda")

    run.stage("读取画面")
    t0 = time.time()
    video = pt.Frames(frames.paths, (hw[1], hw[0]))
    video.preload(lambda d, t: progress(d, t, "读取画面"))
    read_s = time.time() - t0

    model = run.model("CoTracker3 模型", load_model, job.repo_dir, checkpoint, p.mode, hw, device)

    n, f = len(queries), len(frames)
    if p.mode == "offline":
        # at most OFFLINE_BUDGET, less when the card is smaller or busy
        free, _ = torch.cuda.mem_get_info()
        budget = min(OFFLINE_BUDGET, 0.8 * free)
        window, per_pass = offline_plan(f, n, 0 if support is None else len(support), hw, budget)
    else:
        window, per_pass = WINDOW_LEN["online"], n
    passes = [slice(i, min(n, i + per_pass)) for i in range(0, n, per_pass)]
    # input pixels (centre at +0.5) -> model pixels (upstream: centre at integer)
    sx, sy = hw[1] / width, hw[0] / height
    q_model = np.stack([queries.xy[:, 0] * sx - 0.5, queries.xy[:, 1] * sy - 0.5], -1).astype(np.float32)

    total = 0  # frames stepped (progress only)
    for sl in passes:
        q = queries.t[sl]
        back = q >= first_window_end(p.mode, f, window)
        total += (f if p.mode == "offline" else f - int(q.min())) + (int(q[back].max()) + 1 if back.any() else 0)
    done = 0

    def tick(k: int) -> None:
        nonlocal done
        done = min(total, done + max(0, k))
        progress(done, total, "跟踪")

    run.stage(f"跟踪 {n} 个点（{p.mode}" + (f"，每段 {window} 帧" if p.mode == "offline" and window < f else "") + "）")
    t2 = time.time()
    xy = np.empty((n, f, 2), np.float32)
    vis = np.empty((n, f), np.float32)
    conf = np.empty((n, f), np.float32)
    offsets = np.empty((n, 2), np.float32)
    for sl in passes:
        xy[sl], vis[sl], conf[sl], offsets[sl] = track(p.mode, model, video, queries.t[sl], q_model[sl], support,
                                                       window, device, tick)
    track_s = time.time() - t2
    run.frame_seconds.extend([track_s / f] * f)  # the tracking is one pass over the shot: shared out per frame
    offset_px = np.linalg.norm(offsets / np.array([sx, sy], np.float32), axis=-1)  # input pixels

    tracks = np.stack([(xy[..., 0] + 0.5) / sx, (xy[..., 1] + 0.5) / sy], -1).astype(np.float32)
    confidence = (vis * conf).astype(np.float32)
    visible = (vis if p.mode == "offline" else confidence) > VISIBLE_THRESHOLD[p.mode]
    # tracks pass through the query points (anchored); a query point is visible (upstream predictor)
    idx = np.arange(n)
    tracks[idx, queries.t] = queries.xy  # removes float round-off only
    visible[idx, queries.t] = True
    if np.isnan(tracks).any():
        say("W-COTRACKER-NAN", count=int(np.isnan(tracks).any(-1).sum()))

    pt.write_result(
        run,
        tracks=tracks, visible=visible, confidence=confidence, queries=queries, frame_numbers=frame_numbers,
        width=width, height=height, p=p, user_points=n_user, query_offset_px=offset_px,
        read_seconds=read_s, track_seconds=track_s,
        model="CoTracker3",
        mode=p.mode,
        checkpoint=CHECKPOINTS[p.mode],
        visible_rule=("visibility > 0.9 (upstream offline predictor)" if p.mode == "offline"
                      else "visibility x confidence > 0.6 (upstream online predictor)"),
        confidence_rule="visibility probability x CoTracker3 confidence (probability the point is within 12 px at 512x384)",
        processing_resolution=[hw[1], hw[0]],
        processing_note=("squeezed to upstream's 512x384" if p.resolution is None else "aspect ratio kept")
        + ", area-averaged resize",
        offline_window=window if p.mode == "offline" else None,
        offline_windows=[list(w) for w in offline_windows(f, window)] if p.mode == "offline" else None,
        offline_overlap=OFFLINE_OVERLAP if p.mode == "offline" and window < f else None,
        points_per_pass=per_pass,
        passes=len(passes),
        support_grid=support is not None,
        direction="forwards from each query frame, backwards (reversed frames) before it where needed",
    )


if __name__ == "__main__":
    serve(main)
