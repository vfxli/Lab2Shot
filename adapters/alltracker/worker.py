"""AllTracker worker: dense long-range correspondence to a reference frame. Runs inside third_party/alltracker/.venv
with the pinned repo on the path; never imports Lab2Shot core.

    python worker.py <job.json>

AllTracker estimates the flow from one query frame to every other frame of the shot, for every pixel of the query
frame (upstream forward_sliding: windows of 16 frames, stride 8; the frames before the query frame are tracked
backwards over the reversed shot, as upstream's demo does). The worker writes two contracts:

  lab2shot_worker/correspondence.py  per frame t, for every pixel of t, where it is on the reference frame: the
                                     inverse of the model's reference -> t flow (below), and its confidence;
  lab2shot_worker/point_tracks.py    the grid points (on the reference frame) and clicked points followed through the
                                     shot (tracks.npz), sampled from the same flow.

Inverting the flow. The model says where each reference pixel p is on frame t: q = p + F(p). An ST-map needs the
opposite: for each pixel x of frame t, the p with p + F(p) = x. Each visible reference pixel is splatted to the
frame-t pixel it lands on (where several land on one, the most confident wins), pixels nobody lands on start from
x - F(x), and a few fixed-point steps p <- x - F(p) refine them, each pixel keeping whichever step lands closest to x.
A pixel is trusted where the refined p lands within half a processing pixel of x, weighted by the model's
visibility x confidence at p (upstream's score, visible at >= 0.6).

Resolution. Upstream's paper runs at up to 768 x 1024 on a 40 GB card. The worker runs at `resolution` (long side,
a multiple of 8) and the node brings the results to the plate's size (positions are resampled, not re-estimated:
fine detail smaller than a processing pixel is interpolated).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import lab2shot_worker.point_tracks as pt
from lab2shot_worker import fail, fit_size, progress, reason, resident, serve, stage
from lab2shot_worker import correspondence as corr
from lab2shot_worker.run import Run

CHECKPOINT = "alltracker/alltracker.pth"
WINDOW = 16  # upstream Net(seqlen=16)
ITERS = 4  # upstream demo's inference_iters
VISIBLE = 0.6  # upstream evaluation: visible where visibility x confidence >= 0.6
REFINE_STEPS = 8
TRUSTED_PX = 0.5  # the inverse lands within this many processing pixels of the pixel


@resident
def load_model(checkpoint: Path, device):
    from nets.alltracker import Net

    model = Net(WINDOW, init_weights=False)  # no ImageNet ConvNeXt download: the checkpoint holds every weight
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval().to(device)


def sample(field: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """field [C, h, w] at continuous pixel positions xy [..., 2] (pixel centres at +0.5), bilinear, border-clamped ->
    [..., C]."""
    c, h, w = field.shape
    grid = torch.stack([xy[..., 0] / w * 2 - 1, xy[..., 1] / h * 2 - 1], -1).reshape(1, 1, -1, 2)
    out = F.grid_sample(field[None], grid, mode="bilinear", padding_mode="border", align_corners=False)
    return out[0, :, 0].T.reshape(*xy.shape[:-1], c)


def invert(flow: torch.Tensor, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """flow [2, h, w] (reference pixel -> its move to frame t), score [h, w] (visibility x confidence) -> for every pixel
    of frame t its position on the reference frame [h, w, 2] and how much to trust it [h, w]."""
    _, h, w = flow.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    grid = torch.stack([xs, ys], -1).float() + 0.5  # pixel centres
    moved = flow.permute(1, 2, 0)
    target = grid + moved
    ix, iy = torch.floor(target[..., 0]).long(), torch.floor(target[..., 1]).long()
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h) & (score >= VISIBLE)
    cell = (iy * w + ix)[ok]
    best = torch.full((h * w,), -1.0, device=flow.device).scatter_reduce(0, cell, score[ok], "amax")
    wins = score[ok] >= best[cell]
    start = grid - sample(flow, grid)  # where nothing lands: one step from x - F(x)
    start = start.reshape(-1, 2)
    start[cell[wins]] = grid[ok][wins]
    p = start.reshape(h, w, 2)

    def miss(p):
        return torch.linalg.norm(p + sample(flow, p) - grid, dim=-1)

    err = miss(p)
    for _ in range(REFINE_STEPS):
        step = grid - sample(flow, p)
        step_err = miss(step)
        better = step_err < err
        p = torch.where(better[..., None], step, p)
        err = torch.where(better, step_err, err)
    inside = (p[..., 0] >= 0) & (p[..., 0] <= w) & (p[..., 1] >= 0) & (p[..., 1] <= h)  # not outside the reference frame
    trust = sample(score[None], p)[..., 0] * (err <= TRUSTED_PX) * inside
    return p, trust


def main(job_path: str) -> None:
    run = Run.start(job_path, "alltracker.track", "AllTracker")
    job = run.job
    if len(job.frames) < 2:
        fail("E-WORKER-TOOFEWFRAMES", least=2, have=len(job.frames), why=reason("I-WORKER-WHYCOMPARE"))
    frames = run.frames()
    numbers = frames.numbers
    p = pt.parse_params(job.params, numbers)
    checkpoint = job.weights_dir / CHECKPOINT
    run.weights(checkpoint)
    width, height = job.width, job.height
    w, h = fit_size(width, height, job.params["resolution"], 8, upscale=False)
    # the points to sample from the flow: none asked for (no grid, nothing clicked) -> the ST-maps only
    queries = pt.build_queries(job, p, width, height, allow_empty=True) if p.grid or "points" in job.inputs else None
    if queries is not None and not len(queries):  # nothing to sample (no grid, nothing clicked): the ST-maps only
        queries = None
    device = torch.device("cuda")

    run.stage("读取画面")
    video = pt.Frames(frames.paths, (w, h))
    video.preload(lambda d, t: progress(d, t, "读取画面"))
    model = run.model("AllTracker 模型", load_model, checkpoint, device)

    ref, n = p.query_index, len(frames)
    run.stage(f"稠密跟踪（{w}×{h}，参考帧 {numbers[ref]}）")
    t0 = time.time()

    def sweep(order: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow and visibility x confidence from order[0] (the reference) to each frame of `order`, on the CPU (the
        frames stay on the CPU too: upstream moves them to the GPU a chunk at a time)."""
        itself = (torch.zeros(1, 2, h, w), torch.ones(1, h, w))  # the reference is where it is
        if len(order) == 1:
            return itself
        clip = torch.from_numpy(np.stack([video[i] for i in order])).permute(0, 3, 1, 2)[None].float()  # 0..255
        with torch.inference_mode():
            flows, visconf, _, _ = model.forward_sliding(clip, iters=ITERS, window_len=WINDOW, is_training=False)
        if len(order) == 2:  # upstream's two-frame path gives the one flow, without a time axis
            flows, visconf = flows[:, None], visconf[:, None]
            return torch.cat([itself[0], flows[0]]), torch.cat([itself[1], visconf[0, :, 0] * visconf[0, :, 1]])
        return torch.cat([itself[0], flows[0, 1:]]), torch.cat([itself[1], visconf[0, 1:, 0] * visconf[0, 1:, 1]])

    after, score_after = sweep(list(range(ref, n)))
    progress(n - ref, n, "稠密跟踪")
    before, score_before = sweep(list(range(ref, -1, -1)))  # the frames before the reference: over the reversed shot
    flows = torch.cat([before.flip(0)[:-1], after])  # [n, 2, h, w]: reference -> each frame
    scores = torch.cat([score_before.flip(0)[:-1], score_after])
    track_s = time.time() - t0
    run.frame_seconds.extend([track_s / n] * n)  # the tracking is one pass over the shot: shared out per frame

    run.stage("反求 ST-map")
    to_plate = torch.tensor([width / w, height / h], device=device)
    clicked = {int(t) for t in queries.t if t != ref} if queries is not None else set()  # keep their frames' inverse
    inverse = {}
    for i, (f, _) in run.each(frames.pairs, "反求 ST-map"):
        pos, trust = invert(flows[i].to(device), scores[i].to(device))
        corr.save_frame(job.raw_dir, f, (pos * to_plate).cpu().numpy(), trust.cpu().numpy())
        if i in clicked:
            inverse[i] = pos
    stats = {} if queries is None else sample_tracks(job, queries, flows, scores, inverse, ref, to_plate, numbers)
    run.finish(numbers, target={"width": width, "height": height}, reference=numbers[ref], size=[w, h], **stats)


def sample_tracks(job, queries: pt.Queries, flows: torch.Tensor, scores: torch.Tensor, inverse: dict, ref: int,
                  to_plate: torch.Tensor, numbers: list[int]) -> dict:
    """The grid and clicked points followed through the shot, sampled from the reference -> frame flow (a point
    clicked on another frame is first taken back to the reference by that frame's inverse) -> raw/tracks.npz."""
    device, n = to_plate.device, len(numbers)
    width, height = job.width, job.height
    stage("取样 2D 跟踪点")
    q = torch.from_numpy(queries.xy).to(device) / to_plate  # processing pixels
    on_ref = q.clone()
    for k in np.nonzero(queries.t != ref)[0]:  # a point clicked on another frame: back to the reference first
        on_ref[k] = sample(inverse[int(queries.t[k])].permute(2, 0, 1), q[k][None])[0]
    tracks = np.empty((len(q), n, 2), np.float32)
    confidence = np.empty((len(q), n), np.float32)
    for i in range(n):
        flow, score = flows[i].to(device), scores[i].to(device)
        tracks[:, i] = ((on_ref + sample(flow, on_ref)) * to_plate).cpu().numpy()
        confidence[:, i] = sample(score[None], on_ref)[:, 0].cpu().numpy()
    pt.anchor(tracks, queries.t, queries.xy)
    visible = confidence >= VISIBLE
    visible[np.arange(len(q)), queries.t] = True  # a query point is visible where it was placed
    pt.save_tracks(job.raw_dir, tracks, visible, confidence, queries, numbers)
    return pt.track_stats(tracks, visible, width, height)


if __name__ == "__main__":
    serve(main)
