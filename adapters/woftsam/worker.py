"""WOFTSAM worker: long-term planar tracking. Runs inside third_party/woftsam/.venv with the pinned WOFTSAM repo
(repo/src on PYTHONPATH, the repo as working directory: its configs load each other by relative path) and the
authors' SAM 2 fork; never imports Lab2Shot core.

    python worker.py <job.json>

job["params"]
    corners   [[x, y] x 4] the plane's corners on the start frame, image pixels (pixel centres at +0.5), in order
              around the quad
    frame     the start frame (frame number)
    resolution  processing size, long side in pixels; null: the plate's (each side rounded to a multiple of 8, as
              RAFT needs)

raw/tracks.npz (the point-track contract, lab2shot_worker.point_tracks, for the four corners)
    tracks        float32 [4, F, 2]  the corners on every frame, input pixels, centres at +0.5
    visible       bool    [4, F]     the corner is in the picture and the plane was found on that frame
    confidence    float32 [4, F]     1 found by the flow, 0.5 found again by SAM-H, 0 not confirmed
    query_frames  int32   [4]        the start frame
    frames        int32   [F]
    user          bool    [4]        all True (the corners were given)
    homography    float64 [F, 3, 3]  the plane from the start frame's pixels to each frame's (centres at +0.5)
    state         int8    [F]        0 start, 1 followed by the flow (WOFT), 2 lost by the flow and found again by
                                     SAM-H, 3 neither confirmed it (SAM-H's guess, or held from the last frame)

WOFTSAM tracks forwards from the start frame; frames before it are tracked over the reversed shot. Its own loop
(woft_wrapper.woftsam_track) is copied here to read out, per frame, which of its three steps found the plane
(upstream throws that away): the flow from the template (pre-warped by the last good pose), the flow pre-warped by
SAM-H's pose, or SAM-H's pose as it is.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from lab2shot_worker import fail, fit_size, local_hub, progress, quiet, read_frame, resident, serve
from lab2shot_worker.run import Run

SAM2_CHECKPOINT = "sam2.1-hiera-tiny/sam2.1_hiera_tiny.pt"  # as the installer stores it (lab2shot_worker.hf_dest)
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"  # inside the sam2 package (hydra)
START, FLOW, REFOUND, UNSURE = 0, 1, 2, 3
CONFIDENCE = {START: 1.0, FLOW: 1.0, REFOUND: 0.5, UNSURE: 0.0}


@contextmanager
def in_folder(folder: Path):
    """WOFTSAM's configs load each other by paths relative to the working directory (its repo)."""
    before = os.getcwd()
    os.chdir(folder)
    try:
        yield
    finally:
        os.chdir(before)


@resident
def load_models(repo: Path, sam2_dir: Path, sam2_checkpoint: Path, dinov2: Path):
    """WOFTSAM's configuration (with SAM-H's DINOv2 extractor), the SAM 2.1 tiny video predictor and the WOFTSAM
    tracker (its weighted RAFT)."""
    if str(sam2_dir) not in sys.path:
        sys.path.insert(0, str(sam2_dir))
    local_hub({"dinov2": dinov2}, "WOFTSAM")
    with in_folder(repo):
        from flatsam.config import load_config
        from flatsam.woftsam import WOFTSAM
        from sam2.build_sam import build_sam2_video_predictor

        conf = load_config("configs/woftsam.py")
        # upstream's get_predictor, with the installed checkpoint instead of ./checkpoints/
        predictor = build_sam2_video_predictor(SAM2_CONFIG, str(sam2_checkpoint), hydra_overrides_extra=[
            f"++model.memory_temporal_stride_for_eval={conf.sam.memory_stride}",
            "++model.do_not_update_when_not_present=True"])
        tracker = WOFTSAM(conf)
    return conf, predictor, tracker


def to_woft(h: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """A homography between input pixels (centres at +0.5) as one between processing pixels, centres at 0
    (WOFTSAM's, OpenCV's): x_woft = x * s - 0.5."""
    a = np.array([[sx, 0, -0.5], [0, sy, -0.5], [0, 0, 1.0]])
    return a @ h @ np.linalg.inv(a)


def from_woft(h: np.ndarray, sx: float, sy: float) -> np.ndarray:
    a = np.array([[sx, 0, -0.5], [0, sy, -0.5], [0, 0, 1.0]])
    return np.linalg.inv(a) @ h @ a


def warp(h: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.c_[xy, np.ones(len(xy))] @ h.T
    return p[:, :2] / p[:, 2:]


def track_one_way(conf, predictor, tracker, frames: list[np.ndarray], init: np.ndarray, tick):
    """WOFTSAM over `frames` (BGR, processing size) from the first, its plane given by `init` (2 x 4, processing
    pixels, centres at 0). Yields (index, H_init_to_frame, state) per frame (upstream's woftsam_track, reading
    out which step found the plane)."""
    import flatsam.utils.vis as vu
    from flatsam.flatsam import flatsam_track

    init_mask = vu.draw_mask(init, frames[0].shape[:2]).astype(np.uint8)
    found: list[bool] = []  # this frame's pose checks: the flow's own, then the one pre-warped by SAM-H's pose
    estimate = tracker.estimate_H_with_prewarp

    def checked(prewarp_h, img):
        h, ok = estimate(prewarp_h, img)
        found.append(bool(ok))
        return h, ok

    tracker.estimate_H_with_prewarp = checked
    last = np.eye(3)
    try:
        for idx, _mask, _vis, info in flatsam_track(predictor, conf.flatsam, frames, init, "plane"):
            if idx == 0:
                tracker.init(frames[0].copy(), init_mask)
                h_cur2init, state = np.eye(3), START
            else:
                robust = None
                try:
                    robust = np.linalg.inv(info["output_H"]) if "output_H" in info else info["output_H2init"]
                except (np.linalg.LinAlgError, KeyError, TypeError):
                    pass
                found.clear()
                try:
                    h_cur2init, _meta = tracker.track(frames[idx].copy(), robust_H2init=robust)
                    state = FLOW if found[:1] == [True] else REFOUND if found[1:2] == [True] else UNSURE
                except Exception:  # upstream's loop keeps the last pose too
                    h_cur2init, state = last, UNSURE
            # `last` is the last pose that inverted: set before the inversion, `inv(last)` below would invert the
            # same singular matrix again and fail the job. Upstream (demo.py:75-77) also keeps the previous pose
            try:
                h = np.linalg.inv(h_cur2init)
                last = h_cur2init   # only a pose that inverted becomes the fallback
            except np.linalg.LinAlgError:
                h, state = np.linalg.inv(last), UNSURE
            tick()
            yield idx, h, state
    finally:
        del tracker.estimate_H_with_prewarp  # the resident tracker keeps its own method


def main(job_path: str) -> None:
    run = Run.start(job_path, "woftsam.track", "WOFTSAM")
    job = run.job
    params = job.params
    frames = run.frames()
    numbers = frames.numbers
    if params["frame"] not in numbers:
        fail("E-WOFTSAM-FRAMEOUT", frame=params["frame"], first=numbers[0], last=numbers[-1])
    start = numbers.index(params["frame"])
    corners = np.asarray(params["corners"], np.float64).reshape(4, 2)
    sam2_dir = Path(os.environ["LAB2SHOT_SAM2_DIR"])
    dinov2 = Path(os.environ["LAB2SHOT_DINOV2_DIR"])
    checkpoint = job.weights_dir / SAM2_CHECKPOINT
    run.weights(checkpoint, dinov2, job.weights_dir / "torch" / "hub" / "checkpoints")

    width, height = job.width, job.height
    long_side = params["resolution"] or max(width, height)
    w, h = fit_size(width, height, min(long_side, max(width, height)), 8, minimum=64)
    sx, sy = w / width, h / height

    run.stage("读取画面")
    bgr = []
    for i, (_, path) in run.each(frames.pairs, "读取画面"):
        img = read_frame(path)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LINEAR)
        bgr.append(np.ascontiguousarray(img[..., ::-1]))

    conf, predictor, tracker = run.model("WOFTSAM（加权 RAFT、SAM 2.1、DINOv2）", load_models, job.repo_dir, sam2_dir,
                                         checkpoint, dinov2)

    init = (corners * [sx, sy] - 0.5).T  # 2 x 4, processing pixels, centres at 0
    n = len(frames)
    homography = np.tile(np.eye(3), (n, 1, 1))
    state = np.zeros(n, np.int8)  # the start frame: START
    # forwards from the start frame, then backwards over the reversed shot (upstream's loop needs two frames)
    passes = [order for order in (list(range(start, n)), list(range(start, -1, -1))) if len(order) > 1]
    total, done = sum(map(len, passes)), 0

    def tick():
        nonlocal done
        done += 1
        progress(done, total, "跟踪平面")

    run.stage("跟踪平面" + ("（先向后，再从起始帧倒着向前）" if len(passes) == 2 else ""))
    t2 = time.time()
    with quiet(), in_folder(job.repo_dir):
        for order in passes:
            for k, hh, st in track_one_way(conf, predictor, tracker, [bgr[i] for i in order], init, tick):
                if k:
                    homography[order[k]] = from_woft(hh, sx, sy)
                    state[order[k]] = st
    track_s = time.time() - t2
    run.frame_seconds.extend([track_s / max(1, n - 1)] * (n - 1))  # the start frame is given: n - 1 frames stepped

    tracks = np.stack([warp(homography[j], corners) for j in range(n)], 1).astype(np.float32)  # [4, F, 2]
    inside = (tracks[..., 0] >= 0) & (tracks[..., 0] <= width) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= height)
    visible = inside & (state != UNSURE)[None, :]
    confidence = np.tile(np.array([CONFIDENCE[int(s)] for s in state], np.float32), (4, 1))
    job.raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(job.raw_dir / "tracks.npz", tracks=tracks, visible=visible, confidence=confidence,
                        query_frames=np.full(4, params["frame"], np.int32), frames=np.asarray(numbers, np.int32),
                        user=np.ones(4, bool), homography=homography, state=state)
    counts = {name: int((state == s).sum()) for name, s in (("flow", FLOW), ("refound", REFOUND), ("unsure", UNSURE))}
    run.finish(
        numbers,
        kind="point_tracks",
        model="WOFTSAM",
        files="tracks.npz: tracks [4,F,2], visible, confidence [4,F], query_frames, frames, user, homography [F,3,3], state [F]",
        convention="input pixels, pixel centres at +0.5; homography from the start frame's pixels to each frame's",
        processing_size=[w, h],
        start_frame=params["frame"],
        frames_flow=counts["flow"],
        frames_refound=counts["refound"],
        frames_unsure=counts["unsure"],
        track_seconds=round(track_s, 1),
    )


if __name__ == "__main__":
    serve(main)
