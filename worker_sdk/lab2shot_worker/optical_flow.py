"""The optical-flow worker contract, shared by every optical-flow worker (MEMFOF, WAFT). numpy only; never imports
Lab2Shot core or a model.

job["frames"]  the shot's frames in order (sRGB PNGs); the flow goes between neighbours in this list
job["params"]  (as the node defines them)
    resolution   long side, pixels, the model runs at; None = the plate's own size

raw/frame_<n>.npz, one per frame, on a grid of the processing size h x w (the node resamples it to the plate):
    forward              float32 [h, w, 2]  where each pixel of frame n is on the next frame, minus where it is:
                                            +x right, +y down, in pixels of the processing size. Absent on the last
                                            frame.
    backward             float32 [h, w, 2]  the same towards the previous frame. Absent on the first frame.
    forward_confidence   float16 [h, w]     0..1, how sure the model is of each vector (confidence()); absent when the
    backward_confidence                     model gives none
raw/result.json  {"model": ..., "size": [w, h], ...the Run's standard timing and memory fields}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import fail
from .files import fit_size, read_frame, save_npz
from .run import Run


class Shot:
    """What every optical-flow worker starts from: the job's frames (at least two, else fail()), the plate's size and
    the processing size (w, h); shot[i] is frame i at the processing size, uint8 [h, w, 3] RGB, read when it is first
    needed and kept until release(i) lets the frames before i go."""

    def __init__(self, job):
        self.frames = job.frames
        if len(self.frames) < 2:
            fail("E-FLOW-TWOFRAMES")
        self.width, self.height = job.width, job.height
        self.w, self.h = processing_size(job.width, job.height, job.params["resolution"])
        self._kept: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> np.ndarray:
        if i not in self._kept:
            img = read_frame(self.frames[i][1])
            if img.shape[:2] != (self.h, self.w):
                import cv2

                img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
            self._kept[i] = img
        return self._kept[i]

    def release(self, i: int) -> None:
        for k in [k for k in self._kept if k < i]:
            del self._kept[k]

    def done(self, run: Run, model: str, **info) -> None:
        """raw/result.json: the model, the processing size, and the Run's standard timing and GPU-memory fields
        (run.finish)."""
        run.finish([f for f, _ in self.frames], model=model, size=[self.w, self.h], **info)


def processing_size(width: int, height: int, resolution: int | None) -> tuple[int, int]:
    """(w, h) the model runs at: the plate's own size (the models pad it themselves), or its long side brought down
    to `resolution` (never up), both sides a multiple of 8."""
    if resolution is None or max(width, height) <= resolution:
        return width, height
    return fit_size(width, height, resolution, 8, upscale=False)


def confidence(info: np.ndarray, var_min: float, var_max: float) -> np.ndarray:
    """How sure a SEA-RAFT-style model (MEMFOF, WAFT) is of each vector, 0..1. `info` [4, h, w] is its head's output:
    the logits of a mixture of two Laplace distributions of the error and their raw log scales, the first clamped to
    [0, var_max], the second to [var_min, 0] (as in training). The confidence is 1 / b, b the mixture's geometric-mean
    scale in the model's own pixels, at most 1: 1 means the model expects to be within about a pixel, 0.1 within about
    ten. In the model's pixels, so it reads the same whatever the processing size."""
    info = np.asarray(info, np.float32)
    first = 1.0 / (1.0 + np.exp(np.clip(info[1] - info[0], -60.0, 60.0)))  # softmax of two logits: the first's weight
    log_b0, log_b1 = np.clip(info[2], 0.0, var_max), np.clip(info[3], var_min, 0.0)
    return np.clip(np.exp(-(log_b1 + first * (log_b0 - log_b1))), 0.0, 1.0)


def save_frame(raw: Path, frame: int, *, forward: np.ndarray | None = None, backward: np.ndarray | None = None,
               forward_confidence: np.ndarray | None = None, backward_confidence: np.ndarray | None = None) -> Path:
    """One frame's raw/frame_<n>.npz: [h, w, 2] vectors in float32, [h, w] confidences in float16 (plenty for 0..1);
    stored uncompressed (flow hardly compresses: zlib saves a few percent and costs a second per HD frame)."""
    arrays = {k: np.asarray(v, np.float32) for k, v in (("forward", forward), ("backward", backward)) if v is not None}
    arrays |= {k: np.asarray(v, np.float16) for k, v in (("forward_confidence", forward_confidence),
                                                         ("backward_confidence", backward_confidence)) if v is not None}
    return save_npz(Path(raw) / f"frame_{frame}.npz", **arrays)
