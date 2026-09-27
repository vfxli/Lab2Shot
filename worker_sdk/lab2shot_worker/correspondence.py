"""The dense-correspondence worker contract, shared by the workers that find where every pixel of the plate is in
another picture: AllTracker (in the shot's reference frame), RoMa v2 (in another image). numpy only; never imports
Lab2Shot core or a model.

raw/frame_<n>.npz, one per plate frame, on a grid of any size h x w covering the plate (the node resamples it to the
plate, pixel centres aligned):
    xy          float32 [h, w, 2]  where the pixel is in the other picture, in that picture's pixels (its full size,
                                   raw/result.json "target"): +x right, +y down, pixel centres at +0.5 (the top-left
                                   pixel covers [0, 1] x [0, 1])
    confidence  float16 [h, w]     0..1: the pixel is seen in the other picture and xy is right
raw/result.json  {"target": {"width": W, "height": H}, ...}
raw/matches.npz  (optional) correspondences sampled for other programs (PnP, COLMAP):
    xy_a        float32 [N, 2]     on the plate, plate pixels (the same convention as xy)
    xy_b        float32 [N, 2]     in the other picture, its pixels
    frame_a     int64 [N]          the plate frame of each match
    frame_b     int64 [N]          the other picture's frame it is in
    confidence  float32 [N]        0..1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .files import save_npz


def save_frame(raw: Path, frame: int, xy: np.ndarray, confidence: np.ndarray) -> Path:
    """One plate frame's raw/frame_<n>.npz (uncompressed: positions hardly compress)."""
    return save_npz(Path(raw) / f"frame_{frame}.npz", xy=np.asarray(xy, np.float32),
                    confidence=np.clip(np.asarray(confidence, np.float32), 0.0, 1.0).astype(np.float16))


def save_matches(raw: Path, xy_a: np.ndarray, xy_b: np.ndarray, frame_a: np.ndarray, frame_b: np.ndarray,
                 confidence: np.ndarray) -> Path:
    return save_npz(Path(raw) / "matches.npz", compression=1, xy_a=np.asarray(xy_a, np.float32).reshape(-1, 2),
                    xy_b=np.asarray(xy_b, np.float32).reshape(-1, 2), frame_a=np.asarray(frame_a, np.int64),
                    frame_b=np.asarray(frame_b, np.int64), confidence=np.asarray(confidence, np.float32))
