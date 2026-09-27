"""Sequence logic single-image methods lack: people-box tracking over a shot and zero-phase temporal smoothing
(the SAM 3D Body family and its ViTDet people detector node; the core's camera smoothing). Pure numpy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lab2shot_shared.smoothing import gaussian_smooth, smooth_angles, smooth_quats  # noqa: F401


# --------------------------------------------------------------------------- tracking


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of [N,4] and [M,4] xyxy boxes -> [N,M]."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass
class Track:
    boxes: dict[int, np.ndarray] = field(default_factory=dict)  # frame -> xyxy

    @property
    def last_frame(self) -> int:
        return max(self.boxes)

    @property
    def prominence(self) -> float:
        """Screen presence: summed box area over the shot."""
        return float(sum((b[2] - b[0]) * (b[3] - b[1]) for b in self.boxes.values()))


def track_boxes(
    detections: dict[int, np.ndarray],
    min_iou: float = 0.3,
    max_gap: int = 10,
) -> list[Track]:
    """Greedy IoU tracking. detections: frame -> [N,4]. Returns tracks, most prominent first."""
    tracks: list[Track] = []
    frames = sorted(detections)
    for frame in frames:
        boxes = np.asarray(detections[frame], dtype=np.float64).reshape(-1, 4)
        alive = [t for t in tracks if frame - t.last_frame <= max_gap]
        taken: set[int] = set()
        if alive and len(boxes):
            prev = np.stack([t.boxes[t.last_frame] for t in alive])
            scores = iou(prev, boxes)
            for flat in np.argsort(-scores, axis=None):
                ti, bi = np.unravel_index(flat, scores.shape)
                if scores[ti, bi] < min_iou:
                    break
                if bi in taken or frame in alive[ti].boxes:
                    continue
                alive[ti].boxes[frame] = boxes[bi]
                taken.add(int(bi))
        for bi, box in enumerate(boxes):
            if bi not in taken:
                tracks.append(Track({frame: box}))
    return sorted(tracks, key=lambda t: -t.prominence)


# --------------------------------------------------------------------------- smoothing


def segments(frames: list[int], max_gap: int = 1) -> list[list[int]]:
    """Split sorted frame numbers into runs; smoothing never crosses a gap."""
    runs: list[list[int]] = []
    for f in frames:
        if runs and f - runs[-1][-1] <= max_gap:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def smoothing_sigma(strength: float) -> float:
    """UI strength 0..1 -> Gaussian sigma in frames (0.5 -> 1.5 frames)."""
    return 3.0 * float(strength)
