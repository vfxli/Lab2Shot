"""The mask-guided video matting worker contract, shared by every such worker
(MatAnyone 2, VideoMaMa): one job contract, so the node code is shared. Needs
numpy and OpenCV; never imports Lab2Shot core.

Job: job["node"] = "<extension>.matte"; job["frames"] = [{frame, path}] sRGB PNGs;
job["inputs"]["mask"] = JSON {"frames": {"<frame>": path}} of coarse 0..1 guide
masks (EXR: channel "A" or the only channel; PNG/JPG: grey 0..255);
job["params"] = {resolution, warmup, erode_dilate, fp16}, each model validates its own
ranges and defaults.

Output: raw/frame_<n>.npz with alpha float32 [H,W] in 0..1 at the input frame's
resolution (+ foreground float32 [H,W,3] sRGB 0..1 when a model predicts one),
raw/result.json.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import Job, fail, nothing, read_frame, read_mask, say
from .frame_io import FrameReader, Writer
from .run import Run

GUIDE_THRESHOLD = 0.5  # a guide pixel is foreground above this
OUTPUT_FILES = "frame_<n>.npz: alpha float32 [H,W] in 0..1 at the input resolution"
CONVENTIONS = {
    "alpha": "straight (unpremultiplied) coverage 0..1, 1 = subject; same pixel grid as the input frame",
    "resize": "processed at a smaller size when the frame's long side exceeds resolution; alpha is resized "
              "back bilinearly (pixel-centre aligned)",
}


# --------------------------------------------------------------------------- frames


def frame_reader(paths: list[Path], size: tuple[int, int]) -> FrameReader:
    """The shot's frames, a few decoded ahead (get(index, the indices still to come in processing order)); every frame
    must be `size` (height, width)."""

    def read(path: Path) -> np.ndarray:
        rgb = read_frame(path)
        if rgb.shape[:2] != size:
            fail("E-MATTE-FRAMESIZE", file=path.name)
        return rgb

    return FrameReader(paths, read, threads=3, ahead=6)


# --------------------------------------------------------------------------- guide masks


class GuideMasks:
    """The coarse guide masks of job["inputs"]["mask"], per frame number, at the frame size."""

    def __init__(self, job: Job, height: int, width: int):
        if "mask" not in job.inputs:
            fail("E-MATTE-NOMASK")
        self.files = job.listing("mask")
        self.height, self.width = height, width
        self.resized = False
        wanted = {f for f, _ in job.frames}
        if not wanted & set(self.files):
            fail("E-MATTE-NOOVERLAP")

    def __contains__(self, frame: int) -> bool:
        return frame in self.files

    def get(self, frame: int) -> np.ndarray | None:
        """float32 [H,W] 0..1 at the frame size, or None when the frame has no mask."""
        path = self.files.get(frame)
        if path is None:
            return None
        mask = np.nan_to_num(read_mask(path), nan=0.0, posinf=1.0, neginf=0.0)
        if mask.shape != (self.height, self.width):
            if not self.resized:
                say("W-MATTE-MASKSIZE", mask_width=mask.shape[1], mask_height=mask.shape[0], width=self.width, height=self.height)
                self.resized = True
            mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    def missing(self, frames: list[int]) -> list[int]:
        """The frames with no guide mask at all, said once (W-MATTE-NOGUIDE). A model guided on every frame mattes
        those from an empty mask, so they come out empty: the shot still finishes, the node says what was missing
        (an empty result is not an error). Said here once for every such worker, not in each adapter."""
        gone = [f for f in frames if f not in self.files]
        if gone:
            say("W-MATTE-NOGUIDE", count=len(gone), first=gone[0])
        return gone

    def first_nonempty(self, frames: list[int]) -> tuple[int, np.ndarray]:
        """(index into `frames`, mask) of the first frame whose guide has foreground. A guide empty on every frame (the
        segmentation upstream found no one) ends the job with nothing(): an empty matte, not an error."""
        for i, f in enumerate(frames):
            if f in self.files:
                mask = self.get(f)
                if mask is not None and (mask > GUIDE_THRESHOLD).any():
                    return i, mask
        nothing("N-MATTE-EMPTYGUIDE")


def morph(binary: np.ndarray, pixels: int) -> np.ndarray:
    """uint8 0/1 mask grown (pixels > 0) or shrunk (< 0) by an elliptical kernel of |pixels| diameter."""
    if pixels == 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(pixels), abs(pixels)))
    return (cv2.dilate if pixels > 0 else cv2.erode)(binary, kernel, iterations=1)


# --------------------------------------------------------------------------- output


class Output:
    """Writes raw/frame_<n>.npz on a thread and keeps the numbers for result.json."""

    def __init__(self, job: Job):
        self.raw = job.raw_dir
        self.raw.mkdir(parents=True, exist_ok=True)
        self.writer = Writer(threads=2, max_pending=16)  # bounded: a slow disk must not pile frames up in RAM
        self.coverage: dict[int, float] = {}
        self.empty: list[int] = []

    def put(self, frame: int, alpha: np.ndarray, foreground: np.ndarray | None = None) -> None:
        alpha = np.ascontiguousarray(np.clip(alpha, 0.0, 1.0), dtype=np.float32)
        arrays = {"alpha": alpha}
        if foreground is not None:
            arrays["foreground"] = np.ascontiguousarray(np.clip(foreground, 0.0, 1.0), dtype=np.float32)
        self.coverage[frame] = float((alpha > 0.5).mean())
        if alpha.max() < 0.5:
            self.empty.append(frame)
        self.writer.npz(self.raw / f"frame_{frame}.npz", **arrays)

    def close(self) -> None:
        self.writer.close()

    def finish(self, run: Run, frames: list[int], **info) -> None:
        """result.json for a matte job (kind "matte"): the family's words, the coverage numbers, the worker's own fields,
        and the Run's standard timing and memory fields (frames = [first, last] and count among them)."""
        self.close()
        missing = [f for f in frames if f not in self.coverage]
        if missing:
            fail("E-MATTE-MISSING", count=len(missing), frames=missing[:5])
        if self.empty:
            say("W-MATTE-EMPTYALPHA", count=len(self.empty), frames=sorted(self.empty)[:10])
        cov = list(self.coverage.values())
        run.finish(
            frames,
            kind="matte",
            files=OUTPUT_FILES + (info.pop("extra_files", "")),
            conventions=CONVENTIONS,
            coverage={"min": round(min(cov), 4), "max": round(max(cov), 4), "mean": round(float(np.mean(cov)), 4)},
            empty_frames=sorted(self.empty),
            **info,
        )


def mb(n_bytes: float) -> int:
    return int(math.ceil(n_bytes / 2**20))
