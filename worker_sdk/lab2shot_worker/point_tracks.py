"""The point-tracking worker contract, shared by every point-tracking worker
(TAPNext++, CoTracker3, WOFTSAM's plane corners; in 3D, TAPIP3D and Track4World:
see the end). Needs numpy and OpenCV (masks and depth: OpenEXR or OpenImageIO);
never imports Lab2Shot core or a model.

job["node"]   "tapnext.track" | "cotracker.track" | "alltracker.track" (its sampled points) | a 3D tracker
job["frames"] [{frame, path}] sRGB PNGs of one shot, in frame order
job["params"]   (as the node defines them)
    grid         int: points per side of a regular grid
                 of query points on the query frame (20 -> 400 points; 0 = no grid).
                 The n x n grid cells tile the frame (or, with a mask, the mask's
                 bounding box) and each point sits at a cell centre.
    query_frame  frame number where the grid points start; null = first frame
    mode         how it runs (CoTracker3: whole shot / sliding); absent for a tracker with one way
    model        which checkpoint (TAPNext++: 512 / 256); absent for a tracker with one
    resolution   processing resolution, long side in pixels; null = the model's native
job["inputs"] (optional)
    points       JSON {"points": [{"frame": int, "x": float, "y": float}, ...]}:
                 user-chosen query points, input pixels (tracked as well as the grid,
                 and listed first in the output, in this order)
    mask         JSON {"frames": {"<frame>": exr path}}: grid points only where the
                 mask > 0.5 on the query frame (to track one surface or object)

raw/tracks.npz
    tracks        float32 [N, F, 2]  pixel x, y at the input resolution; pixel
                                     centres at +0.5 (the top-left pixel covers
                                     [0, 1] x [0, 1]); +X right, +Y down
    visible       bool    [N, F]     the point is visible (not occluded / off-frame)
    confidence    float32 [N, F]     0..1, the model's own score (see result.json)
    query_frames  int32   [N]        frame number each point starts at
    frames        int32   [F]        frame numbers
    user          bool    [N]        point came from the "points" input (else grid)
Positions are given for every frame, also before the query frame (tracked
backwards) and while the point is hidden (the model's guess; check `visible`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import fail, nothing, read_frame, read_mask, save_npz, say
from .frame_io import FrameReader
from .run import Run

MASK_THRESHOLD = 0.5
CONVENTION = (
    "image pixels at the input resolution: +X right, +Y down, origin at the top-left corner "
    "of the top-left pixel, pixel centres at +0.5"
)


# ---------------------------------------------------------------------- params


@dataclass
class TrackParams:
    grid: int
    query_index: int  # index into the job's frames
    # 「方式」: how this tracker runs (CoTracker3 whole shot / sliding windows). None for a tracker with a single mode
    mode: str | None
    resolution: int | None  # 「处理分辨率」: the size the picture is scaled to before tracking; None = the model's own
    # 「模型」: which weights to use (TAPNext++ 512 / 256). This is separate from mode, hence two fields rather than one
    # shared name (a name refers to one thing only). None for a tracker with a single set of weights
    model: str | None = None


def parse_params(params: dict, frame_numbers: list[int]) -> TrackParams:
    query_frame = params["query_frame"]
    if query_frame is not None and query_frame not in frame_numbers:
        fail("E-TRACKS-QUERYFRAME", frame=query_frame, first=frame_numbers[0], last=frame_numbers[-1])
    query_index = 0 if query_frame is None else frame_numbers.index(query_frame)
    return TrackParams(grid=params["grid"], query_index=query_index, mode=params.get("mode"),
                       resolution=params["resolution"], model=params.get("model"))


# ---------------------------------------------------------------------- queries


@dataclass
class Queries:
    xy: np.ndarray  # float32 [N, 2], input pixels, pixel centres at +0.5
    t: np.ndarray  # int64 [N], index into the job's frames
    user: np.ndarray  # bool [N]

    def __len__(self) -> int:
        return len(self.t)


def grid_points(n: int, box: tuple[float, float, float, float]) -> np.ndarray:
    """n x n cell centres of the box (x0, y0, x1, y1) -> float32 [n*n, 2], row-major."""
    x0, y0, x1, y1 = box
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    ys = y0 + (np.arange(n) + 0.5) * (y1 - y0) / n
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def load_mask(job, frame: int, width: int, height: int) -> np.ndarray:
    files = job.listing("mask")
    if frame not in files:
        fail("E-TRACKS-MASKFRAME", frame=frame, count=len(files))
    alpha = read_mask(files[frame])
    if alpha.shape != (height, width):
        alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    return alpha > MASK_THRESHOLD


def build_queries(job, p: TrackParams, width: int, height: int, allow_empty: bool = False) -> Queries:
    """User points first (in the file's order), then the grid on the query frame. A mask with nothing in it on the query
    frame places no grid point (said once); no point left at all is nothing to track: nothing(). A node whose tracks
    are only a by-product (AllTracker's ST-maps) passes `allow_empty` and gets no queries instead."""
    inputs = job.inputs
    frame_numbers = [f for f, _ in job.frames]
    xy, t, user = [], [], []
    if "points" in inputs:
        spec = json.loads(inputs["points"].read_text(encoding="utf-8"))
        for i, pt in enumerate(spec.get("points", [])):
            try:
                frame, x, y = int(pt["frame"]), float(pt["x"]), float(pt["y"])
            except (KeyError, TypeError, ValueError):
                fail("E-TRACKS-POINTFORMAT", index=i + 1, point=repr(pt))
            if frame not in frame_numbers:
                fail("E-TRACKS-POINTFRAME", index=i + 1, frame=frame, first=frame_numbers[0], last=frame_numbers[-1])
            if not (0 <= x <= width and 0 <= y <= height):
                fail("E-TRACKS-POINTOUTSIDE", index=i + 1, x=x, y=y, width=width, height=height)
            xy.append((x, y))
            t.append(frame_numbers.index(frame))
            user.append(True)
    if p.grid > 0:
        box = (0.0, 0.0, float(width), float(height))
        mask = None
        if "mask" in inputs:
            mask = load_mask(job, frame_numbers[p.query_index], width, height)
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                say("N-TRACKS-EMPTYMASK", frame=frame_numbers[p.query_index])
            else:  # the grid spans the object, so a small object still gets grid x grid cells
                box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
        pts = grid_points(p.grid, box) if mask is None or mask.any() else np.zeros((0, 2), np.float32)
        if mask is not None and len(pts):
            c = np.clip(pts[:, 0].astype(int), 0, width - 1)
            r = np.clip(pts[:, 1].astype(int), 0, height - 1)
            pts = pts[mask[r, c]]
        xy.extend(map(tuple, pts))
        t.extend([p.query_index] * len(pts))
        user.extend([False] * len(pts))
    if not t and not allow_empty:  # nothing to track: empty tracks and a notice, not an error
        if p.grid == 0:  # neither a grid nor a point asked for
            nothing("N-TRACKS-NOQUERY")
        nothing("N-TRACKS-NOTHINGLEFT", frame=frame_numbers[p.query_index])
    return Queries(np.asarray(xy, np.float32).reshape(-1, 2), np.asarray(t, np.int64), np.asarray(user, bool))


# ---------------------------------------------------------------------- frames


def frame_size(path: Path) -> tuple[int, int]:
    img = read_frame(path)
    return img.shape[1], img.shape[0]


FRAME_CACHE_BYTES = 2 * 2**30  # resized frames kept in RAM; longer / larger shots are streamed from disk


class Frames:
    """The shot's frames resized to size = (w, h), uint8 [h, w, 3] RGB, read lazily.

    frames[i] -> one frame, frames[a:b] -> uint8 [n, h, w, 3]; reversed_upto(last)
    -> the frames last, last-1, ..., 0 as another Frames (same cache). Up to
    FRAME_CACHE_BYTES of resized frames stay in RAM (a whole normal shot);
    beyond that the oldest are dropped and read again when needed, and the next
    frames are decoded ahead in the background. Area-averaged downscale (no
    aliasing); cv2's pixel-centre convention matches the output's (a coordinate
    scales by w / W).
    """

    def __init__(self, paths: list[Path], size: tuple[int, int], order=None, reader: FrameReader | None = None):
        self.paths = paths
        self.size = size
        self.order = list(range(len(paths))) if order is None else list(order)
        if reader is None:
            w, h = size
            capacity = max(32, FRAME_CACHE_BYTES // (w * h * 3))
            reader = FrameReader(paths, self._resized, threads=min(8, os.cpu_count() or 4), ahead=4, keep=capacity,
                                 budget_bytes=FRAME_CACHE_BYTES)
        self.reader = reader  # shared by reversed_upto's views: one cache

    def __len__(self) -> int:
        return len(self.order)

    def _resized(self, path: Path) -> np.ndarray:
        w, h = self.size
        img = read_frame(path)
        if img.shape[:2] == (h, w):
            return img
        interp = cv2.INTER_AREA if img.shape[1] > w or img.shape[0] > h else cv2.INTER_LINEAR
        return cv2.resize(img, (w, h), interpolation=interp)

    def __getitem__(self, key):
        if isinstance(key, slice):
            idx = self.order[key]
            return np.stack(self.reader.take(idx)) if idx else np.zeros((0, self.size[1], self.size[0], 3), np.uint8)
        pos = key if key >= 0 else len(self.order) + key
        return self.reader.get(self.order[pos], self.order[pos + 1:])  # the next frames decode in the background

    def reversed_upto(self, last: int) -> "Frames":
        return Frames(self.paths, self.size, self.order[: last + 1][::-1], self.reader)

    def preload(self, on_progress=None) -> None:
        """Decode the first frames (as many as the cache holds) in parallel."""
        head = self.order[: self.reader.keep]
        self.reader.request(head)
        for done, i in enumerate(head, 1):
            self.reader.take([i])
            if on_progress:
                on_progress(done, len(head))


# ---------------------------------------------------------------------- output


def anchor(xy: np.ndarray, q_t: np.ndarray, q_xy: np.ndarray) -> np.ndarray:
    """Shift each track (xy [N, F, 2], in place) so it passes exactly through its
    query point (q_xy at frame index q_t). Returns the offsets [N, 2].

    A tracker's estimate on the query frame is usually a fraction of a pixel off
    the clicked point (TAPNext++ predicts on a 256-step grid: up to ~0.5 px), and
    that offset stays the same along the track. Moving the whole track keeps it
    smooth; overwriting only the query frame, as evaluation code does, would make
    it pop on that frame.
    """
    offset = q_xy - xy[np.arange(len(q_t)), q_t]
    xy += offset[:, None, :]
    return offset


def save_tracks(raw: Path, tracks: np.ndarray, visible: np.ndarray, confidence: np.ndarray,
                queries: Queries, frame_numbers: list[int]) -> Path:
    frames = np.asarray(frame_numbers, np.int32)
    raw.mkdir(parents=True, exist_ok=True)
    return save_npz(
        raw / "tracks.npz",
        tracks=np.ascontiguousarray(tracks, np.float32),
        visible=np.ascontiguousarray(visible, bool),
        confidence=np.ascontiguousarray(confidence, np.float32),
        query_frames=frames[queries.t].astype(np.int32),
        frames=frames,
        user=queries.user,
    )


def track_stats(tracks: np.ndarray, visible: np.ndarray, width: int, height: int) -> dict:
    """Plain numbers for result.json: how much of the shot the points survive."""
    n, f = visible.shape
    inside = (tracks[..., 0] >= 0) & (tracks[..., 0] <= width) & (tracks[..., 1] >= 0) & (tracks[..., 1] <= height)
    return {
        "points": int(n),
        "visible_fraction": round(float(visible.mean()), 4) if visible.size else 0.0,
        "visible_fraction_last_frame": round(float(visible[:, -1].mean()), 4) if n else 0.0,
        "visible_outside_frame": int((visible & ~inside).sum()),
    }


TRACK_FILES = ("tracks.npz: tracks [N,F,2], visible [N,F], confidence [N,F], "
               "query_frames [N], frames [F], user [N]")
ANCHORING = "each track shifted by its query-frame offset so it passes through the query point"


def write_result(run: Run, *, tracks: np.ndarray, visible: np.ndarray, confidence: np.ndarray, queries: Queries,
                 frame_numbers: list[int], width: int, height: int, p: TrackParams, user_points: int,
                 query_offset_px: np.ndarray, read_seconds: float | None = None, track_seconds: float | None = None,
                 **info) -> Path:
    """Write the point-tracking family's result: `raw/tracks.npz` plus `raw/result.json`.

    The part every worker of this family writes is written here once: pixel conventions, file description, anchoring
    description, coverage statistics and how far query points were moved. What differs per model (model name,
    visibility rule, processing resolution, chunking, ...) is passed by the worker as `**info` and placed between the
    common head and tail. Total time, load time, per-frame time and peak VRAM are Run's standard fields (run.finish);
    the times of reading frames and of tracking are passed in by the worker that measured them (read_seconds /
    track_seconds).

    (`rig_motion.write_result` follows the same pattern: the family module wraps the basic `write_result`.)
    """
    save_tracks(run.job.raw_dir, tracks, visible, confidence, queries, frame_numbers)
    timings = {k: round(v, 1) for k, v in (("read_seconds", read_seconds), ("track_seconds", track_seconds))
               if v is not None}
    return run.finish(
        frame_numbers,
        kind="point_tracks",
        files=TRACK_FILES,
        convention=CONVENTION,
        **info,
        anchoring=ANCHORING,
        query_offset_px_median=round(float(np.median(query_offset_px)), 3),
        query_offset_px_p90=round(float(np.percentile(query_offset_px, 90)), 3),
        width=width,
        height=height,
        frames=frame_numbers,  # the whole list, not the standard [first, last]: the node reads every frame by it
        grid=p.grid,
        query_frame=frame_numbers[p.query_index],
        user_points=user_points,
        **track_stats(tracks, visible, width, height),
        **timings,
    )


# ---------------------------------------------------------------------- 3D point tracks
#
# A 3D point tracker (TAPIP3D, Track4World) writes the 2D contract above (the points on the picture: tracks.npz, whose
# visibility is theirs) and
#
#   raw/tracks3d.npz  xyz  float32 [N, F, 3]  metres, in the result's world (result.json "world"):
#                                             "camera": the world of the camera the node sent (recon.load_camera: the
#                                             shot's, Y up), "own": the method's own (OpenCV axes), its cameras in
#                                             raw/cameras.npz (recon.save_cameras); nan where not known
#   result.json       world ("camera" | "own"), metric (own world: whether distances are real)
#
# It may be given the plate's depth (job input "depth": the node's depth maps, files.read_depth) and camera (job input
# "camera": recon.load_camera).


def save_tracks3d(raw: Path, xyz: np.ndarray, tracks: np.ndarray, visible: np.ndarray, confidence: np.ndarray,
                  queries: Queries, frame_numbers: list[int]) -> None:
    """tracks3d.npz and the 2D contract's tracks.npz of the same points (result.json says the world)."""
    save_tracks(raw, tracks, visible, confidence, queries, frame_numbers)
    save_npz(raw / "tracks3d.npz", xyz=np.ascontiguousarray(xyz, np.float32))


def project(xyz: np.ndarray, focal_px: np.ndarray, cam_to_world: np.ndarray, width: int, height: int) -> np.ndarray:
    """World points xyz [N, F, 3] (metres) seen by per-frame OpenCV cameras (focal [F] pixels, principal point at the
    picture's centre, cam_to_world [F, 4, 4]) -> pixels [N, F, 2], centres at +0.5; nan behind the camera."""
    w2c = np.linalg.inv(cam_to_world)
    cam = np.einsum("fij,nfj->nfi", w2c[:, :3, :3], xyz) + w2c[None, :, :3, 3]
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([cam[..., 0] / z * focal_px[None] + width / 2, cam[..., 1] / z * focal_px[None] + height / 2], -1)
    uv[z <= 1e-6] = np.nan
    return uv.astype(np.float32)
