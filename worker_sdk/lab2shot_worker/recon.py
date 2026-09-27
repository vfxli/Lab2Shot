"""Chunked multi-view reconstruction, shared by every worker that writes the `reconstruction` contract (VGGT, Pi3,
Depth Anything 3, MapAnything, CUT3R, MonST3R, COLMAP masks). Needs numpy only.

A shot longer than one forward pass / optimisation is cut into overlapping chunks (plan_chunks). Each chunk is
solved in its own world; the Stitcher brings every chunk into the first one's world by a similarity transform:

    rotation            rotation="cameras" (default): mean relative orientation of the cameras of the frames both
                        chunks solved (well defined even for a telephoto's narrow, nearly flat point cloud, where a
                        point-only fit drifts: best for feed-forward models); rotation="points": fitted with scale
                        and translation on the points (better where a chunk's end cameras are the least reliable
                        part, e.g. an optimisation with temporal smoothing);
    scale, translation  least squares on the 3D points of the same pixels of those frames (weights 1/depth^2, so
                        near and far points count alike; pairs beyond 3x the median relative residual dropped).

Across the shared frames cameras, intrinsics, depth and confidence cross-fade from the earlier chunk to the later
one (no step at the seam). Cameras come out relative to the first frame's camera (the contract's world).

Chained chunk after chunk, the fits' errors add up. Loop closure (Stitcher(keep=folder)): windows of frames far apart
that see the same place are reconstructed together as loop chunks, and the chunk transforms are optimised over every
fit (optimize_chunks) before any frame is placed (see Stitcher).

The raw contract (node side: lab2shot/nodes/results.py reconstruction()):

    raw/cameras.npz      frames [F] int64, K [F,3,3] pixels at the input resolution (pixel (i, j) covers
                         [j, j+1] x [i, i+1]), cam_to_world [F,4,4] OpenCV camera, world = first frame's camera
    raw/frame_<n>.npz    depth [H,W] camera Z, confidence [H,W], mask [H,W] bool (reliable static geometry)
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lab2shot_shared.motion import matrix_to_rotvec, rotvec_to_matrix  # noqa: F401
from lab2shot_shared.poses import (blend_poses, interpolate_poses, mean_rotation, rotation_deg,  # noqa: F401
                                   umeyama, unproject)

from . import Job, fail, read_mask, save_npz

BOX_MARGIN = 0.10  # people boxes grow by 10 % of their width / height on every side (arms, hair, shadows)
MASK_THRESHOLD = 0.5  # mask input value above which a pixel belongs to a moving object
ALIGN_SAMPLES = 200_000  # 3D point pairs used to fit one chunk-to-world similarity
MIN_PAIRS = 100
CONVENTION = (
    "OpenCV camera: +X right, +Y down, +Z forward; cam_to_world maps camera to world; world = the first frame's "
    "camera; K in pixels at the input resolution, pixel (i, j) covers [j, j+1] x [i, i+1] (a centred principal "
    "point is (W/2, H/2)); depth = camera Z"
)


# ---------------------------------------------------------------------- moving objects


class MovingMasks:
    """frame -> bool [H,W] at the input resolution, True on moving objects: the job's "mask" input (> 0.5) and its
    "boxes" input (people boxes grown by BOX_MARGIN). Read on demand, so long shots never hold every mask."""

    def __init__(self, job: Job, width: int, height: int) -> None:
        self.width, self.height = width, height
        self.files = job.listing("mask")
        self.boxes: dict[int, list] = {}
        for boxes in job.people().values():
            for frame, box in boxes.items():
                self.boxes.setdefault(frame, []).append(box)
        self.kinds = [k for k in ("mask", "boxes") if k in job.inputs]  # for result.json

    def __len__(self) -> int:
        return len(set(self.files) | set(self.boxes))

    def __contains__(self, frame: int) -> bool:
        return frame in self.files or frame in self.boxes

    def get(self, frame: int) -> np.ndarray | None:
        if frame not in self:
            return None
        w, h = self.width, self.height
        moving = np.zeros((h, w), dtype=bool)
        if frame in self.files:
            m = read_mask(self.files[frame]) > MASK_THRESHOLD
            if m.shape != (h, w):
                m = m[np.arange(h) * m.shape[0] // h][:, np.arange(w) * m.shape[1] // w]  # nearest, no OpenCV needed
            moving |= m
        for x1, y1, x2, y2 in self.boxes.get(frame, ()):
            mx, my = BOX_MARGIN * (x2 - x1), BOX_MARGIN * (y2 - y1)
            c0, c1 = max(0, int(np.floor(x1 - mx))), min(w, int(np.ceil(x2 + mx)))
            r0, r1 = max(0, int(np.floor(y1 - my))), min(h, int(np.ceil(y2 + my)))
            if c1 > c0 and r1 > r0:
                moving[r0:r1, c0:c1] = True
        return moving


# ---------------------------------------------------------------------- chunks


EDGE_RTOL = 0.05  # depth edge ("flying pixels"): the 3x3 neighbourhood spans a depth ratio above 1 + EDGE_RTOL


def depth_edges(depth: np.ndarray, rtol: float = EDGE_RTOL) -> np.ndarray:
    """bool, same shape as depth [..., H, W]: pixels whose 3x3 neighbourhood spans a depth ratio > 1 + rtol."""
    d = np.maximum(np.asarray(depth, np.float32), 1e-6)
    pad = [(0, 0)] * (d.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(d, pad, mode="edge")  # repeating the border adds no new values to any window
    h, w = d.shape[-2:]
    shifts = [p[..., y:y + h, x:x + w] for y in range(3) for x in range(3)]
    return np.max(shifts, axis=0) / np.min(shifts, axis=0) > 1.0 + rtol


def plan_chunks(n: int, max_frames: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) index ranges covering 0..n: the fewest chunks of at most `max_frames`, all the same length and
    spread evenly, consecutive ones sharing at least `overlap` frames (clamped to max_frames - 1)."""
    if n <= max_frames:
        return [(0, n)]
    overlap = min(overlap, max_frames - 1)
    k = math.ceil((n - overlap) / (max_frames - overlap))
    length = min(max_frames, math.ceil((n - overlap) / k) + overlap)
    starts = [round(i * (n - length) / (k - 1)) for i in range(k)]
    return [(s, s + length) for s in starts]


# ---------------------------------------------------------------------- geometry


def fit_similarity(src: np.ndarray, dst: np.ndarray, depth: np.ndarray,
                   R: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray, float]:
    """dst ~ s R src + t on 3D point pairs, robustly: weights 1/depth^2 (relative errors), refitted on the pairs
    within 3x the median relative residual. R given: only scale and translation are fitted; None: a full weighted
    Umeyama. Returns s, R, t and the RMS relative residual (fraction of depth) of the kept pairs."""
    w = 1.0 / np.maximum(depth, 1e-6) ** 2
    keep = np.ones(len(src), dtype=bool)
    fixed = R is not None
    for _ in range(3):
        wk = w[keep] / w[keep].sum()
        if fixed:
            rs = src[keep] @ R.T
            mu_s, mu_d = wk @ rs, wk @ dst[keep]
            xs, xd = rs - mu_s, dst[keep] - mu_d
            s = float(wk @ (xs * xd).sum(1) / (wk @ (xs**2).sum(1)))
            t = mu_d - s * mu_s
        else:
            s, R, t = umeyama(src[keep], dst[keep], wk)
        rel = np.linalg.norm(s * src @ R.T + t - dst, axis=1) / np.maximum(depth, 1e-6)
        keep = rel <= 3.0 * np.median(rel[keep]) + 1e-12
    return s, R, t, float(np.sqrt(np.mean(rel[keep] ** 2)))


def apply_similarity(s: float, R: np.ndarray, t: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """X_new = s R X + t applied to camera-to-world poses: rotations turn, positions move (cameras stay rigid)."""
    out = cam_to_world.copy()
    out[..., :3, :3] = R @ cam_to_world[..., :3, :3]
    out[..., :3, 3] = s * cam_to_world[..., :3, 3] @ R.T + t
    return out


# ---------------------------------------------------------------------- stitching


@dataclass
class Chunk:
    """One chunk in its own world, per frame of the chunk, at the model resolution h x w."""

    cam_to_world: np.ndarray  # float64 [n,4,4] OpenCV
    K: np.ndarray  # float64 [n,3,3] model pixels (same convention in every chunk of a worker)
    depth: np.ndarray  # float32 [n,h,w] camera Z
    confidence: np.ndarray  # float32 [n,h,w]
    usable: np.ndarray  # bool [n,h,w] reliable static geometry: used to align chunks, becomes the output mask
    extra: dict[str, np.ndarray] = field(default_factory=dict)  # more per-frame maps [n,...] carried along
    # These `extra` maps are lengths (the model's own camera-space point maps [n,h,w,3], in the same unit as depth), so
    # when chunks are joined they are multiplied by the chunk's scale T.s like `depth`; maps not listed here (boolean
    # masks, sky, moving objects) are carried as they are. Declared explicitly rather than guessed from names: `_place`
    # recognizes only this list.
    scaled: tuple[str, ...] = ()
    looks: np.ndarray | None = None  # [n,D] what each frame looks like (thumbnails()): loop closure's proposals


@dataclass
class Frame:
    """A stitched frame, handed out by Stitcher.pop: camera relative to the first frame's camera."""

    index: int
    cam_to_world: np.ndarray
    K: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    usable: np.ndarray
    extra: dict[str, np.ndarray]


@dataclass(frozen=True)
class Sim3:
    """x -> s R x + t."""

    s: float = 1.0
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __matmul__(self, other: Sim3) -> Sim3:
        return Sim3(self.s * other.s, self.R @ other.R, self.s * (self.R @ other.t) + self.t)

    def inverse(self) -> Sim3:
        return Sim3(1.0 / self.s, self.R.T, -(self.R.T @ self.t) / self.s)


GRAPH_ITERATIONS = 30


def optimize_chunks(start: list[Sim3], edges: list[tuple[int, int, Sim3]], units: list[float]) -> list[Sim3]:
    """Chunk-to-world transforms T_k (T_0 held) agreeing best with measured relative ones: an edge (i, j, S) says
    chunk j's coordinates map into chunk i's by S, T_i^-1 T_j = S. Levenberg-Marquardt on x_k = (log s, rotation
    vector, translation), residual per edge the disagreement E = S^-1 T_i^-1 T_j as (log scale, rotation vector,
    translation / units[j], the chunk's median depth: a translation error counts like the angle it spans)."""
    n = len(start)
    if n < 2 or not edges:
        return list(start)

    def unpack(x: np.ndarray) -> list[Sim3]:
        return [start[0]] + [Sim3(math.exp(v[0]), rotvec_to_matrix(v[1:4]), v[4:7]) for v in x.reshape(-1, 7)]

    def residuals(x: np.ndarray) -> np.ndarray:
        T = unpack(x)
        out = []
        for i, j, S in edges:
            E = S.inverse() @ T[i].inverse() @ T[j]
            out.append(np.concatenate([[math.log(E.s)], matrix_to_rotvec(E.R), E.t / units[j]]))
        return np.concatenate(out)

    x = np.concatenate([np.concatenate([[math.log(T.s)], matrix_to_rotvec(T.R), T.t]) for T in start[1:]])
    scale = np.tile([1.0, 1.0, 1.0, 1.0, *([max(np.median(units), 1e-9)] * 3)], n - 1)  # steps in comparable units
    r = residuals(x)
    lam = 1e-4
    for _ in range(GRAPH_ITERATIONS):
        J = np.empty((len(r), len(x)))
        for p in range(len(x)):
            dx = np.zeros_like(x)
            dx[p] = 1e-6 * scale[p]
            J[:, p] = (residuals(x + dx) - r) / dx[p]
        H, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), -g)
            r_new = residuals(x + step)
            if r_new @ r_new < r @ r:
                x, r, lam = x + step, r_new, max(lam / 10, 1e-9)
                break
            lam *= 10
            if lam > 1e8:
                return unpack(x)
        if np.abs(step / scale).max() < 1e-10:
            break
    return unpack(x)


LOOK_SIZE = 32  # thumbnails for loop proposals: LOOK_SIZE x LOOK_SIZE grey
LOOP_LOOK = 0.9  # thumbnails this alike (cosine) are worth a loop chunk, wherever the drifted solve puts them
LOOP_SEEN = 0.6  # ... or this share of one frame's points in view of the other (the solve so far), looking alike
LOOP_ANGLE_DEG = 30.0  # ... within this angle
LOOP_RESIDUAL = 0.05  # a loop chunk fits both chunks this well (relative 3D residual) or it is no loop
LOOP_AGREE_DEG = 3.0  # after the optimisation a loop still disagrees this much with the rest: it was a wrong one
POINTS_PER_FRAME = 256  # points of each frame kept to see which frames see the same place


def thumbnails(images: np.ndarray) -> np.ndarray:
    """[n,D] what frames look like: grey LOOK_SIZE x LOOK_SIZE area averages, zero mean, unit length (a cosine of two
    compares their framing). images [n,3,h,w] or [n,h,w,3] in 0..1 or 0..255."""
    a = np.asarray(images, np.float32)
    grey = a.mean(1) if a.shape[1] == 3 and a.shape[-1] != 3 else a.mean(-1)
    n, h, w = grey.shape
    rows = np.linspace(0, h, LOOK_SIZE + 1).astype(int)[:-1]
    cols = np.linspace(0, w, LOOK_SIZE + 1).astype(int)[:-1]
    area = np.outer(np.diff([*rows, h]), np.diff([*cols, w]))  # pixels per cell: cells of a small picture differ
    small = (np.add.reduceat(np.add.reduceat(grey, rows, axis=1), cols, axis=2) / area).reshape(n, -1)
    small -= small.mean(1, keepdims=True)
    return small / np.maximum(np.linalg.norm(small, axis=1, keepdims=True), 1e-9)


class Stitcher:
    """Chunks (added in order) -> one world, each chunk placed by the similarity fitted on the frames it shares with
    the one before (T_k = T_k-1 S_k). Frames stay until popped: pop only frames no later chunk shares.

    Loop closure (keep = a folder): the chunks chained one after another drift, however well each is fitted. With a
    folder to keep chunks in until the end, nothing is placed while chunks come in; loop_candidates() then names
    pairs of windows of frames far apart in the shot that see the same place, the worker reconstructs each pair
    together as one more chunk (add_loop), and close() finds the chunk transforms agreeing best with every fit, the
    chunk-to-chunk ones and the loops (optimize_chunks), before it places the chunks and hands out their frames."""

    def __init__(self, rotation: str = "cameras", samples: int = ALIGN_SAMPLES, seed: int = 0, keep: Path | None = None) -> None:
        if rotation not in ("cameras", "points"):
            raise ValueError(f"rotation must be 'cameras' or 'points', not {rotation!r}")
        self.rotation = rotation
        self.frames: dict[int, dict] = {}  # placed, waiting to be popped
        self.alignments: list[dict] = []
        self.samples = samples
        self.rng = np.random.default_rng(seed)
        self.origin: np.ndarray | None = None  # world -> first frame's camera
        self.keep = keep
        self.chunks: list[tuple[int, int]] = []  # (start, count) of each chunk added
        self.T: list[Sim3] = []  # chunk -> world
        self.edges: list[tuple[int, int, Sim3]] = []  # (i, j, S): chunk j into chunk i, measured
        self.loops: list[dict] = []  # what each loop chunk gave
        self.units: list[float] = []  # each chunk's median depth
        self._last: tuple[int, Chunk] | None = None
        self._views: dict[int, dict] = {}  # loop proposals: per frame its chunk, camera, lens, points, look
        self.loop_correction: dict = {}  # how far close() moved the chunks from where chaining put them
        self.kept_bytes = 0  # the chunks kept on disk until close()
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)

    # ---- fitting

    def _fit(self, src: Chunk, src_frames: list[int], dst: Chunk, dst_frames: list[int]) -> tuple[Sim3, dict]:
        """The similarity bringing `src` into `dst`'s world, fitted on the frames they both have (src_frames[j] is
        src's j-th frame's index in the shot)."""
        shared = [(j, dst_frames.index(f)) for j, f in enumerate(src_frames) if f in dst_frames]
        if not shared:
            raise ValueError("chunk shares no frame with the stitched ones")
        pts_src, pts_dst, zs = [], [], []
        for j, i in shared:
            rows, cols = np.nonzero(src.usable[j] & dst.usable[i])
            if len(rows):
                pts_src.append(unproject(src.depth[j], src.K[j], src.cam_to_world[j], rows, cols))
                pts_dst.append(unproject(dst.depth[i], dst.K[i], dst.cam_to_world[i], rows, cols))
                zs.append(dst.depth[i][rows, cols].astype(np.float64))
        pairs = sum(len(p) for p in pts_src)
        if pairs < MIN_PAIRS:
            raise ValueError(f"only {pairs} reliable shared pixels between chunks")
        pts_src, pts_dst, zs = np.concatenate(pts_src), np.concatenate(pts_dst), np.concatenate(zs)
        if len(pts_src) > self.samples:
            pick = self.rng.choice(len(pts_src), self.samples, replace=False)
            pts_src, pts_dst, zs = pts_src[pick], pts_dst[pick], zs[pick]
        relative = np.stack([dst.cam_to_world[i][:3, :3] @ src.cam_to_world[j][:3, :3].T for j, i in shared])
        s, R, t, rel_rms = fit_similarity(pts_src, pts_dst, zs, mean_rotation(relative) if self.rotation == "cameras" else None)
        return Sim3(s, R, t), {"rotation_from": self.rotation, "scale": s, "rotation_deg": float(rotation_deg(R)),
                               "points": len(pts_src), "residual_rms_relative": rel_rms,
                               "orientation_disagreement_deg": float(rotation_deg(R.T @ relative).max()), "shared": len(shared)}

    def add(self, start: int, chunk: Chunk) -> dict:
        """Chunk `chunk` (frames start ..) after the others; returns its fit to the one before (identity for the first)."""
        n = len(chunk.cam_to_world)
        k = len(self.chunks)
        info: dict = {"start": start, "count": n}
        if k == 0:
            T = Sim3()
        else:
            last_start, last = self._last
            S, fit = self._fit(chunk, list(range(start, start + n)), last, list(range(last_start, last_start + len(last.depth))))
            info.update(fit)
            T = self.T[-1] @ S
            self.edges.append((k - 1, k, S))
        self.alignments.append(info)
        self.chunks.append((start, n))
        self.T.append(T)
        self.units.append(float(np.median(chunk.depth[chunk.usable])) if chunk.usable.any() else 1.0)
        self._last = (start, chunk)
        if self.keep is None:
            self._place(start, chunk, T)
        else:
            self._remember(k, start, chunk)
            save_npz(self.keep / f"chunk_{k}.npz", cam_to_world=chunk.cam_to_world, K=chunk.K, depth=chunk.depth,
                     confidence=chunk.confidence, usable=chunk.usable, scaled=np.array(chunk.scaled, dtype="<U64"),
                     **{f"extra_{name}": v for name, v in chunk.extra.items()})
            self.kept_bytes += (self.keep / f"chunk_{k}.npz").stat().st_size
        return info

    def _load(self, k: int) -> Chunk:
        d = np.load(self.keep / f"chunk_{k}.npz")
        return Chunk(d["cam_to_world"], d["K"], d["depth"], d["confidence"], d["usable"],
                     {name.removeprefix("extra_"): d[name] for name in d.files if name.startswith("extra_")},
                     tuple(str(s) for s in d["scaled"]) if "scaled" in d.files else ())

    def _place(self, start: int, chunk: Chunk, T: Sim3) -> None:
        """The chunk's frames into the world by T, cross-fading from the earlier chunk across the frames both have."""
        n = len(chunk.cam_to_world)
        shared = [j for j in range(n) if start + j in self.frames]
        cams = apply_similarity(T.s, T.R, T.t, chunk.cam_to_world)
        for j in range(n):
            new = {"cam_to_world": cams[j], "K": chunk.K[j], "depth": chunk.depth[j] * T.s, "confidence": chunk.confidence[j],
                   "usable": chunk.usable[j],
                   # extras that are lengths (Chunk.scaled: the model's own 3D point maps) are brought to this chunk's
                   # scale with depth
                   "extra": {k: (v[j] * T.s if k in chunk.scaled else v[j]) for k, v in chunk.extra.items()}}
            old = self.frames.get(start + j)
            if old is not None:
                w = (shared.index(j) + 1) / (len(shared) + 1)
                new = {"cam_to_world": blend_poses(old["cam_to_world"], new["cam_to_world"], w),
                       "K": (1 - w) * old["K"] + w * new["K"],
                       "depth": (1 - w) * old["depth"] + w * new["depth"],
                       "confidence": (1 - w) * old["confidence"] + w * new["confidence"],
                       "usable": old["usable"] & new["usable"],
                       "extra": {k: (old["extra"][k] | v) if v.dtype == bool else (1 - w) * old["extra"][k] + w * v
                                 for k, v in new["extra"].items()}}
            self.frames[start + j] = new
        if self.origin is None:
            self.origin = np.linalg.inv(self.frames[min(self.frames)]["cam_to_world"])

    def pop(self, until: int | None = None) -> Iterator[Frame]:
        """Frames with index < `until` (all when None), in order, removed from the stitcher."""
        for k in sorted(i for i in self.frames if until is None or i < until):
            f = self.frames.pop(k)
            yield Frame(k, self.origin @ f["cam_to_world"], f["K"], f["depth"], f["confidence"], f["usable"], f["extra"])

    # ---- loop closure

    def _remember(self, k: int, start: int, chunk: Chunk) -> None:
        """What loop proposals look at, per frame (the first chunk that has it): a few of its points in its camera,
        its camera and lens in its chunk, its look."""
        h, w = chunk.depth.shape[1:]
        for j in range(len(chunk.depth)):
            if start + j in self._views:
                continue
            rows, cols = np.nonzero(chunk.usable[j])
            if len(rows) > POINTS_PER_FRAME:
                pick = self.rng.choice(len(rows), POINTS_PER_FRAME, replace=False)
                rows, cols = rows[pick], cols[pick]
            self._views[start + j] = {
                "chunk": k, "cam": chunk.cam_to_world[j], "K": chunk.K[j], "size": (w, h),
                "points": unproject(chunk.depth[j], chunk.K[j], np.eye(4), rows, cols),
                "look": None if chunk.looks is None else chunk.looks[j]}

    def _world(self, index: int) -> np.ndarray:
        v = self._views[index]
        T = self.T[v["chunk"]]
        return apply_similarity(T.s, T.R, T.t, v["cam"][None])[0]

    def _seen(self, a: int, b: int) -> float:
        """Share of frame a's points that frame b's camera (as chained so far) has in view, 0 when the two look more
        than LOOP_ANGLE_DEG apart."""
        ca, cb = self._world(a), self._world(b)
        if rotation_deg(ca[:3, :3].T @ cb[:3, :3]) > LOOP_ANGLE_DEG:
            return 0.0
        va, vb = self._views[a], self._views[b]
        if not len(va["points"]):
            return 0.0
        world = va["points"] @ ca[:3, :3].T + ca[:3, 3]
        cam = (world - cb[:3, 3]) @ cb[:3, :3]  # into b's camera
        z = cam[:, 2]
        u = vb["K"][0, 0] * cam[:, 0] / np.maximum(z, 1e-9) + vb["K"][0, 2]
        v = vb["K"][1, 1] * cam[:, 1] / np.maximum(z, 1e-9) + vb["K"][1, 2]
        w, h = vb["size"]
        return float(np.mean((z > 0) & (u >= 0) & (u <= w) & (v >= 0) & (v <= h)))

    def loop_candidates(self, window: int, most: int) -> list[tuple[list[int], list[int]]]:
        """Up to `most` pairs of frame windows (each inside one chunk, the two chunks sharing no frame) that seem to
        see the same place: frames that look alike (LOOP_LOOK), or whose points the other has in view by the solve
        so far (LOOP_SEEN). One pair per two chunks, the likeliest first."""
        if self.keep is None or len(self.chunks) < 3:
            return []
        stride = max(1, window // 2)
        best: dict[tuple[int, int], tuple[float, int, int]] = {}
        frames = sorted(self._views)[::stride]
        for a in frames:
            for b in frames:
                ka, kb = self._views[a]["chunk"], self._views[b]["chunk"]
                if kb <= ka or self._overlap(ka, kb):
                    continue
                la, lb = self._views[a]["look"], self._views[b]["look"]
                look = float(la @ lb) if la is not None and lb is not None else 0.0
                seen = min(self._seen(a, b), self._seen(b, a))
                score = max(look - LOOP_LOOK, (seen - LOOP_SEEN) * (look > 0.5 or la is None))
                if score >= 0 and score + 1 > best.get((ka, kb), (-1.0,))[0]:
                    best[(ka, kb)] = (score + 1, a, b)
        chosen = sorted(best.values(), reverse=True)[:most]
        return [(self._window(a, window), self._window(b, window)) for _, a, b in chosen]

    def _overlap(self, ka: int, kb: int) -> bool:
        (sa, na), (sb, nb) = self.chunks[ka], self.chunks[kb]
        return sa < sb + nb and sb < sa + na

    def _window(self, index: int, size: int) -> list[int]:
        """`size` frames around `index` inside its chunk."""
        start, n = self.chunks[self._views[index]["chunk"]]
        size = min(size, n)
        first = min(max(start, index - size // 2), start + n - size)
        return list(range(first, first + size))

    def add_loop(self, first: list[int], second: list[int], chunk: Chunk) -> dict:
        """A loop chunk: the frames `first` + `second` (two windows loop_candidates gave) reconstructed together. It is
        fitted to each window's chunk; if both fits hold (LOOP_RESIDUAL), chunk b's coordinates into chunk a's are
        one more measurement for close()."""
        ka, kb = (next(k for k, (start, n) in enumerate(self.chunks) if start <= w[0] and w[-1] < start + n)
                  for w in (first, second))  # a chunk that has the whole window
        info: dict = {"frames": [first[0], first[-1], second[0], second[-1]], "chunks": [ka, kb]}
        try:
            loop = list(first) + list(second)
            (Sa, fa), (Sb, fb) = (self._fit(chunk, loop, self._load(k), list(range(self.chunks[k][0], sum(self.chunks[k]))))
                                  for k in (ka, kb))  # the loop chunk into each chunk (it has frames of both)
        except ValueError as exc:
            info.update(used=False, why=str(exc))
            self.loops.append(info)
            return info
        worst = max(fa["residual_rms_relative"], fb["residual_rms_relative"])
        info.update(residual_rms_relative=[fa["residual_rms_relative"], fb["residual_rms_relative"]], used=worst <= LOOP_RESIDUAL)
        if info["used"]:
            info["edge"] = len(self.edges)
            self.edges.append((ka, kb, Sa @ Sb.inverse()))
        else:
            info["why"] = f"the loop chunk does not fit both chunks (relative residual {worst:.3f})"
        self.loops.append(info)
        return info

    def close(self) -> Iterator[Frame]:
        """Every frame not handed out yet, in order. With loops: the chunk transforms optimised over all fits first
        (a loop still disagreeing LOOP_AGREE_DEG with the rest afterwards was a wrong one: dropped, optimised again),
        then the chunks placed with them one by one; the kept chunks are deleted."""
        if self.keep is not None:
            chained = list(self.T)
            while any(loop.get("used") for loop in self.loops):
                self.T = optimize_chunks(chained, self.edges, self.units)
                wrong = [loop for loop in self.loops if loop.get("used") and self._disagreement(loop) > LOOP_AGREE_DEG]
                if not wrong:
                    break
                worst = max(wrong, key=self._disagreement)
                worst.update(used=False, why=f"disagrees {self._disagreement(worst):.1f} deg with the rest after optimising")
                self.edges = [e for i, e in enumerate(self.edges) if i != worst["edge"]]
                for loop in self.loops:  # edge numbers after the removed one move down
                    if loop.get("used") and loop["edge"] > worst["edge"]:
                        loop["edge"] -= 1
            else:
                self.T = chained
            for k, (start, _) in enumerate(self.chunks):
                self._place(start, self._load(k), self.T[k])
                yield from self.pop(self.chunks[k + 1][0] if k + 1 < len(self.chunks) else None)
                (self.keep / f"chunk_{k}.npz").unlink()
            moved = [float(np.linalg.norm(T.t - C.t) / self.units[k]) for k, (T, C) in enumerate(zip(self.T, chained))]
            for loop in self.loops:
                loop.pop("edge", None)
            self.loop_correction = {"max_move_over_depth": max(moved), "max_turn_deg": max(float(rotation_deg(T.R @ C.R.T))
                                    for T, C in zip(self.T, chained)), "loops_used": sum(bool(lp.get("used")) for lp in self.loops),
                                    "kept_mb": round(self.kept_bytes / 2**20)}
        yield from self.pop()

    def _disagreement(self, loop: dict) -> float:
        """Degrees a loop edge's rotation still disagrees with the optimised chunk transforms."""
        i, j, S = self.edges[loop["edge"]]
        return float(rotation_deg((S.inverse() @ self.T[i].inverse() @ self.T[j]).R))


# ---------------------------------------------------------------------- a camera the node sends (the "camera" input)


@dataclass
class InputCamera:
    """A camera of the plate a node sends its worker (results.send_camera): per frame its focal length and where it
    is, in the camera's own world (the shot's, Y up), OpenCV camera axes, metres."""

    frames: np.ndarray  # [F] frame numbers
    focal_px: np.ndarray  # [F], principal point at the image centre
    cam_to_world: np.ndarray  # [F,4,4] OpenCV, metres
    rotation_only: bool = False
    """True 时**这不是一台相机，只是每帧的旋转**（kit/cameras.py send_rotation 写的）：
    位移全是 0，因为节点上根本没有「相机」输入口，用户接的是「相机旋转」那一个参数（真要每帧旋转时
    不能直接接一台相机，那会把它的位移一起丢掉）。
    **拿它当一台相机用的地方必须先看这个标志**：把人整体搬到「这台相机的世界」那种事，
    对着一台位移恒为 0 的相机做是没有意义的。"""

    def at(self, frames) -> tuple[np.ndarray, np.ndarray]:
        """(focal [N], cam_to_world [N,4,4]) at these frame numbers (nearest sample for missing ones)."""
        idx = [int(np.argmin(np.abs(self.frames - f))) for f in frames]
        return self.focal_px[idx], self.cam_to_world[idx]


def load_camera(job) -> InputCamera | None:
    """The "camera" input (npz: frames, focal_px [F], cam_to_world [F,4,4] OpenCV, metres); None without it.
    `rotation_only` in the file marks a rotation the node got on its 「相机旋转」 parameter rather than a camera
    (kit/cameras.py send_rotation): the poses then carry no position (see InputCamera.rotation_only)."""
    path = job.inputs.get("camera")
    if path is None:
        return None
    d = np.load(path)
    frames = np.asarray(d["frames"]).astype(np.int64)
    focal = np.asarray(d["focal_px"], np.float64).reshape(-1)
    if "cam_to_world" not in d:
        fail("E-RECON-NOPOSES")
    c2w = np.asarray(d["cam_to_world"], np.float64).reshape(-1, 4, 4)
    if not (len(frames) == len(focal) == len(c2w)) or not len(frames):
        fail("E-RECON-CAMLENGTHS")
    if not np.all(focal > 0):
        fail("E-RECON-CAMFOCAL")
    return InputCamera(frames, focal, c2w, bool(d["rotation_only"]) if "rotation_only" in d else False)


# ---------------------------------------------------------------------- outputs


def save_cameras(raw: Path, frames: list[int], K: np.ndarray, cam_to_world: np.ndarray, width: int, height: int,
                 **extra) -> None:
    """raw/cameras.npz of the contract (K at the input resolution, cameras relative to the first frame's)."""
    save_npz(raw / "cameras.npz", frames=np.asarray(frames, np.int64), K=np.asarray(K, np.float64),
             cam_to_world=np.asarray(cam_to_world, np.float64), width=np.int64(width), height=np.int64(height), **extra)


def at_input_size(frame: Frame, width: int, height: int,
                  exclude: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A joined frame scaled back to the input resolution, as (depth, confidence, usable mask).

    Depth and confidence are bilinear, the usable mask nearest-neighbour. `exclude` (a boolean mask at the input
    resolution: moving objects, sky; used by workers such as LingBot-Map that carry them in `Frame.extra`) is removed
    from the usable mask; None removes nothing. A caller adding its own conditions to the mask can apply `&=` to the
    result: all are bitwise AND, so the order does not matter.
    """
    import cv2  # this module does not require OpenCV otherwise (files.py imports it on demand too); only upscaling to the input resolution needs it

    d = cv2.resize(frame.depth, (width, height), interpolation=cv2.INTER_LINEAR)
    c = cv2.resize(frame.confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(frame.usable.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    if exclude is not None:
        m &= ~exclude
    return d, c, m


def save_frame(raw: Path, frame: int, depth: np.ndarray, confidence: np.ndarray, mask: np.ndarray, **extra) -> None:
    """raw/frame_<n>.npz of the contract, at the input resolution."""
    save_npz(raw / f"frame_{frame}.npz", depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
             confidence=np.asarray(confidence, np.float32), mask=np.asarray(mask, bool), **extra)


def summary(cam_to_world: np.ndarray, K: np.ndarray, width: int, depth_median: list[float]) -> dict:
    """result.json numbers every reconstruction worker reports: focal, and camera motion relative to the scene
    (a locked-off shot stays near zero)."""
    focal = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
    scene = float(np.nanmedian(depth_median)) if np.isfinite(depth_median).any() else float("nan")
    travel = float(np.linalg.norm(cam_to_world[:, :3, 3] - cam_to_world[0, :3, 3], axis=1).max())
    path = float(np.linalg.norm(np.diff(cam_to_world[:, :3, 3], axis=0), axis=1).sum()) if len(focal) > 1 else 0.0
    return {
        "focal_px": {"median": float(np.median(focal)), "min": float(focal.min()), "max": float(focal.max()),
                     "per_frame": [round(float(f), 2) for f in focal]},
        "fov_x_deg_median": math.degrees(2 * math.atan(width / (2 * float(np.median(focal))))),
        "depth_median": [None if not np.isfinite(d) else round(float(d), 4) for d in depth_median],
        "scene_depth_median": scene,
        "drift": {"max_translation": travel,
                  "max_translation_relative_to_scene_depth": travel / scene if scene > 0 else None,
                  "max_rotation_deg": float(rotation_deg((np.linalg.inv(cam_to_world[0]) @ cam_to_world)[:, :3, :3]).max()),
                  "path_length": path},
    }
