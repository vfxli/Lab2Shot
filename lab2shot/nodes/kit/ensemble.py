"""Ensemble kernels: several results of one kind of data, already in one camera space, combined into one result.

The nodes (nodes/core/ensemble.py: ensemble_matte; each other kind its own kernel, kit/ensemble_<data>.py) read the candidates' packets and write what these
functions return; the evaluation against ground truth calls the very same functions, so the numbers it reports are
the node's. numpy only (the core environment has no SciPy / OpenCV); every function here is pure and works on arrays.

Nothing here knows a model. A candidate is an array plus what the data says of it (its guidance group); which candidate is trusted for what is decided from the data itself:
- a judge independent of the candidates where there is one (the shot's own tracks for a camera, the user's own
  selection of the object), else the candidates' agreement;
- an object-level check first, so a candidate that went wrong as a whole (the wrong person) is
  dropped before any per-pixel combination;
- candidates that share a guide (two matting models fed by one rough mask) count once in every vote (`groups`).

Alpha works on 0..1.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

import numpy as np

# ------------------------------------------------------------------ objectives

# What an ensemble may aim at, per data kind: the nodes' 「目标」 parameter is exactly this (Literal[...]); every other
# kind's is beside its kernel (kit/ensemble_<data>.py). Each has its own strategy and its own measure in the evaluation;
# every objective is offered, passed or not, and each node's description says what it gave (集成评测/一期报告.md).
MATTE_OBJECTIVES = ("accuracy", "robust", "edges")


# ------------------------------------------------------------------ names

NAME_MOST = 48  # longest layer name a result is given before its models are abbreviated
_TOKEN = re.compile(r"[^a-z0-9]+")


def _token(s: str) -> str:
    return _TOKEN.sub("", s.lower()) or "x"


def ensemble_name(data: str, objective: str, models: Sequence[str], most: int = NAME_MOST) -> str:
    """The name a combined result goes by (its layer in a multi-layer EXR, its curves' prefix), the same rule for every
    ensemble node: `<data>_ensemble_<objective>_<model>_<model>…`, the models sorted and each once. Only letters,
    digits and `_` (a Nuke layer name; a "." would split it into a layer and a channel).

    Too long: each model is cut to its first four characters (a clash gets a digit), and if that is still too long the
    models are given as their count and a six-character checksum of their full names, so a name never depends on who
    named what nor on the order of the wires."""
    head = f"{_token(data)}_ensemble_{_token(objective)}"
    names = sorted({_token(m) for m in models})
    full = "_".join([head, *names])
    if len(full) <= most:
        return full
    short: list[str] = []
    for n in names:
        cut, k = n[:4], 2
        while cut in short:
            cut, k = f"{n[:3]}{k}", k + 1
        short.append(cut)
    cut_name = "_".join([head, *short])
    if len(cut_name) <= most:
        return cut_name
    digest = f"{zlib.crc32('+'.join(names).encode()):08x}"[:6]  # a fixed checksum, not a cache key
    return f"{head}_{len(names)}models_{digest}"


def strictest_licence(commercial: Sequence[bool]) -> bool:
    """A combined result may be used commercially only when every candidate may (its strictest input's licence)."""
    return all(commercial)


# ------------------------------------------------------------------ robust statistics

MAD_TO_SIGMA = 1.4826  # median absolute deviation of a normal distribution -> its standard deviation


def wmedian(x: np.ndarray, w: np.ndarray | float = 1.0) -> np.ndarray:
    """Weighted median along axis 0 of `x` [N, ...] (NaN = no value there), weights `w` broadcast to it (>= 0).
    Where the weight is split exactly in half the two middle values are averaged (an unweighted even count gives the
    usual median). NaN where no candidate has a value."""
    x = np.asarray(x, np.float64)
    w = np.where(np.isfinite(x), np.broadcast_to(np.asarray(w, np.float64), x.shape), 0.0)
    order = np.argsort(np.where(np.isfinite(x), x, np.inf), axis=0, kind="stable")
    xs = np.take_along_axis(x, order, 0)
    ws = np.take_along_axis(w, order, 0)
    c = np.cumsum(ws, 0)
    total = c[-1]
    half = 0.5 * total
    n = x.shape[0]
    lo = np.minimum((c < half - 1e-12 * np.maximum(total, 1)).sum(0), n - 1)
    lo_val = np.take_along_axis(xs, lo[None], 0)[0]
    at_half = np.abs(np.take_along_axis(c, lo[None], 0)[0] - half) <= 1e-9 * np.maximum(total, 1)
    hi = np.minimum(lo + 1, n - 1)
    # the next value with weight (a zero-weight value in between is skipped by taking the next finite one)
    hi_val = np.take_along_axis(xs, hi[None], 0)[0]
    out = np.where(at_half & np.isfinite(hi_val), 0.5 * (lo_val + hi_val), lo_val)
    return np.where(total > 0, out, np.nan)


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over the (2r+1)² window around every pixel (clipped at the borders), through an integral image."""
    a = np.asarray(a, np.float64)
    if r <= 0:
        return a.copy()
    h, w = a.shape
    s = np.zeros((h + 1, w + 1))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    y0 = np.clip(np.arange(h) - r, 0, h)
    y1 = np.clip(np.arange(h) + r + 1, 0, h)
    x0 = np.clip(np.arange(w) - r, 0, w)
    x1 = np.clip(np.arange(w) + r + 1, 0, w)
    return s[y1][:, x1] - s[y0][:, x1] - s[y1][:, x0] + s[y0][:, x0]


# ------------------------------------------------------------------ matte


def binary_iou(a: np.ndarray, b: np.ndarray, at: float = 0.5) -> float:
    """Intersection over union of two alphas cut at `at` (1 when both are empty)."""
    pa, pb = a > at, b > at
    u = (pa | pb).sum()
    return float((pa & pb).sum() / u) if u else 1.0


def group_means(A: np.ndarray, groups: Sequence[int]) -> tuple[list[int], np.ndarray]:
    """The groups in order of first appearance and each group's mean alpha [G,H,W]: candidates that share a guide
    (one rough mask behind several matting models) are one voice."""
    order = list(dict.fromkeys(groups))
    g = np.asarray(groups)
    return order, np.stack([A[g == k].mean(0) for k in order])


def identity_vote(reps: np.ndarray, prior: np.ndarray | None, agree: float = 0.9,
                  rank: np.ndarray | None = None) -> np.ndarray:
    """Which groups [G] show the right object in this frame (the object-level check).

    Groups that agree with one another (binary IoU ≥ `agree`) form a party; the biggest party wins. A tie (two groups
    that disagree, or no majority) is settled by `prior`, what is known to be the object here: the user's own selection
    on the frames it covers, else the combined result of the frame before (the object does not jump from frame to
    frame; a model that suddenly takes another person or adds one does). `rank` [G], when the user selected the object:
    how closely each group drew that selection where it was made (selection_fit); a tie then goes to the closest on
    every frame, since a model that follows a rough mask drifts with it while the one that drew the selection keeps to
    it. Without either a tie keeps everyone."""
    n = len(reps)
    if n == 1:
        return np.ones(1, bool)
    iou = np.array([[binary_iou(reps[i], reps[j]) if i != j else 1.0 for j in range(n)] for i in range(n)])
    party = iou >= agree
    sizes = party.sum(1)
    best = sizes.max()
    leaders = np.flatnonzero(sizes == best)
    if len(leaders) == 1 or np.all([party[leaders[0], j] for j in leaders]):
        return party[leaders[0]].copy()
    if rank is not None:
        return party[leaders[np.argmax(np.asarray(rank)[leaders])]].copy()
    if prior is None:
        return np.ones(n, bool)
    fit = np.array([binary_iou(reps[i], prior) for i in range(n)])
    win = leaders[np.argmax(fit[leaders])]
    return party[win].copy()


def disagreement_band(A: np.ndarray, low: float = 0.05, high: float = 0.95, grow: int = 2) -> np.ndarray:
    """Where the kept candidates do not all say 'background' (< low) nor all say 'foreground' (> high), grown by `grow`
    pixels: the unknown region of a trimap, the only place an edge source is asked."""
    fg = (A > high).all(0)
    bg = (A < low).all(0)
    band = ~(fg | bg)
    if grow > 0:
        band = box_sum(band.astype(np.float64), grow) > 0
    return band


def matte_confidence(A: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """0..1, higher = more trusted: 1 − 2·σ of the kept candidates' alpha (σ ≤ 0.5), 1 where they all agree."""
    k = kept[:, None, None] if kept.ndim == 1 else kept
    sel = np.where(k, A, np.nan)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(sel, 0)
    return np.clip(1.0 - 2.0 * np.nan_to_num(sd), 0.0, 1.0)


def matte_fuse(A: np.ndarray, kept: np.ndarray, w: np.ndarray, source: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """One frame's combined alpha and which candidate each pixel came from ([H,W] int: the source inside the band,
    else the kept candidate nearest the value). The kept candidates' weighted mean; with an edge `source` (kept), the
    band where they disagree is taken from it instead."""
    k = np.asarray(kept, bool)
    if not k.any():
        k = np.ones(len(A), bool)
    ww = np.where(k, np.asarray(w, np.float64), 0.0)
    if ww.sum() <= 0:
        ww = k.astype(np.float64)
    out = np.tensordot(ww / ww.sum(), A, axes=1).astype(np.float32)
    idx = np.flatnonzero(k)
    near = idx[np.argmin(np.abs(A[idx] - out[None]), 0)]
    if source >= 0 and k[source]:
        band = disagreement_band(A[k])
        out = np.where(band, A[source], out).astype(np.float32)
        near = np.where(band, source, near)
    return np.clip(out, 0.0, 1.0), near


# ------------------------------------------------------------------ leaning on the most trusted candidate


def trusted_share(disagreement: np.ndarray, floor: float) -> np.ndarray:
    """How much of a pixel goes to the most trusted candidate, 0..1: u/(u + floor) of a disagreement u (0 where the
    candidates agree, so the combination stands; one half at u = floor; towards 1 where they disagree widely). The
    same mapping as the confidence (0.01/(0.01 + u)) the other way round: what is not confidence in the combination
    goes to the candidate trusted most. NaN counts as no disagreement."""
    u = np.maximum(np.nan_to_num(np.asarray(disagreement, np.float64), nan=0.0), 0.0)
    return u / (u + floor)


def lean_on_trusted(value: np.ndarray, trusted: np.ndarray, disagreement: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """The per-pixel gate between the combination and the single candidate trusted most (the one that was best on
    its own on the tuning clips, marked by the card): where the candidates agree the combination is kept (it is
    better there), where they disagree the result leans on the trusted one in proportion (a mean of disagreeing
    alphas is often worse than the best of them: a hair edge doubled, a soft edge smeared).
    No ground truth involved: the disagreement is among the candidates themselves. Returns (result, share [H,W]);
    where either has no value the other is taken as is, where the combination has none it stays none."""
    t = trusted_share(disagreement, floor)
    both = np.isfinite(value) & np.isfinite(trusted)
    out = np.where(both, t * np.where(both, trusted, 0.0) + (1.0 - t) * np.where(both, value, 0.0), value)
    return out, np.where(both, t, 0.0)


def matte_deviation(A: np.ndarray, kept: np.ndarray, trusted: int) -> np.ndarray:
    """How far the trusted alpha is from the mean of the other kept candidates ([H,W], 0..1; 0 when it is the only
    one kept): the alpha ensemble's disagreement for lean_on_trusted."""
    others = [k for k in np.flatnonzero(kept) if k != trusted]
    if not others:
        return np.zeros(A.shape[1:])
    return np.abs(A[trusted] - A[others].mean(0))


# ------------------------------------------------------------------ constants (fitted on the tuning clips only;
# 集成评测/一期报告.md lists the values tried and what each gave)

MATTE_AGREE = 0.9  # two groups whose binary alphas overlap less than this show different objects
SELECTION_MARGIN = 0.08  # a candidate this much further (IoU) from the user's selection than the best one is left out
SELECTION_MARGIN_ROBUST = 0.03  # … for the robust objective: stricter, fewer wrong objects at some cost in accuracy
# Leaning on the most trusted candidate where the candidates disagree (lean_on_trusted; 集成评测/一期b报告.md): the
# disagreement at which the result is half the combination, half the trusted candidate. Alpha only (for depth the same
# gate was measured too and made the held-out clips worse, so the depth ensemble does not lean).
TRUSTED_MATTE_DEVIATION = 0.2  # |trusted α − the other kept candidates' mean α|


def selection_fit(A: np.ndarray, selection: np.ndarray) -> np.ndarray:
    """Each alpha's binary IoU [N] with the user's selection."""
    return np.array([binary_iou(a, selection) for a in A])


def selection_check(A: np.ndarray, selection: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Which candidates [N] draw the object the user selected (the 「对象」 mask, on a frame it covers): one whose binary
    alpha overlaps the selection more than `margin` (IoU) less than the best one's takes in more, or less, than the
    selected object (another person, the chair the person sits on) and is left out of the whole shot. Judged per
    candidate, not per guide group: a model that refines its rough mask may well have mended what the mask got wrong."""
    fit = selection_fit(A, selection)
    return fit >= fit.max() - margin


# ------------------------------------------------------------------ poses
#
# Robust averages of poses, for the ensembles that combine cameras (the camera kernel:
# kit/ensemble_camera.py).


def rotation_median(R: np.ndarray, w: np.ndarray | None = None, steps: int = 10) -> np.ndarray:
    """The L1 (Weiszfeld) mean of rotations [N,3,3] on SO(3), weights `w` [N] (Hartley et al.'s rotation averaging):
    from the chordal mean, repeatedly step along the weighted mean of the unit directions to each rotation. One wrong
    rotation pulls it far less than it pulls an average."""
    from lab2shot_shared.motion import matrix_to_rotvec, rotvec_to_matrix
    from lab2shot_shared.poses import mean_rotation

    R = np.asarray(R, np.float64)
    w = np.ones(len(R)) if w is None else np.asarray(w, np.float64)
    m = mean_rotation(R * w[:, None, None])
    for _ in range(steps):
        v = matrix_to_rotvec(np.swapaxes(m, -1, -2)[None] @ R)  # each rotation seen from m
        d = np.linalg.norm(v, axis=-1)
        k = w / np.maximum(d, 1e-9)
        step = (k[:, None] * v).sum(0) / max(k.sum(), 1e-12)
        if np.linalg.norm(step) < 1e-9:
            break
        m = m @ rotvec_to_matrix(step)
    return m


def geometric_median(x: np.ndarray, w: np.ndarray | None = None, steps: int = 50) -> np.ndarray:
    """The point [3] with the least weighted sum of distances to points x [N,3] (Weiszfeld)."""
    x = np.asarray(x, np.float64)
    w = np.ones(len(x)) if w is None else np.asarray(w, np.float64)
    m = (w[:, None] * x).sum(0) / max(w.sum(), 1e-12)
    for _ in range(steps):
        d = np.maximum(np.linalg.norm(x - m, axis=1), 1e-9)
        k = w / d
        new = (k[:, None] * x).sum(0) / k.sum()
        if np.linalg.norm(new - m) < 1e-9 * (1 + np.linalg.norm(m)):
            return new
        m = new
    return m
