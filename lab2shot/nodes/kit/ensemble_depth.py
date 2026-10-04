"""Depth ensemble kernel: depth maps of several models, aligned to one reference (one camera space), combined in log
depth into one.

The node (nodes/core/ensemble_depth.py: ensemble_depth) reads the candidates' packets and writes what these functions
return; the evaluation against ground truth (集成评测/ens, 集成评测/实测/depth) calls the very same functions. numpy
only, pure functions on arrays. Depth works on log depth (a ratio is the error that matters at every distance).

Nothing here knows a model: a candidate that went wrong as a whole (a collapsed depth) is dropped by its disagreement
with all the others before any per-pixel combination.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .ensemble import MAD_TO_SIGMA, box_sum, wmedian

# What the depth ensemble may aim at (the node's 「目标」 parameter), each measured on its own (集成评测/一期报告.md).
DEPTH_OBJECTIVES = ("accuracy", "scale", "temporal")


def blur(a: np.ndarray, r: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Mean over the (2r+1)² window of the valid pixels (NaN counts as not valid); NaN where none is."""
    ok = np.isfinite(a) if valid is None else (valid & np.isfinite(a))
    num = box_sum(np.where(ok, a, 0.0), r)
    den = box_sum(ok.astype(np.float64), r)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def guided_lowpass(src: np.ndarray, guide: np.ndarray | None, r: int, eps: float = 1e-3) -> np.ndarray:
    """An edge-keeping low-pass of `src` (NaN = no value): the guided filter (He et al. 2010) steered by `guide` (a grey
    picture 0..1, same size), so the low band keeps the picture's edges; without a guide a plain box mean."""
    ok = np.isfinite(src)
    if guide is None:
        out = blur(src, r, ok)
        return np.where(ok, out, np.nan)
    g = np.asarray(guide, np.float64)
    den = box_sum(ok.astype(np.float64), r)
    safe = np.maximum(den, 1e-12)

    def mean(a):
        return box_sum(np.where(ok, a, 0.0), r) / safe

    p = np.where(ok, src, 0.0)
    mi, mp = mean(g), mean(p)
    var = mean(g * g) - mi * mi
    cov = mean(g * p) - mi * mp
    a = cov / (var + eps)
    b = mp - a * mi
    ma, mb = blur(a, r, den > 0), blur(b, r, den > 0)
    out = ma * g + mb
    return np.where(ok & np.isfinite(out), out, np.nan)


# ------------------------------------------------------------------ depth: log stacks and consensus


def log_depths(depths: Sequence[np.ndarray]) -> np.ndarray:
    """Candidates' depths [H,W] (cm, 0 or less = no value) as one stack of log depth [N,H,W], NaN where none."""
    z = np.stack([np.asarray(d, np.float64) for d in depths])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((z > 0) & np.isfinite(z), np.log(np.where(z > 0, z, 1.0)), np.nan)


@dataclass
class Consensus:
    """A per-pixel combination of log depths: the value, the spread among the candidates (log units, about a relative
    error), which candidates were kept at each pixel, and the index of the candidate nearest the value."""

    value: np.ndarray  # [H,W] log depth, NaN where nothing
    spread: np.ndarray  # [H,W] robust spread (σ from the weighted MAD)
    kept: np.ndarray  # [N,H,W] bool
    nearest: np.ndarray  # [H,W] int, -1 where nothing


def consensus(L: np.ndarray, w: np.ndarray | float = 1.0, k: float = 3.0, floor: float = 0.01,
              least: int = 1) -> Consensus:
    """Weighted median, spread s = 1.4826·weighted MAD, candidates further than k·max(s, floor) from the median
    dropped, the weighted mean of the rest. `least`: at least this many candidates with a value, else no value."""
    m = wmedian(L, w)
    dev = np.abs(L - m[None])
    s = MAD_TO_SIGMA * wmedian(dev, w)
    keep = np.isfinite(L) & (dev <= k * np.maximum(np.nan_to_num(s, nan=0.0), floor)[None])
    ww = np.where(keep, np.broadcast_to(np.asarray(w, np.float64), L.shape), 0.0)
    total = ww.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(total > 0, (ww * np.nan_to_num(L)).sum(0) / np.maximum(total, 1e-12), np.nan)
    enough = np.isfinite(L).sum(0) >= least
    value = np.where(enough, value, np.nan)
    near = np.where(np.isfinite(value), np.nanargmin(np.where(np.isfinite(L), np.abs(L - np.nan_to_num(value)[None]), np.inf), 0), -1)
    return Consensus(value, np.where(np.isfinite(value), s, np.nan), keep, near.astype(np.int64))


def centred(L: np.ndarray) -> np.ndarray:
    """Each candidate's log depth less its own median over the frame (its shape, whatever its overall scale)."""
    med = np.nanmedian(L.reshape(L.shape[0], -1), axis=1)
    return L - med[:, None, None]


def disagreement(L: np.ndarray) -> np.ndarray:
    """Per candidate [N]: the median |shape − the others' median shape| over the frame — how far it is, as a whole,
    from what the other candidates agree on (leave one out, so it never judges itself). NaN with fewer than three."""
    n = L.shape[0]
    if n < 3:
        return np.full(n, np.nan)
    c = centred(L)
    out = np.empty(n)
    for i in range(n):
        others = np.delete(c, i, 0)
        out[i] = np.nanmedian(np.abs(c[i] - np.nanmedian(others, 0)))
    return out


def outliers(score: np.ndarray, ratio: float = 2.0, least: float = 0.05, groups: Sequence[int] | None = None) -> np.ndarray:
    """Candidates [N] whose score (a disagreement: lower is better) is both above `least` and more than `ratio` times
    the median of the others' — dropped as a whole. Never with fewer than three independent candidates (groups:
    candidates of one group count once), and never all of them."""
    score = np.asarray(score, np.float64)
    n = len(score)
    if len(set(range(n) if groups is None else groups)) < 3 or not np.isfinite(score).any():
        return np.zeros(n, bool)
    out = np.zeros(n, bool)
    for i in range(n):
        others = np.delete(score, i)
        ref = np.nanmedian(others)
        out[i] = bool(np.isfinite(score[i]) and score[i] > least and score[i] > ratio * ref)
    if out.all():
        out[:] = False
    return out


# ------------------------------------------------------------------ depth: the multi-view judge


def reprojection_error(z_a: np.ndarray, z_b: np.ndarray, K_a: np.ndarray, K_b: np.ndarray, a_to_b: np.ndarray,
                       stride: int = 2) -> np.ndarray:
    """Multi-view geometric consistency of one candidate with itself (Lab2Shot's judge: independent of the other
    candidates). Frame a's depth (cm) is lifted to 3D through its camera, moved into frame b's camera by `a_to_b`
    (4×4, OpenCV axes: x right, y down, z forward) and compared with frame b's own depth where it lands: the relative
    difference, on a grid every `stride` pixels of frame a ([H/stride, W/stride], NaN where it leaves frame b or
    either has no value). Consistent depth agrees from both views; a depth that is wrong in a way the other view
    sees (a flattened step, a frame whose scale jumped) does not."""
    h, w = z_a.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    za = z_a[ys, xs].astype(np.float64)
    ok = np.isfinite(za) & (za > 0)
    pts = np.linalg.inv(K_a) @ np.stack([xs[ok] + 0.5, ys[ok] + 0.5, np.ones(ok.sum())]) * za[ok]
    q = a_to_b[:3, :3] @ pts + a_to_b[:3, 3:4]
    uv = K_b @ q
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = uv[0] / uv[2] - 0.5, uv[1] / uv[2] - 0.5
    ui, vi = np.round(u), np.round(v)
    hit = (q[2] > 0) & (ui >= 0) & (ui < z_b.shape[1]) & (vi >= 0) & (vi < z_b.shape[0])
    err = np.full(ok.sum(), np.nan)
    zb = z_b[vi[hit].astype(np.int64), ui[hit].astype(np.int64)]
    with np.errstate(divide="ignore", invalid="ignore"):
        err[hit] = np.where(zb > 0, np.abs(q[2][hit] - zb) / zb, np.nan)
    out = np.full(ys.shape, np.nan)
    out[ok] = err
    return out


# ------------------------------------------------------------------ depth: bands, anchor, scale


def split_bands(low_from: np.ndarray, high_from: np.ndarray, r: int, guide: np.ndarray | None = None) -> np.ndarray:
    """Low band of one log depth, high band of another: lowpass(low_from) + (high_from − lowpass(high_from)). The low
    band carries the overall scale and the slow shape, the high band the edges and the fine detail."""
    lo = guided_lowpass(low_from, guide, r)
    hi = high_from - guided_lowpass(high_from, guide, r)
    out = lo + hi
    return np.where(np.isfinite(out), out, np.where(np.isfinite(high_from), high_from, low_from))


def pick(scores: np.ndarray, allowed: np.ndarray) -> int:
    """The allowed candidate with the lowest score (−1 none)."""
    s = np.where(allowed & np.isfinite(scores), scores, np.inf)
    return int(np.argmin(s)) if np.isfinite(s).any() else -1


def scale_consensus(fits: Sequence[float | None], least: int = 2) -> float | None:
    """The factor that brings the reference's scale to the candidates' own: each metric candidate was multiplied by
    `fit` to match the reference (depth_align's scale), so 1/fit is how much bigger the candidate thinks the scene
    is; their median over the candidates that say. None with fewer than `least`."""
    got = [1.0 / f for f in fits if f is not None and np.isfinite(f) and f > 0]
    return float(np.exp(np.median(np.log(got)))) if len(got) >= least else None


def jitter(L0: np.ndarray, L1: np.ndarray) -> float:
    """How much a candidate's shape changes from one frame to the next: median |shape(t+1) − shape(t)| of its own
    log depth (each less its median). The camera's motion moves every candidate alike, so the candidates compare by
    it without a camera and without judging one by the others (a steady model is not penalised for not following
    the others' flicker). Lower is steadier."""
    d = np.abs(centred(L1[None])[0] - centred(L0[None])[0])
    return float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")


SPREAD_FLOOR = 0.01  # the relative error at which the confidence is 0.5 (the same mapping as UniK3D's log_error)


def depth_confidence(spread: np.ndarray, gain: float = 1.0, power: float = 1.0) -> np.ndarray:
    """0..1, higher = more trusted, from the candidates' spread (≈ relative error). The spread is calibrated against
    the truth first (expected error ≈ gain · spread^power, constants fitted on the tuning clips, CONF_GAIN / CONF_POWER),
    then mapped like a model's own log error: 0.01/(0.01 + e)."""
    e = gain * np.power(np.maximum(np.nan_to_num(spread, nan=1.0), 0.0), power)
    return SPREAD_FLOOR / (SPREAD_FLOOR + e)


def band_radius(shape: tuple[int, ...]) -> int:
    """The low band's radius: a 32nd of the picture's width (design 4.1)."""
    return max(2, int(round(shape[-1] / BAND_DIVISOR)))


# ------------------------------------------------------------------ constants (fitted on the tuning clips only;
# 集成评测/一期报告.md lists the values tried and what each gave)

DEPTH_K = 3.0  # per pixel: a candidate further than DEPTH_K·max(σ, DEPTH_FLOOR) from the median is left out there
DEPTH_FLOOR = 0.01
DEPTH_REJECT_RATIO = 2.0  # whole candidate: its disagreement over twice the others' median …
DEPTH_REJECT_LEAST = 0.05  # … and above 5 % (log units)
ANCHOR_AGREE = 0.1  # a time anchor's shape must agree with the consensus within this (median |Δ log depth|)
BAND_DIVISOR = 32
SCALE_WARN = 0.3  # the reference's scale off the candidates' own by more than this: warned
SCALE_APPLY = 0.2  # … and by more than this: the candidates' own scale is used
CONF_GAIN = 1.0  # expected relative error ≈ gain · spread^power (calibrated on the tuning clips)
CONF_POWER = 1.0
