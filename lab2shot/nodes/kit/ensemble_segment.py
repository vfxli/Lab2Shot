"""Object segmentation ensemble kernels: several masks of one object (segmentation models, binarised mattes) combined
into one.

The node (nodes/core/ensemble_segment.py) reads the candidates and writes what these functions return. numpy only.
A candidate is a 0..1 map (a mask, or an alpha that votes with its soft value); a label map (a segmentation with
object numbers) becomes one by `pick_objects`, the objects the user's selection covers.

Every objective checks the object first (kit/ensemble.py selection_check, identity_vote: a candidate that drew another
object is left out), and candidates that share a guide are one vote. Then:
- accuracy: the majority of the votes per pixel (`segment_vote` above one half);
- stable: as accuracy, with hysteresis: a pixel that was foreground stays so until its share drops below
  STABLE_LOW (fewer flips between frames);
- reliability: STAPLE (Warfield 2004): each vote's sensitivity and specificity estimated by EM, the posterior above
  one half (needs three votes; fewer: the majority).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

SEGMENT_OBJECTIVES = ("accuracy", "stable", "reliability")
STABLE_LOW = 0.3
PICK_INSIDE = 0.5  # an object of a label map is the user's when more than this share of it lies inside the selection


def pick_objects(labels: np.ndarray, selection: np.ndarray) -> list[int]:
    """Object numbers of a label map whose pixels lie more than PICK_INSIDE inside the selection; none: the one that
    overlaps it best (IoU); nothing overlaps: []."""
    labels = np.rint(np.asarray(labels)).astype(np.int64)
    sel = np.asarray(selection, bool)
    chosen, best, best_iou = [], None, 0.0
    for i in np.unique(labels):
        if i <= 0:
            continue
        m = labels == i
        both = (m & sel).sum()
        if both / max(m.sum(), 1) > PICK_INSIDE:
            chosen.append(int(i))
        iou = both / max((m | sel).sum(), 1)
        if iou > best_iou:
            best, best_iou = int(i), iou
    return chosen or ([best] if best is not None else [])


def segment_vote(A: np.ndarray, kept: np.ndarray, groups: Sequence[int], w: np.ndarray | None = None) -> np.ndarray:
    """The foreground share [H,W] (0..1): each kept group's members' weighted mean, the groups averaged with their
    mean weights (one guide = one vote). 0 where no group is kept."""
    A = np.asarray(A, np.float64)
    kept = np.asarray(kept, bool)
    w = np.ones(len(A)) if w is None else np.asarray(w, np.float64)
    g = np.asarray(groups)
    num = np.zeros(A.shape[1:])
    den = 0.0
    for k in dict.fromkeys(groups):
        m = (g == k) & kept & (w > 0)
        if not m.any():
            continue
        gw = w[m].mean()
        num += gw * np.tensordot(w[m] / w[m].sum(), A[m], axes=1)
        den += gw
    return num / den if den > 0 else num


def group_votes(A: np.ndarray, kept: np.ndarray, groups: Sequence[int], w: np.ndarray | None = None) -> np.ndarray:
    """Each kept group's soft vote [G,H,W] (its members' weighted mean)."""
    A = np.asarray(A, np.float64)
    w = np.ones(len(A)) if w is None else np.asarray(w, np.float64)
    g = np.asarray(groups)
    out = []
    for k in dict.fromkeys(groups):
        m = (g == k) & np.asarray(kept, bool) & (w > 0)
        if m.any():
            out.append(np.tensordot(w[m] / w[m].sum(), A[m], axes=1))
    return np.stack(out) if out else np.zeros((0,) + A.shape[1:])


def staple(V: np.ndarray, iters: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STAPLE on binary votes V [G,H,W]: (posterior foreground probability [H,W], sensitivities [G], specificities [G])."""
    V = np.asarray(V, bool)
    G = len(V)
    f = float(np.clip(V.mean(), 1e-4, 1 - 1e-4))
    p, q = np.full(G, 0.9), np.full(G, 0.99)
    W = V.mean(0).astype(np.float64)
    for _ in range(iters):
        la = np.log(f) + sum(np.where(V[j], np.log(p[j]), np.log1p(-p[j])) for j in range(G))
        lb = np.log1p(-f) + sum(np.where(V[j], np.log1p(-q[j]), np.log(q[j])) for j in range(G))
        W = 1.0 / (1.0 + np.exp(np.clip(lb - la, -50, 50)))
        sw, sn = W.sum(), (1 - W).sum()
        p = np.clip(np.array([(W * V[j]).sum() / max(sw, 1e-9) for j in range(G)]), 1e-3, 1 - 1e-3)
        q = np.clip(np.array([((1 - W) * ~V[j]).sum() / max(sn, 1e-9) for j in range(G)]), 1e-3, 1 - 1e-3)
    return W, p, q


def segment_decide(share: np.ndarray, prior: np.ndarray | None = None, low: float | None = None) -> np.ndarray:
    """Foreground where the share is above one half; exactly one half (a tie) goes to `prior` (the frame before) when
    given. With `low` and a prior: a pixel that was foreground stays so while its share is above `low`."""
    fg = share > 0.5 + 1e-9
    if prior is not None:
        prior = np.asarray(prior, bool)
        fg |= (np.abs(share - 0.5) <= 1e-9) & prior
        if low is not None:
            fg |= prior & (share > low)
    return fg


def segment_share(A: np.ndarray, kept: np.ndarray, groups: Sequence[int], w: np.ndarray, objective: str) -> np.ndarray:
    """The foreground share [H,W] the objective decides on (see the module)."""
    if objective == "reliability":
        V = group_votes(A, kept, groups, w)
        if len(V) >= 3:
            return staple(V > 0.5)[0]
    return segment_vote(A, kept, groups, w)


def segment_confidence(A: np.ndarray, kept: np.ndarray, result: np.ndarray) -> np.ndarray:
    """0..1: the share of kept candidates (cut at one half) that agree with the result there."""
    kept = np.asarray(kept, bool)
    if not kept.any():
        return np.zeros(result.shape, np.float32)
    agree = ((np.asarray(A)[kept] > 0.5) == np.asarray(result, bool)[None]).mean(0)
    return agree.astype(np.float32)
