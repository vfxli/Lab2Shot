"""The depth ensemble: depth maps of several models of one shot, aligned to one reference, combined in log depth.

One node per kind of data (nodes/core/ensemble.py: what every ensemble shares, _Ensemble); the kernel is
nodes/kit/ensemble_depth.py. The node only combines: aligning every candidate to the base camera is the card's work
(depth_align), done before; candidates aligned to different references or of different scale meanings are refused.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ... import i18n
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..expects import SameReference
from ..kit import ensemble as E
from ..kit import ensemble_depth as ED
from .ensemble import CANDIDATES, _Ensemble

PRE_FRAMES = 16  # frames looked at before combining (whole-candidate check, time anchor), evenly spread


class DepthCandidate(NodeParams):
    """One row of 「候选」 = one input: its port (stable: wires and graph files name it), its prior weight and what it is
    used for (shape: combined pixel by pixel; anchor: only as the time anchor of the temporal objective, a depth whose
    overall level is steady but whose edges are soft). The display name last (the table's text column)."""

    name: Literal[CANDIDATES] = P(..., widget="fixed")
    weight: float = P(1.0, ge=0.0)
    use: Literal["shape", "anchor"] = P("shape")
    label: dict[str, str] | str = P("")


class EnsembleDepth(_Ensemble, NodeDef):
    """Depth maps of several models, aligned to one reference (one camera space), combined in log depth.

    Objectives, each measured on its own against ground truth (13 clips with true depth, D02/D03 failed for every
    model and are not counted: 6 tuning, 5 held out; parameters set on the tuning clips only, the held-out clips run
    once; 集成评测/一期报告.md, re-run on this kernel in 集成评测/实测/depth with the same numbers). None passed the
    acceptance rule (held-out median ≥ 5 % better than both the tuning clips' best single model and the held-out
    clips' best one, no held-out clip more than 5 % worse than either), so no card uses this node yet:
    - accuracy (shape AbsRel): the shape candidates' weighted consensus per pixel, outliers dropped per pixel and, when
      one candidate disagrees with all the others as a whole, for the whole shot. Held-out median 0.0525: 15.2 %
      better than MoGe (best on the tuning clips), 7.6 % better than UniK3D (best held out), no clip 5 % worse than
      UniK3D, but D01 and D04 8 % and 14 % worse than MoGe. The multi-view judge (kit/ensemble_depth.py
      reprojection_error) was measured too and made the tuning clips worse: it is not used.
    - scale (delivered AbsRel): as accuracy, then the whole result brought to the candidates' own metric scale where
      the reference's is more than SCALE_APPLY off it (the reference's scale is the error every aligned candidate
      shares). Held-out median level with UniK3D (0.1468 / 0.1467), F06 worse than UniK3D and D04 than MoGe.
    - temporal (TAE): the low band from the steadiest candidate that agrees with the consensus (the time anchor,
      usually ViPE), the high band from the consensus. 33 % less steady than ViPE alone on the held-out median.
    Edges (a high band from the candidate whose edges sit best on the picture) and robust (the unweighted median)
    were measured and gave nothing over accuracy: not offered. Leaning on the single best model where the candidates
    disagree (MoGe trusted) kept every tuning clip within 5 % of MoGe but made the held-out clips worse (median shape
    error 0.0525 → 0.0600, 一期b报告.md): not used.
    """

    id = "ensemble_depth"
    category = "geo_maps"
    data_kind = "depth"
    result_port = "depth"
    main = "depth"
    on_node = ("objective",)
    inputs = ()
    outputs = (
        Port("depth", "image.1", means=("scale",)),
        Port("confidence", "image.1"),
        Port("choice", "image.1"),
        Port("scores", "curves"),
    )

    class Params(NodeParams):
        objective: Literal[ED.DEPTH_OBJECTIVES] = P("accuracy", group="ensemble")
        candidates: list[DepthCandidate] = P(
            [{"name": n, "weight": 1.0, "use": "shape", "label": ""} for n in CANDIDATES[:2]], widget="table",
            group="ensemble", max_length=len(CANDIDATES), validate_default=True,
        )

    @classmethod
    def candidate_expects(cls) -> tuple:
        return (SameReference(),)

    @staticmethod
    def sampled(frames: list[int], most: int = PRE_FRAMES) -> list[int]:
        return [frames[i] for i in np.unique(np.linspace(0, len(frames) - 1, min(len(frames), most)).astype(int))]

    @classmethod
    def cook(cls, ctx):
        from ...data.maps import map_at, same_size
        from ...data.payloads import window_of

        cands = cls.candidates(ctx)
        cls.identify(cands)
        same_size({c["title"]: c["packet"] for c in cands})
        cls._check_space(cands)
        commercial = cls.say_licence(ctx, cands)
        frames = cls.common_frames(cands)
        objective = ctx.params["objective"]
        shape = [k for k, c in enumerate(cands) if c["row"].get("use", "shape") == "shape"]
        if not shape:
            raise Invalid(Msg("E-ENSEMBLE-NOSHAPE"))
        prior = np.array([float(c["row"].get("weight", 1.0)) for c in cands])
        first = cands[0]["packet"]
        window = window_of(first)

        def log_at(k: int, f: int) -> np.ndarray:
            got = map_at(cands[k]["packet"], f)
            z = got[0][..., 0].astype(np.float64)
            return ED.log_depths([np.where(got[1] > 0, z, 0.0)])[0]

        ctx.stage("judge_candidates")
        look = cls.sampled(frames)
        stacks = {f: np.stack([log_at(k, f) for k in shape]) for f in look}
        dis = np.nanmedian(np.stack([ED.disagreement(stacks[f]) for f in look]), 0) if len(shape) >= 3 else np.full(len(shape), np.nan)
        dropped = ED.outliers(dis, ED.DEPTH_REJECT_RATIO, ED.DEPTH_REJECT_LEAST)
        for k, d in zip(shape, dropped):
            if d:
                ctx.say("N-ENSEMBLE-DROPPED", model=cands[k]["title"])
        kept = [k for k, d in zip(shape, dropped) if not d]

        # the candidates' own weights only: weighting them by the multi-view judge made the tuning clips worse
        # (一期报告.md, 深度·准确), so the node does not
        weights = prior

        anchor, scale_factor = -1, 1.0
        if objective == "temporal":
            anchor = cls._anchor(ctx, cands, kept, look, stacks, shape, log_at, weights)
        if objective == "scale":
            scale_factor = cls._scale(ctx, cands, kept)

        name = E.ensemble_name("depth", objective, [cands[k]["token"] for k in sorted(set(kept) | {anchor} - {-1})])
        summary = {"objective": objective, "candidates": [c["title"] for c in cands], "kept": [cands[k]["title"] for k in kept],
                   "dropped": [cands[k]["title"] for k, d in zip(shape, dropped) if d], "commercial": commercial,
                   "anchor": cands[anchor]["title"] if anchor >= 0 else None, "scale_factor": scale_factor}
        meta = {"scale": first.meta.get("scale")}
        if first.meta.get("aligned"):
            meta["aligned"] = first.meta["aligned"]
        w = cls.writers(ctx, cands, window, name, summary, **meta)
        n = len(cands)
        curve = np.full((len(frames), n), np.nan)

        def one(j_f):
            j, f = j_f
            L = np.stack([log_at(k, f) for k in kept])
            c = ED.consensus(L, weights[kept][:, None, None], ED.DEPTH_K, ED.DEPTH_FLOOR)
            value, near = c.value, np.where(c.nearest >= 0, np.array(kept)[np.maximum(c.nearest, 0)], -1)
            if objective == "temporal" and anchor >= 0:
                value = ED.split_bands(log_at(anchor, f), value, ED.band_radius(value.shape))
            elif objective == "scale":
                value = value + np.log(scale_factor)
            ok = np.isfinite(value)
            w["result"].add(f, np.where(ok, np.exp(np.where(ok, value, 0.0)), 0.0).astype(np.float32), ok)
            if "confidence" in w:
                w["confidence"].add(f, ED.depth_confidence(c.spread, ED.CONF_GAIN, ED.CONF_POWER).astype(np.float32), ok)
            if "choice" in w:
                w["choice"].add(f, (near + 1).astype(np.float32))
            row = np.zeros(n)
            row[kept] = c.kept.reshape(len(kept), -1).mean(1)
            return row

        ctx.stage("combine")
        for j, row in enumerate(ctx.each_done(list(enumerate(frames)), one)):
            curve[j] = row
        ctx.say("I-ENSEMBLE-RESULT", objective=cls.option_label("objective", objective), count=len(kept),
                models=i18n.Both.of(lambda: i18n.separator().join(cands[k]["title"] for k in kept)), name=name)
        names = [f"{c['token']}_used" for c in cands]
        return cls.finish(ctx, w, frames, names, curve, summary)

    # ------------------------------------------------------------------ depth: helpers

    @staticmethod
    def _check_space(cands: list[dict]) -> None:
        """One camera space: one scale meaning, and one reference: every aligned candidate aligned to the same one, and
        an unaligned candidate only when it is that reference itself (the time anchor, ViPE's depth) or when none is
        aligned."""
        scales = {c["packet"].meta.get("scale") for c in cands}
        refs = {(c["packet"].meta.get("aligned") or {}).get("reference") for c in cands} - {None}
        loose = [c for c in cands if not (c["packet"].meta.get("aligned") or {}).get("reference")]
        if len(scales) > 1 or len(refs) > 1 or (refs and any(c["packet"].fingerprint not in refs for c in loose)):
            raise Invalid(Msg("E-ENSEMBLE-UNALIGNED"))

    @classmethod
    def _anchor(cls, ctx, cands, kept, look, stacks, shape, log_at, weights) -> int:
        """The time anchor: of the candidates whose shape agrees with the consensus, the one whose overall level jumps
        least from frame to frame against it."""
        ref = {f: ED.consensus(stacks[f][[shape.index(k) for k in kept]], 1.0, ED.DEPTH_K, ED.DEPTH_FLOOR).value for f in look}
        n = len(cands)
        agree, steady = np.full(n, np.nan), np.full(n, np.nan)
        for k in range(n):
            Ls = np.stack([log_at(k, f) for f in look])
            R = np.stack([ref[f] for f in look])
            agree[k] = float(np.nanmedian([np.nanmedian(np.abs(ED.centred(a[None])[0] - ED.centred(b[None])[0])) for a, b in zip(Ls, R)]))
            steady[k] = float(np.nanmedian([ED.jitter(log_at(k, f), log_at(k, g)) for f, g in zip(look, look[1:])]))
        ok = np.isfinite(agree) & (agree <= ED.ANCHOR_AGREE)
        best = ED.pick(steady, ok)
        if best < 0:
            ctx.say("N-ENSEMBLE-NOANCHOR")
        else:
            ctx.say("I-ENSEMBLE-ANCHOR", model=cands[best]["title"])
        return best

    @classmethod
    def _scale(cls, ctx, cands, kept) -> float:
        """The factor from the reference's scale to the candidates' own (metric candidates aligned by depth_align)."""
        fits = [(c["packet"].meta.get("aligned") or {}) for c in (cands[k] for k in kept)]
        factor = ED.scale_consensus([a.get("scale") for a in fits if a.get("source_scale") == "metric" and a.get("model") == "scale"])
        if factor is None:
            ctx.say("N-ENSEMBLE-NOSCALE")
            return 1.0
        off = abs(np.log(factor))
        if off > np.log(1 + ED.SCALE_WARN):
            ctx.say("W-ENSEMBLE-SCALE", percent=f"{(factor - 1) * 100:+.0f}")
        return factor if off > np.log(1 + ED.SCALE_APPLY) else 1.0



NODES = (EnsembleDepth,)
