"""Object segmentation ensemble: masks of one object from several models combined into one hard mask.

Candidates: masks (SAM 3's point-picked mask), label maps (VidEoMT's panoptic segments: the objects the user's
selection covers on its first frame are taken, kit/ensemble_segment.py pick_objects) and mattes (an alpha votes with
its soft value). The user's selection (「对象」, the card's SAM 3 point-picked mask) is the judge of which object it
is: a candidate drawing more or less than it is left out for the shot (kit/ensemble.py selection_check), and per
frame candidates that disagree on which object it is are settled by the majority of the guide groups
(identity_vote). Candidates sharing a guide (SAM 3's mask and MatAnyone 2 on it; BiRefNet and VideoMaMa on its rough
mask) are one vote (nodes/core/ensemble.py guide_groups).

Outputs as every ensemble: the mask, a confidence map (the share of kept candidates agreeing), which candidate each
pixel is closest to and per-frame scores (each candidate's IoU with the result, whether it was kept).

Measured (集成评测/实测/segment, verdict.txt; the card's four candidates and weights, current SAM 3.1 / VidEoMT code; 12
clips: DAVIS, MOSE, SegTrackV2, FBMS and 8 matting clips with alpha > 0.5 as truth, 6 tuning / 6 held out): no
objective passed the strict rule. Held-out median IoU 0.988 / 0.985 / 0.988 (accuracy / stable / reliability) against
MatAnyone 2 0.990 (best on the tuning clips) and BiRefNet 0.992 (best on the held-out ones): flat, no clip more than
5 % worse and no wrong object (BiRefNet took the wrong object on 2 held-out clips, MatAnyone 2 on none), so it neither
beats both by 5 % nor stops a failure of the tuning-best model. No card uses this node.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ... import i18n
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..kit import ensemble as E
from ..kit import ensemble_segment as ES
from .ensemble import CANDIDATES, _Ensemble


class SegmentCandidate(NodeParams):
    """One row of 「候选」 = one input: its port, its prior weight, its display name."""

    name: Literal[CANDIDATES] = P(..., widget="fixed")
    weight: float = P(1.0, ge=0.0)
    label: dict[str, str] | str = P("")


class EnsembleSegment(_Ensemble, NodeDef):
    """Masks of one object from several models, the object checked first, then voted per pixel
    (kit/ensemble_segment.py): accuracy = majority; stable = majority with hysteresis; reliability = STAPLE."""

    id = "ensemble_segment"
    category = "mask"
    data_kind = "segment"
    result_port = "mask"
    main = "mask"
    on_node = ("objective",)
    inputs = (Port("object", "image.1", optional=True),)
    outputs = (
        Port("mask", "image.1"),
        Port("confidence", "image.1"),
        Port("choice", "image.1"),
        Port("scores", "curves"),
    )

    class Params(NodeParams):
        objective: Literal[ES.SEGMENT_OBJECTIVES] = P("accuracy", group="ensemble")
        candidates: list[SegmentCandidate] = P(
            [{"name": n, "weight": 1.0, "label": ""} for n in CANDIDATES[:2]], widget="table",
            group="ensemble", max_length=len(CANDIDATES), validate_default=True,
        )

    @classmethod
    def cook(cls, ctx):
        from ...data.maps import map_at, same_size
        from ...data.payloads import UNIT, window_of

        cands = cls.candidates(ctx)
        cls.identify(cands, derived=True)
        same_size({c["title"]: c["packet"] for c in cands})
        commercial = cls.say_licence(ctx, cands)
        frames = cls.common_frames(cands)
        objective = ctx.params["objective"]
        groups = cls.guide_groups(cands)
        if len(set(groups)) < 2:
            ctx.say("N-ENSEMBLE-SHAREDGUIDE")
        n = len(cands)
        prior_w = np.array([float(c["row"].get("weight", 1.0)) for c in cands])
        labelled = [bool(c["packet"].meta.get("classes")) for c in cands]
        mask = ctx.input("object")

        def raw(k, f):
            v, ok = map_at(cands[k]["packet"], f)
            return np.where(ok > 0, v[..., 0], 0.0)

        # the user's selection on the first frame that has it; without one, the masks' majority on the first frame
        f0, selection = None, None
        if mask is not None:
            for f in frames:
                got = map_at(mask, f)
                if got is not None and (got[0][..., 0] > 0.5).any():
                    f0, selection = f, got[0][..., 0] > 0.5
                    break
        if selection is None:
            plain = [k for k in range(n) if not labelled[k]]
            if any(labelled) and not plain:
                raise Invalid(Msg("E-ENSEMBLE-NOSELECTION"))
            f0 = frames[0]
            if plain:
                selection = np.mean([np.clip(raw(k, f0), 0, 1) for k in plain], 0) > 0.5
        picked = {}
        for k in range(n):
            if labelled[k]:
                picked[k] = ES.pick_objects(raw(k, f0), selection)
                if not picked[k]:
                    ctx.say("N-ENSEMBLE-NOOBJECT", model=cands[k]["title"])

        def stack_at(f):
            out = []
            for k in range(n):
                v = raw(k, f)
                out.append(np.isin(np.rint(v).astype(np.int64), picked[k]).astype(np.float32) if labelled[k]
                           else np.clip(v, 0.0, 1.0).astype(np.float32))
            return np.stack(out)

        name = E.ensemble_name("segment", objective, [c["token"] for c in cands])
        summary = {"objective": objective, "candidates": [c["title"] for c in cands], "groups": groups,
                   "commercial": commercial}
        window = window_of(cands[0]["packet"])
        w = cls.writers(ctx, cands, window, name, summary, validity=False, value_range=UNIT, half=True)
        shot_ok, rank_of = np.ones(n, bool), None
        if selection is not None and mask is not None:
            A0, chosen = stack_at(f0), selection.astype(np.float32)
            shot_ok = E.selection_check(A0, chosen, E.SELECTION_MARGIN)
            rank_of = E.selection_fit(A0, chosen)
            for k in np.flatnonzero(~shot_ok):
                ctx.say("N-ENSEMBLE-NOTSELECTED", model=cands[k]["title"])
        low = ES.STABLE_LOW if objective == "stable" else None
        curve = np.zeros((len(frames), 2 * n))
        dropped = np.zeros(n, int)
        previous = None
        ctx.stage("combine")
        for j, f in enumerate(ctx.each(frames)):  # in order: each frame's object check leans on the frame before
            A = stack_at(f)
            sel = shot_ok if shot_ok.any() else np.ones(n, bool)
            voters = [g for g, o in zip(groups, sel) if o]
            order, reps = E.group_means(A[sel], voters)
            rank = None if rank_of is None else np.array([max(rank_of[k] for k in range(n) if sel[k] and groups[k] == g) for g in order])
            said = map_at(mask, f) if mask is not None else None
            judge = (said[0][..., 0] > 0.5).astype(np.float32) if said is not None else \
                (previous.astype(np.float32) if previous is not None else None)
            trusted = E.identity_vote(reps, judge, E.MATTE_AGREE, rank)
            kept = np.array([g in order and trusted[order.index(g)] for g in groups]) & sel
            share = ES.segment_share(A, kept, groups, prior_w, objective)
            result = ES.segment_decide(share, previous, low)
            previous = result
            dropped += ~kept
            w["result"].add(f, result.astype(np.float32))
            if "confidence" in w:
                w["confidence"].add(f, ES.segment_confidence(A, kept, result))
            if "choice" in w:
                dist = np.where(kept[:, None, None], np.abs(A - result[None]), np.inf)
                w["choice"].add(f, np.where(np.isfinite(dist.min(0)), dist.argmin(0) + 1, 0).astype(np.float32))
            curve[j, :n] = [E.binary_iou(A[k], result.astype(np.float32)) for k in range(n)]
            curve[j, n:] = kept
        for k in range(n):
            if dropped[k] and shot_ok[k]:
                ctx.say("N-ENSEMBLE-OTHEROBJECT", model=cands[k]["title"], count=int(dropped[k]))
        ctx.say("I-ENSEMBLE-RESULT", objective=cls.option_label("objective", objective), count=n,
                models=i18n.Both.of(lambda: i18n.separator().join(c["title"] for c in cands)), name=name)
        names = [f"{c['token']}_iou" for c in cands] + [f"{c['token']}_kept" for c in cands]
        return cls.finish(ctx, w, frames, names, curve, summary)


NODES = (EnsembleSegment,)
