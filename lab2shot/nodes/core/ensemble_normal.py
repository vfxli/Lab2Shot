"""Normal ensemble: normal maps of several models of one shot, in one camera space, combined on the unit sphere.

Candidates: normals a model gives itself (MoGe), normals of a model that knows only people (Sapiens2: its row's
「区域」 is 「人体」, used only inside the 「人体区域」 input), normals computed from a depth model's depth
(normal_from_depth, which takes the card's base camera). The node does not move anything between spaces: every
candidate must say the same space (camera or world); bringing them into one camera is the card's work.

Which model each candidate is comes from the graph (lineage, _Ensemble.identify with derived=True): the node the
wire comes from, else the nearest node above it that the other derived candidates do not share (a ViPE camera every depth model took its focal length
from is nobody's own), else the nearest one. Candidates of one model (MoGe's normals and the normals of MoGe's depth)
are one vote.

Outputs as every ensemble (nodes/core/ensemble.py): the result, a confidence map, which candidate each pixel is
closest to (label map) and per-frame scores (each candidate's median angle to the result).

Measured (集成评测/实测/normal/summary.txt; 12 clips with ground truth, 7 tuning, 5 held out): neither objective passed
the strict rule (held-out median angular error 1.1 % / 1.7 % better than MoGe, D04 16.8 % / 5.3 % worse), so no card
uses this node.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ... import i18n
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..kit import ensemble as E
from ..kit import ensemble_normal as EN
from ..kit.ports import normal_port
from .ensemble import CANDIDATES, _Ensemble

NORMAL_THREADS = 4  # frames combined at once: a 1080p frame of five candidates holds about 1 GB while it is combined


class NormalCandidate(NodeParams):
    """One row of 「候选」 = one input: its port, its prior weight, where it counts (everywhere, or only inside the
    「人体区域」 input), its display name."""

    name: Literal[CANDIDATES] = P(..., widget="fixed")
    weight: float = P(1.0, ge=0.0)
    region: Literal["everywhere", "people"] = P("everywhere")
    label: dict[str, str] | str = P("")


class EnsembleNormal(_Ensemble, NodeDef):
    """Normals of several models combined per pixel on the unit sphere (kit/ensemble_normal.py).

    - accuracy: the spherical median, candidates further than three spreads (at least 5°) from it left out there,
      the weighted mean direction of the rest;
    - robust: the spherical median itself.
    """

    id = "ensemble_normal"
    category = "geo_maps"
    data_kind = "normal"
    result_port = "normal"
    main = "normal"
    on_node = ("objective",)
    ports_from_type = "image.3"
    inputs = (Port("people", "image.1", optional=True),)
    outputs = (
        normal_port(),
        Port("confidence", "image.1"),
        Port("choice", "image.1"),
        Port("scores", "curves"),
    )

    class Params(NodeParams):
        objective: Literal[EN.NORMAL_OBJECTIVES] = P("accuracy", group="ensemble")
        candidates: list[NormalCandidate] = P(
            [{"name": n, "weight": 1.0, "region": "everywhere", "label": ""} for n in CANDIDATES[:2]], widget="table",
            group="ensemble", max_length=len(CANDIDATES), validate_default=True,
        )

    @classmethod
    def cook(cls, ctx):
        from ...data.maps import map_at, same_size
        from ...data.payloads import SIGNED, window_of

        cands = cls.candidates(ctx)
        cls.identify(cands, derived=True)
        same_size({c["title"]: c["packet"] for c in cands})
        spaces = {c["packet"].meta.get("space", "camera") for c in cands}
        if len(spaces) > 1:
            raise Invalid(Msg("E-ENSEMBLE-NORMALSPACE"))
        space = spaces.pop()
        commercial = cls.say_licence(ctx, cands)
        frames = cls.common_frames(cands)
        objective = ctx.params["objective"]
        groups = cls.model_groups(cands)
        people = ctx.input("people")
        regional = [c["row"].get("region") == "people" for c in cands]
        if any(regional) and people is None:
            for c, r in zip(cands, regional):
                if r:
                    ctx.say("N-ENSEMBLE-NOPEOPLE", model=c["title"])
        prior = EN.vote_weights([float(c["row"].get("weight", 1.0)) for c in cands], groups)
        window = window_of(cands[0]["packet"])
        name = E.ensemble_name("normal", objective, [c["token"] for c in cands])
        summary = {"objective": objective, "candidates": [c["title"] for c in cands], "groups": groups,
                   "commercial": commercial}
        out = cls.writers(ctx, cands, window, name, summary, channels=3, value_range=SIGNED, half=True, space=space)
        n = len(cands)
        curve = np.full((len(frames), n), np.nan)
        ctx.stage("combine")

        def combine(job):
            j, f = job
            N, Wt = [], []
            region = None
            if people is not None and any(regional):
                got = map_at(people, f)
                region = (got[0][..., 0] > 0.5) & (got[1] > 0) if got is not None else None
            for k, c in enumerate(cands):
                v, ok = map_at(c["packet"], f)
                N.append(EN.unit_normals(v[..., :3], ok > 0))
                w = np.full(ok.shape, prior[k])
                if regional[k]:
                    w = w * region if region is not None else np.zeros(ok.shape)
                Wt.append(w)
            N, Wt = np.stack(N), np.stack(Wt)
            got = EN.normal_consensus(N, Wt, objective)
            has = np.isfinite(got.value).all(-1)
            out["result"].add(f, np.nan_to_num(got.value).astype(np.float32), has)
            if "confidence" in out:
                out["confidence"].add(f, EN.normal_confidence(got.spread).astype(np.float32), has)
            if "choice" in out:
                out["choice"].add(f, (got.nearest + 1).astype(np.float32))
            for k in range(n):
                curve[j, k] = EN.median_angle(np.where((Wt[k] > 0)[..., None], N[k], np.nan), got.value)

        list(ctx.each_done(list(enumerate(frames)), combine, most=NORMAL_THREADS))
        ctx.say("I-ENSEMBLE-RESULT", objective=cls.option_label("objective", objective), count=n,
                models=i18n.Both.of(lambda: i18n.separator().join(c["title"] for c in cands)), name=name)
        names = [f"{c['token']}_deg" for c in cands]
        return cls.finish(ctx, out, frames, names, curve, summary)


NODES = (EnsembleNormal,)
