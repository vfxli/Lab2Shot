"""Ensembles: several results of one kind of data, already in one camera space, combined into one.

One node per kind of data (the user's rule: two kinds of data never share a node); within one kind, the 「目标」
parameter picks what the combination aims at, each its own strategy (kit/ensemble.py) measured on its own against
ground truth (集成评测/一期报告.md and the measurements each module names). Every objective stays offered, measured or
not passed alike; each node's description says what its objectives gave.

Here: what every ensemble shares (_Ensemble) and ensemble_matte; the other kinds each have their own module
(ensemble_camera.py, ensemble_depth.py, ensemble_normal.py, ensemble_segment.py). All five are registered; only
ensemble_matte's robust objective passed the strict rule, and only it is on a card (2D 抠像（模型集成）).

The nodes only combine. Bringing the candidates into one camera space is the card's work (camera_space), done before:
a camera ensemble refuses candidates moved into different base cameras' worlds.

Every ensemble gives the same four things: the combined result, its confidence (a 0..1 map, the 置信度 convention of the
models' own; per frame for a camera), which candidate each pixel came from (a label map with the candidates as its
classes; per frame for a camera) and per-frame per-candidate scores (curves). The result is named by one rule from the models and the objective
(kit/ensemble.py ensemble_name: its EXR layer when the output row leaves the name empty), and it carries the strictest
licence of its candidates (W-ENSEMBLE-NONCOMMERCIAL says which ones).

Which model each candidate is, which candidates share a guide (two matting models fed by one rough mask vote once) and
their licences come from the graph, per wire (NodeDef.reads_lineage, CookContext.lineage): never from a list of model
names here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ... import i18n
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..kit import ensemble as E

CANDIDATES = tuple(f"candidate{i}" for i in range(1, 9))


class MatteCandidate(NodeParams):
    """One row of 「候选」 = one input: its port, its prior weight, what it is used for (blend: averaged; trusted:
    averaged too, and, for the edges objective, the one the result leans on where it and the others disagree), its
    display name."""

    name: Literal[CANDIDATES] = P(..., widget="fixed")
    weight: float = P(1.0, ge=0.0)
    use: Literal["blend", "trusted"] = P("blend")
    label: dict[str, str] | str = P("")


class _Ensemble:
    """(Mixed into each ensemble node, ahead of NodeDef.) What every ensemble shares: one input per row of 「候选」, the
    candidates' identities and licences from the graph, the four outputs and their names."""

    ports_from = "candidates"
    ports_from_side = "inputs"
    ports_from_type = "image.1"
    ports_from_names = CANDIDATES
    picture = "candidate1"
    reads_lineage = True
    data_kind = ""  # "matte" / "camera": the first word of the result's name
    candidate_data = True  # a candidate input carries values, never a picture (a camera's: neither)

    @classmethod
    def made_ports(cls, params: dict) -> tuple[Port, ...]:
        """One input per candidate, named by the row's label in the language now."""
        rows = [{**r, "label": i18n.pick(r.get("label"))} if isinstance(r, dict) else r for r in params.get("candidates") or ()]
        ports = super().made_ports({**params, "candidates": rows})
        return tuple(replace(p, alpha=False, expects=cls.candidate_expects(), **({"data": True} if cls.candidate_data else {}))
                     for p in ports)

    @classmethod
    def candidate_expects(cls) -> tuple:
        return ()

    # ------------------------------------------------------------------ candidates and who they are

    @classmethod
    def candidates(cls, ctx) -> list[dict]:
        """The wired rows, in row order: {row, packet, wire (its lineage, CookContext.lineage: the source node's id
        and label, what made it as delivered), sources (every third-party project above it and itself that gave
        something, upstream first: models and classical solvers alike), learned (those of them that run a model),
        commercial (all of its sources allow it: the wire's own, the strictest licence)}."""
        out = []
        for row in ctx.params.get("candidates") or ():
            packet = ctx.input(row["name"])
            if packet is None:
                continue
            wire = (ctx.lineage.get(row["name"]) or [{}])[0]
            sources = list(wire.get("sources", ()))
            out.append({"row": row, "packet": packet, "wire": wire, "sources": sources,
                        "learned": [s for s in sources if s.get("learned")],
                        "commercial": bool(wire.get("commercial", all(s.get("commercial", True) for s in sources)))})
        if len(out) < 2:
            raise Invalid(Msg("E-ENSEMBLE-FEW", count=len(out)))
        return out

    @staticmethod
    def identify(cands: list[dict], derived: bool = False) -> None:
        """Each candidate's own model (`models`), its name for the result (`token`: those projects' extensions) and its
        display name (`title`), from the third-party projects above its wire (`sources`: whether or not they run a
        model, so a classical solver is a candidate like any other). Its own: the node the wire comes from when it is
        one of them (by node id); else the ones above it that are not everyone's — a reference every candidate was
        aligned to is nobody's — or, with `derived` (a kind where some candidates are computed from another's result:
        normals from a depth), the nearest of those not common to the derived candidates; else the nearest above it."""
        def direct(c: dict) -> list[dict]:
            return [s for s in c["sources"] if s.get("id") is not None and s.get("id") == c["wire"].get("id")]

        if derived:
            pool = [c for c in cands if not direct(c)]
            common = set.intersection(*({s["node"] for s in c["sources"]} for c in pool)) if len(pool) > 1 else set()

            def others(c: dict) -> list[dict]:
                return [s for s in c["sources"] if s["node"] not in common][-1:]
        else:
            shared = set.intersection(*({s.get("id") for s in c["sources"]} for c in cands)) if cands else set()

            def others(c: dict) -> list[dict]:
                return [s for s in c["sources"] if s.get("id") not in shared]
        for c in cands:
            own = direct(c) or others(c) or c["sources"][-1:]
            c["models"] = own
            c["token"] = "_".join(dict.fromkeys(s["node"].split(".")[0] for s in own)) or c["wire"].get("label") or c["row"]["name"]
            c["title"] = i18n.pick(c["row"].get("label")) or " + ".join(dict.fromkeys(s["project"] for s in own)) \
                or c["wire"].get("label") or c["row"]["name"]

    @staticmethod
    def model_groups(cands: list[dict]) -> list[int]:
        """Candidates whose own model (identify) is the same node type are one group (one vote)."""
        keys = [frozenset(s["node"] for s in c["models"]) or frozenset([c["token"]]) for c in cands]
        order: list[frozenset] = []
        out = []
        for k in keys:
            hit = next((i for i, o in enumerate(order) if o & k), None)
            if hit is None:
                order.append(k)
                hit = len(order) - 1
            out.append(hit)
        return out

    @staticmethod
    def guide_groups(cands: list[dict]) -> list[int]:
        """Candidates that share a learned node type anywhere above them (VideoMaMa and SDMatte on BiRefNet's rough
        mask, and BiRefNet itself) are one group: their mistakes are one mistake, so they are one vote."""
        n = len(cands)
        parent = list(range(n))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        kinds = [{s["node"] for s in c["learned"]} for c in cands]
        for i in range(n):
            for j in range(i + 1, n):
                if kinds[i] & kinds[j]:
                    parent[root(j)] = root(i)
        roots = [root(i) for i in range(n)]
        order = list(dict.fromkeys(roots))
        return [order.index(r) for r in roots]

    @classmethod
    def say_licence(cls, ctx, cands: list[dict]) -> bool:
        """The strictest licence of the candidates is the result's: said when it is not commercial, naming every
        project above them that does not allow it (models or not)."""
        commercial = E.strictest_licence([c["commercial"] for c in cands])
        if not commercial:
            projects = sorted({s["project"] for c in cands for s in c["sources"] if not s.get("commercial", True)})
            ctx.say("W-ENSEMBLE-NONCOMMERCIAL", projects=projects)
        return commercial

    @staticmethod
    def common_frames(cands: list[dict]) -> list[int]:
        frames = sorted(set.intersection(*(set(c["packet"].meta["frames"]) for c in cands)))
        if not frames:
            raise Invalid(Msg("E-ENSEMBLE-NOFRAMES"))
        return frames

    # ------------------------------------------------------------------ outputs every map ensemble shares

    @classmethod
    def writers(cls, ctx, cands: list[dict], window, name: str, summary: dict, channels: int = 1, validity: bool = True,
                **result) -> dict:
        """The writers of the result (`channels`, with a validity channel or not, `result`: its value range, half,
        what it means), of the confidence map (0..1, the result's validity) and of which candidate each pixel came
        from (a label map, the candidates as its classes): those two only when wanted."""
        from ...data.payloads import ExrWriter

        out = {"result": ExrWriter(ctx.outputs[cls.result_port], channels, validity=validity, window=window, layer=name,
                                   ensemble=summary, **result)}
        if "confidence" in ctx.wanted:
            out["confidence"] = ExrWriter(ctx.outputs["confidence"], 1, validity=validity, half=True, value_range=(0.0, 1.0),
                                          window=window)
        if "choice" in ctx.wanted:
            classes = [{"index": i + 1, "name": c["title"]} for i, c in enumerate(cands)]
            out["choice"] = ExrWriter(ctx.outputs["choice"], 1, value_range=(0.0, float(len(cands))), classes=classes,
                                      window=window)
        return out

    @classmethod
    def scores_packet(cls, ctx, frames: list[int], names: list[str], values: np.ndarray, summary: dict):
        from ...data.payloads import curves_packet

        return curves_packet(ctx.outputs["scores"], frames, names, values, ensemble=summary)

    result_port = "result"

    @classmethod
    def finish(cls, ctx, writers: dict, frames, names, values, summary) -> dict:
        from ..base import empty_packet

        out = {cls.result_port: writers["result"].packet()}
        for port in ("confidence", "choice"):
            if port in ctx.wanted:
                out[port] = writers[port].packet()
        if "scores" in ctx.wanted:
            out["scores"] = cls.scores_packet(ctx, frames, names, values, summary)
        for port in ("confidence", "choice", "scores"):
            if port in ctx.outputs and port not in out:
                out[port] = empty_packet(ctx, port)
        return out


class EnsembleMatte(_Ensemble, NodeDef):
    """Alphas of several matting models combined, the object checked first.

    The user's selection (「对象」, a SAM 3 point-picked first-frame mask on the card) leaves out, for the whole shot, a
    candidate that draws more or less than it (selection_check). Per frame, candidates that share a guide are one vote
    (guide_groups); groups that disagree on which object it is (binary IoU below MATTE_AGREE) are settled by the
    majority, a tie by the candidate that drew the selection best (or, without one, by the previous frame's result);
    the losers are left out of that frame. The kept candidates' weighted mean is the result.
    - accuracy: selection margin SELECTION_MARGIN. Not passed (held-out SAD 8.7 % worse than VideoMaMa).
    - robust: the stricter SELECTION_MARGIN_ROBUST; the only ensemble objective that passed the strict rule (held-out
      median ≥ 5 % better than the best single model, no held-out clip more than 5 % worse). Measured (集成评测/
      一期报告.md; 117 clips with ground truth, thresholds set on 60 tuning clips, run once on 57 held-out ones):
      temporal stability (dtSSD) 7.8 % better than VideoMaMa on the held-out median with no clip more than 5 % worse;
      no wrong object on any of the 117 clips; edges (Grad) 24.6 % better on the median, SAD 8.7 % worse.
    - edges: as accuracy, then, where the candidate marked trusted (the one best on its own: VideoMaMa on the card)
      and the others' mean disagree, the result leans on it (kit/ensemble.py lean_on_trusted, 集成评测/一期b报告.md):
      held-out Grad 22 % better than VideoMaMa on the median with 2 clips more than 5 % worse (the plain mean: 8),
      SAD 2 % worse than VideoMaMa. Not passed by the strict rule either.
    """

    id = "ensemble_matte"
    version = 2  # 2：alpha 不再带有效性通道（交付时成了第二个 A）
    category = "matte_alpha"
    data_kind = "matte"
    result_port = "alpha"
    main = "alpha"
    on_node = ("objective",)
    inputs = (Port("object", "image.1", optional=True),)
    outputs = (
        Port("alpha", "image.1", means=("matte",)),
        Port("confidence", "image.1"),
        Port("choice", "image.1"),
        Port("scores", "curves"),
    )

    class Params(NodeParams):
        objective: Literal[E.MATTE_OBJECTIVES] = P("robust", group="ensemble")
        candidates: list[MatteCandidate] = P(
            [{"name": n, "weight": 1.0, "use": "blend", "label": ""} for n in CANDIDATES[:2]], widget="table",
            group="ensemble", max_length=len(CANDIDATES), validate_default=True,
        )

    @classmethod
    def cook(cls, ctx):
        from ...data.maps import map_at, same_size
        from ...data.payloads import window_of

        cands = cls.candidates(ctx)
        cls.identify(cands)
        same_size({c["title"]: c["packet"] for c in cands})
        commercial = cls.say_licence(ctx, cands)
        frames = cls.common_frames(cands)
        objective = ctx.params["objective"]
        groups = cls.guide_groups(cands)
        if len(set(groups)) < 2:
            ctx.say("N-ENSEMBLE-SHAREDGUIDE")
        prior_w = np.array([float(c["row"].get("weight", 1.0)) for c in cands])
        first = cands[0]["packet"]
        window = window_of(first)
        mask = ctx.input("object")
        lean_to = -1  # edges: the trusted candidate the result leans on where it and the others disagree
        if objective == "edges":
            marked = [k for k, c in enumerate(cands) if c["row"].get("use") == "trusted"]
            if len(marked) > 1:
                ctx.say("N-ENSEMBLE-TRUSTEDMANY", model=cands[marked[0]]["title"])
            if marked:
                lean_to = marked[0]
            else:
                ctx.say("N-ENSEMBLE-NOTRUSTED")

        def stack_at(f):
            out = []
            for c in cands:
                got = map_at(c["packet"], f)
                out.append(np.where(got[1] > 0, np.clip(got[0][..., 0], 0.0, 1.0), 0.0))
            return np.stack(out).astype(np.float32)

        name = E.ensemble_name("matte", objective, [c["token"] for c in cands])
        summary = {"objective": objective, "candidates": [c["title"] for c in cands], "groups": groups,
                   "commercial": commercial, "trusted": cands[lean_to]["title"] if lean_to >= 0 else None}
        # an alpha has a value everywhere: no validity channel (delivered, it would be a second A beside the alpha's own)
        w = cls.writers(ctx, cands, window, name, summary, validity=False, matte=True)
        n = len(cands)
        shot_ok = np.ones(n, bool)
        rank_of = None  # each candidate's fit to the user's selection: ties of the object check go to the closest
        if mask is not None:  # the user's selection, on the first frame it covers: who draws that object at all
            f0 = next((f for f in frames if map_at(mask, f) is not None), None)
            if f0 is not None:
                margin = E.SELECTION_MARGIN_ROBUST if objective == "robust" else E.SELECTION_MARGIN  # edges: as accuracy
                A0, chosen = stack_at(f0), (map_at(mask, f0)[0][..., 0] > 0.5).astype(np.float32)
                shot_ok = E.selection_check(A0, chosen, margin)
                rank_of = E.selection_fit(A0, chosen)
                for k in np.flatnonzero(~shot_ok):
                    ctx.say("N-ENSEMBLE-NOTSELECTED", model=cands[k]["title"])
        curve = np.zeros((len(frames), 2 * n))
        dropped_frames = np.zeros(n, int)
        previous = None
        ctx.stage("combine")
        # in frame order, one after the other: each frame's object check leans on the frame before (identity_vote)
        for j, f in enumerate(ctx.each(frames)):
            A = stack_at(f)
            sel = shot_ok if shot_ok.any() else np.ones(n, bool)
            voters = [g for g, o in zip(groups, sel) if o]
            order, reps = E.group_means(A[sel], voters)
            rank = None if rank_of is None else np.array([max(rank_of[k] for k in range(n) if sel[k] and groups[k] == g) for g in order])
            said = map_at(mask, f) if mask is not None else None
            prior = (said[0][..., 0] > 0.5).astype(np.float32) if said is not None else previous
            trusted = E.identity_vote(reps, prior, E.MATTE_AGREE, rank)
            kept = np.array([g in order and trusted[order.index(g)] for g in groups]) & sel
            alpha, choice = E.matte_fuse(A, kept, prior_w)
            if lean_to >= 0 and kept[lean_to]:
                leaned, share = E.lean_on_trusted(alpha.astype(np.float64), A[lean_to].astype(np.float64),
                                                  E.matte_deviation(A, kept, lean_to), E.TRUSTED_MATTE_DEVIATION)
                alpha = np.clip(leaned, 0.0, 1.0).astype(np.float32)
                choice = np.where(share > 0.5, lean_to, choice)
            previous = (alpha > 0.5).astype(np.float32)
            dropped_frames += ~kept
            w["result"].add(f, alpha)
            if "confidence" in w:
                w["confidence"].add(f, E.matte_confidence(A, kept).astype(np.float32))
            if "choice" in w:
                w["choice"].add(f, (choice + 1).astype(np.float32))
            curve[j, :n] = [E.binary_iou(A[k], alpha) for k in range(n)]
            curve[j, n:] = kept
        for k in range(n):
            if dropped_frames[k] and shot_ok[k]:
                ctx.say("N-ENSEMBLE-OTHEROBJECT", model=cands[k]["title"], count=int(dropped_frames[k]))
        ctx.say("I-ENSEMBLE-RESULT", objective=cls.option_label("objective", objective), count=n,
                models=i18n.Both.of(lambda: i18n.separator().join(c["title"] for c in cands)), name=name)
        names = [f"{c['token']}_iou" for c in cands] + [f"{c['token']}_kept" for c in cands]
        return cls.finish(ctx, w, frames, names, curve, summary)


NODES = (EnsembleMatte,)
