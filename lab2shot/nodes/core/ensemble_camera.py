"""The camera ensemble: cameras of several solvers of one shot, already in one world, combined into one.

One node per kind of data (nodes/core/ensemble.py: what every ensemble shares, _Ensemble); the kernel is
nodes/kit/ensemble_camera.py. The node only combines: bringing every candidate into the base camera's world is the
card's work (camera_space, fit "whole"), done before; candidates moved into different base cameras' worlds are refused.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import numpy as np

from ... import i18n
from ...errors import Invalid
from ...messages import Msg
from ..base import NodeDef, NodeParams, P, Port
from ..expects import SameReference
from ..kit import ensemble as E
from ..kit import ensemble_camera as EC
from .ensemble import CANDIDATES, _Ensemble
from .scene import CameraSpaceConvert


class CameraCandidate(NodeParams):
    """One row of 「候选」 = one input (a camera already moved into the base camera's world): its port, its prior weight,
    its display name."""

    name: Literal[CANDIDATES] = P(..., widget="fixed")
    weight: float = P(1.0, ge=0.0)
    label: dict[str, str] | str = P("")


class EnsembleCamera(_Ensemble, NodeDef):
    """Cameras of several solvers of one shot, already in the base camera's world (camera_space on the card), combined
    into one.

    The judge is the shot's own 2D tracks (「跟踪点」, CoTracker's): per candidate, each track is triangulated from half
    of its views and reprojected into the other half (kit/ensemble_camera.py held_out_errors), the deviation a
    matchmover reads, independent of every candidate. A candidate whose whole-shot error is over CAMERA_GATE times the
    best one's is not used. Objectives (kit/ensemble_camera.py combine):
    - accuracy: per stretch of CAMERA_WINDOW frames the candidate whose cameras the tracks fit best, switches
      cross-faded;
    - robust: the candidate the tracks fit best over the whole shot;
    - smooth: per frame the rotation median and the geometric median of the centres of the candidates used.
    Without tracks there is no judge: every objective takes the median of all candidates and says so.
    When the base camera (the one camera_space handed through as its own target; else the first) barely moves (centre
    spread under camera_space's MOVED_CM), camera_space could not read the others' scale and the judge does not see
    it: the base camera is used unchanged, whatever the objective.

    Each frame's lens is the one of the candidate its pose comes from, given in the first candidate's filmback (the
    same field of view and principal point in pixels), so a pose never meets a lens it was not solved with.

    Measured (ViPE and COLMAP as candidates, 11 clips with ground truth, 6 tuning / 5
    held out, ATE sim3): no objective passed the strict rule — robust 2.414 cm on the held-out median against COLMAP's
    1.856 (30 % worse), accuracy 3.069, smooth 2.541 — and robust stopped ViPE's failure on one clip but chose wrong on
    two. Its card (3D 跟踪匹配（模型集成）) was removed; no card uses this node.
    """

    id = "ensemble_camera"
    category = "camera_traj"
    data_kind = "camera"
    result_port = "camera"
    main = "camera"
    picture = ""
    candidate_data = False  # a camera: neither values of a map nor a picture
    on_node = ("objective",)
    ports_from_type = "scene.camera"
    inputs = (Port("tracks", "tracks2d", optional=True, multi=True),)
    outputs = (
        Port("camera", "scene.camera"),
        Port("confidence", "curves"),
        Port("choice", "curves"),
        Port("scores", "curves"),
    )

    class Params(NodeParams):
        objective: Literal[EC.CAMERA_OBJECTIVES] = P("robust", group="ensemble")
        candidates: list[CameraCandidate] = P(
            [{"name": n, "weight": 1.0, "label": ""} for n in CANDIDATES[:2]], widget="table",
            group="ensemble", max_length=len(CANDIDATES), validate_default=True,
        )

    @classmethod
    def candidate_expects(cls) -> tuple:
        """Every candidate camera in one base camera's world (camera_space on the card)."""
        return (SameReference(),)

    @classmethod
    def cook(cls, ctx):
        from ...data.camera import CameraSamples

        cands = cls.candidates(ctx)
        cls.identify(cands)
        commercial = cls.say_licence(ctx, cands)
        frames = cls.common_frames(cands)
        if len({(c["packet"].meta.get("aligned") or {}).get("reference") for c in cands}) > 1:
            raise Invalid(Msg("E-ENSEMBLE-NOTONESPACE"))
        objective = ctx.params["objective"]
        cams = [CameraSamples.from_packet(c["packet"], frames) for c in cands]
        first = cams[0]
        tracks = cls._tracks(ctx, frames)
        width = int((tracks[2] if tracks else 0) or first.width or cands[0]["packet"].meta.get("width") or 0)
        c2w = np.stack([np.asarray(c.poses(), np.float64) for c in cams])  # [C,F,4,4] GL axes, cm
        K = np.stack([EC.camera_K(c, width) for c in cams])  # [C,F,3,3] pixels
        prior = np.array([float(c["row"].get("weight", 1.0)) for c in cands])
        n = len(cands)

        ctx.stage("judge_candidates")
        judge = np.full((n, len(frames)), np.nan)
        if tracks is None:
            ctx.say("N-ENSEMBLE-NOTRACKS")
        else:
            for k in range(n):
                judge[k] = EC.frame_errors(EC.held_out_errors(c2w[k] @ EC.GL_TO_CV, K[k], tracks[0], tracks[1]))
            if not np.isfinite(judge).any():
                ctx.say("N-ENSEMBLE-FEWTRACKS", least=EC.TRACK_MIN_PER_FRAME)

        ctx.stage("combine")
        # The base camera (camera_space handed it through as the target itself) barely moving: camera_space read no
        # scale for the others (W-CAMSPACE-STILL), and the judge does not see scale, so a switch could change the
        # world's size. Keep the base camera.
        least = CameraSpaceConvert.MOVED_CM
        base = EC.still_base(c2w, [(c["packet"].meta.get("aligned") or {}).get("fit") for c in cands], least)
        still = base is not None
        if still:
            ctx.say("N-ENSEMBLE-CAMERASTILL", model=cands[base]["title"], spread=EC.spread_cm(c2w[base]), least=least)
        result, choice, used = EC.combine(c2w, judge, prior, objective, only=base)
        whole = EC.whole_error(judge, prior)
        if np.isfinite(whole).any() and not still:
            best = float(np.min(whole[np.isfinite(whole)]))
            for k in np.flatnonzero(~EC.allowed(whole)):
                if np.isfinite(whole[k]):
                    ctx.say("N-ENSEMBLE-CAMERAOFF", model=cands[k]["title"], px=float(whole[k]), best=best, times=EC.CAMERA_GATE)
                else:
                    ctx.say("N-ENSEMBLE-CAMERANOJUDGE", model=cands[k]["title"])
        idx = np.flatnonzero(used).tolist()
        name = E.ensemble_name("camera", objective, [cands[k]["token"] for k in idx])
        lensK = K[choice, np.arange(len(frames))]  # [F,3,3]: each frame's lens from the candidate its pose comes from
        out_cam = cls._with_lens(first, frames, result, lensK, width)
        out_cam = replace(out_cam, info={**first.info, "ensemble": name})
        summary = {"objective": objective, "candidates": [c["title"] for c in cands], "used": [cands[k]["title"] for k in idx],
                   "commercial": commercial,
                   "judge_px": [None if not np.isfinite(e) else float(e) for e in whole]}
        rest = {k: v for k, v in {"aligned": cands[0]["packet"].meta.get("aligned")}.items() if v is not None}
        out = {"camera": out_cam.write(ctx.outputs["camera"], name=name, ensemble=summary, layer=name, **rest)}
        if "confidence" in ctx.wanted:
            mine = (EC.frame_errors(EC.held_out_errors(result @ EC.GL_TO_CV, lensK, tracks[0], tracks[1]))
                    if tracks is not None else np.full(len(frames), np.nan))
            out["confidence"] = cls._curves(ctx, "confidence", frames, ["confidence"], EC.camera_confidence(mine)[:, None], summary)
        if "choice" in ctx.wanted:
            out["choice"] = cls._curves(ctx, "choice", frames, ["candidate"], (choice + 1)[:, None].astype(float), summary,
                                        classes=[{"index": i + 1, "name": c["title"]} for i, c in enumerate(cands)])
        if "scores" in ctx.wanted:
            out["scores"] = cls._curves(ctx, "scores", frames, [f"{c['token']}_px" for c in cands], judge.T, summary)
        from ..base import empty_packet

        for port in ("confidence", "choice", "scores"):
            if port in ctx.outputs and port not in out:
                out[port] = empty_packet(ctx, port)
        ctx.say("I-ENSEMBLE-RESULT", objective=cls.option_label("objective", objective), count=len(idx),
                models=i18n.Both.of(lambda: i18n.separator().join(cands[k]["title"] for k in idx)), name=name)
        return out

    @staticmethod
    def _with_lens(base, frames, c2w, K, width):
        """`base` (the first candidate) at `frames` with poses c2w [F,4,4] and per-frame intrinsics K [F,3,3] in pixels
        of a picture `width` wide, written in base's filmback and pixel aspect."""
        h_ap = float(np.median(base.h_aperture_mm))
        per_mm = width / h_ap
        height = float(base.height) * width / float(base.width) if base.width else 0.0
        focal = K[:, 0, 0] / per_mm
        center = np.stack([(K[:, 0, 2] - width / 2) / per_mm,
                           (height / 2 - K[:, 1, 2]) / (per_mm * float(base.pixel_aspect))], -1) if height else np.zeros((len(K), 2))
        n = len(frames)
        return replace(base, frames=tuple(int(f) for f in frames), cam_to_world=np.asarray(c2w, np.float64),
                       focal_mm=focal, h_aperture_mm=np.full(n, h_ap),
                       v_aperture_mm=np.full(n, float(np.median(base.v_aperture_mm))), center_mm=center)

    @staticmethod
    def _curves(ctx, port, frames, names, values, summary, **meta):
        from ...data.payloads import curves_packet

        return curves_packet(ctx.outputs[port], frames, names, values, ensemble=summary, **meta)

    @staticmethod
    def _tracks(ctx, frames):
        """Every wired track set at the candidates' common frames, one after the other: (xy [N,F,2], visible [N,F],
        the picture width they are in pixels of, 0 when unknown); None when nothing is wired."""
        from ...data.payloads import read_tracks

        xs, vs, width = [], [], 0
        for p in ctx.inputs.get("tracks") or []:
            if p is None or p.meta.get("empty"):
                continue
            t = read_tracks(p)
            width = width or int(p.meta.get("width") or 0)
            own = {int(f): i for i, f in enumerate(p.meta["frames"])}
            xy = np.full((len(t["tracks"]), len(frames), 2), np.nan)
            vis = np.zeros((len(t["tracks"]), len(frames)), bool)
            for j, f in enumerate(frames):
                i = own.get(int(f))
                if i is not None:
                    xy[:, j] = t["tracks"][:, i]
                    vis[:, j] = t["visible"][:, i]
            xs.append(xy)
            vs.append(vis)
        if not xs:
            return None
        return np.concatenate(xs), np.concatenate(vs), width


NODES = (EnsembleCamera,)
