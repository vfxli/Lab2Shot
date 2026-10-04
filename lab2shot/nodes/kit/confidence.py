"""模型自己给的置信度：一个节点一处声明（Confidence），每种给法各有一条映射到 0-1，有置信度的节点统一多一个输出口和一份写出去的代码。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ...data.packet import Packet
from ...data.payloads import UNIT, ExrWriter
from ..base import Port, empty_packet


@dataclass(frozen=True)
class ConfidenceScale:
    """How a model gives its confidence, and the one mapping to 置信度 (0–1, higher = more trusted; a model's order kept,
    never normalised per shot: a shot's scores then compare with another shot's of the same model)."""

    # how the model gives it and how it is mapped, as the output's tooltip says: confidence.scale.<id>
    to_unit: Callable[[np.ndarray], np.ndarray]


# Declared per node by what its model computes (WorkerNode.confidence), mapped only here.
CONFIDENCE_SCALES: Mapping[str, ConfidenceScale] = MappingProxyType({
    # already a probability (Pi3's sigmoid; the worker SDK's optical flow and correspondence give 0-1 too)
    "probability": ConfidenceScale(lambda c: c),
    # c = 1 + exp(x) >= 1, the DUSt3R family's way (VGGT, MapAnything, CUT3R, MonST3R, LingBot-Map, Depth Anything 3):
    # 1 - 1/c = exp(x) / (1 + exp(x)) is the sigmoid of x, the same kind of number as Pi3's
    "exp_plus_one": ConfidenceScale(lambda c: 1.0 - 1.0 / np.maximum(c, 1.0)),
    # UniDepth: c = exp(the depth error it expects, metres), trained with log c against |depth - truth| after a median
    # rescale, so larger is LESS trusted (typically 0.77 on smooth walls, 1.7-2.8 on edges, up to hundreds on sky).
    # 1/(1 + c) is the sigmoid of -log c: 0.5 where it expects no error, lower as the error it expects grows
    "exp_error": ConfidenceScale(lambda c: 1.0 / (1.0 + np.maximum(c, 0.0))),
    # UniK3D: c = the |log depth| error it expects (about the relative error: 0.008 is 0.8 %), trained against it
    # directly, so larger is LESS trusted (typically 0.007 on smooth surfaces, 0.017 on edges). 0.01 / (0.01 + c): an
    # expected 1 % error is 0.5, 0.25 % is 0.8, 4 % is 0.2
    "log_error": ConfidenceScale(lambda c: 0.01 / (0.01 + np.maximum(c, 0.0))),
})


@dataclass(frozen=True)
class Confidence:
    """A node's declaration that its model computes a confidence per pixel: how it gives it (a CONFIDENCE_SCALES key)
    and, when the generic tooltip does not say it well, what the score means for this model."""

    scale: str
    # True: the node says what the score means itself (node.<type>.port.confidence.help); an id: a family's words
    # (confidence.<id>)
    help: str | bool = ""
    # Whether the family's own point cloud (点云) leaves out pixels scored below 0.5 (kit/maps.py family_points). False
    # when the official code does not filter its points by this score and the score is only a ranking within a picture,
    # not a probability (MapAnything: infer(apply_confidence_mask=False) by default, and when asked it cuts a percentile
    # of each view, model.py:2039-2040; its scores sit at 1.00-1.02, all below 0.5 once mapped). The 置信度 output is
    # given either way.
    gates_points: bool = True

    def __post_init__(self):
        if self.scale not in CONFIDENCE_SCALES:
            raise ValueError(f"confidence scale {self.scale!r} is not one of CONFIDENCE_SCALES ({', '.join(CONFIDENCE_SCALES)})")
        if isinstance(self.help, str) and self.help and not (self.help.isidentifier() and self.help.isascii()):
            raise ValueError(f"confidence help is True or a family's words id (confidence.<id>), not {self.help!r}")

    def port(self, node_type) -> Port:
        """The 置信度 output this declaration gives `node_type` (nodes/applies.py all_outputs lists it by the type order)."""
        return confidence_port(node_type)


def to_confidence(raw: np.ndarray, scale: str) -> np.ndarray:
    """A model's own confidence [H,W] -> 置信度 0..1 (CONFIDENCE_SCALES); what is not finite is 0."""
    unit = CONFIDENCE_SCALES[scale].to_unit(np.nan_to_num(np.asarray(raw, np.float32), nan=0.0, posinf=1e30))
    return np.clip(unit, 0.0, 1.0).astype(np.float32)


def confidence_port(node_type) -> Port:
    """The 置信度 output of a node that declares a Confidence: its tooltip says whose it is and how it was mapped."""
    c = node_type.confidence
    from ... import i18n
    from ..text import scope_of

    # its words: the node's own port.confidence.help when it has one (Confidence(help=True)), the family's
    # (confidence.<id>), else the general one, naming the node and how its scale was mapped
    if isinstance(c.help, str) and c.help:
        given = i18n.t(f"confidence.{c.help}")
    else:
        given = i18n.t("confidence.help", node=node_type.subtitle, scale=i18n.t(f"confidence.scale.{c.scale}"))
    return Port("confidence", "image.1", may_be_empty=True, help=given).owned(getattr(node_type, "id", ""), scope_of(node_type) or "", "output")


class ConfidenceWriter:
    """The 置信度 output of a node that declares a Confidence (WorkerNode.confidence), frame by frame at the plate's size;
    does nothing for a node without one."""

    def __init__(self, ctx, plate: Packet, node_type):
        c = node_type.confidence
        self.ctx = ctx
        self.scale = c.scale if c is not None else ""
        from ...data.payloads import window_of

        self.size = window_of(plate).canvas  # the pixels the worker was sent (a canvas is wider than the plate frame)
        self.writer = None
        if c is not None and "confidence" in ctx.outputs:
            self.model = node_type.project.title
            from ...data.payloads import window_of

            self.writer = None if "confidence" not in ctx.wanted else \
                ExrWriter(ctx.outputs["confidence"], 1, value_range=UNIT,
                          window=window_of(plate), model=self.model)

    def add(self, frame: int, scores: np.ndarray | None) -> None:
        """The model's own scores on a frame (any size, its scale); None: it gave none on this frame."""
        if self.writer is None or scores is None:
            return
        from ...data.maps import resize

        self.writer.add(frame, resize(to_confidence(scores, self.scale), *self.size))

    def packet(self) -> dict[str, Packet]:
        if self.writer is None:
            return {}
        if not self.writer.files:  # the model gave none this time (Depth Anything 3's Metric-Large)
            self.ctx.say("N-CONFIDENCE-EMPTY", model=self.model)
            return {"confidence": empty_packet(self.ctx, "confidence")}
        return {"confidence": self.writer.packet()}
