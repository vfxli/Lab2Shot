"""光流和稠密对应：画面帧之间每个像素怎么动，或者对到另一张图上。"""

from __future__ import annotations

import numpy as np

from ...data.packet import Packet
from ...errors import Invalid
from ...messages import Msg
from ...data.payloads import UNIT, ExrWriter
from ...data.units import DEFAULT_WIDTH
from ..base import NodeParams, Port
from .base import Job, RawOutput, WorkerNode
from ..kit.confidence import Confidence, ConfidenceWriter
from ..applies import Cost


def motion_vectors(ctx, raw: RawOutput, image: Packet, node_type) -> dict[str, Packet]:
    """The optical-flow raw contract (lab2shot_worker.optical_flow: per frame forward / backward [h,w,2] at the
    processing size, +y down, and their confidences) -> motion vectors in Nuke's convention (the plate's pixels per
    frame, forward.u / forward.v to the next frame, backward.u / backward.v to the previous one, u right, v up; 0 where
    there is no neighbour) and one 置信度 per pixel: the lower of the two directions' (the worker SDK gives them 0–1),
    empty when the model gives none."""
    from ...data.maps import resize
    from ...data.payloads import window_of

    w, h = window_of(image).canvas  # the pixels the worker was sent
    ctx.stage("写出运动矢量")
    flow_map = ExrWriter(ctx.outputs["flow"], 4, window=window_of(image))
    scores = ConfidenceWriter(ctx, image, node_type)
    for f, d in raw.frames(ctx, image.meta["frames"]):
        vectors = np.zeros((h, w, 4), np.float32)
        sure = np.ones((h, w), np.float32)
        for k, key in enumerate(("forward", "backward")):
            if key not in d:
                continue
            flow = resize(d[key], w, h) * np.array([w / d[key].shape[1], h / d[key].shape[0]], np.float32)
            vectors[..., 2 * k] = flow[..., 0]
            vectors[..., 2 * k + 1] = -flow[..., 1]  # Nuke: v up
            if f"{key}_confidence" in d:
                sure = np.minimum(sure, resize(d[f"{key}_confidence"], w, h))
        flow_map.add(f, vectors)
        scores.add(f, sure if any(k.endswith("_confidence") for k in d.files) else None)
    return {"flow": flow_map.packet(), **scores.packet()}


class OpticalFlowParams(NodeParams):
    """Parameters every optical-flow node shares; each declares its own 处理分辨率 (flow_resolution_param, with the
    limits measured on that model)."""


# 处理分辨率留空（None）= 原生分辨率，4K/8K 素材会直接以原尺寸进模型、超过模型能稳定处理的上限。
# 这个隐式默认值钳到安全上限；用户显式填的数字不动。
# 上限和 DEFAULT_WIDTH 数值相同只是巧合（一个是模型的安全处理尺寸，一个是画面尺寸未知时的默认值），
# 复用常量只是为了不再多写一遍这个数字
NATIVE_FLOW_CEILING = DEFAULT_WIDTH


def clamped_flow_side(ctx, image, resolution: int | None) -> int | None:
    """`resolution` 留空时按 NATIVE_FLOW_CEILING 自动封顶；明确填的数字不动，那是用户自己的选择。"""
    native = max(image.meta["width"], image.meta["height"])
    if resolution is None and native > NATIVE_FLOW_CEILING:
        ctx.say("W-FLOW-CAPPED", side=native, ceiling=NATIVE_FLOW_CEILING, param="resolution")
        return NATIVE_FLOW_CEILING
    return resolution


class OpticalFlow(WorkerNode):
    """Optical flow of a plate: motion vectors to the next and the previous frame (Nuke's motion layer) and how sure the
    model is (motion_vectors()). Occlusion masks, warps and ST-maps come from the generic motion-vector nodes.
    A missing frame fails. Job.notes: none."""

    on_node = ("resolution",)
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (Port("flow", "image.4", "运动矢量"),)
    confidence = Confidence("probability", help="运动矢量模型自己估的每个矢量有多可信（0–1，越大越可信；前后两个方向取低的）。"
                                                "只在这个模型的结果之间比高低；当遮罩用先接「置信度转遮罩」，要遮挡的地方用「遮挡遮罩」")
    cost = Cost(gpu=True)
    Params = OpticalFlowParams

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        frames = image.meta["frames"]
        if len(frames) < 2:
            raise Invalid(Msg("E-FLOW-ONEFRAME"))
        gaps = [b - a for a, b in zip(frames, frames[1:]) if b - a != 1]
        if gaps:
            ctx.say("W-FLOW-GAPS", count=len(gaps))
        resolution = clamped_flow_side(ctx, image, ctx.params["resolution"])
        return Job(image, extra={"resolution": resolution} if resolution != ctx.params["resolution"] else {})

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        return motion_vectors(ctx, raw, job.plate, cls)


def correspondence(ctx, raw: RawOutput, image: Packet, node_type, frames: list[int] | None = None) -> dict[str, Packet]:
    """The dense-correspondence raw contract (lab2shot_worker.correspondence: per plate frame, where each pixel is in
    the other picture, and how sure; raw/result.json "target": the other picture's size) -> an ST-map sequence on the
    plate that takes the other picture there (Nuke's STMap: R = x / W, G = 1 - y / H of the other picture's size) and
    its 置信度 (0–1: the pixel is seen in the other picture and the match is right, the model's own score, as the node's
    `confidence` declares). `frames`: the plate frames solved (default all)."""
    from ...data.maps import resize
    from ...data.payloads import window_of

    target = raw.result()["target"]
    tw, th = target["width"], target["height"]
    w, h = window_of(image).canvas  # the pixels the worker was sent
    ctx.stage("写出 ST-map")
    stmap = ExrWriter(ctx.outputs["stmap"], 2, value_range=UNIT, window=window_of(image))
    scores = ConfidenceWriter(ctx, image, node_type)
    for f, d in raw.frames(ctx, frames or image.meta["frames"]):
        xy = resize(d["xy"], w, h)
        stmap.add(f, np.stack([xy[..., 0] / tw, 1.0 - xy[..., 1] / th], -1))
        scores.add(f, d["confidence"])
    return {"stmap": stmap.packet(), **scores.packet()}
