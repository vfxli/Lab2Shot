"""运动矢量：运动矢量变形、遮挡遮罩、运动矢量转 ST-map. Written once for any producer of motion vectors (MEMFOF,
WAFT, vectors rendered from CG or made by Nuke's VectorGenerator, read with 「读取序列」): what is derived from motion
vectors lives here, so a new optical-flow method needs no code of its own beyond its vectors.

Motion vectors are Nuke's (map.motion): per pixel of a frame, forward = where it is on the next frame of the sequence,
backward = where it is on the previous one, minus where it is, in pixels, u right and v up. In image pixels (+y down)
a pixel at (x, y) of frame t is at (x + u, y - v) on the neighbour.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ...errors import Invalid
from ...messages import Msg
from ...data.contracts import Shape, warped_by
from ..base import NodeDef, NodeParams, P, Port
from ..expects import SameShot

RELATIVE = 0.01  # forward-backward check: the part of the vectors' length two ways may differ by (Sundaram et al. 2010)


def tolerance_param():
    return P(1.0, label="容差", unit="px", ge=0.05, le=20.0, group="检查",
             help="前后一致性检查：一个像素顺着运动矢量到邻帧、再顺着邻帧的矢量回来，差得超过这么多像素（外加矢量长度的 1%）"
                  "就算被挡住或新露出来。噪点多、遮挡边缘太碎就调大；漏掉了细的遮挡就调小")


def _grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:h, :w].astype(np.float32) + 0.5
    return x, y


def _vectors(motion, frame: int, backward: bool) -> np.ndarray:
    """One direction of a frame's motion vectors in image pixels (+y down): [H, W, 2]."""
    from ...data.maps import map_at

    data = map_at(motion, frame)[0]
    uv = data[..., 2:4] if backward else data[..., 0:2]
    return np.stack([uv[..., 0], -uv[..., 1]], -1)


def consistent(there: np.ndarray, back_there: np.ndarray, inside: np.ndarray, tolerance: float) -> np.ndarray:
    """Where going `there` [H,W,2] and coming back by the neighbour's vectors found there (`back_there`) returns to the
    same pixel (bool [H,W]); pixels that leave the picture never are."""
    miss = np.sum((there + back_there) ** 2, -1)
    return inside & (miss <= RELATIVE * (np.sum(there**2, -1) + np.sum(back_there**2, -1)) + tolerance**2)


def neighbours(frames: list[int]) -> dict[int, tuple[int | None, int | None]]:
    """frame -> (previous, next) in the sequence (None at its ends): what the vectors of that frame point at."""
    return {f: (frames[i - 1] if i else None, frames[i + 1] if i + 1 < len(frames) else None) for i, f in enumerate(frames)}


class MotionWarp(NodeDef):
    id = "core.motion_warp"
    picture = "src"
    version = 2  # image packets now always say whether they have an alpha
    category = "img_warp"
    on_node = ("direction",)
    inputs = (Port("src", "image", "源", alpha=True), Port("flow", "image.4", "运动矢量", expects=(SameShot(of="src"),)))
    outputs = (Port("image", "image", "结果", type_from="input:src", shape=warped_by("flow")),)

    class Params(NodeParams):
        direction: Literal["next", "previous"] = P(
            "next", label="取哪一帧", group="变形", option_labels={"next": "下一帧", "previous": "上一帧"},
            help="下一帧：每一帧取下一帧的内容，用 forward 矢量对齐到这一帧（最后一帧没有下一帧，不输出）；"
                 "上一帧：取上一帧的内容，用 backward 矢量（第一帧不输出）")

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import (ExrWriter, file_at, has_alpha, has_validity, is_data, is_labels, read_map,
                                      read_picture)
        from ...data.types import channels_of
        from ...data.maps import same_size, sample
        from ...data.contracts import own_meta
        from ...data.payloads import window_of

        src, motion = ctx.input("src"), ctx.input("flow")
        same_size({"源": src, "运动矢量": motion})
        backward = ctx.params["direction"] == "previous"
        # 它什么二维数据都收，所以「这是画面还是数值图」「这是编号图吗」由数据自己带的东西说，不由通道数猜
        picture, labels = not is_data(src), is_labels(src)
        keeps_alpha = has_validity(src)  # 源图自己带着「哪里有值」那条通道，结果也带着
        # the warp lies on the motion vectors' window, and reads the source on its plate frame (the vectors are in its pixels)
        window, src_box = window_of(motion), window_of(src).display_box
        writer = ExrWriter(ctx.outputs["image"], channels_of(src.type), validity=keeps_alpha,
                           colorspace=src.meta["colorspace"] if picture else None,
                           value_range=tuple(src.meta["range"]) if not picture else None,
                           window=window, **own_meta(src),
                           **({"alpha": True} if picture and has_alpha(src) else {}))
        near = {f: pair[0 if backward else 1] for f, pair in neighbours(motion.meta["frames"]).items()}
        frames = [f for f, other in near.items() if other is not None and file_at(src, other) is not None]
        lacking = [f for f in motion.meta["frames"] if f not in frames]
        if lacking:
            ctx.say("N-MOTION-NOPREVIOUS" if backward else "N-MOTION-NONEXT", count=len(lacking), first=lacking[0])
        if not frames:
            raise Invalid(Msg("E-MOTION-NOFRAMES"))
        x, y = _grid(motion.meta["height"], motion.meta["width"])
        for f in ctx.each(frames):
            path = file_at(src, near[f])
            if picture:  # an alpha pulled along with the picture (premultiplied: sampled as it is)
                values, alpha = read_picture(src, path, src_box), None
            else:
                values, alpha = read_map(path, src_box)
            v = _vectors(motion, f, backward)
            # 编号图（带类别表）取最近的一个，编号之间插值会插出不存在的编号；别的都双线性，和 Nuke 的 IDistort 一样
            out, valid = sample(values, x + v[..., 0], y + v[..., 1], alpha if keeps_alpha else None,
                                nearest=labels)
            writer.add(f, out, valid)
        return {"image": writer.packet()}


class MotionOcclusion(NodeDef):
    picture = "flow"  # its result lies on the motion vectors' window
    id = "core.motion_occlusion"
    category = "mask_make"
    on_node = ("tolerance",)
    inputs = (Port("flow", "image.4", "运动矢量"),)
    outputs = (Port("occluded", "image.1", "遮挡"), Port("revealed", "image.1", "新露出"))

    class Params(NodeParams):
        tolerance: float = tolerance_param()

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter
        from ...data.maps import sample

        motion, tol = ctx.input("flow"), ctx.params["tolerance"]
        frames = motion.meta["frames"]
        if len(frames) < 2:
            raise Invalid(Msg("E-MOTION-ONEFRAME"))
        h, w = motion.meta["height"], motion.meta["width"]
        x, y = _grid(h, w)
        near = neighbours(frames)
        from ...data.payloads import window_of

        window = window_of(motion)
        out = {port: ExrWriter(ctx.outputs[port], 1, value_range=UNIT, half=True, window=window) for port in ("occluded", "revealed")}
        got = {port: False for port in out}
        for f in ctx.each(frames):
            for port, backward in (("occluded", False), ("revealed", True)):
                other = near[f][0 if backward else 1]
                if other is None:  # the end of the shot: nothing to check against
                    out[port].add(f, np.zeros((h, w), np.float32))
                    continue
                there = _vectors(motion, f, backward)
                back, inside = sample(_vectors(motion, other, not backward), x + there[..., 0], y + there[..., 1])
                picked = ~consistent(there, back, inside > 0, tol)
                got[port] = got[port] or bool(picked.any())
                out[port].add(f, picked)
        empty = [cls.outputs[i].label for i, port in enumerate(("occluded", "revealed")) if not got[port]]
        if empty:  # 一个像素都没判出来：交出空遮罩并留一句（空结果不是错误，但要留提醒）
            ctx.say("N-MOTION-NOOCCLUSION", which="、".join(empty), tolerance=tol)
        return {port: writer.packet() for port, writer in out.items()}


class MotionStmap(NodeDef):
    picture = "flow"  # its result lies on the motion vectors' window
    id = "core.motion_stmap"
    category = "img_warp"
    on_node = ("query_frame",)
    inputs = (Port("flow", "image.4", "运动矢量"),)
    main = "stmap"
    outputs = (Port("stmap", "image.2", "ST-map", shape=Shape(lens="unknown")), Port("valid", "image.1", "有效区域"))

    class Params(NodeParams):
        query_frame: int | None = P(None, label="参考帧", group="传播", placeholder="第一帧",
                                  help="在哪一帧上画（帧号）。每一帧的 ST-map 都指回这一帧；选一帧要修补的物体清楚、正对镜头的帧，"
                                       "前后都会串过去")
        tolerance: float = tolerance_param()

    @classmethod
    def cook(cls, ctx):
        from ...data.payloads import UNIT, ExrWriter
        from ...data.maps import sample

        motion, tol = ctx.input("flow"), ctx.params["tolerance"]
        frames = motion.meta["frames"]
        ref = frames[0] if ctx.params["query_frame"] is None else ctx.params["query_frame"]
        if ref not in frames:
            raise Invalid(Msg("E-MOTION-REFFRAME", frame=ref, first=frames[0], last=frames[-1]))
        h, w = motion.meta["height"], motion.meta["width"]
        x, y = _grid(h, w)
        from ...data.payloads import window_of

        window = window_of(motion)
        stmap = ExrWriter(ctx.outputs["stmap"], 2, value_range=UNIT, window=window)
        valid = ExrWriter(ctx.outputs["valid"], 1, value_range=UNIT, half=True, window=window)
        identity = np.stack([x, y, np.ones_like(x)], -1)

        def write(f, chain):
            stmap.add(f, np.stack([chain[..., 0] / w, 1.0 - chain[..., 1] / h], -1))
            valid.add(f, np.clip(chain[..., 2], 0.0, 1.0))

        write(ref, identity)
        # away from the reference both ways: after it back by the backward vectors, before it on by the forward ones.
        # chain [H,W,3]: where each pixel is on the reference frame (x, y), and whether the way back holds (0..1)
        i = frames.index(ref)
        steps = [(f, True) for f in frames[i + 1:]] + [(f, False) for f in frames[:i][::-1]]
        towards, chain = ref, identity
        for f, backward in ctx.each(steps):
            if not backward and towards != frames[frames.index(f) + 1]:  # the second leg starts at the reference again
                towards, chain = ref, identity
            there = _vectors(motion, f, backward)
            nx, ny = x + there[..., 0], y + there[..., 1]
            chain, inside = sample(chain, nx, ny)
            back, _ = sample(_vectors(motion, towards, not backward), nx, ny)
            chain[..., 2] *= consistent(there, back, inside > 0, tol)
            write(f, chain)
            towards = f
        return {"stmap": stmap.packet(), "valid": valid.packet()}


NODES = (MotionWarp, MotionOcclusion, MotionStmap)
