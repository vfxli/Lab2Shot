"""画面上的点，从头跟到尾（「2D 跟踪点」）：点跟踪器撒的网格点和视图里点的手动点，以及别的家族顺手算出来的
2D 关键点（人身上的肩、肘、腕，手上的指节）。

这里是 `tracks2d` 唯一的产出处：点跟踪家族走 tracks()，别的家族（全身动作 WorldHumans）走 keypoints2d()，
按「人 → 点」分组写进同一种数据，下游的 2D 跟踪点输出、CornerPin、3DE 文件因此一视同仁。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..kit.ports import rgb_port
from ...data.packet import Packet
from ...errors import Invalid, NothingToCook
from ...messages import Msg
from ...data.payloads import tracks_packet
from ..base import NodeParams, P, Port
from ..handles import Handle
from .base import Job, RawOutput, WorkerNode
from ..kit.ports import plate_mask_port
from ..applies import Cost, Fact


def tracks(ctx, raw: RawOutput, image: Packet, extension: str, min_confidence: float | None = None,
           names: list[str] | None = None, into=None, **meta) -> dict[str, Packet]:
    """raw/tracks.npz: tracks [N,F,2] pixels at the input resolution, visible [N,F], query_frames [N];
    optional confidence [N,F] 0..1, user [N] (points that came from the viewer, listed first) and homography [F,3,3]
    (points on one plane: the plane from their query frame to each frame).

    `min_confidence` replaces the model's own visible/hidden decision by confidence >= it
    (a point stays visible on the frame it starts at); the confidence itself is kept in the packet either way.
    `names`: the points' names (default user_NN / grid_NNNN); `meta`: more the packet says of them."""
    d = raw.arrays("tracks.npz")
    frames = image.meta["frames"]
    visible = d["visible"].astype(bool)
    if min_confidence is not None:
        if "confidence" not in d:
            raise Invalid(Msg("E-TRACKS-NOCONFIDENCE"))
        visible = d["confidence"] >= min_confidence
        start = np.searchsorted(frames, d["query_frames"])
        visible[np.arange(len(visible)), np.clip(start, 0, len(frames) - 1)] = True
    user = d["user"] if "user" in d else np.zeros(len(visible), bool)
    names = names or [f"user_{k + 1:02d}" if u else f"grid_{k + 1 - int(user.sum()):04d}" for k, u in enumerate(user)]
    # `into`：不把 2D 点当输出口交出去的节点（families/tracks3d.py gives_tracks2d=False）
    # 内部照样要这份数据算三维点，写进它给的临时文件夹
    return {"tracks": tracks_packet(into if into is not None else ctx.outputs["tracks"], frames, image.meta["width"],
                                    image.meta["height"], d["tracks"], visible, d["query_frames"], names,
                                    d["homography"] if "homography" in d else None,
                                    d["confidence"] if "confidence" in d else None, extension=extension, **meta)}


# ------------------------------------------------------------------ 2D 关键点（别的家族顺手算出来的点）


@dataclass(frozen=True)
class Keypoints2D:
    """一个节点的声明：它的方法在解算之前本来就在画面上找了一套 2D 关键点（ViTPose 的全身 17 点、手上的 21 点……）。
    声明了它，节点就多一个「2D 关键点」输出口，写出走 keypoints2d() 一份实现。

    `said`：这套点是什么、谁找的，写在输出口的悬停提示里（给美术看：点数、模型名）。"""

    said: str

    def port(self, node_type) -> Port:
        return Port("keypoints", "tracks2d", "2D 关键点", may_be_empty=True,
                    help=f"{self.said}。画面上的点，不是三维结果的投影：可以直接接「2D 跟踪点输出设置」交给 "
                         "3DEqualizer、Nuke，也可以叠在画面上看解出来的人贴不贴。按人分组，一个人一组")


@dataclass(frozen=True)
class PersonKeypoints:
    """一个人（或一只手、一张脸）在画面上的一套点：`name` 这一组叫什么（和这个人在 USD 里的名字一样），
    `frames` 他被解出来的帧，`xy` [F,K,2] 那些帧上每个点的像素位置（画面原尺寸），`confidence` [F,K] 模型自己的
    把握（0–1，没有就是 None），`names` [K] 每个点叫什么。"""

    name: str
    frames: list[int]
    xy: np.ndarray
    confidence: np.ndarray | None
    names: tuple[str, ...]


def keypoints2d(ctx, port: str, image: Packet, people: list[PersonKeypoints], **meta) -> dict[str, Packet]:
    """一组人的 2D 关键点 -> 一份「2D 跟踪点」，按「人 → 点」分组（data/items.py 的 groups：「逐项开始」「取一条」
    就能按人取）。点的名字是「这一组_这个点」，整份数据里不重名。

    这个人没被解出来的帧：位置按最近的一帧保持不变、标成「被挡住」（不可见），模型的把握照原样留在数据里 —— 和
    tracks() 一样，可见与否是结论，分数本身不丢。一个人都没有：空数据 + 一条提醒，不是错误。"""
    from ..base import empty_packet

    frames = image.meta["frames"]
    people = [p for p in people if p.frames and len(p.names)]
    if not people:
        ctx.say("N-KEYPOINTS-NONE")
        return {port: empty_packet(ctx, port)}
    index = np.asarray(frames)
    xy, visible, queries, names, groups, confidence = [], [], [], [], [], []
    for person in people:
        own = np.asarray(person.frames)
        nearest = np.clip(np.searchsorted(own, index), 0, len(own) - 1)  # 没解出的帧：保持最近一帧的位置
        before = np.maximum(nearest - 1, 0)
        take = np.where(np.abs(own[before] - index) <= np.abs(own[nearest] - index), before, nearest)
        k = len(person.names)
        xy.append(np.asarray(person.xy, np.float32)[take].transpose(1, 0, 2))  # [K,F,2]
        visible.append(np.repeat(np.isin(index, own)[None], k, 0))
        queries.append(np.full(k, int(own[0]), np.int64))
        names += [f"{person.name}_{n}" for n in person.names]
        groups.append({"name": person.name, "first": sum(g["count"] for g in groups), "count": k})
        if person.confidence is not None:
            confidence.append(np.asarray(person.confidence, np.float32)[take].T)  # [K,F]
    # 有一个人没有把握值，这份数据就整个不带把握值：半份分数没法比高低
    scores = np.concatenate(confidence) if len(confidence) == len(people) else None
    return {port: tracks_packet(ctx.outputs[port], frames, image.meta["width"], image.meta["height"],
                                np.concatenate(xy), np.concatenate(visible), np.concatenate(queries), names,
                                confidence=scores, groups=groups, **meta)}


def track_queries(ctx, image: Packet, picks: list[str]) -> dict[str, Path]:
    """Worker inputs of a point tracker: a mask to place the grid in and the viewer's clicked points (points.json)."""
    from ..handles import parse_picks, say_bad_entries

    inputs = ctx.input_files("mask")
    points = []
    say_bad_entries(ctx, "picks")
    for frame, x, y, _ in parse_picks(ctx.node_type, "picks", picks):
        if frame not in image.meta["frames"]:
            ctx.say("N-TRACKS-PICKOUTSIDE", frame=frame, x=x, y=y, param="picks")
        else:
            points.append({"frame": frame, "x": x, "y": y})
    if points:
        inputs["points"] = ctx.work / "points.json"
        inputs["points"].write_text(json.dumps({"points": points}), encoding="utf-8")
    return inputs


class TrackParams(NodeParams):
    """Parameters every point tracker shares; each adds its own model mode, processing size and 网格点数 (grid: the
    settings measured on that tracker, nodes/kit/ports.py measured_param)."""

    query_frame: int | None = P(None, label="参考帧", group="跟踪", placeholder="第一帧")
    picks: list[str] = P([], label="手动点", widget="picks", group="跟踪", placeholder="显示本节点，在 2D 视图里点要跟的位置", worker=False)
    min_confidence: float | None = P(None, label="可见门槛", ge=0.05, le=0.95, group="跟踪", placeholder="模型默认", worker=False)


class PointTracker(WorkerNode):
    """Point tracking: grid points (optionally inside a mask) and points clicked in the viewer, through the shot
    (tracks()). Job.notes: none."""

    on_node = ("grid", "query_frame")
    # how many points it gives, worked out from its own parameters: the node below it can say which of its choices
    # that suits (nodes/applies.py Incoming — 「2D 跟踪点输出设置」's CornerPin takes exactly four)
    fact_labels = {"points": "跟踪点数"}
    inputs = (rgb_port(), plate_mask_port("遮罩", every_frame=False))
    outputs = (Port("tracks", "tracks2d", "2D 跟踪点", may_be_empty=True),)
    cost = Cost(gpu=True)
    handles = (Handle("points", {"points": "picks"}),)

    @classmethod
    def facts(cls, params: dict) -> dict:
        """How many points it gives: the grid (n x n) plus each point clicked in the viewer. Known from the
        parameters alone, before anything cooks — a mask only moves the grid's points, it never changes how many."""
        from ..base import parse_picks

        grid = int(params.get("grid") or 0)
        picked = len(parse_picks(cls, "picks", list(params.get("picks") or [])))  # an entry that does not read is no point
        return {"points": Fact(grid * grid + picked, cls.fact_labels["points"])}

    @classmethod
    def prepare(cls, ctx) -> Job:
        image = ctx.input("image")
        inputs = track_queries(ctx, image, ctx.params["picks"])
        if not ctx.params["grid"] and "points" not in inputs:  # nothing asked for: an empty result, not an error
            raise NothingToCook(Msg("N-TRACKS-NOQUERY"))
        return Job(image, inputs=inputs)

    @classmethod
    def convert(cls, ctx, raw: RawOutput, job: Job) -> dict[str, Packet]:
        return tracks(ctx, raw, job.plate, cls.runtime, ctx.params["min_confidence"])
