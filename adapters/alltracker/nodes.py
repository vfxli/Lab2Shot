"""Nodes provided by the AllTracker extension (MIT: commercial use allowed)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, Job, P, PointTracker, Port, TrackParams, correspondence,
                          empty_packet, track_queries, tracks, Cost, Measured)


class Track(PointTracker):
    """点跟踪家族的一员（网格点、点选的点、「跟踪点数」这件事实都来自家族），另多一个稠密的 ST-map 输出；
    没有遮罩输入（上游只吃画面），网格点和点选都没有时也照样出 ST-map。"""

    id = "alltracker.track"
    # 上游 demo.py forward_video：rgbs -> traj_maps_e（每帧指回参考帧的稠密轨迹图）、trajs_e（按 --rate 抽样的 2D 点）
    official = Official(
        cite="third_party/alltracker/repo/demo.py:130-149",
        takes={"image": "rgbs"},
        gives={"stmap": "traj_maps_e", "tracks": "trajs_e"},
        note="上游 forward_sliding(images, iters, sw, is_training, window_len, "
             "stride)（nets/alltracker.py:363）只吃画面：没有查询点、没有遮罩。2D 点是把稠密轨迹图按 --rate "
             "的步长抽样出来的（demo.py:148）。所以节点没有遮罩输入口：要只跟一块区域，"
             "在图上接「人物框转遮罩」→「图像合成」（留下）把画面挡住再送进「RGB」口",
    )
    # 每一帧均指回参考帧，向前、向后均跟踪（整段）；长镜头仅受内存限制（长边 1024 时约每 100 帧 1.5 GB），一次完成计算
    runtime = "alltracker"
    # vram_gb：RTX 4090，处理分辨率 1024
    cost = Cost(gpu=True, vram_gb=11.1, seconds_per_frame=0.095)
    inputs = PointTracker.inputs[:1]  # the family's 「RGB」 without its mask (upstream takes none)
    outputs = (Port("stmap", "image.2", "ST-map"), *PointTracker.outputs)
    confidence = Confidence("probability", help="每个像素在这一帧里看不看得见、跟得准不准（AllTracker 的可见度 × 置信度，0–1）。当遮罩用先接「置信度转遮罩」")

    class Params(TrackParams):
        query_frame: int | None = P(None, label="参考帧", group="跟踪", placeholder="第一帧")
        # the grid is sampled from the dense result: any count costs the same
        grid: Literal[0, 10, 20, 30, 50] = measured_param(
            "网格点数", {n: Measured(flat=True) for n in (0, 10, 20, 30, 50)},
            default=0, group="跟踪")
        resolution: Literal[512, 768, 1024] = measured_param(
            "处理分辨率", {512: Measured(below=1024), 768: Measured(below=1024), 1024: Measured(gb=11.1)},
            default=1024, group="跟踪")

    min_frames = 2  # a track needs a second frame (refused before cooking, nodes/expects.py FrameCount)

    @classmethod
    def prepare(cls, ctx):
        """与家族的不同只在于：没有查询点时也要算（ST-map 是整段的稠密结果）。"""
        image = ctx.input("image")
        return Job(image, inputs=track_queries(ctx, image, ctx.params["picks"]))

    @classmethod
    def convert(cls, ctx, raw, job):
        image = job.plate
        out = correspondence(ctx, raw, image, cls)
        if raw.path("tracks.npz").exists():
            out |= tracks(ctx, raw, image, cls.runtime, ctx.params["min_confidence"])
        else:  # no grid and nothing clicked: the ST-maps only
            out["tracks"] = empty_packet(ctx, "tracks")
        return out


NODES = (Track,)
