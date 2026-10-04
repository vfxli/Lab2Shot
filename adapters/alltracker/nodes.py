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
        )
    # 每一帧均指回参考帧，向前、向后均跟踪（整段）；长镜头仅受内存限制（长边 1024 时约每 100 帧 1.5 GB），一次完成计算
    runtime = "alltracker"
    # vram_gb：RTX 4090，处理分辨率 1024
    cost = Cost(gpu=True, vram_gb=11.1, seconds_per_frame=0.095)
    inputs = PointTracker.inputs[:1]  # the family's 「RGB」 without its mask (upstream takes none)
    outputs = (Port("stmap", "image.2"), *PointTracker.outputs)
    confidence = Confidence("probability")  # its tooltip: node.alltracker.track.port.confidence.help
    needs_points = False  # 网格点和点选都没有时照样出 ST-map：不拦（PointTracker.wiring_notes）

    class Params(TrackParams):
        query_frame: int | None = P(None, group="tracking")
        # the grid is sampled from the dense result: any count costs the same
        grid: Literal[0, 10, 20, 30, 50] = measured_param({n: Measured(flat=True) for n in (0, 10, 20, 30, 50)},
            default=0, group="tracking")
        resolution: Literal[512, 768, 1024] = measured_param({512: Measured(below=1024), 768: Measured(below=1024), 1024: Measured(gb=11.1)},
            default=1024, group="tracking")

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
