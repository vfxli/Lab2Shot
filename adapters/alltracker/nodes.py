"""Nodes provided by the AllTracker extension (MIT: commercial use allowed)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, Confidence, Handle, Job, P, Port, TrackParams, WorkerNode, correspondence,
                          empty_packet, track_queries, tracks, Cost, Measured)


class Track(WorkerNode):
    id = "alltracker.track"
    # 上游 demo.py forward_video：rgbs -> traj_maps_e（每帧指回参考帧的稠密轨迹图）、trajs_e（按 --rate 抽样的 2D 点）
    official = Official(
        cite="third_party/alltracker/repo/demo.py:130-149",
        takes={"image": "rgbs"},
        gives={"stmap": "traj_maps_e", "tracks": "trajs_e"},
        note="上游 forward_sliding(images, iters, sw, is_training, window_len, stride)（nets/alltracker.py:363）"
             "只吃画面：没有查询点、没有遮罩。2D 点是把稠密轨迹图按 --rate 的步长抽样出来的（demo.py:148）。"
             "原来节点上有一个「遮罩」输入，是我们自己加的：它一个字节都没进模型，只在 build_queries 里决定"
             "网格点撒在哪几个像素上。已删除——要只跟一块区域，在图上接"
             "「人物框转遮罩 → 图像相乘」把画面挡住再送进来。",
    )
    on_node = ("grid", "query_frame")
    # 每一帧均指回参考帧，向前、向后均跟踪（整段）；长镜头仅受内存限制（长边 1024 时约每 100 帧 1.5 GB），一次完成计算
    runtime = "alltracker"
    # vram_gb：RTX 4090，处理分辨率 1024
    cost = Cost(gpu=True, vram_gb=11.1, seconds_per_frame=0.095)
    inputs = (Port("image", "image.3", "RGB"),)
    outputs = (Port("stmap", "image.2", "ST-map"), Port("tracks", "tracks2d", "2D 跟踪点"))
    confidence = Confidence("probability", help="每个像素在这一帧里看不看得见、跟得准不准（AllTracker 的可见度 × 置信度，0–1）。当遮罩用先接「置信度转遮罩」")
    handles = (Handle("points", {"points": "picks"}),)

    class Params(TrackParams):
        query_frame: int | None = P(None, label="参考帧", group="跟踪", placeholder="第一帧",
                                    help="在哪一帧上画（帧号）：每一帧的 ST-map 都指回这一帧，网格点也撒在这一帧，前后都会跟过去。"
                                         "选一帧要修补的东西清楚、正对镜头的帧")
        # the grid is sampled from the dense result: any count costs the same
        grid: Literal[0, 10, 20, 30, 50] = measured_param(
            "网格点数", {n: Measured("从稠密结果里取样：显存和时间不随点数变（1024 长边 0.095 秒/帧）", flat=True) for n in (0, 10, 20, 30, 50)},
            default=0, group="跟踪",
            help="每边的点数：从 ST-map 里在参考帧上均匀取 n×n 个点（20 = 400 个点）；默认 0 = 只跟你在 2D 视图里点的那些点（点完可以拖着调位置）和 ST-map；撒网格点就选 10 / 20 / 30 / 50")
        resolution: Literal[512, 768, 1024] = measured_param(
            "处理分辨率", {512: Measured("比实测的一档省", below=1024), 768: Measured("更快", below=1024), 1024: Measured("1080×1920 120 帧 0.095 秒/帧；内存约每 100 帧 1.5 GB", gb=11.1)},
            default=1024, group="跟踪",
            help="长边缩到这个像素再跟踪，结果放大回原尺寸（位置跟着放大，比这更细的细节是插值出来的）")

    min_frames = 2  # a track needs a second frame (refused before cooking, nodes/expects.py FrameCount)

    @classmethod
    def prepare(cls, ctx):
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
