"""Nodes provided by the TAPNext++ extension (Apache-2.0)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, P, PointTracker, TrackParams, Cost, Licence, Measured)

class Track(PointTracker):
    id = "tapnext.track"
    # 上游 TAPNext.forward(video, query_points, state) -> tracks、track_logits、visible_logits
    official = Official(
        cite="third_party/tapnext/repo/tapnet/tapnext/tapnext_torch.py:241-300",
        takes={"image": "video", "mask": "query_points"},
        gives={"tracks": "tracks"},
        note="上游没有遮罩参数：遮罩在我们这边只决定 query_points 撒在哪，进模型的还是 query_points。"
             "可见性（visible_logits）在我们的 2D 跟踪点数据里，不是单独的口。",
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   mask：实测（TAP-Vid、PointOdyssey）：遮罩只决定网格点撒在哪；点进去的跟踪点精度（AJ）没区别
    # 逐帧往前推的在线跟踪器，显存不随镜头长度增长（300 帧仍是 2.6 GB），几千帧也能跑；
    # 点可以在任意一帧撒，之前的帧倒着往回跟；固定机位和跟拍上背景点偏离中位数 0.38 px
    runtime = "tapnext"
    # RTX 4090，默认的 512 模型
    cost = Cost(gpu=True, vram_gb=2.6, seconds_per_frame=0.047)
    licence = Licence(note="代码和两个模型都是 Apache-2.0，可以商用。")

    class Params(TrackParams):
        grid: Literal[0, 10, 16, 20] = measured_param(
            "网格点数", {0: Measured(below=20), 10: Measured(below=20), 16: Measured(below=20), 20: Measured(gb=2.6)}, default=0, group="跟踪")
        model: Literal["512", "256"] = P(
            "512", label="模型", group="模型",
            option_labels={"512": "512", "256": "256"},
        )
        resolution: Literal[256, 512] | None = measured_param(
            "处理分辨率", {256: Measured(gb=1.7), 512: Measured(gb=2.6)}, auto="模型默认", group="模型")



NODES = (Track,)
