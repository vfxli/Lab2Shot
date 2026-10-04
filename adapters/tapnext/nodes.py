"""Nodes provided by the TAPNext++ extension (Apache-2.0)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, P, PointTracker, TrackParams, Cost, Licence, Measured)

class Track(PointTracker):
    id = "tapnext.track"
    version = 2  # 2：支撑点半径按官方取模型输入的 32 像素（256 模型为画面 1/8）
    # 上游 TAPNext.forward(video, query_points, state) -> tracks、track_logits、visible_logits
    official = Official(
        cite="third_party/tapnext/repo/tapnet/tapnext/tapnext_torch.py:241-300",
        takes={"image": "video", "mask": "query_points"},
        gives={"tracks": "tracks"},
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   mask：实测（TAP-Vid、PointOdyssey）：遮罩只决定网格点撒在哪；点进去的跟踪点精度（AJ）没区别
    # 逐帧往前推的在线跟踪器，显存不随镜头长度增长（300 帧仍是 2.6 GB），几千帧也能跑；
    # 点可以在任意一帧撒，之前的帧倒着往回跟；固定机位和跟拍上背景点偏离中位数 0.38 px
    runtime = "tapnext"
    # RTX 4090，默认的 512 模型
    cost = Cost(gpu=True, vram_gb=2.6, seconds_per_frame=0.047)
    licence = Licence(note=True)

    class Params(TrackParams):
        grid: Literal[0, 10, 16, 20] = measured_param(
            {0: Measured(below=20), 10: Measured(below=20), 16: Measured(below=20), 20: Measured(gb=2.6)}, default=0, group="tracking")
        model: Literal["512", "256"] = P("512", group="model")
        resolution: Literal[256, 512] | None = measured_param(
            {256: Measured(gb=1.7), 512: Measured(gb=2.6)}, auto=True, group="model")



NODES = (Track,)
