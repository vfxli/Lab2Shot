"""Nodes provided by the CoTracker3 extension (CC-BY-NC-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, P, PointTracker, TrackParams, Cost, Measured)


class Track(PointTracker):
    id = "cotracker.track"
    version = 2  # 2：长镜头分段重叠半窗、接缝在置信度最高的帧重新起跟、重叠区按置信度混合
    # 上游 CoTrackerPredictor.forward：video + queries / segm_mask -> tracks、visibilities
    official = Official(
        cite="third_party/cotracker/repo/cotracker/predictor.py:36-68",
        takes={"image": "video", "mask": "segm_mask"},
        gives={"tracks": "tracks"},
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   mask：实测（TAP-Vid、PointOdyssey）：遮罩只决定网格点撒在哪；点进去的跟踪点精度（AJ）没区别
    # 默认「整段」模式一次看一大段、前后双向一起算；4090 上 400 个点一段最多 240 帧，段间重叠 1 帧
    runtime = "cotracker"
    # 数值来自 RTX 4090、默认「整段」模式
    cost = Cost(gpu=True, vram_gb=5.6, seconds_per_frame=0.012)

    class Params(TrackParams):
        grid: Literal[0, 10, 16, 20] = measured_param({0: Measured(below=20), 10: Measured(below=20), 16: Measured(below=20), 20: Measured(gb=5.6)}, default=0, group="tracking")
        mode: Literal["offline", "online"] = P(
            "offline", group="model",
        )
        resolution: Literal[768] | None = measured_param({768: Measured(gb=6.7)}, group="model")


NODES = (Track,)
