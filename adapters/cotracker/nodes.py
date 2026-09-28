"""Nodes provided by the CoTracker3 extension (CC-BY-NC-4.0: research only)."""

from __future__ import annotations

from typing import Literal

from lab2shot.sdk import (Official, measured_param, P, PointTracker, TrackParams, Cost, Licence, Measured)


class Track(PointTracker):
    id = "cotracker.track"
    # 上游 CoTrackerPredictor.forward：video + queries / segm_mask -> tracks、visibilities
    official = Official(
        cite="third_party/cotracker/repo/cotracker/predictor.py:36-68",
        takes={"image": "video", "mask": "segm_mask"},
        gives={"tracks": "tracks"},
        note="segm_mask 是上游自己的输入（predictor.py:46「Segmentation mask of shape (B, 1, H, W)」）："
             "给了它，网格点就只撒在遮罩里（第 43 行注释）。可见性（visibilities）在我们的 2D 跟踪点数据里，"
             "不是单独的口。",
    )
    # 公开基准上的实测（接不接、接什么的差别）：
    #   mask：实测（TAP-Vid、PointOdyssey）：遮罩只决定网格点撒在哪；点进去的跟踪点精度（AJ）没区别
    # 默认「整段」模式一次看一大段、前后双向一起算；4090 上 400 个点一段最多 240 帧，段间重叠 1 帧
    runtime = "cotracker"
    # 数值来自 RTX 4090、默认「整段」模式
    cost = Cost(gpu=True, vram_gb=5.6, seconds_per_frame=0.012)
    licence = Licence(note="代码和模型都是 CC-BY-NC-4.0，只能研究用，不能商用。")

    class Params(TrackParams):
        grid: Literal[0, 10, 16, 20] = measured_param(
            "网格点数", {0: Measured(below=20), 10: Measured(below=20), 16: Measured(below=20), 20: Measured(gb=5.6)}, default=0, group="跟踪")
        mode: Literal["offline", "online"] = P(
            "offline", label="方式", group="模型",
            option_labels={"offline": "整段", "online": "逐段滑动"},
        )
        resolution: Literal[768] | None = measured_param(
            "处理分辨率", {768: Measured(gb=6.7)}, auto="512×384", group="模型")


NODES = (Track,)
